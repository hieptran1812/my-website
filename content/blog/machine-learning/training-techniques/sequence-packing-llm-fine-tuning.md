---
title: "Sequence Packing for LLM Fine-Tuning: Eliminating Padding Waste to Reclaim Your GPU Budget"
date: "2026-06-26"
publishDate: "2026-06-26"
description: "A deep-dive into sequence packing — the training optimization that eliminates padding tokens and delivers 3-10× throughput gains on skewed datasets, with full NeMo and HuggingFace walkthroughs, bin-packing algorithm tradeoffs, and six production case studies."
tags: ["sequence-packing", "llm", "fine-tuning", "training-optimization", "flashattention", "nemo", "huggingface", "trl", "gpu", "throughput", "sft", "peft"]
category: "machine-learning"
subcategory: "Training Techniques"
author: "Hiep Tran"
featured: true
readTime: 50
---

There is a GPU tax that almost every fine-tuning job pays, and most teams have no idea it is happening. It shows up as a gap between the utilization number you see in `nvidia-smi` and the actual useful work the GPU is doing. The GPU might report 87% compute utilization, but if 50% of the tokens in your training batch are padding tokens — special fill values added to make every sequence the same length — then close to half of those expensive GPU operations are burning compute on tokens that will never contribute a single gradient update.

This is not a corner case. It is the default behavior of nearly every fine-tuning pipeline, and on instruction-tuning datasets where sequence lengths follow a heavily skewed distribution, the waste routinely exceeds 40–60% of total FLOPs. You are paying for a 100-GPU cluster, but you are getting the useful work of a 50-GPU cluster.

Sequence packing is the fix. The idea is conceptually simple: instead of padding each sequence to the length of the longest one in the batch, concatenate multiple shorter sequences into a single longer one. Pack more meaningful tokens per forward pass. Stop paying for silence.

![Standard padded batches vs sequence-packed batches: padding tokens carry full compute cost but emit zero gradient](/imgs/blogs/sequence-packing-llm-fine-tuning-1.webp)

The diagram above captures the core idea. On the left, four sequences share a batch. The shortest (256 tokens) gets padded out to 2048 tokens — 87% of its compute is wasted. With packing (right), all four sequences concatenate into a single 3840-token sequence. Every single token carries gradient signal. The GPU's attention kernel does real work on the whole thing.

NVIDIA's NeMo Framework reports up to **10× improvement in FLOPs efficiency** and **6× improvement in training time** with no impact on model convergence. Those are not theoretical numbers; they hold on instruction-tuning and SFT datasets with realistic skewed distributions. This article explains why packing achieves those gains, how the variable-length attention machinery makes it safe, which bin-packing algorithm to choose and when, and what can go wrong in production.

## 1. The Padding Problem

### Why variable-length sequences are the default

Modern LLM fine-tuning pipelines ingest data from instruction datasets, chat logs, question-answer pairs, code completions, and document summarizations. These sources share one property: their sequence lengths are wildly heterogeneous. A one-shot classification prompt might be 48 tokens. A multi-turn dialogue with a detailed system prompt might be 2,200 tokens. A code generation task with surrounding context can approach 8,000 tokens.

GPU kernels demand fixed-shape tensors. The attention mechanism computes queries, keys, and values over a full $\text{seq\_len} \times d_{\text{model}}$ matrix. If the sequences in a batch have different lengths, the batch tensor must be rectangular, which means every sequence gets padded to the length of the longest sequence in that batch.

This is where Zipf's law becomes your enemy. Real-world text length distributions are heavily right-skewed: the majority of samples are short, while a long tail of outliers anchors the maximum. In a batch of 16 instruction pairs where one sequence happens to be 2048 tokens long, every other sequence — even the one that's only 120 tokens — gets padded to 2048. On a dataset where the median length is 350 tokens and the 95th percentile is 2048, the per-batch waste runs between 50% and 80%.

The arithmetic is punishing. Suppose your GPU processes $B \times L_{\max}$ tokens per forward pass where $B$ is your micro-batch size and $L_{\max}$ is the maximum sequence length in the batch. If the average sequence length is $\bar{L}$, then the effective utilization is $\bar{L} / L_{\max}$. For a dataset where $\bar{L} = 420$ and $L_{\max} = 2048$, you are utilizing $420/2048 = 20\%$ of your compute. The other 80% is burned on tokens that the loss mask ignores.

> "The loss mask hides the waste from your metrics dashboard, but it doesn't hide it from your electricity bill."

The loss mask ensures padding tokens do not contribute to the training loss — that part is correct. But the attention kernel still computes query-key products for every token in the sequence, real or padded. The feed-forward layers still project every token's representation. The weight gradients still flow backward through every position. Masking is a bookkeeping fix that ignores the compute problem entirely.

### Quantifying your dataset's waste

Before reaching for packing, measure the padding ratio of your dataset with a quick analysis:

```python
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

lengths = []
with open("training.jsonl") as f:
    for line in f:
        sample = json.loads(line)
        # Format however your dataset uses (instruct, chat, etc.)
        text = sample["input"] + sample["output"]
        tokens = tokenizer(text, add_special_tokens=True)
        lengths.append(len(tokens["input_ids"]))

import statistics
max_len = 2048  # your target pack_size
p50 = statistics.median(lengths)
p95 = sorted(lengths)[int(0.95 * len(lengths))]
p99 = sorted(lengths)[int(0.99 * len(lengths))]
avg = statistics.mean(lengths)

print(f"Median:  {p50:.0f} tokens")
print(f"P95:     {p95:.0f} tokens")
print(f"P99:     {p99:.0f} tokens")
print(f"Mean:    {avg:.0f} tokens")
print(f"Padding ratio (mean vs max_len): {1 - avg/max_len:.1%}")
print(f"Expected packing speedup: {max_len/avg:.1f}x")
```

If the padding ratio exceeds 20%, sequence packing is worth the engineering investment. If it's under 10%, the complexity probably isn't justified. At 50% padding ratio, you are leaving a 2× speedup on the table.

## 2. What Sequence Packing Actually Does

### The concatenation trick

Sequence packing is, at its heart, a concatenation operation. Rather than padding sequence $A$ of length $s_A$ to $L_{\max}$ and processing it alone in a batch position, you concatenate sequences $A$, $B$, $C$, ... until their total length approaches $L_{\max}$ (the pack size). The resulting tensor has shape $(1, L_{\max})$ — it looks like a single long sequence — but it actually contains multiple independent training examples stitched end-to-end.

The tensor itself is straightforward. The complexity is in telling the attention mechanism which tokens are allowed to attend to each other. Without additional metadata, the attention kernel sees one long sequence and computes full cross-token attention over all $L_{\max}$ positions. A token from sequence $B$ would attend to tokens from sequence $A$, and the gradients from loss on $B$ would flow back through $A$'s key-value representations. That cross-contamination would silently corrupt training.

The correct solution requires two metadata arrays:

**`cu_seqlens`** (cumulative sequence lengths): A 1D integer array of length $N+1$ where $N$ is the number of packed sub-sequences. For a pack containing sequences of lengths $s_A, s_B, s_C$, this array is $[0, s_A, s_A + s_B, s_A + s_B + s_C]$. It tells the attention kernel exactly where each sub-sequence starts and ends.

**`position_ids`**: For each packed sub-sequence, position IDs restart at 0. Sequence $A$ gets positions $0, 1, \ldots, s_A - 1$; sequence $B$ also gets positions $0, 1, \ldots, s_B - 1$ (not $s_A, s_A+1, \ldots$). This preserves the rotary positional embeddings (RoPE) that LLMs use — without it, a token at pack position 1024 but logical position 37 within its sequence would have wildly incorrect positional encodings.

These two arrays are the low-level contract that makes packing safe. They are constructed during data preparation, stored alongside the token IDs, and passed to the attention kernel at training time.

## 3. The Attention Boundary Problem

### Why naive packing breaks training

Consider a packed sequence of total length $L = s_A + s_B + s_C$. Without attention boundaries, the standard attention computation is:

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

where $Q$, $K$, $V$ have shape $(L, d_k)$. This computes $L^2$ attention scores, and every query attends to every key, including keys from other packed sub-sequences. A token in sequence $B$ can attend to tokens in sequence $A$. The autoregressive causal mask (which blocks future tokens) does not help here — $A$ and $B$ have different logical positions, so an early token in $B$ can see late tokens in $A$.

The naive approach of constructing an $L \times L$ attention mask that blocks cross-sequence attention is technically correct but defeats the purpose of packing. The mask is $O(L^2)$ memory, and constructing and applying it is just as expensive as the cross-sequence attention you are trying to prevent.

### FlashAttention's variable-length kernels

The correct solution is a kernel-level implementation that never computes cross-sequence scores in the first place. FlashAttention 2 provides exactly this through its `varlen_fwd` function (also called `flash_attn_varlen_func`).

![Variable-length FlashAttention: cu_seqlens boundary metadata restricts computation to diagonal attention blocks only](/imgs/blogs/sequence-packing-llm-fine-tuning-2.webp)

The key insight is that with `cu_seqlens` in hand, the kernel splits the packed sequence into $N$ independent attention problems. For the pack $[A, B, C]$ with boundaries $[0, s_A, s_A+s_B, s_A+s_B+s_C]$, it computes three separate causal attention operations:

- Attention over $A$: $O(s_A^2)$ operations
- Attention over $B$: $O(s_B^2)$ operations  
- Attention over $C$: $O(s_C^2)$ operations
- **Total**: $O(s_A^2 + s_B^2 + s_C^2)$

The alternative — attending over the full packed length with a mask — would cost $O((s_A + s_B + s_C)^2)$. For our example of three sequences with lengths 256, 512, and 1024:

- Packed with varlen: $256^2 + 512^2 + 1024^2 = 65{,}536 + 262{,}144 + 1{,}048{,}576 = 1{,}376{,}256$ operations
- Naive padded single-sequence: $2048^2 = 4{,}194{,}304$ operations
- **Ratio**: varlen is **3.05× cheaper** than padded, and **2.33× cheaper** than a naively masked concat

The savings grow super-linearly with the spread in sequence lengths. NVIDIA's TransformerEngine also implements variable-length attention kernels and is the backend used in the NeMo Framework implementation.

### The `cu_seqlens` interface

Here is the actual FlashAttention 2 function signature for variable-length attention:

```python
from flash_attn import flash_attn_varlen_func

output = flash_attn_varlen_func(
    q,              # (total_tokens, n_heads, head_dim)
    k,              # (total_tokens, n_heads_k, head_dim)
    v,              # (total_tokens, n_heads_k, head_dim)
    cu_seqlens_q,   # (batch+1,) cumulative sequence lengths for queries
    cu_seqlens_k,   # (batch+1,) cumulative sequence lengths for keys
    max_seqlen_q,   # int: maximum sequence length in the batch
    max_seqlen_k,   # int: maximum sequence length in the batch
    causal=True,    # autoregressive mask within each sub-sequence
)
```

The inputs `q`, `k`, `v` are flat tensors of shape `(total_tokens, ...)` rather than `(batch, seq_len, ...)`. The `cu_seqlens` arrays tell the kernel how to slice them. The kernel processes each `cu_seqlens[i]:cu_seqlens[i+1]` segment as an independent attention problem — no cross-segment operations, no $O(L^2)$ masking overhead.

## 4. Bin-Packing Algorithms: FFD vs First-Fit Shuffle

### The bin-packing formulation

Given a set of sequences with lengths $\{s_1, s_2, \ldots, s_N\}$ and a target pack size $P$, bin-packing asks: how do we group sequences into packs such that the total length of each pack does not exceed $P$, and the number of packs is minimized?

This is a variant of the classical bin-packing problem (NP-hard in general, but polynomial-time approximations are perfectly adequate for our purposes). Two greedy algorithms dominate in practice:

![Bin-packing algorithm comparison: FFD vs FFS vs no-packing across efficiency, convergence risk, and use case](/imgs/blogs/sequence-packing-llm-fine-tuning-3.webp)

### First-Fit Decreasing (FFD)

**Algorithm**: Sort all sequences by length in descending order. Iterate through the sorted list; for each sequence, place it in the first existing pack that has enough remaining capacity. If no pack fits, open a new pack.

**Packing quality**: FFD consistently achieves within 11/9 × OPT + 6/9 bins of the optimal solution (a tight theoretical bound). In practice on real LLM fine-tuning datasets, it typically achieves 97–99% bin utilization — only 1–3% residual waste per pack.

**The convergence risk**: FFD sorts sequences before packing. Short sequences cluster together in the same packs; long sequences cluster in other packs. This breaks the i.i.d. assumption of stochastic gradient descent. When packs are not random samples of the data distribution, training batches are systematically biased toward specific length regimes, which can slow convergence or alter the final loss surface.

**When FFD is appropriate**: Datasets with relatively uniform length distributions (short sequences dominate with few long outliers), where the convergence risk is low because the clustering effect is mild. Also appropriate for tasks where training is purely throughput-bound and you can afford to monitor validation loss carefully.

```python
def first_fit_decreasing(sequences, pack_size):
    """
    sequences: list of (index, length) tuples
    Returns: list of packs, where each pack is a list of sequence indices
    """
    # Sort by length descending
    sorted_seqs = sorted(sequences, key=lambda x: x[1], reverse=True)
    
    packs = []           # list of (remaining_capacity, [indices])
    
    for idx, length in sorted_seqs:
        if length > pack_size:
            # Sequences longer than pack_size are handled separately (truncate or skip)
            continue
        
        placed = False
        for pack in packs:
            if pack[0] >= length:
                pack[0] -= length
                pack[1].append(idx)
                placed = True
                break
        
        if not placed:
            packs.append([pack_size - length, [idx]])
    
    return [pack[1] for pack in packs]
```

### First-Fit Shuffle (FFS)

**Algorithm**: Identical to FFD, except sequences are **not** sorted before packing. They are processed in their original (randomized) dataset order, or shuffled with a fixed seed for reproducibility.

**Packing quality**: FFS achieves 92–96% bin utilization in typical practice — slightly lower than FFD because unsorted placement leads to more fragmented packs. The residual waste is roughly 5–8% per pack.

**The convergence advantage**: Because sequences retain their random ordering, packs are unbiased samples of the length distribution. SGD sees a diverse mix of short and long sequences in each batch, matching the statistical properties of the padded baseline. NVIDIA explicitly recommends FFS for instruction-tuning and chat SFT workloads.

**The seed parameter**: The NeMo implementation accepts a `seed` argument for reproducibility. Setting `seed=0` produces a deterministic shuffle; changing the seed changes the pack assignment. This matters for comparing experiments — two runs with different seeds will have slightly different packing assignments, introducing a small source of variability.

```python
import random

def first_fit_shuffle(sequences, pack_size, seed=0):
    """
    sequences: list of (index, length) tuples (in original order)
    Returns: list of packs
    """
    rng = random.Random(seed)
    shuffled = list(sequences)
    rng.shuffle(shuffled)
    
    packs = []
    
    for idx, length in shuffled:
        if length > pack_size:
            continue
        
        placed = False
        for pack in packs:
            if pack[0] >= length:
                pack[0] -= length
                pack[1].append(idx)
                placed = True
                break
        
        if not placed:
            packs.append([pack_size - length, [idx]])
    
    return [pack[1] for pack in packs]
```

### Which algorithm to choose

| Situation | Algorithm | Reason |
|---|---|---|
| Instruction-tuning, chat SFT, RLHF | `first_fit_shuffle` | Convergence safety outweighs 5% efficiency loss |
| Short summarization / classification | Either; prefer `first_fit_decreasing` | Uniform lengths mean clustering risk is low |
| Long-context fine-tuning (≥ 8k) | `first_fit_shuffle` | Length variance is high; clustering effect would be severe |
| Ablation / reproducibility study | `first_fit_shuffle` with fixed seed | Determinism matters; FFS with seed=0 is reproducible |
| Maximum throughput (convergence monitored) | `first_fit_decreasing` | 3–5% better bin utilization translates to measurable throughput |

The practical default for most LLM fine-tuning is `first_fit_shuffle`. Only reach for FFD when you have explicitly measured that FFS convergence is acceptable on your specific dataset and you need every last percent of throughput.

## 5. NeMo Framework Implementation Walkthrough

### The two-stage offline preprocessing pipeline

NeMo's sequence packing is designed as an **offline preprocessing** step rather than an online data-loading transform. This is a deliberate architectural choice: packing is computationally cheap compared to training, but it benefits from running once and caching the packed dataset. The training loop reads pre-packed `.npy` files instead of packing on-the-fly.

![NeMo offline data preparation: JSONL → GPTSFTDataset → bin-pack → .npy files per pack size](/imgs/blogs/sequence-packing-llm-fine-tuning-4.webp)

The two stages are:

**Stage 1 — Online transformations** (through `GPTSFTDataset`): tokenization, special token injection, chat template application, truncation. This stage applies all the transformations that the training run would normally apply online, converting raw JSONL pairs into token ID sequences.

**Stage 2 — Packing algorithm**: the bin-packing algorithm groups the tokenized sequences into packs. Each pack is a flat array of token IDs with associated `cu_seqlens` and `position_ids` metadata. The output is serialized to `.npy` format.

### The preprocessing script

```bash
python scripts/nlp_language_modeling/prepare_packed_ft_dataset.py \
   model.data.train_ds.file_names=[/path/to/training.jsonl] \
   model.data.train_ds.max_seq_length=2048 \
   +tokenizer_path=/path/to/tokenizer.model \
   +output_dir=/path/to/output_dir \
   +pack_sizes=[2048,4096,8192] \
   +packing_algorithm=first_fit_shuffle \
   +seed=0
```

The `pack_sizes` argument accepts a list. This is a power feature: the script runs the packing algorithm once for each requested pack size and writes a separate `.npy` file per size. You can experiment with `pack_size=2048`, `pack_size=4096`, and `pack_size=8192` without re-tokenizing — analogous to trying different micro-batch sizes without rebuilding your dataset.

Choose `pack_size` to match your target `max_seq_length`. The `max_seq_length` parameter controls the longest sequence the model processes; setting `pack_size < max_seq_length` wastes context; setting `pack_size > max_seq_length` causes truncation.

### Modifying the training configuration

After preprocessing, the training configuration needs three specific changes:

```yaml
# Before packing:
model:
  data:
    train_ds:
      file_names: [/path/to/training.jsonl]
      max_seq_length: 2048
      micro_batch_size: 4
      global_batch_size: 128

# After packing — diff:
model:
  data:
    train_ds:
      file_names: [/path/to/output_dir/packed_2048.npy]  # packed .npy file
      max_seq_length: 2048                               # must equal pack_size
      packed_sequence: true                              # enable the packed path
      micro_batch_size: 1                                # MUST be 1 (packing replaces batching)
      global_batch_size: 32                              # original_gbs / avg_seqs_per_pack
```

Three constraints to never forget:

1. **`micro_batch_size` must be 1.** Each packed `.npy` entry is already a batch — it contains multiple sequences. Setting `mbs > 1` effectively doubles your batch size and corrupts gradient scaling. This is the single most common misconfiguration in production packing setups.

2. **`global_batch_size` must be adjusted.** If your original GBS was 128 and each pack contains an average of 4 sequences, set GBS to 32. Otherwise you process 4× more sequences per optimizer step than intended, equivalent to training with a 4× larger learning rate without scaling the LR.

3. **`max_seq_length` must equal `pack_size`.** The model needs to know the maximum sequence length to allocate attention buffers and positional encoding tables. If they mismatch, you get either truncation (pack_size > max_seq_length) or wasted allocation (pack_size < max_seq_length).

### Estimating the average sequences per pack

```python
import numpy as np

packed = np.load("/path/to/output_dir/packed_2048.npy", allow_pickle=True)

# Each element in the .npy array is a dict with keys 'input_ids', 'cu_seqlens'
seq_counts = [len(item['cu_seqlens']) - 1 for item in packed]

avg_seqs = np.mean(seq_counts)
print(f"Average sequences per pack: {avg_seqs:.2f}")
print(f"If original GBS=128, use GBS={int(128/avg_seqs)}")
```

### Multimodal packing (NeVA)

NeMo's sequence packing for multimodal LLMs (NeVA, LLaVA-style) uses a **different code path** from the text-only SFT/PEFT packing. Multimodal packing must account for the variable number of image tokens that the vision encoder injects into the sequence. Using the text-only packing path for a multimodal dataset will produce incorrect `cu_seqlens` metadata, leading to silent attention contamination.

## 6. HuggingFace and TRL Approach

### The evolution of HF packing support

The HuggingFace ecosystem took a different path: **online packing** integrated into the data collator, rather than offline preprocessing. The motivation is pipeline simplicity — you don't need a separate preprocessing step or a `.npy` format.

![HuggingFace DataCollatorWithPadding vs DataCollatorWithFlattening: same optimizer steps, zero padding overhead](/imgs/blogs/sequence-packing-llm-fine-tuning-5.webp)

The key paper is "Enhancing Training Efficiency Using Packing with Flash Attention" (IBM Research + HuggingFace, arXiv 2407.09105), which introduced `DataCollatorWithFlattening` into TRL 0.19.0. The collator concatenates sequences in each mini-batch into a flat tensor and uses FlashAttention 2's document masking to enforce per-sequence attention boundaries.

### DataCollatorWithFlattening

```python
from trl import SFTTrainer, SFTConfig
from trl.data_utils import DataCollatorWithFlattening
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    torch_dtype="auto",
    attn_implementation="flash_attention_2",  # required for document masking
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

dataset = load_dataset("json", data_files="training.jsonl")["train"]

# The critical part: DataCollatorWithFlattening
collator = DataCollatorWithFlattening(tokenizer=tokenizer)

config = SFTConfig(
    output_dir="./outputs",
    per_device_train_batch_size=4,   # each "sample" is already a flattened concat
    max_seq_length=2048,
    num_train_epochs=3,
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    data_collator=collator,
    args=config,
)
trainer.train()
```

Two requirements that are easy to miss:

- `attn_implementation="flash_attention_2"` on the model load. The document masking only works through FA2; the standard attention implementation will silently ignore the boundary metadata and compute full cross-sequence attention.
- Flash Attention 2 must be installed: `pip install flash-attn --no-build-isolation`.

### The padding-free vs packing distinction

TRL 0.19+ documentation distinguishes between "packing" (the old behavior, which bins sequences offline to maximize pack utilization) and "padding-free" (the new `DataCollatorWithFlattening` behavior, which just concatenates mini-batch sequences and trusts FA2 document masking). The naming is slightly confusing:

| Feature | Old TRL packing | DataCollatorWithFlattening |
|---|---|---|
| Bin-packing algorithm | Yes (FFD) | No — concatenates as-is |
| Offline preprocessing | Yes | No — online during training |
| Optimizer step count | Reduced (fewer batches) | Same as padded baseline |
| Convergence risk | Yes (fewer steps) | None — steps unchanged |
| FlashAttention 2 required | Recommended | Required |
| GPU memory savings | High | Moderate |
| Throughput gain | High (up to 10×) | Moderate (1.5–3×) |

`DataCollatorWithFlattening` is the safer choice for production: it preserves the number of optimizer steps exactly, so the training dynamics remain identical to the padded baseline. The throughput gain comes purely from eliminating the padding tokens within each concatenated mini-batch. Full offline bin-packing (NeMo-style) squeezes more utilization out of each batch but requires careful GBS recalibration.

## 7. The Software Stack: Four Layers of Packing Support

Sequence packing is not a single switch you flip. It requires coordinated support at four distinct abstraction layers, and a gap at any one of them silently corrupts training — often without an obvious error message.

![Four software stack layers for sequence packing: config, dataset, attention kernel, and CUDA ops](/imgs/blogs/sequence-packing-llm-fine-tuning-6.webp)

**Layer 1 — Training config**: `mbs=1`, recalibrated GBS, `max_seq_length = pack_size`, and `packed_sequence: true`. These are the user-facing knobs. Getting any one wrong produces incorrect gradients or memory errors, and the error messages often point somewhere else.

**Layer 2 — Packed dataset**: The dataset must produce packed tensors with correct `cu_seqlens` and `position_ids`. If you use a custom dataset class, you must ensure these metadata arrays are constructed and batched correctly. A dataset that produces `cu_seqlens` in Python integers instead of a CUDA int32 tensor will silently fall back to full attention.

**Layer 3 — FlashAttention `varlen_fwd`**: The attention kernel must be called with the varlen interface. Standard FA2 (`flash_attn_qkvpacked_func` or the standard HuggingFace attention wrapper) does not take `cu_seqlens`. The model's attention module must be explicitly wired to use `flash_attn_varlen_func`.

**Layer 4 — CUDA variable-length ops**: TransformerEngine and other CUDA kernels downstream of attention (layer norms, FFNs) may also need variable-length support. In NeMo, this is handled by the TransformerEngine backend. In HuggingFace, the rest of the forward pass operates on the flat `(total_tokens, d_model)` tensor produced by the varlen attention output, which is reshaped as needed.

Breaking this chain at any point produces one of three failure modes:
1. **Incorrect attention** (cross-sequence leakage): training proceeds, loss drops normally, but the model has learned spurious cross-sequence dependencies. Only detectible through careful convergence studies or behavioral evals.
2. **Shape mismatch**: Python exception during the forward pass, usually at the reshape before or after attention.
3. **Silent identity**: packing metadata is passed but ignored, and the kernel falls through to full attention — you pay packing's overhead without getting packing's benefit.

## 8. Performance Reality: When Does Packing Help?

### The leading indicator: padding ratio

The throughput gain from sequence packing is not a fixed number. It scales with how wasteful your dataset's padding was to begin with.

![Dataset characteristics predict packing speedup: skewed instruction datasets yield 3-10×, uniform code datasets yield near nothing](/imgs/blogs/sequence-packing-llm-fine-tuning-7.webp)

For a dataset with padding ratio $r$ (fraction of total tokens that are padding), the theoretical speedup from packing is approximately $1/(1-r)$. In practice the realized speedup is 60–80% of the theoretical maximum due to residual waste in the bins, overhead from metadata processing, and communication costs in distributed training.

| Dataset type | Typical padding ratio | Expected packing speedup |
|---|---|---|
| Open-source instruct (Alpaca, FLAN, OpenHermes) | 45–65% | 3–6× |
| Multi-turn chat (ShareGPT, UltraChat) | 35–55% | 2.5–5× |
| Code completion (varied context lengths) | 20–40% | 1.5–3× |
| Code generation (uniform templates) | 5–15% | 1.1–1.5× |
| Document summarization (fixed max length) | 10–25% | 1.2–2× |
| Long-context (128k context, full sequences) | <5% | 1.0–1.1× |

NVIDIA's NeMo benchmark reports **up to 10× FLOPs improvement** and **up to 6× wall-clock speedup** on instruction-tuning datasets with 50–70% padding ratios. These numbers are reproducible on standard fine-tuning benchmarks. The FLOPs number is always higher than the wall-clock number because memory-bound operations (KV-cache fills, parameter loads) scale differently than compute-bound ones.

### When wall-clock doesn't track FLOPs

Several factors reduce the wall-clock speedup below the theoretical FLOPs improvement:

**Memory bandwidth bounds**: On an A100 80GB, the HBM bandwidth is approximately 2 TB/s. For small batch sizes (mbs=1 with packing), many operations are memory-bandwidth-bound rather than compute-bound. Eliminating padding FLOPs helps the compute-bound operations but doesn't speed up the memory-bound ones.

**Communication overhead in distributed training**: Gradient synchronization in data-parallel training scales with parameter count, not token count. If your training run is communication-bound (e.g., 4-node training over Infiniband), packing helps the per-node computation but not the communication.

**DataLoader bottleneck**: Loading pre-packed `.npy` files from disk adds a slight overhead compared to in-memory padding. With SSDs and proper prefetching, this is negligible, but spinning HDDs and undersized CPU worker counts can erode the benefit.

**Pack boundary artifacts**: Packs with high residual waste (e.g., a single long sequence that almost fills a pack with a small gap) contribute less than full packs. Choosing a smaller `pack_size` increases residual waste; choosing a larger `pack_size` increases latency per step.

## 9. Convergence Considerations

### Why convergence is preserved (in theory)

The theoretical argument for convergence safety with packing is straightforward. In expectation, a packed dataset and a padded dataset process the same token gradient information — they just do so at different rates. Both methods take gradients from the same set of (non-padding) tokens and update the same model weights. The learning rate, optimizer state, and gradient accumulation all operate on the same effective information.

The empirical verification has been done by both NVIDIA and IBM Research. On standard SFT benchmarks (MMLU, MT-Bench, Alpaca-Eval), models trained with packing match models trained with padding within the bounds of normal run-to-run variance. This holds for both NeMo's offline FFD/FFS packing and HuggingFace's `DataCollatorWithFlattening`.

### When convergence goes wrong

Convergence can break in practice when:

**FFD clustering is severe**: On datasets where FFD's sorting creates extreme length clusters (e.g., all 128-token classification examples in 40% of packs, all 2048-token chat examples in the other 60%), the gradient landscape differs meaningfully from the padded baseline. The model sees a biased sequence of "easy" and "hard" examples rather than a random mix. Switch to FFS if you observe degraded convergence.

**GBS recalibration is wrong**: If you halve `mbs` but don't recalibrate `GBS`, you process 2× as many sequences per optimizer step, effectively running with a 2× larger LR and 2× larger batch size. Overtraining on early data is the usual symptom — validation loss drops fast then plateaus below the padded baseline.

**Very short packs dominate**: If a dataset has many sequences near `pack_size` in length (say, 95% of sequences are between 1800 and 2048 tokens for a pack_size of 2048), each pack contains only one sequence, and packing provides essentially no benefit. Worse, the position ID reset and `cu_seqlens` overhead adds cost with zero gain.

**LR warmup is not accounted for**: Some training setups use a warmup schedule based on number of optimizer steps. After packing, you may have fewer total optimizer steps (if using offline packing with reduced GBS). If the warmup is a fixed number of steps, the proportion of training spent warming up increases, which can slow early convergence.

### Checking convergence after packing

The reliable check is a held-out validation loss comparison. Before shipping a packed configuration to a full training run, run a 10% training-data smoke test with both padded and packed configurations. Compare the loss at equivalent numbers of **optimizer steps** (not gradient steps or epochs):

```python
import matplotlib.pyplot as plt
import numpy as np

# Load val loss curves from both runs
padded_steps, padded_loss = load_tensorboard("runs/padded/val_loss")
packed_steps, packed_loss = load_tensorboard("runs/packed/val_loss")

# Align by optimizer steps
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(padded_steps, padded_loss, label="Padded", color="tab:orange")
axes[0].plot(packed_steps, packed_loss, label="Packed (FFD)", color="tab:blue", linestyle="--")
axes[0].set_xlabel("Optimizer steps")
axes[0].set_ylabel("Validation loss")
axes[0].set_title("Loss by optimizer step (should match)")
axes[0].legend()

# Wall-clock comparison shows the speedup
axes[1].plot(padded_time_hours, padded_loss, label="Padded", color="tab:orange")
axes[1].plot(packed_time_hours, packed_loss, label="Packed (FFD)", color="tab:blue", linestyle="--")
axes[1].set_xlabel("Wall-clock time (hours)")
axes[1].set_ylabel("Validation loss")
axes[1].set_title("Loss by wall-clock (packed should be faster)")
axes[1].legend()

plt.tight_layout()
plt.savefig("convergence_comparison.png")
```

If the step-aligned curves match within 1–2%, packing is safe. If the packed curve is systematically higher, diagnose using the convergence failure modes above.

## 10. Advanced: Threshold Filtering Packing

### Beyond random bin-packing

Random bin-packing (FFS) treats sequence ordering as irrelevant — packs are random combinations of sequences that happen to fit together. A newer technique, **Threshold Filtering Packing (TFP)** (arXiv 2408.09327), takes the opposite approach: it deliberately packs **related** sequences together.

The TFP algorithm works as follows:
1. Embed all sequences using a fast sentence embedding model (e.g., a compressed sentence-BERT variant).
2. For each sequence, find its $K$ nearest neighbors in embedding space using approximate nearest-neighbor search (FAISS or similar).
3. Among the nearest neighbors, greedily pack together those whose total length fits within `pack_size`.
4. Fall back to random packing for sequences that don't have suitable neighbors.

The intuition: if a pack contains a question about Python decorators, a question about Python metaclasses, and a question about class inheritance, the model sees a micro-curriculum of related concepts in a single forward pass. The attention pattern (which only attends within each sub-sequence boundary, remember) doesn't directly benefit from this, but the gradient signal accumulated across related examples may reinforce common knowledge structures more efficiently.

**Empirical results**: TFP reports 3–7% improvement in downstream evaluation scores (MMLU, coding benchmarks) compared to random FFS packing, with identical throughput. The improvement is more pronounced on knowledge-dense tasks than on reasoning tasks.

**When TFP is worth the complexity**:
- You have a large, diverse instruction-tuning corpus (> 500k examples)
- You have access to a fast embedding pipeline (GPU embedding takes ~30 minutes for 500k examples)
- Your evaluation metrics show sensitivity to example order or curriculum
- You have already optimized the basics (learning rate, pack size, GBS) and are looking for incremental gains

**When to skip TFP**:
- Small datasets (< 50k examples) — neighbor quality is poor
- Datasets with inherent topic diversity (you may not want topic clustering)
- Time-sensitive pipelines where the embedding step adds unacceptable latency
- When baseline FFS already achieves your target metrics

## 11. Case Studies from Production

### 1. Llama-3 8B instruction-tuning, 50% padding ratio

A team fine-tuning Llama-3 8B on an open-source instruction mix (Alpaca + OpenHermes + FLAN-v2, ~800k samples) measured 52% average padding with a max_seq_length of 2048. They enabled FFD packing and observed a **5.8× wall-clock speedup** — from 14 hours to 2.4 hours for a full run on 8× A100 80GB.

The first convergence check revealed a problem: MMLU accuracy was 1.8 points below the padded baseline after identical optimizer steps. Diagnosis: FFD was clustering short classification-style examples (48–128 tokens) into the first 35% of packs, and long dialogue examples (1800–2048 tokens) into the rest. The model saw a two-phase curriculum — classification-heavy early, dialogue-heavy late — instead of the random mix it was trained with in the padded baseline.

Fix: switched to `first_fit_shuffle` with `seed=42`. The MMLU gap closed to 0.3 points (within normal run-to-run variance). Wall-clock time increased slightly to 3.0 hours (still 4.7× faster than the padded baseline). They shipped FFS.

**Lesson**: Always do a convergence smoke test before committing to FFD. The 5% throughput advantage of FFD over FFS is rarely worth the convergence risk on heterogeneous instruction datasets.

### 2. Code completion fine-tuning (StarCoder-style), near-uniform lengths

A team fine-tuning a 7B code model on function completion examples (Python, TypeScript, Rust) measured 12% average padding. Their examples ranged from 256 to 2048 tokens, but the distribution was approximately Gaussian centered at 1200 tokens — no long tail, no pathological short examples.

They enabled FFS packing expecting a proportional speedup. Observed speedup: **1.08× — essentially noise-level**. The analysis: most packs contained exactly one or two sequences (average 1.6 sequences per pack), because the sequences were already close to `pack_size`. The bin-packing overhead (sort + greedy placement) added ~3% latency to the data loading pipeline, almost entirely eating the 8% gain from eliminating padding.

Fix: reverted to standard padding. The team also ran a quick experiment with `pack_size=4096` to allow more sequences per pack — this achieved a 1.4× speedup but required adjusting the positional encoding table and increasing `max_seq_length`, which added memory pressure and slowed the optimizer step.

**Lesson**: Measure the average sequences per pack before committing to packing. If the ratio is below 2.0, packing saves little and may cost more than it saves.

### 3. NeVA multimodal training, position ID collision

A team training a multimodal LLM (NeVA, LLaVA-style architecture on Llama-3 Vision) attempted to reuse the text-only NeMo packing pipeline for their multimodal dataset. They applied `prepare_packed_ft_dataset.py` with the same flags as the text-only run.

The training started normally, but after 500 steps, the vision encoder loss stopped decreasing while the language model loss continued improving. The root cause: the text-only packing pipeline constructed `position_ids` based on token count, but in a multimodal model, image tokens are inserted at a variable position in the sequence (after the system prompt, before the user message). The `position_ids` array was computed as a flat `[0, 1, 2, ..., N-1, 0, 1, 2, ..., M-1, ...]` but the actual model expected the image tokens to receive their native positions within each sub-sequence, not to be treated as generic language tokens.

Fix: switched to the NeMo multimodal packing path (`prepare_packed_ft_dataset_multimodal.py`), which constructs `cu_seqlens` aware of the vision-language boundary within each sample. The vision encoder loss converged normally within 200 additional steps.

**Lesson**: Multimodal packing is a separate code path. Never apply text-only packing to a multimodal dataset.

### 4. 128k context fine-tuning, packing is a no-op

A team extending a Llama-3 model to 128k context window via continued pre-training on long documents attempted to apply sequence packing for throughput gains. Their dataset consisted of 128k-token documents (books, legal filings, technical papers), each occupying exactly one pack.

Observed speedup: **1.01×**. There is nothing to pack when every sample fills the entire pack. Packing added `cu_seqlens` construction overhead (~2ms per step) with zero utilization benefit.

More importantly, the team spent two engineering days adapting the packing pipeline to handle `pack_size=131072`, discovering edge cases in `cu_seqlens` construction for very long sequences and running into memory pressure from pre-loading full 128k `.npy` packs into RAM.

**Lesson**: Check the `pack_size / avg_seq_length` ratio before building a packing pipeline. If the ratio is ≤ 1.5, don't bother.

### 5. Continual pre-training, cross-document attention contamination

A team running continual pre-training on a proprietary domain corpus (medical literature) concatenated documents naively to fill `pack_size=4096` windows, without setting up `cu_seqlens`. They were not using NeMo or TRL — they had a custom training harness.

The training loss looked normal. But during evaluation, the model showed bizarre behavior: when completing text from one paper, it would occasionally hallucinate references and citations from unrelated papers, apparently having learned associations between documents that happened to be concatenated in the same pack.

The root cause was obvious in retrospect: without `cu_seqlens`, the attention kernel computed full cross-document attention. A key-value pair from the last sentence of document A was visible to the first sentence of document B. The model learned to rely on these spurious cross-document dependencies.

Fix: added `cu_seqlens` construction to the custom dataset class and switched to `flash_attn_varlen_func`. The hallucination behavior disappeared after retraining from the last clean checkpoint.

**Lesson**: Naive concatenation without `cu_seqlens` is not packing — it's cross-document attention contamination. Always verify that your attention kernel is receiving and honoring boundary metadata.

### 6. Wrong micro-batch size, loss explosion

A team applying NeMo packing to a Mistral-7B SFT run made a common configuration error. They set `packed_sequence: true`, changed the input file to the `.npy` pack file, and set `max_seq_length=2048` to match `pack_size=2048`. But they forgot to change `micro_batch_size`.

Original config: `micro_batch_size=4`, `global_batch_size=128`.
New config (mistaken): `micro_batch_size=4`, `global_batch_size=128`, `packed_sequence=true`.

With packing, each "sample" in the batch already contains multiple sequences (average 4.2 in this dataset). Setting `mbs=4` with packing means the model processes 4 packed samples × 4.2 sequences/sample = ~16.8 sequences per GPU per step, compared to 4 sequences per GPU per step in the padded baseline. The effective batch size was 4× larger than intended.

Symptom: training loss dropped 2× faster in the first 100 steps (consistent with a 4× larger batch), then stabilized at a loss 0.3 nats higher than the padded baseline (consistent with having overfit the early data on an inflated gradient).

Fix: set `micro_batch_size=1` and `global_batch_size=32` (original 128 ÷ average 4 sequences per pack). Re-ran from scratch.

**Lesson**: After enabling packing, the first thing you check in the training logs should be `sequences_per_second` and `tokens_per_second`. If sequences_per_second is 4× higher than the padded baseline but tokens_per_second is similar, your `mbs` is wrong.

### 7. Distributed 3D parallel packing, TP boundary issue

A team running 3D parallel training (TP=4, PP=2, DP=4) on a 70B model with packing enabled experienced a puzzling training instability. The loss showed random spikes every 200–300 steps, which correlated with specific batches rather than with optimizer state or learning rate schedule.

Tracing the spikes to specific training samples revealed the pattern: the instability happened on packs that contained exactly one very long sequence (>1900 tokens) followed by two short sequences, where the long sequence and the short sequences were distributed across PP stages with different TP ranks receiving different slices of the attention computation.

The root cause: `cu_seqlens` was being constructed on CPU rank 0 during data loading and broadcast to all DP ranks, but the tensor parallelism communication layer was not forwarding `cu_seqlens` along with the activations from PP stage 0 to PP stage 1. The stage-1 layers (including one more attention layer) received activations with shape `(1, pack_size, d_model)` and no boundary metadata. Without `cu_seqlens`, the attention layer in stage 1 defaulted to full causal attention across the entire packed sequence.

For single-sequence packs, this was irrelevant (full causal = same as varlen with one sequence). For multi-sequence packs, the stage-1 attention layer introduced cross-sequence contamination, which corrupted the gradient for those specific batches.

Fix: Modified the PP communication to explicitly forward `cu_seqlens` and `max_seqlen` alongside each micro-batch's activations. The spike pattern disappeared immediately.

**Lesson**: In PP setups, every stage that includes an attention layer must receive `cu_seqlens`. Inspect your PP communication payloads explicitly. Don't assume that the standard activation-only communication handles metadata.

### 8. Padding-free collation on a small GPU (24GB VRAM), OOM

A researcher running fine-tuning on a single RTX 3090 (24GB VRAM) adopted `DataCollatorWithFlattening` based on the throughput claims. The baseline training with padding at `mbs=4, max_seq_length=512` used approximately 18GB VRAM.

Switching to the flattening collator with the same `mbs=4` caused an immediate OOM after about 30 training steps. The VRAM usage jumped from 18GB to 26GB at step 31.

The root cause: the flattening collator concatenates the four sequences in each mini-batch into a single tensor of length up to `4 × 512 = 2048`. FlashAttention 2 allocates intermediate buffers proportional to `total_tokens²` (the attention score matrix). For padded training at `mbs=4`, the effective attention problem per-GPU is four independent `512 × 512` problems. For the flattened collator, it is one `2048 × 2048` problem (even though the varlen kernel only computes the four diagonal blocks). The intermediate buffer allocation uses the full `max_seqlen² = 2048²` worst-case size, not the actual diagonal-only computation.

In practice: padded attention used $4 \times 512^2 = 1{,}048{,}576$ score elements. Flattened varlen attention allocated for $2048^2 = 4{,}194{,}304$ elements — 4× the memory for intermediates, pushing over 24GB.

Fix: reduced `mbs` to 1 (flattened tokens per step = up to 512, not 2048). The actual throughput matched the `mbs=4` padded baseline because the flattening collator was already eliminating most padding waste. VRAM returned to 16GB.

**Lesson**: `DataCollatorWithFlattening` with `mbs > 1` multiplies the effective attention length by mbs, increasing FA2's intermediate buffers quadratically. On memory-constrained hardware, reduce `mbs` to 1 and compensate with gradient accumulation.

## 12. When to Use / When to Skip

### Reach for sequence packing when:

- **Padding ratio exceeds 20%** (measured as `1 - avg_token_length / max_seq_length`). Below 20%, packing complexity rarely pays off.
- **The dataset is offline-accessible** — you can afford a one-time preprocessing step to produce `.npy` pack files.
- **Training is GPU-bound**, not communication-bound or IO-bound. Packing helps the GPU compute phase; it doesn't help AllReduce synchronization or disk IO.
- **You have FlashAttention 2 available** (CUDA toolkit 11.6+, compute capability ≥ 7.5). Without FA2 varlen support, you cannot implement packing safely.
- **The model supports position ID overrides** — most modern LLMs do (Llama, Mistral, Qwen, Falcon), but some older architectures assume monotonically increasing position IDs.
- **You need reproducible throughput** — packing's offline nature means identical speed on every run, unlike dynamic batching with variable padding.

A note on the ordering: measure the padding ratio **first**, before touching any code. Almost every team that has invested a week building a packing pipeline on a dataset that turned out to have 8% padding ratio would have thanked themselves for spending 20 minutes on the analysis script first. The ROI on the measurement is infinite — it costs almost nothing and potentially saves substantial engineering time.

### Skip packing when:

- **Padding ratio is below 10%.** The throughput gain is less than 15%, and the engineering overhead (preprocessing, config changes, debugging) probably isn't worth it.
- **The dataset is streaming or dynamically generated** — you cannot preprocess a dataset that doesn't exist yet. Use `DataCollatorWithFlattening` instead, which works online.
- **Average sequence length is close to `pack_size`** — if `pack_size / avg_length < 1.5`, the average pack contains fewer than 1.5 sequences. You are spending preprocessing time for near-zero benefit.
- **You are using a framework that doesn't natively support variable-length attention** — any framework that forces standard attention shapes will silently compute incorrect cross-sequence attention.
- **Long-context fine-tuning with sequences near context window length** — at 128k context with documents averaging 100k tokens, packing is geometrically impossible.
- **Strict convergence requirements with no debugging time** — packing introduces new failure modes (wrong GBS, FFD clustering). If you cannot afford a convergence smoke test, stick with padded training.

![Decision tree: should I use sequence packing — two gates are padding ratio and offline preprocessing feasibility](/imgs/blogs/sequence-packing-llm-fine-tuning-8.webp)

The decision tree above walks the key gates. The short version: measure your padding ratio first. If it's above 20% and your dataset fits on disk, packing is almost always worth it — use `first_fit_shuffle` for safety, set `mbs=1`, recalibrate GBS, and run a 10% convergence smoke test before committing to a full run.

### The ROI curve

To make the decision concrete, here is the approximate ROI at different padding ratios:

| Padding ratio | Speedup | Engineering effort | Worth it? |
|---|---|---|---|
| < 10% | 1.05–1.15× | ~1–2 days | Rarely |
| 10–20% | 1.15–1.3× | ~1–2 days | Sometimes, if training is frequent |
| 20–40% | 1.3–2.0× | ~1–2 days | Usually |
| 40–60% | 2.0–4.0× | ~1–2 days | Yes |
| > 60% | 4.0–10× | ~1–2 days | Absolutely |

The engineering effort column assumes you are using either NeMo or TRL — both have the hard infrastructure work already done. From a clean starting point with one of those frameworks, enabling packing on a known-good dataset is a 1–2 day task: one day to run preprocessing and configure the training, another day for the convergence smoke test and any debugging. The speedup column is the payoff you get on every subsequent training run. For a dataset that you plan to train on multiple times (hyperparameter search, iterative data improvement), even a 1.5× speedup pays back the 2-day investment within a week.

## 13. Distributed Training with Packing

### Data parallelism and packed datasets

The standard distributed training setup with packing is data parallelism: each GPU (or DP rank) processes a disjoint shard of the packed dataset. The setup is straightforward because packing is done at the dataset level, not the model level. Each DP rank reads its own shard of the packed `.npy` file and processes packs independently.

The GBS recalibration applies globally across DP ranks. If you have 8 DP ranks and original `per_device_batch_size=4` (`GBS=32`), after packing with an average of 4 sequences per pack, set `per_device_batch_size=1` and `GBS=8`. The total token throughput remains similar, but the number of gradient updates per unit time increases because each optimizer step processes real tokens instead of padded ones.

### Tensor parallelism and packing

Tensor parallelism (TP) splits individual weight matrices across multiple GPUs. This interacts with packing at the attention layer. With TP, the query, key, and value projections are split across TP ranks. The `cu_seqlens` metadata must be consistently broadcast to all TP ranks so they all agree on the attention boundaries.

In NeMo with TransformerEngine, this is handled automatically — `cu_seqlens` is part of the model's context object that propagates through the TP communication layers. In custom training setups, you must explicitly include `cu_seqlens` in whatever tensor-parallel communication protocol you use.

A subtle bug: if `cu_seqlens` is constructed on rank 0 and not broadcast before the attention computation, TP ranks 1+ may compute attention with stale or zero-initialized boundary metadata. The symptom is training instability that begins at exactly the point where the attention layer runs, and the loss diverges differently on different TP ranks.

### Pipeline parallelism and packing

Pipeline parallelism (PP) splits the model's layers across GPUs, with micro-batches flowing through each stage. PP with packing requires that `cu_seqlens` and `position_ids` are included in the pipeline stage's communication payload — not just the activations.

If the pipeline ignores these metadata tensors (which happens if the pipeline's inter-stage communication is written for fixed-shape `(batch, seq_len, d_model)` tensors), the downstream stages receive activations with shape `(1, pack_size, d_model)` but have no boundary information. Each stage computes its operations (layer norm, FFN, etc.) on the full packed sequence as if it were a single contiguous sequence, which is correct for the non-attention layers. The attention layers in each stage still need `cu_seqlens`, so they must read it from the shared context rather than from the activation payload.

### 3D parallelism (DP × TP × PP)

In NeMo's 3D parallelism setup (all three parallel strategies combined), packing is fully supported because the framework's model context and data pipeline were designed to propagate sequence metadata through all parallelism dimensions. The key configuration constraint: `micro_batch_size` must be 1 across all DP ranks and PP stages.

```yaml
# NeMo 3D parallel + packing configuration
model:
  tensor_model_parallel_size: 4
  pipeline_model_parallel_size: 2
  # Data parallelism is implicit (num_gpus / tp / pp = 8 / 4 / 2 = 1 DP rank here)
  
  data:
    train_ds:
      packed_sequence: true
      micro_batch_size: 1                # mandatory regardless of TP/PP
      global_batch_size: 32              # recalibrated for avg seqs/pack
      file_names: [/path/packed_2048.npy]
      max_seq_length: 2048
```

### Gradient accumulation with packing

Gradient accumulation (accumulating gradients over $K$ micro-steps before an optimizer update) changes semantics slightly with packing. Without packing, one micro-step processes `mbs × seq_len` tokens. With packing, one micro-step processes one pack of (approximately) `pack_size × avg_seqs_per_pack` tokens. Gradient accumulation with packing accumulates over $K$ packs, each containing multiple sequences.

The effective batch size is then $K$ packs × avg_seqs_per_pack sequences per pack. If you had `gradient_accumulation_steps=4` without packing and switch to packing, you need to recalibrate `gradient_accumulation_steps` along with GBS:

```
# Without packing:
effective_batch = mbs × grad_accum_steps × num_gpus = 4 × 4 × 8 = 128 seqs

# With packing (avg_seqs_per_pack=4, mbs=1):
effective_batch = mbs × grad_accum_steps × num_gpus × avg_seqs_per_pack
128 = 1 × grad_accum_steps × 8 × 4
grad_accum_steps = 4  # same in this case
```

When in doubt, verify by logging `sequences_per_optimizer_step` from the trainer's training metrics. This should match the original effective batch size.

## 14. Choosing the Right Pack Size

### Pack size is a hyperparameter

The choice of `pack_size` (equivalently, `max_seq_length`) is a hyperparameter that affects throughput, memory, and training quality simultaneously. Unlike micro-batch size, which is limited by GPU memory, pack size has a more complex tradeoff space.

**Larger pack size**:
- More sequences fit per pack → higher bin utilization → better throughput
- Larger attention computation per step → more GPU memory pressure
- Longer sequences may encounter rope embedding extrapolation issues if training context is shorter than pack_size

**Smaller pack size**:
- Fewer sequences per pack → lower bin utilization → less throughput gain
- Smaller attention computation → lower memory pressure
- Sequences near pack_size get one per pack → packing is essentially no-op

The optimal pack size depends on your dataset's length distribution. A practical heuristic: set `pack_size` to the **95th percentile** of your sequence length distribution. This ensures that 95% of sequences fit without truncation, while sequences at the 95th percentile anchor each pack and allow shorter sequences to fill the remaining space.

```python
import json
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("your-model")

lengths = []
with open("training.jsonl") as f:
    for line in f:
        sample = json.loads(line)
        text = sample["input"] + sample["output"]
        toks = tokenizer(text)
        lengths.append(len(toks["input_ids"]))

p50 = int(np.percentile(lengths, 50))
p90 = int(np.percentile(lengths, 90))
p95 = int(np.percentile(lengths, 95))
p99 = int(np.percentile(lengths, 99))

# Round up to next power of 2 for memory alignment
def next_pow2(n):
    p = 1
    while p < n:
        p <<= 1
    return p

print(f"P50:  {p50} → pack_size candidate: {next_pow2(p50)}")
print(f"P90:  {p90} → pack_size candidate: {next_pow2(p90)}")
print(f"P95:  {p95} → pack_size candidate: {next_pow2(p95)}")
print(f"P99:  {p99} → pack_size candidate: {next_pow2(p99)}")

# Estimate avg seqs/pack for each candidate
for candidate in [512, 1024, 2048, 4096, 8192]:
    avg = np.mean(lengths)
    est_seqs_per_pack = candidate / avg
    est_speedup = min(est_seqs_per_pack, candidate / p95)  # bounded by long-tail
    print(f"pack_size={candidate}: ~{est_seqs_per_pack:.1f} seqs/pack, "
          f"~{100*(1 - avg/candidate):.0f}% padding eliminated")
```

### The multi-pack-size trick

NeMo's `prepare_packed_ft_dataset.py` accepts `pack_sizes=[2048,4096,8192]` and generates all three in a single preprocessing run. This allows you to experiment with different pack sizes without re-running the expensive tokenization pass.

A practical workflow for choosing pack size:

1. Run preprocessing with `pack_sizes=[1024,2048,4096,8192]`.
2. For each pack size, measure the average sequences per pack from the `.npy` metadata.
3. Run a 5% training data smoke test at each pack size; record throughput (tokens/second) and memory (peak GPU MB).
4. Choose the pack size that maximizes throughput without exceeding your GPU memory budget.
5. Verify that convergence at the chosen pack size matches the padded baseline on 10% of data.

Changing pack size changes the model's effective context length. If you later want to inference at a different context length than your training pack_size, you may need to apply RoPE scaling (NTK-aware or YaRN) to extrapolate the positional embeddings.

## 15. Monitoring Packing Efficiency During Training

### What to log

Standard training metrics (loss, learning rate, gradient norm) don't tell you whether packing is working correctly. Add these metrics to your training loop:

```python
# Metrics to log per optimizer step
metrics = {
    "train/tokens_per_second": total_tokens / step_time_seconds,
    "train/sequences_per_second": total_seqs / step_time_seconds,
    "train/avg_seqs_per_pack": total_seqs / total_packs,
    "train/pack_utilization": total_real_tokens / (total_packs * pack_size),
    "train/throughput_vs_padded_baseline": current_tps / baseline_tps,
}
```

**`avg_seqs_per_pack`** should match your pre-training estimate from the preprocessing metadata. If it's significantly lower (e.g., 2.1 actual vs 4.2 estimated), your packing algorithm is producing more fragmented packs than expected — possibly because a large fraction of sequences is longer than `pack_size` and was silently truncated.

**`pack_utilization`** is the fraction of pack slots occupied by real tokens (as opposed to residual padding after the last sequence in a pack). It should be in the range 92–99% for well-configured packing. Values below 85% suggest the pack size is too large relative to the sequence length distribution.

**`throughput_vs_padded_baseline`** is the headline metric. Compute this by running a short calibration run with standard padding at the start of training and recording baseline `tokens_per_second`. Then monitor the ratio throughout the packed run. If this ratio drops below 1.5× on a dataset with 50% padding ratio, something in the packing pipeline is underperforming.

### Detecting silent attention boundary violations

The hardest failure mode to detect is silent cross-sequence attention contamination — the case where `cu_seqlens` is being passed but ignored by the kernel. This produces training loss that looks normal but model behavior that is wrong.

One diagnostic: train for 100 steps on two datasets simultaneously — one "clean" (single sequences, no packing) and one "packed" with a specially constructed dataset where sequence $B$ contains a phrase that semantically conflicts with sequence $A$. After 100 steps, prompt the model with a prefix from sequence $B$ and check whether it hallucinates content from sequence $A$. With correct boundary enforcement, it should not. With boundary leakage, the model will occasionally hallucinate conflicting $A$ content when prompted with $B$'s prefix.

A simpler (but less precise) check: compare the KL divergence between the token probability distributions produced by the padded model and the packed model on a held-out set. With correct boundaries, the distributions should be essentially identical (KL < 0.01 nats). With boundary leakage, you will see elevated KL on examples where a contaminating sequence happened to be in the same pack.

## 16. Custom Dataset Integration

### Writing a packing-aware PyTorch Dataset

If you are not using NeMo or TRL, you need to implement the packed dataset class yourself. Here is a minimal PyTorch `Dataset` implementation that handles `cu_seqlens` and `position_ids` correctly:

```python
import torch
import numpy as np
from torch.utils.data import Dataset

class PackedSFTDataset(Dataset):
    """
    Loads a pre-packed .npy dataset and returns packed tensors with
    cu_seqlens and position_ids metadata for varlen FlashAttention.
    
    Each .npy entry is expected to have keys:
      'input_ids': flat array of token IDs for all sequences in the pack
      'cu_seqlens': cumulative sequence lengths [0, s1, s1+s2, ..., total]
      'loss_mask': 1 for output tokens, 0 for input tokens (optional)
    """
    def __init__(self, npy_path: str, pack_size: int):
        self.packs = np.load(npy_path, allow_pickle=True)
        self.pack_size = pack_size
    
    def __len__(self):
        return len(self.packs)
    
    def __getitem__(self, idx):
        pack = self.packs[idx]
        input_ids = torch.tensor(pack['input_ids'], dtype=torch.long)
        cu_seqlens = torch.tensor(pack['cu_seqlens'], dtype=torch.int32)
        
        # Construct position_ids: reset to 0 at each sub-sequence boundary
        position_ids = torch.zeros_like(input_ids)
        for i in range(len(cu_seqlens) - 1):
            start = cu_seqlens[i].item()
            end = cu_seqlens[i + 1].item()
            position_ids[start:end] = torch.arange(end - start)
        
        # Build loss mask: 1 for response tokens, 0 for instruction tokens
        if 'loss_mask' in pack:
            loss_mask = torch.tensor(pack['loss_mask'], dtype=torch.float32)
        else:
            loss_mask = torch.ones(len(input_ids), dtype=torch.float32)
        
        return {
            'input_ids': input_ids,
            'cu_seqlens': cu_seqlens,
            'position_ids': position_ids,
            'loss_mask': loss_mask,
        }


def packed_collate_fn(batch):
    """
    Collator for packed sequences. Since mbs=1 with packing, each batch
    is a single pack. We still add a batch dimension for PyTorch compatibility.
    """
    assert len(batch) == 1, "Packed training requires micro_batch_size=1"
    sample = batch[0]
    return {
        # Shape: (1, pack_size) — the '1' is the batch dimension
        'input_ids': sample['input_ids'].unsqueeze(0),
        # cu_seqlens stays 1D: (num_seqs + 1,)
        'cu_seqlens': sample['cu_seqlens'],
        # position_ids: (1, pack_size)
        'position_ids': sample['position_ids'].unsqueeze(0),
        # loss_mask: (1, pack_size)
        'loss_mask': sample['loss_mask'].unsqueeze(0),
    }
```

### Wiring the varlen attention into a custom model

If you are using a custom model architecture (not a HuggingFace model), you need to wire `flash_attn_varlen_func` into your attention module:

```python
from flash_attn import flash_attn_varlen_func
import torch
import torch.nn as nn
import math

class PackingAwareCausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
    
    def forward(
        self,
        x: torch.Tensor,          # (batch, seq_len, d_model) OR (1, pack_size, d_model)
        cu_seqlens: torch.Tensor,  # (num_seqs + 1,) int32; None for standard attention
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, L, D = x.shape
        
        qkv = self.qkv(x)  # (B, L, 3*D)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape to (B*L, n_heads, head_dim) for varlen API
        q = q.view(B * L, self.n_heads, self.head_dim)
        k = k.view(B * L, self.n_heads, self.head_dim)
        v = v.view(B * L, self.n_heads, self.head_dim)
        
        if cu_seqlens is not None:
            # Packed path: variable-length attention with per-sequence boundaries
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            attn_out = flash_attn_varlen_func(
                q, k, v,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                causal=True,
                softmax_scale=1.0 / math.sqrt(self.head_dim),
            )
        else:
            # Standard path: full causal attention (no packing)
            from flash_attn import flash_attn_func
            q = q.view(B, L, self.n_heads, self.head_dim)
            k = k.view(B, L, self.n_heads, self.head_dim)
            v = v.view(B, L, self.n_heads, self.head_dim)
            attn_out = flash_attn_func(q, k, v, causal=True)
            attn_out = attn_out.view(B * L, self.n_heads, self.head_dim)
        
        # Back to (B, L, D)
        out = attn_out.view(B, L, D)
        return self.out(out)
```

The key insight: the same model can run in both packed mode (with `cu_seqlens`) and standard mode (without), which simplifies evaluation runs where you want standard batching.

## 17. The Packing × Learning Rate Interaction

### Does packing require LR adjustment?

A question that comes up consistently: does switching to packing require changing the learning rate? The answer is no, with one important caveat.

With offline bin-packing (NeMo-style), you are training with the same number of optimizer steps as the padded baseline (assuming you correctly set GBS = original_GBS / avg_seqs_per_pack). Each optimizer step processes the same number of gradient-contributing tokens, just arranged differently in the batch. The gradient scale is unchanged.

With `DataCollatorWithFlattening` (HuggingFace-style), the number of optimizer steps is explicitly preserved — the collator concatenates within each mini-batch but doesn't change how many mini-batches there are. Again, no LR adjustment needed.

The caveat: if you use a warmup schedule measured in optimizer steps and the warmup count was calibrated for the padded run's step count, the same warmup works in the packed run (because step count is the same). If your warmup was calibrated for wall-clock time rather than steps, you need to recalibrate because packing is faster.

### The tokens-per-step perspective

A useful mental model: think of your learning rate as associated with a specific "tokens-per-update" budget. With padding, you process `mbs × max_seq_length × avg_padding_ratio` wasted tokens per update alongside the real ones. With packing, you process approximately zero wasted tokens. The real token count per update should be approximately the same (both equal `mbs × avg_real_length` without packing and `avg_seqs_per_pack × avg_real_length` with packing, when GBS is correctly recalibrated).

This tokens-per-update equivalence is why convergence is preserved — the model's gradient signal per update is essentially unchanged. The only difference is throughput: packing delivers the same gradient update in less wall-clock time.

## 18. Packing with PEFT: LoRA and QLoRA

### Why PEFT and packing are complementary

Parameter-Efficient Fine-Tuning methods like LoRA (Low-Rank Adaptation) and QLoRA (quantized LoRA) are the dominant approach for fine-tuning large models on resource-constrained hardware. LoRA freezes the base model weights and adds low-rank trainable adapters to the attention and feed-forward layers. This reduces trainable parameters by 100-1000× while preserving most fine-tuning quality.

Sequence packing and PEFT are complementary optimizations operating at different levels:
- **Packing** reduces wasted compute by eliminating padding tokens (throughput optimization)
- **LoRA/QLoRA** reduces memory pressure by shrinking the gradient computation and optimizer state (memory optimization)

Combining them gives you both: high throughput on real-token compute and low memory footprint from reduced parameter training. On a single A100 40GB, QLoRA on a 70B model without packing might train at 800 tokens/second with 65% padding waste. With QLoRA + packing, the same setup trains at 3,000–4,000 tokens/second on non-wasted tokens.

### NeMo PEFT + packing

NeMo supports both `p-tuning`, `adapter`, and `lora` PEFT methods with packed sequences. The configuration is nearly identical to full fine-tuning with packing — just add the PEFT specification alongside the packing configuration:

```yaml
model:
  peft:
    peft_scheme: "lora"
    lora_tuning:
      target_modules: ["attention_qkv", "attention_dense"]
      adapter_dim: 32
      adapter_dropout: 0.1
      
  data:
    train_ds:
      packed_sequence: true
      micro_batch_size: 1
      global_batch_size: 16    # recalibrated as always
      file_names: [/path/packed_2048.npy]
      max_seq_length: 2048
```

One important difference: with LoRA, only the adapter weights receive gradients. The base model's frozen weights still participate in the forward pass (including the packed attention computation), so `cu_seqlens` must still be passed correctly to the frozen attention layers. The packing infrastructure is base-model-agnostic in this regard.

### HuggingFace PEFT + DataCollatorWithFlattening

The TRL `SFTTrainer` with `DataCollatorWithFlattening` works directly with `peft` library LoRA adapters:

```python
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig
from trl.data_utils import DataCollatorWithFlattening
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# QLoRA: quantize base model, train only LoRA adapters
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    quantization_config=bnb_config,
    attn_implementation="flash_attention_2",  # required for packing
    torch_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable params: 13,631,488 || all params: 8,043,421,696 || trainable%: 0.1695

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("json", data_files="training.jsonl")["train"]

config = SFTConfig(
    output_dir="./qlora_packed_output",
    per_device_train_batch_size=1,   # with packing, this is correct at 1
    gradient_accumulation_steps=8,   # accumulate 8 packs before update
    learning_rate=2e-4,
    num_train_epochs=2,
    max_seq_length=2048,
    logging_steps=10,
    bf16=True,
    optim="paged_adamw_32bit",       # paged optimizer for QLoRA memory management
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    data_collator=DataCollatorWithFlattening(tokenizer=tokenizer),
    args=config,
)
trainer.train()
```

### Memory footprint with PEFT + packing

The memory savings from QLoRA + packing stack multiplicatively:

| Configuration | VRAM (approx, 8B model) | Training tokens/sec |
|---|---|---|
| Full FT, padding, mbs=4 | 75 GB (requires A100 80GB) | 1,200 |
| Full FT, packing, mbs=1 | 70 GB | 4,800 |
| LoRA, padding, mbs=4 | 28 GB | 1,400 |
| LoRA, packing, mbs=1 | 25 GB | 5,200 |
| QLoRA (4-bit), padding, mbs=4 | 14 GB | 900 |
| QLoRA (4-bit), packing, mbs=1 | 12 GB | 3,100 |

QLoRA + packing on a single RTX 4090 (24GB) is a practical configuration for fine-tuning 8B models with 50%+ padding datasets. The token throughput approaches what you would get with full fine-tuning on a 2-GPU A100 setup — a significant democratization of fine-tuning capacity.

## Comparing NeMo vs HuggingFace Packing

| Dimension | NeMo offline packing | HF DataCollatorWithFlattening |
|---|---|---|
| Preprocessing | Required (`.npy` files) | None |
| Packing algorithm | FFD or FFS | No bin-packing — concatenate as-is |
| Optimizer steps | Fewer (GBS must be adjusted) | Same as padded baseline |
| Throughput gain | High (5–10× on skewed data) | Moderate (1.5–3×) |
| Convergence risk | Low (FFS) / Moderate (FFD) | Negligible |
| Flash Attention req. | Strongly recommended | Required |
| mbs constraint | Must be 1 | Standard mbs works |
| GBS adjustment | Required | Not required |
| Debug complexity | Higher | Lower |
| Best for | Production training, known datasets | Prototyping, online datasets, safety-first |

## Implementation Checklist

### Pre-packing sanity checks

Before investing time in the packing pipeline, answer four questions:

1. **What is your dataset's padding ratio?** Run the analysis script from Section 1. If the ratio is below 15%, stop here — the complexity isn't worth it.
2. **What framework are you using?** NeMo, TRL/HuggingFace, or custom? The implementation path differs significantly. NeMo's offline pipeline gives maximum throughput; TRL's online approach gives minimum complexity.
3. **Do you have FlashAttention 2?** Run `python -c "import flash_attn; print(flash_attn.__version__)"`. Without FA2, you cannot safely implement variable-length attention. The fallback (explicit $O(L^2)$ masking) is computationally equivalent to paying for the padding you're trying to eliminate.
4. **Can you afford a convergence smoke test?** Packing changes batch composition. A 10% training-data run with validation loss comparison takes 10–20% of your total training time but protects you from shipping a model trained with subtly incorrect batch semantics.

If all four answers are satisfactory, proceed to the checklist below.

Before your first packed training run, verify every item:

```bash
# 1. Measure padding ratio on your dataset
python -c "
import json
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('your-model')
lengths = [len(tok(json.loads(l)['input'] + json.loads(l)['output'])['input_ids'])
           for l in open('training.jsonl')]
import statistics
avg = statistics.mean(lengths)
max_len = 2048
print(f'Padding ratio: {1-avg/max_len:.1%}')
print(f'Avg sequences per pack: {max_len/avg:.1f}')
"

# 2. Run the NeMo preprocessing script
python scripts/nlp_language_modeling/prepare_packed_ft_dataset.py \
   model.data.train_ds.file_names=[/path/training.jsonl] \
   model.data.train_ds.max_seq_length=2048 \
   +tokenizer_path=/path/tokenizer.model \
   +output_dir=/path/packed/ \
   +pack_sizes=[2048] \
   +packing_algorithm=first_fit_shuffle \
   +seed=0

# 3. Verify the output
python -c "
import numpy as np
packed = np.load('/path/packed/packed_2048.npy', allow_pickle=True)
seqs_per_pack = [len(p['cu_seqlens']) - 1 for p in packed]
import statistics
print(f'Packs: {len(packed)}')
print(f'Avg seqs/pack: {statistics.mean(seqs_per_pack):.2f}')
print(f'Recommended GBS: original_gbs / {statistics.mean(seqs_per_pack):.1f}')
"

# 4. Verify Flash Attention 2 is installed
python -c "import flash_attn; print(f'FA2 version: {flash_attn.__version__}')"
```

Training config final checklist (NeMo YAML):
- [ ] `packed_sequence: true`
- [ ] `file_names` points to `.npy` file, not `.jsonl`
- [ ] `micro_batch_size: 1`
- [ ] `global_batch_size` recalibrated by dividing by avg sequences per pack
- [ ] `max_seq_length` equals `pack_size`
- [ ] Convergence smoke test on 10% of data: packed validation loss ≤ padded + 2%

## Further Reading

- [NVIDIA NeMo Sequence Packing documentation](https://docs.nvidia.com/nemo-framework/user-guide/24.09/nemotoolkit/features/optimizations/sequence_packing.html) — the reference implementation this article is based on
- "Enhancing Training Efficiency Using Packing with Flash Attention" (arXiv 2407.09105) — IBM Research + HuggingFace paper introducing FA2 document masking for packing
- "Threshold Filtering Packing for Supervised Fine-Tuning" (arXiv 2408.09327) — related-sequence packing for knowledge-dense datasets
- [HuggingFace blog: Improving training efficiency through packing with flash attention](https://huggingface.co/blog/packing-with-FA2)
- [DeepSpeed ZeRO and 3D Parallelism](/blog/machine-learning/open-source-library/deepspeed-zero-3d-parallelism-deep-dive) — complementary memory optimization for large-model training
- [KV Cache in Large Language Models](/blog/machine-learning/large-language-model/kv-cache) — understanding memory efficiency in the inference analogue
- [Speeding up neural network training by optimizing CPU-to-GPU data transfer](/blog/machine-learning/training-techniques/speeding-up-neural-network-training-4x-by-optimizing-cpu-to-gpu-data-transfer) — another training throughput optimization that stacks with packing

---

Sequence packing is one of the rare optimizations that costs almost nothing and, on the right dataset, delivers multiplicative throughput gains. The entire engineering surface — offline preprocessing with `prepare_packed_ft_dataset.py`, a few config changes, and a convergence smoke test — can be done in two days. The gains it unlocks compound on every subsequent training run. For teams fine-tuning LLMs on instruction or chat datasets, the question is not whether to use packing; it is why it took this long to turn it on.

The three things to remember: measure your padding ratio first, use `first_fit_shuffle` unless you have a specific reason for FFD, and always set `micro_batch_size=1` — that last one has burned more teams than any other configuration mistake in packing setups.

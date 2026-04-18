---
title: "vLLM: A Complete Guide to High-Throughput LLM Serving"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "Large Language Model"
tags:
  [
    "llm",
    "vllm",
    "inference",
    "serving",
    "paged-attention",
    "optimization",
    "deep-learning",
    "kv-cache",
    "continuous-batching",
  ]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "vLLM is the most widely adopted LLM serving framework, built around PagedAttention — the innovation that solved the KV cache memory fragmentation problem. This guide covers its architecture, every major optimization, production deployment, and interview-ready depth on LLM serving systems."
---

## What Is vLLM?

vLLM (Virtual Large Language Model) is an open-source LLM serving framework developed at UC Berkeley that achieves high-throughput inference through **PagedAttention** — a memory management technique inspired by operating system virtual memory. Since its release in 2023, vLLM has become the most widely deployed LLM serving engine in production, powering inference at companies from startups to major cloud providers.

The central problem vLLM solves: **KV cache memory is wasted in traditional serving systems.** Before vLLM, LLM serving frameworks pre-allocated contiguous memory blocks for each request's KV cache, sized to the maximum possible sequence length. Since most sequences are much shorter than the maximum, 60-80% of allocated GPU memory sat unused. vLLM's PagedAttention eliminates this waste by managing KV cache in small, dynamically allocated pages — like how an operating system manages RAM.

```
Before vLLM (static allocation):
  Request 1: [██████░░░░░░░░░░░░░░]  Actual: 300 tokens, Allocated: 2048
  Request 2: [████░░░░░░░░░░░░░░░░]  Actual: 200 tokens, Allocated: 2048
  Request 3: [██████████░░░░░░░░░░]  Actual: 500 tokens, Allocated: 2048
  
  GPU memory: 80 GB → fits ~3 concurrent requests (wasteful!)
  Memory utilization: ~16%

After vLLM (paged allocation):
  Request 1: [██████]               Actual: 300 tokens, Allocated: 304 (19 pages)
  Request 2: [████]                 Actual: 200 tokens, Allocated: 208 (13 pages)
  Request 3: [██████████]           Actual: 500 tokens, Allocated: 512 (32 pages)
  
  GPU memory: 80 GB → fits ~30+ concurrent requests
  Memory utilization: >96%
```

## The KV Cache Memory Problem (In Depth)

To understand why vLLM matters, let's quantify exactly how bad the memory problem is without it.

### The Memory Math

For a model with $L$ layers, $n_\text{kv}$ KV heads, and head dimension $d$, the KV cache per token is:

$$\text{KV per token} = 2 \times L \times n_\text{kv} \times d \times \text{bytes}$$

For Llama 3.1 70B ($L=80$, $n_\text{kv}=8$ GQA heads, $d=128$, FP16):

$$\text{KV per token} = 2 \times 80 \times 8 \times 128 \times 2 = 327,680 \text{ bytes} = 320 \text{ KB}$$

For a max sequence length of 8192:

$$\text{KV per sequence} = 320 \text{ KB} \times 8192 = 2.56 \text{ GB}$$

### Three Types of Memory Waste

**1. Internal fragmentation**: Pre-allocating max_seq_len tokens when the actual sequence is much shorter. If average output is 200 tokens but max is 8192, you waste 97.6% of each allocation.

**2. External fragmentation**: As requests start and finish at different times, freed memory leaves gaps that are too small for new allocations but too large to ignore. Over time, the memory becomes a patchwork of used and free regions.

**3. Reservation waste**: Memory reserved for potential future growth (the sequence hasn't reached max_len yet) can't be used by other requests, even though it's currently empty.

```
External fragmentation example:

After several requests complete:
  [████][    free    ][██████][  free ][████████][ free ][██]
  
  Total free memory: 45 GB
  Largest contiguous block: 12 GB
  New request needs: 15 GB contiguous
  → Request REJECTED even though plenty of total memory exists!
```

PagedAttention eliminates all three types of waste.

## PagedAttention: The Core Innovation

### The OS Virtual Memory Analogy

PagedAttention directly mirrors how operating systems manage physical RAM:

| OS Concept | vLLM Equivalent |
|-----------|-----------------|
| Virtual address space | Logical KV cache (sequence positions) |
| Physical pages (4KB frames) | KV cache blocks (e.g., 16 tokens/block) |
| Page table | Block table |
| Page allocation | On-demand KV block allocation |
| Copy-on-write (COW) | Shared prefix / beam search optimization |
| Swap to disk | KV cache offloading to CPU RAM |

### How It Works

**Step 1: Divide KV cache into fixed-size blocks**

Instead of allocating a contiguous buffer for each sequence, the KV cache is divided into small **blocks** of fixed size (default: 16 tokens per block). Each block stores the K and V tensors for 16 tokens across all layers and KV heads.

```
One KV block (16 tokens):
  Layer 0: K[0:16] shape=(16, d), V[0:16] shape=(16, d)
  Layer 1: K[0:16] shape=(16, d), V[0:16] shape=(16, d)
  ...
  Layer L-1: K[0:16] shape=(16, d), V[0:16] shape=(16, d)
  
  Block size (Llama 3.1 70B, FP16): 16 × 320 KB/token = 5 MB per block
```

**Step 2: Maintain a block table per sequence**

Each sequence has a **block table** that maps logical block indices to physical block locations in GPU memory — just like a page table maps virtual addresses to physical addresses.

```
Sequence A (45 tokens):
  Block table: [Block 7] → [Block 23] → [Block 41]
  Block 7:  tokens 0-15   (full)
  Block 23: tokens 16-31  (full)
  Block 41: tokens 32-44  (partially filled, 13/16 slots used)

Sequence B (22 tokens):
  Block table: [Block 3] → [Block 15]
  Block 3:  tokens 0-15   (full)
  Block 15: tokens 16-21  (partially filled, 6/16 slots used)
```

**Step 3: Allocate blocks on demand**

When a new token is generated and the current last block is full, a new block is allocated from a global free list. No pre-allocation of max_seq_len blocks needed.

```python
class BlockManager:
    """Manages KV cache block allocation."""
    
    def __init__(self, num_total_blocks, block_size=16):
        self.block_size = block_size
        self.free_blocks = list(range(num_total_blocks))
        self.block_tables = {}  # sequence_id -> list of block indices
    
    def allocate_sequence(self, seq_id, num_tokens):
        """Allocate blocks for a new sequence."""
        num_blocks = (num_tokens + self.block_size - 1) // self.block_size
        if len(self.free_blocks) < num_blocks:
            raise OutOfMemoryError("Not enough KV cache blocks")
        
        blocks = [self.free_blocks.pop() for _ in range(num_blocks)]
        self.block_tables[seq_id] = blocks
        return blocks
    
    def append_token(self, seq_id):
        """Called when a new token is generated for a sequence."""
        blocks = self.block_tables[seq_id]
        current_block = blocks[-1]
        tokens_in_last_block = self._tokens_in_block(seq_id, len(blocks) - 1)
        
        if tokens_in_last_block >= self.block_size:
            # Last block is full — allocate a new one
            if len(self.free_blocks) == 0:
                raise OutOfMemoryError("KV cache full")
            new_block = self.free_blocks.pop()
            blocks.append(new_block)
    
    def free_sequence(self, seq_id):
        """Free all blocks when a sequence completes."""
        blocks = self.block_tables.pop(seq_id)
        self.free_blocks.extend(blocks)
```

**Step 4: Modified attention kernel**

The attention computation must be modified to handle non-contiguous KV blocks. Instead of accessing a contiguous K tensor of shape `(seq_len, d)`, the kernel follows the block table to gather K/V from the correct physical locations:

```python
# Pseudocode: attention with paged KV cache
def paged_attention(query, block_table, kv_cache, block_size):
    """
    Compute attention with non-contiguous KV blocks.
    
    query: (1, num_heads, head_dim) — single new token query
    block_table: list of physical block indices for this sequence
    kv_cache: (num_blocks, 2, num_heads, block_size, head_dim)
    """
    all_keys = []
    all_values = []
    
    for block_idx in block_table:
        k_block = kv_cache[block_idx, 0]  # (num_heads, block_size, head_dim)
        v_block = kv_cache[block_idx, 1]
        all_keys.append(k_block)
        all_values.append(v_block)
    
    keys = torch.cat(all_keys, dim=1)    # (num_heads, seq_len, head_dim)
    values = torch.cat(all_values, dim=1)
    
    # Standard attention
    scores = torch.matmul(query, keys.transpose(-1, -2)) / sqrt(head_dim)
    attn = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn, values)
    
    return output
```

In practice, the gathering is fused into the attention CUDA kernel to avoid materializing the full K/V tensors in contiguous memory — this is the key engineering contribution of the PagedAttention kernel.

### Copy-on-Write for Shared Prefixes

When multiple sequences share a prefix (beam search, parallel sampling, or common system prompts), they can share the same physical KV blocks via **copy-on-write**:

```
Beam search with 3 beams, shared prefix "The capital of":

Beam 1: [Block A] → [Block B] → [Block C₁]  "France is Paris"
Beam 2: [Block A] → [Block B] → [Block C₂]  "Japan is Tokyo"
Beam 3: [Block A] → [Block B] → [Block C₃]  "Germany is Berlin"

Blocks A and B are SHARED (reference count = 3)
Blocks C₁, C₂, C₃ are unique per beam

Without COW: 9 blocks needed (3 × 3)
With COW:    5 blocks needed (2 shared + 3 unique) = 44% memory savings
```

The block table tracks reference counts. When a shared block needs to be modified (because beams diverge), it's copied first — hence "copy-on-write."

## Continuous Batching

Before vLLM, most serving systems used **static batching**: collect $B$ requests, process them as a batch, wait for all to finish, then process the next batch. This is wasteful because short sequences finish early but must wait for the longest sequence in the batch.

**Continuous batching** (also called iteration-level batching or inflight batching) processes requests at the token level rather than the request level:

```
Static batching:
  Batch 1: [R1: 500 tokens][R2: 100 tokens][R3: 300 tokens]
  R2 finishes after 100 steps but GPU still processes R1's remaining 400 tokens
  → R2 wastes 400 decode slots; new requests wait in queue
  
Continuous batching:
  Step 0:   [R1][R2][R3]
  Step 100: [R1][R3][R4 ← new!]    R2 finished, R4 enters immediately
  Step 200: [R1][R4][R5 ← new!]    R3 finished at step 200
  Step 300: [R4][R5][R6 ← new!]    R1 finished at step 300
  → No wasted GPU cycles. New requests enter as soon as a slot opens.
```

### Implementation in vLLM

vLLM's scheduler runs a tight loop:

```python
# Simplified vLLM scheduler loop
while True:
    # 1. Check for finished sequences (reached EOS or max_tokens)
    finished = [seq for seq in running if seq.is_finished()]
    for seq in finished:
        running.remove(seq)
        free_kv_blocks(seq)
        return_result(seq)
    
    # 2. Schedule new requests from the waiting queue
    while waiting_queue and has_enough_kv_blocks():
        new_req = waiting_queue.pop(0)
        allocate_kv_blocks(new_req)
        running.append(new_req)
    
    # 3. Run one forward pass for all running sequences
    #    - Prefill for newly added sequences
    #    - Decode for continuing sequences
    outputs = model.forward(running)
    
    # 4. Update sequences with new tokens
    for seq, output in zip(running, outputs):
        seq.append_token(output.token)
        append_kv_to_cache(seq, output.kv)
```

### Prefill and Decode Scheduling

vLLM handles two types of work in each iteration:

**Prefill**: New requests need their entire prompt processed. This is compute-heavy (many tokens in one pass). Prefill requests create a large matrix multiplication that utilizes the GPU's compute capacity.

**Decode**: Running requests generate one token each. This is memory-bandwidth-heavy (loading KV cache for one token). Many decode requests together form a batched matrix multiplication that better utilizes the GPU.

vLLM interleaves prefill and decode operations in the same batch. The scheduler prioritizes decode (to avoid latency spikes for ongoing generations) but slots in prefill work to keep the pipeline full.

### Chunked Prefill

Long prompts (e.g., 10K+ tokens) can monopolize the GPU for seconds during prefill, blocking decode steps for all other sequences. **Chunked prefill** splits long prompts into smaller chunks:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    enable_chunked_prefill=True,
    max_num_batched_tokens=2048,  # max tokens per iteration (prefill + decode)
)
```

Without chunked prefill, a 10K-token prompt processed all at once blocks the GPU for ~2 seconds. With `max_num_batched_tokens=2048`, the prompt is split into 5 chunks of 2048 tokens, each interleaved with decode steps for other sequences.

## Prefix Caching

vLLM supports **automatic prefix caching** to avoid redundant prefill computation for shared prefixes (system prompts, few-shot examples, shared document context).

### How It Works

vLLM uses a **hash-based** approach. Each KV block's content is determined by the token IDs it contains plus the tokens in all preceding blocks. vLLM computes a hash of this content and stores it in a global hash table:

```
Request 1: "System prompt tokens... | User: What is AI?"
  Block 0: hash(tokens[0:16]) = 0xABCD → compute and cache
  Block 1: hash(tokens[0:32]) = 0x1234 → compute and cache
  ...
  Block 93: hash(tokens[0:1500]) = 0x5678 → compute and cache (end of system prompt)
  Block 94: hash(tokens[0:1508]) = 0x9999 → compute (user query, unique)

Request 2: "System prompt tokens... | User: Explain gravity"
  Block 0: hash(tokens[0:16]) = 0xABCD → CACHE HIT! Reuse block.
  Block 1: hash(tokens[0:32]) = 0x1234 → CACHE HIT!
  ...
  Block 93: hash(tokens[0:1500]) = 0x5678 → CACHE HIT! (entire system prompt cached)
  Block 94: hash(tokens[0:1512]) = 0xBBBB → compute (different user query)
```

```python
# Enable prefix caching in vLLM
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    enable_prefix_caching=True,  # opt-in (not default)
)
```

### Limitations Compared to SGLang's RadixAttention

- **Hash computation overhead**: Each block requires hashing, which adds CPU overhead
- **Opt-in**: Not enabled by default; must be explicitly configured
- **Flat structure**: Hash table doesn't capture hierarchical prefix relationships (system prompt → few-shot → user query) — each block is matched independently
- **No LPM scheduling**: vLLM doesn't prioritize requests by prefix match length (SGLang does with its LPM scheduler)

## Quantization Support

vLLM supports multiple quantization methods for reducing model memory and improving throughput:

### Weight Quantization

```python
# FP8 quantization (best for H100/H200)
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    quantization="fp8",
    tensor_parallel_size=2,
)

# AWQ 4-bit quantization
llm = LLM(
    model="TheBloke/Llama-3-70B-AWQ",
    quantization="awq",
    tensor_parallel_size=2,
)

# GPTQ 4-bit quantization
llm = LLM(
    model="TheBloke/Llama-3-70B-GPTQ",
    quantization="gptq",
    tensor_parallel_size=2,
)
```

### KV Cache Quantization

```python
# FP8 KV cache — halves KV cache memory with minimal quality loss
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    kv_cache_dtype="fp8_e5m2",  # FP8 for KV cache
)
```

FP8 KV cache is one of the highest-impact optimizations available: it doubles the number of concurrent sequences (or doubles max context length) with negligible quality degradation.

## Speculative Decoding

vLLM supports multiple speculative decoding strategies:

```python
# Draft model speculation
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.1-8B-Instruct",
    num_speculative_tokens=5,
)

# N-gram speculation (no draft model needed)
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="[ngram]",
    num_speculative_tokens=5,
    ngram_prompt_lookup_max=4,
)

# Medusa heads
llm = LLM(
    model="path/to/medusa-model",
    speculative_model="[medusa]",
    num_speculative_tokens=5,
)
```

Speculative decoding is most effective at low batch sizes (batch 1-4) where decode is heavily memory-bandwidth-bound. At high batch sizes, the GPU is already well-utilized and speculation adds overhead.

## Multi-GPU Parallelism

### Tensor Parallelism (TP)

Splits each layer's weight matrices across GPUs. Every GPU participates in every forward pass via all-reduce communication.

```python
# 4-way tensor parallelism
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
)
```

**Best for**: Latency-sensitive serving (all GPUs work on every token, minimizing per-token time). Required when a single GPU can't hold the model.

### Pipeline Parallelism (PP)

Splits layers across GPUs. GPU 0 runs layers 0-19, GPU 1 runs layers 20-39, etc. Sequences flow through the pipeline.

```python
# 4-way pipeline parallelism
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    pipeline_parallel_size=4,
)
```

**Best for**: Throughput-oriented serving where multiple micro-batches can fill the pipeline. Less inter-GPU communication than TP, but higher per-token latency (tokens must traverse all pipeline stages).

### Combined TP + PP

```python
# 2-way TP × 4-way PP = 8 GPUs
llm = LLM(
    model="meta-llama/Llama-3.1-405B-Instruct",
    tensor_parallel_size=2,
    pipeline_parallel_size=4,
)
```

### Choosing the Right Strategy

```
Model fits on 1 GPU?
  ├── Yes → No parallelism needed (fastest)
  └── No → How many GPUs needed?
       ├── 2-4 GPUs → Use TP (lowest latency)
       ├── 4-8 GPUs → TP=4 + PP=2 (or TP=8 if on NVLink)
       └── 8+ GPUs → Combine TP+PP, benchmark for optimal split

Key factors:
  - TP requires fast interconnect (NVLink best, PCIe slow)
  - PP has pipeline bubbles (reduced GPU utilization)
  - TP=8 on NVLink often beats TP=4 × PP=2
```

## Memory Management: Preemption and Swapping

When KV cache memory is exhausted, vLLM has two strategies:

### Preemption (Recomputation)

Pause a running sequence, free its KV blocks, and recompute its KV cache from scratch when resources are available:

```
Memory full → preempt lowest-priority sequence:
  1. Save the sequence's generated tokens so far
  2. Free all its KV cache blocks
  3. Move sequence back to waiting queue
  4. When memory is available, re-prefill from the prompt + generated tokens
```

**Pros**: No CPU memory needed, simple implementation
**Cons**: Wasted compute — recomputing the prefill is expensive for long sequences

### Swapping (CPU Offload)

Copy the preempted sequence's KV cache to CPU RAM, and restore it when resources are available:

```
Memory full → swap to CPU:
  1. Copy KV cache blocks from GPU → CPU RAM (PCIe transfer)
  2. Free GPU blocks
  3. When memory is available, copy KV cache from CPU → GPU
  4. Resume decoding (no recomputation needed)
```

**Pros**: No wasted compute
**Cons**: PCIe transfers add latency; requires enough CPU RAM to hold swapped caches

```python
# Configure swap space (CPU RAM for KV cache offloading)
llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    swap_space=32,  # 32 GB of CPU RAM for KV cache swap
)
```

## Production Deployment

### Server Launch

```bash
# Basic serving
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --port 8000

# Production-optimized
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --kv-cache-dtype fp8_e5m2 \
    --enable-prefix-caching \
    --enable-chunked-prefill \
    --max-num-batched-tokens 4096 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --port 8000
```

### OpenAI-Compatible API

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="none")

# Chat completion
response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-70B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0.7,
    max_tokens=256,
    stream=True,
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tensor-parallel-size` | 1 | Number of GPUs for tensor parallelism |
| `--pipeline-parallel-size` | 1 | Number of GPUs for pipeline parallelism |
| `--gpu-memory-utilization` | 0.90 | Fraction of GPU memory for model + KV cache |
| `--max-model-len` | Model default | Override maximum context length |
| `--max-num-batched-tokens` | Auto | Max tokens per iteration (controls chunked prefill) |
| `--max-num-seqs` | 256 | Max concurrent sequences |
| `--kv-cache-dtype` | auto | KV cache precision: `auto`, `fp8_e5m2` |
| `--quantization` | None | Weight quantization: `fp8`, `awq`, `gptq`, `squeezellm` |
| `--enable-prefix-caching` | false | Enable hash-based prefix caching |
| `--enable-chunked-prefill` | false | Enable chunked prefill for long prompts |
| `--swap-space` | 4 | CPU swap space in GB for preempted KV cache |
| `--block-size` | 16 | Tokens per KV cache block |
| `--enforce-eager` | false | Disable CUDA graphs (for debugging) |

### CUDA Graphs

vLLM uses **CUDA graphs** to capture and replay GPU kernel sequences, eliminating CPU-side kernel launch overhead. During decode, the same sequence of CUDA kernels runs every step — CUDA graphs capture this pattern once and replay it with minimal CPU involvement:

```
Without CUDA graphs:
  CPU: launch_kernel_1 → launch_kernel_2 → ... → launch_kernel_50 (each launch has overhead)
  Time dominated by CPU kernel launch latency at small batch sizes

With CUDA graphs:
  CPU: replay_captured_graph (single call, all 50 kernels pre-recorded)
  GPU executes the entire graph without waiting for CPU between kernels
```

This is particularly impactful at small batch sizes where CPU overhead is significant relative to GPU compute time. vLLM captures CUDA graphs for different batch sizes and selects the appropriate one at runtime.

### Monitoring

```python
# Server metrics endpoint
curl http://localhost:8000/metrics

# Key Prometheus metrics:
# vllm:num_requests_running         — current batch size
# vllm:num_requests_waiting         — queue depth (should be low)
# vllm:gpu_cache_usage_perc         — KV cache utilization
# vllm:cpu_cache_usage_perc         — CPU swap utilization
# vllm:avg_prompt_throughput_toks   — prefill throughput
# vllm:avg_generation_throughput_toks — decode throughput
# vllm:time_to_first_token_seconds  — TTFT distribution
# vllm:time_per_output_token_seconds — TPOT distribution
# vllm:num_preemptions_total        — preemption count (should be rare)
```

## Request Lifecycle: End-to-End Trace

```
1. REQUEST ARRIVES
   POST /v1/chat/completions
   Body: {"messages": [...], "max_tokens": 256, "temperature": 0.7}
   │
2. PREPROCESSING
   ├── Tokenize using the model's tokenizer
   ├── Apply chat template (add <|begin_of_text|>, role tokens, etc.)
   ├── Create a SequenceGroup (handles beam search / parallel sampling)
   └── Add to waiting queue
   │
3. SCHEDULING (runs every iteration)
   ├── Check for finished sequences → free their KV blocks
   ├── Check for preempted sequences → return to waiting or swap back
   ├── Schedule waiting requests that fit in available KV blocks:
   │   ├── Prefix caching check: look up block hashes for cache hits
   │   ├── Allocate new blocks for cache misses
   │   └── Add to running batch
   └── Build the execution batch (mix of prefill + decode sequences)
   │
4. MODEL EXECUTION
   ├── Prefill sequences: process all prompt tokens in one pass
   │   ├── Compute Q, K, V for all prompt tokens
   │   ├── Store K, V in allocated KV cache blocks
   │   └── Produce logits for the last position
   ├── Decode sequences: process one new token per sequence
   │   ├── Compute Q, K, V for the single new token
   │   ├── Append K, V to the last KV cache block (allocate new block if full)
   │   └── Attend to all cached K, V via PagedAttention kernel
   └── Combined via FlashAttention/FlashInfer kernels
   │
5. SAMPLING
   ├── Apply temperature scaling
   ├── Apply top-p / top-k filtering
   ├── Sample next token (or argmax for greedy)
   └── Check stopping conditions (EOS token, max_tokens, stop strings)
   │
6. POSTPROCESSING
   ├── Detokenize new tokens
   ├── Stream via SSE (if streaming enabled)
   └── When complete: return full response, free KV blocks
```

## Offline Batch Inference

Beyond serving, vLLM excels at **offline batch processing** — processing large datasets of prompts without a server:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
)

# Process thousands of prompts efficiently
prompts = ["Summarize: " + doc for doc in documents]  # 10,000 prompts

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=512,
)

# vLLM automatically batches, manages memory, and maximizes throughput
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    generated_text = output.outputs[0].text
```

Offline batch inference achieves higher throughput than online serving because:
- No latency constraints — can use maximum batch sizes
- All prompts known in advance — optimal scheduling and prefix sharing
- No streaming overhead

## Advanced Features

### Structured Output (Guided Decoding)

```python
from vllm import SamplingParams

# JSON schema constraint
params = SamplingParams(
    guided_decoding={"json_schema": {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }}
)

# Regex constraint
params = SamplingParams(
    guided_decoding={"regex": r"\d{3}-\d{3}-\d{4}"}  # phone number format
)
```

vLLM integrates with the **Outlines** library for guided decoding. Unlike SGLang's native FSM with jump-forward optimization, vLLM's approach applies logit masks at each step without skipping deterministic tokens.

### LoRA Serving

vLLM can serve multiple LoRA adapters simultaneously on the same base model:

```python
from vllm import LLM
from vllm.lora.request import LoRARequest

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_lora=True,
    max_loras=4,           # max concurrent LoRA adapters
    max_lora_rank=32,
)

# Serve different adapters per request
output1 = llm.generate(
    "Write a poem about the sea",
    lora_request=LoRARequest("poetry-adapter", 1, "path/to/poetry-lora"),
)

output2 = llm.generate(
    "Fix this code: def foo():",
    lora_request=LoRARequest("code-adapter", 2, "path/to/code-lora"),
)
```

This enables serving many specialized models (one per customer, one per task) with the memory cost of a single base model plus small LoRA weights.

### Multi-Modal Models

```python
from vllm import LLM

llm = LLM(
    model="llava-hf/llava-v1.6-mistral-7b-hf",
    max_model_len=4096,
)

# Image + text input
from vllm import ImagePixelData
outputs = llm.generate({
    "prompt": "<image>\nDescribe this image in detail.",
    "multi_modal_data": {"image": ImagePixelData(image)},
})
```

## Interview Questions and Answers

### Q: What is PagedAttention and why was it a breakthrough?

PagedAttention manages KV cache in small, fixed-size blocks (like OS memory pages) instead of pre-allocating contiguous buffers sized to the maximum sequence length. Before PagedAttention, serving systems wasted 60-80% of GPU memory on internal and external fragmentation — sequences much shorter than the maximum waste their pre-allocated space, and freed memory becomes fragmented into unusable gaps.

PagedAttention solves all three types of waste: **internal fragmentation** (only the last block is partially filled, wasting at most `block_size - 1` tokens), **external fragmentation** (any free block can be used by any sequence — no contiguous requirement), and **reservation waste** (blocks allocated on demand as sequences grow). It also enables **copy-on-write** sharing for beam search and common prefixes, avoiding duplicating KV cache for shared prefixes.

The result: vLLM achieves >96% KV cache memory utilization (vs ~20% for static allocation), enabling 3-5x more concurrent sequences on the same hardware. This directly translated to 2-4x higher serving throughput when vLLM was released.

### Q: How does the PagedAttention kernel work differently from standard attention?

Standard attention computes $\text{softmax}(QK^T/\sqrt{d}) \cdot V$ with K and V stored in contiguous memory of shape `(seq_len, head_dim)`. PagedAttention modifies this to handle non-contiguous KV blocks.

The kernel receives a **block table** (mapping logical position → physical block index) and iterates over blocks rather than contiguous memory. For each block, it loads the K/V data, computes partial attention scores, and accumulates the weighted values. The key engineering challenge is doing this efficiently on GPU — the block table adds an extra level of indirection that can hurt memory access patterns.

The vLLM kernel uses several optimizations: (1) each CUDA thread block processes one query against all blocks (parallelism over queries), (2) block-level softmax is computed using the online softmax trick (avoiding materializing the full attention matrix), (3) K/V loads are coalesced within each block (since blocks are internally contiguous).

### Q: Explain continuous batching. What are its advantages over static batching?

**Static batching**: Collect $B$ requests, process them together, wait for all to finish, then start the next batch. If one sequence generates 500 tokens and another generates 50, the short sequence's GPU slot sits idle for 450 steps.

**Continuous batching**: Process at the iteration (token) level. After each decode step, check if any sequence has finished. If so, immediately add a new request from the queue to fill the slot. No GPU cycles are wasted waiting for the slowest sequence.

Advantages:
- **Higher throughput**: GPU slots never sit idle waiting for long sequences
- **Lower latency**: New requests don't wait for an entire batch to complete
- **Better utilization**: The batch size adapts dynamically to the workload
- **Natural fit for streaming**: Each token is available immediately

The throughput improvement depends on the variance in output lengths. For workloads with highly variable lengths (chatbots), continuous batching can provide 2-3x higher throughput. For uniform-length workloads (fixed-length completions), the improvement is smaller.

### Q: How does vLLM handle memory pressure? Compare preemption strategies.

When KV cache memory is exhausted and new requests are waiting:

**Recomputation**: Free the preempted sequence's KV blocks, save its tokens, and re-prefill from scratch when memory is available. Simple, no CPU memory needed, but wastes compute — especially expensive for long sequences that took seconds to prefill.

**Swapping**: Copy KV blocks from GPU to CPU RAM via PCIe, free GPU blocks. When memory is available, copy back from CPU to GPU and resume. Preserves compute but adds PCIe transfer latency (16-32 GB/s for PCIe Gen4, much slower than GPU HBM at 2-3 TB/s).

**Which to choose**: Swapping is better for long sequences (where reprefill is expensive) and when CPU RAM is plentiful. Recomputation is better for short sequences (cheap to re-prefill) and when CPU memory is tight. vLLM defaults to recomputation but supports configurable swap space.

**Best practice**: Avoid preemption entirely by right-sizing `--max-model-len` and `--max-num-seqs`. Monitor `vllm:num_preemptions_total` — any preemptions indicate memory pressure and should be addressed by reducing batch capacity or enabling KV cache quantization.

### Q: How does vLLM's prefix caching work? How does it compare to SGLang's RadixAttention?

vLLM uses **hash-based prefix caching**. Each KV block is identified by hashing its content (the token IDs it contains plus the hash of all preceding blocks). When a new request arrives, vLLM computes block hashes for the prompt and checks the hash table for matches. Matched blocks are reused; only unmatched blocks need prefill.

**Key differences from SGLang's RadixAttention**:

| Aspect | vLLM | SGLang |
|--------|------|--------|
| Data structure | Hash table | Radix tree |
| Enabled | Opt-in flag | Always on (default) |
| Matching | Block-level hash lookup | Tree traversal from root |
| Hierarchy | Flat (each block independent) | Hierarchical (parent-child) |
| Scheduling | No prefix-aware scheduling | LPM (Longest Prefix Match) policy |
| Eviction | LRU on blocks | LRU on tree branches |

SGLang's radix tree is more naturally suited for hierarchical prefix sharing (system prompt → few-shot → user) and provides prefix-aware scheduling (LPM prioritizes requests with long cache hits). vLLM's hash-based approach is simpler and effective for common cases but doesn't capture the structure of prefix relationships.

### Q: What are CUDA graphs and why does vLLM use them?

CUDA graphs capture a sequence of GPU kernel launches as a single replayable graph. Without them, the CPU issues each kernel launch individually — for a forward pass with ~50 kernels, the CPU launch overhead can be significant, especially at small batch sizes where GPU compute is fast.

During decode, the same sequence of kernels runs every step (same model architecture, same operations). vLLM captures CUDA graphs for several common batch sizes during startup. At runtime, it selects the graph matching the current batch size and replays it with a single CPU call.

**Impact**: At batch size 1, CUDA graphs can improve throughput by 10-30% by eliminating CPU-side overhead. At large batch sizes, the impact diminishes because GPU compute dominates.

**Limitations**: CUDA graphs require fixed tensor shapes. vLLM handles this by padding batches to pre-captured sizes and by disabling graphs for operations with dynamic shapes (like variable-length prefill). The `--enforce-eager` flag disables CUDA graphs for debugging.

### Q: Walk through how vLLM handles beam search efficiently.

Beam search generates $k$ candidate sequences (beams) that share a growing prefix and diverge at branch points. Without optimization, each beam needs its own complete KV cache — $k \times$ the memory.

vLLM uses **copy-on-write** via PagedAttention:

1. **Initial prefill**: All beams share the same prompt KV blocks (reference count = $k$)
2. **Divergence**: When beams generate different tokens, the shared block is COW-copied — a new physical block is allocated, the old block's data is copied, and each beam gets its own copy to write to
3. **Pruning**: When a beam is eliminated (lowest score), its unique blocks are freed. Shared blocks have their reference count decremented.
4. **Convergence**: If beams reconverge (same token), their blocks can be re-shared

Memory savings: For beam width $k=4$ with a 2000-token prompt, standard allocation needs $4 \times 2000 = 8000$ tokens of KV cache. With COW, the shared prefix uses only $2000$ tokens, and each beam only needs unique blocks for divergent tokens — typically saving 60-80% memory.

### Q: How do you right-size a vLLM deployment? Walk through the capacity planning.

**Step 1: Model memory**

$$\text{Model memory} = \text{params} \times \text{bytes per param}$$

Llama 3.1 70B in FP16: $70B \times 2 = 140$ GB → needs at least 2× A100 80GB (TP=2)

With FP8 quantization: $70B \times 1 = 70$ GB → fits on 1× A100 80GB (tight)

**Step 2: KV cache budget**

$$\text{KV budget} = \text{GPU memory} \times \text{utilization} - \text{model memory}$$

Example: 4× A100 80GB, TP=4, FP16 model:
$320 \text{ GB} \times 0.9 - 140 \text{ GB} = 148$ GB for KV cache

**Step 3: Max concurrent sequences**

$$\text{max sequences} = \frac{\text{KV budget}}{\text{KV per sequence}}$$

With max_model_len=4096, Llama 3.1 70B:
$\text{KV per seq} = 320 \text{ KB/token} \times 4096 = 1.28$ GB
$\text{max sequences} = 148 / 1.28 \approx 115$ concurrent sequences

With FP8 KV cache: KV per seq halved → ~230 concurrent sequences

**Step 4: Throughput estimation**

Decode throughput ≈ (batch_size × tokens/sec per sequence)

At batch 115 on 4× A100: ~30-40 tokens/sec per sequence → ~3500-4600 total tokens/sec

**Step 5: Validate with benchmarking**

Always benchmark with realistic traffic — theoretical calculations don't account for scheduling overhead, prefill-decode contention, or varying sequence lengths.

### Q: Compare vLLM with TensorRT-LLM and SGLang for production deployment.

| Feature | vLLM | TensorRT-LLM | SGLang |
|---------|------|-------------|--------|
| **Primary strength** | Ecosystem & flexibility | Raw NVIDIA GPU performance | Prefix caching & structured output |
| **Memory management** | PagedAttention | Paged blocks + NVIDIA-optimized | RadixAttention |
| **Prefix caching** | Hash-based (opt-in) | Supported | Radix tree (always on, LPM scheduling) |
| **Constrained decoding** | Outlines integration | Limited | Native FSM + jump-forward |
| **Hardware support** | NVIDIA, AMD, TPU, CPU | NVIDIA only | NVIDIA primarily |
| **Quantization** | FP8, AWQ, GPTQ, SqueezeLLM | FP8 (native H100), INT4/INT8 | FP8, AWQ, GPTQ |
| **Pipeline parallelism** | Mature | Excellent | Limited |
| **Community & ecosystem** | Largest | NVIDIA-backed | Growing rapidly |
| **LoRA serving** | Native multi-LoRA | Limited | Supported |
| **Best for** | General purpose, multi-hardware | Max throughput on NVIDIA | Multi-turn chat, structured output |

**Decision framework**:
- **Default choice**: vLLM — largest community, broadest hardware support, well-tested in production
- **Maximum NVIDIA performance**: TensorRT-LLM — custom CUDA kernels optimized for H100/H200, best raw throughput
- **Multi-turn / structured output**: SGLang — RadixAttention for conversation reuse, jump-forward for JSON/regex output
- **Multi-hardware (AMD, TPU)**: vLLM — broadest backend support

### Q: What are the most common production issues with vLLM and how do you debug them?

**1. High TTFT (Time-to-First-Token)**
- Cause: Long prefill queue, large prompts without chunked prefill
- Debug: Check `vllm:num_requests_waiting` — if high, prefill is the bottleneck
- Fix: Enable chunked prefill, reduce `--max-model-len`, add TP to distribute prefill compute

**2. High TPOT (Time-Per-Output-Token)**
- Cause: KV cache too large (memory-bandwidth-bound), batch too small
- Debug: Check `vllm:gpu_cache_usage_perc` — if high, KV cache dominates bandwidth
- Fix: Enable FP8 KV cache, reduce `--max-model-len`, increase batch size (more sequences amortize weight loading)

**3. Frequent preemptions**
- Cause: KV cache memory exhausted
- Debug: Check `vllm:num_preemptions_total` — any non-zero value indicates pressure
- Fix: Reduce `--max-model-len` (most impactful), enable FP8 KV cache, increase `--gpu-memory-utilization`, add more GPUs

**4. OOM errors**
- Cause: Model + KV cache + activations exceed GPU memory
- Fix: Reduce `--gpu-memory-utilization` (to leave more buffer), enable quantization, increase TP

**5. Throughput drops under load**
- Cause: Scheduling overhead, excessive preemption/swapping, or CUDA graph misses
- Debug: Profile with `py-spy` or `nsys`, check if batch sizes match captured CUDA graph sizes
- Fix: Pre-warm CUDA graphs for expected batch sizes, tune `--max-num-seqs`

## References

1. Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
2. Yu, G., et al. "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention." 2023.
3. [vLLM GitHub Repository](https://github.com/vllm-project/vllm)
4. [vLLM Documentation](https://docs.vllm.ai/)
5. Dao, T., et al. "FlashAttention: Fast and Memory-Efficient Exact Attention." NeurIPS 2022.
6. Zheng, L., et al. "SGLang: Efficient Execution of Structured Language Model Programs." 2024.
7. Leviathan, Y., et al. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.
8. Agrawal, A., et al. "Sarathi: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills." 2024.
9. NVIDIA. "TensorRT-LLM: NVIDIA's Library for LLM Inference." 2024.

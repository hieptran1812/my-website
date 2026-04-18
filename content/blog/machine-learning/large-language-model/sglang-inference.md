---
title: "SGLang: A Complete Guide to Fast LLM Inference"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "Large Language Model"
tags:
  [
    "llm",
    "sglang",
    "inference",
    "serving",
    "radix-attention",
    "optimization",
    "deep-learning",
    "kv-cache",
    "vllm",
  ]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "SGLang is one of the fastest LLM serving frameworks, pioneering RadixAttention for automatic KV cache reuse. This guide covers its architecture, core innovations, how it compares to vLLM, and everything you need for interviews on LLM serving systems."
---

## What Is SGLang?

SGLang (Structured Generation Language) is an open-source LLM serving framework developed at UC Berkeley that achieves state-of-the-art inference throughput through a combination of novel optimizations. It consists of two tightly integrated components:

1. **SGLang Runtime (SRT)**: The backend serving engine that handles model execution, KV cache management, scheduling, and GPU kernel optimization
2. **SGLang Frontend**: A domain-specific language (DSL) for expressing complex LLM programs — multi-turn conversations, branching logic, constrained generation, and tool use — in a way that the runtime can optimize

The key insight behind SGLang: **real-world LLM workloads have massive redundancy.** Users share system prompts, multi-turn conversations reuse previous turns, and structured generation follows predictable patterns. SGLang is designed from the ground up to exploit this redundancy, primarily through its flagship innovation — **RadixAttention**.

```
┌─────────────────────────────────────────────────────┐
│                    SGLang Stack                       │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │           Frontend (Python DSL)                │  │
│  │  Multi-turn programs, branching, constraints   │  │
│  └──────────────────────┬────────────────────────┘  │
│                         │                            │
│  ┌──────────────────────▼────────────────────────┐  │
│  │           SGLang Runtime (SRT)                  │  │
│  │  ┌──────────┐ ┌────────────┐ ┌─────────────┐  │  │
│  │  │ Radix    │ │ Continuous │ │ Constrained  │  │  │
│  │  │ Attention│ │ Batching   │ │ Decoding     │  │  │
│  │  └──────────┘ └────────────┘ └─────────────┘  │  │
│  │  ┌──────────┐ ┌────────────┐ ┌─────────────┐  │  │
│  │  │ Chunked  │ │ FlashInfer │ │ Speculative  │  │  │
│  │  │ Prefill  │ │ Kernels    │ │ Decoding     │  │  │
│  │  └──────────┘ └────────────┘ └─────────────┘  │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## The Core Problem: KV Cache Waste in Production

To understand why SGLang exists, let's look at what happens in a typical production LLM deployment.

### Scenario: Customer Support Bot

You deploy a chatbot with a 1500-token system prompt. Every user message triggers a new request, and each request starts with the same 1500 tokens:

```
Request 1: [system prompt (1500 tokens)] + "How do I reset my password?" (8 tokens)
Request 2: [system prompt (1500 tokens)] + "What are your business hours?" (6 tokens)
Request 3: [system prompt (1500 tokens)] + "I need a refund" (5 tokens)
...
Request 1000: [system prompt (1500 tokens)] + "Cancel my subscription" (4 tokens)
```

**Without prefix caching**: The system runs prefill on 1500 + ~6 tokens = ~1506 tokens for every single request. Across 1000 requests, that's **1,500,000 redundant prefill tokens** — computing the exact same KV cache entries 1000 times.

**With SGLang's RadixAttention**: The system computes the system prompt's KV cache once, stores it in a radix tree, and reuses it across all 1000 requests. Each new request only prefills the unique user portion (~6 tokens). Total: 1500 + 1000 × 6 = **7,500 prefill tokens**. That's a **200x reduction** in prefill compute.

### The Problem Is Even Bigger Than System Prompts

Prefix sharing isn't just about system prompts. In real workloads:

- **Multi-turn conversations**: Turn 3 reuses the KV cache from turns 1-2
- **Few-shot prompting**: All requests share the same few-shot examples
- **RAG applications**: Multiple queries over the same document share the document context
- **Parallel generation**: Generating multiple outputs for the same prompt (as in GRPO training or best-of-N sampling)
- **Tree-of-thought / branching**: Different reasoning branches share a common prefix

vLLM added prefix caching later (hash-based, opt-in), but SGLang's radix tree approach is fundamentally more flexible — it automatically detects and reuses any shared prefix at any depth, not just at the system prompt level.

## RadixAttention: The Core Innovation

RadixAttention is SGLang's most important contribution. It organizes all cached KV blocks in a **radix tree** (also called a prefix tree or trie), keyed by token sequences.

### What Is a Radix Tree?

A radix tree is a compressed trie where each edge represents a sequence of tokens (not just a single token). Nodes correspond to cached KV blocks.

```
Radix tree after 3 requests:

Root
 │
 ├── "System prompt tokens..." (1500 tokens, cached KV blocks)
 │    │
 │    ├── "How do I reset my password?" (KV blocks for this suffix)
 │    │
 │    ├── "What are your business hours?" (KV blocks)
 │    │
 │    └── "I need a refund" (KV blocks)
 │
 └── "Translate the following..." (different prefix, cached separately)
      │
      ├── "Hello world" → "Bonjour le monde"
      │
      └── "Good morning" → "Bon matin"
```

### How RadixAttention Works

**On each new request:**

1. **Prefix matching**: Walk the radix tree from the root, matching the new request's tokens against existing edges. Find the longest matching prefix.
2. **KV reuse**: For the matched prefix, reuse the cached KV blocks directly — no prefill computation needed for these tokens.
3. **Prefill the suffix**: Only compute KV for the novel tokens after the longest match.
4. **Insert into tree**: After generation completes, insert the new token sequence (including generated tokens) into the tree for potential future reuse.

```python
# Pseudocode for RadixAttention lookup
class RadixTree:
    def __init__(self):
        self.root = RadixNode()
    
    def match_prefix(self, token_ids):
        """
        Find longest matching prefix in the tree.
        
        Returns:
            matched_length: Number of tokens that match existing cache
            kv_blocks: Cached KV blocks for the matched prefix
        """
        node = self.root
        matched = 0
        kv_blocks = []
        
        while matched < len(token_ids):
            # Find child edge that matches the next tokens
            child = node.find_child(token_ids[matched:])
            if child is None:
                break
            
            # Match as many tokens as possible along this edge
            edge_tokens = child.edge_tokens
            match_len = common_prefix_length(
                token_ids[matched:], edge_tokens
            )
            
            if match_len == 0:
                break
            
            kv_blocks.extend(child.kv_blocks[:match_len])
            matched += match_len
            
            if match_len < len(edge_tokens):
                break  # Partial edge match
            
            node = child
        
        return matched, kv_blocks
    
    def insert(self, token_ids, kv_blocks):
        """Insert a new sequence into the tree."""
        # ... standard radix tree insertion with KV block attachment
```

### LRU Eviction

GPU memory is finite. When the tree grows too large, SGLang uses **LRU (Least Recently Used) eviction** to remove the least-recently-accessed tree branches:

```
Before eviction (GPU memory full):
Root
 ├── Prefix A (last used: 2 min ago) ── Suffix A1, A2, A3
 ├── Prefix B (last used: 30 sec ago) ── Suffix B1
 └── Prefix C (last used: 10 min ago) ── Suffix C1, C2  ← LRU candidate

After eviction:
Root
 ├── Prefix A (last used: 2 min ago) ── Suffix A1, A2, A3
 └── Prefix B (last used: 30 sec ago) ── Suffix B1

Prefix C evicted → its KV blocks freed → memory available for new requests
```

The eviction is at the **branch level**, not the token level. Entire subtrees are evicted together, which keeps the tree structure clean and avoids fragmented partial caches.

### RadixAttention vs vLLM's Prefix Caching

| Aspect | SGLang RadixAttention | vLLM Prefix Caching |
|--------|----------------------|---------------------|
| Data structure | Radix tree (compressed trie) | Hash table of block contents |
| Matching granularity | Any shared prefix at any depth | Block-aligned hashes |
| Automatic detection | Yes — any common prefix found automatically | Yes — hash-based lookup |
| Multi-turn reuse | Natural — previous turns form tree branches | Supported via hash matching |
| Eviction strategy | LRU on tree branches | LRU on blocks |
| Overhead | Tree traversal (minimal) | Hash computation per block |
| Enabled by default | Yes | Opt-in (`enable_prefix_caching=True`) |

The practical difference: SGLang's radix tree provides **structural** prefix management — you can see the tree, reason about sharing patterns, and the system naturally handles hierarchical reuse (e.g., system prompt → few-shot examples → user query, where each level can be independently shared). vLLM's hash-based approach is more "flat" — it identifies reuse at the block level without a hierarchical structure.

## Continuous Batching and Scheduling

Like vLLM, SGLang uses **continuous batching** — requests are added to and removed from the running batch dynamically, rather than waiting for all requests in a static batch to complete.

### The Scheduler

SGLang's scheduler handles two types of operations:

**Prefill requests**: New requests that need their prompt processed. These are compute-heavy (large matrix multiplications on many tokens) and have higher priority for latency-sensitive workloads.

**Decode requests**: In-progress sequences generating tokens one at a time. These are memory-bandwidth-heavy (loading KV cache) and benefit from being batched together.

```
Timeline of continuous batching:

Time 0: [Prefill R1] [Decode R2] [Decode R3]
Time 1: [Decode R1]  [Decode R2] [Decode R3] [Prefill R4]
Time 2: [Decode R1]  [R3 done ✓] [Decode R4] [Prefill R5]
Time 3: [Decode R1]  [Decode R4]  [Decode R5]
Time 4: [R1 done ✓]  [Decode R4]  [Decode R5] [Prefill R6]
...
New requests enter immediately. Finished requests leave immediately.
No wasted GPU cycles waiting for a batch to complete.
```

### Chunked Prefill

For long prompts, prefill can monopolize the GPU for seconds, blocking decode steps for all other sequences. SGLang implements **chunked prefill**: break long prompts into smaller chunks (e.g., 1024 tokens) and interleave them with decode steps.

```python
# SGLang server with chunked prefill
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --chunked-prefill-size 2048  # process prefill in 2048-token chunks
```

Benefits:
- **Lower P99 latency**: Other sequences get regular decode slots between prefill chunks
- **Better GPU utilization**: Mix compute-heavy prefill chunks with memory-heavy decode steps
- **More predictable latency**: No long pauses caused by processing a 10K-token prompt in one shot

## Constrained Decoding with Jump-Forward

SGLang has built-in support for **constrained decoding** — forcing the model's output to follow a specific format (JSON schema, regex, grammar). What makes SGLang's approach unique is the **jump-forward optimization**.

### The Problem with Naive Constrained Decoding

Standard constrained decoding applies a mask to the logits at each step, zeroing out tokens that violate the constraint. But this is wasteful when the constraint dictates the next several tokens:

```
Generating JSON with schema: {"name": string, "age": int}

Standard constrained decoding:
  Step 1: Generate '{' (only valid option → mask forces '{')
  Step 2: Generate '"' (only valid option)
  Step 3: Generate 'n' (only valid option)
  Step 4: Generate 'a' (only valid option)
  Step 5: Generate 'm' (only valid option)
  Step 6: Generate 'e' (only valid option)
  Step 7: Generate '"' (only valid option)
  Step 8: Generate ':' (only valid option)
  Step 9: Generate ' ' (only valid option)
  Step 10: Generate '"' (only valid option)
  → 10 forward passes to generate 10 deterministic tokens!
```

### Jump-Forward Optimization

When the constraint dictates multiple consecutive tokens, SGLang **jumps forward** — directly appending the deterministic tokens without running the model:

```
SGLang jump-forward:
  Step 1: Constraint dictates '{"name": "' → jump forward 10 tokens (no forward pass!)
  Step 2: Generate the actual name value (model runs here — multiple valid tokens)
  Step 3: Constraint dictates '", "age": ' → jump forward 9 tokens
  Step 4: Generate the actual age value (model runs here)
  Step 5: Constraint dictates '}' → jump forward 1 token
  → Only 2 forward passes for model decisions + deterministic token appending!
```

This can provide **3-5x speedup** for highly structured outputs like JSON, SQL, or code with fixed syntax.

### How It Works Internally

SGLang uses a **Finite State Machine (FSM)** compiled from the constraint specification (regex, JSON schema, or grammar). The FSM tracks which tokens are valid at each position. When only one token (or a fixed sequence) is valid, the system jumps forward:

```python
# Pseudocode for jump-forward constrained decoding
def constrained_decode_with_jumpforward(model, fsm, input_ids):
    state = fsm.initial_state
    
    while not fsm.is_terminal(state):
        # Check how many tokens are deterministic from this state
        deterministic_tokens = fsm.get_deterministic_sequence(state)
        
        if len(deterministic_tokens) > 0:
            # Jump forward: append tokens without running the model
            input_ids = append(input_ids, deterministic_tokens)
            state = fsm.advance(state, deterministic_tokens)
        else:
            # Multiple valid next tokens — run the model
            logits = model(input_ids)
            valid_mask = fsm.get_valid_token_mask(state)
            logits[~valid_mask] = -float('inf')
            
            token = sample(logits)
            input_ids = append(input_ids, token)
            state = fsm.advance(state, token)
    
    return input_ids
```

## FlashInfer: Optimized GPU Kernels

SGLang uses **FlashInfer** as its attention kernel backend — a library of highly optimized CUDA kernels specifically designed for LLM serving workloads.

### Why Custom Kernels Matter

The standard attention computation ($\text{softmax}(QK^T/\sqrt{d}) \cdot V$) has different computational profiles depending on the phase:

- **Prefill**: Long sequence of queries, long sequence of keys. This is a large matrix multiplication → compute-bound. FlashAttention-style kernels work well.
- **Decode**: Single query, long sequence of cached keys. This is a vector-matrix product → memory-bandwidth-bound. Standard FlashAttention is suboptimal because it's designed for compute-bound scenarios.

FlashInfer provides specialized kernels for each case:

| Kernel | Phase | Optimization |
|--------|-------|-------------|
| `BatchPrefillWithPagedKVCache` | Prefill | Fused attention with paged KV, tiling for large sequences |
| `BatchDecodeWithPagedKVCache` | Decode | Memory-bandwidth-optimized, parallel KV page loading |
| `BatchPrefillWithRaggedKVCache` | Mixed | Handles variable-length prefill + decode in one batch |
| `MergeState` | Both | Efficiently merges attention states from split computations |

### Paged KV Cache in FlashInfer

Like vLLM, SGLang uses paged KV cache — but FlashInfer's kernels are designed from the ground up for paged memory access:

```
Standard (contiguous) KV cache:
  [token0_K][token1_K][token2_K][...][tokenN_K]  ← continuous in memory
  
  Efficient sequential access, but fragmentation issues

Paged KV cache (FlashInfer):
  Page 0: [token0_K][token1_K][..][token15_K]    ← 16 tokens per page
  Page 1: [token16_K][token17_K][..][token31_K]
  ...
  
  Block table maps sequence positions to pages:
  Seq A: [Page 0] → [Page 3] → [Page 7] → [Page 12]
  Seq B: [Page 1] → [Page 4] → [Page 8]
```

FlashInfer's kernels handle the page table indirection within the GPU kernel, avoiding the overhead of gathering pages into a contiguous buffer before attention computation.

## Data Parallelism and Tensor Parallelism

### Tensor Parallelism (TP)

For large models that don't fit on a single GPU, SGLang splits the model across GPUs using tensor parallelism:

```python
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4  # split across 4 GPUs
```

Each GPU holds a portion of each layer's weight matrices. Every forward pass requires all-reduce communication between GPUs to combine partial results. TP reduces per-GPU memory but adds communication overhead.

### Data Parallelism (DP)

For high-throughput serving, SGLang supports running multiple independent model replicas:

```python
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --dp 4  # 4 independent replicas
```

Each replica handles its own requests independently, with a load balancer distributing traffic. No inter-GPU communication during inference. This is ideal for smaller models where each replica fits on one GPU.

### Combined DP + TP

For large models with high throughput requirements:

```python
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 --dp 2  # 2 replicas, each split across 4 GPUs (8 GPUs total)
```

## The SGLang Frontend: Programming LLM Workloads

Beyond the runtime, SGLang provides a Python DSL for expressing complex LLM programs that the runtime can optimize.

### Basic Usage

```python
import sglang as sgl

@sgl.function
def multi_turn_qa(s, question1, question2):
    s += sgl.system("You are a helpful assistant.")
    s += sgl.user(question1)
    s += sgl.assistant(sgl.gen("answer1", max_tokens=256))
    s += sgl.user(question2)
    s += sgl.assistant(sgl.gen("answer2", max_tokens=256))

# The runtime automatically:
# 1. Caches the system prompt's KV across calls
# 2. Caches turn 1's KV when processing turn 2
# 3. Batches generation calls efficiently
```

### Branching (Fork/Join)

SGLang can fork a single prefix into multiple branches — each branch shares the prefix KV cache and generates independently:

```python
@sgl.function
def best_of_n(s, prompt, n=5):
    s += sgl.user(prompt)
    
    # Fork into n parallel branches sharing the same prefix
    forks = s.fork(n)
    for i, f in enumerate(forks):
        f += sgl.assistant(sgl.gen(f"response_{i}", max_tokens=512, temperature=0.8))
    
    # All n branches share the prefix KV cache
    # Only the divergent suffixes are computed independently
```

This is much more efficient than sending $n$ independent requests, because the prompt's KV cache is computed once and shared across all branches.

### Constrained Generation

```python
@sgl.function
def generate_json(s, task_description):
    s += sgl.user(task_description)
    s += sgl.assistant(
        sgl.gen("output", max_tokens=512, 
                regex=r'\{"name": "[^"]+", "age": \d+, "city": "[^"]+"\}')
    )
```

The regex constraint is compiled into an FSM, and jump-forward optimization is applied automatically.

### Batched Execution

```python
# Process 1000 questions in parallel with automatic batching
questions = ["What is gravity?", "Explain photosynthesis", ...]

states = multi_turn_qa.run_batch([
    {"question1": q, "question2": "Can you elaborate?"}
    for q in questions
])

# SGLang automatically batches these, shares common prefixes,
# and maximizes GPU utilization
```

## SGLang vs vLLM: A Detailed Comparison

This is the most common interview question about SGLang. Let's be thorough.

### Architecture Comparison

| Component | SGLang | vLLM |
|-----------|--------|------|
| **KV cache management** | Radix tree (RadixAttention) | Block manager with page tables |
| **Prefix caching** | Automatic via radix tree (always on) | Hash-based (opt-in flag) |
| **Attention kernels** | FlashInfer | FlashAttention / FlashInfer |
| **Constrained decoding** | FSM with jump-forward (native) | Outlines integration |
| **Scheduling** | Custom scheduler with priority | Continuous batching scheduler |
| **API** | OpenAI-compatible + native DSL | OpenAI-compatible |
| **Speculative decoding** | EAGLE, EAGLE-2 support | Draft model, n-gram, Medusa |
| **Multi-modal** | Vision-language models (LLaVA, Qwen-VL, etc.) | Vision-language models |
| **Quantization** | FP8, INT4 (AWQ, GPTQ), GGUF | FP8, INT4 (AWQ, GPTQ) |

### Performance Characteristics

**SGLang tends to win when:**
- Workloads have significant prefix sharing (multi-turn chat, shared system prompts)
- Structured output is needed (JSON, regex-constrained generation)
- Complex LLM programs with branching/forking
- Multi-turn conversations where turn reuse is important
- Deployments where RadixAttention's automatic prefix caching adds value

**vLLM tends to win when:**
- Simple, one-shot request patterns with minimal prefix sharing
- Established ecosystem integration is important (more adapters, more quantization formats, larger community)
- Pipeline parallelism is needed (vLLM has more mature PP support)
- Specific hardware optimizations (vLLM has broader hardware backend support)

**In practice**: The performance difference depends heavily on workload characteristics. Benchmarks on pure throughput (random prompts, no prefix sharing) show similar numbers. Benchmarks on realistic workloads with prefix sharing show SGLang with significant advantages (2-5x on prefix-heavy workloads).

### Quick Decision Guide

```
Choose SGLang when:
  ├── Multi-turn chat serving
  ├── Structured output (JSON, SQL) generation
  ├── Complex LLM programs (fork/join, branching)
  ├── Shared system prompts across requests
  └── You want automatic prefix caching without configuration

Choose vLLM when:
  ├── Simple request-response serving
  ├── Need broadest ecosystem support
  ├── Pipeline parallelism across many GPUs
  └── Production stability is the top priority (larger community, more battle-tested)
```

## Deployment Guide

### Basic Server Launch

```bash
# Single GPU, basic setup
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --port 8000

# Multi-GPU with tensor parallelism
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 \
    --port 8000

# With FP8 quantization
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 2 \
    --quantization fp8 \
    --port 8000

# With KV cache quantization
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 \
    --kv-cache-dtype fp8_e5m2 \
    --port 8000

# With EAGLE speculative decoding
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path yuhuili/EAGLE-LLaMA3.1-Instruct-70B \
    --speculative-num-steps 5 \
    --port 8000
```

### OpenAI-Compatible API

SGLang exposes an OpenAI-compatible API, making migration from OpenAI or vLLM straightforward:

```python
import openai

client = openai.Client(base_url="http://localhost:8000/v1", api_key="none")

response = client.chat.completions.create(
    model="meta-llama/Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ],
    temperature=0.7,
    max_tokens=256,
)

print(response.choices[0].message.content)
```

### Key Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--tp` | 1 | Tensor parallelism degree |
| `--dp` | 1 | Data parallelism degree |
| `--mem-fraction-static` | 0.88 | Fraction of GPU memory for KV cache |
| `--chunked-prefill-size` | 8192 | Max tokens per prefill chunk |
| `--max-running-requests` | auto | Maximum concurrent decoding sequences |
| `--schedule-policy` | `lpm` | Scheduling policy: `lpm` (longest prefix match), `fcfs`, `random` |
| `--kv-cache-dtype` | auto | KV cache precision: `auto`, `fp8_e5m2` |
| `--quantization` | None | Weight quantization: `fp8`, `awq`, `gptq` |
| `--context-length` | model default | Override max context length |
| `--disable-radix-cache` | false | Disable RadixAttention (for benchmarking) |

### Monitoring and Observability

```python
# Health check
curl http://localhost:8000/health

# Server metrics
curl http://localhost:8000/get_server_info

# Key metrics to monitor:
# - cache_hit_rate: How often RadixAttention finds a matching prefix
# - num_running_requests: Current batch size
# - num_waiting_requests: Queue depth (should be low)
# - token_usage: Total KV cache memory utilization
# - avg_prefill_latency: Time to process prompts
# - avg_decode_latency: Time per output token
```

## Internals: Request Lifecycle

Let's trace a request through SGLang end-to-end:

```
1. REQUEST ARRIVES
   POST /v1/chat/completions
   Messages: [system: "...", user: "What is AI?"]
   │
2. TOKENIZATION
   Convert messages to token IDs using the model's tokenizer
   Apply chat template (e.g., <|begin_of_text|><|start_header_id|>system...)
   Result: [1, 128006, 9125, 128007, 10, ..., 128009]  (e.g., 1520 tokens)
   │
3. RADIX TREE LOOKUP
   Walk the radix tree matching token IDs:
   ├── Match: 1500 tokens (system prompt cached from previous request!)
   └── No match: 20 tokens (user message is new)
   │
   Hit! Reuse 1500 tokens of KV cache. Only prefill 20 tokens.
   │
4. SCHEDULING
   Add to the prefill queue (20 tokens to process)
   Wait for next scheduling cycle
   │
5. PREFILL
   Compute Q, K, V for the 20 new tokens
   Attend to: 1500 cached K,V + 20 new K,V = 1520 total
   Append new K,V to the KV cache (and radix tree)
   Predict first output token
   │
6. DECODE LOOP
   Move to decode queue
   For each step:
     a. Compute Q, K, V for 1 new token
     b. Append K, V to cache
     c. Attend to full KV cache
     d. Predict next token
     e. Check stopping conditions (EOS, max_tokens)
   │
7. RESPONSE
   Detokenize output tokens
   Stream back via SSE (Server-Sent Events) or return complete response
   │
8. CLEANUP
   KV cache blocks remain in radix tree (available for future reuse)
   If memory pressure → LRU eviction of oldest unused branches
```

## Advanced Features

### Overlap Scheduling

SGLang can overlap prefill computation of new requests with decode computation of existing requests, maximizing GPU utilization:

```
Without overlap:
  Time: |---prefill R1---|--decode R1--|--decode R1--|---prefill R2---|--decode R1+R2--|
                                                     ↑ GPU idle during scheduling gap

With overlap:
  Time: |---prefill R1 + decode existing---|--decode all--|---prefill R2 + decode all---|
         ↑ prefill and decode run together (different memory access patterns complement each other)
```

### Multi-Modal Support

SGLang natively supports vision-language models:

```python
python -m sglang.launch_server \
    --model-path Qwen/Qwen2-VL-7B-Instruct \
    --tp 1 \
    --port 8000

# Usage with image input
response = client.chat.completions.create(
    model="Qwen/Qwen2-VL-7B-Instruct",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
            {"type": "text", "text": "What's in this image?"},
        ],
    }],
)
```

### Embedding and Reward Models

SGLang can serve embedding models and reward models, not just generative models:

```bash
# Embedding model
python -m sglang.launch_server \
    --model-path BAAI/bge-large-en-v1.5 \
    --is-embedding \
    --port 8000

# Reward model
python -m sglang.launch_server \
    --model-path Skywork/Skywork-Reward-Llama-3.1-8B \
    --port 8000
```

## Interview Questions and Answers

### Q: What is SGLang and what problem does it solve?

SGLang is an LLM serving framework that achieves high inference throughput by exploiting redundancy in real-world LLM workloads. Its core innovation is **RadixAttention** — a radix tree-based KV cache management system that automatically detects and reuses shared prefixes across requests. Real workloads have massive redundancy: shared system prompts, multi-turn conversation history, common few-shot examples. SGLang eliminates the redundant prefill computation for these shared prefixes, providing 2-5x throughput improvements on prefix-heavy workloads compared to naive serving.

Beyond RadixAttention, SGLang provides: a constrained decoding engine with jump-forward optimization, FlashInfer-based GPU kernels optimized for both prefill and decode, a Python DSL for expressing complex LLM programs, and support for speculative decoding, multi-modal models, and quantization.

### Q: Explain RadixAttention in detail. How does the radix tree work?

RadixAttention organizes all cached KV blocks in a radix tree (compressed trie) keyed by token sequences. Each edge represents a sequence of tokens, and each node stores pointers to the corresponding KV cache blocks in GPU memory.

**On request arrival**: The system walks the tree from the root, matching the new request's token sequence against existing edges. It finds the longest matching prefix — this could be 0 tokens (no match), the system prompt (partial match), or the entire prompt (full cache hit).

**Cache reuse**: For the matched prefix, the cached KV blocks are reused directly — no prefill computation needed. Only the novel suffix tokens after the match require prefill.

**After generation**: The full sequence (prompt + generated output) is inserted into the tree, creating new edges and nodes. Future requests with overlapping prefixes will find these cached entries.

**Memory management**: When GPU memory is full, LRU eviction removes the least-recently-used tree branches, freeing their KV blocks. Eviction is at the branch level, not individual tokens, keeping the tree structure clean.

The key advantage over hash-based prefix caching (vLLM) is that the radix tree provides **hierarchical** prefix management: system prompt → few-shot examples → conversation history → current query, where each level can be independently cached and shared. The tree structure also makes it easy to visualize and debug cache sharing patterns.

### Q: How does SGLang's constrained decoding with jump-forward work?

SGLang compiles output constraints (JSON schema, regex, grammar) into a **Finite State Machine (FSM)**. At each decoding step, the FSM determines which tokens are valid at the current state.

The **jump-forward optimization** exploits a key observation: when the constraint dictates that only one token (or a specific sequence of tokens) is valid, there's no need to run the model — the output is deterministic. SGLang directly appends these deterministic tokens without any forward pass, advancing the FSM state accordingly. The model only runs when multiple valid tokens exist and an actual decision must be made.

For structured outputs like JSON with a fixed schema, this can skip 50-80% of the tokens, providing 3-5x speedup. For example, generating `{"name": "Alice", "age": 30}` requires model decisions only for the actual values ("Alice" and "30") — all structural tokens (`{"name": "`, `", "age": `, `}`) are deterministic from the schema.

### Q: Compare SGLang and vLLM in detail. When would you choose each?

**Architecture**: SGLang uses a radix tree for KV cache (RadixAttention), vLLM uses paged blocks with a block manager (PagedAttention). Both support continuous batching, chunked prefill, and paged KV cache.

**Prefix caching**: SGLang's RadixAttention is automatic and always-on — any shared prefix is detected and reused. vLLM's prefix caching is hash-based and opt-in. SGLang's tree structure provides hierarchical cache management; vLLM's approach is "flat" (block-level hashes).

**Constrained decoding**: SGLang has native FSM-based constrained decoding with jump-forward optimization (3-5x faster for structured output). vLLM integrates Outlines for constrained decoding but without jump-forward.

**When to choose SGLang**: Multi-turn chat (turn reuse via radix tree), structured output generation (jump-forward), shared system prompts across many requests, complex LLM programs with branching.

**When to choose vLLM**: Broadest hardware support, largest community and ecosystem, more mature pipeline parallelism, established production track record.

**Performance**: On workloads with significant prefix sharing, SGLang typically outperforms vLLM by 2-5x. On simple, one-shot requests with unique prompts, performance is comparable.

### Q: Explain the difference between prefill and decode phases and how SGLang optimizes each.

**Prefill** processes the prompt tokens all at once. It's **compute-bound** — dominated by large matrix multiplications (many tokens × weight matrices). SGLang optimizes prefill with:
- RadixAttention: skip prefill entirely for cached prefixes
- Chunked prefill: break long prompts into chunks to avoid blocking decode
- FlashInfer BatchPrefill kernels: tiled computation for large sequences
- Overlap scheduling: run prefill alongside decode from other sequences

**Decode** generates one token per step. It's **memory-bandwidth-bound** — the GPU spends most time loading the KV cache and model weights, not computing. SGLang optimizes decode with:
- FlashInfer BatchDecode kernels: optimized for bandwidth-limited vector-matrix products
- Continuous batching: maximize batch size to improve compute-to-bandwidth ratio
- KV cache quantization (FP8): reduce memory bandwidth per decode step
- Speculative decoding: generate multiple tokens per KV cache load

### Q: What is the scheduling policy in SGLang and how does LPM (Longest Prefix Match) work?

SGLang's default scheduling policy is **LPM (Longest Prefix Match)**: when choosing which waiting request to schedule next, prioritize the request whose prompt has the longest match in the radix tree.

**Why LPM**: A request with a long prefix match requires minimal prefill (most of its KV cache is already computed), so it can start generating quickly. This maximizes the cache hit rate and reduces average Time-to-First-Token (TTFT).

**Alternative policies**:
- `fcfs` (First Come First Served): Process requests in arrival order. Simple, fair, but ignores caching opportunities
- `random`: Random selection. Useful for load balancing across replicas but no caching optimization

LPM is the right choice for most deployments because it naturally prioritizes requests that benefit from cached state. The one exception is extreme fairness requirements where FCFS is preferred to prevent starvation of requests with unique (non-cached) prefixes.

### Q: How does SGLang handle memory pressure? What happens when the KV cache is full?

SGLang uses a multi-level strategy:

1. **LRU eviction of radix tree branches**: When GPU memory for KV cache is exhausted, the least-recently-used branches of the radix tree are evicted. Their KV blocks are freed and made available for new requests. The eviction is at the branch level (entire subtrees), not individual tokens.

2. **Request queuing**: If eviction doesn't free enough memory, new requests are queued until running requests complete and free their KV blocks.

3. **Memory fraction control**: The `--mem-fraction-static` parameter (default 0.88) controls what fraction of GPU memory is reserved for KV cache. The remaining 12% is buffer for activations, temporary tensors, and overhead.

**Key design decision**: SGLang prefers evicting cached (reusable) state over preempting running requests. This is because a preempted request must either be recomputed from scratch or its KV cache swapped to CPU (both expensive). An evicted cache branch only costs a future cache miss, which is less disruptive.

### Q: What is FlashInfer and why does SGLang use it instead of FlashAttention?

FlashInfer is a library of CUDA kernels specifically designed for LLM **serving** (inference), while FlashAttention was primarily designed for **training**.

The key difference is in the decode phase: during training and prefill, attention is a large matrix-matrix product (compute-bound) — FlashAttention excels here. During decode, attention is a vector-matrix product (bandwidth-bound) — FlashAttention is suboptimal because its tiling strategy is designed for compute-bound workloads.

FlashInfer provides:
- **BatchDecodeWithPagedKVCache**: Bandwidth-optimized kernel for single-query attention against a paged KV cache. Uses different tiling and memory access patterns than FlashAttention, specifically tuned for the low arithmetic intensity of decode.
- **BatchPrefillWithPagedKVCache**: Similar to FlashAttention but with native paged KV cache support (no gather/scatter overhead).
- **Ragged batching**: Handles variable-length sequences in a batch without padding, processing different-length prefills and decodes in a single kernel launch.

SGLang uses FlashInfer because it provides better performance across the full serving workload (prefill + decode + mixed batches), not just the compute-bound prefill portion.

### Q: How would you benchmark SGLang against vLLM for a production deployment?

A rigorous benchmark should test:

**1. Throughput benchmarks**:
- Fixed input/output length, increasing request rate → measure max sustainable throughput (requests/sec) before latency degrades
- Vary input lengths (short: 128, medium: 1024, long: 8192) to test prefill scaling
- Vary output lengths to test decode efficiency

**2. Latency benchmarks**:
- Time-to-First-Token (TTFT): measures prefill + scheduling latency
- Time-Per-Output-Token (TPOT): measures decode efficiency
- Report P50, P95, P99 latencies (averages hide tail latency issues)

**3. Prefix sharing benchmarks** (where SGLang's advantage appears):
- Fixed system prompt (1K-2K tokens) + varying user queries
- Multi-turn conversations (2-5 turns)
- Measure cache hit rate and its effect on TTFT

**4. Realistic workload benchmarks**:
- Use production traffic distributions (not uniform)
- Mix of short and long requests
- Bursty arrival patterns

**Tools**: Use SGLang's built-in benchmarking (`python -m sglang.bench_serving`) or common tools like `genai-perf`, `llmperf`.

**Key mistake to avoid**: Benchmarking with random, unique prompts of uniform length. This hides SGLang's prefix caching advantage and doesn't reflect real-world traffic patterns.

### Q: How does speculative decoding integrate with SGLang?

SGLang supports EAGLE and EAGLE-2 speculative decoding. The integration works as follows:

1. A lightweight **EAGLE head** (trained on the target model's features) generates draft tokens
2. The target model verifies all draft tokens in a single forward pass using tree-structured attention
3. Accepted tokens are committed to the KV cache; rejected tokens trigger rollback

SGLang's RadixAttention complements speculative decoding: the draft and verification both benefit from cached prefixes. The KV cache rollback for rejected tokens is handled by the radix tree's branch management — rejected branches are simply pruned.

Configuration:
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path yuhuili/EAGLE-LLaMA3.1-Instruct-70B \
    --speculative-num-steps 5
```

Speculative decoding in SGLang is most beneficial at low batch sizes (latency-sensitive interactive use). At high batch sizes (throughput-oriented), the overhead typically outweighs the benefit.

### Q: Design a production SGLang deployment for a high-traffic chatbot with 10K concurrent users.

**Requirements analysis**:
- 10K concurrent users → many requests share the system prompt
- Multi-turn conversations → turn reuse is important
- Target: TTFT < 500ms, TPOT < 50ms, throughput: 10K+ tokens/sec

**Architecture**:

```
Load Balancer (nginx/envoy)
    │
    ├── SGLang Replica 1 (TP=4, 4×H100)
    ├── SGLang Replica 2 (TP=4, 4×H100)
    ├── SGLang Replica 3 (TP=4, 4×H100)
    └── SGLang Replica 4 (TP=4, 4×H100)
```

**Configuration per replica**:
```bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tp 4 \
    --kv-cache-dtype fp8_e5m2 \       # 2x KV cache reduction
    --chunked-prefill-size 2048 \     # prevent prefill blocking
    --schedule-policy lpm \            # maximize cache hits
    --context-length 8192 \            # right-sized (not 128K default)
    --mem-fraction-static 0.90 \       # maximize KV cache capacity
    --port 8000
```

**Key decisions**:
1. **FP8 KV cache**: Doubles the number of concurrent sequences per GPU with negligible quality loss
2. **Context length 8192** (not 128K): Most chat conversations fit in 8K; this allows 8-16x more concurrent sequences
3. **LPM scheduling**: Maximizes radix tree cache hits for shared system prompts
4. **4 replicas with DP**: Each replica handles ~2.5K concurrent users; load balancer distributes traffic
5. **Monitoring**: Track cache hit rate (should be >80% for shared system prompt), TTFT P99, TPOT P99, KV cache utilization

**Scaling**: Add more DP replicas as traffic grows. Each replica is independent, so scaling is linear.

## References

1. Zheng, L., et al. "SGLang: Efficient Execution of Structured Language Model Programs." 2024.
2. Zheng, L., et al. "Efficiently Programming Large Language Models using SGLang." 2023.
3. Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
4. Ye, Z., et al. "FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving." 2024.
5. Li, Y., et al. "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees." 2024.
6. [SGLang GitHub Repository](https://github.com/sgl-project/sglang)
7. [SGLang Documentation](https://docs.sglang.ai/)
8. [FlashInfer GitHub Repository](https://github.com/flashinfer-ai/flashinfer)
9. [mini-sglang: Educational Implementation](https://github.com/sgl-project/mini-sglang)

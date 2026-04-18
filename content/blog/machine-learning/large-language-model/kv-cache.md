---
title: "KV Cache in Large Language Models: A Complete Guide"
publishDate: "2026-03-16"
category: "machine-learning"
subcategory: "Large Language Model"
tags: ["llm", "kv-cache", "transformer", "attention", "inference", "optimization", "deep-learning"]
date: "2026-03-16"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "KV Cache is the single most important optimization for LLM inference. This guide explains what it is, why it exists, how it works step-by-step with intuitive examples, and the techniques used to make it more memory-efficient."
---

## Introduction

If you've ever used ChatGPT, Claude, or any large language model, you've noticed something: the model generates text **one token at a time**. It doesn't produce the entire response instantly — it streams words out sequentially, each one depending on everything that came before.

Here's the problem: **generating each new token requires the model to "look at" all previous tokens through the attention mechanism.** Naively, this means recomputing attention over the entire sequence for every single new token. For a 1000-token response, the model would recompute attention over tokens 1-999 just to generate token 1000 — even though those computations were already done in previous steps.

This is where **KV Cache** comes in. It's a deceptively simple idea: **cache the intermediate results from previous tokens so you never compute them twice.** Without KV Cache, modern LLM inference would be orders of magnitude slower. It is arguably the single most important optimization in LLM serving.

Let's build up the intuition step by step.

## Prerequisites: How Attention Works

To understand KV Cache, we need to understand what the **attention mechanism** computes. If you're already familiar with self-attention, feel free to skip to the next section.

### The Core Idea of Attention

Attention is how a transformer decides **which previous words to focus on** when processing the current word. Consider the sentence:

> "The cat sat on the mat because **it** was tired."

When the model processes "it", attention is the mechanism that figures out "it" refers to "cat" (not "mat"). It does this by computing a relevance score between "it" and every preceding word.

### The Q, K, V Framework

Attention uses three vectors for each token, derived from the token's embedding through learned linear transformations:

- **Query (Q)**: "What am I looking for?" — represents the current token's question
- **Key (K)**: "What do I contain?" — represents each token's label/identity
- **Value (V)**: "What information do I carry?" — represents each token's actual content

Think of it like a **library analogy**:

```
You walk into a library looking for information about "cooking pasta."

- Your QUERY is: "cooking pasta" (what you're searching for)
- Each book's KEY is its title/index entry: "Italian Cuisine", "History of Rome",
  "Pasta Making 101", "Quantum Physics" (how each book describes itself)
- Each book's VALUE is its actual content: the pages inside

Step 1: You compare your QUERY against every book's KEY
         → "Pasta Making 101" scores highest, "Italian Cuisine" scores medium,
            "Quantum Physics" scores near zero

Step 2: You read the VALUES weighted by those scores
         → You read mostly from "Pasta Making 101", some from "Italian Cuisine",
            and essentially nothing from "Quantum Physics"
```

### The Math

For each attention layer, given input embeddings $X$:

$$Q = X \cdot W_Q, \quad K = X \cdot W_K, \quad V = X \cdot W_V$$

Where $W_Q$, $W_K$, $W_V$ are learned weight matrices with shapes:
- $W_Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$
- $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$

The attention output is:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q \cdot K^T}{\sqrt{d_k}}\right) \cdot V$$

Where $d_k$ is the dimension of the key vectors. The $\sqrt{d_k}$ scaling prevents the dot products from growing too large, which would push the softmax into regions with extremely small gradients.

In plain English: compute similarity between each query and all keys, normalize with softmax, then use those weights to combine the values.

### Multi-Head Attention

In practice, transformers don't use a single attention computation. They use **multi-head attention (MHA)**: the model runs multiple attention operations in parallel, each with its own set of $W_Q$, $W_K$, $W_V$ matrices. Each "head" can learn to attend to different types of relationships:

```
Head 1: might learn to focus on syntactic relationships (subject-verb)
Head 2: might learn to focus on coreference ("it" → "cat")
Head 3: might learn to focus on recent context
Head 4: might learn to focus on positional patterns
...

Final output = Concat(head_1, head_2, ..., head_h) × W_O
```

This is important for understanding KV Cache because **each head maintains its own K and V cache**. If a model has 32 layers with 32 heads each, that's $32 \times 32 = 1024$ separate K caches and 1024 separate V caches.

### Causal (Autoregressive) Masking

One more critical detail: during generation, LLMs use **causal masking**. This means each token can only attend to tokens **at or before** its position — never to future tokens. The attention matrix looks like this:

```
              Key positions →
            The  weather  today  is
Query   The  ✓     ✗       ✗     ✗
pos.  weather ✓     ✓       ✗     ✗
  ↓   today  ✓     ✓       ✓     ✗
        is   ✓     ✓       ✓     ✓

✓ = can attend, ✗ = masked (set to -∞ before softmax)
```

This causal property is what makes KV Cache possible: since token "The" at position 0 never looks at future tokens, its K and V vectors are determined entirely by its own position and content. They never change as new tokens are generated. This invariance is the mathematical foundation of the entire optimization.

## The Inference Problem: Why We Need KV Cache

### How LLMs Generate Text

Large language models are **autoregressive** — they generate one token at a time, and each new token depends on all previous tokens:

```
Input:  "The weather today is"
Step 1: Generate "really"      → looks at: "The weather today is"
Step 2: Generate "nice"        → looks at: "The weather today is really"
Step 3: Generate "outside"     → looks at: "The weather today is really nice"
Step 4: Generate "."           → looks at: "The weather today is really nice outside"
```

### The Naive Approach (Without KV Cache)

Without any caching, here's what happens at each step. Let's trace step 3 as an example:

To generate "outside", the model must:

1. Compute Q, K, V for **every** token: "The", "weather", "today", "is", "really", "nice"
2. Compute attention scores between the new query and all keys
3. Compute the weighted sum of all values
4. Pass through the rest of the transformer to produce the next token

But wait — at step 2, we already computed K and V for "The", "weather", "today", "is", "really". And at step 1, we computed K and V for "The", "weather", "today", "is". **We're recomputing the same K and V vectors over and over again.**

Let's visualize the wasted computation:

```
Step 1: Compute K,V for ["The", "weather", "today", "is"]
        → Generate "really"

Step 2: Compute K,V for ["The", "weather", "today", "is", "really"]
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          Already computed in Step 1! Wasted!
        → Generate "nice"

Step 3: Compute K,V for ["The", "weather", "today", "is", "really", "nice"]
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                          Already computed in Steps 1 & 2! Wasted!
        → Generate "outside"
```

The computational cost grows **quadratically** with sequence length. For a sequence of length $n$, the total work across all generation steps is:

$$\text{Total operations} \propto 1 + 2 + 3 + \cdots + n = \frac{n(n+1)}{2} = O(n^2)$$

For a 4096-token generation, that's about **8.4 million** redundant K/V computations. This is unacceptable for production systems.

### The Restaurant Analogy

Imagine you're a waiter at a restaurant. Every time a new customer sits down, you take their order. But in the naive approach, **you also re-take every existing customer's order** — even though their orders haven't changed:

```
Customer 1 arrives: Take order for Customer 1
Customer 2 arrives: Re-take order for Customer 1, Take order for Customer 2
Customer 3 arrives: Re-take order for Customer 1, Re-take order for Customer 2,
                    Take order for Customer 3
...
```

Obviously, you should just **write down each customer's order once** and keep a running list. That's exactly what KV Cache does.

## How KV Cache Works

### The Key Insight

Here's the crucial observation: **for previously generated tokens, K and V vectors never change.** Once we compute K and V for "The" at position 0, those vectors are the same no matter how many subsequent tokens we generate. This is because:

- K and V depend only on the token itself and its position
- In autoregressive (causal) models, previous tokens cannot attend to future tokens
- The weight matrices $W_K$ and $W_V$ are fixed during inference

So the solution is simple: **store (cache) the K and V vectors after computing them, and reuse them for all future steps.**

### Step-by-Step Example

Let's trace through generation with KV Cache:

```
Prompt: "The weather today is"

═══ Step 0: Process the prompt (Prefill Phase) ═══

Compute K,V for all prompt tokens at once:
  K_cache = [K_"The", K_"weather", K_"today", K_"is"]
  V_cache = [V_"The", V_"weather", V_"today", V_"is"]

Compute Q for all tokens, run full attention → predict next token: "really"


═══ Step 1: Generate "really" (Decode Phase) ═══

Compute K,V for ONLY the new token "really":
  K_new = K_"really"
  V_new = V_"really"

Append to cache:
  K_cache = [K_"The", K_"weather", K_"today", K_"is", K_"really"]
  V_cache = [V_"The", V_"weather", V_"today", V_"is", V_"really"]

Compute Q for ONLY "really", attend to entire K_cache/V_cache
→ predict next token: "nice"


═══ Step 2: Generate "nice" (Decode Phase) ═══

Compute K,V for ONLY "nice":
  K_new = K_"nice"
  V_new = V_"nice"

Append to cache:
  K_cache = [K_"The", K_"weather", K_"today", K_"is", K_"really", K_"nice"]
  V_cache = [V_"The", V_"weather", V_"today", V_"is", V_"really", V_"nice"]

Compute Q for ONLY "nice", attend to entire K_cache/V_cache
→ predict next token: "outside"
```

### What Changed?

| Aspect | Without KV Cache | With KV Cache |
|--------|-----------------|---------------|
| K,V computation per step | All tokens (growing) | 1 token (constant) |
| Q computation per step | All tokens (growing) | 1 token (constant) |
| Attention computation | Q(all) × K(all) | Q(1) × K(all cached) |
| Total K,V computations | $O(n^2)$ | $O(n)$ |
| Memory usage | Low (recompute each time) | High (store all K,V) |

The trade-off is classic in computer science: **we trade memory for speed**. We use more memory to store the cache, but we avoid an enormous amount of redundant computation.

### The Two Phases of Inference

KV Cache creates a natural split in the inference process:

**1. Prefill Phase (Processing the prompt)**
- Process all prompt tokens in parallel (like training)
- Compute and cache K, V for every prompt token
- This is compute-bound (lots of matrix multiplications)
- Happens once at the start

**2. Decode Phase (Generating new tokens)**
- Process one new token at a time
- Compute K, V only for the new token, append to cache
- Compute attention between the new Q and all cached K, V
- This is memory-bound (loading the large KV cache from memory)
- Repeats for every generated token

Understanding this split is important because the two phases have very different computational profiles, and optimizing them requires different strategies.

### Under the Hood: What Exactly Is Stored?

Let's be very precise about what the KV cache contains. For a model with $L$ layers, $h$ attention heads, and head dimension $d$, the cache at sequence position $t$ stores:

```
KV Cache structure (per sequence):

Layer 0:
  Head 0: K[0:t] shape=(t, d), V[0:t] shape=(t, d)
  Head 1: K[0:t] shape=(t, d), V[0:t] shape=(t, d)
  ...
  Head h-1: K[0:t] shape=(t, d), V[0:t] shape=(t, d)

Layer 1:
  Head 0: K[0:t] shape=(t, d), V[0:t] shape=(t, d)
  ...

...

Layer L-1:
  Head 0: K[0:t] shape=(t, d), V[0:t] shape=(t, d)
  ...
  Head h-1: K[0:t] shape=(t, d), V[0:t] shape=(t, d)
```

In most implementations, this is stored as a 4D tensor of shape `[L, 2, h, t, d]` where the `2` dimension holds K and V respectively. As each new token is decoded, a single slice of shape `[L, 2, h, 1, d]` is computed and appended along the `t` dimension.

### A Concrete Code Walkthrough

Here's a simplified Python/PyTorch-style pseudocode showing what happens with and without KV Cache:

```python
# ═══ WITHOUT KV Cache (naive) ═══
def generate_naive(model, prompt_tokens, max_new_tokens):
    tokens = prompt_tokens
    for _ in range(max_new_tokens):
        # Recompute EVERYTHING from scratch every step
        Q = model.compute_Q(tokens)    # shape: (seq_len, d)
        K = model.compute_K(tokens)    # shape: (seq_len, d)  ← REDUNDANT
        V = model.compute_V(tokens)    # shape: (seq_len, d)  ← REDUNDANT

        # Full attention over all tokens
        attn = softmax(Q @ K.T / sqrt(d)) @ V
        next_token = model.predict(attn[-1])  # only need last position
        tokens = concat(tokens, next_token)
    return tokens


# ═══ WITH KV Cache (optimized) ═══
def generate_with_kv_cache(model, prompt_tokens, max_new_tokens):
    # Prefill: process entire prompt at once
    Q = model.compute_Q(prompt_tokens)
    K_cache = model.compute_K(prompt_tokens)
    V_cache = model.compute_V(prompt_tokens)
    attn = softmax(Q @ K_cache.T / sqrt(d)) @ V_cache
    next_token = model.predict(attn[-1])

    tokens = concat(prompt_tokens, next_token)
    for _ in range(max_new_tokens - 1):
        # Decode: only compute for the NEW token
        q = model.compute_Q(next_token)     # shape: (1, d) ← just 1 token!
        k_new = model.compute_K(next_token) # shape: (1, d)
        v_new = model.compute_V(next_token) # shape: (1, d)

        # Append to cache
        K_cache = concat(K_cache, k_new)    # grows by 1 each step
        V_cache = concat(V_cache, v_new)

        # Attention: new query against ALL cached keys/values
        attn = softmax(q @ K_cache.T / sqrt(d)) @ V_cache  # (1, d)
        next_token = model.predict(attn)
        tokens = concat(tokens, next_token)
    return tokens
```

Notice the critical difference: in the cached version, `compute_K` and `compute_V` operate on a **single token** during decoding, while in the naive version they operate on the **entire growing sequence**.

### Computational Complexity: A Detailed Breakdown

Let's be precise about the savings. For a sequence of total length $n$ (prompt + generated tokens), with model dimension $d$ and $L$ layers:

**Without KV Cache:**
- At step $t$, K/V computation: $O(t \cdot d^2)$ per layer (linear projections for $t$ tokens)
- At step $t$, attention: $O(t^2 \cdot d)$ per layer
- Total across all $n$ steps: $O(n^2 \cdot d^2 + n^3 \cdot d)$ per layer

**With KV Cache:**
- At step $t$, K/V computation: $O(d^2)$ per layer (only 1 token)
- At step $t$, attention: $O(t \cdot d)$ per layer (1 query against $t$ keys)
- Total across all $n$ steps: $O(n \cdot d^2 + n^2 \cdot d)$ per layer

The K/V computation savings alone reduce a quadratic term to linear. For Llama 2 7B generating 4096 tokens:

```
Without cache: ~4096² = 16.7M K/V forward passes (per layer)
With cache:    ~4096   = 4K K/V forward passes (per layer)

Speedup for K/V computation: ~4096x
```

The attention computation itself is still $O(n^2)$ total (because each step attends to a growing cache), but this is unavoidable — you need to compare against all previous tokens.

## How Much Memory Does KV Cache Use?

This is where things get interesting — and concerning. The KV cache can consume a **massive** amount of memory.

### The Formula

For a single sequence, the KV cache size is:

$$\text{KV Cache Size} = 2 \times n_{\text{layers}} \times n_{\text{heads}} \times d_{\text{head}} \times n_{\text{tokens}} \times \text{bytes per element}$$

The "2" is for K and V (we store both).

### Concrete Example: Llama 2 7B

Let's calculate for Llama 2 7B with a 4096-token sequence:

```
Parameters:
  - Layers: 32
  - Attention heads: 32
  - Head dimension: 128
  - Sequence length: 4096
  - Precision: FP16 (2 bytes per element)

KV Cache = 2 × 32 × 32 × 128 × 4096 × 2 bytes
         = 2 × 32 × 32 × 128 × 4096 × 2
         = 2,147,483,648 bytes
         ≈ 2 GB per sequence
```

**2 GB just for the KV cache of a single sequence!** The model weights themselves are about 14 GB in FP16. Now imagine you're serving 32 concurrent users — that's 64 GB just for KV caches, far exceeding the model weights.

### For Larger Models

| Model | Layers | Heads | Head Dim | KV Cache per 4K tokens |
|-------|--------|-------|----------|----------------------|
| Llama 2 7B | 32 | 32 | 128 | ~2 GB |
| Llama 2 13B | 40 | 40 | 128 | ~3.2 GB |
| Llama 2 70B | 80 | 64 | 128 | ~10 GB |
| GPT-3 175B | 96 | 96 | 128 | ~18 GB |

This memory pressure is why KV cache optimization has become one of the hottest research areas in LLM inference.

## KV Cache Optimization Techniques

Given the enormous memory cost, researchers have developed several techniques to make KV cache more efficient.

### 1. Multi-Query Attention (MQA)

**Idea**: Instead of each attention head having its own K and V, **all heads share a single set of K and V**. Each head still has its own Q.

```
Standard Multi-Head Attention (MHA):
  Head 1: Q₁, K₁, V₁
  Head 2: Q₂, K₂, V₂
  Head 3: Q₃, K₃, V₃
  Head 4: Q₄, K₄, V₄
  → KV cache stores: K₁,V₁, K₂,V₂, K₃,V₃, K₄,V₄  (8 tensors)

Multi-Query Attention (MQA):
  Head 1: Q₁, K_shared, V_shared
  Head 2: Q₂, K_shared, V_shared
  Head 3: Q₃, K_shared, V_shared
  Head 4: Q₄, K_shared, V_shared
  → KV cache stores: K_shared, V_shared              (2 tensors)
```

**Memory reduction**: If you have 32 heads, MQA reduces KV cache by **32x**. The trade-off is a small quality degradation since heads can no longer specialize their K/V representations.

Used in: PaLM, Falcon, StarCoder.

### 2. Grouped-Query Attention (GQA)

**Idea**: A middle ground between MHA and MQA. Instead of all heads sharing one K/V or each head having its own, **groups of heads share K/V**.

```
Grouped-Query Attention (GQA) with 2 KV groups:
  Group 1: Q₁, Q₂ share K_group1, V_group1
  Group 2: Q₃, Q₄ share K_group2, V_group2
  → KV cache stores: K_g1,V_g1, K_g2,V_g2           (4 tensors)
```

**Memory reduction**: With $g$ groups instead of $h$ heads, reduction is $h/g$ times. For Llama 2 70B with 8 KV groups and 64 Q heads, that's an **8x reduction**.

GQA preserves most of the quality of MHA while getting most of the memory savings of MQA. It's currently the most popular approach.

Used in: Llama 2 70B, Llama 3, Mistral, Gemma.

### 3. KV Cache Quantization

**Idea**: Store cached K and V in lower precision. Instead of FP16 (16 bits), use INT8 (8 bits) or even INT4 (4 bits).

```
FP16 KV Cache:  2 bytes per element → 2 GB for Llama 7B @ 4K tokens
INT8 KV Cache:  1 byte per element  → 1 GB (50% reduction)
INT4 KV Cache:  0.5 bytes           → 0.5 GB (75% reduction)
```

The key insight is that K and V values don't need full precision for the attention computation to work well. Research shows that INT8 quantization of KV cache causes negligible quality loss, while INT4 requires more careful calibration.

An important nuance: **K and V have different quantization sensitivities.** Keys are used in dot products that directly determine attention scores, so they're more sensitive to precision errors. Values are combined via weighted sums, which are more forgiving. Some advanced schemes (like KVQuant) use **mixed precision** — e.g., INT4 for V and INT8 for K — to maximize compression while preserving quality.

```python
# Pseudocode: asymmetric KV cache quantization
k_cache = quantize(k_new, bits=8)     # INT8 for keys (sensitive)
v_cache = quantize(v_new, bits=4)     # INT4 for values (less sensitive)

# During attention: dequantize on-the-fly
k_dequant = dequantize(k_cache)       # back to FP16 for attention math
v_dequant = dequantize(v_cache)
attn_output = softmax(q @ k_dequant.T / sqrt(d)) @ v_dequant
```

### 4. Paged Attention (vLLM)

**Idea**: Inspired by **virtual memory** in operating systems. Instead of allocating a contiguous block of memory for each sequence's KV cache, divide the cache into fixed-size **pages** and allocate them on demand.

```
Traditional KV Cache allocation:
  Sequence 1: [████████████░░░░░░░░]  ← Pre-allocated for max length
  Sequence 2: [██████░░░░░░░░░░░░░░]  ← Lots of wasted space!
  Sequence 3: [██████████████░░░░░░]

PagedAttention:
  Pages: [██][██][██][██][██][██][██][██][██]...
  Seq 1 uses pages: 1, 4, 7, 9, 11, 13  (non-contiguous, no waste)
  Seq 2 uses pages: 2, 5, 8              (grows as needed)
  Seq 3 uses pages: 3, 6, 10, 12, 14    (pages shared if possible)
```

Benefits:
- **Near-zero memory waste** (no pre-allocation for max sequence length)
- **Memory sharing**: Common prefixes (like system prompts) can share KV cache pages across requests
- **Dynamic allocation**: Memory grows with actual sequence length

This technique, introduced by the vLLM project, improved serving throughput by **2-4x** compared to naive allocation and is now standard in production LLM serving.

### 5. Sliding Window Attention

**Idea**: Instead of caching K/V for **all** previous tokens, only keep the most recent $w$ tokens. This caps the KV cache size regardless of sequence length.

```
Full attention (cache grows forever):
  Token 100 attends to: tokens 1-99    (99 cached KV pairs)
  Token 1000 attends to: tokens 1-999  (999 cached KV pairs)

Sliding window (w=256):
  Token 100 attends to: tokens 1-99    (99 cached KV pairs)
  Token 1000 attends to: tokens 744-999 (256 cached KV pairs, capped!)
```

The assumption is that for many tasks, the most relevant context is recent. Information from very early tokens can still propagate through the network via the stacking of multiple layers (each layer's sliding window lets information "hop" further back).

Used in: Mistral 7B (window size = 4096).

### 6. Token Eviction / Pruning

**Idea**: Not all cached tokens are equally important. Identify and evict less important tokens from the cache based on attention scores.

Techniques like **H2O (Heavy-Hitter Oracle)** observe that a small fraction of tokens (called "heavy hitters") receive most of the attention across all layers. By keeping only these high-attention tokens plus recent tokens, you can dramatically reduce cache size with minimal quality loss.

```
Full cache: [The] [weather] [today] [is] [really] [quite] [nice] [and] [pleasant] [outside]

H2O cache (budget=6): [The] [weather] [is] [nice] [pleasant] [outside]
                       ↑ heavy hitters (high attention)    ↑ recent tokens
                       Evicted: [today], [really], [quite], [and]
```

### 7. Prefix Caching (Prompt Caching)

**Idea**: Many requests share the same prefix — system prompts, few-shot examples, or shared document contexts. Instead of recomputing KV cache for these common prefixes every time, **cache and reuse them across requests**.

```
Request 1: [System prompt] + "What is Python?"
Request 2: [System prompt] + "Explain recursion."
Request 3: [System prompt] + "How does HTTP work?"

Without prefix caching:
  Request 1: Prefill 500 system tokens + 4 user tokens   = 504 token prefill
  Request 2: Prefill 500 system tokens + 3 user tokens   = 503 token prefill
  Request 3: Prefill 500 system tokens + 5 user tokens   = 505 token prefill
  Total: 1,512 token prefills

With prefix caching:
  First request: Prefill 500 system tokens + cache them
  Request 1: Load cached 500 tokens + prefill 4 tokens   = 4 token prefill
  Request 2: Load cached 500 tokens + prefill 3 tokens   = 3 token prefill
  Request 3: Load cached 500 tokens + prefill 5 tokens   = 5 token prefill
  Total: 512 token prefills (66% reduction!)
```

This is especially powerful for production APIs where every request starts with the same long system prompt. SGLang's **RadixAttention** takes this further by organizing cached prefixes in a radix tree data structure, enabling efficient lookup and sharing of any common prefix — not just system prompts.

### 8. Multi-Level Caching (GPU/CPU/Disk)

**Idea**: When GPU memory is full, instead of discarding KV cache entries, **offload them to CPU RAM or even disk**, and bring them back when needed.

```
┌──────────────────────────┐
│   GPU HBM (fastest)       │  ← Active sequences' KV cache
│   ~80 GB on A100          │
├──────────────────────────┤
│   CPU RAM (medium)        │  ← Paused/preempted sequences
│   ~512 GB typical server  │
├──────────────────────────┤
│   NVMe SSD (slow)         │  ← Long-idle sequences
│   ~TB scale               │
└──────────────────────────┘
```

This hierarchical approach allows serving systems to handle far more concurrent users than GPU memory alone would allow. The trade-off is latency: resuming a sequence from CPU RAM or disk requires transferring the cache back to GPU, adding latency to the first token.

## Putting It All Together: A Complete Picture

Here's how these techniques combine in a modern LLM serving stack:

```
┌─────────────────────────────────────────────────────┐
│                   User Request                       │
│         "Explain quantum computing simply"           │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              PREFILL PHASE                           │
│                                                      │
│  Process all prompt tokens in parallel               │
│  Compute K,V for each token at each layer            │
│  Store in KV Cache (using PagedAttention)            │
│  Predict first output token                          │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              DECODE PHASE (repeated)                  │
│                                                      │
│  For each new token:                                 │
│    1. Compute Q, K, V for new token only             │
│    2. Append K, V to cache (quantized to INT8)       │
│    3. Attend: new Q × all cached K → scores          │
│    4. Weighted sum of cached V → attention output     │
│    5. Rest of transformer → next token prediction     │
│                                                      │
│  Cache uses GQA (8 KV groups instead of 64 heads)    │
│  Pages allocated dynamically via PagedAttention       │
│  Old low-attention tokens evicted if memory tight     │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│              Generated Response                      │
│  "Quantum computing uses quantum bits (qubits)..."   │
└─────────────────────────────────────────────────────┘
```

## Common Questions

### Why not cache Q as well?

Q (Query) represents **what the current token is looking for**. It's only used once — to compute attention with all cached K vectors. After that, the current token's Q is never needed again because future tokens will compute their own Q vectors. So caching Q would waste memory with no benefit.

K and V are different: they represent information **about** a token that future tokens will need to query against. Every future token needs to attend to past K and V, so caching them saves repeated computation.

### Does KV Cache affect model output?

No. KV Cache is a **mathematically exact** optimization. The output is identical with or without the cache — we're simply avoiding redundant computation, not approximating anything.

(Note: some optimizations built on top of KV cache, like quantization or token eviction, do introduce approximations. But the basic KV cache itself is lossless.)

### What about during training?

KV Cache is only used during **inference** (generation). During training, the model processes the entire sequence at once in parallel (using teacher forcing), so there's no sequential token-by-token generation and no redundant computation to cache.

### Why is the decode phase memory-bound?

During decoding, we process a single token, which means very little computation (small matrix multiplications). But we need to load the **entire KV cache** from GPU memory to compute attention. For a long sequence, the KV cache can be gigabytes, and loading it from HBM (High Bandwidth Memory) becomes the bottleneck — not the actual math.

Let's quantify this with an example:

```
Llama 3.1 8B, sequence length = 4K, FP16 KV cache

KV cache size: ~0.5 GB
Compute for one decode step: ~16 GFLOPS
A100 HBM bandwidth: 2 TB/s
A100 compute: 312 TFLOPS (FP16)

Time to load KV cache: 0.5 GB / 2 TB/s = 0.25 ms
Time to compute: 16 GFLOPS / 312 TFLOPS = 0.00005 ms

Ratio: memory loading takes 5000x longer than compute!
```

This is why techniques that reduce KV cache size (GQA, quantization, eviction) directly improve decode throughput: less data to load from memory per step. The decode phase is almost entirely bottlenecked on how fast you can read the KV cache from GPU HBM.

### How does KV Cache interact with batch size?

When serving multiple requests simultaneously, each request has its own KV cache. The total memory is:

```
Total KV Cache Memory = batch_size × per_sequence_KV_cache_size
```

This means KV cache is often the **primary limiter of batch size** in production. A larger batch improves GPU utilization (more compute per memory load), but each additional sequence in the batch requires its own KV cache allocation. This is the fundamental tension in LLM serving: you want large batches for throughput, but each batch slot costs significant memory.

### What happens when KV cache runs out of memory?

Serving frameworks handle this differently:

- **vLLM**: Preempts (pauses) lower-priority sequences, either swapping their KV cache to CPU RAM or discarding it entirely (to be recomputed later)
- **SGLang**: Evicts least-recently-used prefix cache entries from the radix tree, then preempts sequences if needed
- **TensorRT-LLM**: Queues incoming requests until memory is available

In all cases, running out of KV cache memory means **lower throughput** (fewer concurrent sequences) or **higher latency** (queuing/preemption). This is why capacity planning for KV cache is critical.

### Can KV Cache be shared across users?

Yes, but only for **common prefixes**. If two users have the same system prompt, the KV cache for that system prompt is identical and can be shared (this is what prefix caching does). However, once the sequences diverge (different user messages), each needs its own KV cache from that point forward.

This has architectural implications: if you design your prompts so that the common prefix is as long as possible (system prompt first, then few-shot examples, then user input last), you maximize sharing opportunities.

### How does KV Cache work with beam search?

In beam search, you maintain multiple candidate sequences that share a common prefix but diverge at different points. KV cache management becomes interesting here:

```
Beam search with 3 beams:

Step 1: All beams share prefix "The weather is"
  KV cache: 1 shared copy (copy-on-write)

Step 2: Beams diverge:
  Beam 1: "The weather is really nice"
  Beam 2: "The weather is quite warm"
  Beam 3: "The weather is absolutely beautiful"
  KV cache: shared prefix + 3 separate suffixes

Step 3: Beam 3 is pruned (lowest score)
  KV cache: free beam 3's suffix, keep shared prefix
```

vLLM implements **copy-on-write** semantics: beams share KV cache blocks via reference counting, and only allocate new blocks when beams write different tokens. This avoids duplicating the shared prefix across all beams.

## Best Practices for KV Cache in Production

Based on experience deploying LLMs in production, here are practical recommendations:

### 1. Right-Size Your Max Sequence Length

The single biggest lever for KV cache memory is the **maximum sequence length** you configure. Many deployments set this to the model's maximum (e.g., 128K for Llama 3.1) by default, but most requests are much shorter.

```
Example: Serving Llama 3.1 8B

  Max context = 128K tokens:
    KV cache per sequence ≈ 32 GB (!) — only 2 sequences on an 80GB A100

  Max context = 8K tokens (covers 95% of your actual traffic):
    KV cache per sequence ≈ 2 GB — 20+ concurrent sequences on an 80GB A100

  Impact: 10x more throughput by matching config to actual usage
```

**Best practice**: Analyze your actual request length distribution. Set the max sequence length to cover your P95 or P99, not the theoretical maximum. Use request routing to send the rare long-context requests to a separate, appropriately configured pool.

### 2. Use GQA Models When Possible

If you're choosing which model to deploy, **prefer models with Grouped-Query Attention** over standard Multi-Head Attention. The KV cache reduction directly translates to higher serving throughput.

| Model | Attention Type | KV Cache Reduction |
|-------|---------------|-------------------|
| Llama 2 7B | MHA (32 KV heads) | 1x (baseline) |
| Llama 3 8B | GQA (8 KV groups) | 4x smaller |
| Mistral 7B | GQA (8 KV groups) | 4x smaller |
| Llama 3.1 70B | GQA (8 KV groups) | 8x smaller |

At similar quality levels, a GQA model can serve 4-8x more concurrent users. This is often the difference between needing 1 GPU vs. 4 GPUs for a given traffic level.

### 3. Enable KV Cache Quantization in Your Serving Framework

Most modern serving frameworks support KV cache quantization as a simple configuration flag:

```python
# vLLM example
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    kv_cache_dtype="fp8_e5m2",  # FP8 KV cache — ~50% memory reduction
    # kv_cache_dtype="auto",    # default: same precision as model
)
```

```python
# SGLang example
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --kv-cache-dtype fp8_e5m2
```

**Best practice**: Start with FP8 quantization — it provides ~50% KV cache memory savings with negligible quality loss for virtually all use cases. Only go to INT4 if you've benchmarked quality on your specific task and confirmed acceptable degradation.

### 4. Use PagedAttention-Based Serving Frameworks

If you're deploying LLMs in production, **always use a framework that implements PagedAttention** (vLLM, SGLang, TensorRT-LLM). The memory efficiency gains are significant and essentially free:

```
Memory utilization comparison for 100 concurrent requests:

  Naive allocation (HuggingFace generate):
    Pre-allocates max_seq_len per request
    Memory waste: 60-80% (most sequences don't reach max length)
    Effective batch size: limited by worst-case allocation

  PagedAttention (vLLM/SGLang):
    Allocates pages on demand as sequences grow
    Memory waste: <4%
    Effective batch size: 2-4x larger for same GPU memory
```

Never use `model.generate()` from HuggingFace Transformers for production serving. It's designed for simplicity, not efficiency.

### 5. Leverage Prefix Caching for Repeated Prefixes

If your application uses the same system prompt for all requests (which most do), enable prefix caching:

```python
# vLLM with automatic prefix caching
llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    enable_prefix_caching=True,  # automatic prefix sharing
)

# SGLang has RadixAttention enabled by default — no config needed
```

**When this matters most:**
- Long system prompts (1000+ tokens): saves significant prefill compute
- Few-shot examples in every request: shared across all users
- RAG applications with shared document context: common prefix from retrieved docs

**When it doesn't help:**
- Every request has a unique prefix (no sharing opportunity)
- Very short prompts (overhead of cache management exceeds savings)

### 6. Monitor KV Cache Usage in Production

KV cache is often the bottleneck that limits throughput. Monitor these metrics:

```
Key metrics to track:

1. KV Cache Utilization (%)
   - How full is the KV cache memory pool?
   - Alert at >85% — approaching OOM, requests will queue

2. Prefill vs. Decode Latency
   - Prefill TTFT (Time to First Token): indicates compute bottleneck
   - Decode TPOT (Time per Output Token): indicates memory bandwidth bottleneck
   - If TPOT is high, KV cache is too large → reduce max_seq_len or enable quantization

3. Request Preemption Rate
   - How often are sequences evicted from GPU to make room?
   - High preemption = GPU memory too small for your traffic pattern

4. Cache Hit Rate (if using prefix caching)
   - How often are prefixes reused?
   - Low hit rate = prefix caching overhead without benefit → consider disabling
```

### 7. Chunked Prefill for Better Latency

For long prompts, the prefill phase can take seconds, blocking decode steps for other sequences. **Chunked prefill** breaks long prompts into smaller chunks that are interleaved with decode steps:

```
Without chunked prefill:
  Time: |------ Long prefill (2s) ------||decode|decode|decode|...
  Other sequences: BLOCKED for 2 seconds (latency spike!)

With chunked prefill (chunk_size=512):
  Time: |prefill chunk 1|decode|prefill chunk 2|decode|prefill chunk 3|decode|...
  Other sequences: get decode slots between chunks (smooth latency)
```

Both vLLM and SGLang support this. In SGLang it's called `--chunked-prefill-size`.

### 8. Speculative Decoding: Reduce the KV Cache Bandwidth Tax

Since the decode phase is memory-bandwidth-bound (loading the KV cache is the bottleneck, not compute), **speculative decoding** helps by generating multiple tokens per KV cache load:

```
Standard decoding:
  Load KV cache → generate 1 token → load KV cache → generate 1 token → ...
  KV cache loads: N loads for N tokens

Speculative decoding:
  Draft model (small, fast) proposes 5 tokens: [t1, t2, t3, t4, t5]
  Target model verifies all 5 in ONE forward pass (loading KV cache once)
  Accept first 3 that match: [t1, t2, t3] ← 3 tokens for 1 KV cache load!

  Effective KV cache loads: N/acceptance_rate (typically 2-4x fewer)
```

This doesn't reduce KV cache memory, but it reduces how often the cache needs to be loaded, directly improving decode throughput.

### 9. Choose the Right Batch Size

Batching multiple sequences together is key to efficient GPU utilization, but KV cache memory limits how many sequences you can batch:

```
GPU memory budget = Model weights + KV Cache(batch_size × seq_len) + Activations

Example: 80GB A100 with Llama 3.1 8B (FP16)
  Model weights: ~16 GB
  Activations:   ~2 GB
  Available for KV cache: ~62 GB

  With 4K max seq_len: batch_size ≈ 62 GB / (0.5 GB/seq) = 124 sequences
  With 32K max seq_len: batch_size ≈ 62 GB / (4 GB/seq) = 15 sequences
  With 128K max seq_len: batch_size ≈ 62 GB / (16 GB/seq) = 3 sequences
```

**Best practice**: Use **continuous batching** (implemented in vLLM/SGLang) instead of static batching. Continuous batching adds new requests to the batch as soon as existing ones finish, keeping GPU utilization high even when sequences have different lengths.

### 10. Multi-GPU Strategies: Tensor Parallel vs. Pipeline Parallel

For large models that don't fit on a single GPU, the parallelism strategy affects KV cache distribution:

```
Tensor Parallelism (TP):
  - Splits attention heads across GPUs
  - Each GPU stores KV cache for its assigned heads
  - KV cache is naturally distributed
  - Best for: latency-sensitive serving (all GPUs work on every token)

Pipeline Parallelism (PP):
  - Splits layers across GPUs
  - Each GPU stores KV cache for its assigned layers
  - KV cache is naturally distributed
  - Best for: throughput-oriented serving (GPUs work on different sequences)

Example: Llama 3.1 70B on 4× A100 80GB
  TP=4: Each GPU stores KV for 16 of 64 layers → ~2.5 GB per GPU per seq
  PP=4: Each GPU stores KV for 20 of 80 layers → ~2.5 GB per GPU per seq
  TP=2, PP=2: Hybrid — balanced latency and throughput
```

## Deep Dive: KV Cache in Real-World Serving Systems

Let's look at how major serving frameworks implement KV cache management:

### vLLM

vLLM pioneered PagedAttention and remains one of the most popular serving frameworks. Its KV cache management:

- **Block manager**: Allocates KV cache in fixed-size blocks (default 16 tokens per block)
- **Copy-on-write**: For beam search and parallel sampling, shared prefixes use reference counting — copies only happen when a sequence diverges
- **Preemption**: When GPU memory is full, sequences can be swapped to CPU or recomputed later
- **Prefix caching**: Maintains a hash table of block contents for automatic prefix sharing

### SGLang

SGLang focuses on complex LLM programs with multiple generation calls. Its distinctive feature:

- **RadixAttention**: Organizes all cached KV blocks in a radix tree, where shared prefixes form common branches. This naturally supports sharing at any granularity — not just the system prompt, but any common substring across requests
- **Automatic cache management**: LRU eviction of least-recently-used tree branches
- **Efficient for multi-turn**: In chat applications, previous turns' KV cache is automatically reused

### TensorRT-LLM (NVIDIA)

NVIDIA's optimized inference framework:

- **KV cache manager with paged blocks**: Similar to vLLM but with NVIDIA-specific CUDA kernel optimizations
- **In-flight batching**: Combines continuous batching with chunked prefill
- **FP8 KV cache**: Native support for Hopper GPU's FP8 format, maximizing HBM bandwidth

## Summary

KV Cache is the foundation that makes autoregressive LLM inference practical:

| Concept | What It Does |
|---------|-------------|
| **KV Cache** | Stores K,V from previous tokens to avoid recomputation |
| **Prefill** | Processes the full prompt, populates the initial cache |
| **Decode** | Generates tokens one by one, appending to cache each step |
| **GQA/MQA** | Reduces cache size by sharing K,V across attention heads |
| **Quantization** | Stores cache in lower precision (FP8/INT4) |
| **PagedAttention** | Manages cache memory like OS virtual memory pages |
| **Sliding Window** | Caps cache size by only keeping recent tokens |
| **Token Eviction** | Removes low-importance tokens from cache |
| **Prefix Caching** | Reuses KV cache for shared prefixes across requests |
| **Chunked Prefill** | Breaks long prefills into chunks to avoid decode latency spikes |
| **Speculative Decoding** | Generates multiple tokens per KV cache load |

The fundamental trade-off is **memory for speed**: KV cache consumes significant GPU memory but eliminates the quadratic recomputation cost, making real-time LLM generation possible.

The key takeaways for practitioners:

1. **Always use a PagedAttention-based framework** (vLLM, SGLang, TensorRT-LLM) for production serving
2. **Right-size your max sequence length** to match actual traffic, not theoretical maximum
3. **Enable FP8 KV cache quantization** — it's nearly free quality-wise and halves cache memory
4. **Prefer GQA models** when choosing which model to deploy — 4-8x KV cache reduction
5. **Enable prefix caching** if your requests share common prefixes (system prompts, few-shot examples)
6. **Monitor KV cache utilization** — it's usually the bottleneck before anything else

Every major optimization in LLM serving — from vLLM's PagedAttention to Mistral's sliding window to SGLang's RadixAttention — is ultimately about making this cache more efficient. Understanding KV cache deeply is essential for anyone building or operating LLM systems at scale.

## Interview Questions and Answers

### Q: What is KV Cache and why is it necessary for LLM inference?

KV Cache stores the Key and Value vectors computed for all previously processed tokens during autoregressive generation, so they don't need to be recomputed at each decoding step. Without KV Cache, generating each new token requires recomputing K and V for the entire sequence from scratch — the total work grows quadratically as $O(n^2)$ with sequence length. With KV Cache, each decode step only computes K and V for the single new token and appends them to the cache, reducing total K/V computation to $O(n)$. For a 4096-token generation, this eliminates ~4096x redundant computations per layer. The trade-off is memory: the cache must be stored in GPU memory, and for large models with long sequences, this can consume tens of gigabytes.

### Q: Walk through exactly what happens during the prefill and decode phases.

**Prefill phase**: The entire prompt is processed in a single forward pass. All tokens are run through the model in parallel (like training). Q, K, V are computed for every prompt token, the full attention matrix is computed, and the KV cache is populated with all prompt tokens' K and V. This phase is **compute-bound** — dominated by large matrix multiplications on the full prompt. Output: the first generated token, plus a populated KV cache.

**Decode phase**: Runs once per generated token. Only the single new token is passed through the model. Q, K, V are computed for that one token. K and V are appended to the cache. Attention is computed between the new Q (1 token) and all cached K/V (entire history). This phase is **memory-bandwidth-bound** — the main cost is loading the large KV cache from GPU HBM, not the tiny matrix multiplications for a single token. The compute-to-memory ratio (arithmetic intensity) is extremely low, which is why decode throughput is primarily limited by memory bandwidth, not FLOPS.

### Q: How do you calculate the KV cache memory for a given model?

The formula is:

$$\text{KV Cache} = 2 \times L \times n_\text{kv\_heads} \times d_\text{head} \times n_\text{tokens} \times \text{bytes}$$

Where: 2 is for K and V, $L$ is the number of layers, $n_\text{kv\_heads}$ is the number of KV heads (equals $n_\text{heads}$ for MHA, fewer for GQA), $d_\text{head}$ is the head dimension, $n_\text{tokens}$ is the sequence length, and bytes is the per-element size (2 for FP16, 1 for FP8).

**Concrete example — Llama 3.1 8B** ($L=32$, $n_\text{kv\_heads}=8$ (GQA), $d_\text{head}=128$, FP16):
- Per token: $2 \times 32 \times 8 \times 128 \times 2 = 131,072$ bytes = 128 KB
- Per 4K sequence: $128 \text{ KB} \times 4096 = 512$ MB
- Per 128K sequence: $128 \text{ KB} \times 131072 = 16$ GB

For batch serving, multiply by the number of concurrent sequences. This is why KV cache is usually the primary bottleneck for serving throughput.

### Q: Explain Multi-Query Attention (MQA), Grouped-Query Attention (GQA), and their impact on KV cache.

**MHA (Multi-Head Attention)**: Each of $h$ attention heads has its own Q, K, V projections. KV cache stores $h$ independent K and $h$ independent V tensors per layer. Full expressiveness, maximum memory cost.

**MQA (Multi-Query Attention)**: All heads share a single K and V projection, but each head has its own Q. KV cache is reduced by $h\times$ (e.g., 32x for 32 heads). The quality trade-off is that heads can no longer specialize their key/value representations, which can hurt performance on complex tasks.

**GQA (Grouped-Query Attention)**: Compromise — $h$ query heads are divided into $g$ groups, each group sharing one K/V projection. KV cache is reduced by $h/g$ times. Example: Llama 3 8B has 32 query heads and 8 KV groups → 4x cache reduction. GQA retains most of MHA's quality while capturing most of MQA's memory savings. It's the current industry standard — used by Llama 3, Mistral, Gemma 2, and most modern models.

The practical impact is enormous: with GQA, you can serve 4-8x more concurrent users on the same hardware, or equivalently, support 4-8x longer sequences.

### Q: What is PagedAttention and why was it a breakthrough for LLM serving?

Before PagedAttention (introduced by vLLM), serving frameworks pre-allocated a contiguous memory block for each sequence's KV cache sized to the maximum possible sequence length. If you set max_seq_len=4096 but the average response is 200 tokens, you waste ~95% of the allocated memory. Worse, memory fragmentation made it impossible to fully utilize GPU memory.

PagedAttention borrows the concept of **virtual memory paging** from operating systems. KV cache is divided into fixed-size **blocks** (e.g., 16 tokens per block). Blocks are allocated on demand as the sequence grows and can be non-contiguous in physical memory. A block table maps logical sequence positions to physical memory locations.

Key benefits:
1. **Near-zero internal fragmentation** — only the last block of each sequence may be partially filled
2. **No external fragmentation** — any free block can be used by any sequence
3. **Copy-on-write sharing** — common prefixes (system prompts, beam search) share physical blocks via reference counting; copies only happen when content diverges
4. **Dynamic allocation** — memory is consumed proportional to actual sequence length, not maximum possible length

The result was 2-4x higher throughput compared to naive allocation, and PagedAttention is now standard in all production serving frameworks.

### Q: Why is the decode phase memory-bandwidth-bound and not compute-bound?

During decode, the model processes a single token. The compute is tiny — one token's worth of linear projections and one row of attention scores. But the model must load the **entire KV cache** from GPU HBM to compute attention.

Quantitatively for Llama 3.1 8B with a 4K sequence:
- KV cache to load: ~512 MB
- Compute for one decode step: ~16 GFLOPS
- A100 HBM bandwidth: 2 TB/s → loading 512 MB takes ~0.25 ms
- A100 FP16 compute: 312 TFLOPS → 16 GFLOPS takes ~0.00005 ms

The memory load takes 5000x longer than the compute. The GPU's arithmetic units are idle >99.99% of the time during decode, waiting for data from memory. This is called being **memory-bandwidth-bound** — throughput is determined by how fast you can shuttle data from HBM to the compute units.

This explains why every KV cache optimization (GQA, quantization, eviction) directly improves decode speed: less data to load per step. It also explains why **batching** helps — processing multiple sequences together amortizes the cost of loading model weights, improving the arithmetic intensity toward the compute-bound regime.

### Q: What is prefix caching / prompt caching and when is it useful?

Prefix caching stores and reuses the KV cache of common prompt prefixes across multiple requests. If 1000 requests all start with the same 2000-token system prompt, the system computes the KV cache for those 2000 tokens once and shares it across all requests, saving 1000 redundant prefill computations.

**How it works in SGLang (RadixAttention)**: All cached KV blocks are organized in a radix tree keyed by token sequences. When a new request arrives, the system finds the longest matching prefix in the tree, reuses its cached KV blocks, and only runs prefill on the remaining novel tokens. LRU eviction reclaims tree branches when memory is tight.

**When it's most useful**:
- Long, shared system prompts (1000+ tokens) — saves significant prefill compute
- Few-shot examples included in every request — shared across all users
- RAG applications where the same retrieved documents appear across requests
- Multi-turn chat — previous conversation turns are the prefix for new turns

**When it doesn't help**:
- Every request has a unique prompt (no sharing opportunity)
- Very short prompts where caching overhead exceeds savings

### Q: What is sliding window attention and what are its trade-offs?

Sliding window attention limits each token to only attend to the most recent $w$ tokens, rather than the entire sequence. This caps the KV cache at $w$ entries per layer, regardless of total sequence length.

**Memory benefit**: KV cache is bounded at $O(w)$ instead of $O(n)$, enabling arbitrarily long sequences without growing memory.

**How information propagates beyond the window**: Through layer stacking. If each layer has a window of $w$ tokens and the model has $L$ layers, information can theoretically propagate $w \times L$ tokens back through the network. At layer 1, token 1000 sees tokens 744-999. At layer 2, the representation of token 744 already contains information from tokens 488-743. So layer 2 indirectly "sees" back to token 488.

**Trade-offs**:
- Works well for tasks where local context dominates (chat, code completion)
- Can struggle with tasks requiring precise long-range retrieval ("what was the third item on the list from page 1?")
- Mistral 7B uses $w=4096$, which covers most practical use cases
- Some models combine sliding window for most layers with full attention for a few layers, getting the best of both worlds

### Q: How does KV cache quantization work and what are the trade-offs?

KV cache quantization compresses the stored K and V tensors from FP16 (16 bits) to lower precision formats like FP8 (8 bits) or INT4 (4 bits). Values are quantized before storage and dequantized on-the-fly during attention computation.

**Key nuance — K and V have different sensitivities**:
- **Keys** participate in dot products that determine attention scores. Small errors in K can flip which tokens receive attention, causing large output changes. Keys are more sensitive.
- **Values** are combined via weighted summation. Errors in V are averaged out across multiple values, making them more tolerant. Values are less sensitive.

Advanced schemes like KVQuant use mixed precision: INT8 for K, INT4 for V. This maximizes compression while minimizing quality impact.

**Practical guidance**: FP8 KV cache quantization (supported by vLLM, SGLang, TRT-LLM) provides ~50% memory reduction with negligible quality loss on virtually all tasks. INT4 saves 75% memory but requires per-task benchmarking to verify acceptable quality. Always benchmark on your specific workload, not just perplexity.

### Q: How does KV cache interact with batching? What limits batch size?

Each sequence in a batch has its own independent KV cache. Total GPU memory splits into: model weights (fixed) + KV cache (scales with batch_size × seq_len) + activations (small).

$$\text{max batch size} \approx \frac{\text{GPU memory} - \text{model weights} - \text{overhead}}{\text{KV cache per sequence}}$$

For an 80GB A100 serving Llama 3.1 8B (FP16, ~16GB weights):
- 4K max context: $(80 - 16 - 2) / 0.5 \approx 124$ concurrent sequences
- 32K max context: $(80 - 16 - 2) / 4 \approx 15$ concurrent sequences
- 128K max context: $(80 - 16 - 2) / 16 \approx 3$ concurrent sequences

This is why **right-sizing max_seq_len** to your actual traffic distribution (not the model's theoretical maximum) is the single most impactful configuration decision. Setting max_seq_len to your P95 request length instead of the model maximum can increase batch size (and thus throughput) by 10-30x.

**Continuous batching** (vLLM, SGLang) further improves utilization by dynamically adding/removing sequences from the batch as they start/finish, rather than waiting for all sequences in a batch to complete.

### Q: What is speculative decoding and how does it relate to KV cache?

Speculative decoding uses a small, fast **draft model** to propose multiple candidate tokens, then verifies them all at once with the large **target model** in a single forward pass. If the draft model proposes 5 tokens and 3 are accepted, you generate 3 tokens for the cost of 1 KV cache load (plus the cheap draft model overhead).

**Relation to KV cache**: Since decode is memory-bandwidth-bound (dominated by KV cache loading), speculative decoding amortizes that loading cost across multiple tokens. It doesn't reduce KV cache memory, but reduces the number of times the cache must be loaded from HBM. If the acceptance rate is $\alpha$ and the draft length is $k$, the effective tokens per KV cache load is approximately $1/(1-\alpha)$ on average.

**When it helps most**: Memory-bandwidth-bound scenarios (single-sequence generation, long contexts, large models). It helps less when the serving is already compute-bound (large batch sizes) because the draft verification adds compute.

### Q: Compare the KV cache management strategies of vLLM, SGLang, and TensorRT-LLM.

| Feature | vLLM | SGLang | TensorRT-LLM |
|---------|------|--------|---------------|
| **Core KV strategy** | PagedAttention with block manager | RadixAttention (radix tree of cached blocks) | Paged blocks with CUDA-optimized kernels |
| **Prefix caching** | Hash-based block matching (opt-in) | Radix tree (automatic, any-depth matching) | Supported via KV cache reuse |
| **Memory pressure** | Preempt (swap to CPU or recompute) | LRU eviction of radix tree branches | Request queuing |
| **Beam search** | Copy-on-write block sharing | Fork/join on tree branches | Block-level sharing |
| **KV quantization** | FP8, INT8 | FP8 | FP8 (native H100), INT8 |
| **Best for** | General-purpose serving | Multi-turn chat, complex LLM programs | Maximum throughput on NVIDIA hardware |

SGLang's radix tree approach is particularly elegant for multi-turn conversations — when a user sends a follow-up message, the KV cache from the entire previous conversation is automatically reused (it's already in the tree). vLLM requires explicit prefix caching configuration for this.

### Q: How would you diagnose and fix KV cache-related performance problems in production?

**Symptom: High Time-Per-Output-Token (TPOT)**
- Diagnosis: Decode is memory-bandwidth-bound → KV cache loading is the bottleneck
- Fixes: Enable KV cache quantization (FP8), use GQA models, reduce max_seq_len, increase batch size to improve arithmetic intensity

**Symptom: Frequent request preemptions**
- Diagnosis: KV cache memory is exhausted, sequences are being evicted
- Fixes: Reduce max_seq_len, enable KV quantization, add more GPUs, use prefix caching to share memory

**Symptom: High Time-To-First-Token (TTFT)**
- Diagnosis: Prefill phase is too slow (long prompts, compute-bound)
- Fixes: Enable chunked prefill (prevents blocking decode), use tensor parallelism to distribute prefill compute, enable prefix caching for shared prefixes

**Symptom: Low GPU utilization during decode**
- Diagnosis: Batch size too small (arithmetic intensity too low)
- Fixes: Increase batch size (may require KV quantization to fit more sequences), use continuous batching, reduce max_seq_len to allow more concurrent sequences

**Monitoring checklist**: KV cache utilization %, preemption rate, TTFT/TPOT distributions, cache hit rate (if prefix caching enabled), GPU memory breakdown (weights vs. KV cache vs. activations).

### Q: What is chunked prefill and why does it matter for serving latency?

When a long prompt (e.g., 10K tokens) arrives, the prefill phase can monopolize the GPU for seconds. During this time, all other sequences in the batch are blocked from their decode steps, causing latency spikes in their TPOT (Time Per Output Token).

**Chunked prefill** breaks the long prompt into smaller chunks (e.g., 512 or 1024 tokens) and interleaves them with decode steps for other sequences:

```
Without chunked prefill:
  Time: |──── 10K token prefill (2s) ────||decode|decode|...
  Other requests: stuck waiting for 2 seconds

With chunked prefill (chunk=1024):
  Time: |chunk1|decode|chunk2|decode|...|chunk10|decode|decode|...
  Other requests: get regular decode slots (~10ms gaps)
```

The total prefill time is slightly longer (overhead of splitting), but the P99 TPOT for concurrent requests drops dramatically. This is critical for production systems with SLA requirements on per-token latency.

### Q: If you were designing a new LLM architecture, what would you do to minimize KV cache overhead?

This is a system design question that tests understanding of the full stack:

1. **Use GQA with few KV groups** (e.g., 8 groups for 32-64 query heads) — proven to maintain quality with 4-8x cache reduction
2. **Reduce head dimension** if possible — smaller $d_\text{head}$ directly reduces cache per token, but may hurt representation quality
3. **Use a hybrid attention pattern** — full attention for a few layers (for long-range dependencies) + sliding window for the rest (bounded cache). DeepSeek-V2 does this effectively
4. **Support native FP8 KV cache** — design the attention mechanism to be numerically stable with FP8 inputs
5. **Consider Multi-Head Latent Attention (MLA)** — compress K and V into a lower-dimensional latent before caching, as used in DeepSeek-V2. This reduces cache size by up to 93% compared to standard MHA while maintaining quality through learned up-projection during attention
6. **Design for prefix sharing** — ensure the model's positional encoding scheme (e.g., RoPE) allows efficient KV cache reuse for shared prefixes
7. **Consider linear attention variants** — mechanisms like RetNet or Mamba that replace the growing KV cache with a fixed-size recurrent state ($O(1)$ memory per step), at the cost of reduced attention expressiveness

## References

1. Vaswani et al., "Attention Is All You Need" (2017) — The original transformer paper introducing the attention mechanism
2. Shazeer, "Fast Transformer Decoding: One Write-Head is All You Need" (2019) — Multi-Query Attention
3. Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints" (2023) — Grouped-Query Attention
4. Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (2023) — vLLM and PagedAttention
5. Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models" (2023) — Token eviction strategies
6. Hooper et al., "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization" (2024) — Advanced KV cache quantization
7. Zheng et al., "SGLang: Efficient Execution of Structured Language Model Programs" (2024) — RadixAttention and prefix caching
8. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023) — Speculative decoding
9. Pope et al., "Efficiently Scaling Transformer Inference" (2023) — Comprehensive analysis of memory vs. compute tradeoffs in LLM inference

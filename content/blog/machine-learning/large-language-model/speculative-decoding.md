---
title: "Speculative Decoding in LLMs: A Complete Guide"
publishDate: "2026-04-17"
category: "machine-learning"
subcategory: "Large Language Model"
tags:
  [
    "llm",
    "speculative-decoding",
    "inference",
    "optimization",
    "deep-learning",
    "transformer",
    "draft-model",
    "serving",
  ]
date: "2026-04-17"
author: "Hiep Tran"
featured: false
aiGenerated: true
excerpt: "Speculative decoding is the most important algorithmic breakthrough for LLM inference speed. This guide explains the core idea, the math behind why it's lossless, every major variant, and practical deployment considerations — all at interview-ready depth."
---

## The Core Problem: Why Are LLMs Slow?

To understand speculative decoding, you first need to understand why LLM inference is slow — and crucially, **what kind of slow** it is.

LLMs generate tokens **one at a time**. Each token requires a full forward pass through the model. For a 70B-parameter model, each forward pass involves:

- Loading ~140 GB of weights from GPU memory (FP16)
- Performing matrix multiplications with those weights
- Loading the KV cache (which grows with sequence length)

Here's the critical insight: during the **decode phase** (generating tokens one by one), the GPU is massively underutilized. The model processes just a single token per step, meaning the matrix multiplications are tiny (a single row times a matrix). The GPU spends almost all its time **waiting for data to be loaded from memory**, not computing.

```
What the GPU is doing during decode:

Time: |==load weights==||calc||==load weights==||calc||==load weights==||calc|
       ████████████████ ░░    ████████████████ ░░    ████████████████ ░░
       ~99% loading            ~99% loading            ~99% loading
       ~1% computing           ~1% computing           ~1% computing
```

This is called being **memory-bandwidth-bound**. The GPU has enormous compute capacity (312 TFLOPS on an A100), but generating one token uses only a tiny fraction of it. The arithmetic intensity (FLOPS per byte loaded) is extremely low.

**The key observation**: If you could somehow process **multiple tokens** in a single forward pass, you'd use the same memory bandwidth (load the weights once) but do proportionally more compute. Instead of generating 1 token per weight load, you could generate 3-5 tokens. The memory loading cost is amortized, and you get closer to actually using the GPU's compute capacity.

But there's a fundamental constraint: autoregressive generation is **sequential** — each token depends on the previous one. You can't just generate 5 tokens in parallel because you don't know what the 2nd token is until the 1st is generated.

**Speculative decoding breaks this constraint.** It uses a clever trick: **guess** multiple future tokens cheaply, then **verify** all of them at once in a single forward pass through the large model.

## The Big Idea

![Speculative decoding flow: small draft model proposes γ tokens, target model verifies in one parallel forward pass, accepted prefix is emitted (+1 bonus) or rejected tokens are resampled from corrected distribution](/imgs/blogs/spec-dec-01-flow.png)

Speculative decoding works in three steps:

1. **Draft**: A fast, cheap model (the "draft model") quickly generates $K$ candidate tokens
2. **Verify**: The large target model processes all $K$ candidates in a single forward pass
3. **Accept/Reject**: Compare the draft model's predictions with the target model's — accept correct tokens, reject the first wrong one, and keep everything up to that point

```
Standard autoregressive decoding:
  Step 1: [target model] → token₁
  Step 2: [target model] → token₂
  Step 3: [target model] → token₃
  Step 4: [target model] → token₄
  Step 5: [target model] → token₅
  → 5 forward passes through the large model

Speculative decoding:
  Draft:  [draft model] → token₁', token₂', token₃', token₄', token₅'  (fast!)
  Verify: [target model] verifies all 5 at once in 1 forward pass
  Result: token₁'✓, token₂'✓, token₃'✓, token₄'✗ → accept 3 + sample 1 new = 4 tokens
  → 1 forward pass through the large model for 4 tokens!
```

**The restaurant analogy**: Imagine you're at a restaurant with a very slow, expert chef (the target model) who makes perfect dishes. Speculative decoding is like having a fast apprentice (draft model) who prepares several dishes in advance based on what they think the chef would make. The chef then inspects all the pre-made dishes at once — approves the ones that match their standards, remakes the first one that doesn't, and the kitchen has saved a lot of time.

## Why Is This Lossless?

This is the most remarkable property of speculative decoding and the first thing interviewers will probe: **the output distribution is mathematically identical to the target model's distribution**. You get the exact same quality as running the target model alone, just faster.

This is not an approximation. Not "close enough." The distribution is **exactly the same**.

### The Rejection Sampling Algorithm

The magic is in how we accept or reject draft tokens. Let's walk through it precisely.

Let $p(x)$ be the target model's probability distribution for the next token, and $q(x)$ be the draft model's distribution. Given a draft token $x$ sampled from $q$:

**Step 1**: Accept $x$ with probability:

$$\min\left(1, \frac{p(x)}{q(x)}\right)$$

**Step 2**: If rejected, sample a new token from a **modified distribution**:

$$p'(x) = \frac{\max(0,\; p(x) - q(x))}{\sum_{x'} \max(0,\; p(x') - q(x'))}$$

This is the normalized residual — the "leftover" probability mass where the target model assigns more probability than the draft model.

### Why This Gives Exact $p(x)$

Let's prove that the combined acceptance probability equals $p(x)$ for any token $x$:

**Case 1: $p(x) \geq q(x)$** (target assigns more probability than draft)

The probability of generating $x$ is:
- Draft samples $x$ with probability $q(x)$
- Accepted with probability $\frac{p(x)}{q(x)} = 1$ (since $p(x)/q(x) \geq 1$, capped at 1)
- Wait, that's not right. Let me be more careful.

Actually, acceptance probability is $\min(1, p(x)/q(x))$. When $p(x) \geq q(x)$, this is 1. So:

$$P(\text{accept } x) = q(x) \cdot 1 = q(x)$$

But we also get $x$ from the rejection sampling step when some other token $x'$ was rejected and then $x$ is drawn from $p'$:

$$P(\text{reject and draw } x) = P(\text{reject}) \cdot p'(x)$$

The total probability works out to exactly $p(x)$ when you sum both paths. The full proof uses the fact that:

$$P(\text{token} = x) = q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) + \left(1 - \sum_{x'} q(x') \cdot \min\left(1, \frac{p(x')}{q(x')}\right)\right) \cdot p'(x) = p(x)$$

This can be verified by expanding and simplifying for both cases ($p(x) \geq q(x)$ and $p(x) < q(x)$).

### Intuition for Why It Works

Think about it this way:

- When the draft model agrees with the target model ($q(x) \approx p(x)$), we accept almost everything → big speedup
- When the draft model disagrees ($q(x)$ far from $p(x)$), we reject and fall back to the target model's distribution → no speedup, but also no quality loss
- The rejection sampling correction ensures we always end up with the target distribution, regardless of how good or bad the draft model is

**A worse draft model doesn't make the output worse — it just makes speculation less effective** (lower acceptance rate → fewer tokens per verification step → less speedup).

### The Acceptance Rate

The expected number of tokens accepted per speculation round is:

$$\mathbb{E}[\text{accepted tokens}] = \sum_{t=1}^{K} \prod_{i=1}^{t} P(\text{accept token } i)$$

For a draft length of $K$ and per-token acceptance rate $\alpha$:

$$\mathbb{E}[\text{tokens per round}] = \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

Plus we always get at least 1 token (either accepted or from the residual sampling), so the expected tokens per target model forward pass is at least 1 and at most $K + 1$.

**Typical acceptance rates in practice**: 70-90% for well-matched draft-target pairs (e.g., Llama 3.1 1B drafting for Llama 3.1 70B), 40-60% for weaker draft models.

```python
import numpy as np

def expected_tokens_per_step(alpha, K):
    """
    Expected tokens generated per target model forward pass.
    
    Args:
        alpha: Per-token acceptance rate (0 to 1)
        K: Draft length (number of speculated tokens)
    
    Returns:
        Expected tokens including the bonus token from rejection
    """
    # Expected accepted tokens + 1 (always get at least 1 from residual)
    return (1 - alpha**(K + 1)) / (1 - alpha)

# Examples:
print(f"α=0.8, K=5: {expected_tokens_per_step(0.8, 5):.2f} tokens/step")  # ~3.36
print(f"α=0.9, K=5: {expected_tokens_per_step(0.9, 5):.2f} tokens/step")  # ~4.69
print(f"α=0.7, K=5: {expected_tokens_per_step(0.7, 5):.2f} tokens/step")  # ~2.74
print(f"α=0.5, K=5: {expected_tokens_per_step(0.5, 5):.2f} tokens/step")  # ~1.97
```

## The Complete Algorithm

Here's the full speculative decoding algorithm, step by step:

```python
import torch
import torch.nn.functional as F

def speculative_decode(target_model, draft_model, input_ids, K=5, max_tokens=100):
    """
    Speculative decoding with rejection sampling.
    
    Args:
        target_model: Large, slow, high-quality model
        draft_model: Small, fast, approximate model
        input_ids: Initial prompt token IDs
        K: Number of tokens to speculate per round
        max_tokens: Maximum total tokens to generate
    
    Returns:
        Generated token IDs (distribution identical to target_model)
    """
    generated = input_ids.clone()
    total_generated = 0
    
    while total_generated < max_tokens:
        # ═══ PHASE 1: DRAFT ═══
        # Generate K candidate tokens using the fast draft model
        draft_tokens = []
        draft_probs = []
        draft_input = generated.clone()
        
        for _ in range(K):
            with torch.no_grad():
                logits = draft_model(draft_input).logits[:, -1, :]
                q = F.softmax(logits, dim=-1)         # draft distribution
                token = torch.multinomial(q, 1)        # sample from draft
                draft_tokens.append(token)
                draft_probs.append(q)
                draft_input = torch.cat([draft_input, token], dim=-1)
        
        draft_tokens = torch.cat(draft_tokens, dim=-1)  # (1, K)
        
        # ═══ PHASE 2: VERIFY ═══
        # Run target model on ALL draft tokens in a single forward pass
        verify_input = torch.cat([generated, draft_tokens], dim=-1)
        with torch.no_grad():
            target_logits = target_model(verify_input).logits
        
        # Extract target model probabilities for each draft position
        # target_logits[:, len(generated)-1 : len(generated)-1+K] gives
        # p(token | prefix) for each speculated position
        
        # ═══ PHASE 3: ACCEPT/REJECT ═══
        accepted = 0
        for i in range(K):
            target_pos = len(generated) - 1 + i  # position in the verify input
            p = F.softmax(target_logits[:, target_pos, :], dim=-1)  # target dist
            q = draft_probs[i]                                       # draft dist
            
            draft_token = draft_tokens[:, i]
            
            # Acceptance probability: min(1, p(x)/q(x))
            p_token = p[0, draft_token[0]]
            q_token = q[0, draft_token[0]]
            accept_prob = min(1.0, (p_token / q_token).item())
            
            if torch.rand(1).item() < accept_prob:
                # Accept this token
                accepted += 1
                generated = torch.cat([generated, draft_token.unsqueeze(-1)], dim=-1)
            else:
                # Reject: sample from residual distribution
                residual = torch.clamp(p - q, min=0)
                residual = residual / residual.sum()  # normalize
                new_token = torch.multinomial(residual, 1)
                generated = torch.cat([generated, new_token], dim=-1)
                accepted += 1  # we still get 1 token from residual
                break
        else:
            # All K tokens accepted — sample bonus token from target at position K
            bonus_pos = len(generated) - 1
            # Reuse the already-computed target logits at the last position
            p_bonus = F.softmax(target_logits[:, target_pos + 1, :], dim=-1)
            bonus_token = torch.multinomial(p_bonus, 1)
            generated = torch.cat([generated, bonus_token], dim=-1)
            accepted += 1
        
        total_generated += accepted
    
    return generated
```

### Visualization of One Speculation Round

```
Prompt: "The capital of France is"

═══ PHASE 1: Draft model generates K=5 tokens quickly ═══

Draft: "The capital of France is" → [Paris] [,] [which] [is] [a]
       q("Paris")  = 0.85          (draft is pretty confident)
       q(",")      = 0.70
       q("which")  = 0.30          (draft is less sure here)
       q("is")     = 0.40
       q("a")      = 0.50

═══ PHASE 2: Target model processes all 5 in ONE forward pass ═══

Target probabilities at each position:
       p("Paris")  = 0.92          (target agrees)
       p(",")      = 0.65          (target roughly agrees)
       p("which")  = 0.15          (target disagrees — prefers "a")
       p("is")     = ...           (not reached)
       p("a")      = ...           (not reached)

═══ PHASE 3: Accept/Reject ═══

Token 1 "Paris":  accept_prob = min(1, 0.92/0.85) = 1.0   → ✓ ACCEPT
Token 2 ",":      accept_prob = min(1, 0.65/0.70) = 0.93  → ✓ ACCEPT (rolled 0.41)
Token 3 "which":  accept_prob = min(1, 0.15/0.30) = 0.50  → ✗ REJECT (rolled 0.73)

→ Sample from residual: p'(x) = normalize(max(0, p(x) - q(x)))
  p("a") - q("a") > 0 → "a" gets mass in residual
  → Sample "a" from residual

Result: "Paris , a" → 3 tokens from 1 target forward pass!
        (Without speculation: would have needed 3 separate forward passes)
```

## Draft Model Strategies

![Draft model strategies: separate small LM, self-speculation (skip layers), Medusa heads, EAGLE feature-level autoregression, n-gram/lookup — with training-cost tradeoffs](/imgs/blogs/spec-dec-02-drafts.png)

The choice of draft model is crucial. There are several approaches, each with different trade-offs:

### 1. Independent Small Model

Use a separately trained smaller model from the same family.

```
Target: Llama 3.1 70B
Draft:  Llama 3.1 8B (or even 1B)
```

**Pros**: Simple to set up; draft model is independently useful; well-studied
**Cons**: Draft model may have different "personality" → lower acceptance rate; need to host two models; vocabulary must match exactly

**Acceptance rates**: Typically 70-85% for same-family models (e.g., Llama 70B + Llama 8B).

### 2. Self-Drafting (Layer Skipping)

Use a subset of the target model's own layers as the draft model. Skip every other layer, or use only the first $N$ layers with an early exit head.

```
Target: Llama 3.1 70B (80 layers)
Draft:  Llama 3.1 70B layers 1-20 + exit head (same weights, fewer layers)
```

**Pros**: No additional model needed; shares weights → saves memory; vocabulary is identical; distribution is naturally similar
**Cons**: Need to train the early exit head; less flexible; draft quality limited by how well early layers predict final output

**Variants**:
- **DRAFT (Draft & Verify)**: Skip alternate layers for the draft pass
- **LayerSkip (Meta)**: Train the model with early exit losses, then use early layers as the draft
- **Kangaroo**: Uses a fixed subset of layers plus a lightweight adapter

### 3. Medusa Heads

Add multiple small prediction heads on top of the target model, each predicting a different future token position.

```
Target model backbone → hidden state at position t
                         ├── Head 0: predicts token at t+1 (original LM head)
                         ├── Head 1: predicts token at t+2
                         ├── Head 2: predicts token at t+3
                         └── Head 3: predicts token at t+4
```

```python
class MedusaHead(nn.Module):
    """One Medusa head that predicts a future token."""
    def __init__(self, hidden_size, vocab_size):
        super().__init__()
        # Simple residual MLP
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        self.act = nn.SiLU()
    
    def forward(self, hidden_states):
        h = self.act(self.linear1(hidden_states))
        h = h + hidden_states  # residual
        return self.linear2(h)

class MedusaModel(nn.Module):
    def __init__(self, base_model, num_heads=4):
        super().__init__()
        self.base = base_model  # frozen
        hidden_size = base_model.config.hidden_size
        vocab_size = base_model.config.vocab_size
        self.medusa_heads = nn.ModuleList([
            MedusaHead(hidden_size, vocab_size)
            for _ in range(num_heads)
        ])
    
    def forward(self, input_ids):
        # Single forward pass through the base model
        outputs = self.base(input_ids, output_hidden_states=True)
        hidden = outputs.hidden_states[-1]
        
        # Each head predicts a different future position
        predictions = [head(hidden) for head in self.medusa_heads]
        return outputs.logits, predictions
```

**Pros**: Single model — no separate draft model to host; adds only ~0.5-1% parameters; very fast drafting (one forward pass produces all candidates); easy to fine-tune the heads
**Cons**: Not lossless (uses a tree-based verification with approximation, unless combined with rejection sampling); heads must be trained; quality of speculation degrades for positions further ahead

### 4. EAGLE (Extrapolation Algorithm for Greater Language-model Efficiency)

EAGLE uses the target model's **second-to-last layer features** (not just the output logits) to predict future tokens, achieving higher acceptance rates than Medusa.

```
Standard Medusa:
  hidden_state → MLP → next token prediction
  (predicts from the last layer's output, losing intermediate information)

EAGLE:
  feature_t (second-to-last layer) + embedding_{t+1} → lightweight model → feature_{t+1}
  → autoregressively generates draft features, then maps to tokens
```

**Pros**: Higher acceptance rates than Medusa (reportedly 1.5-2x faster than vanilla speculation); features carry more information than output logits
**Cons**: Requires training the feature predictor; slightly more complex implementation

### 5. Lookahead Decoding

Instead of a draft model, use **n-gram patterns** from the model's own generation history to predict future tokens. Extract common n-grams from the Jacobi iteration trajectory and verify them in parallel.

**Pros**: No draft model or additional training needed; works with any model out of the box
**Cons**: Lower acceptance rate than learned draft models; most effective for repetitive content

### Comparison Table

| Method | Extra Memory | Training Needed | Acceptance Rate | Speedup | Lossless? |
|--------|-------------|-----------------|-----------------|---------|-----------|
| Independent draft | Full draft model (~1-8B) | No (use existing model) | 70-85% | 2-3x | Yes |
| Self-drafting / LayerSkip | Exit head only (~0.1%) | Exit head training | 65-80% | 1.5-2.5x | Yes |
| Medusa | +0.5-1% params | Head fine-tuning | 60-80% | 2-3x | Approximate* |
| EAGLE | +1-2% params | Feature predictor training | 75-90% | 2-4x | Yes (v2) |
| Lookahead | None | None | 40-60% | 1.3-1.8x | Yes |

*Medusa can be made lossless with typical rejection sampling but uses tree verification with approximation by default.

## Tree-Based Speculation

![Tree speculation: draft proposes a tree of candidate continuations, target verifies all branches in one forward with a causal tree mask, dramatically widening accepted paths](/imgs/blogs/spec-dec-03-tree.png)

A powerful extension: instead of generating a single chain of $K$ draft tokens, generate a **tree** of candidates exploring multiple possible continuations.

### Why Trees?

A linear chain is fragile — if token 2 is wrong, tokens 3-5 are wasted regardless of their quality. A tree explores multiple branches:

```
Linear speculation:
  "The" → "cat" → "sat" → "on" → "the"
  If "sat" is wrong, everything after is wasted.

Tree speculation:
                   ┌→ "sat" → "on"
  "The" → "cat" ──┤
                   └→ "is" → "very"
              │
              └→ "dog" ──→ "ran"
  
  Multiple paths verified simultaneously.
  Even if "cat sat" is wrong, "cat is" might be accepted.
```

### How Tree Verification Works

The target model can verify an entire tree in a single forward pass using **attention masking**. Each node in the tree only attends to its ancestors (the path from root to that node):

```python
def build_tree_attention_mask(tree_structure):
    """
    Build a causal attention mask for tree-structured speculation.
    
    Each token can attend to:
    1. All prefix tokens (the verified context)
    2. Its ancestors in the tree (the path from root to this node)
    """
    num_nodes = len(tree_structure)
    mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool)
    
    for node_idx, parent_idx in enumerate(tree_structure):
        # Each node attends to itself
        mask[node_idx, node_idx] = True
        
        # Walk up the tree to root, enabling attention to all ancestors
        current = parent_idx
        while current is not None:
            mask[node_idx, current] = True
            current = tree_structure[current]  # parent's parent
    
    return mask
```

### Optimal Tree Structure

The tree structure should be chosen to maximize expected accepted tokens per verification. This depends on the draft model's confidence at each position:

- **High confidence**: Extend the chain deeper (linear-like)
- **Low confidence**: Branch out wider (more alternatives at that position)

Medusa and EAGLE-2 learn optimal tree structures by profiling the draft model's accuracy at different positions.

## The Speedup Formula

![Speedup formula with acceptance rate α, draft length γ, and draft/target cost ratio c — typically 2-3× on natural text, best at low QPS and latency-sensitive serving, worst at already-compute-bound large-batch](/imgs/blogs/spec-dec-04-speedup.png)

The theoretical speedup of speculative decoding depends on several factors:

$$\text{Speedup} = \frac{\text{tokens per round}}{\text{time per round} / \text{time per standard step}}$$

Let:
- $K$ = draft length
- $\alpha$ = per-token acceptance rate
- $c$ = cost ratio (time for draft to generate $K$ tokens / time for one target forward pass)
- $\tau$ = overhead ratio (target verify with $K$ tokens / target with 1 token)

Expected tokens per round: $\frac{1 - \alpha^{K+1}}{1-\alpha}$

Time per round: $c + \tau$ (drafting time + verification time)

$$\text{Speedup} \approx \frac{1 - \alpha^{K+1}}{(1-\alpha)(c + \tau)}$$

### What This Tells Us

- **Higher $\alpha$ → bigger speedup**: When draft and target agree more, more tokens are accepted
- **Lower $c$ → bigger speedup**: When the draft model is much faster than the target, the drafting overhead is negligible
- **$\tau$ is usually close to 1**: Verifying $K$ tokens in one forward pass takes about the same time as generating 1 token, because the decode phase is memory-bandwidth-bound (same weights loaded regardless of whether you process 1 or $K$ tokens)
- **Diminishing returns on $K$**: Beyond $K \approx 5-8$, the probability that all tokens are accepted drops exponentially, while draft cost increases linearly

### Practical Speedups

| Configuration | Acceptance Rate | Draft Cost | Typical Speedup |
|--------------|-----------------|------------|-----------------|
| Llama 3.1 70B + 8B draft | ~80% | ~0.1x | 2.5-3x |
| Llama 3.1 70B + 1B draft | ~70% | ~0.02x | 2-2.5x |
| 70B + Medusa (4 heads) | ~75% | ~0.01x | 2.5-3x |
| 70B + EAGLE-2 | ~85% | ~0.05x | 3-4x |
| 70B + self-draft (layer skip) | ~65% | ~0.25x | 1.5-2x |

## When Speculative Decoding Helps (and When It Doesn't)

### When It Helps Most

**1. Single-sequence generation (batch size = 1)**

At batch size 1, decode is maximally memory-bandwidth-bound. The GPU loads all model weights to process just one token — enormous waste. Speculative decoding generates multiple tokens per weight load, dramatically improving utilization.

**2. Long outputs**

The more tokens you generate, the more you benefit from amortized drafting costs.

**3. "Easy" / predictable text**

When the target model's output is highly predictable (common phrases, structured output, boilerplate), the draft model has high acceptance rates. JSON output, code with standard patterns, and formulaic text benefit most.

**4. Large models**

Larger models have more "wasted" compute during decode (more weights loaded per token), so the amortization benefit is larger.

### When It Helps Less

**1. Large batch sizes**

At large batch sizes, decode becomes compute-bound rather than memory-bandwidth-bound. Processing 128 sequences already uses the GPU's compute effectively, so adding more tokens per step doesn't help much — the bottleneck shifts from memory bandwidth to actual FLOPS.

```
Batch size 1:   Load 140GB weights → process 1 token → waste 99% compute
Batch size 128: Load 140GB weights → process 128 tokens → good utilization

Adding speculation at batch 128 means:
  Load 140GB weights → process 128 × (K+1) tokens → exceeds compute capacity
  → Speculation adds latency instead of reducing it!
```

**2. Creative / uncertain generation**

When the target model's output is highly uncertain (creative writing, open-ended reasoning), the draft model has low acceptance rates. Many drafted tokens are rejected, and the overhead of drafting is wasted.

**3. Very short outputs**

If the output is only 5-10 tokens, the overhead of setting up the draft pipeline may not be worth it.

**4. When memory is tight**

Hosting both a draft and target model requires more GPU memory. If memory is the bottleneck, the memory used by the draft model might be better spent on larger batch sizes or longer KV caches.

## Implementation: Practical Considerations

### KV Cache Management

Both the draft model and target model maintain their own KV caches. When tokens are rejected, the KV cache must be **rolled back**:

```python
class SpeculativeKVCacheManager:
    def __init__(self):
        self.target_kv_cache = None
        self.draft_kv_cache = None
    
    def after_verification(self, num_accepted, total_drafted):
        """
        After verification, align KV caches.
        
        - Draft cache: roll back to match accepted length
        - Target cache: already correct (verification populated it)
        """
        num_rejected = total_drafted - num_accepted
        
        if num_rejected > 0:
            # Truncate draft KV cache: remove rejected positions
            self.draft_kv_cache = self.draft_kv_cache[:, :, :, :-num_rejected, :]
        
        # Target KV cache was populated during the verification forward pass
        # It already contains entries for all accepted tokens + the bonus token
        # Truncate it to match the final accepted length
        final_length = self.target_kv_cache.shape[3] - num_rejected
        self.target_kv_cache = self.target_kv_cache[:, :, :, :final_length, :]
```

### Temperature and Sampling

Speculative decoding works with any temperature. For temperature $T > 0$, the draft model samples with temperature $T$ and the verification uses the same temperature-adjusted distributions:

```python
def speculative_sample_with_temperature(p, q, draft_token, temperature):
    """Rejection sampling with temperature scaling."""
    # Apply temperature to both distributions
    p_scaled = F.softmax(target_logits / temperature, dim=-1)
    q_scaled = F.softmax(draft_logits / temperature, dim=-1)
    
    # Same rejection sampling algorithm
    accept_prob = min(1.0, p_scaled[draft_token] / q_scaled[draft_token])
    
    if random() < accept_prob:
        return draft_token, True
    else:
        residual = torch.clamp(p_scaled - q_scaled, min=0)
        residual = residual / residual.sum()
        return torch.multinomial(residual, 1), False
```

For **greedy decoding** (temperature = 0), the algorithm simplifies dramatically: accept if the draft token matches the target's argmax, reject otherwise. No probabilistic rejection needed.

```python
def speculative_greedy(target_logits, draft_token):
    """Speculative decoding with greedy (temperature=0) sampling."""
    target_token = target_logits.argmax(dim=-1)
    if draft_token == target_token:
        return draft_token, True  # accept
    else:
        return target_token, False  # reject, use target's choice
```

### Dynamic Draft Length

Fixed draft length $K$ is suboptimal — sometimes the draft model is confident and could draft more tokens, sometimes it's uncertain and should draft fewer. **Adaptive draft length** adjusts $K$ based on the draft model's confidence:

```python
def adaptive_draft(draft_model, input_ids, max_K=8, confidence_threshold=0.4):
    """Generate draft tokens, stopping early if confidence drops."""
    draft_tokens = []
    draft_probs = []
    
    for k in range(max_K):
        logits = draft_model(input_ids).logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        
        top_prob = probs.max().item()
        
        # Stop drafting if the model is too uncertain
        if top_prob < confidence_threshold and k > 0:
            break
        
        token = torch.multinomial(probs, 1)
        draft_tokens.append(token)
        draft_probs.append(probs)
        input_ids = torch.cat([input_ids, token], dim=-1)
    
    return draft_tokens, draft_probs
```

### Integration with Serving Frameworks

Major serving frameworks support speculative decoding:

```python
# vLLM with speculative decoding
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    speculative_model="meta-llama/Llama-3.1-8B-Instruct",
    num_speculative_tokens=5,
    # Or use self-drafting:
    # speculative_model="[ngram]",  # n-gram based speculation
)

params = SamplingParams(temperature=0.7, max_tokens=512)
output = llm.generate("Explain quantum computing", params)
```

```python
# SGLang with EAGLE speculation
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path yuhuili/EAGLE-LLaMA3.1-Instruct-70B \
    --speculative-num-steps 5
```

## Speculative Decoding in Production

### Throughput vs Latency

Speculative decoding primarily improves **latency** (time to generate a single sequence), not necessarily **throughput** (total tokens per second across all requests).

- **Latency**: At batch size 1, speculation can reduce per-request latency by 2-3x
- **Throughput**: At high batch sizes, the GPU is already well-utilized, and speculation adds overhead without proportional benefit. In some cases, throughput actually **decreases**

The right deployment strategy depends on your workload:

| Workload | Batch Profile | Use Speculation? |
|----------|--------------|------------------|
| Interactive chat | Low batch, latency-sensitive | Yes — reduces TPOT significantly |
| Batch processing | High batch, throughput-sensitive | Probably not — better to increase batch size |
| Mixed | Varies | Adaptive — enable when batch is low, disable when high |

### Memory Budget

Hosting both models costs memory:

```
Without speculation:
  Target model (70B FP16): 140 GB
  KV cache: variable
  Total: 140 GB + KV cache

With independent draft model (8B FP16):
  Target model: 140 GB
  Draft model: 16 GB
  KV cache (both models): variable
  Total: 156 GB + KV caches

With Medusa/EAGLE:
  Target model: 140 GB
  Extra heads: ~1-2 GB
  KV cache (one model): variable
  Total: 142 GB + KV cache  ← much less overhead!
```

This is why Medusa and EAGLE are appealing for memory-constrained deployments — they add speculation capability with minimal extra memory.

### Choosing the Optimal Draft Length $K$

The optimal $K$ balances accepted tokens against draft cost:

$$K^* = \arg\max_K \frac{\mathbb{E}[\text{accepted tokens}(K)]}{\text{draft time}(K) + \text{verify time}}$$

In practice:
- $K = 3-5$ works well for most configurations
- Higher $K$ helps when acceptance rate is very high ($\alpha > 0.85$) and draft is very fast ($c < 0.05$)
- Lower $K$ (2-3) is better when acceptance rate is moderate ($\alpha \approx 0.6-0.7$)

Some frameworks tune $K$ dynamically per request based on observed acceptance rates.

## Connections to Other Techniques

### Speculative Decoding + KV Cache Quantization

Speculative decoding reduces the **number of KV cache loads** (fewer target model forward passes), while KV cache quantization reduces the **size of each load**. They're complementary:

```
Standard decode:        100 loads × 2 GB per load = 200 GB total bandwidth
Speculation (3x):       34 loads × 2 GB per load  = 68 GB total bandwidth
Spec + FP8 KV cache:    34 loads × 1 GB per load  = 34 GB total bandwidth
                                                     6x reduction!
```

### Speculative Decoding + Tensor Parallelism

With tensor parallelism (model split across GPUs), speculation still works but the draft model adds complexity:

- If the draft model is small enough, run it on a single GPU while the target model spans multiple GPUs
- Some approaches run the draft model on the CPU while the target uses the GPU
- EAGLE/Medusa avoid this issue since they share the target model's infrastructure

### Speculative Decoding + Continuous Batching

In a continuous batching system (vLLM, SGLang), speculation interacts with the scheduler:

- Different sequences may have different acceptance rates → some finish speculation quickly, others slowly
- The scheduler must handle variable numbers of accepted tokens per sequence
- Sequences that reject early can be immediately rescheduled without waiting

## Interview Questions and Answers

### Q: What is speculative decoding and why does it work?

Speculative decoding uses a fast draft model to propose multiple future tokens, then verifies all of them in a single forward pass through the target model. It works because the decode phase is memory-bandwidth-bound — the GPU loads the entire model weights to process just one token, wasting massive compute. By verifying $K$ tokens in one forward pass, we load the weights once and amortize that cost across multiple tokens. The output distribution is mathematically identical to the target model thanks to a rejection sampling algorithm.

### Q: Prove that speculative decoding is lossless.

The acceptance probability for a draft token $x$ sampled from $q(x)$ is $\min(1, p(x)/q(x))$, where $p(x)$ is the target distribution. When rejected, we sample from the normalized residual $p'(x) = \text{normalize}(\max(0, p(x) - q(x)))$.

For any token $x$, the total probability of outputting $x$ is:

$$P(x) = q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) + P(\text{reject}) \cdot p'(x)$$

**Case 1: $p(x) \geq q(x)$**: $P(\text{accept } x) = q(x) \cdot 1 = q(x)$, plus the residual contribution. The residual $p'(x) \propto p(x) - q(x)$. Summing gives $P(x) = p(x)$.

**Case 2: $p(x) < q(x)$**: $P(\text{accept } x) = q(x) \cdot p(x)/q(x) = p(x)$, and $p'(x) = 0$ since $p(x) - q(x) < 0$. So $P(x) = p(x)$.

In both cases, $P(x) = p(x)$. The output distribution exactly matches the target.

### Q: What is the acceptance rate and how does it determine speedup?

The per-token acceptance rate $\alpha$ is the probability that a draft token is accepted: $\alpha = \mathbb{E}[\min(1, p(x)/q(x))]$ where $x \sim q$. With draft length $K$, the expected tokens per verification step is $(1-\alpha^{K+1})/(1-\alpha)$.

Speedup $\approx$ (expected tokens per step) / (relative cost per step). At $\alpha = 0.8, K=5$: expected tokens = 3.36 per target forward pass. If draft cost is negligible and verification cost ≈ 1 standard step, speedup ≈ 3.36x.

Higher $\alpha$ gives exponentially better returns (at $\alpha = 0.9, K=5$: 4.69 tokens/step). This is why the draft-target alignment is the most important factor.

### Q: What makes a good draft model?

A good draft model optimizes three properties:

1. **Distribution alignment with target** — $q(x) \approx p(x)$ gives high acceptance rate. Same model family helps (Llama 8B for Llama 70B). Exactly matching vocabularies is required.
2. **Speed** — must be significantly faster than the target. The draft cost ratio $c$ (draft time / target time) should be < 0.15. A draft model that's half the target's speed is useless even with 100% acceptance.
3. **Memory efficiency** — shouldn't consume too much of the GPU memory budget that could be used for batching or KV cache.

The ideal draft model is 10-50x smaller than the target, from the same model family, with the same tokenizer. Medusa/EAGLE heads are even better on criteria 2 and 3, since they add minimal cost.

### Q: When would you NOT use speculative decoding?

1. **Large batch sizes**: At batch size 128+, decode is already compute-bound — the GPU is well-utilized. Speculation adds overhead (draft model compute + verification of extra tokens) without benefit.
2. **Memory-constrained**: If the draft model's memory would be better spent increasing batch size or KV cache capacity.
3. **Highly creative/uncertain generation**: Low acceptance rates make speculation overhead-dominant.
4. **Very short outputs**: Setup cost not amortized over enough tokens.
5. **Throughput-optimized serving**: Speculation helps latency more than throughput. For batch processing workloads, increasing batch size is more effective.

### Q: Compare Medusa, EAGLE, and independent draft models.

**Independent draft model** (e.g., Llama 8B drafting for 70B): Simplest approach. Requires hosting a separate model (extra memory). Acceptance rate 70-85%. The draft model's vocabulary and tokenizer must match the target exactly. Proven and well-supported in production frameworks.

**Medusa**: Adds small prediction heads to the target model. Each head predicts a different future position (t+1, t+2, t+3...). Only ~0.5-1% extra parameters. Drafting is extremely fast (one forward pass = all candidates). Uses tree verification for efficiency. Heads need fine-tuning on the target model. Default version uses approximate verification; can be made lossless with rejection sampling at some cost.

**EAGLE**: Uses the target model's intermediate features (not just output logits) to predict future tokens. An autoregressive feature predictor generates draft features, which are then mapped to tokens. Higher acceptance rates than Medusa because intermediate features carry more information. ~1-2% extra parameters. Currently achieves the best speedups (3-4x).

### Q: How does tree-based speculation improve upon linear speculation?

Linear speculation generates a single chain of $K$ tokens. If token $i$ is rejected, tokens $i+1$ through $K$ are wasted regardless. Tree speculation generates multiple branches, so if one path is rejected, another path might be accepted.

The target model verifies the entire tree in a single forward pass by using a tree-structured attention mask — each node attends only to its ancestors. The verification cost is proportional to the number of nodes in the tree, not the number of paths.

For the same total number of draft tokens, a well-structured tree typically yields more accepted tokens than a linear chain, especially when the draft model's accuracy varies across positions.

### Q: How does speculative decoding interact with continuous batching in production?

In continuous batching, sequences enter and leave the batch dynamically. Speculation complicates this because:

1. **Variable-length results**: Different sequences may accept different numbers of tokens (1 to $K+1$), making the batch irregular after verification
2. **Draft synchronization**: The draft phase must complete for all sequences before verification, but some may draft faster than others
3. **Memory management**: KV cache rollback for rejected tokens must be handled per-sequence
4. **Mixed decode modes**: Some sequences may be in prefill while others are speculating — the scheduler must handle both

Modern frameworks (vLLM, SGLang) solve this by treating speculation as a scheduling primitive: draft tokens are added as tentative entries in the KV cache, and rollback is handled by the paged memory manager.

### Q: Explain the memory bandwidth argument for speculative decoding more precisely.

During decode at batch size $B$, one step requires:
- **Loading model weights**: $2P$ bytes (for a $P$-parameter model in FP16)
- **Loading KV cache**: $2 \times L \times n_\text{kv} \times d \times S \times B \times 2$ bytes (sequence length $S$)
- **Compute**: $2P \times B$ FLOPS (one matmul per parameter per batch element)

The arithmetic intensity is:

$$\text{AI} = \frac{2PB}{2P + \text{KV cache}} \approx \frac{B}{1 + \text{KV/weights}} \text{ FLOPS/byte}$$

For batch size 1 with a 70B model: $AI \approx 1$ FLOP/byte. An A100 has 2 TB/s bandwidth and 312 TFLOPS → the break-even AI is $312/2 = 156$ FLOPS/byte. At $AI = 1$, we use $1/156 = 0.6\%$ of the GPU's compute capacity.

Speculative decoding with $K=5$ verified tokens effectively multiplies $B$ by $\sim$5 for the verification step (processing 5 tokens instead of 1), increasing AI by ~5x. Still memory-bound, but significantly better utilization.

At batch size 128, $AI \approx 128$, which is already close to the 156 break-even. Adding speculation would push past the compute capacity, causing slowdown rather than speedup. This is precisely why speculation helps at low batch sizes and hurts at high batch sizes.

### Q: What is the relationship between speculative decoding and parallel decoding methods like Jacobi decoding?

Both aim to generate multiple tokens per step, but from different angles:

**Speculative decoding**: Uses a separate model to guess tokens, then verifies. Lossless via rejection sampling. Works with any draft-target pair.

**Jacobi decoding**: Treats autoregressive generation as a fixed-point iteration. Initialize all $K$ future tokens randomly, then iteratively refine them in parallel using the target model until convergence. Each iteration is a single forward pass over all $K$ positions. Convergence happens when the model's prediction at each position matches the current token at that position.

**Connection**: Jacobi decoding can be seen as a form of self-speculation where the model is its own draft. Lookahead decoding (mentioned earlier) combines Jacobi iterations with n-gram caching to accelerate convergence.

**Trade-off**: Jacobi decoding requires no draft model but converges slowly (often needs 5-10 iterations per token accepted, reducing the benefit). Speculative decoding with a good draft model converges in 1 step (the verification) but requires hosting the draft.

## References

1. Leviathan, Y., Kalman, M., & Matias, Y. "Fast Inference from Transformers via Speculative Decoding." ICML 2023.
2. Chen, C., et al. "Accelerating Large Language Model Decoding with Speculative Sampling." 2023.
3. Cai, T., et al. "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads." ICML 2024.
4. Li, Y., et al. "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty." ICML 2024.
5. Li, Y., et al. "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees." 2024.
6. Elhoushi, M., et al. "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding." ACL 2024.
7. Fu, Y., et al. "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding." 2024.
8. Santilli, A., et al. "Accelerating Transformer Inference for Translation via Parallel Decoding." ACL 2023.
9. Kwon, W., et al. "Efficient Memory Management for Large Language Model Serving with PagedAttention." SOSP 2023.
10. Miao, X., et al. "SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference and Verification." ASPLOS 2024.

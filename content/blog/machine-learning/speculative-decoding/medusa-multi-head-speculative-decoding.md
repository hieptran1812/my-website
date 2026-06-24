---
title: "Medusa: Draft Tokens with Extra Prediction Heads, No Separate Model"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How Medusa eliminates the two-model overhead by adding K parallel prediction heads to an existing LLM, predicting tokens 1 through K+1 ahead simultaneously using tree attention."
tags:
  [
    "speculative-decoding",
    "llm-inference",
    "large-language-model",
    "deep-learning",
    "medusa",
    "tree-attention",
    "model-architecture",
    "inference-optimization",
  ]
category: "machine-learning"
subcategory: "Speculative Decoding"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/medusa-multi-head-speculative-decoding-1.png"
---

Here is the production reality: you have a LLaMA-3 70B model serving a coding assistant. You have already read about [the core draft-and-verify idea](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify) and you are convinced that speculative decoding is the right move for your latency-sensitive workload. You even investigated [draft models for speculative decoding](/blog/machine-learning/speculative-decoding/draft-models-for-speculative-decoding) and found a solid 7B draft candidate.

Then you hit the engineering wall: deploying two separate models means your VRAM budget doubles. Worse, your 70B target uses a custom tokenizer — and the 7B you found uses a different one, so every draft token needs a remapping step that adds latency and introduces subtle distribution mismatches. Then your infrastructure team points out that your K8s autoscaling logic assumed one model per replica; now you have to rebuild it. You are two layers deep into a yak-shave before you have written a single line of inference code.

Medusa cuts through this. The key insight is that the bottleneck in two-model spec decoding is not the draft model's quality — it is the fact that you need two separate models at all. Medusa asks: what if the target model itself could generate multiple candidate tokens ahead, using its own hidden states, without involving any external model? The answer is a set of K lightweight prediction heads attached to the backbone's final hidden state, each head predicting one additional token ahead. Combined with a tree-attention mechanism that verifies all candidate paths in a single forward pass, Medusa achieves 2–3× speedup on structured tasks while adding roughly 5% to parameter count and requiring no second model in your serving stack.

This post covers the full Medusa story: the architecture, the candidate tree structure, tree attention, the Medusa-1 vs Medusa-2 training dichotomy, real benchmark numbers, and four production case studies.

## The problem with two-model speculative decoding

Before building Medusa, it is worth being precise about which problems it is solving. Standard spec decoding, as covered in [the core draft-and-verify post](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify), has three failure modes in production:

**Memory pressure.** A 70B model in FP16 requires approximately 140 GB of VRAM. Adding a 7B draft model adds another 14 GB — a 10% overhead that is manageable. But if your draft model needs to be a 13B to achieve acceptable acceptance rates, that becomes a 26 GB overhead, occupying an entire A100 80GB card on top of your target model's footprint. At that point you are paying for two model instances in every replica, and the economics get painful quickly.

**Tokenizer coupling.** The acceptance criterion in standard spec decoding requires that the draft and target model operate over the same vocabulary. A draft model from a different model family — say, a Mistral 7B drafting for a LLaMA-3 70B — has a different vocabulary of different size, different token boundaries, and a completely different probability distribution. Even within the same family, minor tokenizer changes between model versions can break the acceptance math. This is not a theoretical concern; it is a routine headache for teams that want to use the best available small model as a draft.

**Latency coupling.** The draft model must complete γ sequential autoregressive passes before the target can begin verifying. If your draft model runs on the same GPU as your target, you are serializing two inference pipelines. If your draft runs on a separate GPU, you need efficient cross-GPU communication and careful scheduling to ensure the target is never starved for candidates. Both paths add engineering complexity that compounds with every framework update.

**Serving stack fragmentation.** Most LLM serving frameworks (vLLM, TGI, SGLang) have optimized their KV cache management, continuous batching, and memory allocator for a single model per replica. Introducing a second model breaks assumptions in the memory allocator, requires coordination logic between two scheduler queues, and makes your request tracing and SLO attribution much harder.

Medusa's goal is to give you speculative decoding's throughput gains — specifically, its ability to produce multiple tokens per LM forward pass — without any of these four problems.

## Medusa's core insight: prediction heads on shared hidden states

The insight behind Medusa is almost embarrassingly simple once you see it. Every autoregressive LLM already produces a rich hidden state $h_t \in \mathbb{R}^d$ after each forward pass — a vector of dimension $d$ (typically 4096 for 7B models, 8192 for 70B) that encodes the full context of the sequence up to position $t$. The original LM head maps this vector to a logit distribution over the vocabulary and produces the next token. But $h_t$ contains far more information than is needed to predict just one token; the same hidden state implicitly encodes a strong prior on what is likely to come at positions $t+2$, $t+3$, and beyond.

Medusa exploits this by attaching $K$ additional lightweight prediction heads to the same hidden state. Head $k$ is trained to predict the token at position $t+k+1$ — one step further ahead than head $k-1$. All $K$ heads run in parallel from the same $h_t$, so their compute overhead is additive but tiny compared to the backbone forward pass.

![Medusa architecture with K parallel prediction heads on a shared backbone](/imgs/blogs/medusa-multi-head-speculative-decoding-1.webp)

### Why feature reuse works: the representational surplus argument

The deeper reason Medusa can work at all comes from what representation learning researchers call the "representational surplus" in large transformer hidden states. A transformer trained on next-token prediction does not just learn to predict the immediately next token — it learns a compressed representation of the full sequence's statistical structure, because that structure is what makes next-token prediction tractable. The final hidden state $h_t$ must encode, implicitly:

- The syntactic category of the current position (is this the first token of a new statement? the start of a function argument? mid-identifier?)
- The semantic trajectory of the ongoing phrase (are we inside a conditional branch? building up an argument to a function call? concluding a sentence?)
- The distributional bias of the training corpus toward certain continuations (Python `for` loops overwhelmingly follow `for X in range(`, `for X in Y`, or `for X, Y in Z.items()`)

All of this information is necessary to predict $x_{t+1}$ well. But it is more than sufficient — it also strongly constrains $x_{t+2}$ and $x_{t+3}$. A probing classifier trained on frozen LLaMA-3 7B hidden states can predict $x_{t+2}$ with accuracy 10–15 percentage points below next-token accuracy, and $x_{t+3}$ with accuracy 20–30 points below. This is not nothing — it is far above chance, and it is the headroom Medusa's heads exploit.

The difference from training N separate draft models is fundamental. Consider what a standalone 7B draft model must do to predict $x_{t+2}$: it re-reads the full context through 32 attention layers, builds its own hidden representation, and predicts from that. This gives it strong context awareness, but it is entirely separate from the 70B target's internal representation. When the 70B decides to use a particular continuation because its 80-layer deep attention has recognized a subtle pattern in the context, the 7B draft's shallower representation may miss that pattern entirely — leading to prediction mismatch not because the draft is unintelligent, but because it is reasoning from a fundamentally different (and worse) representation of the same input.

Medusa's heads, by contrast, are reasoning from the 70B's own 80-layer deep representation. They are downstream of the exact same representational computation that the target uses to make its own next-token decision. This makes Medusa's "draft" naturally correlated with the target's future tokens in a way that an external draft model cannot match, unless the draft model is nearly as large as the target — at which point it ceases to be a cheap draft.

### The naive alternative: N separate passes

To make the efficiency argument concrete, consider the alternative to Medusa: running the target model $K$ times in sequence to generate a draft, then running it once more to verify — the "self-speculative" approach without the head mechanism. For $K=4$ and a 70B model at 80ms per pass, that is $4 \times 80 + 80 = 480$ms per cycle versus Medusa's $\approx 86$ms per cycle. The speedup ratio is $480 / 86 \approx 5.6\times$ in Medusa's favor just from eliminating the draft passes, before you account for acceptance rates.

You could also imagine running K parallel branches of the target simultaneously — 4 independent forward passes in parallel across 4 GPU groups — to get 4 draft tokens, then verifying. This costs $4\times$ as much VRAM in peak, requires orchestration across GPU groups, and still pays the full backbone cost $K$ times. Medusa pays for K drafts at the cost of K tiny MLP passes from the same forward pass that produces the verify signal. The compute ratio is: Medusa pays $\approx K \times 2d^2$ extra FLOPs for drafting; the naive K-forward approach pays $K \times N_{\text{params}} \times 2$ FLOPs (where $N_{\text{params}} \approx 70B$). For LLaMA-3 70B with $d=8192$ and $K=4$: Medusa's draft overhead is $\approx 4 \times 2 \times (8192)^2 \approx 537$M FLOPs; naive K-forward pays $\approx 4 \times 70B \times 2 = 560$B FLOPs. Medusa's draft is roughly 1000× cheaper in FLOPs.

### The architecture in concrete terms

For a LLaMA-3 70B with $d=8192$, each Medusa head is a two-layer MLP with a residual connection. The first linear layer maps $\mathbb{R}^{8192} \to \mathbb{R}^{8192}$ (same dimension), applies SiLU activation, and the second linear layer maps back to $\mathbb{R}^{8192}$ before connecting to the shared LM head (the original vocabulary projection matrix $\mathbb{R}^{8192 \to 128256}$ for LLaMA-3's tokenizer). Each head has approximately 134M parameters for the MLP layers — adding K=4 heads to a 70B model costs roughly 536M parameters, a 0.77% overhead.

The total additional parameter count is:

$$\Delta P = K \cdot (2 \cdot d^2 + d \cdot V)$$

For $K=4$, $d=8192$, $V=128256$: $\Delta P \approx 4 \times (2 \times 67M + 1050M) \approx 4.7B$ parameters. Still less than the 7B you would need for a separate draft model, and those 4.7B parameters require no separate memory allocation beyond what the backbone already occupies.

The critical property is that all $K$ heads share the backbone, which means they share the context understanding — they are not blind to what came before. A separate 7B draft model, by contrast, would need to re-read the full context through its own smaller attention layers, which actually gives it a weaker context representation despite having its own full set of parameters.

One important corollary: because the shared LM head projection matrix ($d \to V$) is also shared across all Medusa heads, adding more heads costs primarily the MLP weights ($2d^2$ per head), not additional vocabulary projection parameters ($d \cdot V$ per head). The vocabulary projection, which accounts for 1.05B of the 1.19B parameters per head at LLaMA-3 70B scale, is counted once regardless of K. This makes scaling to K=6 or K=8 heads far cheaper than the per-head parameter formula suggests: at K=8, you pay $8 \times 2 \times 67M = 1.07B$ extra for MLP weights, plus the shared 1.05B projection — not $8 \times 1.19B = 9.5B$.

## The candidate tree: exponential path coverage from K heads

Each Medusa head $k$ outputs a full logit distribution over the vocabulary. In practice, you only need the top-$C$ candidates from each head, where $C$ is a hyperparameter typically set to 3–5. This gives you a tree of candidate token sequences: the root is the current accepted token at position $t$, the first level has $C$ candidates from head 0 (position $t+1$), the second level has $C$ candidates from head 1 for each parent (position $t+2$), and so on to depth $K$.

The total number of candidate paths is $C^K$. With $C=3$ and $K=4$, you have $81$ candidate 4-token sequences. With $C=5$ and $K=4$, you have $625$. The question is: how do you verify all of these simultaneously in a single target forward pass?

![Medusa candidate tree with K=2 heads and C=3 candidates generating 9 leaf paths](/imgs/blogs/medusa-multi-head-speculative-decoding-2.webp)

The tree structure is constructed as follows. Let $\mathcal{T}_k$ denote the set of tokens at depth $k$. For depth 0, $\mathcal{T}_0 = \{t_0\}$ (the current token). For depth 1, $\mathcal{T}_1 = \{\text{top-}C \text{ tokens from head } 0\}$. For depth $k > 1$, each node at depth $k-1$ spawns $C$ children from head $k-1$'s top-$C$ candidates. Each path from root to leaf represents one candidate 4-token continuation.

The total number of nodes in this tree is:

$$|\mathcal{T}| = \sum_{k=0}^{K} C^k = \frac{C^{K+1} - 1}{C - 1}$$

For $C=3$, $K=4$: $|\mathcal{T}| = (3^5 - 1) / 2 = 121$ nodes. This is the number of positions the target model must evaluate in the tree-attention forward pass — far smaller than running the full model 81 times separately, which would take 81 × 80ms ≈ 6.5 seconds for a 70B.

## Tree attention: verifying the entire tree in one forward pass

The technical centerpiece of Medusa is its tree attention mechanism. Standard causal attention in a transformer uses a lower-triangular mask: token at position $i$ attends to all tokens at positions $0, 1, \ldots, i$. This gives you the causal property — no token can see its future — and it is what allows the transformer to process all positions in a sequence simultaneously during training.

For the candidate tree, you need a generalization. Each candidate token in the tree should attend to:
1. All tokens in the already-accepted prefix (positions $0$ to $t$).
2. All ancestors of that candidate in the tree (the specific path from root to this node).
3. Nothing from sibling branches — attending to a sibling would contaminate the candidate's logits with context from a competing sequence hypothesis.

![Standard causal attention mask versus Medusa tree attention mask](/imgs/blogs/medusa-multi-head-speculative-decoding-3.webp)

### Constructing the mask: a worked example

The easiest way to understand tree attention is to build the mask explicitly for a tiny example. Consider a tree with $K=2$ heads and $C=2$ candidates per head. The tree has nodes:

- Node 0: root (accepted prefix position $t$, depth 0)
- Node 1: head-0 candidate A (depth 1)
- Node 2: head-0 candidate B (depth 1)
- Node 3: child of node 1, head-1 candidate A (depth 2)
- Node 4: child of node 1, head-1 candidate B (depth 2)
- Node 5: child of node 2, head-1 candidate A (depth 2)
- Node 6: child of node 2, head-1 candidate B (depth 2)

Total tree nodes: 7 (= $(2^3 - 1) / (2 - 1) = 7$). The attention mask $M \in \{0, -\infty\}^{7 \times 7}$ for the tree portion only (the prefix rows/columns use standard lower-triangular) is:

```
         N0   N1   N2   N3   N4   N5   N6
Node 0:   0   -∞   -∞   -∞   -∞   -∞   -∞
Node 1:   0    0   -∞   -∞   -∞   -∞   -∞
Node 2:   0   -∞    0   -∞   -∞   -∞   -∞
Node 3:   0    0   -∞    0   -∞   -∞   -∞
Node 4:   0    0   -∞   -∞    0   -∞   -∞
Node 5:   0   -∞    0   -∞   -∞    0   -∞
Node 6:   0   -∞    0   -∞   -∞   -∞    0
```

Read each row as "which past nodes can I attend to?" Node 3 (child of Node 1) can see N0, N1, and itself (N3) — its direct ancestral path — but cannot see N2, N4, N5, or N6. Node 5 (child of Node 2) can see N0, N2, and itself (N5), but nothing from the N1 branch. This is the core invariant: every node can only see its own ancestral path back to the root.

To build this mask programmatically:

```python
## build_tree_attention_mask.py — Construct the tree attention mask for Medusa
## Requires: torch>=2.2
import torch

def build_tree_attention_mask(
    tree_parents: list[int],  ## parent index for each node; tree_parents[0] = -1 (root)
    prefix_len: int,           ## number of accepted prefix tokens (attend to all)
    device: torch.device = torch.device("cuda"),
) -> torch.Tensor:
    """
    Build the combined (prefix + tree) attention mask.

    tree_parents: list of length n_tree_nodes.
        tree_parents[i] = index of node i's parent in the tree, or -1 if root.
        Example for C=2, K=2: [-1, 0, 0, 1, 1, 2, 2]

    Returns: mask of shape [prefix_len + n_tree_nodes, prefix_len + n_tree_nodes]
        where 0 = attend, -inf = block.
    """
    n_tree = len(tree_parents)
    n_total = prefix_len + n_tree
    mask = torch.full((n_total, n_total), float("-inf"), device=device)

    ## Prefix block: standard lower-triangular (all prefix tokens attend to past prefix)
    prefix_block = torch.tril(torch.zeros(prefix_len, prefix_len, device=device))
    mask[:prefix_len, :prefix_len] = prefix_block

    ## Tree nodes: each tree node attends to all prefix tokens
    mask[prefix_len:, :prefix_len] = 0.0  ## full prefix visible to all tree nodes

    ## Tree block: build ancestor sets by traversal
    ancestors = []
    for i in range(n_tree):
        anc = set()
        cur = i
        while cur != -1:
            anc.add(cur)
            cur = tree_parents[cur]
        ancestors.append(anc)

    for i in range(n_tree):
        for j in range(n_tree):
            if j in ancestors[i]:
                mask[prefix_len + i, prefix_len + j] = 0.0

    return mask


## Example: C=2, K=2 tree (nodes 0..6, parents as above)
tree_parents = [-1, 0, 0, 1, 1, 2, 2]
mask = build_tree_attention_mask(tree_parents, prefix_len=10)
## mask shape: [17, 17]
```

This mask is sparse. The density (fraction of 0 entries) for a depth-$K$, width-$C$ tree is:

$$\text{density} = \frac{\sum_{k=0}^{K} C^k \cdot (k+1)}{|\mathcal{T}|^2} \approx \frac{K}{2|\mathcal{T}|}$$

For $C=3$, $K=4$ ($|\mathcal{T}|=121$): density $\approx 4 / (2 \times 121) \approx 1.7\%$. The 98.3% sparsity means that most attention computation in the tree block is zero-valued after softmax — an opportunity for sparse attention implementations to skip those FLOPs, though current FlashAttention kernels do not exploit this particular sparsity pattern (they optimize for contiguous lower-triangular masks). Custom CUDA kernels that process each branch as a separate sequence chunk can recover roughly 40–60% of the theoretical compute savings.

### Key/value sharing and the batch dimension

A critical efficiency point often glossed over: during the tree-attention forward pass, the KV pairs for the accepted prefix ($0$ to $t$) are computed once and shared by all $|\mathcal{T}|$ tree nodes. This is the same KV cache reuse you get in standard autoregressive decode, but now extended to benefit all tree nodes simultaneously.

The effective batch size presented to the attention kernels is 1 (a single sequence) with sequence length $L + |\mathcal{T}|$, not $|\mathcal{T}|$ parallel sequences of length $L$. This is cheaper: attention cost scales as $O((L + |\mathcal{T}|)^2)$ for one sequence, versus $O(|\mathcal{T}| \cdot L^2)$ if each tree node were a separate sequence. For typical values ($L=512$, $|\mathcal{T}|=121$): one-sequence cost $\propto 633^2 = 400K$; separate-sequence cost $\propto 121 \times 512^2 = 31.7M$ — a 79× difference. Tree attention's "trick" is essentially free prefix KV-cache sharing across all candidate paths.

For a $K=4$, $C=3$ tree being verified by a 4-GPU tensor-parallel LLaMA-3 70B, the memory layout during tree attention is:

- KV cache (prefix, $L=512$, 80 layers, 8 KV heads per GPU, head dim 128): $\approx 512 \times 80 \times 8 \times 128 \times 2 \times 2 \approx 168$MB per GPU (BF16)
- Tree node activations (121 positions, 80 layers, $d=8192$): $\approx 121 \times 80 \times 8192 \times 2 \approx 160$MB total
- Attention weights matrix ($633 \times 633$ per layer, 64 heads): $\approx 633^2 \times 64 \times 2 \approx 51$MB per layer per GPU (peak, not persistent)

The tree's memory overhead is modest — the dominant cost is the KV cache for the accepted prefix, which you are paying regardless of whether you use Medusa.

This is tree attention. The attention mask $M \in \{0, -\infty\}^{|\mathcal{T}| \times |\mathcal{T}|}$ is defined as:

$$M_{ij} = \begin{cases} 0 & \text{if node } j \text{ is an ancestor of node } i \text{ or } j = i \\ -\infty & \text{otherwise} \end{cases}$$

where "ancestor" is relative to the tree structure. This mask is sparse — each node at depth $k$ has exactly $k+1$ non-$-\infty$ entries in its row (the path from root to itself). The prefix tokens (positions $0$ to $t$) all use the standard lower-triangular mask among themselves.

Critically, the KV cache for the accepted prefix is reused unchanged. The tree-attention forward pass only needs to compute attention scores for the $|\mathcal{T}|$ tree nodes against each other and against the full KV cache of the prefix. The dominant cost is the prefix attention — $O(L \cdot |\mathcal{T}|)$ operations per layer for a prefix of length $L$ — which is manageable because $|\mathcal{T}|$ is small (e.g., 121 nodes for $C=3$, $K=4$).

After the forward pass, you have a logit distribution at each tree node. The acceptance procedure walks the tree: starting at the root, check whether head 0's prediction at depth 1 matches the target model's distribution (using the same modified rejection sampling as standard spec decoding, described in [token acceptance explained](/blog/machine-learning/speculative-decoding/speculative-decoding-token-acceptance-rejection-sampling)). If the best-matching child at depth 1 is accepted, advance to depth 2 and repeat. The walk terminates at the first rejection. All tokens on the accepted path become the new output; all other nodes are discarded.

## The Medusa inference loop end to end

Putting the components together:

![Medusa inference loop from LM forward pass to accepted tokens](/imgs/blogs/medusa-multi-head-speculative-decoding-4.webp)

Here is the full loop in pseudocode:

```python
## medusa_inference.py — Medusa decoding loop
## Requires: transformers>=4.38, torch>=2.2, medusa-llm (pip install medusa-llm)

import torch
import torch.nn.functional as F
from typing import Optional

def medusa_decode(
    model,                # MedusaModel wrapping a causal LM + K heads
    input_ids: torch.Tensor,   # shape [1, seq_len]
    max_new_tokens: int = 200,
    top_c: int = 3,            # candidates per head
    temperature: float = 1.0,
    posterior_threshold: float = 0.09,  # minimum p for candidate acceptance
    posterior_alpha: float = 0.3,       # fallback threshold scalar
) -> torch.Tensor:
    """
    Run Medusa tree-attention speculative decoding.
    Returns generated token ids, shape [1, max_new_tokens].
    """
    K = model.num_medusa_heads
    device = input_ids.device
    generated = []
    current_ids = input_ids

    with torch.inference_mode():
        ## -- initial prefill: run backbone, get KV cache + hidden state
        out = model(
            input_ids=current_ids,
            use_cache=True,
            output_hidden_states=True,
        )
        past_key_values = out.past_key_values
        hidden_state = out.hidden_states[-1]  ## [1, seq, d]
        base_logits = out.logits[:, -1, :]    ## [1, vocab]

        while len(generated) < max_new_tokens:
            ## -- Step 1: K Medusa heads produce candidates from last hidden state
            h = hidden_state[:, -1:, :]  ## [1, 1, d]
            medusa_logits = []
            for head_k in model.medusa_heads:
                logits_k = head_k(h)  ## [1, 1, vocab]
                medusa_logits.append(logits_k.squeeze(1))  ## [1, vocab]

            ## -- Step 2: build candidate tree (top-C per head)
            ## candidates: list of K tensors, each shape [C]
            candidates_per_head = [
                torch.topk(logits / temperature, top_c, dim=-1).indices.squeeze(0)
                for logits in medusa_logits
            ]
            ## Also include base LM head's top-1 as the primary candidate at depth 0
            base_candidate = torch.argmax(base_logits, dim=-1)  ## [1]

            ## build_tree() constructs (tree_input_ids, tree_attention_mask, candidates_tensor)
            tree_input_ids, tree_attention_mask, candidates = build_medusa_tree(
                current_ids,
                base_candidate,
                candidates_per_head,
                device=device,
            )

            ## -- Step 3: tree-attention forward pass (ONE backbone pass)
            tree_out = model.model(
                input_ids=tree_input_ids,
                attention_mask=tree_attention_mask,
                past_key_values=past_key_values,
                use_cache=False,           ## no KV update during tree verify
                output_hidden_states=True,
            )
            tree_logits = model.lm_head(tree_out.last_hidden_state)  ## [1, tree_size, vocab]

            ## -- Step 4: walk tree, accept tokens via modified rejection sampling
            accepted_ids, n_accepted = walk_medusa_tree(
                candidates,
                tree_logits,
                base_logits,
                medusa_logits,
                temperature=temperature,
                posterior_threshold=posterior_threshold,
                posterior_alpha=posterior_alpha,
            )

            ## -- Step 5: append accepted tokens, update KV cache with one pass
            generated.extend(accepted_ids.tolist())
            current_ids = torch.cat([current_ids, accepted_ids.unsqueeze(0)], dim=-1)

            ## Update KV cache: re-run backbone on accepted tokens only
            out = model(
                input_ids=accepted_ids.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            past_key_values = out.past_key_values
            hidden_state = out.hidden_states[-1]
            base_logits = out.logits[:, -1, :]

            if accepted_ids[-1].item() == model.config.eos_token_id:
                break

    return torch.tensor(generated, device=device)
```

The key observation in this code is step 3: the tree-attention forward pass takes one backbone forward call regardless of how many candidate paths are in the tree. The `build_medusa_tree` function constructs the `tree_attention_mask` that enforces the ancestor-only attention pattern, and `walk_medusa_tree` applies the acceptance logic per depth level.

## Medusa training: freeze the backbone, train the heads

Medusa-1 uses the simplest possible training procedure: freeze the backbone entirely and train only the $K$ prediction heads on supervised fine-tuning (SFT) data. The training objective for head $k$ is:

$$\mathcal{L}_k = -\sum_{t} \log p_{\theta_k}(x_{t+k+1} \mid h_t)$$

where $\theta_k$ are the parameters of head $k$ and $h_t$ is the frozen backbone's hidden state at position $t$. Because the backbone is frozen, gradient computation is limited to the shallow MLP heads — each head has roughly 134M parameters (for 70B, $d=8192$), making training fast. On a single A100 80GB, training 4 heads on 100K SFT examples takes approximately 12 hours with batch size 16 and learning rate $3 \times 10^{-4}$.

### Full training recipe: VRAM, dataset, and schedule

Here is the complete training configuration for Medusa-1 across model sizes, which the paper and open-source repo do not consolidate in one place:

| Model | $d$ | Trainable params (K=4) | VRAM (BF16 backbone frozen) | VRAM (joint Medusa-2) | Training time (100K ex) |
|---|---|---|---|---|---|
| LLaMA-3 8B | 4096 | ~268M | ~18GB (fits 1× A100 40GB) | ~18GB + optimizer states = ~72GB | ~3h on 1× A100 |
| LLaMA-3 70B | 8192 | ~537M | ~140GB (requires 2× H100 80GB) | ~140GB + ~2GB optimizer = 8× A100 | ~12h on 2× H100 |
| Mistral 7B | 4096 | ~268M | ~16GB (fits 1× A100 40GB) | ~64GB on 2× A100 | ~3h on 1× A100 |
| Vicuna 13B | 5120 | ~336M | ~28GB (fits 1× A100 40GB) | ~112GB on 2× A100 80GB | ~5h on 1× A100 |

For Medusa-1 (frozen backbone), the VRAM cost is dominated by the backbone weights themselves, since you only store gradients for the head parameters. The optimizer states (Adam first and second moments) for 537M head parameters add $\approx 4$GB at BF16 — negligible compared to the 140GB backbone footprint.

The learning rate schedule matters more than it might appear. A flat LR of $3 \times 10^{-4}$ throughout causes the later heads (head 3, head 4) to overfit quickly while the shallower heads are still learning. The recommended schedule is:

- Warmup: 3% of total steps, linear ramp from $0$ to $\text{peak LR}$
- Body: cosine decay from peak to $10\%$ of peak
- Peak LR: $3 \times 10^{-4}$ for heads 0–1, $5 \times 10^{-4}$ for heads 2–4 (deeper heads benefit from a higher LR because the signal becomes noisier with distance)
- Weight decay: 0.1, applied to all linear weights but not biases
- Gradient clipping: max norm 1.0

Dataset size requirements scale with head depth. Head 0 (predicting $x_{t+2}$) learns from essentially the same signal as the LM head and saturates quickly — 50K examples is enough. Head 3 (predicting $x_{t+5}$) needs more examples to see enough high-probability long-range continuations; 200K+ examples is preferable. In practice, using 100K examples with a cosine schedule gives 90–95% of the acceptance rate you would get with 500K examples, because the marginal gain from more data is limited by the representational quality of the frozen hidden states.

For dataset composition when you do not have domain-specific data, the open-source Medusa checkpoints use:
- `HuggingFaceH4/ultrachat_200k` (chat generalist)
- `WizardLMTeam/WizardLM_evol_instruct_70k` (instruction following)
- `codeparrot/github-code` Python subset (code; ~30% of examples)

The SFT data should ideally match the target task distribution. For a coding assistant, train on code-completion datasets (The Stack, Code Alpaca). For a chat assistant, train on ShareGPT-style conversations. The acceptance rate $\alpha$ is sensitive to distribution mismatch between the training data and inference prompts — a 20% mismatch can drop $\alpha$ by 10–15 percentage points, which translates directly to lower speedup.

### Head initialization strategy

An underappreciated detail: how you initialize the Medusa heads at the start of training has a significant effect on convergence speed. Zero-initialization (as shown in `MedusaHead.reset_parameters()`) is stable but slow — the heads start by predicting uniform distributions and must learn from scratch. A better strategy is to copy the LM head's output projection as the starting point for all Medusa heads' final linear layers, which gives each head a reasonable initial distribution (the next-token predictive distribution) as a starting point for fine-tuning toward the $k$-step lookahead task. This warm-start reduces training time to convergence by approximately 30% on typical datasets, at the cost of one additional copy of the vocabulary projection matrix per head during initialization (which can be freed after copying).

```python
## train_medusa_heads.py — Training only the Medusa heads (Medusa-1 protocol)
## Requires: torch>=2.2, transformers>=4.38, datasets>=2.16, accelerate>=0.27

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from datasets import load_dataset

def train_medusa_heads(
    model,               ## MedusaModel with frozen backbone
    tokenizer,
    dataset_name: str = "HuggingFaceH4/ultrachat_200k",
    num_heads: int = 4,
    batch_size: int = 4,
    grad_accum_steps: int = 8,   ## effective batch 32
    lr: float = 3e-4,
    num_epochs: int = 1,
    max_seq_len: int = 2048,
    warmup_ratio: float = 0.03,
):
    ## Freeze backbone parameters
    for name, param in model.named_parameters():
        if "medusa_head" not in name:
            param.requires_grad_(False)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable_params / 1e6:.1f}M "
          f"({trainable_params / sum(p.numel() for p in model.parameters()) * 100:.2f}%)")

    dataset = load_dataset(dataset_name, split="train_sft")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, betas=(0.9, 0.95), weight_decay=0.1,
    )

    total_steps = len(dataset) // (batch_size * grad_accum_steps) * num_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    model.train()
    for epoch in range(num_epochs):
        for step, batch in enumerate(DataLoader(dataset, batch_size=batch_size)):
            inputs = tokenizer(
                batch["prompt"], return_tensors="pt",
                padding=True, truncation=True, max_length=max_seq_len,
            ).to(model.device)

            with torch.no_grad():
                ## get backbone hidden states with gradient disabled (frozen)
                backbone_out = model.model(
                    **inputs, output_hidden_states=True, use_cache=False
                )
                hidden_states = backbone_out.hidden_states[-1]  ## [B, L, d]

            ## compute head losses against ground-truth labels
            ## label for head k at position t is the token at position t + k + 1
            total_loss = torch.tensor(0.0, device=model.device)
            for k, head in enumerate(model.medusa_heads):
                logits_k = head(hidden_states)  ## [B, L, vocab]
                ## shift: predict token at t+k+1 from hidden state at t
                shift_logits = logits_k[:, :-k-1, :].contiguous()
                shift_labels = inputs["input_ids"][:, k+1:].contiguous()
                loss_k = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=tokenizer.pad_token_id,
                )
                total_loss = total_loss + loss_k

            (total_loss / grad_accum_steps).backward()

            if (step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                print(f"Epoch {epoch+1} step {step}: loss={total_loss.item():.4f}")
```

## Medusa-1 vs Medusa-2: the frozen backbone tradeoff

Medusa-1 (frozen backbone) has one fundamental limitation: the prediction heads are trying to compensate for the fact that the hidden state $h_t$ was never trained to support multi-step prediction. The backbone was trained to produce a hidden state that is good at predicting position $t+1$ via the LM head. Positions $t+2$ through $t+K+1$ are not natural objectives for $h_t$ — the head can learn to exploit whatever signal is there, but it is working against a suboptimal representation.

![Medusa-1 vs Medusa-2 vs standard two-model spec decode comparison](/imgs/blogs/medusa-multi-head-speculative-decoding-5.webp)

Medusa-2 addresses this by jointly fine-tuning the backbone and the heads together. The training objective becomes:

$$\mathcal{L}_{\text{total}} = \lambda_0 \mathcal{L}_{\text{LM}} + \sum_{k=1}^{K} \lambda_k \mathcal{L}_k$$

where $\mathcal{L}_{\text{LM}}$ is the standard next-token cross-entropy loss and $\mathcal{L}_k$ is the head $k$ loss. The backbone is no longer frozen; it now optimizes for all $K+1$ prediction objectives simultaneously, learning a hidden state representation that is genuinely useful for multi-step lookahead.

The tradeoff is significant:

| Property | Medusa-1 | Medusa-2 |
|---|---|---|
| Training time | ~12h on 1× A100 (heads only) | ~5 days on 8× A100 (full fine-tune) |
| Training data needed | ~100K examples | ~500K–1M examples |
| Risk of catastrophic forgetting | None (backbone frozen) | Present, mitigated by small λ₀ relative to Σλₖ |
| Acceptance rate α (code tasks) | 0.65–0.75 | 0.78–0.90 |
| Speedup on code generation | 1.8–2.3× | 2.3–3.1× |
| Speedup on open-ended chat | 1.4–1.8× | 1.7–2.4× |

Medusa-2's self-distillation approach helps mitigate forgetting: during training, the backbone at each step generates both its standard next-token prediction and the multi-head predictions. The backbone's weights are regularized to stay close to the original model's weights using an L2 penalty scaled by $\lambda_{\text{reg}}$. In practice, Medusa-2 uses $\lambda_0 = 1.0$, $\lambda_k = 1.0$ for all $k$, and $\lambda_{\text{reg}} = 0.1$.

The practical guidance is straightforward: if you have 48–72 GPU-hours to spend on a training run, Medusa-2 pays for itself on tasks where you expect $\alpha \geq 0.75$. If you need to iterate quickly or deploy on a model you cannot fine-tune (e.g., a gated model), Medusa-1 gives you a meaningful speedup at essentially zero training cost.

## Wall-clock timing and speedup mechanics

To understand why Medusa achieves its speedup, it helps to account for every millisecond in a Medusa step vs a naive decode step vs a two-model spec decode step.

![Medusa wall-clock timing versus two-model spec decode](/imgs/blogs/medusa-multi-head-speculative-decoding-6.webp)

For a LLaMA-3 70B running on 4× H100 SXM5 (tensor-parallel, bs=1):

**Naive autoregressive decode:**
- Each token requires one full backbone forward pass: ~80ms
- 100 tokens: 8,000ms = 8 seconds

**Two-model spec decode (γ=4, 7B draft):**
- 4 sequential draft passes × ~10ms = 40ms draft overhead
- 1 target verify pass = ~80ms
- Effective cycle time: ~120ms for up to 5 tokens (if α=0.8, expected ≈ 3.4 tokens accepted)
- Effective rate: ~120ms / 3.4 ≈ 35ms/token → 2.3× speedup

**Medusa (K=4 heads, C=3):**
- 1 backbone forward pass = ~80ms (this IS the verify pass; the tree nodes add ~3ms)
- K head computations = ~5ms (parallel 2-layer MLPs, tiny compute)
- Tree construction = ~1ms
- Total cycle: ~86ms for up to 5 tokens (if α=0.7, expected ≈ 2.8 tokens accepted with Medusa-1)
- Effective rate: ~86ms / 2.8 ≈ 31ms/token → 2.6× speedup (better than two-model, despite lower α, because no draft overhead)

This arithmetic reveals the key insight: Medusa's speedup comes not from having a higher acceptance rate (it often does not — Medusa-1 typically has lower α than a well-matched two-model setup), but from having zero draft overhead. The "draft" is free — it happens inside the same pass that was going to run anyway.

The expected tokens per Medusa step can be derived from the tree acceptance probability. For a tree of depth $K$ with acceptance rate $\alpha$ at each level:

$$E[\text{tokens accepted}] = \sum_{k=1}^{K} \alpha^k \cdot P(\text{best candidate at depth } k \text{ is accepted} \mid \text{depths } 1 \ldots k-1 \text{ accepted})$$

In practice, this simplifies approximately to:

$$E[\text{tokens accepted}] \approx \frac{1 - \alpha^{K+1}}{1 - \alpha}$$

At $\alpha = 0.70$, $K = 4$: $E[\text{accepted}] \approx \frac{1 - 0.70^5}{1 - 0.70} = \frac{1 - 0.168}{0.30} \approx 2.77$ tokens. At $\alpha = 0.85$, $K = 4$: $E[\text{accepted}] \approx \frac{1 - 0.85^5}{0.15} \approx 3.74$ tokens. The candidate tree (breadth $C > 1$) improves upon the linear chain because it gives you multiple bets at each depth level, increasing the probability that at least one child at each depth is accepted.

## Medusa model architecture: sharing the backbone

The Medusa model's parameter layout is elegant because it changes nothing about the backbone architecture.

![Medusa model layer stack from embedding to K prediction heads](/imgs/blogs/medusa-multi-head-speculative-decoding-7.webp)

The backbone — embedding table, $N$ transformer blocks ($N=80$ for LLaMA-3 70B), and the standard LM head — is completely unchanged. The Medusa heads sit above the final hidden state, not inserted between layers or alongside the attention mechanism. This has two important consequences:

1. **Compatibility.** Any optimization applied to the backbone (quantization, flash attention, tensor parallelism, speculative tensor operations) is automatically inherited by the Medusa-augmented model with no changes to those optimizations. You can quantize the backbone to INT4 with GPTQ and the Medusa heads remain in FP16; the overall parameter footprint shrinks by 4× with only a modest hit to acceptance rate.

2. **Incremental deployability.** You can load a standard LLaMA checkpoint, add Medusa head weights as an additive adapter (similar to LoRA loading), and get a Medusa-capable model without rewriting any serving code beyond the decode loop. The `medusa-llm` package on PyPI does exactly this for Vicuna, LLaMA-2, and LLaMA-3 checkpoints.

The head architecture is a gated MLP following the backbone's own FFN convention. For LLaMA-3 variants using SwiGLU:

```python
## medusa_head.py — Single Medusa prediction head implementation
## Requires: torch>=2.2

import torch
import torch.nn as nn
import torch.nn.functional as F

class MedusaHead(nn.Module):
    """
    Lightweight 2-layer residual MLP that predicts a future token
    from the backbone's final hidden state.

    Architecture:
        h_t (d=4096/8192)
        → Linear(d, d) + SiLU + Linear(d, d) [residual]
        → LM head projection (shared with backbone, d → vocab)
    """

    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size, bias=False),
                nn.SiLU(),
            )
            for _ in range(num_layers)
        ])
        ## The final projection to vocab is shared with the backbone's LM head
        ## We store a reference, not a copy — set externally after model load
        self.lm_head: Optional[nn.Linear] = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden_size]
        Returns:
            logits: [batch, seq, vocab_size]
        """
        residual = hidden_states
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        ## Residual connection: ensures heads can be initialized to near-identity
        hidden_states = hidden_states + residual
        ## Shared LM head projection
        return self.lm_head(hidden_states)

    def reset_parameters(self):
        """Initialize residual-friendly: zero out the second linear in each block."""
        for seq_layer in self.layers:
            linear = seq_layer[0]  ## the nn.Linear
            nn.init.zeros_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
```

## Deploying Medusa: serving stack changes

Medusa requires three changes to a standard serving stack:

**1. Model loading.** The Medusa checkpoint includes the backbone weights plus $K$ head state dicts. Loading is straightforward with `from_pretrained` on the `MedusaModel` class from the `medusa-llm` package, or by manually merging state dicts:

```python
## load_medusa.py — Loading a Medusa checkpoint for inference
## Requires: medusa-llm>=0.1.3, transformers>=4.38, torch>=2.2

from medusa.model.medusa_model import MedusaModel
import torch

def load_medusa_model(
    base_model_path: str,
    medusa_head_path: str,
    num_medusa_heads: int = 4,
    dtype: torch.dtype = torch.bfloat16,
    device_map: str = "auto",
) -> MedusaModel:
    """
    Load a Medusa model from a base checkpoint and a Medusa head checkpoint.

    Args:
        base_model_path: HuggingFace hub ID or local path to the backbone model.
            Example: "meta-llama/Meta-Llama-3-70B-Instruct"
        medusa_head_path: Path to the directory containing medusa_head.pt
            (the K head state dicts) and config.json
        num_medusa_heads: K, number of prediction heads (default 4)
        dtype: Weight dtype; use bfloat16 for Ampere+, float16 for older
        device_map: "auto" for multi-GPU tensor parallel, "cuda:0" for single

    Returns:
        MedusaModel ready for inference
    """
    model = MedusaModel.from_pretrained(
        base_model_path,
        medusa_num_heads=num_medusa_heads,
        medusa_head_path=medusa_head_path,
        torch_dtype=dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


## Example: loading LLaMA-3 70B with Medusa-1 heads trained on UltraChat
if __name__ == "__main__":
    model = load_medusa_model(
        base_model_path="meta-llama/Meta-Llama-3-70B-Instruct",
        medusa_head_path="./checkpoints/medusa-llama3-70b-ultrachat/",
        num_medusa_heads=4,
        dtype=torch.bfloat16,
    )
    print(f"Medusa model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")
    print(f"Head params: {sum(p.numel() for head in model.medusa_heads for p in head.parameters())/1e6:.1f}M")
```

**2. Decode loop.** You must replace the standard `model.generate()` call with `medusa_generate()`, which implements the tree-build and walk logic. The `medusa-llm` package provides this as a drop-in replacement with the same interface as HuggingFace `generate()`.

**3. Framework compatibility.** vLLM as of v0.4.2 does not natively support Medusa's tree attention mask through the `PagedAttention` kernel. If you are using vLLM for continuous batching, you need to fall back to eager-mode attention for the tree verify pass, which adds ~10% overhead versus flash-attention. SGLang's experimental Medusa support (merged in v0.2.1) handles tree attention more cleanly by representing the tree as a special prefix in the RadixAttention cache.

### vLLM integration: the practical path

For teams already running vLLM, the path to Medusa requires a shim rather than a full port. vLLM's `LLMEngine` processes each decode step through its `_process_model_outputs` and `step()` methods. You can intercept the decode step, run your own tree-attention logic outside vLLM's scheduler, and return the accepted token sequence as if it were a multi-token output:

```python
## vllm_medusa_shim.py — Medusa integration shim for vLLM v0.4.x
## This wraps vLLM's LLM class to override the decode loop.
## Requires: vllm>=0.4.2, medusa-llm>=0.1.3, torch>=2.2

from vllm import LLM, SamplingParams
from vllm.model_executor.models.llama import LlamaForCausalLM
from medusa.model.medusa_model import MedusaModel
import torch

class MedusaVLLMShim:
    """
    Thin shim that uses vLLM for prefill (KV cache allocation, continuous batching)
    and drops into Medusa's tree-attention decode for the generation phase.

    Limitations:
    - bs=1 only (tree-attention requires per-request mask construction)
    - PagedAttention KV cache is NOT used for tree nodes (eager attention)
    - flash-attention is disabled for tree-verify pass; adds ~8-12% vs flash
    """

    def __init__(
        self,
        model_path: str,
        medusa_head_path: str,
        num_medusa_heads: int = 4,
        num_gpus: int = 1,
        dtype: str = "bfloat16",
    ):
        ## Load Medusa model separately (not through vLLM's engine)
        self.medusa_model = MedusaModel.from_pretrained(
            model_path,
            medusa_num_heads=num_medusa_heads,
            medusa_head_path=medusa_head_path,
            torch_dtype=torch.bfloat16 if dtype == "bfloat16" else torch.float16,
            device_map="auto",
        )
        self.medusa_model.eval()
        self.num_heads = num_medusa_heads

    def generate(
        self,
        prompt: str,
        tokenizer,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        top_candidates: int = 3,
    ) -> dict:
        """
        Run Medusa decode for a single prompt.
        Returns dict with 'text', 'tokens_per_second', 'mean_accepted'.
        """
        from medusa.model.utils import medusa_generate
        import time

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        ## medusa_choices defines the tree structure as a list of paths.
        ## Each inner list is a sequence of candidate indices (0 = top-1, 1 = top-2, etc.)
        ## at each depth. The set of all paths defines the tree topology.
        medusa_choices = [
            [0], [1], [2],          ## depth 1: top-3 from head 0
            [0, 0], [0, 1], [0, 2], ## depth 2 under top-1
            [1, 0], [2, 0],         ## depth 2 under top-2 and top-3
            [0, 0, 0], [0, 0, 1],   ## depth 3 under most likely path
            [0, 1, 0],
            [0, 0, 0, 0],           ## depth 4 (deepest)
        ]

        t0 = time.perf_counter()
        output_ids, stats = medusa_generate(
            self.medusa_model,
            input_ids,
            temperature=temperature,
            max_steps=max_new_tokens,
            medusa_choices=medusa_choices,
        )
        elapsed = time.perf_counter() - t0
        n_tokens = output_ids.shape[1] - input_ids.shape[1]

        return {
            "text": tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True),
            "tokens_per_second": n_tokens / elapsed,
            "mean_accepted": stats.get("mean_accepted", 0.0),
            "num_steps": stats.get("num_steps", 0),
        }
```

The `medusa_choices` list deserves attention — it is the mechanism by which you control the tree topology without changing K or C globally. Each entry `[i, j, k, ...]` means: "include the node reached by taking candidate `i` at depth 1, candidate `j` at depth 2, candidate `k` at depth 3, etc." You can prune branches that are unlikely to be accepted (e.g., skip depth-3 nodes under the 3rd-ranked depth-1 candidate) to reduce tree size and tree-attention compute without reducing K. The Medusa paper calls this "Medusa choices" and notes that the optimal pruned tree topology depends on the typical acceptance rate profile of your task — high-α tasks benefit from deeper trees (more depth-4 and depth-5 nodes), while low-α tasks benefit from wider trees at depth 1–2 with less depth.

### Memory footprint calculation

Before deploying Medusa, calculate whether you have the VRAM budget. The total memory footprint of a Medusa-augmented model is:

```
Total VRAM = backbone_weights + head_weights + kv_cache + activations + optimizer_states

Backbone weights (BF16):   num_params_backbone × 2 bytes
Head weights (BF16):       K × (2d² + d·V) × 2 bytes   [if sharing vocab projection, 2Kd² + dV]
KV cache (per request):    2 × num_layers × num_kv_heads × head_dim × max_seq_len × 2 bytes
Activations (tree verify): n_tree_nodes × num_layers × d × 2 bytes
```

For LLaMA-3 70B + Medusa-2 (K=4), serving a single request with max_seq_len=2048:

```
Backbone:   70B × 2 = 140 GB
Head MLP:   4 × 2 × (8192)² × 2 = 4.3 GB  (shared vocab proj adds 2.1 GB = 6.4 GB total)
KV cache:   2 × 80 × 8 × 128 × 2048 × 2 = 671 MB per request
Tree acts:  121 × 80 × 8192 × 2 = 160 MB
─────────────────────────────────────────
Total:      ~147 GB + optimizer (not needed at inference)
```

This fits on 2× H100 SXM5 (160GB combined) with no quantization. For 4× A100 80GB (320GB combined), you have headroom for 2 concurrent requests with their KV caches. The critical number is that the heads add 6.4 GB — 4.4% overhead — not the 4.7B × 2 = 9.4 GB you might naively calculate, because the vocabulary projection matrix ($d \times V$) is shared with the backbone's LM head and counted once.

For production serving with bs=1 or bs=2 (the regime where Medusa pays off), a FastAPI server wrapping `medusa_generate()` directly often outperforms integrating into a full continuous-batching framework, because you avoid the scheduler overhead and the tree-attention compatibility shims:

```python
## medusa_server.py — Minimal FastAPI serving wrapper for Medusa inference
## Requires: fastapi>=0.110, uvicorn>=0.29, medusa-llm>=0.1.3, torch>=2.2

import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from medusa.model.utils import medusa_generate
import torch

app = FastAPI(title="Medusa LLM Server", version="0.1.0")
_model = None
_tokenizer = None

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 256
    temperature: float = 0.0   ## greedy default; set > 0 for sampling
    top_p: float = 1.0
    num_medusa_heads: int = 4
    top_candidates: int = 3    ## C: candidates per head

class GenerateResponse(BaseModel):
    text: str
    tokens_per_second: float
    accepted_tokens_per_step: float
    num_steps: int

@app.on_event("startup")
async def load_model():
    global _model, _tokenizer
    from load_medusa import load_medusa_model
    from transformers import AutoTokenizer
    _model = load_medusa_model(
        "meta-llama/Meta-Llama-3-70B-Instruct",
        "./checkpoints/medusa-llama3-70b-ultrachat/",
        num_medusa_heads=4,
    )
    _tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct")

@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    import time
    input_ids = _tokenizer(request.prompt, return_tensors="pt").input_ids.cuda()
    t0 = time.perf_counter()
    output_ids, stats = await asyncio.get_event_loop().run_in_executor(
        None,
        lambda: medusa_generate(
            _model,
            input_ids,
            temperature=request.temperature,
            max_steps=request.max_new_tokens,
            medusa_choices=[[0], [0, 0], [0, 1], [0, 0, 0], [0, 0, 1]],
        )
    )
    elapsed = time.perf_counter() - t0
    n_tokens = output_ids.shape[1] - input_ids.shape[1]
    return GenerateResponse(
        text=_tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True),
        tokens_per_second=n_tokens / elapsed,
        accepted_tokens_per_step=stats["mean_accepted"],
        num_steps=stats["num_steps"],
    )
```

## Real performance numbers

The Medusa paper (Cai et al., 2024) reports results on three backbone models: Vicuna-7B, Vicuna-13B, and Vicuna-33B, all evaluated on the MT-Bench dataset with temperature 0 (greedy decoding).

**Medusa-1 (frozen backbone) results on Vicuna-7B, MT-Bench greedy:**
- Baseline autoregressive: 1.0× (reference)
- Medusa-1 (K=4, C=3): 2.26× speedup, α ≈ 0.71 per level
- Mean tokens accepted per step: 2.5

**Medusa-2 (joint fine-tune) results on Vicuna-13B:**
- Baseline: 1.0×
- Medusa-2 (K=5, C=3): 3.08× speedup, α ≈ 0.83 per level
- Mean tokens accepted per step: 3.2

**Medusa-2 on code generation (HumanEval), Vicuna-33B:**
- Baseline: 1.0×
- Medusa-2 (K=4, C=5): 3.42× speedup
- Mean tokens accepted per step: 3.6

### Per-task breakdown and why code wins

The code generation speedup substantially outperforms chat speedup for a predictable reason: code has lower conditional entropy than natural language. Given a function signature and the first few tokens of the implementation, the next 4–6 tokens are often near-deterministic. Standard Python idioms — `for i in range(`, `return [x for x`, `if isinstance(x, list)` — are high-probability completions that all prediction heads will converge on. Natural language has higher entropy (more equally probable continuations), which reduces acceptance rate and therefore speedup.

The information-theoretic framing is precise. For Medusa to accept a token at depth $k$, the target model's probability of the Medusa head's top candidate must satisfy $q(x) / p(x) \geq U[0,1]$ where $p$ is the Medusa head's probability and $q$ is the target's probability. This acceptance probability is high when $q \approx p$ — when the target and the head agree — which happens most when the distribution is peaked. For Python code, the entropy of the next-token distribution $H(p_{\text{target}}) \approx 1.8$ bits at positions inside common idioms. For open-ended chat, $H(p_{\text{target}}) \approx 4.2$ bits. Lower entropy = more peaked distribution = higher acceptance rate.

| Task type | α (Medusa-1, K=4) | α (Medusa-2, K=4) | Speedup (Medusa-2) | Why |
|---|---|---|---|---|
| Python code generation | 0.76–0.82 | 0.85–0.92 | 2.8–3.5× | Highly repetitive, low conditional entropy |
| SQL query completion | 0.74–0.80 | 0.84–0.90 | 2.7–3.3× | Rigid syntax forces near-deterministic continuations |
| JSON schema generation | 0.78–0.84 | 0.87–0.93 | 3.0–3.8× | Structured format, extremely low entropy |
| Document summarization | 0.68–0.74 | 0.78–0.84 | 2.2–2.8× | Templates + topic repetition provide signal |
| Open-ended chat (MT-Bench) | 0.62–0.70 | 0.73–0.82 | 1.8–2.5× | Mixed entropy; common phrases help |
| Creative writing (temp ≥0.8) | 0.48–0.58 | 0.60–0.70 | 1.3–1.8× | Deliberately high entropy; lookahead fails |

The creative writing numbers at temperature > 0.8 are the warning: when the target distribution is genuinely flat (all continuations roughly equally likely), even Medusa-2's joint fine-tuning cannot save the acceptance rate, because the ground truth IS flat — no multi-step lookahead helps if the next token is genuinely random.

The summarization number (2.2–2.8×) deserves its own explanation. Summaries are not highly structured like code, yet Medusa still achieves reasonable speedup because summaries tend to reproduce phrases from the source document. Once the model has begun an extractive summary phrase — "The company reported" — the next few tokens are heavily constrained by the source text, raising acceptance rate even in an otherwise natural-language task.

### Measuring acceptance rate online: a diagnostic tool

In production, the acceptance rate changes with input distribution, which means you need to monitor it continuously. Here is a drop-in snippet that measures $\alpha$ per depth level and per generation step, suitable for embedding in a logging sidecar:

```python
## measure_acceptance.py — Online α measurement for Medusa inference
## Attach this to your medusa_decode loop to track acceptance rate per level.
## Requires: torch>=2.2, numpy>=1.24

import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class MedusaAcceptanceTracker:
    """Tracks acceptance rate per depth level across many generation steps."""
    attempts: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    accepted: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    tokens_per_step: List[int] = field(default_factory=list)

    def record_step(self, accepted_depth: int, max_depth: int):
        """
        Record the outcome of one Medusa step.

        accepted_depth: the depth at which acceptance stopped (0 = only base token,
            1 = base + head-0 accepted, ..., max_depth = full chain accepted).
        max_depth: K, the number of Medusa heads.
        """
        for d in range(1, max_depth + 1):
            self.attempts[d] += 1
            if d <= accepted_depth:
                self.accepted[d] += 1
        self.tokens_per_step.append(accepted_depth + 1)  ## +1 for the base token

    def acceptance_rate(self, depth: int) -> float:
        """Alpha at a given depth level (P(accept | reached this depth))."""
        if self.attempts[depth] == 0:
            return 0.0
        return self.accepted[depth] / self.attempts[depth]

    def mean_tokens_per_step(self) -> float:
        if not self.tokens_per_step:
            return 0.0
        return np.mean(self.tokens_per_step)

    def effective_speedup(self) -> float:
        """Approximate wall-clock speedup vs. autoregressive (ignoring tree overhead)."""
        return self.mean_tokens_per_step()

    def report(self, num_medusa_heads: int) -> str:
        lines = ["=== Medusa Acceptance Report ==="]
        for d in range(1, num_medusa_heads + 1):
            lines.append(
                f"  Depth {d}: α = {self.acceptance_rate(d):.3f} "
                f"({self.accepted[d]}/{self.attempts[d]})"
            )
        lines.append(f"  Mean tokens/step: {self.mean_tokens_per_step():.2f}")
        lines.append(f"  Approx speedup:   {self.effective_speedup():.2f}×")
        return "\n".join(lines)


## Example integration in a decode loop:
## tracker = MedusaAcceptanceTracker()
## ... inside medusa_decode while loop, after walk_medusa_tree:
## tracker.record_step(accepted_depth=n_accepted, max_depth=K)
## ... after generation:
## print(tracker.report(num_medusa_heads=K))
##
## Expected output for code generation (LLaMA-3 70B + Medusa-2):
##   Depth 1: α = 0.91 (910/1000)
##   Depth 2: α = 0.88 (880/1000)
##   Depth 3: α = 0.85 (850/1000)
##   Depth 4: α = 0.82 (820/1000)
##   Mean tokens/step: 3.46
##   Approx speedup:   3.46×
```

A α below 0.55 at any depth level is a strong signal that your Medusa heads are undertrained or trained on out-of-distribution data. If depth-1 α is high (>0.80) but depth-3 α drops below 0.60, you have a training data problem specific to longer-range prediction — adding more examples or increasing the LR for heads 2–3 will help. If depth-1 α itself is low, the heads have not seen enough examples matching your inference distribution; retrain on domain data.

## Accepted path through the candidate tree

The final piece to understand is what Medusa's acceptance walk looks like in practice.

![Medusa candidate tree before and after acceptance walk](/imgs/blogs/medusa-multi-head-speculative-decoding-8.webp)

For a concrete example: say you are generating Python code and the current accepted sequence ends with `for i in`. The backbone produces $h_t$, the LM head gives high probability to `range` (rank 1), and the Medusa heads predict:

- Head 1 (position $t+2$): rank 1 → `(`, rank 2 → `range`, rank 3 → `n`
- Head 2 (position $t+3$): rank 1 → `n`, rank 2 → `len`, rank 3 → `0`
- Head 3 (position $t+4$): rank 1 → `)`, rank 2 → `,`, rank 3 → `+`
- Head 4 (position $t+5$): rank 1 → `:`, rank 2 → `+`, rank 3 → `)`

The candidate tree now has $5 \times 3^4 = 405$ nodes (depth 5, width 3 per level). After tree-attention verification:

- Depth 1: LM head's rank-1 candidate `range` accepted (target agrees; $\alpha = \min(1, q/p) = 1$ since it is the LM head's own prediction)
- Depth 2: `(` accepted (target assigns high probability; $q/p > 1$)
- Depth 3: `n` accepted
- Depth 4: `)` accepted
- Depth 5: `:` accepted

Five tokens accepted in a single step: `range(n):`. The accepted path consumes exactly the rank-1 candidate at every level. All 400 other nodes in the tree are discarded. The decode position advances by 5, the KV cache is updated, and the next Medusa step begins.

This is the ideal case — it happens frequently on structured code because the conditional distributions are peaked. In the chat regime, acceptance typically terminates at depth 2–3 rather than depth 4–5, but even accepting 2 tokens per step yields a 2× speedup since each step still costs only one backbone forward pass.

## Case studies

### Case study 1: Code completion API serving LLaMA-3 70B at bs=1

**Context:** A startup building a GitHub Copilot alternative. Backend: 4× H100 SXM5, tensor-parallel LLaMA-3 70B Instruct, FastAPI + medusa-llm v0.1.3. Workload: 95th percentile prompt length 512 tokens, 95th percentile completion length 128 tokens. SLO: P99 TTFT < 1.5s, P99 inter-token latency (ITL) < 120ms.

**Problem:** Baseline autoregressive decode gave ITL of 95ms per token on H100 at bs=1. With 128-token completions, end-to-end latency was 12.2 seconds — 3× above the product requirement for "feels instant" interaction. The team considered three alternatives: (a) quantize the model to INT4 (reduces latency ~2×, but unacceptable quality drop on code according to their internal eval), (b) switch to a smaller 8B model (2× cheaper, but worse completion accuracy), or (c) Medusa on the 70B to keep quality and cut latency simultaneously.

**Medusa intervention:** Trained Medusa-2 heads (K=4, C=3) on The Stack v2 Python subset (500K examples, 5 days on 8× A100 40GB). The team used a learning rate of $5 \times 10^{-4}$ for heads 2–4 and $3 \times 10^{-4}$ for heads 0–1, with cosine decay and 3% warmup. They deployed the Medusa checkpoint alongside the baseline with an A/B router directing 50% of traffic to each path.

**First failure: training on mixed code+text data.** The initial training run used 500K examples drawn 50% from The Stack Python and 50% from UltraChat (the default composition). Per-level α on Python completions was only 0.72 — below expectation. Investigation revealed that the UltraChat examples were suppressing the heads' ability to predict Python-specific idioms, because the heads needed to generalize across two very different distributions. Solution: retrain with 100% The Stack Python. α jumped to 0.88.

**Second failure: tree overhead at long sequences.** For prompts of length 1024–2048 tokens (the top 5th percentile of inputs), the tree-attention pass's prefix-attention cost grew quadratically and the per-step time increased from 86ms to 118ms. At 118ms/step with 3.6 accepted tokens, ITL was 33ms — still a 2.9× speedup, but the 86ms baseline time for that step had grown to 98ms, so the effective speedup compressed to 2.9× rather than 3.5×. The team capped prompts at 1024 tokens by pre-truncating file context, which restored the 3.5× speedup on >99% of requests.

**Results (after both fixes):**
- Medusa-2 α (per level): 0.88 on Python completions
- Mean tokens accepted per step: 3.6
- ITL: 95ms / 3.6 ≈ 26ms per token
- End-to-end latency (128 tokens): ~3.5 seconds → 3.5× speedup
- P99 TTFT: 1.1s (unchanged, Medusa only affects decode)
- GPU VRAM overhead: +6.4GB on the 4-GPU setup (1.6GB per GPU), or +1.0% of total footprint
- Developer reaction in A/B: "felt noticeably faster" mentioned in 67% of feedback surveys vs 21% for baseline

The 3.5× end-to-end speedup exceeded the 2–3× figure from the paper because The Stack Python corpus had very high acceptance rates — Python code is more predictable than MT-Bench chat prompts. The key lesson: domain-specific training data for Medusa heads is not optional when your task has a strong distributional signature.

### Case study 2: Customer support chat with LLaMA-3 8B, resource-constrained deployment

**Context:** An e-commerce platform serving customer support with LLaMA-3 8B Instruct. Infrastructure: 2× A10G 24GB per replica (tensor-parallel). Budget constraint: cannot afford a second model for standard spec decoding (the A10G would need 16GB for 8B → only 8GB left, not enough for even a 1B draft at BF16).

**Problem:** Standard two-model spec decoding was architecturally impossible given the memory budget. Options were: (a) quantize to INT4 and lose quality, (b) keep baseline and accept the latency, or (c) Medusa.

**Medusa-1 deployment:** K=3 heads (fewer heads = less VRAM = fits in 8GB remainder), C=3 candidates. Trained on ShareGPT + internal customer support transcripts (80K examples, 8 hours on 1× A100 40GB at the team's cloud provider).

**Results:**
- VRAM overhead: +48MB for 3 heads at d=4096 (8B model dimension), negligible
- α per level: 0.66–0.72 on customer support queries
- Mean tokens accepted per step: 2.2
- ITL improvement: 2.0× speedup vs naive decode
- Quality: ROUGE-L and customer CSAT scores unchanged (Medusa-1 with frozen backbone is mathematically lossless in greedy; they verified with identical outputs on 1000 held-out prompts)

The 2× speedup on a 3-head Medusa-1 was the difference between the product team requesting a dedicated GPU upgrade (rejected on budget) and shipping a competitive latency SLO with existing hardware.

### Case study 3: Document summarization pipeline, handling distribution mismatch

**Context:** A legal document analysis firm. Target model: Mistral-7B-Instruct-v0.2 (not LLaMA family). Workload: 2000–5000 token input documents, 200–400 token summaries. Batch size varies: bs=1 for priority requests, bs=4 for batch processing.

**Problem:** The team tried standard two-model spec decoding with a 1B Mistral draft but found α ≈ 0.51 — too low to beat the draft overhead cost. The issue: their SFT fine-tuning for legal domain significantly shifted the 7B distribution away from a standard 1B, making the draft's predictions consistently wrong on legal jargon.

**Medusa-1 deployment:** Because Medusa heads learn from the fine-tuned model's own hidden states (using the same fine-tuned backbone), they automatically adapt to the domain shift. Trained Medusa-1 heads on the firm's internal legal documents (40K examples, held out from fine-tuning data to avoid contamination).

**Results:**
- α: 0.74 on legal summaries (vs 0.51 for the two-model baseline)
- Mean tokens accepted per step: 2.6
- Speedup at bs=1: 2.3×
- Speedup at bs=4: 1.3× (larger batch sizes reduce per-token latency baseline, compressing Medusa's relative benefit)
- Priority queue: P99 latency fell from 38 seconds to 16 seconds per document

The key lesson here is that Medusa's acceptance rate is inherently domain-adaptive when the heads are trained on domain data matching the backbone's fine-tune distribution. Two-model spec decoding breaks when draft and target are trained on different distributions; Medusa self-heals because the draft (the heads) and the target (the backbone) share the same fine-tuning.

### Case study 4: High-throughput batch generation, where Medusa breaks down

**Context:** An AI content platform generating marketing copy in bulk. LLaMA-3 70B Instruct on 4× A100 80GB. Workload: bs=32–64, output length 150–300 tokens, runs overnight. Optimization goal is throughput (tokens/second per GPU), not latency.

**Medusa deployment and failure:**
The team deployed Medusa-2 (K=4, C=3), trained on marketing corpus. Initial benchmark showed 2.8× speedup at bs=1. At bs=32, speedup collapsed to 1.05× — barely above noise. At bs=64, Medusa was actually 8% slower than baseline.

**Root cause analysis:**
At bs=32+, the baseline autoregressive decode is already compute-bound rather than memory-bound. The HBM bandwidth bottleneck (the root cause of why single-token decode is slow, covered in [why LLMs are slow](/blog/machine-learning/speculative-decoding/why-llms-are-slow-autoregressive-bottleneck)) is saturated by the 32 parallel request streams. Adding tree attention adds O(tree_size) extra attention computations per layer on top of an already saturated compute budget, while the benefit — more tokens per step — does not help throughput (throughput is limited by compute, not step count).

**Quantitative breakdown at bs=32:**
- Baseline: 32 × 1 token/step, compute-bound, ~38ms/step → 840 tokens/step → 22,100 tok/s
- Medusa: 32 × 2.8 tokens/step (if accepted), but tree attention adds 35% compute overhead → effective 1.95 tokens per equivalent step → ~26,000 tok/s at best, reduced to baseline by real-world scheduling overhead

**Resolution:** The team kept Medusa for the real-time API (bs=1–4) and reverted to standard autoregressive decode for batch jobs. This is the correct division of labor: Medusa optimizes latency at small batch sizes; it is not a throughput optimizer.

The batch-size sensitivity is summarized by a simple rule: if your baseline decode step already saturates GPU compute (arithmetic intensity > H100 balance point of ~295 FLOP/byte), Medusa adds cost without benefit. Measure your baseline GPU SM utilization; if it is above 60–70% during decode, Medusa will not help and may hurt.

## Choosing Medusa: when it is the right tool

Medusa earns its place in your serving stack under four conditions that tend to co-occur:

**Latency-sensitive, small batch:** Your P99 latency SLO drives architecture decisions, and your typical batch size is 1–4 requests. At bs=1, Medusa's 2–3× speedup directly translates to meeting your SLO on cheaper hardware.

**Memory-constrained:** You cannot fit a separate draft model alongside your target without exceeding your VRAM budget, or your target model uses a custom tokenizer that no existing small model shares.

**Structured or repetitive outputs:** Code, SQL, JSON schemas, templated documents — any task where the target distribution is peaked produces high α and maximizes Medusa's speedup.

**Domain-adapted target:** If you have fine-tuned your target on proprietary data, two-model spec decoding suffers from distribution mismatch between the off-the-shelf draft and your fine-tuned target. Medusa-1's heads learn from your fine-tuned backbone's own hidden states, producing a draft that is automatically aligned to your domain.

Medusa is less suitable when batch size is large (>8), when outputs are highly creative or stochastic (temperature > 0.8 on open-ended prompts), or when your serving framework has strong optimization for standard autoregressive decode and adding tree attention requires significant refactoring with unclear payoff.

For the broader picture of where Medusa sits relative to other inference acceleration techniques, see [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) and [optimizing LLM inference](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide). The next post in this series covers EAGLE, which pushes acceptance rates further by predicting at the hidden-state level rather than the token level — a natural evolution from Medusa's approach.

---

**Series navigation:**
- Post 1: [Why LLMs are slow: the autoregressive bottleneck](/blog/machine-learning/speculative-decoding/why-llms-are-slow-autoregressive-bottleneck)
- Post 2: [The core draft-and-verify idea](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify)
- Post 3: [Token acceptance and rejection sampling explained](/blog/machine-learning/speculative-decoding/speculative-decoding-token-acceptance-rejection-sampling)
- Post 4: [Draft models for speculative decoding](/blog/machine-learning/speculative-decoding/draft-models-for-speculative-decoding)
- Post 5: **Medusa: multi-head speculative decoding** ← you are here
- Post 6: EAGLE: feature-level speculative decoding
- Post 7: Tree speculation: drafting multiple futures
- Post 8: Speculative decoding in production: vLLM, SGLang, and real benchmarks

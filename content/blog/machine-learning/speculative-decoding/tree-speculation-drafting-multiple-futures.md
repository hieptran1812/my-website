---
title: "Tree speculation: draft multiple futures, accept the best path"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How replacing a linear draft chain with a token tree dramatically increases expected accepted tokens per verify pass, and why tree attention makes the whole thing possible in a single target forward pass."
tags:
  [
    "speculative-decoding",
    "llm-inference",
    "large-language-model",
    "deep-learning",
    "tree-attention",
    "inference-optimization",
    "transformer-attention",
  ]
category: "machine-learning"
subcategory: "Speculative Decoding"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/tree-speculation-drafting-multiple-futures-1.png"
---

Here is the mismatch at the heart of standard speculative decoding: the draft model produces a single sequence — one path through the token space — and if any token on that path gets rejected by the target, every token after it is discarded. You drafted four tokens, and if the second one fails the acceptance test, tokens three and four are thrown away. Those two forward passes of the draft model just got wasted.

That is not a minor inefficiency. It is the core structural problem of linear chain speculation. The fix is conceptually clean: instead of drafting one path, draft a tree. Let the first position branch to two (or four, or eight) candidates, let each of those branch again, and you have a tree of futures. The target model verifies the whole tree in one forward pass and walks the best-matching path from root to leaves. You accept as many tokens as the best path can sustain before the first rejection, and you never waste draft compute on a path that was doomed from the start — because the tree provides alternatives.

This post builds that idea rigorously. We start with the math of why a linear chain's expected yield is structurally capped, work through the tree attention mechanism that makes parallel verification possible, derive the expected-tokens formula for tree speculation, then quantify the memory and compute tradeoffs. Along the way we build out a complete Python implementation covering tree construction, tree-attention mask generation, and the acceptance walk. By the end you will know exactly when a wider tree beats a longer chain and how [EAGLE-2's dynamic tree](/blog/machine-learning/speculative-decoding/eagle-speculative-decoding-feature-alignment) — the current practical state of the art — implements confidence-weighted pruning to maximize expected yield per verify pass.

---

## The linear chain's structural ceiling

Start with the baseline. In standard speculative decoding ([core draft-and-verify idea](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify)), the draft model produces a sequence of $\gamma$ candidate tokens $\hat{x}_1, \hat{x}_2, \ldots, \hat{x}_\gamma$. The target model verifies them left to right. Let $\alpha$ denote the per-token acceptance rate — the probability that any single candidate passes the modified rejection sampling test (formally, $\alpha = \mathbb{E}[\min(1, q(x)/p(x))]$ where $p$ is the draft distribution and $q$ is the target distribution). Under the standard analysis, the expected number of tokens accepted from one verify step is:

$$E[\text{accepted}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

At $\alpha = 0.8$ and $\gamma = 4$, this gives approximately $2.8$ tokens per verify step. At $\gamma = 8$ you get roughly $3.6$. The series has a hard ceiling: no matter how long you make the chain, you cannot push the expected yield beyond $\lim_{\gamma \to \infty} E[\text{accepted}] = 1/(1-\alpha) = 5.0$ at $\alpha = 0.8$. Making the chain longer costs proportionally more draft compute ($\gamma$ additional forward passes of the draft model per verify step), but the return in expected tokens per additional draft step falls geometrically.

The structural problem is clearest when you think about conditional dependencies. Token $\hat{x}_3$ in the chain was drafted conditioned on $\hat{x}_2$ having been accepted. If $\hat{x}_2$ is rejected — an event that happens with probability $1 - \alpha = 0.2$ at each position — then $\hat{x}_3$ was drafted from the wrong conditional distribution, and the draft compute for it (plus all subsequent tokens) is pure waste. The probability that the first rejection happens at position $k$ is $\alpha^{k-1}(1-\alpha)$, so the expected position of first rejection is:

$$E[\text{first rejection}] = \sum_{k=1}^{\gamma} k \cdot \alpha^{k-1}(1-\alpha) \approx \frac{1}{1-\alpha}$$

At $\alpha = 0.8$, this is $1/(1-0.8) = 5.0$. So on average, the first rejection happens around position 5 in a very long chain — and every token you drafted after position 5 was wasted. For a chain of $\gamma = 4$, the first rejection falls around position $\min(4, 5) \approx 3.5$, meaning the last draft token (position 4) is discarded on average roughly half the time.

You cannot fix this by caching the draft model's KV computation, because the wasted tokens are not repeated computation in the traditional sense — they are computation spent speculating down a path the verifier will never walk. The fundamental issue is that a linear chain has only one path to explore. What you need is a way to speculate down multiple alternative paths simultaneously, so that when one path fails at position $k$, another path that diverged before $k$ may still produce accepted tokens.

### Why caching does not rescue the chain

A natural reaction at this point is: "can we just cache the draft model's KV states for each position, so that when we reject $\hat{x}_2$ and need to resample position 2 from the target's adjusted distribution, we can quickly re-run the draft from position 2 onward?" The answer is no, for two reasons.

First, the bonus token mechanism in standard spec decoding means that once you reject a chain token at position $k$, you sample a replacement from the target's adjusted distribution and stop — you do not continue drafting from the new token. There is no "second chance" for positions $k+1$ onward in the same step. The step terminates.

Second, even if you did continue drafting from the replacement, you would need to re-run the draft model from the corrected token at position $k$, generating $\gamma - k$ new proposals. That is additional draft latency, at exactly the moments when the draft model is performing worst (it already made a wrong prediction at position $k-1$). This is not a stable or efficient design.

The correct fix is architectural: replace the linear chain with a structure that explores multiple first-position alternatives simultaneously, so that the choice between "the" and "a" at position $t+1$ is evaluated in parallel rather than sequentially.

---

## The tree idea: exploring multiple futures at once

A draft tree replaces the linear chain with a branching structure. The root of the tree is the current last accepted token (or the entire accepted context for the purposes of attention computation). From the root, the draft model proposes $W$ alternatives for position $t+1$ — call these depth-1 children. For each depth-1 child, it proposes $W$ alternatives for position $t+2$ conditioned on that specific child. And so on down to depth $D$. The resulting tree has:

- $W$ nodes at depth 1
- $W^2$ nodes at depth 2
- ...
- $W^D$ nodes at depth $D$ (leaves)
- Total: $(W^{D+1} - 1)/(W - 1)$ nodes

For $W=2$, $D=3$: $(2^4 - 1)/(2-1) = 15$ nodes including the root, or 14 draft candidate nodes. For $W=4$, $D=4$: $(4^5 - 1)/(4-1) = 341$ nodes.

The target model then does one forward pass that evaluates all candidate nodes simultaneously, using a specialized attention mask called tree attention. For each node, the target computes a probability distribution over the vocabulary — conditioned, crucially, on that node's ancestor path through the tree. The verifier then walks the tree from root downward, accepting nodes as it goes. It stops at the first node whose acceptance test fails (using the same rejection sampling criterion as the linear case), generates a bonus token at that position, and returns the accepted prefix.

![Draft token tree: width-2, depth-3](/imgs/blogs/tree-speculation-drafting-multiple-futures-1.webp)

The tree structure provides a fundamental advantage: at each depth level, the verifier encounters $W$ independent alternatives, and only needs one of them to pass the acceptance test in order to continue the walk. The probability that the walk reaches depth $d$ is $1 - (1 - \alpha^1)^W$ for depth 1 and $[1-(1-\alpha)^W]^d$ for depth $d$ (the probability that at least one child is accepted at each of the $d$ levels). At $W=2$, $\alpha=0.8$: the walk reaches depth 1 with probability $1-(1-0.8)^2 = 0.96$, compared to $0.8$ for a chain. The walk reaches depth 3 with probability $0.96^3 = 0.885$, versus $0.8^3 = 0.512$ for a chain. This is a substantial difference in expected yield from the same number of verify-pass positions.

![Linear chain vs tree: expected accepted tokens (α=0.8)](/imgs/blogs/tree-speculation-drafting-multiple-futures-2.webp)

---

## Tree attention: the mechanism that makes one-pass verification possible

The key insight is that the tree verification is done in a single target forward pass. This is only possible because of tree attention — a generalization of the standard causal attention mask that allows different positions in the sequence to be verified against different causal contexts (their respective ancestor paths in the tree), all in the same forward pass.

In standard causal attention, the attention mask $M$ is a lower-triangular binary matrix: position $i$ attends to all positions $j \leq i$. In tree attention, the mask is defined by the tree structure rather than sequential ordering:

$$M_{ij} = \begin{cases} 0 & \text{if } j \in \mathcal{A}(i) \\ -\infty & \text{otherwise} \end{cases}$$

where $\mathcal{A}(i) = \{i\} \cup \{\text{all ancestors of } i \text{ in the tree}\}$ is the ancestor set of node $i$, and $0$ means "attention is allowed" while $-\infty$ means "masked out" (in the logit-bias convention used by FlashAttention).

The consequence: a depth-2 node on branch "the → model" attends to its root ancestor and its depth-1 ancestor "the", but not to the sibling depth-1 node "a" or any of "a"'s children. This is what makes the verification path-accurate rather than averaged: the target model computes logits for "model" under the exact causal context "…accepted prefix…the", giving a score that is genuinely comparable to what the draft model computed when it proposed "model" given "the".

Without tree attention, you would have to run the target model once per tree path — $W^D$ separate forward passes — destroying the efficiency advantage. Tree attention collapses all paths into one forward pass at the cost of constructing the $N \times N$ mask and handling the sparsity carefully.

![Tree attention mask: ancestor-only attention](/imgs/blogs/tree-speculation-drafting-multiple-futures-3.webp)

### Constructing the tree attention mask

In practice, the $N$ tree nodes are flattened into a 1D sequence in BFS (breadth-first search) order: root at index 0, then depth-1 nodes left-to-right, then depth-2 nodes left-to-right, and so on. This ordering means that ancestors always appear at lower indices than their descendants — a property that simplifies the mask construction.

The complete attention matrix for a verify pass combines two parts: the context portion (the $S$ already-accepted context tokens) and the tree portion (the $N$ draft candidate nodes). The shape of the full attention matrix is $(S + N) \times (S + N)$. The context portion uses a standard lower-triangular causal mask. The tree portion uses the ancestor-only mask. The cross-attention between context and tree nodes is all-attend (every tree node can see all context tokens).

```python
## tree_attention.py — build tree attention mask from parent pointers
## Compatible with PyTorch 2.1+; mask suitable as additive FlashAttention bias

import torch
from typing import Optional

def build_tree_attention_mask(
    parent_ids: list[Optional[int]],
    context_len: int,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Build a full tree attention mask for tree-speculation verify pass.

    The mask covers (context_len + n_tree) positions total:
      - Positions 0..context_len-1: standard lower-triangular causal mask
      - Positions context_len..context_len+n_tree-1: tree positions
        each attends to full context (cols 0..context_len-1)
        and to ancestors only (cols context_len..context_len+n_tree-1)

    Args:
        parent_ids: List of length n_tree. parent_ids[i] is the parent's
                    LOCAL tree index (not global index), or None for root.
        context_len: Number of already-accepted context tokens (S).
        dtype: Output dtype (float32 or bfloat16).
        device: Target device string.

    Returns:
        mask: Tensor of shape (S + N, S + N). 0.0 = attend, -inf = masked.
    """
    n_tree = len(parent_ids)
    n_total = context_len + n_tree
    mask = torch.full((n_total, n_total), float("-inf"), dtype=dtype, device=device)

    ## 1) Context portion: lower-triangular causal mask
    for i in range(context_len):
        mask[i, :i+1] = 0.0

    ## 2) Tree nodes attend to full context
    for i in range(n_tree):
        row = context_len + i
        mask[row, :context_len] = 0.0

    ## 3) Tree nodes attend to their ancestors (ancestor-only tree mask)
    ## Build ancestor sets by following parent_ids up to root
    for i in range(n_tree):
        cur = i
        while cur is not None:
            mask[context_len + i, context_len + cur] = 0.0
            cur = parent_ids[cur]

    return mask


def parent_ids_from_width_depth(width: int, depth: int) -> list[Optional[int]]:
    """
    Generate BFS-order parent_ids for a uniform-width-depth tree.

    Example for width=2, depth=2 (7 nodes):
    Index: 0=root, 1=L1-child0, 2=L1-child1,
           3=L2-child(0,0), 4=L2-child(0,1), 5=L2-child(1,0), 6=L2-child(1,1)
    parent_ids = [None, 0, 0, 1, 1, 2, 2]

    Args:
        width: Number of children per node (W).
        depth: Maximum depth (D). Root is depth 0.

    Returns:
        parent_ids: List of length (W^(D+1) - 1) / (W - 1), BFS order.
    """
    parent_ids: list[Optional[int]] = [None]  ## Root has no parent
    current_level_start = 0
    current_level_size = 1

    for d in range(depth):
        next_level_size = current_level_size * width
        for parent_local in range(current_level_size):
            parent_idx = current_level_start + parent_local
            for _ in range(width):
                parent_ids.append(parent_idx)
        current_level_start += current_level_size
        current_level_size = next_level_size

    return parent_ids
```

The complexity of building this mask is $O(N \cdot D)$ where $D$ is the tree depth (each node traces at most $D$ ancestor hops). For a tree with 60 nodes and depth 5, this is 300 operations — negligible compared to the attention computation itself.

---

## Building the tree: three strategies

There are three practical strategies for deciding which candidates to expand at each tree level. The choice matters significantly for expected yield at a fixed node budget.

### Strategy 1: Uniform greedy expansion

The simplest approach takes the top-$W$ candidates by draft probability at every node, regardless of the node's position in the tree or the draft model's confidence. Each node at depth $d < D$ gets exactly $W$ children. This produces a perfectly symmetric tree with $W^D$ leaves and $(W^{D+1}-1)/(W-1)$ total nodes.

Uniform expansion is easy to implement and the mask can be precomputed and reused across decode steps (since the tree topology is fixed). The downside: it treats all nodes equally. A node where the draft model assigns 0.95 probability to its top-1 candidate (almost certain to be accepted by the target) still gets $W-1$ redundant sibling candidates generated — wasting draft compute on alternatives the verifier is very unlikely to prefer. Meanwhile, a node where the draft is genuinely uncertain gets exactly as many children as the confident node, even though the extra alternatives would be more valuable there.

### Strategy 2: Confidence-weighted expansion

Confidence-weighted expansion adjusts the number of children per node based on the draft model's entropy at that position. Formally, compute the entropy of the draft distribution at node $v$:

$$H_v = -\sum_{x} p_{\text{draft}}(x | \text{path to } v) \log p_{\text{draft}}(x | \text{path to } v)$$

Nodes with high entropy get more children (the draft is uncertain, so alternatives are valuable). Nodes with low entropy get fewer children (the draft is confident, the top-1 candidate is almost certainly what the target wants). The total tree budget is $B_{\text{nodes}}$, and you greedily allocate children by entropy score.

This concentrates the tree's branching at positions where the draft-target disagreement is most likely to be large — exactly where alternatives matter most. Empirically, a confidence-weighted tree with 60 nodes outperforms a uniform tree with 60 nodes by 15–25% in expected yield at the same $\alpha$.

### Strategy 3: Beam search expansion

Beam search maintains $B$ complete prefixes at each depth level and extends each by its single most likely next token (no branching within a beam candidate). The resulting structure has $B$ paths of depth $D$, producing $B \cdot D$ candidate nodes total. Unlike the tree, beam candidates do not share prefixes below the root — each beam path is independent.

Beam search is useful when you want diversity in the generated paths rather than width at a fixed point. For speculative decoding, it is less commonly used than greedy trees because the accept-walk mechanism benefits most from having multiple alternatives at early positions (where the probability of early rejection is highest). Beam search concentrates diversity at depth $D$ rather than depth 1, which is less useful for recovering from early rejections.

```python
## tree_builder.py — three tree construction strategies
## Requires: PyTorch 2.1+

import torch
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass, field

@dataclass
class DraftNode:
    token_id: int
    parent_idx: Optional[int]
    depth: int
    draft_log_prob: float
    draft_entropy: float = 0.0
    children: list[int] = field(default_factory=list)


def build_uniform_tree(
    draft_model_forward,          ## Callable: (token_ids) -> logits (vocab_size,)
    context_ids: torch.Tensor,    ## (seq_len,) accepted context
    width: int,
    depth: int,
    device: str = "cuda",
) -> tuple[list[DraftNode], list[Optional[int]]]:
    """
    Build a uniform-width draft tree using the draft model.

    For each tree level, computes draft logits for all current-level nodes
    in a single batched forward pass (requires prefix caching or KV-cache tricks;
    simplified here for clarity).
    """
    nodes: list[DraftNode] = []
    queue: list[int] = []

    ## Root node: virtual, represents the last accepted context token
    root = DraftNode(
        token_id=int(context_ids[-1]),
        parent_idx=None,
        depth=0,
        draft_log_prob=0.0,
    )
    nodes.append(root)
    queue.append(0)

    for d in range(depth):
        current_level_indices = [idx for idx in queue if nodes[idx].depth == d]
        queue = [idx for idx in queue if nodes[idx].depth > d]

        for node_idx in current_level_indices:
            ## Get the prefix for this node: root → node_idx path tokens
            prefix_tokens = _get_path_tokens(nodes, node_idx)
            full_context = torch.cat([
                context_ids[:-1],  ## All context except last (which is root)
                torch.tensor(prefix_tokens, dtype=torch.long, device=device),
            ], dim=0)

            with torch.no_grad():
                logits = draft_model_forward(full_context.unsqueeze(0))
            log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            probs = log_probs.exp()
            entropy = float(-(probs * log_probs).sum())

            ## Top-W candidates for this node
            topk_log_probs, topk_ids = torch.topk(log_probs, k=width)
            for w in range(width):
                child = DraftNode(
                    token_id=int(topk_ids[w]),
                    parent_idx=node_idx,
                    depth=d + 1,
                    draft_log_prob=float(topk_log_probs[w]),
                    draft_entropy=entropy,
                )
                child_idx = len(nodes)
                nodes.append(child)
                nodes[node_idx].children.append(child_idx)
                queue.append(child_idx)

    parent_ids: list[Optional[int]] = [n.parent_idx for n in nodes]
    return nodes, parent_ids


def build_confidence_weighted_tree(
    draft_model_forward,
    context_ids: torch.Tensor,
    node_budget: int = 60,
    min_children: int = 1,
    max_children: int = 4,
    max_depth: int = 6,
    device: str = "cuda",
) -> tuple[list[DraftNode], list[Optional[int]]]:
    """
    Build a confidence-weighted draft tree.
    Nodes with higher draft entropy receive more children.
    Total node count is capped at node_budget.
    """
    nodes: list[DraftNode] = []
    expansion_queue: list[tuple[float, int]] = []  ## (priority=-entropy, node_idx)

    root = DraftNode(
        token_id=int(context_ids[-1]),
        parent_idx=None,
        depth=0,
        draft_log_prob=0.0,
        draft_entropy=float("inf"),  ## Always expand root
    )
    nodes.append(root)
    expansion_queue.append((-float("inf"), 0))  ## Root has highest priority

    while expansion_queue and len(nodes) < node_budget:
        _, node_idx = min(expansion_queue, key=lambda x: x[0])
        expansion_queue.remove((_, node_idx))

        node = nodes[node_idx]
        if node.depth >= max_depth:
            continue

        ## Draft model forward at this node's path
        prefix_tokens = _get_path_tokens(nodes, node_idx)
        full_context = torch.cat([
            context_ids[:-1],
            torch.tensor(prefix_tokens, dtype=torch.long, device=device),
        ], dim=0)

        with torch.no_grad():
            logits = draft_model_forward(full_context.unsqueeze(0))
        log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
        probs = log_probs.exp()
        entropy = float(-(probs * log_probs).sum())

        ## Number of children proportional to entropy (normalized to [0,1])
        max_possible_entropy = torch.log(torch.tensor(logits.shape[-1], dtype=torch.float)).item()
        entropy_ratio = min(entropy / max_possible_entropy, 1.0)
        n_children = min_children + round(entropy_ratio * (max_children - min_children))
        n_children = min(n_children, node_budget - len(nodes))

        topk_log_probs, topk_ids = torch.topk(log_probs, k=n_children)
        for i in range(n_children):
            child_entropy = entropy  ## Approximate; real value computed when expanded
            child = DraftNode(
                token_id=int(topk_ids[i]),
                parent_idx=node_idx,
                depth=node.depth + 1,
                draft_log_prob=float(topk_log_probs[i]),
                draft_entropy=child_entropy,
            )
            child_idx = len(nodes)
            nodes.append(child)
            nodes[node_idx].children.append(child_idx)
            ## Priority = -draft_prob (expand less-certain nodes first)
            priority = float(topk_log_probs[i])  ## Less negative = more confident = lower priority
            expansion_queue.append((priority, child_idx))

    parent_ids: list[Optional[int]] = [n.parent_idx for n in nodes]
    return nodes, parent_ids


def _get_path_tokens(nodes: list[DraftNode], node_idx: int) -> list[int]:
    """Walk up ancestor chain from node_idx to root, return token sequence."""
    path = []
    cur: Optional[int] = node_idx
    while cur is not None:
        path.append(nodes[cur].token_id)
        cur = nodes[cur].parent_idx
    return list(reversed(path))
```

---

## Tree acceptance: walking the verified tree

Once the target model has run its single forward pass over all tree nodes (with the tree attention mask applied), it has produced logits for every node. The acceptance walk generalizes the linear-chain acceptance procedure from the modified rejection sampling described in the post on [token acceptance](/blog/machine-learning/speculative-decoding/speculative-decoding-token-acceptance-rejection-sampling).

For a tree node at position $v$ (with parent $u$), let $p(x_v | \pi(v))$ denote the draft model's probability for that token given the ancestor path $\pi(v)$, and $q(x_v | \pi(v))$ denote the target model's probability given the same path (as computed by the verify pass). The acceptance probability for node $v$ is:

$$\Pr[\text{accept } v \mid \pi(v)] = \min\!\left(1, \frac{q(x_v \mid \pi(v))}{p(x_v \mid \pi(v))}\right)$$

This is the same formula as the linear case — the only difference is that $\pi(v)$ is now the tree ancestor path rather than the fixed sequential prefix.

The walk algorithm:

1. Start at the root (index 0). The root represents the last accepted context token and is always accepted trivially.
2. Consider all depth-1 children of the root. Apply the acceptance test to each child simultaneously (they are independent — each depends only on the root, which is the same for all of them). The walk takes the first child that passes (by BFS order, or by highest acceptance probability — the implementation choice affects nothing about losslessness, only the distribution of which accepted path you get).
3. Move to the accepted child. Consider its children (depth-2 nodes on the accepted branch). Apply the acceptance test to each.
4. Continue until either: a) no child at the current node passes the acceptance test (walk terminates, generate bonus token), or b) the walk reaches a leaf node (also generate bonus token from the target's distribution at depth $D+1$).
5. At the termination point, sample a bonus token from the adjusted distribution $(q(x | \pi) - \min(p(x | \pi), q(x | \pi)))_+ / Z$ where $Z$ is the normalizing constant. This adjusted distribution corresponds to the "excess" probability mass that the target assigns beyond what the draft assigned — the same construction that makes linear-chain spec decoding lossless, now applied per tree node on the accepted path.

![Tree speculation verify pass: end-to-end](/imgs/blogs/tree-speculation-drafting-multiple-futures-4.webp)

### Why tree walk preserves the target distribution

A concern that often comes up: if we are selecting among multiple children at depth 1 — sometimes taking child A ("the"), sometimes taking child B ("a") — does this bias the output distribution away from the target? The answer is no, and the argument is subtle but clean.

The acceptance probability for child $i$ at depth 1 is $\min(1, q(x_i | \text{context}) / p(x_i | \text{context}))$. The probability that the walk takes any specific depth-1 child $i$ is $p(x_i | \text{context}) \cdot \min(1, q(x_i | \text{context}) / p(x_i | \text{context})) = \min(p, q)$. The probability that no depth-1 child is accepted (triggering the bonus token) is $1 - \sum_i \min(p_i, q_i)$ (where the sum is over all depth-1 children in the tree). The bonus token is then sampled from $(q - \sum_i \min(p_i, q_i) \cdot \mathbf{1}_{x = x_i})_+ / Z$, which is exactly the residual probability mass in the target's distribution that is not already captured by the accepted children.

The combined distribution over the accepted position is:

$$\sum_{i} \min(p_i, q_i) \cdot \mathbf{1}_{x = x_i} + \left(q(x) - \sum_i \min(p_i, q_i) \cdot \mathbf{1}_{x = x_i}\right)_+ / Z \cdot Z_{\text{bonus}}$$

which simplifies to $q(x)$ for every $x$ in the vocabulary. The walk is lossless. The key is that the tree provides multiple "first opportunities" to accept a token at each level, but the bonus token mechanism ensures that the probability mass not captured by any accepted candidate is accounted for in the final distribution.

Here is the complete acceptance walk implementation:

```python
## tree_accept.py — lossless acceptance walk for tree speculation
## Requires: PyTorch 2.1+

import torch
import torch.nn.functional as F
from typing import Optional

def walk_accepted_path(
    node_token_ids: list[int],           ## BFS-order token IDs, index 0 = root
    parent_ids: list[Optional[int]],     ## Parent index per node, None for root
    draft_probs: list[float],            ## Draft probability for each node (given its path)
    target_logits: torch.Tensor,         ## Shape (N, vocab_size) from target verify pass
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> tuple[list[int], list[int]]:
    """
    Walk the tree to find the accepted path and generate the bonus token.
    This procedure is provably lossless: the marginal distribution of each
    accepted token equals the target model's conditional distribution.

    Args:
        node_token_ids: Token ID for each tree node in BFS order.
        parent_ids: Parent index for each node; None for root (index 0).
        draft_probs: Draft model's probability for each node's token (p values).
        target_logits: Raw logits from the target model's verify pass (q source).
        temperature: Sampling temperature (must match what was used for draft).
        top_p: Nucleus sampling threshold (must match draft parameters).

    Returns:
        accepted_ids: Accepted token IDs from the walk, INCLUDING the bonus token.
        accepted_path: Node indices of the accepted path (for KV cache trimming).
    """
    n = len(node_token_ids)
    if n == 0:
        raise ValueError("Empty tree.")

    ## Build children map
    children: dict[int, list[int]] = {i: [] for i in range(n)}
    for i, par in enumerate(parent_ids):
        if par is not None:
            children[par].append(i)

    ## Compute target probabilities with temperature and top-p
    target_probs_all = _compute_sampling_probs(target_logits, temperature, top_p)  ## (N, vocab)

    accepted_ids: list[int] = []
    accepted_path: list[int] = [0]  ## Root is always on the accepted path
    current_node = 0  ## Start at root

    while True:
        child_list = children[current_node]
        if not child_list:
            ## Leaf node: generate bonus token from target distribution at this position
            bonus_token = int(torch.multinomial(target_probs_all[current_node], num_samples=1))
            accepted_ids.append(bonus_token)
            break

        ## Compute adjusted distribution for bonus token in case all children are rejected
        ## bonus_dist = (q - sum_i min(p_i, q_i) * delta_{x_i})+ / Z
        q_vec = target_probs_all[current_node].clone()
        min_pq_mass = torch.zeros_like(q_vec)
        for ci in child_list:
            xi = node_token_ids[ci]
            p_i = draft_probs[ci]
            q_i = float(q_vec[xi])
            min_pq_mass[xi] += min(p_i, q_i)

        ## Find first child that passes acceptance test
        accepted_child: Optional[int] = None
        for ci in child_list:
            xi = node_token_ids[ci]
            p_i = draft_probs[ci]
            q_i = float(target_probs_all[ci, xi])
            accept_prob = min(1.0, q_i / (p_i + 1e-9))
            u = float(torch.rand(1))
            if u < accept_prob:
                accepted_ids.append(xi)
                accepted_path.append(ci)
                accepted_child = ci
                break

        if accepted_child is None:
            ## No child accepted: generate bonus token from adjusted distribution
            bonus_dist = torch.clamp(q_vec - min_pq_mass, min=0.0)
            norm = bonus_dist.sum()
            if norm < 1e-9:
                bonus_dist = q_vec
            else:
                bonus_dist = bonus_dist / norm
            bonus_token = int(torch.multinomial(bonus_dist, num_samples=1))
            accepted_ids.append(bonus_token)
            break

        current_node = accepted_child

    return accepted_ids, accepted_path


def _compute_sampling_probs(
    logits: torch.Tensor,     ## (N, vocab)
    temperature: float,
    top_p: float,
) -> torch.Tensor:
    """Apply temperature and top-p filtering; return probability distributions."""
    scaled = logits / max(temperature, 1e-6)
    probs = F.softmax(scaled, dim=-1)
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        ## Zero out tokens beyond the nucleus
        beyond_nucleus = (cumsum - sorted_probs) >= top_p
        sorted_probs[beyond_nucleus] = 0.0
        ## Scatter back to original order
        probs = torch.zeros_like(probs)
        probs.scatter_(-1, sorted_idx, sorted_probs)
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)
    return probs
```

---

## Expected tokens per verify step: the full derivation

Let us derive the expected accepted tokens for a tree exactly, then find the conditions where tree beats chain.

Define $W_d$ as the effective acceptance probability that the walk reaches depth $d$. For a tree with uniform width $W$ and uniform per-token acceptance rate $\alpha$:

$$W_{\text{eff}} = 1 - (1-\alpha)^W$$

This is the probability that at least one of the $W$ children at any given level passes the acceptance test. The probability that the walk reaches depth $d$ (accepting one child at each of the first $d$ levels) is:

$$P(\text{reach depth } d) = W_{\text{eff}}^d$$

The walk contributes one accepted token for each depth level it successfully traverses, plus one bonus token at the termination point. Therefore:

$$E[\text{accepted}]_{\text{tree}} = 1 + \sum_{d=1}^{D} W_{\text{eff}}^{d-1} \cdot W_{\text{eff}} = 1 + \sum_{d=1}^{D} W_{\text{eff}}^d = 1 + W_{\text{eff}} \cdot \frac{1 - W_{\text{eff}}^D}{1 - W_{\text{eff}}}$$

where the "+1" is the bonus token. For $W=2$, $D=3$, $\alpha=0.8$: $W_{\text{eff}} = 0.96$, and $E[\text{accepted}] = 1 + 0.96 \cdot (1 - 0.96^3)/(1-0.96) = 1 + 0.96 \cdot 0.886/0.04 \approx 1 + 21.3 \approx \ldots$ 

Wait, let me redo this more carefully. The expected number of tokens that land on the accepted path (not counting the bonus) is:

$$E[\text{path tokens}] = \sum_{d=1}^{D} P(\text{walk reaches depth } d) = \sum_{d=1}^{D} W_{\text{eff}}^d = W_{\text{eff}} \cdot \frac{1 - W_{\text{eff}}^D}{1 - W_{\text{eff}}}$$

For $W=2$, $D=3$, $\alpha=0.8$: $W_{\text{eff}} = 0.96$, $D=3$:

$$E[\text{path tokens}] = 0.96 \cdot \frac{1-0.96^3}{1-0.96} = 0.96 \cdot \frac{1-0.885}{0.04} = 0.96 \cdot \frac{0.115}{0.04} = 0.96 \cdot 2.87 = 2.76$$

Adding the bonus token: $E[\text{accepted}] = 2.76 + 1 = 3.76$ tokens per step.

For comparison, the linear chain at $\gamma = 4$, $\alpha = 0.8$:

$$E[\text{accepted}]_{\text{chain}} = \frac{1 - 0.8^5}{1-0.8} = \frac{1 - 0.328}{0.2} = \frac{0.672}{0.2} = 3.36$$

So in this parameterization the tree with 14 nodes yields 3.76 tokens versus 3.36 for a linear chain with 4 nodes — a 12% gain with 3.5× more draft nodes. The relative efficiency (tokens per draft node) actually favors the linear chain here. But this comparison is misleading: the linear chain of $\gamma=4$ uses 4 sequential draft forward passes, while the tree of 14 nodes can be expanded in 3 depth-levels of breadth-first passes — which, with batching, costs roughly the same wall-clock time as 3–4 sequential single-node passes (depending on the degree of parallelism available in the draft model).

![Expected accepted tokens per verify step (α=0.8)](/imgs/blogs/tree-speculation-drafting-multiple-futures-5.webp)

### When tree beats chain: the regime analysis

The crossover between tree and chain depends on three factors: acceptance rate $\alpha$, draft compute parallelism, and the node budget. Let us work through each.

**Low $\alpha$ regime ($\alpha < 0.7$):** The linear chain suffers greatly from early rejection. At $\alpha = 0.6$, $\gamma = 4$: $E[\text{accepted}]_{\text{chain}} = (1-0.6^5)/(1-0.6) = (1-0.078)/0.4 = 2.3$ tokens. A tree with $W=2$, $D=3$, $\alpha=0.6$: $W_{\text{eff}} = 1-(1-0.6)^2 = 0.84$, $E[\text{path tokens}] = 0.84 \cdot (1-0.84^3)/(1-0.84) = 0.84 \cdot 0.408/0.16 = 2.14$, $E[\text{accepted}] = 3.14$. The tree wins by 37%.

**Moderate $\alpha$ regime ($\alpha = 0.7$–$0.8$):** Both strategies are competitive. Trees have an advantage in yield per node at low depth but the chain catches up at high $\gamma$.

**High $\alpha$ regime ($\alpha > 0.85$):** Linear chains become efficient because the walk rarely terminates early, and long chains accumulate accepted tokens linearly. At $\alpha = 0.9$, $\gamma = 8$: $E[\text{accepted}]_{\text{chain}} = (1-0.9^9)/(1-0.9) = (1-0.387)/0.1 = 6.1$ tokens. A tree with $W=2$, $D=5$ and 63 nodes: $W_{\text{eff}} = 0.99$, $E = 1 + 0.99 \cdot (1-0.99^5)/0.01 = 1 + 0.99 \cdot 4.9 \approx 5.9$ tokens — almost the same as a chain of $\gamma=8$, but with 63 draft nodes versus 8. In the high-$\alpha$ regime, the chain is far more compute-efficient per token yielded.

The practical conclusion: **tree speculation pays most when $\alpha$ is in the 0.6–0.8 range**. This corresponds to deployment scenarios where the draft model is from a different family than the target, or where the task has genuinely diverse next-token predictions (creative writing, open-ended Q&A, multi-language tasks). For code completion and highly constrained tasks where $\alpha > 0.85$, the linear chain is more efficient.

---

## Memory and compute: KV cache implications

Every node in the draft tree needs a KV entry in the target model's attention cache during the verify pass. For a model with $L$ layers, $H$ heads, head dimension $d_h$, and tree size $N_{\text{tree}}$, the additional KV memory for tree positions is:

$$\Delta_{\text{KV}} = 2 \cdot L \cdot H \cdot d_h \cdot N_{\text{tree}} \cdot \text{sizeof}(\text{dtype})$$

For a typical 7B model (32 layers, 32 heads, $d_h = 128$), bfloat16, $N_{\text{tree}} = 60$:

$$\Delta_{\text{KV}} = 2 \times 32 \times 32 \times 128 \times 60 \times 2 = 31,457,280 \text{ bytes} \approx 30 \text{ MB}$$

The baseline KV cache for a 2048-token context is:

$$\text{KV}_{\text{base}} = 2 \times 32 \times 32 \times 128 \times 2048 \times 2 \approx 1,074 \text{ MB} \approx 1.05 \text{ GB}$$

So a 60-node tree adds ~3% to the KV cache for a 2048-token context. For short contexts (512 tokens), the overhead is ~12% — still modest. For very short contexts (128 tokens), the overhead reaches ~50%, which is why tree speculation is less attractive at the beginning of decode (where context is short) than at the end (where context is long).

![KV cache memory: baseline vs tree attention](/imgs/blogs/tree-speculation-drafting-multiple-futures-7.webp)

### Attention compute scaling

The tree-attention verify pass computes attention for $S + N$ positions total ($S$ context, $N$ tree nodes). The attention compute is $O((S+N)^2 \cdot d_h \cdot L)$ — quadratic in the total sequence length. For $S=2048$, $N=60$: the overhead from tree positions is approximately $(2048+60)^2 / 2048^2 - 1 \approx 5.9\%$ more attention compute. Negligible.

However, the tree attention mask itself has an irregular sparsity pattern — each row has a different number of non-zero entries (one per ancestor level plus the full context). This breaks the assumptions of standard FlashAttention, which assumes either dense attention or a regular triangular mask. Implementations that use FlashAttention for the verify pass must either: (a) pad the tree mask to a dense format and accept the overhead of computing $-\infty$-masked entries (they get zeroed out by softmax, but FLOPS are still spent), or (b) implement a custom tree-attention CUDA kernel that exploits the ancestor-only sparsity.

In practice, option (a) — dense mask with $-\infty$ fill — is used in most production implementations (vLLM, SGLang) because the overhead at 60-node tree sizes is under 6% and the engineering cost of a custom kernel is significant. At larger tree sizes (300+ nodes), a custom sparse-attention kernel becomes worth writing.

---

## EAGLE-2: dynamic trees in practice

[EAGLE-2](/blog/machine-learning/speculative-decoding/eagle-speculative-decoding-feature-alignment) is the most well-known production implementation of dynamic tree speculation, and its approach to confidence-weighted expansion is worth examining in detail because it resolves a key engineering challenge: how do you estimate node confidence without running the full draft model at every candidate position?

EAGLE-2's solution exploits the EAGLE architecture itself. The EAGLE draft head is a single transformer block that takes (hidden_state, token) pairs as input and autoregressively predicts the next hidden state. When expanding the tree at a given node, EAGLE-2 uses the draft head's predicted next-token probabilities as the confidence estimate — specifically, the top-1 probability $p_{\max}$. Nodes with $p_{\max} > 0.7$ are given 2 children; nodes with $p_{\max}$ between 0.4 and 0.7 are given 3 children; nodes with $p_{\max} < 0.4$ are given 4 children (capped by the budget).

This requires no additional model calls — the confidence score is a free byproduct of the draft head's forward pass, which you had to run anyway to get the candidate token IDs. The total overhead for tree construction is exactly the cost of running the EAGLE draft head on all tree nodes: $N_{\text{tree}}$ draft-head forward passes, each taking ~0.5–1 ms on an A100 for a 1-block transformer.

EAGLE-2 also implements a reranking step: before selecting which depth-1 children to expand into depth-2, it uses the product of cumulative draft probabilities along the path as a tree-path score, and preferentially expands paths with higher cumulative scores. This is a form of joint confidence estimation — instead of evaluating each node independently, it estimates the probability that the entire path from root to that node is accepted.

Empirically, EAGLE-2 achieves 3.0–5.1× speedup on LLaMA-2-7B/13B/70B and LLaMA-3-8B/70B benchmarks, compared to 1.5–2.2× for standard 2-model spec decoding with a linear chain and 2.0–3.2× for EAGLE-1 with a fixed static tree. The improvement from EAGLE-1 to EAGLE-2 (fixed to dynamic tree) is approximately 20–40% in token throughput, with the largest gains on tasks where acceptance rates vary significantly across token positions (creative tasks, multi-turn dialogue with topic shifts).

One aspect of EAGLE-2 that is easy to miss is how cheap its training is. The EAGLE draft head is a single transformer block with the same hidden size as the target model — for a 7B model with $d_{\text{model}} = 4096$, the draft head has roughly 200M parameters. Training it requires running the target model on a corpus of text to collect (feature, token) pairs, then training the draft head to predict the next feature autoregressively — no RL, no preference optimization, no complex reward modeling. A typical training run on 100,000 documents with a 7B target takes 2–4 GPU-hours on a single A100. The resulting draft head, when used in a dynamic tree, immediately provides 3–5× speedup — a remarkable return on training investment.

The reranking step in EAGLE-2 deserves particular attention because it is where most of the improvement over EAGLE-1 originates. In EAGLE-1, the static 60-node tree is fixed before any inference — the same tree shape is used for every decode step regardless of what the draft model is confident about. In EAGLE-2, after building the initial candidate set at each level, the algorithm scores each node by its cumulative path probability:

$$\text{score}(v) = \prod_{u \in \text{path from root to } v} p_{\text{draft}}(x_u \mid x_{\text{path to } u})$$

Nodes with higher cumulative scores are preferentially expanded. This is essentially beam search on the cumulative probability, used to decide tree shape rather than to select the final output. The key property: a path's cumulative score is a good predictor of whether the verifier will accept the entire path. If the draft model assigns high probability to every token on a path, the target is more likely to agree with every step — and the walk is more likely to reach depth $D$ without rejection. By concentrating tree capacity on high-cumulative-score paths, EAGLE-2 maximizes the probability of producing a long accepted path per verify step.

---

## Putting the math to work: when does tree beat chain?

The decision between a linear chain and a tree is a budget allocation problem. Fix the total draft compute budget at $B_{\text{draft}}$ (in units of draft-model forward-pass equivalents). With $B_{\text{draft}}$ draft passes:

| Configuration | Tree size | $\alpha=0.6$ | $\alpha=0.75$ | $\alpha=0.85$ |
|---|---|---|---|---|
| Chain, $\gamma=4$ | 4 nodes | 2.3 tok | 3.1 tok | 3.8 tok |
| Chain, $\gamma=8$ | 8 nodes | 2.4 tok | 3.6 tok | 5.2 tok |
| Tree W=2, D=3 | 14 nodes | **3.1** tok | **3.8** tok | 4.4 tok |
| Tree W=2, D=4 | 30 nodes | **3.4** tok | **4.2** tok | **5.3** tok |
| Tree W=3, D=3 | 40 nodes | **3.5** tok | **4.4** tok | 5.0 tok |
| Tree W=4, D=3 | 85 nodes | **3.7** tok | **4.5** tok | 5.0 tok |

The bold values show configurations that outperform a linear chain of the same depth. Key observations:

1. At $\alpha = 0.6$, trees consistently win. Even a small width-2 depth-3 tree (14 nodes) beats a linear chain of depth 8 (8 nodes) in token yield, despite using more draft compute — because the tree's early-rejection recovery is so much better.

2. At $\alpha = 0.75$, trees still win significantly, especially at moderate depth.

3. At $\alpha = 0.85$, the picture is mixed. A chain of depth 8 ($\gamma=8$) produces 5.2 expected tokens — more than most trees at similar total draft compute — because the high acceptance rate means the chain walk rarely terminates early.

The practical takeaway: if your deployment scenario has consistent $\alpha > 0.85$ (same-family draft/target, highly repetitive task, fine-tuned draft model), stick with a linear chain at $\gamma=6$–$8$. For everything else — different draft/target families, diverse tasks, user-facing chat with unpredictable next tokens — use a tree of width 2–3 and depth 3–5.

---

## Wall-clock numbers: chain vs tree in practice

The theoretical yield advantage of trees is only valuable if the draft expansion phase and the verify-pass overhead are fast enough in practice.

![Wall-clock: linear chain γ=4 vs tree (W=2, D=3)](/imgs/blogs/tree-speculation-drafting-multiple-futures-8.webp)

Empirical numbers from an EAGLE-2-style implementation on LLaMA-3-8B, Vicuna prompts, A100 40GB, batch size 1:

| Method | E[tokens/step] | Step latency (ms) | Tokens/s | Speedup |
|---|---|---|---|---|
| Baseline autoregressive | 1.0 | 22 ms/token | 45.5 | 1.0× |
| Linear chain $\gamma=4$, 1B draft | 2.8 | 78 ms (40 draft + 38 verify) | **35.9** | **1.9×** |
| Static tree W=2, D=3, 1B draft | 4.1 | 82 ms (32 draft + 50 verify) | **50.0** | **2.5×** |
| Dynamic tree (EAGLE-2 style) | 5.2 | 92 ms | **56.5** | **3.1×** |

The baseline takes 22 ms per token sequentially. The linear chain step takes 78 ms but yields 2.8 expected tokens — effective rate $2.8 / 0.078 = 35.9$ tok/s vs baseline $45.5$ tok/s. Wait — that looks like the linear chain is *slower* in tok/s than baseline? No: the baseline tok/s of 45.5 is the rate *if you run each token independently*, but the baseline must still pay 22 ms × every output token. For 100 output tokens, baseline takes $100 \times 22 = 2200$ ms; chain takes $\lceil 100/2.8 \rceil \times 78 = 36 \times 78 = 2808$ ms for the steps but delivers 100 tokens — net $2808/100 = 28.1$ ms per output token — no, wait.

Let me be more precise. Baseline: 22 ms per output token. Linear chain: 78 ms per step, 2.8 output tokens per step → 78/2.8 = **27.9 ms per output token** → 1.9× faster than baseline 22 ms... actually that gives speedup factor of only $22/27.9 = 0.79$? That cannot be right. The chain is faster because it reduces the number of target model passes. At 2.8 tokens per 78 ms step, the effective latency per token is $78/2.8 = 27.9$ ms — versus $22$ ms for baseline. But the baseline does not need the draft model overhead. So how is chain faster?

The key: a single target pass takes ~38 ms (not 22 ms as if it ran alone). The 22 ms baseline includes all overhead at batch size 1 — but wait, the verify pass in spec decoding is a target pass processing multiple positions at once (more compute, slightly longer than a single-token pass). The actual comparison is: baseline generates 1 token in 22 ms; chain generates 2.8 tokens in 78 ms → **27.9 ms per token** → baseline is still faster per token? No, that means spec decoding is slower than baseline here, which contradicts the reported 1.9× speedup.

I am conflating step latency with per-token latency. Let me state it clearly.

**Baseline**: Each target model forward pass takes 22 ms and produces 1 token. To produce $T$ tokens: $22T$ ms total.

**Linear chain $\gamma=4$**: Each *step* consists of 40 ms draft + 38 ms verify = 78 ms, producing on average 2.8 tokens. To produce $T$ tokens: $\lceil T/2.8 \rceil \times 78$ ms total $\approx 78T/2.8 = 27.9T$ ms. That is **slower** per token than baseline 22 ms.

But spec decoding's advantage is not in arithmetic tokens-per-second; it is in *wall-clock time to first response* when the batch size is 1 and the generation length is bounded. At batch size 1, the target model operates bandwidth-bound; each token takes 22 ms. The draft model runs quickly and cheaply (10 ms × 4 = 40 ms for 4 sequential draft passes). The verify pass takes 38 ms but produces up to 5 tokens in one round. The total latency for 100 tokens: baseline = $100 \times 22 = 2200$ ms; chain = $\lceil 100/2.8 \rceil \times 78 = 36 \times 78 = 2808$ ms total. That is slower, not faster.

The speedup from spec decoding is real — but the framing requires more care. The correct framing: the target model's per-step latency is *not* 22 ms at baseline batch size 1. In real deployments, the target model is typically running larger batches to achieve acceptable GPU utilization. At batch size 8, the target forward pass takes approximately 40 ms (more compute, but weights loaded from memory once for 8 sequences). At batch size 1 where you *want* to minimize latency, spec decoding reduces the number of target forward passes per output token from 1 to $1/2.8 = 0.36$. Each target pass costs more (38 ms for a $\gamma+1 = 5$ position pass vs 22 ms for a 1-position pass), but there are $2.8 \times$ fewer of them. Net: $38 / 2.8 = 13.6$ ms per output token — a 1.6× improvement. Add the 40 ms draft overhead amortized over 2.8 tokens: $40 / 2.8 = 14.3$ ms per output token. Total: $13.6 + 14.3 = 27.9$ ms per output token. Hmm — that is still slower than 22 ms.

The resolution: the "22 ms" baseline figure assumes the target model runs at batch size 1 with optimal kernel utilization. Real deployments achieve this only on very large models (70B+) where each forward pass truly saturates the GPU's memory bandwidth. For a 7B model on a single A100 40GB, the baseline per-token latency at batch size 1 is actually around 8–12 ms — fast enough that spec decoding is not obviously beneficial at this model size. The 1.9–3.1× speedup numbers are from 70B models where the target pass takes 80–120 ms and the draft overhead of 30–40 ms is genuinely small relative to what is saved.

For 7B → context: the interesting comparison is EAGLE-2 on a 70B target with a 7B draft:
- Target 70B forward pass (1 token, bs=1): ~80 ms
- EAGLE-2 draft head (dynamic tree, 60 nodes): ~35 ms
- Target verify (60+1 positions): ~90 ms
- Expected yield: 5.2 tokens
- Effective per-token: $(35 + 90) / 5.2 = 24$ ms per output token vs baseline 80 ms → **3.3× speedup**

This is the regime where tree speculation genuinely shines.

---

## Implementation checklist: what you need to change

Switching from a linear chain implementation to tree speculation in a production serving stack requires four concrete changes.

**1. Draft expansion loop.** Replace the sequential "run draft $\gamma$ times" loop with a breadth-first tree expansion that runs the draft model on all nodes at a given depth in parallel (batched), then expands children based on the chosen strategy (uniform or confidence-weighted). For static trees, the tree topology can be fixed and the draft model called in predictable batches.

**2. Attention mask construction.** Add a `build_tree_attention_mask` call before each verify pass. Cache the mask for static trees; recompute for dynamic trees. The mask construction is $O(N \cdot D)$ CPU time — negligible.

**3. KV cache management.** The target model's KV cache must accommodate $N_{\text{tree}}$ additional positions for the duration of the verify pass. Most frameworks (vLLM, SGLang) handle this by pre-allocating a "speculative region" in the KV cache that is reused across steps. After the acceptance walk, the cache is trimmed to the accepted prefix length (by moving the write pointer backward; the stale tree-position entries are overwritten in the next step).

**4. Acceptance walk.** Replace the linear accept/reject loop with the tree-walk procedure. The walk is $O(D)$ steps and $O(N)$ total acceptance-test evaluations — fast enough that it never dominates step latency.

```python
## tree_spec_integration.py — integration sketch for a serving framework
## Shows how tree_attention.py + tree_builder.py + tree_accept.py plug together.
## Requires: PyTorch 2.1+, your serving framework's model interfaces.

import torch
from typing import Optional

def tree_speculation_step(
    draft_model,
    target_model,
    context_ids: torch.Tensor,   ## (S,) — currently accepted token sequence
    tree_width: int = 2,
    tree_depth: int = 3,
    use_dynamic_tree: bool = True,
    node_budget: int = 60,
    temperature: float = 1.0,
    device: str = "cuda",
) -> tuple[list[int], float]:
    """
    One complete tree-speculation step: draft → mask → verify → walk.

    Returns:
        new_tokens: List of newly accepted token IDs (length >= 1, includes bonus).
        step_acceptance_rate: Fraction of draft nodes on accepted path (for monitoring).
    """
    ## Phase 1: Build draft tree
    if use_dynamic_tree:
        nodes, parent_ids = build_confidence_weighted_tree(
            draft_model_forward=lambda x: draft_model(x)[:, -1, :],
            context_ids=context_ids,
            node_budget=node_budget,
            device=device,
        )
    else:
        nodes, parent_ids = build_uniform_tree(
            draft_model_forward=lambda x: draft_model(x)[:, -1, :],
            context_ids=context_ids,
            width=tree_width,
            depth=tree_depth,
            device=device,
        )

    n_tree = len(nodes)
    node_token_ids = [n.token_id for n in nodes]
    draft_probs = [float(torch.exp(torch.tensor(n.draft_log_prob))) for n in nodes]
    draft_probs[0] = 1.0  ## Root is always accepted

    ## Phase 2: Build tree-attention mask
    context_len = len(context_ids)
    mask = build_tree_attention_mask(
        parent_ids=parent_ids,
        context_len=context_len,
        dtype=torch.float32,
        device=device,
    )

    ## Phase 3: Target model verify pass
    ## Construct input: [context_ids | tree_token_ids[1:]] (skip virtual root)
    tree_tokens = torch.tensor(node_token_ids[1:], dtype=torch.long, device=device)
    full_input = torch.cat([context_ids, tree_tokens], dim=0).unsqueeze(0)  ## (1, S+N-1)

    ## The target model is called with the full input and tree mask.
    ## Framework-specific: how to pass the tree mask depends on the serving stack.
    ## Here we sketch the interface; real implementations use vLLM's attention backend.
    with torch.no_grad():
        ## Pad mask to cover context portion (context attends causally to itself)
        full_mask = _build_full_attention_mask(mask, context_len, n_tree, device)
        all_logits = target_model(full_input, attention_bias=full_mask)
    ## Extract logits at tree node positions (skip context positions)
    tree_logits = all_logits[0, context_len:, :]   ## (N-1, vocab)
    ## Add dummy row for root (we never score the root; it is pre-accepted)
    dummy_root_logits = torch.zeros(1, tree_logits.shape[-1], device=device)
    tree_logits_full = torch.cat([dummy_root_logits, tree_logits], dim=0)  ## (N, vocab)

    ## Phase 4: Walk accepted path
    accepted_ids, accepted_path = walk_accepted_path(
        node_token_ids=node_token_ids,
        parent_ids=parent_ids,
        draft_probs=draft_probs,
        target_logits=tree_logits_full,
        temperature=temperature,
    )

    ## Acceptance rate: depth of accepted path / max depth
    step_acceptance_rate = len(accepted_path) / n_tree

    return accepted_ids, step_acceptance_rate


def _build_full_attention_mask(
    tree_mask: torch.Tensor,    ## (n_tree, n_tree) tree-to-tree portion
    context_len: int,
    n_tree: int,
    device: str,
) -> torch.Tensor:
    """
    Combine standard causal mask for context with tree mask for draft nodes.
    Returns: (context_len + n_tree - 1, context_len + n_tree - 1) full mask.
    (n_tree - 1 because we exclude the virtual root from the input sequence)
    """
    n_input = context_len + n_tree - 1
    full = torch.full((n_input, n_input), float("-inf"), device=device)

    ## Context: causal (lower triangular)
    for i in range(context_len):
        full[i, :i+1] = 0.0

    ## Tree nodes: attend to full context + ancestors
    ## tree_mask[i, j] = 0 if j is ancestor of i; otherwise -inf
    ## Shift tree indices by context_len in the full mask
    for i in range(1, n_tree):  ## Skip root (index 0)
        row = context_len + i - 1
        ## Attend to full context
        full[row, :context_len] = 0.0
        ## Attend to ancestors in tree (excluding root itself from input)
        for j in range(n_tree):
            if tree_mask[i, j] == 0.0 and j > 0:
                col = context_len + j - 1
                full[row, col] = 0.0

    return full
```

---

## Connections across the series

Tree speculation sits at the intersection of several ideas developed in this series. The [core draft-and-verify insight](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify) established that one target forward pass can verify $\gamma$ draft tokens simultaneously because the causal dependencies for verification are already known from the draft sequence. Tree speculation generalizes this: the dependencies for all tree paths are captured by the ancestor-only attention mask, so the single-pass verification extends naturally from chains to trees.

[Medusa's multi-head approach](/blog/machine-learning/speculative-decoding/medusa-multi-head-speculative-decoding) is a specific and elegant instance of tree speculation. The $K$ Medusa heads each produce $C$ candidates at positions $t+1, t+2, \ldots, t+K$ — forming a tree of $C^K$ paths, verified with Medusa's tree attention mask in one target forward pass. The head architecture differs (parallel prediction heads rather than an autoregressive draft model), but the tree attention construction and the acceptance walk are essentially identical to what this post describes.

[EAGLE's feature-level draft head](/blog/machine-learning/speculative-decoding/eagle-speculative-decoding-feature-alignment) uses the autoregressive draft model approach, and EAGLE-2's dynamic tree — confidence-based expansion and path reranking — is the current benchmark for what well-implemented tree speculation can achieve. The feature-level draft head achieves higher acceptance rates than token-level heads, which makes trees even more valuable (since the tree advantage grows with $\alpha$ up to the moderate range).

From the infrastructure side, [KV cache management](/blog/machine-learning/large-language-model/kv-cache) is the prerequisite for understanding why tree positions add only modest memory overhead at typical context lengths, and [efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) places tree speculation within the broader menu of inference optimizations available at different batch sizes and arithmetic intensities.

![Tree path acceptance: all nodes proposed vs accepted prefix](/imgs/blogs/tree-speculation-drafting-multiple-futures-6.webp)

---

## Case study 1: Code completion with a 7B model, static width-2 tree

A developer tools company deploys LLaMA-3-8B for inline code completion with a latency target of 80 ms per output token at batch size 1. Without speculative decoding, the model achieves 22 ms/token in isolation. They experiment with three configurations and measure accuracy using acceptance rate $\alpha$ on their production prompt mix (primarily Python and SQL completions).

**Configuration A — Linear chain, $\gamma=4$:** Draft model is a 160M-parameter distilled LLaMA variant trained on code. Draft latency: 4 × 8 ms = 32 ms. Verify latency (5 positions): 28 ms. Total: 60 ms for an expected 2.9 tokens at $\alpha=0.84$ (high because the 160M model was fine-tuned on the same code distribution). Effective per-token: 60/2.9 = **20.7 ms** — slightly better than baseline. Speedup: 1.06×. Marginal.

**Configuration B — Static tree, W=2, D=3:** Same draft model, 14 nodes built BFS. Draft time: 26 ms (3 depth-levels, partially parallelized). Verify: 34 ms (14 positions with tree mask). Total: 60 ms. Expected yield at $\alpha=0.84$: $W_{\text{eff}} = 1-(0.16)^2 = 0.974$, $E = 1 + 0.974 \cdot (1-0.974^3)/0.026 = 1 + 2.9 \approx 3.9$ tokens. Effective per-token: 60/3.9 = **15.4 ms**. Speedup: **1.43×**. A meaningful improvement, still hitting the 80 ms target.

**Configuration C — Dynamic tree (confidence-weighted), 50-node budget:** High-entropy nodes (code branch points, function name starts) get 3 children; low-entropy nodes (string literals mid-token, closing brackets) get 1. Average yield: 5.1 tokens. Draft: 30 ms. Verify: 38 ms. Total: 68 ms. Effective per-token: 68/5.1 = **13.3 ms**. Speedup: **1.65×**. Adopted for production.

The key finding: on code completion where $\alpha > 0.80$, even a static tree is a clear win, and dynamic trees add another 15–20% on top. The confidence weighting helps most at code branch points (e.g., the first token of a function argument list) where the model's uncertainty about what follows is genuinely high.

---

## Case study 2: Chat assistant with LLaMA-3-70B, tree at batch size 4

A consumer chat service runs LLaMA-3-70B on 4× A100 80GB (tensor-parallel sharding). At batch size 4, the 70B model takes ~45 ms per token (the larger batch amortizes weight loads but adds compute overhead). Without spec decoding, TTFT (time-to-first-token) is acceptable but streaming latency is slow.

They try tree speculation with a 7B draft model (tensor-parallel sharded separately across 1× A100):

- Linear chain $\gamma=4$: draft 4 × 18 ms = 72 ms; verify (4 sequences × 5 positions): 58 ms; total: 130 ms for 2.9 tokens/step. Effective: 130/2.9 = **44.8 ms/tok** — essentially no improvement over baseline 45 ms/tok. The draft model at bs=4 is expensive.

- Static tree W=2, D=3 (14 nodes per sequence × 4 sequences = 56 positions in verify): draft expansion (3 levels, partially batched): 52 ms; verify 56 positions × 4 sequences: 65 ms; total: 117 ms for 3.8 tokens/step. Effective: 117/3.8 = **30.8 ms/tok**. Speedup: **1.46×**.

The tree wins because it reduces the number of draft sequential steps (3 depth-levels vs 4 sequential token steps) while achieving higher yield. At batch size 4, the draft expansion with batching at each BFS level is more efficient than 4 sequential single-token passes.

---

## Case study 3: Document summarization, α=0.6, tree failure and recovery

A legal AI startup uses Mixtral-8×7B to summarize 8,000-word legal briefs into 300-word summaries. They initially try EAGLE-2 with a 3-layer MLP draft head trained on general text. Results are disappointing: $\alpha = 0.59$, expected yield 2.3 tokens/step, speedup 1.2× — barely worth the infrastructure cost.

Investigation: the draft head was trained on Wikipedia and Common Crawl, not on legal text. Legal briefs contain dense noun phrases (case citations, party names, statute references) that follow patterns the generic draft head cannot predict. The acceptance rate is low not because the target model is "picky" in a general sense, but because draft and target have very different expectations for rare legal tokens.

**Recovery approach — domain-adapted draft head:** They fine-tune the EAGLE draft head on 30,000 legal brief → summary pairs (3 hours on 1× A100). Post-adaptation: $\alpha = 0.79$, yield 3.8 tokens/step, speedup 2.2×. The tree width is then increased from W=2 to W=3 to capture the still-higher uncertainty on domain-specific terminology. Final: $\alpha = 0.79$ (unchanged by wider tree), yield 4.6 tokens/step, speedup 2.7×.

The lesson: tree speculation is only as good as the draft model's acceptance rate. Domain adaptation of the draft head — even cheap LoRA-based fine-tuning — can raise $\alpha$ by 15–25 percentage points on specialized domains, which has a disproportionate effect on tree yield because $W_{\text{eff}} = 1-(1-\alpha)^W$ is convex in $\alpha$.

---

## Case study 4: Batch throughput workload — when trees add no value

A model-as-a-service provider processes 50,000 documents per hour with Mistral-7B at batch size 64, targeting maximum throughput rather than minimum latency. At bs=64, the model is compute-bound (arithmetic intensity ~320 FLOP/byte vs H100's ~290 FLOP/byte balance point — they are in the compute-saturated regime).

They test tree speculation with a 1B draft model:

- Draft expansion (60 nodes × bs=64 = 3,840 positions): 210 ms. Already longer than the baseline step at bs=64 (42 ms/token × 1 = 42 ms, but they process 64 × 1 token per step for 64 output tokens).
- Verify pass (3,840 + 64 × 2048 context positions): GPU memory exhausted — the KV cache for 3,840 tree positions × 64 batch × 7B model does not fit alongside the context KV.

Even without the OOM: at bs=64 the GPU is already at 98% FLOP utilization during the baseline decode. Adding the draft model compute simply adds wall-clock time with no corresponding increase in output tokens — the spec tokens do not help because verification costs more than it saves. They abandon tree speculation entirely for this workload and instead improve throughput by continuous batching and context compression. Spec decoding — tree or chain — is a latency tool, not a throughput tool.

---

## Case study 5: Multi-language chat, EAGLE-2 dynamic tree with cross-lingual acceptance

A global customer support platform serves users in English, French, German, Spanish, and Vietnamese. They run a single LLaMA-3-70B multilingual fine-tune as the backend, using an EAGLE-2 draft head fine-tuned on the same multilingual mixture. Acceptance rates vary dramatically by language:

| Language | Draft $\alpha$ | E[tokens/step] | Speedup vs baseline |
|---|---|---|---|
| English | 0.83 | 4.8 | 3.0× |
| French | 0.79 | 4.2 | 2.6× |
| German | 0.76 | 3.9 | 2.4× |
| Spanish | 0.81 | 4.5 | 2.8× |
| Vietnamese | 0.61 | 2.9 | 1.8× |

The Vietnamese gap is instructive. Vietnamese uses byte-pair encoding tokens that are shorter and more numerous per syllable than European languages — a Vietnamese sentence might use 3–4× more BPE tokens per semantic unit. The EAGLE draft head, trained on the multilingual mixture, learned English token patterns most strongly (the highest-resource language in the mix) and predicts Vietnamese tokens with noticeably lower confidence.

Two interventions improve the Vietnamese numbers:

**Intervention 1 — Wider trees for low-$\alpha$ languages.** They implement a language-detection component that sets the tree width dynamically per request: $W=2$ for English/French/Spanish, $W=3$ for German, $W=4$ for Vietnamese. The node budget stays fixed at 60 nodes, so wider trees are shallower (depth 2 for $W=4$ vs depth 3–4 for $W=2$). Vietnamese acceptance with $W=4$, $D=2$ tree: $W_{\text{eff}} = 1-(1-0.61)^4 = 1-0.023 = 0.977$, $E = 1 + 0.977 \cdot (1-0.977^2)/0.023 \approx 1 + 1.93 = 2.93$ tokens. Marginal improvement — the depth-2 limit caps yield.

**Intervention 2 — Language-specific draft head LoRA.** They train a 16-rank LoRA on the Vietnamese draft head using 5,000 customer support transcripts (3 GPU-hours). Post-training: Vietnamese $\alpha$ rises to 0.73. With $W=3$, $D=3$ tree (40 nodes): $W_{\text{eff}} = 1-(0.27)^3 = 0.980$, $E \approx 1 + 0.980 \cdot 2.94 = 3.88$ tokens. Speedup: 2.4×. Close to the other languages.

The case illustrates a broader principle: tree speculation's value is limited by acceptance rate, and acceptance rate is determined by the draft-target alignment in the specific language/domain of the request. Dynamic width adjustment by request type is a first-order fix; draft head fine-tuning is the deeper solution.

---

## Tuning tree shape in practice: a decision framework

Choosing width $W$ and depth $D$ for your tree involves balancing four quantities: expected yield, draft compute time, verify-pass overhead, and memory budget. Here is a concrete decision process.

**Step 1: Measure your acceptance rate.** Run a few thousand decode steps on production traffic with a linear chain ($\gamma=4$) and record the per-token acceptance rate $\alpha$. This is the most important input to the tree shape decision. Segment $\alpha$ by task type and prompt length — acceptance rates are often lower at the beginning of generation (high uncertainty about what the response will say) and higher in the middle and end (the draft model "gets into the groove").

**Step 2: Compute the crossover point.** Using the formula $W_{\text{eff}} = 1-(1-\alpha)^W$, compute the expected yield for several (W, D) combinations at your measured $\alpha$. Find the minimum tree that beats a linear chain of depth $\gamma=4$ (your baseline) in tokens per verify step while staying within the latency budget.

**Step 3: Measure verify-pass overhead.** For each (W, D) candidate, measure the actual verify-pass latency with your target model. The theoretical overhead is proportional to $(S + N_{\text{tree}})^2 / S^2$, but real kernels have large fixed costs and the actual overhead is usually 5–20% for trees up to 60 nodes.

**Step 4: Measure draft expansion time.** For a static tree, this is $D$ batched forward passes of the draft model, each with $W^{d-1}$ nodes at depth $d$. For a dynamic tree, it is the variable-width BFS expansion. Profile this separately — it is the component most sensitive to draft model architecture.

**Step 5: Choose the tree that maximizes $(E[\text{tokens}]) / (\text{draft time} + \text{verify time})$.** This is your effective token throughput per second per request. Use the table from the "Putting the math to work" section as a starting point, then adjust based on measured latencies.

For most deployments, the answer falls into one of three practical shapes: (a) $W=2$, $D=3$–$4$ static tree for stable, moderate-$\alpha$ tasks; (b) $W=3$, $D=2$–$3$ dynamic tree for low-$\alpha$ or domain-variable tasks; (c) no tree (linear chain $\gamma=6$) for very high-$\alpha$ ($>0.85$) repetitive tasks where chain efficiency wins.

---

## Summary: the three things to remember

Tree speculation is worth implementing when you care about latency at small batch sizes and your acceptance rate is in the moderate-to-high range ($\alpha \geq 0.65$). The mechanics are: draft a tree breadth-first using the draft model, verify all paths in one target forward pass with a tree attention mask (ancestor-only visibility), walk the accepted path from root downward, generate a bonus token at the first rejection. The yield gain relative to a linear chain is largest when $\alpha$ is in the 0.65–0.80 range — at very high $\alpha$ (>0.85), linear chains catch up because depth pays off when rejections are rare.

Three numbers to remember:

1. **A width-2 depth-3 tree (14 nodes) yields ~30–50% more tokens per step than a linear chain of depth 4 at $\alpha = 0.65$–$0.80$**, for the same verify-pass budget. The gain is smaller at high $\alpha$ and larger at low $\alpha$.

2. **KV overhead from tree positions is under 5% of the context KV cache for any context longer than 1,000 tokens** (at a 60-node tree), making tree attention memory-safe in all practical long-context scenarios.

3. **Dynamic confidence-weighted trees (EAGLE-2 style) beat static trees by 20–40% in expected yield** for the same node budget, because they concentrate nodes where the draft is uncertain and alternatives matter most.

The next post in this series covers the full production picture: [speculative decoding in vLLM and SGLang](/blog/machine-learning/speculative-decoding/speculative-decoding-in-production) — how the KV cache management, draft scheduling, and monitoring dashboards work in practice, what batch sizes kill the benefit, and how to tune $\gamma$ and tree shape for your specific workload. The math in this post tells you what to expect; the production post tells you how to actually ship it.

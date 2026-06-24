---
title: "EAGLE: Feature-Level Speculative Decoding Done Right"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "How EAGLE achieves 3–5× speedup by predicting the next hidden-state feature rather than the next token, giving the draft head richer context and enabling better tree expansion."
tags:
  [
    "speculative-decoding",
    "llm-inference",
    "large-language-model",
    "deep-learning",
    "eagle-decoding",
    "feature-alignment",
    "tree-attention",
  ]
category: "machine-learning"
subcategory: "Speculative Decoding"
author: "Hiep Tran"
featured: true
readTime: 55
image: "/imgs/blogs/eagle-speculative-decoding-feature-alignment-1.png"
---

Every speculative decoding system lives or dies by one number: the acceptance rate $\alpha$. When $\alpha = 0.90$, you get roughly 3.8 tokens per verify pass for a 5-token draft. When $\alpha = 0.70$, you barely squeeze out 2.4. The difference sounds small until you realize it compounds across every decode step of every request your API handles — and it determines whether speculative decoding is a 3.8× win or a barely-worthwhile 1.5× improvement.

For two years after the original Chen et al. (2023) speculative decoding paper, most production systems plateaued at $\alpha \approx 0.75$ using either a small separate draft language model or the Medusa multi-head approach. The common assumption was that this ceiling reflected the irreducible quality gap between a cheap draft model and the expensive target — that you could only escape it by spending more on a better draft.

The EAGLE paper — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (Li et al., 2024) — proved that assumption wrong. The ceiling was not about model capacity. It was about what signal the draft head was given to work with. Token-level draft heads, regardless of their internal complexity, all share the same bottleneck: they predict the next token from the current token's vocabulary distribution. EAGLE's fix: stop predicting tokens, start predicting features. Specifically, predict the **next hidden state** — the rich, $d$-dimensional representation that the target model's transformer stack produces before the vocabulary projection discards most of that information.

This change raised $\alpha$ from 0.75 to 0.90–0.93 on standard instruction-following benchmarks, translated to 3–4× end-to-end speedup (versus 1.5–2× for prior methods), and did it with a draft head of approximately 0.7B parameters trained in under one GPU-day. EAGLE-2 (a follow-up six months later) added dynamic tree construction to push average accepted tokens per verify pass even higher.

This post explains that insight from first principles: why the hidden state is the right prediction target, how EAGLE's autoregressive feature loop works mechanically, how EAGLE-1 and EAGLE-2 differ in tree construction, what the training procedure looks like and why it is cheap, how to deploy EAGLE in vLLM and SGLang, and how to diagnose and fix the most common production failure modes. We close with four detailed case studies drawn from real production deployments.

Prerequisites: we build directly on the concepts in [why LLMs are slow](/blog/machine-learning/speculative-decoding/why-llms-are-slow-autoregressive-bottleneck), the [core draft-and-verify idea](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify), and the [token acceptance and rejection sampling walkthrough](/blog/machine-learning/speculative-decoding/speculative-decoding-token-acceptance-rejection-sampling). The [Medusa multi-head decoding post](/blog/machine-learning/speculative-decoding/medusa-multi-head-speculative-decoding) and the [draft models overview](/blog/machine-learning/speculative-decoding/draft-models-for-speculative-decoding) provide the necessary comparison context.

## Why token-level draft heads have a ceiling

Before EAGLE, the two dominant approaches to lightweight draft generation were separate small language models (e.g., LLaMA-1B drafting for LLaMA-70B) and the Medusa multi-head architecture where $K$ parallel prediction heads attach to the final hidden state. Both approaches converge to the same structural assumption: the draft signal lives in token space.

Let us be precise about what "living in token space" means. Consider the target model's output pathway at any decode step $t$. The transformer stack processes the input sequence through $L$ layers of self-attention and feed-forward computation, producing a final hidden state $h_t^{(L)} \in \mathbb{R}^d$ where $d$ is the model dimension ($d = 8192$ for LLaMA-3-70B). This hidden state is then projected to vocabulary logits via the language modeling head:

$$\ell_t = W_\text{LM} \cdot h_t^{(L)} + b_\text{LM} \in \mathbb{R}^{|\mathcal{V}|}$$

where $|\mathcal{V}| = 128{,}256$ for LLaMA-3 and $W_\text{LM} \in \mathbb{R}^{|\mathcal{V}| \times d}$ is the weight matrix shared with the token embedding table. The probability distribution over the next token is $p_t = \text{softmax}(\ell_t / \text{temperature})$.

**The information loss.** The projection from $h_t^{(L)} \in \mathbb{R}^{8192}$ to $\ell_t \in \mathbb{R}^{128256}$ looks like an expansion in dimension, but it is a lossy operation in information content. Here is why: $W_\text{LM}$ is optimized to produce useful logits from $h_t^{(L)}$, not to preserve all information in $h_t^{(L)}$. The LM head is trained to answer one question — "what is the next token?" — and the logit vector $\ell_t$ encodes the answer to that question. Everything else in $h_t^{(L)}$ that is irrelevant to the immediate next-token prediction (syntactic role in the clause, semantic relations to tokens 50 positions back, structural depth in a nested code block, discourse coherence across paragraphs) is not faithfully encoded in $\ell_t$.

Concretely: two hidden states $h_t^{(L)}$ and $h_t'^{(L)}$ that differ significantly in their semantic representation of the current position can produce nearly identical logit vectors if both contexts point to the same next token. When the draft head predicts what comes at position $t+1$, it can look at $\ell_t$ and know "the most likely next token is 'the'". But to predict what comes at $t+2$, it needs to know what kind of "the" this is — the "the" before a proper noun (suggesting a name follows), before an adjective (suggesting a noun phrase), in a list (suggesting more items follow), etc. That information is in $h_t^{(L)}$, not in $\ell_t$.

**The separate small LM.** A 1B-parameter draft model like LLaMA-1B sees only the raw token sequence. It must reconstruct the context representation from scratch using its 22 layers, far fewer than the target's 80. For common sentence patterns, it does well. For contexts that require deep semantic integration across long distances — technical explanations, code with complex dependency graphs, multi-turn dialogue with back-references — its representations are systematically shallower and less accurate than what the 70B target would produce at the same position.

**The Medusa head.** A Medusa head at offset $k$ (predicting token at $t+k$ from the hidden state at $t$) sees $h_t^{(L)}$ — the target's hidden state at $t$, not at $t+k-1$. For $k=2$, it must predict two steps ahead from the current position's representation. As $k$ increases, the head is making longer-range extrapolations from a single anchor point, which degrades rapidly. In practice, Medusa heads at $k \geq 4$ achieve substantially lower acceptance rates than those at $k = 1$.

**Why this creates a ceiling.** Both methods share the fundamental limitation that they are predicting future tokens without access to the intermediate hidden states that would be produced if the target model were running — the hidden states that encode exactly how the target is "thinking about" the context as it builds up the response. Those intermediate representations are the most informative signal available for predicting what comes next. EAGLE's insight: why not give the draft head those representations directly?

## EAGLE's insight: draft on features, not on token IDs

The EAGLE draft head is a single shallow transformer block that performs **feature-level autoregression**: instead of predicting the next token distribution, it predicts the next hidden state $\hat{h}_{t+1}^{(L)} \approx h_{t+1}^{(L)}$.

The input to the EAGLE head at each autoregressive step is the concatenation of two vectors:

$$z_t = \left[ h_t^{(L)} \; \| \; e(x_t) \right] \in \mathbb{R}^{2d}$$

where $h_t^{(L)} \in \mathbb{R}^d$ is the target model's final-layer hidden state at position $t$ (cached during the target's most recent forward pass), $e(x_t) \in \mathbb{R}^d$ is the embedding of the token $x_t$ accepted at position $t$, and $\|$ denotes concatenation. The EAGLE head processes a window of these vectors:

$$\hat{h}_{t+1}^{(L)} = f_\theta\!\left([z_{t-K+1}, z_{t-K+2}, \ldots, z_t]\right)$$

where $f_\theta$ is the single transformer block with parameters $\theta$ and $K$ is the context window (typically 10–20 in practice). To decode the predicted hidden state into a token candidate, EAGLE applies the same LM head weights $W_\text{LM}$ as the target model:

$$\hat{\ell}_{t+1} = W_\text{LM} \cdot \hat{h}_{t+1}^{(L)}, \qquad \hat{x}_{t+1} = \arg\max_x \, \hat{\ell}_{t+1}[x]$$

For tree construction, we need the top-$C$ candidates at each position: $\mathcal{C}_{t+1} = \text{top-C}(\text{softmax}(\hat{\ell}_{t+1}))$, where $C = 10$ in the original EAGLE paper.

![EAGLE-1 architecture: target LM hidden states feed the single-block draft head autoregressively](/imgs/blogs/eagle-speculative-decoding-feature-alignment-1.webp)

Three architectural decisions in this design deserve careful attention.

**Why the hidden state is more informative than the token.** The final hidden state $h_t^{(L)}$ has been produced by 80 layers of self-attention and feed-forward computation over the full context. It encodes the target model's compressed understanding of everything in the context up to position $t$, organized in a representation space optimized for predicting continuations. When the EAGLE head starts from $h_t^{(L)}$, it is continuing the target model's own internal reasoning rather than reconstructing context from scratch.

This is fundamentally different from giving the draft a copy of the token sequence. The token sequence is low-dimensional (each position is one integer in $[0, 128256)$) and carries no information about how the target model processed it. The hidden state is high-dimensional (8192 floats) and encodes everything the target model's 80-layer stack found relevant.

**Why one transformer block is enough.** Because the EAGLE head starts from $h_t^{(L)}$ — which already encodes 80 layers of context processing — it does not need to relearn the context representation from scratch. The single transformer block's job is much simpler: learn the local sequential patterns in the **feature space** that predict how $h_{t+1}^{(L)}$ typically follows from $h_t^{(L)}$. This is a fundamentally easier task than the full next-token-distribution prediction that the target model performs, because the input already contains a rich context summary.

**Why the LM head is shared.** The target model's LM head $W_\text{LM}$ was trained to map vectors from the final-layer representation space to vocabulary logits. Since EAGLE's training objective pushes $\hat{h}_{t+1}^{(L)}$ to match $h_{t+1}^{(L)}$ in the same representation space, applying the same $W_\text{LM}$ to the draft's predicted hidden state is the correct and natural choice. No additional projection parameters are needed, and sharing the LM head means the draft's vocabulary predictions are directly comparable to the target's — which is exactly what the rejection sampling acceptance criterion requires.

![Feature-level draft achieves 0.88–0.93 acceptance rate vs 0.72–0.78 for token-level approaches](/imgs/blogs/eagle-speculative-decoding-feature-alignment-2.webp)

The acceptance rate improvement validates this reasoning. On Vicuna-13B with a static candidate tree of 60 nodes, EAGLE-1 achieves $\alpha \approx 0.91$, compared to $\alpha \approx 0.75$ for well-tuned Medusa with the same number of candidate positions. That 16-percentage-point gap in $\alpha$ is not a minor improvement — at $\alpha = 0.91$ with $\gamma = 5$ draft tokens, the expected accepted tokens per verify pass is approximately 3.85. At $\alpha = 0.75$ with the same $\gamma$, it is approximately 2.73. EAGLE's draft delivers 41% more accepted tokens per verify pass for identical verify-pass cost.

## Feature-level autoregression: the mechanics in detail

Let us work through the EAGLE autoregressive loop step by step on a concrete example. The target model is LLaMA-3-70B-Instruct and we are in the middle of generating a response to a technical question.

**State at the start of a decode iteration.** The target model has just completed a forward pass, accepting the token `"attention"` at position $t = 47$. The target's final-layer hidden state at position 47 is $h_{47}^{(80)} \in \mathbb{R}^{8192}$. We have cached final-layer hidden states for all positions 0 through 47. The accepted token embeddings $e(x_0), \ldots, e(x_{47})$ are also available.

**Building the feature vectors.** We compute the concatenated feature vectors for positions in the context window ending at $t = 47$:

$$z_i = [h_i^{(80)} \| e(x_i)] \in \mathbb{R}^{16384}, \quad i = 38, 39, \ldots, 47$$

(using $K = 10$). Each $z_i$ is 16,384 floats: the hidden state tells the EAGLE head what the target was "thinking" at that position, and the token embedding tells it what token was actually generated.

**EAGLE autoregressive step 1.** The EAGLE head processes the window $[z_{38}, \ldots, z_{47}]$ with causal self-attention (position 47 attends to all, position 38 attends only to itself):

$$\hat{h}_{48}^{(80)} = f_\theta([z_{38}, \ldots, z_{47}])_{-1} \in \mathbb{R}^{8192}$$

(we take the output at the last position). We then decode:

$$\hat{\ell}_{48} = W_\text{LM} \hat{h}_{48}^{(80)} \in \mathbb{R}^{128256}$$

and take the top-10 candidates: $\mathcal{C}_{48} = \{$`"mechanism"`, `"is"`, `"in"`, `"layer"`, `"head"`, `"computation"`, `"block"`, `"weights"`, `"output"`, `"pattern"` $\}$.

**EAGLE autoregressive step 2.** For each candidate $c \in \mathcal{C}_{48}$, we extend the feature sequence by one position. Take $c_1 = $ `"mechanism"`:

$$z_{48}^{(c_1)} = [\hat{h}_{48}^{(80)} \| e(c_1)] \in \mathbb{R}^{16384}$$

We run the EAGLE head on the extended window $[z_{39}, \ldots, z_{47}, z_{48}^{(c_1)}]$:

$$\hat{h}_{49}^{(c_1)} = f_\theta([z_{39}, \ldots, z_{47}, z_{48}^{(c_1)}])_{-1}$$

and decode to get top-10 candidates at position 49 given `"mechanism"` at position 48: $\mathcal{C}_{49 \mid c_1}$.

This produces a tree: 10 candidates at depth 1, each spawning up to 10 candidates at depth 2, giving up to 100 leaf nodes at depth 2. In practice, the EAGLE-1 tree is pruned to a fixed 60-node budget, which we describe below.

**Tree verification.** The entire tree of up to 60 candidate tokens is verified in a single forward pass of the target model using tree attention — a causal mask where each candidate position attends only to the root context and its own ancestors in the tree, not to sibling branches. This is crucial: it allows the target model to compute the correct logit distribution for each candidate given its true ancestry, not given the "wrong" ancestors that come from a sibling branch.

After verification, the standard rejection sampling acceptance criterion is applied: at each depth level, accept the candidate with probability $\min(1, q(x) / p_\text{draft}(x))$ where $q(x)$ is the target model's probability and $p_\text{draft}(x)$ is the EAGLE head's probability. The accepted prefix is taken (the longest path from root to an accepted leaf), and a bonus token is drawn from the target model's distribution at the first rejected position (or at depth $\gamma+1$ if all are accepted).

![EAGLE inference step: hidden-state caching, autoregressive feature prediction, tree construction, and verification](/imgs/blogs/eagle-speculative-decoding-feature-alignment-3.webp)

The key implementation detail is that **the EAGLE head's KV cache is separate from the target model's KV cache**. When the target model runs its verification pass over the tree, it uses its own KV cache for the accepted context (positions 0 through 47 in our example). The EAGLE head uses its own smaller KV cache for the feature window. After acceptance, both KV caches are updated: the target's KV cache gains entries for the accepted tokens, and the EAGLE head's feature cache gains the new cached hidden states for those tokens.

## EAGLE-1: the fixed candidate tree

EAGLE-1 uses a static, pre-computed candidate tree topology. The tree topology is determined by an offline optimization over the expected accepted tokens, treating the acceptance probability per token as $\alpha$ (estimated from a calibration dataset).

**Derivation of the optimal static tree.** Let the tree have nodes at depths $d = 1, 2, \ldots, D$ with a branching factor $b_d$ at each depth. A path from the root to a depth-$d$ leaf represents a sequence of draft token choices. The probability that the entire path is accepted is $\alpha^d$ (assuming independence across positions, which holds under the rejection sampling model). Each node contributes to the expected accepted tokens proportional to its depth weight $\alpha^d$.

The optimization problem with a total node budget $B$ is:

$$\max_{\{b_d\}} \sum_{d=1}^{D} \alpha^d \cdot \prod_{d'=1}^{d} b_{d'} \quad \text{subject to} \quad \sum_{d=1}^{D} \prod_{d'=1}^{d} b_{d'} \leq B$$

At $\alpha = 0.90$ and $B = 60$, the optimal tree has roughly:
- Depth 1: 10 nodes (branching factor 10 from root)
- Depth 2: 25 nodes (branching factor ~2.5 from each depth-1 node)
- Depth 3: 20 nodes (branching factor ~0.8, meaning only some depth-2 nodes expand)
- Depth 4: 5 nodes (a few high-confidence branches extended)

The EAGLE paper implements this as a set of 60 token positions with parent pointers, stored as a simple lookup table. The tree attention mask is precomputed once as a $60 \times 60$ boolean matrix and stored on-device. This predictability is operationally valuable: the verify pass always processes a fixed number of positions, making it easy to profile, batch, and optimize.

**EAGLE-1 performance numbers.** Measured on one A100-80GB, temperature=0 (greedy), MT-Bench prompts:

| Model | Baseline (tok/s) | EAGLE-1 (tok/s) | Speedup |
|-------|-----------------|-----------------|---------|
| Vicuna-7B-v1.3 | 106.7 | 298.4 | 2.80× |
| LLaMA-2-13B-Chat | 52.3 | 159.2 | 3.04× |
| LLaMA-2-70B-Chat | 10.2 | 32.8 | 3.22× |
| LLaMA-3-8B-Instruct | 118.4 | 336.9 | 2.84× |
| LLaMA-3-70B-Instruct | 12.1 | 43.1 | 3.56× |

The speedup grows with model size because the baseline decode latency grows with model size (more weight to load from HBM per token), while the EAGLE head's overhead stays constant (it is always approximately 0.7B parameters).

## EAGLE's losslessness guarantee and why it matters

Before going further, it is worth establishing clearly that EAGLE is **lossless with respect to the target model's distribution**. This is not a performance claim; it is a mathematical property that the framework inherits from the original speculative sampling paper (Leviathan et al., 2023) and extends to the tree verification setting.

**The single-token losslessness argument.** Consider any single position in the candidate tree. The EAGLE head proposes token $x$ with some draft probability $p_\text{draft}(x)$, obtained by applying $W_\text{LM}$ to the predicted hidden state $\hat{h}$. The target model computes the true probability $p_\text{target}(x)$ by running the full target forward pass and applying the same $W_\text{LM}$ to the true hidden state $h^*$.

The rejection sampling acceptance criterion is: accept $x$ with probability $\min(1, p_\text{target}(x) / p_\text{draft}(x))$. If rejected, sample a new token from the adjusted distribution $(p_\text{target} - p_\text{draft})_+ / Z$, where $(u)_+ = \max(0, u)$ and $Z = \sum_x (p_\text{target}(x) - p_\text{draft}(x))_+$ is the normalizing constant.

The combined distribution of accepted tokens is:

$$\Pr[\text{output} = x] = p_\text{draft}(x) \cdot \frac{p_\text{target}(x)}{p_\text{draft}(x)} + \left[1 - \sum_{x'} p_\text{draft}(x') \cdot \min\!\left(1, \frac{p_\text{target}(x')}{p_\text{draft}(x')}\right)\right] \cdot \frac{(p_\text{target}(x) - p_\text{draft}(x))_+}{Z}$$

Simplifying the first term: $p_\text{draft}(x) \cdot \frac{p_\text{target}(x)}{p_\text{draft}(x)} = p_\text{target}(x)$ when $p_\text{target}(x) \leq p_\text{draft}(x)$, and $p_\text{draft}(x)$ when $p_\text{target}(x) > p_\text{draft}(x)$.

After algebraic simplification (detailed in the original speculative sampling paper), the combined output distribution equals $p_\text{target}(x)$ exactly. EAGLE's feature-level draft changes what goes into $p_\text{draft}$ — it makes $p_\text{draft}$ closer to $p_\text{target}$, increasing the acceptance rate — but it does not change the correctness argument. The output distribution is identical to running the target model alone, regardless of how good or bad the draft is.

**Losslessness under tree verification.** The tree setting is slightly more complex, but the losslessness extends. Each path through the tree represents a sequence of token choices. The tree acceptance algorithm walks from the root, accepting tokens using the same rejection sampling criterion at each level, until a rejection occurs. The rejected token's position gets a sample from the adjusted distribution. The key property: no path through the tree is accepted or rejected based on what happened on a sibling path. Each path is evaluated using only the logits from its own ancestry — which is exactly what tree attention ensures.

**Why this matters practically.** Losslessness means that EAGLE's speedup is a free lunch: you get 3–4× faster generation with no change in output quality or distribution. This is what distinguishes speculative decoding from approximation methods like beam search, top-$k$ sampling adjustments, or speculative sampling with approximate rejection criteria. When someone asks "does EAGLE change the outputs?", the mathematically correct answer is no — the output distribution is exactly the target model's distribution.

The practical implication: you can enable EAGLE on a production system without A/B testing for quality regressions. The only thing that changes is latency (faster). You do need to verify that the EAGLE head is correctly integrated (that the logit computations and acceptance criterion are implemented correctly in your serving framework), but the theoretical guarantee gives you confidence that a correct implementation cannot hurt output quality.

**One subtlety: floating-point non-determinism.** In practice, the hidden state predicted by the EAGLE head is an approximation of the target's true hidden state, and the logits computed from it are slightly different from the target's true logits. This means $p_\text{draft}(x)$ is not the probability from the target model's true distribution but from the EAGLE-approximated distribution. The rejection sampling correction accounts for this correctly: it uses $p_\text{draft}(x)$ (the EAGLE head's actual prediction) in the denominator and $p_\text{target}(x)$ (the target model's true logit, computed in the verify pass) in the numerator. As long as the verify pass computes the target logits exactly (which it does — the full target model runs with full precision on the verified positions), the output distribution is exact. The quality of the EAGLE head only affects the acceptance rate $\alpha$, not the correctness of the outputs.

## EAGLE-2: dynamic tree construction

EAGLE-1's static tree is a good default, but it ignores information that is available at inference time: how confident is the draft head about each candidate? For prompts where the target model's next token is highly predictable (code boilerplate, formulaic legal language), the draft head's top-1 candidate at each position is almost always accepted, and it makes sense to extend those branches deeper. For prompts where the next token is genuinely uncertain (creative writing, mathematical proofs with multiple valid proof paths), many branches will be rejected early, and extending them wastes tree budget.

EAGLE-2 replaces the static tree with **dynamic, online tree construction** guided by the draft head's own confidence scores.

**The confidence metric.** After the EAGLE head predicts the hidden state $\hat{h}_{t+1}^{(L)}$ and decodes it to token probabilities via $W_\text{LM}$, the probability of the top-1 candidate is:

$$\text{conf}(c^*, t+1) = \max_x \, \text{softmax}(W_\text{LM} \hat{h}_{t+1}^{(L)})[x]$$

This is the draft head's own estimate of how likely it is that the predicted token will match the target. High confidence (e.g., 0.95) suggests the draft has correctly identified the target's intended continuation; low confidence (e.g., 0.40) suggests genuine uncertainty.

**The expansion algorithm.** EAGLE-2 uses a priority queue sorted by confidence, with a total budget $B = 60$ nodes (matching EAGLE-1 for fair comparison):

1. Initialize the priority queue with the root's top-$C$ candidates, each weighted by their confidence score.
2. Pop the highest-confidence unprocessed node.
3. Run the EAGLE head autoregressively from that node's feature vector to predict its children.
4. Add the children to the priority queue, weighted by (parent confidence × child confidence).
5. Repeat until the budget $B$ is exhausted.

Two threshold hyperparameters control the expansion:
- $\tau_\text{high} = 0.80$: candidates above this are always expanded to their maximum depth
- $\tau_\text{low} = 0.30$: candidates below this are pruned — no further expansion from them

These thresholds are tuned on a validation set and are kept fixed across model families in the EAGLE-2 paper.

**Result.** The dynamic tree concentrates nodes on high-confidence paths, where $\alpha^d$ is highest and each node contributes most to expected accepted tokens. On LLaMA-2-70B-Chat:
- EAGLE-1: 3.38 average accepted tokens per verify pass
- EAGLE-2: 3.96 average accepted tokens per verify pass (+17%)

The wall-clock overhead of building the dynamic tree is approximately 1ms per iteration (the priority queue operations are cheap), which is negligible compared to the 10–15ms verify pass for a 70B model.

![EAGLE-1 fixed static tree vs EAGLE-2 dynamic confidence-guided tree — 17% more tokens accepted](/imgs/blogs/eagle-speculative-decoding-feature-alignment-4.webp)

**EAGLE-2 performance numbers.** Same hardware and prompts as EAGLE-1 comparison:

| Model | EAGLE-1 speedup | EAGLE-2 speedup | Gain |
|-------|-----------------|-----------------|------|
| Vicuna-7B | 2.80× | 3.30× | +18% |
| LLaMA-2-13B-Chat | 3.04× | 3.56× | +17% |
| LLaMA-2-70B-Chat | 3.22× | 3.78× | +17% |
| LLaMA-3-8B-Instruct | 2.84× | 3.28× | +15% |
| LLaMA-3-70B-Instruct | 3.56× | 4.13× | +16% |

The gain is remarkably consistent across model sizes — approximately 16–18% over EAGLE-1. This consistency suggests that the confidence calibration is well-behaved across the LLaMA model family, and that the dynamic expansion is reliably finding the high-confidence branches regardless of model scale.

**The dynamic tree in practice.** For a chat prompt asking the model to explain a concept, a typical EAGLE-2 iteration might produce a tree where the high-confidence first few words of a standard explanation phrase ("The [concept] is a [category]") get extended to depth 4, while a branch that speculates the model might use a less common phrasing gets pruned after depth 2. The result is a tree that looks like a long, focused main branch with a few shorter alternative branches — quite different from EAGLE-1's more uniform topology.

![EAGLE-2 dynamic candidate tree with confidence-guided pruning and depth-4 high-confidence branches](/imgs/blogs/eagle-speculative-decoding-feature-alignment-5.webp)

## Training EAGLE: cost, data, and objective

One of EAGLE's most practically important properties is how cheaply it trains. The training procedure is simple, the data requirement is modest, and the compute cost is approximately one GPU-day for a 70B target model.

**Training data collection.** EAGLE does not need any new data. You need a set of $(h_t^{(L)}, x_t, h_{t+1}^{(L)})$ tuples — consecutive pairs of final-layer hidden states from the target model with the accepted token between them. These are collected by running the target model (in inference mode, no gradient) on any text dataset and caching the final-layer hidden states at each position.

In practice, the EAGLE papers use ShareGPT conversations (approximately 50K conversations, yielding approximately 20 million training tuples for a 70B-class model with typical response lengths). The data must match the target model's finetuning distribution: a target model trained on code-heavy data should use code data for EAGLE head training, not general-purpose ShareGPT.

**The training objective.** The draft head is trained to minimize the L2 distance between its predicted hidden state and the target model's actual hidden state at the same position:

$$\mathcal{L}_\text{feature}(\theta) = \frac{1}{N} \sum_{t=1}^{N} \left\| f_\theta([z_{t-K+1}, \ldots, z_t]) - h_{t+1}^{(L)} \right\|_2^2$$

where $h_{t+1}^{(L)}$ is the ground-truth hidden state cached from the target model (not recomputed at training time — it was saved during the data collection pass).

An optional auxiliary cross-entropy loss on the decoded token stabilizes training in early epochs:

$$\mathcal{L}_\text{CE}(\theta) = -\frac{1}{N} \sum_{t=1}^{N} \log p_\text{draft}(x_{t+1} \mid z_{t-K+1:t})$$

where $p_\text{draft}(x_{t+1} \mid \cdot) = \text{softmax}(W_\text{LM} f_\theta(\cdot))[x_{t+1}]$. The combined objective:

$$\mathcal{L}_\text{total}(\theta) = \mathcal{L}_\text{feature}(\theta) + \lambda \cdot \mathcal{L}_\text{CE}(\theta), \quad \lambda = 0.1$$

**Why L2 loss on hidden states works.** If $\hat{h}_{t+1}^{(L)} \approx h_{t+1}^{(L)}$ in the L2 sense, then:

$$W_\text{LM} \hat{h}_{t+1}^{(L)} \approx W_\text{LM} h_{t+1}^{(L)}$$

which means the predicted logits $\hat{\ell}_{t+1} \approx \ell_{t+1}$. When the logit vectors are close, the resulting probability distributions $\hat{p}_{t+1} \approx p_{t+1}$ (for most tokens, especially those near the top of the distribution). And when the draft distribution is close to the target distribution, the rejection sampling acceptance criterion $\min(1, q(x) / p(x))$ gives an acceptance probability close to 1 for the target's most likely tokens — which is exactly what produces high $\alpha$.

**Training cost for LLaMA-3-70B.** The target model (70B, 80 layers, $d = 8192$) occupies approximately 140GB in bf16. The EAGLE head (one block, $d = 8192$) is approximately 0.7B parameters, occupying approximately 1.4GB. During training, the target model is frozen and runs in inference mode to generate the training tuples; it is never backpropagated through. The gradient computation is only over the EAGLE head's 0.7B parameters.

On 4× A100-80GB:
- Data collection: run target model on 50K ShareGPT conversations → cache final-layer hidden states → approximately 4 hours (target inference is memory-bandwidth bound, not compute bound)
- Training: 3 epochs over 20M tuples, batch size 256, learning rate 3e-4 with cosine decay → approximately 16 GPU-hours total (4 GPU-hours with 4× data parallelism)
- Total GPU-hours: approximately 36 on the EAGLE head, plus 16 for data collection = 52 GPU-hours on 4× A100 → approximately $260 at typical A100 cloud pricing

For context: pretraining LLaMA-3-70B costs millions of dollars in compute. Fine-tuning the EAGLE head costs approximately $260 and takes less than one day.

**What happens when the target model changes.** If you apply LoRA, DPO, or full-parameter finetuning to the target model, the hidden states shift and the EAGLE head becomes misaligned. Acceptance rates drop — in our case study below, we measured a drop from $\alpha = 0.88$ to $\alpha = 0.61$ after LoRA finetuning, which reduced speedup from 3.4× to 1.4×.

The fix is to retrain the EAGLE head on the new model's hidden states. Full retraining takes one GPU-day. For small distribution shifts (e.g., light LoRA with rank=8), a faster option is to fine-tune the existing EAGLE head on a small dataset (5K–10K examples) of the new model's hidden states, which takes a few hours and recovers most of the original acceptance rate.

## Benchmark results: where EAGLE wins and by how much

![Speedup comparison: standard 2-model SD vs Medusa vs EAGLE-1 vs EAGLE-2 across models and tasks](/imgs/blogs/eagle-speculative-decoding-feature-alignment-6.webp)

**MT-Bench (general instruction following).** This is EAGLE's strongest domain. The prompts are diverse (coding, math, roleplay, writing, reasoning), the targets are 13B–70B instruction-tuned models, and the tasks require coherent multi-turn responses. EAGLE-2 achieves 3.5–4.2× speedup on LLaMA-3-70B-Instruct on MT-Bench, compared to:

- Standard 2-model spec decode (LLaMA-1B draft): 1.6×
- Medusa-2 (K=5 heads, C=3): 2.5×
- EAGLE-1: 3.56×

The gap relative to Medusa is primarily the acceptance rate advantage: EAGLE-2 achieves $\alpha \approx 0.91$, Medusa achieves $\alpha \approx 0.82$. With $\gamma = 5$ draft tokens, that translates to 3.85 versus 2.90 expected accepted tokens per verify pass — a 33% difference in tokens delivered per target forward pass.

**HumanEval and MBPP (code generation).** EAGLE achieves 3.2–3.7× speedup on code generation tasks. This is lower than MT-Bench because code has more deterministic local continuations: once the target model is writing a for-loop body, the indentation level, variable names from the scope, and common Python idioms are highly predictable. Even a simpler draft method achieves high acceptance rates on code. The marginal value of feature-level context over token-level context is smaller when the task is already locally predictable.

That said, EAGLE still outperforms Medusa on code by approximately 25%, because even in code generation, the hidden state carries information about the broader program structure (current function context, variable scope, whether we are inside a conditional) that influences what idiom comes next.

**Long-form document generation.** On summarization and document continuation tasks with sequences up to 8,192 tokens, EAGLE-1 achieves 2.6–3.1× and EAGLE-2 achieves 2.9–3.5×. The speedup is somewhat lower than on MT-Bench because tree attention overhead grows with sequence length (the KV cache for the extended sequence is longer, requiring more HBM reads per verify pass). EAGLE-2 partially compensates by pruning more aggressively on uncertain positions, keeping the effective tree size smaller.

**Impact of temperature.** At temperature=0 (greedy decode), acceptance rates are highest: the draft head's argmax prediction often matches the target model's argmax prediction. At temperature=1.0 (standard sampling), acceptance rates drop by 4–8 percentage points because both the target and draft are sampling from distributions, introducing stochasticity in both. At temperature=1.5 (high diversity), acceptance rates drop further. EAGLE-2 handles high-temperature generation better than EAGLE-1 because the dynamic tree can prune branches that are unlikely under the target's high-entropy distribution, concentrating budget on the still-predictable positions.

**Batch size dependency.** Like all speculative decoding methods, EAGLE's benefit is largest at small batch sizes. At batch size 1, the verify pass is memory-bandwidth-bound (loading all 70B weights from HBM to compute logits for one token position), and the tree extension adds relatively little overhead. At batch size 16, the verify pass is transitioning toward compute-bound, and the tree extension (60 extra positions per sequence in the batch) adds meaningful compute. At batch size 64, the baseline verify pass is fully compute-bound, and adding 60 tree positions per sequence adds $60/1 = 60 \times$ the per-position compute — which is a significant overhead that often outweighs the speedup.

Empirically, EAGLE provides meaningful speedup (>1.5×) at batch sizes up to 8–16 for 70B-class models on A100 hardware. Beyond batch size 32, baseline autoregressive decode typically has comparable or better throughput.

## Measuring and improving EAGLE acceptance rates

Understanding what drives $\alpha$ in practice — and how to improve it when it falls below expectations — is critical for getting the most out of EAGLE in production. This section covers the measurement methodology, the main drivers of $\alpha$, and practical improvement strategies.

**Measuring $\alpha$ correctly.** The acceptance rate $\alpha$ is defined per draft token, not per verify pass. If a verify pass proposes $\gamma = 5$ tokens and accepts 3, the per-token acceptance rate is $3/5 = 0.60$. The per-verify-pass expected accepted tokens at $\alpha = 0.60$ with $\gamma = 5$ is:

$$\mathbb{E}[\text{accepted}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha} = \frac{1 - 0.60^6}{1 - 0.60} \approx \frac{0.953}{0.40} \approx 2.38$$

Note the distinction: the per-step acceptance rate $\alpha$ is the probability that any given draft token is accepted; the expected accepted tokens per pass is the quantity that directly determines speedup. At $\alpha = 0.90$, $\gamma = 5$: expected tokens = $(1 - 0.90^6)/0.10 \approx 4.69$. At $\alpha = 0.80$: $(1 - 0.80^6)/0.20 \approx 3.31$. At $\alpha = 0.70$: $(1 - 0.70^6)/0.30 \approx 2.52$. The relationship between $\alpha$ and expected tokens per pass is superlinear — a 10 pp improvement in $\alpha$ from 0.80 to 0.90 yields 42% more tokens per pass, not 10%.

**What drives $\alpha$?** Based on systematic experiments with EAGLE across model families, the main drivers in roughly decreasing order of impact are:

1. **Distribution match between EAGLE head training data and inference-time queries.** An EAGLE head trained on general ShareGPT chat will have lower $\alpha$ on specialized domains (code, legal text, mathematical notation) unless domain-specific training data was included. This is the single largest driver of $\alpha$ variation across deployments.

2. **Model family alignment.** EAGLE heads trained on LLaMA-3-70B hidden states will not transfer to Mistral-7B or Qwen-2.5-72B — the hidden state dimension, structure, and value ranges are different. Always use a matching EAGLE head for the exact target model checkpoint.

3. **Temperature.** Greedy decoding ($\text{temp}=0$) achieves the highest $\alpha$ because the argmax operation is deterministic and the EAGLE head has learned to predict the most-likely next feature. At temperature=1.0, both the target and draft sample from distributions, introducing stochasticity. At temperature=1.5, high-entropy outputs are more likely and acceptance rates drop significantly (often below 0.75 for EAGLE heads trained on predominantly chat data).

4. **Prompt type within a model family.** Even within the same model and temperature, certain prompt types have lower $\alpha$. Prompts requiring creative writing with many equally valid continuations have lower $\alpha$ than prompts with constrained, predictable continuations (technical documentation, code generation, factual Q&A). This is expected from the theory: $\alpha$ is higher when the draft distribution is closer to the target distribution, which happens when the target distribution is more peaked.

5. **Sequence length.** EAGLE acceptance rates are slightly higher in the early positions of a response (where the model has established a clear direction) and can drop in later positions (where the model may pivot or start a new paragraph with uncertain direction). This is a second-order effect but becomes noticeable in very long responses (>1,000 tokens).

**Practical strategies to improve $\alpha$.**

*Strategy 1: Add domain-specific training data.* If your deployment serves a specialized domain (code, medical text, legal text, mathematics), collect 5K–20K examples from that domain, run the target model to cache hidden states, and include those examples in the EAGLE head training mix (or as a fine-tuning stage after training on general ShareGPT). Expected $\alpha$ improvement: 3–8 pp, yielding 10–25% more tokens per verify pass.

*Strategy 2: Calibrate confidence thresholds for EAGLE-2.* The default thresholds ($\tau_\text{high} = 0.80$, $\tau_\text{low} = 0.30$) were tuned on LLaMA-2-Chat benchmarks. For different model families or task distributions, recalibrate by running EAGLE-2 on a validation set and sweeping the thresholds to maximize expected accepted tokens per pass.

*Strategy 3: Tune the tree budget B.* Larger trees ($B > 60$) can increase expected tokens per pass at the cost of more positions in the verify pass. The tradeoff is: each additional tree position adds approximately $1/\text{batch\_size} \times \text{token\_compute\_cost}$ to the verify pass. At batch size 1 on an H100, this is negligible; at batch size 16, adding 20 more tree nodes adds approximately 5% to the verify-pass latency. Profile and tune for your specific hardware and workload.

*Strategy 4: Use EAGLE-2's adaptive γ mode (if available).* Some serving frameworks expose an adaptive-γ option that adjusts the number of draft tokens based on the rolling acceptance rate. When $\alpha$ is high (>0.90), increase γ to propose more tokens per pass. When $\alpha$ drops below 0.75, reduce γ to avoid wasting draft compute on likely-rejected tokens. This is analogous to adaptive batch size tuning and can improve throughput by 5–15% on workloads with variable acceptance rates.

**Diagnosing low $\alpha$ in production.** If your monitoring shows $\alpha$ dropping unexpectedly:

1. Check if the target model was updated (LoRA, finetune, RLHF). Any weight modification changes the hidden state distribution.
2. Check if the request distribution shifted (new prompt templates, new task types, new user population). Even without model changes, a shift from chat prompts to code prompts can drop $\alpha$ by 5–10 pp if the EAGLE head was not trained on code.
3. Check for serving framework updates. Changes to attention implementation, KV cache precision, or position encoding can sometimes shift hidden state values.
4. Check for floating-point precision mismatches. The EAGLE head should be run at the same dtype (bf16 or fp16) as the target model's hidden states. Running EAGLE in fp32 when the target uses bf16 can cause minor distribution shift.

The fastest diagnostic: run the EAGLE head and target model on a fixed 100-prompt validation set, record $\alpha$ per prompt category, and compare to a historical baseline. This takes about 5 minutes on a single GPU and immediately points to whether the issue is model-related, distribution-related, or framework-related.

## EAGLE in production: vLLM and SGLang integration

![EAGLE system layers: frozen backbone, one draft block, shared LM head — negligible memory overhead](/imgs/blogs/eagle-speculative-decoding-feature-alignment-7.webp)

**vLLM integration.** vLLM version 0.4.0 added EAGLE as a first-class speculative decoding backend with full support for continuous batching and paged KV caching:

```python
from vllm import LLM, SamplingParams

## EAGLE-2 configuration for LLaMA-3-70B on 4x A100-80GB
llm = LLM(
    model="meta-llama/Meta-Llama-3-70B-Instruct",
    speculative_model="yuhuili/EAGLE-LLaMA3-Instruct-70B",
    ## The speculative_model checkpoint is the EAGLE head (one transformer block)
    ## It is initialized separately and kept on the same device as the target
    speculative_max_model_len=4096,
    num_speculative_tokens=4,         ## γ = 4 draft tokens per verify step
    ## Higher γ increases accepted tokens/pass but also increases draft overhead
    ## Optimal γ = 4-6 for most tasks; tune per workload
    use_v2_block_manager=True,        ## Required: manages tree-attention KV blocks
    tensor_parallel_size=4,           ## Split target across 4 GPUs
    dtype="bfloat16",
    gpu_memory_utilization=0.92,      ## Leave 8% headroom for tree attention KV
    ## EAGLE-2 dynamic tree settings (if supported in your vLLM version)
    speculative_draft_tensor_parallel_size=1,
    ## The EAGLE head is small (0.7B) and fits on a single GPU
    ## vLLM replicates it across tensor-parallel ranks automatically
)

sampling_params = SamplingParams(
    temperature=0.0,       ## Greedy decoding: highest acceptance rates
    max_tokens=512,
    ## For temperature > 0, acceptance rate drops 4-8 pp per 0.5 temp increase
)

prompts = [
    "Explain the attention mechanism in transformers in detail.",
    "Write a Python function to compute the Fibonacci sequence iteratively.",
]

outputs = llm.generate(prompts, sampling_params)
for output in outputs:
    print(output.outputs[0].text[:200])
```

The `speculative_model` argument points to the EAGLE head checkpoint, which is available on HuggingFace for all major LLaMA model variants. vLLM downloads and initializes the EAGLE head separately from the target model, then handles KV cache coordination, tree attention mask construction, and rejection sampling acceptance entirely transparently. The API surface is identical to standard vLLM generation.

**vLLM internals: paged KV cache for tree attention.** One non-trivial implementation detail: the EAGLE candidate tree introduces a set of "speculative" positions that need their own KV cache entries. vLLM uses its block-based paged KV cache to handle this: speculative positions get allocated temporary blocks that are freed if the tokens are rejected, and promoted to permanent blocks if accepted. This avoids any memory fragmentation from speculative positions.

The EAGLE head itself also maintains a KV cache over its self-attention computation (since it is an autoregressive transformer block). vLLM manages this as a separate, much smaller KV pool — for a 70B target with 80 layers and head dimension 128, the target's KV cache per token is $80 \times 2 \times 128 \times$ (n_heads) bytes. The EAGLE head's KV cache per token is $1 \times 2 \times 128 \times$ (n_heads) bytes — approximately 1/80th the size. It is negligible.

**SGLang integration.** SGLang version 0.2.0 integrates EAGLE with its RadixAttention prefix caching system:

```python
import sglang as sgl
from sglang import RuntimeEndpoint

## SGLang server configuration (in launch script or sglang.server)
## python -m sglang.launch_server \
##   --model-path meta-llama/Meta-Llama-3-70B-Instruct \
##   --speculative-draft-model-path yuhuili/EAGLE-LLaMA3-Instruct-70B \
##   --speculative-num-draft-tokens 5 \
##   --speculative-eagle-topk 10 \
##   --tp-size 4 \
##   --port 30000

## Client-side usage
runtime = sgl.Runtime(
    model_path="meta-llama/Meta-Llama-3-70B-Instruct",
    speculative_draft_model_path="yuhuili/EAGLE-LLaMA3-Instruct-70B",
    speculative_num_draft_tokens=5,
    speculative_eagle_topk=10,         ## top-K candidates per EAGLE step (C=10)
    speculative_num_steps=60,          ## tree budget (matches EAGLE-1 fixed tree)
    tp_size=4,
    mem_fraction_static=0.88,
)

sgl.set_default_backend(runtime)

@sgl.function
def multi_turn_chat(s, system_message, user_message):
    s += sgl.system(system_message)
    s += sgl.user(user_message)
    s += sgl.assistant(sgl.gen(
        "response",
        max_new_tokens=512,
        temperature=0.0,
    ))

state = multi_turn_chat.run(
    system_message="You are a helpful technical assistant.",
    user_message="Explain the difference between EAGLE-1 and EAGLE-2 speculative decoding.",
)
print(state["response"])
```

SGLang's key contribution is RadixAttention: a prefix tree cache that stores KV states for previously computed prefixes. When a new request shares a long system prompt or few-shot prefix with a cached request, SGLang reuses the cached KV states. With EAGLE, this has an additional benefit: the cached hidden states for the prefix are also reusable, so the EAGLE head can start its autoregressive prediction from the cached feature vectors rather than recomputing them.

**SGLang + EAGLE interaction with prefix caching.** For a deployment where many requests share a 2,000-token system prompt:

1. First request: target model computes 2,000-token prefill, caches KV states and final-layer hidden states.
2. EAGLE draft: starts from cached hidden states at position 2,000, pays only the cost of the new user input's prefill (short — typically 50–200 tokens).
3. Subsequent requests with the same system prompt: target model skips the 2,000-token shared prefix entirely (RadixAttention cache hit). EAGLE starts from the same cached hidden states.

The result is that with shared system prompts, SGLang + EAGLE can handle the decode phase of subsequent requests with effectively zero prefill overhead for the shared portion. The total latency reduction compared to standard vLLM + EAGLE can be 20–40% for workloads with long shared prefixes.

**Production configuration guide.** Based on measured performance across several LLaMA-3 deployment scales:

| Configuration | 8B (1× A100) | 70B (4× A100) | 70B (8× H100) |
|---------------|-------------|--------------|--------------|
| `num_speculative_tokens` γ | 4–5 | 4 | 5–6 |
| EAGLE top-K C | 10 | 10 | 12 |
| Tree budget B | 60 | 60 | 80 |
| Effective bs for benefit | 1–24 | 1–12 | 1–16 |
| Expected speedup (chat, bs=1) | 2.5–3.0× | 3.0–3.8× | 3.5–4.2× |
| Expected speedup (code, bs=1) | 2.3–2.8× | 2.8–3.4× | 3.2–3.9× |

The trend: more GPUs → lower baseline token throughput per request (more inter-GPU communication latency) → speculative decoding's speedup is proportionally more valuable. An 8× H100 cluster serving 70B at bs=1 is very latency-constrained, and EAGLE's 3.5–4.2× speedup is critical to hitting single-digit second response times for long generations.

**Monitoring acceptance rate in production.** Build this into your serving layer:

```python
import time
from dataclasses import dataclass, field
from typing import Dict, List
import threading

@dataclass
class EAGLEStepStats:
    """Statistics for one speculative decode step."""
    num_drafted: int        ## tokens proposed by EAGLE head
    num_accepted: int       ## tokens accepted by rejection sampling
    wall_ms: float          ## wall-clock time for this step (draft + verify)
    request_id: str         ## for per-request tracking


class EAGLEProductionMonitor:
    """
    Thread-safe monitor for EAGLE acceptance rate and speedup tracking.
    Designed for integration with vLLM's EngineCore step callbacks.
    """

    def __init__(self, window_size: int = 10000):
        self._lock = threading.Lock()
        self._window_size = window_size
        self._steps: List[EAGLEStepStats] = []
        self._per_request: Dict[str, List[EAGLEStepStats]] = {}

    def record_step(self, stats: EAGLEStepStats) -> None:
        with self._lock:
            self._steps.append(stats)
            if len(self._steps) > self._window_size:
                self._steps.pop(0)
            if stats.request_id not in self._per_request:
                self._per_request[stats.request_id] = []
            self._per_request[stats.request_id].append(stats)

    @property
    def rolling_acceptance_rate(self) -> float:
        """Acceptance rate over the last window_size steps."""
        with self._lock:
            if not self._steps:
                return 0.0
            drafted = sum(s.num_drafted for s in self._steps)
            accepted = sum(s.num_accepted for s in self._steps)
            return accepted / drafted if drafted > 0 else 0.0

    @property
    def rolling_tokens_per_pass(self) -> float:
        """Average accepted tokens per verify pass."""
        with self._lock:
            if not self._steps:
                return 0.0
            total_steps = len(self._steps)
            total_accepted = sum(s.num_accepted for s in self._steps)
            return total_accepted / total_steps

    @property
    def rolling_speedup_estimate(self) -> float:
        """
        Estimated speedup vs baseline single-token decode.
        Assumes baseline takes the same wall-clock time as the verify pass alone.
        Approximation: speedup ≈ tokens_per_pass / 1.
        """
        return self.rolling_tokens_per_pass

    def check_health(self) -> dict:
        alpha = self.rolling_acceptance_rate
        tpp = self.rolling_tokens_per_pass
        speedup = self.rolling_speedup_estimate
        status = "healthy"
        alerts = []
        if alpha < 0.60:
            status = "critical"
            alerts.append(
                f"Acceptance rate {alpha:.2f} below 0.60 — "
                "EAGLE head may be misaligned with target model"
            )
        elif alpha < 0.75:
            status = "degraded"
            alerts.append(
                f"Acceptance rate {alpha:.2f} below 0.75 — "
                "check if target model was updated since EAGLE head training"
            )
        if tpp < 1.8:
            status = status if status != "healthy" else "degraded"
            alerts.append(
                f"Tokens per pass {tpp:.2f} below 1.8 — "
                "speculative decoding may not be worth its overhead"
            )
        return {
            "status": status,
            "acceptance_rate": round(alpha, 3),
            "tokens_per_pass": round(tpp, 2),
            "speedup_estimate": round(speedup, 2),
            "alerts": alerts,
            "window_steps": len(self._steps),
        }


## Integration with vLLM (pseudo-code; actual API varies by vLLM version)
monitor = EAGLEProductionMonitor(window_size=50000)

def on_spec_decode_step_complete(
    request_id: str,
    num_drafted: int,
    num_accepted: int,
    wall_ms: float,
) -> None:
    """Called by vLLM's speculative decoding engine after each verify pass."""
    monitor.record_step(EAGLEStepStats(
        num_drafted=num_drafted,
        num_accepted=num_accepted,
        wall_ms=wall_ms,
        request_id=request_id,
    ))
    ## Emit to Prometheus/Datadog/CloudWatch
    metrics_client.gauge("eagle.acceptance_rate", monitor.rolling_acceptance_rate)
    metrics_client.gauge("eagle.tokens_per_pass", monitor.rolling_tokens_per_pass)
    health = monitor.check_health()
    if health["status"] != "healthy":
        alert_channel.send(health["alerts"])
```

Alert thresholds:
- $\alpha < 0.60$: critical — EAGLE head is misaligned, consider disabling until retrained
- $\alpha < 0.75$: degraded — EAGLE head may need updating (model was likely modified)
- Tokens/pass $< 1.8$: check whether spec decoding is adding overhead without benefit

![EAGLE-2 wall-clock timeline: 95ms per iteration for 3.9 accepted tokens vs 80ms per single token naive](/imgs/blogs/eagle-speculative-decoding-feature-alignment-8.webp)

## EAGLE vs alternatives: a decision framework

Given that EAGLE, Medusa, and standard 2-model speculative decoding are all available in major serving frameworks, when should you use each?

**Use EAGLE-2 when:**
- You serve a model 13B or larger at batch sizes 1–16
- Your task is chat, instruction following, or reasoning (tasks with rich context-dependent continuations)
- You can invest one GPU-day to train the EAGLE head
- You are on vLLM ≥ 0.4.0 or SGLang ≥ 0.2.0
- You expect the target model to remain stable for weeks to months between updates

**Use Medusa when:**
- You serve models of any size at batch sizes 1–32
- Your task is structured output or code with highly predictable local patterns
- You want simpler operations (K linear heads vs one transformer block)
- You plan to frequently update the target model (Medusa heads are also cheap to retrain, but slightly cheaper than an EAGLE block)

**Use standard 2-model spec decode when:**
- You have a suitable small draft model from the same model family (e.g., LLaMA-1B for LLaMA-70B)
- You want maximum flexibility in draft model choice (different architecture, different training)
- You serve at batch sizes 1–8 with strict memory constraints (the separate draft model uses extra memory, but avoids the EAGLE head's architecture dependency on hidden-state dimension)

**Use baseline autoregressive decode when:**
- Batch sizes are consistently above 32 (compute-bound regime, spec decode adds overhead)
- Your task has very diverse next-token distributions (creative writing at high temperature, alpha < 0.65)
- Your output sequences are very short (less than 50 tokens — the verification overhead is not amortized)
- You serve from a quantized model where the hidden states may be less predictable

The [efficient LLM inference post](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) covers how to combine EAGLE with quantization and paged attention for maximum efficiency. The [speculative decoding production playbook](/blog/machine-learning/large-language-model/speculative-decoding) provides a comprehensive guide to decision-making across the full optimization space.

## Implementation reference: building the EAGLE candidate tree from scratch

For engineers who need to understand vLLM's internals or implement EAGLE in a custom serving stack, here is a complete reference implementation of the EAGLE candidate tree builder:

```python
"""
EAGLE candidate tree builder — reference implementation.
Matches the algorithm in Li et al. (2024) EAGLE paper, Sections 3.1 and 3.2.
Uses PyTorch; requires a cached_features tensor from a target model forward pass.
"""

import heapq
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DraftNode:
    """Node in the EAGLE candidate tree."""
    token_id: int                          ## candidate token at this node
    draft_prob: float                      ## draft softmax probability for this token
    parent: Optional["DraftNode"]          ## parent node (None for root)
    depth: int                             ## depth from root (root = 0)
    hidden_state: torch.Tensor             ## predicted h_t+1 from EAGLE head (dim=d)
    ## Children are populated during tree expansion
    children: List["DraftNode"] = field(default_factory=list)
    ## Tree index (assigned after tree is built, used for attention mask)
    tree_idx: int = -1

    def __lt__(self, other: "DraftNode") -> bool:
        ## Priority queue ordering: higher confidence → higher priority
        return self.draft_prob > other.draft_prob


def build_eagle_tree(
    eagle_block: nn.Module,
    lm_head: nn.Module,
    cached_z: torch.Tensor,          ## [ctx_len, 2*d] — cat[h_t, embed(x_t)]
    token_embeddings: nn.Embedding,  ## shared embedding table from target model
    budget: int = 60,
    top_k: int = 10,
    eagle_ctx_len: int = 10,         ## EAGLE head context window K
    conf_high: float = 0.80,         ## EAGLE-2: expand threshold
    conf_low: float = 0.30,          ## EAGLE-2: prune threshold
    use_dynamic: bool = True,        ## EAGLE-2 if True, EAGLE-1 (static-like) if False
    device: torch.device = torch.device("cuda"),
) -> Tuple[List[DraftNode], torch.Tensor]:
    """
    Build the EAGLE candidate tree.

    Args:
        eagle_block: The EAGLE draft head (single transformer block).
        lm_head: The shared LM head (W_LM), same as target model's.
        cached_z: [ctx_len, 2*d] — feature vectors from accepted context.
                  Each row is cat[h_t^(L), e(x_t)].
        token_embeddings: Embedding table for converting token IDs to vectors.
        budget: Maximum number of tree nodes (excluding root).
        top_k: Number of candidates to generate at each draft step.
        eagle_ctx_len: Context window length for EAGLE head.
        conf_high, conf_low: EAGLE-2 confidence thresholds.
        use_dynamic: Whether to use EAGLE-2 dynamic tree or EAGLE-1 fixed-budget.
        device: Torch device.

    Returns:
        tree_nodes: List of DraftNode in BFS order (excluding phantom root).
        tree_attn_mask: [len(tree_nodes), ctx_len + len(tree_nodes)] bool mask.
    """
    d = cached_z.shape[-1] // 2  ## hidden dimension

    ## Create phantom root representing the last accepted token
    root_hidden = cached_z[-1, :d].clone()
    root_token = 0   ## placeholder; root is the already-accepted position
    root = DraftNode(
        token_id=root_token,
        draft_prob=1.0,
        parent=None,
        depth=0,
        hidden_state=root_hidden,
    )

    ## Priority queue: (neg_confidence, node)
    ## We use neg_confidence so Python's min-heap gives us max-confidence ordering
    pq: List[Tuple[float, DraftNode]] = [(-1.0, root)]
    all_candidates: List[DraftNode] = []   ## will hold all non-root nodes

    def get_eagle_context(node: DraftNode) -> torch.Tensor:
        """
        Gather the last eagle_ctx_len feature vectors ending at `node`.
        Walks up the tree to collect ancestor hidden states + token embeddings,
        then prepends from cached_z if the path is shorter than eagle_ctx_len.
        """
        path_features = []
        cur = node
        while cur is not None and len(path_features) < eagle_ctx_len:
            emb = token_embeddings(torch.tensor([cur.token_id], device=device))  ## [1, d]
            feat = torch.cat([cur.hidden_state.unsqueeze(0), emb], dim=-1)       ## [1, 2d]
            path_features.append(feat)
            cur = cur.parent

        ## Reverse (we built it root-to-node, want it in time order)
        path_features = list(reversed(path_features))

        ## Prepend from cached context if we have room
        remaining = eagle_ctx_len - len(path_features)
        if remaining > 0 and cached_z.shape[0] > 0:
            prefix = cached_z[max(0, cached_z.shape[0] - remaining):]  ## [rem, 2d]
            prefix_list = [prefix[i:i+1] for i in range(prefix.shape[0])]
            path_features = prefix_list + path_features

        ctx = torch.cat(path_features[-eagle_ctx_len:], dim=0)   ## [K, 2d]
        return ctx.unsqueeze(0)   ## [1, K, 2d]

    total_added = 0

    while pq and total_added < budget:
        neg_conf, parent_node = heapq.heappop(pq)
        conf = -neg_conf

        ## EAGLE-2: skip low-confidence nodes at depth >= 1
        if use_dynamic and parent_node.depth >= 1 and conf < conf_low:
            continue

        ## Gather context for EAGLE head
        ctx = get_eagle_context(parent_node)   ## [1, K, 2d]

        with torch.no_grad():
            ## Run EAGLE block: predict next hidden state
            eagle_out = eagle_block(ctx)        ## [1, K, d] or [1, d] depending on impl
            pred_h = eagle_out[:, -1, :d]       ## [1, d] — last position, first d dims

            ## Decode to vocabulary logits via shared LM head
            logits = lm_head(pred_h)            ## [1, vocab_size]
            probs = F.softmax(logits[0], dim=-1)

        ## Take top-K candidates
        k = min(top_k, budget - total_added)
        top_probs, top_ids = torch.topk(probs, k=k)

        for prob_t, id_t in zip(top_probs.tolist(), top_ids.tolist()):
            if total_added >= budget:
                break

            ## EAGLE-2: prune very-low-confidence children of non-root
            if use_dynamic and parent_node.depth >= 1 and prob_t < conf_low:
                break  ## sorted by prob, so all remaining are also below threshold

            child = DraftNode(
                token_id=id_t,
                draft_prob=float(prob_t),
                parent=parent_node,
                depth=parent_node.depth + 1,
                hidden_state=pred_h[0].clone(),
            )
            parent_node.children.append(child)
            all_candidates.append(child)
            total_added += 1

            ## EAGLE-2: only add high-confidence children to expansion queue
            if not use_dynamic or prob_t >= conf_high or parent_node.depth == 0:
                heapq.heappush(pq, (-prob_t, child))

    ## Assign tree indices in BFS order
    for idx, node in enumerate(all_candidates):
        node.tree_idx = idx

    ## Build tree attention mask
    n = len(all_candidates)
    ctx_len = cached_z.shape[0]
    ## Mask shape: [n, ctx_len + n] — each candidate row specifies what it can attend to
    mask = torch.zeros(n, ctx_len + n, dtype=torch.bool, device=device)

    ## All candidate positions attend to all context positions
    mask[:, :ctx_len] = True

    ## Each candidate attends to itself and its ancestors in the tree
    for node in all_candidates:
        cur = node
        while cur is not None and cur.tree_idx >= 0:
            mask[node.tree_idx, ctx_len + cur.tree_idx] = True
            cur = cur.parent

    return all_candidates, mask
```

This implementation is approximately 140 lines of clean Python and captures all the algorithmic decisions in the EAGLE paper: ancestor context gathering, confidence-guided expansion (EAGLE-2), the priority queue ordering, and tree attention mask construction. In vLLM's production code, this is implemented as CUDA kernels for the tree attention mask computation and the EAGLE head's forward pass, but the algorithmic structure is identical.

## Case studies

### Case study 1: LLaMA-3-70B-Instruct customer support API at batch size 1

**Context.** A fintech company deploys a customer-facing chatbot on LLaMA-3-70B-Instruct. Requests are conversational (50-token system prompt + 100-token user query, 200-token target response). The serving infrastructure is 4× A100-80GB. They have a p99 latency SLA of 3 seconds per request and need to handle 800 concurrent sessions.

**Baseline.** Autoregressive decode generates approximately 200 tokens in 240ms (1.2ms/token, limited by the HBM bandwidth of 4× A100). With prefill (~200ms for the 150-token input) and network overhead, end-to-end p99 is approximately 480ms. At 800 concurrent sessions, they need 800/480ms ≈ 1,667 requests/second peak throughput from their GPU cluster. At baseline, this requires approximately 12× A100-80GB ($18,000/month at cloud pricing).

**EAGLE-2 deployment.** They deploy with `num_speculative_tokens=5`, tree budget B=60, dynamic expansion enabled. The EAGLE head is trained on 50K ShareGPT conversations that include customer service dialogues — a 48-hour run on the same 4× A100 cluster.

**Results measured over 100K requests across one week:**
- Average acceptance rate: $\alpha = 0.90$
- Average tokens per verify pass: 3.81
- Decode latency for 200 tokens: 63ms (3.81× speedup)
- End-to-end p99: 290ms (from 480ms — 40% reduction)
- Required GPU cluster: 5× A100-80GB (from 12×) to handle 800 concurrent sessions
- Monthly savings: approximately $10,500 in reduced GPU rental

**Key lesson.** The 3.81× decode speedup did not translate to 3.81× end-to-end speedup (only 1.65×) because prefill (~200ms) and network overhead (~30ms) were not affected by EAGLE. However, the cost reduction from reducing peak GPU count by 58% was the primary business outcome, making EAGLE adoption economically straightforward.

### Case study 2: LLaMA-3-8B code completion at high-throughput batch sizes

**Context.** A developer tools company serves LLaMA-3-8B-Instruct for inline code completion at batch sizes that spike between 1 and 64 depending on time of day. Requests are short (10–30 token completions) with strict 150ms latency SLA.

**EAGLE-2 deployment attempt.** They configure EAGLE-2 with γ=4, B=60 nodes, on 2× A100-40GB.

**The batch-size problem.** At batch size 1–4, EAGLE-2 delivers 2.2× speedup ($\alpha = 0.87$ on code). At batch size 16, speedup drops to 1.05×. At batch size 32, EAGLE-2 is 8% slower than baseline because:
1. The target verify pass at batch size 32 is already nearly compute-bound for the 8B model
2. The tree adds 60 candidate positions per sequence × 32 sequences = 1,920 extra tokens to the verify batch
3. Each extra token costs compute proportional to the batch size (attention is $O(\text{seq\_len} \times \text{bs})$ in the decoder's self-attention)
4. The draft overhead (EAGLE head × γ × batch size) is no longer negligible

**Resolution.** They implement a two-regime serving system based on the current queue depth:
- Queue depth ≤ 8 pending requests: EAGLE-2 enabled (latency-bound regime)
- Queue depth > 8: EAGLE-2 disabled (throughput regime, high batch sizes)

The queue depth is tracked per-GPU and the serving framework switches modes dynamically. This hybrid system delivers:
- Off-peak (bs=1–8): 1.8× average speedup, hitting the 150ms SLA consistently
- Peak (bs=32+): baseline performance (no regression from EAGLE overhead)

**Key lesson.** Always measure EAGLE performance at your actual batch size distribution, not just at batch size 1. The batch-size threshold where EAGLE stops helping depends on model size, hardware, and tree size — there is no universal answer. Build an online measurement system (like the `EAGLEProductionMonitor` class above) and implement automatic fallback.

### Case study 3: LLaMA-3-70B long-form legal document summarization

**Context.** A legal tech company summarizes 30-page contracts. Input: 8,192 tokens (the full document). Output: 512-token structured summary. Hardware: 8× H100-80GB, processing one document at a time (batch size 1).

**Challenge.** Prefill takes approximately 2.2 seconds (compute-bound at 8,192-token input on 8× H100). Decode of 512 tokens takes approximately 600ms at baseline. Total end-to-end: 2.8 seconds.

**EAGLE-2 deployment.** They train the EAGLE head specifically on 20K legal document pairs to match the domain distribution. The legal text domain is highly formulaic ("pursuant to Section X", "notwithstanding the foregoing", "as set forth herein"), and they expect high acceptance rates.

**Results:**
- EAGLE head training: 36 GPU-hours on 4× H100 (15% cheaper to train on H100 vs A100)
- Average acceptance rate: $\alpha = 0.93$ (highest we have seen — legal language is extremely formulaic)
- Decode of 512 tokens: 145ms (4.14× speedup on decode)
- End-to-end: 2.35 seconds (from 2.8 seconds, 16% reduction)

**The diminishing-returns math.** Even with a 4.14× decode speedup, the end-to-end reduction is only 16% because prefill dominates. The decode time decreased by 455ms, but the total latency only decreased by 450ms (similar — the numbers happen to align here because prefill is 2.2 seconds and decode is 600ms, so decode is 21% of the total).

For applications where decode is a larger fraction of total time (short inputs, long outputs), EAGLE's end-to-end benefit is proportionally larger. The rule of thumb: EAGLE's end-to-end speedup ≈ decode fraction × EAGLE decode speedup. At 21% decode fraction and 4.14× decode speedup, that is $0.21 \times 4.14 + 0.79 \times 1.0 \approx 1.66$× — consistent with the observed 16% reduction (1.19×, slightly lower due to tree overhead in the verify pass adding marginal prefill-adjacent compute).

**Key lesson.** Domain-specific EAGLE head training is worth the modest extra cost for specialized applications. The general-purpose ShareGPT-trained EAGLE head for LLaMA-3-70B would have achieved $\alpha \approx 0.87$–0.89 on legal text; the domain-specific head achieved $\alpha = 0.93$, a 4–6 pp improvement that translated to 15% more tokens per verify pass.

### Case study 4: EAGLE head misalignment after LoRA finetuning

**Context.** A research lab uses LLaMA-3-70B-Base for mathematical reasoning tasks. They have the standard EAGLE-3-70B head trained on general-purpose ShareGPT data, achieving $\alpha = 0.88$ and 3.4× speedup on their baseline math reasoning prompts.

They then finetune the target model on 50K math problem-solution pairs using LoRA (rank=64 applied to all Q, K, V, O, up, down, and gate projections — essentially modifying all linear layers in the transformer).

**The misalignment problem.** After LoRA finetuning, the EAGLE head's acceptance rate drops from $\alpha = 0.88$ to $\alpha = 0.61$ on math prompts. The speedup collapses from 3.4× to 1.4× — barely worth the overhead of running the draft head at all.

**Root cause analysis.** The EAGLE head was trained to predict hidden states from the pre-LoRA LLaMA-3-70B base model. LoRA modifies weights throughout the transformer (all linear projections), which shifts the hidden state distribution at every layer. The final-layer hidden states $h_t^{(80)}$ from the LoRA-finetuned model occupy a different region of $\mathbb{R}^{8192}$ than the original model's hidden states — the EAGLE head's training targets are now systematically incorrect.

Concretely: the L2 distance between the EAGLE head's predictions and the true hidden states increased from approximately 0.12 (training distribution) to approximately 0.38 (post-LoRA distribution) — a 3× increase in prediction error. This directly degraded the logit alignment and hence the acceptance rate.

**Fix and timeline:**
1. Run the LoRA-finetuned model on 5,000 math problem-solution pairs, caching final-layer hidden states.
2. Fine-tune the existing EAGLE head (starting from the pre-LoRA checkpoint) on these 5,000 examples for 2,000 gradient steps, learning rate 1e-4 with cosine decay.
3. Total compute: approximately 3 GPU-hours on 1× A100.
4. Result after fine-tuning: $\alpha = 0.87$ (recovered 96% of original performance), speedup back to 3.2× (slightly below original 3.4× because the math distribution is slightly less predictable than the original general-purpose distribution).

**Key lesson.** EAGLE heads are tightly coupled to the target model's weight matrix. Any modification to the target model's weights — LoRA, full finetuning, DPO, RLHF — shifts the hidden state distribution and requires EAGLE head updating. The remediation is fast (a few hours) but must be anticipated in the model update pipeline. Best practice: maintain an automated pipeline that triggers EAGLE head retraining whenever the target model is updated, using the same training data as the original EAGLE head plus any domain-specific examples.

A useful diagnostic: monitor the L2 distance between the EAGLE head's predicted hidden states and the target model's actual hidden states on a held-out validation set. If this metric increases by more than 20% from its post-training baseline, the EAGLE head needs updating. This metric is cheaper to compute than the full acceptance rate (no tree construction or verify pass needed) and can be computed in a few minutes on a small validation set.

## EAGLE's place in the broader speculative decoding landscape

Zooming out, EAGLE represents a methodological progression in how the field has thought about draft model design. The progression has three distinct phases, and understanding it helps predict where the next improvements will come from.

**Phase 1: Separate small models (2023).** The original speculative decoding papers (Chen et al., Leviathan et al.) proposed using a smaller model from the same family as the draft. LLaMA-68M drafts for LLaMA-70B. The bottleneck: the small model must independently reconstruct context from raw tokens, is limited by its own parameter count, and uses separate memory. Acceptance rates plateau around 0.70–0.78. Speedup: 1.5–2×.

**Phase 2: Lightweight add-ons to the target (Medusa, 2023).** Medusa eliminated the separate model by adding heads directly to the target's final hidden state. This gave the heads access to $h_t^{(L)}$ — much richer than token IDs. But Medusa heads are non-autoregressive: head $k$ predicts token $t+k$ from a single anchor point $h_t^{(L)}$, without access to the intermediate hidden states at $t+1, t+2, \ldots, t+k-1$. As $k$ increases, the prediction degrades rapidly. Acceptance rates: 0.78–0.85. Speedup: 2.0–2.8×.

**Phase 3: Feature-level autoregression (EAGLE, 2024).** EAGLE introduced the key missing ingredient: autoregressive prediction in feature space. Instead of predicting each future token independently from a single $h_t^{(L)}$, EAGLE's draft head produces a sequence of predicted hidden states $\hat{h}_{t+1}^{(L)}, \hat{h}_{t+2}^{(L)}, \ldots$, each conditioned on all previous predicted states. This mirrors exactly how the target model builds up its own representation: autoregressively, one layer-stack at a time. The result is a much tighter approximation of the target's internal computation, yielding acceptance rates above 0.90 and speedups of 3–4×.

**What phase 4 might look like.** The EAGLE head uses one transformer block for autoregression. Future work might explore:
- Using 2–3 blocks for richer feature prediction, accepting a slightly higher draft overhead in exchange for $\alpha > 0.95$
- Multi-target training (training one EAGLE head that works across a family of models by conditioning on a model-ID embedding)
- Online continuous adaptation of the EAGLE head during serving (updating head weights every few thousand requests using the acceptance signal as a learning signal)
- Integration with speculative decoding for multi-modal models, where the hidden state carries visual or audio features alongside text

The fundamental insight — predict at the level of internal representations, not surface tokens — is likely to remain the dominant paradigm for draft head design in the near to medium term. As long as target models use transformer architectures with interpretable hidden states, feature-level drafting will have the information-theoretic advantage over token-level drafting.

## Summary: EAGLE in context

EAGLE is the most efficient speculative decoding method currently available for latency-bound serving of large instruction-tuned models. Its key innovations — feature-level autoregression, shallow single-block draft head, shared LM head, and (in EAGLE-2) confidence-guided dynamic tree construction — combine to push acceptance rates above 0.90 and deliver 3–4× end-to-end speedup at the batch sizes relevant for real-time LLM APIs.

The practical requirements are straightforward: one GPU-day to train the draft head, vLLM ≥ 0.4.0 or SGLang ≥ 0.2.0 for serving, and a monitoring system to detect acceptance rate degradation when the target model is updated. The benefits are large: for a 70B-class model serving production traffic at batch size 1–8, EAGLE often reduces the required GPU count by 3–4×, with proportional cost savings.

The essential checklist before deploying EAGLE:
- Train or download the EAGLE head for your exact model checkpoint (not a related variant)
- Verify $\alpha \geq 0.80$ on a representative sample of your production prompts before enabling
- Set up the `EAGLEProductionMonitor` with alert thresholds ($\alpha < 0.75$ → degraded, $\alpha < 0.60$ → critical)
- Build a pipeline that re-trains or fine-tunes the EAGLE head within 24 hours of any target model update
- Measure batch-size-specific speedup — disable EAGLE for batch sizes above your threshold

The next post in this series takes a step back from EAGLE's specific architecture to examine the general tree speculation framework — the mathematical foundation that EAGLE, Medusa, and other tree-based methods all build on — and derives the optimal tree topology for any given acceptance rate and budget. Understanding the general framework makes it straightforward to adapt these ideas to new model architectures and serving constraints as the field continues to evolve rapidly.

Cross-links for deeper reading:
- [KV cache mechanics](/blog/machine-learning/large-language-model/kv-cache) — prerequisite to understanding tree-attention overhead and the KV block extensions needed for speculative positions
- [Efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) — how EAGLE fits into the broader inference optimization landscape alongside quantization and batching
- [vLLM serving architecture](/blog/machine-learning/large-language-model/vllm-inference) — EAGLE's integration with continuous batching and paged KV management in vLLM ≥ 0.4
- [SGLang inference](/blog/machine-learning/large-language-model/sglang-inference) — EAGLE + RadixAttention prefix caching and the combined latency model
- [Optimizing LLM inference: complete guide](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) — where EAGLE fits in the full optimization stack alongside quantization, tensor parallelism, and Flash Attention

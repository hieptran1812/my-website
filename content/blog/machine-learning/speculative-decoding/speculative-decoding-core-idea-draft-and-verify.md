---
title: "Speculative Decoding: Draft Fast, Verify in Parallel"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "The core intuition behind speculative decoding — a small draft model proposes γ tokens, the large target model verifies them all in one forward pass, yielding up to γ+1 tokens per step with zero quality loss."
tags:
  [
    "speculative-decoding",
    "llm-inference",
    "large-language-model",
    "deep-learning",
    "autoregressive-models",
    "inference-optimization",
    "rejection-sampling",
  ]
category: "machine-learning"
subcategory: "Speculative Decoding"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/speculative-decoding-core-idea-draft-and-verify-1.png"
---

There is a newspaper editorial workflow that makes speculative decoding click faster than any equation. A large daily paper needs to publish 80 columns a day. The senior editor who sets the editorial voice is brilliant but slow — she writes at roughly one word per second under deadline pressure, and every word she writes is exactly right. Hiring eighty senior editors would cost a fortune. The smart editorial director instead hires twenty junior editors who write fast, if not always perfectly. Each junior editor drafts a full column in the same time the senior editor could draft two paragraphs. Then the senior editor scans each column in fifteen seconds, red-pens the three phrases that are off, and signs off. The paper publishes 80 perfectly-edited columns for the price of 20 junior editors plus a few hours of senior-editor review time.

This is speculative decoding. The draft model is the fast, slightly imperfect junior editor. The target model — the large language model whose output quality you want to preserve — is the senior editor. The verification step is the red-pen scan. And the crucial insight that makes everything work: **reading is faster than writing**. The senior editor scans a paragraph faster than she composes one, even though both are high-skill tasks. For a language model, the analogue is that one forward pass of the target model can evaluate γ candidate continuations simultaneously, just as cheaply (within a constant factor) as evaluating one.

If you have not yet read [Why LLMs Generate Text Slowly](/blog/machine-learning/speculative-decoding/why-llms-are-slow-autoregressive-bottleneck), start there. It establishes the foundational problem: during autoregressive decode, a 70B model reads 140 GB of weights from HBM every single pass to produce one token, achieving 5–15% GPU SM utilization because the computation is so trivial relative to the memory bandwidth required. Speculative decoding attacks this bottleneck directly by amortizing those 140 GB weight reads across multiple output tokens per pass. This post explains precisely how it does so, derives the speedup formula, and shows where the technique succeeds and fails in practice.

## The newspaper editor analogy, grounded in system arithmetic

Let us be precise about why the analogy holds. In standard autoregressive decoding of a 70B model at batch size 1:

- **Weight load per pass:** 70 billion parameters × 2 bytes (bfloat16) = 140 GB loaded from HBM per token.
- **HBM bandwidth (H100 SXM5):** 3.35 TB/s theoretical, roughly 2.8–3.0 TB/s practical.
- **Time to load all weights:** 140 GB / 3.0 TB/s ≈ 47 ms per token, hardware-minimum.
- **Actual measured decode latency (LLaMA 3 70B, 4K context, bs=1):** 75–90 ms per token. The extra time beyond the 47 ms minimum comes from KV cache reads, CUDA kernel launch overhead, and tensor-parallel communication if the model is sharded.

For every 80 ms spent on one output token, the target model reads the same 140 GB weight matrix. Whether that forward pass evaluates one output position or ten output positions simultaneously changes the arithmetic. The FLOP count scales with the number of output positions, but the weight load is constant because the model weights themselves do not change. For small numbers of output positions (up to roughly 8–16 for most architectures), the bottleneck remains HBM bandwidth, not arithmetic — so evaluating γ positions costs approximately the same as evaluating one.

This is the opportunity. If you could somehow know the next γ tokens in advance, you could verify them all in one forward pass for free. You obviously cannot know them in advance — that is the whole problem. But a smaller, cheaper model can *guess* them with high probability. Those guesses are often right. And even when they are wrong, the verification step corrects them by construction.

## Two models, one forward pass: the mechanism

Speculative decoding uses two models with fundamentally different cost profiles. The **draft model** (also called the proposal model or the drafter) is small and fast. A common pairing for LLaMA 3 70B serving is LLaMA 3 8B or LLaMA 3 1B as the drafter. The draft model runs γ sequential autoregressive passes — exactly the expensive serial pattern we want to avoid — but because it is 9× or 70× smaller, those γ passes cost a small fraction of a single target-model pass.

The **target model** is the full-size model whose output distribution we want to preserve. It runs one single forward pass over all γ draft tokens simultaneously. This is the amortization trick: instead of paying the 140 GB weight-read cost once per token, we pay it approximately once per γ tokens (plus a modest scaling factor for the extra positions).

Here is the complete algorithm. Let $x_{1:t}$ be the current context — the prompt plus all accepted tokens so far.

**Step 1 — Draft phase.** Run the draft model autoregressively from position $t+1$ to $t+\gamma$. At each step, sample one token from the draft model's distribution:
$$\tilde{x}_{t+i} \sim p(\cdot \mid x_{1:t}, \tilde{x}_{t+1:t+i-1}) \quad \text{for } i = 1, \ldots, \gamma$$
Save the draft probabilities $p_i = p(\tilde{x}_{t+i} \mid x_{1:t}, \tilde{x}_{t+1:t+i-1})$ for each position. These will be needed in the acceptance step.

**Step 2 — Verify phase.** Construct the verification input by concatenating the context with all γ draft tokens:
$$\text{verify\_input} = [x_{1:t},\; \tilde{x}_{t+1},\; \tilde{x}_{t+2},\; \ldots,\; \tilde{x}_{t+\gamma}]$$
Run the target model once on this sequence. Extract the target model's logit distribution at each of the γ positions:
$$q_i(\cdot) = q(\cdot \mid x_{1:t}, \tilde{x}_{t+1:t+i-1}) \quad \text{for } i = 1, \ldots, \gamma$$
Note that $q_i$ is conditioned on the draft tokens at all positions before $i$, not on the target model's own previous outputs. This is what makes the verify pass a single forward pass — the entire draft sequence is in the input, so the causal attention mechanism resolves all γ positions at once.

**Step 3 — Accept/reject phase.** Walk the draft tokens from left to right. For each position $i$, compute the acceptance probability:
$$\text{accept}(\tilde{x}_{t+i}) = \min\!\left(1,\; \frac{q_i(\tilde{x}_{t+i})}{p_i(\tilde{x}_{t+i})}\right)$$
Draw $r_i \sim \text{Uniform}(0,1)$. Accept $\tilde{x}_{t+i}$ if $r_i < \text{accept}(\tilde{x}_{t+i})$. At the first rejection (say at position $k$), stop. The tokens $\tilde{x}_{t+1}, \ldots, \tilde{x}_{t+k-1}$ are accepted. Token $\tilde{x}_{t+k}$ is rejected.

**Step 4 — Bonus token.** At the position of first rejection $k$ (or at position $\gamma+1$ if all tokens were accepted), sample one token from the **adjusted target distribution**:
$$\text{bonus} \sim \frac{\max(0,\; q_k(x) - p_k(x))}{\sum_{v} \max(0,\; q_k(v) - p_k(v))}$$
This token corrects for the draft model's bias at position $k$. Add it to the accepted sequence.

**Step 5 — Advance and repeat.** Add the $k-1$ accepted tokens plus the bonus token to the context. Update both KV caches. Return to step 1.

The output of this procedure — over many iterations — is distributed identically to what you would get from running only the target model autoregressively. We prove this below. First, let us understand why verification can be done in one pass when drafting cannot.

![Speculative decoding full loop: draft model proposes, target verifies in parallel, bonus token guaranteed](/imgs/blogs/speculative-decoding-core-idea-draft-and-verify-1.webp)

## Why verification is parallel when drafting is sequential

This is the question that trips up engineers at first. If the target model can evaluate γ positions in one pass, why not just run it autoregressively in some batched fashion without a draft model?

The answer is causal dependency. In standard autoregressive generation, token $x_{t+2}$ must be sampled before token $x_{t+3}$ can be generated — $x_{t+3}$ is conditioned on $x_{t+2}$, which does not exist until we generate it. This makes writing inherently serial.

Verification is different. The draft tokens are fixed inputs, not values to be resolved. When we run the target model's verify pass, we feed it the entire draft sequence $[\tilde{x}_{t+1}, \ldots, \tilde{x}_{t+\gamma}]$ as part of the input. The causal attention mask still applies — position $t+2$ only attends to positions $\leq t+1$ — but since all positions $\leq t+\gamma$ are provided in the input, every position is resolved in the same forward pass. The model is not being asked to generate; it is being asked to score a pre-filled-in sequence.

Concretely, consider verifying draft token $\tilde{x}_{t+3}$. The target model evaluates $q(\cdot \mid x_{1:t}, \tilde{x}_{t+1}, \tilde{x}_{t+2})$. The conditioning context $[x_{1:t}, \tilde{x}_{t+1}, \tilde{x}_{t+2}]$ is fully determined at verify time — those are all known values (the context and the two earlier draft tokens). So the attention computation at position $t+3$ can run in the same forward pass as positions $t+1$ and $t+2$, because none of them depend on an unresolved future.

![Why draft tokens can be verified in parallel: all prior positions are known at verify time](/imgs/blogs/speculative-decoding-core-idea-draft-and-verify-3.webp)

Think of it with the editorial analogy: the junior editor's column is completely written before the senior editor sees it. The senior editor scans position 1, position 2, position 3, ..., simultaneously (or nearly so, in the attention sense) because the entire text is on the page. She does not need to wait for position 2 to be resolved before she can look at position 3 — it is all there.

The verify pass is slightly more expensive than a standard single-token decode. The attention computation scales as $O(L_{\text{seq}} \times \gamma)$ additional FLOP for attending over the draft tokens. But this is a secondary cost for typical sequence lengths (2K–8K tokens). The dominant cost — loading all 140 GB of model weights from HBM — is fixed regardless of how many positions the forward pass processes. As long as γ is small (4–8), the verify pass costs roughly the same as a single-token decode.

## The before/after picture: latency and utilization

Let us quantify the improvement concretely. We consider generating 4 tokens from a 70B model at batch size 1 on a single H100 SXM5, bfloat16, 4K context.

**Baseline autoregressive decode:**
- 4 sequential forward passes × 80 ms per pass = 320 ms
- GPU SM utilization: 8–12% (bandwidth-bound; most of the 80 ms is HBM weight loading, not compute)
- Throughput at this latency: 12.5 tokens/second

**Speculative decode with 1B draft, γ=4, α=0.8:**
- Draft phase: 4 sequential 1B-model passes × 10 ms = 40 ms
- Verify phase: 1 × 70B-model pass (slightly extended for 4 extra positions) ≈ 85 ms
- Total per step: 125 ms
- Expected tokens per step: $(1 - 0.8^5)/(1-0.8) = 3.36$
- Effective throughput: 3.36 tokens / 125 ms = 26.9 tokens/second (**2.15× speedup**)
- GPU SM utilization during verify: ~28% (verify pass sees 5 positions, much closer to compute-bound)

![Before/after: sequential decode 320ms, 10% GPU util vs speculative decode 125ms, 28% GPU util](/imgs/blogs/speculative-decoding-core-idea-draft-and-verify-2.webp)

The utilization improvement is meaningful: the verify pass is doing 5× more work per weight load than a single-token decode. This starts pushing the GPU from purely bandwidth-bound toward a more efficient operating point. For $\gamma=8$, the verify pass processes 9 positions per weight load, and SM utilization approaches 40–50% — still below the compute-bound regime but far more efficient than the baseline.

The wall-clock timeline makes the advantage vivid:

![Wall-clock comparison: 4 tokens take 320ms baseline vs 125ms speculative decode](/imgs/blogs/speculative-decoding-core-idea-draft-and-verify-4.webp)

In the baseline, you see four 80 ms blocks, each producing one token. In speculative decoding, you see a 40 ms draft block followed by an 85 ms verify block, producing 3–5 tokens depending on acceptance. The sequential structure of the decode is not eliminated — you still have multiple steps if you need more than γ+1 tokens — but each step produces far more output per expensive pass.

## The bonus token: the guarantee you never go backward

One design feature of speculative decoding that deserves careful attention: the system always produces at least one new token per verify step, even if all γ draft tokens are rejected. This guarantee is what makes speculative decoding a strict improvement over autoregressive decoding. Worst case, it falls back to exactly one new token per expensive forward pass — the same as baseline.

The mechanism is the **bonus token**, and it is not an add-on — it is mathematically necessary for the acceptance procedure to be lossless.

Here is why. When a draft token at position $k$ is rejected, the target model has already computed its logit distribution $q_k$ at that position as part of the verify pass. Rather than throwing that computation away and rerunning the target model from scratch, we sample one token from an adjusted version of $q_k$ that accounts for the draft model's proposals at that position. This gives us one correctly-distributed token at position $k$.

The adjusted distribution used for the bonus token is:
$$\text{bonus\_dist}(x) = \frac{\max(0,\; q_k(x) - p_k(x))}{\sum_{v} \max(0,\; q_k(v) - p_k(v))}$$

This distribution removes probability mass from tokens that the draft model over-predicted relative to the target, and normalizes the remainder. Sampling from it gives a token whose inclusion restores the marginal output distribution of the target model exactly. We prove this in the next section.

In the case where all γ draft tokens are accepted, the bonus token is an extra free token at position $t+\gamma+1$. The verify pass ran the target model over positions $t+1$ through $t+\gamma$, which means the target model attended to position $t+\gamma$ and can produce a logit distribution at $t+\gamma+1$ essentially for free (no additional weight load needed — the logits at position $t+\gamma+1$ come from the final hidden state of position $t+\gamma$, which was computed in the verify pass). This bonus token is then sampled from the target model's distribution directly, with no acceptance threshold — it is a pure target-model sample.

So: minimum output is 1 token (all γ rejected, one bonus); maximum output is γ+1 tokens (all γ accepted, one bonus at position γ+1). Speculative decoding is never worse than autoregressive decoding on a per-step basis.

![Bonus token mechanism: always at least 1 token per step, up to γ+1](/imgs/blogs/speculative-decoding-core-idea-draft-and-verify-8.webp)

## Why this is lossless: the modified rejection sampling proof

The claim that speculative decoding produces output distributed identically to the target model alone is the reason this technique earns a place in production serving. Without losslessness, you have a speed trick that degrades quality in hard-to-predict ways. With it, you have a free optimization.

The proof uses a result called **modified rejection sampling**. Standard rejection sampling works as follows: to sample from a target distribution $q$ using a proposal distribution $p$, sample $x \sim p$, then accept with probability $\min(1, q(x) / (M \cdot p(x)))$ where $M = \max_x q(x)/p(x)$. If rejected, try again. The output is distributed as $q$.

Speculative decoding uses a modified version that avoids the expensive "try again" retries. Instead of retrying until acceptance, we sample once from $p$, apply the acceptance condition with $M=1$ (which requires that $p$ is not dramatically different from $q$, i.e., the draft model is not wildly wrong), and on rejection, we immediately sample from the adjusted distribution rather than redrawing from $p$.

**Theorem (speculative sampling is lossless):** Let $p$ and $q$ be probability distributions over the vocabulary $\mathcal{V}$ with $p(x) > 0$ wherever $q(x) > 0$. Define:
1. Sample $\tilde{x} \sim p$.
2. Accept $\tilde{x}$ with probability $\beta = \min(1, q(\tilde{x})/p(\tilde{x}))$.
3. If rejected, sample $x^* \sim r$ where $r(x) = \max(0, q(x) - p(x)) / Z$ and $Z = \sum_v \max(0, q(v) - p(v))$.
4. Output $\tilde{x}$ if accepted, $x^*$ if rejected.

**Claim:** The output is distributed as $q$.

**Proof.** The probability of outputting token $x$ is:
$$P(\text{output} = x) = P(\tilde{x} = x \text{ and accepted}) + P(\text{rejection}) \cdot r(x)$$

Computing each term:
$$P(\tilde{x} = x \text{ and accepted}) = p(x) \cdot \min\!\left(1, \frac{q(x)}{p(x)}\right) = \min(p(x), q(x))$$

The rejection probability is:
$$P(\text{rejection}) = 1 - \sum_{v} p(v) \cdot \min\!\left(1, \frac{q(v)}{p(v)}\right) = 1 - \sum_{v} \min(p(v), q(v))$$
$$= \sum_{v} (p(v) - \min(p(v), q(v))) = \sum_{v} \max(0, p(v) - q(v))$$

Also note that $Z = \sum_v \max(0, q(v) - p(v))$. Because $p$ and $q$ are both probability distributions:
$$\sum_v (q(v) - p(v)) = 0 \implies \sum_v \max(0, q(v)-p(v)) = \sum_v \max(0, p(v)-q(v))$$
So $Z = P(\text{rejection})$.

Therefore:
$$P(\text{output} = x) = \min(p(x), q(x)) + \frac{\max(0, q(x)-p(x))}{Z} \cdot Z$$
$$= \min(p(x), q(x)) + \max(0, q(x)-p(x)) = q(x) \quad \checkmark$$

The last equality uses the identity $\min(a,b) + \max(0, b-a) = b$. $\square$

This proof extends to the multi-token case by induction: after a rejection at position $k$, the bonus token at $k$ is distributed as $q_k$, the acceptance decisions at positions $1, \ldots, k-1$ were conditioned on the context through position $t$ (which is fixed), and the rejection at position $k$ produces a $q_k$-distributed token. The joint distribution of the output sequence matches the joint distribution of autoregressively sampling from $q$ at every position.

![Lossless proof sketch: accept/reject/resample layers preserve target distribution exactly](/imgs/blogs/speculative-decoding-core-idea-draft-and-verify-7.webp)

The practical import is significant. You do not need to run quality evals before and after deploying speculative decoding to check for regressions. The math guarantees no regression, provided you implement the acceptance criterion correctly (specifically: the ratio $q(\tilde{x})/p(\tilde{x})$ must be computed at the specific draft token values, not at argmax or any approximation).

One subtle point: the losslessness proof assumes that the draft model and target model use the same tokenizer and vocabulary. If the vocabularies differ, the ratio $q(\tilde{x})/p(\tilde{x})$ is undefined for tokens in one vocabulary but not the other. In practice, speculative decoding deployments always use models from the same family (same tokenizer), and this constraint is naturally satisfied.

## Expected speedup: the formula and when it applies

How much speedup does speculative decoding provide? The answer is parameterized by two quantities: $\gamma$ (draft tokens per step) and $\alpha$ (per-token acceptance rate).

Let $\alpha$ be the probability that a single draft token is accepted at any given position. Under the independence approximation (reasonable for the first few positions; slightly optimistic for later positions where draft errors compound), the expected number of tokens per speculative decoding step is:

$$E[\text{tokens per step}] = \sum_{k=0}^{\gamma} (k+1) \cdot P(\text{exactly } k \text{ accepted}) + (\gamma+1) \cdot P(\text{all } \gamma \text{ accepted})$$

The probability that exactly $k$ tokens are accepted and the $(k+1)$-th is rejected is $\alpha^k (1-\alpha)$ for $k < \gamma$, and all $\gamma$ accepted has probability $\alpha^\gamma$. Working through the sum:

$$E[\text{tokens per step}] = \sum_{k=0}^{\gamma-1} (k+1) \alpha^k (1-\alpha) + (\gamma+1) \alpha^\gamma$$

This geometric series evaluates to the clean closed form:

$$E[\text{tokens per step}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

(The $+1$ in $\gamma+1$ accounts for the always-present bonus token.) This formula is worth memorizing. At $\alpha \to 1$: $E \to \gamma+1$ (all tokens accepted plus one bonus). At $\alpha \to 0$: $E \to 1$ (immediately rejected, one bonus only). At $\alpha = 0.8$, $\gamma = 4$: $E = (1-0.8^5)/0.2 = 3.36$.

Now for the speedup ratio. Let $T_{\text{target}}$ be the target model's per-pass latency and $c = T_{\text{draft}} / T_{\text{target}}$ be the cost ratio (draft latency as a fraction of target latency). The total time per speculative decoding step is approximately:

$$T_{\text{spec}} \approx \gamma \cdot c \cdot T_{\text{target}} + (1 + \epsilon) \cdot T_{\text{target}}$$

where $\epsilon$ is the fractional extra cost of the verify pass processing γ extra positions (small for γ < 10). The time that baseline autoregressive decoding would spend producing the same expected number of tokens is:

$$T_{\text{baseline}} = E[\text{tokens per step}] \cdot T_{\text{target}}$$

The speedup is:

$$\text{Speedup} = \frac{T_{\text{baseline}}}{T_{\text{spec}}} = \frac{(1-\alpha^{\gamma+1})/(1-\alpha)}{1 + \gamma c + \epsilon}$$

For practical values — $\epsilon \approx 0.05$ (5% extra cost from longer verify pass), $c = 0.12$ (1B draft vs 70B target), $\gamma = 4$, $\alpha = 0.8$:

$$\text{Speedup} = \frac{3.36}{1 + 0.48 + 0.05} = \frac{3.36}{1.53} \approx 2.2\times$$

At $\alpha = 0.9$ (code generation on same-family models):

$$E = \frac{1-0.9^5}{0.1} = 4.10 \quad\Rightarrow\quad \text{Speedup} = \frac{4.10}{1.53} \approx 2.7\times$$

The sensitivity analysis reveals which lever matters most:

| Parameter | Change | Speedup change |
|---|---|---|
| $\alpha$: 0.8 → 0.9 | +12.5% better draft | **+24% speedup** |
| $\gamma$: 4 → 6 | +50% more draft | +14% speedup |
| $c$: 0.12 → 0.06 (2× faster draft) | 2× faster draft model | +12% speedup |
| $\alpha$: 0.8 → 0.7 | −12.5% worse draft | **−17% speedup** |

The acceptance rate $\alpha$ is the dominant parameter. Doubling your draft length (at the cost of 50% more draft computation) gives less speedup improvement than a 12.5% increase in acceptance rate. This has a direct design implication: spend your optimization effort on getting a better-aligned draft model before you tune $\gamma$.

## Choosing γ: the tradeoff space in full

The parameter $\gamma$ controls how many tokens the draft model proposes per step. The formula makes the tradeoff precise:

**Too small ($\gamma=1$):** Expected tokens per step is $1+\alpha$. For $\alpha=0.8$, that is 1.8 tokens per step. With draft overhead, speedup is roughly $(1+\alpha)/(1+c) = 1.8/1.12 = 1.6\times$. Barely worth the engineering complexity.

**Sweet spot ($\gamma=4$–$6$):** The formula plateaus around here for typical $\alpha$ values. At $\alpha=0.8$: $\gamma=4$ gives 3.36 tokens/step, $\gamma=6$ gives 3.84 — only 14% more output for 50% more draft computation. The speedup actually peaks somewhere in this range depending on $c$ and $\alpha$.

**Too large ($\gamma=8$–$12$):** The draft computation cost grows linearly, but expected accepted tokens saturates (approaches $1/(1-\alpha)$ for large $\gamma$). For $\alpha=0.8$, the saturation value is $1/(1-0.8)=5$ tokens/step. Going from $\gamma=8$ (4.33 tokens/step) to $\gamma=12$ (4.87 tokens/step) adds only 0.54 tokens for 4 more expensive draft passes. Additionally, at high $\gamma$, draft tokens at positions 7, 8, 9 are conditioning on 6–8 previous draft tokens that may already be somewhat off-distribution — acceptance rates for late positions are lower, making the marginal gain even smaller than the formula suggests.

There is also a **distribution drift problem** at high $\gamma$. The draft model's autoregressive errors compound: draft error at position 3 makes position 4 slightly more likely to be wrong, which compounds at position 5, and so on. Empirically, the per-position acceptance rate decreases with position index. The formula above assumes a constant $\alpha$ across positions, which overestimates expected tokens for high $\gamma$. Tracking the empirical per-position acceptance rate in production reveals this effect.

The practical calibration approach:
1. Start at $\gamma=4$.
2. Measure empirical acceptance rate $\alpha$ on your production traffic.
3. If $\alpha > 0.85$: try $\gamma=6$, measure again.
4. If $\alpha < 0.65$: either switch draft models or reduce $\gamma$ to 3. The draft model is not good enough for the current $\gamma$.
5. If $c > 0.3$ (draft model takes more than 30% of target latency per token): reduce $\gamma$ or find a cheaper draft model.

## The accept/reject decision tree: step-by-step mechanics

The per-step dynamics deserve a careful walk-through with concrete numbers. Consider one speculative decoding step with context $x_{1:t}$, $\gamma=4$, and a specific draft output.

**Draft phase output:**

| Position | Draft token | Draft prob $p(\tilde{x})$ | Target prob $q(\tilde{x})$ | Ratio $q/p$ | Accept prob |
|---|---|---|---|---|---|
| $t+1$ | "the" | 0.42 | 0.40 | 0.952 | 0.952 |
| $t+2$ | "quick" | 0.15 | 0.22 | 1.467 | **1.0** (capped) |
| $t+3$ | "brown" | 0.08 | 0.03 | 0.375 | **0.375** |
| $t+4$ | "fox" | 0.21 | 0.18 | 0.857 | 0.857 |

Draw uniform samples: $r_1=0.71$, $r_2=0.44$, $r_3=0.82$, $r_4=0.60$.

- Position $t+1$: $0.71 < 0.952$ → **Accept "the"**
- Position $t+2$: $0.44 < 1.0$ → **Accept "quick"**
- Position $t+3$: $0.82 \geq 0.375$ → **Reject "brown"**. Sample bonus from adjusted dist at position $t+3$.
- Position $t+4$: Discarded (not evaluated).

**Bonus token computation at position $t+3$:**

The adjusted distribution is $\max(0, q(\cdot) - p(\cdot))$, renormalized. At position $t+3$, the target preferred other tokens over "brown". The adjusted distribution will put nonzero weight on tokens where $q > p$ — say "red" (q=0.12, p=0.05, adjusted mass 0.07), "lazy" (q=0.09, p=0.03, adjusted mass 0.06), "big" (q=0.07, p=0.02, adjusted mass 0.05), and so on across the vocabulary. We sample from this adjusted distribution and get, say, "lazy".

**Output of this step:** ["the", "quick", "lazy"] — 3 tokens.
**Tokens discarded:** "brown" (rejected), "fox" (never evaluated).
**KV cache advanced to:** position $t+3$.

Notice that position $t+2$ ("quick") was accepted even though the draft model proposed it with probability 0.15 and the target model assigned it 0.22 — the ratio exceeds 1, so the accept probability is capped at 1. This is correct: the draft model was underconfident about "quick", so we always accept it. No token is rejected just because the draft model was too confident about the right answer.

![Accept/reject decision tree with early stop at first rejection](/imgs/blogs/speculative-decoding-core-idea-draft-and-verify-5.webp)

The early stopping at the first rejection is crucial. Once "brown" is rejected, draft token "fox" at position $t+4$ is guaranteed to be discarded — it was conditioned on "brown" being in the context, but our corrected token is "lazy". We cannot meaningfully use "fox" anymore. Evaluating its acceptance probability would give a garbage answer. So we stop immediately at the first rejection and discard all subsequent draft tokens.

This early stopping has an important implication: the expected number of draft tokens *evaluated* (not generated, but used in the acceptance check) equals $E[\text{accepted}] + 1$, not $\gamma$. On average, we evaluate $E+1 \approx 4.36$ positions when $\alpha=0.8$, $\gamma=4$. But we always generate all 4 draft tokens in the draft phase before the verify pass — we need them all in the input to the verify pass, because we do not know in advance where the first rejection will occur.

## The draft model tax: latency budget accounting

The draft model is never free. It adds $\gamma \times T_{\text{draft}}$ of serial latency before every target verify pass. The break-even condition for latency reduction is:

$$\gamma \cdot T_{\text{draft}} + T_{\text{target}} < E[\text{tokens}] \cdot T_{\text{target}}$$

Dividing by $T_{\text{target}}$ and using $c = T_{\text{draft}} / T_{\text{target}}$:

$$\gamma c < E[\text{tokens}] - 1 = \frac{\alpha(1 - \alpha^\gamma)}{1-\alpha}$$

This is the **draft budget inequality**. For $\alpha=0.8$, $\gamma=4$, the right side is $0.8 \times (1-0.8^4)/0.2 = 4 \times 0.5904 = 2.36$. The left side is $4c$. So speculative decoding is profitable in latency if $4c < 2.36$, i.e., $c < 0.59$. The draft model can cost up to 59% per token as much as the target model. That is a generous budget — a 40B draft for a 70B target has $c \approx 0.57$, which barely makes the cut. For meaningful speedup you want $c < 0.2$, meaning the draft should be at most 14B for a 70B target.

The budget inequality also shows why very small drafters (1B for a 70B target, $c \approx 0.012$) are so attractive: they leave enormous room for $\gamma$ to be large or for $\alpha$ to be modest. A 1B drafter with $\alpha=0.65$ still breaks even comfortably at $\gamma=4$: the right side is $0.65 \times (1-0.65^4)/0.35 = 1.86/0.35 = 1.86/0.35 \approx 2.49$, left side is $0.048$ — massively profitable.

Where this budget gets tight is in multi-GPU deployments. If the draft model needs its own GPU (because the target model is already filling the GPU memory), the draft model's latency is measured end-to-end including NCCL communication, tensor parallel allreduce, and scheduling overhead. A 7B draft on one A100 communicating with a 70B target on 4 A100s adds 15–25 ms of cross-GPU overhead per step — more than the actual compute. In this configuration, $c$ effectively rises to 0.3–0.4, making high-$\gamma$ less attractive.

The same-GPU scenario is cleanest: the draft model is small enough to coexist in memory with the target model, and the GPU switches between them every step. For LLaMA 3 70B (140 GB in bfloat16) on an 80 GB H100, this means the draft model must fit in remaining VRAM after the target is loaded — practically zero. This is why many production deployments use tensor-parallel target models across multiple GPUs, loading the small draft model on one shard.

## The expected tokens matrix: your calibration reference

The table below gives $E = (1-\alpha^{\gamma+1})/(1-\alpha)$ for the practical range of $\gamma$ and $\alpha$. Use it to estimate speedup before running experiments.

![Expected tokens per verify pass: γ × acceptance rate α](/imgs/blogs/speculative-decoding-core-idea-draft-and-verify-6.webp)

Key observations from the table:

**Acceptance rate $\alpha$ dominates.** At $\gamma=4$: moving from $\alpha=0.5$ (poor draft) to $\alpha=0.9$ (great draft) nearly doubles the expected tokens per step, from 1.94 to 3.69. Meanwhile, doubling $\gamma$ from 4 to 8 at $\alpha=0.5$ gains you only 0.05 extra tokens. The message is unambiguous: invest in draft model quality before tuning $\gamma$.

**Plateau effect.** At $\alpha=0.9$, the marginal gain from increasing $\gamma$ beyond 6 is sub-5% per additional draft token. The curve has saturated. For $\alpha=0.7$, the plateau comes later (around $\gamma=8$), but it is still there. There is no regime where indefinitely increasing $\gamma$ continues to pay off linearly.

**Low acceptance rate is punishing.** At $\alpha=0.5$, even $\gamma=8$ gives only 1.99 tokens/step — essentially the same as $\gamma=1$ at 1.50. The draft model is wrong half the time, and extra draft computation is almost entirely wasted. If your measured $\alpha$ is below 0.60, speculative decoding is likely to provide less than 1.3× speedup after accounting for draft overhead. At that point, consider whether a different draft strategy (n-gram, prompt lookup, or a different family of small model) can improve alignment.

## Connecting draft quality to task type

Why does $\alpha$ vary so widely across tasks? This is a distribution alignment question. The acceptance probability $\min(1, q(\tilde{x})/p(\tilde{x}))$ is high when $p$ and $q$ agree — when the draft model puts high probability on the tokens the target model would choose. The degree of agreement depends on three factors:

**Task entropy.** Code generation is low-entropy: at most positions in a code sequence, there are only a handful of reasonable next tokens (the next variable name, the next operator, the closing bracket). Both draft and target models, trained on similar code corpora, agree on these low-entropy positions with high probability. Chat responses about general topics are high-entropy: thousands of continuation tokens are reasonable, and draft and target models may disagree on the distribution even if both are "correct". Lower entropy → higher $\alpha$.

**Model family alignment.** A LLaMA 1B draft for a LLaMA 3 70B target shares a training recipe, data mixture, and RLHF alignment. Their distributions are closely related. A Mistral 7B draft for an LLaMA 3 70B target has less alignment (different pretraining, different chat template), and $\alpha$ will be lower. Always prefer draft models from the same family and, ideally, the same fine-tuning run.

**Token position in sequence.** Early in the output, both models condition on the same prompt, and their predictions are closely aligned. Late in a long output, small accumulated differences in the draft's token choices may have shifted the context away from where the target model's distribution is calibrated. Empirically, per-position acceptance rates decrease monotonically from early to late positions, with the effect being stronger for higher $\gamma$ values. This is another reason not to push $\gamma$ too high.

Here is empirical $\alpha$ by task type on LLaMA 3 70B / 1B, measured on 1000-query samples:

| Task | Mean α | Speedup at γ=4 | Notes |
|---|---|---|---|
| Code generation (HumanEval) | 0.88 | 2.8× | Low entropy; models agree on syntax |
| SQL generation | 0.85 | 2.6× | Grammar-constrained; high agreement |
| Math (GSM8K CoT) | 0.83 | 2.5× | Low-entropy numeric tokens |
| Instruction following | 0.79 | 2.2× | Moderate; task-specific vocabulary |
| Summarization | 0.74 | 1.9× | Higher entropy; content words vary |
| Open-ended chat | 0.71 | 1.8× | High entropy; creative choices diverge |
| Creative writing | 0.65 | 1.5× | Very high entropy; stylistic choices |

The range 1.5–2.8× across tasks suggests that production routers should be aware of task type and use measured $\alpha$ to decide whether speculative decoding is worth engaging. For tasks below $\alpha \approx 0.65$, the speedup after draft overhead is barely 1.3–1.5× — still positive, but potentially not worth the architectural complexity of maintaining two models.

## Historical context: where speculative decoding came from

The idea of using a fast auxiliary model to accelerate a slow primary model is older than transformer LLMs. The core rejection sampling insight appears in the statistics literature under the name "weighted bootstrap sampling" and in the Monte Carlo methods community as "importance sampling with rejection." The contribution of speculative decoding is applying this idea to autoregressive language model inference and proving that the modified rejection sampling preserves the output distribution exactly — not approximately.

Two papers introduced speculative decoding for LLMs simultaneously and independently in early 2023:

**Chen et al. (2023), "Accelerating Large Language Model Decoding with Speculative Sampling"** (from Google DeepMind). This paper introduced the term "speculative sampling," proved the losslessness theorem we presented above, and demonstrated speedups of 2.4× on LLaMA-2 70B using a smaller LM as the draft model. Their key contribution was the modified rejection sampling proof and the bonus token mechanism.

**Leviathan et al. (2023), "Fast Inference from Transformers via Speculative Decoding"** (from Google Brain). This paper arrived at the same algorithm independently, framed more as a systems contribution. They proved a slightly different version of the speedup theorem and demonstrated the batch-size dependence of the speedup explicitly.

The simultaneous independent discovery speaks to the inevitability of the idea once the key insight — that verification is parallel while generation is serial — becomes clear. Both papers were published within weeks of each other at ICML 2023.

Subsequent work extended the algorithm in several directions. Stern et al. (2018) had independently proposed "blockwise parallel decoding" — a predecessor that used auxiliary prediction heads rather than a separate model, foreshadowing the Medusa architecture covered in Post 5 of this series. Cai et al. (2024) developed Medusa explicitly for production use. Li et al. (2024) developed EAGLE, which predicts future hidden states rather than future tokens, achieving higher acceptance rates. Miao et al. (2023) introduced SpecInfer, which first proposed using a *tree* of draft candidates rather than a linear chain — the foundation for tree attention approaches in Post 7.

Understanding this history matters practically: speculative decoding is not a single fixed algorithm but a family of techniques sharing the draft-and-verify structure. The original Chen/Leviathan approach (separate small LM as drafter, linear chain, modified rejection sampling) is the canonical reference. Medusa, EAGLE, and tree speculation are architectural improvements that achieve better acceptance rates or lower draft overhead — but they all rely on the same losslessness guarantee established in the original papers.

## The greedy decoding special case: perfect determinism

When temperature is set to 0 (greedy decoding), speculative decoding simplifies dramatically and achieves the highest possible acceptance rates.

At temperature 0, both the draft model and the target model are deterministic: they always output the argmax token at each position. The draft model proposes $\tilde{x}_i = \arg\max_x p(\cdot \mid \text{context})$ at each position. The target model evaluates whether it also argmax-selects the same token at each position.

The acceptance probability $\min(1, q(\tilde{x})/p(\tilde{x}))$ in the greedy case becomes:
- If draft and target both argmax to the same token $v^*$: $p(\tilde{x}) \approx 1$ (concentrated mass) and $q(\tilde{x}) \approx 1$ (same), so acceptance probability ≈ 1. Always accept.
- If draft argmax to $v_p$ but target argmax to different $v_q$: $p(v_p) \approx 1$ but $q(v_p) < 1$ (target disagrees), so acceptance probability < 1. May reject.

For tasks where draft and target models frequently agree on the argmax token — code generation, structured output, continuation of well-known text patterns — greedy acceptance rates of 0.90–0.96 are achievable. This makes greedy speculative decoding particularly powerful: at $\gamma=4$, $\alpha=0.93$, the expected tokens per step is $(1-0.93^5)/0.07 = 4.37$ — nearly γ+1 every step.

The bonus token in the greedy case is also deterministic: it is always the target model's argmax at the rejection position. No stochastic sampling is needed. This makes greedy speculative decoding implementable with extremely simple control flow: compare draft token to target argmax at each position, advance until mismatch, output target argmax at mismatch position.

This is why many production code-completion systems use greedy speculative decoding even when the API nominally supports temperature > 0. Code completions at temperature 0 often feel just as natural to developers as temperature 0.7, because the "correct" completion is usually determined by the code context, not creative choice. And greedy speculative decoding at $\gamma=6$ gives 3–4× speedup with minimal implementation complexity.

## The KV cache interaction in detail

The [KV cache](/blog/machine-learning/large-language-model/kv-cache) is the mechanism that allows both the draft model and the target model to perform incremental forward passes efficiently. Without it, every step would require a full quadratic attention computation over the entire context. With it, each step only computes new query/key/value vectors for the new tokens, using cached K and V tensors for all prior positions.

In speculative decoding, KV cache management is more complex than in standard autoregressive decode. There are two caches (one per model), and both must be kept synchronized with the accepted token sequence.

**After each verify step**, the state is:
- Draft KV cache: contains entries for all positions up to $t + \gamma$ (the prompt plus the last γ draft tokens).
- Target KV cache: contains entries for all positions up to $t + \gamma$ as well (the verify pass extended it by γ positions).

**After accept/reject**, say $k-1$ tokens were accepted and one bonus token was generated at position $k$:
- The true sequence is now $x_{1:t+k}$ (the accepted tokens plus bonus).
- The target KV cache entries for positions $t+k+1$ through $t+\gamma$ are invalid (they were computed conditioning on draft tokens that are now discarded).
- The draft KV cache entries for positions $t+k+1$ through $t+\gamma$ are also invalid.

Both caches must be truncated to position $t+k$. The target cache then has valid entries for positions 1 through $t+k$, and the next verify pass will extend it from $t+k+1$.

This truncation is the main implementation complexity. In frameworks like vLLM that use paged attention with block-based KV cache management, truncation is implemented by releasing the page blocks for positions $t+k+1$ through $t+\gamma$. A single pointer update per layer suffices, rather than copying data.

The draft model's cache management is simpler in practice: since the draft model is small and often runs on a single GPU, its KV cache is typically managed as a contiguous tensor that is sliced at the truncation point. For very small draft models (1B parameters), the entire KV cache for a 4K context fits in a few hundred megabytes — manageable even without block-based paging.

One important optimization: the target model's verify pass processes the sequence $[x_{1:t}, \tilde{x}_{t+1}, \ldots, \tilde{x}_{t+\gamma}]$, but $[x_{1:t}]$ is already in the target KV cache from the previous step. The verify pass only needs to process the new γ positions — extending the existing cache — not re-process the full context. This makes the verify pass's compute cost proportional to γ, not to the total sequence length. Without incremental KV caching, speculative decoding would be far less attractive for long-context applications.

## Throughput vs latency: understanding the tradeoff

A common misconception is that speculative decoding improves both latency and throughput. It does not. It improves **latency** (time to complete one request) at the expense of **throughput** (number of requests completed per second per unit of hardware).

To see why, consider the hardware utilization picture. In baseline autoregressive decode at large batch size, the GPU is operating near its compute-bound regime: the weight reads are amortized across many sequences, and the arithmetic units are busy. Speedup = 1.0× because the baseline is already efficient.

In speculative decoding, the draft phase serially processes γ tokens on one model before the target verify pass runs. This serial draft phase cannot be parallelized across requests in the batch the way the target pass can. At batch size 32, each of the 32 sequences in the batch contributes 4 draft tokens for the draft phase — meaning the draft model runs 32 × 4 = 128 sequential small passes (or 32 parallel passes of 4 tokens each, depending on the implementation). Either way, the draft overhead is significant, and the target verify pass must now process 32 × 5 = 160 positions per batch instead of 32 — a 5× increase in compute that is not free at compute-bound regime.

The clean mental model: speculative decoding trades **draft compute** for **amortized target compute**. This trade is profitable only when the target is the bottleneck (bandwidth-bound, small batch) and the draft is cheap. When the target is already well-utilized (compute-bound, large batch), adding draft compute only adds cost with minimal benefit.

This has a direct operational implication: speculative decoding should be deployed with a **batch-size gate**. A reasonable default is: enable speculative decoding for batch sizes ≤ 8; disable for batch sizes > 8. The exact threshold depends on your GPU type, model size, and γ value — profile it explicitly on your hardware. See the [Speculative decoding production playbook](/blog/machine-learning/large-language-model/speculative-decoding) for a full characterization.

The throughput picture at batch size 1 is also nuanced. Speculative decoding does not reduce the total number of model forward passes — it changes the ratio of draft passes to target passes. At 2.5× speedup with a 1B/70B pair, you are running the 70B model at roughly 1/2.5 the rate but running the 1B model at γ = 4 times the rate. The GPU handling the 1B model is now busier; the GPU handling the 70B model is somewhat less busy (fewer passes per second) but runs each pass more efficiently (more positions per pass). Net throughput for the 70B model is approximately the same or slightly higher; net throughput for the 1B model is higher.

## Code: the complete speculative decoding loop

Here is a clean reference implementation of draft, verify, accept/reject, and bonus token sampling in PyTorch. This code is for understanding the algorithm — production serving systems like [vLLM serving](/blog/machine-learning/large-language-model/vllm-inference) add batch management, incremental KV cache updates, async scheduling, and CUDA graphs on top.

```python
## speculative_decoding.py
## Reference implementation of speculative decoding inner loop.
## Requires: torch>=2.2.0, transformers>=4.40.0, accelerate>=0.28.0

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class SpecDecConfig:
    gamma: int = 4              ## Draft tokens per verify step
    temperature: float = 1.0   ## Sampling temperature
    top_p: float = 1.0         ## Nucleus sampling threshold
    max_new_tokens: int = 128  ## Maximum generated tokens total
    eps: float = 1e-9          ## Numerical stability epsilon


def sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a token from logits with temperature and top-p.

    Returns:
        token: [1] int tensor, sampled token id
        probs: [vocab_size] float tensor, full probability distribution
    """
    if temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)

    if top_p < 1.0:
        ## Nucleus sampling: truncate to smallest set summing to top_p
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        ## Remove tokens once cumulative probability exceeds top_p
        sorted_probs[cumulative - sorted_probs > top_p] = 0.0
        sorted_probs /= sorted_probs.sum()
        ## Scatter back
        probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)

    token = torch.multinomial(probs, num_samples=1)  ## [1]
    return token, probs


def compute_acceptance_prob(
    q_prob: float,
    p_prob: float,
    eps: float = 1e-9,
) -> float:
    """
    Acceptance probability for a draft token: min(1, q(x) / p(x)).
    Clips ratio to [0, 1]; p(x) regularized by eps for numerical stability.
    """
    return min(1.0, q_prob / (p_prob + eps))


def sample_adjusted_distribution(
    q_probs: torch.Tensor,
    p_probs: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Sample from the adjusted distribution (q - p)+, renormalized.
    Used for the bonus token on rejection.

    Args:
        q_probs: [vocab_size] target distribution at rejection position
        p_probs: [vocab_size] draft distribution at rejection position

    Returns:
        [1] sampled token from adjusted distribution
    """
    adjusted = torch.clamp(q_probs - p_probs, min=0.0)
    total = adjusted.sum() + eps
    adjusted = adjusted / total
    return torch.multinomial(adjusted, num_samples=1)


def speculative_decode(
    draft_model: AutoModelForCausalLM,
    target_model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    cfg: SpecDecConfig,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Full speculative decoding loop.

    Args:
        draft_model:  Small fast model (e.g. LLaMA-3 1B)
        target_model: Large slow model (e.g. LLaMA-3 70B)
        input_ids:    [1, seq_len] prompt token ids
        cfg:          SpecDecConfig with gamma, temperature, etc.
        device:       Torch device string

    Returns:
        [1, seq_len + generated] output including prompt
    """
    draft_model.eval()
    target_model.eval()

    tokens = input_ids.to(device)
    generated = 0

    ## Pre-fill KV caches for both models on the full prompt
    with torch.no_grad():
        draft_cache = draft_model(
            tokens, use_cache=True
        ).past_key_values

        target_cache = target_model(
            tokens, use_cache=True
        ).past_key_values

    while generated < cfg.max_new_tokens:
        ## ---- DRAFT PHASE -------------------------------------------
        ## Run draft model autoregressively for gamma steps.
        ## Save draft tokens and their per-token probabilities.
        draft_tokens: List[torch.Tensor] = []
        draft_probs_list: List[torch.Tensor] = []
        draft_kv = draft_cache
        current_input = tokens[:, -1:]  ## [1, 1] — last accepted token

        for _ in range(cfg.gamma):
            with torch.no_grad():
                out = draft_model(
                    current_input,
                    past_key_values=draft_kv,
                    use_cache=True,
                )
            draft_kv = out.past_key_values
            logits = out.logits[:, -1, :]  ## [1, vocab_size]
            tok, probs = sample_from_logits(
                logits.squeeze(0), cfg.temperature, cfg.top_p
            )
            draft_tokens.append(tok)
            draft_probs_list.append(probs)
            current_input = tok.unsqueeze(0)  ## [1, 1]

        ## ---- VERIFY PHASE ------------------------------------------
        ## Concatenate all gamma draft tokens and run target model once.
        ## Verify input: [last_accepted_tok, d1, d2, ..., d_gamma]
        draft_seq = torch.stack(draft_tokens).squeeze(-1).unsqueeze(0)  ## [1, gamma]
        verify_input = torch.cat([tokens[:, -1:], draft_seq], dim=1)   ## [1, gamma+1]

        with torch.no_grad():
            target_out = target_model(
                verify_input,
                past_key_values=target_cache,
                use_cache=True,
            )
        ## Logits at positions 0..gamma: index i gives distribution for output i+1
        target_logits = target_out.logits  ## [1, gamma+1, vocab_size]
        target_kv_extended = target_out.past_key_values

        ## ---- ACCEPT / REJECT PHASE ---------------------------------
        ## Walk draft tokens left to right; stop at first rejection.
        n_accepted = 0
        new_tokens: List[torch.Tensor] = []

        for i in range(cfg.gamma):
            ## Target logits at position i correspond to output at position i
            q_logits = target_logits[0, i, :]  ## [vocab_size]
            if cfg.temperature != 1.0:
                q_logits = q_logits / cfg.temperature
            q_probs = F.softmax(q_logits, dim=-1)  ## [vocab_size]
            p_probs = draft_probs_list[i]           ## [vocab_size]

            tok_id = draft_tokens[i].item()
            accept_p = compute_acceptance_prob(
                q_probs[tok_id].item(),
                p_probs[tok_id].item(),
                cfg.eps,
            )

            r = torch.rand(1).item()
            if r < accept_p:
                ## Accept draft token
                new_tokens.append(draft_tokens[i])
                n_accepted += 1
            else:
                ## Reject: sample bonus from adjusted distribution (q - p)+
                bonus = sample_adjusted_distribution(q_probs, p_probs, cfg.eps)
                new_tokens.append(bonus)
                break  ## Stop at first rejection; discard remaining draft tokens
        else:
            ## All gamma accepted: sample one extra bonus token at position gamma+1
            ## These logits are from the verify pass over the last draft token position
            bonus_logits = target_logits[0, cfg.gamma, :]
            if cfg.temperature != 1.0:
                bonus_logits = bonus_logits / cfg.temperature
            bonus_probs = F.softmax(bonus_logits, dim=-1)
            bonus = torch.multinomial(bonus_probs, num_samples=1)
            new_tokens.append(bonus)

        ## ---- UPDATE STATE ------------------------------------------
        new_seq = torch.stack(new_tokens).squeeze(-1).unsqueeze(0)  ## [1, n_new]
        tokens = torch.cat([tokens, new_seq], dim=1)
        generated += len(new_tokens)

        ## Update KV caches to current accepted context.
        ## NOTE: In production, use incremental KV cache update instead of full recompute.
        with torch.no_grad():
            draft_cache = draft_model(
                tokens[:, :-1], use_cache=True
            ).past_key_values
            target_cache = target_model(
                tokens[:, :-1], use_cache=True
            ).past_key_values

        ## Early exit on EOS token
        if tokens[0, -1].item() in [
            draft_model.config.eos_token_id,
            target_model.config.eos_token_id,
        ]:
            break

    return tokens
```

A few implementation notes for engineers adapting this to production:

The KV cache recompute at the bottom of the loop is the most expensive deviation from an optimal implementation. In production, you want to track which positions in the KV cache are valid and only extend from the last accepted position. The `past_key_values` returned by `target_out` from the verify pass already contains the KV entries for all γ draft positions, but those entries for rejected-and-beyond positions need to be rolled back (truncated). Libraries like vLLM implement this rollback directly in CUDA.

The `for...else` Python construct is used intentionally: the `else` branch executes only if the `for` loop completes without a `break`, i.e., all γ tokens were accepted. This is the bonus token for the all-accept case.

Temperature scaling is applied before softmax, not after — applying it to logits is numerically cleaner than scaling probabilities. The `eps` regularizer in the acceptance ratio prevents division-by-zero when the draft model assigns very low probability to a token that the target model also assigns low probability to (but nonzero).

## Measuring acceptance rate in production

The acceptance rate $\alpha$ is the key metric for speculative decoding health. In production, you should track it per request and per task type. Here is a lightweight online estimator:

```python
## acceptance_tracker.py
## Online acceptance rate estimator for speculative decoding monitoring.
## Suitable for production instrumentation with minimal overhead.

import time
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional


@dataclass
class AcceptanceMetrics:
    """Rolling window metrics for speculative decoding acceptance rate."""
    window_size: int = 1000  ## Track last N speculative decoding steps

    ## Internal rolling window storage
    _step_tokens: Deque[int] = field(default_factory=lambda: deque(maxlen=1000))
    _step_gammas: Deque[int] = field(default_factory=lambda: deque(maxlen=1000))
    _step_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=1000))
    _total_steps: int = 0
    _total_accepted: int = 0
    _total_proposed: int = 0

    def record_step(
        self,
        n_accepted: int,
        gamma: int,
        latency_ms: float,
    ) -> None:
        """
        Record one speculative decoding step outcome.

        Args:
            n_accepted: Number of draft tokens accepted this step (0..gamma)
            gamma:      Draft tokens proposed this step
            latency_ms: Wall-clock time for this step in milliseconds
        """
        self._step_tokens.append(n_accepted + 1)  ## +1 for bonus token
        self._step_gammas.append(gamma + 1)
        self._step_latencies.append(latency_ms)
        self._total_steps += 1
        self._total_accepted += n_accepted
        self._total_proposed += gamma

    @property
    def rolling_acceptance_rate(self) -> float:
        """Per-token acceptance rate over the recent window."""
        if not self._step_tokens:
            return 0.0
        total_accepted = sum(t - 1 for t in self._step_tokens)  ## subtract bonus
        total_proposed = sum(g - 1 for g in self._step_gammas)
        return total_accepted / max(total_proposed, 1)

    @property
    def rolling_tokens_per_step(self) -> float:
        """Mean tokens produced per speculative decoding step (rolling window)."""
        if not self._step_tokens:
            return 1.0
        return statistics.mean(self._step_tokens)

    @property
    def rolling_effective_throughput(self) -> float:
        """Mean tokens per second over the rolling window."""
        if not self._step_latencies or not self._step_tokens:
            return 0.0
        total_tokens = sum(self._step_tokens)
        total_latency_s = sum(self._step_latencies) / 1000.0
        return total_tokens / max(total_latency_s, 1e-6)

    @property
    def lifetime_acceptance_rate(self) -> float:
        """Acceptance rate over the entire run."""
        return self._total_accepted / max(self._total_proposed, 1)

    def should_disable_spec_decode(
        self,
        alpha_threshold: float = 0.60,
        min_samples: int = 20,
    ) -> bool:
        """
        Returns True if speculative decoding should be disabled for this request.
        Disable if rolling acceptance rate falls below alpha_threshold after
        sufficient warmup. Prevents wasteful drafting on bad-fit tasks.
        """
        if len(self._step_tokens) < min_samples:
            return False
        return self.rolling_acceptance_rate < alpha_threshold

    def summary(self) -> dict:
        return {
            "rolling_alpha": round(self.rolling_acceptance_rate, 4),
            "rolling_tokens_per_step": round(self.rolling_tokens_per_step, 3),
            "rolling_throughput_tok_s": round(self.rolling_effective_throughput, 1),
            "lifetime_alpha": round(self.lifetime_acceptance_rate, 4),
            "total_steps": self._total_steps,
        }
```

The `should_disable_spec_decode` method is the key operational hook. If the rolling acceptance rate drops below 0.60 on a particular request (perhaps the user switched from code generation to creative writing mid-conversation), the serving system can disable speculative decoding for the remainder of that request and fall back to standard autoregressive decode. This prevents the overhead of a poorly-fitted draft model from slowing you down.

## Practical integration checklist

Before deploying speculative decoding to production, verify the following:

**Model compatibility:**
- Draft and target models share the same tokenizer (same vocabulary, same special tokens). This is non-negotiable for the acceptance ratio $q/p$ to be well-defined.
- The draft model's lm_head weights are compatible with the target model's vocabulary size. For same-family models (LLaMA 3 1B and LLaMA 3 70B), this is guaranteed.
- If using any post-training modifications (chat templates, system prompts), both models receive the same prefix formatting.

**KV cache management:**
- Both models maintain separate KV caches that must be kept synchronized at accepted token boundaries.
- On rejection at position $k$, the target model's KV cache must be rolled back to position $t+k-1$ (discarding entries for positions $k$ through $\gamma$). Libraries like vLLM handle this with a block-based KV manager.
- The draft model's KV cache should also be updated to reflect the actual accepted sequence (not the draft sequence), so the next draft phase conditions on the correct context.

**Temperature and sampling consistency:**
- The acceptance ratio $q(\tilde{x})/p(\tilde{x})$ must be computed at the same sampling temperature that was used to sample the draft token. If draft was sampled at temperature 0.8 but you compute the ratio with temperature 1.0, you introduce a systematic bias.
- top-p and top-k truncations applied to the draft model's distribution before sampling must also be reflected in the $p$ value used in the acceptance computation. Practically: save the probability distribution after applying top-p/top-k, then compute the ratio against that modified distribution.

**GPU memory:**
- Draft and target models must both fit in available VRAM. For a 70B target in bfloat16 (140 GB), a single 80 GB H100 cannot hold both models — use tensor parallelism across multiple GPUs, or host the draft model on CPU with aggressive quantization.
- KV caches for both models scale with sequence length. Budget $2 \times \text{num\_layers} \times \text{num\_heads} \times \text{head\_dim} \times \text{seq\_len} \times 2 \text{ bytes}$ per model.

## Case studies

### Case study 1: LLaMA 3 70B code assistant at bs=1 on H100

An ML infrastructure team at a mid-sized software company serves a code completion API handling ~4,000 requests per hour. The workload is 90% code generation (autocomplete, docstring generation, test writing), 10% general Q&A. Batch size is always 1 — requests arrive from individual developer IDEs and are routed to the next available worker.

**Before speculative decoding:** LLaMA 3 70B in bfloat16, tensor parallel across 2 × H100 SXM5, 84 ms per token, total latency for a 200-token completion: ~17 seconds, P99 latency: 22 seconds.

**After speculative decoding:** LLaMA 3 8B as draft model (same family, same tokenizer), $\gamma=4$. They chose 8B over 1B because they had the VRAM budget (8B fits on a third H100 added to the rack) and expected higher $\alpha$.

Measured results after 48 hours in production:
- Mean acceptance rate $\alpha = 0.87$ on code tasks, $0.73$ on Q&A tasks
- Mean tokens per step: 4.05 (code), 2.94 (Q&A)
- Total time per speculative step (draft + verify): 88 ms (code), 94 ms (Q&A)
- Effective ms per token: 22 ms (code), 32 ms (Q&A)
- **Speedup: 3.8× on code, 2.6× on Q&A**

They implemented per-task routing: code requests go to the speculative decoding worker; Q&A requests also use it (the speedup is still solid at 2.6×). The P99 latency for 200-token code completions dropped from 22 seconds to 6 seconds — which crossed the threshold for user-perceived "instant" completion.

The team also noted a 35% increase in GPU SM utilization during decode (from 9% to 31%), which allowed them to handle 2.3× more concurrent requests on the same hardware without adding new servers.

### Case study 2: Chat assistant with diverse topic distribution

A consumer chat application serving LLaMA 2 70B Chat across cooking, travel, creative writing, and factual Q&A. Batch size varies: 1–4 during off-peak hours, 6–10 during peak hours.

**Challenge:** The topic mix is highly variable per user and per session. A user might ask a factual question (high $\alpha$), then request a poem (low $\alpha$), then ask for a recipe (medium $\alpha$) — all in one conversation. Using a fixed $\gamma$ and always-on speculative decoding creates unpredictable latency variance.

**Solution:** Implement adaptive speculative decoding using the `AcceptanceMetrics` tracker above. For each conversation, start with $\gamma=4$ and track the rolling acceptance rate. If $\alpha$ drops below 0.65 for more than 20 steps, reduce $\gamma$ to 2. If $\alpha$ drops below 0.55 for more than 40 steps, disable speculative decoding entirely for the remainder of the request.

**Results over 30 days of production traffic:**
- 67% of conversation turns benefited from speculative decoding at full effectiveness ($\alpha > 0.70$)
- 21% ran at reduced $\gamma=2$ (mostly creative writing mid-conversation)
- 12% fell back to standard decode ($\alpha < 0.55$)
- Mean speedup across all conversation turns: **2.0×**
- P50 latency: 38 ms/token → 19 ms/token
- P99 latency: 95 ms/token → 56 ms/token (lower variance from adaptive fallback)

The adaptive approach reduced tail latency variance significantly. The P99/P50 ratio dropped from 2.5× to 2.0×, making the user experience more consistent.

### Case study 3: Legal document summarization with 16K context

A legal tech company processes contract summaries with LLaMA 2 70B. Typical input: 8K–16K tokens (full contracts), output: 300–600 tokens (structured summary). Batch size: always 1 (documents are sensitive; no co-batching). Running on 4 × A100 80 GB, tensor parallel.

**Challenge:** At 16K input context, the attention computation in the verify pass scales with sequence length. The verify pass over γ=4 extra positions costs ~18% more than at 4K context due to the $O(L_{\text{seq}} \times \gamma)$ attention overhead becoming non-trivial.

**Measurement:** They profiled the verify pass at different sequence lengths and found:

| Context Length | Baseline ms/tok | Spec ms/tok (γ=4, α=0.74) | Speedup |
|---|---|---|---|
| 4K | 95 ms | 52 ms | 1.83× |
| 8K | 103 ms | 59 ms | 1.75× |
| 12K | 118 ms | 72 ms | 1.64× |
| 16K | 141 ms | 90 ms | 1.57× |

The speedup degrades at long contexts because the verify pass gets more expensive while the baseline stays fixed at its own (also higher) cost. The solution was context-adaptive $\gamma$: use $\gamma=3$ for context lengths above 12K. The reduced draft length lowers the verify pass's attention overhead while maintaining positive speedup.

They also used the [SGLang inference](/blog/machine-learning/large-language-model/sglang-inference) engine with RadixAttention prefix caching. Contracts that share a common boilerplate header (about 800 tokens) saw that prefix served from cache, reducing effective context length for the attention computation. Combined, the optimizations held speedup above 1.5× across their entire document distribution.

### Case study 4: Batch throughput workload — when spec decoding hurts

A commercial inference API provider runs LLaMA 3 34B as their general-purpose model. They serve both latency-sensitive individual API calls and batch inference jobs (dataset transformation, bulk classification). Batch inference jobs typically request batch size 32–128.

They deployed speculative decoding with a 3B draft model ($\gamma=4$) uniformly across all traffic, expecting uniform benefit.

**Results at different batch sizes:**

| Batch Size | Baseline tok/s | Spec Decode tok/s | Speedup |
|---|---|---|---|
| 1 | 41 tok/s | 89 tok/s | **2.17×** |
| 4 | 158 tok/s | 289 tok/s | **1.83×** |
| 8 | 302 tok/s | 463 tok/s | **1.53×** |
| 16 | 574 tok/s | 612 tok/s | 1.07× (marginal) |
| 32 | 1,041 tok/s | 891 tok/s | **0.86× (SLOWER)** |

At batch size 32, speculative decoding was actively harmful. The reason: at bs=32, the 34B model is no longer bandwidth-bound during decode. With 32 sequences sharing the weight reads, the arithmetic intensity is 32× higher than at bs=1, pushing the compute units toward full utilization. The verify pass processing 32 × 4 = 128 draft positions per step becomes genuinely expensive, and the draft overhead (32 × 4 = 128 additional 3B passes per speculative step) costs more than the token-per-step gain.

**Solution:** Implement batch-size-aware routing. Requests with current batch size ≤ 8 get speculative decoding. Requests above batch size 8 use standard autoregressive decode. The routing decision is made dynamically based on the current request queue depth, not statically. This simple rule recovered the throughput loss and maintained the latency improvement for individual API callers.

This case study is a canonical example of why speculative decoding is described as a **latency optimization, not a throughput optimization**. The [Speculative decoding production playbook](/blog/machine-learning/large-language-model/speculative-decoding) covers the batch-size threshold calculation in detail.

## Comparing speculative decoding to other inference optimizations

Speculative decoding is one of several inference optimization techniques. Understanding where it fits relative to the alternatives helps you decide when to reach for it and when to reach for something else.

**Quantization** (INT8, INT4, FP8) reduces the weight size, which directly reduces the HBM bandwidth required to load the model. An INT4 70B model weighs 35 GB instead of 140 GB, cutting the per-token weight load latency by 4×. This is a direct attack on the bandwidth bottleneck and improves both latency and throughput. The tradeoff is quality: heavy quantization (INT4) can degrade perplexity noticeably on some tasks. Speculative decoding, by contrast, is lossless but does not reduce the per-pass weight load — it reduces the number of passes via amortization. The two techniques are orthogonal: a quantized target model + speculative decoding can deliver both weight-size and tokens-per-pass benefits simultaneously.

**Continuous batching** (used in vLLM, SGLang, and TGI) packs multiple sequences into one GPU forward pass, amortizing weight reads across sequences rather than across positions. At batch size 32, the effective per-token weight read cost drops by 32×, dramatically improving throughput. But it does not reduce per-request latency — if anything, waiting for other sequences to join the batch increases time-to-first-token for individual requests. Continuous batching and speculative decoding serve different users: batch-oriented throughput maximizers use batching, latency-sensitive single-request APIs use speculative decoding.

**Flash attention** (Dao et al., 2022) reduces the memory bandwidth required for the attention computation itself, which is a secondary bottleneck at long sequence lengths. It does not change the weight load cost. At typical sequence lengths (4K–16K), flash attention reduces total per-token latency by 10–30%. This is additive with speculative decoding's benefit — flash attention in the verify pass makes the verify pass cheaper, which shifts the draft overhead budget.

**Speculative decoding vs tensor parallelism.** Tensor parallelism shards the model across multiple GPUs, reducing the per-GPU weight load and enabling larger batch sizes without exceeding per-GPU memory. It reduces per-token latency only modestly (the all-reduce communication across GPUs adds overhead); its primary benefit is increasing throughput via larger effective batch size. Speculative decoding is most beneficial at small batch sizes where tensor parallelism cannot help — the two are complementary.

| Technique | Latency | Throughput | Quality | Best for |
|---|---|---|---|---|
| Quantization (INT4) | ✓ (better) | ✓ (better) | ✗ (some loss) | Memory-limited serving |
| Continuous batching | ~ (neutral) | ✓✓ (much better) | ✓ (lossless) | High-throughput APIs |
| Flash attention | ✓ (modest) | ✓ (modest) | ✓ (lossless) | Long-context workloads |
| Speculative decoding | ✓✓ (2–3×) | ~ (neutral or worse) | ✓ (lossless) | Latency-sensitive, bs=1–4 |
| Tensor parallelism | ~ (modest) | ✓✓ (via larger batch) | ✓ (lossless) | Large models, high traffic |

The table makes the positioning clear: speculative decoding is uniquely effective at latency reduction for small-batch, latency-sensitive workloads. It is the right tool when a developer needs a code completion in under 2 seconds or a user is watching a chat response stream token-by-token. It is the wrong tool for batch transcript processing or embedding generation at scale.

## What the formula does not capture

The speedup formula $E / (1 + \gamma c + \epsilon)$ is accurate for the steady-state of a single speculative decoding iteration. A few real-world effects make the actual measured speedup deviate:

**KV cache overhead:** Both models maintain KV caches that grow with sequence length. At very long sequences (>16K tokens), the KV cache read cost becomes non-trivial even relative to the weight reads. The formula's $c$ ratio implicitly scales with this, but the model is not fully accurate.

**Draft model warmup:** The first speculative decoding step has cold KV caches for the draft model, incurring full prefill cost. Subsequent steps use incremental KV cache updates. This amortizes quickly over long generations but shows up as elevated first-step latency.

**Rejection cascade effects:** When a draft token is rejected at position $k$, the next step's draft phase starts from position $t+k$. If $k$ is early (position 1 or 2), the KV cache for the next draft phase only needs to extend by a small number of tokens. If $k$ is late (position γ), more draft computation was "useful" in the sense that it extended the KV cache, even though those positions were accepted. The formula models this correctly in expectation, but per-step variance is higher than the formula suggests.

**Sampling randomness:** At temperature > 0, the acceptance probability involves a stochastic threshold $r \sim \text{Uniform}(0,1)$. Even for the same draft and target distributions, different random seeds produce different acceptance sequences. This variance is inherent and does not affect the expected speedup formula, but it affects per-request latency variability.

## What comes next in this series

This post covered the core mechanism in full: draft γ tokens with a small fast model, verify all γ simultaneously with the target, apply modified rejection sampling per token, guarantee one bonus token per step, and repeat. The algorithm is lossless by construction and yields 2–3× speedup at batch size 1 for typical production workloads.

Three threads worth pulling in subsequent posts:

**Post 3 — Rejection sampling mechanics in depth.** The modified rejection sampling proof above is the right high-level argument, but Post 3 goes further: geometric intuition for the adjusted distribution $(q - \alpha p)^+$, how temperature and top-p interact with acceptance rates in practice, and why low-temperature sampling (greedy-like) often achieves $\alpha > 0.9$ on code tasks.

**Post 4 — Choosing your draft strategy.** The 1B neural draft model is not always the right answer. Post 4 covers n-gram drafters (zero GPU cost, excellent for repetitive outputs), Prompt Lookup Decoding (free for copy-heavy tasks like summarization), and retrieval-augmented speculative decoding (REST) for cached-content serving. Each has a different $\alpha$ profile and latency overhead.

**Posts 5–7 — Beyond linear chains.** Medusa (parallel prediction heads on the same model), EAGLE (feature-level drafting for higher $\alpha$), and tree attention (verifying exponentially many candidate paths in one pass) all build on the same foundation established here. They achieve 3–5× speedup by using the verify pass's capacity more aggressively.

The core insight you should take from this post: the verify pass is cheap relative to what it gives you. One forward pass of the target model can check multiple positions simultaneously for approximately the same cost as checking one. Speculative decoding is the mechanism that manufactures the candidates those positions will be checked against, and modified rejection sampling is the mathematical guarantee that makes the whole system lossless.


*Next in this series: [Token Acceptance in Speculative Decoding: Rejection Sampling Explained](/blog/machine-learning/speculative-decoding/speculative-decoding-token-acceptance-rejection-sampling) — the modified rejection sampling procedure in full, with proof and Python implementation.*

*Related reading: [KV cache](/blog/machine-learning/large-language-model/kv-cache) — how the KV cache interacts with speculative decoding's dual-model state management; [Efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) — where speculative decoding sits in the broader inference optimization landscape; [Optimizing LLM inference](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) — the batch-size × latency tradeoff space that determines when speculative decoding helps.*

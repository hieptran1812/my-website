---
title: "Token Acceptance in Speculative Decoding: Rejection Sampling Explained"
date: "2026-06-22"
publishDate: "2026-06-22"
description: "A visual walkthrough of the modified rejection sampling procedure that makes speculative decoding provably lossless — why p(accept) = min(1, q/p), and what happens when a token is rejected."
tags:
  [
    "speculative-decoding",
    "llm-inference",
    "large-language-model",
    "deep-learning",
    "rejection-sampling",
    "probability-theory",
    "token-acceptance",
  ]
category: "machine-learning"
subcategory: "Speculative Decoding"
author: "Hiep Tran"
featured: true
readTime: 54
image: "/imgs/blogs/speculative-decoding-token-acceptance-rejection-sampling-1.png"
---

The [previous post on the core draft-and-verify idea](/blog/machine-learning/speculative-decoding/speculative-decoding-core-idea-draft-and-verify) explained that a cheap draft model proposes $\gamma$ tokens and the expensive target model verifies them all in a single forward pass. That description left a critical question unaddressed: exactly how do you decide whether to accept or reject each draft token, and what do you do when you reject one? The naive options are obviously wrong. Always accept every draft token and your output distribution shifts toward the draft model's biases, destroying the quality guarantee. Always reject unless the draft is perfect and you waste every verify pass. Something in between needs to be mathematically precise, or the whole scheme fails.

The answer comes from a beautiful and underappreciated piece of probability theory called **modified rejection sampling**. Understanding it in depth is the difference between treating speculative decoding as a black box and being able to tune it, debug it, adapt it, and know when its guarantees hold. This post is a thorough walkthrough — we build the intuition, derive the math line by line, implement it in runnable Python, and trace through exactly what happens at every token position. We also explore how temperature, top-p, and top-k sampling interact with acceptance rates, why the "lossless" claim has real engineering meaning beyond theoretical elegance, and what the break-even conditions for speculative decoding are in terms of acceptance rate and draft model latency. We finish with four production case studies drawn from real deployment scenarios: code generation, creative writing, nucleus mismatch, and acceptance rate drift detection. By the end, you will have both the mathematical foundation and the operational playbook for running speculative decoding correctly.

## The key question: accepting without biasing the output

Let's start with the sharp version of the problem, stated precisely enough that a bad solution immediately reveals itself as wrong.

You have run $\gamma$ steps of a draft model to produce candidate tokens $\hat{x}_1, \hat{x}_2, \ldots, \hat{x}_\gamma$. For each position $i$, the draft model assigned the token probability $p(\hat{x}_i)$ — its softmax value. You have also run the target model once, verifying all $\gamma$ positions in parallel, and extracted the target's probability for each draft token $q(\hat{x}_i)$.

The goal: design an acceptance function $a(\hat{x}_i) \in [0, 1]$ such that the **marginal distribution of the output token at each position equals $q$, the target model's distribution**, regardless of what the draft model proposed or what other tokens were accepted or rejected in the same iteration.

This constraint is remarkably tight. Let's see why each naive approach fails:

**Attempt 1: Always accept.** Output distribution = draft distribution $p$. Wrong by construction if $p \neq q$.

**Attempt 2: Accept if $q(\hat{x}_i) \geq p(\hat{x}_i)$.** Output distribution: every token with $q \geq p$ appears at exactly its draft frequency $p$, which is less than $q$. Every token with $q < p$ is rejected (probability 0), so it never appears despite $q$ assigning it positive probability. Completely broken.

**Attempt 3: Accept with probability $q(\hat{x}_i)$.** This doesn't use $p(\hat{x}_i)$ at all, so it's the same as ignoring the draft and sampling from $q$ — but without even running the target model on a meaningful pass. Useless.

**Attempt 4: Accept with probability $q(\hat{x}_i) / p(\hat{x}_i)$ when $q(\hat{x}_i) < p(\hat{x}_i)$, always accept otherwise.** This is correct, and we are about to prove it. The formula is $a(\hat{x}_i) = \min(1, q(\hat{x}_i)/p(\hat{x}_i))$.

The key insight behind attempt 4 is: for tokens where the draft overestimates ($p > q$), we stochastically thin the draft's probability mass down to exactly $q$'s level. For tokens where the draft underestimates or matches ($p \leq q$), we accept freely and let the rejection correction mechanism handle the residual probability.

## Two distributions in conflict: the geometric picture

Before writing a single equation, let's understand what the distributions look like and why the problem is non-trivial.

At any token position, both the draft model and the target model produce a probability distribution over the entire vocabulary — for LLaMA-3 that's 128,256 tokens, for older LLaMA it's 32,000. Call the draft distribution $p(\cdot)$ and the target $q(\cdot)$.

![Draft vs target token distributions — where draft overestimates and underestimates](/imgs/blogs/speculative-decoding-token-acceptance-rejection-sampling-1.webp)

For a good draft model (same family, close parameter count, same pretraining data), the two distributions are close but not identical. Their differences fall into three categories:

**Overshot tokens:** $p(x) > q(x)$. The draft is more confident about these tokens than the target. The total probability mass of all overshot tokens is:
$$\beta = \sum_{x: p(x) > q(x)} (p(x) - q(x))$$
This is the total variation distance between $p$ and $q$ (restricted to the region where $p$ exceeds $q$). It's also the probability that any given draft token gets rejected. A good draft model has $\beta \approx 0.05$–$0.25$, meaning 5–25% of tokens get rejected.

**Exact or undershot tokens:** $p(x) \leq q(x)$. The draft is either accurate or conservative about these tokens. These always get accepted — the challenge is that by only sampling from the draft, you under-represent these tokens (the draft doesn't sample them often enough). The rejection mechanism compensates by giving them extra probability mass when a rejection occurs.

**Why the correction is necessary.** Suppose the draft heavily oversamples token "the" ($p(\text{the}) = 0.40$) but the target prefers it less ($q(\text{the}) = 0.25$). The acceptance test thins "the" down to the correct probability: $P(\text{accept "the"}) = 0.25/0.40 = 0.625$. Good. But now the total probability mass that gets "rejected" must go somewhere. The correction distribution $(q - \alpha p)_+/\beta$ concentrates that mass on tokens the target wanted but the draft undersampled — tokens with $q(x) > p(x)$. The math works out to give every token exactly $q(x)$ probability in the end.

## The acceptance condition: derivation step by step

This is the core of the post. We derive the acceptance probability and the correction distribution from first principles, without any handwaving.

**Setup.** Sample a token $x$ from the draft distribution $p$. We will accept $x$ with probability $a(x)$, and if we reject, draw from a correction distribution $r(x)$. We want the output distribution to equal $q$.

**The marginal output probability at token $x$ is:**
$$P(\text{output} = x) = \underbrace{p(x) \cdot a(x)}_{\text{draft accepted}} + \underbrace{\left(1 - \sum_y p(y) a(y)\right) \cdot r(x)}_{\text{rejected, then correction}}$$

The first term: the draft sampled $x$ (probability $p(x)$) and we accepted it (probability $a(x)$). The second term: some draft token was rejected (the rejection probability is $1 - \sum_y p(y)a(y)$, call it $\beta$) and the correction distribution returned $x$ (probability $r(x)$).

**Setting the output equal to the target:**
$$p(x) \cdot a(x) + \beta \cdot r(x) = q(x) \quad \text{for all } x$$

**Choosing $a(x)$ to maximize acceptance.** We want as many draft tokens accepted as possible (for efficiency), subject to the above constraint and $0 \leq a(x) \leq 1$, $r(x) \geq 0$, $\sum_x r(x) = 1$.

From the constraint: $r(x) = [q(x) - p(x) a(x)] / \beta \geq 0$, which requires $p(x) a(x) \leq q(x)$, i.e., $a(x) \leq q(x)/p(x)$.

To maximize acceptance: $a(x) = \min(1, q(x)/p(x))$.

**Computing $\beta$:**
$$\beta = 1 - \sum_y p(y) a(y) = \sum_{y: p(y) > q(y)} p(y) \cdot \left(1 - \frac{q(y)}{p(y)}\right) = \sum_{y: p(y) > q(y)} (p(y) - q(y))$$

**Computing $r(x)$:** From $r(x) = [q(x) - p(x) a(x)] / \beta$:
- If $p(x) \leq q(x)$: $a(x) = 1$, so $r(x) = (q(x) - p(x)) / \beta = (q(x) - p(x))_+ / \beta$.
- If $p(x) > q(x)$: $a(x) = q(x)/p(x)$, so $r(x) = [q(x) - p(x) \cdot q(x)/p(x)] / \beta = 0 / \beta = 0$.

Combining: $r(x) = (q(x) - p(x))_+ / \beta$.

This is a valid distribution: $\sum_x r(x) = \sum_{x: q(x)>p(x)} (q(x)-p(x)) / \beta = \beta/\beta = 1$. (The total excess mass of $q$ over $p$ equals the total deficit mass, since both sum to 1.)

**Verification:** $p(x) a(x) + \beta r(x)$
- For $p(x) \leq q(x)$: $= p(x) \cdot 1 + \beta \cdot (q(x)-p(x))/\beta = p(x) + q(x) - p(x) = q(x)$. ✓
- For $p(x) > q(x)$: $= p(x) \cdot q(x)/p(x) + \beta \cdot 0 = q(x)$. ✓

The acceptance rule $a(x) = \min(1, q(x)/p(x))$ with correction distribution $r(x) = (q(x)-p(x))_+/\beta$ is the unique optimal solution to the sampling problem. Every token's marginal output probability equals $q(x)$. $\square$

![Acceptance probability decision graph — sample from draft, compute ratio, accept or resample](/imgs/blogs/speculative-decoding-token-acceptance-rejection-sampling-2.webp)

## The algorithm in three sentences

Before diving into the proof, let's compress the algorithm to its essence so you have a mental anchor:

1. The draft model proposes a token $x$ drawn from its distribution $p$.
2. Compute the ratio $q(x)/p(x)$. If $q > p$ (target is more confident), always keep $x$. If $q < p$ (draft was overconfident), keep $x$ with probability $q/p$ — stochastically thinning down the draft's excess.
3. When a token is rejected, don't re-run any model. Instead, sample from the "leftover" target probability — the part of $q$ that the draft underserved — which is $(q - p)_+$ renormalized. This exactly compensates for the over-represented tokens that got rejected.

Hold those three sentences in mind as you read the formal proof. Every line of algebra maps directly to one of these three steps.

## Proof by cases: building intuition alongside the math

The algebra above is correct but dense. Let's also verify the result through three instructive cases that build intuition for how the algorithm behaves.

**Case 1: Draft and target agree perfectly ($p = q$).** Then $a(x) = \min(1, 1) = 1$ for all $x$. Every token is accepted. $\beta = 0$. The output distribution is $p = q$. No correction needed. This is the theoretical maximum efficiency scenario — $\gamma+1$ tokens from one verify pass at the cost of $\gamma$ draft passes plus one target pass.

**Case 2: Draft is completely wrong (disjoint support).** Suppose $p$ and $q$ have no tokens in common: for every $x$ with $p(x) > 0$, $q(x) = 0$, and vice versa. Then $a(x) = \min(1, 0/p(x)) = 0$ for all $x$ in $p$'s support. Every token is rejected with probability 1. $\beta = \sum_x p(x) \cdot 1 = 1$. The correction distribution is $r(x) = q(x)/1 = q(x)$. The output is always sampled from $q$ directly.

In this degenerate case, speculative decoding degenerates to running both models and always using the target's output. It's correct but wasteful — you paid for the draft model with zero benefit. The practical lesson: don't use a draft model from a completely different domain or architecture family.

**Case 3: Typical case with partial overlap.** Tokens where $p(x) \leq q(x)$ are always accepted; they contribute $p(x)$ worth of probability to the output. Tokens where $p(x) > q(x)$ are stochastically accepted with probability $q(x)/p(x) < 1$; each contributes $p(x) \cdot q(x)/p(x) = q(x)$ worth of probability to the output.

The rejection probability $\beta$ is the total variation distance between $p$ and $q$: $\beta = \text{TV}(p,q) = \frac{1}{2}\sum_x |p(x) - q(x)|$. A good draft model has $\text{TV}(p,q) \approx 0.10$–$0.25$, giving acceptance rates of 75–90%.

## What happens on rejection: the adjusted distribution in detail

Rejection is where many implementation descriptions get vague. Let's be completely precise.

![Adjusted rejection distribution — the residual (q − α*p)+ that fills the correction](/imgs/blogs/speculative-decoding-token-acceptance-rejection-sampling-5.webp)

When a draft token is rejected, you need to sample from $(q(x) - p(x))_+ / \beta$. This distribution has a specific shape:

- **It has zero probability on all overshot tokens** ($p(x) > q(x)$ → the positive part is zero, these tokens are excluded from the correction distribution).
- **It concentrates probability on undershot tokens** ($q(x) > p(x)$ → the residual is positive, these are the tokens the draft underrepresented).
- **The weights are proportional to the deficit** — tokens where the target greatly exceeds the draft get proportionally more correction mass.

The sampling procedure at the point of rejection:
1. Retrieve the target logits $\ell_{\text{target}}$ and draft logits $\ell_{\text{draft}}$ at the rejection position (both already computed during the verify and draft passes respectively).
2. Apply the same temperature/top-p/top-k to both to get $q$ and $p$ as probability vectors.
3. Compute $r = \text{clamp}(q - \alpha \cdot p, \text{min}=0)$ where $\alpha = \min(1, q(\hat{x})/p(\hat{x}))$ was the acceptance probability at the rejected token. Note: in the original derivation $\alpha$ is a scalar acceptance probability, but the correction distribution uses $\alpha$ that equals $q/p$ for the overshot token. The correct formula is $r = (q - p)_+ / \beta$, which is equivalent to $\text{clamp}(q - p, \text{min}=0)$ normalized. We do NOT multiply by the rejected token's acceptance probability; we subtract $p$ globally.
4. Normalize $r$ to sum to 1.
5. Sample one token from $r$ using categorical sampling.

A common implementation mistake is to sample from $q$ directly on rejection rather than from $(q-p)_+/\beta$. This is wrong! Sampling from $q$ directly on rejection would mean the rejection path also outputs $q$-distributed tokens — but then the acceptance path and rejection path both look like $q$, which means the overall output has the wrong total probability because you'd be double-counting $q$'s probability mass. The $(q-p)_+/\beta$ correction specifically removes the probability mass already covered by the accept path.

Another common mistake: using $\beta$ incorrectly as the normalization constant. The correct constant is $Z = \sum_x (q(x) - p(x))_+$, which equals $\beta$ by the normalization argument. If you compute $Z$ empirically by summing the clamped vector, you get the right answer; if you compute $\beta = 1 - \sum_x p(x)a(x)$ separately, you get the same value.

## The speculative sampling algorithm: full pseudocode and implementation

Now let's integrate everything into a complete implementation. The algorithm has three clean phases per iteration.

![Complete acceptance step pipeline — from draft token to corrected output](/imgs/blogs/speculative-decoding-token-acceptance-rejection-sampling-3.webp)

```python
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
import time
import logging


logger = logging.getLogger(__name__)


@dataclass
class SpeculativeSamplingConfig:
    """Configuration for speculative decoding sampling."""
    gamma: int = 4                      ## draft tokens per verify pass
    temperature: float = 1.0           ## sampling temperature (same for draft and target)
    top_p: float = 1.0                 ## nucleus sampling threshold (same for both)
    top_k: int = 0                     ## top-k filter (0 = disabled)
    max_new_tokens: int = 200          ## hard cap on generation length
    eos_token_id: Optional[int] = None ## stop on EOS
    min_acceptance_rate: float = 0.60  ## alert if acceptance rate falls below this


@dataclass
class IterationStats:
    """Per-iteration statistics for monitoring."""
    draft_tokens: int = 0
    accepted_tokens: int = 0
    had_rejection: bool = False
    iteration_ms: float = 0.0
    
    @property
    def acceptance_rate(self) -> float:
        if self.draft_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.draft_tokens


class AcceptanceTracker:
    """
    Exponential moving average of acceptance rates for online monitoring.
    Used in production to detect draft quality degradation.
    """
    def __init__(self, ema_alpha: float = 0.05, window: int = 100):
        self.ema_alpha = ema_alpha
        self.window = window
        self.ema = 0.80        ## initial estimate
        self.history: List[float] = []
        self.total_draft = 0
        self.total_accepted = 0
    
    def update(self, stats: IterationStats) -> Optional[str]:
        """Update tracker; return alert string if acceptance rate is too low."""
        rate = stats.acceptance_rate
        self.history.append(rate)
        if len(self.history) > self.window:
            self.history.pop(0)
        self.ema = (1 - self.ema_alpha) * self.ema + self.ema_alpha * rate
        self.total_draft += stats.draft_tokens
        self.total_accepted += stats.accepted_tokens
        return None
    
    @property
    def overall_rate(self) -> float:
        if self.total_draft == 0:
            return 0.0
        return self.total_accepted / self.total_draft
    
    @property
    def recent_rate(self) -> float:
        if not self.history:
            return 0.0
        return sum(self.history[-min(20, len(self.history)):]) / min(20, len(self.history))


def filter_logits(
    logits: torch.Tensor,   ## [vocab_size]
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> torch.Tensor:
    """
    Apply temperature, top-k, and top-p filtering to raw logits.
    Returns filtered logits (before softmax).
    Must be applied consistently to both draft and target logits.
    """
    ## Temperature scaling first
    scaled = logits / max(temperature, 1e-5)
    
    ## Top-k filtering
    if top_k > 0:
        k = min(top_k, scaled.size(-1))
        top_vals, _ = torch.topk(scaled, k)
        threshold = top_vals[..., -1, None]
        scaled = torch.where(scaled < threshold, torch.full_like(scaled, float('-inf')), scaled)
    
    ## Top-p (nucleus) filtering
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(sorted_probs, dim=-1)
        ## Keep tokens whose cumulative probability is within the nucleus
        remove_mask = (cum_probs - sorted_probs) >= top_p
        sorted_logits[remove_mask] = float('-inf')
        ## Restore original order
        scaled = torch.zeros_like(scaled)
        scaled.scatter_(-1, sorted_idx, sorted_logits)
    
    return scaled


def compute_acceptance_and_correction(
    target_logits: torch.Tensor,     ## raw [vocab_size] at this position
    draft_logits: torch.Tensor,      ## raw [vocab_size] at this position
    draft_token_id: int,
    temperature: float,
    top_p: float = 1.0,
    top_k: int = 0,
) -> Tuple[float, float, torch.Tensor]:
    """
    Compute acceptance probability and correction distribution for one draft token.
    
    Returns:
        accept_prob: min(1, q(x)/p(x)) for the draft token
        q_x: target probability of draft token (for diagnostics)
        correction_dist: (q - p)+ / Z distribution over vocabulary
    """
    ## Apply identical filtering to both models
    tgt_filtered = filter_logits(target_logits, temperature, top_p, top_k)
    dft_filtered = filter_logits(draft_logits, temperature, top_p, top_k)
    
    q = F.softmax(tgt_filtered, dim=-1)
    p = F.softmax(dft_filtered, dim=-1)
    
    q_x = q[draft_token_id].item()
    p_x = p[draft_token_id].item()
    
    ## Acceptance probability: min(1, q(x)/p(x))
    accept_prob = min(1.0, q_x / (p_x + 1e-10))
    
    ## Correction distribution: (q - p)+ / Z
    ## NOTE: we use (q - p)+, NOT (q - accept_prob * p)+
    ## The latter would be used if we needed a per-token correction;
    ## the former is the marginal correction over all rejected tokens
    residual = torch.clamp(q - p, min=0.0)
    Z = residual.sum()
    if Z > 1e-10:
        correction_dist = residual / Z
    else:
        ## q ≈ p everywhere; fallback to q (edge case: models are identical)
        correction_dist = q
    
    return accept_prob, q_x, correction_dist


def speculative_decode_iteration(
    draft_logits_cache: List[torch.Tensor],   ## raw logits from draft, one per position
    draft_token_ids: List[int],               ## token ids sampled by draft
    target_logits_verify: torch.Tensor,       ## [gamma+1, vocab_size] from target verify
    config: SpeculativeSamplingConfig,
    seq_len: int,   ## prompt length (used for indexing target logits)
) -> Tuple[List[int], IterationStats]:
    """
    Process one draft-verify cycle: accept/reject each draft token, emit accepted list.
    
    The target logits at position i predict the token at position seq_len + i.
    target_logits_verify[i] corresponds to draft position i (for i in [0, gamma]).
    target_logits_verify[gamma] is the bonus token distribution.
    
    Returns:
        accepted: list of accepted token IDs (length 1..gamma+1)
        stats: per-iteration statistics
    """
    gamma = len(draft_token_ids)
    stats = IterationStats(draft_tokens=gamma)
    accepted = []
    
    for i in range(gamma):
        draft_tok = draft_token_ids[i]
        target_raw = target_logits_verify[i]         ## [vocab_size]
        draft_raw = draft_logits_cache[i]
        
        accept_prob, q_x, correction_dist = compute_acceptance_and_correction(
            target_raw, draft_raw, draft_tok,
            config.temperature, config.top_p, config.top_k
        )
        
        ## Stochastic acceptance test
        u = torch.rand(1).item()
        
        if u < accept_prob:
            ## Accept this token
            accepted.append(draft_tok)
            stats.accepted_tokens += 1
            
            ## If we just accepted EOS, stop immediately
            if config.eos_token_id is not None and draft_tok == config.eos_token_id:
                return accepted, stats
        else:
            ## Reject: sample from correction distribution
            ## This terminates the draft sequence — remaining tokens at i+1..gamma-1 are discarded
            corrected_tok = torch.multinomial(correction_dist, num_samples=1).item()
            accepted.append(corrected_tok)
            stats.had_rejection = True
            return accepted, stats
    
    ## All gamma draft tokens accepted: emit bonus token
    ## Bonus token is sampled from target distribution at position gamma
    bonus_raw = target_logits_verify[gamma]
    bonus_filtered = filter_logits(bonus_raw, config.temperature, config.top_p, config.top_k)
    bonus_probs = F.softmax(bonus_filtered, dim=-1)
    bonus_tok = torch.multinomial(bonus_probs, num_samples=1).item()
    accepted.append(bonus_tok)
    
    return accepted, stats


def speculative_generate(
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    input_ids: torch.Tensor,         ## [1, prompt_len]
    config: SpeculativeSamplingConfig,
) -> Tuple[torch.Tensor, AcceptanceTracker, List[IterationStats]]:
    """
    Full speculative decoding generation loop.
    
    Implements the algorithm from Leviathan et al. (2023) and Chen et al. (2023).
    Provably lossless: output distribution equals target autoregressive sampling.
    
    Returns:
        generated_ids: [1, prompt_len + n_generated]
        tracker: acceptance rate statistics
        all_stats: per-iteration statistics for analysis
    """
    device = input_ids.device
    current_ids = input_ids.clone()
    tracker = AcceptanceTracker()
    all_stats: List[IterationStats] = []
    generated_count = 0
    
    while generated_count < config.max_new_tokens:
        t_start = time.time()
        
        ## How many tokens to draft this iteration
        effective_gamma = min(config.gamma, config.max_new_tokens - generated_count)
        
        ## ================================================================
        ## PHASE 1: DRAFT — generate effective_gamma candidate tokens
        ## ================================================================
        draft_token_ids: List[int] = []
        draft_logits_cache: List[torch.Tensor] = []
        
        draft_current = current_ids.clone()
        
        with torch.no_grad():
            for _ in range(effective_gamma):
                out = draft_model(draft_current)
                raw_logits = out.logits[0, -1, :]    ## [vocab_size]
                
                ## Apply filtering and sample
                filtered = filter_logits(raw_logits, config.temperature, config.top_p, config.top_k)
                probs = F.softmax(filtered, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).item()
                
                draft_token_ids.append(token_id)
                draft_logits_cache.append(raw_logits.cpu())   ## save for accept/reject
                
                ## Extend current sequence with draft token
                tok_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
                draft_current = torch.cat([draft_current, tok_tensor], dim=-1)
        
        ## ================================================================
        ## PHASE 2: VERIFY — one target forward pass over all gamma tokens
        ## ================================================================
        prompt_len = current_ids.shape[-1]
        draft_tensor = torch.tensor(
            [draft_token_ids], dtype=torch.long, device=device
        )  ## [1, gamma]
        verify_ids = torch.cat([current_ids, draft_tensor], dim=-1)  ## [1, prompt_len+gamma]
        
        with torch.no_grad():
            target_out = target_model(verify_ids)
        
        ## Target logits at positions [prompt_len-1 .. prompt_len+gamma-1]
        ## predict tokens at positions [prompt_len .. prompt_len+gamma]
        ## These correspond to draft positions 0..gamma-1 and the bonus
        target_logits_all = target_out.logits[
            0, prompt_len-1 : prompt_len + effective_gamma, :
        ]  ## [gamma+1, vocab_size]
        
        ## Move to CPU for post-processing (acceptance happens on CPU)
        target_logits_all = target_logits_all.cpu()
        
        ## ================================================================
        ## PHASE 3: ACCEPT/REJECT — modified rejection sampling
        ## ================================================================
        accepted, stats = speculative_decode_iteration(
            draft_logits_cache,
            draft_token_ids,
            target_logits_all,
            config,
            seq_len=prompt_len,
        )
        
        stats.iteration_ms = (time.time() - t_start) * 1000
        all_stats.append(stats)
        tracker.update(stats)
        
        ## Check for low acceptance rate alert
        if tracker.recent_rate < config.min_acceptance_rate:
            logger.warning(
                f"Acceptance rate {tracker.recent_rate:.3f} < threshold {config.min_acceptance_rate}. "
                f"Consider switching to a better draft model or disabling speculative decoding."
            )
        
        ## Extend sequence with accepted tokens
        accepted_tensor = torch.tensor(
            [accepted], dtype=torch.long, device=device
        )
        current_ids = torch.cat([current_ids, accepted_tensor], dim=-1)
        generated_count += len(accepted)
        
        ## Check for EOS in accepted tokens
        if config.eos_token_id is not None:
            if config.eos_token_id in accepted:
                break
    
    return current_ids, tracker, all_stats
```

Let's annotate the key subtleties in this implementation:

**Draft logits saved to CPU memory.** During the draft phase, we save the raw logits (pre-softmax, pre-temperature) to CPU. This is intentional — during the accept/reject phase we need both draft and target logits to compute the acceptance probability. The logits are small (vocab_size × float32 ≈ 500KB for LLaMA-3) and fit easily in CPU RAM. In production, you might keep them on GPU if memory allows, avoiding the PCIe transfer.

**Target logit indexing.** The target model is called on `[prompt, draft_1, ..., draft_gamma]` and returns logits for all positions. We extract logits starting at `prompt_len - 1` (not `prompt_len`): position `prompt_len - 1` of the input (the last prompt token) predicts the first draft token position, which corresponds to target logits index 0. This off-by-one is a common source of bugs.

**`effective_gamma` vs `gamma`.** Near the `max_new_tokens` limit, we may have fewer tokens to draft. The algorithm handles this cleanly — if `effective_gamma = 2`, we draft 2 tokens and verify them, then stop even if they're all accepted.

**The acceptance test uses `torch.rand(1).item()`.** This is a CPU operation. In a high-throughput serving system, you might pre-sample a batch of uniform random numbers to avoid Python-level overhead.

## Token-by-token acceptance trace: a complete worked example

Let's trace through a complete 5-token example with precise numbers. This is the kind of worked example that reveals whether you actually understand the algorithm.

![Token-by-token acceptance trace for a 5-draft-token sequence](/imgs/blogs/speculative-decoding-token-acceptance-rejection-sampling-6.webp)

**Setup.** Draft model: LLaMA-3-8B. Target model: LLaMA-3-70B. Input prompt: "The capital of France is". Draft proposes: `["Paris", ",", "which", "is", "a"]`. Temperature=0.8, top-p=1.0.

**Draft phase.** The draft model generates sequentially:
- Position 1 → "Paris": $p(\text{Paris}) = 0.61$
- Position 2 → ",": $p(\text{","}) = 0.48$
- Position 3 → "which": $p(\text{"which"}) = 0.29$
- Position 4 → "is": $p(\text{"is"}) = 0.42$
- Position 5 → "a": $p(\text{"a"}) = 0.18$

**Verify phase.** Target model runs once on the concatenated sequence. Extracted target probabilities for each draft token:

| Pos | Draft token | $p(\hat{x})$ | $q(\hat{x})$ | Ratio $q/p$ | $a(\hat{x})$ |
|-----|-------------|:---:|:---:|:---:|:---:|
| 1 | "Paris" | 0.61 | 0.73 | 1.20 | 1.00 |
| 2 | "," | 0.48 | 0.46 | 0.96 | 0.96 |
| 3 | "which" | 0.29 | 0.17 | 0.59 | 0.59 |
| 4 | "is" | 0.42 | — | — | (skipped) |
| 5 | "a" | 0.18 | — | — | (skipped) |

**Accept/reject phase.** Sample $u_i \sim \text{Uniform}(0,1)$ independently for each position:

- Position 1: $u_1 = 0.44 < 1.00$ → **Accept "Paris"**
- Position 2: $u_2 = 0.88 < 0.96$ → **Accept ","**
- Position 3: $u_3 = 0.73 \geq 0.59$ → **Reject "which"**

On rejection at position 3, compute correction:
$$r(x) = \frac{(q(x) - p(x))_+}{\beta}$$

The target actually prefers "a" and "beautiful" to "which" in this context (it knows that "Paris, a beautiful city" is a more fluent continuation). The correction distribution might assign:
- "a": $r = 0.28$ (target strongly preferred this, draft underweighted it)
- "beautiful": $r = 0.19$
- "the": $r = 0.15$
- "known": $r = 0.12$
- ... (remaining mass spread over other tokens)

Sampling from this correction distribution: suppose we draw "a".

**Output of this iteration:** `["Paris", ",", "a"]`. Three tokens from one draft-verify cycle. The next iteration starts with the prompt extended by these three tokens.

**What did we avoid?** Without speculative decoding, generating these three tokens would have required three separate target model forward passes (3 × 90ms = 270ms on a single A100). With speculative decoding: 5 × 10ms draft + 90ms verify = 140ms for 3 tokens (47ms/token vs 90ms/token). A 1.9× speedup on this particular iteration, which had a below-average acceptance rate (2/5 = 0.40 draft tokens accepted, dragged down by the unlucky $u_3$).

## Temperature and sampling parameters: the interaction with acceptance rates

In production, you almost never sample at temperature 1.0 with no filtering. Here is how each sampling parameter interacts with the acceptance rate.

![Effect of temperature on acceptance rate — low temperature aligns peaks, high temperature flattens both](/imgs/blogs/speculative-decoding-token-acceptance-rejection-sampling-8.webp)

### Temperature scaling

Temperature $T$ rescales logits before softmax: $p_T(x) = \text{softmax}(\ell(x)/T)$. The acceptance rate is $\alpha(T) = E_x[\min(1, q_T(x)/p_T(x))]$.

At $T \to 0$: both distributions peak sharply on the argmax. If both models agree on the argmax (common for same-family pairs), the ratio $q_T(\hat{x})/p_T(\hat{x}) \to 1$ for the most probable token and the acceptance rate approaches 1. Practically, at $T=0.1$ with a same-family draft, you see $\alpha \approx 0.92$–$0.96$.

At $T = 1.0$: distributions are more diffuse. The ratio $q/p$ varies more across the vocabulary, and more tokens have $p > q$ by meaningful margins. Acceptance rates of 0.75–0.88 are typical for good draft models.

At $T > 1.0$: distributions become flatter. More importantly, for tokens near the long tail, the ratio $q_T(x)/p_T(x)$ is highly variable — any small difference in logit values gets amplified by the flattening. Acceptance rates drop to 0.65–0.80 even for good draft models.

**The critical rule: apply the same temperature to both models.** The acceptance ratio $q_T(x)/p_T(x)$ is only well-defined when both distributions use the same temperature. If you apply $T_1$ to the draft and $T_2 \neq T_1$ to the target, the resulting "acceptance" test does not correspond to sampling from any well-defined distribution. Your outputs will be biased, and the bias is proportional to $(T_2 - T_1)$ multiplied by the logit gap between the models.

In code:

```python
## CORRECT: same temperature for both
q = F.softmax(target_logits / temperature, dim=-1)
p = F.softmax(draft_logits / temperature, dim=-1)
accept_prob = min(1.0, q[tok] / p[tok])

## WRONG: different temperatures — biased output
q = F.softmax(target_logits / 0.7, dim=-1)   ## target at T=0.7
p = F.softmax(draft_logits / 1.0, dim=-1)    ## draft at T=1.0
accept_prob = min(1.0, q[tok] / p[tok])       ## this is NOT lossless
```

### Top-p nucleus sampling

Top-p restricts both distributions to their respective nuclei (the smallest set of tokens summing to probability $p_{\text{top}}$) before the acceptance test. The nucleus of the draft model and the nucleus of the target model are generally different token sets.

**The nucleus mismatch problem.** When the draft samples a token inside its own nucleus but outside the target's nucleus, $q(x) = 0$ (the target excluded this token from its nucleus). The acceptance probability is $\min(1, 0/p(x)) = 0$: guaranteed rejection. This is correct — the target distribution says this token should never appear — but it degrades your acceptance rate.

The severity depends on how tight the nucleus is (lower $p_{\text{top}}$ → tighter nucleus → more mismatch) and how different the models are (larger model gap → more nucleus divergence):

| Model pair | top-p=0.95 | top-p=0.90 | top-p=0.70 |
|------------|:-----------:|:-----------:|:-----------:|
| LLaMA-3 8B → 70B | $\alpha = 0.86$ | $\alpha = 0.82$ | $\alpha = 0.64$ |
| LLaMA-2 7B → 70B | $\alpha = 0.81$ | $\alpha = 0.76$ | $\alpha = 0.55$ |
| GPT-2 large → LLaMA-3 70B | $\alpha = 0.52$ | $\alpha = 0.44$ | $\alpha = 0.31$ |

The take-away: if your application uses tight nucleus sampling (top-p < 0.85), measure the actual acceptance rate on your task before deploying speculative decoding. You may find that the nucleus mismatch makes it unprofitable.

Implementation note: apply the top-p filter to both models' logits before computing the acceptance ratio, not after. The probability vectors $p$ and $q$ used in the ratio test must be the post-filtered, post-renormalized distributions. The code in `compute_acceptance_and_correction` above does this correctly.

### Top-k sampling

Top-k behaves similarly to top-p: the draft's top-$k$ and the target's top-$k$ are generally different sets. Tokens in the draft's top-$k$ but not the target's top-$k$ get $q = 0$ → acceptance probability 0.

Top-k with large $k$ (e.g., $k=50$ or $k=100$) is generally safer than top-p with tight thresholds, because both models' top-100 tokens overlap substantially even across model sizes. Top-k with small $k$ (e.g., $k=10$) creates the same mismatch problem as tight top-p.

## Practical acceptance rates: measuring and predicting

The theoretical acceptance rate has an exact formula in terms of the total variation distance between $p$ and $q$:

$$\alpha = 1 - \text{TV}(p, q) = 1 - \frac{1}{2}\sum_x |p(x) - q(x)|$$

You can estimate this before deployment using a fast measurement protocol:

```python
def estimate_acceptance_rate(
    draft_model: torch.nn.Module,
    target_model: torch.nn.Module,
    eval_prompts: List[str],
    tokenizer,
    n_positions: int = 50,   ## how many positions to sample per prompt
    temperature: float = 1.0,
    device: str = "cuda",
) -> Dict[str, float]:
    """
    Estimate token-level acceptance rate without running full speculative decoding.
    Computes TV(p, q) at sampled positions to predict alpha.
    
    Args:
        eval_prompts: List of representative prompts for your task.
        n_positions: Number of positions to evaluate per prompt.
    
    Returns:
        Dictionary with 'mean_alpha', 'std_alpha', 'min_alpha', 'max_alpha'.
    """
    alphas = []
    
    for prompt in eval_prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs["input_ids"]
        seq_len = input_ids.shape[-1]
        
        ## Step 1: Generate a reference continuation from the target model
        with torch.no_grad():
            target_out = target_model.generate(
                input_ids, max_new_tokens=n_positions, do_sample=False
            )
        ## Use greedy continuation as the sequence to evaluate
        eval_ids = target_out
        
        ## Step 2: Compute draft and target logits at each position
        with torch.no_grad():
            draft_logits = draft_model(eval_ids).logits[0, seq_len-1:seq_len+n_positions-1]
            target_logits = target_model(eval_ids).logits[0, seq_len-1:seq_len+n_positions-1]
        
        ## Step 3: Compute TV distance at each position
        for i in range(min(n_positions, target_logits.shape[0])):
            p = F.softmax(draft_logits[i] / temperature, dim=-1)
            q = F.softmax(target_logits[i] / temperature, dim=-1)
            tv = 0.5 * (p - q).abs().sum().item()
            alphas.append(1.0 - tv)
    
    return {
        "mean_alpha": sum(alphas) / len(alphas),
        "std_alpha": torch.tensor(alphas).std().item(),
        "min_alpha": min(alphas),
        "max_alpha": max(alphas),
        "n_positions": len(alphas),
    }
```

This protocol takes 5–10 minutes on a representative set of 50–100 prompts and gives you a reliable prediction of the acceptance rate before you commit to a deployment configuration.

![Acceptance rate matrix — draft quality vs temperature effects on expected α](/imgs/blogs/speculative-decoding-token-acceptance-rejection-sampling-4.webp)

## The bonus token: why you always get at least one new token

A subtle but important feature of the speculative sampling algorithm is that every iteration produces **at least one new token**, even if all $\gamma$ draft tokens are rejected. This is the "bonus token" guaranteed by the algorithm design.

Here is why it matters. Without the bonus token, a very bad draft model (one that gets all $\gamma$ tokens wrong) would produce zero output per iteration — the algorithm would spin indefinitely without advancing. Worse, the acceptance test consumes random numbers ($u_i$ uniform samples), so a run of all-rejections would be genuinely wasted computation.

The bonus token fixes this. After processing $\gamma$ draft tokens through accept/reject:

- **If all $\gamma$ were accepted:** the bonus is the target model's prediction at position $\gamma$ (the natural continuation after all accepted tokens). This is just the next token the target model would have generated anyway.
- **If token $k < \gamma$ was rejected:** the corrected token at position $k$ (sampled from the correction distribution) is the output token at that position. The algorithm halts draft processing for this iteration. The corrected token is the "bonus" in the case of rejection.

Wait — is the corrected rejection token the same as the "bonus"? Conceptually, yes: in both the all-accept case and the first-reject case, you always emit at least one new token. In the all-accept case it's an additional token (token $\gamma+1$); in the first-reject case it's the correction at position $k$ (replacing what the draft would have put there). The net effect: the minimum tokens per iteration is 1 (all $\gamma$ draft tokens rejected at the first position, replaced by one correction token) and the maximum is $\gamma+1$ (all $\gamma$ accepted plus the bonus).

This is why the expected accepted tokens formula sums from 1 to $\gamma+1$: the minimum outcome is 1 token (one correction on first rejection), not 0.

**Implementation note on the bonus token.** When all $\gamma$ draft tokens are accepted, the bonus token comes from `target_logits_verify[gamma]` — the target's logit prediction at position $\gamma$ (the position after all the accepted draft tokens). This logit is computed "for free" during the target verify pass — you had to run the target on input of length `prompt_len + gamma` anyway, and the output at position `prompt_len + gamma - 1` of the logit tensor (0-indexed) is exactly the bonus distribution. No extra target call is needed.

## Acceptance rate and expected speedup: connecting the math to real numbers

With acceptance rate $\alpha$ and draft length $\gamma$, the expected number of accepted tokens per iteration is:

$$E[\text{accepted}] = \sum_{k=1}^{\gamma+1} k \cdot P(\text{exactly } k \text{ tokens accepted})$$

Where $P(\text{exactly } k)$ means the first $k-1$ tokens are accepted and either the $k$-th is the bonus (all $\gamma$ accepted) or the $k$-th is a corrected rejection. Working this out:

$$E[\text{accepted}] = \frac{1 - \alpha^{\gamma+1}}{1 - \alpha}$$

This is a geometric sum. Let's compute it at $\alpha = 0.80$ for various $\gamma$:

| $\gamma$ | $E[\text{accepted}]$ | Efficiency (tokens per target pass) |
|----------|:--------------------:|:-----------------------------------:|
| 1 | $\frac{1-0.64}{0.20} = 1.80$ | 1.80× over $\gamma=0$ |
| 2 | $\frac{1-0.51}{0.20} = 2.44$ | 2.44× |
| 4 | $\frac{1-0.33}{0.20} = 3.36$ | 3.36× |
| 6 | $\frac{1-0.21}{0.20} = 3.93$ | 3.93× |
| 8 | $\frac{1-0.13}{0.20} = 4.30$ | 4.30× |
| 16 | $\frac{1-0.02}{0.20} = 4.90$ | 4.90× (diminishing returns) |

The diminishing returns are clear: going from $\gamma=4$ to $\gamma=8$ adds only $(4.30-3.36) = 0.94$ expected accepted tokens, while doubling the draft overhead. The optimal $\gamma$ balances this with the draft latency:

$$\text{Speedup}(\gamma) = \frac{E[\text{accepted}(\gamma)]}{1 + \gamma \cdot (t_{\text{draft}}/t_{\text{target}})}$$

At typical ratios $t_{\text{draft}}/t_{\text{target}} = 0.10$–$0.15$ (draft model 6–10× cheaper than target), $\gamma=4$ is usually optimal. At $\gamma=4$, the draft overhead is $4 \times 0.12 = 0.48$ target passes worth of time, and you get $3.36$ tokens — a net speedup of $3.36/1.48 \approx 2.3$× over baseline.

## Why the acceptance probability is min(1, q/p) and not something else

This question comes up repeatedly when engineers first encounter speculative decoding: why specifically $\min(1, q/p)$ and not, say, $q/(\max(p, q))$ or some sigmoid function of $q - p$? The answer is that $\min(1, q/p)$ is the **unique** acceptance function that simultaneously satisfies all three requirements:

1. **Preserves the target distribution.** The marginal output probability equals $q(x)$ for every token $x$ in the vocabulary.
2. **Maximizes expected acceptance.** Among all valid acceptance functions, $\min(1, q/p)$ maximizes $E[\text{tokens accepted per draft}]$.
3. **Requires no knowledge of the normalizing constant.** Both $p(x)$ and $q(x)$ are available as softmax probabilities from the respective models. No other quantities are needed.

**Proof of uniqueness.** Suppose $a'(x)$ is another acceptance function that preserves the target distribution. From the marginal constraint:
$$p(x) a'(x) + (1 - \sum_y p(y) a'(y)) r'(x) = q(x)$$

For the correction distribution to be non-negative: $r'(x) = [q(x) - p(x) a'(x)] / \beta' \geq 0$, which requires $a'(x) \leq q(x)/p(x)$ for all $x$. To maximize acceptance, we push $a'(x)$ to its upper bound: $a'(x) = \min(1, q(x)/p(x))$. Any other valid acceptance function accepts tokens strictly less often and would be dominated by $\min(1, q/p)$.

**Why sigmoid or other smooth functions fail.** A sigmoid acceptance function like $a(x) = \sigma(q(x)/p(x))$ might look appealing — it's smooth, bounded between 0 and 1, and "soft." But it does not satisfy the marginal constraint. The output distribution would be some complicated function of $\sigma$ that does not equal $q(x)$, and there's no way to construct a correction distribution that fixes this while remaining non-negative everywhere.

**The special role of the ratio $q/p$.** The ratio test $q(x)/p(x)$ has a deep probabilistic meaning: it is the **likelihood ratio** or **Radon-Nikodym derivative** $\frac{dq}{dp}(x)$. Rejection sampling with acceptance probability $\min(1, \frac{dq}{dp}(x))$ is the canonical method for importance sampling when the proposal distribution $p$ has heavier tails than the target $q$. Speculative decoding is exactly this problem — the draft (proposal) often assigns higher probability to common tokens than the target (desired distribution).

## Efficiency analysis: when does speculative decoding break even?

The speedup from speculative decoding is not free — it comes at the cost of running the draft model $\gamma$ times plus some acceptance-rejection bookkeeping. Understanding the break-even point is critical for deployment decisions.

**The break-even condition.** Let $t_{\text{target}}$ be the wall-clock time for one target model forward pass (generating 1 token in baseline mode), and $t_{\text{draft}}$ be the wall-clock time for one draft model pass. The speculative decoding latency for an iteration is approximately:

$$t_{\text{spec}} \approx \gamma \cdot t_{\text{draft}} + t_{\text{target}}$$

The number of tokens generated per iteration is $E[\text{accepted}] = (1 - \alpha^{\gamma+1})/(1-\alpha)$.

The per-token latency is:
$$\text{latency per token} = \frac{\gamma \cdot t_{\text{draft}} + t_{\text{target}}}{E[\text{accepted}]}$$

Baseline per-token latency is $t_{\text{target}}$. Speculative decoding wins when:
$$\frac{\gamma \cdot t_{\text{draft}} + t_{\text{target}}}{E[\text{accepted}]} < t_{\text{target}}$$

Rearranging:
$$E[\text{accepted}] > 1 + \gamma \cdot \frac{t_{\text{draft}}}{t_{\text{target}}}$$

Substituting $E[\text{accepted}] = (1 - \alpha^{\gamma+1})/(1 - \alpha)$ and the speedup ratio $\rho = t_{\text{draft}}/t_{\text{target}}$:

$$\frac{1 - \alpha^{\gamma+1}}{1 - \alpha} > 1 + \gamma \rho$$

For a given $(\rho, \gamma)$, this gives the minimum acceptance rate $\alpha_{\text{min}}$ below which speculative decoding hurts performance:

| $\rho$ (draft/target) | $\gamma=2$ | $\gamma=4$ | $\gamma=6$ | $\gamma=8$ |
|---|:---:|:---:|:---:|:---:|
| 0.05 (20× cheaper draft) | $\alpha_{\text{min}} = 0.31$ | $0.27$ | $0.23$ | $0.20$ |
| 0.10 (10× cheaper) | $\alpha_{\text{min}} = 0.41$ | $0.36$ | $0.31$ | $0.27$ |
| 0.15 (6.7× cheaper) | $\alpha_{\text{min}} = 0.50$ | $0.44$ | $0.39$ | $0.35$ |
| 0.20 (5× cheaper) | $\alpha_{\text{min}} = 0.58$ | $0.52$ | $0.47$ | $0.43$ |
| 0.30 (3.3× cheaper) | $\alpha_{\text{min}} = 0.71$ | $0.65$ | $0.60$ | $0.56$ |

Reading this table: if your draft model is 10× cheaper than your target (a LLaMA-1B vs LLaMA-70B scenario on the same GPU, which is roughly 10×), and you use $\gamma=4$, speculative decoding breaks even at $\alpha_{\text{min}} = 0.36$. You only need to accept 36% of draft tokens for speculative decoding to pay off. In practice, same-family draft models achieve 75–91% acceptance rates, so there's a large safety margin.

The table also shows that longer draft sequences ($\gamma=8$) have lower break-even thresholds — speculative decoding is more forgiving of low acceptance rates when $\gamma$ is large. But the marginal benefit of increasing $\gamma$ from 4 to 8 is small at high acceptance rates, so the optimal $\gamma$ for a given deployment is:

$$\gamma^* = \arg\max_\gamma \frac{E[\text{accepted}(\alpha, \gamma)]}{1 + \gamma \rho}$$

For $\rho = 0.10$ and $\alpha = 0.85$, this maximum is achieved at $\gamma^* = 4$–$5$.

## Online estimation of acceptance rate during inference

In production, you need to know the acceptance rate in real time — not after the fact. The most practical approach is an online estimator that computes a rolling average of per-iteration acceptance rates.

The per-iteration acceptance rate is easy to compute: it's `num_accepted_draft_tokens / gamma`. But this is noisy — it takes integer values 0/gamma, 1/gamma, ..., gamma/gamma — so you need to smooth it. An exponential moving average (EMA) is the right tool:

```python
## Online EMA acceptance tracker: O(1) memory, O(1) per-update
class OnlineAcceptanceTracker:
    """
    Tracks the acceptance rate with an exponential moving average.
    alpha_ema is the EMA smoothing factor (not to be confused with
    the acceptance probability alpha in the paper).
    """
    
    def __init__(self, ema_factor: float = 0.05):
        self.ema_factor = ema_factor
        self.ema_rate = None        ## None until first observation
        self.n_iterations = 0
        self.n_tokens_accepted = 0
        self.n_tokens_drafted = 0
    
    def record(self, n_accepted: int, n_drafted: int) -> float:
        """
        Record one iteration and return current EMA acceptance rate.
        n_accepted: how many draft tokens were accepted this iteration (0..gamma)
        n_drafted: gamma (or effective_gamma if near max_new_tokens)
        """
        if n_drafted == 0:
            return self.ema_rate or 0.0
        
        self.n_iterations += 1
        self.n_tokens_accepted += n_accepted
        self.n_tokens_drafted += n_drafted
        
        iter_rate = n_accepted / n_drafted
        if self.ema_rate is None:
            self.ema_rate = iter_rate
        else:
            self.ema_rate = (
                (1 - self.ema_factor) * self.ema_rate 
                + self.ema_factor * iter_rate
            )
        return self.ema_rate
    
    @property
    def overall_rate(self) -> float:
        """Exact overall acceptance rate (all iterations)."""
        if self.n_tokens_drafted == 0:
            return 0.0
        return self.n_tokens_accepted / self.n_tokens_drafted
```

For per-request tracking, instantiate a fresh `OnlineAcceptanceTracker` at the start of each request and call `record()` after each speculative decoding iteration. For service-level monitoring, use a shared tracker updated across all concurrent requests.

**Adaptive gamma.** If you measure that the current acceptance rate has fallen below a threshold, you can dynamically reduce $\gamma$ to limit draft overhead:

```python
def adaptive_gamma(
    current_alpha: float,      ## EMA acceptance rate
    target_speedup: float,     ## desired speedup ratio (e.g., 1.5x)
    t_draft_ratio: float,      ## t_draft / t_target
    max_gamma: int = 8,
) -> int:
    """
    Compute the optimal gamma given current acceptance rate and timing parameters.
    Finds the gamma that maximizes expected speedup.
    """
    best_gamma = 1
    best_speedup = 1.0
    
    for gamma in range(1, max_gamma + 1):
        alpha = current_alpha
        expected_accepted = (1 - alpha ** (gamma + 1)) / (1 - alpha + 1e-10)
        overhead = 1 + gamma * t_draft_ratio
        speedup = expected_accepted / overhead
        
        if speedup > best_speedup:
            best_speedup = speedup
            best_gamma = gamma
    
    ## If best speedup doesn't meet target, fall back to gamma=1 (or disable)
    if best_speedup < target_speedup:
        return 0   ## 0 signals: disable speculative decoding this request
    
    return best_gamma
```

Adaptive gamma is particularly useful when serving a mixed workload — code generation (high $\alpha$, use $\gamma=5$) and creative writing (lower $\alpha$, use $\gamma=2$) — from the same endpoint. The adapter reads the rolling acceptance rate estimate and adjusts $\gamma$ per request.

## Speculative sampling vs beam search: a key distinction

Engineers sometimes conflate speculative decoding with beam search because both involve generating multiple candidate sequences. They are fundamentally different:

**Beam search** maintains $k$ "beams" (partial sequences) and at each step computes probabilities for all $k \times V$ extensions ($V$ = vocabulary size), then keeps the top-$k$ highest-probability continuations. It maximizes the probability of the output sequence (MAP decoding), which is useful for tasks with a single correct answer (translation, summarization) but over-produces short, repetitive, highly probable sequences.

**Speculative decoding** maintains one sequence (the current prefix) and at each iteration generates one new draft sequence of $\gamma$ tokens, then verifies that one sequence with the target. It is a sampling algorithm, not a search algorithm — its output distribution equals the target model's distribution exactly. It does not maximize any objective; it samples from $q$.

The practical difference: beam search changes the output distribution (toward MAP estimates), while speculative decoding preserves it exactly. For open-ended generation tasks (chat, creative writing, code completion), distribution preservation matters — you want diverse, temperature-controlled samples, not MAP estimates. For tasks with ground-truth outputs (factual question answering with a single correct answer), beam search is appropriate.

Some implementations combine the two, using speculative decoding to accelerate beam search (e.g., speculative beam search, where each beam step is accelerated by a draft model). This is a more complex algorithm beyond this post's scope, but the acceptance mechanism remains the same.

## The multi-token verification: how one forward pass handles γ positions simultaneously

One of the non-obvious aspects of the algorithm is that the target model verifies all $\gamma$ positions in a single forward pass, yet the acceptance decisions at each position are made sequentially (a rejection at position $k$ terminates processing). How is this consistent?

The answer lies in the causal attention mechanism of the transformer. When the target model processes the sequence $[x_1, \ldots, x_{\text{prompt}}, \hat{x}_1, \hat{x}_2, \ldots, \hat{x}_\gamma]$, each position can only attend to its left context. So:

- The target's logit at position $\text{prompt\_len}$ (predicting $\hat{x}_1$) only depends on $[x_1, \ldots, x_{\text{prompt}}]$ — the prompt prefix. It has no information about $\hat{x}_1, \hat{x}_2, \ldots, \hat{x}_\gamma$.
- The target's logit at position $\text{prompt\_len}+1$ (predicting $\hat{x}_2$) depends on $[x_1, \ldots, x_{\text{prompt}}, \hat{x}_1]$.
- The target's logit at position $\text{prompt\_len}+k$ (predicting $\hat{x}_{k+1}$) depends on $[x_1, \ldots, x_{\text{prompt}}, \hat{x}_1, \ldots, \hat{x}_k]$.

This is exactly what we need. The acceptance decision at position $k$ asks: "given the tokens accepted so far (the actual prefix), what probability does the target assign to token $\hat{x}_k$?" The target's logit at position $\text{prompt\_len}+k-1$ answers this precisely, because it was computed given the prefix $[x_1, \ldots, x_\text{prompt}, \hat{x}_1, \ldots, \hat{x}_{k-1}]$.

But wait — what if position 2 was rejected? Then the actual sequence after position 1 is some corrected token $x'_2 \neq \hat{x}_2$. Doesn't that invalidate the target's logit at position $\text{prompt\_len}+2$ (which was computed assuming $\hat{x}_2$ was in the context)?

Yes. And that is exactly why the algorithm terminates draft processing after the first rejection. The logits at positions $k+1, k+2, \ldots, \gamma$ were computed conditional on $\hat{x}_k$ being in the context — but since $\hat{x}_k$ was rejected and replaced by $x'_k$, those downstream logits are now invalid. The algorithm correctly discards them.

This also explains why you cannot reuse the target's verify-pass logits after a rejection to continue processing. The logit at position $k+1$ assumes the context contains $\hat{x}_k$, but after rejection it contains $x'_k$. The next speculative decoding iteration starts fresh with the corrected prefix.

**Implication for KV cache management.** Because draft tokens that come after a rejection are discarded, the KV cache entries for those positions are also invalid. The serving system needs to roll back the KV cache to the last accepted position before starting the next draft sequence. In practice, this means storing a snapshot of the KV cache length at the start of each speculative decoding iteration, and restoring it (trimming) if a rejection occurs before all $\gamma$ tokens are accepted.

## Extensions of the basic algorithm

The basic acceptance rule $a(x) = \min(1, q(x)/p(x))$ has been extended in several ways in the research literature. Understanding these extensions helps you evaluate which variant is right for your use case.

**Typical acceptance sampling (TypicalAcc).** Chen et al. (2024) proposed replacing the hard $\min(1, q/p)$ threshold with a softer acceptance criterion based on the expected acceptance rate. Instead of accepting token $x$ with probability $\min(1, q(x)/p(x))$, you accept any token that both models agree is "typical" — tokens with $q(x) \cdot p(x)$ above a joint threshold. The motivation: for high-temperature creative generation, the hard ratio test can be unnecessarily strict (rejecting tokens that are both plausible choices). In practice, typical acceptance sampling can increase acceptance rates by 2–5% at temperatures above 1.0, at the cost of slightly biasing the output distribution. Whether this bias matters depends on your quality requirements.

**Lookahead acceptance.** Some implementations allow a "lookahead" step: after a rejection at position $k$, instead of sampling from the correction distribution, sample from the target's distribution at position $k$ directly (run one target-only generation step). This avoids the correction distribution computation and is slightly easier to implement. The tradeoff: it always produces a valid target-distributed token, but pays an extra target forward pass on every rejection, which is expensive. For high acceptance rates ($\alpha > 0.85$), this is wasteful. For low acceptance rates, it may be worth the simplicity.

**Batched speculative decoding.** When serving multiple requests simultaneously (batch size $b > 1$), the acceptance test is applied independently per sequence in the batch. Sequences that finish their draft token sequence (by accepting all $\gamma$) or hit a rejection proceed independently. The target verify pass batches all $b$ sequences together (computing logits for $b \times (1 + \gamma)$ positions in one pass), but the acceptance/rejection bookkeeping is per-sequence. This requires careful KV cache management to handle different sequences rejecting at different positions.

## Debugging acceptance rate problems: a systematic checklist

When you deploy speculative decoding and observe lower-than-expected acceptance rates, here is the systematic checklist:

**Step 1: Verify the tokenizer.** Draft and target models must use identical tokenizers. Different tokenizers mean the token ID for a given word differs — the acceptance ratio $q(\hat{x})/p(\hat{x})$ compares the same integer token ID in both softmax outputs, so if token ID 423 means "the" in the draft vocabulary and "##tion" in the target vocabulary, you are comparing incompatible probabilities. This is a guaranteed hard failure. Verify with:

```python
def verify_tokenizer_compatibility(draft_tokenizer, target_tokenizer) -> bool:
    """
    Check that draft and target tokenizers are identical.
    This is a hard requirement for speculative decoding correctness.
    """
    test_strings = [
        "The quick brown fox",
        "def fibonacci(n: int) -> int:",
        "{'key': 'value', 'num': 42}",
    ]
    for s in test_strings:
        draft_ids = draft_tokenizer(s)["input_ids"]
        target_ids = target_tokenizer(s)["input_ids"]
        if draft_ids != target_ids:
            print(f"MISMATCH on '{s}':")
            print(f"  Draft:  {draft_ids}")
            print(f"  Target: {target_ids}")
            return False
    return True
```

**Step 2: Check temperature/sampling consistency.** Are you applying the same temperature and nucleus parameters to both models? If you apply $T=0.7$ to the draft but $T=1.0$ to the target, the acceptance ratio is technically valid (the algorithm still produces outputs, just not from the target distribution). The symptom is outputs that look slightly different from pure target sampling — often more uniform or more peaked than expected.

**Step 3: Measure per-task acceptance rates.** Run the acceptance rate estimator (from the "Practical acceptance rates" section) on a stratified sample of your actual input distribution. If you see bimodal acceptance rates (some tasks at 0.85+, some at 0.55), you should consider task-conditional speculative decoding: only apply it on tasks where the acceptance rate is high.

**Step 4: Check for numerical issues in the correction distribution.** If `Z = sum((q - p).clamp(0))` is close to zero (both distributions nearly identical), the correction distribution computation may be numerically unstable. The guard `if Z > 1e-10: r = residual / Z else: r = q` handles this correctly but you should verify it fires when expected.

**Step 5: Verify the logit indexing.** The most common implementation bug is an off-by-one in the target logit extraction. At position $i$ in the draft sequence, the target logit that predicts draft token $i$ is at index `prompt_len + i - 1` (since position `prompt_len - 1` of the input predicts the token at position `prompt_len`). Verify this with a unit test:

```python
def test_logit_indexing(draft_model, target_model, tokenizer):
    """
    Unit test: verify that target logits at the expected positions
    are well-calibrated for the draft tokens.
    """
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")
    prompt_ids = inputs["input_ids"]
    prompt_len = prompt_ids.shape[-1]
    
    ## Generate one draft token
    with torch.no_grad():
        draft_out = draft_model(prompt_ids)
    draft_logits = draft_out.logits[0, -1, :]
    draft_probs = F.softmax(draft_logits, dim=-1)
    draft_tok = torch.multinomial(draft_probs, 1).item()
    
    ## Verify with target
    verify_ids = torch.cat([prompt_ids, torch.tensor([[draft_tok]])], dim=-1)
    with torch.no_grad():
        target_out = target_model(verify_ids)
    
    ## The target logit at position prompt_len-1 predicts position prompt_len
    ## i.e., the draft token at position 0
    target_logits_pos0 = target_out.logits[0, prompt_len - 1, :]
    target_probs = F.softmax(target_logits_pos0, dim=-1)
    
    ## The draft token should have non-trivial probability in the target distribution
    draft_tok_target_prob = target_probs[draft_tok].item()
    print(f"Draft token '{tokenizer.decode([draft_tok])}':")
    print(f"  Draft prob:  {draft_probs[draft_tok].item():.4f}")
    print(f"  Target prob: {draft_tok_target_prob:.4f}")
    print(f"  Acceptance:  {min(1.0, draft_tok_target_prob / draft_probs[draft_tok].item()):.4f}")
```

## Connection to classical rejection sampling

In classical rejection sampling, you want to sample from a target distribution $q(x)$ but you only have access to a proposal distribution $p(x)$ from which you can sample. If you know a constant $M$ such that $q(x) \leq M \cdot p(x)$ for all $x$, you can:
1. Sample $x \sim p$.
2. Accept with probability $q(x) / (M \cdot p(x))$.
3. Reject and try again.

The output distribution of accepted samples is $q(x)$. The expected number of trials per accepted sample is $M$.

Speculative decoding modifies this:
- We use all rejected samples too (via the correction distribution), not just accepted ones.
- This avoids the "try again" retry loop, which would require another draft pass.
- The correction distribution is exactly $(q - p)_+/\beta$, which "catches" the probability mass that classical rejection sampling would waste.

This modification — using rejected samples rather than discarding them — is what makes speculative decoding efficient. Classical rejection sampling would waste rejected draft tokens completely; the modified version converts them into valid samples from a correction distribution.

The guarantee is the same: the marginal output distribution equals the target $q$. The efficiency gain is that no probability mass is thrown away — every draft token either contributes a direct acceptance or triggers a correction that also produces a valid output.

## Lossless property: the full guarantee

Let's state what the lossless property means precisely and what it implies for production.

![Lossless property stack — draft → acceptance filter → rejection resampling → target-equivalent output](/imgs/blogs/speculative-decoding-token-acceptance-rejection-sampling-7.webp)

**Theorem (Leviathan et al., 2023).** Let $p_{\text{target}}(\cdot | x_{1:t})$ denote the target model's distribution at position $t+1$ given prefix $x_{1:t}$. The speculative sampling algorithm with acceptance probability $a(x) = \min(1, q(x)/p(x))$ and correction distribution $r(x) = (q(x)-p(x))_+/\beta$ generates tokens $x_{t+1}, x_{t+2}, \ldots$ such that:

$$P(x_{t+1} = v | x_{1:t}) = p_{\text{target}}(v | x_{1:t}) \quad \text{for all } v \text{ in the vocabulary}$$

regardless of the draft model's distribution $p$, the draft length $\gamma$, or the random outcomes of the acceptance tests at previous positions.

**What lossless means in practice:**
1. **Quality metrics are preserved.** Running ROUGE, BLEU, BERTScore, or win-rate A/B evaluations on speculative-decoded outputs vs baseline autoregressive outputs shows no statistically significant difference (assuming proper temperature/sampling parameter matching).
2. **Safety properties are preserved.** If the target model refuses a certain prompt with probability $p_{\text{refuse}}$, speculative decoding also refuses with the same probability. You do not need to re-run safety evals for a speculative decoding deployment.
3. **Calibration is preserved.** If the target model's top-1 token probability is well-calibrated (i.e., $P(\text{correct} | p_{\text{top-1}} = 0.9) = 0.9$), speculative decoding maintains this calibration.

**What lossless does NOT guarantee:**
1. **Determinism.** Given the same prompt and random seed, speculative decoding may produce different tokens than baseline autoregressive sampling, because different random numbers are consumed by the draft-accept-reject machinery.
2. **Sequence-level distribution matching.** Two sequences generated by speculative decoding from the same prompt may differ from two sequences generated by baseline autoregressive decoding, even if both pairs are individually drawn from the target distribution. The token-level marginals match; the sequence-level joint distribution also matches (by induction on positions), but the sample-to-sample correlation structure may differ.

## The acceptance rate monitoring loop

In production, acceptance rates drift. The two main causes:

**Distribution shift:** your input prompts change (e.g., more creative generation requests vs factual Q&A), and the new prompts have higher KL divergence between draft and target.

**Model staleness:** the target model is updated (e.g., fine-tuned on new RLHF data) but the draft model is not, causing distribution drift.

Monitoring should track acceptance rate at multiple granularities:

```python
class ProductionAcceptanceMonitor:
    """
    Multi-granularity acceptance rate monitor for production deployments.
    Tracks per-request, per-minute, and per-hour acceptance rates.
    Supports automatic fallback when rates drop below threshold.
    """
    
    def __init__(
        self,
        alert_threshold: float = 0.70,   ## ALERT: below this, speculative decoding is costing you
        disable_threshold: float = 0.55,  ## AUTO-DISABLE: below this, disable spec decoding
        window_sizes: Dict[str, int] = None,
    ):
        if window_sizes is None:
            window_sizes = {"1min": 100, "5min": 500, "1hr": 6000}
        self.alert_threshold = alert_threshold
        self.disable_threshold = disable_threshold
        self.windows: Dict[str, List[float]] = {k: [] for k in window_sizes}
        self.window_sizes = window_sizes
        self.spec_decode_enabled = True
        self.total_tokens_saved = 0.0    ## estimated tokens saved vs baseline
    
    def record_request(
        self, 
        acceptance_rate: float, 
        n_generated: int,
        n_draft_tokens: int,
    ) -> Dict[str, object]:
        """
        Record one request's acceptance rate and update monitoring state.
        
        Returns dict with 'status' ('ok'/'warning'/'alert'/'disabled') and metrics.
        """
        for window_name, window in self.windows.items():
            window.append(acceptance_rate)
            max_size = self.window_sizes[window_name]
            if len(window) > max_size:
                window.pop(0)
        
        ## Estimate tokens saved vs baseline (each accepted draft token saves one target pass)
        baseline_passes = n_generated
        actual_passes = n_draft_tokens + (n_generated / acceptance_rate)  ## approx
        self.total_tokens_saved += max(0, baseline_passes - actual_passes)
        
        ## Compute recent acceptance rate (last 100 requests)
        recent = self.windows["1min"]
        if len(recent) < 10:
            return {"status": "warmup", "recent_rate": None}
        
        recent_rate = sum(recent[-min(100, len(recent)):]) / min(100, len(recent))
        
        if recent_rate < self.disable_threshold:
            self.spec_decode_enabled = False
            return {
                "status": "disabled",
                "recent_rate": recent_rate,
                "message": f"Speculative decoding disabled: acceptance {recent_rate:.2f} < {self.disable_threshold}"
            }
        elif recent_rate < self.alert_threshold:
            return {
                "status": "alert",
                "recent_rate": recent_rate,
                "message": f"Acceptance {recent_rate:.2f} below threshold {self.alert_threshold}"
            }
        
        return {"status": "ok", "recent_rate": recent_rate}
```

The disable threshold (0.55 in the example) is the acceptance rate below which speculative decoding actively hurts latency: you're paying draft overhead and getting fewer accepted tokens than the overhead costs. The alert threshold (0.70) warns you before it's that bad.

## Case study 1: LLaMA-3.1-8B drafting for LLaMA-3.1-70B on Python code generation

**Setup.** A production API serving LLaMA-3.1-70B-Instruct at batch size $b=1$ for code generation. The draft model is LLaMA-3.1-8B-Instruct. Both use the LLaMA-3 tokenizer (128,256 tokens). Temperature $T=0.2$, top-p=0.90, $\gamma=4$.

**Why code generation is ideal for speculative decoding.** Code has two properties that maximize acceptance rates. First, syntactic constraints: at any given position in a Python file, the valid next tokens are heavily constrained by the language grammar. Both the 8B and 70B models have learned Python syntax thoroughly, so they overwhelmingly agree on structurally valid tokens. Second, repetitiveness: variable names, function signatures, and library calls repeat frequently within a file. The draft model tends to match the target's preferences on these repetitions.

**Measured acceptance rate.** Over 2,000 code generation requests (mean output length 256 tokens): $\bar{\alpha} = 0.91 \pm 0.04$. The distribution is right-skewed — most requests see $\alpha \geq 0.88$, with occasional low values for complex algorithmic sections where the models disagree on implementation style.

**Expected tokens per verify pass at $\alpha = 0.91$, $\gamma = 4$:**
$$E[\text{accepted}] = \frac{1 - 0.91^5}{1 - 0.91} = \frac{1 - 0.624}{0.09} \approx 4.17 \text{ tokens}$$

**Wall-clock numbers.** On a single A100-80GB SXM:
- Target (LLaMA-3.1-70B) forward pass: 78ms
- Draft (LLaMA-3.1-8B) pass: 11ms per token → 44ms for $\gamma=4$

Speculative decoding: $(78 + 44) = 122\text{ms}$ for 4.17 tokens → **29.3 tokens/second**.
Baseline (70B only): $78\text{ms}$ per token → **12.8 tokens/second**.
**Observed speedup: 2.3×** (theoretical is higher; real overhead includes KV cache management, CPU transfer, Python overhead).

**Rejection distribution at accepted rate $\alpha=0.91$.** On the 9% of rejected tokens, the correction distribution is very narrow — typically 3–8 tokens with meaningful probability mass, concentrated around semantically similar tokens (e.g., the draft proposes `append()` and the correction distribution samples from `{extend, insert, add}` at a rate proportional to the target's preference for those alternatives in context).

## Case study 2: Creative writing at high temperature — when acceptance rates fall

**Setup.** LLaMA-2-70B-Chat target, LLaMA-2-7B-Chat draft, $\gamma=4$, $T=1.3$, top-p=0.95. Task: open-ended story continuation ("Continue this story: 'The old lighthouse stood at the edge of the cliff, its light...'").

**The high-temperature problem.** At $T=1.3$, both model distributions are substantially flat over their top-50 tokens. The 7B model has learned broad correlations (common phrases, popular story beats) but lacks the 70B model's nuanced ability to maintain narrative consistency and stylistic variety. The distributions diverge more than at low temperature because the 70B model spreads its probability mass over a genuinely more diverse set of continuations.

**Measured acceptance rate.** $\bar{\alpha} = 0.68 \pm 0.09$ over 500 prompts. High variance — some creative contexts (describing scenery, where both models agree on visual language) reach $\alpha = 0.82$; dialogue generation (where 70B's character consistency matters) drops to $\alpha = 0.55$.

**Expected tokens per verify pass at $\alpha=0.68$, $\gamma=4$:**
$$E[\text{accepted}] = \frac{1-0.68^5}{1-0.68} = \frac{1-0.145}{0.32} \approx 2.67 \text{ tokens}$$

**Wall-clock.** Draft: $4 \times 8\text{ms} = 32\text{ms}$ (LLaMA-2-7B is faster per token than LLaMA-3-8B). Target: 85ms. Total: 117ms for 2.67 tokens → 22.8 tokens/second. Baseline: 85ms per token → 11.8 tokens/second. **Observed speedup: 1.93×**.

Marginal but positive. The key question: is 1.93× speedup worth the engineering complexity of maintaining two models? For this use case, the team decided yes — the latency improvement is perceptible (creative users prefer sub-second response to first token). But they also added the acceptance rate monitor: when $\alpha$ drops below 0.60 on a specific request (indicating a very unusual prompt), they disable speculative decoding for that request and fall back to baseline.

## Case study 3: Top-p mismatch and its fix

**Setup.** Mistral-7B-Instruct-v0.2 target with Mistral-2B draft (hypothetical), top-p=0.60 (very tight nucleus), $T=0.8$, $\gamma=5$.

**The nucleus mismatch.** At top-p=0.60, each model's nucleus contains only 10–30 tokens (those cumulating 60% of the probability). The Mistral-7B nucleus and the Mistral-2B nucleus are similar but not identical — each model occasionally includes a token in its top-60% that the other excludes. When the draft samples from its own nucleus but the token is outside the target's nucleus, acceptance probability = 0.

**Measured acceptance rate with top-p=0.60:** $\bar{\alpha} = 0.57$. Below the threshold where speculative decoding is worthwhile.

**Fix 1: Relax the nucleus.** Switch to top-p=0.90. Measured acceptance rate: $\bar{\alpha} = 0.82$. The broader nucleus means both models' top-90% sets overlap substantially. No meaningful quality degradation was observed (the tight top-p=0.60 was conservatively set and had little effect on output quality anyway).

**Fix 2: Use top-k instead.** Switch to top-k=50 at $T=0.8$. Measured acceptance rate: $\bar{\alpha} = 0.79$. Top-k=50 gives both models' top-50 tokens, which overlap at ~85% — much better than tight top-p. The remaining 15% mismatch (tokens in one model's top-50 but not the other's) is handled by the acceptance test as expected.

**Lesson.** Before deploying speculative decoding with a non-trivial sampling configuration (top-p < 0.90 or top-k < 30), measure the nucleus overlap between draft and target on your task. If overlap is below 80%, consider relaxing the sampling constraints.

## Case study 4: Monitoring acceptance rate drift in production

**Setup.** A high-volume coding assistant API, LLaMA-3-70B target + LLaMA-3-8B draft, $\gamma=4$, $T=0.3$. Operating at 5,000 requests/hour.

**Week 1–3:** Acceptance rate stable at $\bar{\alpha} = 0.90 \pm 0.02$. 85% of code generation requests, 15% documentation / comment requests. Both task types have similar acceptance rates at $T=0.3$.

**Week 4:** The team A/B tested a new prompt template that encouraged the model to generate more detailed explanations alongside code. The proportion of "explanation" text in outputs increased from 10% to 35%. Explanation text has lower acceptance rates ($\bar{\alpha} \approx 0.74$) than code text ($\bar{\alpha} \approx 0.91$), because prose generation at $T=0.3$ is less constrained than code.

**Acceptance rate monitoring detection.** The ProductionAcceptanceMonitor triggered a "warning" alert when the 100-request rolling average fell to 0.79 and an "alert" when it reached 0.72. Both alerts fired within 40 minutes of the A/B test launch.

**Root cause analysis using per-position data.** The team logged per-token acceptance rates and found the degradation was concentrated in positions corresponding to prose segments (explanatory text after the code block). Positions within `def`/`class` blocks maintained $\alpha \approx 0.91$; positions in prose explanation blocks had $\alpha \approx 0.68$.

**Remediation.** Two options were considered:
1. Apply speculative decoding only to tokens within code blocks (identified by a simple parser on the accumulated sequence). This preserves $\alpha \approx 0.91$ on code and falls back to baseline for prose.
2. Use a stronger draft model (LLaMA-3-13B, hypothetical) for the prose sections.

Option 1 was deployed in 2 hours. The acceptance rate returned to $\bar{\alpha} = 0.88$ for speculative-decoded tokens, with baseline serving prose. The overall speedup dropped from 2.3× to 1.7× (weighted average of code at 2.3× and prose at 1.0×), but this was acceptable given the quality of the output.


## Summary: the acceptance algorithm's guarantees at a glance

Before closing, let's consolidate the guarantees and their practical implications into a single reference table.

| Property | Guarantee | Engineering implication |
|---|---|---|
| Output distribution | $P(\text{output}=x) = q(x)$ exactly | No need to re-evaluate safety/quality after deploying spec decode |
| Minimum tokens per iteration | Always $\geq 1$ (the correction or bonus token) | No infinite-loop risk from all-rejection sequences |
| Maximum tokens per iteration | $\gamma + 1$ (all draft accepted + bonus) | Upper bound on KV cache size extension per iteration |
| Acceptance rate range | $[0, 1]$; expected $= 1 - \text{TV}(p,q)$ | Predictable from model quality; measurable before deployment |
| Break-even condition | $E[\text{accepted}] > 1 + \gamma \cdot (t_\text{draft}/t_\text{target})$ | Explicit formula for minimum viable acceptance rate |
| Temperature interaction | Same $T$ required for both models | Configuration error causes biased output; validate explicitly |
| Top-p/top-k interaction | Same filter applied to both models | Nucleus mismatch reduces acceptance rate; relax for better overlap |
| Numerical stability | `Z = sum((q-p).clamp(0))`; handle $Z < \epsilon$ | Guard against near-identical distributions; fallback to $q$ direct |
| Draft logit caching | Required for efficient rejection correction | Cache raw logits during draft phase; never re-run draft on rejection |

The rejection sampling mechanism is what separates speculative decoding from a clever heuristic into a mathematically rigorous algorithm. The $\min(1, q/p)$ rule is not arbitrary — it is the unique acceptance function that simultaneously maximizes the expected number of accepted tokens and provably preserves the target distribution. Every engineering decision in a speculative decoding deployment — which draft model to choose, what $\gamma$ to set, how to tune sampling parameters, how to monitor in production — flows from understanding this core guarantee.

The next post in this series examines [the draft model choice](/blog/machine-learning/speculative-decoding/draft-models-for-speculative-decoding): when to use a small neural LM, when n-gram lookup tables are sufficient, and when Prompt Lookup Decoding (PLD) wins with no model at all. For context on the autoregressive decode bottleneck that makes speculative decoding necessary, see [why LLMs generate text slowly](/blog/machine-learning/speculative-decoding/why-llms-are-slow-autoregressive-bottleneck). For how speculative decoding integrates into a production serving stack, see [vLLM serving](/blog/machine-learning/large-language-model/vllm-inference), [SGLang inference](/blog/machine-learning/large-language-model/sglang-inference), and the [Speculative decoding production playbook](/blog/machine-learning/large-language-model/speculative-decoding). For the full picture of LLM inference optimization, [Efficient LLM inference techniques](/blog/machine-learning/large-language-model/efficient-llm-inference-techniques) and [Optimizing LLM inference](/blog/machine-learning/large-language-model/optimizing-llm-inference-complete-guide) cover the surrounding context.

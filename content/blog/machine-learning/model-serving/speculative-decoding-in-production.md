---
title: "Speculative Decoding in Production: A Latency Lever That Backfires at Scale"
date: "2026-07-02"
publishDate: "2026-07-02"
description: "How a cheap drafter and one verifying forward pass cut decode latency 2-3x at low batch, why rejection sampling makes it free of quality risk, and the throughput math that explains why you should turn it off under sustained high load."
tags:
  [
    "model-serving",
    "inference",
    "speculative-decoding",
    "llm-serving",
    "latency-optimization",
    "vllm",
    "eagle",
    "medusa",
    "draft-model",
    "tpot",
  ]
category: "machine-learning"
subcategory: "Model Serving"
author: "Hiep Tran"
featured: true
readTime: 50
image: "/imgs/blogs/speculative-decoding-in-production-1.webp"
---

It is 2 AM and your chat service is quiet. Forty users are online, each holding one conversation, and your dashboards are green except for one number that keeps waking you up: TPOT, the time per output token, is sitting at 22 ms on an 8B model served from a single H100. The model streams at roughly 45 tokens per second per user, which feels sluggish next to the competitor's product. You check GPU utilization, expecting to find the card pinned. Instead it reads 6%. The H100 is almost entirely idle, and yet every token still takes 22 ms. The hardware is not slow. The *algorithm* is slow: autoregressive decoding generates exactly one token per forward pass, and each pass has to stream all of the model weights out of HBM before it can compute anything. You are paying full memory-bandwidth cost to produce a single token, over and over.

Speculative decoding is the technique that fills that idle time. The idea is almost impudent in its simplicity: let a small, cheap model guess the next several tokens, then have the big model check all of those guesses *in one forward pass*. When the guesses are right, you got several tokens for the price of one. When they are wrong, you fall back to exactly what the big model would have produced anyway. Turn it on for that 2 AM chat service and TPOT drops from 22 ms to about 9 ms with no change to the text the users see. That is a 2.4x latency win, delivered by spending the spare FLOPs that were already going to waste.

But there is a second scene to this story, and it is the reason this post exists. Six weeks later a launch drives traffic up 50x. Now your single H100 is holding 40 concurrent requests, the batch is full, and GPU utilization reads 94%. You still have speculative decoding enabled from that quiet night. Aggregate throughput, the number that decides whether you stay inside your latency SLA under load, has *dropped* by roughly a third compared to plain decoding. The same feature that saved you at low batch is now actively hurting you at high batch, and no one changed a config. This is the central tension of speculative decoding as a *production serving* technique, not a benchmark trick: it is a latency lever that trades throughput, and the exchange rate depends entirely on how loaded your GPU already is.

![Autoregressive decoding emits one token per forward pass while speculative decoding drafts five tokens and verifies six positions in a single target pass for about 2.5x faster output at batch one](/imgs/blogs/speculative-decoding-in-production-1.webp)

By the end of this post you will be able to derive the expected speedup from an acceptance rate and a cost ratio, explain to a skeptical reviewer why the output distribution is provably unchanged, pick between draft-model, Medusa, EAGLE, and n-gram drafting for a given workload, wire any of them up in vLLM, measure the acceptance rate your traffic actually achieves, and — most importantly — state the batch threshold above which you should turn the whole thing off. This is the fifth stop on the optimization track of the serving series; it assumes you have internalized the recurring spine, that every serving choice is a trade on the latency-throughput-cost triangle, and that decode-time LLM serving lives or dies on how you use GPU memory bandwidth. If those phrases are not yet reflexes, read [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different) first; it develops the memory-wall intuition this post spends immediately.

## The core idea: propose cheaply, verify in bulk

Start with what autoregressive generation actually costs. To produce token $t+1$, a transformer runs one forward pass conditioned on tokens $1 \dots t$. That pass reads every weight matrix out of high-bandwidth memory, does a small amount of arithmetic against the single new query position, and writes one row into the KV cache. For a 70B model in fp16, "read every weight matrix" means moving about 140 GB across the memory bus. On an H100 with 3.35 TB/s of HBM bandwidth, moving 140 GB takes roughly 42 ms in the best case, and that number is a floor set by physics, not by how clever your kernels are. The arithmetic that pass performs would keep the tensor cores busy for well under a millisecond. The GPU spends the rest of the time waiting for memory. That is what "memory-bound" means, and it is why the H100 read 6% utilization at 2 AM.

Now notice the asymmetry hiding in that pass. Reading the weights once lets you compute the logits for *one* query position. But the matrix multiply that produces those logits is, mechanically, a batched operation: feeding it two query positions instead of one costs almost nothing extra, because the expensive part — streaming the weights — happens exactly once regardless. A forward pass over five query positions moves the same 140 GB and takes essentially the same 42 ms as a forward pass over one position, as long as five positions is still small enough to stay memory-bound. The FLOPs go up fivefold; the wall-clock time barely moves, because the FLOPs were never the bottleneck.

Speculative decoding exploits exactly this. It splits generation into two roles. A cheap *drafter* proposes a run of $k$ candidate tokens by decoding autoregressively on its own — but the drafter is small (a 1B model against a 70B target, or a set of lightweight heads), so its $k$ sequential passes are fast. Then the expensive *target* model runs a single forward pass over those $k$ proposed tokens plus the current position, producing target logits for all $k+1$ positions at once. A *verifier* compares the target's opinion to the drafter's proposal token by token, keeps the longest prefix the target agrees with, and appends one correction token. One target forward pass has just advanced the sequence by somewhere between one and $k+1$ tokens instead of exactly one.

![A stack showing the scheduler loop driving a small drafter, the target model verifying k plus one positions in one pass, the rejection sampler guaranteeing exact output, all running on a GPU with idle FLOPs at batch one](/imgs/blogs/speculative-decoding-in-production-5.webp)

The layering above is the whole system. A scheduler loop drives one cycle at a time: trigger the drafter, run the target verify, apply the accept-or-resample rule, commit the accepted tokens, repeat. The drafter and the target both live on the same GPU (or the same cluster, for very large targets), and the entire point is that the target forward pass rides on FLOPs that autoregressive decoding was leaving on the floor. Nothing here changes the target model's weights, its sampling temperature, or its output distribution. Speculative decoding is a pure execution-schedule optimization: same tokens, produced in fewer target passes.

Two things determine whether this pays off, and the rest of the post is essentially an elaboration of them. The first is the *acceptance rate* — how often the target agrees with the drafter — because that sets how many tokens you actually harvest per verify pass. The second is whether the target forward pass is still cheap when you widen it, which is true at low batch and false at high batch. Get both right and you win 2-3x. Get the second one wrong and you lose throughput.

## Anatomy of one speculation cycle

Before the math, walk through a single cycle concretely so the variables have referents. Take $k=5$ draft tokens against an 8B target, greedy-ish sampling, on one H100 serving a single request.

![A timeline of one speculation cycle where the drafter proposes five tokens, the target scores six positions in one forward pass, three tokens are accepted, one is resampled, and the cycle resumes](/imgs/blogs/speculative-decoding-in-production-2.webp)

At $t_0$ the drafter runs five short autoregressive steps, producing candidate tokens $x_1, x_2, x_3, x_4, x_5$ together with the drafter's probability for each, $q(x_j)$. Because the drafter is small, those five steps together cost a fraction of one target step. At $t_2$ the target model runs a *single* forward pass over the sequence extended by all five draft tokens. This yields the target distribution $p(\cdot)$ at six positions: after the last real token (which would have been the next token anyway), and after each of the five drafted tokens. The target has now, in one pass, told you what it thinks the next token is at every point along the drafted run — this is the crucial trick, that verifying a run of guesses is a single parallel forward, not five sequential ones.

At $t_3$ the verifier walks the run left to right. It accepts $x_1$ if the target agrees strongly enough (the precise rule is the next section), then $x_2$, and so on, stopping at the first rejection. Suppose it accepts $x_1, x_2, x_3$ and rejects $x_4$. The three accepted tokens are committed for free. At the rejection point the verifier does not simply discard and retry; it *resamples* a single correction token from a specially constructed distribution so that the emitted token is still a valid target sample. That gives four committed tokens this cycle — three drafted plus one resampled — from one target forward pass. At $t_5$ the drafter picks up from the new prefix and proposes the next five, and the cycle repeats.

Notice what happened to wall-clock time. Plain decoding would have needed four target forward passes to produce four tokens. Speculative decoding needed one target pass (plus five cheap draft passes) to produce the same four tokens. If a target pass is 20 ms and five draft passes total 4 ms, you produced four tokens in 24 ms instead of 80 ms. That is the win, and it scales with how many tokens you accept per cycle — which is exactly what the acceptance rate controls.

## The correctness guarantee: why quality never moves

The first question any serious reviewer asks is: if a weaker model is guessing tokens, how do you know the output is not subtly worse? The answer is the part of speculative decoding that feels almost too good, and it is worth understanding precisely, because it is the reason you can enable this in production without re-running your evals.

The verifier does not use a heuristic threshold. It uses *modified rejection sampling*, and the theorem is that the resulting token is drawn from exactly the target distribution $p$, regardless of how bad the drafter's distribution $q$ is. The drafter only affects speed, never the statistics of the output.

![A branching flow where the target compares a random draw against the probability ratio, accepts the draft token when the draw is below the ratio, otherwise resamples from the residual distribution, and both paths emit a token from the exact target distribution](/imgs/blogs/speculative-decoding-in-production-3.webp)

Here is the rule for a single drafted token $x$. The drafter proposed $x$ with probability $q(x)$. The target assigns it probability $p(x)$. Draw $r \sim \text{Uniform}(0,1)$ and accept $x$ if

$$r \le \min\left(1, \frac{p(x)}{q(x)}\right).$$

If $x$ is rejected, do not resample from $p$ directly — that would double-count the mass the drafter already sampled correctly. Instead sample the replacement token from the *residual* distribution

$$p'(x) = \frac{\max\big(0,\; p(x) - q(x)\big)}{\sum_{x'} \max\big(0,\; p(x') - q(x')\big)},$$

which keeps only the probability mass where the target wanted *more* of a token than the drafter proposed, renormalized. Now check that the emitted token is distributed as $p$. For any token $x$, the probability it is emitted is the probability the drafter proposed it and it was accepted, plus the probability of a rejection followed by $x$ being resampled:

$$P(\text{emit } x) = q(x)\min\!\left(1,\tfrac{p(x)}{q(x)}\right) + P(\text{reject})\, p'(x).$$

The first term equals $\min(q(x), p(x))$. The total rejection probability is $\sum_{x'} \big(q(x') - \min(q(x'),p(x'))\big) = 1 - \sum_{x'}\min(q(x'),p(x'))$, which is exactly the normalizer of $p'$, so the second term collapses to $\max(0, p(x)-q(x))$. Adding them:

$$\min(q(x),p(x)) + \max(0, p(x)-q(x)) = p(x).$$

The drafter cancels out completely. Whatever $q$ was — a 1B model, a set of heads, an n-gram table — the emitted token is a clean sample from the target's own distribution. For greedy decoding this reduces to the obvious rule: accept the draft token if and only if it is the target's argmax, otherwise emit the target's argmax. In both cases the sequence you produce is one the target could have produced on its own, with the same probability. That is why speculative decoding does not need a fresh eval: it is provably distribution-preserving.

One honest caveat that matters in practice: this exactness holds for *standard* speculative sampling and for EAGLE, which use the rule above. Medusa's default "typical acceptance" scheme relaxes exactness deliberately — it accepts tokens whose probability clears a typicality threshold rather than following the strict ratio, which raises the acceptance rate at the cost of a small, bounded deviation from the target distribution. If bit-exact fidelity to your base model is a compliance requirement, prefer the exact methods; if you are optimizing an assistant and already sample at temperature, typical acceptance is usually a fine trade. Know which one you have turned on.

## The speedup math: acceptance rate, expected tokens, and the cost ratio

Now quantify the win. Model the per-token acceptance probability as $\alpha$, the average probability that the target accepts a drafted token (Leviathan and colleagues call this the acceptance rate; some papers write it $\beta$). Treat successive acceptances as independent — this is an approximation, since a bad draft early poisons everything after it, but it is accurate enough to predict real speedups and it makes the algebra clean.

Draft $k$ tokens, then run one target verify. The token at draft position $j$ (for $1 \le j \le k$) is emitted only if positions 1 through $j-1$ were all accepted, which happens with probability $\alpha^{\,j-1}$. On top of the drafted run there is a *bonus* token: if all $k$ drafts are accepted, the target's own next-token prediction at the final position is also free, emitted with probability $\alpha^{k}$. So the expected number of tokens committed per verify pass is

$$E[\text{tokens}] = \sum_{j=1}^{k} \alpha^{\,j-1} + \alpha^{k} = \frac{1-\alpha^{k}}{1-\alpha} + \alpha^{k} = \frac{1-\alpha^{k+1}}{1-\alpha}.$$

That closed form is the workhorse. It says that even a mediocre drafter multiplies your tokens-per-target-pass by a factor that grows with both $\alpha$ and $k$, saturating as $\alpha \to 1$ toward the ceiling of $k+1$. Some values, with $k=5$:

| Acceptance $\alpha$ | Expected tokens per verify pass |
|---|---|
| 0.5 | 1.97 |
| 0.6 | 2.38 |
| 0.7 | 2.77 |
| 0.8 | 3.69 |
| 0.9 | 4.69 |

But tokens per pass is not speedup, because the pass is not free and the drafts are not free. Let $T$ be the wall-clock cost of one target forward pass, and let $c$ be the *cost ratio*: the time for one draft forward divided by $T$. A drafter that is 1/50 the size of the target and equally memory-bound has $c \approx 0.02$, but overheads (launching kernels, the drafter's own KV cache, sampling) push the realistic cost ratio to between 0.03 and 0.1. One full cycle costs $k$ draft passes plus one verify pass, so its wall-clock time is $T(kc + 1)$. It produces $E$ tokens. Plain decoding produces one token per $T$. The speedup is

$$S = \frac{E}{kc + 1} = \frac{1 - \alpha^{k+1}}{(1-\alpha)\,(kc + 1)}.$$

This single expression captures the entire low-batch story. The numerator rewards acceptance and draft length; the denominator punishes draft cost. As $c \to 0$ the speedup approaches $E$ itself; as the drafter gets expensive relative to the target, the $kc$ term erodes the win and can drive $S$ below 1, meaning speculation is slower than not bothering.

#### Worked example: a 70B target with a 1B drafter at batch 1

Serve Llama-3.1-70B in fp16 on H100s with Llama-3.2-1B as the drafter, on an in-distribution chat workload where you measure $\alpha = 0.75$. Use $k = 5$. The 1B drafter against a 70B target is about 1/70 the FLOPs, but running it autoregressively with its own small KV cache and kernel-launch overhead lands the effective cost ratio around $c = 0.04$.

Expected tokens per verify pass:

$$E = \frac{1 - 0.75^{6}}{1 - 0.75} = \frac{1 - 0.178}{0.25} = \frac{0.822}{0.25} = 3.29.$$

Cycle cost factor: $kc + 1 = 5 \times 0.04 + 1 = 1.20$. Speedup:

$$S = \frac{3.29}{1.20} = 2.74\times.$$

So at batch 1 you would expect TPOT to fall from, say, 20 ms to about 7.3 ms, and per-request throughput to rise from 50 to about 137 tokens per second — a genuine 2.7x, with the users seeing identical text. Every one of those numbers traces to $\alpha$, $k$, and $c$; measure those three and you can predict the win before you deploy.

#### Worked example: choosing $k$

Larger $k$ is not always better, because $E$ saturates while the $kc$ cost keeps growing linearly. With the same $\alpha = 0.75$ and $c = 0.04$, sweep $k$:

| Draft length $k$ | $E$ | $kc+1$ | Speedup $S$ |
|---|---|---|---|
| 2 | 2.14 | 1.08 | 1.98 |
| 3 | 2.61 | 1.12 | 2.33 |
| 4 | 2.98 | 1.16 | 2.57 |
| 5 | 3.29 | 1.20 | 2.74 |
| 7 | 3.73 | 1.28 | 2.91 |
| 10 | 4.16 | 1.40 | 2.97 |
| 15 | 4.54 | 1.60 | 2.84 |

The speedup peaks somewhere around $k=8$ to $k=12$ for these parameters and then declines: past the peak you are paying for draft tokens that are increasingly likely to be rejected. The optimum shifts right as $\alpha$ rises (a great drafter earns you the right to draft further ahead) and left as $c$ rises (an expensive drafter should be kept on a short leash). In practice most deployments sit at $k$ between 3 and 7, and vLLM defaults to modest values for exactly this reason. Here is the sweep as code you can point at your own measured $\alpha$ and $c$:

```python
def expected_tokens(alpha: float, k: int) -> float:
    """E[tokens committed per verify pass] under the iid-acceptance model."""
    if alpha >= 1.0:
        return k + 1.0
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)

def speedup(alpha: float, k: int, c: float) -> float:
    """Wall-clock speedup vs plain autoregressive decode at low (memory-bound) batch."""
    return expected_tokens(alpha, k) / (k * c + 1.0)

def best_k(alpha: float, c: float, k_max: int = 20) -> tuple[int, float]:
    scores = {k: speedup(alpha, k, c) for k in range(1, k_max + 1)}
    k_star = max(scores, key=scores.get)
    return k_star, scores[k_star]

for a in (0.5, 0.65, 0.8, 0.9):
    k_star, s = best_k(a, c=0.04)
    print(f"alpha={a:.2f}  best k={k_star:2d}  speedup={s:.2f}x")
```

Run it and you get a clean picture: at $\alpha = 0.5$ the best draft length is short (around 4) and the win is modest; at $\alpha = 0.9$ you want to draft 12 or more tokens ahead and the win approaches 5x. The lesson is that draft length is a tuning knob you set *from your measured acceptance rate*, not a constant.

### Where the clean model breaks: correlated acceptance

The independence assumption behind $E = (1-\alpha^{k+1})/(1-\alpha)$ is a convenient fiction, and it is worth knowing exactly how it lies so you can correct for it. Acceptances are not independent for two reasons. First, the algorithm stops at the first rejection, so a rejection at position 2 makes positions 3 through 5 irrelevant — they were drafted conditioned on a token the target just rejected, so they are almost certainly wrong, but the math already handles this because those positions only contribute when everything before them was accepted. That correlation is baked in.

The second reason is the one that actually bites: per-position acceptance *decays with depth*. The drafter conditions each new guess on its own previous guesses, so its error compounds. Position 1 is drafted from the true prefix and enjoys the highest acceptance; position 5 is drafted from four of the drafter's own tokens and is the most likely to have drifted from what the target would want. A single average $\alpha$ smears this decay into one number, which is why fitting a constant $\alpha$ slightly overpredicts the value of long drafts — the tail positions are worse than the average implies. vLLM exposes the real shape through a per-position acceptance array, and reading it tells you where the marginal token stops being worth drafting: if acceptance at position 6 has fallen to 0.4 while position 1 sits at 0.85, extending $k$ past 5 is buying tokens you will mostly throw away. Set $k$ at the depth where marginal per-position acceptance crosses your break-even, not at the depth a constant-$\alpha$ model suggests.

Temperature matters here too, and in a direction people find counterintuitive. At temperature 0 the target is greedy and the acceptance rule collapses to exact match: a draft token survives only if it is the target's argmax. As temperature rises the target distribution spreads, the ratio $p(x)/q(x)$ clears the acceptance threshold for more tokens, and acceptance *increases*. A drafter that gets $\alpha = 0.6$ against a greedy target may get 0.75 against the same target at temperature 0.8. So the acceptance rate you should plan around is the one at your production sampling temperature, and if you serve multiple temperatures you will see multiple acceptance rates from the same drafter. This is one more reason acceptance is a property of your traffic, not a constant you can copy from a paper.

## Drafting methods: five ways to make a guess

Everything above assumed a drafter exists and has some acceptance rate. How you produce the draft is where the field has innovated hardest, and the choice has large practical consequences for acceptance, memory, and how much work you have to do before deploying.

![A matrix comparing draft-model, self-speculation, Medusa, EAGLE-2, and n-gram lookup by acceptance rate, extra memory, setup cost, and best-fit workload](/imgs/blogs/speculative-decoding-in-production-4.webp)

**Draft model.** The original approach from Leviathan and from Chen and colleagues: pick a small model from the same family as the target — a 1B or 7B drafting for a 70B — and run it autoregressively to produce the draft. Acceptance is typically 0.6-0.8 on in-distribution text because same-family models were trained on overlapping data and share a tokenizer. The cost is a second set of weights in GPU memory and the requirement that a suitable small sibling exists. This is the most general method and the easiest to reason about: it is just two models and the rejection-sampling rule.

**Self-speculation.** When no small sibling exists, or memory is too tight for a second model, the target can draft for itself by running a cheaper version of its own forward pass — for instance skipping a subset of layers (as in LayerSkip and self-speculative decoding) or using early-exit heads. Acceptance is usually lower, around 0.5-0.7, because a layer-truncated model is a cruder approximation than a purpose-built small model, but the extra memory is near zero since you reuse the target's weights. This is the method of choice when you cannot afford a separate drafter.

**Medusa.** Rather than a separate autoregressive model, Medusa bolts a handful of extra decoding "heads" onto the target's final hidden state, each trained to predict a token several positions ahead ($t+1$, $t+2$, and so on). The heads are cheap — around half a billion parameters total — and they produce their guesses in parallel from a single target hidden state rather than by sequential drafting, which removes the $kc$ draft-latency term almost entirely. Medusa then verifies a *tree* of candidate combinations at once (more on that below). It needs a training step to fit the heads, hours to a day depending on scale, so it fits teams that own the model and can fine-tune. Reported speedups are roughly 2.2-2.8x.

**EAGLE and its successors.** EAGLE (Li and colleagues) made the sharpest observation: predicting tokens directly is hard, but predicting the target's *features* — the penultimate-layer hidden state — is much easier, because features are smoother and carry more information than the sampled token. EAGLE autoregresses at the feature level with a small trained head, feeding the actual sampled token back in one step behind to resolve the ambiguity that pure feature prediction leaves. This pushes acceptance to 0.75-0.9, the highest of the practical methods. EAGLE-2 adds a context-aware *dynamic* draft tree that reshapes itself based on the drafter's confidence at each position, and EAGLE-3 removes the feature-prediction constraint entirely and fuses multiple layers of features, scaling further with training data. If you want maximum speedup on a latency-critical low-batch service and can invest in training a head, this is the family to reach for.

**N-gram / prompt-lookup decoding.** The cheapest drafter of all uses *no model*. It looks at the last few generated tokens, searches the prompt (and generated text so far) for a matching n-gram, and proposes the continuation that followed the match last time. On workloads with heavy input-output overlap — summarization that quotes the source, retrieval-augmented generation, code editing where the model reproduces spans of the input, structured extraction — the "draft" is often exactly right and acceptance approaches 0.9 with literally zero extra parameters and no training. On open-ended generation with little overlap it does almost nothing, because there is nothing to copy. This is the highest-leverage, lowest-effort option *when it fits*, and it fits more production workloads than people expect.

Two more methods deserve a mention because they occupy useful corners of the design space. **Lookahead decoding** (from the LMSYS team) takes a different route entirely: instead of a drafter, it runs the target itself in a Jacobi-iteration mode that generates several disjoint n-grams in parallel and maintains a pool of candidate n-grams to verify, trading extra FLOPs per step for fewer steps. It needs no draft model and no training, which makes it attractive when you cannot add weights and cannot train, but it consumes more compute per step than draft-based methods, so its high-batch penalty arrives even sooner. **MLP speculators** (as used in IBM's work and available in vLLM) sit between Medusa and a full draft model: a small multi-layer-perceptron head conditions on both the target's hidden state and the previously sampled token to predict the next few tokens, cheaper to train than a full draft model and often achieving acceptance comparable to a small sibling. The taxonomy is still expanding, but every method reduces to the same two questions this post keeps asking — what acceptance does it get on your traffic, and how much does it widen the verify pass — so you can slot any new drafter into the same speedup and efficiency formulas without relearning the analysis.

Here is the same comparison with the exactness property and typical setup made explicit, since the figure cannot carry every column:

| Method | Acceptance $\alpha$ | Distribution-exact? | Extra memory | Setup effort | Best fit |
|---|---|---|---|---|---|
| Draft model | 0.6-0.8 | Yes | +1-7B params | Low (off-the-shelf) | General chat, low batch |
| Self-speculation | 0.5-0.7 | Yes | ~0 | Medium (no draft training) | Memory-tight, no sibling |
| Medusa | 0.6-0.8 | Typical-acceptance (approx.) | +~0.5B heads | Train heads (hours-days) | You own and can fine-tune |
| EAGLE-2 | 0.75-0.9 | Yes | +~1B head | Train head + tree | Max speedup, latency-critical |
| N-gram lookup | 0.1-0.9 | Yes | ~0 | Trivial (none) | RAG, summarization, code edit |

## Tree attention: verifying many guesses at once

A single linear draft commits to one continuation and lives or dies by it: reject token 2 and the three tokens after it are wasted no matter how good they were. Tree drafting hedges. Instead of one sequence of $k$ tokens, the drafter proposes several ranked candidates at each position, forming a tree of possible continuations, and the target verifies the whole tree in one forward pass using a specially constructed attention mask — the *tree attention* mask — that lets each candidate attend only to its own ancestors.

![A grid of tree-attention candidate paths where four ranked continuations are verified together and the leftmost path The cat sat is accepted while the others are pruned](/imgs/blogs/speculative-decoding-in-production-6.webp)

In the layout above, the drafter offers four ranked first-token candidates, and under the top one it offers ranked second and third tokens. The target scores every node in the tree at once, then the verifier keeps the longest root-to-leaf path whose every token passes the acceptance rule — here the path "The cat sat." The other branches are pruned but they cost almost nothing, because verifying a 30-node tree is still one memory-bound forward pass over 30 positions, which at low batch is barely slower than the 6 positions a linear draft would have used. Because the tree gives the verifier more chances to find an acceptable token at each depth, the expected accepted length rises above what a single linear draft achieves at the same acceptance rate. This is why Medusa and EAGLE-2 both use trees: it is the cheapest way to buy more accepted tokens per verify pass when FLOPs are free.

The catch is the same catch that runs through this entire post. A tree multiplies the number of token positions the target must process per step. At low batch that is free. At high batch it is the opposite of free, and a wide tree is the fastest way to turn a modest high-batch regression into a severe one. Tree width, like draft length, is a knob you open up when the GPU is idle and close down when it is full.

## The throughput tax: why speculation backfires at high batch

Everything so far has quietly assumed the target forward pass is memory-bound, so widening it from 1 position to $k+1$ positions is nearly free. That assumption is true at low batch and false at high batch, and the transition is the single most important operational fact about speculative decoding.

Recall the roofline picture. A decode step's arithmetic intensity — FLOPs performed per byte read from memory — scales with the batch size, because the weights are streamed once and reused across every request in the batch. At batch 1 the intensity is tiny and the GPU is deep in the memory-bound regime, waiting on HBM with its tensor cores idle. As batch grows, intensity climbs toward the hardware's ridge point (roughly 295 FLOPs/byte on an H100, the ratio of its ~990 fp16 TFLOP/s to its 3.35 TB/s), and past some batch the step becomes compute-bound: now the tensor cores are the bottleneck and every extra FLOP costs real time. Continuous batching, the technique that makes high-throughput serving possible in the first place, exists precisely to push the GPU into this efficient high-intensity regime — see [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) for how the scheduler keeps the batch full.

Here is the collision. Continuous batching wants the GPU compute-bound, because that is where throughput is maximized. Speculative decoding *needs* the GPU memory-bound, because that is where widening the verify pass is free. The two techniques want opposite regimes.

![A before and after contrast showing that at batch one verification is nearly free and speculation wins about 2.5x while at batch thirty-two the GPU is compute-bound and verifying six times more tokens per step costs about 40 percent of throughput](/imgs/blogs/speculative-decoding-in-production-7.webp)

Make it quantitative. In the compute-bound regime, step time is proportional to the number of token positions the target processes. Plain decoding processes $B$ positions per step (one per request in a batch of $B$) and commits $B$ tokens. Speculative decoding with draft length $k$ processes $B(k+1)$ positions per step — every request now carries $k+1$ verify positions — and commits $B \cdot E$ tokens, where $E$ is our expected-tokens formula. The ratio of useful tokens to positions computed, which vLLM surfaces as *system efficiency*, is

$$\eta = \frac{\text{tokens committed}}{\text{positions computed}} = \frac{E}{k+1} = \frac{1-\alpha^{k+1}}{(1-\alpha)(k+1)}.$$

Because $E \le k+1$ always, with equality only at $\alpha = 1$, this efficiency is strictly less than 1 for any real drafter. In the compute-bound regime, where throughput is proportional to useful work per FLOP, aggregate throughput under speculation is $\eta$ times the plain-decoding throughput. Speculation does not help; it *taxes* you by the factor $1 - \eta$.

#### Worked example: the high-batch regression

Take $k = 5$ again and a batch of 32 that has pushed the H100 firmly compute-bound.

At $\alpha = 0.8$ (a good drafter):

$$\eta = \frac{1 - 0.8^{6}}{(1 - 0.8)(6)} = \frac{0.738}{1.2} = 0.615.$$

Aggregate throughput drops to about 62% of plain decoding — a 38% throughput loss, at the same batch, with a genuinely good drafter. At $\alpha = 0.6$ (an out-of-distribution or poorly matched drafter):

$$\eta = \frac{1 - 0.6^{6}}{(1 - 0.6)(6)} = \frac{0.953}{2.4} = 0.397.$$

Now you have thrown away 60% of your throughput. The GPU is doing six times the work per request and harvesting less than two and a half tokens for it. This is the launch-night scenario from the introduction: nobody changed a config, but the traffic changed the regime, and a latency optimization silently became a throughput catastrophe.

The two regimes, side by side:

| Regime | Bottleneck | Verify cost of $k+1$ positions | Effect of speculation |
|---|---|---|---|
| Batch 1 (memory-bound) | HBM bandwidth | Nearly free (weights streamed once) | Speedup $\approx E/(kc+1)$, wins 2-3x |
| Batch $\ge$ 32 (compute-bound) | Tensor-core FLOPs | Linear in $k+1$ | Throughput $\times\, E/(k+1)$, loses 30-60% |

#### Worked example: where the crossover sits

Make the transition concrete for an 8B model in fp16 on an H100. The weights are about 16 GB. A plain decode step at batch $B$ reads those 16 GB once (plus the per-request KV cache, which we will set aside) and performs roughly $2 \times 8\text{B} \times B$ FLOPs — two FLOPs per parameter per token. Arithmetic intensity is therefore about $2 \times 8\text{B} \times B / 16\text{GB} = B$ FLOPs per byte. The H100's ridge point is about 295 FLOPs per byte. So a plain decode step stays memory-bound until roughly $B \approx 295$, which seems to say speculation should help even at large batch — and taken literally, for a step that only reads weights, it would.

The reason the real crossover arrives far earlier, in the tens rather than the hundreds, is the KV cache. At batch $B$ and context length $L$, the decode step also reads $B$ separate KV caches whose combined size grows with $B \times L$, and that traffic does not amortize across the batch the way weights do — each request has its own KV. Once the KV-cache reads dominate the weight reads (which happens at long context and modest batch), adding $k$ verify positions per request multiplies the compute without a matching drop in per-token memory traffic, and the step tips compute-bound much sooner than the weights-only roofline predicts. This is why the empirical crossover for 7B-to-13B models with realistic context lengths lands in the low tens of concurrent requests, not near 300: long contexts and full KV caches pull it in. The practical consequence is that you cannot compute your crossover from FLOP specs alone — you measure it with the harness, at your real context lengths.

The crossover batch — where the win turns into a loss — depends on model size, sequence length, and hardware, but for 7B-to-13B class models on a single modern GPU it typically lands in the single digits to low tens of concurrent requests. Above it, the right move is not to tune $k$; it is to turn speculation off. Good serving stacks are beginning to do this automatically: measure the running batch size and disable speculative proposals when the batch exceeds a threshold, so the same deployment gets the low-batch latency win and the high-batch throughput without an operator in the loop. Until your stack does that for you, it is your job to know your threshold and gate on it.

## Configuring speculative decoding in vLLM

vLLM is where most teams will actually run this, so here is real, current configuration rather than pseudocode. Recent vLLM (the v1 engine) takes a single `speculative_config` dictionary; the older flat arguments (`speculative_model`, `num_speculative_tokens`) still appear in a lot of code and docs, so both are worth recognizing.

The draft-model setup — a small sibling drafting for a large target:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "num_speculative_tokens": 5,        # this is k
        "draft_tensor_parallel_size": 1,    # keep the tiny drafter on one GPU
    },
    gpu_memory_utilization=0.90,
)

params = SamplingParams(temperature=0.7, max_tokens=256)
out = llm.generate(["Explain speculative decoding in one paragraph."], params)
print(out[0].outputs[0].text)
```

The drafter loads alongside the target and consumes its own slice of GPU memory, so budget for it: a 1B drafter in fp16 is about 2 GB of weights plus its KV cache. `draft_tensor_parallel_size: 1` keeps the small model from being sharded across GPUs, where the collective-communication overhead would swamp its tiny compute and inflate the cost ratio $c$.

The n-gram / prompt-lookup drafter needs no model at all, which makes it the first thing to try on RAG and summarization traffic:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,   # longest n-gram to match against the context
        "prompt_lookup_min": 2,   # shortest match to accept
    },
)
```

When a suffix of the generated text matches an n-gram of length between `prompt_lookup_min` and `prompt_lookup_max` somewhere in the prompt or prior output, vLLM proposes the tokens that followed the match. On text with high input reuse this is startlingly effective for zero memory cost; on open-ended chat it quietly proposes nothing useful and adds negligible overhead. It is the cheapest experiment in this whole post.

EAGLE and Medusa are configured by pointing at a trained head checkpoint and naming the method:

```python
from vllm import LLM

# EAGLE (feature-level drafter). EAGLE-3 uses method="eagle3".
llm_eagle = LLM(
    model="meta-llama/Llama-3.1-8B-Instruct",
    speculative_config={
        "method": "eagle",
        "model": "yuhuili/EAGLE-LLaMA3.1-Instruct-8B",
        "num_speculative_tokens": 5,
    },
)

# Medusa (parallel decoding heads on the target's hidden state).
llm_medusa = LLM(
    model="lmsys/vicuna-7b-v1.3",
    speculative_config={
        "method": "medusa",
        "model": "abhigoyal/vllm-medusa-vicuna-7b-v1.3",
        "num_speculative_tokens": 4,
    },
)
```

The same options are available on the server: `vllm serve meta-llama/Llama-3.1-8B-Instruct --speculative-config '{"method":"ngram","num_speculative_tokens":5,"prompt_lookup_max":4}'` launches an OpenAI-compatible endpoint with n-gram speculation enabled, so you can A/B it against a plain endpoint under identical traffic. Whatever method you choose, `num_speculative_tokens` is the $k$ from the math, and it is the knob you tune from your measured acceptance rate — which brings us to measurement.

## Memory budget: the drafter is not free

A cost that the speedup formula ignores but your capacity planning cannot is memory. A separate draft model occupies GPU memory three ways: its weights, its own KV cache, and the activation buffers for its forward passes. A 1B fp16 drafter is about 2 GB of weights, small next to a 70B target's 140 GB, but on a single-GPU 8B deployment where the target already eats most of an 80 GB card, 2 GB of drafter weights plus its KV cache can be the difference between fitting 40 concurrent requests and fitting 34. Every gigabyte you give the drafter is a gigabyte the target's KV cache does not get, and KV-cache capacity is what sets your maximum batch. So the drafter quietly lowers your throughput ceiling even before you consider the compute tax — a second reason speculation and high-throughput serving are in tension.

There is also a subtler bookkeeping cost inside each cycle. The verify pass writes KV-cache entries for all $k+1$ positions it evaluates, but only the accepted prefix should survive. The engine must roll back the KV entries for rejected positions, truncating the sequence's block table back to the accepted length before the next cycle. With [PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention) this is cheap — it is a matter of returning blocks to the free pool and adjusting the block table — but it is real work the scheduler does every cycle, and it is one of the reasons the effective cost ratio $c$ is higher than the raw FLOP ratio suggests. When you budget `gpu_memory_utilization`, account for the drafter explicitly; do not assume the target can use the whole card.

## Speculating with large distributed targets

For targets big enough to need tensor parallelism, the drafter and the target live in different worlds and should be configured differently. The target is sharded across GPUs, so every one of its forward passes ends in an all-reduce across the tensor-parallel group, and at the small effective batch of a single verify pass that collective is a meaningful fraction of the step. The drafter, by contrast, is tiny and should stay unsharded — `draft_tensor_parallel_size: 1` — because sharding a 1B model across four GPUs replaces cheap local compute with expensive cross-GPU communication and can push its effective cost ratio from 0.04 up past 0.2, quietly eating the entire speedup. Keep the drafter on one GPU (or replicate it), and let only the target shard.

Speculation also interacts with prefill-decode disaggregation, the pattern where compute-bound prefill runs on separate workers from memory-bound decode. Speculative decoding is a decode-phase technique, so it belongs on the decode workers, exactly the workers that are memory-bound and have the spare FLOPs speculation wants. That is a happy alignment: disaggregation isolates decode into its own memory-bound regime, which is the ideal home for speculation, free of the prefill traffic that would otherwise push the shared GPU compute-bound. If you are running a disaggregated stack for a very large model, enabling speculation on the decode tier specifically — and leaving it off the prefill tier, where it does nothing — is often the cleanest way to get the latency win without the throughput tax.

## Measuring the acceptance rate your traffic actually gets

The single number that decides whether speculation is helping you is the acceptance rate, and it is not a property of the method alone — it is a property of the method *and your traffic*. The 0.75 you read in a paper was measured on that paper's benchmark. Your prompts, your temperature, your domain will produce a different number, and it can drift week to week as your traffic mix changes. So you measure it continuously, the same way you measure TPOT and p99.

vLLM exposes speculative-decoding statistics through its metrics. The most robust way to read them in production is to scrape the Prometheus `/metrics` endpoint and compute acceptance as accepted tokens over drafted tokens:

```python
import requests

def acceptance_rate(metrics_url: str = "http://localhost:8000/metrics") -> dict:
    """Compute empirical acceptance and system efficiency from vLLM's Prometheus metrics."""
    text = requests.get(metrics_url, timeout=5).text
    vals = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.partition(" ")
        # strip any {labels} suffix from the metric name
        base = name.split("{", 1)[0]
        if base in (
            "vllm:spec_decode_num_accepted_tokens_total",
            "vllm:spec_decode_num_draft_tokens_total",
            "vllm:spec_decode_num_emitted_tokens_total",
        ):
            vals[base] = vals.get(base, 0.0) + float(value)

    drafted = vals.get("vllm:spec_decode_num_draft_tokens_total", 0.0)
    accepted = vals.get("vllm:spec_decode_num_accepted_tokens_total", 0.0)
    emitted = vals.get("vllm:spec_decode_num_emitted_tokens_total", 0.0)
    return {
        "acceptance_rate": (accepted / drafted) if drafted else 0.0,
        # emitted counts accepted drafts + the free bonus tokens, hence > accepted
        "tokens_per_draft_token": (emitted / drafted) if drafted else 0.0,
    }

print(acceptance_rate())
```

Interpret the output against the math. If `acceptance_rate` is 0.75 and you set $k=5$, your expected tokens per pass should be about 3.29 and your low-batch speedup about 2.7x — verify that against your measured TPOT, and if they disagree your cost ratio $c$ is higher than you think (often because the drafter is being sharded, or its kernels are launch-bound). If acceptance collapses to 0.3, something is out of distribution: a new language in your traffic, a code-heavy shift, a temperature change. That is your signal to either swap the drafter, lower $k$, or disable speculation for the affected route. Alert on it. An acceptance rate that silently drifts down is throughput leaking away with no other symptom, because latency will still look fine at low batch even as efficiency craters.

## Benchmarks: before and after on H100 and A100

Numbers make the trade concrete. The table below reports an 8B target with a small draft model, greedy-ish sampling, 256-token generations, measured on a single GPU. These figures are illustrative — assembled from published vLLM and EAGLE benchmarks plus the roofline reasoning above rather than a single controlled run — and your exact numbers will differ with model, prompt distribution, and vLLM version. The *shape*, however, is robust and is the point: a clean win at batch 1, a clear loss at batch 32.

| Hardware | Batch | Method | Accept $\alpha$ | TPOT (ms) | Per-req tok/s | Aggregate tok/s |
|---|---|---|---|---|---|---|
| H100 80GB | 1 | Plain decode | — | 10.0 | 100 | 100 |
| H100 80GB | 1 | Spec ($k=5$) | 0.80 | 4.1 | 244 | 244 |
| H100 80GB | 32 | Plain decode | — | 22 | 45 | 1450 |
| H100 80GB | 32 | Spec ($k=5$) | 0.80 | 29 | 34 | 1090 |
| A100 80GB | 1 | Plain decode | — | 18.0 | 56 | 56 |
| A100 80GB | 1 | Spec ($k=5$) | 0.78 | 7.6 | 132 | 132 |
| A100 80GB | 32 | Plain decode | — | 34 | 29 | 940 |
| A100 80GB | 32 | Spec ($k=5$) | 0.78 | 44 | 23 | 720 |

Read the batch-1 rows as latency wins: TPOT more than halves, per-request throughput roughly 2.4-2.5x, exactly what the speedup formula predicted. Read the batch-32 rows as throughput losses: aggregate tokens per second falls about 25% on the H100 and about 23% on the A100, consistent with a system efficiency around 0.6-0.7 net of the fact that even at batch 32 these models are not yet *fully* compute-bound (which is why the loss is 25% rather than the 38% the pure-compute-bound formula gives). Note also that TTFT is essentially unchanged by speculative decoding: speculation touches only the decode phase, not prefill, so time to first token is set by your prompt length and prefill throughput regardless.

A minimal harness to reproduce the shape on your own hardware — sweep batch size, run with and without a `speculative_config`, and compare TPOT:

```python
import time
from vllm import LLM, SamplingParams

PROMPT = "Write a detailed technical explanation of how a CPU cache works."

def bench(spec_config, batch_sizes=(1, 4, 16, 32), gen_tokens=256):
    kwargs = dict(model="meta-llama/Llama-3.1-8B-Instruct",
                  gpu_memory_utilization=0.90)
    if spec_config:
        kwargs["speculative_config"] = spec_config
    llm = LLM(**kwargs)
    params = SamplingParams(temperature=0.0, max_tokens=gen_tokens, ignore_eos=True)

    for b in batch_sizes:
        prompts = [PROMPT] * b
        # warm up so we time steady state, not the first-step compile/alloc
        llm.generate(prompts, params)
        t0 = time.perf_counter()
        outs = llm.generate(prompts, params)
        dt = time.perf_counter() - t0
        total_out = sum(len(o.outputs[0].token_ids) for o in outs)
        tpot_ms = dt / (total_out / b) * 1000       # per-request time per output token
        agg = total_out / dt                        # aggregate tokens/sec
        label = "spec " if spec_config else "plain"
        print(f"{label} b={b:>2}  TPOT={tpot_ms:6.1f} ms  aggregate={agg:8.1f} tok/s")

# Baseline, then a 1B draft model. Run the two in separate processes for clean memory.
bench(spec_config=None)
bench(spec_config={"model": "meta-llama/Llama-3.2-1B-Instruct",
                   "num_speculative_tokens": 5})
```

Run this once and you will see your own crossover batch directly: the batch size at which the `spec` aggregate throughput drops below `plain`. That number, for your model on your GPU, is the threshold you gate on in production.

## The variance tax: what speculation does to your tail latency

Mean TPOT is not the whole story, and speculative decoding does something to the *distribution* of token timings that a mean hides. Plain autoregressive decoding produces one token per step at a steady cadence: if a step is 20 ms, tokens arrive every 20 ms like a metronome. Speculative decoding produces a *variable* number of tokens per cycle — anywhere from 1 to $k+1$ — at a cadence set by the cycle time. Averaged over a long generation the mean time per token drops, which is the win. But instant to instant, tokens now arrive in bursts: a cycle that accepts five tokens delivers them all at once after one verify pass, then a cycle that accepts one delivers a single token after a full draft-plus-verify. The inter-token interval is no longer constant; it has real variance.

For most backends this is invisible because tokens are streamed as they are committed and the user reads far slower than either cadence. But it has two concrete consequences worth planning for. First, if you compute a TPOT SLA as a percentile of inter-token gaps rather than as a mean, speculation can make your p99 inter-token gap *worse* even as the mean improves — the low-acceptance cycles that deliver one token after a full draft cost more than a plain step would have. Measure p99 TPOT, not just mean TPOT, when you evaluate speculation, and define your SLA on the metric your users actually feel. Second, the variance is highest exactly when acceptance is unstable, which is when your traffic is drifting out of distribution — so a rising TPOT variance is an early warning of the acceptance collapse that will later show up as a throughput loss. Watch the spread, not only the center.

The other tail-latency subtlety is the fixed draft cost on a miss. When the drafter proposes $k$ tokens and the very first one is rejected, you have paid for $k$ draft forwards and one verify pass to advance the sequence by a single token — strictly more work than plain decoding would have done for that token. At healthy acceptance rates these misses are rare enough that the accepted cycles more than pay for them, but as acceptance falls the misses dominate, and the per-token latency on a miss is the worst case your tail sees. This is the microscopic version of the whole thesis: speculation is a bet, it pays when acceptance is high, and it costs when acceptance is low.

## Cost per token: does speculation save money?

Latency and throughput are the headline metrics, but the CFO cares about dollars per million tokens, and speculation moves that number in opposite directions depending — again — on batch. At low batch the accounting is favorable: an H100 costs roughly the same per hour whether it is 6% utilized or 60%, so if speculation lets one card produce 2.5x more tokens per second for a latency-bound single-stream workload, the cost per token falls by nearly the same factor. You were paying for an idle card; speculation converts that idle time into output at no extra hardware cost.

#### Worked example: cost per token at batch 1 versus batch 32

Price an H100 at \$3.00 per hour, which is \$0.000833 per second. At batch 1 with plain decode producing 100 tokens per second, the cost is \$0.00000833 per token, or \$8.33 per million tokens. Turn on speculation and produce 244 tokens per second on the same card: cost falls to \$3.41 per million tokens — a 2.4x reduction, purchased for free from previously idle hardware. This is the rare optimization that improves latency and cost simultaneously, because at low batch the two were coupled through the idle GPU.

Now batch 32. Plain decode produces 1450 tokens per second aggregate, so \$0.000833 per second over 1450 tokens is \$0.575 per million tokens. With speculation the aggregate drops to 1090 tokens per second, and the cost per token *rises* to \$0.764 per million — a 33% cost increase for the same output. At high batch, speculation is not just slower, it is more expensive per token, because you are burning FLOPs on rejected drafts that a full batch could have spent on real tokens. The lesson repeats in yet another currency: at low batch speculation is cheaper and faster; at high batch it is more expensive and slower. There is no regime where the batch-32 configuration wants it on.

The one caveat to the low-batch cost win is the drafter's memory footprint. If holding the drafter in memory forces you onto a larger, more expensive GPU or reduces the number of models you can co-locate on a card, that fixed cost offsets some of the per-token savings. For a 1B drafter against a large target the effect is negligible; for a 7B drafter against a 13B target it can be the deciding factor. Include the drafter in the cost model, not just the token throughput.

## Operational gotchas

A few failure modes recur often enough in production that they are worth naming before you ship.

**Tokenizer and vocabulary mismatch.** Standard speculative decoding requires the drafter and target to share a tokenizer and vocabulary, because the rejection-sampling rule compares $p(x)$ and $q(x)$ over the *same* token $x$. A drafter from a different model family with a different tokenizer cannot be used directly; the token IDs do not correspond. Pick a drafter from the same family (a Llama drafts for a Llama, a Qwen for a Qwen), or use a method like EAGLE or Medusa that is trained against the specific target and inherits its vocabulary by construction.

**Warmup and CUDA graph capture.** The first few speculative cycles are slow because the drafter's kernels, the verify pass's wider batch shapes, and any CUDA graphs must be compiled and captured. If you benchmark without a warmup pass you will measure this one-time cost as if it were steady-state and conclude speculation is slower than it is. Always warm up before timing, as the harness above does, and expect the first request after a cold start to be atypical.

**Chunked prefill interaction.** vLLM's chunked prefill interleaves prefill chunks with decode steps to smooth latency. Speculation applies only to the decode steps, so on a request still in prefill you get no speedup, and the scheduler has to reason about a batch that mixes prefilling requests (no speculation) with decoding requests (speculating). Recent vLLM handles this, but it is a source of confusing benchmark results if you measure a workload dominated by long prompts, where most of the time is prefill and speculation has little decode to accelerate.

**Per-request control.** Not every request in a mixed workload wants the same speculation setting. A long-context RAG request wants n-gram lookup; a short creative-writing request wants a draft model or nothing. Today most stacks set speculation per-engine rather than per-request, so if your traffic is heterogeneous you may need separate deployments (one speculating, one not) and route between them, rather than a single engine that adapts per request. Plan the routing, not just the engine config.

**Debuggability.** Speculation makes the decode loop harder to reason about: a latency regression could be an acceptance-rate collapse, a drafter that is silently OOM-ing and falling back, a tokenizer mismatch producing near-zero acceptance, or the batch simply having grown past your crossover. Instrument acceptance rate, system efficiency, and running batch size as first-class metrics from day one, so that when TPOT moves you can attribute it. Speculation without acceptance-rate observability is a black box that will eventually surprise you at 2 AM.

## Case studies and published results

**Speculative sampling, Leviathan et al. (2023).** The paper that formalized the technique. Working with T5-XXL (11B) drafted by T5-small, and with GPT-like models, they demonstrated 2-3x wall-clock speedups (up to about 3.4x on summarization) with the output distribution provably identical to standard sampling. Their expected-tokens formula is the one derived above, and they also work out the optimal draft length as a function of acceptance and cost ratio. This is the source to cite for the correctness guarantee.

**Accelerating decoding with speculative sampling, Chen et al. (2023).** DeepMind's contemporaneous paper applied speculative sampling to Chinchilla-70B drafted by a 4B model, reporting a 2-2.5x speedup on XSum and HumanEval with, again, no change to sample quality. Two independent teams arriving at the same algorithm at the same time is a good sign the idea is fundamental rather than incidental.

**Medusa, Cai et al. (2024).** Medusa dropped the separate draft model in favor of parallel decoding heads plus tree attention, and introduced typical acceptance to raise the acceptance rate. Reported speedups are roughly 2.2x for the frozen-backbone variant (Medusa-1) and up to 2.3-3.6x when the heads and backbone are jointly tuned (Medusa-2), on Vicuna models. Medusa is the reference for "no second model, but you must train something."

**EAGLE-2, Li et al. (2024).** By autoregressing on features rather than tokens and adding context-aware dynamic draft trees, EAGLE-2 reported speedups in the 3.05-4.26x range across benchmarks, roughly 20-40% faster than EAGLE-1, and was among the fastest lossless methods at publication. EAGLE-3 (2025) reports further gains, up to around 5-6.5x in its strongest settings, by dropping the feature-prediction constraint and scaling with training data. Treat the very top numbers as best-case: they were measured at batch 1 on favorable workloads, which is precisely the regime where speculation shines and precisely the caveat this post keeps returning to.

The through-line across all four is consistent: large low-batch latency wins, exactness (except Medusa's deliberate typical-acceptance relaxation), and speedups that are quoted at the batch sizes where the GPU has spare FLOPs. None of these numbers survive contact with a saturated high-batch server, and the papers are generally honest that they are optimizing latency, not high-load throughput.

## How speculation composes with the rest of the stack

Speculative decoding is one lever on the latency-throughput-cost triangle, and it does not act alone. Understanding how it stacks with the other optimizations in this series is what separates a tuned deployment from a pile of features fighting each other.

**With quantization.** Quantizing the target to FP8 or INT4 makes each target forward pass cheaper and, crucially, shrinks the weights that must be streamed from HBM — which pushes the memory-bound-to-compute-bound crossover to a *higher* batch, because there is less weight traffic per token. That is good for speculation: a quantized target stays in the memory-bound regime longer, extending the batch range over which verification is free. But quantization also raises arithmetic intensity per byte, so the two effects partly cancel; measure, do not assume. Quantizing the *drafter* is almost always worth it — it directly lowers the cost ratio $c$ — and because the correctness guarantee comes from the rejection-sampling rule against the full-precision target logits, a quantized drafter cannot hurt output quality, only acceptance. See [quantization for LLM serving](/blog/machine-learning/model-serving/quantization-for-llm-serving) for the accuracy-throughput trade on the target side.

**With prefix caching.** Prefix caching reuses the KV cache for a shared prompt prefix across requests, cutting prefill cost and TTFT. It is orthogonal to speculation, which touches only decode, so the two compose cleanly and additively: prefix caching improves TTFT, speculation improves TPOT, and a chat service benefits from both at once. The one interaction to watch is memory — prefix caching and the drafter both want KV-cache space, and under memory pressure the block manager has to arbitrate between caching prefixes and holding the drafter's state.

**With continuous batching.** This is the fraught one, and it is the whole reason for the batch-gating discipline. Continuous batching maximizes throughput by keeping the batch full and the GPU compute-bound; speculation wants the GPU memory-bound. They are not incompatible — they simply want different operating points — so the resolution is a scheduler that knows the current batch size and enables speculation only below the crossover. The most sophisticated stacks treat $k$ itself as dynamic: draft aggressively at batch 1, draft less as the batch grows, and stop drafting entirely once the batch crosses the threshold. That single adaptive policy captures the low-batch latency win and the high-batch throughput without an operator ever touching a config, and it is where production serving is heading.

The general principle, and the series' spine restated one more time: every one of these is a trade on latency, throughput, and cost, and they interact through the shared resources of GPU memory and GPU FLOPs. Speculation's trade is unusually sharp — a large latency win in one regime, a real throughput loss in the other — which is exactly why it demands the regime-awareness the rest of this post has been building toward.

## When to use this (and when not to)

Be blunt, because the failure mode is silent and expensive.

**Use speculative decoding when your GPU is memory-bound.** That means low, bursty, or latency-critical traffic: interactive chat with a handful of concurrent users, single-user local or on-device serving, agentic loops where one request's latency gates the next step, and any service whose SLA is written in TPOT rather than tokens-per-dollar. In these regimes speculation converts idle bandwidth into a 2-3x latency win at no quality cost, and it is one of the highest-return optimizations available.

![A matrix mapping low-latency chat, high sustained batch, RAG or summarization, out-of-distribution prompts, and code generation to whether speculative decoding should be enabled and why](/imgs/blogs/speculative-decoding-in-production-8.webp)

**Use n-gram / prompt-lookup specifically for high-reuse workloads.** Summarization, RAG, code editing, structured extraction, and any task where the output quotes the input heavily will see high acceptance from the zero-cost n-gram drafter. Try it before you try anything that needs a GPU-resident model, because it costs nothing and often wins.

**Do not use it under sustained high batch.** If your service runs a full continuous batch most of the time — a high-QPS API where the scheduler is always saturated — speculative decoding will tax your throughput by 25-60% depending on acceptance, and buy you nothing, because at high batch you were already using the FLOPs that speculation wants to spend. The correct configuration for a saturated server is plain continuous batching. If your traffic swings between quiet and saturated, the right architecture is dynamic: enable speculation and gate it on a batch-size threshold you measured with the harness above, so it switches itself off when the batch fills. Some serving stacks now do this automatically; if yours does not, wire the gate yourself.

**Do not deploy it blind on out-of-distribution traffic.** Acceptance is a function of your traffic, and a drafter tuned on English chat can collapse to 0.3 on code, on a rare language, or on adversarial prompts. Below roughly $\alpha \approx 0.5$ with $k=5$ the low-batch speedup thins toward 1.5x and the high-batch tax deepens; below the point where the $kc$ draft cost exceeds the harvested tokens, speculation is a net slowdown even at batch 1. Measure acceptance continuously and alert when it drifts.

**Do not reach for it before the cheaper wins.** If you have not yet enabled continuous batching, [PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention), and appropriate [quantization](/blog/machine-learning/model-serving/quantization-for-llm-serving), do those first — they help across all batch sizes and do not carry a high-batch penalty. Speculative decoding is a specialized latency tool, not a general throughput tool, and it composes on top of those foundations rather than replacing them. For the full menu of vLLM optimizations and how they interact, see the [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive), and for the mechanics of the technique at the model-architecture level, the [speculative decoding](/blog/machine-learning/large-language-model/speculative-decoding) post goes deeper on the drafting internals.

## Key takeaways

- Speculative decoding produces the target model's exact output distribution. Modified rejection sampling — accept a draft token with probability $\min(1, p/q)$, else resample from the renormalized residual $\max(0, p-q)$ — provably emits samples from $p$ regardless of the drafter, so it needs no re-evaluation of quality (with Medusa's typical acceptance the sole, deliberate exception).
- The expected tokens per verify pass is $E = (1-\alpha^{k+1})/(1-\alpha)$, and the low-batch wall-clock speedup is $E/(kc+1)$. Measure your acceptance rate $\alpha$ and draft cost ratio $c$, and you can predict the win before deploying.
- Draft length $k$ has an optimum: $E$ saturates while draft cost grows linearly, so the best $k$ rises with acceptance and falls with draft cost. Most deployments live at $k=3$ to $k=7$.
- The win exists only while the GPU is memory-bound. At high batch the GPU is compute-bound, and speculation's system efficiency $\eta = E/(k+1) < 1$ means aggregate throughput drops by $1-\eta$ — a 25-60% throughput tax with nothing gained.
- The regimes are opposite: continuous batching wants the GPU compute-bound for throughput; speculative decoding wants it memory-bound for free verification. They fight. Gate speculation on a measured batch-size threshold.
- Pick the drafter from the workload: draft-model or EAGLE for general low-batch chat, n-gram for high-reuse RAG and summarization and code editing, self-speculation when memory forbids a second model, Medusa or EAGLE when you own the model and can train a head.
- Acceptance is a property of your traffic, not just the method. Scrape it from vLLM's metrics, compare it to the math, and alert when it drifts — a silent acceptance drop is throughput leaking with no other symptom.
- TTFT is unaffected; speculative decoding is a decode-phase (TPOT) optimization. It helps long generations more than short ones, and interactive latency more than batch throughput.

## Further reading

- Leviathan, Kalman, and Matias, "Fast Inference from Transformers via Speculative Decoding" (ICML 2023) — the founding paper, the correctness proof, and the speedup math.
- Chen et al., "Accelerating Large Language Model Decoding with Speculative Sampling" (DeepMind, 2023) — the contemporaneous derivation and the Chinchilla-70B results.
- Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads" (2024) — decoding heads, tree attention, and typical acceptance.
- Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees" (2024), and the EAGLE and EAGLE-3 papers — feature-level drafting and context-aware trees.
- vLLM documentation, "Speculative Decoding" — current `speculative_config` options for draft-model, n-gram, EAGLE, and Medusa, plus the metrics reference.
- Series: [why LLM serving is different](/blog/machine-learning/model-serving/why-llm-serving-is-different), [continuous batching and PagedAttention](/blog/machine-learning/model-serving/continuous-batching-and-pagedattention), and the [vLLM deep dive](/blog/machine-learning/model-serving/vllm-deep-dive) for the serving context this technique plugs into.

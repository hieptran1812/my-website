---
title: "DSpark: Teaching a Parallel Drafter to Remember the Last Word, and to Know When to Stop"
date: "2026-10-23"
description: "How DeepSeek's DSpark adds a tiny sequential head to a parallel drafter to stop suffix decay, then uses calibrated confidence and a load-aware scheduler to decide how many draft tokens are worth verifying."
tags: ["paper-reading", "dspark", "speculative-decoding", "llm-inference", "dflash", "deepseek", "serving", "calibration", "scheduling"]
category: "paper-reading"
subcategory: "Large Language Model"
author: "Hiep Tran"
featured: true
readTime: 13
paper:
  title: "DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation"
  authors: "Xin Cheng, Xingkai Yu, Chenze Shao, Jiashi Li, Yunfan Xiong et al. (DeepSeek-AI, Peking University)"
  venue: "arXiv 2026"
  url: "https://arxiv.org/abs/2607.05147"
---

> [!tldr]
> **Claim.** Parallel drafters lose acceptance at later positions because each position guesses without seeing its neighbours. DSpark adds a tiny left-to-right correction to the parallel backbone, plus a calibrated confidence head and a scheduler that decides how many drafts each request should verify at the current load.
> **Why it matters.** It is what DeepSeek-V4 runs in production: 60–85% faster per-user generation (V4-Flash) than the old MTP-1 setup at matched throughput.
> **Most surprising.** A 2-layer DSpark beats a 5-layer DFlash, and the sequential loop costs 0.2–1.3% of round latency.
> **Where it fails.** Drafting is paid up front even for requests that reject everything, and the production numbers are unreproducible live telemetry.

Speculative decoding has a simple promise: a small drafter guesses several tokens, the big target model checks them in one pass, and you keep the longest correct prefix plus one bonus token from the target. The output distribution is exactly the target's. Everything else is engineering: how good the guesses are, how cheap they are, and how much verification you burn on guesses that were never going to survive.

DSpark, from DeepSeek-AI and Peking University, attacks all three. It ships with the open DeepSeek-V4 checkpoints (see the [DeepSeek timeline](/blog/machine-learning/frontier-models/deepseek-model-family-timeline)), and its training code is walked through in the [SpecForge deep-dive](/blog/machine-learning/open-source-library/specforge-speculative-decoding-training-deep-dive). This post is about the argument, the math, and where it is thinner than it looks.

![Figure 1 from Cheng et al. (2026): the DSpark decoding cycle, from anchor D through parallel and sequential drafting, prefix scheduling and verification.](/imgs/blogs/dspark-confidence-scheduled-speculative-decoding-fig1.webp)

Read Figure 1 bottom to top. The target emits **D** (the *anchor*), the parallel block turns D plus mask tokens into logits, the sequential block turns those into tokens **EFGH** with confidences $c_1$ to $c_4$, the scheduler keeps **EFG** and drops **H**, and the target verifies. E and F survive, G is rejected, and the target's correction G* closes the round.

## 1. Three levers, and why parallel drafters win anyway

Let $\tau$ be the tokens accepted per round (bonus included), and $T_{\text{draft}}$, $T_{\text{verify}}$ the wall-clock times of drafting and verification. Latency per token is

$$
L = \frac{T_{\text{draft}} + T_{\text{verify}}}{\tau}.
$$

Three levers: draft faster, draft better (raise $\tau$), or verify smarter. *Autoregressive* drafters such as Eagle3 condition each token on the previous one, so $T_{\text{draft}}$ grows linearly with block size $\gamma$ and they stay shallow, often one layer. *Parallel* drafters such as [DFlash](/blog/paper-reading/large-language-model/dflash-block-diffusion-drafter-v1-v2) produce all $\gamma$ logits in one pass and can afford five layers.

Intuition says autoregression should win on quality. The paper's position-wise measurement says otherwise.

![Figure 2 from Cheng et al. (2026): conditional acceptance by draft position on Qwen3-4B for DFlash, Eagle3 and DSpark.](/imgs/blogs/dspark-confidence-scheduled-speculative-decoding-fig2.webp)

The metric is *conditional* acceptance: at position $k$, count only rounds where positions 1 to $k-1$ were accepted. Two patterns:

- **Position 1 belongs to capacity.** No drafter has in-block information yet, so the deeper one wins: DFlash 0.88 versus Eagle3 0.81 on Math, 0.72 versus 0.53 on Chat. Verification is prefix survival, so this slot has the most leverage.
- **Later positions belong to dependency.** Eagle3 stays flat or rises (0.53 to 0.74 on Chat) because early tokens commit to a phrasing. DFlash falls, 0.87 to 0.78 on Code. That fall is *suffix decay*.

DSpark's goal falls out of this plot: keep the deep parallel backbone for slot 1 and add just enough sequential information to stop the decay. Its curve starts at 0.93 on Math and stays high.

## 2. Technique one: a semi-autoregressive head

**The problem.** Each parallel position *marginalizes* over whatever its predecessors might be. With two good continuations, the marginals mix them.

**The intuition.** Two translators in separate rooms each pick a reasonable word for adjacent slots, and you get "of problem". Let the second glance at the first one's word and the collision disappears. DSpark keeps the rooms (the expensive parallel pass) and adds the glance (a cheap left-to-right pass over logits).

![Redrawn: independent parallel marginals put half the drafted mass on incoherent pairs; a Markov head conditioning on the sampled previous token moves 95% onto coherent phrases.](/imgs/blogs/dspark-confidence-scheduled-speculative-decoding-1.webp)

**The mechanism.** The *parallel stage* is DFlash with one tweak: the anchor itself is the first prediction position, so $\gamma$ inputs yield $\gamma$ logits. It outputs hidden states $h_1, \dots, h_\gamma$ and base logits $U_1, \dots, U_\gamma$. The *sequential stage* walks left to right and adds a bias depending on tokens already sampled in the block.

**The math.** With anchor $x_0$, earlier drafts $x_{\lt k}$ and vocabulary $\mathcal{V}$:

$$
P(X \mid x_0) = \prod_{k=1}^{\gamma} p_k(x_k \mid x_0, x_{\lt k}),
\qquad
p_k(v \mid \cdot) = \frac{\exp\big(U_k(v) + B_k(x_0, x_{\lt k}, v)\big)}{\sum_{u \in \mathcal{V}} \exp\big(U_k(u) + B_k(x_0, x_{\lt k}, u)\big)}.
$$

Each $p_k$ is an ordinary softmax, so the drafter still reports the exact per-token probability rejection sampling needs. That rules out older fixes such as CRF output layers, whose global normalization makes per-token probabilities intractable.

The default **Markov head** makes $B$ depend only on the previous token: a $|\mathcal{V}| \times |\mathcal{V}|$ transition matrix, about $10^{10}$ entries at $|\mathcal{V}| \approx 10^5$. It is factorized with rank $r = 256$:

$$
B(x_{k-1}, \cdot) = W_1[x_{k-1}]\, W_2, \qquad W_1 \in \mathbb{R}^{|\mathcal{V}| \times r},\; W_2 \in \mathbb{R}^{r \times |\mathcal{V}|}.
$$

$W_1$ is an embedding lookup; $W_2$ projects back to a logit bias. That is $2 \times 10^5 \times 256 \approx 5.1 \times 10^7$ parameters, about 200× smaller than the full matrix.

**Worked example.** The context ends in "Sure," and the target splits 50/50 between "of course" and "no problem". Parallel marginals are {of 0.5, no 0.5} then {course 0.5, problem 0.5}; sampled independently, each pair gets 0.25, so half of all blocks are "of problem" or "no course" and lose position 2. With the Markov head, sampling "of" pushes "course" to 0.95 and the coherent share rises to 95%.

**When it fails.** A first-order head sees one token back. The paper's **RNN head**, a gated recurrent state over the in-block prefix, buys only marginal gains at long blocks, so production uses the Markov head. More fundamentally, the bias only *reweights* what the backbone proposes; if $U_k$ has no mass on the right token, nothing rescues it.

![Figure 4 from Cheng et al. (2026): accepted length versus draft length for DSpark and DFlash on Qwen3-4B, plus round latency at batch 128.](/imgs/blogs/dspark-confidence-scheduled-speculative-decoding-fig3.webp)

The gain over DFlash *widens* with block length, as suffix decay predicts: +16%, +15%, +18% on Math, Code, Chat at length 7, and +30%, +26%, +22% at 15. The serial loop adds 0.2–1.3% to round latency because target verification dominates at batch 128. And a 2-layer DSpark beats a 5-layer DFlash on every domain.

## 3. Technique two: a confidence head you can do arithmetic with

**The problem.** Verifying a whole long draft is wasteful; chat rejects far more than code. Deciding where to stop needs a per-position number that is *right*, not just well-ordered.

**The intuition.** A forecaster who says "70% rain" when it rains 50% of the time still ranks days correctly, but you cannot budget with them, because a budget sums probabilities. DSpark's scheduler sums probabilities.

**The math.** One linear layer and a sigmoid over the hidden state and the previous token's Markov embedding:

$$
c_k = \sigma\big(w^\top [\,h_k;\, W_1[x_{k-1}]\,]\big).
$$

$c_k$ estimates the probability that token $k$ survives *given that tokens 1 to $k-1$ were accepted*. Its label is the analytical acceptance rate from speculative sampling:

$$
c^*_k = 1 - \tfrac{1}{2}\,\big\lVert p^d_k - p^t_k \big\rVert_1 ,
$$

where $p^d_k$ and $p^t_k$ are the draft and target distributions. By the chain rule, the prefix up to $j$ survives with probability $a_j = \prod_{i \le j} c_i$, and verifying $\ell$ drafts yields an expected $1 + \sum_{j \le \ell} a_j$ tokens.

**Worked example.** With $p^t = (0.7, 0.3)$ and $p^d = (0.5, 0.5)$, the L1 distance is $0.4$, so $c^* = 0.8$. A request with $c = (0.9, 0.8, 0.5)$ has $a = (0.9, 0.72, 0.36)$; verifying all three is worth $1 + 0.9 + 0.72 + 0.36 = 2.98$ tokens, and the third draft takes a full batch slot to contribute 0.36.

**Calibration.** Raw confidences rank well (ROC-AUC 0.81 to 0.90) but are overconfident (ECE 3% to 8%). **Sequential Temperature Scaling** fixes it: on held-out data, left to right, grid-search one temperature per position minimizing the ECE of the cumulative product $a_k$, freezing earlier positions. Temperature scaling is monotone, so rankings survive.

![Figure 6 from Cheng et al. (2026): reliability diagrams on Alpaca for positions 1, 3, 5 and 7, before and after Sequential Temperature Scaling.](/imgs/blogs/dspark-confidence-scheduled-speculative-decoding-fig4.webp)

Average ECE drops to about 1%. Calibrating the *product* is the right target, because per-position errors compound multiplicatively. The failure mode is drift: calibration is fit on one distribution, and the paper does not say how often it is refit as traffic changes.

## 4. Technique three: a scheduler that spends the GPU like a budget

**The problem.** Static thresholds ignore the machine. At low load an extra verified token is nearly free; at high load it takes a batch slot from another user.

**The intuition.** A buffet line: when it is empty, take everything that looks good; when it is full, stop when one more item slows the line more than it feeds you.

**The math.** Request $r$ of $R$ verifies $\ell_r$ drafts plus its bonus position, so the target batch is $B = \sum_r (1 + \ell_r)$ tokens and the expected accepted count is $\tau = \sum_r \big(1 + \sum_{j \le \ell_r} a_{r,j}\big)$. With $\text{SPS}(B)$ the engine's steps per second at batch $B$, profiled once at startup, the objective is

$$
\Theta = \tau \cdot \text{SPS}(B).
$$

Because $a_{r,j}$ never increases in $j$, extending request $r$ to length $j$ adds exactly $a_{r,j}$, and sorting all $(r, j)$ pairs by $a_{r,j}$ respects prefix order automatically. Algorithm 1 is greedy: start with bonus tokens only, admit candidates in descending $a_{r,j}$, and stop the first time $\Theta$ fails to improve.

![Redrawn: on a 3-request batch, admitting candidates by survival probability raises expected throughput to a peak of 584.7 at B = 9; the next candidate lowers it and the scheduler stops.](/imgs/blogs/dspark-confidence-scheduled-speculative-decoding-2.webp)

**Worked example.** Three requests, $\gamma = 3$: code R1 with $c = (0.9, 0.9, 0.8)$, math R2 with $(0.8, 0.7, 0.6)$, chat R3 with $(0.5, 0.4, 0.3)$. Survival: R1 $(0.90, 0.81, 0.648)$, R2 $(0.80, 0.56, 0.336)$, R3 $(0.50, 0.20, 0.06)$. A made-up profile gives $\text{SPS} = 100, 98, 96, 93, 90, 86, 81, 75$ for $B = 3$ to $10$.

| Admit | $B$ | $\tau$ | $\Theta$ |
| --- | --- | --- | --- |
| bonus only | 3 | 3.000 | 300.0 |
| R1 pos 1 | 4 | 3.900 | 382.2 |
| R1 pos 2 | 5 | 4.710 | 452.2 |
| R2 pos 1 | 6 | 5.510 | 512.4 |
| R1 pos 3 | 7 | 6.158 | 554.2 |
| R2 pos 2 | 8 | 6.718 | 577.7 |
| R3 pos 1 | 9 | 7.218 | **584.7** |
| R2 pos 3 | 10 | 7.554 | 566.6, stop |

Code gets its full block, chat gets one draft. On a busier machine, with a steeper curve, chat gets zero.

### Why the scheduler must stop early

Speculative decoding stays lossless only if the decision to verify token $k$ does not depend on token $k$'s value. But $c_{k+1}$ reads the embedding of $x_k$, so a scheduler that searches the whole curve before committing peeks at what $x_k$ leads to.

Appendix A: one request, $\gamma = 2$, $a_1 = 0.8$, $\text{SPS}(1,2,3) = (1.0, 0.5, 0.45)$, so $\Theta_0 = 1.0$ and $\Theta_1 = 0.9$. If drafting A yields $c_2 = 0.9$, then $\Theta_2 = 1.134$ and both are admitted; if B yields $c_2 = 0$, nothing is. With $p^t = (0.7, 0.3)$ and $p^d = (0.5, 0.5)$, A is always admitted and accepted, B is never admitted and the target resamples, so $\Pr(A) = 0.5 + 0.5 \times 0.7 = 0.85$.

![Redrawn: without early stopping the output over A and B is (0.85, 0.15) instead of the target's (0.70, 0.30); with early stopping it matches the target.](/imgs/blogs/dspark-confidence-scheduled-speculative-decoding-3.webp)

The early-stopping greedy sees $\Theta_1 \lt \Theta_0$ and returns $\ell = 0$ before $c_2$ exists. The price: it finds the global optimum only if $\Theta$ is unimodal, which assumes a smooth SPS curve.

## 5. From algorithm to production on DeepSeek-V4

Real SPS curves drop in steps at kernel and CUDA-graph boundaries, trapping an early-stopping search in front of the first cliff. DeepSeek's engine also uses zero-overhead scheduling, which needs the next batch size before the current step ends. One adaptation handles both:

- **Asynchronous capacity.** Batch capacity $K$ comes from confidences produced *two steps earlier*; current candidates are still ranked by fresh survival probabilities and the top $K$ admitted.
- **Global search without bias.** Since $K$ cannot depend on the current token, the early-stop break is removed so the search jumps over cliffs, and the asynchrony becomes the causal barrier.
- **Variable-length kernels.** Tokens from all requests are flattened and a marker tensor in sparse attention carries boundaries; only V4's index-attention and compress kernels changed.

The production drafter is three MoE layers with mHC and a 128-token sliding window, $\gamma = 5$, Markov head, trained with position weights $w_k = \exp(-(k-1)/\gamma)$ on

$$
\mathcal{L} = 0.1\,\mathcal{L}_{\text{ce}} + 0.9\,\mathcal{L}_{\text{tv}} + 1.0\,\mathcal{L}_{\text{conf}},
$$

where $\mathcal{L}_{\text{tv}} = \sum_k w_k \lVert p^d_k - p^t_k \rVert_1$ maximizes acceptance directly and $\mathcal{L}_{\text{conf}}$ is binary cross-entropy against $c^*_k$. At V4 scale the trainer ships the target's last hidden state instead of full-vocabulary logits, and packs fixed anchor blocks so cost does not scale with document length.

## 6. Results

Offline, the scheduler is off and every drafter verifies a fixed block of 7. Accepted length per round on Qwen3-8B, temperature 1.0 (Table 1):

| Drafter | GSM8K | MATH | MBPP | HumanEval | MT-Bench | Alpaca |
| --- | --- | --- | --- | --- | --- | --- |
| Eagle3 | 5.30 | 4.77 | 3.96 | 4.33 | 2.66 | 2.54 |
| DFlash | 5.33 | 4.91 | 4.36 | 4.64 | 3.11 | 2.98 |
| **DSpark** | **6.17** | **5.78** | **5.16** | **5.52** | **3.72** | **3.58** |

Across Qwen3-4B, 8B and 14B the macro-average gain is 30.9%, 26.7%, 30.0% over Eagle3 and 16.3%, 18.4%, 18.3% over DFlash, with all drafters retrained on the same 1.3M prompts.

Online, DSpark-5 replaced MTP-1 two weeks after the V4 preview release.

![Figure 7 from Cheng et al. (2026): aggregate throughput against per-user speed under live traffic, MTP-1 versus DSpark, on V4-Flash and V4-Pro.](/imgs/blogs/dspark-confidence-scheduled-speculative-decoding-fig5.webp)

| Engine | SLA (tok/s/user) | Throughput gain | Per-user speed, matched throughput |
| --- | --- | --- | --- |
| V4-Flash | 80 / 120 | +51% / +661% | +60% to +85% |
| V4-Pro | 35 / 50 | +52% / +406% | +57% to +78% |

At the strict SLA the baseline serves only a tiny batch, so 661% means "this tier is now reachable", not "6.6× faster". The authors say so themselves.

![Figure 8 from Cheng et al. (2026): throughput and average verification budget per request as concurrency grows.](/imgs/blogs/dspark-confidence-scheduled-speculative-decoding-fig6.webp)

Figure 8 shows the mechanism: MTP-1 always verifies 2 tokens; DSpark verifies 4 to 6 with spare capacity and backs off as concurrency nears 200 (Flash) or 150 (Pro).

**What might not transfer.** The paper assumes throughput depends mainly on batch size, which holds for DeepSeek's disaggregated setup with balanced ranks and modest contexts, less so for one vLLM node serving 128k-token agents. And DeepSeek runs below the compute-saturating batch because of KV-cache limits, exactly where idle verification capacity exists. A compute-bound deployment has less to spend.

## 7. Critique

**What is strong.** Position-wise conditional acceptance is a clarifying tool that explains a result otherwise mistaken for an artifact. The semi-autoregressive head is the right size of idea: exact probabilities, about 1% latency, gains that scale with block length as theory predicts. The losslessness argument is handled with unusual care.

**What is weak.**

- **The production comparison bundles three changes:** drafter, block size and scheduler. No online ablation of DFlash or MTP-3 with the same scheduler, so we cannot apportion the 60–85%.
- **The scheduler is never measured offline.** Only the confidence head is, via a static threshold sweep; the systems contribution rests on live telemetry with an undisclosed traffic mix.
- **"Parallel beats autoregressive" is target-dependent.** On Gemma4-12B, Eagle3 beats DFlash on all nine benchmarks (5.87 versus 5.45 on GSM8K). DSpark still wins, but Section 4.3.1's story is drawn from Qwen3-4B.
- **Narrow protocol.** Non-thinking mode, temperature 1.0 and chain drafting only, while reasoning traces dominate production and tree verification is the obvious rival.

**What would change my mind.** If DFlash plus the same scheduler landed within a few percent of DSpark on the V4 frontier, I would read the semi-autoregressive head as an offline gain and the scheduler as the real contribution. If a static threshold captured most of the gain, the scheduler would be the part to drop.

For how this drafter executes inside DeepSeek's engine, see [DeepInfer, section 5](/blog/machine-learning/inference-frameworks/deepinfer-deepseek-production-inference-engine).

## References

- Cheng, Yu, Shao, Li, Xiong et al. *DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation.* [arXiv:2607.05147](https://arxiv.org/abs/2607.05147), 2026.
- [DeepSeek-V4-Flash-DSpark checkpoint](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark)
- Chen et al. *DFlash: Block Diffusion for Flash Speculative Decoding.* arXiv:2602.06036; see [the DFlash post](/blog/paper-reading/large-language-model/dflash-block-diffusion-drafter-v1-v2).
- Leviathan, Kalman and Matias. *Fast Inference from Transformers via Speculative Decoding.* ICML 2023.
- [SpecForge deep-dive](/blog/machine-learning/open-source-library/specforge-speculative-decoding-training-deep-dive) and [DeepInfer deep-dive](/blog/machine-learning/inference-frameworks/deepinfer-deepseek-production-inference-engine)

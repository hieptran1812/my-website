---
title: "A is for Absorption: Why Sparse Autoencoder Latents Quietly Stop Firing"
date: "2026-10-15"
publishDate: "2026-10-15"
description: "How the sparsity penalty in sparse autoencoders makes a clean-looking 'starts with S' latent go silent on the word 'short', how Chanin et al. prove it, measure it, and find it in every SAE they tested."
tags: ["paper-reading", "sparse-autoencoders", "feature-absorption", "feature-splitting", "mechanistic-interpretability", "linear-probing", "gemma-scope", "neurips-2025-oral"]
category: "paper-reading"
subcategory: "AI Interpretability"
author: "Hiep Tran"
featured: true
readTime: 21
paper:
  title: "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders"
  authors: "David Chanin, James Wilken-Smith, Tomas Dulka, Hardik Bhatnagar, Satvik Golechha, Joseph Bloom"
  venue: "NeurIPS 2025 (oral)"
  url: "https://arxiv.org/abs/2409.14507"
---

> [!tldr]
> **The claim.** When two concepts form a hierarchy (every "short" token also "starts with S"), a sparse autoencoder can save sparsity by folding the parent direction into the child's latent and switching the parent latent off. Reconstruction stays perfect, so nothing in the training loss objects. The authors call this **feature absorption**.
> **Why it matters.** A latent that looks like a clean "starts with S" detector, with an F1 of 0.81, silently fails on arbitrary tokens. If you use SAE latents as classifiers for bias, deception, or any high-stakes concept, those holes are false negatives you cannot see.
> **Most surprising result.** A three-line proof shows the sparsity loss falls linearly as absorption grows, with slope equal to the co-occurrence probability. Absorption is not a training accident. It is the optimum.
> **Where it fails.** The detection metric needs causal ablations, so it only works up to layer 17 of Gemma 2 2B, and it deliberately ignores partial absorption. The measured rates are a floor, not an estimate.

Sparse autoencoders (SAEs) are the current workhorse of mechanistic interpretability. You train a wide, sparse dictionary on a model's activations, then read off latents like "Golden Gate Bridge" or "starts with the letter S". The promise is that each latent is a **monosemantic** unit: it fires when, and only when, its concept is present.

This paper tests the "only when" direction's twin, recall. Does the "starts with S" latent fire on *every* token that starts with S? The answer is no, and the reason is baked into the objective. Below we walk through the paper's toy model, its proof, its case study on the token "short", and the metric it builds to count the damage across hundreds of Gemma Scope SAEs.

![Figure 1 from Chanin et al. (2025): the ideal interpretable solution fires two latents on "short", the absorption solution fires one](/imgs/blogs/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders-fig1.webp)

Read Figure 1 as two answer keys to the same exam. On the left, latent 1 means "starts with S" and latent 2 means "short", and the word "short" lights up both. On the right, latent 1's encoder has learned "starts with S **and not** short", while latent 2's decoder writes "short" plus "starts S". The right-hand key reconstructs every input just as well and fires one latent instead of two. That is what the SAE learns.

## Background: SAEs, hierarchies, and probes

### The sparse autoencoder

**The problem.** A residual-stream activation in a language model mixes many concepts in superposition: more concept directions than dimensions, overlapping slightly. Individual neurons are therefore polysemantic and hard to read.

**The mechanism.** An SAE projects the activation into a much wider space, applies a nonlinearity that zeroes most entries, and then reconstructs the activation from the few survivors. Training trades reconstruction error against how many latents fire.

**The math.** Given an input activation $a \in \mathbb{R}^{d}$ (one residual-stream vector), the SAE computes

$$
f = \sigma(W_{enc}\, a + b_{enc}), \qquad \hat{a} = W_{dec}\, f + b_{dec}
$$

where $W_{enc} \in \mathbb{R}^{m \times d}$ is the encoder, $W_{dec} \in \mathbb{R}^{d \times m}$ is the decoder, $m \gg d$ is the dictionary width (16k or 65k in this paper), $b_{enc}, b_{dec}$ are biases, and $\sigma$ is a ReLU or a variant such as JumpReLU. The vector $f \in \mathbb{R}^{m}$ holds the latent activations and $\hat{a}$ is the reconstruction. Row $i$ of $W_{enc}$ decides *when* latent $i$ fires; column $i$ of $W_{dec}$ decides *what* it writes back.

That split between encoder and decoder is the whole story of this paper. The encoder is a classifier, the decoder is a dictionary entry, and the two are free to disagree.

**The loss.** Training minimizes reconstruction plus a sparsity term:

$$
\mathcal{L} = \underbrace{\mathbb{E}\,\lVert a - \hat{a} \rVert_2^2}_{\text{reconstruction}} \;+\; \lambda \underbrace{\mathbb{E}\sum_i \lvert f_i \rvert}_{\text{L1 sparsity}}
$$

with $\lambda > 0$ trading the two. TopK and BatchTopK SAEs replace the L1 term with a hard cap on how many latents may fire, but the pressure is the same: fewer active latents is better. People summarize an SAE's sparsity by **L0**, the average number of latents active per token.

### Hierarchical features

The paper's key definition is short. Features $f_1$ and $f_2$ form a **hierarchy**, with $f_1$ the parent and $f_2$ the child, if $f_2 \Rightarrow f_1$: every time the child fires, the parent fires too. "Short" implies "starts with S". "Dog" implies "starts with D". "Algebra" implies "math". Language is full of these.

### Linear and k-sparse probes

A **linear probe** is a logistic-regression (LR) classifier trained on hidden activations. It is the paper's ground-truth yardstick: if an LR probe can read "starts with S" from the residual stream, the information is linearly there.

A **k-sparse probe** is a linear probe restricted to $k$ SAE latents. The authors pick the $k$ latents by training an LR probe with an L1 penalty (coefficient 0.01) on SAE activations and keeping the $k$ largest weights, then refit a plain probe on just those. Growing $k$ from 1 to 15 tells you how many latents it takes to carry a concept.

## The toy model: watching absorption happen

**The problem.** Real LLM features have no ground truth, so you cannot prove an SAE learned the "wrong" thing. A toy model with planted features fixes that.

**The setup.** Four true features, each a random unit vector in $\mathbb{R}^{50}$, mutually orthogonal, each firing with magnitude 1.0. Feature 0 fires with probability 0.25; features 1, 2 and 3 fire with probability 0.05. An input is the sum of whichever features fired. The SAE has exactly 4 latents, L1 coefficient 3e-5, learning rate 3e-4, trained on 100M activations with SAELens.

**Run 1, independent features.** The SAE recovers all four perfectly. Each encoder row and decoder column has cosine similarity 1 with exactly one true feature.

**Run 2, a hierarchy.** Now feature 1 may fire only when feature 0 fires, still at an overall 5% rate. Features 2 and 3 stay independent.

![Figure 2 from Chanin et al. (2025): encoder and decoder cosine similarity with the true features, independent (left) vs feature 1 nested under feature 0 (right)](/imgs/blogs/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders-fig2.webp)

The right half of Figure 2 is the signature of absorption, and it has two parts:

- **Encoder of latent 0 has a hole.** It is positive on feature 0 and *negative* on feature 1. Latent 0 now means "feature 0 and not feature 1".
- **Decoder of latent 1 is a blend.** It points at feature 0 plus feature 1, so firing latent 1 alone rebuilds both.

Latents 2 and 3 are untouched. Only the hierarchical pair gets rewired, and it gets rewired asymmetrically: the encoder and decoder no longer agree. Keep that asymmetry in mind, because the authors suggest it could become a detector.

## Why sparsity rewards absorption: the δ-absorption proof

**The problem.** The toy model shows absorption once. Is it a fluke of initialization, or does the objective actually prefer it?

**The analogy.** Think of two staff at a store entrance. One greets every customer whose name starts with S; another greets anyone named "Short". When Mr. Short walks in, both greet him, which costs two greetings. If the "Short" greeter learns to also say "your name starts with S", the S greeter can stay quiet for Mr. Short. The customer hears the same information with half the effort. Nobody is penalized for the S greeter's new blind spot.

**The mechanism.** Interpolate between the clean solution and the absorbed solution with one knob $\delta \in [0, 1]$. As $\delta$ grows, latent 1's encoder subtracts more of the child direction, and latent 2's decoder adds more of the parent direction. Check that reconstruction never changes, then compute what happens to the sparsity cost.

**The math.** Assume orthonormal features $f_1$ (parent) and $f_2$ (child), a bias-free ReLU SAE, and the family

$$
W_{e,1} = f_1 - \delta f_2, \quad W_{e,2} = f_2, \quad W_{d,1} = f_1, \quad W_{d,2} = f_2 + \delta f_1
$$

where $W_{e,i}$ is latent $i$'s encoder row and $W_{d,i}$ its decoder column. Let $p_{11}$ be the probability that parent and child co-occur, $p_{10}$ the probability that the parent fires alone, and $p_{00}$ the probability neither fires. The hierarchy forces $p_{01} = 0$.

Walk the three possible inputs $h$:

| Input $h$ | $z_1 = \text{ReLU}(W_{e,1} \cdot h)$ | $z_2$ | Reconstruction | $\lvert z_1 \rvert + \lvert z_2 \rvert$ |
| --- | --- | --- | --- | --- |
| $f_1$ (parent alone) | 1 | 0 | $f_1$ | 1 |
| $f_1 + f_2$ (both) | ${1 - \delta}$ | 1 | $(1-\delta) f_1 + f_2 + \delta f_1 = f_1 + f_2$ | ${2 - \delta}$ |
| ${0}$ (neither) | 0 | 0 | ${0}$ | 0 |

Every row reconstructs exactly, for every $\delta$. The reconstruction loss is identically zero across the whole family. The expected sparsity cost is the probability-weighted last column:

$$
L_{sp}^{(1,2)}(\delta) = p_{11}(2 - \delta) + p_{10}, \qquad \frac{dL_{sp}^{(1,2)}}{d\delta} = -p_{11}
$$

The derivative is negative whenever the child ever appears. Gradient descent on $\mathcal{L} = L_{rec} + \lambda L_{sp}$ therefore walks $\delta$ to 1: full absorption.

![delta-absorption: the clean solution and the absorbed solution reconstruct "short" identically, but absorption fires one latent instead of two](/imgs/blogs/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders-1.webp)

**Worked example.** Plug in the toy model's own rates. Feature 1 fires 5% of the time, always with feature 0, so $p_{11} = 0.05$. Feature 0 fires 25% of the time overall, so it fires alone $p_{10} = 0.25 - 0.05 = 0.20$. Then:

- No absorption: $L_{sp}(0) = 0.05 \times 2 + 0.20 = 0.30$.
- Full absorption: $L_{sp}(1) = 0.05 \times 1 + 0.20 = 0.25$.

That is a free 1/6 cut in this pair's sparsity cost (my arithmetic, from the paper's formula), bought with zero reconstruction error. Any $\lambda > 0$ takes the deal.

**When it fails, or rather, when it bends.** The proof covers one hand-picked family of weights; it shows absorption is *a* descent direction, not that it is the global optimum of every SAE. The authors' extended toy runs fill in the messier cases:

- **Magnitude variance gives partial absorption.** If feature 0's magnitude varies (variance 0.1), the absorbing decoder can no longer encode the right ratio, so the parent latent fires weakly as a correction. With both features at 1.0, it fires at 0.22. At parent magnitude 0.90 it fires at 0.12, and at 0.75 it switches off entirely.
- **Imperfect co-occurrence still absorbs.** If feature 1 rides with feature 0 95% of the time, the parent latent fires at 0.67 when both are present versus 1.00 alone.
- **TopK does not escape.** A batch TopK SAE on a 12-feature toy with no L1 term shows the same encoder hole and blended decoder. Fewer active latents buy better reconstruction at fixed $k$, so the incentive survives.

## The LLM testbed: spelling the first letter

**Why first letters.** Ground truth is free. Every token made of English letters has a known first letter, so you can train a perfect-label probe and check each SAE latent against it. And the hierarchy is guaranteed: any token-identity feature ("short") implies a first-letter feature ("starts with S").

**The prompt.** The authors use in-context examples of the form `{token} has the first letter: {LETTER}`, for instance:

```
 tartan has the first letter: T
 mirth has the first letter: M
 dog has the first letter:
```

They read residual activations at the `dog` position, train LR probes on an 80/20 split, and evaluate everything on the held-out 20%. When they splice an SAE into the model, they keep the SAE error term so the model's output does not change.

**The causal metric.** To measure whether a latent matters, they zero it and watch the logit of the right letter against the others:

$$
m = g[y] - \frac{1}{\lvert L \rvert - 1} \sum_{l \in L \setminus \{y\}} g[l]
$$

where $g$ is the final-token logit vector, $L$ is the set of 26 uppercase letter tokens, and $y$ is the correct letter. A latent's **ablation effect** is $m$ before minus $m$ after setting that latent to zero.

**Worked example.** Say the logit for "D" is 10 and the other 25 letters average 2. Then $m = 10 - 2 = 8$. If zeroing a latent drops "D" to 4 while the others hold at 2, then $m$ becomes 2 and the latent's ablation effect is 6: it carried most of the model's confidence about the first letter. (These numbers are illustrative, mine.)

The models are Gemma-2-2B with the full Gemma Scope residual SAE suite at widths 16k and 65k, plus the authors' own L1 SAEs on the first 8 layers of Qwen2 0.5B and Llama 3.2 1B and TopK SAEs on Llama 3.2 1B.

## Do SAEs learn first-letter latents at all?

For each letter, the authors pick the SAE latent whose encoder has the highest cosine similarity with that letter's LR probe, yielding 26 "first-letter" latents per SAE. Usually one latent, or at most a couple, stands out clearly. Choosing by k=1 sparse probing instead gives nearly identical picks.

![Figure 4 from Chanin et al. (2025): mean F1 of the best first-letter SAE latent by layer (against the LR probe) and by L0](/imgs/blogs/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders-fig4.webp)

Three findings from Figure 4 and its companion Figure 5:

- **No SAE matches the probe.** The LR probe line sits above every SAE at every layer. The information is in the residual stream; the SAE latents just do not capture all of it.
- **L0 sets the precision/recall trade.** Low-L0 SAEs learn high-precision, low-recall latents; high-L0 SAEs learn low-precision, high-recall ones.
- **The sweet spot moves with depth.** Layers 0 to 12 learn the cleanest first-letter latents at L0 near 25 to 50; layers 13 to 25 at L0 near 50 to 100. The authors caution against reading an "optimal L0" off this, because the per-letter breakdown varies widely.

Low recall is the symptom. The next section finds the cause.

## The case study: "short" and the missing S

Take the Gemma Scope layer 3, 16k-wide SAE with L0 of 59. Its latent 6510 looks like a textbook "starts with S" classifier, with F1 of 0.81. Yet it does not fire on the token `short`, which the probe classifies correctly and the model can spell.

![Figure 6 from Chanin et al. (2025): latent 6510 fires on S-tokens except "short", where token latent 1085 fires instead; right, decoder cosine similarity with the S probe](/imgs/blogs/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders-fig6.webp)

In the left panel of Figure 6, latent 6510 lights up on snake, soggy, steam and the rest, then drops to zero on `short`. Latent 1085, a token-aligned latent that fires on variants of "short" (" short", "SHORT"), fires instead. The right panel shows why that is enough:

| Latent | Role | Decoder cos sim with "S" probe | Activation on `short` |
| --- | --- | --- | --- |
| 6510 | main "starts with S" | 0.52 | 0 |
| 1085 | "short" token | 0.12 | about 5x a typical 6510 activation |

Latent 1085 has roughly a fifth of the probe alignment but fires at roughly five times the magnitude, so it writes a similar amount of "starts with S" into the residual stream. Small alignment times big activation equals the same signal. This is the toy model's blended decoder, found in the wild.

### Proving it is causal

Cosine similarity is suggestive, not causal. So the authors run the ablation sweep on the `short` token.

![Figure 7 from Chanin et al. (2025): ablation effects on the "short" token before (left) and after (right) projecting the probe direction out of latent 1085](/imgs/blogs/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders-fig7.webp)

On the left, latent 1085 has an ablation effect far larger than any other latent (the appendix puts it above 6, with everything else well below 0.5). Then the control: project the probe direction *out* of latent 1085's decoder and ablate again. On the right, 1085 disappears from the chart. The causally important part of the "short" latent is exactly its sliver of "starts with S".

The authors call the result a false sense of understanding: label 6510 "starts with S" from its top activations and you are silently wrong on an arbitrary set of tokens.

## Feature splitting: the benign cousin

**The problem.** Before counting absorption, you have to rule out something that looks similar but is harmless. **Feature splitting**, first described by Bricken et al., is when one latent in a small SAE becomes several more specific latents in a larger one.

**The example.** In the Gemma Scope layer 0, 16k-wide, L0 105 SAE, two latents align with the "L" probe. Latent 7112 fires on tokens starting with lowercase "l"; latent 7657 on uppercase "L". Together they cover "starts with L". That is fine: both are easy to find, and depending on your use either granularity could be the useful one.

**The mechanism.** Detect splits with k-sparse probing. If going from $k$ to $k+1$ latents raises the probe's F1 by more than a threshold, the extra latent carries real signal and the set is a split:

$$
\text{split at } k+1 \iff F1(k+1) - F1(k) > \tau, \qquad \tau = 0.03
$$

For "L", F1 jumps by about 0.08 from $k=1$ to $k=2$, then each further step adds less than 0.01. For "N", every step adds less than 0.01: no split. Any $\tau$ between 0.01 and 0.05 separates the two; 0.03 sits in the middle. Across Gemma Scope, splits per letter grow as L0 falls, that is, with higher sparsity.

The difference that matters: a split leaves you with a *set* of latents that covers the concept. Absorption leaves holes that no small set of concept latents covers, because the missing cases live inside hundreds of token latents.

## Counting absorption: the metric

**The problem.** One case study proves existence. To compare hundreds of SAEs, you need a rate.

**The mechanism.** For each letter, the metric hunts for tokens where the main latents went silent and a different latent carried the letter instead.

![The absorption-rate pipeline: probe, split detection, false negatives, ablation, and a two-threshold test](/imgs/blogs/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders-2.webp)

1. Train the LR probe for the letter.
2. Use k-sparse probing with $\tau = 0.03$ to collect the letter's main latents, including any splits.
3. Find **false negatives**: test tokens the probe classifies correctly but on which *every* main latent stays at zero.
4. Run an integrated-gradients approximation of the ablation sweep on each false negative (sampling 200 if there are more).
5. Call it absorption if the latent with the largest negative ablation effect has probe cosine similarity above 0.025 **and** leads the runner-up by at least 1.0.

**The math.**

$$
\text{absorption rate} = \frac{\text{num absorptions}}{\text{LR probe true positives}}
$$

**Worked example.** Suppose for letter "D" the probe gets 1,000 true positives, of which 300 are false negatives for the SAE. The metric samples 200 and flags 40 as absorption. Scaling the sample back up gives about ${40 \times 300 / 200 = 60}$ absorptions and a rate of ${60 / 1000 = 0.06}$. (Illustrative numbers and scaling are mine; the paper states only the 200-sample cap.)

**Why the thresholds.** The 0.025 cosine floor is a cheap filter to skip ablations on latents with no probe component; typical absorbing latents sit at 0.05 to 0.15, and anything between 0 and 0.05 works. The 1.0 lead demands a single dominant absorber; anything from 0.5 to 3.0 works. The authors chose round midpoints of ranges that give similar results.

**When it fails.** The metric is built to be conservative, and it says so:

- It misses **multiple absorbers** firing together, since it demands one dominant latent.
- It misses **partial absorption**, since the main latents must be exactly zero.
- It only works where ablation works. In Gemma 2 2B, attention moves the first-letter information from the word token to the answer position around layer 18, so ablating at the word token later has no effect. The analysis therefore stops at layer 17.

The appendix sketches a looser variant that replaces ablation with probe projection, so a latent $l$ counts as absorbing if its reconstruction share of the probe direction clears a threshold: $\tau_c \lt \hat{a}_l \cdot d_p \,/\, (a \cdot d_p)$, where $d_p$ is the probe direction and $\hat{a}_l$ is latent $l$'s activation times its decoder vector. That version could run at every layer and count partial absorption. The paper does not use it.

## Results: absorption everywhere

![Figure 9 from Chanin et al. (2025): mean absorption rate vs layer and vs L0 for Gemma Scope, and vs layer for Llama 3.2 1B and Qwen2 0.5B SAEs](/imgs/blogs/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders-fig9.webp)

| Question | Setting | Finding (source) |
| --- | --- | --- |
| Does it depend on layer? | Gemma Scope 16k and 65k, layers 0 to 17 | No obvious pattern (Fig. 9a) |
| Does sparsity matter? | Gemma Scope, varying L0 | Lower L0 (sparser) means more absorption (Fig. 9b) |
| Does width matter? | 16k vs 65k | Wider SAEs absorb more (Fig. 9b) |
| Is it Gemma-specific? | Own L1 SAEs on Qwen2 0.5B and Llama 3.2 1B, TopK on Llama, layers 0 to 8 | Absorption in all of them (Fig. 9c) |
| Is it L1-specific? | Toy batch TopK SAE | Same encoder hole and blended decoder (App. A.3) |
| Can latents steer the letter? | Zero the true-letter latent, set a new letter's latent to its mean | Works best with 16k SAEs at L0 75 to 150, layers 7 to 9, with large per-token variance (App. A.6) |

The custom SAEs used L1 coefficient 2.5 and 500M training tokens on one 40 GB A100 each. The Qwen2 SAEs landed at L0 25 to 50 with explained variance 0.77 to 0.83; the Llama SAEs at L0 27 to 110 with explained variance 0.74 to 0.89.

The uncomfortable part is the direction of the trends. The knobs practitioners turn to get "more interpretable" dictionaries, wider and sparser, are the knobs that increase absorption. Varying width or sparsity does not make it go away.

### What is load-bearing and might not transfer

- **The task is unusually clean.** First letters give perfect labels and a guaranteed hierarchy. Most concepts people care about ("deception", "bias") have fuzzy labels, so you cannot even compute false negatives without a probe you trust.
- **The probe is the ground truth.** A token counts as a miss only if the LR probe gets it right. Where the probe is weak, absorption is invisible to this metric.
- **The layer window is narrow.** Results cover layers 0 to 17 of Gemma 2 2B and 0 to 8 of the smaller models. The authors believe absorption continues deeper but cannot certify it causally.

## Splitting, absorption, and partial absorption side by side

![Feature splitting vs full absorption vs partial absorption: what you observe, what the main latent does, and whether it hurts](/imgs/blogs/a-is-for-absorption-studying-feature-splitting-and-absorption-in-sparse-autoencoders-3.webp)

The three differ in one question: on a positive example, what does the main latent do? Under splitting, one of a small, findable set fires. Under full absorption it reads exactly zero, which the metric catches. Under partial absorption it fires weakly, which the metric deliberately does not count.

## Critique

**What is strong.** The paper ties a theory (the δ-family proof), a controlled toy, a single vivid case, and a population-scale measurement together, and each step uses a causal check rather than activation-pattern eyeballing. The control in Figure 7, projecting out the probe direction and watching the effect vanish, is the kind of test interpretability papers often skip. The metric is honest about being a lower bound.

**What is weak.**

- **The proof is local.** It shows absorption lowers loss within one hand-built family. It does not show the trained SAE reaches that family, or rule out other solutions with the same loss. The toy experiments cover that gap empirically, not formally.
- **Thresholds are hand-tuned.** The 0.03, 0.025 and 1.0 values come from manual inspection. The authors show ranges that behave similarly, but there is no sensitivity curve for the final rate.
- **No absolute headline rate in the text.** The trends are clear, but a reader has to read plot axes to know whether "absorption" means 2% of positives or 30%. I would have liked a table of rates per SAE.
- **One task.** Spelling is a perfect lab specimen and possibly an unusual one: first-letter information is dense and every token carries it. Sparser parent features may absorb less.
- **A small inconsistency.** The appendix text describes the TopK toy with $k = 2$, while the caption says $k = 12$. It does not change the conclusion.

**Missing ablation.** The paper names candidate fixes (Meta-SAEs, attribution dictionary learning, group lasso, hierarchical sparse coding) but tests none. Even one toy-model run with a structured-sparsity penalty would show whether the incentive can be broken. Later work did take this up: Matryoshka SAEs explicitly target absorption by nesting dictionaries.

**What would change my mind.** If an SAE trained with a hierarchy-aware objective showed near-zero absorption on this exact first-letter benchmark *and* matched the LR probe's F1, I would stop treating absorption as a fundamental cost of sparse dictionaries and start treating it as a fixable objective bug. Conversely, if the probe-projection variant of the metric showed rates far above the conservative one at deep layers, I would conclude this paper understates the problem by a wide margin.

### What I would do with this in practice

This section is my extrapolation, not the paper's.

- **Never treat a single SAE latent as a complete detector.** Pair it with an LR probe on the same concept and check the latent's recall against the probe's, the way the paper does. This is also the lesson of the sparse-probing case study.
- **Expect circuits to be wider than they look.** If a concept leaks into hundreds of token latents, a [sparse feature circuit](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models) built from the "main" latent will miss edges.
- **Look at encoder/decoder disagreement.** A latent whose decoder has a component its encoder actively subtracts elsewhere is the absorption fingerprint from the toy model. The authors flag this as a possible detector.

## References

- Chanin, Wilken-Smith, Dulka, Bhatnagar, Golechha, Bloom. *A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders.* NeurIPS 2025. [arXiv:2409.14507](https://arxiv.org/abs/2409.14507)
- Code: [github.com/lasr-spelling/sae-spelling](https://github.com/lasr-spelling/sae-spelling). Interactive results explorer: [feature-absorption.streamlit.app](https://feature-absorption.streamlit.app)
- Bricken et al. *Towards Monosemanticity: Decomposing Language Models with Dictionary Learning.* Transformer Circuits Thread, 2023.
- Gurnee et al. *Finding Neurons in a Haystack: Case Studies with Sparse Probing.* 2023.
- Marks et al. *Sparse Feature Circuits.* See our analysis: [Sparse Feature Circuits](/blog/paper-reading/ai-interpretability/sparse-feature-circuits-discovering-and-editing-interpretable-causal-graphs-in-language-models)
- Related on this site: Learning Multi-Level Features with Matryoshka SAEs, Are Sparse Autoencoders Useful? A Case Study in Sparse Probing

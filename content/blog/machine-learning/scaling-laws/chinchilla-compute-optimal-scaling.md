---
title: "Chinchilla: compute-optimal training and the 20-tokens-per-parameter rule"
date: "2026-06-15"
description: "Learn how DeepMind's Chinchilla paper rewrote the rules of LLM training: the parametric loss law, the IsoFLOP method, and the 20-tokens-per-parameter recipe you can apply to any compute budget."
tags: ["scaling-laws", "chinchilla", "compute-optimal", "large-language-models", "training-compute", "isoflop", "tokens-per-parameter", "deepmind", "loss-curves", "pretraining", "gopher"]
category: "machine-learning"
subcategory: "Scaling Laws"
author: "Hiep Tran"
featured: true
readTime: 52
---

In early 2022 the conventional wisdom about how to spend a training budget was simple, confident, and wrong. If you had a pile of GPUs, you built the biggest model you could fit and fed it whatever data you happened to have. GPT-3 was 175B parameters trained on roughly 300B tokens. Gopher was 280B parameters on 300B tokens. MT-NLG pushed to 530B parameters on about 270B tokens. The implicit theory, inherited from the [Kaplan 2020 scaling laws](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models), was that parameters were where the loss lived: spend on size, stop training early, move on. Then DeepMind trained a 70B model called Chinchilla on 1.4 trillion tokens, and it beat every one of those larger models at the same training compute. The diagram below is the mental model for the whole post: a fixed compute budget should split roughly evenly between how big you make the model and how many tokens you train it on, which lands you at about 20 tokens per parameter.

![A graph showing how a fixed training compute budget of six times N times D splits through three estimation methods into roughly equal exponents for model size and tokens, converging on the twenty tokens per parameter rule](/imgs/blogs/chinchilla-compute-optimal-scaling-1.png)

That single result, from Hoffmann et al. 2022 ("Training Compute-Optimal Large Language Models," arXiv:2203.15556), reorganized how the field thinks about pretraining. It said that a generation of headline models were *severely undertrained* — that they had spent their compute on the wrong axis. It introduced a clean, fittable loss law with concrete constants. And it gave practitioners a recipe so simple it fits on an index card: for a budget $C$, pick model size $N_{opt} \propto \sqrt{C}$ and feed about $20N$ tokens. This post is a full tour of that result. We will build the intuition first, then derive the math, then work numeric examples in FLOPs and dollars, then look at exactly how Chinchilla beat the giants, and finally turn it into practical guidance — including the important caveat that the exact coefficients are softer than the original paper made them look.

> [!important] The one number to remember: about 20 tokens per parameter
> - **The loss law is $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$** — an irreducible floor $E$ plus two penalties that shrink as you add parameters $N$ and tokens $D$. DeepMind's original fit: $E = 1.69$, $A = 406.4$, $\alpha = 0.34$, $B = 410.7$, $\beta = 0.28$.
> - **Because $\alpha \approx \beta$, you should scale $N$ and $D$ about equally.** Both compute-optimal exponents land near one-half: $N_{opt} \propto C^{0.5}$ and $D_{opt} \propto C^{0.5}$.
> - **At the compute-optimal point, $D/N \approx 20$ tokens per parameter.** Chinchilla itself is 70B parameters on 1.4T tokens (about 20:1); Gopher was 280B on 300B (about 1:1).
> - **Three independent methods agree.** Training curves, IsoFLOP profiles, and a parametric fit all put the exponents near 0.5 — strong evidence the result is real and not an artifact of one estimator.
> - **Most 2020–2022 flagship models were undertrained.** A compute-optimal 70B model beats Gopher 280B, GPT-3 175B, and MT-NLG 530B at *equal training compute*, and is far cheaper to serve.
> - **The recipe:** for budget $C$ FLOPs, set $N_{opt} \approx \sqrt{C/120}$ and $D \approx 20 N_{opt}$, using $C \approx 6ND$.
> - **Caveat:** a 2024 replication (Besiroglu et al.) found the original parametric fit reproduces poorly with implausibly tight error bars. The ~20:1 conclusion survives; the exact constants do not. The full story is in the [reconciliation post](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation).

## Why this is different from what came before

The right way to feel the Chinchilla result is to notice how badly it contradicts the prior consensus, and how confidently that consensus had been stated. Here is the before-and-after, framed as the kind of assumption-versus-reality table a senior engineer keeps in their head.

| Question | The 2020–2021 assumption (post-Kaplan) | The Chinchilla reality (2022) |
|---|---|---|
| Where does loss reduction come from? | Mostly parameters; bigger is the lever | Parameters *and* data, in nearly equal measure |
| Given budget $C$, how big a model? | Very large: $N_{opt} \propto C^{0.73}$ | Moderate: $N_{opt} \propto C^{0.5}$ |
| How many tokens? | Relatively few: $D_{opt} \propto C^{0.27}$ | Many: $D_{opt} \propto C^{0.5}$ |
| Tokens per parameter at optimum? | Roughly a few (GPT-3 ≈ 1.7:1) | About 20:1 |
| Should you train to convergence? | No — stop early, the model is what matters | Train far longer than people did; data is the missing half |
| Was GPT-3 well-allocated? | Yes, by design | No — it was undertrained for its size |

The two rows that matter most are the exponents. Kaplan's analysis said that when you get more compute, you should pour roughly three-quarters of the new budget into making the model bigger and only a quarter into more tokens. Chinchilla said you should split the new budget about evenly. That sounds like a small numerical disagreement. It is not. Over several orders of magnitude of compute, the difference between an exponent of 0.73 and 0.5 compounds into models that differ by *4x or more in size* at the same budget — and at the scales the labs were operating, that gap was the difference between a 280B model trained on 300B tokens and a 70B model trained on 1.4T tokens.

> If you take one thing from this post: the loss is not stored in the parameters. It is stored, in roughly equal halves, in the parameters and the tokens. Starve either half and you waste the other.

Why did the field get this wrong for two years? The honest answer is bookkeeping, and it is worth a sentence here even though the [full forensic account](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation) is its own post. Kaplan counted *non-embedding* parameters and made some small-scale measurement choices that, at the model sizes he studied, biased the exponent upward. Chinchilla counted *total* parameters and measured across a much wider, cleaner sweep. The networks were always scaling the same way; the two papers were just counting differently. Keep that in your back pocket — it is the deepest lesson of this whole series — but for now we will take Chinchilla's numbers as the corrected answer and understand them on their own terms.

### A short history of how we got here

It is worth tracing the chain of ideas, because Chinchilla did not arrive from nowhere — it was the third step in a research program, and understanding the lineage makes the result less surprising and more inevitable.

The first step was empirical, and it predates the language-model boom. Hestness et al. at Baidu (2017) showed that generalization error falls as a power law in dataset size across vision, speech, and language — a straight line on log-log axes, with an exponent that depends on the problem but barely on the architecture. That paper established the core fact the whole field rests on: loss is *predictable* before you train, because it lives on a smooth power-law curve. Rosenfeld et al. (2019) extended this to a joint function of model size *and* data size, with a smooth transition out of the small-data random-guessing regime. The seed of the $L(N, D)$ form was already there.

The second step was Kaplan et al. (2020), which made the program concrete for transformers. It fit single-variable power laws for loss versus parameters, versus data, and versus compute, over seven orders of magnitude, and — critically — derived a compute-optimal allocation. Its conclusion, $N_{opt} \propto C^{0.73}$, told everyone to spend mostly on parameters. This was not a careless paper; it was rigorous, influential, and it directly shaped GPT-3's design (175B parameters, ~300B tokens — almost exactly what the Kaplan exponent recommends). For two years, this was simply *how you trained a large language model*. The [Kaplan post](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models) covers it in full.

The third step is Chinchilla. DeepMind, having built Gopher under the Kaplan-era assumptions, went back and re-measured the frontier with a much larger and more careful experimental sweep — over 400 models in some accounts — and three independent estimation methods. They found the exponents were near 0.5, not 0.73. And then, instead of just publishing a number, they did the thing that made the result undeniable: they trained a model at the new optimum (70B/1.4T) on the same budget as Gopher (280B/300B) and showed it won. That head-to-head, same-budget comparison is why Chinchilla is the paper people cite, even though the conceptual groundwork was laid years earlier. The lesson of the lineage is that scaling laws are an *empirical* discipline: the form was guessed early, but the constants — and the allocation they imply — had to be measured carefully, and a measurement done at the wrong scale or with the wrong bookkeeping can mislead an entire field.

## 1. The loss law: three terms that explain everything

**Senior rule of thumb: before you fit anything, write down the functional form you expect, and make sure each term has a physical meaning you can defend.** Chinchilla's whole edifice rests on one equation, and its power comes from the fact that every piece of it means something.

$$L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}$$

Here $L$ is the loss in nats per token (cross-entropy on held-out text), $N$ is the number of parameters, and $D$ is the number of training tokens. The three terms are additive, and that additivity is the entire intuition. The figure below pulls them apart.

![Diagram breaking the Chinchilla loss law into three additive boxes: an irreducible entropy floor of one point six nine nats, a finite-model penalty that shrinks with parameters, and a finite-data penalty that shrinks with tokens, with the corrected replication constants noted](/imgs/blogs/chinchilla-compute-optimal-scaling-2.png)

The first term, $E$, is the **irreducible loss** — the entropy floor of natural language itself. Even a perfect model with infinite parameters trained on infinite data would still pay this, because language has genuine residual uncertainty (you cannot predict the next token of a sentence you have never seen with zero error). DeepMind fit $E = 1.69$ nats per token. No amount of model or data removes it; it is the asymptote everything else approaches.

The second term, $A/N^{\alpha}$, is the **finite-model penalty**: the cost of having too few parameters to represent the function you are trying to learn. It vanishes as $N \to \infty$. The fitted constants are $A = 406.4$ and $\alpha = 0.34$. The exponent $\alpha$ tells you how fast loss falls when you add parameters and hold data fixed — bigger $\alpha$ means parameters buy you more.

The third term, $B/D^{\beta}$, is the **finite-data penalty**: the cost of not having seen enough tokens to estimate the function well. It vanishes as $D \to \infty$. The fitted constants are $B = 410.7$ and $\beta = 0.28$. The exponent $\beta$ tells you how fast loss falls when you add tokens and hold parameters fixed.

### Why $\alpha \approx \beta$ is the whole story

Look at those two exponents: $\alpha = 0.34$ and $\beta = 0.28$. They are close. That near-equality is not a curiosity; it is the result. If parameters and data buy loss reduction at roughly the same rate, then a rational budget should buy roughly equal amounts of each. If $\alpha$ were much larger than $\beta$ (parameters cheap, data expensive in loss terms), you would want a big model on little data — which is exactly what Kaplan's higher $a$ exponent implied. The fact that the two penalty terms fall at nearly the same rate is the mathematical reason the compute-optimal split is roughly 50/50.

There is a subtlety worth flagging. The constants $A = 406.4$ and $B = 410.7$ are also nearly equal, but you should not read too much into that — they carry units that depend on how $N$ and $D$ are measured, so their numeric closeness is partly a coincidence of scaling. The load-bearing comparison is between the *exponents*, because exponents are dimensionless slopes on a log-log plot and are directly comparable. When someone tells you "Chinchilla says scale data and model equally," the evidence they are pointing at is $\alpha \approx \beta$.

### A quick sanity-check in code

It helps to make the law concrete. Here is the loss surface evaluated at a few points, using the original constants, so you can see the three terms competing.

```python
import numpy as np

# DeepMind's original Approach-3 fit (Hoffmann et al. 2022, arXiv:2203.15556).
E, A, alpha = 1.69, 406.4, 0.34
B, beta     = 410.7, 0.28

def loss(N, D):
    """Predicted loss in nats/token for N params, D tokens."""
    return E + A / N**alpha + B / D**beta

# Gopher: 280B params, 300B tokens (about 1 token/param).
# Chinchilla: 70B params, 1.4T tokens (about 20 tokens/param).
for name, N, D in [("Gopher",     280e9, 300e9),
                   ("Chinchilla",  70e9, 1.4e12),
                   ("GPT-3",      175e9, 300e9)]:
    Lm = A / N**alpha     # finite-model penalty
    Ld = B / D**beta      # finite-data penalty
    print(f"{name:11s}  L={loss(N,D):.3f}  model-pen={Lm:.3f}  data-pen={Ld:.3f}")
```

Run it and the pattern jumps out. Gopher's *data* penalty dominates its *model* penalty — it is a huge model bottlenecked by too few tokens. Chinchilla, with one quarter the parameters but nearly five times the tokens, balances the two penalties and lands at a lower total loss. That imbalance is the visual signature of an undertrained model: one penalty term far larger than the other. The whole compute-optimal program is, in one sentence, *the art of balancing the two penalty terms subject to a compute constraint.*

### A note on units: nats, bits, and perplexity

It is worth pausing on what $L = 1.69$ actually means, because the unit trips people up. The loss is cross-entropy measured in *nats* per token — natural-log units. To convert to bits, divide by $\ln 2 \approx 0.693$: so $1.69$ nats is about $2.44$ bits per token, meaning a perfect model at the irreducible floor would still need about 2.44 bits, on average, to encode each token of text under this tokenizer. To convert to *perplexity*, the metric many practitioners actually report, exponentiate: perplexity $= e^{L}$, so $e^{1.69} \approx 5.4$. A perplexity of 5.4 at the floor says that, on average, the model is as uncertain as if it were choosing uniformly among about 5.4 equally-likely next tokens — even with infinite capacity and data. Every improvement you buy with more parameters or tokens is a reduction *toward* that floor, never below it. This is why chasing perplexity has diminishing returns: you are not racing to zero, you are asymptoting to a fixed entropy floor set by the language and the tokenizer, and the closer you get the harder each increment becomes.

The dependence on the tokenizer is also why borrowed constants do not transfer cleanly. If you change the tokenizer — a different vocabulary size, byte-level versus subword, a different language mix — you change how much information each "token" carries, which shifts $E$ and rescales $A$ and $B$. The exponents are more robust because they describe *slopes* (how fast loss falls), not absolute levels, but even they can drift with the data distribution. Always interpret a loss number together with the tokenizer it was measured under; a perplexity comparison across different tokenizers is meaningless.

### The compute constraint: $C \approx 6ND$

We have not yet connected the loss law to compute. The bridge is a back-of-envelope accounting fact that you should commit to memory: **training a dense transformer costs about $6ND$ floating-point operations**, where the 6 comes from roughly 2 FLOPs per parameter for the forward pass and about 4 for the backward pass (the backward pass touches each weight twice — once for the gradient with respect to the weight, once for the gradient with respect to the input). So a forward-backward step over the whole dataset is approximately $6 \times (\text{params}) \times (\text{tokens})$.

This is the constraint that turns the loss law into an optimization problem. You do not get to make both $N$ and $D$ large for free; their product is pinned by your budget $C$. Inference, for contrast, costs about $2N$ FLOPs per generated token (forward pass only) — a number that will matter enormously in the [inference-aware scaling post](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws), because it means a model you train once gets *served* millions of times, and serving cost scales with $N$, not with how many tokens you trained on.

So the question Chinchilla actually answers is: **minimize $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$ subject to $6ND = C$.** Everything that follows is the answer to that constrained-optimization problem, computed three different ways.

## 2. Solving the optimization: where the square root comes from

**Senior rule of thumb: when a result has a clean closed form, derive it once so you understand which assumptions it depends on.** The $\sqrt{C}$ scaling is not magic; it falls directly out of minimizing the loss law under the compute constraint, and the derivation tells you exactly when it breaks.

We want to minimize $L(N, D) = E + A N^{-\alpha} + B D^{-\beta}$ subject to $C = 6ND$. Use a Lagrange multiplier, or just substitute $D = C/(6N)$ and minimize over $N$ alone. Substituting:

$$L(N) = E + A N^{-\alpha} + B \left(\frac{C}{6N}\right)^{-\beta} = E + A N^{-\alpha} + B \left(\frac{6}{C}\right)^{\beta} N^{\beta}$$

Take the derivative with respect to $N$ and set it to zero:

$$-\alpha A N^{-\alpha - 1} + \beta B \left(\frac{6}{C}\right)^{\beta} N^{\beta - 1} = 0$$

Rearranging to isolate $N$:

$$\alpha A N^{-\alpha} = \beta B \left(\frac{6}{C}\right)^{\beta} N^{\beta}$$

$$N^{\alpha + \beta} = \frac{\alpha A}{\beta B} \left(\frac{C}{6}\right)^{\beta}$$

$$N_{opt} \propto C^{\frac{\beta}{\alpha + \beta}}$$

There it is. The compute-optimal model size is a power of compute with exponent $a = \beta/(\alpha + \beta)$, and by symmetry the optimal token count is $D_{opt} \propto C^{a'}$ with $a' = \alpha/(\alpha + \beta)$. Plug in the fitted exponents:

$$a = \frac{\beta}{\alpha + \beta} = \frac{0.28}{0.34 + 0.28} = \frac{0.28}{0.62} \approx 0.45$$

$$a' = \frac{\alpha}{\alpha + \beta} = \frac{0.34}{0.62} \approx 0.55$$

So the parametric fit gives $N_{opt} \propto C^{0.45}$ and $D_{opt} \propto C^{0.55}$ — both near one-half, and they must sum to 1 because $aN \cdot a'D$ scales as $C^{a + a'} = C^1 = C$, which is the constraint. The "20 tokens per parameter" rule is the $D_{opt}/N_{opt}$ ratio evaluated at the budget DeepMind cared about. The crucial structural fact is that **the two exponents always sum to 1** (it is forced by the constraint), so the only free question is how the unit budget splits — and the answer depends entirely on $\alpha$ versus $\beta$.

This is why the exponents being near-equal is everything. If $\alpha = \beta$ exactly, then $a = a' = 0.5$ and you scale model and data identically. The small gap ($0.34$ vs $0.28$) tilts the split slightly toward more tokens than parameters, which is why the data exponent ($0.55$) edges out the model exponent ($0.45$). The derivation also tells you the failure mode: if your fitted $\alpha$ and $\beta$ are wrong (and we will see that they are softer than advertised), your split is wrong by exactly the corresponding amount. Garbage exponents in, garbage allocation out.

## 3. Three roads to the same frontier

**Senior rule of thumb: when a single estimator gives a surprising answer, the result is only trustworthy if independent methods agree.** Chinchilla's authors knew the result was going to overturn the field, so they did not estimate the frontier once. They estimated it three different ways, with three different sets of assumptions, and checked that the answers converged. The matrix below summarizes all three.

![Matrix comparing the three Chinchilla estimation methods by what is held fixed, the N exponent, the D exponent, and the predicted model and token counts at the Gopher compute budget](/imgs/blogs/chinchilla-compute-optimal-scaling-4.png)

The three approaches differ in what they hold fixed and what they vary, which means they make different mistakes — and the fact that their mistakes do not push the answer in the same direction is the strongest possible evidence the answer is real. Let us walk each one.

### Approach 1: fix the model size, vary the tokens

The first method is the most direct and the easiest to picture. Take a model of a given size — say 1B parameters — and train it, logging the loss at every step. Because steps consume tokens at a fixed rate, that training curve *is* a curve of loss versus tokens for that model size. Now repeat for a family of model sizes, from small to large. You get a fan of training curves.

For any horizontal slice at a fixed compute budget $C$ (remember $C = 6ND$, so a fixed $C$ is a hyperbola in the $N$–$D$ plane), you can ask: among all these models, which one reaches the lowest loss by the time it has spent $C$ FLOPs? That model is the compute-optimal size for that budget. Trace the minimum across budgets and you have the frontier. DeepMind ran this and found $a \approx 0.50$ and $b \approx 0.50$ — model and data scaling almost exactly equally. Extrapolated to Gopher's compute budget, this approach predicts a compute-optimal model of about **67B parameters on 1.5T tokens.** Gopher was 280B on 300B. The method is saying, point-blank, that Gopher should have been roughly a quarter the size and trained on five times the data.

### Approach 2: IsoFLOP profiles (the one to internalize)

The second method is the most visually compelling and, in my experience, the one that makes the whole result click. Instead of fixing the model size, you fix the *compute budget* and vary the model size. For a single budget — say $6 \times 10^{18}$ FLOPs — you train a range of models of different sizes, each for exactly that many FLOPs. A small model finishes its FLOP budget having seen an enormous number of tokens (because $D = C/6N$ is large when $N$ is small); a large model finishes having seen very few tokens. You plot the final loss of each against its parameter count.

What you get is a **U-shaped curve**, called an IsoFLOP curve ("iso" = equal, so "equal-FLOP"). The figure below shows three of them, one per budget.

![A hand-drawn line chart showing three U-shaped IsoFLOP valleys of training loss versus model size at three fixed compute budgets, with the valley bottoms moving right and down as budget grows to trace the compute-optimal frontier](/imgs/blogs/chinchilla-compute-optimal-scaling-3.png)

Read the U from left to right. On the far left, the model is too small. It has burned its entire FLOP budget chewing through a vast number of tokens, but the model simply does not have the capacity to absorb them — it has effectively *overtrained* a tiny network, and the extra data is wasted because the model saturated long ago. Loss is high. As you move right and the model grows, loss falls: the model is now big enough to use the data it sees. You reach the bottom of the valley — the single best model size for this exact budget. Keep going right and loss rises again: now the model is too big, it has burned its FLOP budget before seeing enough tokens, and it is *undertrained*. The right wall of the valley is where Gopher, GPT-3, and MT-NLG all sat.

The valley bottom is the compute-optimal model for that budget. Now do this for nine budgets, spanning $6 \times 10^{18}$ through intermediate budgets like $1 \times 10^{20}$ up to $3 \times 10^{21}$ FLOPs, and watch what happens to the valley bottoms: they march steadily to the right (bigger optimal model) and down (lower achievable loss) as the budget grows. Connect them and you have traced the compute-optimal frontier directly, with no functional-form assumption at all. The IsoFLOP method gave $a \approx 0.49$ and $b \approx 0.51$ — again, almost exactly one-half each. Extrapolated to Gopher's budget it predicts about **63B parameters on 1.4T tokens.** That 70B-on-1.4T target is, of course, exactly what Chinchilla turned out to be.

The reason I love the IsoFLOP picture is that it makes "undertrained" a *geometric* fact, not a moral judgment. Gopher is not bad; it is just sitting on the right wall of its valley. The same compute, reallocated to the valley bottom, buys a lower loss. There is no cleverness involved — just moving along a budget curve to its minimum.

### Approach 3: fit the parametric loss law to everything

The third method is the one we have already used in the derivation. Take *all* the training runs — every model size, every token count, every intermediate checkpoint — and fit the full parametric law $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$ to the whole cloud of points at once, using a robust regression (DeepMind used a Huber loss to resist outliers). Once you have $\alpha$ and $\beta$, you plug them into the closed form $a = \beta/(\alpha + \beta)$ from Section 2 and read off the exponents. This gave $a \approx 0.46$ and $b \approx 0.54$, extrapolating to roughly **40B parameters on about 1T tokens** at Gopher's budget — slightly smaller than the other two methods, but still firmly in the "moderate model, lots of data" regime.

The three methods give predicted model sizes of 67B, 63B, and ~40B at the same budget. That is a real spread — the parametric fit is the outlier on the low side — but all three agree on the qualitative conclusion that flipped the field, and two of the three land essentially on the 70B/1.4T point that DeepMind actually trained. When three methods with different failure modes converge on "scale equally, around 20:1," you believe it.

### How the IsoFLOP fit actually works in code

The IsoFLOP method is worth implementing once, because doing so demystifies it completely. For each fixed budget you have a handful of (model size, final loss) points forming a U. You fit a parabola in log-parameter space — loss is locally quadratic near its minimum — and the vertex of that parabola is the optimal model size for that budget. Then you regress the vertices against budget on log-log axes to recover the exponent. Here is the whole thing.

```python
import numpy as np

# For one FLOP budget: (params, final_loss) along the IsoFLOP curve.
# Small models overtrain (high loss); large models undertrain (high loss);
# the bottom of the U is compute-optimal for this budget.
params = np.array([1e8, 3e8, 1e9, 3e9, 1e10])      # model sizes tried
losses = np.array([3.10, 2.78, 2.71, 2.80, 3.05])  # final loss at fixed C

# Fit loss as a quadratic in log10(N); the vertex is the optimal N.
x = np.log10(params)
a, b, c = np.polyfit(x, losses, 2)        # loss ~= a x^2 + b x + c
x_star = -b / (2 * a)                      # vertex of the parabola
N_opt = 10**x_star
print(f"compute-optimal model size for this budget: {N_opt:.2e} params")

# Repeat across budgets, collect (C, N_opt), then fit the frontier exponent:
#   log N_opt = a_exp * log C + const   ->   N_opt ~ C^a_exp
C_grid   = np.array([6e18, 6e19, 6e20, 3e21])
Nopt_grid = np.array([4.0e8, 1.3e9, 4.1e9, 1.0e10])  # vertices from each U
a_exp, _ = np.polyfit(np.log10(C_grid), np.log10(Nopt_grid), 1)
print(f"frontier exponent a (N_opt ~ C^a): {a_exp:.3f}")  # ~0.5 for Chinchilla
```

Two things are worth noticing about this procedure. First, it makes *no assumption about the global functional form* of the loss — it only assumes the loss is locally parabolic near each valley bottom, which is true for any smooth minimum. That is why Approach 2 is more robust than Approach 3: it does not require the $E + A/N^{\alpha} + B/D^{\beta}$ form to hold everywhere, only that each U has a well-defined bottom. Second, the quality of the exponent depends entirely on how well you have bracketed each valley. If your model-size grid is too coarse, or if it does not straddle the minimum, the parabola fit is garbage and so is the vertex. A practitioner running their own IsoFLOP sweep should always confirm that for each budget they have at least one model on each side of the minimum — points that are clearly overtrained *and* points that are clearly undertrained — so the vertex is interpolated, not extrapolated.

### Why three methods instead of one

There is a meta-point here that generalizes far beyond scaling laws. Approach 1 (training curves) is sensitive to learning-rate schedule effects, because reading loss off intermediate points of a cosine-decayed run conflates "loss at this token count" with "loss for a run *planned* to stop at this token count" — they differ. Approach 2 (IsoFLOP) sidesteps this by training each model to completion at its budget, but it costs more compute (you throw away the intermediate points). Approach 3 (parametric fit) uses every data point efficiently but bets everything on the functional form being correct globally, which the Besiroglu replication later showed was a shakier bet than it looked. Each method trades one weakness for another. The convergence of all three is what licenses the conclusion — not the precision of any single one. This is the right template for any high-stakes empirical claim: estimate it three ways that fail differently, and trust the agreement, not the individual point estimates.

## 4. The 20-tokens-per-parameter rule

**Senior rule of thumb: a ratio is more portable than a curve. Carry the ratio, derive the curve when you need it.** The single most-cited output of Chinchilla is not the loss law or the exponents — it is the number 20. At the compute-optimal point, the ratio of training tokens to parameters is about 20:1. The figure below plots real models against that line.

![A scatter plot of training tokens against parameters on log axes, with the diagonal D equals twenty N line, showing Chinchilla on the line, Gopher and GPT-3 and MT-NLG below it as undertrained, and LLaMA-3 8B far above it as deliberately overtrained](/imgs/blogs/chinchilla-compute-optimal-scaling-6.png)

The diagonal dashed line is $D = 20N$ — the compute-optimal locus. A model that sits on the line is using its compute budget efficiently in the Chinchilla sense. A model below the line has too many parameters for the number of tokens it saw: it is undertrained, and feeding it more tokens would lower its loss for the same model size. A model above the line has been trained on more tokens than compute-optimality requires: it is overtrained, which wastes *training* FLOPs but, as we will see, can be a deliberate and smart choice when you care about inference cost.

Where does the 20 come from, exactly? It is the $D_{opt}/N_{opt}$ ratio evaluated at the specific compute scale DeepMind studied. Importantly, **the ratio is not a universal constant** — because $N_{opt} \propto C^{a}$ and $D_{opt} \propto C^{a'}$ with $a' > a$ slightly, the ratio $D_{opt}/N_{opt} \propto C^{a' - a}$ creeps upward as the budget grows. With $a' - a \approx 0.55 - 0.45 = 0.10$, the ratio grows slowly: roughly speaking, every two orders of magnitude of compute nudges the compute-optimal ratio up by a small factor. But over the range of budgets relevant to almost everyone, "20 tokens per parameter" is an excellent rule of thumb, and treating it as a constant will not steer you badly wrong. Some practitioners round to "20:1" and some use a slightly higher number for very large budgets; either is fine.

Let us read the famous models off the line:

| Model | Parameters | Training tokens | Tokens/param | Verdict |
|---|---|---|---|---|
| Chinchilla | 70B | 1.4T | ~20 | on the line (compute-optimal) |
| Gopher | 280B | 300B | ~1 | far below (severely undertrained) |
| GPT-3 | 175B | 300B | ~1.7 | below (undertrained) |
| MT-NLG | 530B | 270B | ~0.5 | far below (severely undertrained) |
| LLaMA-1 7B | 7B | 1T | ~143 | above (overtrained) |
| LLaMA-2 70B | 2T tokens | 2T | ~29 | just above the line |
| LLaMA-3 8B | 8B | 15T | ~1875 | far above (deliberately overtrained) |

The top four rows are the 2020–2022 generation, all clustered below the line — the undertrained giants. The bottom three are the LLaMA family, which moved decisively above the line over time. LLaMA-3 8B at roughly 1,875 tokens per parameter is about 94x past Chinchilla-optimal. That is not a mistake; it is the [inference-aware scaling](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) argument made flesh, and we will get to why in Section 7.

### Worked example: how fast does the ratio actually drift?

Let us make the "20 is not a constant" point quantitative, because it matters at the frontier. The compute-optimal ratio is $D_{opt}/N_{opt} \propto C^{a' - a}$. Using the parametric-fit exponents $a' \approx 0.55$ and $a \approx 0.45$, the drift exponent is $a' - a \approx 0.10$. So if you increase your compute budget by a factor of 100 (two orders of magnitude), the optimal ratio grows by $100^{0.10} \approx 1.58$ — roughly a 58% increase. Start at 20:1 at one budget, and 100x more compute moves you to about 32:1. Another 100x and you are near 50:1. The ratio climbs, but slowly, and over the range most practitioners operate in, treating it as a flat 20:1 introduces only a small error.

This slow drift is also why different sources quote slightly different "tokens per parameter" numbers — they are evaluating the ratio at different compute scales, and they are all approximately right. If you are at the absolute frontier with a budget orders of magnitude beyond Chinchilla's, you should nudge your target ratio upward; if you are doing a modest run, 20:1 is fine. The honest framing is: 20:1 is the compute-optimal ratio *at roughly Chinchilla's compute scale*, and it drifts upward gently with budget. Anyone who states it as a universal law has over-simplified, but anyone who uses 20:1 as a planning default is not far wrong.

### Worked example: is your training run undertrained?

Suppose you have a 13B-parameter model and a corpus of 260B tokens, and you are wondering whether to gather more data. The ratio is $260\text{B} / 13\text{B} = 20$. You are exactly on the Chinchilla line — compute-optimal for that pair. If instead you had 13B parameters and only 100B tokens, your ratio is about 7.7:1, well below the line: you are undertrained, and your compute would be better spent on more tokens than on the model you have. Conversely, with a 1B model and 100B tokens you are at 100:1 — overtrained relative to Chinchilla, which is wasteful for training but might be exactly right if you intend to serve that 1B model to a billion requests. The ratio tells you which side of the line you are on; what you *do* about it depends on whether you are optimizing training cost or lifetime cost.

## 5. Chinchilla vs the giants: the result that changed the field

**Senior rule of thumb: a controlled comparison at equal cost is worth a hundred uncontrolled comparisons at different cost.** The reason Chinchilla landed so hard is that DeepMind held training compute fixed. Chinchilla 70B was trained on *the same compute budget as Gopher 280B* — they just allocated it differently. The before-and-after is stark.

![A two-column comparison of Gopher and Chinchilla at the same training compute, showing Gopher at 280B parameters and 300B tokens losing the eval suite, versus Chinchilla at 70B parameters and 1.4T tokens winning broadly with cheaper inference](/imgs/blogs/chinchilla-compute-optimal-scaling-5.png)

Same FLOP budget. Gopher spent it on 280B parameters and 300B tokens — about 1 token per parameter, the far-right wall of its IsoFLOP valley. Chinchilla spent it on 70B parameters and 1.4T tokens — about 20 tokens per parameter, the valley bottom. The model got 4x smaller and the data got roughly 4.7x larger, holding the product (and thus the compute) approximately constant. And the smaller model won.

It did not win narrowly, and it did not win on a cherry-picked metric. Chinchilla outperformed Gopher on the large majority of a broad evaluation suite — language modeling, reading comprehension, common-sense reasoning, MMLU-style knowledge tasks. The same comparison held against GPT-3 175B and MT-NLG 530B, both of which Chinchilla beat at equal or less training compute despite being a fraction of their size. The matrix below lays out all four.

![A matrix comparing Chinchilla against Gopher, GPT-3, and MT-NLG by parameters, training tokens, tokens per parameter, and eval-suite verdict, showing the 70B compute-optimal model beating all three larger undertrained models](/imgs/blogs/chinchilla-compute-optimal-scaling-7.png)

The numbers in that matrix are the entire argument. MT-NLG, the largest model of its day at 530B parameters, was trained on only about 270B tokens — roughly half a token per parameter. By the IsoFLOP picture, it was sitting far up the right wall of a valley whose bottom was a model perhaps a tenth its size. The field had built a 530B-parameter model and left most of its potential on the table by starving it of data.

### Why does the smaller model win, mechanistically?

It is worth resisting the temptation to treat "smaller model, more data, lower loss" as a black box. There is a mechanistic intuition. A model's parameters are its *capacity* — the number of distinct patterns it can store. Tokens are its *experience* — the number of examples it gets to learn those patterns from. An undertrained giant like Gopher has enormous capacity but thin experience: it is like a vast library with most of the shelves empty, because nobody ran enough books past it to fill them. The capacity is physically present in the weights, but it is not *used*, because the training signal to set those weights well never arrived. Conversely, a tiny model on enormous data has rich experience but no shelf space — it has seen everything but cannot retain it, so it saturates and the extra data is wasted.

The compute-optimal point is where capacity and experience are balanced: every parameter has enough data behind it to be set to a useful value, and every token of data has somewhere to go. The IsoFLOP valley bottom is, mechanistically, the point where you stop wasting capacity (left of it, model too small) and stop wasting data (right of it, model too big). Gopher was wasting capacity *and* a generation of practitioners' intuition; the same FLOPs at the valley bottom filled the shelves. This is also why the win is broad rather than narrow: it is not that Chinchilla got better at one task, it is that its weights were *better estimated* across the board, which lifts performance on essentially everything that depends on those weights. A better-estimated model is better at the whole distribution, not at a subset.

### The second-order win: inference

There is a second-order consequence that is easy to miss and that, in hindsight, may be more important than the benchmark wins. A 70B model is not just cheaper to *train* than a 280B model at the same loss — it is dramatically cheaper to *serve*. Inference cost scales with $N$ (about $2N$ FLOPs per token), so a 4x smaller model is roughly 4x cheaper per generated token, for its entire deployed life. Gopher would have cost about 4x more per token to run than Chinchilla, forever, while being *worse*. When you are serving a model to millions of users, the cumulative inference bill dwarfs the one-time training bill, so "smaller model, same quality" is a gift that keeps giving. This observation is the seed of the entire inference-aware line of work, which argues that you should sometimes go *past* 20:1 — deliberately overtraining a small model — precisely to drive down that lifetime serving cost.

## 6. The budgeting recipe you can actually use

**Senior rule of thumb: a result you cannot turn into a number you can act on is trivia.** The beautiful thing about Chinchilla is that it reduces to a three-step recipe. Given a compute budget $C$ in FLOPs, here is how you turn it into a model size and a token count.

![A pipeline showing the compute-optimal budgeting recipe: fix the FLOP budget, size the model with a square root, set tokens to twenty times the model size, sanity-check that six times N times D matches the budget, then train to twenty to one and stop](/imgs/blogs/chinchilla-compute-optimal-scaling-8.png)

Step one: fix your compute budget $C$. This is usually set by your hardware and your schedule — number of GPUs times their throughput times how many days you can run.

Step two: size the model. Because $C = 6ND$ and $D \approx 20N$ at the optimum, substitute to get $C \approx 6N \cdot 20N = 120N^2$, which rearranges to:

$$N_{opt} \approx \sqrt{\frac{C}{120}}$$

Step three: set the token count to $D \approx 20 N_{opt}$, then sanity-check that $6 N_{opt} D \approx C$. If it does, train to that token count and stop. That is the whole recipe.

### Worked example: a one-million-GPU-hour budget

Let us make it concrete. Suppose you have 512 H100 GPUs for 30 days. Each H100 delivers roughly $10^{15}$ FLOP/s of usable BF16 throughput at, say, 40% model FLOPs utilization (a realistic MFU for a well-tuned dense transformer), so about $4 \times 10^{14}$ effective FLOP/s per GPU. Total:

$$C \approx 512 \times 4 \times 10^{14} \,\tfrac{\text{FLOP}}{\text{s}} \times 30 \times 86400 \,\text{s} \approx 5.3 \times 10^{23} \text{ FLOPs}$$

Now apply the recipe:

$$N_{opt} \approx \sqrt{\frac{5.3 \times 10^{23}}{120}} = \sqrt{4.4 \times 10^{21}} \approx 6.6 \times 10^{10} = 66\text{B parameters}$$

$$D \approx 20 \times 66\text{B} \approx 1.3 \text{T tokens}$$

Sanity check: $6 \times 66\text{B} \times 1.3\text{T} \approx 5.1 \times 10^{23}$ FLOPs — close to our budget, good. So with this hardware budget, the Chinchilla-optimal choice is a roughly 66B model on about 1.3T tokens. Notice that this is essentially Chinchilla's own configuration, which is reassuring: the recipe reproduces the paper when you feed it the paper's budget.

### Worked example: the same budget, allocated the old way

For contrast, suppose you had taken the Kaplan-era advice with the same $5.3 \times 10^{23}$ FLOP budget. With $N_{opt} \propto C^{0.73}$, you would have built a much larger model — comfortably north of 200B parameters — and, because $C = 6ND$ pins the product, you would have trained it on far fewer tokens, on the order of a few hundred billion. You would have landed near where Gopher and GPT-3 sat: a giant model on a thin diet of data, sitting on the right wall of its IsoFLOP valley. Same electricity bill, worse model, and a 3x larger inference cost forever. That is the concrete price of getting the exponent wrong.

### Worked example: dollars, not just FLOPs

Engineers think in FLOPs; finance thinks in dollars, and the translation is worth doing because it reframes the decision. At public cloud rates, an H100 runs somewhere around \$2–\$4 per GPU-hour depending on commitment. Take \$3. Our 512-GPU, 30-day run is $512 \times 30 \times 24 = 368{,}640$ GPU-hours, or about **\$1.1M** for the training run. The Kaplan-style 200B+ model on the same compute costs the same \$1.1M to train but produces a worse model — so the \$1.1M is simply spent less efficiently. Worse, the inference math compounds: if you then serve that 200B model instead of the 66B Chinchilla-optimal one, every \$1 of serving cost on the small model becomes about \$3 on the large one. For a product that spends, say, \$500K/month on inference, choosing the undertrained giant turns that into \$1.5M/month — \$12M more per year — to ship a model that scores *lower*. The exponent is not an academic detail; it is a line item.

### Second-order gotcha: the $6ND$ approximation has its own assumptions

Before trusting the recipe's output, know where its central approximation leaks. The $C \approx 6ND$ estimate counts the matrix-multiply FLOPs of a dense transformer and ignores several things: attention's quadratic-in-sequence-length cost, the embedding and unembedding layers, layer norms, activations, and the overhead of any sparsity or mixture-of-experts routing. For models with long context or small parameter counts, the attention and embedding terms are *not* negligible — and, as it happens, the embedding-parameter question is precisely the bookkeeping discrepancy that split Kaplan from Chinchilla, because at small scale the embedding FLOPs and parameters are a large fraction of the total. For a standard dense model at the scales Chinchilla studied, $6ND$ is accurate to within a few percent, which is good enough for budgeting. But if you are training a mixture-of-experts model (where only a fraction of parameters are active per token), or a very-long-context model (where attention dominates), you must use the right FLOP count for *your* architecture, or your $N_{opt}$ will be off. The recipe is exact for the model class it was derived on; verify it applies to yours before you trust the number.

### Second-order gotcha: the recipe assumes you can get the tokens

The recipe has a hidden assumption that bites in practice: it assumes you actually have $20N$ unique, high-quality tokens to feed the model. For a 66B model that is 1.3T tokens — fine, the web has that. But scale up: a Chinchilla-optimal 500B model wants 10T tokens, and a 1T model wants 20T tokens, which starts to strain the supply of high-quality text on the internet. This is the *data wall*, and it is exactly the regime where the [data-constrained scaling](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation) question — how much can you repeat data before the law breaks down — becomes the binding constraint instead of compute. When you cannot get $20N$ fresh tokens, the clean recipe stops applying and you are into a different optimization entirely.

## 7. When to break the 20:1 rule on purpose

**Senior rule of thumb: compute-optimal and cost-optimal are different objectives, and you should know which one you are solving.** Chinchilla minimizes *training* compute. Almost nobody actually wants to minimize training compute — they want to minimize the total cost of getting useful predictions out of a model over its lifetime, which is training plus inference. And the moment you add inference to the objective, the optimum moves.

The reason is the asymmetry we noted earlier: training costs $\approx 6ND$ once, but inference costs $\approx 2N$ per token *every time you serve a request*, for the model's entire deployment. If you are going to serve trillions of tokens of inference, the $2N$-per-token term can dominate the total, and you would gladly spend extra training FLOPs to shrink $N$ — even if that means training a smaller model far past 20:1. A smaller, overtrained model can match a larger Chinchilla-optimal model's quality while being permanently cheaper to run.

This is precisely why LLaMA-3 8B was trained on 15T tokens — about 1,875 tokens per parameter, roughly 94x past Chinchilla-optimal. Meta deliberately blew through the 20:1 rule because an 8B model is cheap to serve at massive scale, and the extra training cost is amortized over the model's enormous deployment. The [inference-aware scaling post](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) works out the full objective — minimize $6ND_{tr} + 2ND_{inf}$ subject to hitting a target loss — and shows that for high-traffic deployments the optimum shifts to a smaller model on far more tokens, sometimes saving a double-digit percentage of total FLOPs and even more in dollars (because inference often runs at much lower hardware utilization than training, which amplifies the per-token cost).

### Worked example: where is the inference crossover?

Here is the back-of-envelope that decides whether to overtrain. Suppose you have a Chinchilla-optimal 70B model and are considering a 35B model overtrained to match its quality. Training the 35B model to parity costs *more* training FLOPs than the compute-optimal point (you are pushing it well past 20:1), but it saves on every inference token. The question is whether the inference savings repay the extra training cost over the model's life.

Training cost scales with the extra FLOPs you spend overtraining; inference savings scale with $2 \times (70\text{B} - 35\text{B}) = 2 \times 35\text{B}$ FLOPs saved per generated token. Roughly, the crossover is at the inference volume where cumulative savings equal the extra training spend. Plugging in realistic numbers — and noting that inference often runs at perhaps half the hardware utilization of training, which amplifies the per-token *dollar* savings beyond the raw FLOP savings — the crossover for a 70B-class target typically lands somewhere in the low trillions of inference tokens. Below that volume, just train the compute-optimal 70B; above it, the overtrained 35B is cheaper over its lifetime. For a high-traffic product serving many trillions of tokens per year, you are comfortably past the crossover, which is exactly why production models drift above the line. The [inference-aware post](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) gives a concrete worked case: a 30B-Chinchilla-quality target with $10^{13}$ inference tokens optimizes to about a 13.6B model on roughly 2.8x Chinchilla data, cutting total FLOPs by about 28%.

So the rule is not "always train to 20:1." The rule is: **train to 20:1 if you are minimizing training cost; train past 20:1 if you are minimizing lifetime cost and you expect heavy inference.** Chinchilla answered the first question definitively. The second question is what the rest of this series is about.

| Your situation | Where to sit relative to 20:1 |
|---|---|
| Research model, trained once, evaluated, rarely served | On the line (~20:1) — minimize training cost |
| Production model with moderate traffic | Slightly above the line (30–100:1) — modest overtraining |
| Production model, very high traffic, latency-sensitive | Far above the line (hundreds to thousands :1) — overtrain aggressively |
| You cannot acquire $20N$ unique tokens | Below the line, but repeat data carefully (data-constrained regime) |
| You will quantize the model heavily at inference | Be cautious about extreme overtraining (precision interacts with D/N) |

## 8. How solid are the numbers? The replication caveat

**Senior rule of thumb: a single published coefficient set is a point estimate with real uncertainty, not a law of nature. Treat it accordingly.** The Chinchilla result is robust at the level of its qualitative conclusion. It is shakier at the level of its exact constants, and an honest practitioner should know the difference.

In 2024, Besiroglu et al. published "Chinchilla Scaling: A replication attempt" (arXiv:2404.10102). They reconstructed the data from the original Approach-3 parametric fit (the third method, the full regression) and tried to reproduce the published coefficients. They found two problems. First, the published fit *reproduces poorly* — refitting the same functional form to the same reconstructed data does not land on the paper's reported $E$, $A$, $B$, $\alpha$, $\beta$. Second, the original paper's reported confidence intervals on those coefficients were *implausibly tight* — far narrower than the data could actually support, which is a statistical red flag suggesting the uncertainty was understated.

Their corrected fit gave roughly $E \approx 1.82$, $A \approx 482$, $B \approx 2085$, $\alpha \approx 0.348$, $\beta \approx 0.366$. Look at what changed and what did not. The exponents $\alpha$ and $\beta$ moved a little and, crucially, came out *even closer to each other* (0.348 vs 0.366) than in the original fit — which actually *strengthens* the "scale equally" conclusion, because near-equal exponents are exactly what drives the 50/50 split. The compute-optimal exponents from the corrected fit still land essentially at one-half, and the ~20:1 ratio still holds. What did *not* survive is the precise value of the constants, especially $B$, which moved by a factor of five.

The lesson is not "Chinchilla was wrong." The lesson is the one that runs through this whole series: **the structural result is solid (scale model and data about equally, roughly 20 tokens per parameter), but the exact coefficients are soft, and you should never hard-code a published constant as if it were $\pi$.** If you are doing your own scaling study, re-fit on your own data, your own architecture, your own tokenizer — and report honest, wide error bars. The [reconciliation post](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation) goes deeper into exactly how fragile these fits can be and how a parameter-counting choice split the entire field for two years.

### What the replication does and does not threaten

It helps to be precise about the blast radius of the Besiroglu finding, because it is easy to over- or under-react. What it threatens: any use of the *exact* published constants for absolute loss prediction, and any claim that DeepMind's reported confidence intervals were trustworthy. If you plugged $E = 1.69$, $A = 406.4$, $B = 410.7$ into a spreadsheet to predict a specific loss value, that prediction was built on sand — not because the numbers are absurd, but because they are point estimates with much wider true uncertainty than advertised, and they were fit on a distribution that is not yours.

What it does *not* threaten: the qualitative conclusion. Three methods, only one of which (Approach 3) is the parametric fit that failed to reproduce, all pointed at near-equal exponents. The IsoFLOP method (Approach 2) makes essentially no global functional-form assumption and still landed at ~0.5. The Gopher-versus-Chinchilla head-to-head is an empirical fact independent of any fit — two real models on the same budget, one of which won. And the corrected exponents came out *closer to equal*, reinforcing the central claim. So the structure survives the replication comfortably; only the false precision of the constants does not. The healthy posture is to hold the shape with confidence and the constants with suspicion — which is, frankly, the right posture toward every empirical scaling result, Chinchilla included.

## Case studies from the trenches

Theory is cheap. Here are the concrete situations where the Chinchilla framework either saved a project or where ignoring it cost real money. Each is the kind of incident a team actually runs into.

### 1. The 530B model that a 70B model embarrassed

MT-NLG (Megatron-Turing NLG) was, at 530B parameters, the largest dense language model of its moment — a joint Microsoft-NVIDIA effort that represented an enormous engineering achievement in distributed training. It was trained on about 270B tokens, roughly half a token per parameter. By the IsoFLOP picture, it was sitting near the top of the right wall of its valley: an immense model starved of data. When Chinchilla 70B beat it across the eval suite at a fraction of the size and compute, the lesson was not that big models are bad — it was that the field had a systematic bias toward parameters over tokens, and that bias had produced a half-trillion-parameter model leaving most of its potential unrealized. The fix was not a better architecture; it was a better *allocation*. The same compute, spent at the valley bottom, would have produced a far stronger model.

### 2. Gopher and the four-fold reallocation

Gopher (280B, 300B tokens) and Chinchilla (70B, 1.4T tokens) came from the same lab, DeepMind, trained on the same compute budget. That is what makes the pair the cleanest controlled experiment in scaling-law history. The team did not change the hardware, the codebase, or the budget — they changed the allocation, guided by their own fitted frontier, and shrank the model 4x while growing the data 4.7x. The result quantified the cost of the prior consensus exactly: a 4x oversized model had been *worse* than its right-sized sibling. Internally, this is the experiment that turns "we think we should train longer" into "we have proof, here are two models on the same budget." If you are trying to convince a skeptical leadership team to spend on data instead of parameters, this is the slide.

### 3. The undertrained internal model nobody noticed

A common pattern inside companies: a team trains a 7B model on whatever curated corpus they happen to have — say 150B tokens — because that is the data that was ready. The ratio is about 21:1, which looks Chinchilla-optimal by coincidence, so nobody questions it. But the *reason* it is 21:1 is that the data ran out, not that someone optimized for it. When the team later acquires another 500B tokens and retrains the same 7B model to 650B tokens (~93:1), the loss drops noticeably and downstream task accuracy climbs — because the original run was compute-limited by available data, not by model capacity. The diagnostic move is to always compute the ratio explicitly and ask *why* it is what it is. A ratio that lands near 20:1 by accident is not the same as one chosen deliberately.

### 4. The "we'll just make it bigger" budget meeting

A team has a fixed quarterly GPU budget and wants the best possible model. The instinct, inherited from the GPT-3 era, is to spend it all on the largest model that fits in memory. Running the Chinchilla recipe instead — $N_{opt} \approx \sqrt{C/120}$ — typically returns a model two to four times smaller than the instinct, trained on far more tokens. The pushback in the meeting is always the same: "but the bigger model will be smarter." The IsoFLOP curve is the answer: at *this* budget, the bigger model is on the right wall of the valley, and the smaller one at the bottom will have lower loss. The bigger model is only smarter if you also give it the tokens to match — which your fixed budget does not allow. The recipe converts a vibe ("bigger is better") into a falsifiable prediction you can check with a small sweep.

### 5. The data wall at the frontier

A frontier lab plans a compute-optimal run and discovers, on applying the recipe, that the Chinchilla-optimal token count exceeds the high-quality data they can actually source. For a 500B-parameter compute-optimal model you want 10T tokens; after dedup and quality filtering, the available pool is smaller. Now the binding constraint is no longer compute — it is data. The team has three options: (a) repeat data for a few epochs (which is roughly free up to a point, but degrades past it), (b) lower the model size to match the available tokens (sacrificing the compute-optimal point to stay on a real data budget), or (c) invest in data acquisition and filtering. This is the regime where Chinchilla's clean recipe hands off to the data-constrained scaling literature, and where "just follow 20:1" stops being actionable advice.

### 6. The over-trained small model that won on cost

A product team needs to serve an 8B-class model to tens of millions of users with tight latency budgets. The Chinchilla-optimal token count for 8B is about 160B tokens. They train on 15T tokens instead — nearly 100x the Chinchilla ratio. By compute-optimal accounting this is wildly wasteful: they spent far more training FLOPs than needed for that model size. But by *lifetime-cost* accounting it is the obviously correct call: the model is small enough to serve cheaply at enormous scale, the extra training cost is a one-time payment amortized over trillions of inference tokens, and the over-training pushed the small model's quality up to where it competes with models several times larger. This is the LLaMA-3 8B story, and it is the clearest real-world demonstration that "compute-optimal" and "cost-optimal" are genuinely different targets.

### 7. The coefficient that did not transfer

A team reads the Chinchilla paper, hard-codes $E = 1.69$, $A = 406.4$, $\alpha = 0.34$, $B = 410.7$, $\beta = 0.28$ into their own planning spreadsheet, and uses it to predict the loss of a planned run on their proprietary code-heavy corpus with a custom tokenizer. The prediction is off — sometimes substantially — because those constants were fit on DeepMind's data distribution, their tokenizer, and their architecture, none of which match. The exponents transfer reasonably well (they reflect something more universal about how networks scale), but the *constants* are distribution-specific. The fix is to run a small IsoFLOP sweep on your own setup, fit your own $E$, $A$, $B$, and only borrow the qualitative structure ($\alpha \approx \beta$, scale equally, ~20:1) from the paper. This is the operational form of the Besiroglu caveat: borrow the shape, fit the constants.

### 8. The intermediate-checkpoint mirage

A team fits a scaling law using only fully-converged final-checkpoint losses and gets clean exponents. A second team fits using intermediate checkpoints (loss at many points along each training curve, as Approach 1 does) and gets slightly different exponents, because early-training dynamics — warmup, learning-rate schedule, the cosine decay tail — distort the loss-versus-tokens relationship away from the asymptotic power law. This is not a contradiction; it is a reminder that *what you measure* shapes *what you fit*. The cosine learning-rate schedule in particular means a model's loss at 50% of its planned tokens is not the same as the final loss of a model planned to stop at that point. Mismatching these is one of the subtle bookkeeping errors that contributed to the Kaplan-Chinchilla discrepancy, and it is worth being deliberate about whether your fit uses final or intermediate losses.

### 9. The MMLU jump that justified the reallocation

When a team reallocates from a Kaplan-style big-undertrained model to a Chinchilla-style right-sized one at fixed compute, the most convincing internal evidence is rarely the held-out perplexity — it is the downstream benchmark jump. Lower pretraining loss translates into measurable gains on knowledge and reasoning evals, and those are the numbers leadership cares about. The Chinchilla paper's broad eval-suite win over Gopher was not an abstract loss improvement; it showed up as concrete accuracy gains on the tasks the field uses to rank models. The practical move when you run your own reallocation is to track both the loss (to confirm you moved along the frontier as predicted) and the downstream evals (to confirm the loss gain converted into capability), because the two together make an argument that neither makes alone.

### 10. The memory-bound model that could not be right-sized

A team computes their Chinchilla-optimal model size at, say, 66B parameters, but their inference hardware can only hold a 34B model in the memory budget they have allocated per replica. The instinct is to give up on compute-optimality and just train the 34B model on whatever tokens the budget allows. The better move is to recognize that they are no longer in a pure training-compute optimization at all — the binding constraint is *serving memory*, which fixes $N$ from the outside. Once $N$ is fixed by deployment, the only free variable is $D$, and the right thing to do is train that 34B model on as many tokens as the training budget allows, which will land it far above 20:1. The lesson: real constraints frequently come from outside the loss law (memory, latency SLAs, regulatory limits on data), and when they do, you optimize the remaining free variable rather than pretending the constraint does not exist. Chinchilla tells you the *unconstrained* optimum; your job is to find the optimum subject to *your* constraints.

### 11. The multilingual corpus with a different exponent

A team building a model for a low-resource-language mix assumes the Chinchilla exponents transfer and plans a 20:1 run. Their own small IsoFLOP sweep reveals that on their data distribution, the data-penalty exponent $\beta$ is meaningfully different from English-web text — repetition and morphology change how quickly loss falls with tokens. The compute-optimal ratio for their distribution turns out to be different from 20:1. This is not a failure of Chinchilla; it is a reminder that the *exponents themselves* are properties of the data distribution and the architecture, not universal constants. The qualitative structure (additive penalties, a single valley per budget) holds, but the numbers must be re-measured. Teams that skip the small sweep and import 20:1 blindly leave performance on the table in exactly the cases where it matters most — the non-standard distributions where intuition from English benchmarks does not apply.

### 12. The premature-stop that looked compute-optimal

A team is training a 30B model toward a planned 600B tokens (20:1) but, under schedule pressure, stops at 400B tokens (~13:1) and ships. The loss curve still looked like it was descending, so it feels like they left value on the table — and they did, but the subtlety is *how much*. Because the data-penalty term $B/D^{\beta}$ has a small exponent ($\beta \approx 0.3$), the marginal loss reduction per token is shrinking as you go: the last third of the tokens buys less loss reduction than the first third. Stopping at 13:1 instead of 20:1 sacrifices real but modest quality. The deeper point is that the power-law shape means there is no sharp cliff at exactly 20:1 — the penalty for being somewhat below the line is gentle, and the penalty for being somewhat above it (overtraining) is also gentle. The 20:1 rule is the bottom of a *broad, shallow* valley, not a knife-edge, which is exactly why slightly-off allocations like LLaMA-2 70B at 29:1 cost almost nothing in quality while buying inference savings. Knowing the valley is shallow is what lets you trade along it confidently.

## What this means in practice

If you are planning a pretraining run, here is the distilled, opinionated guidance.

**Always compute the ratio first.** Before anything else, write down your planned parameters $N$ and tokens $D$ and compute $D/N$. If it is far below 20, you are about to train an undertrained model — stop and reconsider whether you should shrink the model or get more data. If it is far above 20, make sure that is a deliberate inference-cost decision and not an accident.

**Use the recipe as a default, then adjust.** Start from $N_{opt} \approx \sqrt{C/120}$ and $D \approx 20N$. That is your compute-optimal baseline. Then move *up* the ratio (smaller model, more tokens) in proportion to how much inference you expect, because lifetime cost, not training cost, is usually what you actually pay.

**Run a small IsoFLOP sweep on your own data.** Do not trust borrowed constants. Train a handful of model sizes at two or three small FLOP budgets, plot the U-shaped curves, and read off your own valley bottoms. This is cheap relative to the main run and it catches distribution-specific surprises — a code-heavy or multilingual corpus can have meaningfully different exponents.

**Distinguish compute-optimal from cost-optimal explicitly.** Write down which objective you are solving. "Minimize training FLOPs" gives you Chinchilla 20:1. "Minimize training plus lifetime inference" gives you something above 20:1, and the heavier your traffic the higher you go. Confusing the two is the most common expensive mistake in this whole area.

**Treat every published coefficient as a point estimate.** The exponents are reasonably universal; the constants are not. Borrow the structure, fit the numbers, and report honest uncertainty. The Besiroglu replication is a permanent reminder that even a landmark paper's exact coefficients can fail to reproduce.

**Mind the data wall.** At the frontier, $20N$ unique high-quality tokens may simply not exist. When data, not compute, is the binding constraint, the clean recipe stops applying and you are into data repetition, filtering, and acquisition trade-offs — a genuinely different optimization.

**Validate the frontier as you climb it.** A scaling law is a forecast, and forecasts should be checked. When you run the big training job, log loss against tokens and confirm it tracks the curve your small sweep predicted. If the real run diverges from the forecast — loss falling faster or slower than expected — that is a signal worth investigating before you have spent the whole budget: it can reveal a data-quality problem, a learning-rate mistuning, or that your small-scale fit did not extrapolate. The entire value proposition of scaling laws is that they let you predict the expensive run from cheap ones; that promise is only real if you actually compare the prediction to the outcome and tighten the model when they disagree. Treat the first 10% of a large run as a live test of your scaling fit, not just as the opening of the run.

The deepest takeaway is the one we opened with: loss is not stored in the parameters. It is split, in roughly equal halves, between the parameters and the tokens. Chinchilla's lasting contribution was to measure that split carefully enough to overturn two years of consensus, hand the field a recipe simple enough to use in a budget meeting, and — through its replication saga — teach a lasting lesson about how much trust to place in any single fitted constant. The constants will keep getting refined; the shape — additive penalties, a single valley per budget, scale model and data together — is the durable part. Carry the shape into every planning meeting and re-fit the constants on your own data, and you will not repeat the half-trillion-parameter mistake.

## Further reading

- Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models" (the Chinchilla paper): https://arxiv.org/abs/2203.15556
- Kaplan et al. 2020, "Scaling Laws for Neural Language Models" (the prior consensus): https://arxiv.org/abs/2001.08361
- Besiroglu et al. 2024, "Chinchilla Scaling: A replication attempt": https://arxiv.org/abs/2404.10102
- Sardana & Frankle et al. 2024, "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws": https://arxiv.org/abs/2401.00448
- Sibling posts on this blog: [Kaplan 2020: the first scaling laws for language models](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models), [Kaplan vs Chinchilla: how a parameter-counting bug split the field](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation), and [Beyond Chinchilla: scaling laws that account for inference cost](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws).

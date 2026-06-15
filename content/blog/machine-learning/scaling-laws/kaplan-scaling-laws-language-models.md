---
title: "Kaplan 2020: the first scaling laws for language models"
date: "2026-06-15"
description: "Understand Kaplan's loss-vs-N, loss-vs-D, and loss-vs-compute power laws, the combined envelope, critical batch size, and the compute-optimal allocation that built GPT-3 — and why Chinchilla later corrected it."
tags: ["scaling-laws", "kaplan", "language-models", "compute-optimal", "power-law", "gpt-3", "loss-curves", "pretraining", "transformer", "deep-learning", "chinchilla"]
category: "machine-learning"
subcategory: "Scaling Laws"
author: "Hiep Tran"
featured: true
readTime: 53
---

> [!important]
> **The five things to remember about Kaplan 2020:**
> - Language-model loss is a smooth **power law** in three resources — parameters $N$, training tokens $D$, and compute $C$ — and a power law is a *straight line on log-log axes*, which is why you can extrapolate it from cheap small runs.
> - The three exponents are shallow: $\alpha_N \approx 0.076$ (parameters), $\alpha_D \approx 0.095$ (data), $\alpha_C \approx 0.050$ (compute). Shallow means a 10x increase in scale buys only ~16% lower loss — progress is bought in powers of ten.
> - **Architecture barely matters.** At fixed $N$, depth-versus-width, feed-forward ratio, and head count move the loss by a couple of percent; scale moves it by orders of magnitude.
> - The **critical batch size** $B_{\text{crit}}(L) \propto L^{-1/\alpha_B}$ with $\alpha_B \approx 0.21$ grows as the loss falls, so the efficient batch is small early in training and large late.
> - The headline allocation $N_{\text{opt}} \propto C^{0.73}$ says *spend most extra compute on a bigger model, little on more data*. This built GPT-3 (175B params on ~300B tokens) — and was **later corrected by Chinchilla** to roughly equal scaling. The one number to carry forward: $\alpha_N \approx 0.076$, the slope of the most important line in modern ML.

Here is the bet that OpenAI made in January 2020, and it sounds reckless until you see the data behind it: *you can predict, to within a few percent, how good a language model will be — before you train it.* Not roughly. Not "bigger is probably better." A specific number. Give me the parameter count, the token count, and the compute budget, and I will tell you the cross-entropy loss the model will reach, even if that model is a thousand times larger than anything you have run so far.

That is what Kaplan et al.'s "Scaling Laws for Neural Language Models" (arXiv:2001.08361) claimed and, for the regime they measured, demonstrated. The loss of a transformer language model is not a chaotic function of a hundred hyperparameters. It is a smooth power law in three quantities — the number of parameters $N$, the number of training tokens $D$, and the compute $C$ — and almost nothing else matters very much. Depth versus width, the feed-forward ratio, the number of attention heads: those move the loss by a percent or two. Scale moves it by orders of magnitude.

![A layered stack with test loss on top, three blue strong-lever layers for parameters, tokens, and compute, and a thin amber architecture layer at the bottom](/imgs/blogs/kaplan-scaling-laws-language-models-1.png)

The diagram above is the mental model for the entire paper. At the top sits the outcome, the test loss, predictable over roughly seven orders of magnitude. Below it are the three strong levers in blue — parameters $N$, tokens $D$, and compute $C$ — each setting the loss through its own power law with the exponents we will derive. At the very bottom, drawn as a thin amber layer to signal how little it matters, is architectural shape: depth, width, head count. It is real but weak — it shifts the offset within a narrow band and does not change the exponent of the law. The rest of this post is a tour of that picture. We will derive each of the three power laws, show why a power law is a straight line on log-log axes (and why that is the whole game for extrapolation), work through the combined envelope, look at the critical batch size that decides how much you can parallelize, and then arrive at the headline that defined an era: the compute-optimal allocation $N_{\text{opt}} \propto C^{0.73}$, which says spend most of your budget on a bigger model and relatively little on more data. That advice built GPT-3. It was also, as we will flag repeatedly, *wrong* in a way that took two more years to diagnose — and we will set up exactly that tension without resolving it here.

If you read only one number from this post, make it the exponent of the parameter law, $\alpha_N \approx 0.076$. It is the slope of the most important straight line in modern machine learning, and it is shockingly shallow. We will spend a lot of time on what "shallow" buys you and what it costs.

## Why predictable loss is different from "bigger is better"

> Everyone in 2019 believed bigger models were better. Almost nobody could tell you *how much* better, or *how to spend a fixed budget*. Kaplan turned a vibe into an equation.

The assumption that scaling laws overturn is subtle. It is not that scale helps — that was folklore by 2019. The assumption is that the relationship between resources and quality is *messy*, that you discover it by training the model and measuring, and that each new scale is a fresh experiment with an unknown answer. Under that worldview, the only way to find out whether a 10-billion-parameter model is worth the money is to train it.

Kaplan's contribution is to replace "train it and find out" with "fit a curve on cheap runs and forecast." Here is the practical difference, stated as a table because the gap between the two worldviews is exactly the kind of assumption-versus-reality mismatch that gets budgets wrong.

| Question | Pre-scaling-law instinct | Kaplan's answer |
|---|---|---|
| Is a bigger model worth it? | Train it and measure | Read it off the $L(N)$ line |
| How much data do I need? | Train until validation stops improving | Pick $D$ from the combined law for your $N$ |
| Where should an extra \$1M of compute go? | Mostly model size, by feel | $N_{\text{opt}} \propto C^{0.73}$ — quantified |
| Does my new architecture help? | Run the eval suite and hope | Compare offsets; the exponent rarely moves |
| Will training converge if I stop early? | Risky, avoid it | Expected — bigger models are sample-efficient |

The right column is what makes this a *law* rather than an observation. A law lets you plan. It lets a research lead stand in front of a budget committee and say "this \$5M run will reach a validation loss of 2.1 nats per token, which corresponds to roughly this benchmark performance, and here is the curve we fit on \$50k of small runs to prove it." That sentence was not possible to say with a straight face before 2020.

The unit matters, so fix it now. Throughout, loss $L$ is the autoregressive cross-entropy in **nats per token** — the average negative log-likelihood the model assigns to the next token, measured in natural-log units. Lower is better. A perfect model that knew the true distribution would still have nonzero loss equal to the entropy of language itself; Kaplan's single-variable laws are fit in a regime above that floor, which is why they look like clean power laws rather than power-laws-plus-a-constant. (The constant — the irreducible entropy floor — is exactly what the later Chinchilla form makes explicit, and it is one of the threads that ties this post to its sequel.)

### What came before Kaplan

Kaplan did not invent the idea that error scales predictably; it built on a lineage worth knowing, because it tells you which parts were new. The earlier predictability post in this series covers this in depth, but the short version: in 2017, Hestness et al. ("Deep Learning Scaling is Predictable, Empirically") showed that generalization error falls as a power law in training-set size across machine translation, language modeling, speech, and vision — $\varepsilon(m) \approx \alpha \cdot m^{\beta_g}$, a straight line on log-log axes, with the exponent depending on the problem. They also identified the three-region structure of a learning curve: a small-data region where the model is near chance, a power-law middle region (the predictable part), and an irreducible-error plateau at the entropy floor. Crucially, they observed that architecture changes shift the *offset* of the curve but rarely the *exponent* — the seed of Kaplan's "architecture barely matters" finding. In 2019, Rosenfeld et al. extended this to a joint functional form in both data and model size with a smooth transition out of the random-guess regime, and showed you could fit on small scales and extrapolate to large ones, saving compute.

So what did Kaplan add? Three things. First, it pinned the laws specifically to *transformer language models* and measured the exponents precisely for that family. Second, it tied parameters, data, *and* compute together into a single coherent framework with the $C \approx 6ND$ bridge, and derived a *compute-optimal allocation* — the prescriptive step that turns description into a budgeting tool. Third, it pushed the measured range to roughly seven orders of magnitude, far enough that the straight-line claim became hard to dismiss as a small-scale coincidence. Hestness said "error is predictable"; Kaplan said "here is the equation, here are the constants, and here is how to spend your money." The prescriptive third step is exactly the one that later needed correction — description survived, prescription got revised — which is a tidy summary of the whole saga.

One more piece of bookkeeping that will matter enormously later: in Kaplan's paper, $N$ is the count of **non-embedding** parameters. The token and position embeddings, and the final unembedding projection, are excluded. At GPT-3 scale that exclusion is a rounding error. At the small end of Kaplan's sweep — models down to a few hundred hidden units — the embedding matrices are a *large* fraction of the total, and that single bookkeeping choice is, we now know, the seed of the discrepancy with Chinchilla. We are flagging it three times on purpose; the reconciliation post pulls on exactly this thread.

## 1. Power laws and the log-log straight line

A power law is the simplest non-trivial relationship between two positive quantities:

$$y = a \cdot x^{-c}$$

with $a > 0$ and exponent $c > 0$. Take the logarithm of both sides:

$$\log y = \log a - c \log x.$$

That is a straight line. Plot $\log y$ against $\log x$ and you get a line of slope $-c$ and intercept $\log a$. This is the single most important fact in the whole field, and it is worth dwelling on why it is so powerful for forecasting.

![A hand-drawn log-log plot of test loss against parameter count showing a straight descending line with small fitted points and an extrapolation to the frontier](/imgs/blogs/kaplan-scaling-laws-language-models-2.png)

The figure above is the actual shape of $L(N)$ on log-log axes — a straight line descending left to right, fit from a handful of small, cheap runs (the blue squares on the upper left) and extrapolated to a frontier model (the green square on the lower right). The slope of that line *is* the exponent. For the parameter law it is $-\alpha_N \approx -0.076$, which is very shallow: the line drops slowly. Read it as a sentence: every time you multiply the parameter count by 10, the loss falls by a constant *multiplicative* factor, namely $10^{-0.076} \approx 0.84$. Ten times the model buys you a 16% reduction in loss. Another factor of ten buys another 16%, compounding.

That compounding-by-constant-factor behavior is what distinguishes a power law from the two relationships people reach for by default. Consider three ways the loss might fall as $N$ grows:

- **Linear**, $L = b - mN$. Doubling $N$ subtracts a fixed amount. This cannot be right because it eventually predicts negative loss, which is impossible for cross-entropy.
- **Exponential**, $L = b \cdot e^{-kN}$. Doubling $N$ multiplies the *gap above zero* by a constant — but the relevant axis is linear in $N$, so on log-log axes an exponential curves sharply. Real loss curves do not look like this; they would predict that a modest model already reaches the floor.
- **Power law**, $L = a N^{-c}$. Doubling $N$ multiplies $L$ by $2^{-c}$, a constant factor, *regardless of where you are on the curve*. On log-log axes it is a straight line forever. This is what the data actually do.

The economic consequence is the entire reason anyone cares. Because the line is straight, two cheap points determine it, and a straight line extrapolates trivially. You do not need to train the expensive model to know roughly where it lands. You fit on models you can afford and you read off the prediction for models you cannot yet afford. The risk in extrapolation is whether the line stays straight — whether some new regime bends it — and the empirical surprise of Kaplan's work is how astonishingly straight it stays, across roughly seven orders of magnitude of model size and compute. That straightness is the load-bearing assumption of the whole research program, and the figure above is its picture: the line you fit on the cheap runs is the line you ride to the frontier.

A subtlety worth internalizing: the shallowness of $\alpha_N$ cuts both ways. Shallow means *robust* — you have to be off by a large factor in $N$ before the prediction is meaningfully wrong, so the law is forgiving. But shallow also means *expensive* — to halve the loss you need an enormous increase in scale, because $2 = 10^{\,0.30}$ and $0.30 / 0.076 \approx 4$, so you need roughly **four orders of magnitude** more parameters to cut the loss in half. That single arithmetic fact is why the field is in a permanent arms race for compute: the exponent is small, so progress is bought in powers of ten.

### Why nats per token, and what the floor means

A short aside on units, because it changes how you read the numbers. Cross-entropy in nats relates directly to perplexity: $\text{perplexity} = e^{L}$. A loss of $2.0$ nats is a perplexity of about $7.4$ — the model is, on average, as uncertain as if it were choosing uniformly among ~7.4 equally likely next tokens. Dropping the loss from $2.0$ to $1.9$ is a perplexity improvement from $7.4$ to $6.7$, which sounds small but, near the floor, corresponds to real qualitative gains in capability. The power-law exponents are measured in nats precisely because nats make the loss additive and the law multiplicative, which is the cleanest possible setting for a straight log-log line.

### Reading a slope off the page

There is a manual trick that every practitioner should be able to do, because it is the fastest sanity check on a fitted law. Given two points on a log-log plot, $(N_1, L_1)$ and $(N_2, L_2)$, the slope is just the ratio of the log differences:

$$-\alpha_N = \frac{\log L_2 - \log L_1}{\log N_2 - \log N_1}.$$

So if you train a 100M model to loss 3.0 and a 1B model to loss 2.5, the slope is $(\log 2.5 - \log 3.0)/(\log 10^9 - \log 10^8) = (\ln 2.5 - \ln 3.0)/(\ln 10) = (0.916 - 1.099)/2.303 = -0.079$. That is within a hair of Kaplan's $0.076$, computed from two cheap runs on the back of an envelope. The reason this works is the straightness of the line: two points determine it, and any third point is a prediction. The moment a third point falls off the line you drew through the first two, you have learned something — either a regime change, a bug, or a bookkeeping inconsistency. The whole discipline of scaling-law work is "draw the line, then look hard at the points that miss it."

### Why a power law and not something else

It is fair to ask *why* loss should obey a power law at all, rather than some other curve. The honest answer is that there is no single airtight first-principles derivation — the power law is, first and foremost, an empirical regularity that holds across an embarrassing range of scales. But there are suggestive arguments. One family of explanations comes from the geometry of data: if the data lives near a low-dimensional manifold of intrinsic dimension $d$, then the loss of a model that effectively places $N$ "resolution elements" over that manifold falls roughly as $N^{-\alpha}$ with $\alpha$ tied to $1/d$ — more data dimensions mean a shallower exponent, which matches the observation that harder, higher-dimensional problems have shallower slopes. Another family comes from the spectrum of the data covariance: if the eigenvalues of the feature covariance themselves follow a power law (a common empirical fact for natural data), the generalization error inherits a power-law tail. The theory of *why* power laws arise is rich enough to be its own topic, and the foundations post in this series treats it; for Kaplan, the relevant fact is simply that the law holds with remarkable fidelity, and that the exponent encodes problem difficulty. The shallower the slope, the harder the problem is to improve on by scale alone — and language, at $\alpha_N \approx 0.076$, is a *hard* problem in this sense.

## 2. The three single-variable power laws

Kaplan fits three power laws, one for each resource, in the regime where that resource is the binding constraint and the others are effectively unlimited. Each is a clean straight line on log-log axes, and each has its own measured exponent and scale constant.

![A matrix listing the three single-variable laws with their formulas, exponents, and scale constants](/imgs/blogs/kaplan-scaling-laws-language-models-3.png)

The matrix above is the heart of the paper in three rows. Let us take them one at a time.

**Loss versus parameters**, holding data and steps unconstrained:

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \qquad \alpha_N \approx 0.076, \quad N_c \approx 8.8 \times 10^{13}.$$

Here $N$ is non-embedding parameters and $N_c$ is a scale constant — the parameter count at which this naive form would predict a loss of one nat per token (it is far beyond any model ever trained, which is fine; it is just the line's intercept expressed as a scale). The exponent $\alpha_N \approx 0.076$ is the shallow slope we discussed.

**Loss versus data**, holding the model large and training to convergence on $D$ tokens (with early stopping to avoid overfitting):

$$L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \qquad \alpha_D \approx 0.095, \quad D_c \approx 5.4 \times 10^{13}.$$

The data exponent $\alpha_D \approx 0.095$ is slightly steeper than the parameter exponent. That difference — $0.095$ versus $0.076$ — looks tiny but it is doing real work in the allocation argument later, because the optimal split between $N$ and $D$ is governed by the *ratio* of these exponents.

**Loss versus compute**, where $C_{\min}$ is the minimum compute to reach a given loss using the most efficient (compute-optimal) model size at each budget:

$$L(C_{\min}) = \left(\frac{C_c}{C_{\min}}\right)^{\alpha_C}, \qquad \alpha_C \approx 0.050, \quad C_c \approx 3.1 \times 10^{8} \ \text{PF-days}.$$

A PF-day is one petaflop sustained for one day, about $8.64 \times 10^{19}$ floating-point operations — a natural unit when budgets are measured in cluster-days. The compute exponent $\alpha_C \approx 0.050$ is the shallowest of the three. That is expected: $C$ is downstream of both $N$ and $D$, so its law inherits a kind of average behavior.

The relationship $C \approx 6ND$ is the bridge between the worlds. A forward-and-backward pass through a dense transformer costs about six floating-point operations per parameter per token — roughly two for the forward multiply-accumulate and four for the backward pass and weight update. So a training run that sees $D$ tokens with an $N$-parameter model costs about $6ND$ FLOPs total. (Inference is cheaper, about $2N$ per token, because there is no backward pass; that asymmetry is the seed of the inference-aware scaling work later in this series.) The factor of six is approximate and architecture-dependent, but it is accurate enough that the entire field budgets in $6ND$, and we will use it freely.

### A worked numeric example with the parameter law

Let us make $L(N)$ concrete. Suppose you train a 125M-parameter model (non-embedding, roughly GPT-3-Small scale) and observe a loss of about 3.0 nats per token, and you want to know what a 1.3B-parameter model should reach, all else equal. The ratio of the predicted losses is governed purely by the exponent:

$$\frac{L(N_2)}{L(N_1)} = \left(\frac{N_1}{N_2}\right)^{\alpha_N} = \left(\frac{1.25 \times 10^8}{1.3 \times 10^9}\right)^{0.076}.$$

The size ratio is about $0.096$ (roughly a 10.4x increase). Raising $0.096$ to the $0.076$ power: $0.096^{0.076} = e^{0.076 \ln 0.096} = e^{0.076 \times (-2.34)} = e^{-0.178} \approx 0.837$. So the bigger model should reach about $3.0 \times 0.837 \approx 2.51$ nats. A 10x bigger model, a 16% loss reduction — exactly the compounding factor from before. You did not have to train the 1.3B model to make that forecast; you read it off the line.

Now extrapolate two more decades, to a 130B-parameter model. Another factor of 100 in $N$ is another factor of $100^{-0.076} = 10^{-0.152} \approx 0.705$. So $2.51 \times 0.705 \approx 1.77$ nats. The shallow exponent is doing its quiet work: a factor of a *thousand* in parameters, from 125M to 130B, takes the loss from 3.0 to about 1.77, a reduction of roughly 41%. That is the whole story of the GPT-2-to-GPT-3 leap in one line of arithmetic.

### Fitting a power law in practice

The arithmetic above is something you would normally hand to a few lines of code. Here is the realistic version — fit $L(N)$ on a ladder of small runs and extrapolate — using nothing exotic:

```python
import numpy as np
from scipy.optimize import curve_fit

# Cheap small runs: non-embedding params N and measured loss L (nats/token).
N = np.array([1.2e7, 4.5e7, 1.25e8, 3.5e8, 7.6e8])      # 12M ... 760M
L = np.array([3.85,   3.39,   3.02,   2.78,   2.61])    # observed losses

# Kaplan's single-variable form: L(N) = (Nc / N)**alpha_N.
# Fit in log space so the optimizer sees the straight line directly.
def log_loss(logN, alpha_N, logNc):
    return alpha_N * (logNc - logN)

(alpha_N, logNc), _ = curve_fit(log_loss, np.log(N), np.log(L))
print(f"alpha_N = {alpha_N:.3f}, Nc = {np.exp(logNc):.2e}")

# Extrapolate to a 13B-parameter model we have NOT trained yet.
N_target = 1.3e10
L_pred = np.exp(log_loss(np.log(N_target), alpha_N, logNc))
print(f"predicted loss at {N_target:.0e} params: {L_pred:.3f} nats/token")
```

The pattern is the entire methodology in fifteen lines: collect a handful of cheap points, fit the straight line in log space, and read off the prediction for a model you cannot yet afford. In real labs this is wrapped in a few hundred lines of bookkeeping — careful early-stopping, learning-rate re-tuning per scale, consistent parameter counting — but the core is exactly this. The hard part is never the fit; it is making sure the small runs are *comparable* to the large run you are forecasting, which is precisely where Kaplan's setup later turned out to have subtle biases.

### A worked example with the data law

Mirror the parameter example on the data side, because the two are not symmetric and the asymmetry matters. Suppose a large model trained on 10B tokens reaches 2.8 nats, and you want to know what 100B tokens buys, holding the model large enough that data is the binding constraint. Using $L(D) = (D_c/D)^{\alpha_D}$ with $\alpha_D \approx 0.095$:

$$\frac{L(D_2)}{L(D_1)} = \left(\frac{D_1}{D_2}\right)^{\alpha_D} = (0.1)^{0.095} = 10^{-0.095} \approx 0.804.$$

So 10x the data drops the loss to $2.8 \times 0.804 \approx 2.25$ nats — a 20% reduction for an order of magnitude more tokens. Compare that to the parameter law's 16% per decade. Data is a *slightly steeper* lever than parameters ($0.095 > 0.076$), which means, at the margin, a token is worth marginally more loss reduction than a parameter — a hint, again, that pouring everything into parameters might not be optimal. Kaplan's allocation nonetheless favors parameters because the *cost* of a parameter and a token are not symmetric under the $6ND$ constraint; the allocation balances loss reduction against FLOP cost, not against raw count.

### Second-order subtlety: the laws are fits, not physics

It is tempting to treat $\alpha_N = 0.076$ as a constant of nature. It is not. It is a least-squares fit to a particular family of decoder-only transformers, trained with a particular optimizer and learning-rate schedule, on a particular dataset (WebText-derived), evaluated in a particular regime. The exponent is stable across a wide range — that stability is the discovery — but it is not universal across modalities (vision and translation have different exponents, as the earlier work in this series documents) and it is sensitive to bookkeeping choices like whether $N$ includes embeddings. Treat the number as a well-supported point estimate with real uncertainty, not a physical law. This humility is the through-line of the reconciliation post; hold onto it.

## 3. The combined envelope: when both resources bind

The three single-variable laws each assume the other resources are unlimited. Real training runs have *both* a fixed model and a fixed token budget, and the loss depends on both at once. A huge model fed too little data plateaus; a small model fed infinite data plateaus. Neither single-variable curve is honest about its own ceiling.

![A before-and-after comparison contrasting two single-variable laws that each stall against the combined two-variable envelope](/imgs/blogs/kaplan-scaling-laws-language-models-4.png)

The before-and-after figure above is the conceptual shift. On the left, the single-variable view: $L(N)$ is optimistic because it pretends data is free, and $L(D)$ is optimistic because it pretends parameters are free. Each curve stalls at a floor the moment the *other* resource becomes the binding constraint. On the right, the combined envelope: a single surface $L(N, D)$ that adds an $N$-term and a $D$-term so that the loss is governed by whichever resource is scarcer, and predicts every real $(N, D)$ run rather than just the idealized ones.

Kaplan's actual combined form is:

$$L(N, D) = \left[ \left(\frac{N_c}{N}\right)^{\alpha_N / \alpha_D} + \frac{D_c}{D} \right]^{\alpha_D}.$$

This looks awkward at first, so let us decode it. The exponent ratio $\alpha_N / \alpha_D \approx 0.076 / 0.095 \approx 0.80$ converts the parameter term into the same "currency" as the data term so they can be added. When $D$ is enormous, the $D_c/D$ term vanishes and you recover the pure parameter law $L(N) = (N_c/N)^{\alpha_N}$ — check the algebra: the bracket becomes $(N_c/N)^{\alpha_N/\alpha_D}$, raised to $\alpha_D$ gives $(N_c/N)^{\alpha_N}$. When $N$ is enormous, the first term vanishes and you recover $L(D) = (D_c/D)^{\alpha_D}$. The combined form is the smooth interpolation between the two limits, and its shape says something important: to keep moving down the curve as you grow $N$, you must also grow $D$, or the $D_c/D$ term starts to dominate and the loss stalls. The two resources are complements, not substitutes.

This is the structural reason the later Chinchilla form, $L(N, D) = E + A/N^\alpha + B/D^\beta$, is so similar in spirit. Both say: loss is an irreducible floor plus a parameter penalty plus a data penalty. Kaplan's form folds the floor into the limiting behavior rather than writing it as an explicit $E$, and it ties the two exponents together through the ratio. Chinchilla decouples them and fits the floor explicitly. The two are cousins, and the difference in how they parameterize the surface is one reason their compute-optimal recommendations diverge — but that is the next post.

### A worked example with the combined law

Let us watch the combined law do something the single-variable laws cannot: tell you when a model is *starved*. Take a 1B-parameter model ($N = 10^9$) and ask what loss it reaches at two very different token budgets, 20B tokens versus 200B tokens, using the combined form with $N_c = 8.8\times10^{13}$, $D_c = 5.4\times10^{13}$, $\alpha_N/\alpha_D = 0.80$, $\alpha_D = 0.095$.

The parameter term is the same in both cases: $(N_c/N)^{0.80} = (8.8\times10^{13}/10^9)^{0.80} = (8.8\times10^4)^{0.80}$. Now $\log_{10}(8.8\times10^4) = 4.94$, times $0.80$ is $3.96$, so the parameter term is $\approx 10^{3.96} \approx 9.1\times10^3$.

At $D = 2\times10^{10}$ tokens, the data term is $5.4\times10^{13}/2\times10^{10} = 2700$. The bracket is $9100 + 2700 = 11800$, and $L = 11800^{0.095}$: $\log_{10}(11800) = 4.07$, times $0.095$ is $0.387$, so $L \approx 10^{0.387} \approx 2.44$ nats.

At $D = 2\times10^{11}$ tokens (10x more data), the data term drops to $270$. The bracket is $9100 + 270 = 9370$, and $L = 9370^{0.095}$: $\log_{10}(9370) = 3.97$, times $0.095$ is $0.377$, so $L \approx 2.38$ nats.

Ten times the data moved the loss from 2.44 to 2.38 — barely. Why so little? Because the parameter term ($9100$) utterly dominates the data term ($2700 \to 270$); the model is *parameter-starved*, not data-starved, so adding data is nearly wasted. The combined law tells you exactly this: when one term dwarfs the other, you are spending on the wrong resource. For this 1B model, you would do far better to spend that 10x of compute on a bigger model than on more tokens — which is the parameter-favoring instinct Kaplan's allocation encodes. The catch, and the Chinchilla critique, is that this reasoning was carried *too far* at the frontier, where the published models ended up so parameter-heavy that the data term dominated in the *other* direction and they became data-starved. The combined law is the diagnostic; how you read it determines whether you build GPT-3 or Chinchilla.

### The overfitting boundary

There is a practical corollary buried in the combined form. If you fix $N$ and keep feeding tokens, the loss does not improve forever — past some point the model has extracted what its capacity allows and additional unique data yields almost nothing, while *repeated* data eventually overfits. Kaplan operationalizes this: for a model of size $N$, there is a token count beyond which you are wasting data, and below which you are starving the model. The combined law makes that boundary computable. The lesson for a practitioner is to size $D$ to $N$ deliberately rather than "train until it stops improving," which wastes compute on the flat part of the curve. How aggressively you can stretch a fixed unique-token budget by repeating data is itself a scaling question, and a later post in this series quantifies the roughly-four-epochs-are-free rule; for now, the takeaway is that $N$ and $D$ have a matched ratio and overshooting either is waste.

## 4. The critical batch size: how much you can parallelize

So far the laws describe *what loss you reach*. The critical batch size describes *how fast and how efficiently you get there*, and it is the bridge from scaling laws to the brutally practical question of how many GPUs you can usefully throw at a single run.

Every training step processes a batch of tokens, computes a gradient, and updates the weights. If the batch is small, each gradient is noisy, and you need many steps. If the batch is large, each gradient is cleaner, but past a point you get diminishing returns — doubling the batch stops halving the number of steps you need, so you are burning extra tokens to buy a shrinking amount of wall-clock speedup. The crossover is the **critical batch size** $B_{\text{crit}}$, and Kaplan (building on McCandlish et al.'s gradient-noise-scale work) shows it follows its own power law:

$$B_{\text{crit}}(L) \propto L^{-1/\alpha_B}, \qquad \alpha_B \approx 0.21.$$

![A hand-drawn curve of critical batch size rising as the loss falls during training, with a free-parallelism region and a token-cost region](/imgs/blogs/kaplan-scaling-laws-language-models-5.png)

The curve above shows the consequence, and it is genuinely useful intuition. The critical batch size is *not* a fixed property of the model — it grows as training progresses and the loss falls. Early in training, when the loss is high, the critical batch is small: gradients are informative even from few examples, so a small batch is efficient and a huge batch wastes tokens. Late in training, when the loss is low and the remaining signal is subtle, the critical batch is large: you need many examples to extract a clean gradient, so you can profitably scale the batch way up. The green region below the curve is "parallelism nearly free" — you are under $B_{\text{crit}}$, so adding more parallel workers buys near-linear speedup. The amber region above the curve is "tokens bought for speed" — you are over $B_{\text{crit}}$, so each additional doubling of the batch costs extra tokens for a sub-linear time savings.

The exponent $\alpha_B \approx 0.21$ tells you how fast that window opens. Because it appears as $-1/\alpha_B \approx -4.8$ in the exponent on $L$, the critical batch grows *steeply* as the loss falls: a small drop in loss corresponds to a large increase in the batch you can use efficiently. This is why large runs use enormous batches near the end of training but cannot productively start there — early on, the same batch would be far past critical and most of the compute would be wasted noise-averaging.

### The serial-versus-parallel tradeoff in numbers

The critical batch size sets a hard ceiling on the speed-versus-efficiency tradeoff. There is a clean relationship: if $E_{\min}$ is the minimum number of training examples (tokens) needed to reach a loss and $S_{\min}$ is the minimum number of serial optimization steps, then for a run at batch size $B$:

$$\frac{S}{S_{\min}} \approx \frac{1}{1 - B_{\text{crit}}/B} \quad\text{(steps inflate as } B \to \infty\text{)}, \qquad \frac{E}{E_{\min}} \approx \frac{1}{1 - B/B_{\text{crit}}} \quad\text{(tokens inflate as } B \to 0\text{).}$$

Read the two limits. Train at exactly $B = B_{\text{crit}}$ and you pay roughly $2\times$ the minimum steps and $2\times$ the minimum tokens — the balanced point. Train at $B \gg B_{\text{crit}}$ (maximum parallelism, minimum wall-clock) and your step count approaches $S_{\min}$ but your token count blows up: you finish fast but burn data. Train at $B \ll B_{\text{crit}}$ (maximum data efficiency) and you use near-minimal tokens but need a huge number of serial steps, so it takes forever. There is no free lunch; $B_{\text{crit}}$ is the exchange rate between the two currencies, and it drifts upward as the model learns.

For a practitioner this is directly actionable. If your run is wall-clock-bound and you have spare GPUs, push the batch toward and past $B_{\text{crit}}$ late in training when the window is wide — you will trade some token efficiency for real speedup. If your run is data-bound (a fixed corpus you cannot grow), keep the batch near or below $B_{\text{crit}}$ so you do not waste tokens. The single most common mistake is using one fixed batch size for the entire run; the critical batch nearly always wants to grow, and many production recipes ramp the batch over training for exactly this reason.

### The gradient-noise-scale intuition

Where does $B_{\text{crit}}$ come from physically? It is essentially the gradient-noise scale from McCandlish et al.'s "An Empirical Model of Large-Batch Training." A gradient estimated from a batch is the true gradient plus noise; the noise averages down as $1/B$. The critical batch is the point where the noise has been averaged down to roughly the same magnitude as the true gradient signal — below that point you are signal-limited and a bigger batch helps a lot; above it you are already signal-dominated and a bigger batch mostly buys redundant precision. As training progresses and the loss falls, the true gradient gets *smaller* relative to its per-example variance (the easy signal has been learned, leaving subtler structure), so you need a larger batch to average the noise back down below the shrinking signal. That is the mechanism behind $B_{\text{crit}}$ rising as $L$ falls, and it is why the law has the form $B_{\text{crit}} \propto L^{-1/\alpha_B}$ — a decreasing loss drives an increasing critical batch.

Put a number on it. Suppose early in training $B_{\text{crit}}$ is around $5\times10^4$ tokens and you measure the loss as it falls by a factor of two over the run. With $\alpha_B \approx 0.21$, the critical batch grows by $2^{1/0.21} = 2^{4.76} \approx 27\times$ — so by the end of training the efficient batch is well over a million tokens. A run that started at $5\times10^4$ and held it would be leaving a $27\times$ parallelism opportunity on the floor by the end; a run that started at the final $1.4\times10^6$ would have wasted enormous early compute averaging noise that a small batch handled fine. Neither fixed choice is right. The batch wants to grow by more than an order of magnitude over a single run, and the $B_{\text{crit}}(L)$ curve tells you the schedule.

## 5. Bigger models are more sample-efficient

One Kaplan finding sits underneath the allocation argument and deserves its own treatment, because it is counterintuitive and it directly justifies the "stop early" advice. **Larger models reach a given loss in fewer tokens than smaller models.** They are more *sample-efficient*: per token of data consumed, a big model extracts more signal.

This sounds backwards if you think of a big model as "more to train." But the scaling-law view flips it. Plot loss against tokens seen, for several model sizes, and the bigger models' curves sit *below* the smaller ones at every point and reach any target loss after fewer tokens. A big model is a better learner per example, not just a higher-ceiling one. The mechanism is roughly that a larger model has more capacity to represent the regularities in each batch, so it captures more of what each token has to teach before moving on.

This is the engine of the "undertrain on purpose" advice. If bigger models learn more per token, then under a fixed compute budget the efficient thing is to make the model big and *not* run it all the way to convergence — stop while it is still on the steep part of its learning curve, because at that point it is still extracting a lot of loss reduction per token, and the marginal token is better spent than it would be late in a smaller model's run. Combined with the near-constant optimal step count, this is why a Kaplan-optimal model is large and lightly trained. It is internally consistent: sample efficiency plus the allocation plus constant steps all point the same way, toward big-and-undertrained.

The irony, visible only in hindsight, is that "more sample-efficient" was measured in the *early* part of training, where big models do shine. Run training much longer — feed 20x the tokens, as Chinchilla does — and the smaller-but-longer-trained model catches up and passes the bigger-but-undertrained one at equal compute. Sample efficiency early does not imply compute optimality overall. Kaplan measured the right phenomenon and drew a conclusion that held in the regime measured but not in the regime that mattered for deployment. This is a recurring shape in the scaling-law story: a true local observation, extrapolated one regime too far.

## 6. The compute-optimal allocation — the headline that built GPT-3

We now arrive at the result that defined a generation of models, and the one this series exists to interrogate. The question is: given a fixed compute budget $C$, how should you split it between a bigger model (more $N$) and more training (more $D$)? Since $C \approx 6ND$, every extra FLOP can go into either factor, and the split determines the loss you reach.

![A pipeline showing a compute budget split mostly into parameters and only weakly into data with steps held nearly constant](/imgs/blogs/kaplan-scaling-laws-language-models-6.png)

Kaplan's answer, derived by minimizing the combined loss $L(N, D)$ subject to $6ND = C$, is the pipeline above:

$$N_{\text{opt}} \propto C^{a}, \quad a \approx 0.73; \qquad D_{\text{opt}} \propto C^{b}, \quad b \approx 0.27; \qquad S \approx \text{nearly constant}.$$

Read those exponents carefully, because they are the entire argument. As you scale the compute budget, the model size should grow as $C^{0.73}$ — almost linearly — while the data should grow only as $C^{0.27}$ — very slowly. The number of optimization steps stays roughly constant. In plain terms: **when you get more compute, spend almost all of it on a bigger model and very little on more data.** A 10x bigger budget means a $10^{0.73} \approx 5.4$x bigger model but only a $10^{0.27} \approx 1.9$x increase in tokens.

Why does the math come out this way? Because in the combined law the parameter term and the data term trade off, and the optimal balance depends on the exponents. With Kaplan's measured values, the parameter term is the cheaper lever per FLOP at the margin, so the optimizer pours budget into $N$. The danger box at the end of the pipeline states the resulting advice bluntly: train a very large model on relatively modest data, and *stop well before convergence* — because the steps stay constant while the model grows, a compute-optimal Kaplan model is deliberately undertrained relative to what it could absorb.

### Where the 0.73 comes from

It is worth seeing, at least in sketch, why the optimization produces $a \approx 0.73$ rather than some other number, because the structure of the derivation is exactly what Chinchilla later redoes with different inputs. You want to minimize the combined loss $L(N, D)$ subject to the budget constraint $C = 6ND$. Substitute $D = C/(6N)$ into the loss to get a function of $N$ alone at fixed $C$, then set the derivative to zero. Near the optimum, the two terms of the combined law are being traded against each other, and the first-order condition balances the marginal loss reduction from spending the next FLOP on $N$ against spending it on $D$.

When you carry the algebra through with Kaplan's combined form, the optimal $N$ scales as a power of $C$ whose exponent is determined by the two single-variable exponents. Schematically, the exponent on $N$ comes out near

$$a = \frac{\alpha_D}{\alpha_N + \alpha_D} \approx \frac{0.095}{0.076 + 0.095} \approx 0.56 \quad\text{(naive balance)},$$

but Kaplan's *measured* allocation, fit directly to the empirical compute-optimal frontier rather than read off the naive balance of the two exponents, lands at $a \approx 0.73$. The gap between the naive $0.56$ and the measured $0.73$ is itself a clue: it means the empirical frontier was steeper in $N$ than a clean two-term balance predicts, which is exactly the kind of distortion the reconciliation work later traces to the small-scale parameter-counting and warmup artifacts. The naive balance, run with Chinchilla's near-equal exponents, gives $a \approx 0.5$ — the corrected answer. So the *form* of the derivation is shared by both papers; only the inputs (the exponents and the fitting methodology) differ, and that difference is the whole controversy. Do not memorize $0.73$; understand that it is the output of a constrained minimization whose inputs were slightly off.

### From the formula to a 175-billion-parameter model

This is not abstract. Kaplan's allocation is the direct intellectual lineage of GPT-3. The recipe — biggest model your budget allows, train it on a comparatively modest token count, stop before convergence — produced GPT-3 as **175 billion parameters trained on roughly 300 billion tokens**. Run the ratio: that is fewer than two tokens per parameter. The model was, by design, enormous and lightly trained, because the scaling laws of the day said that was the compute-optimal thing to do.

Let us sanity-check the compute. With $C \approx 6ND$ and $N = 1.75 \times 10^{11}$, $D = 3 \times 10^{11}$:

$$C \approx 6 \times 1.75 \times 10^{11} \times 3 \times 10^{11} = 3.15 \times 10^{23} \ \text{FLOPs}.$$

Converting to PF-days at $8.64 \times 10^{19}$ FLOPs each, that is about $3.6 \times 10^{3}$ PF-days — thousands of petaflop-days, a landmark training run for 2020. The point is that the *shape* of that run — the 175B/300B split — was a decision made by the scaling law, not by trial and error. Kaplan said "spend it on parameters," and GPT-3 spent it on parameters.

### The tension, set up and deliberately left open

Here is where we plant the flag and walk away. The allocation $N_{\text{opt}} \propto C^{0.73}$ is the part of Kaplan that did *not* survive contact with the next two years. In 2022, DeepMind's Chinchilla paper (Hoffmann et al., arXiv:2203.15556) re-ran the compute-optimal analysis with more careful methodology and found a dramatically different answer: $a \approx b \approx 0.5$, meaning model size and data should grow *together*, roughly as $\sqrt{C}$ each, with about **20 tokens per parameter** at the optimum — an order of magnitude more data per parameter than GPT-3 used. Under that correction, GPT-3 and its 175B-parameter contemporaries were severely *undertrained*: you could reach the same loss with a much smaller model fed far more data, at the same compute.

Here is the disagreement laid out as a table, because the contrast is sharper side by side. Note that almost every row agrees; only the allocation rows differ.

| Aspect | Kaplan (2020) | Chinchilla (2022) |
|---|---|---|
| Loss is a predictable power law? | Yes | Yes |
| Parameter count $N$ means | Non-embedding | Total (incl. embeddings) |
| Compute model | $C \approx 6ND$ | $C \approx 6ND$ |
| $N_{\text{opt}} \propto C^{a}$ | $a \approx 0.73$ | $a \approx 0.50$ |
| $D_{\text{opt}} \propto C^{b}$ | $b \approx 0.27$ | $b \approx 0.50$ |
| Tokens per parameter at optimum | Few (budget-dependent) | ~20 (roughly fixed) |
| Recommended run | Big model, modest data, undertrain | Smaller model, much more data |
| Verdict (per 2024 reconciliation) | Allocation was a measurement artifact | Allocation is correct |

We are not going to resolve which exponent is right here — that is the explicit job of the [Chinchilla post](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) and the [reconciliation post](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation). But notice what is at stake and what is *not*. Both camps agree completely that bigger helps and that loss is a predictable power law. They disagree only on the *split of a fixed budget* — how to divide $C$ between $N$ and $D$. That is a narrower disagreement than it first appears, and the eventual diagnosis (a parameter-counting bookkeeping choice plus small-scale warmup and learning-rate artifacts) is one of the best "don't worship the constants" lessons in the field. The exponent $0.73$ is wrong; the *method* — fit a law, minimize loss under a compute constraint — is exactly right. Hold both thoughts.

## 7. Architecture barely matters next to scale

One of Kaplan's most liberating findings, and the one that most reshaped how labs spend engineering effort, is that the *shape* of the transformer is a weak lever compared to its size. Within wide ranges, depth versus width, the feed-forward expansion ratio, and the number of attention heads at fixed model dimension move the loss by a couple of percent. Change $N$, $D$, or $C$ and the loss moves along the power law, by orders of magnitude.

![A layered graph contrasting weak architecture-shape knobs against the strong levers of total scale](/imgs/blogs/kaplan-scaling-laws-language-models-8.png)

The graph above makes the asymmetry visual. On the left, the shape knobs — aspect ratio, feed-forward ratio, head count — all feed into a single "loss shift of a few percent at fixed $N$" node. On the right, the scale levers feed into "loss moves along the power law, orders of magnitude." The arrow between them, labeled "so spend on," is the practical recommendation: stop tuning architecture and start tuning scale.

This is not to say architecture is irrelevant. It says that *at fixed parameter count*, the loss is remarkably insensitive to how you arrange those parameters, as long as you stay within sane ranges — roughly, aspect ratios from very deep-and-thin to very wide-and-shallow all land within a few percent of each other. A transformer that is 6 layers wide or 48 layers deep, at the same total $N$, reaches nearly the same loss. The practical consequence is that you should pick a reasonable shape, fix it, and pour your effort into scaling, because the shape decision is worth a percent and the scale decision is worth a factor.

There is an important caveat that the engineering community learned the hard way: "barely matters for loss" is not "barely matters for *throughput*." Aspect ratio enormously affects how efficiently you can train — wide-and-shallow models parallelize differently than deep-and-thin ones, with different memory and communication profiles. So architecture choices are still real engineering decisions; they are just decisions about *how cheaply you reach a given loss*, not about *what loss is reachable*. Kaplan's claim is specifically about the latter. Pick the shape that trains fastest on your hardware, and let the scaling law handle the quality.

### The flip side: shape choices that do change the exponent

It is worth naming the boundary of this claim, because it is sometimes overstated. The architectural choices Kaplan found inert are *within-family* knobs — variations of the same decoder-only transformer. Changes that alter the fundamental computation, such as moving to a sparsely-activated mixture-of-experts, or radically changing the attention mechanism, *can* change the effective exponent or the constant, because they change how many parameters are active per token and how information flows. The clean statement is: holding the transformer family and the active-parameter count fixed, shape is a weak lever. Cross-family changes are a different question, and a live research area. Within the regime Kaplan studied, though, the recommendation stands: scale beats shape.

## 8. Putting the laws to work: a complete worked example

Let us run a realistic planning exercise end to end, the kind a research lead actually does, using only Kaplan's laws. The goal is to make the abstractions concrete by turning them into a budget decision.

Suppose you have a compute budget of $C = 100$ PF-days and you want to know, under Kaplan's allocation, what model size and token count to target, and what loss to expect. We will use the proportionalities and anchor them with the published constants.

**Step 1: allocate the budget.** Under Kaplan, $N_{\text{opt}} \propto C^{0.73}$ and $D_{\text{opt}} \propto C^{0.27}$. The proportionality constants come from the fits; the qualitative split is what matters here. With $C = 100$ PF-days, $C^{0.73} = 10^{2 \times 0.73} = 10^{1.46} \approx 29$ and $C^{0.27} = 10^{0.54} \approx 3.5$, both relative to the $C = 1$ baseline. So relative to a 1-PF-day model, you scale parameters by ~29x and tokens by only ~3.5x. The model grows fast; the data crawls. This is the Kaplan signature.

**Step 2: check the constraint.** Whatever $N$ and $D$ you pick, they must satisfy $6ND = C$ in consistent units. Converting $C = 100$ PF-days to FLOPs: $100 \times 8.64 \times 10^{19} = 8.64 \times 10^{21}$ FLOPs. So $ND = C/6 \approx 1.44 \times 10^{21}$. If the allocation puts you at, say, $N \approx 2.7 \times 10^{9}$ parameters, then $D \approx 1.44 \times 10^{21} / 2.7 \times 10^{9} \approx 5.3 \times 10^{11}$ tokens — about 530B tokens for a 2.7B model, roughly 200 tokens per parameter at this small budget. (Note the tokens-per-parameter ratio shifts with budget under Kaplan; the allocation is about exponents, not a fixed ratio like Chinchilla's 20.)

**Step 3: forecast the loss.** Plug $N$ and $D$ into the combined envelope:

$$L(N, D) = \left[ \left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D} \right]^{\alpha_D}.$$

With $N_c \approx 8.8 \times 10^{13}$, $D_c \approx 5.4 \times 10^{13}$, $\alpha_N/\alpha_D \approx 0.80$, $\alpha_D \approx 0.095$, and our $N, D$ from above, the parameter term is $(8.8\times10^{13}/2.7\times10^{9})^{0.80} = (3.26\times10^{4})^{0.80}$. Now $\log_{10}(3.26\times10^{4}) = 4.51$, times $0.80$ is $3.61$, so the term is $\approx 10^{3.61} \approx 4.1 \times 10^{3}$. The data term is $5.4\times10^{13}/5.3\times10^{11} \approx 102$. The bracket is $\approx 4.1\times10^{3} + 102 \approx 4.2\times10^{3}$, and raising to $0.095$: $\log_{10}(4.2\times10^3) = 3.62$, times $0.095$ is $0.344$, so $L \approx 10^{0.344} \approx 2.2$ nats per token. That is your forecast, computed before spending a single GPU-hour on the big run.

**Step 4: notice the imbalance.** The parameter term ($\sim 4100$) dominates the data term ($\sim 102$) by a factor of 40. In the combined law, a term that dominates is the term you are starved on. Kaplan's allocation deliberately keeps the parameter term large — that is what "big model, modest data" means — but a careful reader will already smell the Chinchilla critique here: if the parameter term so thoroughly dominates, maybe the model is *too big for its data* and you would reach the same loss more cheaply with a smaller model and more tokens. That is exactly the argument Chinchilla makes rigorously. We compute it here only to show that the tension is visible in the arithmetic, not to resolve it.

This four-step procedure is the entire practical payload of scaling laws: allocate, constrain, forecast, audit. You can run it on a spreadsheet. The discipline of doing it *before* the big run is what separates a planned training program from an expensive guess.

## 9. From Kaplan to GPT-3 to Chinchilla: the timeline

It helps to see the sequence of events, because the story is as much about how the field updated as it is about the math.

![A timeline from Kaplan's 2020 laws through GPT-3 to the 2022 Chinchilla correction and the 2024 reconciliation](/imgs/blogs/kaplan-scaling-laws-language-models-7.png)

The timeline above traces the arc. January 2020: Kaplan et al. publish the laws, with the allocation $N_{\text{opt}} \propto C^{0.73}$. May 2020: GPT-3 ships as 175B parameters on ~300B tokens — the allocation made concrete, and an enormous success that seemed to vindicate the law. 2021–2022: Gopher (280B on 300B tokens), Megatron-Turing NLG (530B), and a wave of frontier models follow the same big-model recipe, all sized in the Kaplan spirit. March 2022: Chinchilla lands and reframes everything — 70B parameters on 1.4T tokens, beating the much larger Gopher at the same training compute, and arguing that the whole cohort was undertrained. 2024: the reconciliation work (Pearce & Song; Porian et al.) diagnoses *why* Kaplan and Chinchilla disagreed, tracing it largely to the non-embedding parameter counting at small scale plus warmup and learning-rate artifacts, and concludes Chinchilla's exponents are the correct ones.

The intellectual lesson in that arc is worth stating plainly, because it is the reason this series treats Kaplan and Chinchilla together rather than declaring a winner. Kaplan's *framework* — loss is a predictable power law; fit it cheaply and extrapolate; optimize allocation under a compute constraint — is correct and is still exactly how the field plans runs. Kaplan's *specific allocation constant* was an artifact of measurement choices. The framework outlived the constant. If you internalize one meta-lesson from the whole scaling-laws literature, let it be that distinction: trust the method, hold the constants loosely, and always write down your bookkeeping (total versus non-embedding parameters, which FLOPs you count) before you fit anything.

## Case studies: scaling laws in the wild

The theory lands harder when attached to real models and real mistakes. Here are nine episodes, each a concrete instance of the laws helping or the bookkeeping biting.

### 1. GPT-3: the allocation made flesh

GPT-3 is the canonical Kaplan-allocation model: 175B parameters, ~300B tokens, fewer than two tokens per parameter. At the time this looked optimal — the laws said pour budget into parameters. The model was a landmark and its few-shot abilities were genuinely new. But by Chinchilla's accounting, GPT-3 was leaving performance on the table: a smaller model on far more tokens would have matched its loss at the same compute, and would have been dramatically cheaper to *serve*, since inference cost scales with $N$. The lesson is not that GPT-3 was a mistake — it advanced the field enormously — but that "compute-optimal for training" and "best model to deploy" are different objectives, and Kaplan optimized only the first.

### 2. Gopher: 280B on 300B tokens

DeepMind's own Gopher, trained before Chinchilla, is the most poignant case study because the *same lab* later showed it was undertrained. Gopher used 280B parameters on roughly 300B tokens — about one token per parameter, even more parameter-heavy than GPT-3. Chinchilla (70B on 1.4T tokens) then beat Gopher across a broad eval suite *at the same training compute*. Gopher is the clearest single demonstration that Kaplan's allocation, applied at frontier scale, produced models that were too big for their data. It is also a testament to honest science: the team that built Gopher published the result that superseded it.

### 3. The non-embedding parameter trap

The reconciliation work pins much of the Kaplan-Chinchilla gap on one bookkeeping choice: Kaplan counted non-embedding parameters, Chinchilla counted total. At GPT-3 scale this is negligible, but Kaplan's *fits* used models down to a few hundred hidden units, where the embedding and unembedding matrices are a large fraction of the total parameter count. Counting only non-embedding parameters at that small scale makes the $N$-to-compute relationship non-linear in a way that distorts the fitted exponent. Re-running the analysis with consistent total-parameter counting at small scale reproduces the Chinchilla allocation. The lesson: the choice of what to count, made for a sensible reason at large scale, became a systematic bias at small scale. Define your bookkeeping before you fit.

### 4. The warmup artifact at small scale

Porian et al. identified a second culprit: learning-rate warmup. Short training runs at small scale spend a disproportionate fraction of their steps in warmup, where the model is barely learning, which biases the small models toward higher loss and tilts the fitted exponent. The fix is to scale warmup duration to the run length rather than using a fixed warmup. This is a classic "the experiment, not the model, was wrong" finding — the small models were not inherently worse, they were under-warmed-up relative to the long runs, and the fitting procedure mistook the artifact for a scaling effect.

### 5. The learning-rate-schedule mismatch

A third reconciliation finding: the optimal learning rate and schedule shift with scale, and if you do not re-tune them per scale, the smaller models in your sweep are run with sub-optimal hyperparameters, again biasing the fit. Chinchilla's Approach 1 (fixing model size and varying tokens, reading the minimum off each training curve) is partly robust to this because it reads the best achievable loss at each size. Kaplan's setup was more exposed. The practical takeaway is brutal and simple: re-tune learning rate and warmup at every scale in your sweep, or your fitted exponent will encode your tuning laziness rather than the model's scaling behavior.

### 6. Critical batch size at frontier scale

The critical-batch-size law is one part of Kaplan that aged well and is used daily. Frontier runs ramp the batch size over training — starting modest, ending enormous — precisely because $B_{\text{crit}}$ grows as the loss falls. Teams that used a single fixed large batch from step zero discovered the early phase was wildly token-inefficient: most of the early compute went into noise-averaging gradients that a small batch would have produced just as well. Teams that used a single fixed small batch discovered the late phase was wall-clock-bound, leaving GPUs idle relative to what the wide late-training $B_{\text{crit}}$ would have allowed. Ramping the batch along the $B_{\text{crit}}(L)$ curve is the resolution, and it is now standard.

### 7. Forecasting a run before committing the budget

A research team I worked alongside used the Kaplan procedure exactly as intended: they fit $L(N)$ and the combined law on a ladder of small models — a few million to a few hundred million parameters, each cheap — and extrapolated to predict the loss of a planned multi-billion-parameter run. The forecast came within a few percent of the realized loss. The value was not the precision per se; it was the *confidence to commit*. The budget committee approved the run because the curve made the outcome legible. This is the everyday payoff of scaling laws, and it is why the method survives even where Kaplan's specific allocation constant did not.

### 8. The inference-cost blind spot

Kaplan optimizes training compute and ignores inference entirely. For a model trained once and served billions of times, that is the wrong objective. Inference costs roughly $2N$ FLOPs per token, so a model twice as big costs twice as much *every time it answers*. A Kaplan-optimal big model can be far more expensive over its serving lifetime than a smaller model trained on more tokens to the same loss. This blind spot motivated an entire subsequent line of work on inference-aware scaling, which pushes the optimum toward smaller, more-trained models — well past even Chinchilla's 20 tokens per parameter. Kaplan's law is correct for what it optimizes; the trap is forgetting what it does not optimize.

### 9. The shape-versus-throughput confusion

A recurring engineering mistake is to read "architecture barely matters" as "architecture is free to choose at random." It matters for *throughput* even when it does not matter for *loss*. A team picked an extreme aspect ratio because the scaling law said loss was insensitive to it, then watched training run at a fraction of the achievable hardware utilization because that shape parallelized badly on their interconnect. The loss would have been fine; the wall-clock and dollar cost were not. The correct reading of Kaplan is: shape is a weak lever for *quality*, so choose the shape that is a strong lever for *speed* on your specific hardware, and let scale handle quality.

### 10. Megatron-Turing NLG 530B: the high-water mark of big-model thinking

Megatron-Turing NLG, at 530 billion parameters, is the largest dense model trained squarely in the Kaplan-allocation spirit — enormous parameter count, comparatively modest token budget. It was a triumph of distributed-systems engineering, requiring 3D parallelism across thousands of GPUs just to fit and train. And by the Chinchilla accounting that landed shortly after, it was deeply undertrained: a far smaller model on far more tokens would have matched or beaten it at the same compute, while being an order of magnitude cheaper to serve. MT-NLG is the clearest illustration that following the allocation to its logical extreme produced systems that were heroic to build and economically suboptimal to run. The systems lesson is real and durable; the allocation lesson was a dead end.

### 11. The benchmark-versus-loss gap

A subtle trap in using scaling laws: the law predicts *loss*, not *benchmark accuracy*. Loss is smooth and predictable; downstream task accuracy is often not, sometimes appearing to jump at certain scales. A team forecasting a run with Kaplan's laws nailed the loss prediction to within a percent but was surprised when a specific reasoning benchmark improved far more (or less) than the loss curve suggested. The resolution is that loss is the right target for the *scaling law* but a poor proxy for any single downstream metric; benchmark behavior is a noisier, sometimes discontinuous function of loss. Use the law to forecast loss with confidence, and treat the loss-to-capability mapping as a separate, looser relationship. Kaplan never claimed otherwise, but the conflation is common.

### 12. The early-stopping discipline

Kaplan's $L(D)$ law is fit *with early stopping* — you stop before the model overfits the finite token budget. A team that ignored this and trained to a fixed step count, repeating their limited corpus many times, watched their loss diverge from the predicted curve as the model began memorizing rather than generalizing. The fix was to early-stop at the point the combined law implied, sized to their token budget. This case is a reminder that the laws describe a *well-run* training process; they are not robust to procedural mistakes like over-repeating data past the point of diminishing returns. How far you can push repetition before it hurts is itself a scaling question that a later post in this series quantifies, but Kaplan's baseline assumption is a single, well-early-stopped pass through enough unique data.

## Common misreadings of Kaplan, and the correct reading

It is worth collecting the ways people misuse these laws, because each misreading has cost real money.

**"Bigger is always better, so just maximize $N$."** No — the combined law says $N$ and $D$ are complements. A bigger model on the same data eventually stalls at the $D$-limited floor. Maximizing $N$ alone, without growing $D$, walks you off the predictable curve into a plateau. Kaplan's *allocation* over-weights $N$, but even Kaplan never said grow $N$ with fixed $D$.

**"The exponent is a law of nature."** No — it is a fit to a specific model family, optimizer, and dataset, with specific bookkeeping. The reconciliation work proved how sensitive the *allocation* exponent is to those choices. Re-measure for your setting when the stakes justify it.

**"Architecture doesn't matter, so don't think about it."** No — it doesn't matter much for *loss at fixed $N$*, but it matters enormously for *training throughput* and for *cross-family* changes like mixture-of-experts. The claim is narrow and specific.

**"Scaling laws mean we just need more compute."** Partly — but Kaplan optimizes *training* compute only, ignoring inference, data quality, and the data wall. Each of those is its own scaling axis, treated in later posts. Kaplan is the first map, not the whole atlas.

**"GPT-3's size proves Kaplan was right."** No — GPT-3's *success* proves scale helps, which everyone now agrees on. GPT-3's specific 175B/300B *split* is the part Chinchilla overturned. Conflating "scale works" with "this particular allocation is optimal" is the single most common error, and it is the one this whole post is built to prevent.

## When to reach for Kaplan's laws, and when not to

**Reach for Kaplan's framework when:**

- You need to forecast the loss of a large run before committing the budget, and you can afford a ladder of cheap small runs to fit the curve.
- You are setting up a compute-optimal *training* plan and want a principled way to allocate a fixed budget — even if you use Chinchilla's exponents, the method is Kaplan's.
- You are reasoning about the critical batch size and how to ramp it, or about how much parallelism a run can productively absorb at a given point in training.
- You want to justify *not* spending engineering effort on architecture micro-tuning, and redirect it toward scale.
- You are teaching or explaining why loss is predictable at all — the log-log straight line is the cleanest possible entry point.

**Be careful, or reach for a later post, when:**

- You are deciding the *split* of a fixed budget between model size and data. Use [Chinchilla's](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) $a \approx b \approx 0.5$ and the ~20-tokens-per-parameter rule, not Kaplan's $0.73 / 0.27$. The allocation is the one piece of Kaplan that was corrected.
- You care about deployment cost, not just training cost. Kaplan ignores inference; a model that is training-compute-optimal can be badly inference-inefficient, and you should optimize total lifetime cost instead.
- You are fitting your own laws and have models spanning a wide scale range. Watch the bookkeeping — total versus non-embedding parameters, last-layer FLOPs — and re-tune learning rate and warmup per scale, or you will reproduce Kaplan's artifacts rather than the true exponents. The full diagnosis is in the [reconciliation post](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation).
- You are working with mixture-of-experts, radically different attention, or non-text modalities. The "shape barely matters" and the specific exponents are claims about a within-family decoder-only transformer on text; cross-family changes can move the law.
- You are tempted to treat any single exponent as a constant of nature. It is a fit. Treat it as a point estimate with uncertainty, and re-measure for your own setup when the stakes are high.

## What this means in practice

The durable lesson of Kaplan 2020 is not the number $0.73$ — that one was overturned. It is the *posture*: loss is predictable, so plan it. Before a large run, fit the laws on cheap small models, write down your forecast, and audit which term in the combined law dominates so you know whether you are starved on parameters or on data. Pick a transformer shape that trains fast on your hardware and stop fiddling with it, because shape is a weak lever for quality. Ramp the batch size along the critical-batch curve as the loss falls, rather than fixing it. And above all, write your bookkeeping down — total versus non-embedding parameters, which FLOPs you count — before you fit anything, because the single most expensive error in this entire literature was a bookkeeping choice that nobody noticed for two years.

If you are about to plan a real run, here is the checklist the laws imply, in order. First, decide and *write down* your bookkeeping: total or non-embedding parameters, which FLOPs you count (including or excluding the last-layer projection), and your unit for compute. Second, build a ladder of cheap small runs — a handful spanning two or three orders of magnitude in $N$ — and *re-tune the learning rate and warmup at each rung* rather than reusing one schedule, because a fixed schedule biases the small rungs and corrupts the fit. Third, fit the single-variable laws and the combined envelope in log space, and check that your points actually fall on a straight line; a point that misses is a signal, not noise to be smoothed over. Fourth, forecast the loss of the big run and audit which term of the combined law dominates so you know whether you are parameter- or data-starved. Fifth — and this is the step the field had to learn the hard way — use *Chinchilla's* allocation, not Kaplan's, to set the actual $N$-to-$D$ split, and if you will serve the model heavily, push even further toward more tokens to account for inference cost. Sixth, ramp the batch size along the critical-batch curve over the run rather than fixing it. That sequence is the practical distillation of everything in this post, and only the fifth step departs from Kaplan's own recommendation.

If you take the allocation exponent from this post, take it as a historical artifact and a cautionary tale, not as planning guidance. For how to actually split a budget today, read the [Chinchilla post](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling); for why Kaplan and Chinchilla disagreed and who was right, read the [reconciliation post](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation); and for the foundations of why power laws describe loss in the first place, start from the [predictability foundations post](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations) that opens this series. Kaplan gave us the first map of the territory. The map had one road drawn in the wrong place. The territory was real, and the act of mapping it changed how every frontier model since has been planned.

## Further reading

- Kaplan et al., "Scaling Laws for Neural Language Models" (2020) — https://arxiv.org/abs/2001.08361
- Hoffmann et al., "Training Compute-Optimal Large Language Models" (Chinchilla, 2022) — https://arxiv.org/abs/2203.15556
- Pearce & Song, "Reconciling Kaplan and Chinchilla Scaling Laws" (2024) — https://arxiv.org/abs/2406.12907
- Porian et al., "Resolving Discrepancies in Compute-Optimal Scaling of Language Models" (2024) — https://arxiv.org/abs/2406.19146
- Hestness et al., "Deep Learning Scaling is Predictable, Empirically" (2017) — https://arxiv.org/abs/1712.00409
- McCandlish et al., "An Empirical Model of Large-Batch Training" (2018) — https://arxiv.org/abs/1812.06162

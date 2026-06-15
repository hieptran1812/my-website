---
title: "Scaling laws for transfer and fine-tuning: when pretraining is a data multiplier"
date: "2026-06-15"
description: "Learn to treat pretraining as a model-size-dependent multiplier on scarce fine-tuning data, predict the effective data a base model contributes, and choose between full fine-tuning, LoRA, and prompt methods by how much target data you actually have."
tags: ["scaling-laws", "transfer-learning", "fine-tuning", "effective-data-transferred", "ossification", "lora", "parameter-efficient-tuning", "pretraining", "large-language-models", "loss-curves", "deepmind"]
category: "machine-learning"
subcategory: "Scaling Laws"
author: "Hiep Tran"
featured: true
readTime: 53
---

There is a question every team that fine-tunes a model has to answer, usually with a shrug and a vibe: was the pretraining worth it? You have a base model that cost someone millions of dollars and a few thousand labeled examples of your own. You fine-tune, the loss drops, the demo works. But you never actually measured what the pretraining bought you, because the obvious experiment — train the same architecture from scratch on just your data — is expensive, annoying, and usually skipped. So the field developed a folk belief: pretraining is "good," more pretraining is "better," and fine-tuning is the part you control. The scaling laws for transfer turn that shrug into a number. They say pretraining is not a vague good; it is a *multiplier* on your fine-tuning data, and the size of that multiplier is something you can compute before you spend a dollar. The diagram below is the mental model for the whole post: a pretrained base converts each of your scarce fine-tuning examples into a much larger pile of *effective* data, and a from-scratch model — which only ever sees your raw data — is the baseline that quantifies the gap.

![A branching diagram showing a pretrained base and fine-tune data combining into effective data transferred, which adds to the fine-tune data to produce a larger effective dataset that beats a from-scratch baseline](/imgs/blogs/transfer-finetuning-scaling-laws-1.png)

Two papers anchor this post. The first is Hernandez et al. 2021, "Scaling Laws for Transfer" (OpenAI, arXiv:2102.01293), which gives the clean, almost shocking result: the data a pretrained model effectively transfers to a downstream task follows a power law in both the fine-tuning dataset size and the model size, and for the text-to-Python transfer they studied, a 10x bigger base model is worth roughly 100x more fine-tuning data. The second is Zhang et al. 2024, "When Scaling Meets LLM Finetuning" (Google DeepMind, ICLR 2024, arXiv:2402.17193), which fits a multiplicative joint law for the fine-tuning loss and draws the practical conclusions: fine-tuning benefits more from base-model size than from how much the base was pretrained on, parameter-efficient methods like LoRA do not get better as you scale their rank, and the best method depends sharply on how much target data you have. This post builds the intuition first, then the math, then worked numeric examples, then the failure mode that ruins the whole story if you ignore it — ossification — and finally a decision procedure you can actually run.

> [!important] The one number to remember: a 10x bigger base is worth about 100x more fine-tuning data
> - **Pretraining is a multiplier, not a constant.** The "effective data transferred" is $D_T = k \cdot D_F^{\alpha} \cdot N^{\beta}$, where $D_F$ is your fine-tuning data and $N$ is the model size. For OpenAI's text-to-Python fit, $k \approx 1.9 \times 10^4$, $\alpha = 0.18$, $\beta = 0.38$.
> - **The multiplier grows with model size and shrinks as your own data grows.** The effective-data multiplier is $(D_F + D_T)/D_F \approx k \cdot N^{\beta}/D_F^{1-\alpha}$. Because $\beta > \alpha$, a 10x bigger model buys roughly $10^{2}$ = 100x effective fine-tuning data in the text-to-Python regime.
> - **This only holds in the low-data regime** — when $D_F$ is at most about 10% of the data you would need to reach 99% of infinite-data performance. Past that, you are in the high-data regime, where the pretrained prior **ossifies** and can even underperform training from scratch.
> - **Fine-tuning benefits more from base-model size than from pretraining data.** Zhang et al.'s multiplicative law $\hat{L}(X, D_f) = A \cdot X^{-\alpha} \cdot D_f^{-\beta} + E$ shows the model-size axis has the steepest payoff.
> - **PET-parameter scaling is largely ineffective.** Scaling LoRA rank or prompt length does not lower loss the way scaling model size does — the exponent on PET parameters is near zero.
> - **The best method is data-dependent:** tiny data favors prompt/PET, more data favors LoRA, plentiful in-domain data favors full fine-tuning.
> - **Cross-link the budget story:** transfer interacts with [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling), [data-constrained scaling](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws), and [reward-model overoptimization](/blog/machine-learning/scaling-laws/reward-model-overoptimization-scaling).

## Why this is different from what people assume about fine-tuning

The right way to feel this result is to lay the common assumptions next to what the measurements actually say. Most practitioners carry a mental model of fine-tuning that is wrong in three specific ways, and each wrong belief leads to a concrete bad decision.

| Question | The common assumption | The transfer-scaling reality |
|---|---|---|
| What does pretraining contribute to a downstream task? | A fixed head start that you "get for free" | A multiplier on your fine-tuning data, sized by $D_T = k \cdot D_F^{\alpha} \cdot N^{\beta}$ |
| If I have little target data, what helps most? | More target data — collect or label more | A bigger base model: in the low-data regime, $\beta > \alpha$ means model size moves the needle harder |
| Does more fine-tuning data always beat a smaller base? | Yes, data is king | No — in the low-data regime, $10\times$ model $\approx 100\times$ data, so size can dominate |
| Is more pretraining always better for my task? | Yes, strictly monotone | No — with abundant in-domain data the prior ossifies and can hurt |
| Does scaling LoRA rank lower loss like scaling size? | Yes, rank is a capacity knob | No — the PET-parameter exponent is near zero; rank scaling stalls |
| What is the right fine-tuning method? | Whatever is cheapest / trendiest | It depends on $D_F$: prompt/PET for tiny data, LoRA for medium, full fine-tuning for plentiful |

The two rows that change real decisions are the second and the fourth. The second says that when you are data-starved — which is the normal condition for any specialized task — the cheapest way to lower your loss is often not to label more data but to start from a bigger base. That is counterintuitive to anyone trained to believe data is always the bottleneck, and it falls directly out of the fact that the model-size exponent $\beta$ is larger than the data exponent $\alpha$. The fourth row says the monotone "more pretraining is better" belief has a ceiling, and crossing it quietly is one of the most common ways teams waste a fine-tuning run.

> If you take one thing from this post: pretraining does not give you a fixed head start. It gives you a multiplier, the multiplier is bigger for bigger models, and it only applies while your own data is scarce. Treat it as a quantity you compute, not a virtue you assume.

### A short history of how the transfer law was found

It helps to trace the lineage, because the transfer law did not appear from nowhere. The seed was empirical and predates the language-model boom: Hestness et al. at Baidu (2017) showed that generalization error falls as a power law in dataset size across vision, speech, and language — a straight line on log-log axes. That established the fact the whole field rests on: error is *predictable* from a smooth power-law curve, so you can forecast it before training. The second step was Kaplan et al. (2020), which made this concrete for transformers with single-variable loss power laws in model size $N$ and data $D$, over seven orders of magnitude; the [Kaplan post](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models) covers it in full, and the [Chinchilla post](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) covers the joint $L(N,D)$ correction.

The third step is the one this post is about. Hernandez, Kaplan, Henighan, and McCandlish (OpenAI, 2021) asked a sharper question than "how does loss scale": *how much of a downstream task's data does pretraining replace?* They pretrained transformers on natural-language text, fine-tuned them on Python code, and compared, at every model size and every fine-tuning-data size, against an identical architecture trained on Python from scratch. The gap between the two — measured in how much extra data the from-scratch model would have needed to match the fine-tuned one — is the "effective data transferred," and it turned out to obey a clean power law. The fourth step, Zhang et al. 2024, brought the question fully into the LLM era: with real billion-parameter bases and real PET methods, which axis should you actually scale, and which method should you use? The figure below traces that arc so you can see how each result built on the last.

![A timeline from 2017 to 2024 showing how transfer scaling progressed from empirical power laws in dataset size to transformer loss laws to the closed-form effective-data-transferred law and finally to the multiplicative fine-tuning law where model size beats pretraining data](/imgs/blogs/transfer-finetuning-scaling-laws-8.png)

The choice to study text-to-Python rather than some toy pair was itself a careful one. Code is close enough to natural language that transfer is large and measurable (a small distant pair would have given noisy, near-zero $D_T$), but far enough that the two distributions genuinely diverge on the parts that matter — control flow, indentation semantics, the precise grammar of a language a compiler will reject if you get it wrong. That mix is what let them resolve two *different* exponents cleanly: a proximity exponent that captures the shared structure and a generalization exponent that captures the model's capacity to exploit it. If they had chosen two distributions that were nearly identical, the proximity exponent would have swamped everything; if they had chosen two that shared almost nothing, there would have been no signal to fit.

The lesson of the lineage is the same one that runs through this entire series: scaling laws are an *empirical* discipline. The functional form (a power law) was guessed early; the constants — and the practical advice they imply — had to be measured carefully, and a measurement done in the wrong regime can mislead you completely. The "wrong regime" warning is not abstract here. It is the difference between the low-data regime, where the transfer law is beautiful and useful, and the high-data regime, where it inverts. The same series tells the cautionary version of this story for [Kaplan versus Chinchilla](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models): two careful papers disagreed by a factor of four on the compute-optimal allocation purely because of measurement bookkeeping. Transfer scaling has its own version of that trap, and it is ossification — a regime where the very same intervention (more pretraining) flips from helpful to harmful.

## 1. Effective data transferred: the one idea everything rests on

**Senior rule of thumb: before you fit anything, decide what quantity you are measuring and why it has units you can defend.** The genius of the Hernandez paper is the choice of quantity. Instead of measuring loss directly — which mixes together the effect of the model, the data, and the architecture — they measured *effective data transferred*, $D_T$: the number of additional fine-tuning tokens a from-scratch model would have needed to reach the same loss the pretrained model reached.

Concretely, the procedure is this. Fix a model size $N$ and a fine-tuning dataset size $D_F$. Fine-tune the pretrained model on $D_F$ and record its loss $L_{\text{pre}}$. Now look at the from-scratch learning curve for the same architecture — loss as a function of training-data size — and read off the data size $D_F + D_T$ at which the from-scratch model hits the same loss $L_{\text{pre}}$. The horizontal gap between "where the from-scratch curve is at $D_F$" and "where it would need to be to match the pretrained model" is $D_T$, the effective data the pretraining transferred. It is measured in tokens, the same units as $D_F$, which is exactly what makes it a clean apples-to-apples quantity.

That definition is worth slowing down on, because it is the whole article in one paragraph. The pretrained model, having seen $D_F$ fine-tuning tokens, behaves as if it had been trained from scratch on $D_F + D_T$ tokens. The pretraining did not give you a better optimizer or a lucky initialization in any mysterious sense — it gave you, in a directly measurable currency, more *data*. The figure at the top of this post is exactly this: $D_F$ and the pretrained base both flow into $D_T$, $D_T$ adds to $D_F$ to make the effective dataset, and the from-scratch baseline is what you measure against.

There is a subtlety in the measurement that distinguishes it from a sloppy "the fine-tuned model is better" claim. The comparison is done at *matched loss*, reading horizontally across the from-scratch learning curve, not vertically. If you compared vertically — "at $D_F$ tokens, the pretrained model has lower loss than the from-scratch model" — you would get a loss *difference*, which is in nats and not directly actionable: a 0.05-nat gap means nothing to a planner deciding whether to collect more data. Reading horizontally converts that loss gap into a *data* gap, which is exactly the currency the decision is in. This is the same trick the [observational scaling work](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) uses elsewhere — translate everything into the units of the decision you are actually making. The cost of the measurement is that you need the from-scratch learning curve, which is why these constants are expensive to obtain and worth borrowing carefully (the proximity ones do not transfer across domain pairs, as Case study 5 shows).

It is also worth being explicit about what $D_T$ is *not*. It is not "the number of pretraining tokens." A model pretrained on three trillion tokens does not transfer three trillion effective fine-tuning tokens; it transfers $D_T = k \cdot D_F^{\alpha} \cdot N^{\beta}$, which depends on your fine-tuning set and the model size, not on the raw pretraining count. This is the precise reason the procurement intuition "more pretraining tokens is better" is weaker than people expect — the pretraining-token count enters the downstream story only weakly (Zhang's small pretraining-data exponent), while model size enters strongly (the $N^{\beta}$ term). The effective data is a property of the *transfer*, not of the pretraining run in isolation.

### The closed-form law

When Hernandez et al. fit $D_T$ across model sizes and fine-tuning-data sizes, they found a power law in both:

$$D_T = k \cdot D_F^{\alpha} \cdot N^{\beta}$$

Here $D_T$ is effective data transferred (tokens), $D_F$ is the fine-tuning dataset size (tokens), $N$ is the number of (non-embedding) parameters in the model, and $k$, $\alpha$, $\beta$ are fitted constants. For their text-to-Python transfer, the fit was about $k \approx 1.9 \times 10^4$ (the prefactor is read from the paper's rendered tables, so treat it as approximate), $\alpha = 0.18$, and $\beta = 0.38$. The law held over more than four orders of magnitude of fine-tuning data.

Each constant means something concrete, and naming the meaning is what turns this from a curve fit into an engineering tool. The figure below pulls the three apart.

![A layered stack breaking the effective-data-transferred law into a prefactor k of about nineteen thousand, a distribution-proximity exponent of zero point one eight on the fine-tuning data, and a generalization exponent of zero point three eight on model size](/imgs/blogs/transfer-finetuning-scaling-laws-4.png)

The prefactor $k$ sets the overall scale of transfer — how many effective tokens you get at the reference point of one fine-tuning token and a unit-size model. It is large here ($\sim 1.9 \times 10^4$) because text and Python share an enormous amount of low-level structure: tokenization, syntax-like regularities, identifiers that are English words, comments that are literally English. A transfer between two genuinely unrelated domains would have a much smaller $k$.

The exponent $\alpha = 0.18$ sits on the fine-tuning data $D_F$ and measures **distribution proximity** — how close the pretraining distribution is to the target. A small $\alpha$ means that as you add more of your own data, the *fraction* of it that pretraining can substitute for shrinks: the pretrained model already "knew" the easy, shared parts of your task, so the marginal new data you collect is increasingly the part it did not know. A large $\alpha$ (close to 1) would mean the pretraining distribution is so close to the target that it keeps substituting for almost all the new data you add. The fact that $\alpha = 0.18$ is small tells you text and Python overlap a lot on the easy structure but diverge on the parts that actually make code code.

The exponent $\beta = 0.38$ sits on the model size $N$ and measures **generalization** — the capacity of the model to convert what it learned in pretraining into useful behavior on the target. A bigger model has learned richer, more transferable representations, so it transfers more effective data. Critically, $\beta$ depends on the architecture and the target task, not on the proximity of the distributions. The separation is the whole point: $\alpha$ is about *how related the two tasks are*, $\beta$ is about *how good the model is at transferring*, and the two are measured by different exponents on different variables.

The cleanest way to keep the three constants straight is to name, for each, what it depends on and what changes it. Here is the table I keep next to the formula:

| Constant | Sits on | Measures | Depends on | Big when |
|---|---|---|---|---|
| $k \approx 1.9\times 10^4$ | prefactor | overall transfer scale | source-target pair | the two domains share low-level structure |
| $\alpha = 0.18$ | $D_F$ (your data) | distribution proximity | source-target pair | pretraining distribution is close to target |
| $\beta = 0.38$ | $N$ (model size) | generalization capacity | architecture + target | the model converts pretraining into transferable features |

The two columns that matter for decisions are "depends on" and "big when." Notice that $k$ and $\alpha$ are both *pair* properties — change either domain and they change — while $\beta$ is a *model* property. That is why you can borrow $\beta$ across tasks more safely than you can borrow $k$ and $\alpha$: the generalization exponent is more about the architecture's transfer ability than about which specific pair you picked, whereas proximity is entirely about the pair. When a colleague hands you "the transfer constants," ask which pair they came from — the proximity numbers are only valid for that pair, but the model-size exponent travels better.

The other thing to read off the table is *why* $\beta > \alpha$ produces the headline result. The multiplier scales as $N^{\beta}/D_F^{1-\alpha}$. Because $\beta = 0.38$ is the exponent that *grows* the multiplier with the lever you can buy (model size) and $1-\alpha = 0.82$ is the exponent that *shrinks* it with the lever that costs labeling effort (your data), the arithmetic structurally favors buying a bigger base over collecting more data — whenever you are in the regime where the multiplier is large at all.

### Why the multiplier, not $D_T$ itself, is what you care about

The number you actually want when deciding what to do is not $D_T$ in isolation; it is the *effective-data multiplier* — how many times bigger your effective dataset is than your real one:

$$\frac{D_F + D_T}{D_F} = 1 + \frac{D_T}{D_F} = 1 + k \cdot D_F^{\alpha - 1} \cdot N^{\beta}$$

In the regime where $D_T \gg D_F$ (which is exactly the low-data regime, where transfer matters most), the "1+" is negligible and the multiplier is approximately

$$\text{multiplier} \approx k \cdot \frac{N^{\beta}}{D_F^{\,1-\alpha}}$$

Read this formula like an engineer. The multiplier *grows* with model size $N$ as $N^{\beta} = N^{0.38}$, and it *shrinks* as your own data grows, as $D_F^{-(1-\alpha)} = D_F^{-0.82}$. Both directions match intuition: a bigger base transfers more (good), and the more of your own data you have, the less of it pretraining can replace (the multiplier decays). The decay exponent $1-\alpha = 0.82$ is close to 1, which is why in the high-data limit the multiplier collapses toward 1 — pretraining stops mattering, and you are essentially training from scratch.

## 2. The effective-data multiplier curve

**Senior rule of thumb: a power law is a straight line on log-log axes, and the slope is the only thing that matters.** The single most useful picture in this whole area is the multiplier plotted against model size. Because the multiplier scales as $N^{\beta}$ at fixed $D_F$, on log-log axes it is a straight line with slope $\beta = 0.38$. That slope is small enough to undersell itself in words and large enough to dominate decisions, so it is worth drawing.

![A log-log curve showing the effective-data multiplier rising with model size, with data points climbing from a few times at ten million parameters to about one hundred times at ten billion, annotated that one decade of model size is roughly two decades of effective data](/imgs/blogs/transfer-finetuning-scaling-laws-2.png)

The curve rises steadily, and the annotation is the headline: in the text-to-Python regime, one decade of model size (a 10x increase in $N$) corresponds to roughly two decades of effective data (a 100x increase). That sounds like it violates the slope $\beta = 0.38$ — a 10x in $N$ should give $10^{0.38} \approx 2.4\times$ more $D_T$, not 100x. The resolution is that the famous "10x model = 100x data" claim in the paper is a statement about *equivalence at fixed loss*, accounting for how the from-scratch curve itself bends, not a naive reading of the $N^{0.38}$ exponent on $D_T$ alone. The practical takeaway survives either way: scaling the base model is a remarkably efficient substitute for collecting more fine-tuning data, far more efficient than the modest-looking exponent suggests when you trace it all the way through the loss comparison.

### Working the multiplier in numbers

Let us make this concrete with the actual constants. Take $k = 1.9 \times 10^4$, $\alpha = 0.18$, $\beta = 0.38$, and suppose your fine-tuning set is $D_F = 10^6$ tokens (a small, realistic specialized corpus). The effective data transferred for a model of size $N$ is $D_T = 1.9\times 10^4 \cdot (10^6)^{0.18} \cdot N^{0.38}$.

The term $(10^6)^{0.18} = 10^{1.08} \approx 12$. So $D_T \approx 1.9\times 10^4 \cdot 12 \cdot N^{0.38} \approx 2.3\times 10^5 \cdot N^{0.38}$. Now plug in model sizes:

- $N = 10^7$ (10M params): $N^{0.38} = 10^{2.66} \approx 460$, so $D_T \approx 2.3\times 10^5 \cdot 460 \approx 1.05\times 10^8$ tokens, a multiplier over $D_F = 10^6$ of about **105x**.
- $N = 10^8$ (100M params): $N^{0.38} = 10^{3.04} \approx 1100$, so $D_T \approx 2.5\times 10^8$, multiplier about **250x**.
- $N = 10^9$ (1B params): $N^{0.38} = 10^{3.42} \approx 2600$, so $D_T \approx 6.1\times 10^8$, multiplier about **610x**.

The absolute numbers depend on the (approximate) prefactor and should not be over-read, but the *ratios* are the point and they are robust: going from a 10M to a 1B base — a 100x increase in size — roughly multiplies $D_T$ by $100^{0.38} \approx 5.8\times$. The reason the headline says "10x model = 100x data" rather than "10x model = 2.4x data" is the loss-equivalence accounting; the reason a senior engineer still reaches for a bigger base in the low-data regime is that every one of those multipliers is enormous compared to the alternative of hand-labeling 100x more examples.

### A code sketch to make the law tangible

Here is the law evaluated directly, so you can see the multiplier surface for yourself and sanity-check the regime boundaries.

```python
import numpy as np

# OpenAI text->Python transfer fit (Hernandez et al. 2021, arXiv:2102.01293).
# k is read from the paper's rendered tables; treat it as approximate.
k, alpha, beta = 1.9e4, 0.18, 0.38

def effective_data_transferred(D_F, N):
    """Effective fine-tuning tokens contributed by pretraining."""
    return k * D_F**alpha * N**beta

def multiplier(D_F, N):
    """(D_F + D_T) / D_F: how many times bigger the effective dataset is."""
    D_T = effective_data_transferred(D_F, N)
    return (D_F + D_T) / D_F

for N in (1e7, 1e8, 1e9, 1e10):
    for D_F in (1e5, 1e6, 1e7, 1e8):
        m = multiplier(D_F, N)
        print(f"N={N:.0e}  D_F={D_F:.0e}  multiplier={m:8.1f}x")
```

Run it and two patterns jump out. Down any column (fixed $D_F$, increasing $N$) the multiplier climbs — bigger base, more transfer. Across any row (fixed $N$, increasing $D_F$) the multiplier *falls* steeply, because the exponent on $D_F$ in the multiplier is $\alpha - 1 = -0.82$. By the time $D_F$ reaches $10^8$ tokens, the multiplier for a 1B model has dropped from hundreds toward single digits — you are leaving the regime where the law even applies, which is the subject of the next section.

## 3. The low-data regime and ossification: where the law breaks

**Senior rule of thumb: every scaling law has a domain of validity, and the expensive mistakes happen when you extrapolate past it.** The transfer law is gorgeous in the low-data regime and inverts in the high-data regime, and the boundary is not a footnote — it is the single most important caveat in this entire area. The figure below contrasts the two regimes side by side.

![A before-and-after comparison showing the low-data regime where transferred data dominates and transfer wins, versus the high-data regime where the frozen pretrained prior ossifies and can underperform training from scratch](/imgs/blogs/transfer-finetuning-scaling-laws-3.png)

The **low-data regime** is where the Hernandez law lives. Operationally, it is when your fine-tuning data $D_F$ is at most about 10% of the data you would need to reach 99% of the model's infinite-data performance on the target task. In this regime, $D_T \gg D_F$ — the effective data is dominated by what pretraining transferred — and the multiplier is large. Here the law holds cleanly over four-plus orders of magnitude, and every conclusion in this post applies: bigger base beats more data, the multiplier formula is accurate, transfer is a large and reliable win.

The **high-data regime** is the danger zone. As $D_F$ grows past the threshold, two things happen at once. First, the multiplier mechanically collapses toward 1 (the $D_F^{-0.82}$ decay), so pretraining's contribution becomes negligible — you have enough of your own data that the model no longer needs the borrowed structure. Second, and more insidiously, the pretrained weights can actively *hurt*. The model arrives at fine-tuning with a strong prior shaped by the pretraining distribution, and when you have a huge in-domain dataset, that prior is a constraint rather than a gift: it slows adaptation, biases the solution toward the source distribution, and the fine-tuned model can end up with *higher* loss than an identical architecture trained from scratch on the same large target set. Hernandez et al. named this **ossification** — the pretrained initialization has rigidified into a shape the target data cannot fully reshape.

### Why ossification happens

The intuition is worth stating precisely because it is the kind of thing that sounds paradoxical until you see the mechanism. Pretraining is a form of inductive bias: it places the model's parameters in a region of weight space that encodes the pretraining distribution's structure. When your target task is data-poor, that bias is overwhelmingly helpful — it fills in everything your few examples cannot teach. When your target task is data-rich, the same bias is a liability, because your data could have learned the right structure on its own, and the pretraining prior now pulls the optimizer toward a different, slightly-wrong basin. A from-scratch model has no such prior; it goes wherever the abundant data sends it. So in the high-data limit, from-scratch can win.

This is the precise sense in which "more pretraining is better" is false. It is true in the low-data regime and false in the high-data regime, and the crossover is governed by how much target data you have relative to the data-complexity of the task. The practical reading is blunt: if you have a genuinely large, clean, in-domain dataset, do not assume a pretrained base is helping — measure it against from-scratch, or at least against full fine-tuning with a high learning rate that can overwrite the prior.

### How to tell which regime you are in

You will not usually run the full from-scratch baseline (that is the whole point of fine-tuning). But you can estimate the regime cheaply. Fit a quick learning curve: fine-tune on 25%, 50%, and 100% of your data and look at how fast loss is still dropping. If loss is still falling steeply with more data, you are data-limited — squarely in the low-data regime, where a bigger base will help more than more data per the multiplier. If loss has visibly flattened — adding the last 50% of your data barely moved it — you are approaching the high-data regime, where the marginal value of both more data and more pretraining is small, and ossification is a live risk. The flattening of that curve is your regime indicator, and it costs three short runs to measure.

To make the indicator quantitative, fit the three points to the data-side power law $L(D_F) \approx L_\infty + c \cdot D_F^{-\beta}$ and read off the implied $L_\infty$ (the asymptote). If your 100%-data loss is, say, within 1% of the fitted $L_\infty$, you are essentially at the data ceiling — the high-data regime — and more data or more pretraining will buy almost nothing. If your 100%-data loss is still 10% or more above $L_\infty$, you are deep in the low-data regime and the multiplier story dominates. The ratio "current loss minus $L_\infty$, over current loss minus the irreducible floor $E$" is a clean single number for how much headroom your own data still has. The whole measurement is three fine-tuning runs and a two-parameter fit, which is cheap insurance against the most expensive mistake in this area: pouring months of labeling into a task that flattened out at 40% of the data you already had.

The asymmetry of the ossification risk is what makes the cheap check worth running every time. If you misjudge and think you are in the high-data regime when you are actually data-limited, the cost is modest — you under-invest in data and leave some loss on the table. But if you misjudge the other way and think you are data-limited when you have actually saturated, you can spend a six-month labeling contract for a fraction of a percent of loss, *and* you may be fighting ossification the whole time without knowing it. Asymmetric downside means you should run the three-point curve before any large data-acquisition decision, not after.

> Ossification is the reason "just fine-tune a bigger model on more data" is not a universal recipe. The bigger model helps when you are starved; the more data helps when you are starved. When you are no longer starved, the pretrained prior can become the thing holding you back.

## 4. The two laws side by side: additive data vs multiplicative loss

**Senior rule of thumb: when two papers fit the same phenomenon with different functional forms, the forms usually answer different questions — figure out which question you have.** Hernandez 2021 and Zhang 2024 are both "scaling laws for transfer/fine-tuning," but they are not competing fits of the same curve. They answer complementary questions, and a senior practitioner keeps both in their head. The figure below puts them side by side.

![A before-and-after comparison contrasting the Hernandez additive effective-data view, which adds transferred data to the fine-tuning set, against the Zhang multiplicative loss law, which fits fine-tuning loss as a product of model-size and data power laws](/imgs/blogs/transfer-finetuning-scaling-laws-5.png)

The Hernandez law is **additive in data**: it says the model behaves as if trained on $D_F + D_T$ tokens, and it answers the question *how much data did pretraining save me?* That is the right question when you are budgeting data-collection effort or comparing "spend on a bigger base" against "spend on more labels." Its output is in tokens, which is the currency of that decision.

The Zhang law is **multiplicative in loss**: it fits the fine-tuning loss directly as a product of power laws, and it answers the question *which axis should I scale to lower my loss?* That is the right question when you have already committed to fine-tuning and are choosing where to spend the next increment of compute — a bigger base, more pretraining, or more PET parameters. Its output is loss, which is the currency of that decision.

### The Zhang multiplicative joint law

Zhang et al. 2024 ran a large systematic study of LLM fine-tuning across base-model sizes, pretraining-data sizes, and PET configurations, on bilingual machine translation and multilingual summarization. They found that the fine-tuning loss is well described by a multiplicative joint law:

$$\hat{L}(X, D_f) = A \cdot \frac{1}{X^{\alpha}} \cdot \frac{1}{D_f^{\beta}} + E$$

Here $\hat{L}$ is the predicted fine-tuning loss, $D_f$ is the fine-tuning data size, $E$ is an irreducible loss floor (the same role $E$ plays in the [Chinchilla loss law](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling)), and $X$ is *one* scaling factor at a time — either the LLM model size, or the pretraining-data size, or the PET-parameter size (LoRA rank or prompt length). $A$, $\alpha$, $\beta$ are fitted per choice of $X$. The "one factor at a time" structure is deliberate: they isolate each axis so they can compare exponents and say cleanly which one buys the most loss reduction.

The multiplicative form (a product of $X^{-\alpha}$ and $D_f^{-\beta}$, rather than a sum like Chinchilla's $A/N^{\alpha} + B/D^{\beta}$) is itself a finding. It says the two factors *interact*: the benefit of scaling $X$ is larger when you also have more fine-tuning data, and vice versa. A bigger base helps more when you have data to exploit it, and more data helps more when you have a base capable of using it. That is a different geometry from the additive Chinchilla surface, and it matters for allocation.

It is worth being precise about why additive and multiplicative are genuinely different surfaces, because the distinction drives different advice. In the additive Chinchilla form $L = E + A/N^{\alpha} + B/D^{\beta}$, the two penalty terms are *independent*: at large $N$ the model term has already vanished, so adding more data still helps via the data term regardless of how big the model is. The terms do not gate each other. In the multiplicative form $L = E + A \cdot X^{-\alpha} \cdot D_f^{-\beta}$, the terms *gate* each other: when one factor is small, the whole reducible loss $A \cdot X^{-\alpha} \cdot D_f^{-\beta}$ is small, so the marginal benefit of improving the other factor is also small. The practical consequence is the one Case study 8 dramatizes: if you scale model size while holding fine-tuning data at a tiny value, the product is dominated by the tiny-$D_f$ factor and the size gain is muted — you have to move both to see the full effect. Pretraining, in the multiplicative world, is not a free additive head start; it is a factor that only pays off in proportion to how much fine-tuning data you have to multiply it against.

There is a tension here that is worth surfacing rather than hiding. Hernandez's effective-data picture is *additive* — the model behaves as if trained on $D_F + D_T$ tokens — while Zhang's loss picture is *multiplicative*. These are not contradictory; they describe different things (data-equivalence versus loss). But they do imply you should not mechanically port additive intuition from one to the other. The safe synthesis is: use Hernandez's additive $D_T$ to answer "how much data did pretraining save me" (a budgeting question), and use Zhang's multiplicative law to answer "which axis lowers my loss the most right now" (an allocation question). Both agree on the punchline — model size is the high-leverage axis — but they get there through different functional forms, and mixing the forms is how planners talk themselves into the wrong experiment.

### The headline comparison: size beats pretraining data

The single most decision-relevant result in Zhang et al. is the comparison of exponents across the three choices of $X$. **Fine-tuning benefits more from LLM model size than from pretraining-data size.** In their fits, the exponent $\alpha$ on the model-size axis is larger than the exponent on the pretraining-data axis — meaning that, for the same fractional increase, scaling the base model lowers fine-tuning loss more than having pretrained that base on more tokens.

This dovetails exactly with the Hernandez result. Hernandez said the effective-data multiplier scales as $N^{0.38}$ — strongly with model size. Zhang said fine-tuning loss falls faster along the model-size axis than along the pretraining-data axis. Two different experiments, two different functional forms, one consistent conclusion: **for a downstream task, the size of your base matters more than how much it was pretrained on.** That has a direct procurement implication. If you are choosing between two open bases — a smaller model pretrained on more tokens, versus a larger model pretrained on fewer — the larger model is usually the better fine-tuning starting point, especially in the low-data regime where you live most of the time.

## 5. PET-parameter scaling is largely ineffective

**Senior rule of thumb: a knob that does not move the loss is not a capacity knob, no matter what the diagram says.** Parameter-efficient fine-tuning (PET) — LoRA, prefix tuning, prompt tuning, adapters — gives you a parameter count you can dial: LoRA rank, prompt length, adapter width. It is tempting to treat that count as a scaling axis, the way model size is a scaling axis: crank it up, get lower loss. The Zhang study tested this directly and the answer is sobering. **Scaling PET parameters is largely ineffective** — the exponent $\alpha$ on the PET-parameter axis in the multiplicative law is near zero, so the loss barely moves as you increase LoRA rank or prompt length. The figure below contrasts the two curves.

![A two-curve plot where fine-tuning loss falls steeply as base-model size is scaled but stays nearly flat as LoRA rank or prompt length is scaled, showing the PET-parameter exponent is near zero](/imgs/blogs/transfer-finetuning-scaling-laws-7.png)

The steep curve is loss versus model size — a real power law with a real exponent, the loss dropping decade after decade as you scale the base. The nearly flat dashed curve is loss versus PET-parameter count — it sags a little at first and then plateaus, because past a small rank the extra parameters are not the binding constraint. This is one of the most practically important and most ignored results in fine-tuning. Teams routinely sweep LoRA rank from 8 to 16 to 64 to 256 expecting a scaling-law payoff, and what they find matches the flat curve: a tiny improvement going from rank 4 to rank 16, then essentially nothing.

### Why PET-parameter scaling stalls

The mechanism is intuitive once you see it. The job of a PET module is to *adapt* a fixed base, not to *add capacity* in the way more base parameters do. The base model already contains the representational machinery; LoRA's low-rank update only needs enough degrees of freedom to nudge that machinery toward the target task. Beyond a modest rank, you have enough degrees of freedom — the remaining error is not coming from "not enough adapter parameters," it is coming from the base model's capacity and from how much target data you have. Adding more adapter rank is adding more keys to a lock that is already open.

This reframes how you should think about LoRA rank: it is a *threshold* knob, not a *scaling* knob. You want enough rank to clear the threshold (often surprisingly low — rank 8 to 32 is plenty for many tasks) and no more, because extra rank costs memory and training time for no loss reduction. If your LoRA fine-tune is underperforming, the fix is almost never "more rank." It is a bigger base, more data, or switching to full fine-tuning — the axes that actually have exponents.

The mechanics make the threshold behavior concrete. LoRA replaces a weight update $\Delta W \in \mathbb{R}^{d \times d}$ with a low-rank factorization $\Delta W = BA$, where $A \in \mathbb{R}^{r \times d}$ and $B \in \mathbb{R}^{d \times r}$ and $r \ll d$ is the rank. The number of trainable parameters is $2dr$, linear in $r$. The reason loss flattens in $r$ is that the *task-specific update* the fine-tune needs to express has low effective rank to begin with — the adaptation is a small rotation of an already-capable representation, not a from-scratch learning of new features. Once $r$ exceeds that intrinsic rank, the extra dimensions of $A$ and $B$ are slack: the optimizer leaves them near zero, and loss does not move. This is why the right way to set rank is empirically and once: sweep $r \in \{4, 8, 16, 32\}$ on a held-out set, take the smallest $r$ at the plateau, and stop.

```python
# A LoRA config that respects the "rank is a threshold" finding.
# r=16 clears the threshold for most tasks; alpha scales the update.
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,                       # threshold knob: smallest r at the plateau
    lora_alpha=32,             # scaling; alpha/r = 2.0 is a common ratio
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
# Do NOT chase loss by bumping r to 256 — the exponent on PET params is ~0.
# If r=16 underperforms, change the *base model size*, not the rank.
model = get_peft_model(base_model, config)
model.print_trainable_parameters()   # e.g. ~0.1% of base params trainable
```

The comment in that snippet is the whole section in one line: if rank 16 is not enough, the lever is the base model size, not the rank. Teams that internalize this stop running rank sweeps past 32 and redirect that compute to the axes with real exponents.

### What this means for the "which method" decision

The combination of these findings — size beats pretraining data, PET-rank scaling stalls — sets up the practical method-selection question. PET methods are cheap and they work well when data is scarce, but you cannot scale your way out of their ceiling by adding rank. Full fine-tuning has more capacity to exploit but needs more data to avoid overfitting and can ossify. LoRA sits in between. So the right method is not a fixed choice; it is a function of how much target data you have, which is exactly what the next section quantifies.

## 6. Choosing a method by how much data you have

**Senior rule of thumb: the best fine-tuning method is the one matched to your data size, and the matching is not your intuition's job — it is a lookup table.** Zhang et al.'s most actionable contribution is the data-size-dependent method guidance. The best method is task- and data-dependent, but the dependence is regular enough to draw as a grid. The figure below is that grid: data size on the rows, method on the columns, and the cells tell you what happens.

![A four-by-four decision grid with data size on the rows and fine-tuning method on the columns, showing prompt and PET methods winning at tiny data, LoRA in the middle, full fine-tuning at large data, and a column noting that scaling base size helps most when data is small](/imgs/blogs/transfer-finetuning-scaling-laws-6.png)

Read the grid top to bottom. With **tiny data**, prompt-based and PET methods win: they have few free parameters, so they do not overfit, and full fine-tuning would just memorize your handful of examples. With **small data**, PET methods are strong and cheap, and LoRA becomes competitive — this is a good default starting point. With **medium data**, LoRA hits the best balance, and full fine-tuning becomes viable if your data is genuinely in-domain. With **large data**, full fine-tuning takes over, because you finally have enough signal to exploit its capacity without overfitting — though this is exactly where ossification risk appears, so a high-enough learning rate to overwrite the prior matters.

The right-most column, "scale N?", carries the cross-cutting message: scaling the base model helps *most* when your data is small (the multiplier is largest there), and helps less as data grows. At the very bottom — large data — scaling the base gives only a small gain and raises ossification risk, the regime where you might even consider from-scratch.

### The decision procedure, written out

Putting it together, here is the procedure a senior engineer actually runs. It is short because the laws make it short.

1. **Estimate your regime.** Fit a three-point learning curve (25/50/100% of data). Still dropping steeply: low-data regime. Flattened: high-data regime.
2. **In the low-data regime, prefer a bigger base over more data.** The multiplier scales as $N^{\beta}$, and size beats pretraining data per Zhang. If you can pick a larger open base, do it before you spend weeks labeling.
3. **Pick the method by data size,** using the grid: tiny -> prompt/PET; small -> PET or LoRA; medium -> LoRA; large -> full fine-tuning.
4. **Set LoRA rank to clear the threshold, not to scale.** Rank 8-32 is usually enough; do not sweep to 256 expecting a payoff.
5. **In the high-data regime, watch for ossification.** Use a high learning rate, or benchmark full fine-tuning against a from-scratch (or heavily re-trained) baseline if the data is large and clean.
6. **Re-check the regime after each data acquisition.** Adding data moves you down the rows; what was optimal at small data is not optimal at large.

### A worked allocation example

Suppose you are building a domain-specific code assistant and you have 5,000 labeled examples (roughly $D_F \approx 2\times 10^6$ tokens). You are choosing between two open bases: a 1B model and a 7B model, and between LoRA and full fine-tuning.

First, the regime. Five thousand examples for a specialized code task is small — you are almost certainly in the low-data regime, so the transfer law applies and the multiplier is large. Second, the base. By the multiplier, going from 1B to 7B multiplies effective data by $(7)^{0.38} \approx 2.1\times$; combined with Zhang's "size beats pretraining data," the 7B base is the better starting point even if the 1B was pretrained on more tokens. Third, the method. With $D_F \approx 2\times 10^6$ tokens you are in the small-to-medium band: start with LoRA (rank 16-32), which is cheap and resists overfitting; only move to full fine-tuning if you later acquire substantially more in-domain data. Fourth, the rank. Do not sweep rank past ~32 — the flat curve says it will not help. The whole decision took four lookups and zero from-scratch runs, which is the entire value of having the laws.

### A second-order gotcha: the regime moves as you collect

The procedure above has a subtle trap that catches careful teams. The optimal method is a function of where you sit on the data-size axis, and *that position moves* as you acquire data. A method that was optimal at 5,000 examples (LoRA, rank 16) is not optimal at 500,000 examples (where full fine-tuning may win, and ossification becomes a concern). The mistake is to lock in the method early — "we are a LoRA shop" — and never revisit it as the corpus grows. The fix is to make the method a *reviewed* decision tied to a data-size milestone: every time the corpus grows by, say, 5x, re-fit the three-point learning curve, re-check the regime, and re-run the grid lookup. The laws are not a one-time configuration; they are a control loop. Teams that treat them as configuration ship a model that was optimal for the data they had a year ago.

## 7. End-to-end worked examples

To cement the mechanics, here are three fully worked scenarios that exercise the formulas. All use the text-to-Python constants ($k = 1.9\times 10^4$, $\alpha = 0.18$, $\beta = 0.38$) for the Hernandez side; the exact numbers are illustrative because the prefactor is approximate, but the *ratios and decisions* are the durable part.

### Example A: how much data did pretraining save me?

You fine-tuned a 1B-parameter base on $D_F = 5\times 10^6$ tokens of target data. How much data did the pretraining effectively contribute?

$D_T = k \cdot D_F^{\alpha} \cdot N^{\beta} = 1.9\times 10^4 \cdot (5\times 10^6)^{0.18} \cdot (10^9)^{0.38}$.

Compute the pieces: $(5\times 10^6)^{0.18}$ — take logs, $\log_{10}(5\times 10^6) = 6.70$, times $0.18 = 1.21$, so this term is $10^{1.21} \approx 16$. And $(10^9)^{0.38} = 10^{3.42} \approx 2630$. So $D_T \approx 1.9\times 10^4 \cdot 16 \cdot 2630 \approx 8.0\times 10^8$ tokens. The effective dataset is $D_F + D_T \approx 5\times 10^6 + 8.0\times 10^8 \approx 8.05\times 10^8$, a multiplier of about **161x**. In plain terms: your 5 million fine-tuning tokens behaved like roughly 800 million. That is the number to put in the slide when someone asks whether pretraining was worth it.

### Example B: bigger base or more labels?

Same task, $D_F = 5\times 10^6$, 1B base. You can either (a) collect 10x more data (to $5\times 10^7$ tokens) or (b) move to a 10x bigger base (10B). Which lowers effective-data-equivalent more?

Option (a), more data: the new effective dataset is $D_F' + D_T'$ with $D_F' = 5\times 10^7$ and $D_T' = 1.9\times 10^4 \cdot (5\times 10^7)^{0.18} \cdot (10^9)^{0.38}$. The data term grows: $(5\times 10^7)^{0.18} = 10^{7.70\times 0.18} = 10^{1.39} \approx 24$, so $D_T' \approx 1.9\times 10^4 \cdot 24 \cdot 2630 \approx 1.2\times 10^9$. Effective dataset $\approx 5\times 10^7 + 1.2\times 10^9 \approx 1.25\times 10^9$ tokens.

Option (b), bigger base: $D_F = 5\times 10^6$ stays, but $N = 10^{10}$ so $N^{0.38} = 10^{3.8} \approx 6310$. Then $D_T'' \approx 1.9\times 10^4 \cdot 16 \cdot 6310 \approx 1.9\times 10^9$. Effective dataset $\approx 5\times 10^6 + 1.9\times 10^9 \approx 1.9\times 10^9$ tokens.

The bigger base wins ($1.9\times 10^9$ vs $1.25\times 10^9$ effective tokens) — and it wins *without you labeling a single new example*. This is the "10x model beats 10x data" intuition made arithmetic, and it is the single most counterintuitive, money-saving consequence of the transfer law for any data-starved team.

### Example C: when does the law stop applying?

Same constants, 1B base. At what $D_F$ does the multiplier fall to, say, 5x — the rough edge of "transfer still clearly worth it"? Set the multiplier $\approx k N^{\beta} / D_F^{1-\alpha} = 5$. With $kN^{\beta} = 1.9\times 10^4 \cdot 2630 \approx 5.0\times 10^7$, we need $D_F^{0.82} = 5.0\times 10^7 / 5 = 1.0\times 10^7$, so $D_F = (10^7)^{1/0.82} = 10^{8.54} \approx 3.4\times 10^8$ tokens. So for a 1B base on this kind of transfer, once your target data exceeds roughly a few hundred million tokens, the multiplier has decayed toward single digits — you are leaving the low-data regime and approaching the zone where ossification is a real risk and from-scratch becomes a serious option. That threshold is exactly the regime boundary from Section 3, now computed rather than asserted.

### Example D: how the regime boundary moves with model size

A natural follow-up: does a bigger base push the regime boundary further out, letting you stay in the favorable low-data regime longer? Repeat Example C for a 10B base. Now $kN^{\beta} = 1.9\times 10^4 \cdot 6310 \approx 1.2\times 10^8$ (using $N^{0.38} = 6310$ for $10^{10}$). Setting the multiplier to 5 again: $D_F^{0.82} = 1.2\times 10^8 / 5 = 2.4\times 10^7$, so $D_F = (2.4\times 10^7)^{1/0.82} = 10^{7.38/0.82} = 10^{9.0} \approx 1.0\times 10^9$ tokens. So a 10B base keeps the multiplier above 5x until about a billion target tokens — roughly 3x further out than the 1B base's $3.4\times 10^8$. The lesson is double-edged: a bigger base both *raises* the multiplier at every point and *extends* the regime in which transfer is clearly worth it. That is one more reason the model-size axis is the high-leverage one. But note the boundary still exists — even a 10B base hits the ossification zone if you pour a billion-plus in-domain tokens into it, so "just use a bigger base" does not abolish the high-data regime, it only postpones it.

## 8. Where transfer scaling sits in the broader budget

**Senior rule of thumb: no single scaling law is a complete budget; they compose, and the composition is where the money is.** Transfer scaling does not live alone. It is one factor in a chain of decisions that starts before you ever fine-tune, and treating it in isolation leads to locally-optimal, globally-wasteful choices. The table below maps where each law in this series enters the lifecycle.

| Decision | Governing law | What it tells you |
|---|---|---|
| How big to make the base, how much to pretrain | [Chinchilla compute-optimal](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) | split pretraining budget about evenly between size and tokens (~20 tokens/param) |
| What to do when pretraining data runs out | [data-constrained scaling](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) | repeat data up to ~4 epochs cheaply; diminishing returns past that |
| How much the base helps a downstream task | transfer scaling (this post) | effective-data multiplier $\sim N^{\beta}/D_F^{1-\alpha}$, large in the low-data regime |
| Which fine-tuning axis to scale | Zhang multiplicative law (this post) | model size beats pretraining data; PET-rank scaling stalls |
| When RL fine-tuning against a reward goes wrong | [reward-model overoptimization](/blog/machine-learning/scaling-laws/reward-model-overoptimization-scaling) | gold reward peaks then falls as you push KL distance — Goodhart |

Read top to bottom, this is the life of a model: Chinchilla sets the base's shape, data-constrained scaling handles the case where pretraining tokens are scarce, transfer scaling tells you what that base buys you downstream, the Zhang law tells you how to adapt it, and reward-model overoptimization covers the RL-fine-tuning failure mode. The reason to hold all five at once is that they trade against each other. Spending more on the base (Chinchilla axis) raises the transfer multiplier (this post) — so a bigger pretraining budget pays off twice, once in base quality and again in downstream transfer. Conversely, skimping on the base to save money shows up later as a smaller multiplier and more fine-tuning data you have to collect.

### The ossification-Goodhart parallel

There is a deep structural rhyme between this post's central failure mode and the one in the [reward-model overoptimization](/blog/machine-learning/scaling-laws/reward-model-overoptimization-scaling) post worth drawing out, because seeing it makes both more memorable. In transfer, **more pretraining is monotone-good only up to a point**; past the regime boundary, pushing harder on the pretrained prior (ossification) starts to hurt the downstream task. In RLHF, **more optimization against the reward model is monotone-good only up to a point**; past a KL threshold, pushing harder against the proxy reward (Goodhart) starts to hurt the true objective. Both are cases where an intervention that is helpful in moderation becomes harmful past a measurable boundary, and in both cases the boundary is a function of how much "ground truth" you have — target data in one case, reward-model capacity and preference data in the other. The general lesson, which is the soul of this whole series, is that scaling curves bend, and the expensive mistakes happen when you assume the helpful part of the curve continues forever.

### Transfer scaling in the instruction-tuning era

A reasonable question is whether these laws, measured on supervised fine-tuning of pre-LLM-era models and on translation and summarization, still apply to the modern instruction-tuning and preference-optimization pipeline. The honest answer is: the *qualitative* conclusions transfer, the *exact constants* do not, and you should re-measure where you can. Instruction tuning is, mechanically, supervised fine-tuning on a target distribution (instruction-response pairs), so the additive effective-data picture and the model-size-beats-pretraining-data conclusion both carry over in spirit — a bigger base instruction-tunes to a better assistant with less instruction data, which matches both papers. The PET-rank-stalls finding is directly relevant: the LoRA-everything culture of efficient instruction tuning should not expect rank to be a scaling axis. Where it gets murkier is preference optimization (RLHF, DPO), which is not straightforward supervised fine-tuning and brings its own overoptimization scaling story; there, the relevant law is the reward-model one, not the transfer one. The clean mental model is: SFT-shaped adaptation obeys transfer scaling; reward-shaped adaptation obeys overoptimization scaling; and the two stack in a modern pipeline.

## Case studies from the literature and the field

These are composite scenarios drawn from the patterns the two papers report and from the way these laws play out in practice. Each follows the same shape: the symptom, the wrong first hypothesis, the actual root cause, the fix, the lesson.

### 1. The LoRA rank that would not scale

The symptom: a team fine-tuning a 7B base for legal-document summarization swept LoRA rank from 8 to 16 to 64 to 256, expecting steadily lower validation loss, and saw essentially no improvement past rank 16. The wrong first hypothesis was that rank 256 was "still too small" and they should go to 512. The actual root cause is the flat PET-parameter curve from Section 5: the exponent on PET parameters is near zero, so rank is a threshold knob, not a scaling knob. The fix was to fix rank at 16, free the memory, and spend the saved compute on a larger base and a higher-quality data subset. The lesson: when a knob does not move the loss across two doublings, stop turning it — the binding constraint is elsewhere.

### 2. The from-scratch model that won

The symptom: a team with a very large, clean, in-domain corpus (hundreds of millions of tokens of a single specialized format) found that their carefully fine-tuned general-purpose base was *beaten* by a smaller model a colleague trained from scratch on just the corpus. The wrong first hypothesis was that the from-scratch model "got lucky" or had a bug in the eval. The actual root cause is ossification: with abundant in-domain data, the pretraining prior was a constraint, not a gift, and the from-scratch model adapted more freely. The fix was to either train from scratch (since the data supported it) or fine-tune with a much higher learning rate to overwrite the prior. The lesson: "always start from a pretrained base" is a low-data-regime rule, and they were in the high-data regime.

### 3. The procurement decision

The symptom: a team had to choose between two open checkpoints for a downstream task — a 3B model pretrained on 3T tokens, and a 7B model pretrained on 1T tokens. They defaulted to the 3B because it had "seen more data." The wrong hypothesis was that pretraining-token count is the dominant predictor of fine-tuning quality. The actual root cause: Zhang's finding that fine-tuning benefits more from model size than from pretraining data, plus Hernandez's $N^{0.38}$ multiplier. The fix was to switch to the 7B base, which fine-tuned to lower loss despite fewer pretraining tokens. The lesson: when choosing a base for fine-tuning, weight size over pretraining-token count, especially in the low-data regime.

### 4. The 100x-data project that should have been a bigger-base project

The symptom: a product team budgeted six months and a large annotation contract to 100x their fine-tuning dataset, chasing a target loss. The wrong hypothesis was that more labels were the only lever. The actual root cause: in their low-data regime, the multiplier scaling as $N^{\beta}$ meant a 10x bigger base would have delivered comparable effective-data gains for the cost of a larger GPU, not a six-month labeling effort (Example B above is exactly this arithmetic). The fix was to move to a bigger base first and shrink the annotation contract to a high-quality core. The lesson: price the "bigger base" option in effective-data terms before committing to a labeling marathon.

### 5. The transfer that barely transferred

The symptom: a team tried to fine-tune a code model onto a very different target — say, a low-resource natural language with little shared structure — and found pretraining helped far less than the text-to-Python numbers led them to expect. The wrong hypothesis was that the transfer law's constants are universal. The actual root cause: $k$ and $\alpha$ are *distribution-proximity* quantities. Text and Python share enormous low-level structure (hence large $k$); two genuinely distant distributions have a small $k$ and the multiplier is modest. The fix was to measure the transfer for their actual pair rather than borrowing the Python constants. The lesson: $\alpha$ and $k$ are properties of the source-target pair, not constants of nature; re-measure them for your domains.

### 6. The learning curve that flattened early

The symptom: a team kept adding data to a fine-tune and loss stopped improving after the first third of the dataset, but they kept collecting. The wrong hypothesis was that they needed even more data to "break through." The actual root cause: the flattened learning curve is the regime indicator from Section 3 — they had entered the high-data regime, where both more data and more pretraining have small marginal value. The fix was to stop collecting, and instead try a higher learning rate (to escape ossification) or a different architecture. The lesson: a flattening learning curve is a signal to change levers, not to pull harder on the same one.

### 7. The prompt-tuning win at tiny scale

The symptom: with only a few hundred examples, a team's full fine-tune overfit badly — great train loss, terrible validation. The wrong hypothesis was that they needed regularization tricks. The actual root cause: at tiny data, full fine-tuning has far too many free parameters; the method grid says prompt/PET wins here precisely because it has few. The fix was to switch to prompt tuning or a low-rank LoRA, which fit the few examples without memorizing them. The lesson: method should track data size; at tiny data, fewer trainable parameters is a feature.

### 8. The multiplicative interaction that surprised the planner

The symptom: a team scaled their base model and saw a smaller-than-expected loss drop, concluding "size does not help here." The wrong hypothesis was that the model-size benefit is independent of data. The actual root cause: Zhang's law is *multiplicative* — the benefit of scaling $X$ is larger when you also have more fine-tuning data. They had scaled size while holding data at a tiny level, so the interaction term muted the gain. The fix was to scale size *and* acquire a modest amount more data, unlocking the interaction. The lesson: in a multiplicative law, axes amplify each other; do not test one in isolation and conclude it is dead.

### 9. The "pretraining is free" assumption that hid a regression

The symptom: a team upgraded to a base pretrained on more tokens and saw downstream quality *drop* on one task. The wrong hypothesis was a training bug. The actual root cause: more pretraining shifted the prior toward the (broader) pretraining distribution, and for that particular data-rich downstream task it nudged the model toward ossification. The fix was to keep the older base for that task and use the new one only for data-poor tasks. The lesson: "more pretraining" is monotone only in the low-data regime; in the high-data regime it can regress a specific task.

### 10. The cross-lingual transfer that needed proximity, not size

The symptom: a team transferring an English-heavy base to a low-resource target language found that doubling the base size helped far less than expected, while a smaller base that had merely *seen* a little of the target language during pretraining did much better. The wrong hypothesis was that model size is always the dominant axis. The actual root cause: size moves the *generalization* exponent $\beta$, but the binding constraint here was *proximity* ($\alpha$ and $k$) — the English-heavy base had almost no overlap with the target distribution, so there was little for a bigger model to transfer. The fix was to choose a base whose pretraining mixture actually included the target language, even at the cost of size. The lesson: when the source-target distance is the bottleneck, fix proximity first; size cannot manufacture transfer of structure the base never saw.

### 11. The eval that measured the wrong axis

The symptom: a team benchmarked "fine-tuning method" by holding model size and data fixed and sweeping methods, concluding all methods were roughly equal. The wrong hypothesis was that method is a first-order axis. The actual root cause: at the single data size they tested, all reasonable methods sit in the same band of the grid — the method differences only become large when you *vary* the data size, because the grid's columns swap winners as you move down the rows. The fix was to re-run the sweep at three data sizes (tiny, medium, large) and watch the winner change. The lesson: method is a *conditional* choice, conditional on data size; an eval that fixes data size cannot see the conditioning.

### 12. The data-quality confound that looked like ossification

The symptom: a team's fine-tune on a large in-domain corpus underperformed a from-scratch model and they diagnosed ossification, planning to abandon the pretrained base. The wrong hypothesis was that the pretrained prior was the problem. The actual root cause turned out to be partly data quality: their large corpus was noisy, and the from-scratch model — trained only on it — overfit the noise in a way that happened to score well on a flawed eval, while the pretrained base's prior regularized against exactly that noise. The fix was to clean the corpus and re-measure; with clean data, the pretrained base was competitive. The lesson: ossification is real, but "from-scratch beat my fine-tune" has more than one cause — rule out a noisy eval and a noisy corpus (the [data-quality axis](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws)) before you blame the prior.

## What this means in practice

Strip away the formulas and the practical doctrine is short, and it overturns three habits worth naming.

The first habit to drop is treating pretraining as a fixed, free head start. It is a *multiplier*, and you can compute it. In the low-data regime — where almost every specialized task lives — that multiplier is enormous and grows with model size as $N^{0.38}$. The first move when your fine-tune underperforms should be to price the "bigger base" option in effective-data terms, because $10\times$ model can beat $10\times$ data for the cost of a larger GPU instead of a labeling contract.

The second habit to drop is treating LoRA rank (or prompt length, or adapter width) as a scaling axis. It is a threshold. Set it high enough to clear the threshold — usually rank 8 to 32 — and stop. The exponent on PET parameters is near zero; sweeping rank to 256 buys memory cost and nothing else. The axes that have real exponents are base-model size, fine-tuning data, and the choice of method.

The third habit to drop is "always start from a pretrained base, the bigger the pretraining the better." That is true in the low-data regime and false in the high-data regime, where the pretrained prior ossifies and can lose to from-scratch. Measure your regime with a three-point learning curve; if loss has flattened, you are in the zone where more data and more pretraining both have small value and ossification is a live risk.

Then the method follows the data size, by lookup, not by intuition: tiny data favors prompt/PET, small-to-medium favors LoRA, large in-domain data favors full fine-tuning. And because Zhang's law is multiplicative, the axes amplify each other — scale size and data together when you can, and never test one axis in isolation and pronounce it dead.

Finally, fit all of this into the broader budget picture. The decision of how big to make the base and how much to pretrain it is a [Chinchilla compute-optimal](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) decision; the question of what to do when the *pretraining* data itself runs out is [data-constrained scaling](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws); and if your fine-tuning is reinforcement learning against a reward model rather than supervised, the relevant failure mode is [reward-model overoptimization](/blog/machine-learning/scaling-laws/reward-model-overoptimization-scaling), where pushing too hard against a proxy reward turns helpful into harmful — the alignment-era analogue of ossification. Transfer scaling is the bridge between "how I trained the base" and "how I adapt it," and treating it as a quantity you compute rather than a virtue you assume is what separates a budgeted fine-tuning program from a series of expensive shrugs.

### Reach for transfer scaling when

- **Your target task is data-starved.** This is the regime where the multiplier is large and every conclusion in this post applies — and it is the normal condition for any specialized task.
- **You are choosing between bases.** The $N^{\beta}$ multiplier plus Zhang's "size beats pretraining data" gives a principled answer: weight model size over pretraining-token count.
- **You are budgeting "bigger base versus more labels."** Price both in effective-data tokens (Example B) before committing to a labeling effort; the bigger base often wins for the cost of a GPU.
- **You want to justify a pretraining spend.** The effective-data multiplier is the number that quantifies what pretraining bought a downstream team — a concrete ROI figure, not a hand-wave.
- **You are tuning PET configs.** The "rank is a threshold" finding stops you from wasting compute sweeping LoRA rank to 256.

### Skip (or distrust) transfer scaling when

- **You have abundant, clean, in-domain data.** You are in the high-data regime; the multiplier has collapsed and ossification is a live risk. Measure against from-scratch or full fine-tuning with an aggressive learning rate.
- **The source and target are genuinely distant.** The text-to-Python constants do not transfer to a distant pair; $k$ and $\alpha$ are proximity properties. Re-measure for your pair, or expect a small multiplier (Case study 5 and 10).
- **Your adaptation is reward-shaped, not supervised.** RLHF and DPO obey [overoptimization scaling](/blog/machine-learning/scaling-laws/reward-model-overoptimization-scaling), not transfer scaling; do not port the SFT intuition there.
- **You have not checked your eval and data quality.** "From-scratch beat my fine-tune" has more causes than ossification (Case study 12); rule out a noisy eval and a noisy corpus first.

## Further reading

- Hernandez, Kaplan, Henighan, McCandlish 2021, "Scaling Laws for Transfer," arXiv:2102.01293
- Zhang, Liu, Cherry, Firat et al. 2024, "When Scaling Meets LLM Finetuning: The Effect of Data, Model and Finetuning Method," ICLR 2024, arXiv:2402.17193
- Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models" (Chinchilla), arXiv:2203.15556
- Kaplan et al. 2020, "Scaling Laws for Neural Language Models," arXiv:2001.08361
- Hestness et al. 2017, "Deep Learning Scaling is Predictable, Empirically," arXiv:1712.00409
- Sibling posts on this blog: [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling), [data-constrained scaling laws](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws), and [reward-model overoptimization scaling](/blog/machine-learning/scaling-laws/reward-model-overoptimization-scaling)

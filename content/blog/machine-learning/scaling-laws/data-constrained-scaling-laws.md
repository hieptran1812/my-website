---
title: "Scaling data-constrained language models: the 4-epoch rule"
date: "2026-06-15"
description: "Learn how to train a strong language model when you have run out of unique data: the effective-data decay law, why about four epochs of repetition is nearly free, the sixteen-epoch horizon, and how to split compute between epochs and parameters under a data cap."
tags: ["scaling-laws", "data-constrained", "data-repetition", "epochs", "effective-data", "large-language-models", "pretraining", "data-wall", "code-mixing", "perplexity-filtering", "compute-allocation"]
category: "machine-learning"
subcategory: "Scaling Laws"
author: "Hiep Tran"
featured: true
readTime: 53
---

Every scaling law you have read until now quietly assumed an infinite firehose of fresh text. [Chinchilla](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) tells you to feed about 20 tokens per parameter; [Kaplan](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models) told you to feed fewer. Both recipes take for granted that when the recipe says "train on 1.4 trillion tokens," you simply *have* 1.4 trillion unique tokens lying around. For the frontier labs in 2020 that was roughly true. It is no longer true. The high-quality, deduplicated, legally-clean portion of the public internet is finite, and the largest training runs have already swallowed most of it. So the practical question for the rest of us — and increasingly for the frontier too — is not "what is the compute-optimal allocation?" but "what do I do when I run out of unique data and still have compute to spend?" The diagram below is the mental model for the entire post: once you hit your unique-token cap, you repeat the data, and repetition pays off on a saturating curve — about four epochs is nearly free, the gains keep coming with diminishing returns out to roughly sixteen epochs, and past about forty-four epochs more repetition can actively hurt you.

![A decision graph showing a unique-token cap branching into repeat to four epochs, mix in code, and perplexity filter, all flowing toward sixteen epochs and a bias-compute-toward-epochs verdict, with a degradation warning past forty-four epochs](/imgs/blogs/data-constrained-scaling-laws-1.png)

That picture comes from Muennighoff et al. 2023, "Scaling Data-Constrained Language Models" (arXiv:2305.16264), a NeurIPS 2023 Outstanding Paper Runner-Up out of HuggingFace, Harvard, Cornell, Cohere, and the University of Turku. They trained roughly 400 models, up to 9 billion parameters and up to 900 billion tokens, specifically to measure what happens when you train on repeated rather than unique data, and they fit a clean modification of the Chinchilla loss law that predicts it. This post is a full tour of that result. We will build the intuition for why repeated tokens are worth less than fresh ones, derive the effective-data decay law and its key constant, work the headline 8.7-billion-parameter example by hand, walk through the other levers (code mixing, perplexity filtering, deduplication) and which ones actually paid off, and finish with a concrete procedure for splitting a fixed compute budget between more epochs and more parameters when you cannot get more data.

> [!important] The one number to remember: about 4 epochs is free, with a ~16-epoch horizon
> - **Repeated tokens are worth less than fresh ones, and the discount follows a law.** The Chinchilla form gets a twist: $L(N, D) = E + A/(N')^{\alpha} + B/(D')^{\beta}$ where $D'$ is *effective* tokens, not raw tokens, and $D' = U_D + U_D \cdot R_D^{*}(1 - e^{-R_D/R_D^{*}})$.
> - **Up to about 4 epochs of repetition is nearly free.** Concretely: an 8.7B model trained for 4 epochs over 44B unique tokens ended only **0.5% higher** validation loss than the equal-compute all-unique run.
> - **The decay constant is $R_D^{*} \approx 15.39$**, which sets the roughly 16-epoch horizon. From about 4 to 16 epochs you still gain, with diminishing returns; past 16 epochs returns "diminish extremely fast"; around 44 epochs the model can start to degrade.
> - **The repetition gain is strictly capped.** Effective data saturates at $U_D(1 + R_D^{*}) \approx 16.4 \times$ the unique tokens, no matter how many epochs you run. You cannot repeat your way to infinite data.
> - **Code is a free data extender.** Mixing code into a natural-language run gives roughly **2× effective tokens** even for NL-only evaluation, and up to **50% code shows no NL deterioration**.
> - **Perplexity filtering helped; deduplication did not** (on their benchmarks). One popular hygiene step paid off and one did not.
> - **Under a data cap, bias added compute toward more epochs over more parameters** — because the parameter repeat constant $R_N^{*} \approx 5.31$ saturates *earlier* than the data constant.

## Why this is different from every scaling law before it

The right way to feel this result is to notice how thoroughly it violates the assumption baked into the earlier laws. Kaplan and Chinchilla are *single-pass* laws. Every token in $D$ is a fresh, never-before-seen token, and the model sees it exactly once. The loss law $L(N, D) = E + A/N^{\alpha} + B/D^{\beta}$ treats $D$ as a count of distinct training examples, and the math works because more distinct examples carry more independent information about the data distribution. The moment you start repeating data, that identification breaks: the *raw* token count $D$ keeps going up by $U_D$ every epoch, but the *information* the model is extracting from those tokens does not. The second pass over a corpus teaches the model less than the first; the tenth teaches it less than the second. Here is the assumption-versus-reality table a senior engineer keeps in their head.

| Question | The single-pass assumption (Kaplan / Chinchilla) | The data-constrained reality (Muennighoff 2023) |
|---|---|---|
| What is $D$? | A count of fresh, distinct tokens | A count of *token-passes*, which can include repeats |
| Is the second epoch as valuable as the first? | The question never comes up | No — value decays with each repeat |
| Does more $D$ always lower loss? | Yes, monotonically as $B/D^{\beta}$ | Only up to a cap; effective data saturates |
| What does 16 epochs buy you? | Out of scope | About $10.6\times$ unique-token value, not $16\times$ |
| Can you trade compute for data? | Implicitly, by collecting more data | Yes, by repeating — but with sharply diminishing returns |
| When does extra training hurt? | Never (you stop at the optimum) | Past about 44 epochs, loss can rise |

The two rows that matter most are the third and the last. Single-pass laws say loss falls forever as you add tokens. The data-constrained law says that when those "added tokens" are repeats of tokens you already have, the loss reduction *saturates* — it approaches a floor set by how much unique information your corpus actually contains. And the last row is the one that catches people: there is a regime, reached only with extreme repetition, where continuing to train makes the model *worse*, because it is now memorizing the small corpus instead of generalizing. That never happens in the single-pass world.

> If you take one thing from this post: a repeated token is not a fresh token wearing a disguise. It carries a fraction of the information, that fraction shrinks with every pass, and the total information you can extract from a fixed corpus is bounded. Repetition buys time, not infinity.

### A short history of how the data wall arrived

It is worth tracing why this question became urgent, because for a few years it genuinely did not matter. In the Kaplan era (2020), the binding constraint was compute, not data. GPT-3 trained on roughly 300 billion tokens — a small fraction of even the cleaned Common Crawl available at the time — and stopped early because the Kaplan allocation said to spend on parameters and not bother training to convergence. Data was effectively free and abundant relative to the compute you could afford.

Chinchilla (2022) changed the arithmetic by demanding about 20 tokens per parameter. Suddenly a compute-optimal 70-billion-parameter model wanted 1.4 trillion tokens, and a 500-billion-parameter model would want 10 trillion. At those magnitudes the supply of high-quality, deduplicated, non-toxic, legally-usable text stops looking infinite. FineWeb, one of the largest cleaned web corpora, is on the order of 15 trillion tokens *total* after heavy filtering. If you want to train a model that Chinchilla says needs 20 trillion unique tokens, you cannot — the tokens do not exist. This is what people mean by the "data wall," and it is why Ilya Sutskever has called data "the fossil fuel of AI": there is a fixed amount of it, we are burning through it, and we will not get more internet.

So the field arrived at a fork. One branch says: get more tokens by other means — synthetic data, multimodal data, proprietary corpora. The other branch, the subject of this post, asks the humbler question: given the unique tokens you actually have, what is the smartest way to spend extra compute on them? The answer turns out to be a clean, fittable scaling law, and it tells you exactly how far repetition can take you before it stops helping.

### The arithmetic of the wall

It is worth doing the back-of-the-envelope that makes the wall concrete, because the numbers are more alarming than the rhetoric. Take a frontier-class target: a model in the 400-billion-parameter range. Chinchilla at 20 tokens per parameter wants $400\text{B} \times 20 = 8$ trillion unique tokens. That is already most of FineWeb. Push to a trillion-parameter model and Chinchilla wants 20 trillion unique tokens — more high-quality cleaned web text than has ever been assembled. And if you believe the [inference-aware argument](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) that the right ratio for heavily-served models is far above 20:1 — hundreds or thousands of tokens per parameter — the token demand explodes by another order of magnitude. The supply curve and the demand curve crossed somewhere around the 10-trillion-token training runs, and everything to the right of that crossing is the data-constrained regime.

There is also a quality cliff underneath the quantity cliff. The raw Common Crawl is on the order of $10^{14}$ tokens, but after deduplication, language filtering, quality filtering, toxicity removal, and dropping copyrighted or personally-identifying content, the usable fraction collapses by one to two orders of magnitude. So the headline "the internet has trillions of tokens" is true and misleading: the *trainable* internet is much smaller, and it is the trainable internet that sets $U_D$. This is why a paper about repeating data won an Outstanding Paper award at NeurIPS in 2023 — it was the first rigorous answer to the question the whole field had just realized it was about to face.

## 1. Effective data: the one idea that makes everything work

**Senior rule of thumb: when a clean law breaks because an assumption was violated, the fix is usually to replace the raw quantity with an "effective" quantity that restores the assumption.** Chinchilla's law assumes every token is fresh. Repetition violates that. The repair is to stop feeding the law the raw token count and feed it instead an *effective* token count — the number of fresh tokens that would have produced the same loss reduction as your repeated tokens did. The figure below contrasts the naive count with the effective-data count.

![A before-and-after diagram contrasting the naive token count that credits every epoch in full against the effective-data law that discounts repeats and caps the total gain](/imgs/blogs/data-constrained-scaling-laws-4.png)

The naive count, on the left, is what you get if you plug raw tokens into Chinchilla unchanged: $D = \text{epochs} \times U_D$, where $U_D$ is the number of unique tokens. Run sixteen epochs and the naive count says you trained on $16 \times U_D$ tokens, so the law predicts a loss as if you had sixteen times the data. That is wrong, and the experiments show it is wrong: the model trained on sixteen repeated passes does not match the model trained on sixteen times as much fresh data. The naive count overspends on repeats and overpromises on loss.

The effective count, on the right, is the fix. Define the number of *repeats beyond the first pass* as

$$R_D = \max\!\left(\frac{D}{U_D} - 1,\; 0\right)$$

so that a single pass has $R_D = 0$, four epochs have $R_D = 3$, sixteen epochs have $R_D = 15$, and so on. Then the effective token count is

$$D' = U_D + U_D \cdot R_D^{*}\left(1 - e^{-R_D / R_D^{*}}\right)$$

Read this term by term. The first $U_D$ is the value of the first pass, which is worth exactly its unique tokens — no discount, because nothing is repeated yet. The second term is the *bonus* from all the repeats, and it is an exponential saturation: it starts out adding nearly $U_D$ per repeat when $R_D$ is small (so early epochs feel almost like fresh data), but it bends over and approaches a ceiling of $U_D \cdot R_D^{*}$ as $R_D$ grows. The decay constant $R_D^{*}$ controls how fast the bonus saturates. The authors fit $R_D^{*} \approx 15.39$ from their runs (this comes straight out of the released code in github.com/huggingface/datablations).

The whole law then becomes Chinchilla with primes:

$$L(N, D) = E + \frac{A}{(N')^{\alpha}} + \frac{B}{(D')^{\beta}}$$

with a symmetric definition for the *effective parameter* count $N'$ (more on that in the compute-allocation section), and fitted exponents $\alpha = \beta \approx 0.3527$. The structure is identical to Chinchilla; only the inputs are discounted. That is what makes the result so satisfying: it does not throw away the existing theory, it patches the one assumption that repetition breaks.

A good way to check that the patch is principled is to confirm it *recovers* Chinchilla in the single-pass limit. At one epoch, $R_D = 0$, the exponential term $1 - e^{0} = 0$, and so $D' = U_D$ exactly — no discount, because nothing was repeated. The effective-data law reduces to the original Chinchilla law whenever every token is fresh. This is the hallmark of a well-formed generalization: it adds behavior in the regime the old law could not describe (repetition) while leaving the old law untouched in the regime it could (single pass). You can drop the modified law into any pipeline that already fits Chinchilla, and on single-pass data it will give identical predictions.

It also helps to see the raw-versus-effective contrast laid out directly, because the gap between the two columns is the entire cost of running out of data.

| Epochs | Raw tokens $D = \text{epochs} \times U_D$ | Effective tokens $D'$ | Effective fraction $D'/D$ |
|---|---|---|---|
| 1 | $1.0 \times U_D$ | $1.00 \times U_D$ | 100% |
| 4 | $4.0 \times U_D$ | $3.73 \times U_D$ | 93% |
| 16 | $16.0 \times U_D$ | $10.58 \times U_D$ | 66% |
| 44 | $44.0 \times U_D$ | $15.45 \times U_D$ | 35% |
| 100 | $100.0 \times U_D$ | $16.37 \times U_D$ | 16% |

The first two rows are why people say repetition is cheap; the last two are why people who repeat blindly waste most of their compute. The effective fraction is the efficiency of your compute spend, and it collapses well before you reach extreme epoch counts.

### Why an exponential saturation and not a power law

You might ask why the repeat bonus saturates *exponentially* rather than, say, as another power law. The intuition is about diminishing novelty. Think of each pass over the corpus as the model trying to extract residual information it missed last time. On the first repeat there is a lot left to learn — the model has seen each example once and has not yet fit it well, so the second pass behaves almost like new data. By the tenth repeat the model has nearly memorized the easy structure, and each additional pass surfaces only a sliver of genuinely new gradient signal. An exponential approach to a ceiling, $1 - e^{-R_D / R_D^{*}}$, captures exactly this: a roughly constant *fractional* decay in the marginal value of each pass. The constant $R_D^{*}$ is the "e-folding" number of repeats — after about $R_D^{*}$ repeats you have captured roughly 63% of the total available bonus, and after $2 R_D^{*}$ repeats about 86%.

The deep point is that the *information* in a corpus is finite. A million-token corpus contains at most a million tokens' worth of surprise, and usually much less because natural language is redundant. No number of passes manufactures information that was never there. The saturation ceiling $U_D \cdot R_D^{*}$ is the mathematical encoding of that fact: it is the most extra value repetition can ever extract, and it is a fixed multiple of how much unique data you started with.

### How the constant was measured

It is fair to ask where $R_D^{*} \approx 15.39$ comes from, because a scaling law is only as trustworthy as the fit behind it. The authors did not assume the value; they fit it. The experimental design was deliberately broad: roughly 400 training runs, models up to about 9 billion parameters, training budgets up to about 900 billion tokens, sweeping the number of unique tokens and the number of epochs independently so that the repetition effect could be isolated from the model-size effect. For each run they measured held-out loss, then fit the modified law $L(N, D) = E + A/(N')^{\alpha} + B/(D')^{\beta}$ jointly across all runs, with $R_D^{*}$ and $R_N^{*}$ as free parameters inside the effective-count definitions. The released code (in the datablations repository) is where the specific value $R_D^{*} \approx 15.39$ comes from; the paper text emphasizes the qualitative horizon (about four epochs free, about sixteen the practical limit) more than the third decimal place.

That distinction matters for how you should use the number. Treat $R_D^{*} \approx 15.39$ as a well-supported point estimate that pins the *shape* of the curve, not as a universal constant of nature. It will shift with corpus quality, tokenizer, domain, and model scale — a noisier corpus has less unique information per token and may saturate sooner; a cleaner, more diverse corpus may stretch the horizon a little. What is robust across their sweep, and what you should carry into your own runs, is the structure: a first pass at full value, an exponentially-saturating bonus from repeats, a hard ceiling at a low double-digit multiple of unique tokens, and a practical free-zone of a few epochs. If you have the budget, the honest move is to re-fit $R_D^{*}$ on a small sweep of your own data; if you do not, 15.39 is a sound default to plan around.

### A quick numerical sanity check in code

It helps to make the law concrete. Here is the effective-data multiplier evaluated at a range of epoch counts, using the fitted constant, so you can see the saturation directly.

```python
import math

R_D_star = 15.39  # fitted decay constant (Muennighoff et al. 2023, datablations repo)

def effective_multiplier(epochs):
    """D' / U_D as a function of the number of passes over the corpus."""
    R_D = max(epochs - 1, 0)              # repeats beyond the first pass
    bonus = R_D_star * (1.0 - math.exp(-R_D / R_D_star))
    return 1.0 + bonus                    # first pass (1.0) + saturating bonus

for epochs in (1, 2, 4, 8, 16, 32, 44, 64, 100):
    mult = effective_multiplier(epochs)
    efficiency = mult / epochs            # how close to "fresh data" each pass is
    print(f"{epochs:>4} epochs  ->  D'/U_D = {mult:6.2f}   "
          f"(naive would be {epochs:>3})   per-pass efficiency = {efficiency:5.1%}")
```

Running it prints the table that drives the whole post:

```console
   1 epochs  ->  D'/U_D =   1.00   (naive would be   1)   per-pass efficiency = 100.0%
   2 epochs  ->  D'/U_D =   1.97   (naive would be   2)   per-pass efficiency =  98.4%
   4 epochs  ->  D'/U_D =   3.73   (naive would be   4)   per-pass efficiency =  93.1%
   8 epochs  ->  D'/U_D =   6.62   (naive would be   8)   per-pass efficiency =  82.8%
  16 epochs  ->  D'/U_D =  10.58   (naive would be  16)   per-pass efficiency =  66.1%
  32 epochs  ->  D'/U_D =  14.34   (naive would be  32)   per-pass efficiency =  44.8%
  44 epochs  ->  D'/U_D =  15.45   (naive would be  44)   per-pass efficiency =  35.1%
 100 epochs  ->  D'/U_D =  16.37   (naive would be 100)   per-pass efficiency =  16.4%
```

Look at the per-pass efficiency column. At four epochs you are still capturing 93% of the value you would get from genuinely fresh data — that is what "nearly free" means quantitatively. At sixteen epochs you are down to 66% per pass, and the cumulative multiplier is about 10.6 rather than the naive 16. By a hundred epochs you are wasting more than 80% of every pass, and the multiplier has crawled to 16.37, barely above the ceiling of $1 + 15.39 = 16.39$. The corpus is exhausted; you are spending compute to extract essentially nothing.

## 2. The saturation curve, in one picture

**Senior rule of thumb: when a quantity saturates, draw the curve, mark the knee, and never let anyone reason about it as if it were linear.** The single most important visual in this post is the shape of $D'$ as a function of epochs, because that shape is the entire argument. The figure below is that curve, with the naive $D' = \text{epochs}$ line drawn for contrast and the ceiling marked.

![The effective-data saturation curve showing D prime over U_D rising steeply at first then bending toward a ceiling of about sixteen point four times the unique tokens, with the naive linear count diverging above it and markers at four and sixteen epochs](/imgs/blogs/data-constrained-scaling-laws-2.png)

The dashed straight line is the naive count: it credits every epoch in full, so it rises linearly and shoots off the top of the chart. The solid curve is the effective data $D'$. For the first few epochs the two lines nearly coincide — that is the "repetition is free" regime, where each pass is almost as good as fresh data. Then the solid curve bends. By the marked point at four epochs it is at about $3.7 \times U_D$ (still 93% efficient, hugging the naive line). By sixteen epochs it has fallen to about $10.6 \times U_D$, visibly below the naive line. And it asymptotes to the dashed horizontal ceiling at $U_D(1 + R_D^{*}) \approx 16.4 \times U_D$, which it can never cross no matter how many epochs you run.

The decay constant $R_D^{*} \approx 15.39$ is what sets the location of the knee. It is the e-folding scale of the saturation, and it is why the practical horizon for repetition is roughly sixteen epochs: that is about where the curve has bent far enough that additional passes are clearly not worth their compute. The number is not magic — it is fit from data, and it will differ somewhat across corpora and model scales — but the *shape* is robust, and the shape is the lesson. Anyone who tells you "we just trained for 50 epochs to make up for not having enough data" is operating in the flat part of this curve, paying full compute for a multiplier that has already topped out near 16.

### Reading the curve like an economist

There is a cleaner way to think about this curve: marginal value. The slope of the effective-data curve at any point is the marginal value of one more pass over your corpus, measured in fresh-token-equivalents. At one epoch that slope is nearly 1 — your next pass is worth almost a full corpus of fresh data. At four epochs the slope has dropped but is still respectable. At sixteen epochs the slope is shallow. Past forty-four epochs the slope is essentially flat: another pass adds nearly zero effective tokens, so you are paying compute for nothing — and worse, in the actual experiments, that flat region is where overfitting effects can start to push loss *up*, because the model is now memorizing rather than generalizing.

This is why the practical advice is phrased in three bands. **Repeat to about four epochs** because you are on the steep part of the curve where repetition is nearly free. **Continue to about sixteen epochs** if you have spare compute and no more data, accepting diminishing returns. **Stop before about forty-four epochs**, because past there you are firmly in the flat region and at risk of degradation. The bands are not arbitrary thresholds; they are the regions of high, medium, and negligible marginal value on a curve whose knee is governed by $R_D^{*} \approx 15.39$.

There is one more economic framing that practitioners find clarifying: the *exchange rate* between compute and data. In the single-pass world, compute and unique data are bought together — a FLOP and a fresh token arrive as a pair, and Chinchilla tells you the ratio. In the data-constrained world the exchange rate is not fixed; it degrades as you repeat. Early on, one extra unit of compute (a partial epoch) buys you almost a full unit of effective data, so the exchange rate is near 1:1. By sixteen epochs the rate has fallen to roughly 2:3 (you spend three units of compute to get two units of effective data). Past forty-four epochs the rate approaches infinity — you spend unbounded compute for essentially zero effective data. Knowing your position on this exchange-rate curve is the whole game: it tells you when buying more compute to repeat is still a good deal and when you should instead be buying (or borrowing, via code) more unique data.

## 3. The headline result: 4 epochs cost half a percent

**Senior rule of thumb: a scaling law is only believable when its authors put a model on the table and let it be measured against the honest baseline.** The decay law is elegant, but the result that made people pay attention is a single concrete comparison. The figure below shows the loss-versus-epochs picture that the comparison lives on.

![A curve of validation loss against epochs that drops steeply through the first few epochs then flattens, with an annotation that four epochs over 44B tokens ends only half a percent above the all-unique baseline and a warning that loss can rise past forty-four epochs](/imgs/blogs/data-constrained-scaling-laws-3.png)

The setup: take an 8.7-billion-parameter model and a corpus of 44 billion unique tokens. Train it for four epochs — that is, repeat the 44B-token corpus four times, for $4 \times 44 = 176$ billion token-passes of training. Now compare it against the honest baseline: a model trained at the same compute on genuinely unique data (no repetition). The four-epoch repeated run ended with a validation loss only **0.5% higher** than the all-unique run. Half a percent. For four passes over the same data instead of one pass over four times as much.

That is the empirical content of "four epochs is free." It is not literally free — half a percent of loss is not zero — but it is small enough that, when you cannot get four times as much unique data, repeating four times costs you almost nothing. And the loss-versus-epochs curve explains why: the steep initial drop is the model learning the corpus, and most of that drop is captured within the first few passes. After that the curve flattens. The green annotation marks the four-epoch sweet spot; the amber annotation marks the flat region past sixteen epochs where extra passes buy almost nothing; and past about forty-four epochs the curve can turn upward as the model overfits the small corpus.

### Putting numbers on "free"

Let us quantify what you are trading. Suppose you have a fixed compute budget that, on fresh data, would let you train on 176 billion unique tokens. You have two options:

1. **All-unique:** find 176B unique tokens and train one epoch. This is the Chinchilla-style baseline and gives the lowest loss.
2. **Repeat:** you only have 44B unique tokens, so train four epochs over them, using the same 176B token-passes of compute.

Option 2 ends 0.5% higher in loss. The effective-data law predicts roughly this: it says four epochs over 44B unique tokens is worth $3.73 \times 44 = 164$ billion *effective* tokens, versus the 176B unique tokens of option 1. So the law expects you to behave as if you trained on 164B rather than 176B tokens — a 7% shortfall in effective data, which through the $B/(D')^{\beta}$ term with $\beta \approx 0.35$ translates into a small fraction of a percent of loss. The arithmetic and the measurement agree: repeating four times over a corpus a quarter the size costs you a hair of loss, not a generation of capability.

Contrast that with the failure mode. If you instead had only 4 billion unique tokens and tried to hit 176B token-passes, you would need forty-four epochs. The effective-data law says forty-four epochs is worth $15.45 \times 4 = 62$ billion effective tokens — barely a third of your token-passes are doing anything, and you are in the region where the experiments show loss can stop falling and start rising. The same compute, spent on a corpus that is too small relative to the budget, is largely wasted. The lesson is not "repetition is good" or "repetition is bad" — it is that repetition has a budget, set by how much unique data you have, and the budget is roughly four epochs cheap and sixteen epochs total.

The framing that makes this actionable is to think of every training plan as a point in a two-dimensional space of unique tokens versus epochs, and to ask which iso-compute line you are on. Two runs with the same number of token-passes — say 176 billion — can sit anywhere from one epoch over 176B unique tokens (no repetition, full value) to forty-four epochs over 4B unique tokens (heavy repetition, mostly wasted). The compute bill is identical; the effective data differs by nearly a factor of three (164B versus 62B effective tokens), and so does the resulting model. This is the single most important thing to internalize: token-passes are what you pay for, but effective tokens are what you get, and the conversion between them depends entirely on how concentrated your repetition is. A planner who optimizes token-passes is optimizing the bill, not the result. A planner who optimizes effective tokens — by keeping epochs low and unique tokens high, borrowing from code when text runs out — is optimizing the model.

### What actually goes wrong past the ceiling

It is worth being precise about the failure mechanism in the degradation regime, because "the model gets worse" is the part people find hardest to believe. In the single-pass world you never train long enough on any individual example to memorize it; each token is seen once and contributes one gradient step's worth of generalization signal. When you repeat a small corpus dozens of times, the model sees each example dozens of times, and gradient descent does what it always does given the chance: it drives training loss toward zero by memorizing the specific tokens, including their noise. Validation loss — the thing you actually care about — stops tracking training loss and eventually rises, because the parameters are now encoding idiosyncrasies of the training set rather than the structure of the language.

This is classical overfitting, and it is exactly why the single-pass laws never had to mention degradation: they operate in a regime where overfitting cannot happen because nothing is repeated. The effective-data law absorbs the early, benign part of repetition (the saturating bonus) but cannot rescue you from the late, harmful part, because by then the marginal effective tokens are essentially zero while the marginal memorization is not. The practical signature is unmistakable: a widening gap between a still-falling training loss and a flat-then-rising validation loss. If you see that gap opening, you are past the useful horizon — stop, regardless of how much compute is left in the budget. The compute is not the constraint; the information in your corpus is, and you have already extracted it.

## 4. The four regimes of repetition

**Senior rule of thumb: any saturating process has named regimes, and knowing which regime you are in tells you what to do next.** It is worth laying the epoch axis out as a timeline of regimes, because the right action changes qualitatively as you move along it. The figure below names the four bands.

![A timeline of repetition regimes from one epoch through four, sixteen, forty-four, and beyond, labeling them all-unique baseline, nearly free, diminishing returns, saturated, and degradation](/imgs/blogs/data-constrained-scaling-laws-6.png)

**Regime 1 — the all-unique baseline (1 epoch).** Every token is fresh; $D' = U_D$ exactly. This is the world Kaplan and Chinchilla describe, and it is the reference point against which everything else is measured. If you have enough unique data to train one epoch at your compute budget, you are not data-constrained and this whole post is optional reading.

**Regime 2 — nearly free (up to about 4 epochs).** Effective data climbs to about $3.7 \times U_D$, per-pass efficiency stays above 90%, and the loss penalty is about half a percent. This is the regime you want to live in when you are mildly data-constrained. Repeating two to four times is one of the highest-leverage moves in data-constrained training: you get most of the value of a corpus several times larger, for the price of a little extra compute.

**Regime 3 — diminishing returns (about 4 to 16 epochs).** Effective data climbs from $3.7\times$ toward $10.6\times$, but per-pass efficiency drops from 93% to 66%. You are still gaining, and if you have compute and no more data, these epochs are worth running — but you should know you are paying more compute per unit of loss with each pass. The decay constant $R_D^{*} \approx 15.39$ means sixteen epochs (fifteen repeats) is roughly the one-e-folding point: you have captured a large fraction of the total available bonus.

**Regime 4 — saturated and then degrading (past about 16, hard stop near 44).** Past sixteen epochs the returns "diminish extremely fast" toward zero — the curve is nearly flat, effective data crawls from $10.6\times$ toward the ceiling of $16.4\times$, and you are spending real compute for negligible loss reduction. Around forty-four epochs the experiments show the model can begin to *degrade*: with the corpus so heavily memorized, additional training overfits and validation loss can rise. This is the regime to avoid. If your compute budget would push you past sixteen epochs on the data you have, the right move is almost never "run more epochs" — it is to get more data, mix in code, or accept a smaller-but-better-trained model.

| Regime | Epochs | Effective $D'/U_D$ | Per-pass efficiency | What to do |
|---|---|---|---|---|
| All-unique baseline | 1 | 1.0 | 100% | Not data-constrained; follow Chinchilla |
| Nearly free | up to ~4 | ~3.7 | >90% | Repeat freely; highest leverage |
| Diminishing returns | ~4 to ~16 | ~3.7 to ~10.6 | 90% down to 66% | Repeat if compute-rich, data-poor |
| Saturated / degrading | >16, stop by ~44 | ~10.6 up to ~16.4 ceiling | <66%, toward 16% | Stop; get data or shrink the model |

## 5. Stretching the budget: code, filtering, and dedup

**Senior rule of thumb: when you cannot get more of the data you want, the next move is to substitute data you can get and to spend your existing data more wisely — but verify each lever actually helps, because intuition is a poor guide here.** Repetition is the first lever for a data-constrained run. The paper studied three more, and the surprising part is that two of them paid off in ways you might not predict and one popular hygiene step did not help at all on their benchmarks. The figure below summarizes the levers and their measured verdicts.

![A matrix of data levers including repeating to four, sixteen, and past forty-four epochs, mixing code, perplexity filtering, and deduplication, each with its effect on effective data and its verdict at the studied scale](/imgs/blogs/data-constrained-scaling-laws-5.png)

### Code is a natural-language data extender

The most counterintuitive lever is mixing code into a text run. You might expect that adding code (Python, C, and so on) to a natural-language corpus would *help on code* and *hurt on text* — you are spending capacity on a different distribution. The measurement says otherwise: mixing code in gives roughly **2× effective tokens even when you evaluate on natural-language-only tasks**, and up to **50% code shows no deterioration on natural-language performance**. In other words, you can fill half your training mixture with code and your text capability does not suffer, while your effective data budget roughly doubles.

Why would code help text? The honest answer is that the mechanism is not fully settled, but the leading intuitions are that code is highly structured and teaches long-range dependency tracking, variable binding, and compositional reasoning that transfer to natural-language reasoning tasks; and that code is a large, clean, deduplicated, legally-clearer pool of tokens that you can pour into a run that has run dry on text. For a data-constrained team, this is enormous: GitHub and permissively-licensed code repositories are a second reservoir, and tapping it costs your text quality essentially nothing up to a 50% mix.

```python
# A data-constrained mixture that roughly doubles effective tokens without
# hurting natural-language evals (up to ~50% code is the studied safe zone).
mixture = {
    "natural_language_unique": 0.50,   # your text corpus, the constrained part
    "code_unique":             0.50,   # GitHub / permissive licenses: a 2nd reservoir
}
# Net effect reported: ~2x effective tokens for NL-only evaluation,
# with no measured NL deterioration at this 50/50 split.
assert sum(mixture.values()) == 1.0
```

### Perplexity filtering helped; deduplication did not

The other two levers are data-cleaning steps, and here the result is a useful corrective to conventional wisdom. **Perplexity filtering** — using a reference language model to score each document and dropping the highest-perplexity (least typical, often lowest-quality) documents — **helped** on their benchmarks. Keeping the more model-typical text gave better downstream performance per token, which makes sense: high-perplexity documents are often boilerplate, noise, or near-gibberish, and dropping them concentrates your limited token budget on text worth learning from.

**Deduplication**, on the other hand, **did not help** on their benchmarks. This is the surprising one, because deduplication is near-universal advice — and indeed in other settings it clearly helps (see the [data-quality post](/blog/machine-learning/scaling-laws/data-quality-scaling-laws) for the cases where exact and near-duplicate removal cuts memorization and improves perplexity). The nuance is that the benefit of deduplication is context-dependent. In a data-*constrained* regime where you are deliberately repeating data anyway, removing within-corpus duplicates does not obviously help: you are about to repeat everything several times regardless, so collapsing a few naturally-occurring duplicates into one changes little, and may even remove signal about which patterns are common. The lesson is not "never deduplicate" — it is "the value of a data operation depends on the regime, so measure it rather than assuming it."

| Lever | Mechanism | Effect on effective data | Verdict at their scale |
|---|---|---|---|
| Repeat to 4 epochs | Reuse unique tokens | ~3.7× of unique | +0.5% loss, nearly free |
| Repeat to 16 epochs | Reuse further | ~10.6× of unique | Diminishing returns |
| Repeat past 44 epochs | Reuse heavily | Saturates ~16.4× | Can degrade; stop |
| Mix in code | Tap a 2nd token reservoir | ~2× effective NL tokens | 50% code, no NL loss |
| Perplexity filter | Drop atypical documents | Concentrates budget on good text | Helped on evals |
| Deduplication | Remove duplicate documents | Removes redundancy | Did not help here |

### The relationship to data quality

There is a tension worth naming between this post and the [data-quality scaling laws](/blog/machine-learning/scaling-laws/data-quality-scaling-laws), and resolving it sharpens both. The data-quality literature shows that aggressive filtering — keeping only the best documents — shifts the loss-versus-compute curve down, sometimes worth a multiplicative compute factor. But it also shows a Quality-Quantity Tradeoff: a small, heavily-filtered subset loses utility fast when you repeat it, because you are now over-repeating a tiny pool. That is the same saturation curve from this post, viewed from the data-curation side. Filter too hard and you shrink $U_D$; with a smaller $U_D$ you hit the four-epoch and sixteen-epoch horizons sooner, and the filtered set's repeated utility decays.

The synthesis is that filtering aggressiveness should depend on your compute-to-data ratio. If you are data-rich relative to compute (you will train one pass), filter hard — quality dominates and you have tokens to spare. If you are data-poor relative to compute (you will repeat several epochs), filter more gently — you need the diversity, and over-filtering forces you deeper into the repetition curve where the filtered pool saturates. Perplexity filtering helped in the data-constrained study precisely because it is a *mild* filter that drops obvious junk without shrinking the corpus to a sliver. The lesson is to filter to the point where the marginal dropped document is genuinely worse than repeating a kept one — and that crossover moves with your epoch count.

### Combining the levers

These levers stack, and the stacking is multiplicative on your effective budget. The figure below shows the order to apply them: start from the unique text you have, repeat it to about four epochs (a roughly $3.7\times$ multiplier on the text portion), mix in code (roughly $2\times$ on effective tokens for NL evaluation), and perplexity-filter to keep the budget pointed at good text.

![A layered stack showing how to stretch a fixed unique-token budget, starting from unique text, then repeating to four epochs, then mixing in code, then perplexity filtering, then spending leftover compute on parameters or epochs](/imgs/blogs/data-constrained-scaling-laws-7.png)

The practical recipe that falls out is: do not reach for one lever and stop. A team with, say, 100 billion unique tokens of clean text and a Chinchilla appetite for 400 billion can plausibly close the gap by repeating to about four epochs and mixing in code, landing near the effective budget they need without finding a single new text token. Perplexity filtering then ensures the tokens they do have are the ones worth repeating. Only after exhausting these data levers should the leftover compute go toward more parameters or more epochs — which is the allocation question we turn to next.

## 6. How to allocate compute under a data cap

**Senior rule of thumb: when one input is capped, the optimization changes from "how do I get more of everything" to "how do I tilt my marginal dollar toward the input that is still paying off."** In the unconstrained Chinchilla world you scale parameters and tokens together, roughly $N \propto \sqrt{C}$ and $D \propto \sqrt{C}$. Under a data cap you cannot scale tokens with fresh data anymore — you can only scale token-*passes* by repeating, and we have seen that repetition saturates. So the allocation question becomes: with extra compute and a fixed unique-token budget, do you spend it on a bigger model or on more epochs? The figure below frames the choice.

![A decision diagram for allocating extra compute under a data cap, comparing spending on more parameters where the repeat constant saturates early against spending on more epochs where the repeat constant saturates later, concluding that you should tilt toward epochs first](/imgs/blogs/data-constrained-scaling-laws-8.png)

The key insight is that the effective-data law has a *symmetric* term for parameters. Just as effective tokens are

$$D' = U_D + U_D \cdot R_D^{*}\left(1 - e^{-R_D/R_D^{*}}\right),$$

the effective parameters are

$$N' = U_N + U_N \cdot R_N^{*}\left(1 - e^{-R_N/R_N^{*}}\right),$$

with its own decay constant $R_N^{*} \approx 5.31$. Here $U_N$ is the parameter count that would be compute-optimal on the unique data you have, and $R_N$ measures how far you push parameters beyond that point. The crucial comparison is between the two constants: $R_N^{*} \approx 5.31$ versus $R_D^{*} \approx 15.39$. The parameter term saturates *almost three times earlier* than the data term. That means, under a fixed unique-token budget, growing the model beyond its data-matched size stops paying off sooner than running more epochs does. So the verdict is: **bias added compute toward more epochs before more parameters.** Repeat your data to about four epochs (nearly free), push toward sixteen (diminishing but positive), and only then — or in parallel — consider a bigger model.

### Why two saturating terms imply tilting toward epochs

The tilt-toward-epochs conclusion is not a hand-wave; it falls out of comparing the marginal value of compute spent on each axis. Spend a marginal FLOP on the data axis (one more partial epoch) and it raises $D'$ by an amount proportional to the slope of the data saturation curve at your current $R_D$, which is $e^{-R_D / R_D^{*}}$ times $U_D$. Spend the same marginal FLOP on the parameter axis (a slightly bigger model) and it raises $N'$ by an amount proportional to $e^{-R_N / R_N^{*}}$ times $U_N$. Because the loss law is symmetric in the two penalty terms ($\alpha = \beta$), the loss reduction from each marginal FLOP is governed by which saturation factor is larger. And since $R_D^{*} \approx 15.39$ is nearly three times $R_N^{*} \approx 5.31$, the data-axis saturation factor $e^{-R_D/R_D^{*}}$ stays close to 1 for far more repeats than the parameter-axis factor $e^{-R_N/R_N^{*}}$ does. In plain terms: as you push beyond the data-matched configuration, the parameter axis goes flat first. Compute keeps earning loss reduction on the epoch axis after it has stopped earning on the parameter axis, so a rational allocator front-loads epochs.

There is a clean way to see the turning point. The two axes are equally attractive at the margin when their saturation factors are equal, $e^{-R_D/R_D^{*}} = e^{-R_N/R_N^{*}}$, i.e. when $R_D / R_D^{*} = R_N / R_N^{*}$. Because $R_D^{*}$ is the larger constant, this balance is reached at a *larger* number of data-repeats than parameter-repeats — you should be willing to run roughly $15.39 / 5.31 \approx 2.9$ times as many epoch-repeats as parameter-repeats before the two axes deserve equal marginal compute. That ratio is the quantitative content of "tilt toward epochs."

### A worked allocation example

Let us make the tilt concrete. Suppose you have $U_D = 100$ billion unique tokens and a compute budget $C$. Chinchilla on fresh data would tell you to train a model whose compute-optimal size, given 100B tokens at 20 tokens/param, is $U_N = 100\text{B}/20 = 5$ billion parameters. Now suppose your actual budget $C$ is four times larger than what a single epoch of that 5B/100B run would cost. You have $4\times$ the compute and no more data. Where does it go?

- **Option A — all into parameters.** Quadruple the compute on the model axis. With $C \approx 6ND$ and $D$ fixed at 100B, quadrupling $C$ quadruples $N$ to 20B parameters, trained one epoch. But now $D/N = 100\text{B}/20\text{B} = 5$ tokens per parameter — far below the 20:1 the data wants. You have a badly undertrained 20B model, and the effective-parameter saturation ($R_N^{*} \approx 5.31$) means much of that extra size is not earning its keep on only 100B tokens.
- **Option B — into epochs.** Keep $N = 5$B and spend the $4\times$ compute on four epochs over the 100B tokens. Effective data is $3.73 \times 100 = 373$ billion tokens, $D'/N = 373/5 = 75$ effective tokens per parameter — comfortably in well-trained territory, at a loss penalty of about half a percent versus the (unavailable) all-unique 400B run.
- **Option C — tilt toward epochs, then grow a little.** Spend most of the extra compute on epochs and a little on a modest parameter bump — say a 7-8B model trained for three epochs. This sits near the joint optimum the effective-data law predicts: enough epochs to keep effective tokens per parameter healthy, a slightly larger model to use them, neither axis pushed past its saturation knee.

Option B dominates Option A, and Option C edges out Option B by jointly optimizing both effective terms. The general principle, which you can implement by minimizing $L(N', D')$ over $(N, \text{epochs})$ subject to $C \approx 6ND$ with $D = \text{epochs} \times U_D$, is: **fill the epoch budget toward about four (free) and toward sixteen (if data-starved) before pouring compute into parameters, because $R_D^{*} > R_N^{*}$ means the data axis keeps paying after the parameter axis has stopped.**

```python
import math

# Effective-data and effective-parameter decay constants (fitted).
R_D_star, R_N_star, alpha = 15.39, 5.31, 0.3527
E, A, B = 1.69, 406.4, 410.7  # Chinchilla-style floor + coefficients (illustrative)

def eff(unique, repeats, R_star):
    return unique + unique * R_star * (1.0 - math.exp(-repeats / R_star))

def loss(N, epochs, U_D, U_N):
    R_D = max(epochs - 1, 0)
    R_N = max(N / U_N - 1, 0)         # how far params exceed the data-matched size
    Dp  = eff(U_D, R_D, R_D_star)
    Np  = eff(U_N, R_N, R_N_star)
    return E + A / Np**alpha + B / Dp**alpha

U_D = 100e9          # 100B unique tokens
U_N = U_D / 20       # 5B params is the data-matched (20:1) size

# Same ~4x compute budget spent three ways; lower loss is better.
print("A all-params  (20B, 1 epoch):", round(loss(20e9, 1,  U_D, U_N), 5))
print("B all-epochs  (5B,  4 epoch):", round(loss(5e9,  4,  U_D, U_N), 5))
print("C tilt+grow   (8B,  3 epoch):", round(loss(8e9,  3,  U_D, U_N), 5))
```

The exact numbers depend on the coefficients you plug in, but the *ordering* is the robust takeaway: spreading compute across epochs (and a modest parameter bump) beats dumping it all into a bigger, data-starved model. The model that is too big for its data sits on the flat part of the $N'$ curve; the model that runs a few extra epochs sits on the still-paying part of the $D'$ curve.

### The connection to inference-aware scaling

There is a satisfying link here to the [inference-aware scaling laws](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws). That line of work argues that when you account for lifetime inference cost, the optimum shifts toward a *smaller* model trained on *more* tokens — sometimes thousands of tokens per parameter. Data-constrained scaling reinforces the same direction from a different premise: when you cannot get more unique tokens, the way to push tokens-per-parameter higher is to repeat, and repetition keeps a smaller model well-fed past the point where a bigger model would starve. Both literatures, for different reasons, tell the practitioner to resist the urge to simply scale parameters: inference economics says a smaller over-trained model is cheaper to serve; data constraints say a smaller repeated model is what your data can actually support. They point the same way.

## 7. Worked examples from end to end

**Senior rule of thumb: a law you cannot turn into a number for your own situation is a law you do not understand.** Let us run three complete examples that a team might actually face, from a stated data cap and compute budget to a concrete decision.

### Example 1 — the mildly constrained team

You have 200 billion unique tokens of clean text and enough compute for 600 billion token-passes. Chinchilla would want you to train, say, a 30B model on 600B unique tokens — but you only have 200B unique. Three epochs over 200B gives you 600B token-passes at the right compute. The effective-data multiplier at three epochs is $1 + 15.39(1 - e^{-2/15.39}) = 1 + 15.39 \times 0.122 = 2.87$, so effective data is $2.87 \times 200 = 574$ billion tokens. You behave as if you had 574B unique tokens versus the 600B you would have ideally — a 4% effective shortfall, which is a small fraction of a percent of loss. Decision: **train three epochs, ship it, do not go hunting for the last 400B tokens.** You are in the nearly-free regime and the cost of repetition is in the noise.

### Example 2 — the severely constrained team

You have 20 billion unique tokens (a specialized domain — say legal or biomedical text) and compute for 320 billion token-passes. That is sixteen epochs over your 20B corpus. The effective multiplier at sixteen epochs is about 10.58, so effective data is $10.58 \times 20 = 212$ billion tokens — versus the 320B token-passes you are spending. You are getting two-thirds efficiency overall; the last several epochs are clearly diminishing. Decision: **run the sixteen epochs if you truly cannot get more data, but first try the data levers.** Mixing in general-domain or code tokens to roughly double effective data would let you reach the same effective budget at far fewer epochs of your scarce domain text, leaving the model less overfit to the small corpus. If after code mixing you can stay under, say, eight epochs on the domain text, do that instead — you will be on a healthier part of the curve.

### Example 3 — the team about to make a mistake

You have 5 billion unique tokens and a manager who wants a "big model" and has compute for 220 billion token-passes — which is forty-four epochs over 5B tokens. The effective multiplier at forty-four epochs is about 15.45, so effective data is only $15.45 \times 5 = 77$ billion tokens despite 220B token-passes of compute: nearly two-thirds of your compute is buying nothing, and you are at the edge of the degradation regime where loss can rise. Decision: **do not run forty-four epochs.** This is the textbook misuse of repetition. The right moves are, in order: aggressively mix in code and general text to multiply effective tokens (5B of domain text plus a large code/text pool could easily reach the effective budget at a handful of domain epochs); cap domain repetition at four to eight epochs; and size the model to the *effective* data you can muster, not to the compute budget. A smaller model trained on a healthy effective-token budget will beat a model trained forty-four epochs into the ground.

### Example 4 — the team with code to spare

You have 50 billion unique tokens of natural-language text and a large pool of permissively-licensed code, and compute for 400 billion token-passes. Naively, hitting 400B token-passes on 50B of text means eight epochs — well into the diminishing-returns regime, where effective data is about $6.62 \times 50 = 331$ billion tokens. But you have a second reservoir. Build a 50/50 mixture: 50B text plus 50B code, for 100B unique tokens, then run four epochs to reach 400B token-passes. Now your text portion is repeated only four times (nearly free), the code roughly doubles effective tokens for natural-language evaluation, and you have stayed off the steep part of the diminishing-returns curve on your scarce text. Decision: **mix code to halve the text repetition.** The same compute now sits in the nearly-free regime on the constrained axis instead of the diminishing-returns regime, and you get code capability as a bonus. This is the single most common win in practice: a second token reservoir lets you trade epochs you cannot afford for tokens you can.

### A table of the four examples

| Team | Unique tokens | Token-passes | Naive epochs | Better move | Why |
|---|---|---|---|---|---|
| Mildly constrained | 200B | 600B | 3 | Train 3 epochs, ship | Nearly-free regime, ~4% effective shortfall |
| Severely constrained | 20B | 320B | 16 | Mix data, cap text at ~8 epochs | 16 epochs is two-thirds efficient; mixing avoids the flat region |
| About to err | 5B | 220B | 44 | Mix heavily, cap at 4-8, shrink model | 44 epochs wastes ~2/3 of compute, risks degradation |
| Code to spare | 50B text + code | 400B | 8 (text-only) | 50/50 mix, 4 epochs | Halves text repetition, doubles effective tokens |

## Case studies from the literature and the field

The data-constrained regime shows up all over modern training, sometimes by design and sometimes by accident. Here are eight cases, named, with the symptom, the actual cause, the fix, and the lesson.

### 1. The forty-four-epoch wall

**Symptom:** a team with a small high-quality corpus kept adding epochs to use up their compute allocation, and validation loss stopped falling around epoch twenty and began creeping *up* past epoch forty. **Wrong first hypothesis:** the learning rate schedule was wrong, or the model was too small. **Actual cause:** they were deep in the saturated-then-degrading regime predicted by the effective-data law; with effective data pinned near the $16.4\times$ ceiling, extra passes were pure overfitting on a corpus the model had memorized. **Fix:** cap repetition at the sixteen-epoch horizon and spend remaining compute on code mixing instead. **Lesson:** the degradation past about forty-four epochs is real and predictable — the curve flattens and then the model starts memorizing. Compute available is not a reason to keep training; effective data exhausted is a reason to stop.

### 2. GPT-3's accidental frugality

**Symptom:** in hindsight, GPT-3 (175B params, ~300B tokens) looks bizarrely under-fed by Chinchilla standards — about 1.7 tokens per parameter. **Wrong first hypothesis:** OpenAI was data-constrained and could not find more tokens. **Actual cause:** GPT-3 predates both Chinchilla and the data-constrained framing; under the Kaplan allocation it was deliberately trained on few tokens and stopped early, and data abundance was simply not the binding constraint in 2020. **Fix (retrospective):** the [reconciliation post](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation) explains why the Kaplan allocation over-weighted parameters; the data-constrained correction is that even at fixed parameters, more token-passes via repetition would have helped. **Lesson:** "undertrained" and "data-constrained" are different diagnoses. GPT-3 was undertrained by choice, not data-constrained by necessity — but a modern redo would both add tokens and consider repetition.

### 3. The domain-specialist that overfit

**Symptom:** a biomedical language model trained on a curated 15B-token corpus overfit badly after a few epochs, with train loss diving and validation loss stalling. **Wrong first hypothesis:** regularization was too weak; add dropout and weight decay. **Actual cause:** 15B unique tokens is a small corpus, and pushing many epochs over it sat on the flat part of the effective-data curve where the model memorizes. **Fix:** mix in general-domain and code tokens to roughly double effective data, cap domain-text repetition at about four epochs, and reduce model size to match the effective data available. **Lesson:** specialization concentrates you into the data-constrained regime fast. The fix is rarely more epochs on the niche corpus; it is borrowing effective tokens from adjacent distributions (general text, code) that transfer.

### 4. Code that made the text better

**Symptom:** a team added a 40% code mixture to a natural-language run mostly to get code capability, and were surprised that their natural-language reasoning benchmarks *improved* rather than degraded. **Wrong first hypothesis:** the eval set leaked, or it was noise. **Actual cause:** the data-constrained result — up to 50% code shows no NL deterioration and code roughly doubles effective tokens — explains it: code added clean, structured, deduplicated tokens that taught compositional and long-range dependency skills transferring to text reasoning. **Fix:** they leaned in, keeping code near 50% as a deliberate data extender. **Lesson:** code is not just for code. Treat it as a second reservoir of high-quality tokens that helps text reasoning up to about half the mixture.

### 5. The deduplication that didn't pay

**Symptom:** a team in a data-constrained run spent significant engineering effort on aggressive near-duplicate removal expecting a quality bump, and saw essentially no downstream improvement. **Wrong first hypothesis:** the dedup implementation was buggy. **Actual cause:** in a regime where they were already repeating the corpus several epochs, removing naturally-occurring duplicates changed little — and the data-constrained study found deduplication did not help on its benchmarks. **Fix:** redirect the effort to perplexity filtering, which *did* help by concentrating the token budget on model-typical text. **Lesson:** data-cleaning operations are regime-dependent. Deduplication clearly helps in some settings (memorization, train-test overlap; see the [data-quality post](/blog/machine-learning/scaling-laws/data-quality-scaling-laws)), but it is not a universal win — measure, do not assume.

### 6. The perplexity filter that concentrated the budget

**Symptom:** a team with a fixed token budget wanted to squeeze more quality out of it and tried dropping the highest-perplexity documents using a reference model. **Wrong first hypothesis:** filtering would throw away too much data and hurt coverage. **Actual cause:** high-perplexity documents were disproportionately boilerplate and noise, so dropping them concentrated the limited budget on text worth learning — the data-constrained study found perplexity filtering helped. **Fix:** make perplexity filtering a standard step before deciding how many epochs to repeat, so the repeats land on good text. **Lesson:** when your token budget is fixed, *which* tokens you keep matters as much as how many times you repeat them. Spend your repeats on the documents most worth repeating.

### 7. The over-sized model on a small corpus

**Symptom:** a startup with about 30B unique tokens trained a 30B-parameter model (1 token/param) to chase a parameter-count headline, and it underperformed a 7B model trained on the same data for several epochs. **Wrong first hypothesis:** the big model needed more steps to "warm up." **Actual cause:** the 30B model was severely data-starved; on 30B tokens it sat far below the 20:1 ratio, and the effective-parameter term ($R_N^{*} \approx 5.31$) meant most of its size was idle. **Fix:** drop to a 7B model and run four to six epochs, lifting effective tokens-per-parameter into healthy territory. **Lesson:** under a data cap, model size is bounded by data, not by ambition. A smaller model that runs a few epochs beats a bigger model that sees the data once.

### 8. The frontier hitting the data wall

**Symptom:** the largest labs report that the marginal returns to scaling *pretraining* have softened, and several have publicly framed data — not compute — as the binding constraint. **Wrong first hypothesis:** the architecture or optimizer had plateaued. **Actual cause:** at the 10-15 trillion-token scale, the supply of high-quality unique text is genuinely limited; further scaling runs into the data wall, and naive repetition runs into the saturation ceiling. **Fix (industry response):** repeat within the four-to-sixteen-epoch budget, mix in code and multimodal data, lean on synthetic data, and shift effort toward post-training and test-time compute (the subject of later posts in this series). **Lesson:** the data-constrained law is not a niche concern for small teams. It describes the boundary the entire field is now pressing against, and it explains why "just train on more data" stopped being an option.

### 9. The continued-pretraining run that double-counted

**Symptom:** a team doing continued pretraining on a fixed proprietary corpus reported their token count as "we trained on 800B tokens" when the unique corpus was 50B repeated sixteen times, and were puzzled when the model underperformed a competitor's run reported at 800B tokens. **Wrong first hypothesis:** their architecture or data quality was worse. **Actual cause:** the two "800B" numbers were not comparable — the competitor's was 800B unique tokens (effectively 800B effective tokens), while theirs was sixteen epochs over 50B, worth only about $10.6 \times 50 = 530$ billion effective tokens. They were comparing 800 effective tokens against 530. **Fix:** report unique-token count and epoch count separately, and compute effective tokens before benchmarking against external runs. **Lesson:** "tokens trained" is ambiguous in the data-constrained world. Always report unique tokens and epochs, and convert to effective tokens before any cross-run comparison — otherwise you will mistake a data gap for a capability gap.

### 10. The synthetic-data shortcut that hit the same wall

**Symptom:** a team tried to dodge the data wall by generating synthetic text from a teacher model and training on it for many epochs, expecting unlimited effective data. **Wrong first hypothesis:** synthetic data is "new" data and therefore not subject to the repetition discount. **Actual cause:** synthetic data drawn from a single teacher carries limited *independent* information — much of it is paraphrase and recombination of what the teacher already knew, so it behaves more like repeated data than like fresh data, and training many epochs on it saturated similarly. **Fix:** treat synthetic data as a partial extender with its own (often lower) effective-data ceiling, diversify the generators, and ground it in real unique tokens rather than replacing them. **Lesson:** the effective-data discount is about *independent information*, not about whether a token is literally repeated. Synthetic and heavily-paraphrased data can hit a saturation ceiling too, for the same reason: the underlying information is finite.

### Common misconceptions, corrected

It is worth collecting the recurring mistakes in one place, because each of them comes from forgetting that a repeated token is not a fresh token.

| Misconception | Correction |
|---|---|
| "More token-passes always lowers loss" | Only up to the saturation ceiling; past ~44 epochs loss can rise |
| "16 epochs is 16× the data" | It is about 10.6× effective data; the gain is sublinear and capped at ~16.4× |
| "Repeating is a hack to avoid" | Up to ~4 epochs is a measured, nearly-free (~0.5% loss) standard move |
| "A bigger model uses my compute better" | Under a data cap, the parameter axis saturates earlier ($R_N^{*} < R_D^{*}$); tilt toward epochs |
| "Deduplication always helps" | It did not help in this data-constrained study; measure it for your regime |
| "Code only helps code tasks" | Up to ~50% code helps text reasoning and roughly doubles effective NL tokens |
| "Synthetic data is unlimited fresh data" | It carries finite independent information and saturates like repeats |

## What this means in practice

The data-constrained regime rewrites the practical recipe for anyone whose unique-token supply is smaller than their compute appetite — which, increasingly, is everyone. The takeaways compress to a short procedure.

**Reach for repetition when:**

- Your compute budget exceeds what one epoch over your unique data would use, and you cannot cheaply get more unique tokens. Repeating to about four epochs is nearly free (about 0.5% loss) and is one of the highest-leverage moves available.
- You are mildly to moderately data-constrained (you would land in the one-to-sixteen-epoch range). The effective-data law says you keep gaining out to about sixteen epochs, with diminishing returns; if you are compute-rich and data-poor, those epochs are worth running.
- You want to keep a smaller model well-fed. Repetition lifts effective tokens-per-parameter, which keeps a right-sized model on the still-paying part of the curve and dovetails with the inference-aware argument for smaller, longer-trained models.

**Stretch the budget with data levers before adding epochs or parameters:**

- Mix in code: up to about 50% code roughly doubles effective tokens with no measured natural-language degradation. Treat code as a second reservoir, not a separate task.
- Perplexity-filter: drop the highest-perplexity documents to concentrate your fixed token budget on text worth repeating. It helped on their benchmarks.
- Do not assume deduplication helps in this regime: it did not on their benchmarks. Measure it for your setting rather than spending engineering effort on faith.

**Skip more repetition when:**

- You would push past about sixteen epochs and especially past about forty-four. The curve is flat there, effective data is pinned near the $16.4\times$ ceiling, and past forty-four epochs the model can degrade. Spend compute on data levers or a smaller model instead.
- You are tempted to grow the model to consume a compute budget your data cannot support. The effective-parameter constant saturates earlier than the effective-data constant, so under a data cap you should tilt extra compute toward epochs (and a modest parameter bump) before pouring it into a bigger, data-starved model.
- You actually have enough unique data to train one epoch at your budget. Then you are not data-constrained — follow Chinchilla, scale $N$ and $D$ together, and ignore the repetition machinery entirely.

If you want the whole post compressed into rules you can recite at a planning meeting, here they are. First: report unique tokens and epochs separately, never a single conflated "tokens trained" number, and convert to effective tokens before comparing runs. Second: repeat to about four epochs by default whenever you are even mildly data-constrained — it is nearly free and is the highest-leverage move on the board. Third: treat sixteen epochs as the practical ceiling and forty-four as a hard stop, watching the train-validation gap for the degradation signature. Fourth: reach for code and perplexity filtering as data extenders before you reach for more epochs or more parameters, and do not assume deduplication helps in this regime. Fifth: under a data cap, tilt extra compute toward epochs before parameters, because the parameter axis saturates earlier — you can justify running roughly three times as many epoch-repeats as parameter-repeats before the axes deserve equal compute.

The deepest lesson is the one the effective-data curve draws in a single line: a corpus contains a finite amount of information, repetition extracts it on a saturating schedule, and the total it can ever yield is a fixed multiple — about sixteen and a half times — of the unique tokens you started with. Repetition buys you time and a few epochs of nearly-free progress. It does not buy you a bigger internet. When the curve flattens, the answer is not more passes; it is more, better, or different data — and increasingly, more thinking at inference time rather than more memorizing at training time.

## Further reading

- Muennighoff et al. 2023, "Scaling Data-Constrained Language Models," arXiv:2305.16264 (NeurIPS 2023 Outstanding Paper Runner-Up). Code: https://github.com/huggingface/datablations
- Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models" (Chinchilla), arXiv:2203.15556
- Kaplan et al. 2020, "Scaling Laws for Neural Language Models," arXiv:2001.08361
- Sardana & Frankle et al. 2023, "Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws," arXiv:2401.00448
- Penedo et al. 2024, "The FineWeb Datasets," arXiv:2406.17557
- Sibling posts on this blog: [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling), [inference-aware scaling laws](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws), and [data quality as a scaling axis](/blog/machine-learning/scaling-laws/data-quality-scaling-laws)

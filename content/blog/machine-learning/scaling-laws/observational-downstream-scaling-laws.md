---
title: "Predicting downstream performance: observational scaling laws"
date: "2026-06-15"
description: "Learn how to forecast a model's benchmark, agentic, and chain-of-thought performance before training it, using a capability space built from existing public models, over-trained probe runs, and loss-to-error maps."
tags: ["scaling-laws", "observational-scaling", "downstream-prediction", "capability-space", "pca", "emergent-abilities", "over-training", "gpt-4", "loss-to-error", "predictability", "machine-learning"]
category: "machine-learning"
subcategory: "Scaling Laws"
author: "Hiep Tran"
featured: true
readTime: 53
---

Here is the uncomfortable gap that this post exists to close. We can predict the **pretraining loss** of a model a thousand times larger than our biggest experiment, and we have been able to do that since 2020. But loss is not what anyone buys a model for. Nobody ships a product because the cross-entropy hit 1.91 nats per token. They ship because the thing answers questions, writes code, follows instructions, and chains a dozen tool calls without falling over. And for a long time the honest answer to "will the next run be good at *those* things?" was a shrug, because the chart of accuracy-versus-scale looked like a flat line that suddenly jumped — the dreaded "emergent ability" — and you cannot extrapolate a cliff.

That shrug cost real money. A frontier pretraining run is an eight-figure commitment, and the decision to launch it used to rest partly on intuition about whether a capability would "show up" at the target scale. This post is about the body of work, mostly from 2023 and 2024, that turned that intuition into arithmetic. The punchline is that downstream performance — benchmark scores, agentic success rates, chain-of-thought gains, even the abilities people called emergent — is **forecastable cheaply**, often for the price of a dinner rather than a data center, if you choose the right intermediate quantity and the right link function.

> [!important]
> **The seven things to take away**
> - **Loss has always been predictable; downstream wasn't — until we found the right links.** The trick is never to extrapolate a raw accuracy curve directly. Go through an intermediate: a capability vector, a loss value, or a properly-scored metric.
> - **Observational scaling laws (Ruan 2024)** build a scaling law from **~100 existing public models with no new training**, by extracting a **low-dimensional capability space** — PCA over benchmark scores gives **K=3 principal components capturing about 97% of the variance**. The whole fit costs **tens of dollars**.
> - **Loss maps to downstream error exponentially (Gadre 2024):** average top-1 error is `Err = ε − k·exp(−γL)`, equivalently a power law in perplexity. That lets you predict a 6.9B-parameter model's accuracy from roughly **20× less compute**.
> - **GPT-4 predicted its final loss from up to 10,000× less compute and its HumanEval pass rate from up to 1,000× less compute** — but one task (Hindsight Neglect) stayed stubbornly unpredictable. Predictability is the rule, not a law of nature.
> - **"Emergence" is mostly a link-function problem.** A logistic or power-law link turns a discontinuous jump into a smooth sigmoid you can fit and extrapolate.
> - **Emergence itself is now forecastable:** finetuning shifts the emergence point earlier (Snell 2024); massive decode-time sampling gives near-infinite resolution (Hu PassUntil 2023), predicting a 2.4B model's code-generation score to **0.05%** before training.
> - **The one number to remember:** a Ruan-style observational forecast of GPT-4-class agentic behavior costs **tens of dollars** and uses **zero new training runs**.

The diagram below is the mental model for the entire post. There is not one way to forecast downstream performance — there are three, and they differ by what you already have lying around. If you have access to a hundred existing models (and you do, on public leaderboards), you fit an observational law. If you can afford a handful of small training runs, you fit an over-training loss law and map loss to error. If you are already mid-run and have cheap checkpoints, you fit a single-run curve. All three funnel through a link function that smooths out the apparent jumps, and all three land on the same place: a forecast of the score you actually care about.

![A diagram showing three input sources feeding three forecasting methods that converge through a link function into a single downstream forecast of benchmark, agentic, and chain-of-thought scores](/imgs/blogs/observational-downstream-scaling-laws-1.png)

Notice what every path in that figure has in common: none of them extrapolates the target metric directly. The observational path goes through a capability vector; the over-training path goes through loss; the single-run path goes through a clean power law in compute. The intermediate quantity is always something that scales smoothly, and the final, jumpy-looking metric is recovered at the end with a link function. Hold that structure in your head — it is the one idea that unifies the whole literature, and the rest of this post is a tour of it.

## Why downstream is harder than loss

**Senior rule of thumb: loss is an average over tokens, so it is smooth by construction; a benchmark score is a thresholded count of whole-answer successes, so it is jumpy by construction. Never confuse the two.**

To see why the same scaling that makes loss predictable makes accuracy look unpredictable, you have to look at what each quantity actually measures. Pretraining loss is the mean negative log-likelihood the model assigns to the next token, averaged over an enormous validation set. Average a few hundred million smooth per-token numbers and you get a quantity that moves in tiny, continuous increments as you add parameters or data. That is the quantity Kaplan and Chinchilla taught us to extrapolate — see [the foundations post](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations) for why power laws extrapolate at all, and [Chinchilla](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) for the compute-optimal allocation that fixes the loss budget.

A benchmark score is a different animal. Take 5-digit multiplication. The model has to get every digit right; one slip and the answer is wrong. If the per-token probability of being correct is `p`, the probability of an exact-match-correct full answer of length `N` tokens is roughly `p^N`. So even as `p` climbs smoothly from 0.90 to 0.95 to 0.99, the exact-match accuracy crawls along near zero and then — once `p` crosses a threshold where `p^N` stops being tiny — shoots up. The underlying capability improved smoothly; the *metric* compressed that smooth improvement into a cliff. This is the core mechanism behind the "emergence is a mirage" argument, which I cover in depth in [the emergent-abilities post](/blog/machine-learning/scaling-laws/emergent-abilities-scaling); here we care about the practical consequence: you cannot fit a power law to a cliff.

Let me put numbers on the compounding so the cliff is not just a word. Suppose the answer is `N = 10` tokens long and `p` is the per-token probability of getting each one right. Exact-match accuracy is `p^10`:

- `p = 0.50` gives `0.50^10 ≈ 0.001` — essentially zero, indistinguishable from chance.
- `p = 0.70` gives `0.70^10 ≈ 0.028` — still looks like noise on a benchmark.
- `p = 0.85` gives `0.85^10 ≈ 0.197` — now it is suddenly "working."
- `p = 0.95` gives `0.95^10 ≈ 0.599` — and the curve has visibly exploded.

The per-token competence `p` moved in even, boring steps of 0.15 to 0.20. The exact-match score went 0.001, 0.028, 0.197, 0.599 — a textbook hockey stick. If your only measurements are at `p = 0.50` and `p = 0.70`, you see two numbers near zero, and any reasonable extrapolation predicts the model will stay near zero forever. The real curve is about to take off, and your forecast missed it entirely. This is not a subtle statistical effect; it is the multiplicative structure of exact-match acting on a smoothly improving `p`. The fix is to measure and extrapolate `p` (or token-edit distance, which tracks `1 − p`), not `p^N`. Everything downstream in this post is a more sophisticated version of that single substitution.

| Quantity | What it measures | Shape vs scale | Directly extrapolable? |
|---|---|---|---|
| Pretraining loss | Mean per-token NLL over a huge set | Smooth power law | Yes — this is the classical scaling law |
| Perplexity | `exp(loss)` | Smooth | Yes (monotone in loss) |
| Token-edit distance | Continuous per-token error | Smooth | Yes |
| Brier score | Proper scoring rule on probabilities | Smooth | Yes |
| Multiple-choice grade | Thresholded argmax-correct | Jumpy near threshold | No — manufactures jumps |
| Exact-string-match | All-or-nothing whole answer | Cliff (`~p^N`) | No — manufactures jumps |
| Agentic task success | Long horizon, compounding | Very jumpy | Not directly — needs a link |

The right column is the whole story. The quantities that extrapolate cleanly are averages and proper scores; the quantities that look emergent are thresholded counts. Schaeffer and colleagues showed in 2023 that **more than 92% of the hand-annotated emergent BIG-Bench tasks use one of two discontinuous metrics** — Multiple-Choice Grade or Exact-String-Match — and that swapping in a continuous metric on the *same model outputs* makes the curve smooth again. They even induced fake "emergence" in vision models (CIFAR100 autoencoders, CNNs) just by choosing a discontinuous metric. The lesson for a forecaster is not "emergence is fake" — that is too strong, and [the loss-perspective rebuttals](/blog/machine-learning/scaling-laws/emergent-abilities-scaling) show genuine loss-threshold transitions survive — the lesson is **route around the discontinuous metric**.

### The three routes, stated precisely

Every method in this post is one way to route around the discontinuity. Conceptually they are the same move applied to three different starting points, and that shared move is the reason the figure above has three lanes that merge. Concretely:

- **Observational (Ruan).** Build a smooth intermediate — a capability vector — from many existing models, then link it to the jumpy metric.
- **Over-training (Gadre).** Use loss as the smooth intermediate; fit a loss law on cheap runs, then map loss to error.
- **Single-run curve (GPT-4 report, Hu).** Use compute directly as the smooth axis; fit a power law on cheap checkpoints, or boost metric resolution with massive sampling so even a "zero" score becomes a measurable positive number.

We will take them one at a time, with the numbers, and then talk about when each one breaks.

## 1. The loss-to-error map: the cheapest bridge

**Senior rule of thumb: if you can predict loss (you can), and loss maps cleanly to error (it does), then you can predict error. The map is exponential in loss, power-law in perplexity.**

Start with the route that requires the least new machinery, because it bolts directly onto the loss scaling laws you already trust. Gadre and colleagues, in their 2024 paper "Language models scale reliably with over-training and on downstream tasks," trained **104 models from 0.011B to 6.9B parameters**, with token multipliers (data-to-parameter ratio `M = D/N`) pushed as high as **640×**, and then asked a simple question: given a model's pretraining loss, can I predict its average accuracy on a suite of downstream tasks?

The answer is a remarkably clean functional form. Let `L` be the pretraining loss and `Err` the average top-1 error across the downstream suite. Then

$$
\mathrm{Err}(L) = \varepsilon - k \cdot \exp(-\gamma L),
$$

where `ε` is the irreducible error ceiling (roughly the random-guess error you cannot beat), `k` sets the dynamic range, and `γ` controls how fast error falls as loss falls. Because perplexity is `PP = exp(L)`, the identical relationship written in perplexity is a power law:

$$
\mathrm{Err}(\mathrm{PP}) = \varepsilon - k \cdot \mathrm{PP}^{-\gamma}.
$$

The two forms are the same equation because of the identity `exp(−γL) = (exp L)^(−γ) = PP^(−γ)`. That equivalence is worth dwelling on, because it tells you *which axis to plot on*. On a loss axis, the relationship is an exponential decay toward a ceiling; on a log-perplexity axis it is also exponential; but on a raw-perplexity axis it is a power law, which means on log-log axes (log error-gap versus log perplexity) it is a straight line. So if you want to eyeball whether the loss-to-error map holds for your suite, plot `log(ε − Err)` against `log(PP)` and look for a straight line of slope `−γ`. A straight line means the map is valid and you can extrapolate; curvature means you have the wrong `ε` or the suite has two regimes.

The figure below is that map drawn out. Read it right-to-left in loss: a model with high loss sits near the error ceiling `ε` (it is barely better than guessing); as you drive loss down — by scaling up, training longer, or cleaning data — the error drops along a smooth exponential and bottoms out at the floor.

![A plotted curve showing average top-1 error rising and saturating toward a ceiling as pretraining loss increases, with blue dots marking cheap small models on the left and amber dots marking an extrapolated large model on the right](/imgs/blogs/observational-downstream-scaling-laws-2.png)

The two colors in that plot are the whole economic argument. The blue dots on the left are cheap small models — the ones you can actually afford to train. You fit the exponential to them. The amber dots on the right of the dashed divider are where the expensive model would land; you read its predicted error straight off the extended curve. Gadre report that this two-step procedure — predict loss from a cheap loss law, then map loss to error — **predicts the average top-1 error of a 6.9B-parameter model trained on 138B tokens from roughly 20× less compute**.

### Why the over-training law matters here

The loss-to-error map is only useful if you can predict the loss it consumes, and the wrinkle Gadre had to solve is that modern training is *over-trained*: we deliberately train small models on far more tokens than Chinchilla-optimal because inference is cheap when the model is small (this is the whole [inference-aware scaling](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) argument). A vanilla loss law in compute alone does not capture the token-multiplier dimension. Gadre's over-training loss law adds it:

$$
L(C, M) = E + \left(a \cdot M^{\eta} + b \cdot M^{-\eta}\right) \cdot C^{-\eta},
$$

where `C` is compute, `M = D/N` is the token multiplier, `E` is the irreducible loss, and `a, b, η` are fitted constants. The clever part is the bracket: it lets a model trained at a high multiplier (lots of tokens per parameter) sit on a different but still predictable curve. With this, Gadre **predict the loss of a 1.4B model over-trained 32× from a fit that uses about 300× less compute** than that target run. Chain the two laws — loss from `L(C, M)`, then error from `Err(L)` — and you have a downstream forecast that never once looked at a jumpy accuracy curve directly.

The figure below is that chain drawn as a workflow: cheap probe runs in, fitted loss law, the loss-to-error map, and a forecast of the large model's accuracy out the right side.

![A four-stage pipeline showing 104 cheap probe runs feeding a fitted over-training loss law, then a loss-to-error map, then a forecast of a 6.9B model's average error from far less compute](/imgs/blogs/observational-downstream-scaling-laws-7.png)

The discipline that workflow enforces is the discipline of the whole post: the only thing you ever extrapolate is loss, which is smooth; the error number is computed from loss at the very end. Every box to the left of the final forecast deals in a quantity that obeys a clean power law. Nothing in the chain asks you to extend a jumpy accuracy curve, which is the operation that always fails.

### A worked example you can do by hand

Let me make the loss-to-error map concrete with numbers, because the formula is abstract until you have pushed values through it. Suppose you have fit, on cheap runs, the constants `ε = 0.62`, `k = 0.55`, `γ = 0.85` (loss in nats). You now want to know the downstream error of a planned large run whose loss your loss law predicts to be `L = 1.4`.

$$
\mathrm{Err}(1.4) = 0.62 - 0.55 \cdot \exp(-0.85 \times 1.4) = 0.62 - 0.55 \cdot \exp(-1.19).
$$

Now `exp(−1.19) ≈ 0.304`, so `Err ≈ 0.62 − 0.55 × 0.304 ≈ 0.62 − 0.167 ≈ 0.453`. An average top-1 error of about 45%, i.e. roughly 55% accuracy across the suite. Drop the predicted loss to `L = 1.0` (a bigger or better-trained model) and `exp(−0.85) ≈ 0.427`, giving `Err ≈ 0.62 − 0.235 ≈ 0.385` — about 61% accuracy. Notice the shape: a 0.4-nat improvement in loss bought roughly six points of accuracy here, but the same 0.4-nat drop near the floor buys far less, because the exponential is flattening. That diminishing-returns curvature is exactly why you must fit the exponential rather than draw a straight line through two accuracy points — a line would massively over-predict the gains from the last few nats.

To feel the diminishing returns directly, take it one more step. Going from `L = 0.8` to `L = 0.6` (another 0.2 nats, now deep in the good regime): `exp(−0.68) ≈ 0.507` gives `Err ≈ 0.62 − 0.279 ≈ 0.341`, and `exp(−0.51) ≈ 0.600` gives `Err ≈ 0.62 − 0.330 ≈ 0.290`. So that 0.2-nat improvement bought about five points (34.1% → 29.0% error). Compare it to a 0.2-nat improvement up in the bad regime, from `L = 2.0` to `L = 1.8`: `exp(−1.7) ≈ 0.183` gives `Err ≈ 0.62 − 0.101 ≈ 0.519`, and `exp(−1.53) ≈ 0.217` gives `Err ≈ 0.62 − 0.119 ≈ 0.501` — under two points. The *same* loss improvement is worth two-and-a-half times more accuracy when you are already good than when you are bad. This is the opposite of most engineers' intuition (which expects the easy early gains to be the cheap ones), and it is a direct consequence of the exponential map: near the floor, each nat of loss translates into a larger swing in `exp(−γL)`. Forecast accordingly — the last stretch of a strong model's training, which looks like it is "barely moving the loss," can be moving the downstream score the most.

### When the loss-to-error map breaks

The map assumes a single, well-behaved relationship between loss and the *average* of a task suite. Two failure modes:

- **Per-task, not average.** The exponential is tight for the *suite average* but loose for any individual task, because individual tasks have their own thresholds and their own noise. Forecast the average; treat per-task predictions as wide intervals.
- **Distribution shift.** If the downstream suite is drawn from a different distribution than the validation loss measures (e.g. you measure loss on web text but test on competition math), the constants `ε, k, γ` shift. Re-fit per domain.
- **Wrong ceiling.** The ceiling `ε` is usually close to the random-guess error, but for a suite with a mix of answer formats it can sit somewhere odd, and a mis-set `ε` curves the log-log diagnostic plot. If `log(ε − Err)` versus `log(PP)` is not straight, try adjusting `ε` before concluding the map is invalid — the map is exquisitely sensitive to the asymptote.
- **Tokenizer effects on perplexity.** Perplexity depends on the tokenization; two models with different tokenizers are not on the same perplexity axis. The loss-to-error map in perplexity form is only comparable within a fixed tokenizer. This is one more reason the observational capability space exists — it sidesteps the cross-tokenizer perplexity incomparability that the loss-to-error map cannot.

These are not reasons to distrust the map; they are the conditions under which it applies. Inside its regime — one suite, one tokenizer, a correctly identified ceiling, a forecast of the average — it is among the most reliable tools in the kit, and it is the one you should reach for first when you control the training recipe.

## 2. Observational scaling laws: a hundred models, no new training

**Senior rule of thumb: you do not need to train a scaling ladder yourself — the field already trained one for you and posted the results on a leaderboard. Read it as a single, coherent scaling experiment.**

This is the idea that gives the post its title, and it is genuinely clever. Ruan, Maddison, and Hashimoto's 2024 NeurIPS spotlight, "Observational Scaling Laws and the Predictability of LM Performance," makes the following observation. There are by now **about a hundred publicly released language models** spanning many families, many sizes, and many training budgets, each with published benchmark scores. Instead of training a fresh, controlled scaling ladder — expensive, and confounded by your own idiosyncratic recipe — treat that pile of existing models as one big *observational* dataset and fit a scaling law to it directly.

The obstacle is that these models are not comparable on the surface. A 7B model from one lab and a 13B model from another were trained on different data, with different tokenizers, at different token multipliers; their raw FLOP counts are not apples-to-apples. Ruan's insight is to stop thinking in FLOPs and start thinking in **capabilities**. Different models with different recipes can still be placed on a small number of shared capability axes, and those axes turn out to be nearly linear in log-compute *within* a family.

### The capability space

The construction is a principal-component analysis over standardized benchmark scores. Take the ~100 models, take their scores on a battery of benchmarks (MMLU, ARC, HellaSwag, GSM8K, BBH, HumanEval, and so on), standardize each benchmark to zero mean and unit variance, and run PCA. The headline result is that the capability variation across all these models is overwhelmingly low-dimensional: **K = 3 principal components recover about 97% of the variance**.

![A four-row table describing the three retained principal components as general knowledge, reasoning, and coding capability axes plus a discarded residual, with the benchmarks each loads on and its forecasting use](/imgs/blogs/observational-downstream-scaling-laws-3.png)

The table above is how to read those three components. The first PC behaves like a general-capability axis — it loads on broad knowledge benchmarks and explains the lion's share of variance. The second and third behave like reasoning and coding axes. Everything past the third component is per-benchmark idiosyncrasy and measurement noise, and you throw it away. The phrase to internalize, the way this works in practice, is that **a model is a point in a 3-dimensional capability space**, and that point is what you scale, not the raw FLOP count.

Why does the variance collapse so dramatically? Because benchmarks are massively redundant. A model that is good at MMLU is almost certainly good at ARC and HellaSwag; a model that is good at GSM8K is probably decent at BBH. The benchmarks are noisy, correlated measurements of a handful of underlying competences, and PCA is exactly the tool for recovering those competences from correlated measurements. The 97% figure is the empirical statement that "language-model capability," as measured by the public benchmark battery, is *almost three-dimensional*. That is a strong and slightly surprising claim — it says that despite all the architectural and data diversity across a hundred models, three numbers per model capture nearly everything the benchmarks see.

The standardization step is not cosmetic. PCA finds the directions of maximum variance, and variance is scale-dependent: a benchmark scored 0–100 would dominate one scored 0–1 purely because its raw numbers are bigger. Standardizing each benchmark to zero mean and unit variance puts them on equal footing so the components reflect *correlation structure*, not arbitrary scoring ranges. Skip the standardization and your "capability space" is just an artifact of which benchmarks happened to use percentage points.

### The family slope is the key to comparing recipes

The within-family linearity `S_m ≈ θ_f·log(C_m) + ν_f` is what lets observational laws compare models that were never trained the same way. The subscript `f` matters: each family gets its *own* slope `θ_f` and intercept `ν_f`. A family with a better data recipe has a higher intercept `ν_f` — it reaches a given capability at less compute — and possibly a steeper slope. By fitting per-family slopes, the method factors out "whose recipe is better" from "how does capability grow with compute," which is precisely the confound that makes raw FLOP comparisons across labs meaningless. When you then predict a *new* strong model, you place it on its family's line and read off the capability vector, then push that vector through the benchmark and link regressions. The compute axis is shared in *log* terms across families; the capability axes are shared in *absolute* terms. That two-level structure — family-specific compute-to-capability, universal capability-to-benchmark — is the whole reason a hundred incomparable models become one coherent scaling experiment.

### The three links that make it forecastable

Ruan's law is a chain of three linear (or generalized-linear) relationships, and the figure below traces it. First, within a model family, the capability vector is approximately linear in log-compute:

$$
S_m \approx \theta_f \cdot \log(C_m) + \nu_f,
$$

where `S_m` is model `m`'s capability vector, `C_m` its compute, and `θ_f, ν_f` are family-specific slope and intercept. Second, each benchmark score is approximately linear in the capability vector. Third — and this is the move that defeats emergence — a downstream metric `E_m` that looks like it jumps is modeled through a **logistic link**:

$$
\sigma^{-1}(E_m) \approx \beta^{\top} S_m + \alpha,
$$

where `σ` is the logistic sigmoid. Applying the inverse-sigmoid to the metric *linearizes* it against capability, so the apparent jump becomes a straight line you can fit and extrapolate, and the forecast in metric space comes back out as a smooth sigmoid.

![A layered graph showing log-compute driving three capability principal components, which in turn linearly predict an MMLU score, an emergent task via a sigmoid link, and agentic and chain-of-thought performance via a logistic link](/imgs/blogs/observational-downstream-scaling-laws-5.png)

That figure is the engine of the method. Compute feeds the capability axes; the capability axes feed every downstream metric; the link function on the right absorbs whatever nonlinearity the metric has. Because the capability space is shared across families, you can fit the `β` and `α` for an expensive target capability using only cheap, already-existing weaker models — and then read off the prediction for a model stronger than anything in your fit set.

### What it actually predicts, and for how much

The results are the part that should make you sit up. Using only this observational fit over existing models, Ruan predict:

- **Emergent abilities** — the BIG-Bench-style tasks that look like cliffs under exact match — smooth out and become forecastable through the logistic link.
- **GPT-4's agentic performance** — its success on agent-style tasks — predicted from the *non-agentic* benchmark scores of weaker models. The capability axes carry enough signal that you do not need agentic data from strong models to forecast agentic behavior.
- **Chain-of-thought and self-consistency gains** — how much a model improves when you let it reason step by step, or sample and vote — predicted as a function of position in capability space.

And the cost: **tens of dollars**. There is no new training run. The expensive part is downloading scores and running a regression. That is the number to remember from this entire post — a forecast of frontier agentic behavior for the price of a few API calls and an afternoon.

It is worth pausing on how counterintuitive the agentic result is, because it is the strongest evidence that the capability space is real and not a curve-fitting accident. Agentic tasks — multi-step tool use, long-horizon planning, recovering from a failed action — feel qualitatively different from answering a multiple-choice knowledge question. A reasonable person would expect that forecasting agentic skill requires agentic data from strong models. Ruan's result says no: the *non-agentic* benchmark scores of weaker models already locate a model in the capability space precisely enough that agentic success falls out as a (link-transformed) linear function of position. The only way that can be true is if agentic competence is not an independent axis but a combination of the general, reasoning, and coding competences the standard benchmarks already measure. That is a substantive scientific claim about the structure of language-model capability, and the fact that the forecast lands is evidence for it. For practitioners, the operational consequence is liberating: the cheap, ubiquitous benchmarks you already have are not a poor substitute for expensive agentic evals — they are most of the signal, and the expensive evals are confirmation, not discovery.

### The sigmoid link, drawn out

The logistic link deserves its own picture, because it is the single mechanism that converts "unpredictable emergence" into "predictable sigmoid." Consider the way this works on a concrete task. The figure below shows the same set of models twice. On the left, raw exact-match accuracy against compute: flat near zero, then a sudden jump — the textbook emergence shape, and a curve no power law can fit. On the right, the same scores viewed through the link: the inverse-sigmoid of the metric is linear in the capability score, so the relationship is a clean line, and the prediction in metric space is a smooth, monotone sigmoid.

![A two-panel chart with the left panel showing raw exact-match accuracy jumping sharply with compute and the right panel showing the same models forming a smooth sigmoid once viewed through a logistic link against capability score](/imgs/blogs/observational-downstream-scaling-laws-4.png)

The left panel is what a naive forecaster sees and gives up on. The right panel is what Ruan's method sees. Nothing about the models changed between the two panels — same checkpoints, same outputs — only the lens. In essence the link function is doing the same job as choosing a continuous metric in the mirage argument: it refuses to let a thresholding nonlinearity hide the underlying smooth trend. If you remember one technique from this section, make it this: **before declaring a capability unpredictable, try fitting it through a sigmoid or power-law link.**

### Why the sigmoid link is the right transform

The choice of a logistic link is not arbitrary curve-fitting; there is a reason it is the natural transform for a bounded metric. Any accuracy-like metric lives in `[0, 1]`. A linear model `βᵀS + α` lives in `(−∞, +∞)`. If you regress accuracy on capability directly with a linear model, the fit will happily predict accuracies of 1.3 or −0.2 outside the fit range, which is nonsense, and it will systematically misfit near the boundaries where the true curve must flatten. The logistic function `σ(x) = 1/(1 + e^(−x))` is the standard bijection from `(−∞, +∞)` to `(0, 1)`, so modeling `σ^(−1)(E) = βᵀS + α` is the principled way to put a bounded metric on a linear footing. The inverse-sigmoid (the logit) stretches the squashed `[0,1]` metric back out into a linear scale; you fit the line there; you squash back at the end. The S-shape you see in the prediction is not assumed — it is the *image* of a straight line under the sigmoid, which is exactly what a metric bounded in `[0,1]` and driven by an unbounded latent must look like.

This also explains why "emergence" appears precisely where it does. The steepest part of a sigmoid is in its middle, around `σ(0) = 0.5`. As capability `S` climbs and `βᵀS + α` crosses zero, the metric races through the steep middle of the sigmoid — that crossing is the "emergence point." It looks like a phase transition because the sigmoid is genuinely steep there, but it is a smooth, differentiable, *fully predictable* steepness. The forecaster who works in logit space sees a boring straight line crossing zero; the forecaster who works in accuracy space sees a dramatic takeoff. Same data, same crossing, different lens.

### A worked observational forecast

Let me run one number end to end so the method is not abstract. Suppose you have fit, on weaker public models, the logistic-linked regression for an agentic-success metric, and obtained capability weights `β = (1.1, 0.8, 0.3)` (general, reasoning, coding) and intercept `α = −3.0`. A new strong model, placed on its family's compute line, has capability vector `S = (1.6, 1.4, 1.2)` (it is above the population mean on all three axes, in standardized units).

Compute the linear predictor: `βᵀS + α = 1.1×1.6 + 0.8×1.4 + 0.3×1.2 − 3.0 = 1.76 + 1.12 + 0.36 − 3.0 = 0.24`. Push it through the sigmoid: `σ(0.24) = 1/(1 + e^(−0.24)) ≈ 1/(1 + 0.787) ≈ 0.56`. So the forecast is about **56% agentic success** for a model whose own agentic data you never measured — derived entirely from where it sits in a capability space built from weaker models' *non-agentic* scores. Now suppose a competitor's model sits at `S = (2.2, 2.0, 1.8)`: the predictor is `2.42 + 1.6 + 0.54 − 3.0 = 1.56`, and `σ(1.56) ≈ 0.83`. The 0.6-unit jump in each capability axis moved success from 56% to 83% — and crucially, that jump is *readable off the straight line in logit space* (0.24 to 1.56) long before you would believe it from staring at accuracy numbers. This is the entire value proposition in one calculation.

## 3. Predictable scaling at the frontier: the GPT-4 report

**Senior rule of thumb: the strongest evidence that downstream is predictable is that a frontier lab bet a nine-figure run on the prediction and published the receipts.**

The Ruan and Gadre results are academic in the best sense — controlled, reproducible, cheap. The GPT-4 Technical Report (2023) is the industrial proof point. OpenAI describe a "Predictable Scaling" methodology in which they built infrastructure to forecast properties of the full GPT-4 run from much smaller runs, and they report two specific, quantified successes.

First, the **final loss**. They fit a power law of the form

$$
L(C) = a \cdot C^{b} + c
$$

— a power law in compute `C` with an irreducible offset `c` — and predicted GPT-4's final pretraining loss **from runs using up to 10,000× less compute**. Let that sink in: a model four orders of magnitude smaller, in compute terms, was enough to pin the loss of the full run. This is the loss-scaling-law story of [the foundations post](/blog/machine-learning/scaling-laws/scaling-laws-predictability-foundations) executed at the largest scale anyone had attempted, and it worked.

Second, and more impressively for our purposes, a **downstream capability**: pass rate on the HumanEval coding benchmark. They modeled the negative log of the pass rate as a power law in compute,

$$
-\mathbb{E}\!\left[\log\,\mathrm{pass\_rate}(C)\right] = \alpha \cdot C^{-k},
$$

and predicted GPT-4's HumanEval pass rate **from runs using up to 1,000× less compute**. Note the form: they did not extrapolate pass rate directly, which would be jumpy. They extrapolated the *expected negative log pass rate*, which is the smooth intermediate, and exponentiated back at the end. Same move as everywhere else in this post — find the smooth quantity, extrapolate that, transform back.

Why is `−E[log pass_rate]` the smooth quantity rather than `pass_rate` itself? For the same reason the loss-to-error map works on loss and not error. Pass rate on a generative coding task has the `~p^N` problem: a program is a long token sequence, and one wrong token usually breaks it, so raw pass rate compresses smooth improvement into a cliff. Taking the log undoes part of the multiplicative structure (`log p^N = N log p`), and the expectation over problems averages out per-problem thresholds. What is left is a quantity that falls as a clean power law in compute. The general lesson, which you should now be able to predict before I write it: when a metric is a product of many per-step successes, work in log space; when it is bounded in `[0,1]`, work in logit space; when it is an average of thresholded counts, average aggressively. Each of these is a way of recovering the smooth latent that the headline metric is hiding.

Worth noting how much engineering sat behind the published number. Predictable scaling at GPT-4's scale required building a dedicated forecasting infrastructure — a ladder of progressively larger runs, careful matching of data and hyperparameters across the ladder so the only thing varying was scale, and a fitting procedure robust enough to commit to publicly before the full run finished. The fact that they were willing to print the prediction *in advance* of the result is the strongest possible endorsement of the method: a lab does not stake a flagship launch on a curve it does not trust.

### The caveat that keeps you honest: Hindsight Neglect

It would be dishonest to present GPT-4's predictable scaling without its famous counterexample. Not every task was predictable. The report flags **Hindsight Neglect**, a task on which performance did *not* follow the smooth extrapolation — it exhibited inverse scaling (larger models got worse) over part of the range before reversing. The single-run power-law forecast failed there.

This matters for how you use these methods. Predictability is the strong central tendency, not a guarantee for every metric. The tasks that break tend to be the ones with non-monotone or U-shaped scaling — exactly the regime that [broken neural scaling laws](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws) were invented to model. A single power law assumes one smooth segment; when the true curve has a break or a non-monotonic dip, the single-power-law forecast can be confidently wrong. The practical implication is to forecast a *suite average* (where idiosyncratic non-monotonicities wash out) and to flag any individual task whose small-scale trend is non-monotone as high-risk for extrapolation.

### The timeline of the idea

It is worth pausing to see how fast this went from "loss only" to "agentic behavior for tens of dollars." The figure below lays out the four years that mattered.

![A timeline from 2020 to 2024 showing Kaplan establishing loss as a power law, GPT-4 predicting loss and HumanEval from far less compute, Gadre adding an over-training law and loss-to-error map, Ruan building an observational law over a hundred models, and Hu and Snell forecasting emergence](/imgs/blogs/observational-downstream-scaling-laws-6.png)

That progression is the structure of the whole field in one line. Kaplan and Chinchilla gave us loss. The GPT-4 report showed loss prediction works at the frontier and took the first serious swing at a downstream metric. Then 2024 was the year downstream prediction became cheap and general: Gadre's loss-to-error map, Ruan's observational capability space, and the emergence-forecasting work we turn to next.

## 4. Forecasting emergence itself

**Senior rule of thumb: even the abilities that genuinely switch on at a threshold can be predicted — you just need a way to measure them *before* the threshold, where the score would otherwise read as a flat zero.**

The deepest objection to everything above is: "fine, but what about a capability that is truly at zero in every model I can afford, and only switches on past my fit range? No link function can extrapolate from a column of zeros." This is a fair objection, and 2024 produced two distinct answers to it.

### Answer one: finetuning shifts the emergence point (Snell 2024)

Snell and colleagues' "Predicting Emergent Capabilities by Finetuning" makes a counterintuitive observation: **finetuning a model on a small amount of task-relevant data shifts the point of emergence toward weaker (smaller, less-compute) models.** A capability that emerges only at, say, 10^23 FLOPs in the base model might emerge at 10^22 FLOPs after a little finetuning. By finetuning a *ladder* of small models with varying amounts of data, you can watch the emergence point move, fit an "emergence law" to how it moves, and extrapolate where the capability will appear in the *base* (un-finetuned) model. The reported reach is about **4× compute ahead** — modest, but exactly in the regime where you are deciding whether the next run will cross a threshold.

The mechanism is intuitive once you see it: finetuning does not create the capability from nothing, it lowers the bar for expressing a latent capability the model partly has. So the finetuned ladder reveals the underlying smooth trend that the base model's exact-match score was hiding behind a threshold. It is the link-function idea again, implemented through data instead of through a sigmoid.

There is a subtlety in how to use Snell's result responsibly. The "emergence law" you fit describes how the emergence point moves *as a function of finetuning data*, and you extrapolate it to the zero-finetuning (base) model. That extrapolation is itself a forecast with its own error, so the ~4× compute-ahead reach is not a hard limit but a regime where the extrapolation has been validated; push it much further and the law's own uncertainty dominates. The right way to deploy it is as an early-warning system: a few cheap finetuned ladders tell you whether a capability is *about* to switch on in your next base-model generation, which is exactly the question a roadmap planner needs answered, and exactly the question a raw base-model evaluation (reading flat zeros) cannot answer.

### Answer two: infinite-resolution evaluation (Hu PassUntil 2023)

Hu and colleagues' "Predicting Emergent Abilities with Infinite Resolution Evaluation" (ICLR 2024) attacks the zeros directly. The reason a small model scores zero on, say, code generation is not that it can *never* produce a correct program — it is that the probability of producing one in a single sample is tiny, far below the resolution of a normal evaluation that samples once or a few times. **PassUntil** fixes the resolution by sampling enormously: keep decoding samples until you get a success, and estimate the per-sample success probability from how long that took. With enough samples, a "zero" becomes a measurable positive number like 0.0003, and a column of measurable positive numbers is something you can fit a scaling law to.

The headline demonstration: PassUntil **predicted the code-generation performance of a 2.4B-parameter model to within 0.05%, before that model was trained.** That is not a smoothed average over a suite — it is a single model's score on a specific generative task, pinned to four-significant-figure precision by buying resolution with compute at evaluation time. It connects naturally to the [repeated-sampling scaling laws](/blog/machine-learning/scaling-laws/repeated-sampling-scaling-laws): the same coverage-grows-with-samples effect that makes best-of-n work also makes near-zero capabilities measurable.

The trade is compute-for-resolution, and it is favorable precisely where you need it. A capability that scores zero at one sample but has a true per-sample probability of 0.0003 needs on the order of a few thousand samples to estimate that probability with usable precision — expensive per data point, but trivial compared to training the model, and you only pay it on the small models in your fit ladder. Once you have positive, smoothly-scaling per-sample probabilities across the ladder, ordinary power-law extrapolation predicts the strong model's probability, and you convert back to whatever pass@k metric you report. PassUntil is, once again, the same algorithm in a new costume: it manufactures a smooth intermediate (per-sample success probability) where the headline metric (pass@1) read as a flat, uninformative zero. The only novelty is that the smoothing happens at evaluation time, by spending decode compute, rather than at fit time by choosing a transform.

### Putting the methods in a taxonomy

Five named methods is a lot to keep straight, so the figure below organizes them by the question that actually decides which one you use: *what do you already have?*

![A tree organizing downstream forecasting into three method families: observational methods using many existing models, over-training methods using a few small runs, and single-run-curve methods using cheap checkpoints, each with its representative papers](/imgs/blogs/observational-downstream-scaling-laws-8.png)

That taxonomy is the decision procedure. If you have access to many models — and via public leaderboards you always do — the observational route (Ruan) is the cheapest and most general. If you can afford a few small training runs of your own recipe, the over-training route (Gadre) gives you a forecast grounded in *your* data and architecture. If you are already training and have cheap intermediate checkpoints, the single-run-curve route (the GPT-4 power law) or PassUntil's resolution trick applies. And if the specific worry is a threshold capability, Snell's finetuning shift and Hu's resolution boost are the specialized tools.

## 5. The unifying principle: find the smooth quantity

By now the repetition should be conspicuous, and that is the point. Every single method routes the forecast through a quantity that scales smoothly and applies a transform at the end:

| Method | Smooth intermediate | Final transform | Reach / cost |
|---|---|---|---|
| Loss-to-error map (Gadre) | Pretraining loss `L` | `Err = ε − k·e^(−γL)` | 6.9B error from ~20× less compute |
| Over-training loss law (Gadre) | Loss as `L(C, M)` | (feeds the map) | 1.4B/32× loss from ~300× less compute |
| Observational law (Ruan) | Capability vector `S` (3 PCs) | Logistic link `σ^(−1)(E)=βᵀS+α` | Agentic/CoT for tens of dollars |
| GPT-4 loss (OpenAI) | Compute `C` | `L=aC^b+c` | Final loss from up to 10,000× less |
| GPT-4 HumanEval (OpenAI) | `−E[log pass_rate]` | exponentiate | Pass rate from up to 1,000× less |
| Finetuning shift (Snell) | Emergence point vs data | emergence law | ~4× compute ahead |
| PassUntil (Hu) | Per-sample success prob | scaling law on prob | 2.4B code-gen to 0.05% |

Look down the "smooth intermediate" column. Loss, capability vector, compute, expected log pass rate, per-sample probability — these are all quantities whose scaling is a clean power law or linear-in-log relationship. Look down the "final transform" column: exponential, logistic, exponentiate, scaling-law-then-invert. The discontinuity always lives in the transform, never in the thing you extrapolate. Conceptually this is one algorithm wearing five costumes, and the figure of the three lanes at the top of the post was the abstract version of this table.

If you internalize a single operational habit from all of this, make it the two-question checklist you run before any capability forecast. First: *what is the smooth quantity here?* If the metric you care about is bounded, multiplicative, or a thresholded count, it is not the smooth quantity — find the latent it is a transform of (a probability, a loss, a capability coordinate). Second: *what is the transform that turns the smooth quantity back into the metric?* Identify it explicitly (sigmoid for bounded, exp/log for multiplicative, an averaging for thresholded counts), fit your scaling law in the smooth space, and apply the transform only at the very end. Teams that follow this habit forecast capabilities others call unpredictable; teams that skip it extrapolate cliffs and conclude, wrongly, that the future cannot be seen from here.

There is a deeper reason this works, and it ties back to the [emergent-abilities](/blog/machine-learning/scaling-laws/emergent-abilities-scaling) debate. Du and colleagues' 2024 "loss perspective" result — same pretraining loss implies same downstream performance, with abilities emerging below a task-specific loss threshold even under continuous metrics — is precisely what makes the loss-to-error map valid. If downstream performance is a (possibly threshold-shaped) function of loss, and loss is predictable, then downstream is predictable *through* loss. The capability space is the multi-dimensional generalization: instead of one scalar (loss) you carry a 3-vector (capabilities), which captures the cases where a single loss number is too coarse — two models with the same loss but different reasoning-versus-knowledge balance land at different points in capability space and get different (correct) forecasts.

It is worth being precise about the relationship between the loss-perspective result and the capability-space construction, because they are easy to conflate. The loss-perspective claim is a *one-dimensional* sufficiency statement: loss alone predicts downstream. The observational capability space is a *low-dimensional* refinement: loss is a strong summary, but it is not perfectly sufficient, because two models can share a loss while differing in how that loss is distributed across skills. A model trained heavily on code reaches a given loss with more coding competence and less general knowledge than a model trained on web text to the same loss. A scalar cannot see that; a 3-vector can. So the capability space does not contradict the loss perspective — it generalizes it, keeping the "smooth intermediate" idea but giving the intermediate enough dimensions to resolve the cases where one number washes out a real difference. In the limit where all models share a recipe, the capability space collapses toward the single loss axis and the two methods agree.

This is also why the field did not simply stop at the loss-to-error map. The map is the cheapest tool when you control the recipe and can measure loss on a relevant distribution. The capability space is the tool when you want to compare *across* recipes you do not control — the public-model setting — where there is no single comparable loss number to begin with, because everyone measures loss on a different validation set with a different tokenizer. The two methods are complements selected by what is comparable in your situation: a shared loss axis, or a shared capability space.

## 6. The assumptions you are betting on

**Senior rule of thumb: every forecast is a bet that the structure you fit on cheap models still holds on the expensive one. Name the assumption before you wire the result into a decision.**

These methods are powerful, but powerful is not the same as safe, and the difference is entirely in the assumptions. Here are the load-bearing ones, stated plainly, because a forecast you cannot stress-test is a forecast you cannot trust.

**Assumption 1: the link is stationary.** The logistic (or exponential, or power-law) link learned on weaker models is assumed to be the same link the strong model obeys. This usually holds because the link reflects the *metric's* structure (boundedness, multiplicativity), not the model's, and the metric does not change. But a metric whose scoring rule effectively changes with capability — e.g. a grader that gets stricter for stronger models, or a task whose difficulty distribution shifts as models start attempting harder instances — can break it. Check that the link's residuals are flat across the capability range, not just small on average.

**Assumption 2: the capability space is complete enough.** Three principal components capturing 97% of variance is excellent, but the missing 3% is not random noise for every task — for a capability that lives almost entirely in a discarded component, the K=3 fit will be blind to it. If your target metric is exotic (a narrow skill no standard benchmark probes), it may not project cleanly onto the three retained axes, and the forecast will be confidently mediocre. The defense is to check that your target correlates well with the retained components on the models you *can* measure before trusting an extrapolation.

**Assumption 3: no regime change between fit and target.** Every extrapolation assumes the functional form does not break between the last fitted point and the target. This is the [broken-neural-scaling-law](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws) caveat in its most general form: if there is a phase transition, a saturation, or a double-descent dip in the gap you are extrapolating across, a single smooth form will miss it. The honest move is to keep the extrapolation reach short (one step beyond the fit set) and to treat any prediction made across more than an order of magnitude of capability as a wide interval, not a number.

**Assumption 4: the inputs are clean.** Observational laws are only as good as the public scores they ingest. Benchmark contamination, inconsistent evaluation harnesses, and self-reported numbers all inject bias. The capability space launders some of this through PCA (idiosyncratic errors land in the discarded components), but systematic contamination that correlates with the capability axes does not launder out — it tilts them. Prefer benchmarks evaluated under a consistent harness, and be skeptical of scores from models likely to have trained on the test.

None of these assumptions is fatal; each is checkable, and the checks are cheap relative to the run. The discipline is to *do* the checks rather than to admire the point forecast. A forecast with a named, tested assumption is engineering; a forecast with a hidden assumption is a guess in a lab coat.

## 7. How to apply this in practice

**Senior rule of thumb: build the cheapest forecast that answers your actual decision, attach an honest interval to it, and validate the method on held-out models before you trust it on the run you cannot afford.**

Here is the concrete playbook, in the order you should reach for the tools.

### Step 0: Do the economics first

Before any modeling, write down the decision the forecast informs and what it is worth. This is the step teams skip, and it is the step that makes the rest worth doing. Suppose you are weighing a 70B run that will cost, in round numbers, on the order of \$3M of GPU time, and the question is whether it will clear an agentic-success bar your product requires. The value of the forecast is not academic — it is the expected cost of being wrong. If launching a run that misses the bar wastes the \$3M (plus weeks of calendar time and the opportunity cost of the cluster), and a forecast that costs \$40 of API calls and an engineer's afternoon can shift your probability of that mistake from, say, 30% to 10%, the expected savings are on the order of `0.20 × \$3M = \$600,000` for a \$40 spend. That is a five-orders-of-magnitude return, and it is why every serious lab now forecasts before it launches. The forecast does not need to be perfect; it needs to be cheap and better than a gut call, and it is overwhelmingly both.

The asymmetry is the point. The observational method costs **tens of dollars**; the over-training method costs a small ladder of runs (thousands to low tens of thousands of dollars, depending on your smallest viable models); the run they gate costs millions. Even the most expensive forecasting method is a rounding error against the decision it informs. Treat the forecast as insurance you are nearly always underpaying for.

### Step 1: Decide what you are forecasting and why

The method depends on the decision. "Should we launch the 70B run?" needs a suite-average accuracy forecast with an interval. "Will the 70B run be able to do agentic tool use?" is a specific-capability question that suits Ruan's observational method (predict agentic from non-agentic). "Will it cross the threshold for multi-step arithmetic?" is an emergence question that suits Snell or PassUntil. Pick the target metric first.

### Step 2: Build an observational fit for free

Before training anything, do the Ruan procedure on public data. Pull benchmark scores for as many models as you can across families and sizes, standardize, run PCA, and confirm you recover a low-dimensional capability space (you should see something like K=3 PCs dominating). Then fit the logistic-linked regression from capability to your target metric using only the weaker models, and predict the stronger ones. This is the cheapest possible sanity check and it costs, as advertised, tens of dollars. Here is the skeleton:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# rows = models, columns = benchmark scores (MMLU, ARC, GSM8K, HumanEval, ...)
scores = pd.read_csv("public_model_scores.csv", index_col="model")
target = scores.pop("agentic_success")        # the metric we want to forecast
log_compute = np.log(scores.pop("flops").values)

# 1. Standardize benchmarks and extract the capability space.
X = StandardScaler().fit_transform(scores.values)
pca = PCA(n_components=3).fit(X)
S = pca.transform(X)                            # capability vectors, K=3
print("variance explained:", pca.explained_variance_ratio_.sum())  # ~0.97

# 2. Logistic link: linearize the (0,1) metric against capability.
eps = 1e-3
y = np.clip(target.values, eps, 1 - eps)
z = np.log(y / (1 - y))                         # inverse sigmoid of the metric

# 3. Fit beta, alpha on WEAK models only, then predict the strong one.
weak = log_compute < np.quantile(log_compute, 0.8)
link = LinearRegression().fit(S[weak], z[weak])
z_hat = link.predict(S[~weak])
y_hat = 1 / (1 + np.exp(-z_hat))                # back through the sigmoid
print("predicted vs actual (held-out strong models):")
print(np.c_[y_hat, target.values[~weak]])
```

The two lines that matter are the `inverse sigmoid` and its inverse: that is the link doing its job. If the held-out strong models' predictions track their actuals, your capability space generalizes and you can trust a forecast one step beyond your fit set.

### Step 3: If you can afford small runs, fit the over-training law

If you have the budget for a small scaling ladder in *your* recipe (your data, your tokenizer, your architecture), train a handful of small over-trained models, fit `L(C, M)`, fit the loss-to-error map on the same models, and chain them. This grounds the forecast in your actual setup rather than the population of public models, which matters when your recipe is unusual.

```python
from scipy.optimize import curve_fit
import numpy as np

# small over-trained runs: compute C, token multiplier M=D/N, measured loss, error
C, M, loss, err = load_probe_runs()            # e.g. 0.011B-1.4B models

# 1. Over-training loss law  L(C,M) = E + (a*M^eta + b*M^-eta) * C^-eta
def loss_law(X, E, a, b, eta):
    C, M = X
    return E + (a*M**eta + b*M**(-eta)) * C**(-eta)
(p_loss, _) = curve_fit(loss_law, (C, M), loss, p0=[1.5, 1e3, 1e3, 0.08], maxfev=100000)

# 2. Loss-to-error map  Err(L) = eps - k*exp(-gamma*L)
def err_map(L, eps, k, gamma):
    return eps - k*np.exp(-gamma*L)
(p_err, _) = curve_fit(err_map, loss, err, p0=[0.6, 0.5, 0.85], maxfev=100000)

# 3. Forecast a large over-trained target (big C, big M) two steps out.
C_target, M_target = 1e22, 320.0
L_hat = loss_law((C_target, M_target), *p_loss)
err_hat = err_map(L_hat, *p_err)
print(f"predicted loss {L_hat:.3f} -> predicted avg error {err_hat:.3f}")
```

### Step 4: Attach an interval, and validate by leave-one-out

A point forecast with no interval is malpractice at these stakes. The honest way to get an interval is **leave-one-out across the strongest models you do have**: refit the method excluding each strong model in turn, predict it, and look at the spread of (predicted − actual). That residual spread is your empirical interval, and it is far more trustworthy than a parametric confidence band, because it directly measures how well the method extrapolates one step beyond its fit set — which is exactly the operation you are about to perform on the run you cannot afford.

### Step 5: Flag the non-monotone tasks

Before you trust any single-task forecast, plot that task's score against compute for the models you have. If it is monotone and smooth (possibly after a link), extrapolate it. If it is non-monotone — a dip, a U-shape, a Hindsight-Neglect-style reversal — *do not* trust a single power-law extrapolation; either model it with a [broken-neural-scaling-law](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws) form (which can express the break) or treat that task's forecast as unreliable and decide on the suite average instead.

## 8. Case studies and failure modes

The methods are clean on paper; the field learned their edges the hard way. Here are the instructive episodes, each with the symptom, the wrong first guess, the actual cause, and the lesson.

### 1. The 5-digit multiplication cliff

The symptom: 3-digit and 4-digit arithmetic accuracy is flat at zero across a whole family of small models, then jumps sharply at a particular scale — the canonical emergent ability from Wei 2022, switching on somewhere in the 10^22 to 10^24 train-FLOP band depending on the task. The wrong first guess was "the model suddenly learned to multiply." The actual cause, per Schaeffer 2023, is that exact-match accuracy is `~p^N` in the per-token correctness `p`; `p` was climbing smoothly the whole time, but exact-match hid it until `p^N` stopped being negligible. The lesson: the capability was always predictable in token-edit distance or per-token accuracy; only the *metric* was emergent. For forecasting, this is the founding example of "route around the discontinuous metric."

### 2. GPT-4's HumanEval forecast that held

The symptom — in the good sense — was that OpenAI publicly committed to a HumanEval pass-rate prediction for GPT-4 derived from models using up to 1,000× less compute, and it landed. The non-obvious engineering was that they did not extrapolate pass rate; they extrapolated `−E[log pass_rate]`, a smooth power law in compute, and transformed back. The lesson: even at the frontier, the recipe is "find the smooth intermediate." A naive team that extrapolated raw pass rate would have seen a jumpy curve and concluded coding was unpredictable.

### 3. Hindsight Neglect, the task that refused

The symptom: a specific BIG-Bench task on which GPT-4-era models scaled *inversely* over part of the range — bigger models did worse before recovering. The wrong first guess was that the predictable-scaling methodology was broken. The actual cause was that this task is genuinely non-monotone in scale, and a single power law cannot represent a U-shape. The lesson, baked into every responsible forecast since: predictability is a strong tendency, not a theorem; screen for non-monotone tasks and either model the break explicitly or exclude them from point forecasts.

### 4. Agentic performance from non-agentic scores

The symptom that should have been impossible: Ruan predicted GPT-4's agentic task performance using only the *non-agentic* benchmark scores of weaker models. The naive expectation is that you cannot forecast a capability you have no direct measurements of in strong models. The actual mechanism is that agentic success is largely a function of position in the shared 3-dimensional capability space, which the non-agentic benchmarks already pin down. The lesson: capabilities are correlated through a low-dimensional latent, so a metric you have never measured in strong models can still be forecast from metrics you have. The practical upshot for anyone building agents: you do not need a fleet of expensive agentic evaluations on frontier models to estimate whether a planned model will be a competent agent — its position on the cheap, standard benchmarks already carries most of the signal, because agentic competence is not orthogonal to general and reasoning competence, it is a (mostly linear) combination of them.

### 5. The over-trained model that broke the vanilla loss law

The symptom: a small model trained at 32× the Chinchilla-optimal token count had a loss that a compute-only scaling law mis-predicted. The wrong first guess was that over-training "doesn't follow scaling laws." The actual cause was that the token multiplier `M = D/N` is a second axis the compute-only law ignores; Gadre's `L(C, M)` with its `(a·M^η + b·M^−η)` bracket captures it and predicts the over-trained loss from ~300× less compute. The lesson: when your regime (heavy over-training, common today for inference economics) differs from the regime the law was fit in, add the missing axis rather than abandoning the framework.

### 6. The column of zeros that PassUntil turned into signal

The symptom: a 2.4B model scored a flat zero on a code-generation task in a normal one-sample evaluation, so there was nothing to fit. The wrong first guess was "the capability is absent below this scale." The actual cause was insufficient measurement resolution: the per-sample success probability was positive but tiny. PassUntil sampled until success and estimated that probability, turning zero into 0.0003-style numbers, and then fit a scaling law that predicted the 2.4B score to 0.05% before training. The lesson: a "zero" is often an evaluation-resolution artifact, and you can buy resolution with decode-time compute.

### 7. The finetuning shift that revealed a latent skill

The symptom: a capability appeared absent in every small base model a team could afford, defeating direct extrapolation. The non-obvious fix, from Snell 2024, was to finetune the small models on a little task data and watch the emergence point move earlier; fitting an emergence law to that movement let them extrapolate where the *base* model would acquire the skill, about 4× compute ahead. The lesson: finetuning lowers the expression threshold for a latent capability, so a finetuned ladder can surface a trend the base models hide.

### 8. The cross-domain loss-to-error map that drifted

The symptom: a loss-to-error map fit on a general benchmark suite over-predicted accuracy when applied to a narrow domain (competition math). The wrong first guess was that the exponential map was wrong. The actual cause was distribution shift: the constants `ε, k, γ` are suite-specific, and the validation loss measured on web text is only loosely coupled to math accuracy. The lesson: re-fit the map per target domain, and never assume a map calibrated on one suite transfers to a distant one.

### 9. The per-task forecast that was wide when the average was tight

The symptom: the suite-average accuracy forecast was excellent, but individual-task forecasts from the same fit were all over the place. The wrong first guess was that the method was unreliable. The actual cause is statistical: averaging over a suite cancels per-task threshold noise, so the average is far better-determined than any component. The lesson: forecast the average for go/no-go decisions; treat per-task numbers as wide intervals, not point estimates. The intuition is the same as the law of large numbers — each task contributes independent threshold noise, and averaging `T` tasks shrinks that noise by roughly `√T`. A 30-task suite average is therefore about five times tighter than any single task's forecast, which is exactly the precision difference teams observe in practice.

### 10. The leaderboard contamination that flattered a forecast

The symptom: an observational fit predicted a new model's MMLU far better than its actual capability warranted, and the residuals on recent models were suspiciously small. The wrong first guess was that the capability space had simply gotten more accurate over time. The actual cause was benchmark contamination — some of the newer public models had MMLU-like data in their training sets, inflating their measured scores relative to their true capability, which distorted the capability axes and made the fit look artificially good on contaminated models while it would mislead on a clean one. The lesson: observational laws inherit the pathologies of their inputs. Public benchmark scores are not pristine; screen for contamination, prefer benchmarks and held-out variants less likely to be in training data, and be suspicious when residuals are *too* good on exactly the models most likely to be contaminated.

### 11. The single-family fit that did not generalize across recipes

The symptom: a team built a beautiful capability-to-benchmark fit using only one model family's ladder, then applied it to a different family and the predictions were biased. The wrong first guess was that the second family was "just worse." The actual cause was that the capability-to-benchmark mapping had absorbed family-specific quirks (a tokenizer that helped one benchmark, a data mix that favored another), so the learned `β` did not transfer. The lesson: fit the capability-to-benchmark and link regressions across *many* families, not one, so the universal part of the mapping is separated from family idiosyncrasy. The whole power of the observational approach comes from pooling families; a single-family fit throws that away and is really just a within-family scaling law wearing PCA's clothes.

## 9. When to reach for these methods, and when not to

### Reach for observational and downstream scaling laws when

- You face a **large, expensive training decision** and want the go/no-go to rest on arithmetic rather than intuition. This is the canonical use, and the cost-benefit is overwhelming: a forecast that costs tens of dollars to gate a run that costs millions.
- You want to forecast a **specific downstream capability** (coding, agentic, reasoning) and you have public benchmark scores across many models — use Ruan's observational capability space.
- You can train a **small ladder in your own recipe** and want a forecast grounded in your data and architecture — use Gadre's over-training law plus the loss-to-error map.
- The worry is a **threshold capability** that reads as zero in affordable models — use PassUntil's resolution boost or Snell's finetuning shift.
- You need to set **realistic expectations** for stakeholders before a run, with an honest interval from leave-one-out validation.

### Skip them, or use them with heavy caveats, when

- The task is **genuinely non-monotone** (Hindsight-Neglect-style inverse or U-shaped scaling). A single power law will be confidently wrong; use a broken-scaling-law form or exclude the task from point forecasts.
- You are forecasting a **single specific task** rather than a suite average and need a tight point estimate — the per-task variance is large; prefer the average.
- The target is **far outside your fit range** (more than ~1 order of magnitude of capability beyond your strongest model). Extrapolation error grows with reach; the observational method is most trustworthy one step beyond the fit set.
- Your **recipe is radically novel** (new architecture, new data distribution, new objective) such that the public-model capability space or the loss-to-error constants may not transfer. Re-fit on your own runs.
- You are tempted to forecast a metric with a **discontinuous scoring rule directly**. Do not. Always go through the smooth intermediate and apply the link at the end — that is the one rule this entire literature agrees on.

> The forecast is never the goal. The goal is a better decision about where to spend the next eight figures of compute, made before you spend it. These methods do not make the run cheaper; they make the decision to launch it accountable.

A final framing. The arc from Kaplan to Ruan is the arc from "we can predict the loss" to "we can predict the product." Loss was never the thing anyone cared about — it was the thing that happened to be smooth. The intellectual content of the last two years has been the realization that you can manufacture smoothness on demand: through a capability vector, through a loss bridge, through a link function, through evaluation resolution. Once you can make the quantity smooth, the old power-law machinery does the rest, and the cliff that used to look like magic turns out to be a sigmoid you forgot to take the inverse of.

## Further reading

- Ruan, Maddison, Hashimoto. "Observational Scaling Laws and the Predictability of LM Performance." arXiv:2405.10938 (NeurIPS 2024 spotlight).
- Gadre et al. "Language models scale reliably with over-training and on downstream tasks." arXiv:2403.08540.
- OpenAI. "GPT-4 Technical Report" (Predictable Scaling). arXiv:2303.08774.
- Snell et al. "Predicting Emergent Capabilities by Finetuning." arXiv:2411.16035.
- Hu et al. "Predicting Emergent Abilities with Infinite Resolution Evaluation" (PassUntil). arXiv:2310.03262 (ICLR 2024).
- Schaeffer, Miranda, Koyejo. "Are Emergent Abilities of LLMs a Mirage?" arXiv:2304.15004 (NeurIPS 2023).
- Du et al. "Understanding Emergent Abilities from the Loss Perspective." arXiv:2403.15796 (NeurIPS 2024).
- Sibling posts: [emergent abilities](/blog/machine-learning/scaling-laws/emergent-abilities-scaling), [Chinchilla compute-optimal scaling](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling), [broken neural scaling laws](/blog/machine-learning/scaling-laws/broken-neural-scaling-laws).

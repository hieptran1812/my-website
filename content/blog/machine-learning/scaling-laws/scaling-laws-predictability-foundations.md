---
title: "Scaling laws, from scratch: why loss is predictable before you train"
date: "2026-06-15"
description: "Learn what a power law is, why it is a straight line on log-log axes, the three regions of a learning curve, and how to forecast a model's loss from cheap small runs before committing a large training budget."
tags: ["scaling-laws", "power-laws", "log-log", "generalization-error", "hestness", "rosenfeld", "predictability", "compute-budget", "deep-learning", "machine-learning", "training-economics"]
category: "machine-learning"
subcategory: "Scaling Laws"
author: "Hiep Tran"
featured: true
readTime: 53
---

Here is a claim that sounds like marketing and is actually a measured, reproducible fact: you can predict how good a neural network will be **before you train it**. Not perfectly, not for every metric you care about, but for the one that matters most during pre-training — the loss — you can fit a curve on a handful of cheap experiments and read off, with a useful error band, what a run a thousand times larger will achieve. The forecast costs less than one percent of the run it predicts.

That single fact reorganizes how serious labs spend money. If loss is predictable, then the question "should we spend \$2M of GPU time on this run?" stops being a gut call and becomes arithmetic. You run three small models, fit a line, extend the line, and the line tells you whether the expensive run lands where you need it. This post is the on-ramp for an entire series about that line: where it comes from, why it is straight, what its slope means, and how the field learned to trust it.

> [!important]
> **The five things to take away**
> - A **power law** $y = a\,x^{-c}$ becomes a **straight line on log-log axes** with slope $-c$. That straightness is the entire reason extrapolation works.
> - **Doubling $x$ multiplies $y$ by the same constant factor** ($2^{-c}$) no matter where you start. Exponential and linear functions do not have this property, which is why they do not extrapolate cleanly.
> - A learning curve has **three regions**: a small-data random-guess plateau, a straight power-law middle, and an irreducible-error floor (the Bayes limit). Only the middle is predictable, and that is where you measure the slope.
> - **Hestness 2017** showed generalization error follows $\varepsilon(m) \approx \alpha\, m^{\beta_g}$ across language, translation, vision, and speech; **Rosenfeld 2019** made it *constructive* — fit on small scale, extrapolate to large scale, waste less compute.
> - The economic payoff: **fit on cheap runs, forecast the expensive run, decide before you spend.** The one number to remember is that measured language-model exponents are shallow — around $\beta_g \approx -0.07$ — so you need roughly **10x the data to cut error by about 15%**.

The diagram below is the mental model for the whole series. It is a learning curve drawn on log-log axes, split into the three regions we will keep returning to. Spend a minute on it now; every later post — Kaplan, Chinchilla, data-constrained scaling, inference-aware scaling — is a refinement of this picture.

![A learning curve drawn on logarithmic axes showing three regions: a flat random-guess plateau on the left, a straight descending power-law line in the middle, and a flat irreducible-error floor on the right](/imgs/blogs/scaling-laws-predictability-foundations-1.png)

The left of that curve is the small-data region, where the model has too few examples to do better than chance, so the error hugs a ceiling. The right is the irreducible floor: even an infinitely large dataset cannot push error below the entropy of the task itself. The interesting part — the part this entire field is built on — is the straight middle. In that region the error obeys a power law, the log-log plot is a straight line, and a straight line is the easiest object in mathematics to extend. That is the whole game.

## 1. Start with the function, not the model

**Senior rule of thumb: before you reason about transformers, get fluent in the four function shapes, because scaling laws are a statement about one of them.**

Forget neural networks for a moment. We are going to compare four ways a quantity $y$ can depend on a quantity $x$, because the entire reason "scaling laws" are useful is that loss happens to follow one specific shape and not the others.

The four shapes are:

- **Constant:** $y = a$. Nothing changes. Boring, but it is exactly what the irreducible floor looks like, so keep it in mind.
- **Linear:** $y = a - b\,x$. Each unit of $x$ subtracts a fixed amount from $y$. Double $x$ and the *change* in $y$ depends on where you started.
- **Exponential:** $y = a\,r^{x}$ with $0 < r < 1$ for decay. Each unit of $x$ *multiplies* $y$ by $r$. This decays fast and explodes fast.
- **Power law:** $y = a\,x^{-c}$. Each time you *multiply* $x$ by a constant, $y$ gets multiplied by a constant. This is the slow, scale-free decay that loss curves follow.

The distinction that matters is not "how fast does it go down" — it is "what stays constant as $x$ grows." For the power law, the constant thing is the *ratio* of outputs when you take a *ratio* of inputs. Write it out. If $y_1 = a\,x_1^{-c}$ and we double the input to $x_2 = 2x_1$, then

$$
\frac{y_2}{y_1} = \frac{a\,(2x_1)^{-c}}{a\,x_1^{-c}} = 2^{-c}.
$$

The $x_1$ cancels completely. The fraction by which $y$ drops when you double $x$ is $2^{-c}$ — the *same* whether you double from 1 million to 2 million or from 1 billion to 2 billion examples. With $c = 0.07$ (a realistic language-model exponent), $2^{-0.07} \approx 0.953$, so each doubling of data shaves about 4.7% off the loss above the floor, forever, at every scale. That invariance is what makes a power law forecastable.

Now do the same for the exponential. If $y_1 = a\,r^{x_1}$ and $x_2 = 2x_1$, then

$$
\frac{y_2}{y_1} = r^{x_2 - x_1} = r^{x_1},
$$

which still depends on $x_1$. The fractional change when you "double" depends on where you are on the axis. An exponential has a characteristic scale — a half-life — baked into it, so what happened between $x = 1$ and $x = 2$ tells you nothing reliable about what happens between $x = 1000$ and $x = 2000$. Linear is worse: doubling $x$ subtracts $b\,x_1$, which grows without bound. Neither shape lets you measure a behavior cheaply at small scale and trust it at large scale.

![A comparison matrix showing how power-law, exponential, and linear functions each respond when x doubles, what they look like on log-log axes, and how reliably each extrapolates](/imgs/blogs/scaling-laws-predictability-foundations-4.png)

The matrix above is the punchline of this section in one frame: only the power law turns "multiply the input" into "multiply the output by a fixed factor," and only the power law is straight on log-log axes. Hold onto the words "straight on log-log," because that is the next thing we have to earn.

### 1.1 Why log-log turns a power law into a straight line

Take logs of both sides of $y = a\,x^{-c}$. Use any base; natural log is fine:

$$
\ln y = \ln a - c\,\ln x.
$$

Rename $Y = \ln y$ and $X = \ln x$. Then

$$
Y = \ln a - c\,X,
$$

which is the equation of a straight line in the variables $(X, Y)$ with slope $-c$ and intercept $\ln a$. So when you plot $\ln y$ against $\ln x$ — that is, put both axes on a log scale — a power law is *exactly* a straight line. The slope you read off the plot is $-c$; the height where the line crosses $X = 0$ is $\ln a$. This is not an approximation or a visualization trick; it is algebra.

Contrast: on log-log axes an exponential $y = a\,r^x$ becomes $\ln y = \ln a + (\ln r)\,x = \ln a + (\ln r)\,e^{X}$, which is curved because $x = e^X$ appears, not $X$. And a linear function $y = a - bx$ becomes $\ln y = \ln(a - b\,e^{X})$, which is also curved and even goes to $-\infty$ at finite $X$. Power laws are the unique family that is straight on log-log, and that is precisely why a researcher squinting at a log-log plot and seeing a straight line immediately reaches for a power-law fit.

![Two side-by-side plots of the same power law, curved and hard to read on linear axes on the left, and a clean straight line on logarithmic axes on the right](/imgs/blogs/scaling-laws-predictability-foundations-2.png)

The benefit is visible the moment you put the two plots side by side, as above. On the left — linear axes — the curve drops steeply and then crawls, and if you tried to guess where it lands ten times further to the right, you would be guessing. On the right — log-log axes — the identical data is a straight line, and extending a straight line is something a ruler can do. Every scaling-law paper you will ever read shows the right-hand plot. Now you know why: it is the only frame in which the relationship is simple enough to extrapolate.

### 1.2 A worked numeric example you can do by hand

Let me make this concrete with numbers, because "fit a line and extend it" is abstract until you have actually done the arithmetic once.

Suppose you train three small language models and measure validation loss (above the floor, for now) at three dataset sizes:

| Dataset size $m$ (tokens) | Measured excess loss $\varepsilon$ |
|---|---|
| $1 \times 10^{6}$ | 0.500 |
| $4 \times 10^{6}$ | 0.402 |
| $16 \times 10^{6}$ | 0.323 |

Take logs (natural log) of both columns:

| $\ln m$ | $\ln \varepsilon$ |
|---|---|
| 13.816 | $-0.693$ |
| 15.202 | $-0.911$ |
| 16.588 | $-1.130$ |

The points $(\ln m, \ln \varepsilon)$ fall on a line. The slope between the first and last point is

$$
\text{slope} = \frac{-1.130 - (-0.693)}{16.588 - 13.816} = \frac{-0.437}{2.772} \approx -0.158.
$$

So $-c \approx -0.158$, i.e. $c \approx 0.158$, and the intercept follows from $\ln \varepsilon = \ln a - c\,\ln m$ at any point: $\ln a = -0.693 + 0.158 \times 13.816 \approx 1.490$, so $a = e^{1.490} \approx 4.44$. Our fitted law is

$$
\varepsilon(m) \approx 4.44 \, m^{-0.158}.
$$

Now extrapolate. What does this predict at $m = 16 \times 10^{9}$ tokens — a thousand times more data than the largest run we paid for? Plug in:

$$
\varepsilon(1.6 \times 10^{10}) \approx 4.44 \times (1.6 \times 10^{10})^{-0.158}.
$$

Compute the exponent: $\ln(1.6 \times 10^{10}) = 23.498$, times $-0.158$ gives $-3.713$, and $4.44 \times e^{-3.713} = 4.44 \times 0.0244 \approx 0.108$. So three runs costing a few thousand dollars predict that a run with a thousand times more data lands near an excess loss of **0.108**, down from 0.323. We never trained the big model. We extended a ruler.

That is the entire mechanism, and the rest of this series is about doing it carefully: which axis to extrapolate along (data, parameters, or compute), how the floor changes the fit, where the law breaks, and how to allocate a budget once you trust the forecast.

### 1.3 The constant-factor property, side by side with the impostors

To cement why the power law is special and the other shapes are impostors for extrapolation, walk the same numerical experiment through all three. Start each at $x = 10$ with $y = 1$, then multiply $x$ by 10 repeatedly and watch what happens to $y$.

For the power law $y = a\,x^{-c}$ with $c = 0.5$ (steep, for visibility), each tenfold increase in $x$ multiplies $y$ by $10^{-0.5} \approx 0.316$. So the sequence of $y$ values at $x = 10, 100, 1000, 10000$ is $1, 0.316, 0.1, 0.0316$ — a clean geometric sequence with ratio $0.316$. Measure that ratio once on the cheap end and you know it forever; the fifth value follows from the fourth by the same factor as the second followed from the first. That is the property you are buying.

For an exponential decay $y = a\,r^{x}$, the same experiment is a disaster. If you tune $a$ and $r$ so that $y = 1$ at $x = 10$ and $y = 0.316$ at $x = 100$, then at $x = 1000$ the value is not $0.1$ — it is $r^{900}$ times the $x = 100$ value, an astronomically tiny number, because the exponential's decay accelerates without bound on a multiplicative $x$-axis. The behavior between your two cheap measurements tells you almost nothing about the behavior three decades out. For a linear function $y = a - b\,x$, it is worse still: extend it far enough and $y$ goes negative, which is meaningless for a loss. Neither impostor has a constant ratio under multiplication of the input, so neither lets you measure cheap and trust far. The power law is the unique shape where the cheap measurement is the expensive prediction, and that uniqueness is the reason this entire field exists.

## 2. The three regions of a real learning curve

**Senior rule of thumb: the power law only holds in the middle of the curve; do not measure your slope in the random-guess region or near the floor, or you will fit a number that means nothing.**

The clean $y = a\,x^{-c}$ story is true *within a region*. A real learning curve — error as a function of training-set size $m$ — has three regions, and only the middle one is a power law. Hestness and colleagues named these regions in 2017, and the names have stuck because they map onto real failure modes you will hit.

**Region 1, the small-data / random-guess region.** When $m$ is tiny, the model has nowhere near enough examples to generalize, so its error sits close to what you would get by guessing. For a balanced $k$-way classifier that ceiling is roughly $1 - 1/k$; for a language model it is the loss of predicting the unigram distribution, or worse. In this region the curve is flat-ish and *above* the power-law line, because adding a few more examples to a model that has essentially learned nothing does not help much yet. If you fit a power law here, you will measure a slope dominated by the transition out of chance, not by the true scaling behavior.

**Region 2, the power-law region.** This is the predictable middle. Here error genuinely follows $\varepsilon(m) \approx \alpha\, m^{\beta_g}$, the log-log plot is straight, and the slope $\beta_g$ is a stable property of the problem. This is the region you measure in and extrapolate from. Hestness's central empirical finding was that this region exists, is wide (often several orders of magnitude), and has a slope that is remarkably insensitive to architecture choices.

**Region 3, the irreducible-error floor.** As $m \to \infty$, error does not go to zero. It approaches a floor $E$ set by the irreducible noise in the task: ambiguous labels, genuine entropy in language, sensor noise in speech. This is the **Bayes error** — the loss of the best possible predictor given the information available. Past a certain dataset size the curve bends away from the straight power-law line and flattens toward $E$, and more data buys you almost nothing. Fitting a pure power law across this bend will underestimate how good your model can get, because the pure power law keeps going down while reality levels off.

![A learning curve on log-log axes annotated with three labeled regions, a chance ceiling on the left, a straight power-law middle with slope beta, and a flattening approach to the Bayes floor on the right](/imgs/blogs/scaling-laws-predictability-foundations-1.png)

I am deliberately showing the same mental-model figure again, because everything in this section is a tour of it. The amber box on the left is Region 1, the blue box in the middle is Region 2 — the one you extrapolate — and the red box on the right is Region 3, the floor. The practical discipline is: collect enough small runs that at least two or three of them land squarely in the blue region, fit only those, and treat the floor as a separate parameter to estimate rather than something the power law will discover on its own.

### 2.1 Modeling the floor: the parametric form you will see everywhere

The way to handle the floor honestly is to bake it into the functional form. Instead of $\varepsilon = a\,m^{-c}$, write

$$
L(m) = E + a\,m^{-c},
$$

where $E$ is the irreducible floor and $a\,m^{-c}$ is the *reducible* part that you can drive down with more data. This is the one-variable ancestor of the equation that runs through this whole series. When we get to Chinchilla in a later post, the backbone equation will be

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}},
$$

where $N$ is the number of parameters and $D$ is the number of training tokens. Read it left to right: an irreducible floor $E$, plus a term that shrinks as the model grows ($A/N^\alpha$), plus a term that shrinks as you feed more data ($B/D^\beta$). It is the same shape as our one-variable law, generalized to two knobs. If the single equation $L(N,D) = E + A/N^\alpha + B/D^\beta$ looks intimidating now, it should look obvious by the end of the [Chinchilla post](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling): it is "floor plus one power law per axis."

Two practical consequences fall out of adding the floor:

- **The slope you measure flattens near the floor even if the true exponent is constant.** Because the *observed* loss is $E + a\,m^{-c}$, its log-log slope is not constant — it is $-c$ far above the floor and bends toward $0$ as $a\,m^{-c}$ becomes small compared to $E$. So a "slope" read off a single segment near the floor undersells the true $c$. Always fit $E$, $a$, and $c$ jointly; do not eyeball the slope of the tail.
- **The floor sets the ceiling on ROI.** If you are already within, say, 5% of $E$, then doubling your data again is nearly worthless — the reducible part is almost gone. The floor is where "more data" stops being a good investment, and knowing roughly where it sits is half of any data-budget decision.

### 2.2 A short tangent on what "loss" even is here

For language models the loss is cross-entropy in nats (or bits) per token: the negative log-probability the model assigns to the true next token, averaged. It is bounded below by the entropy of the text-generating process, which is the floor $E$. This is why language-model scaling laws are usually stated in loss rather than accuracy: loss is a smooth, continuous quantity that behaves like a power law over many orders of magnitude, whereas downstream task accuracy is jumpy, capped at 100%, and full of thresholds. A subtle but important point that recurs throughout the series: **the smooth, predictable quantity is the pre-training loss, and the jumpy, surprising quantity is downstream capability.** Scaling laws forecast the former cleanly; they forecast the latter only loosely, which is the source of most "emergent ability" debates. We forecast loss because loss is the thing that is forecastable.

### 2.3 How to tell which region you are actually in

The regions are not labeled in the wild — you have to diagnose which one each of your runs sits in, and getting this wrong is the most common way a fit goes silently bad. Three practical tests:

- **Distance from chance.** Compute the chance-level loss for your task explicitly: the unigram-prediction loss for a language model, $\ln k$ for a balanced $k$-way classifier. If a run's loss is within a small factor of chance, it is in Region 1 and must be excluded from the power-law fit. A model that is barely beating a bigram baseline has not entered the predictable middle yet.
- **Local slope stability.** Compute the log-log slope between *consecutive* pairs of runs. In Region 2 those pairwise slopes are roughly constant; in Region 1 they steepen as you climb out of chance, and near the floor they flatten toward zero. If your pairwise slopes are still changing monotonically across your whole ladder, you have not bracketed the power-law region and you need runs at more scales.
- **Residual sign pattern.** After fitting a pure power law, look at the residuals. Random scatter means you are in the clean middle. A systematic pattern — small runs above the line, large runs above the line again (a smile shape) — is the signature of fitting across both a Region-1 transition and a Region-3 bend with a model that has neither, which means you should switch to the floor-aware (and ideally transition-aware) form.

The discipline these tests enforce is simple: never report a slope until you have evidence that at least three of your runs live in the same straight-line region. A slope averaged across a transition is a number with no predictive meaning, and it will betray you the moment you extrapolate.

### 2.4 The floor is the Bayes error, and why that is liberating

It is worth dwelling on what the floor $E$ actually is, because it reframes "diminishing returns" from a disappointment into a target. $E$ is the **Bayes error**: the loss of the theoretically optimal predictor that has perfect knowledge of the data-generating distribution but still cannot do better because the task has genuine, irreducible uncertainty. For next-token prediction on natural text, $E$ is essentially the entropy of language itself — there are many plausible next words, and no model, however large, can assign probability 1 to the one that happened to be written. For a labeling task with noisy annotators, $E$ is set by the rate at which the labels themselves disagree.

The liberating consequence is that $E$ gives you a *finish line*. Without it, "lower loss" is an open-ended grind with no sense of how much is left. With it, the reducible part of your loss is $L - E$, and your real progress metric is how much of that reducible gap you have closed. A model at $L = 2.0$ when $E = 1.7$ has closed most of a $0.3$-nat reducible gap; a model at $L = 2.0$ when $E = 1.0$ has a full $1.0$ nat still to go and is nowhere near done. Two models with the same loss can be at completely different points in their journey depending on the floor. Estimating $E$ — even roughly, as a fitted parameter — is therefore not a technicality; it is what turns loss from an unbounded number into a percentage-of-possible-progress, which is exactly the framing a budget owner needs to decide when to stop.

## 3. Hestness 2017: deep learning scaling is predictable, empirically

**Senior rule of thumb: the slope is a property of the problem; the offset is a property of your engineering. Spend your cleverness on the offset and stop expecting it to bend the slope.**

The paper that turned "loss seems to follow a power law" from folklore into a careful empirical result is Hestness et al., *Deep Learning Scaling is Predictable, Empirically* (Baidu SVAIL, 2017). They trained models across four domains — language modeling, machine translation, image classification, and speech recognition — varied the training-set size $m$ over orders of magnitude, and asked whether generalization error follows a power law in $m$. It does. Across all four domains, in the power-law region, error fits

$$
\varepsilon(m) \approx \alpha\, m^{\beta_g},
$$

a straight line on log-log axes, with a domain-specific exponent $\beta_g < 0$.

The exponents they measured are worth memorizing roughly, because they recalibrate your intuition about how slow this all is. Naive theory (a simple bias-variance or kernel argument) predicts $\beta_g = -0.5$ — error should fall like $1/\sqrt{m}$, so four times the data halves the error. Reality is much shallower:

- **Language modeling:** $\beta_g \approx -0.06$ to $-0.09$. This is brutally slow. With $\beta_g = -0.07$, cutting error in half requires $2^{1/0.07} \approx 2^{14} \approx 16{,}000$ times more data. This is the single most important number in this post for anyone working on LLMs: language is a *shallow-exponent* problem, so progress comes from enormous multiplicative jumps in scale, not incremental ones.
- **Machine translation:** $\beta_g \approx -0.13$ to $-0.36$. Steeper than language modeling — data pays off faster.
- **Image classification and speech recognition:** roughly $-0.07$ to $-0.35$, depending on the specific task.

![A grid of measured scaling exponents across language modeling, machine translation, image classification, and speech recognition, each shown against the textbook minus one half ideal](/imgs/blogs/scaling-laws-predictability-foundations-6.png)

The grid above lines those numbers up. Notice two things. First, every measured exponent is *shallower* than the $-0.5$ ideal — the theory that predicts $1/\sqrt{m}$ is optimistic for deep nets on real data. Second, the exponents cluster by *domain*, not by *model*. That second observation is Hestness's most useful structural claim, and it deserves its own treatment.

### 3.1 Architecture changes the offset, rarely the exponent

Hestness found that swapping architectures within a domain — different depths, widths, attention variants, regularizers — moves the power-law line *up or down* but keeps it roughly *parallel*. In the equation $\varepsilon = \alpha\,m^{\beta_g}$, a better architecture lowers $\alpha$ (the offset) but leaves $\beta_g$ (the slope) essentially unchanged. On a log-log plot you see two parallel lines: the better model is the lower one, but both descend at the same rate.

![Two parallel descending lines on log-log axes representing a baseline and an improved architecture, with the same slope but a vertical offset between them](/imgs/blogs/scaling-laws-predictability-foundations-7.png)

This is one of the most quietly consequential facts in the field, so let me state the second-order implications plainly using the figure above. A new architecture buys you a *constant multiplicative head start* — equivalent to being handed a few times more data at every scale — which is genuinely valuable and is most of what "model improvements" deliver. But it does **not** change how fast you climb the data axis. If you are on the language-modeling line with $\beta_g = -0.07$, no clever block design moves you to $\beta_g = -0.3$. The slope is set by the structure of the problem and the loss; you rent a lower starting point, you do not buy a steeper descent.

The practical takeaway is a reallocation of where you spend research effort. Tweaking the architecture to shave the offset is worthwhile but bounded — you are competing for a constant factor. Moving along the data and compute axes is unbounded but slow and expensive. Knowing the slope tells you the exchange rate between the two: with $\beta_g = -0.07$, a 2x improvement in offset is worth roughly $2^{1/0.07} \approx 16{,}000$x in data, which is why a genuinely better architecture can be worth more than almost any feasible data increase — and also why, past a point, you cannot architect your way out of needing more scale.

### 3.2 Model size grows sublinearly with data

Hestness reported a second power law that matters for budgeting: the model size you *need* to stay on the optimal error curve grows sublinearly with data. If $s(m)$ is the number of parameters required at dataset size $m$, then

$$
s(m) \propto m^{\beta_p}, \qquad \beta_p \in [0.5, 1].
$$

In words: when you get 10x more data, you do not need 10x more parameters to make use of it — you need somewhere between $10^{0.5} \approx 3$x and $10$x more. This is the first hint of the *joint* scaling story that Kaplan and Chinchilla will formalize. Data and parameters scale together, but not one-to-one, and the exact exponent of that relationship is exactly the question that the later [Kaplan](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models) and [Chinchilla](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) posts argue about. For now, hold the shape: more data wants a bigger model, but sublinearly so.

### 3.3 Why the exponent is shallow: a sketch of the intuition

It is worth pausing on *why* deep-net exponents are worse than the $-0.5$ ideal, even though the full theory is its own track. The short version: the $-0.5$ rate assumes you are estimating a fixed-complexity target and your only enemy is sampling noise. Real tasks are not fixed-complexity. Language, for instance, has a long tail of rare constructions, idioms, facts, and reasoning patterns; each new order of magnitude of data mostly buys you slightly better coverage of an ever-rarer tail. The data manifold is high-dimensional and the useful signal is spread thinly across it, so each additional example overlaps heavily with what you already had. Heuristically, if the effective dimension of the problem is $d$, error tends to fall like $m^{-1/d}$-ish, and large $d$ gives a shallow exponent. You do not need this theory to *use* scaling laws — the exponent is something you measure, not derive — but it explains why "just add data" is a slow grind for language and why the multiplicative jumps have to be so large.

### 3.4 Reading the exponent as an exchange rate between data and loss

The single most useful thing you can do with a measured exponent is convert it into the question your stakeholders actually ask: "how much more data to hit *this* loss?" The conversion is one line of algebra. To reduce excess loss by a factor $f$ (say, $f = 0.9$ for a 10% cut), you need to multiply the data by a factor $r$ where $r^{\beta_g} = f$, i.e.

$$
r = f^{1/\beta_g}.
$$

Because $\beta_g$ is negative and small in magnitude for language, $1/\beta_g$ is a large negative number, and $f < 1$ raised to a large negative power is a large multiplier. The table below turns each domain's exponent into the concrete data multiplier required to cut excess loss in half, which is the number that makes the shallow-exponent reality visceral.

| Domain | Typical $\beta_g$ | Data multiplier to halve excess loss ($2^{1/|\beta_g|}$) | Practical reading |
|---|---|---|---|
| Language modeling | $-0.07$ | $\approx 2^{14} \approx 16{,}000\times$ | Plan in orders of magnitude; doublings barely move loss |
| Machine translation | $-0.25$ | $\approx 2^{4} \approx 16\times$ | Data pays off; a 16x corpus is a real lever |
| Image classification | $-0.20$ | $\approx 2^{5} \approx 32\times$ | Moderate; depends heavily on the specific task |
| Speech recognition | $-0.15$ | $\approx 2^{6.7} \approx 100\times$ | Substantial but feasible scale jumps help |
| Textbook ideal | $-0.50$ | $2^{2} = 4\times$ | What naive theory promises and reality never delivers |

Two things jump out of that table. First, the gap between the textbook $-0.5$ ideal (halve error with 4x data) and the language-modeling reality (halve excess loss with roughly 16,000x data) is not a rounding error — it is the difference between "data is a tuning knob" and "data is a multi-year, multi-billion-token capital program." Second, the exponent is *task-shaped*: translation and speech reward data far faster than open-ended language modeling does, which is why a fixed data budget buys very different amounts of progress depending on what you are training. When someone asks "should we collect more data?", the honest first response is "what is the exponent for this task?" — because the same 10x of data is transformative in one domain and nearly invisible in another.

## 4. Rosenfeld 2019: making prediction constructive

**Senior rule of thumb: do not just confirm that a power law exists; fit it on the cheap end and use it to decide whether the expensive end is worth paying for.**

Hestness established that the power law is *there*. Rosenfeld et al., *A Constructive Prediction of the Generalization Error Across Scales* (MIT CSAIL, 2019; ICLR 2020), made it *constructive*: a recipe for fitting a functional form on small-scale experiments and predicting large-scale error, jointly across both dataset size and model size. The word "constructive" is doing real work here — the contribution is not a new observation about nature, it is a *procedure* you can run to save money.

Rosenfeld's functional form handles two things at once that Hestness handled separately: it is joint in data and model size, and it models the smooth *transition out of the random-guess regime* rather than assuming you are already in the clean power-law middle. Writing $n$ for data and $m$ for model size, the core unnormalized term is

$$
\tilde{\varepsilon}(m, n) = a\,n^{-\alpha} + b\,m^{-\beta} + c_\infty,
$$

a sum of two power laws (one per axis) plus a constant $c_\infty$ that plays the role of the floor. The full predicted error wraps this in a correction that bends the curve up toward the random-guess ceiling at small scale:

$$
\hat{\varepsilon}(m, n) = \varepsilon_0 \cdot \frac{\tilde{\varepsilon}(m, n)}{\tilde{\varepsilon}(m, n) - i\,\eta},
$$

where $\varepsilon_0$ is the chance-level error and the $i\,\eta$ term in the denominator is what produces the upward bend in Region 1. You do not need to memorize this formula — later posts use the cleaner Chinchilla form — but notice its anatomy: **two power-law terms (data and model), a floor constant, and an explicit correction for the random-guess plateau.** That is Regions 1, 2, and 3 of our mental-model figure, written as one equation.

### 4.1 The constructive recipe, and why it saves money

The procedure Rosenfeld actually advocates is the one that makes this series economically interesting. Fit the functional form using only *small-scale* runs — in their experiments, models up to roughly $1/16$ of the full size — and then *extrapolate* to predict the error of the full-scale model you have not trained. Because the fit only needs the cheap end of the curve, the cost of the forecast is a small fraction of the cost of the run it predicts.

![A five-stage pipeline showing three cheap small runs feeding a power-law fit, which is extended to a much larger scale to forecast loss and drive a budget decision](/imgs/blogs/scaling-laws-predictability-foundations-3.png)

The pipeline above is the constructive recipe as a workflow. Run a few cheap models on a geometric ladder of dataset sizes (say 1M, 4M, 16M tokens). Fit the line in log-log space to recover slope and offset. Extend the line to your target scale — perhaps a thousandfold larger. Read off the forecast loss with an error band. Then make the budget decision: spend, or stop. The fit costs a small fraction of the budget; the decision it informs is the whole budget. That asymmetry is the reason scaling laws are an industrial tool and not just an academic curiosity.

Rosenfeld is the conceptual bridge of this series. Hestness is "loss is a power law in data." Rosenfeld is "loss is a *joint* power law in data and model, and here is how to fit it cheaply and extrapolate." From there it is a short step to Kaplan and Chinchilla, which work out the joint $N$-$D$ law for language models specifically and turn it into a budget-allocation rule. Each post tightens the previous one; none of them abandons the straight line on the log-log plot.

### 4.2 The error bar is part of the deliverable

A forecast without an error bar is a wish. The honest version of "fit and extrapolate" produces not a single predicted loss but a *band*, and the band comes from two sources you should track separately:

- **Fit uncertainty:** with only three or four points, the slope and offset have real confidence intervals, and extrapolating a thousandfold *amplifies* slope uncertainty enormously. A slope known to $\pm 0.01$ at one decade becomes a wide loss band three decades out. This is why people fit on a *ladder* of several scales, not two points, and why a later post will dwell on how a replication found the published Chinchilla confidence intervals implausibly tight.
- **Regime risk:** the extrapolation assumes you stay in the power-law region. If the big run is actually approaching the floor, or if some training instability (loss spikes, learning-rate mis-tuning at scale) kicks in, the clean power law no longer applies. The mitigation is to push your largest *cheap* run as far right as you can afford, so the extrapolation gap is small and you have early evidence of any bend.

A mature forecast therefore reads like: "we expect loss $2.31 \pm 0.05$ at the target scale, assuming we remain in the power-law region and our learning-rate schedule holds." That sentence is worth more to a budget meeting than any single point estimate, because it is honest about both what is known and what is assumed.

## 5. Why predictability is worth real money

**Senior rule of thumb: the value of a scaling law is not the loss number; it is the option to *not* spend the big budget when the forecast says it will not pay off.**

Everything above is mechanics. This section is why anyone with a budget cares. The economic argument has three moves, and each one is a decision a real lab makes.

**Move one: forecast the payoff before committing the budget.** A frontier pre-training run is a single, large, mostly-irreversible capital expenditure — GPU-months, energy, a data pipeline, engineer-quarters. Without a scaling law, deciding to make that expenditure is a bet on a gut feeling about how much better the big model will be. With a scaling law fit on cheap runs, it is a forecast with an error band. You can answer "if we 10x the data, the loss drops from 2.31 to 2.18 — is that 0.13 nats worth \$2M to us?" The number might be yes or no, but now it is a *priced* question.

![A before-and-after comparison contrasting a blind multi-million-dollar training commitment against a forecast-first workflow that spends a tiny fraction on small runs to price the decision](/imgs/blogs/scaling-laws-predictability-foundations-8.png)

The before-and-after above is the whole economic case in one frame. On the left, the blind path: commit the big budget, discover the result only after training, risk re-running or shipping an undertrained model. On the right, the forecast-first path: spend under one percent on small runs, fit the law, set your parameter and data targets, and walk into the budget decision with numbers. The asymmetry — a few thousand dollars to de-risk a few million — is why every serious lab now runs scaling-law sweeps before large runs. The cost of the forecast rounds to zero against the cost of being wrong about the run.

**Move two: allocate a fixed budget optimally.** A scaling law that is joint in $N$ (parameters) and $D$ (data) does more than forecast — it tells you the *best split* of a fixed compute budget between making the model bigger and training it longer. This is exactly the question Kaplan and Chinchilla answer differently, and the disagreement (and its resolution) is the spine of two later posts. The point for now: once you have a joint law, "given $C$ FLOPs, what $N$ and $D$ minimize loss?" is a constrained optimization, not a vibe. Training FLOPs are well-approximated by $C \approx 6ND$, so the budget constraint is concrete and the optimization is tractable.

**Move three: set targets and hold the project to them.** A forecast is also a contract. "This run should reach loss 2.31; if at 30% of the tokens we are tracking above the predicted curve, something is wrong" turns a scaling law into a live monitoring tool. Loss spikes, a mis-set learning rate, a data bug — all of them show up as a departure from the predicted curve *during* training, early enough to kill or fix the run before you have burned the whole budget. The law you fit to *decide* on the run is the same law you use to *babysit* it.

### 5.1 A dollars-and-FLOPs worked example

Let me put numbers on move one, because the asymmetry is more dramatic than people expect.

Suppose you are considering a \$2,000,000 pre-training run and you want to forecast its loss first. You run three small models on a geometric data ladder. A rough cost model: training FLOPs $\approx 6ND$, and at a blended rate of, say, \$2 per petaFLOP-day of effective compute (numbers vary wildly by hardware and year — treat this as illustrative), three small runs at $10^{-3}$, $4\times 10^{-3}$, and $1.6\times 10^{-2}$ of the big run's compute cost roughly

$$
\$2{,}000{,}000 \times (0.001 + 0.004 + 0.016) = \$2{,}000{,}000 \times 0.021 = \$42{,}000,
$$

with the largest of the three dominating. Round up for overhead and call it under \$50k — about 2.5% of the big run. For that 2.5% you get a fitted power law with an error band, and the answer to "does the \$2M run land where we need it?" If the forecast says the big run reaches your target loss, you spend the \$2M with confidence. If it says you will fall short — that you are data-constrained, or that you are already near the floor and the extra compute mostly buys floor-chasing — you have just saved \$2M minus \$50k by *not* running it, or redirected the budget toward more data instead of more parameters.

The expected-value math is lopsided in the way that makes scaling laws non-optional at scale. Even if the forecast is only right 80% of the time, spending 2.5% to avoid a meaningful fraction of a \$2M misfire is overwhelmingly positive expected value. This is why "we fit scaling curves before big runs" is not a research nicety; it is basic capital discipline once the runs cost more than the sweep.

### 5.2 The second-order effect: scaling laws change what you build

There is a subtler, organizational consequence. Once a lab internalizes that loss is predictable and the slope is fixed by the problem, the *kind* of work that gets prioritized shifts. If you cannot bend the slope by tweaking the model, then beyond a point the highest-leverage moves are the ones that move the offset or extend how far you can ride the curve: better data (a lower-offset, sometimes steeper curve), more efficient training (more effective tokens per dollar), and smarter budget allocation (Chinchilla-optimal splits). Whole research programs — data curation, deduplication, curriculum, efficient architectures — are downstream of the realization that the brute-force slope is what it is, so the wins must come from the offset and the efficiency of climbing. The later posts on [data-constrained scaling](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) and data quality are exactly this: how to win the constant factor when you cannot win the exponent.

### 5.3 The third axis: compute, and where $6ND$ comes from

So far we have talked about two knobs — data $m$ (or $D$) and model size $N$. There is a third quantity that is really a function of the first two but is the one your finance team actually cares about: compute, measured in floating-point operations (FLOPs). The reason scaling laws translate so cleanly into dollars is that training compute has a famously simple approximation,

$$
C \approx 6\,N\,D,
$$

where $N$ is parameters and $D$ is training tokens. It is worth understanding where the 6 comes from, because it makes the whole budget calculus concrete rather than mysterious. A single forward pass through a dense transformer costs roughly $2N$ FLOPs per token — the 2 is one multiply plus one add per parameter, since each weight participates in a multiply-accumulate. The backward pass costs about twice the forward pass (you compute gradients with respect to both activations and weights), so roughly $4N$ FLOPs per token. Forward plus backward is $2N + 4N = 6N$ FLOPs per token, and across $D$ tokens that is $6ND$. The approximation ignores attention's quadratic term (small for the sequence lengths and model sizes where it is usually applied), embedding and last-layer costs (which matter at small scale — a recurring theme), and the difference between training and inference. Inference, for comparison, is only the forward pass, so it costs about $2N$ FLOPs per generated token, three times cheaper per token than training — a fact that the [inference-aware scaling post](/blog/machine-learning/scaling-laws/inference-aware-scaling-laws) builds an entire re-derivation of the optimum on.

The payoff of $C \approx 6ND$ is that it lets you convert a scaling law in $(N, D)$ into a scaling law in compute, and a budget in dollars into a constraint on the optimization. If you have $C$ FLOPs to spend and you want to minimize $L(N, D) = E + A/N^\alpha + B/D^\beta$ subject to $6ND = C$, that is a clean Lagrange-multiplier problem whose solution gives the compute-optimal $N$ and $D$. The shape of that solution — does most of the budget go to a bigger model or to more tokens? — is precisely the Kaplan-versus-Chinchilla question. For now, the structural insight is enough: because compute factorizes as $6ND$, a two-axis loss law becomes a one-constraint optimization, and "what should I build with this budget?" has a mathematical answer rather than a political one.

### 5.4 A compute-conversion worked example

Make it concrete. Suppose Chinchilla-style fitting says your compute-optimal point for a budget is a 7-billion-parameter model trained on 140 billion tokens (the roughly 20-tokens-per-parameter ratio the [Chinchilla post](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) derives). The training compute is

$$
C \approx 6 \times (7 \times 10^{9}) \times (140 \times 10^{9}) = 6 \times 9.8 \times 10^{20} \approx 5.9 \times 10^{21}\ \text{FLOPs}.
$$

Now suppose someone proposes instead a 14-billion-parameter model on the *same* 140 billion tokens — double the parameters, same data. Compute doubles to about $1.2 \times 10^{22}$ FLOPs, so it costs twice as much. Does it help? Only through the $A/N^\alpha$ term, and with $\alpha$ near $0.34$ (a Chinchilla-style value), doubling $N$ multiplies that term by $2^{-0.34} \approx 0.79$ — a 21% reduction of just the model-size part of the reducible loss, while leaving the data part $B/D^\beta$ untouched. Meanwhile you spent 2x the compute. Whether that is a good trade depends on which term dominates the reducible loss at your scale, which is exactly what the joint fit tells you and exactly why you do the fit before committing. The arithmetic is unglamorous, but it is the arithmetic that decides eight-figure budgets.

## 6. How to actually fit and use a scaling law

**Senior rule of thumb: fix your bookkeeping before you fit anything. What counts as a parameter, what counts as a FLOP, and which points are in the power-law region — decide these first, because every later argument in the field came from people defining them differently.**

This section is the practical core. Here is the procedure I would hand a team that has never fit a scaling law, written to avoid the mistakes that the rest of the series catalogs.

**Step 1 — Define your bookkeeping.** Decide and write down: (a) whether $N$ means total parameters or non-embedding parameters; (b) how you count training FLOPs (the standard is $C \approx 6ND$, but the last-layer and embedding FLOPs matter at small scale); (c) what loss you are fitting (validation cross-entropy in nats per token on a fixed held-out set). These choices sound bureaucratic. They are the entire reason two famous papers disagreed by a factor that split the field, a story the [Kaplan-vs-Chinchilla reconciliation post](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation) tells in full. The one-line lesson, available to you now: at small scale, embedding parameters are a *large* fraction of the total, so "non-embedding params" and "total params" diverge, and a slope fit in one convention is not comparable to a slope fit in the other.

**Step 2 — Build a geometric ladder.** Choose scales spaced by a constant *multiplicative* factor (2x, 4x, or 10x apart), because you are going to plot in log space and you want even coverage there. Three points is the bare minimum; five to eight is much better, because the slope's confidence interval shrinks fast with more points and the extrapolation amplifies that uncertainty. Make the largest cheap run as large as your sweep budget allows — it anchors the right end of the line and shrinks the extrapolation gap.

**Step 3 — Re-tune per scale.** This is the step people skip and regret. Learning rate, warmup length, and batch size should be re-tuned (or at least scaled by a known rule) at *each* scale. A warmup that is fine for the big run is too long for a tiny run and biases its loss upward; a learning rate optimal at small scale is wrong at large scale. Un-retuned hyperparameters were one of the documented causes of the Kaplan-Chinchilla discrepancy. If you fit a slope using runs that were all trained with the *same* warmup and learning rate regardless of size, you are fitting hyperparameter mis-tuning as if it were scaling behavior.

**Step 4 — Fit the floor-aware form.** Fit $L = E + a\,m^{-c}$ (or the joint $L = E + A/N^\alpha + B/D^\beta$) by minimizing error in *log space*, typically with a robust loss like Huber on the log-residuals, because a single anomalous run should not dominate the fit. Do not fit a pure power law if any of your points are near the floor; estimate $E$ jointly. Report confidence intervals on every fitted constant, and be suspicious if they come out implausibly tight — that usually means an under-specified optimizer or too few points, not a law of nature.

**Step 5 — Extrapolate with an explicit gap and band.** State your target scale, the extrapolation factor (how many decades beyond your largest run), and the resulting loss band. Treat the band, not the point, as the deliverable. If the extrapolation factor is large (say, more than two decades), say so loudly — the forecast is doing a lot of work and deserves a wide band and a plan to validate it with one intermediate run before the full commitment.

**Step 6 — Validate during the run.** Once the big run starts, plot its loss against the predicted curve at checkpoints. On-track means the law held; consistently above means investigate (data, learning rate, instability); consistently below is a pleasant surprise worth understanding. The forecast becomes a live SLA for the run.

### 6.1 A minimal fit in code

The whole fitting procedure is a few lines once you frame it as linear regression in log space. Here is a runnable sketch — real imports, real flags-of-thought — for the pure power-law case, with a note on extending to the floor-aware form.

```python
import numpy as np
from scipy.optimize import curve_fit

# Three cheap runs: dataset size m (tokens) and measured excess loss eps.
m   = np.array([1e6, 4e6, 16e6])
eps = np.array([0.500, 0.402, 0.323])

# --- Method 1: linear regression in log-log space (pure power law) ---
# log(eps) = log(a) - c * log(m). Slope is -c, intercept is log(a).
logm, logeps = np.log(m), np.log(eps)
slope, intercept = np.polyfit(logm, logeps, deg=1)
c = -slope
a = np.exp(intercept)
print(f"fitted: eps ~= {a:.3f} * m^(-{c:.3f})")   # eps ~= 4.44 * m^(-0.158)

# Extrapolate to 1000x the largest run.
m_target = 16e9
eps_pred = a * m_target ** (-c)
print(f"forecast excess loss at m={m_target:.0e}: {eps_pred:.3f}")  # ~0.108

# --- Method 2: floor-aware nonlinear fit L = E + a * m^(-c) ---
# Use this whenever any point may be near the irreducible floor.
def law(m, E, a, c):
    return E + a * m ** (-c)

# Fit in log space for stability; needs >= 4 well-spread points in practice.
# p0 is a rough guess: small floor, offset near the smallest-m loss, shallow c.
popt, pcov = curve_fit(
    lambda m, E, a, c: np.log(law(m, E, a, c)),
    m, np.log(eps),
    p0=[0.05, 4.0, 0.15],
    maxfev=10000,
)
E, a, c = popt
perr = np.sqrt(np.diag(pcov))          # 1-sigma on each fitted constant
print(f"floor E={E:.3f}+-{perr[0]:.3f}, a={a:.3f}, c={c:.3f}+-{perr[2]:.3f}")
```

Two warnings about that snippet that the rest of the series will earn the right to make. First, `curve_fit` with three points and three free parameters is fitting exactly as many parameters as you have data — the floor will be poorly determined. Use it only with a real ladder of five or more points. Second, the confidence intervals `perr` are only as honest as your noise model; if every run used the same hyperparameters regardless of scale, `perr` understates the true uncertainty because it does not know your points are biased.

### 6.2 Common ways the fit lies to you

Here is a compact field guide to the failure modes, framed as a table because each one is a "looks fine, is wrong" trap.

| Symptom | Wrong first hypothesis | Actual root cause | Fix |
|---|---|---|---|
| Slope looks great but extrapolation overshoots | "The law is wrong" | Largest fit point was near the floor; you fit the bend, not the slope | Fit the floor-aware form; estimate $E$ jointly |
| Two papers report different slopes | "Networks scale differently in their setups" | Different parameter or FLOP bookkeeping (total vs non-embedding) | Re-fit both in one convention before comparing |
| Tiny models all sit above the line | "Small models are just bad" | Warmup too long / learning rate un-tuned for short runs | Re-tune hyperparameters per scale |
| Confidence intervals are suspiciously tight | "We nailed it" | Too few points, or an under-specified optimizer | Add ladder points; report honest CIs; sanity-check against a replication |
| Forecast is confident but wrong at scale | "Bad luck" | Extrapolated too many decades past the largest cheap run | Add an intermediate-scale run to shrink the gap before committing |

Every row of that table corresponds to a real episode in the literature this series covers. Treat the table as a pre-flight checklist before you trust any number you extrapolate.

## 7. A short history, and a map of where this series goes

**Senior rule of thumb: read scaling-law papers in order, because each one fixes a specific limitation of the one before it, and the constants are point estimates, not commandments.**

It helps to see the lineage laid out, because the field moved in clear steps and each step is a later post in this series.

![A timeline of empirical scaling-law milestones from Hestness in 2017 through Rosenfeld, Kaplan, and Chinchilla to the modern data-constrained and inference-aware laws](/imgs/blogs/scaling-laws-predictability-foundations-5.png)

The timeline above is the reading order. **Hestness (2017)** established that generalization error is a power law in data, with domain-specific shallow exponents, and that architecture moves the offset, not the slope. **Rosenfeld (2019)** made prediction constructive and joint in data and model size: fit small, extrapolate large, waste less compute. **Kaplan (2020)** worked out the first explicit scaling laws for language models — separate power laws for loss versus parameters, data, and compute — and drew a now-famous (and later corrected) conclusion about how to split a budget. **Chinchilla (2022)** re-did the budget-allocation analysis more carefully and overturned Kaplan's split, giving the "roughly 20 tokens per parameter" rule that defined a generation of models. Everything after — data-constrained, inference-aware, precision, and test-time-compute laws — refines the picture for the constraints that actually bite in practice.

Two cautions to carry forward, both of which later posts make concrete:

- **The slope is real; the constants are estimates.** A 2024 replication of Chinchilla found that one of its three fitting approaches reproduced poorly and reported implausibly tight confidence intervals; the *conclusion* (roughly equal scaling of $N$ and $D$, around 20 tokens per parameter) held, but the exact coefficients did not. Treat any single published coefficient set as a point estimate with real uncertainty, not a constant of nature.
- **Famous disagreements were usually bookkeeping.** Kaplan and Chinchilla appeared to disagree sharply on how to split a budget. The resolution, covered in its own post, is that most of the gap came from counting parameters and FLOPs differently and from un-retuned hyperparameters at small scale — not from a real difference in how networks scale. This is the strongest possible argument for Step 1 of the fitting procedure: define your bookkeeping first.

### 7.1 The questions each later post answers

To make the map useful, here is what to read next depending on what you are trying to decide:

- **"Given a budget, how big a model and how much data?"** — Start with the [Kaplan post](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models) for the first answer and the [Chinchilla post](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) for the corrected one. Chinchilla's roughly-20-tokens-per-parameter rule is the single most cited operational takeaway in the field.
- **"Why did two papers disagree, and whom do I trust?"** — The [reconciliation post](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation) is the detective story: a parameter-counting convention plus small-scale warmup explains nearly all of the apparent contradiction.
- **"I am out of unique data — can I just repeat it?"** — The [data-constrained scaling post](/blog/machine-learning/scaling-laws/data-constrained-scaling-laws) quantifies how many epochs of repetition are nearly free and where returns collapse.
- **"My model will serve billions of tokens — does Chinchilla still apply?"** — The inference-aware post shows that once you price lifetime inference, the optimum shifts to a smaller model trained on far more tokens.

You do not have to read them in order, but the concepts compound, and all of them assume you are comfortable with the straight line on the log-log plot that this post is about.

## 8. Worked case studies: scaling laws meeting reality

**Senior rule of thumb: the law tells you the shape; the case studies tell you where the shape breaks and why. Collect failure modes the way pilots collect incident reports.**

Abstract rules stick once you have seen them save or sink a real decision. Here are eight short, concrete scenarios — some drawn from the published record, some composite — illustrating how the foundations above play out.

### 8.1 The shallow exponent that justified GPT-scale data

When the language-modeling exponent was first pinned near $-0.07$ to $-0.09$, the implication was stark: to make a language model meaningfully better by data alone, you need *order-of-magnitude* increases, not percentage increases. A team expecting a 2x data increase to noticeably move the needle would have been disappointed — $2^{-0.07} \approx 0.95$, a 5% reduction in excess loss. The lesson teams internalized was to think in decades, not doublings, which is exactly why frontier corpora jumped from billions to hundreds of billions to trillions of tokens within a few years. The shallow slope is not a discouragement; it is a *budget* statement: progress on the data axis is real but priced in orders of magnitude.

### 8.2 The architecture win that did not change the slope

A team ships a new attention variant and measures a clear loss improvement at their current scale — say 8% lower loss. The exciting (wrong) interpretation is that they have found a steeper scaling curve. The careful interpretation, straight from Hestness, is that they have lowered the *offset* by a constant factor and the slope is unchanged. The way to check is to re-measure at two or three scales and confirm the lines are parallel. When they did, they found exactly that: a constant multiplicative head start, valuable and bankable, but not a new exponent. The decision that followed was correct: ship the variant for its offset win, but do not revise the long-range data forecast, because the slope that drives that forecast did not move.

### 8.3 The forecast that killed a run before it started

A team had budget for one large run and two plausible recipes: a bigger model on the same data, or the same model on much more data. They fit a joint law on a small ladder for both recipes. The bigger-model recipe's forecast landed barely below the current model — it was approaching the floor for that data budget, and extra parameters were chasing a nearly-exhausted reducible term. The more-data recipe's forecast landed well below. They ran the more-data recipe, hit the forecast, and never spent a dollar on the bigger-model run that the law said would underwhelm. The forecast's entire value was the run they did *not* do.

### 8.4 The floor that masqueraded as a broken law

A practitioner fit a power law across five points and got a beautiful straight line on the first four, with the fifth sitting stubbornly above the extrapolation. First hypothesis: the fifth run was buggy. Actual cause: the fifth run was the only one large enough to be approaching the irreducible floor, so the *observed* loss had started to bend away from the pure power law toward $E$. The pure-power-law extrapolation was too optimistic precisely because it ignored the floor. Re-fitting with $L = E + a\,m^{-c}$ recovered a consistent fit across all five points and a sane floor estimate. The lesson: a point that "breaks" your power law near the high-data end is often the floor announcing itself, not a bad run.

### 8.5 The tight confidence interval that should have been a red flag

A fit came back with a slope confidence interval so narrow it implied near-certainty about a thousandfold extrapolation. That is not how extrapolation works: projecting three decades past your largest point should *widen* the band, not narrow it. The narrow interval was an artifact of too few points and an over-confident noise model. When a replication added more ladder points and re-tuned the optimizer, the interval widened to something believable and the central estimate shifted slightly. This mirrors the real Chinchilla replication story: implausibly tight intervals are a symptom, not a triumph. Always ask whether your error bar is physically reasonable for the extrapolation distance.

### 8.6 The small models that all sat above the line

Every model below a certain size sat above the fitted line, tempting the conclusion that "small models just underperform the law." The real cause was mundane: the warmup schedule, fixed across all runs, consumed a large fraction of the *short* small-model runs, so those models never reached their proper loss. Re-tuning warmup per scale pulled the small models down onto the line. This is one of the documented contributors to the Kaplan-Chinchilla discrepancy, and it is a reason the fitting procedure insists on re-tuning hyperparameters per scale rather than holding them fixed for convenience.

### 8.7 The two papers that agreed once you fixed the bookkeeping

Two well-known analyses appeared to disagree on how to split a compute budget between model size and data — one favoring much bigger models, the other roughly equal scaling. The apparent contradiction evaporated once both were re-expressed in the same convention: one had counted non-embedding parameters at a scale where embeddings are a large fraction of the total, distorting the parameter-to-compute relationship. Re-fit in a single convention, with hyperparameters re-tuned per scale, the two setups agreed. The episode is the field's canonical lesson that "define your bookkeeping first" is not pedantry — it is the difference between two papers fighting and two papers agreeing.

### 8.8 The monitoring curve that caught a silent data bug

A large run was babysat against its predicted loss curve. For the first 20% of training it tracked perfectly; then it began drifting consistently above the prediction. Because the team had a forecast to compare against, they noticed within hours rather than at the end. The cause was a data-loading bug that had started serving a corrupted shard, quietly degrading the effective data quality. They fixed it, restarted from the last good checkpoint, and the run rejoined its predicted curve. Without a scaling-law forecast to define "expected," the drift would have been invisible until the final eval — and the whole budget would have produced a quietly worse model.

### 8.9 The two-point fit that fooled a roadmap

A planning deck once projected a year of model improvements from a power law fit on exactly *two* runs. Two points define a line perfectly — they always do, with zero residual — so the fit "looked" flawless and the slope was reported with false confidence. The problem is that two points cannot distinguish a genuine power law from a slight curve that happens to pass through both; there is no residual to reveal mis-specification, and there is no way to estimate uncertainty in the slope. When a third and fourth run were finally added, the points were visibly not collinear in log-log space — the true curve was bending toward the floor — and the year-long projection collapsed by a wide margin. The lesson is blunt: a two-point "fit" is not a fit, it is a line drawn through two dots. The minimum honest ladder is three points, and three is still thin; five or more is where the slope's confidence interval becomes trustworthy enough to bet a budget on.

### 8.10 The downstream metric that "emerged" from a smooth loss

A team tracked both pre-training loss and a downstream multi-step reasoning benchmark across a model-size sweep. The loss fell as a clean power law the whole way — utterly boring and predictable. The benchmark, however, sat near random for the small and medium models and then jumped sharply at the largest size, which the team initially described as an "emergent ability" that scaling laws had failed to predict. The reconciliation is that nothing discontinuous happened to the *loss*; the benchmark is a thresholded, all-or-nothing metric (you either chain the steps correctly or you do not), so a smooth improvement in the underlying probability of each step crosses a visible threshold suddenly. The scaling law predicted the loss perfectly; it simply does not, and never claimed to, predict where a brittle downstream metric crosses its threshold. The practical takeaway is the one from Section 2.2 made vivid: forecast loss, and treat capability jumps as a separate, harder forecasting problem layered on top of the smooth loss curve.

### 8.11 The budget that was reallocated mid-planning by one fit

A group had pre-committed, on intuition, to spending most of a compute budget on a very large model trained on relatively little data. Before locking it in, an engineer fit a joint $L(N, D)$ law on a small ladder and ran the constrained optimization under $C \approx 6ND$. The fit said the planned allocation was badly off the compute-optimal frontier — the model was too large for its token budget, so the $B/D^\beta$ data term dominated the reducible loss and the extra parameters were wasted. Reallocating toward a smaller model on far more tokens, at the *same* total compute, forecast a substantially lower loss. They changed the plan before spending anything. This is the Chinchilla lesson in miniature, and it happened entirely on paper: the only cost was the small sweep, and the payoff was avoiding an undertrained flagship model. The episode is the single cleanest illustration of why a joint law plus the $6ND$ identity is worth more than any amount of architectural intuition when a budget is on the line.

## 9. What this means in practice

If you remember nothing else, remember the shape and the discipline. The shape is a straight line on a log-log plot: a power law $L = E + a\,m^{-c}$ with a floor, valid in the middle region of the learning curve, with a slope set by the problem and an offset set by your engineering. The discipline is: fix your bookkeeping, build a geometric ladder of cheap runs, re-tune per scale, fit the floor-aware form in log space, extrapolate with an explicit gap and an honest band, and validate during the big run.

The reason this is worth doing is purely economic. A scaling-law sweep costs a small fraction of the run it forecasts, and it converts the largest, least-reversible decision in a model's life — how much to spend pre-training, and how to split that spend between size and data — from a gut call into a priced one. It also turns the forecast into a live monitor that catches instabilities and data bugs early. The shallow language-modeling exponent (around $-0.07$) tells you the brutal truth that data progress comes in orders of magnitude, and the offset-versus-slope distinction tells you where architecture effort actually pays off: a constant factor, never a steeper descent.

The series goes on from here to make this operational for language models specifically. [Kaplan](/blog/machine-learning/scaling-laws/kaplan-scaling-laws-language-models) gives the first joint laws and a budget split; [Chinchilla](/blog/machine-learning/scaling-laws/chinchilla-compute-optimal-scaling) corrects that split into the roughly-20-tokens-per-parameter rule; the [reconciliation post](/blog/machine-learning/scaling-laws/kaplan-vs-chinchilla-reconciliation) explains why the two disagreed and why the answer was bookkeeping; and the later posts handle the real-world constraints — running out of unique data, paying for inference, training in low precision, and spending compute at test time instead of in pre-training. All of them are refinements of the single picture you now hold: loss is a straight line on a log-log plot, and a straight line is predictable.

When to reach for a scaling-law forecast: any time the run you are about to commit to costs more than the sweep that would forecast it; any time you have to split a fixed budget between model size and data; any time you want a live SLA to babysit a large run against. When to be skeptical: when you are extrapolating many decades past your largest cheap run; when your points may be near the floor; when your confidence intervals look too good; and when you are tempted to forecast downstream *capability* rather than pre-training *loss* — the loss is forecastable, the jumpy capability metrics much less so.

## Further reading

- Hestness et al., 2017. *Deep Learning Scaling is Predictable, Empirically.* arXiv:1712.00409. https://arxiv.org/abs/1712.00409
- Rosenfeld et al., 2019. *A Constructive Prediction of the Generalization Error Across Scales.* arXiv:1909.12673 (ICLR 2020). https://arxiv.org/abs/1909.12673
- Kaplan et al., 2020. *Scaling Laws for Neural Language Models.* arXiv:2001.08361. https://arxiv.org/abs/2001.08361
- Hoffmann et al., 2022. *Training Compute-Optimal Large Language Models* (Chinchilla). arXiv:2203.15556. https://arxiv.org/abs/2203.15556
- Besiroglu et al., 2024. *Chinchilla Scaling: A replication attempt.* arXiv:2404.10102. https://arxiv.org/abs/2404.10102

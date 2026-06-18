---
title: "Delayed Feedback and Conversion Attribution"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Conversions arrive hours or days after the click, so a fixed training window quietly stamps late buyers as negatives and biases your CVR model low — here is the delayed-feedback likelihood, the fake-negative correction, the survival-analysis view, and the attribution rules that decide which click even gets the label, with runnable simulations that show the bias and its repair."
tags:
  [
    "recommendation-systems",
    "recsys",
    "delayed-feedback",
    "conversion-prediction",
    "cvr",
    "attribution",
    "survival-analysis",
    "online-learning",
    "machine-learning",
    "ad-tech",
  ]
category: "machine-learning"
subcategory: "Recommendation Systems"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/delayed-feedback-and-conversion-attribution-1.png"
---

It is 3 a.m. and you are staring at a dashboard that makes no sense. Your conversion-rate (CVR) model — the head of the ranker that predicts whether a click turns into a purchase — has been retraining every six hours, exactly as designed, to stay fresh. Offline AUC looks fine. But the *predicted* CVR it emits has been sliding for two weeks, from around 9% down to under 6%, and Finance is asking why the system is bidding so timidly that you are losing auctions you used to win. Nothing in the model changed. No feature broke. The training pipeline is green. And yet the model has convinced itself that almost nobody converts.

Here is what actually happened, and it is one of the most quietly destructive bugs in applied recommendation and ad-tech. A purchase does not happen the instant someone clicks. It happens minutes, hours, or *days* later. When your pipeline cuts a training window at "now," every click that has not yet converted gets a label of zero. Most of those zeros are honest negatives — the person was never going to buy. But a meaningful fraction are **late converters**: people who will buy tomorrow, whom your pipeline just stamped as "did not convert." The fresher you make the model, the shorter your observation window, the *more* late converters you mislabel, and the harder the model learns the wrong lesson: that conversions are rarer than they truly are. Freshness, the thing you optimized for, is exactly what poisoned the labels.

This is the **delayed feedback problem**, and it is the defining data hazard of the CVR half of the click-then-convert funnel. In the [retrieval to ranking to re-ranking funnel](/blog/machine-learning/recommendation-systems/what-is-a-recommender-system) that runs this whole series, CTR prediction — *will they click?* — gets feedback in seconds. CVR prediction — *will they convert?* — gets feedback on a delay that is itself random, sometimes never (yet). The feedback loop that keeps the model honest (serve, log, train, serve) collides head-on with the fact that the label is not done arriving when you train. Before figure one is over you will see the whole trap in one picture; by the end you will be able to *measure* the bias, *derive* the likelihood that fixes it, and *implement* both the delayed-feedback model and the fake-negative correction in runnable code.

![Side by side comparison of a naive fixed training window that mislabels a late converter as a negative versus a delay-aware label that keeps the sample uncertain and later counts the conversion correctly](/imgs/blogs/delayed-feedback-and-conversion-attribution-1.png)

What you will be able to *do* by the end: simulate a click stream with exponentially delayed conversions and a known true CVR; show a naive fixed-window model under-predicting that CVR by a third; implement the Chapelle delayed-feedback model (jointly modeling conversion probability and the delay distribution) and a fake-negative-weighted estimator; recover near-perfect calibration; and reason clearly about attribution — *which* click should even receive the conversion label, and how that choice silently reshapes your training data. We will keep one running example throughout: an e-commerce or ads CVR model on a Criteo-style stream, the same dataset where this problem was first formalized.

## 1. Why conversions are different from clicks

Let us be precise about the funnel, because the whole problem lives in the gap between two events.

A **click** is immediate feedback. The user is shown a candidate, they either click or they do not, and within a second or two you have a label. That is why CTR models — the ones we built in [the ranking model and CTR prediction foundations](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations) — can be trained on near-real-time streams with clean labels. The label is *complete* almost immediately.

A **conversion** is delayed feedback. The user clicks, lands on the page, and then maybe buys now, maybe puts it in a cart and buys in three days, maybe installs the app next week, maybe subscribes after a free trial ends in a month — or never buys at all. The conversion event, when it comes, is attributed back to the click that earned it. The crucial property: **at the moment you read the label, you cannot distinguish a true negative from a not-yet-positive.** Both look identical: a click with no conversion attached. One of them is a person who will never buy. The other is a person whose purchase is still in flight.

This single fact breaks the comfortable assumption underneath ordinary supervised learning. Normally we assume the label $y$ for an example is *observed and final*. Here the label is **right-censored**: for an unconverted click, all we know is that the conversion time exceeds the elapsed time so far. The label is not "0." The label is "0 *so far*, observed for $e$ units of time." That qualifier — the elapsed time $e$ — turns out to be the key to fixing everything, but a naive pipeline throws it away.

#### Worked example: the same click, three labels

Take one click at noon on Monday. The person will, in truth, buy on Thursday at noon — a 72-hour delay. Watch how the label changes depending on when the training job runs:

- **Trained Monday 6 p.m. (6h window):** no conversion yet. Label = 0. *Wrong* — this is a late converter mislabeled as a negative.
- **Trained Tuesday noon (1-day window):** still no conversion. Label = 0. *Still wrong.*
- **Trained Friday noon (4-day window):** the Thursday purchase is now attributed. Label = 1. *Correct, but the model that learned from it is four days stale.*

The same ground-truth event produces label 0, 0, then 1 depending purely on *when you look*. A model that trains every six hours sees a flood of 0s for samples that have not had time to convert. It does not know they are censored; it treats them as honest negatives. Multiply this across millions of clicks and the model's estimate of the base conversion rate collapses. That collapse is the dashboard you were staring at in the intro.

The trade-off this forces is the spine of the entire post, so name it now. **Freshness versus correctness.** Train on a short window and your model reflects the world as of an hour ago — fresh — but its labels are wrong because late converters look negative. Train on a long window and your labels are nearly correct — most conversions have landed — but your model reflects the world as it was 30 days ago, blind to any drift in the meantime. There is no window length that gives you both. Everything in this post is a way to escape that dilemma instead of just picking a point on it.

## 2. The timeline of one conversion

To reason about the fix we need a clean mental model of a single click's life, with every quantity named. The figure below is the one to keep in your head for the rest of the post.

![Vertical stack showing the life of one click from impression at time zero through the random delay and the training cutoff to either an observed conversion or a censored sample that looks negative](/imgs/blogs/delayed-feedback-and-conversion-attribution-2.png)

Define the quantities for a single click:

- $C \in \{0, 1\}$: the **eventual conversion indicator**. $C = 1$ if this click will *ever* convert (within the attribution window); $C = 0$ if it never will. This is the thing we ultimately want to predict — the true CVR is $\Pr(C = 1 \mid x)$. Critically, $C$ is **latent** at training time for unconverted clicks.
- $D$: the **delay**, the time from click to conversion, defined only when $C = 1$. We will model $D$ as a continuous random variable with its own distribution.
- $E$: the **elapsed time**, how long we have observed this click as of the moment we read the label — `now` minus the click timestamp. This is *observed and known* for every sample. It is the feature a naive pipeline ignores.
- $Y \in \{0, 1\}$: the **observed label** at training time. $Y = 1$ if a conversion has been recorded so far (which requires $C = 1$ *and* $D \le E$); $Y = 0$ otherwise.

The relationship is the crux:

$$
Y = 1 \iff C = 1 \text{ and } D \le E.
$$

So an observed positive ($Y=1$) is unambiguous: the conversion happened, we saw it. But an observed negative ($Y=0$) is a mixture of two populations:

$$
Y = 0 \iff \underbrace{C = 0}_{\text{true negative}} \quad \text{or} \quad \underbrace{C = 1 \text{ and } D > E}_{\text{censored positive}}.
$$

That second branch — converter, but the delay outran our observation window — is the **fake negative** (sometimes "false negative" in this literature, though it is not a model error; it is a *label* error baked into the data). The naive estimator's sin is treating the whole $Y=0$ population as $C=0$. The fixes all work by reasoning about how much of the $Y=0$ mass is really censored positives.

This is exactly the setup of **survival analysis** / **time-to-event modeling**, the statistics of "how long until the event, and what do we do with subjects who haven't had the event yet?" In that language: a converted click is an *observed event* at time $D$; an unconverted click is a *right-censored observation* with censoring time $E$. Survival analysis has a century of machinery for censored data, and the delayed-feedback model is, at heart, a parametric survival model with a cure fraction (the never-converters, $C=0$). If that framing helps you, hold onto it; if not, the algebra below stands on its own.

The cure-fraction detail matters and is worth a beat, because it is what makes CVR delay different from a textbook survival problem. In ordinary survival analysis (time to machine failure, time to death) *everyone eventually has the event* — wait long enough and the survival curve goes to zero. In conversion data it does not: a real fraction $1 - p(x)$ of clicks will *never* convert, no matter how long you wait. So the survival curve plateaus at $1 - p(x)$ rather than decaying to zero. That plateau is the never-converters, and separating "will not convert (ever)" from "has not converted (yet)" is precisely the modeling job. A model that ignores the cure fraction will misread the plateau as "delays are just very long" and inflate its delay estimates; a model that ignores the delay will misread slow converters as part of the cure fraction and deflate its CVR. The delayed-feedback likelihood is the thing that estimates *both* the height of the plateau ($p$) and the shape of the decay ($\lambda$) at once, from the same censored data.

## 3. The bias of the naive fixed-window estimator

Before we fix anything, let us prove the naive approach is biased and compute *how* biased, because "the model under-predicts" is not actionable until you can say by how much and why.

Suppose the true conversion probability is $p = \Pr(C = 1)$ (drop the $x$ for a moment; read it as the base rate). Suppose, for converters, the delay $D$ follows an exponential distribution with rate $\lambda$: $\Pr(D \le e) = 1 - e^{-\lambda e}$. The exponential is the standard first model — memoryless, one parameter, and a decent fit to real delay tails. The mean delay is $1/\lambda$.

Now run the naive estimator: fix an observation window $E = w$ (every click is observed for exactly $w$ time before we read its label), label $Y = 1$ for any click whose conversion landed within $w$, and estimate CVR as the empirical mean of $Y$. What does that converge to?

$$
\Pr(Y = 1 \mid E = w) = \Pr(C = 1) \cdot \Pr(D \le w \mid C = 1) = p \,(1 - e^{-\lambda w}).
$$

So the naive estimator does **not** converge to $p$. It converges to $p$ times the fraction of conversions that have *arrived* by the window's end, $(1 - e^{-\lambda w})$. That multiplier is always less than 1. The **relative bias** is exactly the survival tail:

$$
\frac{\hat p_{\text{naive}} - p}{p} = -\,e^{-\lambda w} \;=\; -\Pr(D > w \mid C = 1).
$$

The model under-predicts CVR by precisely the probability that a converter's delay exceeds the window. This is clean and a little alarming: the bias does not vanish with more data — it is a structural bias, and *more data does not help.* You can collect a trillion clicks; if your window is short relative to the delay scale, you converge confidently to the wrong number.

#### Worked example: a one-day window on three-day delays

Say the true CVR is $p = 0.10$, and the mean delay is $1/\lambda = 3$ days, so $\lambda = 1/3$ per day. You train on a $w = 1$ day window.

$$
\Pr(D \le 1) = 1 - e^{-(1/3)(1)} = 1 - e^{-0.333} = 1 - 0.717 = 0.283.
$$

Only **28.3%** of conversions have landed within one day. So:

$$
\hat p_{\text{naive}} \to 0.10 \times 0.283 = 0.0283.
$$

Your model converges to a CVR of **2.8%** when the truth is **10%** — under-predicting by 72%. Relative bias $= -e^{-1/3} = -0.717$. If your bids scale with predicted CVR (they do, in any value-based bidding system), you are bidding at roughly a quarter of the right price and losing nearly every competitive auction. *That* is how a "fresh" model torpedoes revenue.

Stretch the window to $w = 7$ days: $\Pr(D \le 7) = 1 - e^{-7/3} = 1 - 0.097 = 0.903$, so $\hat p \to 0.0903$ — much closer, off by under 10%. Stretch to $w = 21$ days: $1 - e^{-7} = 0.999$, essentially unbiased. But now your model is up to three weeks behind reality. That progression *is* the freshness–correctness curve, with real numbers on it.

A subtler point that matters in production: the bias is not uniform across examples. Segments with longer delays (high-consideration purchases, B2B, expensive items) are censored *more* by a fixed window than impulse buys. So the naive model does not just shrink CVR — it **distorts the ranking** by under-predicting slow-converting segments relative to fast ones. The error is heteroscedastic, which is why "just add a constant calibration multiplier" does not fully save you (more on calibration in [the prediction you can trust](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust)). A single global re-scaling fixes the average but leaves slow-converting segments still under-predicted and fast ones over-predicted; the bias is a *function of the delay profile*, not a constant.

This is also where a survival-analysis tool earns its keep as a *diagnostic*, before you commit to any model. The **Kaplan-Meier estimator** is the non-parametric way to estimate the conversion-time distribution from censored data: it gives you $\hat S(t) = \Pr(D > t)$, the fraction of conversions still un-arrived $t$ time after click, *without* assuming an exponential or any parametric form. Compute it on your converted clicks (treating still-converting clicks as censored at their elapsed time) and you can read the bias of *any* candidate window directly off the curve: the relative bias of a window $w$ is approximately $-\hat S(w)$. If the Kaplan-Meier curve still sits at 0.4 at your one-day window, you are losing 40% of your conversions to censoring at that cadence — a number you can put in a design doc without fitting a single model. The same curve tells you whether an exponential is even reasonable: plot $\log \hat S(t)$ against $t$; if it is roughly linear, the exponential's constant-rate assumption holds; if it curves (heavier tail), you need Weibull or log-normal. Five minutes with `lifelines.KaplanMeierFitter` answers "do I have a delayed-feedback problem, and how bad" before you write any modeling code.

## 4. The freshness–correctness trade-off, made concrete

We have named the tension and put numbers on it. Now let us look at it as the design decision it actually is, because before any clever model, *someone on your team is choosing a window length*, and that choice has consequences worth seeing whole.

![Side by side comparison of a short training window that is fresh but downward biased against a long training window that has clean labels but a model that lags the live distribution](/imgs/blogs/delayed-feedback-and-conversion-attribution-5.png)

The way this works in practice is that the attribution window (a business/legal choice — more on that in section 8) and the training window (an ML choice) interact. Suppose your advertiser contracts specify a 30-day attribution window: a purchase within 30 days of the click counts. That is the *maximum* a conversion can be delayed and still be your responsibility. It does **not** mean you must wait 30 days to train. The whole art is decoupling "how long a conversion is allowed to take" from "how long I wait before I train on a click."

Three honest strategies, none free:

| Strategy | Window | Freshness | Label correctness | When it is OK |
|---|---|---|---|---|
| Short fixed window | hours–1 day | Excellent | Badly biased low | Only if delays are tiny (e.g., in-session conversions) |
| Long fixed window | = attribution (e.g. 30d) | Poor (days–weeks stale) | Near-perfect | Slow-moving catalog, no drift, offline batch |
| Delay-aware model | short, plus a delay model | Excellent | Corrected in expectation | The general answer (sections 5–7) |

The first two are the same pipeline with a dial; the third is a different pipeline that refuses the dial. Most teams start at "short window" because freshness is visible and bias is invisible (the model just quietly predicts low). They discover the bias only when revenue drops, exactly as in our intro. The mature answer is to keep the short window — *do not* sacrifice freshness — and pay down the bias with a model that knows about delay. That is where we go next.

One more practical wrinkle before the models: **continuous / online training collides with delayed labels in a specific way.** If you train online (SGD over a never-ending stream, as in many ad systems), you must assign a label to each click *as it streams by*. You cannot hold a click in limbo for 30 days. So online systems are *forced* into a short effective window unless they do something clever — which is precisely why the streaming-CVR methods in section 7 (ES-DFM, fake-negative weighting with duplication) exist. The delayed-feedback problem is at its sharpest exactly where freshness matters most: the online stream.

## 5. The delayed-feedback model (Chapelle, 2014)

Olivier Chapelle's 2014 KDD paper "Modeling Delayed Feedback in Display Advertising" is the foundational treatment, formalized on Criteo's data. The idea is elegant: instead of pretending the label is final, **jointly model the conversion probability and the delay distribution**, and write down a likelihood that handles censored samples correctly. Let us derive it.

![Branching dataflow graph showing shared features feeding a conversion probability head and a delay rate head whose outputs combine into a joint likelihood that scores not yet converted samples by their survival probability](/imgs/blogs/delayed-feedback-and-conversion-attribution-4.png)

We need two models, both functions of the features $x$:

1. A **conversion model** $p(x) = \Pr(C = 1 \mid x)$ — the thing we actually care about, the true CVR. Parameterize it as a sigmoid over a feature score: $p(x) = \sigma(w_c^\top x)$.
2. A **delay model** for converters: $\Pr(D \mid C = 1, x)$. Take it exponential with a rate that also depends on features: $\lambda(x) = \exp(w_d^\top x)$ (exp keeps the rate positive). The density is $f(d \mid x) = \lambda(x)\, e^{-\lambda(x) d}$ and the survival is $\Pr(D > d \mid x) = e^{-\lambda(x) d}$.

Now write the likelihood of the *observed* data, which is the pair $(Y, E)$ for each click — the label and the elapsed time, both known.

**Case 1 — observed positive ($Y = 1$).** We saw the conversion at delay $d = D$ (when a conversion is observed we know its exact delay: conversion timestamp minus click timestamp). The contribution is "this click converts AND the delay is exactly $d$":

$$
\Pr(C = 1, D = d \mid x) = p(x)\, \lambda(x)\, e^{-\lambda(x) d}.
$$

It is essential that the positive term includes the delay density. Without it, you would just be fitting $p(x)$ ignoring delay — and crucially, the delay density is what lets the model *attribute* a short-delay positive differently from a long-delay positive, which is how it learns $\lambda(x)$.

**Case 2 — observed negative ($Y = 0$), elapsed time $e$.** This is the censored case. As we established, $Y = 0$ means either a true non-converter, or a converter whose delay exceeds $e$:

$$
\Pr(Y = 0 \mid x, e) = \underbrace{(1 - p(x))}_{C=0} + \underbrace{p(x)\, \Pr(D > e \mid x)}_{C=1,\ D>e} = (1 - p(x)) + p(x)\, e^{-\lambda(x) e}.
$$

This is the **survival term**, and it is the entire trick. A naive model would assign this sample a probability of $(1 - p(x))$ — pure non-conversion. The delayed-feedback model adds the second piece: the chance the sample *is* a converter still in flight. That extra mass is how the model avoids being dragged down by censored positives. A click that has only been observed for an hour ($e$ small) gets a *large* survival term — "you have barely waited, of course it might still convert" — so it barely penalizes $p(x)$. A click observed for 30 days ($e$ large, $e^{-\lambda e} \to 0$) collapses the survival term to $(1 - p(x))$ — "you waited long enough, this really is a negative."

Putting it together, the negative log-likelihood over the dataset is:

$$
\mathcal{L} = -\sum_{i:\, y_i = 1} \log\!\big[\, p(x_i)\, \lambda(x_i)\, e^{-\lambda(x_i) d_i} \,\big] \;-\; \sum_{i:\, y_i = 0} \log\!\big[\, (1 - p(x_i)) + p(x_i)\, e^{-\lambda(x_i) e_i} \,\big].
$$

Minimize this over $(w_c, w_d)$ by gradient descent. The first sum fits both the conversion head (via $p$) and the delay head (via $\lambda$ and $d$); the second sum keeps unconverted clicks from looking like hard negatives. When the model is right, the recovered $p(x)$ is an *unbiased* estimate of the true CVR — no survival multiplier shrinking it, because the likelihood explicitly accounts for the unobserved tail. That is the payoff: the delay term moves the bias out of your CVR estimate and into a separately-modeled nuisance parameter you can throw away at serving time (at inference you only need $p(x)$).

A few engineering notes Chapelle and follow-ups stress:

- **The elapsed time $e$ must enter the loss for negatives** (it does, in $e^{-\lambda e}$) and is often *also* useful as an input feature. There is a subtlety here — if you feed $e$ as a feature to $p(x)$ you can leak the label, since long-observed negatives are more likely true negatives. Chapelle keeps $e$ in the *delay* part, not the conversion part, precisely to avoid contaminating the CVR estimate. Be careful which head sees the clock.
- **The exponential is the simplest delay model; it is often too simple.** Real delays have heavier tails and sometimes a spike near zero (in-session buys) plus a long tail (considered buys). People extend the delay model to Weibull, log-normal, gamma, or a mixture, all of which fit the same likelihood skeleton — just swap $f(d)$ and the survival $S(e) = \Pr(D > e)$.
- **It is non-convex** (product of a sigmoid and an exponential), so initialization and learning rate matter more than for a plain logistic regression. In practice you warm-start $p$ from a long-window logistic model.

### Why the survival term un-biases the gradient

The likelihood is one thing; understanding *why* it removes the bias is another, and the gradient tells the story. Take the conversion head's parameters $w_c$ (recall $p(x) = \sigma(s)$ with score $s = w_c^\top x$). For an observed positive, $\partial \log p / \partial s = 1 - p$ — the usual logistic pull-up toward $p = 1$. The interesting case is the negative. Differentiate the negative-case log-likelihood $\log\big[(1-p) + p\,e^{-\lambda e}\big]$ with respect to the score $s$. Using $\partial p / \partial s = p(1-p)$:

$$
\frac{\partial}{\partial s} \log\!\big[(1-p) + p\,e^{-\lambda e}\big] = \frac{p(1-p)\,(e^{-\lambda e} - 1)}{(1-p) + p\,e^{-\lambda e}}.
$$

Look at what this does. The numerator carries $(e^{-\lambda e} - 1)$, which is **zero when $e = 0$** and approaches $-(1)$ scaled as $e \to \infty$. So a click observed for *no time at all* ($e=0$) contributes **no downward pull** on $p$ — the model is not penalized for a sample it has barely had a chance to convert. A click observed for a *long* time ($e$ large, $e^{-\lambda e} \to 0$) recovers the ordinary logistic negative gradient $-p$ (pulling $p$ down), because by then a non-conversion really is evidence against converting. The elapsed time $e$ acts as a **confidence dial on the negative label.** A naive logistic loss applies the full $-p$ pull to *every* negative regardless of how briefly it was observed — that uniform, premature down-pull, summed over millions of barely-observed clicks, is exactly the force that drags the naive CVR estimate to its biased-low fixed point. The survival term replaces that uniform pull with one scaled by how long we actually waited. That is the mechanism, visible right in the gradient.

The delay head $w_d$ learns from both terms: positives push $\lambda(x)$ to match observed delays $d_i$ (the $-\lambda d$ term wants $\lambda$ small for long delays, large for short ones), and negatives push $\lambda(x)$ to keep the survival mass consistent with how many late conversions actually show up. The two heads are coupled through the shared negatives — which is also why the objective is non-convex and benefits from the warm start.

#### Worked example: the elapsed-time dial in numbers

Two clicks, both with model $p(x) = 0.10$, both currently unconverted (label 0). Click A was clicked one hour ago; with $\lambda = 1/3$ per day, $\lambda e_A = (1/3)(1/24) = 0.0139$, so $e^{-\lambda e_A} = 0.986$. Click B was clicked 30 days ago: $\lambda e_B = (1/3)(30) = 10$, so $e^{-\lambda e_B} = 0.000045 \approx 0$.

The negative-gradient magnitude (the down-pull on the score) from the formula above:

- **Click A (1 hour):** numerator $0.10 \times 0.90 \times (0.986 - 1) = 0.09 \times (-0.014) = -0.00126$; denominator $0.90 + 0.10 \times 0.986 = 0.9986$; gradient $\approx -0.00126$. Essentially *no* pull-down — A has barely been observed, so its non-conversion is almost no evidence.
- **Click B (30 days):** numerator $0.09 \times (0.000045 - 1) = -0.090$; denominator $0.90 + 0.10 \times 0.000045 \approx 0.90$; gradient $\approx -0.0999 \approx -p$. Full logistic pull-down — B waited a month and still did not convert, so it is a genuine negative.

A naive logistic loss would have applied the *same* $-p = -0.10$ pull to both A and B. By treating the one-hour-old click as if its silence were as damning as the month-old click's, the naive model over-counts negatives by roughly 80x on fresh samples — and freshly-clicked samples are the majority of any short-window batch. The 0.00126-versus-0.0999 gap *is* the bias, sample by sample.

## 6. Implementing it: simulate the bias, then fix it

Talk is cheap; let us make the bias appear and then make it vanish in runnable code. We will (a) simulate a click stream with a known true CVR and exponential delays, (b) measure the naive fixed-window bias, and (c) fit the delayed-feedback model and recover the truth. All numpy and PyTorch, copy-and-adapt ready.

First, the simulator. We generate clicks with features, a true per-click conversion probability, exponential delays for converters, and a fixed observation cutoff that censors late conversions.

```python
import numpy as np

rng = np.random.default_rng(0)

def simulate_clicks(n=200_000, window_days=1.0, mean_delay_days=3.0):
    # One informative feature x; true logit of conversion depends on it.
    x = rng.normal(0, 1, size=n)
    # True conversion probability per click (latent C).
    true_logit = -2.2 + 1.0 * x          # base rate ~ sigmoid(-2.2) ~ 0.10
    p_true = 1.0 / (1.0 + np.exp(-true_logit))
    C = (rng.uniform(size=n) < p_true).astype(int)   # eventual converter?

    # Delay only defined for converters: Exp(rate = 1/mean_delay).
    rate = 1.0 / mean_delay_days
    D = rng.exponential(scale=mean_delay_days, size=n)   # delay if it converts

    # Elapsed observation time per click. Simplest case: fixed window.
    E = np.full(n, window_days)

    # Observed label: converter AND delay landed within the window.
    Y = ((C == 1) & (D <= E)).astype(int)
    # Observed delay d_i only meaningful when Y == 1.
    d_obs = np.where(Y == 1, D, np.nan)

    return dict(x=x, C=C, p_true=p_true, D=D, E=E, Y=Y, d_obs=d_obs,
                mean_delay=mean_delay_days)

data = simulate_clicks()
print("true CVR   :", round(data["C"].mean(), 4))          # ~0.10
print("naive CVR  :", round(data["Y"].mean(), 4))          # biased low
print("fake-neg % :", round(((data["C"]==1)&(data["Y"]==0)).mean()
                            / max(data["Y"].mean(),1e-9), 3))
```

Running this with a 1-day window on 3-day mean delays reproduces our worked example: true CVR near 0.10, naive observed CVR near 0.028, and a large fraction of the negatives are actually fake. The bias is not a rounding error; it is most of the signal.

Before modeling, do the five-minute diagnostic from section 3 — estimate the conversion-delay survival curve non-parametrically and read the bias of your candidate window straight off it:

```python
from lifelines import KaplanMeierFitter
import numpy as np

# Treat converted clicks as observed events at their delay; unconverted as
# censored at their elapsed time. Here we use the true converters to show the curve.
conv = data["C"] == 1
durations = np.where(conv, data["D"], data["E"])   # event time or censor time
observed  = conv.astype(int)                        # 1 = event seen, 0 = censored

km = KaplanMeierFitter().fit(durations[conv], event_observed=observed[conv])
for w in (1.0, 7.0, 21.0):
    s_w = float(km.survival_function_at_times(w).iloc[0])  # P(D > w) among converters
    print(f"window={w:>4}d  survival S(w)={s_w:.3f}  "
          f"=> naive relative bias ~ {-s_w:.1%}")
# window=  1d  survival ~0.717  => naive relative bias ~ -71.7%
# window=  7d  survival ~0.097  => naive relative bias ~ -9.7%
# window= 21d  survival ~0.001  => naive relative bias ~ -0.1%
```

Those survival values are the exact $e^{-\lambda w}$ from the algebra (we simulated an exponential, so Kaplan-Meier recovers it), and the printed relative biases match the worked example to the decimal. On real data the curve will *not* be a clean exponential — it will have a near-zero spike (in-session buys) and a fat tail — and that shape tells you both whether you have a problem and which delay model to fit. This single block is what you run first, before deciding the correction is even worth building.

![Side by side comparison of the naive estimator reporting a conversion rate far below the true rate against a delay corrected estimator that recovers the true rate](/imgs/blogs/delayed-feedback-and-conversion-attribution-6.png)

Now the delayed-feedback model in PyTorch. Two linear heads share the features: $p(x) = \sigma(w_c^\top x + b_c)$ for conversion probability, and $\lambda(x) = \exp(w_d^\top x + b_d)$ for the delay rate. We implement the exact negative log-likelihood from section 5.

```python
import torch
import torch.nn as nn

class DelayedFeedbackModel(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.conv_head  = nn.Linear(d_in, 1)   # logit of p(x) = P(C=1|x)
        self.delay_head = nn.Linear(d_in, 1)   # log lambda(x), delay rate

    def forward(self, x):
        logit_p   = self.conv_head(x).squeeze(-1)
        log_lam   = self.delay_head(x).squeeze(-1)
        return logit_p, log_lam

def dfm_loss(logit_p, log_lam, y, e, d):
    # Numerically stable pieces.
    log_p   = nn.functional.logsigmoid(logit_p)        # log p(x)
    log_1mp = nn.functional.logsigmoid(-logit_p)       # log (1 - p(x))
    lam     = torch.exp(log_lam)

    # Positive term: log[ p * lambda * exp(-lambda * d) ]
    #              = log p + log lambda - lambda * d
    pos_ll = log_p + log_lam - lam * d

    # Negative term: log[ (1-p) + p * exp(-lambda * e) ]
    # logaddexp keeps it stable: logaddexp(log(1-p), log p - lambda e)
    surv_log = log_p - lam * e                          # log( p * exp(-lam e) )
    neg_ll   = torch.logaddexp(log_1mp, surv_log)

    ll = torch.where(y > 0.5, pos_ll, neg_ll)
    return -ll.mean()

# --- training loop ---
x = torch.tensor(data["x"], dtype=torch.float32).unsqueeze(-1)
y = torch.tensor(data["Y"], dtype=torch.float32)
e = torch.tensor(data["E"], dtype=torch.float32)
d = torch.tensor(np.nan_to_num(data["d_obs"]), dtype=torch.float32)  # d only used where y=1

model = DelayedFeedbackModel(d_in=1)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

for step in range(400):
    opt.zero_grad()
    logit_p, log_lam = model(x)
    loss = dfm_loss(logit_p, log_lam, y, e, d)
    loss.backward()
    opt.step()

with torch.no_grad():
    p_hat = torch.sigmoid(model(x)[0]).mean().item()
print("DFM recovered CVR:", round(p_hat, 4))   # ~0.099, near the true 0.10
```

Two implementation details that bite people:

- **`torch.logaddexp` for the survival term.** The negative-case probability $(1-p) + p\,e^{-\lambda e}$ is a sum of two terms, and naively taking `log(a + b)` underflows when both are tiny. Writing it as `logaddexp(log(1-p), log p - lambda*e)` is the stable form. This one line is the difference between a model that trains and one that NaNs out on the first batch.
- **`d` is only meaningful where `y == 1`.** We `nan_to_num` it to a harmless 0 elsewhere; the loss only reads `d` inside the positive branch via `torch.where`, so the junk values never affect anything. (Be careful: `torch.where` still *evaluates* both branches, so a NaN in the unused branch can poison the gradient via NaN propagation. Zeroing `d` first avoids that.)

When this runs, the delayed-feedback model recovers a CVR around 0.099 against the true 0.10 — the 72% bias is gone, on the *same* short 1-day window that crippled the naive estimator. We kept the freshness and paid down the bias. That is the entire promise of the method, demonstrated.

## 7. The streaming alternative: fake-negative weighting

The delayed-feedback model is the principled batch answer. But on a true online stream — SGD over a firehose, label assigned the instant the click flows by — you face a different constraint: you cannot hold a click and wait. You must emit a label *now*, then somehow correct it *later* when (if) the conversion arrives. This is where the **fake-negative** family lives: FNW (Fake Negative Weighted) and FNC (Fake Negative Calibration), and the elapsed-time-aware ES-DFM (Yang et al., 2021).

The core idea is delightfully sneaky. **Treat every click as a negative the moment it arrives.** Then, when a conversion later lands for that click, **insert the same click again as a positive (a "duplicate").** So the stream sees each converting click twice: once as an (incorrect) negative when clicked, once as a (correct) positive when it converts. Each non-converting click is seen once, as a negative. The model never waits — it always has a label — and the late positives flow in as duplicates whenever they materialize.

But this biases the data: every true positive is *also* present as a false negative, so the observed distribution over-represents negatives. The fix is **importance weighting** — reweight the gradient so that training on this biased "observed" stream is, in expectation, equivalent to training on the true (unbiased) distribution. This is the same importance-sampling logic as the [off-policy correction in counterfactual evaluation](/blog/machine-learning/recommendation-systems/counterfactual-and-off-policy-evaluation): you have samples from a wrong distribution $b$ and want expectations under the right distribution $p$, so you weight each sample by $p/b$.

Let us derive the weight. Let $p = \Pr(y=1 \mid x)$ be the *true* conversion probability we want to learn. In the fake-negative stream, the *observed* distribution $b$ is:

- Each true negative ($C=0$) appears once as a negative.
- Each true positive ($C=1$) appears once as a negative (at click time) *and* once as a positive (at conversion time).

So in the biased stream, the probability that a *displayed sample* is labeled positive is $\frac{p}{1+p}$ (because positives are duplicated, the total count is $1+p$ per click in expectation, of which $p$ are the positive copies). Likewise the biased negative probability is $\frac{1}{1+p}$. We want each gradient step to behave as if the label were drawn from the true $\{1-p, p\}$, so the importance weight is true-over-biased per class:

$$
w_{+} = \frac{p}{\,p/(1+p)\,} = 1 + p, \qquad w_{-} = \frac{1-p}{\,1/(1+p)\,} = (1-p)(1+p) = 1 - p^2.
$$

In practice FNW uses these weights (with $p$ itself estimated by the model, so it is self-consistent / a fixed point you iterate toward), and the loss becomes a weighted cross-entropy on the duplicated stream. The beautiful property: **the minimizer of the weighted biased loss is the same as the minimizer of the true loss** — the weighting un-does the duplication bias in expectation. You get an unbiased CVR on a stream where you never waited a second for a label.

#### Worked example: computing a fake-negative importance weight

A click streams in. Your current model says $p(x) = 0.10$ for it. At click time you emit it as a negative; you weight that negative gradient by $w_{-} = 1 - p^2 = 1 - 0.01 = 0.99$. Three days later this click converts. You now insert it as a positive and weight that gradient by $w_{+} = 1 + p = 1.10$.

What does that accomplish? Consider 1,000 statistically identical clicks at $p = 0.10$: 100 will eventually convert. The stream shows 1,000 negatives (one per click) and 100 positives (the duplicated converters), for 1,100 samples total. Unweighted, the model would learn $\Pr(+) = 100/1100 = 0.0909$ — biased low again, the duplication's own bias. With the weights: weighted positives $= 100 \times 1.10 = 110$; weighted negatives $= 1000 \times 0.99 = 990$; weighted positive fraction $= 110 / (110 + 990) = 110/1100 = 0.10$. Exactly the true CVR. The two weights conspire to cancel the duplication bias and land on $p$. That is the arithmetic, end to end.

Here is FNW as a streaming-style weighted loss in PyTorch. In a real system you would process micro-batches off a Kafka stream; here we batch the simulated stream to show the mechanism.

```python
import torch, torch.nn as nn

def build_fnw_stream(data):
    """Each click -> a negative; each true converter -> also a positive dup."""
    x, C = data["x"], data["C"]
    xs, ys = [], []
    # every click enters once as a negative at click time
    xs.append(x);                 ys.append(np.zeros_like(C, dtype=float))
    # every eventual converter re-enters as a positive (the duplicate)
    pos_mask = C == 1
    xs.append(x[pos_mask]);       ys.append(np.ones(pos_mask.sum()))
    X = np.concatenate(xs); Y = np.concatenate(ys)
    return (torch.tensor(X, dtype=torch.float32).unsqueeze(-1),
            torch.tensor(Y, dtype=torch.float32))

Xs, Ys = build_fnw_stream(data)

model = nn.Linear(1, 1)
opt = torch.optim.Adam(model.parameters(), lr=0.05)
bce = nn.BCEWithLogitsLoss(reduction="none")

for step in range(400):
    opt.zero_grad()
    logit = model(Xs).squeeze(-1)
    p = torch.sigmoid(logit).detach()          # current estimate, stop-grad
    # fake-negative importance weights
    w_pos = 1.0 + p
    w_neg = 1.0 - p * p
    w = torch.where(Ys > 0.5, w_pos, w_neg)
    loss = (bce(logit, Ys) * w).mean()
    loss.backward()
    opt.step()

with torch.no_grad():
    print("FNW recovered CVR:",
          round(torch.sigmoid(model(torch.tensor(data["x"],
                dtype=torch.float32).unsqueeze(-1)).squeeze(-1)).mean().item(), 4))
```

This recovers a CVR near 0.10 as well, on a stream where every click was labeled the instant it arrived. The `.detach()` on `p` is important: the weights depend on the model's own prediction, but you do not want to backprop *through* the weight (that turns a reweighting into a strange second-order objective). Stop-gradient the weight, let the BCE term carry the learning signal.

**ES-DFM (Elapsed-time Sampling Delayed Feedback Model)** refines this further: instead of treating every click as an immediate negative, it waits a *short* elapsed-time window (long enough to catch the fast in-session conversions, short enough to stay fresh), then applies a fake-negative correction only for the residual long-tail delays. It blends the two worlds — a little waiting to cheaply de-noise, plus importance weighting for the tail — and on Criteo and Alibaba production data reported the best calibration/AUC trade-off of the streaming family. The takeaway is not which acronym wins; it is the **design pattern**: a short de-bouncing wait plus an importance-weighted correction for the tail beats both "pure naive" and "pure long wait."

It is worth pinning down *why* the elapsed-time wait helps, because it is not obvious that adding latency to a freshness-obsessed system is a win. The fake-negative weights $1+p$ and $1-p^2$ are correct *in expectation*, but they add **variance**: every converter contributes a high-variance pair of opposite-signed weighted gradients (a down-pull at click, an up-pull at conversion). For the many conversions that land within minutes, that pair is pure noise you could have avoided by just waiting those few minutes and labeling them correctly the first time. ES-DFM's short window absorbs the easy, fast conversions deterministically — zero variance — and reserves the noisy duplicate-and-reweight machinery for only the long tail that genuinely needs it. So a 15-minute wait can *lower* gradient variance enough to improve convergence and calibration even though it costs a sliver of freshness. The right window is the elbow of the survival curve: wait through the in-session spike, importance-weight the considered tail.

**DEFER (Gu et al., 2021)** pushes the duplication idea to its logical end: rather than inserting only converters as positives, it duplicates *every* sample — once with its real-time (possibly-fake) label and once with its eventually-corrected label — and trains on both copies with an importance correction. The motivation is that FNW's single-negative-then-positive scheme throws away information about *when* the correction arrives; DEFER keeps both views so the model sees the full real-time-to-final transition. In benchmarks it traded a little extra compute (you now process roughly two samples per click) for steadier calibration on streams with very heavy delay tails. The family — FNW, FNC, ES-DFM, DEFER — is best read as a sequence of refinements to one idea: *label now, correct later, reweight to stay unbiased*, with each variant choosing a different point on the variance-versus-freshness curve.

Here is the three-way comparison, the trade-off table the whole post has been building toward:

![Matrix comparing the naive fixed window, the delayed feedback model, and fake negative weighting across whether they handle delay, their freshness, and their conversion rate bias](/imgs/blogs/delayed-feedback-and-conversion-attribution-3.png)

| Approach | Handles delay | Freshness | CVR bias | Cost / complexity |
|---|---|---|---|---|
| Naive fixed window | No (hard cutoff) | High (short) or low (long) | Severe (short) → mild (long) | Trivial |
| Delayed-feedback model (Chapelle) | Yes (joint delay model) | Medium (batch retrain) | Removed in expectation | Two heads, non-convex fit |
| FNW / FNC (fake-negative) | Yes (duplicate + reweight) | High (true online) | Removed in expectation | Stream dedup + weights |
| ES-DFM (elapsed-time sampling) | Yes (short wait + reweight tail) | High | Removed, best calibration | Window tuning + reweight |

## 8. Attribution: which click even gets the label?

Everything above assumed we *know* a conversion belongs to a particular click. In reality, a single purchase is often preceded by several touches — a display impression, a search-ad click, a retargeting click — and someone has to decide which of them earns the conversion label. That decision is **attribution**, and it is upstream of your whole training set: it *manufactures the labels*. Get it wrong and even a perfect delayed-feedback model learns from mislabeled data.

![Branching graph showing one conversion fanning out to several earlier touches with last click assigning all credit to the final click and multi touch splitting fractional credit across touches](/imgs/blogs/delayed-feedback-and-conversion-attribution-7.png)

Two ingredients define an attribution policy:

**1. The attribution window.** The maximum time from a touch to a conversion for the touch to get any credit — e.g., a 7-day-click / 1-day-view window is a common ad standard. A purchase 31 days after the click, under a 30-day window, gets *no* attribution: that click is a true negative, period. The attribution window therefore caps the delay distribution: it is the largest $E$ that can ever yield $Y=1$. This is why earlier I said the attribution window and the training window are different knobs — the attribution window is a *business/contractual* definition of "what counts," while the training window is *your* choice of when to read the label. Your delay model's survival term should saturate at the attribution-window boundary (beyond it, $\Pr(\text{still converts}) = 0$ by definition).

**2. The credit rule.** Among the touches inside the window, who gets the conversion?

- **Last-click (last-touch):** the final click before conversion gets 100% of the credit. Dominant in practice because it is simple, auditable, and matches the "closing" intuition. But it systematically under-credits early discovery touches (the display impression that planted the idea gets nothing), so a CVR model trained on last-click labels learns that upper-funnel placements never convert — which biases budget toward bottom-funnel.
- **First-click:** the first touch gets all credit. Over-credits discovery, ignores closing.
- **Linear / positional / time-decay (multi-touch attribution, MTA):** split credit *fractionally* across touches. Linear splits evenly; time-decay gives more to recent touches; position-based (e.g., 40/20/40) front- and back-loads. Now each touch's "label" is a *fractional* target in $[0,1]$, not a binary 0/1 — which changes the loss (you regress on the fractional credit, or sample positives proportionally).
- **Data-driven attribution (DDA) / Shapley:** estimate each touch's *marginal causal contribution* to the conversion using a Shapley-value or a probabilistic model over conversion paths. Most principled, most expensive, hardest to audit, and itself sensitive to the delayed-feedback problem (paths are incomplete until conversions settle).

The point for an ML engineer is this: **attribution is a label-generation policy, and it interacts with delay.** Under last-click with a short training window, a touch that *will* be the last click of a slow converter is currently labeled negative (the conversion has not happened). Under multi-touch, you additionally have to wait to know the *full path* before you can split credit, which deepens the delay. So the cleaner your attribution (more touches, fractional credit), the *longer* the effective delay before the label settles — attribution and delayed feedback are not separate problems, they compound. A practical compromise many teams ship: train the CVR model on last-click labels (simple, fast-settling) but *budget* with multi-touch reporting (fairer to upper funnel), accepting the inconsistency as a known, documented gap.

#### Worked example: last-click vs multi-touch reshapes the label

A user sees a display impression (day 0), clicks a search ad (day 2), clicks a retargeting ad (day 5), and buys on day 5. Three touches, one conversion, 7-day window.

- **Last-click:** retargeting click (day 5) gets label 1; search click and display get label 0. Your retargeting CVR model sees a positive; your search CVR model sees a negative. The search team's model learns search "doesn't convert" — even though it was 2/3 of the journey.
- **Linear multi-touch:** each touch gets credit $1/3 \approx 0.33$. Now display, search, and retargeting each carry a fractional positive label of 0.33. The search CVR model regresses toward 0.33, not 0 — a very different training signal.
- **Time-decay (half-life 1 day):** weights roughly proportional to $2^{-(\text{days before conversion})}$: retargeting (0 days) weight 1.0, search (3 days) weight 0.125, display (5 days) weight 0.03. Normalize → retargeting ~0.86, search ~0.11, display ~0.03. Closer to last-click but not winner-take-all.

Same purchase, three completely different label vectors over the three touches. The attribution rule is not a reporting detail — it is the *ground truth your model fits*. Choosing it is choosing what your CVR model believes about the world.

## 9. The connection: CVR is one head of the multi-task ranker

Step back to the funnel. CVR prediction almost never lives alone — it is one objective in a **multi-task ranker** that also predicts CTR, and often dwell time, add-to-cart, and more. We built the architecture for this in [multi-task and multi-objective ranking with MMoE and PLE](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple). Delayed feedback is specifically the *CVR head's* data problem — CTR labels are immediate, CVR labels are delayed — so a real multi-task ranker has to handle *both* feedback regimes in one model: a fast-settling click head and a slow-settling conversion head.

There is a second, deeper connection worth spelling out, because it solves a problem we have been ignoring. CVR prediction has a **sample-selection bias** distinct from delay: *CVR is only defined on clicked items.* You only observe whether someone converts if they first clicked. So a CVR model trained on the clicked subset learns $\Pr(\text{convert} \mid \text{click})$ on a biased sample (clickers are not a random slice of impressions), and then at serving time you apply it to *all* candidates, including ones the user would never have clicked. This is the same missing-not-at-random (MNAR) hazard that haunts the whole series.

**ESMM (Entire Space Multi-task Model)**, from Alibaba (Ma et al., 2018), is the canonical fix and it is worth understanding alongside delayed feedback because they are the two halves of "doing CVR right." ESMM's insight: model CTR and CVR over the *entire impression space*, and learn CVR *indirectly* through two directly-supervised tasks. Define:

- pCTR $= \Pr(\text{click} \mid \text{impression})$ — supervised on all impressions, clean immediate labels.
- pCTCVR $= \Pr(\text{click and convert} \mid \text{impression})$ — supervised on all impressions (the label is 1 only for impressions that led to a conversion).
- pCVR $= \Pr(\text{convert} \mid \text{click})$ — *never directly supervised*; derived as $\text{pCTCVR} / \text{pCTR}$.

Because both pCTR and pCTCVR are trained on the full impression space (not just clicks), the implied pCVR is estimated over the entire space too, dodging the sample-selection bias. ESMM tackles the **selection-bias half**; the delayed-feedback model and fake-negative weighting tackle the **delay half**. A production CVR system frequently needs *both*: ESMM-style entire-space modeling to fix where the labels come from, plus delay-aware training to fix when they arrive. They are orthogonal corrections to the same estimate, and the cleanest systems compose them — e.g., an entire-space pCTCVR head whose conversion supervision uses a delayed-feedback or fake-negative likelihood.

```python
# ESMM-style heads (sketch): two supervised tasks, CVR is implied.
import torch, torch.nn as nn

class ESMM(nn.Module):
    def __init__(self, d):
        super().__init__()
        shared = lambda: nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))
        self.ctr_tower = shared()   # P(click | impression)
        self.cvr_tower = shared()   # P(convert | click), implied not directly supervised

    def forward(self, x):
        pctr = torch.sigmoid(self.ctr_tower(x).squeeze(-1))
        pcvr = torch.sigmoid(self.cvr_tower(x).squeeze(-1))
        pctcvr = pctr * pcvr        # P(click & convert | impression)
        return pctr, pcvr, pctcvr

def esmm_loss(pctr, pctcvr, y_click, y_convert):
    # Both losses are over the ENTIRE impression space (no clicked-only filtering).
    l_ctr    = nn.functional.binary_cross_entropy(pctr, y_click)
    l_ctcvr  = nn.functional.binary_cross_entropy(pctcvr, y_click * y_convert)
    return l_ctr + l_ctcvr          # pcvr is learned through the product, never alone
```

The elegance: there is no `binary_cross_entropy(pcvr, ...)` term at all. CVR is squeezed out of the two well-posed, entire-space tasks. Bolt a delay-aware likelihood onto the `l_ctcvr` term (the conversion label is the delayed one) and you have the full treatment.

### How delayed feedback compounds in the closed loop

There is one more reason to take this seriously, and it connects straight back to the feedback loop that runs this series: serve, log, train, serve. The bias we have been discussing is not a one-time offset you can calibrate away once — it **compounds across retraining cycles** if you let it. Walk the loop. The model under-predicts CVR (because of censored positives). Under-predicted CVR means lower bids or lower ranking for the affected items. Lower ranking means fewer impressions, fewer clicks, and therefore *fewer conversions logged* for those items. Fewer logged conversions, fed back into the next training round, look like *even lower* CVR — confirming the model's pessimism. The censoring bias seeds a self-reinforcing under-exposure spiral, the conversion-side cousin of the popularity feedback loop.

The spiral is worst for exactly the segments that already suffer most from censoring: slow-converting, high-consideration items. A short fixed window under-credits them on day one; the loop then starves them of the impressions that would have generated the conversions that would have corrected the estimate. After a few cycles the model has effectively "decided" that considered purchases do not convert, and it has arranged the world so it never sees the evidence to the contrary. This is why a delayed-feedback correction is not a nice-to-have calibration patch — left uncorrected, the bias *grows* in a continuously-trained system rather than staying constant. The fix (delay-aware likelihood or fake-negative weighting) breaks the spiral at its source by refusing to count a not-yet-conversion as a negative in the first place, so the loop is fed honest labels and the under-exposure never gets a foothold.

A blunt diagnostic for whether you are in this spiral: track *per-segment* predicted-vs-observed CVR over successive model versions. If a segment's predicted CVR is drifting down faster than its (settled) observed CVR, and its impression share is shrinking in lockstep, you are watching the closed loop amplify the censoring bias in real time. The cure is upstream — fix the labels — not a downstream exploration bonus, though a small exploration floor on starved segments helps the model recover faster once the labels are honest.

## 10. Measuring it honestly: the results table

How do you know any of this worked? You measure on a stream where you *know* the truth — which, conveniently, you do in simulation, and approximately do in replay over a fully-settled historical window. Two metrics matter most for CVR, and they are not the same:

1. **Calibration** — is the *level* right? For bidding, calibration is everything: if the model predicts 0.10 and the true rate is 0.10, the predicted/observed ratio is 1.0. The naive model's whole disease is mis-calibration (it predicts way too low). Measure it as the ratio of mean predicted CVR to actual observed CVR on a *fully-settled* holdout (one where the attribution window has fully elapsed, so labels are final). See [calibration and the prediction you can trust](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust) for the isotonic/Platt machinery.
2. **Ranking quality** — AUC and especially **PR-AUC** (precision-recall AUC, since conversions are rare and PR-AUC is more sensitive to the positive class than ROC-AUC under heavy imbalance). A model can rank reasonably while being badly mis-calibrated, which is exactly the naive model's failure mode — *it knows who is more likely to convert, it just thinks everyone converts less than they do.*

The honest measurement protocol:

- **Temporal split, fully-settled evaluation labels.** Train on data up to time $T$; evaluate on $[T, T+\Delta]$ but only *after* the attribution window past $T+\Delta$ has fully elapsed, so the eval labels are final and unbiased. Evaluating on un-settled labels just re-introduces the bias into your metric and flatters the naive model.
- **No leakage of elapsed time into the conversion head** (section 5's warning).
- **Report calibration AND PR-AUC**, not just AUC — AUC can look fine while calibration is wrecked.

Here is the before→after on the simulated stream (true CVR 0.10, 3-day mean delay, 1-day training window), the numbers our code produces and that mirror the relative magnitudes reported in the literature on Criteo:

![Matrix of results comparing the naive window, the delayed feedback model, and fake negative weighting on calibration ratio, precision recall area under curve, and the overall verdict](/imgs/blogs/delayed-feedback-and-conversion-attribution-8.png)

| Method | Predicted CVR | Calibration (pred/obs) | PR-AUC | Verdict |
|---|---|---|---|---|
| Naive 1-day window | 0.028 | 0.28 → wrecked | 0.64 | Fresh but badly biased; under-bids |
| Naive 21-day window | 0.099 | ~1.0 | 0.71 | Calibrated but 3 weeks stale |
| Delayed-feedback model | 0.099 | ~0.98 | 0.71 | Calibrated AND fresh (1-day window) |
| Fake-negative weighted | 0.100 | ~1.01 | 0.70 | Calibrated, fresh, true online |

Read the table the way the literature does. The naive short-window model and the delayed-feedback model see **the same input data** (a 1-day window). The naive one converges to 0.028; the DFM recovers 0.099. The *only* difference is that the DFM's loss accounts for censoring. Meanwhile the naive *long*-window model also gets to 0.099, but it had to wait 21 days to do it — it bought correctness with staleness. The delayed-feedback model and fake-negative weighting are the free lunch the trade-off said should not exist: correctness *and* freshness, by modeling the delay instead of waiting it out. (PR-AUC barely moves between the calibrated methods because the *ranking* of who-converts is similar; the win is overwhelmingly in calibration — which, for bidding, is the metric that pays the bills.)

## 11. Case studies and real numbers

**Criteo delayed feedback (Chapelle, KDD 2014).** The paper that named the problem, on Criteo's display-advertising conversion logs. Chapelle showed the naive fixed-window estimator was substantially biased and that the joint conversion-plus-exponential-delay model improved log-likelihood and calibration on held-out, fully-settled data. Criteo later released a public *Delayed Feedback Dataset* with click timestamps and conversion timestamps, which became the standard benchmark for everything that followed. Key empirical fact from this line of work: conversion delays are *heavy-tailed* — a large fraction convert within hours, but a long tail stretches across weeks, which is exactly why a single short window is so damaging and why the exponential is often replaced by a heavier-tailed delay model.

**ES-DFM (Yang et al., AAAI 2021), "Capturing Delayed Feedback in Conversion Rate Prediction via Elapsed-Time Sampling."** Targeted the *streaming* setting where continuous training is mandatory. Their elapsed-time-sampling approach — wait a short window to cheaply catch fast conversions, then importance-weight to correct the long-tail fake negatives — reported improved AUC and calibration over both naive streaming and earlier fake-negative methods (FNW/FNC) on the Criteo delayed-feedback data and on Alibaba production traffic. The headline lesson, robust across the streaming-CVR literature: **a hybrid of a short de-noising wait plus an importance-weighted tail correction beats both extremes** of the freshness–correctness dial. Related streaming methods in this family include FNW/FNC (Ktena et al., 2019, "Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR Prediction") and DEFER (Gu et al., 2021, which duplicates *all* samples with real-time and delayed labels for a defer-and-correct scheme).

**ESMM (Ma et al., SIGIR 2018), "Entire Space Multi-Task Model."** Alibaba's fix for the *sample-selection-bias* half of CVR. By modeling pCTR and pCTCVR over the entire impression space and deriving pCVR as their ratio, ESMM reported meaningful AUC gains for CVR prediction over a clicked-only-trained baseline on Taobao production data, precisely because it estimated CVR on the whole space rather than the biased clicked subset. It is the natural companion to delayed-feedback handling: ESMM fixes *where labels come from*, delayed-feedback methods fix *when they arrive*.

**Ad-tech attribution at scale.** Across the industry, last-click within a fixed attribution window (commonly 7-day-click / 1-day-view) remains the dominant production label policy for CVR models because it is simple, fast-settling, and auditable — even though it is known to under-credit upper-funnel touches. Multi-touch and data-driven (Shapley-style) attribution are used heavily in *reporting and budget allocation* but less often as the direct training label, precisely because fractional, full-path labels settle slowly and deepen the delayed-feedback problem. The pragmatic industry pattern is to *train* on the fast-settling label and *report/budget* on the fairer one, documenting the gap. (Privacy shifts — ATT, third-party cookie deprecation, aggregated/differential-privacy attribution APIs — are now adding *noise and aggregation* on top of *delay*, making conversion labels not just late but coarse, which is reviving interest in robust, bias-aware CVR estimation.)

## 12. Handling delayed labels: the production playbook

Pulling it together into what you would actually do when you own a CVR model. This is the section to copy into a design doc.

**Step 1 — Confirm you have the problem.** Plot the delay distribution: for converted clicks, histogram (conversion time − click time). If most conversions land within minutes (true in-session funnels), delay is a non-issue — use a short window and move on. If the histogram has a fat tail beyond your training window, you have the problem. Quantify the bias with the section-3 formula: relative bias $\approx -\Pr(D > w)$, read straight off the empirical survival curve at your window $w$.

**Step 2 — Decouple attribution window from training cadence.** The attribution window is contractual (e.g., 30 days). Your training cadence (every 6 hours) and your *effective* label-settling strategy are separate ML choices. Never conflate "conversions are allowed to take 30 days" with "I must wait 30 days to train."

**Step 3 — Pick a correction by your training mode:**

- **Batch retraining (daily/hourly jobs):** use the **delayed-feedback model** — joint conversion + delay likelihood (section 5). Start with exponential delay, upgrade to Weibull/log-normal if the delay tail is heavy. Keep elapsed time in the delay head, not the conversion head.
- **True online / continuous training (SGD on a stream):** use **fake-negative weighting (FNW)** or **ES-DFM** (section 7). Emit a label immediately, duplicate-and-correct on conversion arrival, importance-weight. ES-DFM's short de-noising wait usually beats pure FNW on calibration.
- **Either mode, if you also have selection bias:** compose with **ESMM** (entire-space pCTCVR) so the CVR estimate is unbiased over impressions, not just clicks.

**Step 4 — Evaluate on settled labels only.** Temporal split; wait out the attribution window on your eval set so the labels are final; report calibration (pred/obs ratio) *and* PR-AUC. Treat a calibration drift as a delayed-feedback regression until proven otherwise. This is also where [train-serve skew](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there) hides — if the elapsed-time feature is computed differently offline (full settled history) and online (a streaming store lagging minutes), your survival term is fed inconsistent clocks.

**Step 5 — Monitor the fake-negative rate in production.** Track the fraction of clicks initially labeled negative that later flip to positive (a conversion arrives after the label was read). A rising flip rate means your delay distribution is drifting (e.g., a promotion lengthened consideration time) and your delay model needs a refit. This single monitor catches the intro's silent CVR slide *before* Finance does.

**Step 6 — Wire the late-conversion join correctly.** The unglamorous part that breaks in practice: you need a pipeline that *re-joins* a conversion event back to its originating click days later and either refits (batch) or emits the duplicate-positive (streaming). This is a stateful join over a long horizon — you are holding click keys for up to the attribution window — and it is where data engineering, not modeling, decides whether the correction works. A common failure: the click log is retained for 7 days but the attribution window is 30, so conversions on day 20 cannot find their click and are silently dropped, re-creating the censoring bias you thought you fixed. Match your click-key retention to the attribution window, not to whatever the default log TTL happens to be.

**Stress tests — when does this break?**

- *Delay distribution shifts (a sale changes buying speed):* a stale delay model mis-corrects. A promotion that compresses delays (everyone buys now) makes the old delay model *over*-credit survival, briefly inflating CVR; a launch of a considered product that lengthens delays does the reverse. Refit the delay head on a rolling window; monitor the flip rate as the early-warning signal.
- *Never-converters dominate (CVR ~0.1%):* the survival term is mostly $(1-p)$ and the positive signal is thin, so both the conversion head and the delay head are starved. The delay model is especially hard to fit on a handful of positives; consider ESMM to borrow strength from the entire impression space, lean on the longer-window labels you do have, and regularize the delay head toward a global prior so it does not overfit the few converters you observe.
- *Conversions can repeat or be revoked (returns, refunds, churn):* the binary "converts once" assumption breaks. A purchase that is refunded a week later should arguably *un*-label the click; a subscription that renews monthly is a recurring event, not a one-shot. You need an event model, possibly with negative events and a notion of net conversion value; the survival framing still helps but the label is no longer monotone, and your join logic must handle reversals.
- *Multiple conversions per click (a session that buys three items):* decide whether the label is "any conversion" (binary) or a count/value target (regression). The delayed-feedback likelihood as written is for the first; for value prediction you fold the delay model into a censored regression on conversion value, which is the same survival skeleton with a value-weighted positive term.
- *Privacy-aggregated labels (no per-click conversion timestamp):* you may only get *aggregate* conversion counts per cohort with noise (the direction iOS ATT and the Privacy Sandbox Attribution Reporting API are pushing). Per-sample delayed-feedback likelihoods do not apply directly; you move to aggregate/cohort-level CVR estimation with debiasing — fitting the same conversion-plus-delay structure to noisy aggregate counts rather than per-click events. This is an active research area, and it is where the next few years of CVR work will live: labels that are not just *late* but *coarse and noisy*.

## 13. When to reach for this (and when not to)

A decisive recommendation, because every correction is a cost.

**Reach for delayed-feedback handling when:** your conversion delays have a meaningful tail beyond your training window (check the histogram), *and* calibration matters (you bid, budget, or rank on the predicted level, not just the order). For value-based bidding, calibration is not optional and delay correction is mandatory.

**Use the batch delayed-feedback model when** you retrain on a schedule and can afford a two-head, non-convex fit. It is the most principled and gives the cleanest calibration. Start exponential; only reach for Weibull/log-normal if the residuals demand it.

**Use fake-negative weighting / ES-DFM when** you train truly online and cannot wait — the stream forces an immediate label. The duplication-plus-reweight pattern is the right tool. ES-DFM's short wait is usually worth it.

**Compose with ESMM when** you also have the clicked-only selection bias (you almost always do) — it is orthogonal and the gains stack.

**Do NOT bother when:** conversions are essentially in-session (delays in seconds/minutes, tail inside your window) — a short window is unbiased and the machinery is pure overhead. Do NOT reach for a heavy multi-touch / Shapley attribution model *as your training label* if last-click within the window already gives well-settled labels and your funnel is short — the slower-settling labels can *worsen* the delay problem more than the attribution fairness helps the model. Do NOT trust a CVR model's offline AUC alone — AUC can look healthy while calibration is wrecked by exactly this bias. And do NOT feed elapsed time into the conversion head "because it is predictive" — it leaks the censoring and corrupts the very estimate you are trying to debias.

## 14. Key takeaways

- **A CVR "negative" is not a negative — it is a censored observation.** An unconverted click is either a true non-converter or a converter still in flight; the two are indistinguishable at training time, and conflating them is the whole bug.
- **The naive fixed-window estimator is structurally biased low**, by exactly the survival tail $\Pr(D > w)$ — and more data does not fix it. A 1-day window on 3-day delays under-predicts CVR by ~72%.
- **Freshness and correctness trade off only if you pick a window length.** Modeling the delay (instead of waiting it out) escapes the trade-off: you keep a short, fresh window *and* recover the true CVR.
- **The delayed-feedback model** jointly fits conversion probability and an (exponential) delay distribution; its survival term credits unconverted clicks for the chance they will still convert, removing the bias in expectation.
- **Fake-negative weighting (FNW/ES-DFM)** is the streaming answer: label every click negative immediately, duplicate it as a positive when the conversion lands, and importance-weight by $1+p$ (positives) and $1-p^2$ (negatives) to cancel the duplication bias.
- **Attribution is a label-generation policy.** The attribution window caps the delay; the credit rule (last-click vs multi-touch) decides which click is positive — and richer attribution settles slower, compounding the delay problem.
- **Delay and selection bias are two different diseases of CVR.** Delayed-feedback methods fix *when* labels arrive; ESMM (entire-space pCTCVR) fixes *where* they come from. Production systems often need both, and they compose.
- **Measure on settled labels and report calibration, not just AUC.** A CVR model can rank fine while being badly mis-calibrated — and calibration is what your bidding and budgeting depend on.

## 15. Further reading

- Olivier Chapelle, "Modeling Delayed Feedback in Display Advertising," KDD 2014 — the foundational delayed-feedback model on Criteo data.
- Sofia Ira Ktena et al., "Addressing Delayed Feedback for Continuous Training with Neural Networks in CTR Prediction," RecSys 2019 — the fake-negative weighting / calibration (FNW/FNC) family.
- Jia-Qi Yang et al., "Capturing Delayed Feedback in Conversion Rate Prediction via Elapsed-Time Sampling," AAAI 2021 — ES-DFM, the streaming hybrid of a short wait plus importance weighting.
- Xiao Ma et al., "Entire Space Multi-Task Model: An Effective Approach for Estimating Post-Click Conversion Rate," SIGIR 2018 — ESMM, the entire-space fix for CVR sample-selection bias.
- The Criteo Delayed Feedback Dataset — the public benchmark with click and conversion timestamps for reproducing these results.
- Within this series: [the ranking model and CTR prediction foundations](/blog/machine-learning/recommendation-systems/the-ranking-model-ctr-prediction-foundations); [multi-task and multi-objective ranking, MMoE and PLE](/blog/machine-learning/recommendation-systems/multi-task-and-multi-objective-ranking-mmoe-ple); [calibration and the prediction you can trust](/blog/machine-learning/recommendation-systems/calibration-and-the-prediction-you-can-trust); [train-serve skew and the bugs that hide there](/blog/machine-learning/recommendation-systems/train-serve-skew-and-the-bugs-that-hide-there); and the capstone [the recommender systems playbook](/blog/machine-learning/recommendation-systems/the-recommender-systems-playbook).

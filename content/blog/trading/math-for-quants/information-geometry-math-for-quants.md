---
title: "Information geometry for quants: distance between models on a curved manifold"
date: "2026-06-15"
description: "A beginner-friendly, build-from-zero guide to information geometry — why a family of probability models is a curved surface, how Fisher information measures distance on it, why KL divergence is not symmetric, and how the natural gradient trains models faster."
tags: ["information-geometry", "fisher-information", "kl-divergence", "natural-gradient", "cramer-rao", "manifold", "model-risk", "optimization", "quant-finance", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A family of probability models is not a flat list of settings; it is a *curved surface* where the right notion of distance is "how easily could data tell these two models apart?" That single idea — geometry on the space of models — quietly unifies three things quants use every day: Fisher information, KL divergence, and the natural gradient.
>
> - Each setting of a model's knobs is a **point on a manifold** — a curved surface where every point is one probability distribution; the closer two points, the harder their data is to tell apart.
> - The **Fisher information** is the metric — the local ruler — on that surface: it says how many "distinguishable steps" lie between two nearby models, and it is exactly the quantity behind the **Cramér–Rao bound** on how precise any estimate can be.
> - **KL divergence** behaves like a *squared distance* between two close models, but it is **not symmetric** — $D(P\,\|\,Q) \neq D(Q\,\|\,P)$ — which is why "the distance between two distributions" must always be said with care, and why model risk has a direction.
> - The **natural gradient** is plain gradient descent done on the curved surface instead of the flat parameter grid; it is invariant to how you parameterize the model and converges in far fewer steps on badly-scaled problems.
> - The one number to remember: to reliably tell apart two daily-return models whose volatilities differ by one Fisher "unit" takes on the order of **100 days** of data — halve the gap and you need **four times** as long.

Here is a puzzle that sounds like wordplay but turns out to be the whole subject. Suppose you have two risk models for a stock. Model A says daily returns are normal with volatility 1.0%. Model B says they are normal with volatility 1.1%. A second pair: Model C says volatility 4.0%, Model D says 4.1%. In plain arithmetic, the gap is identical — 0.1 percentage points each time. But ask the question a trader actually cares about — *if I watched a month of returns, could I tell which model was right?* — and the two pairs are nothing alike. Telling 1.0% from 1.1% is a fine, hard distinction; telling 4.0% from 4.1% is hopeless in a month. The "distance" between models, the one that matters for money, is not the ruler-distance between their settings. It is something else.

![Flat parameter view on the left contrasted with a curved model manifold on the right](/imgs/blogs/information-geometry-math-for-quants-1.png)

The diagram above is the mental model for the entire post. On the left is how we usually draw a model's parameters: a flat grid, each knob an axis, distance measured with a plain ruler so that "0.1 here" equals "0.1 there." On the right is the truth that information geometry insists on: the family of models is a *curved surface*, and the natural distance on it is how distinguishable two models are by data. Where models are easy to tell apart, the surface is stretched and small parameter changes cover a lot of ground; where they are hard to tell apart, the surface is compressed and you can wander a long way without the data noticing. This post builds that whole view from zero, names every piece — the **manifold**, the **Fisher metric**, **KL divergence**, the **natural gradient** — and grounds each one in a dollar decision a quant makes. The honest framing up front: information geometry is not a magic trading edge. It is a *unifying lens* that explains why three tools you may already use — Fisher information, the natural gradient, relative entropy — are really the same idea seen from three angles, with a concrete payoff in faster optimization and clearer model comparison.

## Foundations: the building blocks of a model manifold

Before we can talk about distance between models, we need to agree on what a "model" is, what a "family" of them is, and what it would even mean for that family to be curved. We will define each term the first time it appears, build the simplest version, and only then reach for the machinery practitioners use. If you have read the [maximum likelihood post](/blog/trading/math-for-quants/mle-method-of-moments-math-for-quants) some of this will be review — but we will quickly turn it in a direction that post did not go.

### What is a model, and what is a family of models?

A **model** here means a single probability distribution: a precise rule for how likely each possible outcome is. "Daily returns are normal with mean 0 and volatility 1.0%" is one model — it assigns a definite probability to every possible return. Change the volatility to 1.1% and you have a *different* model, a different rule.

A **family** of models is the whole collection you get by turning the knobs. "Normal with mean 0 and any volatility $\sigma > 0$" is a one-knob family. "Normal with any mean $\mu$ and any volatility $\sigma$" is a two-knob family. Each setting of the knobs picks out exactly one model from the family. We bundle the knobs into a single symbol $\theta$ (the Greek letter "theta"); for the two-knob normal, $\theta = (\mu, \sigma)$.

The leap that makes this geometry: stop thinking of $\theta$ as "the settings" and start thinking of each $\theta$ as **a point in a space**. The space of all allowed $\theta$ values is called the **parameter space**. For the two-knob normal it is a two-dimensional space — one axis for $\mu$, one for $\sigma$. Every point in that space *is* a model. Information geometry studies the shape of that space when you measure distance the right way.

### What is a manifold?

A **manifold** is the mathematician's word for a smooth surface that may be curved, but looks flat if you zoom in far enough. The surface of the Earth is the standard example: it is curved (a sphere), yet your backyard looks flat because you are zoomed in. A road map of your city is a flat sheet that approximates a curved planet, and it is accurate locally but wrong globally — straight lines on the map are not the shortest paths over the globe.

The family of models is a manifold in exactly this sense. Locally, around any one model, the space of nearby models looks flat and you can use ordinary calculus. Globally, it is curved, and "straight lines" (in the geometric sense we will define) are not straight lines in the flat parameter picture. The whole game is: measure distance correctly *locally* with a ruler that can change from place to place, then add up those local measurements to get true distances. That changing local ruler is the metric, and the right one is the Fisher information.

### What does "distinguishable" mean, precisely?

We keep saying two models are "easy" or "hard" to tell apart. Let us pin it down. Suppose the truth is model $\theta$, and a rival claims model $\theta'$. You collect $n$ data points and run a test: do they look more like $\theta$ or like $\theta'$? Two models are **easy to distinguish** if even a small sample reliably points to the right one; they are **hard to distinguish** if you need a huge sample to be confident, and even then you are often fooled.

Distinguishability is the *physical* meaning of distance on the model manifold. The reason 1.0% vs 1.1% volatility is a real distinction and 4.0% vs 4.1% is not: a 0.1% change in a small volatility reshapes the bell curve a lot relative to its width, but a 0.1% change in a large volatility barely nudges an already-wide curve. The data feels the first change and shrugs at the second. The Fisher information is the precise measure of how much the data feels a small change.

### What is the log-likelihood, and why do we differentiate it?

The **likelihood** of a parameter $\theta$ given a single observation $x$ is just the density $f(x;\theta)$ — how probable that observation is under that model. The **log-likelihood** is its logarithm, $\ell(\theta) = \log f(x;\theta)$. We work with the log because, for $n$ independent observations, the joint likelihood is a product and the log turns it into a friendly sum, $\ell(\theta) = \sum_i \log f(x_i;\theta)$. (The [MLE post](/blog/trading/math-for-quants/mle-method-of-moments-math-for-quants) builds this carefully; we lean on it here.)

The new question information geometry asks is not "where is the log-likelihood biggest?" (that is MLE) but "how *sensitive* is the log-likelihood to a small change in $\theta$?" Sensitivity is a derivative. The derivative of the log-likelihood with respect to the parameter is so important it has its own name — the **score** — and its typical size is the Fisher information. That is the next section, and it is the foundation under everything else.

#### Worked example: distinguishability you can feel before any math

You run two coins through your head. Coin A lands heads 50% of the time; coin B lands heads 60%. You flip 10 times and get 7 heads. Which coin? Seven of ten leans toward B, but 7 heads out of 10 is not at all rare for a fair coin either — it happens about 12% of the time. So 10 flips barely separate a 50% coin from a 60% coin. Now imagine coin A at 50% and coin B at 95%. Flip 10 times, get 7 heads: that screams coin A, because a 95% coin almost never gives only 7 of 10. The same *arithmetic* gap in one case (10 percentage points: 50 vs 60) versus a bigger gap in the other (45 points: 50 vs 95) produces wildly different distinguishability. If each flip cost you \$1 and you were paid \$100 for a correct identification, you would happily play the 50-vs-95 game and refuse the 50-vs-60 game at 10 flips, because in the first the data pays for itself and in the second it does not. The one-sentence intuition: the useful distance between two models is set by how loudly the data can speak, not by the gap between their dials.

## 1. The score and the Fisher information

We now build the local ruler. Everything in the post rests on one quantity, so we will go slowly.

### The score: the slope of the log-likelihood

Fix an observation $x$ and a model $\theta$. The **score** is the derivative of the log-likelihood with respect to the parameter:

$$ s(\theta) = \frac{\partial}{\partial\theta}\,\log f(x;\theta). $$

Read it as: "if I nudge the parameter up a hair, does this observation become more probable (positive score) or less probable (negative score), and by how much?" If you are sitting at the true parameter and you average the score over all possible data, it comes out to exactly zero — at the truth, nudging $\theta$ in either direction is, on average, neither flattering nor damning to the data. That zero-mean fact is the quiet workhorse behind the whole theory.

The everyday analogy: the score is the *steepness of the ground* under your feet on the log-likelihood landscape, in the direction of the parameter axis. At the very top of the hill (the MLE) the slope is zero. Off the top, the slope points you uphill. The score is that slope, evaluated at wherever you happen to be standing.

### Fisher information: how steep the slope typically is

The score is zero *on average* at the truth, but it is not always zero — for any particular dataset it wobbles around zero. The **Fisher information** is the variance of the score: the typical *size* of that wobble, squared.

$$ I(\theta) = \mathbb{E}\!\left[\,s(\theta)^2\,\right] = \mathbb{E}\!\left[\left(\frac{\partial}{\partial\theta}\log f(X;\theta)\right)^{\!2}\,\right]. $$

Here $\mathbb{E}[\cdot]$ means "the average over all possible data drawn from the model," and $X$ is the random observation. There is an equivalent and often easier form using the second derivative (the curvature of the log-likelihood):

$$ I(\theta) = -\,\mathbb{E}\!\left[\frac{\partial^2}{\partial\theta^2}\log f(X;\theta)\right]. $$

The two forms agree under mild conditions, and the second is a gift: it says Fisher information is the *average curvature* of the log-likelihood at the truth. A sharply peaked log-likelihood (high curvature) means the data strongly prefers one parameter value — high information. A flat-topped log-likelihood (low curvature) means many parameter values explain the data about equally well — low information.

![Pipeline from a family of distributions through the score to the Fisher metric to a distance between models](/imgs/blogs/information-geometry-math-for-quants-2.png)

The pipeline above is the assembly line that turns a model family into a distance, and it is worth memorizing because the rest of the post just elaborates each station. Start with the family of distributions on the left. Differentiate the log-likelihood to get the score — the slope. Square the score and average it to get the Fisher information — the typical steepness. That Fisher information becomes the *metric*, the local ruler. Once you have a ruler that can change from place to place, you can finally measure the distance between two models. Picture the optimizer or the statistician walking along the surface, the Fisher ruler in hand, counting how many "distinguishable steps" separate one model from the next. That step count is the only distance that matters for telling models apart.

### Why information adds up

A crucial, almost magical, property: Fisher information from independent observations *adds*. If one daily return carries information $I(\theta)$ about the volatility, then $n$ independent daily returns carry $n\,I(\theta)$. This is why more data sharpens estimates, and it is the reason the precision of an estimate improves like $1/\sqrt{n}$ rather than staying fixed — the information grows linearly, and as we will see, the squared error of the best estimate is one over the information.

### Fisher information in higher dimensions: the matrix

When the model has several knobs, $\theta = (\theta_1, \ldots, \theta_k)$, the score is a vector (one slope per knob) and the Fisher information becomes a **matrix**, the **Fisher information matrix**:

$$ I(\theta)_{ij} = \mathbb{E}\!\left[\frac{\partial \ell}{\partial\theta_i}\,\frac{\partial \ell}{\partial\theta_j}\right]. $$

The diagonal entries are the information each knob carries on its own; the off-diagonal entries say how *entangled* two knobs are — whether the data confuses one for the other. This matrix is the metric tensor on the manifold: it is the local ruler that tells you the squared distance of a small step $d\theta$ as $d\theta^\top I(\theta)\, d\theta$. We will use the matrix form when we get to the natural gradient. For now, the one-knob picture is enough.

#### Worked example: Fisher information of a Bernoulli hit-rate and the Cramér–Rao floor

This is the example every quant should be able to do cold, because hit rate — the fraction of your trades that win — is the most basic edge statistic there is. Model each trade as a coin flip: it wins with probability $p$ and loses with probability $1-p$. The log-likelihood of a single trade with outcome $x$ (where $x = 1$ for a win, $0$ for a loss) is $\ell(p) = x\log p + (1-x)\log(1-p)$. The score is $s(p) = x/p - (1-x)/(1-p)$. Its variance — the Fisher information per trade — works out to a clean and famous formula:

$$ I(p) = \frac{1}{p(1-p)}. $$

Plug in a realistic edge: $p = 0.55$, a 55% win rate. Then $I(0.55) = 1/(0.55 \times 0.45) = 1/0.2475 \approx 4.04$ per trade. With $n = 400$ trades, the total information is $400 \times 4.04 \approx 1616$.

Now the payoff — the **Cramér–Rao bound**, which says no unbiased estimator of $p$ can have a variance smaller than $1/(n\,I)$. So the smallest possible variance of your estimated hit rate is $1/1616 \approx 0.000619$, and the smallest possible *standard error* is $\sqrt{0.000619} \approx 0.0249$, about **2.5 percentage points**. That is the floor: after 400 trades, even the best conceivable estimator pins your true hit rate only to within roughly $\pm 2.5$ points — so a measured 55% is really "somewhere around 52.5% to 57.5%."

The dollar reading: suppose each trade risks \$1,000 to make \$1,000 (even money). At a true 55% hit rate your edge is $0.55 \times \$1{,}000 - 0.45 \times \$1{,}000 = \$100$ per trade. But the 2.5-point uncertainty in $p$ translates to a \$50 uncertainty in your per-trade edge — your "\$100 edge" could honestly be \$50 or \$150 on the evidence of 400 trades. To halve that uncertainty to \$25 you would need *four times* the trades, 1,600 of them, because information adds linearly and standard error falls like $1/\sqrt{n}$. The one-sentence intuition: Fisher information converts directly into how tightly your money-making edge can be known, and the Cramér–Rao bound is the hard wall no clever estimator can climb over.

## 2. The Fisher metric: distance on the manifold

We have the local ruler. Now we turn it into actual distance, and we meet the curvature that the flat parameter view misses.

### The metric: a ruler that changes from place to place

In flat space, the squared distance of a small step $(d\theta)$ is just $(d\theta)^2$ — the same everywhere. On the model manifold, the squared distance of that same small step is

$$ ds^2 = I(\theta)\,(d\theta)^2 \quad\text{(one knob)}, \qquad ds^2 = d\theta^\top I(\theta)\, d\theta \quad\text{(several knobs)}. $$

The Fisher information $I(\theta)$ is the conversion factor between a parameter step and a *distinguishability* step, and crucially it *depends on where you are* — that is what makes the surface curved. Where information is high, a small $d\theta$ buys a large $ds$ (the surface is stretched: models are easy to tell apart). Where information is low, a large $d\theta$ buys only a small $ds$ (the surface is compressed: models are hard to tell apart).

This is the precise sense in which the model space is a manifold with a metric. The metric is the Fisher information. Distance measured with this metric is exactly distinguishability, accumulated step by step.

### Back to the volatility puzzle, now with numbers

For a normal model with known mean and unknown volatility $\sigma$, the Fisher information for $\sigma$ is $I(\sigma) = 2/\sigma^2$ per observation. Notice it *shrinks* as $\sigma$ grows. The Fisher distance between two close volatilities $\sigma$ and $\sigma + d\sigma$ is $\sqrt{I(\sigma)}\,d\sigma = \sqrt{2}\,d\sigma/\sigma$ — it depends on the *ratio* $d\sigma/\sigma$, not the raw gap $d\sigma$.

Now the 1.0%-vs-1.1% versus 4.0%-vs-4.1% puzzle dissolves cleanly. The first pair: $d\sigma/\sigma = 0.1/1.0 = 0.10$. The second pair: $d\sigma/\sigma = 0.1/4.0 = 0.025$. The first pair is *four times* farther apart on the manifold than the second, even though their raw gaps are identical. The data finds the first pair four times easier to distinguish — exactly the felt experience we started with. The flat ruler called them equal; the Fisher metric called them what they are.

### Geodesics: the straight lines of a curved world

A **geodesic** is the shortest path between two points measured with the metric — the manifold's version of a straight line. On the Earth, the geodesic between two cities is the great-circle route, which looks curved on a flat map. On the model manifold, the geodesic between two models is the most efficient sequence of intermediate models, each as distinguishable from its neighbor by the same amount.

For the volatility manifold, because distance depends on $d\sigma/\sigma$, the geodesic in $\sigma$ is uniform steps in $\log\sigma$, not in $\sigma$. That is the deep reason practitioners habitually think and step in log-volatility, log-price, log-returns: the logarithm is the coordinate in which the model manifold is flat, so equal log-steps are equal distinguishability-steps. The habit was always secretly geometric.

### The exponential family makes the geometry friendly

A huge swath of the distributions quants use — normal, Bernoulli, Poisson, exponential, gamma — belong to the **exponential family**, distributions whose log-density is linear in some transformed parameters. For these, the manifold has especially clean geometry: there are two natural flat coordinate systems (the "natural" parameters and the "expectation" parameters), and KL divergence has a tidy closed form. We will not need the full machinery, but it is worth knowing that the friendliest models are friendly *because* their geometry is well-behaved — and that is why the worked examples in this post all have neat numbers.

#### Worked example: the geometric distance between two volatility models

Take the volatility manifold with one daily observation's worth of information, $I(\sigma) = 2/\sigma^2$. You want the distinguishability distance between a 1.0% model and a 1.5% model. Because the metric depends on $\sigma$, we integrate the local ruler along the path: the Fisher distance is

$$ \int_{1.0}^{1.5} \sqrt{2}\,\frac{d\sigma}{\sigma} = \sqrt{2}\,\log\frac{1.5}{1.0} = \sqrt{2}\times 0.405 \approx 0.573 \text{ (per observation)}. $$

That is the per-observation distance. Over $n$ days the information multiplies by $n$ and the distance multiplies by $\sqrt{n}$. After 25 days the distance is $\sqrt{25}\times 0.573 \approx 2.87$ — and a distance of about 2 to 3 on this scale is roughly the line above which two models become reliably separable by a standard test (we make "reliably" precise in section 5). So with about 25 days of returns you can tell a 1.0% world from a 1.5% world; the dollar consequence is that a desk sizing positions off a 1.0% risk estimate when the truth is 1.5% will discover the error within a month — a month in which it was carrying roughly **50% more risk** than it thought, i.e. for a \$10 million book a true daily value-at-risk of about \$350,000 instead of the \$233,000 it had budgeted. The one-sentence intuition: geometric distance on the volatility manifold counts the days of data it takes to notice you mismeasured your risk.

## 3. KL divergence as a squared distance

We have a real distance — the Fisher distance — but it is expensive to compute (you integrate along a path). In practice quants reach for a cheaper, closely related quantity: the Kullback–Leibler divergence. Understanding its relationship to the Fisher metric, *and its one dangerous quirk*, is the heart of the post.

### What KL divergence is

The **Kullback–Leibler (KL) divergence** from a model $Q$ to a model $P$ measures the average extra surprise you suffer when you believe $Q$ but reality is $P$. Surprise, in information theory, is $-\log(\text{probability})$ — a rare event is very surprising, a certain event not at all. KL divergence is the average of the *difference* in surprise between using the wrong model $Q$ and the right model $P$, where the average is taken over reality $P$:

$$ D(P\,\|\,Q) = \mathbb{E}_{P}\!\left[\log\frac{P(X)}{Q(X)}\right] = \sum_x P(x)\,\log\frac{P(x)}{Q(x)}. $$

(For continuous models replace the sum by an integral and the probabilities by densities.) It is measured in **nats** when you use natural logs, or **bits** when you use log base 2. It is always $\ge 0$, and it is zero only when $P$ and $Q$ are the same model. So it *acts* like a distance from $P$ to $Q$: zero when they coincide, growing as they differ. It is also called **relative entropy**.

### Why it is locally a squared distance — the bridge to Fisher

Here is the link that ties this post together. Take two *close* models on the manifold, $\theta$ and $\theta + d\theta$. Expand the KL divergence between them in a Taylor series. The constant term is zero (a model versus itself), the linear term is zero (because the score has mean zero — that quiet fact again), and the first surviving term is quadratic:

$$ D(\theta\,\|\,\theta + d\theta) \approx \tfrac{1}{2}\,d\theta^\top I(\theta)\, d\theta = \tfrac{1}{2}\,ds^2. $$

So for nearby models, KL divergence is *one half the squared Fisher distance*. The metric that defines distinguishability and the divergence that measures relative entropy are the same object, twice removed by a square. This is why KL divergence "feels like distance squared," why $\sqrt{2\,D}$ is approximately the Fisher distance for close models, and why information geometry can treat all three — Fisher information, the Fisher metric, KL divergence — as one idea. The Fisher information is literally the second-derivative (the curvature) of the KL divergence at its zero.

![Stack of natural gradient over KL divergence over Fisher metric over the curved manifold](/imgs/blogs/information-geometry-math-for-quants-3.png)

The stack above is the unification in one image, and it is the figure to keep in your head for the rest of the post. At the bottom sits the curved manifold of distributions — the surface itself. On it, the Fisher metric sets the local ruler: how far apart nearby models are. One layer up, KL divergence is that ruler's squared reading — the relative entropy between models, which near any point is just half the squared Fisher distance. At the top, the natural gradient (next section) uses that same ruler to decide which way is truly downhill. The reason these three tools keep showing up together in papers and libraries is not coincidence; it is that each is a different operation on the same geometry. Once you see the stack, Fisher information, KL divergence, and the natural gradient stop being three separate formulas to memorize and become three uses of one surface.

### The dangerous quirk: KL is not symmetric

Now the warning every practitioner must internalize. Ordinary distance is symmetric: the distance from your house to the store equals the distance back. KL divergence is **not**:

$$ D(P\,\|\,Q) \neq D(Q\,\|\,P) \quad\text{in general}. $$

This is not a technicality; it is a feature with teeth. $D(P\,\|\,Q)$ asks "how bad is it to use $Q$ when reality is $P$?" while $D(Q\,\|\,P)$ asks "how bad is it to use $P$ when reality is $Q$?" Those are genuinely different questions with different answers. The asymmetry has a precise behavioral signature:

- **Forward KL** $D(P\,\|\,Q)$, with the truth $P$ in front, punishes $Q$ ferociously for putting *low* probability where $P$ puts *high* probability. It forces $Q$ to *cover* all of $P$'s mass. A $Q$ fit by forward KL spreads out to avoid being caught flat-footed by any outcome the truth allows — it is "mass-covering."
- **Reverse KL** $D(Q\,\|\,P)$, with the approximation $Q$ in front, punishes $Q$ for putting *high* probability where $P$ puts *low* probability. It lets $Q$ ignore parts of $P$ as long as it does not invent probability where there is none — it is "mode-seeking," and tends to be too confident, too narrow.

For a quant the difference is the difference between a risk model that is paranoid about tails it has barely seen and a risk model that is dangerously confident the tails are not there. Which divergence you minimize when you fit or compare models *encodes a choice about which kind of error you fear more*. Get the direction wrong and you have optimized for the wrong fear.

![Matrix contrasting forward KL D of P given Q against reverse KL D of Q given P](/imgs/blogs/information-geometry-math-for-quants-4.png)

The matrix above makes the asymmetry concrete with the numbers from the worked example below. The forward divergence, reading the truth into the approximation, comes out to 0.18 nats; the reverse divergence, reading the approximation into the truth, comes out to 0.36 nats — twice as large. They disagree, and the gap of 0.18 nats is not a rounding error, it is the whole point: there is no single "distance between $P$ and $Q$," only a distance *with a direction*. When a risk report or a model-comparison dashboard quotes "the KL divergence" without saying which way it runs, treat that number with suspicion until you know whether the truth or the approximation is in front.

### Why not just symmetrize it?

You can — the **Jensen–Shannon divergence** averages the two directions against their midpoint, and is symmetric. So does the symmetrized $\tfrac{1}{2}(D(P\|Q) + D(Q\|P))$. But symmetrizing throws away the very information — *which* error you fear — that made the choice meaningful, and it loses the clean Taylor link to the Fisher metric. The Fisher distance itself *is* a genuine symmetric distance (a metric in the strict sense), and for close models it is the honest symmetric answer. KL's asymmetry is the price of its cheapness and its directional meaning, not a defect to be papered over.

#### Worked example: KL between two normals, and why direction matters for model risk

Let $P$ be the "true" daily-return model: normal, mean 0, volatility $\sigma_P = 1.0\%$. Let $Q$ be your fitted model: normal, mean 0, volatility $\sigma_Q = 1.4\%$ — you over-estimated the risk by 40%. The KL divergence between two zero-mean normals has a clean closed form:

$$ D(P\,\|\,Q) = \log\frac{\sigma_Q}{\sigma_P} + \frac{\sigma_P^2}{2\sigma_Q^2} - \frac{1}{2}. $$

Plug in the forward direction (truth $P$, approximation $Q$): $\log(1.4/1.0) + \tfrac{1}{2}(1.0/1.4)^2 - \tfrac{1}{2} = 0.336 + 0.255 - 0.5 = 0.091$ nats. Now the reverse (swap the roles of $P$ and $Q$, so the approximation is in front): $\log(1.0/1.4) + \tfrac{1}{2}(1.4/1.0)^2 - \tfrac{1}{2} = -0.336 + 0.98 - 0.5 = 0.144$ nats. The two directions give 0.091 and 0.144 — different numbers from the *same two models*, confirming the asymmetry. (For a slightly bigger vol gap, say $\sigma_Q = 1.5\%$, the gap widens to roughly 0.18 versus 0.36, the numbers shown in the figure.)

The dollar reading is where it bites. Imagine a regulator's model-risk charge that scales with the relative entropy between your reported risk model and the realized one — a stylized but real flavor of how model-risk capital works. If you over-state volatility (your $Q$ wider than truth $P$), the forward divergence $D(P\|Q) = 0.091$ is comparatively mild: being too cautious is a smaller information crime than being too confident. If instead you *under*-state volatility — your model $Q$ at 1.0% when the truth $P$ is 1.4% — you would compute $D$ with the narrow model as $Q$ and the wide truth as $P$, and the penalty is the *larger* reverse-style number, because a model that is too confident gets caught flat-footed by the fat real tails. On a \$100 million book where the model-risk add-on is, say, \$1 million per nat of divergence, the direction of your error is the difference between a roughly \$91,000 charge for being too cautious and a roughly \$144,000-and-up charge for being too confident. The one-sentence intuition: KL divergence is the natural currency of model risk, but you must always state the direction, because being wrongly confident costs strictly more than being wrongly cautious.

## 4. The natural gradient: steepest descent on the manifold

So far the geometry has been a way of *measuring*. Now it earns its keep as a way of *optimizing*. This is the most practically useful payoff of information geometry, and it connects straight to the [stochastic gradient and optimizers post](/blog/trading/math-for-quants/stochastic-gradient-optimizers-math-for-quants).

### The problem with the plain gradient

When you train a model or fit a signal, you minimize a loss by repeatedly stepping in the direction of steepest descent — the negative gradient. But "steepest" is a *geometric* notion: it means the direction that drops the loss most per unit of distance moved. The plain gradient measures distance with the flat ruler — it assumes a step of size 0.1 in one parameter is the same size as 0.1 in another. On the model manifold that is false: a 0.1 step in a high-information parameter is a *huge* move (very distinguishable), while a 0.1 step in a low-information parameter is a *tiny* move.

The consequence is the classic pathology of a "badly-scaled" or "ill-conditioned" objective: the loss surface is a long, narrow valley, and plain gradient descent zig-zags across the valley walls making painfully slow progress down its length. It is taking equal-sized flat steps in a space where the right step size differs wildly by direction.

![Plain gradient path zig-zagging versus natural gradient path going straight to the answer](/imgs/blogs/information-geometry-math-for-quants-6.png)

The before-and-after above shows the move in pictures. On the left, the plain gradient ignores the scaling of the problem: it zig-zags across the narrow valley, takes only tiny steps down the valley's length, and burns many iterations to converge. On the right, the natural gradient rescales each step by the local Fisher metric, so it heads almost straight downhill, behaves the same no matter how you happened to parameterize the model, and converges in a handful of steps. Watch the optimizer on the left bounce wall to wall while the one on the right walks the valley floor — that is the entire value proposition, and on a badly-conditioned calibration it is the difference between a fit that finishes overnight and one that does not.

### The fix: precondition by the Fisher information

The **natural gradient** corrects for the curvature by measuring "steepest" with the *Fisher* ruler instead of the flat one. Mechanically, you take the plain gradient $g = \nabla_\theta \mathcal{L}$ and multiply it by the inverse Fisher information matrix:

$$ \tilde{g} = I(\theta)^{-1}\, g, \qquad \theta \leftarrow \theta - \eta\,\tilde{g}. $$

Here $\mathcal{L}$ is the loss, $g$ is its ordinary gradient, $I(\theta)^{-1}$ is the inverse Fisher matrix, $\eta$ is the learning rate, and $\tilde g$ is the natural gradient. The inverse Fisher matrix stretches the step out in low-information directions (where progress is safe and cheap) and shrinks it in high-information directions (where a small move already changes the model a lot). The result heads down the *true* steepest direction on the manifold.

### Two properties that make it special

**Reparameterization invariance.** This is the deep one. If you re-express your model in different coordinates — fit in log-volatility instead of volatility, or rescale a feature — the plain gradient *changes direction*, so plain gradient descent gives different answers depending on arbitrary modeling choices. The natural gradient does not: it follows the same path on the manifold no matter how you label the points, because the Fisher metric transforms in exactly the way that cancels the relabeling. Your optimization stops depending on bookkeeping you should not care about.

**Connection to Newton's method.** Readers of the [optimizers post](/blog/trading/math-for-quants/stochastic-gradient-optimizers-math-for-quants) will recognize the shape: Newton's method preconditions the gradient by the inverse *Hessian* (the curvature of the loss). The natural gradient preconditions by the inverse *Fisher* (the curvature of the KL divergence / the log-likelihood). When the loss *is* the negative log-likelihood, the two curvatures coincide near the optimum, so the natural gradient is essentially Newton's method that stays well-behaved (the Fisher matrix is always positive semdefinite, so it never points you uphill the way a raw Hessian can at a saddle). It is curvature-aware optimization with a safety guarantee.

### The honest catch: the Fisher matrix is expensive

Inverting a Fisher matrix every step costs $O(k^3)$ for $k$ parameters — fine for a handful of parameters, ruinous for a model with millions. This is why, in large-scale machine learning, practitioners use *approximations* of the natural gradient (diagonal approximations, K-FAC, and so on) rather than the exact thing. Adam, the workhorse optimizer, is in part a cheap diagonal stand-in for the natural gradient: its per-parameter scaling by the running average of squared gradients is a rough estimate of the Fisher diagonal. So even quants who never compute a Fisher matrix are using its ghost every time they call Adam. The exact natural gradient is the gold standard you approximate toward, not always the thing you run.

#### Worked example: natural gradient versus plain gradient on a badly-scaled fit

You are calibrating a tiny two-parameter model. The loss is a quadratic bowl, but a badly-scaled one: it is 100 times steeper in the $\theta_1$ direction than the $\theta_2$ direction — a long narrow valley, **condition number** 100 (the ratio of steepest to shallowest curvature). You start 1 unit from the minimum in each direction.

*Plain gradient descent.* To stay stable you must set the learning rate by the *steepest* direction, so $\eta \approx 1/100 = 0.01$. In the shallow $\theta_2$ direction each step then closes only about 1% of the remaining gap. To get within 1% of the minimum in $\theta_2$ you need roughly $\log(0.01)/\log(0.99) \approx 458$ steps. The steep direction converges fast but the shallow one drags the whole fit out to **hundreds of iterations**.

*Natural gradient.* Preconditioning by the inverse Fisher rescales both directions to the same effective curvature — the condition number becomes 1, a round bowl. Now a single well-chosen step of $\eta \approx 1$ lands essentially on the minimum: **one to a few steps** to the same answer. The speed-up is roughly the condition number, here about 100×.

The dollar reading: suppose this calibration must run before the open every morning and each iteration costs 2 seconds of compute. Plain gradient at 458 steps is about 15 minutes; the natural gradient at, say, 3 steps is about 6 seconds. If the model is recalibrated across 500 instruments, plain gradient is over 5 days of serial compute (you would parallelize, but the cloud bill is real) versus under an hour for the natural gradient — and both land on the *same* calibrated parameters, so the same \$-accurate risk numbers, just one of them in time to trade on. The one-sentence intuition: the natural gradient does not find a better answer, it finds the same answer far sooner by stepping in the geometry where the problem is round instead of the geometry where it is a knife-edge valley.

## 5. Distinguishability: how many samples to tell models apart

We promised that Fisher distance counts the data needed to separate two models. Now we cash that promise with a usable rule, because "how much data do I need to trust this?" is the question behind every backtest, every A/B test of a signal, and every claim that one model beats another.

### From Fisher distance to a sample-size rule

Two models at Fisher distance $d$ per observation become distinguishable, with a standard test at conventional confidence, once the *accumulated* distance crosses a threshold of order a few units. Because information adds, the accumulated distance after $n$ observations is $\sqrt{n}\,d$. Setting $\sqrt{n}\,d$ equal to a threshold $c$ (think $c \approx 2.8$ for a roughly 80%-power, 5%-significance test) gives the rule:

$$ n \approx \frac{c^2}{d^2}. $$

Sample size grows as the *inverse square* of the per-observation distance. Halve the distance between two models and you need *four times* the data to tell them apart. This single relationship is the quantitative spine of distinguishability, and it explains an enormous amount of frustration in quant research.

![Matrix of how many days are needed to tell apart return models at decreasing Fisher distance](/imgs/blogs/information-geometry-math-for-quants-7.png)

The matrix above turns the rule into a table you can carry around. Two models that sit two Fisher units apart need only about 25 days of data to separate; one unit apart, about 100 days; half a unit, about 400 days; a quarter unit, about 1,600 days — roughly six and a half years of trading. Each halving of the gap quadruples the data. The pattern is brutal and it is why detecting *small* edges and *small* differences between models is so much harder than intuition suggests: the cost of resolution does not grow linearly with how fine a distinction you want, it grows with the *square*. A signal twice as subtle is four times as expensive to confirm.

### Why this is the same fact as the Cramér–Rao bound

This is not a new principle — it is the Cramér–Rao bound wearing different clothes. Cramér–Rao said the variance of the best estimate is $1/(n I)$, so its standard error is $1/\sqrt{n I}$. Two models are distinguishable when their parameter gap exceeds a couple of standard errors, i.e. when $d\theta \gtrsim c/\sqrt{nI}$, which rearranges to $n \gtrsim c^2/(I\,d\theta^2) = c^2/d^2$ since $d = \sqrt{I}\,d\theta$ is the Fisher distance. The bound on how *precisely* you can estimate and the rule for how much data you need to *distinguish* are the same statement read two ways. Geometry just gave us a single picture — distance on a manifold — that contains both.

#### Worked example: how many days to tell two close return models apart

You suspect a stock's true daily volatility is 1.0%, but a colleague's model says 0.9%. Can a quarter's worth of data — about 63 trading days — settle it? Compute the per-observation Fisher distance on the volatility manifold. Using $d = \sqrt{2}\,\log(\sigma_2/\sigma_1) = \sqrt{2}\,\log(1.0/0.9) = \sqrt{2}\times 0.105 = 0.149$ per day. The required sample size for a standard test is $n \approx c^2/d^2 = 2.8^2/0.149^2 = 7.84/0.0222 \approx 353$ days — about **17 months**, not a quarter. With only 63 days the accumulated distance is $\sqrt{63}\times 0.149 = 1.18$, well short of the $\approx 2.8$ you need; the data simply cannot yet tell 1.0% from 0.9% with confidence.

The dollar reading: if you size a \$5 million position assuming 1.0% volatility, your one-day 95%-ish risk is about $1.65\times 1.0\% \times \$5\text{M} = \$82{,}500$. If the truth is 0.9% you are slightly *over*-hedged — modest cost. But flip the colleague's number to 1.3% and rerun: $d = \sqrt{2}\log(1.3/1.0) = 0.371$, so $n \approx 7.84/0.138 \approx 57$ days — under three months. The bigger the real gap, the faster the data resolves it, exactly as the geometry promises. The practical lesson is to spend your data budget on the distinctions that are far apart on the manifold and to be honest that the close ones may be permanently beyond reach within any sane sample. The one-sentence intuition: before you argue about whether 1.0% or 0.9% is the right risk, the geometry tells you that you do not yet have the data to win the argument — and never will in a single quarter.

## The whole picture in one map

![Tree of information geometry concepts rooted at models forming a curved surface](/imgs/blogs/information-geometry-math-for-quants-5.png)

The tree above is the map of everything we built, and it is worth a slow look before the misconceptions. At the root is the single idea the whole subject grows from: a family of models is a curved surface. From that root, two main branches. The first is the **Fisher metric** — the local ruler — and it itself splits into the **Cramér–Rao floor** (the hard limit on estimation precision) and the **natural gradient** (curvature-aware optimization). The second branch is **KL divergence** — relative entropy — which splits into the **model-risk budget** (relative entropy as the currency of how wrong a model is) and the **sample-size-to-distinguish** rule (how much data separates two models). Every tool in this post hangs off that one root. If you remember nothing else, remember the shape of this tree: it is why Fisher information, the natural gradient, KL divergence, and the Cramér–Rao bound are not four things to memorize but one geometric idea seen from four sides.

## Common misconceptions

**"KL divergence is the distance between two distributions."** It is not a distance in the strict sense, because it is not symmetric and does not obey the triangle inequality. $D(P\|Q) \neq D(Q\|P)$, sometimes by a factor of two or more, as our worked example showed. It *behaves* like squared distance for nearby models — half the squared Fisher distance — but the moment models are far apart or you swap the arguments, treating it as a symmetric metric will mislead you. If you need a true distance on the model manifold, the Fisher (geodesic) distance is the symmetric one; KL is the cheap, directional cousin.

**"Information geometry is a trading strategy."** No. It is a *lens*, not an alpha. It does not predict prices and it will not, by itself, make money. What it does is explain why three tools you already use are the same tool, give you the natural gradient for faster fits, and give you honest sample-size and precision bounds. The payoff is in the back office — better optimization, clearer model comparison, sober expectations about what your data can prove — not in a signal you can trade tomorrow. Posts promising that "the geometry of returns" reveals hidden edges are selling the mystique, not the math.

**"More parameters always let me fit the data better, so a richer model is closer to the truth."** Richer models live on a higher-dimensional manifold, and the Fisher geometry warns you why that backfires: more directions means more low-information directions where your estimate wanders far with little data to pin it down. The Cramér–Rao floor on each extra parameter is real, and adding parameters you cannot estimate within a useful standard error adds variance, not accuracy. The geometry quantifies the bias-variance tradeoff that the [estimators post](/blog/trading/quantitative-finance/estimators-mle-bias-variance-quant-interviews) develops from the statistical side.

**"The natural gradient finds a better optimum than plain gradient descent."** It finds the *same* optimum, usually much faster, and independent of how you parameterized the model. The loss surface and its minimum do not change; only the *path* the optimizer takes to reach it changes. If your plain-gradient fit and your natural-gradient fit land on materially different parameters, one of them has not converged — they should agree at the bottom.

**"Fisher information is about the data I collected."** Subtle but important: Fisher information is an *expectation over data the model could produce*, evaluated at a parameter — it is a property of the *model*, not of your particular sample. The thing computed from your actual data (the second derivative of the realized log-likelihood at the MLE) is the **observed** information, a sample estimate of the Fisher information. They are close for large samples but they are not the same object; conflating them is a common source of slightly-wrong confidence intervals.

**"Symmetrizing KL fixes its problems."** Symmetrizing (Jensen–Shannon, or averaging both directions) buys symmetry at the price of the very thing that made KL meaningful — the *direction* of the error you fear. For model risk, the asymmetry is information, not noise: being wrongly confident (narrow model, fat reality) genuinely costs more than being wrongly cautious, and a symmetric measure erases that asymmetry along with the inconvenience.

## How it shows up in real markets

### 1. Volatility calibration and the log-vol habit

Every options desk calibrates volatility, and almost every one of them works in *log*-volatility — fitting, interpolating, and stepping in $\log\sigma$ rather than $\sigma$. The usual justification is "it keeps volatility positive," which is true but shallow. The deeper reason is geometric: the Fisher metric on the volatility manifold depends on $d\sigma/\sigma$, so the log is the coordinate in which the manifold is flat and equal steps are equal distinguishability. The 2020 volatility spike, when realized vol on major indices jumped from roughly 12% to over 80% in weeks, was a move of about $\log(80/12) \approx 1.9$ in log-vol — a large but *finite* geometric distance, which is why models stated in log-vol absorbed it as a big-but-bounded shock rather than the off-the-charts event a linear-vol model would register. Desks that thought in the right geometry were less surprised.

### 2. Model-risk capital and relative entropy

Regulators and internal model-risk teams increasingly express "how wrong could our model be?" in the language of relative entropy. Robust-control and "model uncertainty" frameworks — the kind that became standard after the 2008 crisis exposed how fragile single-model risk numbers are — define a *worst-case* alternative model within a KL ball of radius $\eta$ around your baseline, and charge capital for the worst loss in that ball. The asymmetry of KL is doing real work here: the worst-case model the framework finds is one that is *fatter-tailed* than your baseline, because confident models are the ones KL punishes hardest. A bank that set its KL radius from calm-period data and never revisited it would have under-reserved going into a turbulent period — the ball was too small to contain the model the world actually presented.

### 3. The natural gradient inside the optimizers you already run

Adam, RMSProp, and the adaptive optimizers that train essentially every modern machine-learning alpha signal are, in part, cheap diagonal approximations to the natural gradient. Their per-parameter rescaling by the running average of squared gradients estimates the diagonal of the Fisher matrix. When a research team switched a slow-training factor model from plain SGD to Adam and saw it converge in a fraction of the epochs, they were not getting a better model — they were getting the natural gradient's reparameterization-friendly geometry on the cheap. Trust-region policy optimization, used in reinforcement-learning-based execution agents, uses a KL constraint explicitly — it caps how far the policy can move *in KL* each step, which is exactly a Fisher-distance trust region, keeping the agent from taking a step that looks small in parameters but is huge in behavior.

### 4. Backtest sample size and the limits of resolution

The $n \approx c^2/d^2$ rule is the quiet executioner of many a promising backtest. A team finds a signal whose Sharpe ratio is, say, 0.3 higher than the benchmark over five years and declares victory; the Fisher-distance arithmetic says that the difference between a true Sharpe of 1.0 and 1.3 over daily data needs far more than five years to confirm at conventional confidence, so the "win" is well inside the noise. This is the same arithmetic behind the famous result that distinguishing a genuinely skilled manager from a lucky one can take decades of returns. The geometry does not tell you the signal is bad; it tells you, honestly, that your data cannot yet say — and that humility has saved more capital than most signals have made.

### 5. The 1.0%-vs-1.1% trap in risk reporting

A risk team reports daily VaR off an estimated volatility and updates it as new data arrives. When realized vol drifts from 1.0% to 1.1%, the model may not flag the change for weeks, because — as the distinguishability rule shows — that small a gap takes hundreds of days to confirm. Meanwhile a drift from 1.0% to 1.5% gets caught in about a month. The asymmetry in *detection speed* across the volatility manifold means risk systems are systematically faster to notice large regime shifts than slow creep — which is fine for crashes but dangerous for the slow grind-up of leverage in a quiet market, the exact pattern that preceded several blow-ups. The geometry tells you which mismeasurements your monitoring will be slow to catch.

### 6. Building and comparing alpha signals

When a research team must choose between two candidate signals, they are comparing two models, and the right question — *how much data do I need to be confident A beats B?* — is a distinguishability question answered by the Fisher distance between them. Teams that frame model comparison geometrically avoid two classic errors: declaring a winner on a difference smaller than the data can resolve, and abandoning a signal whose edge over the benchmark is real but too subtle to confirm in the window they tested. The discipline of asking "how far apart are these models on the manifold, and do I have $c^2/d^2$ observations?" is exactly the rigor the [alpha-signal research workflow](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) tries to instill, now with a number attached.

### 7. The LTCM lesson, told geometrically

Long-Term Capital Management's 1998 collapse is usually told as a story about leverage and fat tails. There is a geometric retelling. LTCM's models lived at a point on the manifold calibrated to a *calm-market* world; the 1998 Russian-default crisis was a different point, far away in Fisher distance, where correlations and volatilities were transformed. Crucially, the distance between "calm regime" and "crisis regime" is large, but the *data needed to detect that you have moved* arrives only as the move is happening — distinguishability is symmetric in distance but asymmetric in time, because you accumulate the separating evidence only forward. By the time enough crisis data had accumulated to confirm the regime had changed, the position was already underwater. The geometry does not excuse the leverage, but it sharpens the lesson: a model calibrated to one region of the manifold has *no information* about how far it is from the region it has never visited, and the Fisher metric only measures distance you have paid for in data.

## When this matters to you and further reading

If you fit models to data — and every quant does — information geometry is the lens that turns three scattered tools into one coherent picture, and that coherence pays off in concrete ways. When you next set a learning rate and watch an optimizer crawl down a narrow valley, you will recognize the flat-ruler pathology and know that a Fisher-preconditioned step, or its cheap stand-in Adam, is the cure. When a dashboard quotes "the KL divergence" between two models, you will ask which direction it runs before you trust it, because you know being wrongly confident costs more than being wrongly cautious. When a backtest claims one model beats another, you will reach for $n \approx c^2/d^2$ and ask whether the data could possibly support the claim. And when someone sizes risk off a volatility estimate, you will know — to the day — how long it would take to notice if they were wrong, and that small errors near a large volatility may never be caught.

The honest bottom line, stated plainly: information geometry will not hand you an edge. What it hands you is *clarity about the limits* — how precisely a thing can be known (Cramér–Rao), how much data it takes to know it (distinguishability), which way the error of a model cuts (KL asymmetry), and how to descend a loss surface in the geometry that makes it round instead of jagged (the natural gradient). In a field where most failures come from over-confidence in a number the data could not support, a tool whose main gift is calibrated humility is worth more than it looks.

To go deeper from here, the natural next steps are the posts this one leans on. The [maximum likelihood and method of moments post](/blog/trading/math-for-quants/mle-method-of-moments-math-for-quants) develops the score, the likelihood, and the Cramér–Rao bound from the estimation side — read it as the statistical foundation under this post's geometry. The [stochastic gradient and optimizers post](/blog/trading/math-for-quants/stochastic-gradient-optimizers-math-for-quants) develops gradient descent, momentum, and Adam — read it to see where the natural gradient fits among the methods you actually run, and why Adam is its frugal cousin. For the bias-variance tradeoff that the high-dimensional geometry quietly enforces, the [estimators, bias, and variance post](/blog/trading/quantitative-finance/estimators-mle-bias-variance-quant-interviews) is the companion piece. And to put model comparison to work on real signals, the [alpha-signal research post](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) is where the distinguishability arithmetic becomes a daily research habit. This is educational material, not investment advice — but if it leaves you a little more skeptical of confident numbers and a little more precise about how much your data can prove, it has done its job.

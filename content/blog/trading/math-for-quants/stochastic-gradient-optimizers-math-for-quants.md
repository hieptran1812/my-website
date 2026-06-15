---
title: "Stochastic gradient and numerical optimizers: how a computer finds the best portfolio and the best model"
date: "2026-06-15"
description: "A from-scratch tour of gradient descent, SGD, momentum, Adam, and quasi-Newton methods, and how each one trains trading signals, scales to huge market datasets, and calibrates pricing models."
tags: ["optimization", "gradient-descent", "stochastic-gradient-descent", "adam", "newton-method", "l-bfgs", "model-calibration", "machine-learning", "quantitative-finance"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Every trading model, risk model, and pricing model is found by the same idea: start with a guess, measure which way is downhill, take a step, and repeat until you cannot go lower. That repeated step is *numerical optimization*, and the family of methods that do it is the engine room beneath all quantitative finance.
>
> - **Gradient descent** walks downhill on a cost surface; the **learning rate** is the size of each step. On a bowl-shaped (convex) cost you are guaranteed to reach the bottom; on a bumpy (non-convex) cost you might get stuck.
> - **Stochastic gradient descent (SGD)** estimates the downhill direction from a tiny random sample of your data instead of all of it. That one trick is what lets a model train on hundreds of millions of market observations and adapt to new data as it arrives.
> - **Momentum, RMSProp, and Adam** are smarter versions of the step that go faster and survive messy, noisy gradients — they are what actually train machine-learning alpha models in practice.
> - **Newton and quasi-Newton methods (BFGS, L-BFGS)** use curvature to leap to the answer in a handful of steps; they are the right tool for calibrating small smooth models like a volatility surface or a GARCH model.
> - The one number to remember: a learning rate that is **10× too big can turn a correct \$1.2 million risk estimate into a meaningless \$4.1 billion one**, while a rate 100× too small can leave you still wrong after a week of compute.

Here is a number that should bother you. A modern systematic fund might fit a model on **300 million** rows of market data — every stock, every minute, every feature, for years. If the computer had to read all 300 million rows just to take *one* step toward a better model, and a good model needs thousands of steps, you would be waiting for the heat death of the universe. And yet these models get trained overnight, on a single machine, before the market opens. How?

The answer is a small, beautiful trick: **you do not need all the data to know which way is downhill.** A random handful tells you the direction well enough to take a step, and if you take enough small noisy steps you arrive at almost exactly the same place as the slow exact method — but thousands of times faster. That trick is *stochastic gradient descent*, and it sits at the heart of a whole family of methods called numerical optimizers. This post builds that family from zero, from the simplest downhill walk to the clever momentum-and-curvature methods that train real trading signals and calibrate real pricing models. Every method gets tied to a dollar problem a desk actually solves.

![Pipeline of the optimizer loop from start guess to gradient to step to converged result](/imgs/blogs/stochastic-gradient-optimizers-math-for-quants-1.png)

## The optimizer loop in one picture

The diagram above is the mental model for the entire post, so let us walk it once, slowly. On the far left is a **start guess** — some initial values for the thing you are trying to choose. That "thing" might be the weights of a portfolio, the coefficients of a regression, or the millions of parameters inside a neural network. We bundle all those choices into one list of numbers and call it $w$. The starting guess can be almost anything; often it is just zeros or small random numbers.

Then the loop. First we **measure the slope** of our cost — the *gradient* — which is a compass that points in the direction the cost *increases* fastest. Because we want the cost *smaller*, we **step the opposite way**, downhill, by an amount controlled by a knob called the learning rate. That gives us new, slightly better values of $w$. Then we **check**: is the slope basically flat now? If yes, there is no downhill left and we stop. If no, we loop back and do it again. When the loop finally halts, the $w$ it hands us is the **converged answer** — the best portfolio, the fitted model, the calibrated parameters.

Everything in this article is a variation on that one loop. Plain gradient descent measures the slope using *all* your data. Stochastic gradient descent measures it using a *sample*. Momentum and Adam change *how* the step is taken so it goes faster and survives noise. Newton's method adds a second measurement — the *curvature* — so it can take a much bigger, smarter step. The loop never changes; only the cleverness of each step does. By the end, this picture will read like a sentence.

A quick reassurance before the math, because the symbols scare people off: nothing here requires you to invent calculus. You will need exactly one idea from calculus — the *slope* of a function, also called its derivative or gradient — and we will define it from scratch in plain words. If you have ever stood on a hillside and known that water runs downhill, you already have the core idea. The rest is bookkeeping.

## Foundations: the building blocks

Before any optimizer, we need to agree on four things: what a *cost function* is, what a *gradient* is, what *convex* and *non-convex* mean, and what a *learning rate* does. If you have never met these, this section is for you. A practitioner can skim; a beginner should not skip.

### What a cost function is, in money terms

A **cost function** (also called a *loss function* or *objective*) is a single number that measures how *bad* your current choice is. Lower is better. The whole game of optimization is to make this number as small as possible.

The crucial thing is that the cost depends on the choices you control. Change the choices, and the number changes. A few examples a quant cares about:

- **Portfolio risk.** If $w$ is how much you hold of each asset and $\Sigma$ is the covariance matrix (a grid measuring how the assets move together), then portfolio variance $w^\top \Sigma w$ is a single number measuring how jumpy your portfolio is. A risk-averse investor wants this small. We built this object from scratch in [the covariance matrix as linear algebra](/blog/trading/math-for-quants/covariance-matrix-linear-algebra-math-for-quants).
- **Model error.** If you are fitting a model that predicts tomorrow's return from today's features, your cost might be the *mean squared error* — the average squared gap between what the model predicted and what actually happened. Small cost means the model fits the data.
- **Calibration error.** If you are fitting a pricing model to match the option prices quoted in the market, your cost is the total squared difference between your model's prices and the market's prices. Drive it to zero and your model "matches the market".

In every case the cost is one number, it depends on parameters $w$ you get to choose, and you want it small. The shape you get when you plot the cost against the choices is called the **cost surface** or **loss landscape** — a hilly terrain where the height at each point is the cost, and your job is to find the lowest valley.

### What a gradient is, in plain words

Stand on a hillside in fog. You cannot see the valley, but you can feel the ground under your feet and tell which way is steepest *uphill*. The direction of steepest uphill, together with how steep it is, is the **gradient**. In one dimension the gradient is just the ordinary slope you met in school: how much the cost rises if you nudge $w$ a little to the right.

When $w$ is a list of many numbers — say a thousand model parameters — the gradient is also a list of the same length. Each entry answers "if I nudge *this one* parameter a tiny bit, how much does the cost change?" We write the gradient as $\nabla f(w)$ (the upside-down triangle $\nabla$ is read "nabla" or "grad"). The key facts, and the only two you must remember:

1. The gradient **points uphill** — toward larger cost. So to go *down*hill we move in the *opposite* direction, $-\nabla f(w)$.
2. Where the cost is at its lowest, the ground is flat in every direction, so the gradient is **zero**. "Find where the gradient is zero" is the same as "find the bottom".

That is the whole engine. Gradient descent literally is: *repeatedly take a step in the direction $-\nabla f(w)$.* For the matrix-calculus rules that let you compute these gradients for portfolios and regressions by hand — for instance that the gradient of portfolio variance $w^\top\Sigma w$ is exactly $2\Sigma w$ — see the companion piece on [matrix calculus for optimization](/blog/trading/math-for-quants/matrix-calculus-optimization-math-for-quants). Here we take the gradient as given and focus on what to *do* with it.

### Convex versus non-convex: bowls and badlands

The shape of the cost surface decides whether the downhill walk is safe or risky.

A **convex** cost surface is shaped like a single smooth bowl. Wherever you start, downhill always leads to the *one* lowest point — there are no false valleys to get trapped in. Portfolio variance is convex (it is a bowl in disguise), and so is ordinary least-squares regression. Convexity is the mathematician's gift: it *guarantees* that if you keep going downhill you will reach the global best, and any flat spot you find is *the* answer, not a trap.

A **non-convex** cost surface is badlands: hills, ridges, and many separate valleys of different depths. Downhill from where you stand might lead into a shallow valley — a **local minimum** — that is worse than the deepest valley somewhere else. There are also **saddle points**, places that are flat (gradient zero) but are a low point in one direction and a high point in another, like the seat of a horse saddle. Neural-network trading models live in this badlands. Most of the practical cleverness in modern optimizers — momentum, Adam, random sampling — exists precisely to survive non-convex terrain.

### The learning rate: how big a step

The **learning rate** (often written $\eta$, the Greek letter "eta", or just $\alpha$) is the single most important knob in all of optimization. It is the size of each downhill step. The update rule for plain gradient descent is one line:

$$ w_{t+1} = w_t - \eta\, \nabla f(w_t). $$

In words: the new $w$ equals the old $w$ minus the learning rate times the gradient. Each symbol: $w_t$ is your current parameters, $\nabla f(w_t)$ is the uphill direction there, $\eta$ scales how far you move against it, and $w_{t+1}$ is where you land. The minus sign is what makes it *descent* — you move against the uphill compass.

Choose $\eta$ too small and you crawl: each step is a baby step, and you may need a million of them. Choose it too big and you overshoot the valley, landing on the far slope higher than you started; do that repeatedly and the cost *grows* — the optimizer **diverges**. There is a Goldilocks zone in between, and finding it is half the art of training models. We will see exactly what too-big and too-small do to a real risk number later.

#### Worked example: one gradient-descent step on a tiny portfolio

Let us make the abstract concrete with the smallest real problem: two assets, and we want the minimum-variance mix. Say asset A has variance 0.04 (a 20% volatility) and asset B has variance 0.09 (a 30% volatility), and they are uncorrelated, so the covariance matrix is

$$ \Sigma = \begin{bmatrix} 0.04 & 0 \\ 0 & 0.09 \end{bmatrix}. $$

We want weights $w = [w_A, w_B]$ that minimize portfolio variance $f(w) = w^\top \Sigma w = 0.04\,w_A^2 + 0.09\,w_B^2$, subject to the weights summing to 1. To keep the arithmetic clean, substitute $w_B = 1 - w_A$ so the cost depends on one number:

$$ f(w_A) = 0.04\,w_A^2 + 0.09\,(1-w_A)^2. $$

The gradient (the slope) is $f'(w_A) = 0.08\,w_A - 0.18\,(1 - w_A) = 0.26\,w_A - 0.18$. Start at a naive 50/50 guess, $w_A = 0.5$, and use learning rate $\eta = 1.0$.

- **Step 0:** $w_A = 0.5$. Slope $= 0.26(0.5) - 0.18 = -0.05$. Negative slope means downhill is to the right (increase $w_A$). New $w_A = 0.5 - 1.0 \times (-0.05) = 0.55$.
- **Step 1:** $w_A = 0.55$. Slope $= 0.26(0.55) - 0.18 = -0.037$. New $w_A = 0.55 + 0.037 = 0.587$.
- **Step 2:** $w_A = 0.587$. Slope $= 0.26(0.587) - 0.18 = -0.0274$. New $w_A = 0.614$.
- **Step 3:** $w_A = 0.614 \to 0.634$. **Step 5:** $\approx 0.66$. **Step 10:** $\approx 0.69$.

Each step moves us less because the slope is shrinking — we are approaching the flat bottom. The exact answer (set slope to zero: $0.26\,w_A = 0.18$) is $w_A = 0.692$, so $w_B = 0.308$. We hold *more* of the calmer asset, exactly as intuition says. The portfolio variance at the optimum is $0.04(0.692)^2 + 0.09(0.308)^2 = 0.0192 + 0.0085 = 0.0277$, a volatility of about $\sqrt{0.0277} = 16.6\%$ — lower than either asset alone, the whole point of diversification. On a \$10 million book, moving from the 50/50 guess (variance 0.0325, vol 18.0%) to the optimum trims the daily one-sigma swing from about \$113,000 to \$105,000. **The intuition: gradient descent does not jump to the answer; it converges to it, fast at first and then in ever-smaller steps as the ground flattens.**

## 1. Gradient descent, properly

We have the one-line update; now let us understand what makes it work, when it stalls, and how the convex-versus-non-convex distinction plays out in practice.

### Convergence on a convex bowl

On a convex cost — portfolio variance, OLS regression, ridge regression, logistic regression — gradient descent with a small enough learning rate is *guaranteed* to converge to the global minimum. There is even a clean rule for how small "small enough" is: for a quadratic bowl whose steepest curvature is $L$ (the largest eigenvalue of the Hessian, the matrix of second derivatives), any learning rate below $2/L$ converges, and the sweet spot is around $1/L$.

The speed depends on the *shape* of the bowl. A perfectly round bowl converges in essentially one step. A long, thin, taco-shaped valley is brutal: the gradient mostly points across the valley, not down its length, so you zig-zag slowly toward the far end. The ratio of the widest to the narrowest curvature is called the **condition number**, and a high condition number means slow, zig-zaggy convergence. This is why quants *precondition* or *standardize* their features before fitting — scaling each feature to similar size rounds out the bowl and speeds everything up. We covered the regression side of this in [OLS, GLS, and regularized regression](/blog/trading/math-for-quants/regression-ols-gls-regularized-math-for-quants).

It is worth being precise about what "convergence" buys you, because the word hides two very different speeds. On a well-conditioned convex bowl, the error shrinks *geometrically* — each step multiplies the remaining gap by a constant factor less than one, so the number of steps to reach a given precision grows only like the logarithm of how precise you want to be. Going from 1% error to 0.01% error costs roughly *twice* the steps, not a hundred times more. That is fast. But on a badly conditioned bowl — condition number in the thousands, which happens whenever two of your features are nearly collinear (two factors that move almost together) — that constant factor is agonizingly close to one, and the same precision can cost hundreds of times more steps. The single most cost-effective thing a quant can do to speed up a fit is therefore not to switch optimizers but to fix the conditioning: standardize features, drop near-duplicates, or add a touch of ridge penalty, all of which shrink the condition number and steepen the path to the bottom.

### Where descent stalls: local minima and saddles

On a non-convex cost, gradient descent does exactly what its name says — goes downhill — and downhill is a *local* instruction. It will happily settle into the nearest valley even if a deeper one exists across a ridge. This is the central anxiety of training neural-network alpha models: did we find the *best* model, or just *a* model?

There is good news that took the field years to appreciate. In very high dimensions — and a neural network has millions of dimensions — true local minima that are much worse than the global one are rare. What is common is **saddle points**: flat spots that are minima in some directions and maxima in others. Plain gradient descent slows to a crawl near a saddle because the gradient is tiny there, but it does not get *permanently* stuck the way it would in a deep false valley. The noise of stochastic methods (next section) and the inertia of momentum (section 3) are exactly what kicks the optimizer off saddles and keeps it moving. The practical upshot: do not lose sleep over local minima in big models; do worry about saddles slowing you down, and use methods built to escape them.

### What this costs and when it breaks

Plain (full-batch) gradient descent reads *every* data point to compute one exact gradient. If you have $N$ data points and $d$ parameters, one step costs roughly $N \times d$ arithmetic operations. For small $N$ this is fine and even ideal — the gradient is exact and the path is smooth. But when $N$ is hundreds of millions, paying that price for *every* step is the bottleneck we opened with. There is a second, subtler cost too: full-batch descent dives straight into the *nearest* minimum it can reach, and on a noisy financial loss surface the nearest minimum is frequently an overfit one that fits last month's quirks. So the full-batch method is not only slow on big data; it is also, perversely, *worse at generalizing* than the cheaper noisy method, because it lacks the random jostling that nudges a model toward a robust, flat valley. That is the problem stochastic gradient descent solves on both counts — speed and generalization — and it is where we go next.

## 2. Stochastic gradient descent: the trick that scales

![Before and after comparison of full-batch descent versus stochastic mini-batch descent](/imgs/blogs/stochastic-gradient-optimizers-math-for-quants-2.png)

The diagram above is the whole idea in one contrast, so let us read it. On the left, **full-batch** gradient descent reads all the data to compute one exact slope, then takes one clean, smooth step toward the bottom. Beautiful, but each step is expensive. On the right, **stochastic** gradient descent reads only a small random *mini-batch* — maybe 256 rows out of 300 million — computes a *noisy* estimate of the slope from just that sample, and takes a step. That step is wobbly; the path zig-zags. But it is hundreds of thousands of times cheaper, so you can take a hundred thousand of them in the time the full-batch method takes one. The zig-zags average out, and you arrive at essentially the same place, far sooner.

### Why a sample tells you the direction

Here is the statistical heart of it. The true gradient is an *average* over all your data of each point's individual gradient. The gradient from a random mini-batch is an average over a small random sample. And a sample average is an **unbiased estimate** of the true average — on average it points the right way. It is noisier (it has higher variance), but it is not *biased*. So each stochastic step is, in expectation, a step in the correct downhill direction. Take enough of them and the law of large numbers does the rest; the noise washes out. (For the machinery behind "the sample average is an unbiased, consistent estimate of the true average", see [the law of large numbers and the central limit theorem](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants).)

The mini-batch size $B$ trades two things. Small $B$ (say 32) means very cheap, very noisy steps — lots of zig-zag but blistering speed per step. Large $B$ (say 8,192) means more expensive, less noisy steps that look more like full-batch. Most practitioners land in the 128–1,024 range, often as large as fits comfortably in GPU memory, because bigger batches also parallelize better on the hardware.

### Online learning: adapting to a market that moves

There is a second gift hidden in SGD. Because each step uses only fresh data, you can keep stepping as *new* data arrives. This is **online learning**: the model never "finishes" training; it continuously nudges its parameters toward whatever the market is doing now. For a trading signal this is gold, because the relationship between features and returns *decays* — an edge that worked last year may be arbitraged away this year. An online SGD-trained model naturally down-weights old data and tracks the moving target. The same property that makes SGD fast on huge static datasets makes it *adaptive* on streaming ones. We discuss the alpha-decay side of this in [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research).

### The noise of SGD as free regularization

Now the subtle and wonderful part. The wobble in SGD is not just a tolerable cost — it is an *asset*. Because each step is noisy, SGD cannot settle into a sharp, narrow valley; the noise rattles it out. It prefers *wide, flat* valleys — and wide flat valleys correspond to models that generalize better, that are less overfit to the quirks of the training data. In effect, the randomness of SGD acts as a built-in form of **regularization** (a penalty that discourages overfitting), without you adding any explicit penalty term. A full-batch optimizer, by contrast, will dive precisely into the deepest, sharpest hole it can find — which is often an overfit one. This is one reason almost no one trains large models with full-batch descent even when they could: the noise is part of *why it works*.

#### Worked example: SGD versus full-batch on a factor regression

Suppose you are fitting a linear factor model — predicting daily stock returns from $d = 50$ features — on $N = 5{,}000{,}000$ stock-days of data. Compare the cost of reaching a good fit two ways.

**Full-batch.** One gradient costs reading all 5,000,000 rows × 50 features = 250 million multiply-adds. Say a good fit needs 200 such steps. Total work: $200 \times 250\text{M} = 50$ **billion** operations. If each step takes 0.4 seconds on your machine, that is $200 \times 0.4 = 80$ seconds — and that is for a clean convex problem; a messier loss would need thousands of steps.

**SGD with batch size 256.** One stochastic gradient costs $256 \times 50 = 12{,}800$ operations — about 20,000× cheaper per step than full-batch. SGD needs *more* steps because each is noisy — say 40,000 steps to get an equally good fit (this is one full pass, an "epoch", over 10.2 million-ish samples with replacement, roughly 8 passes worth). Total work: $40{,}000 \times 12{,}800 = 512$ **million** operations — about **100× less total work** than full-batch, despite 200× more steps. In wall-clock terms, if each tiny step takes 0.05 milliseconds, that is $40{,}000 \times 0.05\text{ms} = 2$ seconds versus 80 seconds.

The dollar framing: if you re-fit this model every night, full-batch costs you 80 seconds of a \$30/hour cloud GPU = about \$0.67 per night, or \$245/year. SGD costs 2 seconds = \$0.017 per night, \$6/year. The \$240/year difference is trivial on one model — but a real research team fits *thousands* of model variants in a backtest sweep, and there the 40× wall-clock speedup is the difference between a research idea you can test today and one you cannot. **The intuition: SGD wins not by taking better steps but by taking vastly cheaper ones, so many that the noise averages away and the total work plummets.**

## 3. Momentum, RMSProp, and Adam: the steps that actually train models

Plain SGD is a good worker with a bad sense of direction: it follows the noisy gradient literally, zig-zagging across narrow valleys and crawling on flat stretches. The next three ideas fix the *step itself*, and together they are what trains essentially every machine-learning model in production, including ML alpha signals.

### Momentum: remember where you were going

Picture a heavy ball rolling downhill instead of a hiker taking discrete steps. The ball builds up speed in directions it has consistently been moving, and its inertia carries it across small bumps and through flat saddles. That is **momentum**. Instead of stepping purely by the current gradient, you keep a running average of recent gradients — a *velocity* — and step by that:

$$ v_{t+1} = \beta\, v_t + (1-\beta)\,\nabla f(w_t), \qquad w_{t+1} = w_t - \eta\, v_{t+1}. $$

Here $v$ is the velocity, and $\beta$ (typically 0.9) controls how much memory you keep. Because consistent downhill directions accumulate while zig-zag directions cancel out, momentum damps the cross-valley wobble and accelerates down the valley's length. In a long thin taco of a loss surface, momentum can be several times faster than plain SGD, and its inertia is exactly what coasts the optimizer off saddle points where the gradient is nearly zero.

### RMSProp: scale each parameter's step to its own gradient

Different parameters can have wildly different gradient sizes. A feature that is naturally large produces large gradients; a tiny feature produces tiny ones. With a single global learning rate, you are forced to set it small enough not to blow up the big-gradient parameters — which means the small-gradient parameters crawl. **RMSProp** fixes this by giving each parameter its *own* effective step, dividing the global rate by a running estimate of that parameter's recent gradient magnitude:

$$ s_{t+1} = \rho\, s_t + (1-\rho)\,\nabla f(w_t)^2, \qquad w_{t+1} = w_t - \frac{\eta}{\sqrt{s_{t+1}} + \epsilon}\,\nabla f(w_t). $$

Here $s$ tracks the average squared gradient per parameter, and dividing by its square root means parameters with consistently large gradients take smaller steps and vice versa. The tiny $\epsilon$ (like $10^{-8}$) just avoids dividing by zero. The effect is a self-tuning per-parameter learning rate that keeps every dimension moving at a sensible pace.

### Adam: momentum and per-parameter scaling together

**Adam** ("adaptive moment estimation") is the workhorse: it simply combines momentum (a running average of the gradient, the first moment) and RMSProp (a running average of the squared gradient, the second moment), with a small bias correction for the early steps. It is the default optimizer for training neural networks and gradient-boosted-tree refinements precisely because it *just works* across a huge range of problems without heroic learning-rate tuning. For a quant building an ML alpha model — a neural net that maps a few hundred market features to a return forecast — Adam is almost always the first thing reached for. It is robust to the noisy, non-stationary gradients that market data produces, and its per-parameter adaptivity handles features that live on very different scales.

The price of Adam is two extra running averages per parameter — three times the memory of plain SGD — and a known tendency to find slightly *worse* (sharper, more overfit) minima than well-tuned plain SGD-with-momentum on some problems. Many large-model teams therefore use Adam to get most of the way fast, then switch to plain SGD with a decaying rate to settle into a flatter, better-generalizing valley. There is no free lunch; there is a good default, and Adam is it.

A practical note that saves real grief: Adam's adaptivity does *not* free you from tuning the learning rate, it only widens the range of rates that work. A common myth is that because Adam "adapts", you can throw any rate at it. In practice the good range for Adam on most problems is around $10^{-4}$ to $10^{-3}$, and stepping outside it still produces the crawl-or-diverge failure of plain descent. Two more knobs matter in finance specifically. First, **weight decay** — a small pull of every parameter toward zero each step — is the cleanest regularizer to pair with Adam, and the AdamW variant implements it correctly (plain Adam tangles weight decay with the adaptive scaling and weakens it). Second, the running averages take a few hundred steps to warm up; on the noisy, fat-tailed gradients that market data produces, a single outlier day can briefly poison the second-moment estimate and make a parameter lurch. Clipping gradients to a sane maximum size before they reach Adam is standard insurance on financial data, where a flash crash or a bad print can otherwise detonate an otherwise healthy training run.

![Stack of learning-rate effects from too small through just right to diverging](/imgs/blogs/stochastic-gradient-optimizers-math-for-quants-4.png)

The stack above previews where we are headed: all the cleverness of Adam still rides on top of a learning rate, and the learning rate is what decides whether the whole machine crawls, converges, or blows up. Consider the difference between a rate that gently shrinks the loss each step and one that overshoots and amplifies it — the same model, the same data, the same gradients, and yet one run lands on a usable model and the other prints garbage. We will quantify that exact failure with a dollar risk number in section 6. First, the other branch of the family tree: methods that use *curvature*.

## 4. Newton and quasi-Newton: using curvature to leap

![Tree of optimizer families splitting into first-order and second-order methods](/imgs/blogs/stochastic-gradient-optimizers-math-for-quants-5.png)

The tree above sorts the whole field, so let us read it as our map. At the root are all numerical optimizers. They split into two branches. **First-order** methods use only the slope (the gradient) — that is everything we have met so far: GD, SGD, momentum, Adam. **Second-order** methods use the slope *and* the curvature — how the slope itself is changing. The curvature is captured by a matrix of second derivatives called the **Hessian**, written $H$ or $\nabla^2 f$. The second-order branch splits again into exact **Newton's method**, which uses the full Hessian, and **quasi-Newton** methods like BFGS and L-BFGS, which cleverly *approximate* it. Each leaf is a method you will actually use; the tree tells you which family it belongs to and what information it exploits.

### Why curvature lets you take a bigger step

First-order methods know which way is downhill but not how *far* the bottom is. They guess the step size with the learning rate. Curvature removes the guess. If you know not just the slope but how fast the slope is flattening, you can compute *exactly* how far to step to reach the bottom — at least for a bowl-shaped (quadratic) region. That is **Newton's method**:

$$ w_{t+1} = w_t - H^{-1}\, \nabla f(w_t). $$

Read it as: instead of a fixed learning rate $\eta$, use the *inverse curvature* $H^{-1}$ as a smart, per-direction step size. In a direction where the bowl is steep (high curvature), it takes a small step; where the bowl is shallow, a big one. On a perfect quadratic — and portfolio variance and OLS *are* perfect quadratics — Newton's method lands on the exact minimum in a **single step**. No learning rate to tune, no thousands of iterations. It is the closest thing to magic in this whole post.

### The catch: the Hessian is expensive

Why isn't everything Newton's method, then? Because the Hessian is a $d \times d$ matrix, and you have to *invert* it. For $d = 2$ parameters that is trivial. For $d = 100$ it is fine. For $d = 1{,}000$ it is getting heavy. For $d = 10{,}000{,}000$ (a neural net) it is utterly impossible — the Hessian alone would not fit in memory, let alone be inverted. Newton's method is a *low-dimensional* tool. And that is exactly where it shines in finance: many calibration problems have only a handful of parameters.

### Quasi-Newton: BFGS and L-BFGS

**Quasi-Newton** methods get most of Newton's speed without computing the Hessian. They *build up* an approximation of the inverse Hessian from the gradients they observe along the way — each step teaches them a little more about the curvature. **BFGS** (named for Broyden, Fletcher, Goldfarb, and Shanno) keeps a full $d \times d$ approximation and is excellent for problems with up to a few thousand parameters. **L-BFGS** ("limited-memory BFGS") stores only the last handful of gradient differences instead of the full matrix, so it scales to tens of thousands of parameters while still using curvature information. L-BFGS is the standard tool for fitting GARCH models, calibrating a parametric volatility surface, maximizing a likelihood, and any other smooth, low-to-medium-dimensional fit where you want Newton-like speed without Newton-like memory.

The dividing line is sharp and worth memorizing: **first-order methods (SGD, Adam) for huge-dimensional, huge-data problems like ML alpha models; second-order methods (Newton, L-BFGS) for small, smooth, low-dimensional calibration problems.** Use the wrong one and you either run out of memory (Newton on a neural net) or wait forever for a precision you could have had in five steps (SGD on a 4-parameter vol model).

![Stack contrasting Newton using curvature against first-order using slope only](/imgs/blogs/stochastic-gradient-optimizers-math-for-quants-7.png)

The stack above makes the contrast vivid. Picture two ways to fit a small smooth model. The first-order method takes the slope, multiplies by a fixed learning rate, and steps — over and over, hundreds of times, inching toward the bottom. Newton's method divides the slope by the *curvature* and lands on a perfect quadratic in one step. The cost of that leap is inverting a small Hessian each step, which for a 2-parameter model is nothing. For the right problem, second-order is not a little faster — it is two orders of magnitude faster, as the next worked example shows.

#### Worked example: one Newton step versus many GD steps to calibrate a 2-parameter model

You are calibrating a tiny pricing model with two parameters, $a$ and $b$, to match market data. Your calibration error (sum of squared pricing gaps) turns out to be a clean quadratic bowl:

$$ f(a, b) = 3(a - 2)^2 + 50(b - 1)^2. $$

The true answer is obviously $a = 2$, $b = 1$ (where the bowl bottoms out at cost 0), but pretend you do not know that. Notice the bowl is 16× steeper in the $b$ direction than the $a$ direction — a condition number of about 17 — which is exactly the kind of stretched bowl that punishes first-order methods. Start at $a = 0$, $b = 0$.

**Gradient descent.** The gradient is $\nabla f = [\,6(a-2),\ 100(b-1)\,]$. To not blow up the steep $b$ direction you must keep $\eta$ below $2/100 = 0.02$; take $\eta = 0.018$. The $b$ coordinate converges in a few dozen steps, but the *shallow* $a$ direction crawls: each step moves $a$ by only $0.018 \times 6 \times (\text{distance to }2)$, about 11% of the remaining gap per step. To close the $a$-gap to within 1% takes roughly $\ln(0.01)/\ln(0.89) \approx 40$ steps; to within 0.01% takes about **80 steps**. Each step re-evaluates the model's prices across the whole market grid — say 200 quotes — so 80 steps is 16,000 price evaluations.

**Newton's method.** The Hessian is constant here, $H = \begin{bmatrix} 6 & 0 \\ 0 & 100 \end{bmatrix}$, so $H^{-1} = \begin{bmatrix} 1/6 & 0 \\ 0 & 1/100 \end{bmatrix}$. One Newton step from $(0,0)$:

$$ w_1 = \begin{bmatrix} 0 \\ 0 \end{bmatrix} - \begin{bmatrix} 1/6 & 0 \\ 0 & 1/100 \end{bmatrix} \begin{bmatrix} 6(0-2) \\ 100(0-1) \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix} - \begin{bmatrix} -2 \\ -1 \end{bmatrix} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}. $$

**One step.** Exactly the answer. The inverse curvature un-stretched the bowl perfectly, so both coordinates arrived at once. The dollar framing: if each model re-evaluation across the market grid costs your calibration engine 5 milliseconds, gradient descent spends $16{,}000 \times 5\text{ms} = 80$ seconds, while Newton's method (one step plus a couple of safety checks, say 4 evaluations) spends 20 milliseconds. On a desk that re-calibrates this model **every time the market ticks** — thousands of times a minute — that is the difference between a calibration that keeps up with the market and one that is hopelessly stale. L-BFGS gives you nearly the same speed on bigger versions of this problem without forming the Hessian explicitly. **The intuition: when the model is small and smooth, curvature information collapses hundreds of slope-only steps into a single exact leap.**

## 5. Step-size schedules and early stopping

We have the families. Now the craft — the practical knobs that decide whether a sound method actually trains a good model.

### Learning-rate schedules: big steps early, small steps late

A fixed learning rate is a compromise: small enough to converge precisely at the end, which means too small to move fast at the start. The fix is a **schedule** that *decays* the rate over time. Common shapes:

- **Step decay:** cut the rate by a factor (say ×0.1) every fixed number of epochs.
- **Exponential / cosine decay:** smoothly shrink the rate toward zero over training, the cosine shape being a gentle curve that spends extra time at both high and low rates.
- **Warmup then decay:** start tiny, ramp *up* over the first few hundred steps (so the model does not lurch while its running averages are still garbage), then decay. This is standard for training large models.

The principle is simple: take big confident steps while you are far from the bottom and the gradient is informative, then shrink the steps to settle precisely into the valley without bouncing. A good schedule routinely halves the number of steps needed versus a fixed rate.

### Early stopping: quit while you are ahead

Here is the most important practical idea in this whole section, because it is where optimization meets the thing a quant actually cares about — *out-of-sample* performance. As you train, you watch the cost on two datasets: the **training set** the optimizer is fitting, and a held-out **validation set** it never sees. Early on, both costs fall together — the model is learning real structure. At some point the validation cost flattens and then starts *rising* even as the training cost keeps falling. That divergence is **overfitting**: the model has started memorizing the noise in the training data, which helps on the training set and hurts everywhere else. **Early stopping** means halting training at the moment validation cost bottoms out, before it turns up.

Early stopping is not just a convenience; it is a genuine form of **regularization**. Stopping early keeps the parameters closer to their small starting values, which is mathematically similar to adding a penalty that discourages large parameters (ridge regression). For a trading model this is the difference between a backtest that looks gorgeous and live trading that bleeds money: the overfit model has fit last year's noise, and last year's noise does not repeat. We unpack the held-out-data discipline in detail in [bootstrap and cross-validation](/blog/trading/math-for-quants/bootstrap-cross-validation-math-for-quants).

### When this matters

The takeaway for a practitioner: spend your tuning energy on the learning rate and the schedule first — they matter far more than which fancy optimizer you pick — and *always* watch a validation curve so you can stop at the right moment. A mediocre optimizer with a good schedule and honest early stopping beats a state-of-the-art optimizer run blind.

## 6. The learning rate, quantified: from a good fit to a \$4 billion error

![Before and after of a good learning rate converging versus a bad one diverging into a wrong risk number](/imgs/blogs/stochastic-gradient-optimizers-math-for-quants-6.png)

The before-and-after above is the warning every quant should internalize. Both panels are the *same* model fitting the *same* data with the *same* gradients — the only difference is the learning rate. On the right, a tuned rate makes the loss fall every step, the weights stay finite, and the model converges to a correct risk number. On the left, a rate that is too large makes the loss *grow* every step; the weights overshoot, then overshoot worse, then become "not-a-number" (the computer's way of saying infinity), and any number you read off the broken model is meaningless. Picture the loss climbing from 100 to 250 to 900 to a million in four steps — that is divergence, and it happens silently unless you are watching. The consequences are not academic, as the worked example shows.

#### Worked example: learning rate too big versus too small, and the dollar cost

You are fitting a small risk model whose job is to estimate the 1-day 99% **Value-at-Risk (VaR)** of a \$100 million portfolio — the loss you expect to exceed only 1 day in 100. The cost surface is a simple bowl whose steepest curvature is $L = 4$. The convergence rule says any learning rate below $2/L = 0.5$ is stable, and the sweet spot is around $1/L = 0.25$. The true converged VaR is **\$1.2 million**.

**Too small ($\eta = 0.0025$, i.e. 100× too small).** Each step shrinks the remaining error by only about $\eta \times L = 1\%$. To get within 0.1% of the true VaR you need roughly $\ln(0.001)/\ln(0.99) \approx 690$ steps. If the model re-fits each step on a chunk of data taking 0.2 seconds, that is 690 × 0.2 = **138 seconds** — and if you only let it run for 30 seconds (150 steps) before the market open, you stop at about 78% of the way there. The model reports a VaR of around **\$0.94 million** instead of \$1.2 million — it *understates* risk by \$260,000, and you size your positions as if you can lose less than you really can. The error here is silent: the loss was still falling, everything *looked* fine, you just ran out of time.

**Too big ($\eta = 1.0$, i.e. 4× too large, above the $2/L = 0.5$ stability limit).** Now each step *amplifies* the error by a factor of $|1 - \eta L| = |1 - 4| = 3$. Start with an error of \$0.5 million in the VaR estimate:

- Step 1 error: \$0.5M × 3 = \$1.5M.
- Step 2: \$4.5M. Step 3: \$13.5M. Step 4: \$40.5M. Step 5: \$121.5M.

After a handful of steps the estimate has blown past the entire \$100M portfolio value and is racing toward infinity. If a careless pipeline reads the VaR off at, say, step 8, it might report a VaR of **\$4.1 billion** on a \$100 million book — a number so absurd it should trip a circuit breaker, but in a complex system it might instead just get clamped, logged, or silently fed into a position-sizing rule that now refuses to trade anything. Either way the risk number is garbage.

The dollar framing in one line: the *same* model, the *same* data, the *same* math — and the learning rate alone decides whether you get a correct **\$1.2 million** risk estimate, a dangerously low **\$0.94 million** one, or a meaningless **\$4.1 billion** one. **The intuition: the learning rate is not a minor tuning detail; below the stability limit you converge, above it you diverge, and the cliff between is steep enough to turn a real risk number into fiction.**

## Common misconceptions

**"More data per step always means faster training."** The opposite is usually true. Full-batch gradient descent uses the most data per step and is almost always *slower* to a good model than mini-batch SGD, because the per-step cost dwarfs the benefit of an exact gradient. A noisy gradient from 256 rows, used a thousand times, beats an exact gradient from 5 million rows used once. The currency that matters is *total work to a good answer*, not gradient precision per step.

**"The noise in SGD is a bug to be minimized."** The noise is a feature. It regularizes — it nudges the optimizer toward wide, flat, well-generalizing valleys and away from sharp, overfit ones, and it kicks the optimizer off saddle points. Crank the batch size up to remove all the noise and you often get a model that trains "cleanly" and then generalizes *worse*. Some randomness is doing real work.

**"Adam is strictly better than SGD."** Adam is a better *default* — it tunes itself across many problems with little fuss. But well-tuned SGD with momentum frequently finds flatter, better-generalizing minima, which is why many production pipelines switch from Adam to SGD for the final phase of training. "Adaptive" is not the same as "optimal"; it is "robust out of the box".

**"Local minima are the big danger in training neural nets."** In high dimensions, bad local minima are rare; *saddle points* are the real obstacle, and they slow you down rather than trap you forever. Momentum and SGD noise both help escape them. The lore about local minima comes from low-dimensional intuition that does not transfer to million-parameter models.

**"Newton's method is just a faster gradient descent, so use it everywhere."** Newton's method needs the Hessian — a $d \times d$ matrix you must invert every step. That is wonderful for a 4-parameter vol-surface calibration and impossible for a 10-million-parameter network. First-order methods scale; second-order methods are precise on small problems. They are different tools for different jobs, not a ranking.

**"If the training loss is going down, the model is getting better."** Only the *validation* loss tells you that. Training loss falls monotonically by construction; the moment validation loss turns up, every further step is making the model worse out-of-sample even as the optimizer reports "progress". Watch the validation curve, and stop when it bottoms.

## How it shows up in real markets

### 1. Overnight re-fitting of factor models

A systematic equity fund typically re-fits its return-forecasting models every night on the latest data — tens of millions of stock-days, dozens to hundreds of features. Full-batch optimization would make the nightly window impossibly tight, so these fits run on mini-batch SGD or Adam, often with a learning-rate schedule tuned to converge in a fixed number of steps that fits the overnight budget. The choice of optimizer and schedule is not academic housekeeping; it is what makes "re-fit every model, every night, before the open" physically possible. The same noise that makes SGD fast also keeps the nightly model from overfitting to a single day's quirks.

### 2. Calibrating the volatility surface in real time

An options desk must keep a parametric volatility surface — a model like SVI with roughly five parameters per expiry — matched to live market quotes that move every second. This is a small, smooth, low-dimensional calibration: the textbook home of L-BFGS or a Levenberg-Marquardt (a Newton-flavored) solver. These methods exploit curvature to re-calibrate in a handful of iterations, fast enough to keep up with the tick. Use SGD here and the surface lags the market by seconds — an eternity when you are quoting two-sided markets and someone is picking off your stale prices. The dimensionality of the problem dictates the optimizer, and getting it right is worth real money in tighter, safer quotes.

### 3. GARCH and the daily volatility estimate

Risk teams fit GARCH-family models — three or four parameters that describe how volatility clusters and decays — to return histories, usually by maximizing a likelihood. The likelihood surface is smooth and low-dimensional, so L-BFGS is the standard solver and converges in a dozen-ish iterations. A too-large step here can send a parameter outside its valid range (volatility persistence must stay below 1, or the model implies infinite long-run variance), so the optimizer is wrapped in bounds and the step size is watched. The fitted parameters feed straight into the daily VaR and the margin a clearinghouse demands — so the calibration's accuracy is, quite literally, money posted as collateral.

### 4. Training a machine-learning alpha signal

When a quant builds a neural-network alpha model — a few hundred features in, a return forecast out — Adam is almost always the first optimizer, chosen for robustness to the noisy, non-stationary gradients market data produces, and for its per-parameter adaptivity across features on wildly different scales. Training watches a validation set (a held-out period the model never sees) and stops early when validation performance peaks. Skip early stopping and the model fits last quarter's noise, backtests beautifully, and loses money live. The whole discipline of fitting these models cross-sectionally — and comparing them against simpler tree models — is covered in [tree models for cross-sectional prediction](/blog/trading/quantitative-finance/tree-models-cross-sectional-prediction-quant-research).

### 5. The exploding-gradient blowup

Every desk that trains models has a war story about a run where the loss went to "NaN" — not-a-number — overnight. The cause is almost always a learning rate (or a data spike, or an un-clipped gradient) that pushed an update past the stability limit, after which the error amplified itself step over step into infinity, exactly as our \$4.1 billion example showed. The standard defenses are *gradient clipping* (cap the gradient's size so a single huge value cannot detonate a step), learning-rate warmup, and validation-loss monitoring with automatic rollback. The episode is a vivid reminder that the optimizer sits between your data and your risk numbers, and a mis-set knob there can fabricate a number with no basis in reality.

### 6. Online learning and alpha decay

Edges in markets decay — a signal that predicted returns last year gets crowded and arbitraged away. Funds that retrain continuously, nudging parameters with fresh data via online SGD, track the moving target better than funds that fit once and freeze. The same property — each step uses only recent data — that makes SGD scale to huge static datasets makes it *adaptive* on streaming ones. The risk is the mirror image: over-adapt to recent noise and you chase ghosts. The cure is the same as everywhere else in this post — a held-out check and a sensible step size that lets the model follow real regime change without lurching at every blip.

### 7. The 1998 lesson, retold through optimization

When Long-Term Capital Management blew up in 1998, the proximate cause was leverage and correlated positions, but a deeper lesson echoes through every model-calibration pipeline since: a model calibrated to fit recent data *perfectly* is a model that has memorized recent noise. An optimizer driven to zero training error on a smooth-looking convex calibration will find parameters that match yesterday's prices exactly and say nothing trustworthy about tomorrow's. Modern practice — regularization, early stopping, holding out data, preferring flatter minima — is in large part a disciplined response to the truth that the *easiest* optimization (drive the cost to zero) is often the *most dangerous* one. The optimizer will do exactly what you ask; the art is asking for the right thing.

## When this matters to you

If you ever train a model — a trading signal, a credit-risk score, a demand forecast, anything — these ideas decide whether it works. The practical hierarchy, in order of how much it matters: **(1)** tune the learning rate and its schedule before anything else; **(2)** use mini-batch SGD or Adam for big-data, high-dimensional problems and L-BFGS or Newton for small smooth calibrations; **(3)** always hold out data and stop early when validation performance peaks; **(4)** treat the noise of SGD as a helper, not an enemy. Get those right and the choice between exotic optimizers barely matters.

For the reader who wants to go deeper, the natural next steps on this blog are the gradient-and-curvature foundations in [matrix calculus for optimization](/blog/trading/math-for-quants/matrix-calculus-optimization-math-for-quants), the regression objectives these optimizers actually minimize in [OLS, GLS, and regularized regression](/blog/trading/math-for-quants/regression-ols-gls-regularized-math-for-quants), the held-out-data discipline behind early stopping in [bootstrap and cross-validation](/blog/trading/math-for-quants/bootstrap-cross-validation-math-for-quants), and the applied side — what you do once you can train a model — in [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) and [tree models for cross-sectional prediction](/blog/trading/quantitative-finance/tree-models-cross-sectional-prediction-quant-research).

A closing honest note: everything here is educational, not investment advice. Optimizers find the parameters that minimize a cost *you* chose, on the data *you* fed them — they cannot tell you whether the cost was the right one or whether the past resembles the future. The math is reliable; the modeling judgment around it is where the real risk, and the real edge, lives.

> The optimizer always finds what you asked for. The whole craft is making sure you asked for the right thing.

## A matrix of the four optimizers, for reference

![Matrix comparing GD, SGD, Adam, and L-BFGS across data scale, curvature use, per-step cost, and best job](/imgs/blogs/stochastic-gradient-optimizers-math-for-quants-3.png)

The matrix above collapses the whole post into a decision table — keep it next to you when you pick an optimizer. Read across each row: **GD** uses no curvature, pays a full data pass per step, and scales only to small data, so it is mostly a teaching tool. **SGD** scales to huge data with cheap per-batch steps and is the workhorse for big factor datasets. **Adam** adds approximate per-parameter adaptivity on top of SGD's scaling and is the default for ML alpha nets. **L-BFGS** uses real curvature and pays a full pass per step but converges in few iterations, making it the right choice for small smooth fits like a volatility surface or GARCH. The single rule the table encodes: match the optimizer to the *shape* of the problem — huge and high-dimensional calls for first-order methods, small and smooth calls for second-order ones. Pick by the problem, not by the fashion, and the rest of training gets dramatically easier.

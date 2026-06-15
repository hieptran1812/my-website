---
title: "The Kalman filter and state-space models: tracking a hidden fair value in real time"
date: "2026-06-15"
description: "A build-from-zero guide to state-space models and the Kalman filter for trading: the predict-update cycle, the Kalman gain as a trust dial, a time-varying hedge ratio, a drifting mean return, and the smoother and nonlinear cousins, all in worked dollar examples."
tags: ["kalman-filter", "state-space-models", "predict-update", "kalman-gain", "hedge-ratio", "pairs-trading", "time-varying-beta", "recursive-least-squares", "quant-finance", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A Kalman filter is a recipe for tracking something you cannot see directly — a fair value, a true mean return, a hedge ratio — using a stream of noisy measurements, and updating your estimate optimally every time a new number arrives. The whole engine is one short loop: predict where the hidden quantity went, look at the new noisy print, then split the difference using a single number called the Kalman gain.
>
> - **State-space form** splits the world into a *hidden state* that drifts over time and a *noisy observation* of it. Markets give you the noisy observation; the state — fair value, true beta, true mean — is what you actually want.
> - **The predict-update cycle** is the entire algorithm: predict the state and your uncertainty, then correct with the new observation, weighted by the **Kalman gain**.
> - **The Kalman gain is a trust dial** between 0 and 1: near 1 you trust the new print and snap to it; near 0 you trust your model and barely move. It is set automatically by which noise — the model's or the measurement's — is larger.
> - **Killer application:** a *time-varying hedge ratio* for pairs trading. A static OLS beta freezes the relationship; a Kalman beta tracks it as it drifts, and on a drifting pair that can be the difference between a hedge that holds and one that bleeds — in our worked example, roughly **\$3,000** of slippage avoided over one drift.
> - The one number to remember: when the gain is **0.5**, you move your estimate exactly **halfway** from your prediction to the new print — a clean 50/50 blend of model and data.

You have a stock that just printed at \$100.40. A moment ago it printed at \$99.30, and before that \$101.10. So what is the stock actually *worth* right now? Not the last print — that one is contaminated by a fat-fingered order, a momentary liquidity gap, the random jostle of the order book. The *true* fair value is hidden behind all that noise, and yet you have to act on a number. If you quote a market, hedge a position, or size a trade, you are implicitly committing to an estimate of a quantity you can never observe cleanly.

This is the single most common situation in quantitative trading, and it has a beautiful, exact answer. The Kalman filter is the mathematics of estimating a hidden, drifting quantity from a stream of noisy observations — and doing it *optimally*, in the precise sense that no other linear method can beat its accuracy when the noise is Gaussian. It was built to land Apollo on the moon by tracking a spacecraft from noisy radar; quants borrowed it to track fair values, hedge ratios, and slowly-drifting means. The diagram below is the mental model: a loop that predicts, observes, corrects, and repeats forever.

![Predict observe update repeat loop of the Kalman filter](/imgs/blogs/kalman-filter-state-space-math-for-quants-1.png)

That loop is the whole filter. Predict where the hidden quantity drifted to; look at the new noisy print; compute how much to trust that print; nudge your estimate toward it by exactly that much; carry the result forward and do it all again on the next tick. Everything else in this post — the formulas, the matrices, the worked dollar examples — is just unpacking those five boxes. Let us build the picture from absolutely nothing, because the payoff is that you will be able to *derive* a working hedge ratio tracker by the end, not just recite one.

## Foundations: how a state-space model works

Before we can filter anything, we need the language of *state-space models*. The name sounds intimidating; the idea is something you already do every day. Let us define every piece from zero.

### The hidden state versus the noisy observation

Picture trying to figure out how warm a room is using a cheap, jittery thermometer. The *true* temperature is one thing — call it the **state**. The thermometer's reading is another thing — call it the **observation**. The reading is the true temperature plus some random error: sometimes it reads a degree high, sometimes a degree low. You never see the true temperature directly; you only ever see noisy readings of it.

A *state-space model* is exactly this split, written down with a little math. It has two ingredients:

1. **A state** — the hidden quantity you actually care about. In trading this might be the fair value of a stock, the true hedge ratio between two assets, or the true expected return of a strategy. We write the state at time $t$ as $x_t$.
2. **An observation** — the noisy number the market actually hands you. The last trade price, a regression's noisy beta estimate, a single trade's profit and loss. We write the observation as $z_t$.

The word *state* is borrowed from physics and engineering: the state is the minimal summary of everything you need to know about the system right now to predict its future. If the state is a stock's fair value, then knowing the fair value (plus how it tends to move) is enough to say where it will probably be next — you do not also need the entire history of every past print.

### The two equations: transition and measurement

A state-space model is defined by two equations, and they map one-to-one onto the two ingredients above.

The first is the **transition equation** (also called the *state equation* or *process equation*). It says how the hidden state evolves from one step to the next:

$$x_t = F\, x_{t-1} + w_t, \qquad w_t \sim \mathcal{N}(0, Q)$$

Here $F$ is the *transition matrix* — the rule for how the state drifts. The term $w_t$ is the **process noise**: the part of the state's evolution you cannot predict, drawn from a Gaussian (bell-curve) distribution with variance $Q$. A *variance* is just a number measuring how spread out a random quantity is; bigger $Q$ means the state wanders more between steps.

If the hidden state is a fair value that just drifts randomly with no built-in trend, then $F = 1$, and the equation reads $x_t = x_{t-1} + w_t$: today's fair value equals yesterday's plus a random nudge. This is called a *random walk*, and it is the workhorse model for a slowly-changing latent quantity.

The second equation is the **measurement equation** (also called the *observation equation*). It says how the noisy print relates to the hidden state:

$$z_t = H\, x_t + v_t, \qquad v_t \sim \mathcal{N}(0, R)$$

Here $H$ is the *measurement matrix* — how the observation maps to the state. The term $v_t$ is the **measurement noise**: the contamination in the print, Gaussian with variance $R$. Bigger $R$ means the prints are noisier and less trustworthy.

In the simplest fair-value model, $H = 1$, so $z_t = x_t + v_t$: the print equals the true fair value plus some random error. That is precisely the thermometer story — reading equals truth plus jitter.

The figure below stacks the parts so you can see how they fit together: a hidden state, a rule for how it drifts, a noisy measurement of it, and the two noise budgets ($Q$ and $R$) that the filter constantly weighs against each other.

![Stack of state-space model components state transition noise measurement](/imgs/blogs/kalman-filter-state-space-math-for-quants-2.png)

That is the entire vocabulary. State $x_t$, observation $z_t$, transition $F$ with process-noise variance $Q$, measurement $H$ with measurement-noise variance $R$. Memorize those five symbols and the rest of this article is arithmetic.

### Glossary of the units we will use

Because we will be juggling money and statistics, let us pin down the jargon now, each defined inline so you never have to guess:

- A *basis point* (bp) is one hundredth of a percent — 0.01%. A hedge ratio that is off by "10 bps" is off by 0.10%.
- *Notional* is the dollar size of a position — if you hold 1,000 shares at \$50, your notional is \$50,000.
- *Variance* measures spread (the square of the standard deviation); *standard deviation* (often $\sigma$, "sigma") measures the typical size of a deviation in the same units as the thing itself.
- A *hedge ratio* (or *beta*) is how many units of one asset you trade against one unit of another to neutralize their shared movement. If stock B moves \$1.20 for every \$1.00 stock A moves, the hedge ratio is 1.20.
- *Filtering* means estimating the state *now* using all data up to now. *Smoothing* (later in the post) means re-estimating a past state using data from after it too.

### Why "optimal" is not marketing

When people say the Kalman filter is *optimal*, they mean something exact, not vague praise. Among all estimators that are linear in the observations, the Kalman filter produces the estimate with the smallest possible mean-squared error when the noise is Gaussian. If the noise is Gaussian, it is the best estimator full stop, linear or not, because for Gaussian problems the best estimator happens to be linear. That is a strong guarantee — it means you are not leaving accuracy on the table by using this particular recipe. The cost is the assumption: the noise has to be roughly Gaussian and the relationships roughly linear, and we will spend a whole section on what happens when they are not.

#### Worked example: one predict-update step on a noisy print

Let us walk the simplest possible case all the way through, because the entire filter is just this step repeated.

You are tracking the fair value of a stock. Your model is a pure random walk: $F = 1$, $H = 1$. Suppose yesterday's filtered estimate of fair value was $\hat{x} = \$100.00$, and the variance on that estimate — your uncertainty, squared — was $P = 0.40$ (in dollars-squared). Your model says fair value drifts a bit each day with process-noise variance $Q = 0.10$. And the market's prints are noisy with measurement-noise variance $R = 0.50$.

Now a new print arrives: $z = \$100.40$. What should your new estimate be?

**Step 1 — Predict the state.** With $F = 1$, the predicted fair value is just yesterday's estimate carried forward: $\hat{x}^- = 1 \times 100.00 = \$100.00$. (The minus superscript means "before seeing the new print.")

**Step 2 — Predict the uncertainty.** The state drifted, so your uncertainty grows by the process noise: $P^- = F^2 P + Q = 1 \times 0.40 + 0.10 = 0.50$. You are a little less sure than you were, because time passed.

**Step 3 — Compute the Kalman gain.** The gain is the predicted uncertainty divided by the total uncertainty (state uncertainty plus measurement noise):
$$K = \frac{P^-}{P^- + R} = \frac{0.50}{0.50 + 0.50} = 0.50.$$
A gain of 0.50 means: trust the new print exactly halfway.

**Step 4 — Update the estimate.** Nudge your prediction toward the print by the gain times the surprise. The *surprise* (or *innovation*) is how far the print is from what you predicted: $100.40 - 100.00 = \$0.40$.
$$\hat{x} = \hat{x}^- + K\,(z - \hat{x}^-) = 100.00 + 0.50 \times 0.40 = \$100.20.$$
Your new fair-value estimate is \$100.20 — halfway between your \$100.00 prediction and the \$100.40 print, exactly as a gain of 0.5 dictates.

**Step 5 — Update the uncertainty.** Seeing data always sharpens you: $P = (1 - K)\,P^- = (1 - 0.50) \times 0.50 = 0.25$. Your uncertainty dropped from 0.50 to 0.25 because the print carried information.

The intuition: the filter never blindly believes the last print nor blindly clings to its old guess — it lands at a precise blend, and that blend is dictated entirely by how the two uncertainties compare.

## 1. The predict-update cycle, in full

Now we generalize the worked example into the canonical loop. The Kalman filter alternates between two phases on every tick: a **predict** phase (also called *time update*) and an **update** phase (also called *measurement update* or *correct*). Walk through the predict-observe-update-repeat loop in the mental-model figure once more and you will see these are just its first four boxes.

### The predict phase

Before the new observation arrives, you extrapolate where the state went and how much your uncertainty grew:

$$\hat{x}_t^- = F\,\hat{x}_{t-1} \qquad\text{(predicted state)}$$
$$P_t^- = F\,P_{t-1}\,F^\top + Q \qquad\text{(predicted covariance)}$$

The first line pushes the state forward by the transition rule. The second line pushes the *uncertainty* forward: $P$ is the **error covariance** — a number (or matrix, when the state has several components) measuring how much your estimate could be off. Crucially, prediction always *increases* uncertainty by adding $Q$. Time passing makes you less sure, because the hidden state may have wandered.

A helpful analogy is forecasting tomorrow's temperature from today's. Your best guess is "about the same," but you are less confident about tomorrow than about right now, and the $+Q$ encodes exactly that loss of confidence. The larger $Q$ is, the faster your confidence decays between observations — a stock whose fair value lurches around all day has a big $Q$, while a stock that barely moves has a tiny one, and the filter behaves very differently in the two cases even with identical prints.

### The update phase

When the observation $z_t$ arrives, you correct. First compute the **innovation** — the surprise, how far the actual print is from what you predicted:

$$y_t = z_t - H\,\hat{x}_t^-$$

Then the **innovation covariance** — how surprised you *should* be, combining your state uncertainty and the measurement noise:

$$S_t = H\,P_t^-\,H^\top + R$$

Then the star of the show, the **Kalman gain**:

$$K_t = P_t^-\,H^\top\,S_t^{-1}$$

And finally the corrections to both the estimate and the uncertainty:

$$\hat{x}_t = \hat{x}_t^- + K_t\,y_t \qquad\text{(updated state)}$$
$$P_t = (I - K_t H)\,P_t^- \qquad\text{(updated covariance)}$$

That is the complete algorithm. Five predict-and-update equations, looped. In the scalar fair-value case where $F = H = 1$, these collapse exactly to the five steps of the worked example above: $S_t = P_t^- + R$, $K_t = P_t^-/(P_t^- + R)$, and so on.

### Watching the uncertainty settle

There is one behavior worth pausing on, because it is what makes the filter feel alive rather than mechanical: the uncertainty $P$ and the gain $K$ are not fixed — they evolve on their own, even if the prints stop surprising you. Run the scalar loop forward with constant $Q$ and $R$ and you will see $P$ climb by $+Q$ in every predict step and shrink by the factor $(1 - K)$ in every update step. These two forces fight to a draw, and $P$ converges to a *steady-state* value where the climb exactly cancels the shrink. At that point the gain stops changing too, settling at a constant value determined entirely by the ratio $Q/R$. This is why a long-running filter eventually behaves like a fixed exponentially-weighted average — the "learning" has stabilized.

The practical upshot is that the *transient* phase, right after you start the filter with a deliberately large initial $P$, is when the filter learns fastest, with high gain, gobbling up information from the first few prints. You exploit this on purpose: when a market regime obviously changes — a major news event, a circuit-breaker halt and reopen — you can manually re-inflate $P$ to tell the filter "forget your confidence, the world just changed," and it will snap to the new reality within a few ticks instead of lazily drifting there. Re-inflating $P$ is the filter's panic button, and knowing when to press it separates a filter that survives regime shifts from one that fights them.

### What costs you and when it breaks

The cost of the predict-update cycle is essentially nothing — it is a handful of arithmetic operations per tick, which is why it runs comfortably at microsecond speeds inside trading systems. What breaks it is bad inputs: if your $Q$ and $R$ are wrong, the gain is wrong, and the filter either lags reality (gain too low) or chatters on noise (gain too high). The filter is optimal *given* the right noise variances; getting those right is the real work, and we will return to it.

> A Kalman filter does not predict the future. It estimates the present — the hidden state right now — better than any single measurement could. Confusing the two is the most common way people misuse it.

## 2. The Kalman gain: the trust dial

If you remember one thing from this entire post, make it the Kalman gain. It is the single number that decides how much the filter believes the new data versus its own model, and once you internalize it as a *trust dial* the whole filter becomes obvious.

Recall the scalar form:

$$K = \frac{P^-}{P^- + R}.$$

The numerator $P^-$ is how uncertain you are about your own prediction. The denominator adds $R$, the measurement noise. So the gain is "my uncertainty as a fraction of total uncertainty." Two limits make this concrete.

### The trust-the-data limit

Suppose your prints are *extremely* clean — the measurement noise $R$ is tiny, near zero. Then $K = P^-/(P^- + 0) \approx 1$. A gain of 1 means $\hat{x} = \hat{x}^- + 1 \times (z - \hat{x}^-) = z$: you throw away your prediction and just adopt the new print. That is correct! If the data is perfect, believe the data.

### The trust-the-model limit

Now suppose your prints are *garbage* — the measurement noise $R$ is huge. Then $K = P^-/(P^- + \text{huge}) \approx 0$. A gain of 0 means $\hat{x} = \hat{x}^- + 0 = \hat{x}^-$: you ignore the print entirely and keep your prediction. Also correct! If the data is worthless, ignore it and trust your model.

The matrix below lays out these two regimes side by side, so you can see the gain slide from near 1 (snap to the prints) down to near 0 (ignore the prints) as measurement noise grows.

![Matrix of Kalman gain trust regimes by measurement noise level](/imgs/blogs/kalman-filter-state-space-math-for-quants-4.png)

Everything in between is a smooth blend. The beauty is that you never set the gain by hand — it falls out of the noise variances automatically, and it *adapts* tick by tick as your uncertainty $P^-$ changes. Early on, when you are unsure, $P^-$ is large and the gain is high, so you learn fast from data. As you accumulate observations, $P^-$ shrinks, the gain falls, and you become appropriately stubborn. The filter has a built-in sense of "I have seen enough, stop overreacting to every print."

### Adapting faster or slower on purpose

You can *tune* the responsiveness by choosing the ratio $Q/R$, the *signal-to-noise* of your model. Raise $Q$ (you believe the true state moves a lot) and the gain stays high — the filter is nimble and tracks fast, at the cost of being jumpier. Lower $Q$ (you believe the true state is nearly constant) and the gain settles low — the filter is smooth and stable, at the cost of lagging real moves. This single ratio is the knob that controls everything about the filter's personality, and the right setting is an empirical question, not a theoretical one.

It helps to translate the gain into a number traders already have a feel for: the *effective memory* of the filter. A steady-state gain of $K$ behaves roughly like an exponentially-weighted average with a half-life of about $\ln(2)/K$ observations — so a gain of 0.10 has a memory of about 7 observations of half-weight and roughly 20 of meaningful weight, while a gain of 0.50 remembers barely one or two. When someone asks "how long does your filter look back," the honest answer is not a fixed window but "however long the gain implies," and the gain is set by $Q/R$, not by you picking a number out of the air. That reframing — from "choose a lookback window" to "choose a signal-to-noise ratio" — is the single biggest mindset shift in moving from moving averages to filters.

#### Worked example: gain extremes and the dollars they cost

Let us make the two regimes pay off in money. You are tracking a fair value, currently estimated at $\hat{x}^- = \$50.00$ with predicted variance $P^- = 0.20$. A print arrives at $z = \$50.60$ — a 60-cent surprise. You hold 10,000 shares, so every cent of estimate is \$100 of mark-to-market.

**Regime A — clean prints ($R = 0.02$, very low).**
$$K = \frac{0.20}{0.20 + 0.02} = 0.91.$$
$$\hat{x} = 50.00 + 0.91 \times 0.60 = \$50.55.$$
You move 55 cents toward the print, marking your book up by $0.55 \times \$100 = \$5{,}500$. With clean prints, you believe the move is real and reprice almost fully.

**Regime B — noisy prints ($R = 2.00$, very high).**
$$K = \frac{0.20}{0.20 + 2.00} = 0.091.$$
$$\hat{x} = 50.00 + 0.091 \times 0.60 = \$50.055.$$
You barely budge — up 5.5 cents, a \$550 mark. With noisy prints, you suspect the 60-cent move is mostly noise and refuse to chase it.

The same \$0.60 surprise produces a \$5,500 reprice in one world and a \$550 reprice in the other. The intuition: the Kalman gain is the exact dollar translation of "how much should I believe what I just saw," and getting $R$ right is worth thousands of dollars of premature or delayed repricing.

## 3. A time-varying hedge ratio for a pair

Here is where the Kalman filter earns its keep in trading. We will build a *dynamic hedge ratio* — a beta between two assets that updates every day — and see why it beats a static regression. This is the filter's signature quant application, and it connects directly to the world of [cointegration and pairs trading](/blog/trading/math-for-quants/bayesian-inference-traders-math-for-quants) where the relationship between two assets is the entire trade. (For the static-regression foundation it improves on, see the [linear regression deep-dive](/blog/trading/quantitative-finance/linear-regression-deep-quant-interviews).)

### The pairs-trading setup

In pairs trading, you find two assets that move together — say two oil majors, or a stock and its sector ETF — and you trade the *spread* between them, betting that when they diverge they will reconverge. To trade the spread cleanly, you must hedge: for each \$1 of asset A you buy, you short some amount $\beta$ of asset B, where $\beta$ is the hedge ratio. The spread you actually hold is:

$$\text{spread}_t = A_t - \beta\,B_t.$$

If $\beta$ is right, the spread is *stationary* — it wobbles around a stable mean and the shared market move cancels out. If $\beta$ is wrong, the spread inherits a chunk of raw market direction and your "market-neutral" trade is quietly a directional bet.

### Why a static beta fails

The standard way to get $\beta$ is ordinary least squares (OLS) regression over a historical window: regress A on B and take the slope. The problem is that $\beta$ is not constant. The relationship between two assets drifts — index reweightings, changing business mix, shifting correlations in different volatility regimes. A static OLS beta computed over the last 120 days is an *average* of a quantity that was different at the start of the window than at the end. By the time you use it, it is stale.

The before-and-after figure below makes the failure visible: a static beta of 1.20 stays frozen as the true relationship drifts to 1.55, so the hedge slips further out of line every week, while the Kalman beta tracks the drift and the hedge holds.

![Before after static OLS beta frozen versus Kalman beta tracking the drift](/imgs/blogs/kalman-filter-state-space-math-for-quants-3.png)

### The state-space form of a dynamic beta

Here is the elegant move. Treat the hedge ratio itself as the *hidden state*, and let it drift as a random walk:

$$\beta_t = \beta_{t-1} + w_t \qquad\text{(the hedge ratio drifts)}$$
$$A_t = \beta_t\,B_t + v_t \qquad\text{(the observation)}$$

Compare to the general state-space form. The state is $x_t = \beta_t$, the transition is $F = 1$ (a random walk), and the measurement matrix is $H = B_t$ — *the price of asset B is the measurement matrix*, and it changes every day. The observation $z_t$ is $A_t$, the price of asset A. The filter then runs its ordinary predict-update loop, and out comes a hedge ratio that updates daily, weighting the newest day's relationship by the Kalman gain.

This is no longer batch regression; it is *recursive* regression that never refits from scratch. Each day's data nudges the estimate, the gain controls how hard, and old data fades naturally as the random-walk uncertainty grows. It is the online, drifting cousin of the regression you would build with [least squares and the pseudo-inverse](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews).

#### Worked example: Kalman beta versus static OLS, in dollars

Let us trade a concrete pair and count the money. Asset A is a stock; asset B is its sector ETF. Over the last 120 days, OLS gives a static hedge ratio of $\beta_{\text{OLS}} = 1.20$. But the true relationship has been drifting: it was genuinely 1.20 four months ago and is genuinely about 1.55 today. The Kalman filter, tracking daily, currently reads $\beta_{\text{KF}} = 1.52$.

You want to hold a market-neutral spread with \$100,000 of asset A. For each dollar of A, you short $\beta$ dollars of B.

**Static hedge.** You short $1.20 \times \$100{,}000 = \$120{,}000$ of B. But the true ratio is 1.55, so a fully neutral hedge would short $\$155{,}000$. You are *under-hedged* by $\$35{,}000$ of B notional. That \$35,000 is unhedged exposure to the sector's direction — a naked directional bet you never intended.

Now suppose the sector drops 4% on a bad macro day. Your unhedged \$35,000 of exposure to that move costs you $0.04 \times \$35{,}000 = \$1{,}400$ of unintended loss on that single day — pure noise injected into a trade that was supposed to be market-neutral.

**Kalman hedge.** You short $1.52 \times \$100{,}000 = \$152{,}000$ of B, almost exactly the true 1.55. Your residual mis-hedge is only $\$3{,}000$ of notional, so the same 4% sector drop costs $0.04 \times \$3{,}000 = \$120$. The Kalman hedge cut the unintended loss from \$1,400 to \$120 on that day alone.

Over a quarter of drifting, with a dozen such macro days and the steady leakage of directional PnL into the spread, the static hedge's accumulated slippage easily reaches the **\$3,000** range that a tracking hedge would have avoided — and worse, it corrupts your signal, because the spread you are trading is no longer the clean mean-reverting series your strategy assumed. The intuition: a hedge ratio is a moving target, and a filter that tracks the target keeps your "market-neutral" trade actually neutral instead of slowly turning into a directional gamble.

### What this costs and when it breaks

The dynamic hedge is not free. Because $\beta$ moves daily, your hedge notional moves daily, which means *rebalancing trades* and the transaction costs they carry. Tune $Q$ too high and the filter chases noise, churning your hedge and bleeding commissions and spread for no benefit. Tune $Q$ too low and you are back to a near-static beta that lags. The art is choosing $Q$ so the beta tracks genuine drift but ignores day-to-day noise — and that is exactly the trust-dial tuning from the last section, now measured in transaction costs versus hedge accuracy.

## 4. Tracking a slowly-drifting mean return

The second great quant application is estimating a *true expected return* that drifts over time. Every strategy has some real edge — a true mean return per trade — but that edge is not constant: it decays as competitors find it, it shifts with regime, it erodes as capacity fills. You only ever see noisy realized returns, and the question is how to estimate the *current* true mean without lagging a year behind reality.

### Rolling average versus Kalman mean

The naive approach is a rolling average: average the last $N$ realized returns. But a rolling average has an ugly trade-off baked in. A short window (small $N$) reacts fast to a changing mean but is noisy. A long window (large $N$) is smooth but lags badly — it is still averaging in returns from a regime that ended months ago. Worse, a rolling average gives *equal* weight to every return in the window and *zero* weight to everything just outside it, which is an arbitrary and abrupt way to forget the past.

The Kalman filter does the same job more gracefully. Model the true mean as a random walk and each realized return as a noisy measurement of it:

$$\mu_t = \mu_{t-1} + w_t \qquad\text{(the true mean drifts)}$$
$$r_t = \mu_t + v_t \qquad\text{(the realized return)}$$

This is the same fair-value model from the foundations, reinterpreted: the hidden state is the mean return, the observation is each period's realized return, the measurement noise $R$ is the volatility of returns around the mean. The filter produces an *exponentially-weighted* estimate where recent returns matter most and old ones fade smoothly — no hard window edge — and the fade rate is set by the $Q/R$ ratio rather than an arbitrary $N$.

The timeline below tracks a latent mean return as it drifts from clearly positive, down through zero into negative, and back — and shows the filter cutting the bet as the estimate turns down.

![Timeline of a drifting latent mean return tracked by the filter](/imgs/blogs/kalman-filter-state-space-math-for-quants-7.png)

### The sizing consequence

Why does this matter for money? Because you size your positions on your estimate of the edge. If you bet bigger when you believe the mean is high and smaller when you believe it has decayed, then a *better* estimate of the mean translates directly into better sizing — bigger when the edge is real, smaller before a drawdown deepens. This is the link to the alpha-research world, where estimating and trusting an edge is the whole game; see [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) for how a tracked mean feeds position sizing.

#### Worked example: filtered mean versus rolling average and the sizing impact

A strategy's true mean return per trade was a healthy +0.10% but has recently decayed to about +0.02% as the edge crowded. Daily realized returns are noisy, with a standard deviation of 1.0% around the true mean.

**The 60-day rolling average.** Half its window still covers the old +0.10% regime, so it reports a mean of roughly +0.06% — three times the true current edge. It is lagging badly because it weights two-month-old returns just as heavily as yesterday's.

**The Kalman filter** (tuned so the gain settles around 0.10, giving an effective memory of about 20 trades) has already faded most of the old regime and reports a mean of about +0.03% — much closer to the true +0.02%.

Now the dollars. You run a sizing rule that risks \$10,000 of capital per unit of estimated edge, scaling position size linearly with the estimated mean (a stylized version of the [Kelly](/blog/trading/math-for-quants/bayesian-inference-traders-math-for-quants)-style "bet proportional to edge" rule).

- **Sized off the rolling average (+0.06%):** you allocate as if the edge were three times its real size, putting on roughly $0.06/0.02 = 3\times$ the position the true edge justifies. On \$100,000 of notional that is \$300,000 deployed where \$100,000 was warranted. If the edge keeps eroding toward zero, that oversized book takes the full brunt of the decay — a drawdown of, say, 2% on \$300,000 is a **\$6,000** loss versus \$2,000 on the right-sized book.
- **Sized off the Kalman mean (+0.03%):** you deploy close to the correct \$150,000, and the same 2% adverse move costs \$3,000. You sized to the live edge, not last quarter's.

The intuition: a rolling average tells you what your edge *was*; the Kalman mean tells you what your edge *is*, and the gap between those two numbers is exactly how much you over- or under-bet at the worst possible moment.

## 5. Smoothing a noisy signal

So far we have *filtered*: estimated the state right now using all data up to now. There is a strictly more accurate cousin called the **smoother**, and it answers a slightly different question.

A filter, running forward, can only use the past. When it estimates the fair value at noon, it has not yet seen the afternoon's prints. A *smoother* runs a second pass *backward* through the data, so its estimate of the noon fair value gets to use the afternoon's information too. The result is a smoother, more accurate reconstruction of the whole path — at the cost of being usable only after the fact, not in real time.

The before-and-after figure below shows the payoff: raw prints jump around the true value, while the filtered (and especially the smoothed) estimate is a clean line that tracks the truth with only a slight lag.

![Before after noisy prints versus smooth filtered estimate of fair value](/imgs/blogs/kalman-filter-state-space-math-for-quants-6.png)

### When to filter and when to smooth

The choice is dictated by whether you can wait:

- **Filter (forward only)** for anything you must act on *now*: live quoting, live hedging, live sizing. You only have the past, and the filter is the optimal use of it.
- **Smooth (forward then backward)** for anything you analyze *after the fact*: research, backtesting, reconstructing what the true fair value or true beta *was* at each historical moment. The smoother gives the cleanest possible historical series.

A subtle but critical trap: never use a *smoothed* series as if it were available in real time inside a backtest. The smoother's estimate of Monday's fair value secretly used Tuesday's and Wednesday's data — it peeked into the future. Backtesting a signal on smoothed values is a classic look-ahead bias that inflates results and evaporates in live trading. Filter for the live signal; smooth only for honest post-mortem analysis.

#### Worked example: filter lag versus smoother accuracy

The true fair value of an instrument took a genuine step from \$100 to \$103 over five days, but every daily print is buried in noise with a 0.80 standard deviation. You hold 5,000 shares, so each \$0.01 of estimate error is \$50 of mark error.

**The filter, on day 3 of the move** (true value \$101.50), has seen three noisy prints of the rising series and estimates \$100.90. It is *lagging* the move by about \$0.60 because it cannot yet tell whether the rise is real or a noisy blip — that is the price of using only past data. At 5,000 shares, that lag is a \$3,000 mark error on day 3.

**The smoother, looking back** at day 3 *after* the full move completed, sees that days 4 and 5 confirmed the rise to \$103. It revises its day-3 estimate up to \$101.40 — within six cents of the true \$101.50, a \$300 mark error. The smoother cut the day-3 error from \$3,000 to \$300 by borrowing future information.

The intuition: the filter is the best you can do *live*; the smoother is the best you can do *in hindsight* — and the difference between them is precisely the value of information you do not yet have, which is why you must never let the smoother's hindsight leak into a backtest.

## 6. The state-space model family: smoother, EKF, UKF, and recursive least squares

The plain Kalman filter assumes two things: the transition and measurement are *linear* (matrices $F$ and $H$), and the noise is *Gaussian*. Real problems sometimes violate both, and the family of extensions exists to handle exactly those violations. The tree below organizes the family from the one linear-Gaussian root.

![Tree of the state-space model family Kalman smoother EKF UKF](/imgs/blogs/kalman-filter-state-space-math-for-quants-5.png)

### When the world is nonlinear: EKF and UKF

Sometimes the relationship between state and observation is not a straight line. An option's price is a nonlinear function of its implied volatility; a yield is a nonlinear function of bond prices. If you tried to force a linear $H$ onto these, the filter would be biased.

The **Extended Kalman Filter (EKF)** handles this with calculus: at each step it *linearizes* the nonlinear function around the current estimate — it takes the derivative (the Jacobian) and pretends the function is a straight line *locally*. It is the duct-tape solution: simple, fast, and good enough when the nonlinearity is mild, but it can drift badly when the curvature is sharp because a tangent line is a poor stand-in for a strongly curved function.

The **Unscented Kalman Filter (UKF)** is more careful. Instead of linearizing the function, it pushes a small, cleverly chosen set of sample points (called *sigma points*) through the true nonlinear function and reconstructs the resulting mean and covariance from where they land. It captures curvature that the EKF misses, at a modest extra cost, and is usually the better choice when the nonlinearity is real. (Beyond both lies the *particle filter*, which drops the Gaussian assumption entirely and represents the state with a cloud of weighted samples — the right tool when the distribution is multi-modal, at much higher computational cost.)

### The link to recursive least squares

Here is a connection that ties this post back to plain regression. **Recursive least squares (RLS)** is an algorithm for updating a regression's coefficients one data point at a time, without refitting from scratch. It turns out that RLS is *exactly* a Kalman filter with a specific choice: the state is the vector of regression coefficients, the transition is the identity ($F = I$, coefficients do not drift), and there is no process noise ($Q = 0$).

In other words, *static* recursive regression and *dynamic* Kalman beta are the same machine with one knob turned. Set $Q = 0$ and you get RLS: the coefficients converge to the ordinary least-squares fit over all data seen so far, and they stop moving as data accumulates. Turn $Q$ above zero and the coefficients are allowed to drift, giving you the time-varying hedge ratio from Section 3. The single parameter $Q$ slides you continuously from "estimate one fixed relationship ever more precisely" to "track a relationship that is genuinely changing." Recognizing this saves you from treating them as separate tools — they are one tool with a dial.

#### Worked example: RLS convergence versus a drifting Kalman beta

You are estimating a hedge ratio between two assets whose true relationship is genuinely constant at $\beta = 1.40$, but your daily estimates are noisy.

**With $Q = 0$ (recursive least squares).** Starting from a guess of $\beta = 1.00$ with high uncertainty, the gain starts large and the estimate jumps quickly toward the truth: after 5 days it might read 1.32, after 20 days 1.38, after 100 days 1.399. The uncertainty $P$ shrinks toward zero and the gain shrinks with it, so the estimate locks onto 1.40 and stops twitching. This is the right behavior *because the truth is constant* — you want to converge and then hold.

**With $Q = 0.001$ (a small drift allowance).** The estimate also reaches ~1.40, but the gain never falls all the way to zero — it settles at a small floor — so the estimate keeps gently wobbling around 1.40 by a few thousandths, reacting to every new day. On a truly constant relationship this is slightly *worse*: you are paying for adaptability you do not need, churning the hedge by tiny amounts and burning a sliver of transaction cost.

Suppose each day's unnecessary wobble triggers a rebalance of \$500 of notional at a 2 bp round-trip cost. That is $0.0002 \times \$500 = \$0.10$ per day, or about **\$25** a year of pure waste on a relationship that never moved. The intuition: $Q$ should match reality — set it to zero when the relationship is fixed and pay nothing for adaptability you will not use; set it positive only when the relationship genuinely drifts and the tracking is worth the churn.

## 7. Tuning the filter: estimating Q and R

The filter is optimal *given* $Q$ and $R$, but in trading you almost never know them — you have to estimate them, and a mis-tuned filter is worse than no filter because it gives you false confidence. This is the part practitioners spend the most time on, so let us be concrete.

### Estimating the measurement noise R

The measurement noise $R$ is the variance of the contamination in your prints. For a fair-value filter, $R$ is the variance of the gap between the print and the true value — proxied by the variance of high-frequency price changes that you believe are noise rather than signal (microstructure jitter, bid-ask bounce). A common quick estimate: take the variance of one-tick price changes over a calm window and attribute a fraction of it to noise. If a stock's tick-to-tick changes have a variance of \$0.04 and you judge that half of that is microstructure noise, you might set $R \approx 0.02$.

### Estimating the process noise Q

The process noise $Q$ is harder because it describes how fast the *hidden* state genuinely moves, which you cannot observe directly. Three practical routes:

1. **Innovation diagnostics.** A correctly-tuned filter produces innovations $y_t$ (the surprises) that look like white noise with variance matching the predicted $S_t$. If your innovations are *autocorrelated* — surprises that keep coming in the same direction — your filter is lagging, which means $Q$ is too low; raise it. If your innovations are smaller than $S_t$ predicts, you may have $Q$ too high. Checking that the innovations behave is the single most useful tuning diagnostic.
2. **Maximum likelihood.** Because the filter spits out the likelihood of each observation as a by-product (a Gaussian centered at the prediction with variance $S_t$), you can choose $Q$ and $R$ to maximize the total likelihood of the data — a one- or two-parameter optimization that calibrates the filter to history.
3. **The $Q/R$ ratio as a single knob.** Often you fix $R$ from microstructure and then treat $Q/R$ as the one free parameter, tuning it on a validation period to balance responsiveness against churn — exactly the trust-dial trade-off from Section 2.

#### Worked example: a mis-tuned filter and its dollar cost

You are running a fair-value filter on a name whose true value genuinely drifts with process variance $Q = 0.05$, but you mistakenly set $Q = 0.0005$ — a hundred times too low. The filter believes the fair value is nearly frozen, so its gain settles very low and it barely reacts to prints.

A real \$0.50 move in fair value occurs over a day. With the correct $Q$, the filter would have tracked perhaps 80% of it by day's end, reaching +\$0.40. With your too-low $Q$, the gain is so small that the filter tracks only 20% of it, reaching +\$0.10 — lagging the truth by \$0.40.

You quote a two-sided market around your estimate. Because your estimate lags the true \$0.50 rise by \$0.40, your offer (sell price) is \$0.40 too low — and informed traders lift your stale offer all day. On 20,000 shares of such adverse fills, the lag costs $0.40 \times 20{,}000 = \$8{,}000$ of pure adverse selection: you sold cheap because your filter refused to believe the move. The intuition: too low a $Q$ does not make you "conservative," it makes you *slow*, and in a market full of faster participants slow is the most expensive thing to be.

## Common misconceptions

**"The Kalman filter predicts the future price."** No. It estimates the *present* hidden state better than any single measurement, using only past and present data. The "predict" step extrapolates one tick using the transition model, but for a random-walk fair value that prediction is simply "the same as now" — it carries no forecasting magic. If your transition model has no real predictive structure, the filter gives you a cleaner *estimate* of where things are, not a forecast of where they will go.

**"It needs lots of data to start working."** It is genuinely *online*. It starts producing estimates from the very first observation, with the early estimates appropriately uncertain (large $P$, high gain) so it learns fast. You initialize with a rough guess and a large uncertainty, and the filter sorts itself out within a handful of observations. This is a major advantage over batch regression, which needs a full window before it produces anything.

**"More gain is always better because it reacts faster."** A high gain reacts fast *and* chatters on noise. A filter with the gain pinned near 1 just echoes the noisy prints — you have built an expensive way to do nothing. The whole value is in choosing a gain (via $Q/R$) that reacts to real moves while ignoring noise. Reactivity and stability are a trade-off, not a free lunch.

**"It only works if the noise is exactly Gaussian."** Gaussian noise is the assumption under which it is provably optimal, but the filter is famously *robust*: it degrades gracefully under mild non-Gaussianity and remains the best *linear* estimator regardless. The real danger is not mild non-normality but *fat tails* — occasional huge prints (a fat-finger, a flash crash tick) that the Gaussian model treats as impossibly surprising and therefore overweights. Guarding against those (with robust filters or outlier rejection on the innovation) matters more than worrying about small departures from the bell curve.

**"A Kalman beta is just a fancy moving average of the OLS beta."** It is structurally different. A moving average weights a fixed window equally and forgets abruptly at the window edge. The Kalman beta weights data by *information content* — it leans hard on data when it is uncertain and gently when it is confident — and its memory fades smoothly, governed by $Q/R$ rather than an arbitrary window length. The two can look similar on calm data and diverge sharply when the relationship moves quickly.

**"Smoothing makes my live signal better, so I should always smooth."** Smoothing uses future data, so it is only valid for after-the-fact analysis. Using smoothed values as a live signal — or, worse, in a backtest — is look-ahead bias that will make a worthless strategy look brilliant and then fail in production. Filter live; smooth only for honest historical reconstruction.

## How it shows up in real markets

### 1. Pairs trading desks and the drifting hedge ratio

Statistical-arbitrage desks that trade hundreds of pairs cannot afford a stale hedge ratio on any of them. The standard production setup models each pair's hedge ratio as a random-walk state and runs a Kalman filter to update it daily (or intraday). When a stock's relationship to its sector ETF shifts — after an earnings surprise changes its risk profile, or an index reconstitution changes its weight — the Kalman beta tracks the shift within days, keeping the spread market-neutral. Desks that froze their betas through the 2020 volatility spike, when correlations and relationships moved violently, found their "neutral" books had quietly accumulated large directional exposures, exactly the \$1,400-per-bad-day leak from our Section 3 example, scaled across an entire book.

### 2. Apollo, the original application

The filter's first real job was navigation: NASA's Apollo program used a Kalman filter in the guidance computer to fuse noisy radar and inertial measurements into a single best estimate of the spacecraft's position and velocity. The state was the craft's position and velocity (a six-component state); the observations were noisy sensor readings; the filter blended them optimally in real time on a computer with less memory than a modern thermostat. The trading use is a direct descendant — replace "spacecraft position" with "fair value" and the math is the same. The lesson quants took: you do not need a perfect sensor, you need to combine imperfect sensors optimally.

### 3. Market-making and microprice estimation

A market maker's core problem is estimating the true mid-price from a noisy, flickering order book. Naively taking the midpoint of the best bid and offer is noisy and gameable. Many desks run a filter — sometimes an explicit Kalman filter, sometimes its close relative the exponentially-weighted estimate — to produce a smooth, lag-minimized fair value (the *microprice*) that incorporates order-book imbalance as an additional measurement. The gain governs how aggressively the quote chases the book: too high and the maker gets picked off on noise; too low and the maker quotes stale prices and gets adversely selected, exactly the \$8,000 adverse-selection cost from our Section 7 example.

### 4. Term-structure and factor models

Fixed-income and macro desks model the yield curve as a small set of hidden factors — level, slope, curvature — that evolve over time, observed through the noisy prices of many bonds. This is a textbook state-space model (the Nelson-Siegel and the dynamic factor models are explicitly cast this way), and the Kalman filter estimates the latent factors each day from the cross-section of observed yields. When the curve is illiquid and some bonds trade rarely, the filter naturally down-weights stale prints (high $R$) and leans on the model, producing a coherent curve even when the raw data is patchy. The same machinery powers latent-factor equity models that track time-varying factor exposures.

### 5. Trend-following and adaptive moving averages

Some trend-following systems replace the ordinary moving average with a Kalman filter on the price level, where the state includes both a level and a velocity (a slope). The filter's estimate of the velocity is a smoothed, lag-minimized trend signal, and its built-in uncertainty tells the system how much to trust the trend right now. The advantage over a fixed-length moving average is the same one from Section 4: the memory adapts to the data instead of being frozen at an arbitrary window, so the signal speeds up in fast markets and steadies in calm ones.

### 6. The 2010 flash crash and outlier prints

On May 6, 2010, US equity prices briefly collapsed and rebounded within minutes, with some prints hitting absurd levels (shares trading at a penny or at \$100,000). A naive filter with a pure-Gaussian measurement model would have treated those prints as real and yanked its fair-value estimate to nonsense, because a Gaussian model assigns essentially zero probability to such a large innovation and therefore is shocked into believing it. Production systems learned to reject or heavily down-weight prints whose innovation $y_t$ is many standard deviations of $S_t$ — a robustification of the update step. The episode is the canonical reminder that the Gaussian assumption is a liability precisely in the tails, where the money is made and lost.

## When this matters to you

If you ever need to estimate a number that you cannot see directly — a fair value behind noisy prints, a hedge ratio that drifts, a true edge hiding inside noisy returns — the Kalman filter is very likely the right first tool, and often the right last one. It is the optimal way to fuse a model of how something *should* move with a stream of imperfect measurements of how it *did* move, and it does so in a single, cheap, online loop you can run on every tick.

The practical mental checklist is short. Can you write your problem as a hidden state that drifts plus a noisy measurement of it? Then you have a state-space model. Do you need the estimate live, or only after the fact? Filter for live, smooth for hindsight — and never let the smoother leak into a backtest. Are the relationships linear and the noise roughly Gaussian? Then the plain filter is optimal; if not, reach for the EKF, the UKF, or a particle filter, and watch out for fat-tailed outlier prints. And above all, respect the tuning: the filter is only as good as your $Q$ and $R$, so check the innovations, calibrate by likelihood, and treat $Q/R$ as the one knob that decides whether your filter is nimble or stable.

A closing honesty note, because this is finance and not engineering: none of this is investment advice, and the filter does not manufacture an edge. It estimates a hidden quantity well; whether that quantity is worth trading is a separate question that the math cannot answer for you. A perfectly tuned filter on a relationship that has no real structure will faithfully track noise. The filter is a precision instrument — its value depends entirely on pointing it at something that is genuinely there.

### Further reading

- [Bayesian inference for traders](/blog/trading/math-for-quants/bayesian-inference-traders-math-for-quants) — the Kalman filter is Bayes' rule applied recursively to a Gaussian state; the predict-update cycle *is* prior-times-likelihood done every tick, and this post builds that foundation.
- [Building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) — where a filtered mean return feeds position sizing and signal construction in a real research workflow.
- [Linear regression deep-dive](/blog/trading/quantitative-finance/linear-regression-deep-quant-interviews) — the static OLS hedge ratio that the Kalman beta improves on, and the recursive-least-squares link from Section 6.
- [Covariance and correlation pitfalls](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews) — why the relationships the filter tracks drift in the first place, and the traps in measuring them.
- Primary sources: R. E. Kalman's 1960 paper "A New Approach to Linear Filtering and Prediction Problems," and the state-space treatment in time-series texts such as Durbin and Koopman.

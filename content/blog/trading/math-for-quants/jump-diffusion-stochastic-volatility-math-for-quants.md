---
title: "Jump-diffusion and stochastic volatility: where Black-Scholes breaks and what fixes it"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Build from zero the two great corrections to Black-Scholes — Merton's jumps and Heston's random volatility — and see exactly how each one creates the volatility smile, fat tails, gap risk, and the downside skew that the textbook model cannot, all with worked dollar examples."
tags:
  [
    "jump-diffusion",
    "stochastic-volatility",
    "merton-model",
    "heston-model",
    "volatility-smile",
    "implied-volatility",
    "black-scholes",
    "options-pricing",
    "gap-risk",
    "calibration",
    "quantitative-finance"
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Black-Scholes assumes prices wiggle smoothly with one constant volatility; real prices jump and their volatility is itself random, and those two facts are exactly what create the volatility smile, fat tails, and gap risk.
>
> - **Plain Black-Scholes is wrong in a structured way.** A constant-volatility, continuous-path model cannot make fat tails or a curved volatility smile, so the market quietly *bends* the one input it controls — implied volatility — into a smile to compensate.
> - **Merton jump-diffusion** adds sudden gaps: $dS/S = \mu\,dt + \sigma\,dW + (J-1)\,dN$, where jumps arrive as a Poisson process. The price becomes a probability-weighted *mixture* of ordinary Black-Scholes prices, which lifts the wings of the smile and prices the risk of an overnight gap.
> - **Heston stochastic volatility** makes variance itself a random, mean-reverting process: $dv = \kappa(\theta - v)\,dt + \xi\sqrt{v}\,dW_2$, correlated with the price by $\rho$. A negative $\rho$ — the *leverage effect* — fattens the downside and tilts the smile into a downward skew.
> - **Calibration** is how these models earn their keep: you tune the parameters until the model reprices today's quoted options, then use it consistently to price and hedge exotics the screen does not quote.
> - The number to remember: in the S&P 500 the at-the-money implied volatility is routinely $3$–$5$ percentage points *lower* than the implied volatility of a $25\%$-out-of-the-money put — that gap is the dollar price of crash insurance, and it is precisely what jumps and a negative $\rho$ exist to explain.

On October 19, 1987, the US stock market fell about $20\%$ in a single day. Under the Black-Scholes model — the formula that, even today, prices a large fraction of the world's options — a move that size has a probability so small it is effectively zero: you would not expect to see it once in the entire age of the universe, repeated billions of times over. And yet it happened, on a Monday, in living memory. The market noticed. Within months, the prices of out-of-the-money put options — the contracts that pay off precisely when the market crashes — were permanently more expensive than the textbook formula said they should be. That repricing never went away. It is now a fixture of every options market on earth, and it has a name: the **volatility smile**. This article is about the two pieces of mathematics that explain it.

![Before and after panels contrasting a smooth Black-Scholes price path with a jumpy real-world path that gaps down](/imgs/blogs/jump-diffusion-stochastic-volatility-math-for-quants-1.png)

The two panels above are the mental model for everything that follows. On the left is the world Black-Scholes assumes: prices drift and wiggle, but they never *gap* — you could in principle hedge continuously and never be surprised. On the right is the world we actually trade in: most days look smooth, but every so often the price jumps, opens at a level it never passed through, and the smooth-path assumption shatters. The first great correction to Black-Scholes — Merton's **jump-diffusion** — puts those gaps into the math directly. The second — Heston's **stochastic volatility** — fixes a subtler defect: the assumption that the size of the daily wiggles is a fixed constant, when in reality volatility is itself a living, random, mean-reverting quantity. Together they account for almost everything Black-Scholes gets wrong.

One honest aside before we start. Nothing here is investment advice. We are going to take apart the assumptions of the most famous formula in finance and rebuild it twice, each time to capture a real feature of markets the original ignored. Knowing *where* a pricing model is wrong is, for an options trader, often worth more than the price the model spits out.

## Foundations: the building blocks

Before we can break Black-Scholes, we need to be precise about a handful of terms a careful reader may never have met formally: what an **option** is, what **volatility** and **implied volatility** mean, what **geometric Brownian motion** is, and what it means for a path to be **continuous**. Each is simpler than its name, and each connects directly to a dollar figure.

### Options, strikes, and the two flavors

An **option** is a contract that gives you the *right, but not the obligation*, to buy or sell something at a fixed price on or before a fixed date. The fixed price is the **strike**, written $K$. The "something" — usually a stock or index — is the **underlying**, and its current price is the **spot**, written $S$. The fixed date is **expiry**, $T$.

There are two flavors. A **call** lets you *buy* at the strike: if the stock is at \$120 and you hold a call with strike \$100, you can buy at \$100 and immediately sell at \$120, pocketing \$20. A **put** lets you *sell* at the strike: if the stock falls to \$80 and you hold a put with strike \$100, you can sell at \$100 something worth \$80, pocketing \$20. A call is a bet that the price goes *up*; a put is insurance against it going *down*. An option whose strike is near the current spot is **at-the-money (ATM)**; a put with a strike well below spot, or a call with a strike well above spot, is **out-of-the-money (OTM)** — it currently has no exercise value and is pure bet on a big move.

The price you pay for the option up front is its **premium**. The whole game of options pricing is figuring out the fair premium, and that is what Black-Scholes, Merton, and Heston are all competing to do.

### Volatility: the one number that matters

**Volatility**, written $\sigma$, is the standard deviation of returns, usually quoted on an annual basis. If a stock has $\sigma = 20\%$, a "typical" annual move is about $20\%$, and — dividing by the square root of the number of trading days, roughly $\sqrt{252} \approx 16$ — a typical *daily* move is about $20\%/16 \approx 1.25\%$. Volatility is the single most important input to any option price, because an option only makes money when the underlying *moves*, and volatility is the measure of how much it moves.

There are two volatilities, and the distinction is the spine of this entire article. **Realized volatility** is what the underlying *actually did* — you measure it after the fact from the price history. **Implied volatility** is what the *option price implies* — it is the value of $\sigma$ you have to plug into Black-Scholes to make the formula's output equal the option's quoted market price. Realized vol is a fact about the past; implied vol is the market's forecast, baked into the price, of the future. When they diverge, money changes hands, and we will spend a whole section on exactly how.

### Geometric Brownian motion and continuous paths

Black-Scholes assumes the stock price follows **geometric Brownian motion (GBM)**:

$$
\frac{dS}{S} = \mu\,dt + \sigma\,dW.
$$

Read this left to right. $dS/S$ is the instantaneous *return* over a tiny slice of time $dt$. The first term, $\mu\,dt$, is the predictable **drift** — the average rate the stock earns per unit time. The second term, $\sigma\,dW$, is the random shock: $\sigma$ is the volatility, and $dW$ is the increment of a **Brownian motion** (also called a Wiener process), the mathematical idealization of a pure random walk. The key property of $dW$ is that it is *continuous*: over any tiny interval the price changes by a tiny, random, normally-distributed amount, and it never teleports. If you zoom in on a GBM path, it is jagged but unbroken — you can trace it without lifting your pen.

That last property — **continuity** — is the assumption we are going to attack first. Continuity is convenient: it is what lets Black-Scholes derive a perfect, riskless hedge by continuously rebalancing. But it is also false. Stocks close at one price and open the next morning at another, with no trading in between. Earnings come out after the bell. A central bank surprises everyone at 2 p.m. The price *gaps*, and no amount of continuous rebalancing can protect you from a move you never had the chance to trade through.

If you want the full derivation of GBM, the Itô calculus behind that $dW$, and the change of measure that makes Black-Scholes a fair-pricing formula, the companion posts on [stochastic differential equations](/blog/trading/quantitative-finance/stochastic-differential-equations-gbm-ou-quant-interviews) and [risk-neutral pricing](/blog/trading/quantitative-finance/risk-neutral-pricing-martingale-measure-quant-interviews) build them from the ground up. Here we take GBM as given and ask: what does it get wrong, and how do we fix it?

## Why plain Black-Scholes is wrong

Black-Scholes is not a *bad* model — it is the foundation everything else builds on, and it is exactly right in a world that does not quite exist. Its two load-bearing assumptions are the two we will replace:

1. **Volatility is constant.** The same $\sigma$ governs every day, every strike, every expiry.
2. **Paths are continuous.** Prices never gap; you can always hedge by trading at the current price.

From these two assumptions, the model makes a sharp, falsifiable prediction: the distribution of the log of the stock price at expiry is exactly **normal** (a bell curve), which means the stock price itself is **lognormal**. A normal distribution has thin tails — the probability of a move falls off extremely fast as the move gets larger. A 5-standard-deviation day has a probability of about 1 in 3.5 million; a 10-standard-deviation day, like October 1987, is so unlikely the number is not worth writing down.

But real return distributions are not thin-tailed. They are **leptokurtic** — a fancy word meaning "fat-tailed and sharp-peaked." Big moves happen far more often than a bell curve allows. Daily S&P 500 returns have a *kurtosis* — a measure of tail-fatness, where the normal distribution scores $3$ — that is routinely above $10$, and over a crisis window it can exceed $30$. The companion post on [ARCH and GARCH volatility models](/blog/trading/math-for-quants/arch-garch-volatility-math-for-quants) shows how time-varying volatility alone produces much of this fat-tailedness; here we will see that jumps add the rest, and that random volatility adds it in a different and complementary way.

There is a second, even more visible piece of evidence. If Black-Scholes were right, then when you back out the implied volatility from the prices of options at *different strikes but the same expiry*, you would get the *same number every time* — because the model says there is only one $\sigma$. Instead, you get a curve.

![Before and after panels showing flat Black-Scholes implied volatility versus a curved market volatility smile across strikes](/imgs/blogs/jump-diffusion-stochastic-volatility-math-for-quants-2.png)

The two panels above show exactly this. On the left is the Black-Scholes prediction: implied volatility is flat across strikes, one number for all of them. On the right is what every options market on earth actually shows: implied volatility is *higher* for OTM options — especially OTM puts — and *lowest* near the money, tracing out a curve. In equity indices the curve is lopsided, higher on the downside (puts) than the upside (calls), and the lopsided version is called the **skew** or **smirk**. The symmetric version, common in currencies, is the **smile**. Either way, the existence of the curve is a flat contradiction of constant volatility. The market is telling us, in the only language it has — prices — that big down-moves are more likely than a bell curve allows. Jumps and stochastic volatility are the two coherent stories that explain why.

It is worth being precise about why this is *evidence* and not just a quirk. Implied volatility is defined by inverting the Black-Scholes formula: given a quoted option price, it is the unique $\sigma$ that makes the formula reproduce that price. The formula is monotonic in $\sigma$ — a higher $\sigma$ always gives a higher option price — so the inversion is unambiguous; every quoted option has exactly one implied vol. If the world truly were Black-Scholes, every option on the same underlying and expiry would invert to the *identical* number, because there would be one true $\sigma$ generating all of them. The fact that they invert to *different* numbers, in a stable and repeatable curve, is direct proof that the price-generating mechanism is not the one Black-Scholes assumes. The shape of the disagreement is not random noise — it is the same skew, day after day, year after year — which means it encodes a structural feature of the real distribution. Our job is to name that feature, and there are exactly two candidates that survive scrutiny: the distribution has jumps, and its volatility is random and downside-correlated.

A third piece of evidence, less famous but just as damning, comes from the *autocorrelation of squared returns*. Under GBM, returns are independent across days, so squared returns — a proxy for daily variance — should show no pattern over time. In reality, squared returns are strongly *positively* autocorrelated: a big-magnitude day is followed by more big-magnitude days. This **volatility clustering** is the time-series shadow of the same fact, and it is the central subject of the [ARCH and GARCH](/blog/trading/math-for-quants/arch-garch-volatility-math-for-quants) models. Heston is, in a sense, the continuous-time option-pricing translation of GARCH: both say volatility is a persistent, mean-reverting random process rather than a constant.

## Merton jump-diffusion

The first fix is the most direct: if prices gap, *put gaps in the equation*. In 1976 Robert Merton — who had already co-authored the option-pricing framework — added a jump term to GBM. The model is:

$$
\frac{dS}{S} = (\mu - \lambda k)\,dt + \sigma\,dW + (J-1)\,dN.
$$

Let us read every piece, because each one is a real market feature.

- $\sigma\,dW$ is the ordinary diffusion — the day-to-day wiggling, exactly as in GBM. Most of the time, this is all that happens.
- $dN$ is the increment of a **Poisson process** with intensity (arrival rate) $\lambda$. A Poisson process is the mathematics of *rare events that arrive at random*: $dN = 1$ when a jump occurs in this instant (probability $\lambda\,dt$), and $dN = 0$ otherwise. The parameter $\lambda$ is the *expected number of jumps per year*. If $\lambda = 1$, you expect roughly one jump a year; if $\lambda = 0.25$, one every four years.
- $(J-1)$ is the *size* of the jump when one occurs. $J$ is the multiplicative factor: if a jump hits and $J = 0.9$, the price is instantly multiplied by $0.9$ — a $10\%$ gap down. $J$ is itself random, usually drawn from a lognormal distribution, so jumps come in a range of sizes.
- $\lambda k$ in the drift, where $k = \mathbb{E}[J-1]$ is the average jump size, is a small **compensator** that corrects the drift so the average growth rate stays equal to $\mu$. Without it, adding jumps would secretly change the stock's expected return; the compensator keeps the books honest. You can ignore it for intuition — it is a bookkeeping term.

So the picture is: the stock diffuses smoothly, and *on top of that*, at random Poisson times, it gaps by a random multiplicative amount. Most of the realism comes from a small number of large negative jumps.

### How jumps make fat tails and the smile

Here is the beautiful part, and the reason Merton's model is tractable. Conditional on knowing *exactly how many jumps happened* between now and expiry — say there were $n$ of them — the stock at expiry is *still lognormal*, just with a shifted mean and a different variance, because $n$ lognormal jumps multiplied together is itself lognormal. And the number of jumps $n$ is Poisson-distributed. So the option price under Merton is a **Poisson-weighted average of ordinary Black-Scholes prices**:

$$
C_{\text{Merton}} = \sum_{n=0}^{\infty} \frac{e^{-\lambda' T}(\lambda' T)^n}{n!}\; C_{\text{BS}}(S,\,K,\,T,\,r_n,\,\sigma_n).
$$

Read the structure, not the subscripts. The fraction out front is the **Poisson probability** of exactly $n$ jumps occurring over the life of the option ($\lambda'$ is the jump rate adjusted for the average jump size). For each possible jump count $n$, you compute a *standard Black-Scholes price* but with the volatility and rate nudged to account for those $n$ jumps ($\sigma_n$ is a little larger, reflecting the extra variance the jumps add). Then you average all those Black-Scholes prices, weighting each by how likely that many jumps is. The whole infinite sum converges fast — in practice the first five or six terms are plenty, because the probability of many jumps in one option's life is tiny.

This mixture is *exactly* what produces fat tails. A bell curve has thin tails; a *mixture* of bell curves — some with the ordinary variance, some with the jump-inflated variance — has fat tails, because the high-variance components put extra probability mass far from the center. And that extra tail mass is worth the most, in relative terms, to the options that only pay off in the tail: the deep OTM puts and calls. So jump-diffusion lifts the *wings* of the smile, exactly where Black-Scholes is most wrong, while barely touching the ATM price.

It is worth dwelling on *why* the Poisson process is the right tool for rare gaps, because the choice is not arbitrary. A Poisson process is the unique counting process with three properties that match how surprises actually arrive: jumps in non-overlapping time windows are *independent* (a gap this morning tells you nothing about whether one comes this afternoon), the rate is *constant* (a jump is no more likely at any one instant than another, absent a known event like earnings), and *two jumps never occur at exactly the same instant*. Those three assumptions force the number of jumps in a window of length $T$ to be Poisson-distributed with mean $\lambda T$, and the *time between jumps* to be exponentially distributed with mean $1/\lambda$. So when you set $\lambda = 2$, you are not just saying "two jumps a year on average" — you are committing to the full statement that gaps arrive memorylessly, with an average gap-free stretch of six months, and that the chance of a gap in the next instant is always the same tiny $\lambda\,dt$. That memorylessness is both the model's great convenience and its honest limitation: real jumps *do* cluster (a crash is often followed by aftershocks), which is one reason production models bolt jumps onto a *stochastic-volatility* engine that can stay elevated after a shock.

![Pipeline from picking a model through calibrating it to market quotes to reproducing the volatility smile](/imgs/blogs/jump-diffusion-stochastic-volatility-math-for-quants-3.png)

The pipeline above is the workflow we will follow for both models: pick a richer model, calibrate its parameters to the option prices the market actually quotes, and check that the calibrated model reproduces the smile that flat Black-Scholes cannot. Merton's model has a handful of free knobs — the diffusion vol $\sigma$, the jump rate $\lambda$, and the mean and spread of the jump size — and turning those knobs lets you fit the wings of a real smile remarkably well, especially for short-dated options where a single big gap dominates the risk.

#### Worked example: expected jumps and the dollar cost of a gap

Suppose you are short (you have sold) $100$ shares' worth of a stock trading at \$50, so your position is worth \$5,000, and you have decided not to hedge overnight. You model jumps with a Poisson rate of $\lambda = 2$ per year — two surprise gaps a year on average — and you hold the position for one month, a horizon of $T = 1/12$ years.

The **expected number of jumps** over your horizon is $\lambda T = 2 \times \tfrac{1}{12} = 0.167$. So on average you expect about $0.17$ jumps in a month. The probability of *at least one* jump is $1 - e^{-\lambda T} = 1 - e^{-0.167} = 1 - 0.846 = 0.154$, or about $15\%$. Roughly a one-in-seven chance of a gap in any given month.

Now the dollar impact. Suppose the typical jump, when it comes, is a $10\%$ gap *down* — bad news for the underlying, but you are *short*, so a down-move is a *gain* for you of $10\% \times \$5{,}000 = \$500$. But jumps cut both ways: a $10\%$ gap *up* is a \$500 *loss*. Because you are short, the up-jump is the one that hurts. If up-jumps and down-jumps were equally likely, your expected jump PnL is roughly zero, but the *variance* it adds to your month is enormous: a single \$500 swing on a \$5,000 position is a $10\%$ move in your equity from one event you cannot hedge against once it happens. The intuition: jumps do not change your *average* outcome much, but they create gap risk — a fat tail of sudden \$500 surprises — that no amount of careful daily hedging can remove.

#### Worked example: how jumps lift the OTM put price

Let us see jumps lift the wings of the smile in dollars. Take a one-month put on a \$100 stock, struck at \$90 — a $10\%$ OTM put, the kind of crash insurance the smile is all about. Under plain Black-Scholes with a flat volatility of $\sigma = 20\%$ and a zero interest rate, this put is cheap, because a $10\%$ fall in one month is more than two standard deviations and the bell curve says that is rare. A standard Black-Scholes calculation gives a premium of about \$0.20 — twenty cents to insure against a $10\%$+ crash for a month.

Now switch on jumps. Add a Poisson jump component with $\lambda = 1$ per year and an average jump of $-15\%$ when one occurs. The probability of a jump in the one-month window is $1 - e^{-1/12} \approx 8\%$, and *when* a jump hits, a $-15\%$ move blows straight through the \$90 strike, making the put pay off handsomely. That $8\%$ chance of a large, in-the-money payoff is worth real money. Re-pricing the put as the Poisson-weighted mixture of Black-Scholes prices raises the premium to roughly \$0.55 — nearly *three times* the Black-Scholes value. To express that \$0.55 price as a Black-Scholes implied volatility, you would have to plug in about $\sigma = 33\%$, not $20\%$. The intuition: jumps make crashes far likelier than a bell curve allows, the OTM put is the contract that profits from crashes, so its price — and its implied volatility — jumps up, which is the left wing of the smile appearing right before your eyes.

## Stochastic volatility: the Heston model

Jumps fix the *gap* problem and the *short-dated* wings of the smile. But they leave the second false assumption untouched: that volatility is a constant. It is not. Anyone who watched markets in March 2020 knows that volatility itself moves — it spikes in crises and subsides in calm, and it does so in a *persistent*, *mean-reverting* way that the [GARCH family](/blog/trading/math-for-quants/arch-garch-volatility-math-for-quants) was built to forecast. The second great fix is to make volatility a random process of its own. The most famous version is the **Heston model** (Steven Heston, 1993):

$$
dS = \mu S\,dt + \sqrt{v}\,S\,dW_1, \qquad dv = \kappa(\theta - v)\,dt + \xi\sqrt{v}\,dW_2,
$$

with the two random drivers correlated: $\mathrm{corr}(dW_1, dW_2) = \rho$.

There is a lot here, so let us take it apart one layer at a time. Notice the first equation is just GBM, except the volatility is now $\sqrt{v}$ where $v$ is the **variance** — and $v$ has its own equation. The second equation is where the new physics lives.

![Stack of the five Heston components: random price, random variance, mean reversion, vol-of-vol, and the correlation](/imgs/blogs/jump-diffusion-stochastic-volatility-math-for-quants-5.png)

The stack above lists the five pieces, and we will define each:

- **The price $S$ is driven by $\sqrt{v}$.** Instead of a fixed $\sigma$, the instantaneous volatility is $\sqrt{v_t}$, which changes every instant. When variance is high, the stock wiggles violently; when it is low, the stock is calm.
- **The variance $v$ is itself random.** It has its own Brownian shock $\xi\sqrt{v}\,dW_2$, so volatility is no longer a number you set once — it is a quantity that drifts and jumps around on its own.
- **Variance mean-reverts to $\theta$ at speed $\kappa$.** The drift term $\kappa(\theta - v)$ is a spring: whenever $v$ is above its long-run level $\theta$, the term is negative and pulls it back down; whenever $v$ is below $\theta$, the term is positive and pushes it back up. $\theta$ is the **long-run variance** (where vol settles in the long run), and $\kappa$ is the **speed of mean reversion** (how fast it gets back there — a larger $\kappa$ means volatility shocks die out faster).
- **$\xi$ is the volatility of volatility — "vol-of-vol."** It controls how *jumpy* the variance itself is. A large $\xi$ means variance swings around wildly, which spreads out the distribution of possible volatilities, which fattens the tails of returns and lifts *both* wings of the smile — it controls the smile's **curvature**.
- **$\rho$ is the correlation between price shocks and variance shocks.** This is the single most important parameter for the *shape* of the equity smile, and it deserves its own section.

The $\sqrt{v}$ in the variance equation is not decoration: it is the **CIR process** (Cox-Ingersoll-Ross), the same square-root diffusion used for [short-rate models](/blog/trading/quantitative-finance/short-rate-models-vasicek-hull-white). The square root guarantees that as variance approaches zero the random shocks shrink to zero too, so — under a mild condition called the **Feller condition**, $2\kappa\theta \ge \xi^2$ — variance can never go negative. A negative variance would be nonsense (you cannot have a negative standard deviation squared), and the square root is what protects against it.

The five parameters are not interchangeable knobs — each one controls a *distinct, visible* feature of the smile, which is precisely what makes Heston calibratable. Raising the long-run variance $\theta$ lifts the *whole* smile up (everything gets more expensive). Raising the mean-reversion speed $\kappa$ makes the smile *flatten faster as maturity grows*, because volatility is yanked back to $\theta$ quickly and has less room to wander on long-dated options. Raising the vol-of-vol $\xi$ deepens the smile's *curvature* — it makes both wings steeper, because a wide spread of possible future volatilities puts more mass in both tails. The starting variance $v_0$ sets where the *short end* of the term structure begins, since over a tiny horizon volatility has had no time to move away from its current value. And $\rho$, as we will see next, controls the *tilt*. Because each parameter maps to a separable feature — level, term-structure slope, curvature, short-end anchor, and tilt — a calibration routine can usually find a clean fit without the parameters fighting each other, which is a large part of why Heston, despite being decades old, remains a workhorse. A crucial subtlety: Heston has a *semi-closed-form* price for European options via a Fourier transform of its characteristic function, so calibration does not require slow Monte Carlo for the vanilla grid — you can reprice thousands of options per second, which is what makes fitting the surface in real time practical.

### The leverage effect and the skew

Now the crown jewel: the correlation $\rho$. In equity markets $\rho$ is strongly *negative* — typically around $-0.7$. What does a negative $\rho$ mean? It means that when the price shock $dW_1$ is negative (the stock falls), the variance shock $dW_2$ tends to be positive (volatility rises). In plain English: **when stocks fall, volatility spikes.** This is one of the most robust facts in all of finance, and it has a name — the **leverage effect** — from the old story that a falling stock raises a firm's debt-to-equity ratio and so makes its equity riskier. Whether or not the leverage story is the true cause, the empirical pattern is ironclad: down-moves and volatility-spikes go together.

![Before and after panels contrasting a symmetric return distribution with a left-skewed distribution produced by negative correlation](/imgs/blogs/jump-diffusion-stochastic-volatility-math-for-quants-6.png)

The two panels above show what $\rho < 0$ does to the distribution of returns. On the left, with $\rho = 0$, the distribution is symmetric: downside and upside moves are equally likely, and puts and calls are priced as mirror images. On the right, with $\rho < 0$, the mechanism kicks in: as the price falls, volatility rises, so further falls become *more* volatile and reach *further* — the left tail fattens and stretches. Meanwhile, as the price rises, volatility tends to fall, so rallies become calmer and reach less far — the right tail thins. The result is a **left-skewed** distribution: a fat, long downside tail and a thin, short upside tail. That asymmetry is the whole reason the equity smile is a *skew* — lopsided, expensive on the put side — rather than a symmetric smile. A negative $\rho$ richens OTM puts relative to OTM calls, which is precisely the shape the market quotes every day.

This is the deepest answer to the question we opened with. The market's permanent overpricing of crash puts after 1987 is not irrationality; it is the price of a real, measurable feature of the world — that crashes and volatility-spikes are the same event — encoded in the one parameter $\rho$.

#### Worked example: a negative rho and the richer put

Let us put numbers on the skew. Take a one-year ATM-region option market on a \$100 index, and suppose a Heston model is calibrated with long-run vol $\sqrt{\theta} = 20\%$, mean-reversion speed $\kappa = 2$, vol-of-vol $\xi = 0.3$, and starting volatility also $20\%$. First set $\rho = 0$ (no leverage effect). The model is roughly symmetric, and the implied volatility of a $90$-strike put (a $10\%$ OTM put) comes out near the ATM level — say about $20.5\%$ — pricing the put at roughly \$3.10.

Now flip the correlation to $\rho = -0.7$, the realistic equity value, and *keep everything else fixed*. The negative correlation fattens the downside tail, so the $90$-strike put now has a meaningfully higher chance of paying off, and its implied volatility rises to about $24\%$. Re-pricing the put at $24\%$ instead of $20.5\%$ lifts its premium from \$3.10 to roughly \$4.05 — almost a *dollar more*, an increase of about $30\%$, on the put alone. Meanwhile the symmetric-side call gets slightly *cheaper*, because the upside tail thinned. The intuition: the single number $\rho = -0.7$ is what makes downside insurance expensive and upside lottery tickets cheap, and that asymmetry is worth nearly a dollar per contract on a \$100 index — the dollar shape of the skew.

## The volatility smile and skew

We have now met the smile from two directions. Jumps lift its wings (especially short-dated); a negative $\rho$ tilts it into a skew (especially longer-dated). It is worth pulling back and treating the smile itself as the central object, because for a practicing options trader the smile, not the model, is what they look at all day.

The cleanest way to think about the smile is this: **implied volatility is the market's translation device.** Black-Scholes is wrong, but it is a *universal language* — every option can be quoted as "the $\sigma$ you'd plug into Black-Scholes to get this price." Because the true distribution has fatter tails and a fatter downside than Black-Scholes assumes, the market has to quote a *higher* $\sigma$ for the options that the fat tails make more valuable (the OTM ones) and an even higher one for the OTM puts that the downside fattening makes most valuable. The smile is not a model; it is the *fingerprint* the true distribution leaves on a model we all agree is wrong. The full geometry of that fingerprint across strikes *and* expiries — the [volatility surface](/blog/trading/quantitative-finance/volatility-surface) — is the central data object of the entire derivatives business.

There is a precise relationship hiding here, worth stating because it makes the smile concrete. The famous **Breeden-Litzenberger** result says that the *curvature* of option prices across strikes is, up to a discount factor, the **risk-neutral probability density** of the stock at expiry:

$$
\frac{\partial^2 C}{\partial K^2} = e^{-rT} f_Q(K).
$$

In words: take the prices of calls at every strike, take the second derivative with respect to strike, and you recover the entire probability distribution the market is using to price. The fatter the smile's wings, the fatter the recovered density's tails. The steeper the skew, the more left-skewed the density. The smile *is* the market's distribution, written in the only ink it has — prices. Jumps and stochastic vol are two competing stories about *why* that distribution has the shape it does.

### Term structure: why the smile flattens with maturity

One more feature the two models split between them. Real smiles are *steepest for short maturities* and *flatter for long maturities*. A one-week smile can be a dramatic V; a two-year smile is a gentle tilt. Why?

Jumps explain the short end. Over a single week, the only way to get a big move is a *jump* — diffusion does not have time to wander far — so the short-dated smile is dominated by jump risk and is sharply curved. But over two years, by the central limit theorem, the *sum* of many small diffusion moves and a few jumps starts to look more normal (jumps average out), so the long-dated smile flattens — which is exactly why pure jump models struggle to fit the long end.

Stochastic volatility explains the long end. Random, mean-reverting volatility takes *time* to bend the distribution: over one week there is not enough time for volatility to wander far from where it started, so a pure stoch-vol model produces almost no short-dated smile. But over two years volatility has plenty of time to swing around, and the $\rho < 0$ leverage effect has time to build a substantial skew. The two mechanisms are complementary, which is why the most realistic models — like the **Bates** model — use *both* jumps and stochastic volatility together.

![Matrix comparing geometric Brownian motion, Merton jump-diffusion, and Heston across volatility, jumps, fat tails, and smile-making](/imgs/blogs/jump-diffusion-stochastic-volatility-math-for-quants-4.png)

The matrix above is the scorecard. Plain GBM (Black-Scholes) has constant vol, no jumps, thin tails, and makes no smile. Merton adds jumps and so produces fat tails and a strong *short-dated* smile, but its volatility is still constant, so it fits the long end poorly. Heston adds random, mean-reverting volatility and produces fat tails and a strong *long-dated* skew, but with no jumps it underprices the very short-dated wings. Each model cures one defect; the smile, in full, demands both.

## Why these models exist: hedging gap and vol-of-vol risk

It is tempting to think richer models exist just to fit a prettier curve. They do not. They exist because Black-Scholes makes a *hedging* promise it cannot keep, and the broken promise costs real money. Understanding the broken promise is the deepest reason to care about jumps and stochastic volatility.

Black-Scholes' central claim is that an option can be *perfectly* replicated and therefore *perfectly* hedged by continuously trading the underlying — the famous **delta hedge**. The delta, written $\Delta$, is how much the option price moves per \$1 move in the underlying; you hold $-\Delta$ shares against a long option and rebalance as the price drifts, and in the Black-Scholes world this leaves you with zero risk and a riskless return. The entire formula is *derived* from this perfect-replication argument. If the hedge is perfect, the option's fair price is forced.

Jumps shatter the delta hedge. A delta hedge protects you against *small* moves — it is a first-order approximation, valid only for the tiny continuous wiggles GBM allows. When the price *gaps* — teleports from \$100 to \$85 with no trading in between — your delta hedge was calibrated for the \$100 neighborhood and is suddenly, catastrophically wrong for the \$85 reality. You had no chance to rebalance through the gap. This residual is **gap risk**, and it is *fundamentally unhedgeable* with the underlying alone, because hedging requires being able to trade at every price you pass through, and a jump skips prices entirely. The only defense is to hold *other options* — to buy a deep OTM put that pays off precisely in the gap — and the price of that defense is exactly the smile's wing. So the model and the hedge are two sides of one coin: the jump model exists to *price* the gap risk that the jump makes impossible to hedge away.

Stochastic volatility breaks a different promise. Even with no jumps, if volatility is random, then an option's value depends on a quantity — the level of volatility — that you have *not* hedged. A delta-hedged option position is still exposed to **vega**, the sensitivity to volatility; and if volatility is itself random and *mean-reverting*, you are exposed to its whole future path, including its **vol-of-vol** $\xi$. You cannot hedge vega by trading the underlying — the underlying has no volatility exposure — so you must hedge it by trading *other options* (which do have vega), and you must hold a portfolio whose net vega is zero. But even a vega-neutral book is exposed to the *second-order* volatility risks: **vanna** (how delta changes as volatility moves) and **volga** (how vega changes as volatility moves), and these are governed precisely by $\rho$ and $\xi$. A stochastic-volatility model exists to *quantify and price* those second-order exposures so a desk can hedge them with the right mix of options across strikes.

This is the honest reason the models exist. Black-Scholes prices an option as if its risk could be fully hedged away; jumps and stochastic volatility admit that some risk *cannot* be hedged away — gap risk and vol-of-vol risk — and they put a *price* on that residual. The smile is the dollar value of unhedgeable risk.

#### Worked example: the unhedgeable gap loss on a delta-hedged book

You are long $1{,}000$ shares of a \$100 stock as a delta hedge against a short call position, and your book is delta-neutral at the spot of \$100 — meaning small moves around \$100 leave you flat. Overnight, on bad news, the stock gaps *down* to \$85 before you can trade. Your long shares lose $1{,}000 \times \$15 = \$15{,}000$. Your short call, now far OTM, gains value back to you — but only by, say, \$9{,}000, because the call's delta was nowhere near $1$ and it could not recover the full share loss. Your *net* hit from a position that was supposedly hedged is about $\$15{,}000 - \$9{,}000 = \$6{,}000$, lost in a single gap you had no opportunity to trade through. Had you instead been *long* a deep OTM put as gap insurance — paying the smile's wing premium up front — that put would have exploded in value on the gap and offset much of the \$6,000. The intuition: a delta hedge is a promise that only holds for moves you can trade through, and the entire premium in the smile's wing is the market's price for the gap losses that promise cannot cover.

## Implied vol versus realized vol

We have been treating implied volatility as a passive translation device. But for a trader it is an *asset* — something you can be long or short, and something that has a fair value relative to what actually happens. The gap between **implied** vol (what you pay, baked into the option premium) and **realized** vol (what the underlying actually does over the option's life) is one of the most important spreads in all of trading.

Here is the core mechanic. When you *sell* an option and **delta-hedge** it — continuously trade the underlying to neutralize your exposure to small price moves — your profit and loss over the option's life is governed by a single, beautiful relationship. You collected the premium, which was priced at the *implied* volatility. You pay out, through the costs of hedging, an amount governed by the *realized* volatility. To a good approximation, the PnL of a continuously delta-hedged short option position is:

$$
\text{PnL} \approx \tfrac{1}{2}\,\Gamma\,S^2\,\big(\sigma_{\text{impl}}^2 - \sigma_{\text{real}}^2\big)\,T,
$$

where $\Gamma$ (gamma) measures how fast the option's delta changes — its curvature. Read the sign of the bracket. If you *sold* the option (you are short), you *want* implied volatility to have been higher than realized: you got paid at the high implied vol, and the world delivered only the lower realized vol, so you keep the difference. If realized comes in *higher* than implied, the bracket flips sign and you *lose*. Selling options is, at its heart, selling insurance: you collect a premium priced at implied vol and you win if the actual turbulence (realized vol) comes in below what you charged.

#### Worked example: a delta-hedged short option that loses money

You sell a one-month ATM straddle — a call plus a put, both struck at the \$100 spot — and collect a premium priced at an implied volatility of $20\%$ (annualized). You delta-hedge it diligently every day, neutralizing your exposure to small moves, so the only thing that matters to your PnL is the gap between the implied vol you sold and the realized vol that shows up.

For the first three weeks, the market is calm: realized volatility runs at about $12\%$, comfortably below the $20\%$ you charged. You are winning — collecting the premium decay while paying out little in hedging costs. On a position with, say, \$200 of gamma exposure per percentage-point of variance, three weeks of $12\%$ realized against $20\%$ implied earns you roughly \$300.

Then, in the final week, an earnings surprise hits and the stock starts swinging $4\%$ a day. Realized volatility over that final week spikes to $55\%$ — far above the $20\%$ you sold. Now the bracket $(\sigma_{\text{impl}}^2 - \sigma_{\text{real}}^2)$ is sharply negative, and because gamma is *highest* for an ATM option near expiry, the losses pour in fast. That one week of $55\%$ realized against $20\%$ implied costs you roughly \$900. You end the month down about \$600 on a position that was comfortably profitable for three of its four weeks. The intuition: a short option is short realized volatility, and one violent week — exactly the kind that jumps and volatility-spikes produce — can erase a month of patient premium collection, which is the dollar reason crash insurance is priced so dearly.

## Calibration: fitting the model to the market

A model with free parameters is worthless until you *set* the parameters, and you do not set them by guessing. You **calibrate**: you choose the parameters so the model reproduces the prices of options the market *already quotes*, and then you trust the calibrated model to price the options it *does not* quote — the exotics, the bespoke structures, the strikes and maturities that are not on the screen. The companion post on [maximum likelihood and method of moments](/blog/trading/math-for-quants/mle-method-of-moments-math-for-quants) covers the estimation machinery; here is the options-specific version.

Concretely, you take the grid of liquid option prices — every quoted strike and expiry — and you solve a least-squares problem: find the model parameters $\Theta$ that minimize the total squared gap between model prices and market prices,

$$
\min_{\Theta}\ \sum_{i}\, w_i\,\big(C^{\text{model}}_i(\Theta) - C^{\text{market}}_i\big)^2,
$$

where the sum runs over every quoted option $i$, $w_i$ is a weight (often $1/\text{vega}^2$, so the fit is on implied vols rather than dollar prices), and $\Theta$ is the parameter vector — for Heston that is $(v_0, \kappa, \theta, \xi, \rho)$; for Merton it is $(\sigma, \lambda, \text{jump mean}, \text{jump vol})$. You feed this to a numerical optimizer (the [optimizers post](/blog/trading/math-for-quants/stochastic-gradient-optimizers-math-for-quants) covers how those work) and out come the parameters that best fit today's smile.

Why does calibration matter so much? Because an options book lives or dies on **consistency**. Suppose you have sold an exotic — say a barrier option that knocks out if the stock touches \$80 — and you want to hedge it with vanilla puts and calls. If your model is calibrated to the vanilla smile, then the exotic's price and its hedge ratios are *consistent* with the prices you can actually trade the hedges at. If your model is *not* calibrated — if it says the vanillas are worth something different from their market price — then your hedge is built on fiction, and you will bleed money every time you rebalance. Calibration is what ties the model to reality.

#### Worked example: calibrating the jump rate to a quoted smile

You are pricing a one-month OTM put book on a \$100 index and you observe the market smile directly: the ATM implied vol is $20\%$, but the $90$-strike (a $10\%$ OTM put) trades at an implied vol of $30\%$. Plain Black-Scholes cannot fit both numbers with one $\sigma$ — that is the whole problem. So you reach for Merton and calibrate the jump parameters.

You fix the diffusion vol at $\sigma = 20\%$ so the ATM price matches, and then you turn the jump knobs to lift the $90$-strike to $30\%$ implied. Trying a jump rate of $\lambda = 1$ per year with an average jump of $-10\%$, you re-price the $90$ put as the Poisson mixture and back out its implied vol — it comes to about $26\%$, not enough. You increase the average jump size to $-15\%$ and the rate to $\lambda = 1.5$; now the $90$ put prices to an implied vol of about $30\%$ — a match. The dollar consequence: at $30\%$ implied the $90$ put is worth about \$0.45, versus the \$0.20 a naive $20\%$-flat Black-Scholes would have charged — so a desk pricing off flat Black-Scholes would *sell* this crash insurance for $55\%$ of its fair value and lose the difference, on average, every time a jump arrives. The intuition: calibration is what converts a quoted smile into model parameters you can trust, so you stop selling crash insurance at a discount.

## Common misconceptions

Even people who use these models daily carry a few wrong beliefs. Here are the most common, each corrected.

**"Black-Scholes is obsolete — nobody uses it anymore."** False, and it misunderstands what Black-Scholes *is* today. Almost no one believes its assumptions, but everyone uses it as the *quoting language*: option prices are universally quoted as implied volatilities, and "implied vol" *means* "the Black-Scholes $\sigma$." The smile is literally a chart of Black-Scholes implied vols. Black-Scholes is not the model of reality; it is the coordinate system in which the real models (Merton, Heston, and their cousins) report their answers.

**"The volatility smile means the market is irrational."** False. The smile is the market being *more* rational than Black-Scholes. It is the price-encoded statement that the true return distribution has fatter tails and a fatter downside than a bell curve, which is empirically correct. A flat smile — the Black-Scholes prediction — would be the irrational thing, because it would mean the market believed crashes were as rare as a thin-tailed bell curve says, which October 1987 decisively refuted.

**"Jumps and stochastic volatility are two names for the same thing."** False, and the difference is practical. Jumps are *discontinuous* — the price teleports, and no continuous hedge can protect you, which is why jumps create *unhedgeable* gap risk. Stochastic volatility is *continuous* — the price never teleports, it just gets more or less wiggly — so in principle you can hedge it by trading volatility instruments. They also live at different ends of the term structure: jumps dominate the short-dated smile, stochastic vol the long-dated skew. The most realistic models use both.

**"A negative correlation $\rho$ just means stocks and volatility move opposite — it's a minor detail."** False; $\rho$ is the single most important parameter for the *shape* of the equity smile. It is what turns a symmetric smile into the lopsided, put-heavy skew that defines equity index options. Set $\rho = 0$ and Heston produces a roughly symmetric smile that looks nothing like the real S&P 500. The skew *is* $\rho$.

**"If I calibrate my model perfectly to today's smile, my prices are correct."** Dangerously false. Calibration guarantees your model reprices *today's quoted options*; it says nothing about whether the model's *dynamics* — how the smile will move tomorrow — are right. Two models can fit today's smile identically and disagree wildly about the price of a forward-starting or barrier option, because those depend on how volatility *evolves*, not just where it is today. A perfect static fit with wrong dynamics is one of the classic ways to blow up a structured-products desk.

## How it shows up in real markets

The mathematics above is not academic. Each piece has left fingerprints all over market history.

### The 1987 crash and the birth of the skew

Before October 1987, equity index implied volatilities were roughly flat across strikes — the market more or less believed Black-Scholes. The $20\%$ single-day crash was a textbook *jump*: the market gapped through dozens of strikes with no chance to trade in between, and the people short OTM puts were annihilated. In the aftermath, OTM put prices repriced permanently higher, and the **equity skew** was born and never disappeared. Today's persistent skew — ATM vol several points below deep-OTM-put vol — is the standing memorial to that single jump, and the clearest real-world evidence that markets price for jumps and a negative $\rho$.

### March 2020: a volatility spike, not just a fall

When COVID hit, the S&P 500 fell about $34\%$ in five weeks — and the VIX, the index of S&P implied volatility, rocketed from around $14$ to above $80$, its highest level since 2008. This is the leverage effect ($\rho < 0$) in its purest form: the price collapse and the volatility explosion were *the same event*. Anyone who was short volatility — short straddles, short variance swaps — discovered that realized vol can blow through implied vol by a factor of several, and the $\frac{1}{2}\Gamma S^2(\sigma_{\text{impl}}^2 - \sigma_{\text{real}}^2)$ losses were catastrophic. Several short-vol funds were wiped out in days.

### February 2018: "Volmageddon" and short-vol blowups

On February 5, 2018, the VIX more than doubled in a single session — its largest one-day move on record in percentage terms. A cluster of products that were *short* volatility (notably the inverse-VIX exchange-traded notes) had been quietly profitable for years collecting the implied-versus-realized spread; the spike, a jump in volatility itself, destroyed them overnight, with one note losing about $96\%$ of its value and terminating. It was a live demonstration that selling volatility is selling insurance: you collect small premiums for a long time and then, on one jumpy day, you pay it all back and more.

### The FX smile is symmetric; the equity smile is a skew

Compare two markets. In equity indices, the smile is a steep, lopsided *skew* — puts far richer than calls — because the leverage effect makes $\rho$ strongly negative: stocks and their volatility move opposite. In major currency pairs like EUR/USD, the smile is much closer to *symmetric*, because there is no built-in "crash direction" for an exchange rate — a big move up in EUR/USD is just as plausible and just as volatility-inducing as a big move down, so $\rho$ is near zero. The *shape* of the smile literally reads out the sign of $\rho$ for each market, exactly as the Heston model predicts.

### Earnings jumps and the term-structure kink

Single stocks show a vivid jump signature around earnings announcements. The options expiring *just after* an earnings date carry visibly elevated implied volatility, because everyone knows a *jump* — the earnings surprise — is coming, even though they do not know its direction. The options expiring *just before* the announcement are calm. This creates a sharp kink in the implied-vol term structure right at the earnings date, and traders explicitly model it as a deterministic jump on a known date — Merton's Poisson jump, but with the timing pinned down instead of random.

### Pricing exotics off a calibrated surface

When a bank sells an exotic — an autocallable, a barrier note, a cliquet — it does not price it off flat Black-Scholes. It calibrates a stochastic-vol-plus-jumps model (often a Heston or Bates variant, or a local-stochastic-vol hybrid) to the full vanilla [volatility surface](/blog/trading/quantitative-finance/volatility-surface), then prices and hedges the exotic within that calibrated model. Because the exotic's payoff depends on the *path* and the *tail* — exactly the features jumps and stochastic vol control — using flat Black-Scholes would misprice it by tens of percent. The mechanics of these path-dependent and tail-dependent payoffs are the subject of the [exotic derivatives](/blog/trading/quantitative-finance/exotic-derivatives) deep-dive.

![Tree of pricing models rooted at Black-Scholes, branching into jump models and stochastic-volatility models](/imgs/blogs/jump-diffusion-stochastic-volatility-math-for-quants-7.png)

The tree above is the family map. Everything descends from Black-Scholes (GBM) at the root. One branch relaxes the continuity assumption and adds jumps, giving Merton's lognormal-jump model and Kou's double-exponential-jump variant. The other branch relaxes the constant-volatility assumption and makes vol random, giving Heston (with its analytic pricing formula) and SABR (the market standard for interest-rate smiles). The most realistic production models — Bates, local-stochastic-volatility — sit where the two branches recombine, using jumps *and* random volatility together because, as the smile's term structure shows, you need both.

## When this matters to you and further reading

If you ever buy a put to protect a stock holding, you are paying the skew — the extra premium that a negative $\rho$ and the memory of 1987 have baked into downside insurance. If you ever sell a covered call, you are collecting on the (cheaper) upside wing. The smile is not an exotic curiosity; it is the price every option buyer and seller actually pays, and understanding *why* it has its shape is the difference between buying insurance at a fair price and overpaying for it.

For the quant building these models, the lesson is sharper still. Black-Scholes gives you a clean, closed-form answer that is wrong in exactly two ways — constant vol and continuous paths — and almost the entire field of derivatives pricing is the disciplined repair of those two flaws. Merton's jumps fix the gaps and the short-dated wings; Heston's stochastic volatility fixes the random, mean-reverting nature of vol and, through $\rho$, the downside skew. Knowing which model fixes which flaw — and where each one still fails — is what lets you choose the right tool for the contract in front of you.

To go deeper, the natural next steps build directly on this post. The [volatility surface](/blog/trading/quantitative-finance/volatility-surface) deep-dive treats the full smile-across-strikes-and-maturities object that calibration targets. The [exotic derivatives](/blog/trading/quantitative-finance/exotic-derivatives) post shows the path-dependent and barrier payoffs that *require* these richer models to price correctly. The [ARCH and GARCH volatility models](/blog/trading/math-for-quants/arch-garch-volatility-math-for-quants) post is the discrete-time, time-series cousin of Heston — it forecasts the very volatility that Heston makes random. And the [risk-neutral pricing](/blog/trading/quantitative-finance/risk-neutral-pricing-martingale-measure-quant-interviews) and [stochastic differential equations](/blog/trading/quantitative-finance/stochastic-differential-equations-gbm-ou-quant-interviews) posts supply the measure-theoretic and SDE foundations the equations in this article quietly stand on. Master those, and the volatility smile stops being a mysterious curve on a screen and becomes exactly what it is: the market's honest correction to a beautiful, useful, and slightly wrong formula.

---
title: "Local time and the reflection principle: pricing what happens at a level"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "A Brownian path spends zero seconds at any single price, and yet that price is exactly what decides whether a barrier option pays. Local time is how the paradox is resolved, the reflection principle is how the path count becomes a formula, and discrete monitoring is where the money is actually lost."
tags: ["local-time", "reflection-principle", "barrier-options", "tanaka-formula", "stochastic-calculus", "brownian-motion", "exotic-options", "continuity-correction", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 22
---

> [!important]
> **TL;DR:** A price path spends zero seconds at any single level, yet that level decides whether a barrier option pays. Local time is the density that resolves the paradox, and the reflection principle is what turns "did it ever touch?" into a closed form.
>
> - Time spent exactly at a level is zero. Time spent in a band around it, divided by the band width, converges to a finite number: **local time**, measured in dollars, not seconds.
> - **Tanaka's formula** is Ito's lemma applied to a kink. The second derivative a kink does not have comes back as a local-time term, which is why a zero-rate at-the-money call is worth exactly half the expected local time at its strike: \$7.04 a share on the example below.
> - The **reflection principle** pairs every path that touches a barrier and ends below it with exactly one mirrored path ending above. On a \$100 stock at 25% volatility, a \$90 barrier is touched **54.3%** of the time within six months.
> - That same mirror gives the price. A six-month \$100-strike call is worth \$8.01 a share, the \$90 down-and-out version \$6.92, a knock-out discount of **\$10,912.36** on 10,000 shares.
> - The closed form assumes continuous monitoring, and no traded contract has it. Daily monitoring on that option is worth **\$2,149.79** more, about 3.1%, and that gap is booked by whichever side gets the convention wrong.

## The paradox: a level the price never sits at

A trader sells a six-month knock-out call on a \$100 stock with the barrier at \$90. The contract says: if the stock ever trades at or below \$90, the option dies. Everything about the payoff hangs on one number, \$90, and on a single question about the path, whether it ever got there.

Now ask how much time the stock actually spends at \$90. The answer is none. A Brownian path is a continuous random function, and the set of times at which it equals any fixed level has [Lebesgue measure](/blog/trading/math-for-quants/lebesgue-integration-math-for-quants) zero. Not "a very short time". Zero, with probability one, for every level including the one it started from.

So the level that decides the entire contract is a level the price is never at. That is the paradox, and it is not a curiosity. Barrier pricing, what happens to a hedge as spot pins to a strike, and the reason a stop-loss strategy quietly bleeds all live in the gap between "zero time" and "clearly matters".

![Table of band half-widths of two dollars, one dollar, fifty cents and twenty-five cents against expected trading days inside the band of 22.75, 11.37, 5.69 and 2.84, with the ratio of days to dollar of width holding constant at 5.69 in every row and in the limit](/imgs/blogs/local-time-barriers-reflection-math-for-quants-1.webp)

The figure above is the resolution in one table. Widen the question from "time at \$100" to "time within a band around \$100" and you get a positive number. Shrink the band and that number goes to zero, exactly as the measure-theory argument says it must. But divide by the band width and the ratio does not move at all. The time vanishes; the *density* of time does not. That density is **local time**, and everything below is built on it.

## Foundations: what you need first

**Brownian motion** $W_t$ is the continuous-time limit of a random walk, [built from scratch here](/blog/trading/math-for-quants/brownian-motion-random-walk-math-for-quants). Three properties matter below: the path is continuous, it is nowhere differentiable, and its **quadratic variation** over $[0,t]$ is exactly $t$.

**Geometric Brownian motion** is what a stock does in the Black-Scholes world, $\mathrm{d}S_t = \mu S_t\,\mathrm{d}t + \sigma S_t\,\mathrm{d}W_t$, so the *logarithm* of the price is Brownian motion with drift. Under the risk-neutral measure that drift is $\nu = r - \sigma^2/2$, a correction the [SDE post](/blog/trading/math-for-quants/sdes-gbm-ou-cir-math-for-quants) derives.

**Ito's lemma** is the chain rule for such processes. For a twice-differentiable $f$,

$$\mathrm{d}f(S_t) = f'(S_t)\,\mathrm{d}S_t + \tfrac{1}{2}f''(S_t)\,\sigma^2 S_t^2\,\mathrm{d}t.$$

The second term is the reason option pricing is not ordinary calculus, and [the Ito post](/blog/trading/math-for-quants/ito-integral-itos-lemma-math-for-quants) unpacks it. Note the requirement, **twice differentiable**, and hold on to it.

**The running maximum** is $M_t = \max_{s \le t} W_s$, and the **first-passage time** to a level $a$ is $\tau_a = \inf\{t : W_t = a\}$. "The path touched $a$ before $T$", "$M_T \ge a$" and "$\tau_a \le T$" are three ways of writing one event, and you will move between them constantly.

**A barrier option** is a vanilla plus a level that kills it or creates it. A *down-and-out* call vanishes if the stock ever trades at or below the barrier $H$; a *down-and-in* call pays nothing unless the barrier is touched, at which point it becomes an ordinary call. Knock-in plus knock-out always equals the vanilla, because the barrier is either touched or it is not. The broader family sits in [the exotic derivatives post](/blog/trading/quantitative-finance/exotic-derivatives).

Throughout, one scenario: spot \$100, strike \$100, barrier \$90, volatility 25%, risk-free rate 4%, six months to expiry, on a position of 10,000 shares. All of it is illustrative arithmetic on assumed inputs, not a quote from a market.

## Local time: the density of occupation

Start with the quantity that is actually well defined. The **occupation time** of a set $A$ up to time $t$ is how long the path spent inside it:

$$\Gamma_t(A) = \int_0^t \mathbf{1}\{W_s \in A\}\,\mathrm{d}s.$$

For a band this is a positive number. For a single point it is zero. The question is what happens in between, and the answer is that $\Gamma_t$, viewed as a measure on levels, has a density. That density is local time:

$$L_t^a = \lim_{\epsilon \to 0} \frac{1}{2\epsilon}\int_0^t \mathbf{1}\{|W_s - a| \le \epsilon\}\,\mathrm{d}s.$$

The limit exists, is finite, and is continuous in both $a$ and $t$. Read it as "seconds per dollar of width at level $a$". Equivalently, and this is the form that does the work, the **occupation-time formula** says that integrating any function along the path is the same as integrating it against local time over levels:

$$\int_0^t f(W_s)\,\mathrm{d}s = \int_{-\infty}^{\infty} f(a)\,L_t^a\,\mathrm{d}a.$$

The left side sweeps time; the right side sweeps space. Local time is the exchange rate between them. For a general [semimartingale](/blog/trading/math-for-quants/martingales-risk-neutral-measure-math-for-quants) $X$ the same statement holds with $\mathrm{d}s$ replaced by the quadratic variation $\mathrm{d}\langle X\rangle_s$, which is the version you need for a stock, where $\mathrm{d}\langle S\rangle_t = \sigma^2 S_t^2\,\mathrm{d}t$.

Two consequences follow immediately.

First, **the units are not time**. Match dimensions: the left side is seconds, the right is levels times ${[L]}$, so ${[L]}$ is seconds per level, and normalising by $\mathrm{d}\langle X \rangle$ for a dollar-denominated process turns that into dollars. Local time is measured in dollars.

Second, a theorem of Lévy: local time at the starting level has the *same distribution as the running maximum*, and both have the same distribution as $|W_t|$. So $\mathbb{E}[L_t^0] = \mathbb{E}|W_t| = \sqrt{2t/\pi}$. That is what turns local time from an abstraction into a dollar number.

#### Worked example 1: what \$14.09 of local time buys you

Take the stock at \$100 with 25% annual volatility. Its dollar volatility near \$100 is ${0.25 \times \$100 = \$25}$ per square root of a year. Approximate the six-month path by arithmetic Brownian motion with that dollar volatility, which is accurate enough over six months to make the point.

1. **Expected local time at \$100.** By Lévy's theorem, $\mathbb{E}[L_T^{100}] = \sigma_{\$}\sqrt{2T/\pi} = 25 \times 0.5641896 = \$14.1047$.
2. **Convert it back into time in a band.** Occupancy of a band of width $w$ is $w \times L / \sigma_{\$}^2$. For a \$1 band, ${\$1 \times 14.1047 / 625 = 0.0225675}$ years, which at 252 trading days is **5.69 days** out of the 126 in the period, or 4.5% of the life of the trade.
3. **Shrink the band.** Halve it to 50 cents and the occupancy halves to 2.84 days. Halve it again and it halves again. The ratio, 5.69 days per dollar of width, never moves. That is the column that converges.
4. **Now the money.** Redone exactly for geometric Brownian motion rather than the arithmetic shortcut, the expected local time is \$14.0864, so the approximation cost 0.13%. Half of it is ${\$14.0864/2 = \$7.0432}$ a share. A zero-rate at-the-money six-month call at 25% volatility is worth ${\$100 \times (2 \times 0.535216 - 1) = \$7.0432}$ a share, or **\$70,432** on 10,000 shares.

Those last two numbers are not close by luck, and the next section says why.

## Tanaka's formula: Ito applied to a kink

Ito's lemma needs $f$ twice differentiable. Every payoff a trader cares about is not. A call payoff $(x-K)^+$ has a corner at the strike: the slope jumps from 0 to 1 and the second derivative does not exist there. So what happens if you apply Ito anyway?

The honest answer is **Tanaka's formula**, and it is the cleanest reason local time exists at all:

$$|W_t - a| = |W_0 - a| + \int_0^t \operatorname{sgn}(W_s - a)\,\mathrm{d}W_s + L_t^a,$$

$$(W_t - a)^+ = (W_0 - a)^+ + \int_0^t \mathbf{1}\{W_s \gt a\}\,\mathrm{d}W_s + \tfrac{1}{2}L_t^a.$$

Compare to Ito. The first two terms are exactly what Ito would give: the starting value plus the integral of the first derivative against the path. The first derivative of $(x-a)^+$ is the indicator $\mathbf{1}\{x \gt a\}$, which exists everywhere except at the corner, and a single point never matters to an integral. The third term is the one Ito cannot produce, because it is the second-derivative correction of a function whose second derivative is not a function at all.

In the language of distributions, the second derivative of $(x-a)^+$ is a unit point mass at $a$. Feed that through Ito's correction $\tfrac{1}{2}\int f''(W_s)\,\mathrm{d}s$ and you get one half, times the time the path spent at $a$, times infinity. Zero times infinity. The product is finite, and it is $\tfrac{1}{2}L_t^a$. **Local time is exactly the correction the missing second derivative demands.**

![Two-panel comparison: a smooth function where Ito's correction is one half f double prime times sigma squared S squared dt spread over every level with finite gamma, against a kinked payoff where f double prime is zero except at the strike, the correction becomes one half of local time at K, and gamma is infinite at that single level](/imgs/blogs/local-time-barriers-reflection-math-for-quants-3.webp)

The figure above is the whole idea. A smooth payoff spreads its convexity across every level the path visits, and the hedger earns or pays gamma continuously. A kinked payoff concentrates all of its convexity at one level, so the accounting cannot be an integral over time. It has to be an integral over *how densely the path visited that level*, which is local time.

That closes the loop on worked example 1. Set $r=0$ and $a = K = S_0$ and take risk-neutral expectations of the second identity: the stochastic integral has expectation zero because it is a martingale, leaving $\mathbb{E}[(S_T-K)^+] = \tfrac{1}{2}\mathbb{E}[L_T^K]$. The at-the-money option's entire value **is** half the expected local time at its strike. Not an analogy. An identity.

## The reflection principle

Now the second half, which answers a different question: not how densely the path visits a level, but whether it *ever reaches* one.

The direct approach is hopeless: finding $\mathbb{P}(\tau_a \le t)$ means integrating over the set of all continuous paths that touch $a$ somewhere, an object with no useful description. The **reflection principle** sidesteps it with a symmetry argument that takes one line once you see it.

Fix a level $a \gt 0$ and take the paths that touch $a$ before time $t$. Each has a first touching time $\tau_a$, and from that instant on, by the strong Markov property, it continues as a fresh Brownian motion started at $a$. A fresh Brownian motion is symmetric, so any continuation and its mirror image about $a$ are equally likely. Pair every touching path with the one that agrees with it up to $\tau_a$ and is reflected about $a$ afterwards: the pairing is a bijection and it preserves probability.

![Chart of stock price against time showing a path from one hundred dollars falling to touch the ninety dollar barrier at time tau and continuing down to eighty-four, together with its mirror image after tau rising to ninety-six, with the two endpoints labelled ends below ninety and mirror ends above ninety](/imgs/blogs/local-time-barriers-reflection-math-for-quants-2.webp)

The figure above shows one pair. The solid path touches \$90 and ends at \$84; its mirror agrees with it up to the touch and then does the opposite of everything, ending at \$96. Under the bijection, touching paths that end *below* the level map one-to-one onto touching paths that end *above* it, and by continuity a path that ends above $a$ must have touched $a$. So

$$\mathbb{P}(M_t \ge a) = 2\,\mathbb{P}(W_t \ge a) = \mathbb{P}(|W_t| \ge a).$$

The impossible path count has become a normal-table lookup, doubled. That factor of two is the entire content of the principle and the most common place to lose a mark: the probability of *touching* is twice the probability of *finishing above*, not the same as it.

Drift breaks the symmetry, because a drifting path and its mirror are no longer equally likely. The repair is a [change of measure](/blog/trading/math-for-quants/girsanov-change-of-measure-math-for-quants): Girsanov removes the drift, reflection applies to the driftless path, and the Radon-Nikodym density comes back as an exponential weight on the reflected term. For a down barrier, with $b = \log(H/S_0) \lt 0$ and $\nu = r - \sigma^2/2$,

$$\mathbb{P}\left(\min_{s \le T} \log \frac{S_s}{S_0} \le b\right) = N\!\left(\frac{b - \nu T}{\sigma\sqrt{T}}\right) + e^{2\nu b/\sigma^2}\,N\!\left(\frac{b + \nu T}{\sigma\sqrt{T}}\right).$$

Set $\nu = 0$ and the weight becomes 1 and the two terms collapse to $2N(b/(\sigma\sqrt{T}))$, the reflection principle exactly.

#### Worked example 2: how often is a 10% barrier touched?

Barrier at \$90 on a \$100 stock, six months, 25% volatility, 4% rates. Here $b = \log(0.9) = -0.105361$ and $\sigma\sqrt{T} = 0.25 \times 0.707107 = 0.176777$.

1. **Driftless first.** ${b/(\sigma\sqrt{T}) = -0.105361/0.176777 = -0.5960}$, and $N(-0.5960) = 0.2756$. Double it: ${2 \times 0.2756 = 0.5512}$, so **55.1%**.
2. **With risk-neutral drift.** ${\nu = 0.04 - 0.03125 = 0.00875}$, so ${\nu T = 0.004375}$. First term: ${N((-0.105361 - 0.004375)/0.176777) = N(-0.62076) = 0.26738}$. Weight: ${e^{2 \times 0.00875 \times (-0.105361)/0.0625} = e^{-0.029501} = 0.97093}$. Second term: ${N((-0.105361+0.004375)/0.176777) = N(-0.57126) = 0.28391}$.
3. Total: ${0.26738 + 0.97093 \times 0.28391 = 0.26738 + 0.27566 = 0.54304}$, so **54.3%**.

A barrier 10% out of the money is touched more often than not inside six months. On the \$69,167.61 of knock-out premium priced below, just over half of all scenarios end with that premium earned and the buyer holding nothing. The drift correction is worth 0.8 percentage points here, small because six months of 4% drift is small next to 17.7% of volatility, but it grows with maturity and it is not optional.

## From reflection to a barrier price

A knock-out price is an expectation over paths that never touch. The density of those paths is the unrestricted density minus the density of the touching ones, and reflection is what evaluates the subtraction. Carrying it through gives the **method of images**: the killed process behaves like the free process minus a weighted copy started at the mirror point $H^2/S$ on the far side of the barrier. For a down-and-out call with $H \le K$ and no dividends,

$$C_{\mathrm{DO}}(S) = C(S) - \left(\frac{S}{H}\right)^{1 - 2r/\sigma^2} C\!\left(\frac{H^2}{S}\right),$$

where $C$ is the ordinary Black-Scholes call with the same strike and maturity. Merton derived this in 1973; Reiner and Rubinstein catalogued all eight barrier types in 1991.

Each piece earns its place. $C(S)$ is the option you would have if the barrier did not exist. The subtracted term is the value carried by the paths that do touch, written as a **real option on a fictitious spot**: reflect the current price about the barrier multiplicatively, ${100 \to 90^2/100 = \$81}$, and price a normal call there. The power ${1 - 2r/\sigma^2}$ is the Girsanov weight, the same exponential that appeared in the touch probability. Set $r = \sigma^2/2$ and the power is zero, the weight is one, and you are back to pure reflection with no drift.

Two sanity checks are worth carrying. At $S = H$ the mirror point is also $H$, the weight is 1, and the terms cancel exactly, so the knock-out is worthless at the barrier. As $S$ grows the mirror point $H^2/S$ goes to zero, the image call goes to zero, and the knock-out converges to the vanilla.

![Line chart of option value per share against spot price from ninety to one hundred and twenty dollars, showing the vanilla call rising from 3.37 to 23.18 and the down-and-out call from zero at the barrier to 23.10, with the gap at one hundred dollars marked as 1.09 per share or 10,912 dollars on ten thousand shares](/imgs/blogs/local-time-barriers-reflection-math-for-quants-4.webp)

The figure above prices the whole strip. The two curves are indistinguishable above \$115 and maximally different at the barrier, where one is worth \$3.37 and the other nothing. That collapsing gap is the barrier option's risk profile in one image: almost all of its unusual behaviour is concentrated in the region nobody plots.

#### Worked example 3: the knock-out discount in dollars

Spot \$100, strike \$100, barrier \$90, $r = 4\%$, $\sigma = 25\%$, $T = 0.5$, 10,000 shares.

1. **Vanilla.** ${d_1 = 0.035625/0.1767767 = 0.2015254}$ and ${d_2 = d_1 - \sigma\sqrt{T} = 0.0247487}$, so ${N(d_1) = 0.5798561}$ and ${N(d_2) = 0.5098723}$. Price ${= 100(0.5798561) - 98.0199(0.5098723) = 57.9856 - 49.9776 = \$8.0080}$ a share.
2. **The image option.** Mirror spot ${= 90^2/100 = \$81}$. At that spot, ${d_1 = -0.9904927}$ and ${d_2 = -1.1672694}$, giving ${N(d_1) = 0.1609667}$ and ${N(d_2) = 0.1215508}$, so the image call is worth ${81(0.1609667) - 98.0199(0.1215508) = 13.0383 - 11.9144 = \$1.1239}$ a share.
3. **The weight.** ${1 - 2r/\sigma^2 = 1 - 0.08/0.0625 = -0.28}$, and ${(100/90)^{-0.28} = e^{-0.28 \times 0.105361} = 0.97093}$. So the subtracted term is ${0.97093 \times 1.1239 = \$1.0912}$ a share.
4. **The knock-out.** ${\$8.0080 - \$1.0912 = \$6.9168}$ a share.
5. **In money.** Carrying the unrounded prices, \$8.007997 and \$6.916761 a share, 10,000 shares of the vanilla cost **\$80,079.97** and of the knock-out **\$69,167.61**. The discount is **\$10,912.36**, or **13.6%** of the vanilla premium, which is what the buyer is paid for accepting a 54.3% chance of being knocked out.

The discount and the image term are the same number because they are the same object: the value that lives on the touching paths.

## Why the formula is not the price: discrete monitoring

Everything above assumes the barrier is watched continuously, and no contract is. A term sheet says the barrier is observed at the official closing price each business day, or at a 4pm London fixing, or at expiry only. A daily-monitored option can trade at \$89.40 intraday, close at \$90.20, and survive. Continuous monitoring would have killed it.

That is not a rounding error. Discrete monitoring makes the option **harder to knock out**, so the knock-out is worth **more** than the continuous formula says, and the gap is first order in $\sqrt{\Delta t}$, not in $\Delta t$. That square root is why it is bigger than intuition suggests: what you are missing is the expected overshoot between observations, and a Brownian path's overshoot scales like the square root of the gap between looks.

Broadie, Glasserman and Kou (1997) turned this into one of the most useful results in exotic pricing. Rather than building a new model, **shift the barrier and reuse the continuous formula**. For a barrier monitored $m$ times over a life of $T$,

$$V_m(H) \approx V\!\left(H\,e^{\pm\beta\sigma\sqrt{T/m}}\right), \qquad \beta = -\frac{\zeta(1/2)}{\sqrt{2\pi}} \approx 0.5826,$$

with the plus sign for an up barrier and the minus for a down barrier. The barrier always moves *away* from the spot, which is the right direction: a discretely watched barrier behaves like a slightly more distant continuously watched one. The constant is a Riemann zeta value because the underlying problem, the expected overshoot of a random walk past a level, has a classical answer.

![Table comparing continuous, daily, weekly and monthly monitoring of the same ninety dollar down-and-out call, showing effective barriers of 90.00, 89.18, 88.20 and 86.29, values per share of 6.92, 7.13, 7.34 and 7.63, position values on ten thousand shares, and gaps versus continuous of 2,149.79, 4,251.20 and 7,176.96 dollars](/imgs/blogs/local-time-barriers-reflection-math-for-quants-5.webp)

#### Worked example 4: what daily monitoring is worth

Same option, now monitored at 126 daily closes, so ${\Delta t = 1/252}$.

1. **The shift.** ${\sqrt{1/252} = 0.0629941}$, so ${\sigma\sqrt{\Delta t} = 0.0157485}$ and ${\beta\sigma\sqrt{\Delta t} = 0.5826 \times 0.0157485 = 0.0091751}$. The factor is ${e^{-0.0091751} = 0.990867}$.
2. **The effective barrier.** ${\$90 \times 0.990867 = \$89.178}$, a shift of 82 cents. The touch probability falls from 54.3% to 50.9%.
3. **Reprice.** Run the same closed form with $H = 89.178$: the mirror spot becomes ${89.178^2/100 = \$79.527}$, the image call is worth \$0.90481, the weight is 0.96844, and the subtracted term is ${0.96844 \times 0.90481 = \$0.8763}$. The knock-out is ${\$8.0080 - \$0.8763 = \$7.1317}$ a share.
4. **The booking.** At the unrounded \$7.131740 a share, 10,000 shares cost **\$71,317.40** against **\$69,167.61** for the continuous formula. The difference is **\$2,149.79**, or **3.1%** of the option's value.

Weekly monitoring widens the gap to \$4,251.20 and monthly to \$7,176.96, 10.4% of the price. A desk quoting the continuous formula on a monthly-monitored barrier is not being slightly conservative. It is mispricing by more than its entire bid-offer spread.

Three practical wrinkles survive the correction. The **observation definition** matters: official close, intraday low and a fixing window are three different contracts, and the correction above assumes the first. **Gap risk** is real, because the barrier can be jumped rather than touched on an earnings or overnight move, and no diffusion model prices a jump through the level. And the **hedge misbehaves**: near the barrier close to expiry a knock-out's delta can exceed one and flip sign, so desks price and hedge against a *shifted* barrier further out than the contract's, and carry the difference as an overhedge reserve.

## Where local time shows up beyond barriers

The other place local time surfaces is in a strategy that looks like free money and is not.

The **stop-loss start-gain** strategy: hold one share whenever the stock is above $K$, nothing whenever it is below. Buy on every upward crossing of $K$, sell on every downward crossing. At expiry you hold a share if and only if $S_T \gt K$, and every trade executed at exactly $K$, so switching cost nothing. The terminal position looks worth $(S_T - K)^+$ for the initial intrinsic value alone. That would replicate a call for free.

Tanaka says where the money went:

$$(S_T - K)^+ = (S_0 - K)^+ + \int_0^T \mathbf{1}\{S_t \gt K\}\,\mathrm{d}S_t + \tfrac{1}{2}L_T^K.$$

The middle term is the strategy's trading gain. The final term is the shortfall, and it is not zero: half the local time at the strike, \$7.04 a share on the running example, which is the entire time value of the option. Carr and Jarrow (1990) identified this as the resolution of the paradox. The path crosses $K$ infinitely often, so the strategy needs infinitely many round trips, and in any real implementation, with discrete monitoring or a bid-offer spread, their accumulated cost is exactly what local time counts. It is also why a stop-loss order in a choppy market bleeds: not because the level was wrong, but because the path kept coming back.

## Common misconceptions

**"Local time is an amount of time."** It is not, and the dimensions prove it. The occupation-time formula forces local time to carry units of seconds per unit of level, which after the standard normalisation by quadratic variation is dollars. Expected local time at \$100 over six months in the example is \$14.09, not 14.09 anything-else. Treating it as a duration will invert your whole intuition, because local time *grows* when the path lingers near a level and its "time interpretation" then gets divided by a vanishing width.

**"The reflection principle is a trick with no content."** The bijection is the content. It converts an integral over a set of paths, which has no closed form, into a statement about the marginal distribution at a single time, which is a normal table lookup. It is also what forces the factor of two, which is the difference between "reached the barrier" and "finished past it" and is worth exactly half the answer. And it is not free: as soon as there is drift, symmetry fails, and you have to buy it back with Girsanov, which is where the exponential weight in every barrier formula comes from.

**"A barrier formula prices a barrier option."** It prices a continuously monitored barrier option, and nobody trades one. Monitoring frequency is a first-order effect, not a refinement: on the example above, moving from continuous to daily changes the price by 3.1% and to monthly by 10.4%, while a plausible change in the volatility model would move it by less. Any pricing conversation that reaches the smile before it reaches the observation schedule has its priorities backwards. The related slip is reading "zero time at the barrier" as "the barrier rarely binds": that barrier is touched 54.3% of the time.

## Sources and further reading

- Tanaka, H. (1963). "Note on continuous additive functionals of the 1-dimensional Brownian path." *Zeitschrift für Wahrscheinlichkeitstheorie* 1, 251-257. The original formula.
- Karatzas, I. and Shreve, S. (1991). *Brownian Motion and Stochastic Calculus*, 2nd ed. Springer. Section 2.6 for the reflection principle, 3.6 and 3.7 for local time and Tanaka. The standard reference.
- Revuz, D. and Yor, M. (1999). *Continuous Martingales and Brownian Motion*, 3rd ed. Springer. Chapter VI, for local time done properly, including Lévy's theorem.
- Merton, R. (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics and Management Science* 4(1), 141-183. The first down-and-out formula.
- Reiner, E. and Rubinstein, M. (1991). "Breaking down the barriers." *Risk* 4(8), 28-35. The full catalogue of the eight single-barrier types.
- Broadie, M., Glasserman, P. and Kou, S. (1997). "A continuity correction for discrete barrier options." *Mathematical Finance* 7(4), 325-349. The barrier shift and the constant 0.5826.
- Carr, P. and Jarrow, R. (1990). "The Stop-Loss Start-Gain Paradox and Option Valuation: A New Decomposition into Intrinsic and Time Value." *Review of Financial Studies* 3(3), 469-492.
- Shreve, S. (2004). *Stochastic Calculus for Finance II*. Springer. Chapter 7 for exotic options built on the joint law of the maximum.

The dollar figures in the worked examples are illustrative arithmetic on assumed inputs, computed with the formulas above, not quotes from any market.

## In the interview room and on the desk

The question arrives as **"how would you price a knock-out call?"**, sometimes dressed as "a client wants a cheaper call and will accept it dying at \$90".

The strong answer moves in a fixed order. First, name the object: a knock-out is a vanilla minus the value carried by the paths that touch, and knock-in plus knock-out equals the vanilla, so you only ever have to price one of them. Second, say how you get that value: the reflection principle, because the path count you cannot do becomes an endpoint probability you can, at the cost of a factor of two, plus a Girsanov weight to handle drift. If pushed, write the image formula, ${C(S) - (S/H)^{1-2r/\sigma^2}\,C(H^2/S)}$, and check it at the barrier where the two terms cancel. Third, and this is the part that separates candidates, **volunteer the monitoring frequency before being asked**. Say that the closed form assumes continuous monitoring, that no term sheet has it, and that the Broadie-Glasserman-Kou correction shifts the barrier by ${\beta\sigma\sqrt{\Delta t}}$ with ${\beta \approx 0.5826}$. Quantify it: on a six-month 10% barrier at 25% volatility, daily monitoring is worth about 3% of the option and monthly about 10%.

The trap is quoting the continuous formula for a daily-monitored barrier and stopping there. It sounds rigorous, the derivation is correct, and the number is wrong by several times the spread you would quote. The second trap is the factor of two: using $\mathbb{P}(S_T \le H)$ where you needed $\mathbb{P}(\min S \le H)$ halves the touch probability and, on a knock-out, shows up as the wrong side of a 54% event. A third, rarer, is asserting a barrier option is always cheaper without checking which side of the strike the barrier is on.

If the conversation goes deeper, local time is the natural follow-up: what is the gamma of a call exactly at the strike at expiry, why is a stop-loss replication not free, what does a digital's hedge look like near the barrier. Jane Street, Citadel and any exotics or structured-products seat weight this material most heavily, because the barrier book is where an elegant formula and a sloppy convention cost real money on the same trade.

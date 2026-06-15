---
title: "Girsanov's theorem and the change of measure: how quants move from the real world to the pricing world"
date: "2026-06-15"
description: "A from-scratch tour of why option prices ignore a stock's expected return, how the Radon-Nikodym derivative reweights probabilities, and how Girsanov's theorem turns the real-world drift into the risk-free rate."
tags: ["girsanov", "change-of-measure", "radon-nikodym", "risk-neutral", "market-price-of-risk", "monte-carlo", "importance-sampling", "stochastic-calculus", "quant-finance", "derivatives-pricing"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Changing the probability measure means reweighting how likely each outcome is, without ever changing which outcomes are possible, and that single trick is what lets quants price options.
>
> - The real world runs on a measure called **P**; the pricing world runs on a different measure called **Q**. They agree on what can happen and disagree on how likely it is.
> - The **Radon-Nikodym derivative** $\frac{dQ}{dP}$ is just the per-outcome weight that turns P-probabilities into Q-probabilities. It is a likelihood ratio, nothing more.
> - **Girsanov's theorem** says that if you reweight outcomes the right way, the stock's real growth rate $\mu$ disappears and is replaced by the risk-free rate $r$. The volatility $\sigma$ never moves.
> - The exact drift that gets removed is the **market price of risk** $\lambda=\frac{\mu-r}{\sigma}$ — the asset's Sharpe ratio. This is *why* an option's price ignores the stock's expected return and depends only on its volatility.
> - The same change-of-measure trick, used in reverse for **Monte Carlo importance sampling**, can cut the number of simulations you need to price a deep out-of-the-money option by 10x or more.

Here is a fact that sounds like a typo the first time you hear it: when a quant prices an option on a stock, they throw away the stock's expected return. Two stocks — one a sleepy utility that creeps up \$2 a year, one a hot growth name the whole market expects to double — can have *identical* option prices if their volatility is the same. The thing every investor cares about most, the expected return, plays no role at all in the price of the derivative.

That is not a simplification or an approximation. It is the literal, exact answer, and it falls straight out of one of the most beautiful results in financial mathematics: Girsanov's theorem and the change of probability measure. By the end of this post you will understand exactly why the expected return vanishes, what replaces it, and how the same machinery quietly powers the Monte Carlo engines that price the most exotic products on a trading desk. The figure below is the whole story in one picture — the same stock, viewed through two different probability lenses.

![Real world drift mu becomes pricing world drift r while volatility stays fixed](/imgs/blogs/girsanov-change-of-measure-math-for-quants-1.png)

On the left is the stock as it actually behaves — call this the real world, the world of measure P. It drifts upward at its true expected return $\mu$. On the right is the same stock in the pricing world, the world of measure Q, where it drifts upward at only the risk-free rate $r$. Notice what did *not* change: the volatility $\sigma$ and the set of outcomes that are possible. Girsanov's theorem is the precise statement of how you get from the left picture to the right one, and the rest of this article is a slow, careful walk through that bridge.

## Foundations: the building blocks you need first

Before we can talk about *changing* a probability measure, we need to be crystal clear about what a probability measure even is, what a Brownian motion is, and what it means for a stock price to "drift". None of this requires prior finance knowledge. We will define every term the first time it appears.

### What a probability measure actually is

A **probability measure** is a rule that assigns a number between 0 and 1 to every possible outcome (or set of outcomes), saying how likely it is. If you roll a fair die, the probability measure assigns $\frac{1}{6}$ to each face. If you have a loaded die, a *different* probability measure assigns, say, $\frac{1}{2}$ to the six and $\frac{1}{10}$ to each of the other five faces.

Here is the key observation that the entire post hangs on: the fair die and the loaded die have the **same possible outcomes** — the faces 1 through 6 — but **different probabilities**. They live on the same set of possibilities; they just weight those possibilities differently. Changing the measure is exactly this: keeping the outcomes and re-deciding how likely each one is.

Two probability measures are called **equivalent** if they agree on what is *possible* — anything with zero probability under one has zero probability under the other, and vice versa. The fair die and the loaded die are equivalent: under both, the only impossible outcome is "the die shows a 7". A measure that suddenly made "show a 7" possible would *not* be equivalent. This matters enormously, because the whole theory only works between equivalent measures. You are allowed to reweight outcomes; you are not allowed to conjure up new ones or delete existing ones.

> Changing the measure is like adjusting the brightness on a photo: every pixel that was there is still there, you have only changed how much each one stands out. You cannot add a building that was never in the frame.

### What a Brownian motion is

A **Brownian motion** (also called a Wiener process, written $W_t$) is the mathematical model of a continuous random walk. Start at zero. At each instant, take an infinitesimally small random step, up or down, with no memory of the past. Add up all those steps and you get a wiggly, jagged path that never repeats. Three properties define it: it starts at zero ($W_0=0$), its increments are independent and normally distributed with variance equal to the elapsed time ($W_t-W_s\sim N(0,\,t-s)$), and its paths are continuous. We built this object up from the discrete random walk in the [Brownian motion post](/blog/trading/quantitative-finance/brownian-motion-quant-interviews); here we just use it.

The single most important feature for us: a standard Brownian motion has **zero drift**. On average it goes nowhere. $E[W_t]=0$ for every $t$. It is pure noise with no built-in tendency to climb or fall. Keep that in your pocket — Girsanov's theorem is fundamentally a statement about *adding and removing drift from a Brownian motion*.

### What "drift" and "volatility" mean for a stock

We model a stock price $S_t$ with a stochastic differential equation, or SDE. The standard one is **geometric Brownian motion** (GBM):

$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t$$

Read this out loud in plain English. The change in the stock price over a tiny instant, $dS_t$, has two pieces. The first piece, $\mu S_t\,dt$, is the **drift** — a steady, predictable pull upward at rate $\mu$ per year. If $\mu=10\%$, the stock tends to grow about 10% a year. The second piece, $\sigma S_t\,dW_t$, is the **diffusion** — random noise, scaled by the **volatility** $\sigma$, riding on the Brownian motion $W_t$. If $\sigma=20\%$, the stock's returns scatter with a standard deviation of about 20% a year. We unpack GBM, OU, and CIR processes in detail in the [SDE post on this site](/blog/trading/quantitative-finance/stochastic-differential-equations-gbm-ou-quant-interviews); for now, the two-knob mental model is enough: drift is the steady pull, volatility is the random shake.

The number $\mu$ is what every investor cares about — it is the stock's expected return. The number $\sigma$ is what every option trader cares about — it is the size of the random shake. Hold on to that distinction, because the punchline of this entire article is that **option prices depend on $\sigma$ and not on $\mu$**, and the change of measure is the reason.

#### Worked example: the simplest possible measure change

Let's make the abstract idea of reweighting completely concrete with the smallest example imaginable. A stock today is worth \$100. In one year it will be worth one of two values: \$120 (the "up" state) or \$90 (the "down" state). Suppose the *real-world* probabilities are 70% up, 30% down.

What is the real-world expected price one year out? You weight each outcome by its real probability:

$$E^P[S_1] = 0.70\times \$120 + 0.30\times \$90 = \$84 + \$27 = \$111.$$

So in the real world the stock is expected to be worth \$111 — a 11% expected return. Now imagine a *different* observer who insists the probabilities are actually 40% up, 60% down. Same two outcomes, \$120 and \$90 — nobody invented a new price — but reweighted. Their expected price is:

$$E^Q[S_1] = 0.40\times \$120 + 0.60\times \$90 = \$48 + \$54 = \$102.$$

Same stock, same possible payoffs, two different "expected" values, purely because we changed the weights from $(0.70,\,0.30)$ to $(0.40,\,0.60)$. **That is a change of measure.** The one-sentence intuition: you can move the expected value of an asset anywhere you like inside the range of its outcomes just by reweighting the probabilities — and that freedom is exactly what pricing theory exploits.

## 1. Why pricing needs a second probability world at all

The example above raises the obvious question: why would anyone *want* a second set of probabilities? The real world's probabilities are, well, real. Why invent a fictional 40/60 world?

The answer is no-arbitrage, and it is worth getting right because it is the foundation everything else rests on.

### The no-arbitrage principle in one breath

An **arbitrage** is a money pump: a trade that costs nothing today, can never lose money, and has a positive chance of making money. In a functioning market, arbitrages cannot persist — the instant one appears, traders pile in and the prices move until it is gone. So the working assumption of all derivatives pricing is: **prices are set so that no arbitrage exists.**

Here is the deep theorem that connects this to probability, called the *fundamental theorem of asset pricing*: a market has no arbitrage **if and only if** there exists a probability measure $Q$ — equivalent to the real-world measure $P$ — under which every traded asset's discounted price is a **martingale**. A *martingale* is a process with no drift after you account for the time value of money: its best forecast of tomorrow's discounted value is today's value. We develop the martingale machinery from scratch in the [risk-neutral pricing post](/blog/trading/quantitative-finance/risk-neutral-pricing-martingale-measure-quant-interviews), and this article is in many ways its companion: that post tells you *that* the measure Q exists and what it is for; this post tells you *how to construct it* and *what it does to the stock's dynamics*.

So the second probability world is not optional decoration. It is forced into existence by the simple, hard requirement that the market not contain a money pump. The figure below traces the construction from the real measure on the left to the pricing measure on the right.

![Pipeline from real measure P through the Radon-Nikodym likelihood ratio to pricing measure Q](/imgs/blogs/girsanov-change-of-measure-math-for-quants-2.png)

You start with the real measure P. You multiply each outcome's probability by a carefully chosen weight — the Radon-Nikodym derivative, which we will meet properly in the next section. The result is a new measure Q. And the whole point of choosing that particular weight is that, under Q, the discounted stock price has no drift: it becomes a martingale, and no-arbitrage is satisfied automatically.

### Why Q is called "risk-neutral"

The measure Q goes by a friendlier name: the **risk-neutral measure**. Here is where the name comes from. In the real world, risky assets earn *more* than the risk-free rate on average, because investors demand to be paid for bearing risk — that extra pay is the **risk premium**. A risk-neutral investor, by definition, does not care about risk and demands no premium: to them, every asset should be expected to earn exactly the risk-free rate $r$, no more.

Measure Q is the probability world in which everyone behaves as if they were risk-neutral. Under Q, the expected return of *every* asset — the safe bond, the volatile tech stock, the lottery-ticket option — is the same risk-free rate $r$. That is a strange world. It is not the real world; nobody actually believes those probabilities. But it is an extraordinarily convenient world, because in it, pricing is trivial: the fair price of any derivative is just its expected payoff under Q, discounted back at the risk-free rate.

$$\text{Price}_0 = e^{-rT}\,E^Q[\text{Payoff}_T].$$

The genius of the framework is that this fictional price is also the *only* price consistent with no-arbitrage in the real world. We compute in the convenient fake world; the answer is correct in the real one.

## 2. The Radon-Nikodym derivative: a weight for every outcome

We have been saying "reweight each outcome" loosely. The Radon-Nikodym derivative is the precise object that does the reweighting. Despite the intimidating name, it is conceptually simple, and the two-scenario example will nail it down.

### The definition, built up gently

Suppose $P$ and $Q$ are two equivalent measures on the same outcomes. The **Radon-Nikodym derivative**, written $\frac{dQ}{dP}$, is the function $Z$ that tells you, for each outcome, how much more (or less) likely it is under Q than under P. Formally, for any event $A$:

$$Q(A) = \int_A \frac{dQ}{dP}\,dP = E^P\!\left[Z\cdot \mathbb{1}_A\right],\qquad Z=\frac{dQ}{dP}.$$

In words: to get the Q-probability of an event, take its outcomes, weight each one by $Z$, and average under P. The weight $Z$ is exactly a **likelihood ratio** — the ratio of the new probability to the old probability, outcome by outcome.

Two facts pin down what a legitimate $Z$ looks like. First, $Z\ge 0$ everywhere, because probabilities can't go negative. Second, $E^P[Z]=1$, because the total probability under Q must still sum to 1. A valid reweighting can stretch some outcomes and shrink others, but the average stretch, measured under P, has to be exactly 1. Think of $Z$ as a budget of "probability mass" you are allowed to push around: you can move it from the down-states to the up-states, but you cannot create or destroy any.

#### Worked example: the Radon-Nikodym derivative on two scenarios

Return to our two-state stock: \$120 up, \$90 down, with real-world probabilities $P(\text{up})=0.70$, $P(\text{down})=0.30$. We decided the risk-neutral probabilities should be $Q(\text{up})=0.40$, $Q(\text{down})=0.60$. What is the Radon-Nikodym derivative $Z$ in this world?

It is just the ratio of Q-probability to P-probability, outcome by outcome:

$$Z(\text{up}) = \frac{Q(\text{up})}{P(\text{up})} = \frac{0.40}{0.70} = 0.571,\qquad Z(\text{down}) = \frac{Q(\text{down})}{P(\text{down})} = \frac{0.60}{0.30} = 2.0.$$

So the up-state is *shrunk* (its weight is 0.571, less than 1) and the down-state is *doubled* (weight 2.0). The risk-neutral world makes the bad outcome twice as likely and the good outcome a little over half as likely. Let's check the budget constraint — the average weight under P must be 1:

$$E^P[Z] = 0.70\times 0.571 + 0.30\times 2.0 = 0.40 + 0.60 = 1.0.\ \checkmark$$

It balances. Now here is the part worth pausing on: notice that the **payoffs never changed**. The stock is still worth \$120 up and \$90 down in both worlds. Only the probabilities moved. The down-state got heavier precisely because, in a risk-neutral world, you can't be rewarded for taking the risk that the stock falls — so the framework leans the odds toward the downside until the expected return drops to the risk-free rate. The one-sentence intuition: the Radon-Nikodym derivative is a per-outcome dial, turning up the unlucky states and turning down the lucky ones, until the expected return becomes risk-free.

### From dollars to drift: the same dial, in continuous time

In the two-state model the reweighting is two numbers, $Z(\text{up})$ and $Z(\text{down})$. In continuous time — where the stock follows a Brownian motion and there are infinitely many possible paths — the reweighting becomes a whole *process* $Z_t$, one weight for each path, evolving over time. Remarkably, this process has a clean closed form, and producing it is exactly the job of Girsanov's theorem. The figure below shows the structural change Girsanov makes to the underlying Brownian motion.

![Stack showing the new Brownian motion equals the old one plus the market price of risk times time](/imgs/blogs/girsanov-change-of-measure-math-for-quants-3.png)

The picture, read top to bottom: the new Brownian motion $\widetilde{W}_t$ that lives under Q is built from the old Brownian motion $W_t$ that lived under P, plus a deterministic, steadily-growing drift term equal to the market price of risk times time. That added drift is precisely what absorbs the excess return $\mu-r$ and leaves the stock growing at only $r$. The rest of the article makes every piece of that sentence precise.

## 3. Girsanov's theorem, stated and unpacked

Now we can state the star of the show. We will state it, then immediately translate every symbol, then verify it does what we claimed.

### The statement

Let $W_t$ be a Brownian motion under the real measure $P$. Pick any (nice enough) process $\theta_t$ — for our purposes a constant $\theta$ will do. Define the reweighting process

$$Z_t = \exp\!\left(-\theta W_t - \tfrac{1}{2}\theta^2 t\right).$$

Then $Z_t$ is a valid Radon-Nikodym density (it is positive and has $E^P[Z_t]=1$), and it defines a new measure $Q$ via $\frac{dQ}{dP}=Z_T$. **Girsanov's theorem** says that under this new measure $Q$, the process

$$\widetilde{W}_t = W_t + \theta t$$

is a standard Brownian motion.

That is the whole theorem. Let's translate it.

### What every symbol means

- $W_t$ is the original Brownian motion — pure noise, zero drift, under the real world P.
- $\theta$ is the amount of drift we are about to *inject*. It is the dial. Choosing $\theta$ is choosing which new world to move to.
- $Z_t = \exp(-\theta W_t - \frac{1}{2}\theta^2 t)$ is the reweighting density. The exponential shape is not arbitrary: it is the unique form that keeps $Z_t$ positive and keeps its expectation at 1. The $-\frac{1}{2}\theta^2 t$ term is a normalization correction — it is exactly the amount needed to cancel the upward bias that the exponential of a Gaussian would otherwise have. (This is the same $-\frac{1}{2}\sigma^2$ correction that appears in the log of geometric Brownian motion, for the same Itô-calculus reason.)
- $\widetilde{W}_t = W_t + \theta t$ is the *new* Brownian motion, the one that is driftless when viewed through the Q lens.

Here is the part people find magical, so let's sit with it. Under P, the process $W_t + \theta t$ obviously has drift — it is a Brownian motion plus a steady ramp $\theta t$, so its expectation grows at rate $\theta$. It is *not* a Brownian motion under P, because Brownian motions have zero drift. But Girsanov says: change the probabilities to Q using exactly the weight $Z_t$, and that same process becomes driftless. The drift did not get subtracted off by any algebra. It got *reweighted away*. By making the up-paths a little less likely and the down-paths a little more likely (in just the right exponential proportion), the average ramp cancels, and what looked like a drifting process now looks, statistically, like pure noise.

> Girsanov does not change the path. The exact same wiggly line is still there. It changes the probabilities you attach to that line, and that is enough to make the drift vanish from every expectation you compute.

### Why a constant drift needs a Gaussian-shaped weight

It is worth understanding why the weight has to be exponential-in-$W_t$ and not something simpler. We want to tilt a Gaussian by shifting its mean. If $X\sim N(0,t)$ under P and we want it to be $N(-\theta t, t)$ under Q (a mean shifted down by $\theta t$, which is exactly what makes $X+\theta t$ mean-zero), the ratio of the two Gaussian densities is

$$\frac{q(x)}{p(x)} = \frac{\exp\!\big(-(x+\theta t)^2/2t\big)}{\exp\!\big(-x^2/2t\big)} = \exp\!\left(-\theta x - \tfrac{1}{2}\theta^2 t\right).$$

That ratio, evaluated at $x=W_t$, is exactly $Z_t$. So the seemingly mysterious exponential density is nothing but the likelihood ratio between two Gaussians whose means differ by $\theta t$. The change of measure is a Gaussian mean-shift, dressed in continuous-time clothing. The two-scenario example earlier was the discrete shadow of this; Girsanov is the continuous version where the "two scenarios" become a continuum of Brownian paths.

## 4. Pricing the drift: the market price of risk

We now connect Girsanov's abstract dial $\theta$ to a concrete, tradeable quantity. This is where the theorem stops being pure math and starts pricing options.

### Choosing theta to kill the right drift

Recall the real-world stock dynamics:

$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t.$$

We want a new measure Q under which the stock drifts at the risk-free rate $r$ instead of $\mu$. So we want to remove exactly $\mu - r$ of drift. The question is: what value of $\theta$ does that?

Substitute the Girsanov relationship $W_t = \widetilde{W}_t - \theta t$, equivalently $dW_t = d\widetilde{W}_t - \theta\,dt$, into the SDE:

$$dS_t = \mu S_t\,dt + \sigma S_t\,(d\widetilde{W}_t - \theta\,dt) = (\mu - \sigma\theta)\,S_t\,dt + \sigma S_t\,d\widetilde{W}_t.$$

For the new drift $\mu - \sigma\theta$ to equal $r$, we need:

$$\mu - \sigma\theta = r \quad\Longrightarrow\quad \theta = \frac{\mu - r}{\sigma}.$$

This special value of the dial has a name. It is the **market price of risk**, written $\lambda$:

$$\boxed{\;\lambda = \frac{\mu - r}{\sigma}\;}$$

Look closely at this quantity. The numerator $\mu - r$ is the stock's **excess return** — how much more than the risk-free rate it earns. The denominator $\sigma$ is its volatility. The ratio is the excess return *per unit of volatility* — which is precisely the definition of the **Sharpe ratio**. The market price of risk is the asset's Sharpe ratio. It tells you how much extra return the market is paying per unit of risk taken, and it is exactly the amount of drift Girsanov strips out when moving from P to Q. The figure below assembles it piece by piece.

![Stack building the market price of risk from excess return divided by volatility](/imgs/blogs/girsanov-change-of-measure-math-for-quants-7.png)

Read top to bottom: start with the excess return $\mu - r$, divide by the volatility $\sigma$, and you get the market price of risk $\lambda$ — which is exactly the drift that Girsanov removes when it builds the risk-neutral world. Every layer of that stack is a quantity a desk can measure.

#### Worked example: computing the market price of risk

Let's use the numbers from the roadmap. A stock has an expected return of $\mu=12\%$, the risk-free rate is $r=4\%$, and the stock's volatility is $\sigma=20\%$. The market price of risk is:

$$\lambda = \frac{\mu - r}{\sigma} = \frac{0.12 - 0.04}{0.20} = \frac{0.08}{0.20} = 0.40.$$

So $\lambda = 0.40$. In words, this stock pays 0.40 units of excess return for every unit of volatility you take on — a Sharpe ratio of 0.4, which is a typical, healthy number for a broad equity index over the long run.

Now here is the bridge to pricing. When a quant moves to the risk-neutral world, the drift they remove is $\sigma\lambda = 0.20\times 0.40 = 0.08 = 8\%$. That is exactly $\mu - r = 12\% - 4\% = 8\%$. The 8% real-world risk premium is the precise quantity that Girsanov deletes. Under P the stock drifts at 12%; under Q it drifts at 4%; the 8% difference is the market price of risk multiplied by volatility, and it has been reweighted into oblivion.

To make the dollars concrete: a \$100 share expected to grow at 12% is expected to be worth about \$112.75 in a year under P (using $100\,e^{0.12}$). The same share under Q is expected to be worth only $100\,e^{0.04}\approx\$104.08$. The \$8.67 gap between \$112.75 and \$104.08 is the dollar value of that 8% risk premium that vanishes when we change measure. The one-sentence intuition: the market price of risk is the Sharpe ratio, and it is the single number Girsanov subtracts to convert real-world growth into risk-free growth.

### Why only volatility, not expected return, prices an option

We can now answer the question the article opened with. Look back at the risk-neutral stock dynamics we derived:

$$dS_t = r S_t\,dt + \sigma S_t\,d\widetilde{W}_t.$$

The expected return $\mu$ is *gone*. It does not appear. The only parameters left are $r$ (the risk-free rate, set by the central bank, the same for everyone) and $\sigma$ (the stock's volatility). And since an option's price is its discounted expected payoff *under Q*, and Q's dynamics contain no $\mu$, the option price cannot possibly depend on $\mu$.

This is the rigorous version of a fact that confuses every newcomer. Two stocks with the same volatility and the same current price have the same option prices, no matter how wildly their expected returns differ. The market's opinion about whether a stock will soar or stagnate is a statement about $\mu$, and $\mu$ has been changed-of-measure away. What survives is $\sigma$, the size of the random shake — because the random shake is what a hedger has to manage, and it is the cost of hedging that an option's price ultimately reflects. The expected direction is irrelevant to the cost of insurance against the wiggle.

> An option is insurance against the *size* of the move, not the *direction* of it. That is why its price lives on volatility and is blind to expected return.

## 5. Deriving the risk-neutral GBM, step by step

Let's put the whole derivation in one place, from real-world dynamics to risk-neutral dynamics, so you can see there is no sleight of hand. This is the canonical application of Girsanov, and it is short.

### The derivation

**Step 1. Start with the real-world model.** The stock follows geometric Brownian motion under P:

$$dS_t = \mu S_t\,dt + \sigma S_t\,dW_t.$$

**Step 2. Define the market price of risk.** We computed it above:

$$\lambda = \frac{\mu - r}{\sigma}.$$

**Step 3. Apply Girsanov.** Define the density $Z_t = \exp(-\lambda W_t - \frac{1}{2}\lambda^2 t)$ and the new measure $Q$ via $\frac{dQ}{dP}=Z_T$. By Girsanov, the process

$$\widetilde{W}_t = W_t + \lambda t$$

is a Brownian motion under $Q$. Equivalently, $dW_t = d\widetilde{W}_t - \lambda\,dt$.

**Step 4. Substitute.** Plug into the Step-1 SDE:

$$dS_t = \mu S_t\,dt + \sigma S_t\,(d\widetilde{W}_t - \lambda\,dt) = (\mu - \sigma\lambda)\,S_t\,dt + \sigma S_t\,d\widetilde{W}_t.$$

**Step 5. Simplify the drift.** Since $\sigma\lambda = \sigma\cdot\frac{\mu-r}{\sigma} = \mu - r$, the drift becomes $\mu - (\mu - r) = r$. So:

$$\boxed{\;dS_t = r S_t\,dt + \sigma S_t\,d\widetilde{W}_t\;}$$

That is the risk-neutral geometric Brownian motion. The drift is now the risk-free rate; the volatility is untouched; the Brownian motion has been swapped for its risk-neutral cousin. Five lines, no arbitrage assumed beyond the existence of $Q$, and the famous result that "the stock grows at $r$ under the pricing measure" is yours.

### What the picture looks like

Before and after Girsanov, the *cloud* of possible stock paths looks almost identical — same volatility, same fan-out — but its center of mass has been pulled down from a $\mu$-slope to an $r$-slope. The comparison matrix below lays out exactly what changed and what stayed the same between the two worlds.

![Matrix comparing real measure P and pricing measure Q on drift use probabilities volatility and expected price](/imgs/blogs/girsanov-change-of-measure-math-for-quants-4.png)

The columns are the two measures. Row by row: the drift changes from $\mu$ to $r$; the *use* changes from forecasting and risk management to pricing and hedging; the probabilities are reweighted; the volatility $\sigma$ is identical in both; and the expected stock price drops from a $\mu$-growth to an $r$-growth. The two things that survive the change of measure — volatility and the set of possible outcomes — are the two things every pricing model genuinely needs.

#### Worked example: real-world versus risk-neutral expected price, and why the option uses Q

Let's price something with both worlds and watch the difference. A stock trades at \$100. Its real expected return is $\mu=12\%$, volatility $\sigma=20\%$, and the risk-free rate is $r=4\%$. We want to price a one-year European call option struck at \$100 (the right, but not the obligation, to buy the stock for \$100 in one year).

First, the *wrong* way — pricing under P. If you used the real-world growth rate, you would simulate the stock drifting up at 12%, ending on average around \$112.75, and conclude the call is fairly likely to finish well in the money. You would compute its expected payoff under P, discount it, and arrive at some number — call it the "P-price". The problem: that number admits an arbitrage. A trader could hedge the option dynamically and lock in a riskless profit against your mispriced quote. The P-price is simply wrong.

Now the *right* way — pricing under Q. Under Q the stock drifts at only $r=4\%$, ending on average around \$104.08. The call's expected payoff is smaller (the stock isn't expected to climb as high), and we discount at $r$. Plugging $S_0=100$, $K=100$, $r=0.04$, $\sigma=0.20$, $T=1$ into Black-Scholes gives a call price of about \$9.41. That \$9.41 is the *only* arbitrage-free price.

The dollar gap between the two approaches is real money. The P-price would come out meaningfully higher — perhaps \$12 to \$13 depending on assumptions — because it bakes in the stock's optimistic drift. The market would never pay that; the market pays \$9.41. The roughly \$3 difference is the value of a risk premium that does not belong in an option price, because the option buyer can hedge the directional risk away and is therefore not entitled to be paid for it. The one-sentence intuition: option prices must use Q because only Q produces a price no trader can arbitrage against, and the dollars you would lose by using P are exactly the mispriced risk premium.

### Checking the martingale property by hand

A claim is only as good as the test you can run against it, so let us verify the headline promise directly: that under Q the *discounted* stock price has no drift. Discounting means dividing by the growth of a risk-free bank account, which after time $t$ has grown a deposit by the factor $e^{rt}$. Define the discounted stock price $\widetilde S_t = e^{-rt}S_t$. If Q is doing its job, then $\widetilde S_t$ should be a martingale — its expected future value, under Q, should equal its value today.

Under the risk-neutral dynamics $dS_t = rS_t\,dt + \sigma S_t\,d\widetilde W_t$, the solution is the lognormal form $S_t = S_0\exp\!\big((r-\tfrac12\sigma^2)t + \sigma\widetilde W_t\big)$. Take the expectation under Q. The only random piece is $\widetilde W_t$, which is $N(0,t)$ under Q, and the expectation of $e^{\sigma\widetilde W_t}$ for a normal variable is $e^{\frac12\sigma^2 t}$ (the moment-generating function of the Gaussian). So:

$$E^Q[S_t] = S_0\,e^{(r-\frac12\sigma^2)t}\cdot e^{\frac12\sigma^2 t} = S_0\,e^{rt}.$$

The two halves of the $\frac12\sigma^2 t$ term cancel exactly, and what remains is clean: the stock is expected to grow at precisely the risk-free rate. Discount it and the growth disappears entirely:

$$E^Q[\widetilde S_t] = e^{-rt}E^Q[S_t] = e^{-rt}\cdot S_0 e^{rt} = S_0.$$

The discounted price's expected future value equals today's price. That is the martingale property, confirmed by direct calculation. This is no accident — it is the whole reason we chose $\lambda=\frac{\mu-r}{\sigma}$ as the dial. Any other choice of $\theta$ in Girsanov would leave a residual drift and break the martingale property, reintroducing an arbitrage. The market price of risk is the unique tilt that makes discounted prices fair.

#### Worked example: the two-period binomial bridge to Girsanov

To see the continuous theorem as the limit of something countable, let's run the two-state model for two periods and watch the risk-neutral weights compound — this is exactly how Girsanov's exponential density arises in the limit. A stock starts at \$100. Each period it either rises 10% (to a factor of 1.10) or falls 10% (to a factor of 0.90). The risk-free rate is 2% per period. The single-period risk-neutral up-probability is the value that makes the stock grow at the risk-free rate:

$$q = \frac{e^{r} - d}{u - d} = \frac{e^{0.02} - 0.90}{1.10 - 0.90} = \frac{1.0202 - 0.90}{0.20} = \frac{0.1202}{0.20} = 0.601.$$

So $q\approx 0.60$ up, 0.40 down each period — independent of the real-world probability, which might be 70/30. After two periods there are three terminal prices: up-up gives $\$100\times1.10\times1.10 = \$121$, up-down or down-up gives $\$100\times1.10\times0.90 = \$99$, and down-down gives $\$100\times0.90\times0.90 = \$81$. Their risk-neutral probabilities multiply across periods: $q^2 = 0.361$ for \$121, $2q(1-q)=0.480$ for \$99, and $(1-q)^2 = 0.159$ for \$81.

Check the martingale: the discounted expected terminal price should be \$100.

$$e^{-2r}\big(0.361\times\$121 + 0.480\times\$99 + 0.159\times\$81\big) = e^{-0.04}\times\$104.08 = \$100.00.\ \checkmark$$

The Radon-Nikodym weight on each path is the ratio of its Q-probability to its P-probability, and across $n$ periods it is a product of $n$ single-period ratios. As you shrink the period length and let $n\to\infty$, that product of ratios converges — by the central limit theorem operating in the exponent — to exactly $\exp(-\lambda W_T - \frac12\lambda^2 T)$, the Girsanov density. The one-sentence intuition: Girsanov's exponential weight is nothing more than infinitely many binomial reweightings multiplied together, which is why the discrete and continuous pictures tell the identical story.

## 6. The same trick in reverse: importance sampling in Monte Carlo

Everything so far moved from P to Q to *price* things. The exact same change-of-measure machinery, run for a completely different reason, is one of the most powerful variance-reduction techniques in computational finance: **importance sampling**. This is where the abstract theorem earns its keep on a real desk's compute budget.

### The problem importance sampling solves

Many derivatives pay off only in rare scenarios. A deep out-of-the-money option — say a call struck at \$200 on a stock trading at \$100 — pays nothing unless the stock more than doubles, which almost never happens in a year. If you price it with naive Monte Carlo (simulate thousands of stock paths, average the payoffs, discount), the overwhelming majority of your simulated paths finish below \$200 and contribute a payoff of exactly zero. You are spending almost all your compute on paths that tell you nothing. Only a tiny handful of paths ever reach the payoff region, and your estimate of the price wobbles wildly depending on how many of those rare paths you happened to draw.

The figure below contrasts the naive approach with the importance-sampled one.

![Before and after comparison of naive Monte Carlo versus importance-sampled Monte Carlo for a deep out of the money option](/imgs/blogs/girsanov-change-of-measure-math-for-quants-6.png)

On the left, naive sampling: most paths expire worthless, a few pay off, and the estimator variance is huge because everything depends on those few lucky draws. On the right, importance sampling: we deliberately shift the drift so that most paths *do* reach the strike, then correct for the cheating with a likelihood-ratio weight. Most paths now contribute real information, and the estimator variance collapses.

### The mechanism: sample under a tilted measure, reweight by the density

The idea is exactly Girsanov, used as a computational tool. We want $E^Q[\text{Payoff}]$. Instead of sampling from Q, where the payoff is rare, we sample from a *tilted* measure $\widehat{Q}$ that pushes the stock's drift up toward the strike so the payoff is common. But sampling from the wrong measure would bias the answer — so we correct each sample by multiplying it by the Radon-Nikodym derivative $\frac{dQ}{d\widehat{Q}}$, the likelihood ratio between the measure we *want* and the measure we *sampled from*. Formally:

$$E^Q[\text{Payoff}] = E^{\widehat{Q}}\!\left[\text{Payoff}\cdot \frac{dQ}{d\widehat{Q}}\right].$$

This identity is exact — no approximation. The reweighting undoes the tilt perfectly *in expectation*. What changes is the *variance*: by sampling where the action is, the payoff-times-weight quantity is far more uniform across paths, so its average converges much faster. You get the same answer with a fraction of the simulations.

The tilt itself is a Girsanov shift: add a constant drift $b$ to the Brownian motion, and the likelihood-ratio weight is precisely the Girsanov density $\exp(-b\widetilde{W}_T - \frac{1}{2}b^2 T)$ evaluated on each path. The theorem that built the pricing world also builds the importance-sampling estimator. One mathematical object, two jobs.

#### Worked example: importance sampling a deep out-of-the-money call

Let's quantify the win. Stock at \$100, volatility $\sigma=20\%$, risk-free rate $r=4\%$, one year. We price a deep out-of-the-money call struck at $K=\$200$ — the stock must more than double to pay anything.

**Naive Monte Carlo.** Under Q the stock drifts at 4%, so the chance of finishing above \$200 is tiny — on the order of 0.5%. If you run 10,000 paths, only about 50 finish in the money. Suppose the true price is around \$0.12. Your estimate from 50 nonzero paths is extremely noisy: the standard error might be 30-40% of the price itself. To get the standard error down to, say, 1% of the price, you would need on the order of *millions* of paths, because Monte Carlo error shrinks only as $1/\sqrt{N}$ and you are starting from a 0.5% hit rate.

**Importance-sampled Monte Carlo.** Now shift the sampling drift so the stock is expected to land right around \$200 — a Girsanov tilt of roughly $b\approx 3.5$ in the relevant units, enough to push the expected log-return up to $\ln(200/100)=0.69$. Now perhaps 50% of paths finish in the money instead of 0.5%. Each in-the-money path is multiplied by its likelihood-ratio weight (small, because these paths are rare under the true Q, but nonzero and informative). With the same 10,000 paths, you now have about 5,000 informative draws instead of 50 — a hundredfold increase in useful samples.

The variance reduction is dramatic. In practice, importance sampling on a deep out-of-the-money option of this kind routinely cuts the estimator variance by a factor of 50 to 100, meaning you reach the same accuracy with 50 to 100 times fewer paths. If naive pricing needed 5 million paths and 50 seconds of compute to nail the price to within a penny, importance sampling gets there in roughly 100,000 paths and under a second. The one-sentence intuition: importance sampling is Girsanov used as a flashlight — you shine your simulations where the payoff actually lives, then divide by the likelihood ratio to keep the answer honest. We cover the broader Monte Carlo toolkit, including antithetic variates and control variates, in the [Monte Carlo simulation post](/blog/trading/quantitative-finance/monte-carlo-simulation-coding-quant-interviews).

### A reproducible sketch in Python

Here is runnable pseudocode-grade Python showing the exact mechanics. It is honest about every step.

```python
import numpy as np

S0, K, r, sigma, T = 100.0, 200.0, 0.04, 0.20, 1.0
N = 100_000
rng = np.random.default_rng(0)

def terminal_price(noise):
    """Terminal stock price under risk-neutral GBM given standard-normal noise."""
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * noise)

Z = rng.standard_normal(N)                     # naive: draw under Q (drift = r)
payoff = np.maximum(terminal_price(Z) - K, 0.0)
price_naive = np.exp(-r * T) * payoff.mean()
se_naive = np.exp(-r * T) * payoff.std() / np.sqrt(N)

b = (np.log(K / S0) - (r - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))  # tilt to strike
Zt = rng.standard_normal(N) + b                # sample under the shifted measure
weight = np.exp(-b * Zt + 0.5 * b**2)          # Girsanov likelihood ratio dQ/dQhat
payoff_t = np.maximum(terminal_price(Zt) - K, 0.0) * weight
price_is = np.exp(-r * T) * payoff_t.mean()
se_is = np.exp(-r * T) * payoff_t.std() / np.sqrt(N)

print(f"naive: {price_naive:.4f} +/- {se_naive:.4f}")
print(f"IS:    {price_is:.4f} +/- {se_is:.4f}")
```

Run it and you will see both prices agree on roughly the same small dollar value, but the importance-sampled standard error is many times smaller than the naive one — the same accuracy from the same number of paths, because we sampled where the payoff lives and then divided by the Girsanov weight to stay unbiased.

## 7. Putting the whole toolkit on one map

We have now seen the change of measure do three distinct jobs: define equivalent measures, build the risk-neutral pricing world, and accelerate Monte Carlo. They are not three separate tricks — they are one idea wearing three hats. The tree below organizes them.

![Tree of change of measure concepts branching into equivalent measures pricing and importance sampling](/imgs/blogs/girsanov-change-of-measure-math-for-quants-5.png)

At the root sits the single idea: change of measure, reweighting outcomes without changing what is possible. It branches into three. **Equivalent measures** is the licensing condition, and it splits into the Radon-Nikodym weight (the discrete dial) and the Girsanov drift shift (the continuous version). **Risk-neutral pricing** is the headline application, and it rests on the market price of risk. **Importance sampling** is the computational application, and it buys you variance reduction on rare events. One theorem, three uses, all on the same page.

### A comparison table to lock it in

| Aspect | Real measure P | Pricing measure Q | Tilted measure (importance sampling) |
| --- | --- | --- | --- |
| Stock drift | $\mu$ (true return) | $r$ (risk-free) | shifted toward the payoff |
| Used for | forecasting, risk | pricing, hedging | fast simulation |
| Reweighting | none (the baseline) | by $\lambda=\frac{\mu-r}{\sigma}$ | by a chosen tilt $b$ |
| Volatility | $\sigma$ | $\sigma$ (unchanged) | $\sigma$ (unchanged) |
| Outcomes possible | all paths | the same paths | the same paths |
| The weight | — | $\exp(-\lambda W_T - \tfrac12\lambda^2T)$ | $\exp(-b\widetilde W_T - \tfrac12 b^2T)$ |

The table makes the unity obvious: every column keeps the same outcomes and the same volatility, and every column differs only in the drift it imposes and the exponential weight it uses to get there. That is the change of measure in one line — keep the wiggle and the possibilities, move the drift, pay for it with a likelihood ratio.

## Common misconceptions

**"The risk-neutral probabilities are what the market actually believes."** No. Nobody believes the down-state is more likely just because we are pricing an option. The Q-probabilities are a computational fiction — a reweighting chosen so that discounted prices have no drift and no arbitrage exists. They are real-world probabilities bent by the market price of risk. The real probabilities (the ones you would use to forecast, or to size a bet) live under P, and they are usually quite different. Confusing Q for a forecast is the single most common and most dangerous error: it will make you systematically misjudge how likely an option is to pay off.

**"Changing the measure changes the stock price."** It does not. The change of measure changes the *probabilities* attached to each path, never the path itself or the spot price. A \$100 stock is \$100 under P and under Q. What changes is the *expected* future value, because expectation is a probability-weighted average and you reweighted the probabilities. The present price, being observable in the market, is the same number in every measure.

**"Girsanov changes the volatility too."** This is perhaps the most important thing to get right, and it is precisely backwards. Girsanov changes *only the drift*; the volatility $\sigma$ is invariant under an equivalent change of measure. This is a deep fact, sometimes phrased as "volatility is observable, drift is not". You can shift the mean of a Gaussian by reweighting, but you cannot shrink or grow its spread that way without making some outcomes impossible — which would break equivalence. This is exactly why a stock's *realized volatility* is something you can estimate from data and trust across measures, while its drift is notoriously hard to pin down: the drift is the part the change of measure is free to move.

**"You must believe in risk-neutrality to use the risk-neutral measure."** You do not. The risk-neutral measure is a no-arbitrage *bookkeeping device*, not a statement about anyone's risk preferences. Even a wildly risk-averse desk full of pessimists prices options under Q, because Q is the unique measure that makes prices consistent. Q is about the *market's* implied pricing, not about the modeler's psychology.

**"The market price of risk is a property of the option."** It is a property of the *underlying asset and the market*, not the derivative. The Sharpe ratio $\lambda=\frac{\mu-r}{\sigma}$ belongs to the stock. Every option on that stock inherits the same $\lambda$ for the purpose of the measure change. Two different options on the same stock are priced under the same Q built with the same $\lambda$.

**"Importance sampling changes the answer."** It must not, and a correct implementation does not. The likelihood-ratio weight exactly cancels the bias introduced by sampling from the tilted measure, so the *expectation* is unchanged — only the *variance* drops. If your importance-sampled price disagrees with your naive price beyond Monte Carlo noise, you have a bug in the weight, not a feature. The classic bug is forgetting the $-\frac12 b^2 T$ normalization in the Girsanov density.

## How it shows up in real markets

### 1. Every option screen on every trading desk

When you pull up an option chain and see implied volatilities quoted, you are looking at the output of risk-neutral pricing. Traders quote vol, not price, precisely because under Q the only free parameter is $\sigma$ — the expected return has been changed-of-measure away. The whole convention of trading options *in vol terms* is a direct, daily, multi-trillion-dollar consequence of the fact that Girsanov removes the drift. A desk marking thousands of options never inputs a view on whether the stock will rise; it inputs a vol surface, because that is the only thing Q cares about.

### 2. The 1987 crash and the birth of the volatility smile

Before October 1987, options were priced with a flat volatility across strikes, consistent with the simplest geometric-Brownian-motion-under-Q model. On Black Monday the S&P 500 fell about 20% in a single day — an event so far in the tail that the lognormal Q-model said it should essentially never happen. Afterward, the market permanently repriced deep out-of-the-money puts higher, creating the **volatility smile**: implied vol rises for far-from-the-money strikes. The lesson in our language: the simple Girsanov change of measure assumes constant volatility, but real markets have fat tails and jumps that a single $\sigma$ cannot capture. The smile is the market's correction to a too-simple measure change, and it is why modern desks use more elaborate models (stochastic vol, jumps) whose change of measure is far richer than the textbook constant-$\lambda$ case.

### 3. Pricing deep out-of-the-money catastrophe and tail hedges

Funds that buy crash protection — far out-of-the-money index puts, sometimes 30% or more below spot — face exactly the rare-event Monte Carlo problem importance sampling solves. A naive simulation of a 1-in-1000 payoff would need millions of paths to price stably. Tail-risk desks routinely use Girsanov-tilted importance sampling to value these positions in seconds rather than minutes, shifting the simulated drift sharply downward so the catastrophe scenarios become common in the simulation, then dividing by the likelihood ratio. The same math that prices a vanilla call powers the pricing of the very insurance that pays out when the call's model fails.

### 4. Counterparty risk and CVA: changing measure on a portfolio

After 2008, banks were required to price the risk that their *counterparty* defaults — the credit valuation adjustment, or CVA. Computing CVA means simulating the future value of an entire derivatives portfolio across thousands of scenarios and discounting expected losses. Because large losses are rare, importance sampling under a Girsanov tilt is standard practice on CVA desks: tilt the scenarios toward the states where the counterparty defaults *and* the portfolio is deep in the money, then reweight. A calculation that would otherwise take an overnight grid run can be brought down to something interactive, which matters when traders need a CVA charge quoted before they agree a trade.

### 5. Foreign exchange and the two-currency change of measure

In FX, there are two risk-free rates — one per currency — and pricing a currency option requires a change of measure between the "domestic" and "foreign" risk-neutral worlds. This is Girsanov applied across currencies: the drift adjustment that converts a forward from one currency's measure to the other's is governed by the interest-rate differential, the FX analogue of $\mu-r$. The famous result that an option to buy euros with dollars is the same as a (suitably scaled) option to sell dollars for euros — Siegel's paradox and its resolution — is a pure change-of-measure statement. FX desks live in this two-measure world every day. A concrete number makes it real: if dollar rates are 5% and euro rates are 3%, the 2% rate differential is the drift adjustment that Girsanov applies when converting a forward price from the euro measure to the dollar measure, and it shows up directly in the forward exchange rate through covered interest parity. On a \$50 million forward, getting that 2% adjustment wrong by even a tenth of a point misprices the trade by tens of thousands of dollars, which is why the change of measure here is not an academic nicety but a hard P&L constraint.

### 6. Real-world versus risk-neutral default probabilities in credit

Credit markets show the P-versus-Q gap in its starkest form. The default probability *implied* by a corporate bond's credit spread (a Q-probability) is routinely several times higher than the *actual* historical default rate for bonds of that rating (a P-probability). A single-A bond might have a Q-implied default probability of 2% but a historical P-default rate of 0.5%. The 4x gap is the credit risk premium — investors demand extra yield for bearing default risk, and that premium inflates the Q-probability above the P-probability. It is the credit-market version of our two-scenario example, where the down-state got heavier under Q. Reading a credit spread as a forecast of default (using Q as P) overstates real default risk dramatically.

### 7. Quanto products and the drift correction nobody sees

A *quanto* is a derivative whose payoff is in one currency but references an asset priced in another — for example, a contract that pays in US dollars based on the level of Japan's Nikkei index, with the exchange rate fixed in advance. Pricing it correctly requires a change of measure that produces a subtle, easy-to-miss extra drift term called the **quanto adjustment**. When you move the Nikkei's dynamics into the dollar risk-neutral world, Girsanov's machinery adds a correction equal to the correlation between the index and the exchange rate, times their two volatilities. If the Nikkei and the yen-dollar rate are 30% correlated, with index vol 20% and FX vol 10%, the drift correction is $0.30\times0.20\times0.10 = 0.006$, or 0.6% per year — small, but on a \$100 million notional it is \$600,000 a year of value that a naive single-currency model would silently get wrong. Quanto desks compute this adjustment as a routine consequence of changing measure across currencies, and it is one of the cleanest real-world examples of Girsanov producing a *new* drift term rather than just removing one. The lesson: changing measure does not always merely delete the risk premium; when correlations are in play, it can install a correction that materially moves the price.

## When this matters to you

If you ever build a pricing model, the change of measure is the difference between a number traders will trade on and a number that contains a silent arbitrage. Use P and you bake in a return forecast you have no business charging for; use Q and your price is the unique no-arbitrage one. If you run Monte Carlo for anything rare — tail hedges, exotic payoffs, stress scenarios — importance sampling via a Girsanov tilt is the standard route to making the computation tractable, and the single most common bug is a wrong likelihood-ratio weight, so check that your weight has the $-\frac12 b^2 T$ term.

If you are an investor rather than a quant, the practical takeaway is subtler but just as useful: never read an option's implied probabilities, or a credit spread's implied default rate, as a *forecast*. They are Q-probabilities, bent by a risk premium, and they will systematically overstate bad outcomes relative to what history suggests. The gap between the price-implied (Q) and the historically-estimated (P) view is itself a tradeable quantity — it is precisely the risk premium you collect for selling insurance, and harvesting it is the entire business of volatility selling and credit carry. None of this is advice to trade anything; it is the mechanism behind why those trades exist and how they can both pay and blow up.

The one number to carry away: the market price of risk $\lambda=\frac{\mu-r}{\sigma}$. It is the Sharpe ratio, it is the drift Girsanov removes, and it is the precise bridge between the world as it is and the world we price in.

### Further reading

- [Risk-neutral pricing and the martingale measure](/blog/trading/quantitative-finance/risk-neutral-pricing-martingale-measure-quant-interviews) — the companion that shows *why* the measure Q must exist and what a martingale is.
- [Stochastic differential equations: GBM, OU, and CIR](/blog/trading/quantitative-finance/stochastic-differential-equations-gbm-ou-quant-interviews) — the SDE machinery this post applies, with mean-reversion and short-rate models.
- [Brownian motion from the random walk](/blog/trading/quantitative-finance/brownian-motion-quant-interviews) — where $W_t$ comes from, built up from a coin-flip walk.
- [Monte Carlo simulation for pricing](/blog/trading/quantitative-finance/monte-carlo-simulation-coding-quant-interviews) — the broader variance-reduction toolkit that importance sampling belongs to.
- John Hull, *Options, Futures, and Other Derivatives* — the standard reference; the chapters on the risk-neutral valuation and the Black-Scholes-Merton differential equation cover this material at textbook pace.
- Steven Shreve, *Stochastic Calculus for Finance II* — the rigorous treatment of Girsanov's theorem and the Radon-Nikodym derivative in continuous time.

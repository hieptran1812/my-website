---
title: "Feynman-Kac and the Black-Scholes PDE: the bridge between solving equations and averaging payoffs"
date: "2026-06-15"
description: "A beginner-friendly, build-from-zero tour of how the Feynman-Kac theorem connects a pricing PDE to an expectation over price paths, how the Black-Scholes PDE is born from delta-hedging and Ito's lemma, and how the Greeks are just the terms of that equation."
tags: ["feynman-kac", "black-scholes", "pde", "delta-hedging", "greeks", "theta-gamma", "monte-carlo", "finite-differences", "option-pricing", "quant-finance", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The Feynman-Kac theorem says that the solution of a certain kind of differential equation is exactly equal to an average of payoffs taken over randomly wandering price paths, so the price of an option can be found two completely different ways and they give the same number.
>
> - There are **two roads to one price**: solve the **Black-Scholes PDE** (a deterministic equation on a grid) or compute a **discounted expectation** over simulated random price paths (Monte Carlo). Feynman-Kac proves the roads meet.
> - The Black-Scholes PDE, $V_t + \tfrac12\sigma^2 S^2 V_{SS} + rS V_S - rV = 0$, falls out of a simple trick: **sell an option, hold delta shares against it**, and use Ito's lemma — the randomness cancels and the leftover must earn the risk-free rate.
> - The **Greeks are the terms of the PDE**: theta is $V_t$ (time decay), gamma is $V_{SS}$ (curvature), delta is $V_S$ (sensitivity to spot). The equation literally says **theta pays for gamma**.
> - The **payoff is the boundary condition**. Change the payoff at expiry, keep the same PDE, and you price any European derivative — calls, puts, digitals, all of them.
> - The one number to remember: a 3-month, at-the-money \$100 call with 20% volatility and a 5% rate is worth about **\$4.62**, and you can get that exact \$4.62 from the formula, from a finite-difference grid, or from averaging a million simulated payoffs.

Here is a fact that sounds impossible the first time you hear it. There is a differential equation — a single line of calculus — whose answer is the fair price of a stock option. And there is a completely separate procedure — simulate thousands of random futures for the stock, see what the option pays off in each, average the results, and discount them back to today — that gives the *same* price. Two methods that look like they belong to different universes, one from the world of physics and heat flow, the other from the world of dice and casinos, land on the identical dollar figure.

The theorem that guarantees they agree is named after Richard Feynman and Mark Kac, and it is one of the most useful bridges in all of quantitative finance. It tells you that whenever you can write down a pricing problem as "the expected value of some future payoff," you can also write it as a differential equation — and vice versa. That freedom is enormous in practice. Some options are easy to price by solving an equation and miserable to simulate; others are the reverse. Feynman-Kac lets a quant pick whichever road is easier and trust that the destination is the same.

![Before and after panels showing the PDE route and the Monte Carlo route arriving at the same price today](/imgs/blogs/feynman-kac-black-scholes-pde-math-for-quants-1.png)

The diagram above is the mental model for this entire post. On the left is the PDE road: you write down the Black-Scholes equation, lay out a grid of prices and times, and step the solution backward from expiry to today. On the right is the expectation road: you simulate many random paths the stock might take, compute the option's payoff at the end of each, discount each payoff back to the present, and average them. The Feynman-Kac theorem is the claim that these two answers are not merely close — they are the same number, exactly, in the limit. By the end of this article you will be able to walk both roads by hand for a real option, see why the Greeks are just the pieces of the PDE, and know when a working quant reaches for one road over the other. Let us start from absolute zero.

## Foundations: the building blocks

Before any of the heavy machinery, we need to agree on a small vocabulary. Every term below is defined the first time it appears, and we build the simplest possible version of each idea before stacking on realism. If you already know what an option, a derivative, and a partial derivative are, you can skim; if you do not, you can still follow every step.

### What is an option?

An *option* is a contract that gives you the right, but not the obligation, to buy or sell a stock at a fixed price on a fixed future date. A *call option* gives you the right to **buy**; a *put option* gives you the right to **sell**. The fixed price is the *strike price*, written $K$. The fixed date is the *expiry* or *maturity*, written $T$. The stock you have the right to trade is the *underlying*, and its price right now is the *spot price*, written $S$ (or $S_0$ for "spot today").

Take a concrete case. Suppose a stock trades at \$100 today and you hold a call option with strike \$100 expiring in three months. If the stock is at \$110 at expiry, you exercise: you buy at \$100 and immediately sell at \$110, pocketing \$10. If the stock is at \$90 at expiry, you simply walk away — you would never choose to buy at \$100 something worth \$90 — and the option expires worthless. So a call's payoff at expiry is "the stock price minus the strike, but never less than zero." In symbols:

$$ \text{call payoff} = \max(S_T - K,\ 0). $$

Here $S_T$ is the stock price at expiry $T$, $K$ is the strike, and $\max(\cdot,0)$ means "take the bigger of this and zero." A put is the mirror image: $\max(K - S_T,\ 0)$, because a put lets you sell at $K$, which is only worth doing when the stock has fallen below $K$.

We call these *European* options because they can only be exercised on the expiry date, not before. (The "European" label has nothing to do with geography; *American* options, which can be exercised any time before expiry, are a harder problem we will mostly set aside.) For this entire post, "option" means a European option, and our running question is: **what is the fair price to pay for one today?**

### What is a derivative (the math kind)?

The word *derivative* is overloaded, and both meanings matter here. In finance, a *derivative* is any contract whose value is *derived* from something else — an option is a derivative because its value depends on the stock. In calculus, a *derivative* measures how fast one quantity changes as another changes. We will need both. To avoid confusion, when we mean the calculus kind we will say "partial derivative" or write the symbols.

A *partial derivative* is just a slope when everything but one variable is held fixed. If the option's value $V$ depends on the spot price $S$ and on time $t$, then $\partial V / \partial S$ — written $V_S$ for short — answers "if the stock ticks up by a tiny amount and nothing else changes, how much does the option's value change?" Likewise $\partial V / \partial t = V_t$ answers "as one instant of time passes and the stock stays put, how much does the option's value change?" And the *second* partial derivative $\partial^2 V / \partial S^2 = V_{SS}$ measures how the slope itself bends — the curvature of value against spot. Hold these three in your head; they are about to become the famous Greeks.

### What is a PDE?

A *differential equation* is an equation that relates a function to its own derivatives. An *ordinary differential equation* (ODE) involves derivatives with respect to a single variable; a *partial differential equation* (PDE) involves partial derivatives with respect to several variables. The Black-Scholes equation is a PDE because the option value $V$ depends on two variables — the spot $S$ and the time $t$ — and the equation ties together $V$, $V_t$, $V_S$, and $V_{SS}$.

The most famous PDE in physics is the *heat equation*, which describes how temperature spreads through a metal bar: $u_t = \alpha u_{xx}$, where $u$ is temperature, $t$ is time, $x$ is position along the bar, and $\alpha$ is how fast heat diffuses. The remarkable thing — and a hint of what is coming — is that the Black-Scholes PDE is, after a change of variables, exactly the heat equation. Option prices diffuse through "price space" the way heat diffuses through a metal bar. The same math that Fourier wrote down in 1822 to study a hot poker prices a trillion dollars of options today.

### What is an expectation, and what is "risk-neutral"?

An *expectation* is a probability-weighted average. If a bet pays \$10 with probability 60% and \$0 with probability 40%, its expectation is $0.6 \times \$10 + 0.4 \times \$0 = \$6$. We write $E[X]$ for the expectation of a random quantity $X$.

Here is the subtle part that trips up every beginner. To price an option as an expectation, you do *not* average the payoff using the real-world probabilities of where the stock might go. You use a special, adjusted set of probabilities called the *risk-neutral measure*, written $Q$, and you write the expectation as $E^Q[\cdot]$. Under this measure, every asset is assumed to grow on average at the risk-free interest rate $r$ — the rate you earn on cash with no risk, like a Treasury bill. The deep reason is no-arbitrage: if you could hedge away all the risk of an option (and you can, as we will see), then its price cannot reward you for risk you are not taking, so the math must price it in a world where nobody is paid for risk. That world is the risk-neutral one. We cover the why of this in depth in the companion piece on [martingales and the risk-neutral measure](/blog/trading/quantitative-finance/risk-neutral-pricing-martingale-measure-quant-interviews); for now, just take it as the rule: **price = discounted expectation of the payoff, under $Q$.** In symbols,

$$ V_0 = e^{-rT}\, E^Q\!\big[\text{payoff}(S_T)\big]. $$

The $e^{-rT}$ factor is *discounting*: a dollar at time $T$ is worth $e^{-rT}$ dollars today, because today's dollar could have grown at rate $r$ in the meantime. We will define the stochastic model for $S_T$ in a moment.

#### Worked example: pricing a coin-flip payoff two ways

Let us do the smallest possible version of the whole post right now, so the "two roads" idea is concrete before we earn it with real math. Forget options for a second. Suppose a contract pays you \$120 in one year if a (risk-neutral) coin comes up heads and \$80 if it comes up tails, each with probability 50% under $Q$. The risk-free rate is 5% per year, so the discount factor is $e^{-0.05} \approx 0.9512$.

**Road A — the expectation.** The expected payoff is $0.5 \times \$120 + 0.5 \times \$80 = \$100$. Discount it: $V_0 = 0.9512 \times \$100 = \$95.12$.

**Road B — a tiny "PDE" (here, a one-step backward recursion).** Build a one-period tree. At expiry the value is \$120 in the up-state and \$80 in the down-state. Step backward one node: the value today is the discounted, probability-weighted average of the two children, $0.9512 \times (0.5 \times 120 + 0.5 \times 80) = \$95.12$. Same number.

The two roads agree because they are doing the same arithmetic in different clothes — averaging-then-discounting versus rolling values backward. **That equality, made continuous and rigorous, is precisely the Feynman-Kac theorem.** Everything that follows is the grown-up version of this \$95.12.

## How the stock moves: geometric Brownian motion

To price a real option we need a model for how the stock price wanders between now and expiry. The standard one is *geometric Brownian motion* (GBM), and it is worth understanding at the level of intuition before we write the symbols.

Picture the stock price as a speck of pollen on water, jiggled randomly by molecules from every side — this is literally *Brownian motion*, named after the botanist Robert Brown who watched it under a microscope in 1827. The speck drifts in some average direction while being buffeted by noise. A stock price does the same: it drifts upward on average (companies grow, and investors demand a return) while being shoved around by a constant stream of news.

![Pipeline showing sell an option, buy delta shares, apply Ito's lemma, randomness cancels, must earn the risk-free rate, leading to the Black-Scholes PDE](/imgs/blogs/feynman-kac-black-scholes-pde-math-for-quants-2.png)

The figure above previews where we are headed: the chain of reasoning that turns this random model into a clean, deterministic equation. We hold an option, hedge it with shares, apply the calculus of random processes, watch the randomness cancel, invoke no-arbitrage, and out pops the Black-Scholes PDE. But first the model itself. The defining equation of GBM is a *stochastic differential equation* (SDE):

$$ dS_t = \mu S_t\, dt + \sigma S_t\, dW_t. $$

Read it left to right. $dS_t$ is the tiny change in the stock price over a tiny instant. The first term, $\mu S_t\, dt$, is the *drift*: the price grows at average rate $\mu$ per unit time, scaled by the current price (a \$200 stock moves twice as many dollars as a \$100 stock for the same percentage move). The second term, $\sigma S_t\, dW_t$, is the *noise*: $\sigma$ is the *volatility* — how violently the price jiggles — and $dW_t$ is the random kick, an increment of Brownian motion. A Brownian increment over a time step $dt$ is a random draw from a normal distribution with mean zero and variance $dt$; the longer the step, the bigger the typical kick. We build GBM from the random walk from scratch in the [Ito's lemma](/blog/trading/quantitative-finance/itos-lemma-quant-interviews) companion; here we just use it.

Two features of GBM make it the workhorse model. First, because the noise is proportional to $S_t$, the price can never go negative — it can crash toward zero but not through it, which is how real stock prices behave (a share cannot be worth less than \$0). Second, the *logarithm* of the price follows an ordinary (arithmetic) Brownian motion, which means the price at expiry is *lognormally* distributed — its log is a bell curve. That single fact is what makes the Black-Scholes formula a clean expression involving the normal distribution.

When we move to the risk-neutral world for pricing, we make one change: we replace the real-world drift $\mu$ with the risk-free rate $r$. Under $Q$, the stock grows on average at $r$:

$$ dS_t = r S_t\, dt + \sigma S_t\, dW_t^Q. $$

This is the engine for the expectation road. Simulate this SDE forward many times, and you get many possible $S_T$ values; that is exactly the Monte Carlo procedure.

#### Worked example: where does the stock land in three months?

Let us make the model produce a number. Take $S_0 = \$100$, volatility $\sigma = 20\%$ per year, risk-free rate $r = 5\%$, and horizon $T = 0.25$ years (three months). Under GBM, the stock price at time $T$ is

$$ S_T = S_0 \exp\!\Big[\big(r - \tfrac12\sigma^2\big)T + \sigma\sqrt{T}\, Z\Big], $$

where $Z$ is a standard normal random draw (mean 0, variance 1). The term $-\tfrac12\sigma^2$ is the famous *Ito correction* — a downward nudge to the drift that comes from the curvature of the exponential; it is why the log-price drifts at $r - \tfrac12\sigma^2$ rather than at $r$. Plug in the numbers: $r - \tfrac12\sigma^2 = 0.05 - 0.5 \times 0.04 = 0.03$, so the deterministic part of the exponent is $0.03 \times 0.25 = 0.0075$, and the random part has size $\sigma\sqrt{T} = 0.20 \times 0.5 = 0.10$.

If the random draw is $Z = 0$ (the median path), $S_T = 100 \times e^{0.0075} \approx \$100.75$. If $Z = +1$ (a one-standard-deviation good draw), $S_T = 100 \times e^{0.0075 + 0.10} \approx \$111.34$. If $Z = -1$, $S_T = 100 \times e^{0.0075 - 0.10} \approx \$91.36$. The spread of outcomes around \$100 — from roughly \$91 to \$111 for a one-sigma move — is the volatility doing its work. **The whole job of an option price is to average the payoff over this cloud of possible landings and discount it back.**

## Deriving the Black-Scholes PDE: the delta-hedge argument

Now we earn the equation. The genius of Fischer Black, Myron Scholes, and Robert Merton in 1973 was to notice that you can *cancel the randomness* of an option by holding the right number of shares against it, and that once the randomness is gone, no-arbitrage pins down the price. Let us walk it slowly.

![Pipeline showing sell an option, buy delta shares, apply Ito's lemma, randomness cancels, earn the risk-free rate, Black-Scholes PDE](/imgs/blogs/feynman-kac-black-scholes-pde-math-for-quants-2.png)

The way this works, step by step: suppose you **sell** one call option for its market price $V$ and you want to protect yourself from the stock moving. The option's value rises when the stock rises (you are short it, so that hurts you), so you offset by **buying** some shares of the stock. How many? You buy exactly $V_S$ shares — the option's delta, the rate at which its value changes per \$1 of stock. This is the *delta hedge*. Your portfolio is: short one option, long $V_S$ shares. Its value is

$$ \Pi = -V + V_S \cdot S. $$

Now let a tiny instant pass and ask how $\Pi$ changes. The share leg changes by $V_S\, dS$ — straightforward. The option leg is the tricky part, because $V$ depends on both $S$ (which is random) and $t$. To find $dV$ we need *Ito's lemma*, the chain rule for random processes. For a function $V(S,t)$ where $S$ follows GBM, Ito's lemma says

$$ dV = V_t\, dt + V_S\, dS + \tfrac12 \sigma^2 S^2 V_{SS}\, dt. $$

This looks like the ordinary chain rule with one extra term: the $\tfrac12\sigma^2 S^2 V_{SS}\, dt$. That extra term is the entire reason options are interesting. It exists because $dS$ is random and, unlike an ordinary smooth change, its square does not vanish — $(dS)^2 = \sigma^2 S^2\, dt$ to leading order, not zero. Randomness has "size" even over an instant, and curvature ($V_{SS}$) converts that size into a real, predictable drift in the option's value. We prove this term carefully in the [Ito's lemma](/blog/trading/quantitative-finance/itos-lemma-quant-interviews) post; here, marvel at what it does next.

Put the two legs together. The change in the hedged portfolio is

$$ d\Pi = -dV + V_S\, dS = -\Big(V_t\, dt + V_S\, dS + \tfrac12\sigma^2 S^2 V_{SS}\, dt\Big) + V_S\, dS. $$

Watch the magic: the two $V_S\, dS$ terms cancel exactly. The random term $dS$ — the only place uncertainty lived — is gone.

$$ d\Pi = -V_t\, dt - \tfrac12\sigma^2 S^2 V_{SS}\, dt. $$

What remains is *deterministic*: it has no $dW$, no randomness, just a known drift over $dt$. We built a portfolio that, over the next instant, has a perfectly predictable change. That is the whole trick — *delta-hedging removes risk*.

Now invoke no-arbitrage. A perfectly riskless portfolio must earn exactly the risk-free rate $r$ — no more, no less, or you could borrow or lend against it and print free money. A portfolio worth $\Pi$ that earns $r$ changes by $r\Pi\, dt$ over an instant. Set the two expressions for $d\Pi$ equal:

$$ -V_t\, dt - \tfrac12\sigma^2 S^2 V_{SS}\, dt = r\Pi\, dt = r\big(-V + V_S S\big)\, dt. $$

Divide through by $dt$, move everything to one side, and you get the **Black-Scholes partial differential equation**:

$$ \boxed{\,V_t + \tfrac12\sigma^2 S^2 V_{SS} + rS V_S - rV = 0.\,} $$

Every European derivative on this stock — call, put, digital, anything — satisfies this exact equation. What distinguishes a call from a put is *not* the equation; it is the *boundary condition*, the payoff plugged in at expiry. We will return to that. For now, sit with what just happened: a model full of randomness collapsed into a deterministic equation with no probabilities in sight. That collapse is the same phenomenon that Feynman-Kac describes from the other direction.

## The Greeks are the terms of the PDE

Here is the most useful reframing in options trading, and it is hiding in plain sight in the boxed equation. Each term of the Black-Scholes PDE is a *Greek* — a named sensitivity that traders quote, monitor, and hedge all day long.

![Stack of the four Black-Scholes PDE terms: theta, the gamma term, the delta growth term, minus the carry term, equals zero](/imgs/blogs/feynman-kac-black-scholes-pde-math-for-quants-3.png)

The stack above lays out the four pieces. Reading the boxed equation term by term:

- $V_t$ is **theta** ($\Theta$): how the option's value changes as time passes. For a long option it is negative — value bleeds away as expiry approaches. This is *time decay*.
- $\tfrac12\sigma^2 S^2 V_{SS}$ contains **gamma** ($\Gamma = V_{SS}$): the curvature of value against spot. A long option has positive gamma, and this term is positive — it is the *reward* for holding curvature.
- $rS V_S$ contains **delta** ($\Delta = V_S$): the sensitivity to spot, multiplied by the financing cost of holding the shares.
- $-rV$ is the cost of *financing the option position itself* — the interest you forgo by tying up $V$ dollars in the option.

So the Black-Scholes PDE, stripped of jargon, reads:

> Time decay + curvature reward + delta financing − option financing = 0.

The two terms that traders obsess over are the first two: **theta and gamma**. Rearrange the equation to isolate them (ignoring the smaller financing terms for intuition): $-\Theta \approx \tfrac12\sigma^2 S^2 \Gamma$. In words, *the rate at which you lose money to time decay equals the rate at which curvature can make you money when the stock moves*. This is the famous **theta-gamma trade-off**, and it is the single most important sentence a beginning options trader can internalize: **theta is the rent you pay to own gamma.**

![Matrix mapping each Greek to what it measures, which partial derivative it is, and which term of the PDE it becomes](/imgs/blogs/feynman-kac-black-scholes-pde-math-for-quants-4.png)

The matrix above lines up each Greek with its partial derivative and its PDE term. Note that *rho* — the sensitivity to interest rates — does not get its own term in the way theta and gamma do; rates enter through the $rSV_S$ and $-rV$ terms collectively rather than as a single isolated derivative. Delta, gamma, and theta are the trio that the PDE is *built out of*, which is why they dominate day-to-day hedging.

### Why gamma and theta always have opposite signs

There is a beautiful conservation law buried here. For a long option, gamma is positive (the value curve bends upward) and theta is negative (value decays). The PDE forces their product-scaled sum to balance. You cannot own positive gamma for free — the market charges you theta for it. And if you are *short* an option, you have negative gamma (you lose when the stock moves a lot in either direction) but positive theta (you collect time decay every day). Selling options is collecting rent while standing in front of a steamroller; buying options is paying rent for the right to profit from chaos. The PDE encodes exactly this tension.

#### Worked example: a delta-hedged option's daily P&L

Let us make theta-gamma concrete with dollars. You buy one at-the-money call: $S_0 = \$100$, $K = \$100$, $\sigma = 20\%$, $r = 5\%$, $T = 0.25$. Suppose this option has gamma $\Gamma = 0.039$ per share (meaning delta changes by 0.039 for each \$1 move in the stock) and theta $\Theta = -\$0.025$ per share per day (it loses 2.5 cents of value per day from time passing). These are realistic numbers for this option. You delta-hedge it — you short delta shares so you are immune to small moves — and you hold overnight.

The daily P&L of a delta-hedged long option is approximately

$$ \text{P\&L} \approx \tfrac12 \Gamma (\Delta S)^2 + \Theta, $$

where $\Delta S$ is the stock's move that day. The first term is the *gamma gain* — it is always positive because $(\Delta S)^2$ is positive whether the stock rose or fell, and your curvature profits from movement either way. The second term is the *theta cost* — always negative for a long option.

**Quiet day:** the stock moves \$0.50. Gamma gain $= \tfrac12 \times 0.039 \times (0.50)^2 = \tfrac12 \times 0.039 \times 0.25 = \$0.0049$. Theta cost $= -\$0.025$. Net P&L $= 0.0049 - 0.025 = -\$0.020$. You lost about two cents — the small move was not enough to pay the rent.

**Big-move day:** the stock moves \$1.50. Gamma gain $= \tfrac12 \times 0.039 \times (1.50)^2 = \tfrac12 \times 0.039 \times 2.25 = \$0.0439$. Theta cost $= -\$0.025$. Net P&L $= 0.0439 - 0.025 = +\$0.019$. The big move more than paid the rent.

**The break-even move:** set the gamma gain equal to the theta cost: $\tfrac12 \times 0.039 \times (\Delta S)^2 = 0.025$, so $(\Delta S)^2 = 0.025 / 0.0195 \approx 1.28$, giving $\Delta S \approx \$1.13$. If the stock moves about \$1.13 in a day, your gamma gain exactly offsets your theta decay and your hedged P&L is zero.

![Before and after panels: a quiet day with theta decay and tiny gamma gain netting a loss, versus a big-move day with large gamma gain netting a gain](/imgs/blogs/feynman-kac-black-scholes-pde-math-for-quants-6.png)

The figure above contrasts the two days. **The intuition is this: a delta-hedged option is a bet on how much the stock moves versus how much the market priced in.** That break-even daily move of \$1.13 corresponds to a particular annualized volatility — and it is exactly the *implied volatility* the option was priced at. If the stock realizes *more* volatility than implied, your gamma gains beat your theta and you make money; if it realizes *less*, theta wins. This is the entire game of *volatility trading*, and it is sitting right inside the PDE.

## Feynman-Kac: the formal bridge

We have now built both roads. On one side, the Black-Scholes PDE — a deterministic equation. On the other, the risk-neutral expectation — an average over random paths. The Feynman-Kac theorem is the precise statement that they are the same.

In its general form, suppose a process follows the SDE $dX_t = \mu(X_t,t)\,dt + \sigma(X_t,t)\,dW_t$, and suppose a function $V(x,t)$ solves the PDE

$$ V_t + \mu(x,t) V_x + \tfrac12 \sigma(x,t)^2 V_{xx} - rV = 0, $$

with the terminal condition $V(x,T) = g(x)$ — that is, at the final time $T$, the function equals some given payoff $g$. Then Feynman-Kac says that the solution is exactly

$$ V(x,t) = E^Q\!\Big[ e^{-r(T-t)} g(X_T) \,\Big|\, X_t = x \Big]. $$

Read that carefully: the value of the PDE solution at a point equals the *expected, discounted payoff* if you start the process at that point and let it run to time $T$. The PDE on the left and the expectation on the right are two faces of one object.

![Tree of pricing approaches: a root splitting into the expectation route and the PDE route, each branching into concrete methods](/imgs/blogs/feynman-kac-black-scholes-pde-math-for-quants-5.png)

The tree above organizes how this plays out in practice. The root is "price a derivative." It splits into the *expectation route* (which branches into the closed-form formula when the integral is solvable, and Monte Carlo when it is not) and the *PDE route* (which branches into finite differences and binomial trees). Feynman-Kac is the trunk connecting the two big branches: it guarantees that whichever leaf you climb to, you arrive at the same price. A quant chooses the leaf by convenience, not by correctness.

To see why it is true at the level of intuition, recall the delta-hedge derivation. We showed that the random $dW$ term cancels in a hedged portfolio. Feynman-Kac is the same cancellation viewed from the expectation side: the drift of the discounted option value under $Q$ is zero (it is a *martingale* — a process with no predictable trend), and a process with no drift, run forward and averaged, recovers exactly the function that has no random component, i.e., the PDE solution. The two statements — "the hedged portfolio is riskless" and "the discounted price is a martingale" — are the same fact in different languages. We make the martingale half rigorous in the [risk-neutral pricing](/blog/trading/quantitative-finance/risk-neutral-pricing-martingale-measure-quant-interviews) companion.

For the Black-Scholes case specifically, the SDE is risk-neutral GBM, $dS = rS\,dt + \sigma S\,dW^Q$, the payoff for a call is $g(S_T) = \max(S_T - K, 0)$, and Feynman-Kac gives

$$ V(S,t) = e^{-r(T-t)}\, E^Q\!\big[\max(S_T - K,\,0)\,\big|\,S_t = S\big]. $$

That expectation, because $S_T$ is lognormal, can be computed in closed form. The result is the Black-Scholes formula.

#### Worked example: the same price three ways

Let us nail the equivalence with our running option: $S_0 = \$100$, $K = \$100$, $\sigma = 20\%$, $r = 5\%$, $T = 0.25$. We will price it by the closed-form formula, sketch the Monte Carlo, and confirm a finite-difference grid lands in the same place.

**Route 1 — closed form.** The Black-Scholes call price is

$$ C = S_0 N(d_1) - K e^{-rT} N(d_2), $$

where $N(\cdot)$ is the standard normal cumulative distribution (the probability a standard bell curve lands below a value), and

$$ d_1 = \frac{\ln(S_0/K) + (r + \tfrac12\sigma^2)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}. $$

Plug in. The log term $\ln(100/100) = 0$. The drift term $(r + \tfrac12\sigma^2)T = (0.05 + 0.02)\times 0.25 = 0.0175$. The denominator $\sigma\sqrt{T} = 0.20 \times 0.5 = 0.10$. So $d_1 = 0.0175 / 0.10 = 0.175$ and $d_2 = 0.175 - 0.10 = 0.075$. From the normal table, $N(0.175) \approx 0.5695$ and $N(0.075) \approx 0.5299$. The discount factor $e^{-rT} = e^{-0.0125} \approx 0.98758$. Then

$$ C = 100 \times 0.5695 - 100 \times 0.98758 \times 0.5299 = 56.95 - 52.33 = \$4.62. $$

So the fair price is about **\$4.62**.

**Route 2 — Monte Carlo (the expectation road).** Simulate the risk-neutral GBM forward: draw a million standard normals $Z$, compute $S_T = 100 \exp[(0.05 - 0.02)\times 0.25 + 0.10\, Z] = 100\exp[0.0075 + 0.10 Z]$ for each, take $\max(S_T - 100, 0)$, average, and multiply by $e^{-0.0125}$. With a million paths the average payoff comes out near \$4.68, and after discounting you land at roughly **\$4.62 ± \$0.01** — the Monte Carlo error shrinks as $1/\sqrt{N}$, so a million paths gives about two-decimal accuracy. It converges to the same \$4.62.

**Route 3 — finite differences (the PDE road).** Lay out a grid of stock prices and time steps, impose the payoff $\max(S-100,0)$ at the final time, and march the PDE backward to today using the discretized $V_t$, $V_S$, $V_{SS}$. A grid of a few hundred price steps and a few hundred time steps returns **\$4.62** as well (we do a tiny one-step version of this below).

Three methods, one number. **That triple agreement is Feynman-Kac made visible: the PDE solution, the discounted expectation, and the closed-form integral are the same \$4.62.**

## The payoff is the boundary condition

We said the Black-Scholes PDE is the *same* equation for every European derivative, and that what changes is the boundary condition. Let us make that precise, because it is the source of the PDE road's enormous flexibility.

A PDE alone does not have a unique solution — it has a whole family of them. To pick out the one you want, you supply *conditions on the boundary* of the region you are solving over. For option pricing, the region is "all stock prices, from now until expiry," and the boundary has three pieces.

![Stack from terminal payoff and the two spot boundaries down through solving the PDE backward to the unique price today](/imgs/blogs/feynman-kac-black-scholes-pde-math-for-quants-7.png)

The stack above shows the three conditions feeding into the backward solve. The first and most important is the *terminal condition*: at expiry $t = T$, the option value must equal its payoff. For a call, $V(S, T) = \max(S - K, 0)$. This is where the contract's identity lives. The other two are *spatial boundaries* at the edges of the price range: as $S \to 0$, a call is worthless, so $V(0, t) = 0$; as $S \to \infty$, a call behaves like the stock minus the discounted strike, so $V(S, t) \approx S - K e^{-r(T-t)}$.

Here is the punchline. To price a *put* instead of a call, you change nothing about the PDE and nothing about the spatial boundaries' logic — you only change the terminal condition to $V(S, T) = \max(K - S, 0)$. To price a *digital* option that pays a fixed \$1 if the stock finishes above the strike and \$0 otherwise, you set $V(S, T) = \$1 \cdot \mathbf{1}\{S > K\}$. To price a *straddle*, you set the terminal condition to the sum of a call and a put payoff. The machinery is identical; the payoff is the dial. This is why a single finite-difference solver, written once, can price a whole zoo of European derivatives — you feed it a different terminal condition and turn the crank.

#### Worked example: one finite-difference step on a tiny grid

Let us actually do a finite-difference step by hand on a comically small grid, then compare to the closed form. Finite differences replace the continuous derivatives in the PDE with differences between neighboring grid points.

Set up a grid with three stock prices — \$90, \$100, \$110 — and a single time step from expiry back to one period earlier. Use our call with $K = \$100$, $\sigma = 20\%$, $r = 5\%$, and a time step of $\Delta t = 0.25$ (we collapse the whole three months into one step to keep the arithmetic visible). At expiry the payoffs are: at \$90, $\max(90 - 100, 0) = \$0$; at \$100, $\max(100-100,0) = \$0$; at \$110, $\max(110-100,0) = \$10$.

We want the value at the middle node (\$100) one step before expiry. The *explicit* finite-difference scheme approximates the derivatives at the middle node using its neighbors. With grid spacing $\Delta S = \$10$, the discrete derivatives are

$$ V_S \approx \frac{V_{up} - V_{down}}{2\Delta S} = \frac{10 - 0}{20} = 0.5, \qquad V_{SS} \approx \frac{V_{up} - 2V_{mid} + V_{down}}{(\Delta S)^2} = \frac{10 - 0 + 0}{100} = 0.1. $$

Now rearrange the Black-Scholes PDE to step backward in time. From $V_t = -\tfrac12\sigma^2 S^2 V_{SS} - rS V_S + rV$, with $V_{mid}$ at expiry equal to \$0:

$$ V_t \approx -\tfrac12 (0.20)^2 (100)^2 (0.1) - (0.05)(100)(0.5) + (0.05)(0) = -\tfrac12(0.04)(10000)(0.1) - 2.5 = -20 - 2.5 = -22.5. $$

The value one step earlier is $V_{mid} - V_t \cdot \Delta t$ (stepping backward, value today is higher than at expiry by the decay): $V \approx 0 - (-22.5)(0.25) = \$5.63$. On this absurdly coarse one-step grid we get about **\$5.63**, in the same ballpark as the exact **\$4.62**. Refine the grid — use hundreds of price nodes and hundreds of time steps instead of three nodes and one step — and the finite-difference answer marches smoothly toward \$4.62.

**The lesson: a finite-difference grid is just the PDE with its derivatives replaced by arithmetic on neighboring boxes, and the more boxes you use, the closer you get to the true price.** (The crude one-step number is high because a single giant time step badly over-weights the convexity; this is exactly the kind of discretization error that finer grids erase, and it is why production solvers use implicit or Crank-Nicolson schemes rather than a single explicit jump.)

## PDE versus Monte Carlo: when to use which

Feynman-Kac gives you a choice, and a working quant makes that choice deliberately. The two roads have very different cost profiles, and the right one depends on the shape of the problem.

The PDE road — finite differences or trees — is fast and accurate in *low dimensions*. If your derivative depends on one or two underlyings, a grid with a few hundred nodes per dimension prices it to high precision in milliseconds, and it naturally handles features like *early exercise* (American options), where at every node you compare "continue" versus "exercise now." But grids suffer the *curse of dimensionality*: a grid in $d$ dimensions needs $n^d$ nodes. One underlying with 500 nodes is 500 points; five underlyings with 500 nodes each is $500^5 \approx 3 \times 10^{13}$ points — utterly infeasible. PDEs die past three or four dimensions.

The Monte Carlo road — simulating paths and averaging — has the opposite profile. Its error shrinks like $1/\sqrt{N}$ regardless of dimension, so adding underlyings barely changes the cost. A basket option on fifty stocks, or a payoff that depends on the whole *path* (an Asian option averaging the price over time, a barrier option that knocks out if the price ever touches a level), is natural for Monte Carlo and a nightmare for a grid. The price you pay is the slow $1/\sqrt{N}$ convergence — to halve the error you must quadruple the simulations — and the difficulty of handling early exercise (you cannot easily look forward along a simulated path to decide whether to exercise; special techniques like Longstaff-Schwartz regression are needed).

Here is a comparison table to keep the tradeoff legible.

| Feature | PDE / finite differences | Monte Carlo |
| --- | --- | --- |
| **Best for** | 1–3 underlyings, simple payoffs | many underlyings, path-dependent payoffs |
| **Convergence** | fast, smooth (grid refinement) | slow, $1/\sqrt{N}$ in number of paths |
| **Dimensionality** | dies past ~3–4 (curse of dimensionality) | barely cares about dimension |
| **Early exercise (American)** | natural — compare at every node | hard — needs regression tricks |
| **Greeks** | come almost for free from the grid | need bumping or pathwise estimators |
| **Path-dependence (Asian, barrier)** | awkward to encode | natural — just track the path |

The rule of thumb that falls out: **low dimension and early exercise → grid; high dimension or path-dependence → Monte Carlo.** And because Feynman-Kac guarantees both give the same answer, a careful quant prices a single underlying both ways as a *cross-check*: if the grid and the simulation disagree by more than the Monte Carlo's stated error, there is a bug in one of them. The equivalence is not just elegant — it is a debugging tool worth real money.

#### Worked example: how many paths to trust a Monte Carlo price?

Suppose you price our \$4.62 call by Monte Carlo and you want the answer accurate to within one cent with high confidence. The standard error of a Monte Carlo estimate is $\text{SE} = s / \sqrt{N}$, where $s$ is the standard deviation of the discounted payoff across paths and $N$ is the number of paths. For this option, the discounted payoff has a standard deviation of roughly $s \approx \$7$ (most paths pay \$0, a few pay a lot — a skewed, high-variance quantity). To get a standard error of \$0.01, you need

$$ N = \Big(\frac{s}{\text{SE}}\Big)^2 = \Big(\frac{7}{0.01}\Big)^2 = 700^2 = 490{,}000 \approx \text{half a million paths.} $$

To tighten the error tenfold to \$0.001, you would need a *hundred times* more paths — 49 million. **The intuition: Monte Carlo accuracy is brutally expensive at the margin, which is exactly why, for a single-underlying European option, nobody simulates — they use the closed form or a grid.** Monte Carlo earns its keep only when the alternatives are impossible.

![Tree of pricing approaches splitting into the expectation route and the PDE route with their concrete methods](/imgs/blogs/feynman-kac-black-scholes-pde-math-for-quants-5.png)

The same tree from earlier is the decision aid: you walk down the expectation branch when the payoff is path-dependent or high-dimensional, and down the PDE branch when it is low-dimensional or has early exercise — knowing the leaves agree on price.

## From PDE to the closed-form Black-Scholes formula

We have used the Black-Scholes call formula as a black box. Let us connect it back to the PDE so it stops feeling like a rabbit pulled from a hat. The formula is what you get when you actually *solve* the PDE with the call's terminal condition — or equivalently, when you actually *compute* the lognormal expectation Feynman-Kac hands you.

The call price is

$$ C = S_0 N(d_1) - K e^{-rT} N(d_2), $$

and the put price (by the same machinery with the put payoff) is

$$ P = K e^{-rT} N(-d_2) - S_0 N(-d_1). $$

Each piece has a clean meaning. The term $S_0 N(d_1)$ is the *present value of receiving the stock if the option finishes in the money* — and it turns out $N(d_1)$ is exactly the option's delta, the number of shares to hold in the hedge. The term $K e^{-rT} N(d_2)$ is the *present value of paying the strike if the option finishes in the money*, where $N(d_2)$ is the risk-neutral probability the call ends in the money. So the formula reads: "(value of the stock you might get) minus (cost of the strike you might pay), each weighted by the chance it happens." It is the discounted expectation, term by term.

The reason this expectation has a closed form is the lognormal landing distribution we computed earlier. When the payoff is a simple $\max(S_T - K, 0)$ and $S_T$ is lognormal, the integral defining the expectation splits into two pieces, each a normal cumulative distribution — hence the two $N(\cdot)$ terms. Change the payoff to something that does not integrate cleanly against the lognormal (a barrier, an average, a basket), and the closed form vanishes; you fall back to the grid or the simulation. This is the deep reason the Black-Scholes formula exists only for vanilla options: it is the lucky case where the Feynman-Kac integral is solvable by hand.

#### Worked example: pricing the matching put and checking parity

Use our numbers — $S_0 = \$100$, $K = \$100$, $r = 5\%$, $\sigma = 20\%$, $T = 0.25$ — and price the put. We have $d_1 = 0.175$, $d_2 = 0.075$, so $-d_1 = -0.175$ and $-d_2 = -0.075$, giving $N(-d_1) \approx 0.4305$ and $N(-d_2) \approx 0.4701$. Then

$$ P = 100 \times 0.98758 \times 0.4701 - 100 \times 0.4305 = 46.43 - 43.05 = \$3.38. $$

So the put is worth about **\$3.38**, less than the \$4.62 call — sensible, because with a positive interest rate and the stock drifting up under $Q$, the call has more upside than the put has.

Now check *put-call parity*, an arbitrage identity that must hold no matter what model you use: $C - P = S_0 - K e^{-rT}$. The left side is $4.62 - 3.38 = \$1.24$. The right side is $100 - 100 \times 0.98758 = 100 - 98.758 = \$1.24$. They match exactly. **The lesson: the call and put prices are not independent — they are locked together by an arbitrage relationship the PDE respects automatically, which is a free correctness check on any pricer you build.**

## Common misconceptions

Even people who can recite the Black-Scholes formula often hold one of these wrong beliefs. Each is worth correcting because the error costs real money.

**"The Black-Scholes PDE assumes the stock will grow at the risk-free rate, which is obviously false."** The PDE does *not* claim stocks earn the risk-free rate in reality. It says that *for pricing purposes*, after you hedge away the risk, the real-world drift $\mu$ disappears from the equation entirely — look back at the boxed PDE and notice $\mu$ is nowhere in it. The price does not depend on whether you think the stock will go up or down, because a delta-hedger does not care about direction. The risk-neutral drift $r$ is a mathematical device for computing a hedge-implied price, not a forecast. This is the single most misunderstood point in the whole subject.

**"Feynman-Kac is just a coincidence — two methods that happen to agree."** It is the opposite of a coincidence. The agreement is forced by no-arbitrage: if the PDE price and the expectation price ever differed, you could buy the cheap one and sell the dear one and lock in a riskless profit, which markets compete away. Feynman-Kac is the mathematical shadow of "there is no free lunch." The two roads agree *because* a riskless arbitrage would otherwise exist.

**"Gamma is good, so you should always be long options."** Long gamma is indeed pleasant — you profit from movement either way — but it is never free. The PDE shows theta is the exact rent for it. If the stock moves *less* than the implied volatility you paid for, your theta bleed beats your gamma gains and you lose money even though you "had positive gamma." Whether long gamma pays depends entirely on realized volatility versus implied, not on the sign of gamma alone.

**"Monte Carlo is the modern, superior method; PDEs are old-fashioned."** Neither dominates. For a single-underlying European or American option, a finite-difference grid is faster and more accurate than Monte Carlo by orders of magnitude, and it gives the Greeks almost for free. Monte Carlo wins only when dimension or path-dependence makes a grid infeasible. Reaching for simulation on a problem a grid handles is a rookie inefficiency.

**"The closed-form formula is the 'real' price and numerical methods are approximations."** The closed form is itself just the exact evaluation of the Feynman-Kac integral for the lucky vanilla case. For any payoff where the integral is not elementary, there *is* no closed form, and the grid or simulation is not approximating some hidden formula — it is computing the price as directly as the formula does for vanillas. All three are evaluating the same expectation; the formula is simply the case where pencil and paper suffice.

**"Volatility $\sigma$ is an observable input like the stock price."** The spot price, strike, rate, and time are all observable. Volatility is not — it is the one input you cannot read off a screen. In practice traders run the formula *backward*: take the option's market price as given and solve for the $\sigma$ that reproduces it. That number is the *implied volatility*, and the fact that it differs across strikes and maturities (the *volatility surface*) is the loudest evidence that the constant-volatility GBM assumption is wrong. We dig into that surface in the [options theory](/blog/trading/quantitative-finance/options-theory) companion.

## How it shows up in real markets

The theory above is not a museum piece. It runs, every second, inside the systems that price and hedge the world's options. Here are concrete places the PDE-expectation bridge does real work.

### 1. The 1987 crash and the birth of the volatility smile

Before October 1987, options traders priced more or less by the flat Black-Scholes formula: one volatility for all strikes. Then on October 19, 1987 — *Black Monday* — the S&P 500 fell 20.5% in a single day, an event the lognormal model said should essentially never happen (a 20-plus-standard-deviation move). Overnight, traders stopped trusting that crashes were impossible and began paying up for downside puts as crash insurance. That demand bid up the implied volatility of low-strike puts relative to at-the-money options, bending the flat volatility line into the *volatility skew* (or smile) that persists to this day. The mechanism is pure Feynman-Kac: the market quotes prices, and you invert the formula to read off the implied volatility at each strike. The skew is the market telling you the true risk-neutral distribution has a fatter left tail than the lognormal — a correction the single-$\sigma$ PDE cannot capture, which is why modern desks solve a *local-volatility* PDE with $\sigma$ varying by spot and time.

### 2. Long-Term Capital Management and the limits of hedging

LTCM, the hedge fund founded by Scholes and Merton themselves (two of the formula's authors, both Nobel laureates), was the apotheosis of delta-hedged, model-driven trading. The delta-hedge argument promises a riskless portfolio, but it assumes you can rebalance continuously, that prices move smoothly, and that you can always trade. In August-September 1998, after Russia defaulted on its debt, liquidity evaporated, prices gapped instead of moving smoothly, and correlations all rushed to one. The continuous hedge that the PDE assumes was impossible to execute, and a fund that had earned ~40% a year lost ~\$4.6 billion in months and required a Fed-organized bailout. The lesson the PDE itself whispers if you read it carefully: the derivation assumes *continuous* hedging and *constant* volatility, and when reality violates those assumptions — gaps, illiquidity, vol spikes — the riskless portfolio is not riskless at all.

### 3. Every options market-making desk runs a PDE grid

A modern equity-options market maker quotes thousands of strikes and maturities and must know its delta, gamma, theta, and vega on the whole book continuously. The standard tool is a finite-difference (or tree) engine that solves the pricing PDE on a grid for each option, reading the Greeks straight off the grid's neighboring nodes — exactly the $V_S$, $V_{SS}$, $V_t$ differences from our worked example, just on a fine grid instead of three points. The desk hedges its aggregate delta in the stock, manages its gamma and theta as a portfolio, and the theta-gamma balance from our worked example is, literally, the desk's daily P&L attribution: it decomposes each day's profit into "gamma made from realized moves" minus "theta paid for time."

### 4. Pricing exotics where only one road works

Consider an *Asian option* whose payoff depends on the *average* stock price over its life, used heavily by corporates hedging commodity costs (an airline averaging jet-fuel prices). The payoff depends on the whole path, not just the endpoint, so the simple PDE in $(S, t)$ does not capture it without adding a dimension for the running average — which pushes you toward the curse of dimensionality. Monte Carlo, by contrast, just tracks the average along each simulated path and is completely natural. Conversely, an *American put* on a single stock — exercisable any day — is awkward for Monte Carlo (you cannot look ahead along a path) but trivial for a grid, where at each node you compare "hold" versus "exercise." Real desks route each product to the road Feynman-Kac says is cheapest, confident the prices would agree.

### 5. The 2008 crisis and discounting assumptions

The Black-Scholes derivation assumes a single risk-free rate $r$ for both growing the stock and discounting. After 2008, the gap between supposedly "risk-free" rates (LIBOR versus overnight-indexed-swap rates) blew out as banks stopped trusting each other, and the industry discovered that the one-$r$ assumption baked into the PDE was a simplification that mispriced collateralized derivatives by meaningful amounts. The fix — *multi-curve* and *OIS discounting* — keeps the Feynman-Kac structure (price still equals a discounted expectation) but replaces the single discount factor $e^{-rT}$ with a more careful one. It is a vivid reminder that the bridge survives, but the specific rates plugged into it are assumptions that markets can break.

### 6. Variance swaps and trading the gamma-theta identity directly

A *variance swap* pays the difference between realized variance and a fixed strike — it is a pure bet on how much the stock moves, with no directional exposure. It is, in effect, a packaged version of the delta-hedged-option P&L from our worked example: the $\tfrac12\Gamma(\Delta S)^2$ gamma gains summed over the life of a continuously hedged option, stripped of the theta and the strike dependence. Dealers replicate a variance swap with a strip of options across all strikes precisely *because* the theta-gamma identity inside the PDE tells them how a delta-hedged option's P&L accumulates realized variance. The instrument is the PDE's gamma term turned into a tradable contract — about \$4 billion of notional changes hands in these on a busy day, all riding on the equation we derived.

## When this matters to you

If you ever buy a single call option on a stock you like, the \$4.62 you pay is this machinery in action — the market's discounted expectation of your payoff, computed by someone's grid or formula, marked up by a spread. Understanding that the price is a *hedge-implied* number, not a forecast of where the stock is going, is genuinely useful: it explains why an option can lose value even when the stock moves your way (theta and falling implied volatility eat the gain) and why "the stock went up so my call must be winning" is not reliable.

If you go further and ever sell options — covered calls, cash-secured puts — the theta-gamma trade-off is your daily reality. You are collecting theta (the rent) while short gamma (standing in front of the steamroller). The PDE tells you exactly what you are short and how a big move hurts you. Sizing that position without knowing your gamma is flying blind.

And if you are heading toward a quant role, this bridge is foundational. Feynman-Kac is why the two halves of a pricing library — the analytics that solve PDEs and the simulation engine that averages paths — must reconcile to the penny, and the first thing a senior quant does with a new pricer is cross-check the two roads on a vanilla option. The equation $V_t + \tfrac12\sigma^2 S^2 V_{SS} + rS V_S - rV = 0$ is one you will write from memory in interviews and read off as Greeks every day on a desk.

A standard caution: this is educational, not financial advice. Options can expire worthless and a sold option can lose far more than its premium; the riskless hedge the math promises is riskless only under assumptions (continuous trading, constant volatility, no jumps) that real markets violate exactly when it matters most.

**Further reading:**

- [Ito's lemma and the stochastic chain rule](/blog/trading/quantitative-finance/itos-lemma-quant-interviews) — the calculus tool that makes the delta-hedge cancellation work, derived from scratch.
- [Martingales and the risk-neutral measure](/blog/trading/quantitative-finance/risk-neutral-pricing-martingale-measure-quant-interviews) — why pricing uses the adjusted $Q$ probabilities and why the discounted price has no drift.
- [The Black-Scholes model in full](/blog/trading/quantitative-finance/black-scholes) — the formula, its assumptions, and its failures, at practitioner depth.
- [Options theory and the volatility surface](/blog/trading/quantitative-finance/options-theory) — what the Greeks mean on a real book and how the constant-volatility assumption breaks.
- Black, F. and Scholes, M. (1973), "The Pricing of Options and Corporate Liabilities," and Merton, R. (1973), "Theory of Rational Option Pricing" — the original papers, surprisingly readable.

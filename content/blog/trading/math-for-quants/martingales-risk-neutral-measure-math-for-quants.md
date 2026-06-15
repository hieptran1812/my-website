---
title: "Martingales and the risk-neutral measure"
date: "2026-06-15"
description: "A build-from-zero tour of martingales as fair games, why no-arbitrage forces a risk-neutral measure to exist, and how every option price is just a discounted expectation that the real-world growth rate quietly drops out of."
tags: ["martingale", "risk-neutral-measure", "no-arbitrage", "option-pricing", "replication", "binomial-model", "fundamental-theorems", "quant-finance", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A martingale is a fair game whose expected future value, given everything you know today, equals its value today; the deepest result in pricing is that banning free money forces the existence of a special probability world (the *risk-neutral measure* $Q$) in which every discounted asset price is a martingale.
>
> - A **martingale** satisfies $E[X_t \mid \mathcal{F}_s] = X_s$: knowing the past, your best forecast of the future is exactly where you are now. That is the formal meaning of "no predictable profit."
> - The **First Fundamental Theorem** says: no arbitrage *if and only if* there exists a risk-neutral measure $Q$ under which discounted prices are martingales. Pricing and the absence of free money are the same fact stated two ways.
> - Under $Q$ the drift of every asset is the risk-free rate $r$, **not** the real-world expected return $\mu$ — which is exactly why an option's price does not depend on whether you are bullish or bearish on the stock.
> - The **price of any derivative** is $\text{price} = E^Q[\text{discounted payoff}]$, and the **Second Fundamental Theorem** says that price is unique when the market is *complete* (every payoff can be replicated).
> - The one number to remember: in our running model a \$105-strike call on a \$100 stock that can go to \$120 or \$90 is worth exactly **\$5**, the cost of the stock-and-bond portfolio that copies it — and the real-world odds of up versus down never enter that \$5.

Here is a fact that should feel impossible the first time you meet it. Two traders sit at the same desk. One is convinced a stock will rip higher; she thinks it returns 30% a year. The other is sure it is going nowhere; he pencils in 2%. They are looking at the same option on that same stock. And yet, if you ask them what the option is *worth*, they will — if they have both done the math correctly — write down the exact same number. The bull's optimism and the bear's gloom, the entire question of which way the stock is headed, simply vanishes from the price.

That is not a quirk. It is the single most surprising and most important idea in all of derivatives pricing, and it falls out of two plain-sounding requirements: that you cannot conjure money out of thin air, and that an asset's price today should be a fair reflection of its uncertain future. Make those two ideas precise and you get *martingales* and the *risk-neutral measure*. This post builds both from absolute zero — no measure theory assumed, every term defined the first time it appears — and ties every formula to a concrete dollar figure you can check on a napkin. By the end, the disappearing-optimism trick will look not like magic but like the only thing that could possibly be true.

![Before and after columns contrasting the real-world measure P with the risk-neutral measure Q](/imgs/blogs/martingales-risk-neutral-measure-math-for-quants-1.png)

The diagram above is the mental model for the whole post. On the left is the *real world*, the measure we call $P$ — the place where probabilities describe how often things actually happen, where a risky stock is expected to outgrow a safe bond because investors demand a reward for bearing risk. On the right is the *pricing world*, the measure we call $Q$ — an artificial but extremely useful re-weighting of those same outcomes, engineered so that every asset is expected to grow at exactly the risk-free rate. The outcomes are identical in both worlds; only the *odds* change. Almost everything below is the story of why $Q$ must exist, what it does for you, and why the price of a derivative lives entirely on the right-hand side of that picture.

## Foundations: the building blocks

Before we can talk about fair games and pricing measures, we need a small vocabulary. We will define each word the first time it appears, build the simplest possible version of every idea, and only then climb toward the real machinery. A reader who already knows what a filtration is can skim; a reader who has never seen one can still follow every step.

### What is a "random process"?

A **random process** (or stochastic process) is just a quantity that changes over time in a way we cannot fully predict. Write it $X_t$, read "X at time $t$." The price of a stock is a random process: $X_0$ is the price now, $X_1$ the price tomorrow, $X_2$ the day after. We do not know $X_1$ yet, but we can describe the *odds* of where it might land. A coin you flip once a day and add up — plus a dollar for heads, minus a dollar for tails — is also a random process: your running total drifts up and down as the days pass.

The whole game of quantitative finance is reasoning carefully about random processes: stock prices, interest rates, the value of a portfolio. Everything in this post is a statement about how some random process behaves over time.

### What is "information," and what is a filtration?

At each moment you *know* some things and not others. Right now you know today's price; you do not know tomorrow's. As time passes, your knowledge only grows — you never forget yesterday's price. The mathematical name for "everything you know at time $t$" is the **information set** at time $t$, written $\mathcal{F}_t$ (script-F-sub-t). The whole growing sequence $\mathcal{F}_0 \subseteq \mathcal{F}_1 \subseteq \mathcal{F}_2 \subseteq \cdots$ — information that only accumulates — is called a **filtration**.

You do not need measure theory to use this idea. Just read $\mathcal{F}_t$ as "the history up to and including time $t$." When we later write $E[X \mid \mathcal{F}_t]$, read it as "the expected value of $X$ given everything we know at time $t$." If you want the rigorous version — $\mathcal{F}_t$ as a $\sigma$-algebra of events, the formal reason a backtest must not peek at the future — it lives in the companion post on [probability spaces and random variables](/blog/trading/math-for-quants/probability-spaces-random-variables-math-for-quants). For everything here, "the history so far" is enough.

### What is an "expectation," conditional or otherwise?

The **expectation** of a random quantity is its probability-weighted average — the number you would converge to if you could repeat the situation a million times and average the results. If a bet pays \$10 with probability 0.3 and \$0 with probability 0.7, its expectation is $0.3 \times \$10 + 0.7 \times \$0 = \$3$. We write it $E[X]$.

A **conditional expectation** $E[X \mid \mathcal{F}_t]$ is the same average, but computed *using the information you have at time $t$*. It is your best forecast of $X$ given what you currently know. Crucially, before time $t$ arrives this forecast is itself uncertain (it depends on what you will learn), so a conditional expectation is itself a random process. That subtlety is exactly what makes the martingale definition below say something non-trivial.

### What is "arbitrage"?

**Arbitrage** is free money: a trading strategy that costs nothing to set up, can never lose, and has a positive chance of making a profit. The classic image is buying gold in London for \$2,000 and selling it in New York for \$2,010 at the same instant — \$10 of risk-free profit per ounce, no capital tied up, no way to lose. In an efficient market such opportunities are competed away almost instantly, because everyone wants free money and their buying and selling moves the prices back into line.

The assumption we lean on throughout is **no arbitrage**: prices in the market are such that no free-money strategy exists. This is not a law of nature; it is an idealization that holds approximately because arbitrageurs enforce it. But it is an extraordinarily powerful one, because — as we will see — "no free money" turns out to be enough, all by itself, to pin down prices.

### What is "the risk-free rate" and "discounting"?

The **risk-free rate** $r$ is the interest you earn on money that carries no risk of loss — in practice, the yield on short-term government debt. If $r = 5\%$ per year, then \$100 lent risk-free for a year grows to \$105. We will often write the one-period growth factor as $R = 1 + r$ (so $R = 1.05$ at 5%), or use continuous compounding $e^{rT}$ over horizon $T$ when that is cleaner.

**Discounting** is running that growth backwards. A dollar a year from now is worth less than a dollar today, because today's dollar could have been earning interest. To find today's value of a future amount, you *discount* it: divide by the growth factor. \$105 a year from now is worth $\$105 / 1.05 = \$100$ today. Discounting is the single most important operation in pricing, because comparing money across time is meaningless until you have put everything on the same footing — usually "value as of today." When we talk about the *discounted* price of a stock, we mean the price expressed in today's dollars by dividing out the growth of the risk-free asset. That risk-free asset — the thing we measure everything else against — is called the **numéraire**, a term we will make precise later.

With those five ideas — random process, filtration, conditional expectation, arbitrage, and discounting — we have the raw materials. Now we assemble the first big object: the fair game.

## 1. The real world and the pricing world

A **martingale** is the mathematical name for a perfectly fair game. The defining property is deceptively short:

$$
E[X_t \mid \mathcal{F}_s] = X_s \qquad \text{for all } s < t.
$$

Read it slowly. The left side is "your best forecast of the process's future value $X_t$, given everything you know up to the earlier time $s$." The right side is "where the process is right now, at time $s$." The equation says those two are the same: **knowing the entire past, your best guess for the future is exactly the present value.** The process has no built-in tendency to drift up or down. It is a fair game in the precise sense that, on average, you neither gain nor lose by waiting.

The everyday analogy is a fair coin-flip betting game. Each flip, you bet \$1; heads you win it, tails you lose it. Your running bankroll wanders up and down, but at any moment, your *expected* bankroll one flip from now equals your bankroll right now, because the coin is fair: $0.5 \times (+\$1) + 0.5 \times (-\$1) = \$0$ expected change. That is a martingale. Now bias the coin to land heads 60% of the time and keep the \$1 stakes: your expected change per flip becomes $0.6 \times (+\$1) + 0.4 \times (-\$1) = +\$0.20$, so your bankroll tends to grow. That is a **submartingale** — a game tilted in your favor, $E[X_t \mid \mathcal{F}_s] \ge X_s$. Tilt it against you and you get a **supermartingale**, $E[X_t \mid \mathcal{F}_s] \le X_s$, a game that bleeds you on average.

Here is the first deep payoff of this vocabulary. The statement "there is no *predictable* profit in this process" is *exactly* the martingale property. If $X_t$ were a submartingale — if your best forecast of the future were reliably above today's value — then a simple strategy (hold the thing) would have positive expected profit with no special skill, which is precisely the kind of free lunch that no-arbitrage forbids. Fair game and no-predictable-profit are two names for the same equation.

### Two measures, same outcomes, different odds

Now the crucial move, the one the opening figure is built around. There is not one set of probabilities floating over the market; there are (at least) two, and keeping them apart is what separates someone who understands pricing from someone who is permanently confused.

![Two-column diagram showing the real-world measure on the left and the risk-neutral measure on the right](/imgs/blogs/martingales-risk-neutral-measure-math-for-quants-1.png)

The first set is the **real-world measure**, written $P$ (sometimes called the *physical* or *objective* measure). These are the probabilities of what *actually happens*: if you collected a thousand years of data, $P$ is the frequency with which the stock rises versus falls. In the real world, risky assets are *not* martingales — a stock is expected to grow faster than cash, because investors will not hold a risky thing unless they are paid extra for the risk. That extra expected growth is the **risk premium**, and it is what makes the stock's real-world drift $\mu$ larger than the risk-free rate $r$. Under $P$, the discounted stock price is a submartingale: it tends to drift up.

The second set is the **risk-neutral measure**, written $Q$ (also called the *equivalent martingale measure* or the *pricing measure*). It is an artificial re-weighting of the *same outcomes* — the stock can still go up to \$120 or down to \$90, exactly as before — but the probabilities are tuned so that *every asset is expected to grow at exactly the risk-free rate*. In this world, nobody is paid a premium for risk; the risky stock and the safe bond have the same expected return $r$. That is why it is called "risk-neutral": it is the probability world inhabited by an investor who is utterly indifferent to risk and demands no reward for it.

The word **equivalent** in "equivalent martingale measure" has a precise, important meaning: $P$ and $Q$ agree on which outcomes are *possible*. Anything that can happen under $P$ can happen under $Q$, and vice versa — they only disagree on the *odds*, never on the *set* of outcomes. Equivalent measures never declare a real possibility impossible or invent an impossible one. This is what lets us move freely between the two worlds without losing or inventing scenarios. (The exact recipe for that move — the density $\frac{dQ}{dP}$ — is Girsanov's theorem, the subject of its own post; here we only need that the move is legitimate.)

So we have two worlds with identical scenery and different lighting. The real world $P$ is where you forecast, manage risk, and decide whether a trade is a *good bet*. The pricing world $Q$ is where you compute what a derivative is *worth*. Confusing them is the root of most beginner errors in pricing — the most common being to discount a payoff using the real-world probabilities, which gives a number that is not the price.

## 2. Pricing by expectation

Here is the punchline that the rest of the post justifies. **The price of any derivative is the expected value, under the risk-neutral measure $Q$, of its discounted payoff:**

$$
\text{price}_0 = E^Q\!\left[ e^{-rT} \cdot \text{payoff}_T \right].
$$

Every symbol earns its place. $\text{payoff}_T$ is the random amount the derivative will pay at its expiry time $T$ — for a call option struck at $K$, that payoff is $\max(S_T - K, 0)$, where $S_T$ is the stock price at expiry. The $e^{-rT}$ is the discount factor that converts that future dollar amount into today's dollars. And the $E^Q[\cdot]$ — the expectation *taken under $Q$, not $P$* — averages over all the ways the future could unfold, weighted by the risk-neutral probabilities. The little superscript $Q$ is doing enormous work; swap it for $P$ and the formula breaks.

![Four-stage pipeline from payoff through risk-neutral expectation and discounting to the price today](/imgs/blogs/martingales-risk-neutral-measure-math-for-quants-2.png)

The pipeline above is the entire recipe, and it never changes no matter how exotic the derivative. Step one: write down the payoff as a function of whatever the underlying does. Step two: take its expectation under the risk-neutral measure $Q$ — average the payoff across every future scenario, using risk-neutral odds. Step three: discount that expected payoff back to today at the risk-free rate. Step four: the number you are left with *is* the price. Pricing a call, a put, a barrier option, a swap, a mortgage-backed security — the payoff in step one changes, but the machine is always payoff → risk-neutral expectation → discount → price.

Why is this the right answer, and not, say, the expectation under the *real-world* $P$? The intuition, which the next sections make rigorous, is this: the price has to be the cost of *building* the payoff out of things you can already trade. If you can construct a portfolio of the stock and a bond that delivers exactly the derivative's payoff in every future scenario, then the law of one price forces the derivative to cost exactly what that portfolio costs — otherwise there is free money. And it happens that the cost of that replicating portfolio equals the discounted *risk-neutral* expectation. The real-world drift $\mu$ never appears because the replicating portfolio is rebalanced to cancel the stock's randomness; what is left is pure risk-free growth, and the risk-free rate is the only growth rate that survives.

> The price of a derivative is not a forecast of what it will pay. It is the cost of manufacturing that payoff out of things you can already buy.

That blockquote is worth taping to your monitor. Pricing is not prediction; it is manufacturing cost. Hold onto it through the worked examples, where we build the manufacturing line by hand.

## 3. The one-period binomial model

To make every abstract claim concrete, we will spend the rest of the post inside the simplest possible market that is still rich enough to need a risk-neutral measure: the **one-period binomial model**. One stock, one risk-free bond, one step into the future, exactly two possible outcomes. It is a toy, but every idea in continuous-time pricing — Black-Scholes included — is already present in it, just without the calculus getting in the way.

![Tree of the one-period binomial model with the stock branching up to 120 or down to 90](/imgs/blogs/martingales-risk-neutral-measure-math-for-quants-3.png)

The tree above lays out the model, and these are the numbers we will use for the rest of the post. The stock is worth $S_0 = \$100$ today. One period from now it will be in one of exactly two states: an **up** state where it is worth $S_u = \$120$, or a **down** state where it is worth $S_d = \$90$. We will write the up-factor as $u = 1.2$ (a 20% gain) and the down-factor as $d = 0.9$ (a 10% loss), so $S_u = uS_0$ and $S_d = dS_0$. There is also a risk-free bond: \$1 invested in it today grows to $R = 1 + r$ dollars next period. For the cleanest possible arithmetic in our first pass, we will start with a risk-free rate of **zero** ($r = 0$, so $R = 1$), then redo everything with a realistic $r = 5\%$ to show how much the answer moves.

The derivative we will price is a **call option** struck at $K = \$105$. A call gives its holder the right, but not the obligation, to buy the stock at the *strike price* $K = \$105$ at expiry. If the stock ends at \$120, the holder exercises: they buy at \$105 something worth \$120, pocketing $\$120 - \$105 = \$15$. If the stock ends at \$90, exercising would mean paying \$105 for something worth \$90 — so they decline, and the option pays \$0. The call's payoff is therefore \$15 in the up state and \$0 in the down state. That is the right column of the tree.

### Solving for the risk-neutral probability

Now we find $Q$ — the risk-neutral probability of the up move, which we call $q$. The defining requirement of $Q$ is that *the stock, under $Q$, is expected to grow at the risk-free rate*. In one period, with the risk-free growth factor $R$, that requirement reads:

$$
E^Q[S_T] = R \cdot S_0 \quad\Longleftrightarrow\quad q \cdot S_u + (1 - q) \cdot S_d = R \cdot S_0.
$$

This is one equation in one unknown $q$. Solving it gives the famous formula for the risk-neutral up-probability:

$$
q = \frac{R \cdot S_0 - S_d}{S_u - S_d} = \frac{R - d}{u - d}.
$$

Every quantity on the right is something the market hands you — the up- and down-factors and the risk-free rate. The *real-world* probability $p$ — your actual belief about how likely the stock is to rise — appears nowhere. That absence is the whole point, and the first worked example proves it pays a real dollar dividend.

#### Worked example: solve for q and price the call

Let us use our numbers with $r = 0$, so $R = 1$, $u = 1.2$, $d = 0.9$. The risk-neutral up-probability is

$$
q = \frac{R - d}{u - d} = \frac{1 - 0.9}{1.2 - 0.9} = \frac{0.1}{0.3} = \frac{1}{3} \approx 0.333.
$$

So under $Q$ the stock goes up with probability one-third and down with probability two-thirds. Sanity-check that this makes the stock a fair game: $E^Q[S_T] = \tfrac{1}{3}(\$120) + \tfrac{2}{3}(\$90) = \$40 + \$60 = \$100 = S_0$. With $r = 0$, "grow at the risk-free rate" means "stay flat on average," and it does. Good.

Now price the call. Its payoff is \$15 up, \$0 down. The price is the discounted risk-neutral expectation of that payoff — and with $r = 0$ the discount factor is just 1:

$$
C_0 = \frac{1}{R}\left[ q \cdot \$15 + (1 - q) \cdot \$0 \right] = \frac{1}{1}\left[ \tfrac{1}{3}(\$15) + \tfrac{2}{3}(\$0) \right] = \$5.
$$

The call is worth exactly **\$5**.

Now watch the magic. Suppose you are a raging bull who believes the *real-world* probability of the up move is $p = 0.6$, not one-third. Your real-world expected payoff from the call is $0.6 \times \$15 + 0.4 \times \$0 = \$9$. A naive trader would say "the option pays \$9 on average, so it should cost something like \$9." That is wrong. The *price* is \$5, full stop, no matter what you believe about $p$. Your bullishness changes whether buying the call at \$5 is a *good trade for you* (you think it is worth \$9 in expectation, so yes), but it does not change the *price*. The intuition: the price is the cost to manufacture the payoff, and as the next examples show, that manufacturing cost is \$5 regardless of anyone's opinion about which way the stock will go.

### The realistic wrinkle: a nonzero rate

The $r = 0$ case is clean but unrealistic. Redo it with $r = 5\%$ per period, so $R = 1.05$. The risk-neutral probability shifts:

$$
q = \frac{1.05 - 0.9}{1.2 - 0.9} = \frac{0.15}{0.30} = 0.5.
$$

A positive interest rate pushes $q$ up — the higher the risk-free rate, the more the risk-neutral world tilts toward the up move, because the stock has to be expected to keep pace with a now-higher risk-free return. The call price becomes

$$
C_0 = \frac{1}{1.05}\left[ 0.5(\$15) + 0.5(\$0) \right] = \frac{\$7.50}{1.05} \approx \$7.14.
$$

The price moved from \$5 to \$7.14 once money has a time value. We will carry the clean $r = 0$ numbers through the replication and arbitrage examples below (where \$5 and round portfolio weights keep the arithmetic transparent), and you should keep in your back pocket that turning on a realistic rate is a small, mechanical adjustment, never a change to the logic.

What does this model cost you in realism, and where does it break? A single up-or-down step is obviously a caricature: real stocks can land anywhere, not just at two prices. The fix is to chain the one-period model into a *multi-period binomial tree* — many small up-or-down steps stacked end to end — which converges, as the steps get small and numerous, to the continuous lognormal model behind Black-Scholes. Each node in that tree solves the very same one-equation problem for its own local $q$, and the price is built by working backwards from the payoffs at expiry, discounting one step at a time. So the toy is not thrown away; it is the atom from which the industrial models are assembled. Where it genuinely breaks is when a single step can produce *more than two* outcomes (a jump, a third state) and you still have only two hedging instruments — then, as the Second Fundamental Theorem warns, replication can fail and the unique price dissolves into a range.

## 4. The discounted stock is a Q-martingale

We motivated $Q$ as "the measure under which the stock grows at the risk-free rate." There is a cleaner, more powerful way to say the same thing, and it is the statement that ties this whole post to the word *martingale*: **under $Q$, the discounted stock price is a martingale.**

To "discount" the stock price means to express it in today's dollars by dividing out the risk-free growth. Define the discounted price $\tilde{S}_t = S_t / R^t$ (or $e^{-rt} S_t$ in continuous time). The claim is

$$
E^Q\!\left[ \tilde{S}_t \mid \mathcal{F}_s \right] = \tilde{S}_s.
$$

In words: in the pricing world, the *discounted* stock price is a fair game. It has no drift left in it — we discounted away exactly the risk-free growth, and $Q$ was built so that the risk-free growth is *all* the growth there is. What is left over is pure noise, a martingale. This is not a coincidence; it is the definition of $Q$, re-expressed. And it generalizes: under $Q$, the discounted price of *every* tradable asset — the stock, the bond, and every derivative — is a martingale. That single sentence is the engine of all arbitrage-free pricing.

#### Worked example: verify the discounted stock is a martingale

Take the $r = 5\%$ model, where $q = 0.5$. Today's discounted price is just today's price, $\tilde{S}_0 = S_0 = \$100$ (nothing to discount at time zero). One period out, the discounted price in the up state is $\tilde{S}_u = \$120 / 1.05 = \$114.29$, and in the down state $\tilde{S}_d = \$90 / 1.05 = \$85.71$.

Now take the risk-neutral expectation of the discounted future price:

$$
E^Q[\tilde{S}_T] = q \cdot \frac{\$120}{1.05} + (1 - q) \cdot \frac{\$90}{1.05} = 0.5(\$114.29) + 0.5(\$85.71) = \$100.
$$

It equals $\tilde{S}_0 = \$100$ exactly. The discounted stock is a martingale under $Q$: its expected discounted future value equals its discounted value today. Equivalently, the *undiscounted* expectation is $E^Q[S_T] = 0.5(\$120) + 0.5(\$90) = \$105 = \$100 \times 1.05 = S_0 \cdot R$, which is the relation $E^Q[S_T] = S_0 e^{rT}$ written for one discrete period.

Contrast this with the real world. If your real-world up-probability is $p = 0.6$, then $E^P[S_T] = 0.6(\$120) + 0.4(\$90) = \$108$ — an expected gross return of 8%, comfortably above the 5% risk-free rate. That extra 3% is the risk premium, the reward the market pays you for holding a risky stock instead of a safe bond. Under $P$ the stock outgrows cash; under $Q$ it merely matches cash. The intuition: $Q$ is the unique re-weighting of the same two outcomes that strips out the risk premium and leaves a fair game, and that fair game is precisely what makes prices computable.

### The numéraire: what "discounted" really means

We have been quietly dividing by the risk-free growth factor $R^t$. That risk-free asset — the money-market account that turns \$1 into $R^t$ dollars — has a name: the **numéraire**, the yardstick we measure all other prices against. Discounting is just "measuring prices in units of the numéraire instead of dollars." The deeper truth, which matters as you go further, is that *the choice of numéraire is yours to make*. You can measure prices in units of the bond (the standard risk-neutral measure), in units of the stock itself, or in units of a zero-coupon bond maturing at $T$ (the "forward measure" used to price interest-rate options). Each choice of numéraire comes paired with its own martingale measure under which prices, measured in that numéraire, are martingales. This *change-of-numéraire* freedom is one of the most powerful tools in the quant's kit, because choosing the right yardstick can turn an ugly expectation into a clean one. For now, the only numéraire we need is the risk-free bond, and "discounted" means "in bond units."

## 5. Arbitrage when a price is wrong

So far we have asserted that the call is worth \$5 and that the real-world odds do not matter. Now we *prove* it, in the most convincing way finance has: by showing that any other price hands a free-money machine to the rest of the market. This is the practical face of the no-arbitrage principle, and it is how trading desks actually keep each other honest.

![Before and after columns contrasting a fairly priced option with an arbitrageable one](/imgs/blogs/martingales-risk-neutral-measure-math-for-quants-6.png)

The figure contrasts the two situations. On the left, the call is quoted at its fair value of \$5: it costs exactly what the replicating portfolio costs, so building the hedge and selling the option nets you nothing, and there is no free money to be had. On the right, someone foolishly quotes the call at \$7. Now the option is overpriced relative to the \$5 portfolio that copies it, and a sharp trader can sell the expensive thing, buy the cheap copy, and lock in the \$2 gap with zero risk. Let us walk that arbitrage step by step with real dollars.

#### Worked example: a concrete \$2 arbitrage

We are in the $r = 0$ world, where the fair call price is \$5 and (jumping ahead to the next section) the replicating portfolio is "hold half a share of stock, financed by borrowing \$45." Suppose a careless market-maker quotes the call at **\$7** — \$2 above fair value. Here is how to extract \$2 of risk-free profit, today, that you keep no matter what the stock does.

**Today (time 0):**

- **Sell** one call at the inflated price. You receive $+\$7$ in cash.
- **Build the replicating portfolio** that will exactly cover your obligation: buy half a share of stock (costing $0.5 \times \$100 = \$50$) and borrow \$45 from the risk-free account (you receive $+\$45$). The net cost of this portfolio is $\$50 - \$45 = \$5$.
- Your net cash today: $+\$7$ (from selling the call) $-\$5$ (to build the hedge) $= +\$2$. You pocket **\$2 right now**, with no money of your own at stake.

**At expiry (time 1), the up state — stock at \$120:**

- The call you sold is exercised against you: you must pay the holder $\$120 - \$105 = \$15$. That is a $-\$15$ cash flow.
- Your hedge: your half share is worth $0.5 \times \$120 = \$60$; you repay the \$45 loan (with $r = 0$, you owe exactly \$45). Net from the hedge: $\$60 - \$45 = +\$15$.
- The $-\$15$ obligation and the $+\$15$ hedge cancel exactly. Net at expiry: \$0.

**At expiry (time 1), the down state — stock at \$90:**

- The call you sold expires worthless; you owe nothing. Cash flow: \$0.
- Your hedge: your half share is worth $0.5 \times \$90 = \$45$; you repay the \$45 loan. Net from the hedge: $\$45 - \$45 = \$0$.
- Net at expiry: \$0.

In *both* future states, your obligations and your hedge cancel to exactly zero. You walk away having kept the \$2 you collected at the start — a riskless profit conjured purely from the mispricing. This is arbitrage: it cost you nothing, it cannot lose, and it made money. The intuition: the only price that does *not* hand someone this free \$2 machine is \$5, which is exactly why \$5 is *the* price. Quote anything higher and sellers feast; quote anything lower and the mirror-image strategy (buy the cheap call, short the portfolio) feasts. The fair price is the unique no-free-lunch price.

This is also why the real-world probability is irrelevant in a way you can now *feel*: nowhere in the arbitrage did we use $p$. The hedge cancels in *every* state, so it does not matter how likely each state is. A strategy that wins in every possible world does not care about the odds of those worlds.

## 6. Replication and the law of one price

We have now used the replicating portfolio twice without deriving it. Time to build it from scratch, because replication is the deepest reason pricing-by-expectation works at all. The principle behind it has a name: the **law of one price**. It says that two things that deliver the *same* cash flows in *every* state of the world must have the *same* price today. If they did not, you would buy the cheap one, sell the expensive one, and collect the difference risk-free — exactly the arbitrage we just ran. The law of one price is no-arbitrage applied to look-alikes.

So if we can build a portfolio of the stock and the bond that pays *exactly* what the call pays — \$15 in the up state, \$0 in the down state — then by the law of one price, the call must cost exactly what that portfolio costs. We do not need to forecast anything; we just need to find the recipe and read off its price tag.

![Stack showing the replicating portfolio of half a share and a borrowed bond summing to a five dollar cost](/imgs/blogs/martingales-risk-neutral-measure-math-for-quants-7.png)

The stack above is the answer we are about to derive: hold half a share of stock and borrow \$45, for a net cost of \$5. Let us find those two numbers from the requirement that the portfolio match the call in both states.

#### Worked example: build the replicating portfolio

We want a portfolio of $\Delta$ shares of stock (the Greek letter delta, the number of shares to hold) and $B$ dollars in the risk-free bond, chosen so it pays exactly the call's payoff in each state. With $r = 0$, the bond holding $B$ is worth $B$ in both future states (no interest). The two matching conditions are:

- **Up state:** $\Delta \cdot \$120 + B = \$15$ (must equal the call's up payoff).
- **Down state:** $\Delta \cdot \$90 + B = \$0$ (must equal the call's down payoff).

Subtract the second equation from the first to eliminate $B$:

$$
\Delta(\$120 - \$90) = \$15 - \$0 \quad\Longrightarrow\quad \Delta = \frac{\$15}{\$30} = 0.5 \text{ shares}.
$$

This $\Delta$ is the **hedge ratio** or option *delta* — the number of shares that exactly offsets one option, and the single most important number on an options desk. Now plug $\Delta = 0.5$ back into the down-state equation to find $B$:

$$
0.5 \times \$90 + B = \$0 \quad\Longrightarrow\quad \$45 + B = \$0 \quad\Longrightarrow\quad B = -\$45.
$$

The negative sign means you *borrow* \$45 rather than invest it. So the replicating portfolio is: **buy half a share of stock and borrow \$45.** Its cost today is

$$
\text{cost} = \Delta \cdot S_0 + B = 0.5 \times \$100 + (-\$45) = \$50 - \$45 = \$5.
$$

There it is again: **\$5**, the exact price we found by risk-neutral expectation. Two completely different routes — discounting an expectation under an artificial measure, and pricing a hand-built copy with no probabilities at all — land on the identical number. That is not luck; it is a theorem. The intuition: the risk-neutral expectation *is* the cost of the replicating portfolio, dressed up in probability notation. The \$5 is the manufacturing cost of the payoff, and every valid pricing method is just a different way of reading that same price tag.

Let us double-check the replication actually works, state by state. In the **up** state the portfolio is worth $0.5 \times \$120 - \$45 = \$60 - \$45 = \$15$ ✓ (matches the call). In the **down** state it is worth $0.5 \times \$90 - \$45 = \$45 - \$45 = \$0$ ✓ (matches the call). The portfolio is a perfect clone in both states, so the law of one price nails the call to its \$5 cost.

### Why a derivative can be hedged at all: martingale representation

There is one more layer, and it answers a question you may not have known to ask: *why* is it always possible to build a replicating portfolio? In our two-state model it worked because we had two unknowns ($\Delta$ and $B$) and two equations (up and down), so a solution exists. But in a continuous-time model with infinitely many possible price paths, that counting argument collapses — why should a single, continuously-rebalanced stock-and-bond portfolio be able to track an option's value through every wiggle?

The answer is a beautiful result called the **martingale representation theorem**. Stripped of its technicalities, it says: in a complete market, *any* martingale can be written as a running sum of bets on the underlying martingale (the discounted stock). Translated into trading: the discounted value of the option is itself a $Q$-martingale, and the theorem guarantees it can be reproduced by a self-financing trading strategy in the stock — and that strategy's holding *is* the delta, $\Delta_t = \partial C / \partial S$. In plain terms, the theorem promises that the hedge always exists and tells you it is exactly the option's sensitivity to the stock. This is the deep reason that delta-hedging works and that derivatives can be manufactured rather than merely guessed at. The "delta" you compute in Black-Scholes is the continuous-time descendant of the $\Delta = 0.5$ we just solved for by hand.

## 7. The two fundamental theorems

We can now state the two results that organize this entire subject. They are called the **fundamental theorems of asset pricing**, and together they are the load-bearing wall of quantitative finance.

![Matrix laying out the first and second fundamental theorems of asset pricing](/imgs/blogs/martingales-risk-neutral-measure-math-for-quants-4.png)

The matrix above lays them out side by side. Read each row as "market property → measure result → what it buys you."

The **First Fundamental Theorem of Asset Pricing** states: *a market is free of arbitrage if and only if there exists at least one risk-neutral (equivalent martingale) measure $Q$ under which all discounted asset prices are martingales.* This is the "existence" theorem, and the *if and only if* is the heart of it. One direction is the easy one we have been feeling our way toward: if such a $Q$ exists, there can be no arbitrage, because under $Q$ the discounted value of any self-financing strategy is a martingale, and a martingale that starts at zero cannot have a guaranteed positive future value (a fair game cannot be rigged to always win). The other direction — that no-arbitrage *forces* a $Q$ to exist — is the deep one, and it is essentially a separating-hyperplane argument: if no portfolio offers free money, then prices live in a configuration that can always be reproduced by *some* set of positive risk-neutral weights. The upshot is breathtaking in its economy: **"no free money" and "prices are discounted expectations" are logically equivalent.** You cannot have one without the other.

The **Second Fundamental Theorem of Asset Pricing** states: *an arbitrage-free market is complete if and only if the risk-neutral measure $Q$ is unique.* A market is **complete** when *every* payoff can be replicated by trading the available assets — when there is nothing you might want to manufacture that you cannot. The theorem says completeness and uniqueness are the same thing. If $Q$ is unique, then the formula $\text{price} = E^Q[\text{discounted payoff}]$ returns a single, unambiguous number for every derivative — there is no wiggle room. If instead there are *many* risk-neutral measures (an incomplete market), then different choices of $Q$ give different prices, and the no-arbitrage principle alone is no longer enough to pin down a unique value; it only gives you a *range* of arbitrage-free prices, and you need something beyond no-arbitrage (a model, a calibration, a view) to choose within that range.

Our one-period binomial model is *complete*: two states, two tradable assets (stock and bond), so every two-state payoff is replicable, and $Q$ is the unique pair $(q, 1-q)$ we solved for. That is exactly why the call had a single, definite price of \$5. Add a third possible state to the same two assets and the market becomes *incomplete* — you would have three payoff equations but only two instruments to solve them with, replication can fail, and a band of arbitrage-free prices opens up. Real markets are incomplete (because of jumps, stochastic volatility, transaction costs, and more), which is precisely why pricing in practice requires a *model* and not just the no-arbitrage principle — and why two desks can disagree on the price of an exotic option while both being arbitrage-free.

### The logic in one chain

It is worth pausing to see the whole argument as a single chain, because the pieces snap together in a particular order.

![Stack of the logical steps from no-arbitrage to discounted prices being martingales](/imgs/blogs/martingales-risk-neutral-measure-math-for-quants-5.png)

The stack above is the chain. Start at the top with the only assumption we ever make: **no arbitrage**, no free money. By the First Fundamental Theorem, that assumption *forces* a risk-neutral measure $Q$ to exist. Under that $Q$, by construction, every asset is expected to grow at exactly the risk-free rate — which is the same as saying the *discounted* price has zero drift. A discounted price with zero drift, with all the risk-free growth divided out, is by definition a **martingale**. So the chain runs: no free money → a pricing measure exists → discounted prices have no drift → discounted prices are fair games. Every link is forced by the one before it. The reason discounted prices are martingales is not an extra assumption you have to swallow; it is the unavoidable consequence of forbidding free money. That is the single most important sentence in this post, and the chain above is its proof in pictures.

## Common misconceptions

The risk-neutral measure trips up nearly everyone the first time, and a few specific wrong beliefs are worth correcting head-on.

**"The risk-neutral probability is what will actually happen."** No. The risk-neutral probability $q$ is a *pricing device*, not a forecast. In our model $q = 1/3$ (or $0.5$ with $r=5\%$), but the stock's *real* chance of rising might be 60% or 20% — $q$ has nothing to say about it. Treating $q$ as a prediction is the single most common error. The frequencies you would observe over a thousand repetitions are governed by $P$; $q$ is the artificial weight that makes prices come out arbitrage-free. They are different numbers serving different jobs.

**"A higher expected return on the stock should make its options more expensive."** It feels obvious — if the stock is going to rip, surely the call to buy it cheaply is worth more. But we proved it false: the call cost \$5 whether you thought the up-probability was one-third or sixty percent. The reason is replication. The call's price equals the cost of a hedge that cancels the stock's randomness in every state, and that hedge costs the same regardless of which way you think the stock is headed. Your bullishness affects whether buying the call is a *good bet* for you; it does not move the *price*.

**"Risk-neutral means investors don't care about risk."** Real investors care enormously about risk — that is why the real-world stock drift $\mu$ exceeds the risk-free rate $r$ by a risk premium. "Risk-neutral" is a property of the *pricing measure* $Q$, an artificial construct, not a claim about human psychology. The trick of pricing is that we can compute the *same* correct price *as if* investors were risk-neutral, because the risk premium gets canceled by the hedge. The risk preferences are still there in reality; they just do not survive into the price.

**"You can use the real-world probabilities if you discount at a risk-adjusted rate instead of $r$."** This is actually true in principle — it is the old "discounted cash flow" approach — but it is a trap in practice, because the correct risk-adjusted discount rate for a *derivative* changes from moment to moment and from state to state, and is fiendishly hard to find. The entire genius of the risk-neutral approach is that it lets you discount at the *single, known, constant* risk-free rate by adjusting the probabilities instead. You move the difficulty from an unknowable discount rate into a computable change of measure. That trade is almost always worth it.

**"Martingale means the price never moves."** A martingale is not constant; it wanders all over the place. What is "fair" about it is only the *expectation*: your best forecast of its future value equals its current value. The discounted stock in our model jumps from \$100 to either \$114.29 or \$85.71 — hardly motionless — yet it is a martingale because the probability-weighted average of those two is back to \$100. Zero *expected* drift, plenty of actual motion.

**"Completeness is a technicality."** It is the difference between a unique price and a range of prices. In a complete market the Second Fundamental Theorem hands you one number; in an incomplete one (which is to say, every real market) you get a band, and you need a model to choose within it. The gap between the bid and the offer on a hard-to-hedge exotic is, in part, the width of that band made visible. Completeness is where the clean theory meets the messy market, and pretending it always holds is how model risk sneaks in.

## How it shows up in real markets

The risk-neutral framework is not a chalkboard curiosity; it is the operating system of every derivatives desk on Earth. Here are concrete places it surfaces.

### 1. Black-Scholes is this argument in continuous time

The Black-Scholes formula, which underpins a derivatives market measured in the hundreds of trillions of dollars of notional, is exactly the one-period story repeated infinitely often. Replace "two states" with "a continuum of price paths," replace the algebraic hedge ratio $\Delta = 0.5$ with the calculus derivative $\partial C / \partial S$, and replace the discrete risk-neutral probabilities with a risk-neutral *lognormal distribution* for the stock — and the binomial price converges to the Black-Scholes price. Famously, the stock's real-world expected return $\mu$ does not appear anywhere in the Black-Scholes formula; only the volatility $\sigma$ and the risk-free rate $r$ do. Everything we said about \$5 not depending on $p$ is *why* $\mu$ is absent from the most important equation in finance. For the full derivation, see the deep-dive on [Black-Scholes](/blog/trading/quantitative-finance/black-scholes).

### 2. Implied volatility: the market quoting $Q$ back to you

When traders quote an option's price, they usually quote it as an *implied volatility* — the value of $\sigma$ that, plugged into the risk-neutral pricing formula, reproduces the observed price. Reading implied vols off the market is, quite literally, reading off the market's risk-neutral distribution for the stock. When implied vol is higher for low strikes than high strikes — the famous *volatility skew* — it means the market's risk-neutral measure $Q$ assigns *fatter left tails* (bigger crash probabilities) than a plain lognormal would. The skew is the risk-neutral measure made visible, and it is the central object in the post on the [volatility surface](/blog/trading/quantitative-finance/volatility-surface). Note that the risk-neutral crash probability embedded in the skew is generally *larger* than the real-world one — investors pay up for crash protection, and that premium shows up as a steeper $Q$ than $P$.

### 3. The 1987 crash and the birth of the skew

Before October 1987, equity options were priced as if the stock followed a clean lognormal — implied vols were roughly flat across strikes, implying a risk-neutral measure with thin tails. Then, on October 19, 1987, the S&P 500 fell about 20% in a single day, an event so many standard deviations out that under the pre-crash $Q$ it had a probability indistinguishable from zero. The market never forgot. Ever since, out-of-the-money equity puts have traded at a persistent premium — a steep volatility skew — because the market's risk-neutral measure now bakes in a meaningful chance of a crash that the old thin-tailed $Q$ ruled out. The skew is, in effect, the market pricing its memory of 1987 into $Q$. It is a vivid reminder that the risk-neutral measure is set by *prices*, not by historical frequencies, and prices encode fear.

### 4. Convertible-bond and capital-structure arbitrage

A convertible bond is a corporate bond that can be swapped for a fixed number of shares — a bond with a call option bolted on. Hedge funds run "convert arb" by buying the convertible and shorting the right amount of stock (the delta), manufacturing a near-riskless position out of a mispriced embedded option. This is replication and the law of one price applied to a real, traded instrument: the fund is exploiting a gap between the convertible's market price and the cost of the replicating stock-and-bond portfolio. When the gap is real, it is the \$2 arbitrage of our worked example, scaled up to millions and hedged continuously. When the hedge slips — because the market is incomplete, liquidity dries up, or the stock gaps — the "riskless" trade turns out to carry the very risk the model assumed away, as several funds learned painfully in 2008.

### 5. Interest-rate derivatives and the forward measure

Pricing a cap, a swaption, or a Bermudan callable bond uses exactly the same recipe — price equals discounted risk-neutral expectation of payoff — but practitioners almost never use the plain money-market numéraire there. They switch to a *forward measure*, taking a zero-coupon bond as the numéraire, because under that measure the relevant forward interest rate becomes a martingale and the expectation simplifies enormously. This is the change-of-numéraire trick in daily industrial use: a trillion-dollar rates market runs on choosing the yardstick that makes the martingale clean. The short-rate models that feed these calculations — Vasicek, Hull-White — are surveyed in [short-rate models](/blog/trading/quantitative-finance/short-rate-models-vasicek-hull-white), and the curve they price off in [yield-curve modeling](/blog/trading/quantitative-finance/yield-curve-modeling).

### 6. Monte Carlo pricing on a trading desk

When a payoff is too complex for a formula — a path-dependent Asian option, a multi-asset basket, a structured note — desks price it by brute force: simulate millions of *risk-neutral* paths of the underlying (drift set to $r$, not $\mu$), compute the payoff on each path, average them, and discount. That average-and-discount is literally $E^Q[e^{-rT}\,\text{payoff}]$ evaluated numerically. The single most common bug in a junior quant's Monte Carlo pricer is simulating paths under the real-world drift $\mu$ instead of the risk-neutral $r$ — which produces a number that is not a price and that no amount of running more paths will fix. The discipline of "simulate under $Q$, not $P$" is the practical face of everything in this post.

### 7. Where the framework genuinely breaks

The clean theory assumes you can trade continuously, at no cost, in any size, with no funding constraints, in a complete market. Reality violates every clause. Transaction costs mean you cannot rebalance the hedge infinitely often, so the replication is approximate and the residual risk is real. Jumps in the price mean a delta hedge calibrated to small moves fails catastrophically in a gap — the 2008 and 2020 crashes both saw "hedged" books take large losses. Incompleteness means $Q$ is not unique and the model you choose to pin it down can be wrong (model risk). And in a crisis, the risk-free rate itself becomes ambiguous and funding costs explode, so even the discounting is uncertain. The framework is indispensable and also a map, not the territory; the best practitioners hold both ideas at once.

## When this matters to you

If you ever buy an option — even a simple covered call on a stock you own, or a put as portfolio insurance — the price you pay was computed by exactly this machinery. Understanding it changes how you read those prices: the option is not "expensive because the market expects the stock to soar," it is expensive because the *cost to manufacture its payoff* is high, which usually means volatility is high. When you see an implied-vol skew, you are looking at the market's risk-neutral fear, not its forecast. And the next time someone tells you an option is cheap because they are bullish on the underlying, you will know that bullishness is a reason to *buy* at the price, not a reason the *price* should be different.

The deeper payoff is conceptual. The risk-neutral idea — that you can price a risky thing by pretending the world is risk-neutral, provided you can hedge — is one of the genuinely great ideas of applied mathematics, on par with calculus or the Fourier transform in its reach. It says that the price of risk is determined not by anyone's appetite for it but by the cost of getting rid of it. Once that clicks, a great deal of finance stops looking like gambling and starts looking like engineering.

A standard, necessary caveat: this is an educational tour of how pricing works, not advice to buy or sell anything. Options can lose their entire value; replicating strategies can fail when the assumptions behind them do; and every number here lives inside a model that real markets will, eventually, violate.

**Further reading.** To go deeper, the natural next steps on this blog are the companion math-for-quants post on [probability spaces and random variables](/blog/trading/math-for-quants/probability-spaces-random-variables-math-for-quants) (the rigorous foundation for filtrations and conditional expectation that underlies the martingale definition), and the quant-interview deep-dives on [risk-neutral pricing and the martingale measure](/blog/trading/quantitative-finance/risk-neutral-pricing-martingale-measure-quant-interviews) (the same ideas drilled in interview form) and on [put-call parity and no-arbitrage](/blog/trading/quantitative-finance/put-call-parity-no-arbitrage-quant-interviews) (the cleanest possible example of pricing forced entirely by the absence of free money). From there, Girsanov's theorem makes the change from $P$ to $Q$ explicit, and Feynman-Kac connects the expectation $E^Q[\cdot]$ to the Black-Scholes partial differential equation — two threads that turn this one-period story into the full continuous-time theory.

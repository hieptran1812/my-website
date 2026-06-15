---
title: "Cointegration and pairs trading: how a tethered gap becomes a money machine"
date: "2026-06-15"
description: "Why two prices can wander forever yet stay tied together, how to test for that tether, and how a quant turns the gap into a mean-reverting trade with real dollar edge."
tags: ["cointegration", "pairs-trading", "statistical-arbitrage", "mean-reversion", "engle-granger", "johansen-test", "error-correction-model", "time-series", "quantitative-finance"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Two stock prices can each wander randomly forever and yet stay invisibly tethered, so that a specific combination of them — their *spread* — keeps snapping back to a fixed level; that tether is called **cointegration**, and trading the snap-back is the foundation of **pairs trading**.
>
> - **Correlation of prices lies.** Two completely unrelated random walks routinely show a sky-high correlation and a regression with an $R^2$ of 0.9 — pure statistical illusion called *spurious regression*. You must never test a trading relationship on the raw prices.
> - **Cointegration is the real test.** Two prices are cointegrated if some combination $y_t - \beta x_t$ is *stationary* — it has a fixed mean it keeps returning to, even though each price alone never settles down. That stationary combination is your tradeable spread.
> - **The trade is mechanical.** Convert the spread to a **z-score** (how many standard deviations it sits from its mean), go in when it stretches to about $\pm 2$, and close when it returns to $0$. The **half-life** of mean reversion tells you how long you'll wait and how big to size the bet.
> - The one number to remember: in our worked example, a single \$1,000,000 market-neutral pair earns about **\$30,000** on one clean round trip from a 2-sigma stretch back to the mean — and loses it all and more if the cointegration quietly breaks.

In 1985, a small group at the investment bank Morgan Stanley noticed something that should not have been profitable. Coca-Cola and Pepsi sell the same thing to the same people; their stock prices march up and down together for the same reasons. Most days the two move in lockstep. But every so often one would jump ahead of the other for no fundamental reason — a big buyer needed Coke shares that afternoon, say — and the gap between them would stretch wider than usual. The group's quiet bet was simple: when the gap stretched, *short the one that ran ahead and buy the one that lagged*, and wait for the gap to close again. It almost always did. That desk, and the strategy it pioneered, came to be known as **statistical arbitrage**, and over the next decade it printed money so reliably that it reshaped Wall Street.

The mathematics that makes this work is one of the most beautiful and most misunderstood ideas in all of quantitative finance: **cointegration**. It is the precise statement of "these two prices are tethered, even though neither one ever sits still." By the end of this post you will be able to spot the trap that fools beginners (two random series that *look* related but are not), test a real pair for the genuine tether, build the spread, turn it into a trading signal, size the position to a dollar risk budget, and understand exactly why even a beautifully backtested pair eventually stops working. We will do all of it with concrete dollar examples on a \$1,000,000 book.

![Before and after panels showing two drifting prices on the left and a flat stationary spread on the right](/imgs/blogs/cointegration-pairs-trading-math-for-quants-1.png)

The figure above is the whole idea in one picture. On the left, two stock prices each climb and dip on their own — neither one has a level it returns to, they just wander. That wandering is what statisticians call a **random walk**, and it is the single most important shape in finance. On the right is the magic: a specific *combination* of those two wandering prices — the first one minus a multiple of the second — does not wander at all. It oscillates tightly around a flat line and keeps coming back. That flat-hugging combination is the **spread**, and every dollar a pairs trader makes comes from betting that the spread will return to its line. The rest of this post is about why that combination exists, how to find it, and how to trade it without blowing up.

## Building blocks: the tools you need

Before any of the real machinery, we need a small vocabulary. If you have never heard the terms "stationary," "random walk," or "unit root," this section builds them from absolutely nothing. A practitioner can skim; a beginner should read every line, because every later idea is just these four pieces rearranged.

### What a "stationary" series is

A time series is just a number recorded over and over through time — a stock's closing price each day, the temperature each noon, your weight each morning. A series is **stationary** if its basic statistical behavior does not change as time passes: it has a fixed average level it hovers around, a fixed amount of bounce (variance), and the way today relates to yesterday stays the same year after year.

The cleanest mental handle is a leash. Picture a dog on a fixed leash tied to a post. The dog runs around, but it can never get more than a leash-length from the post, and on average it sits near the post. That is a stationary series: it has a *home* (the mean) and it always gets pulled back. Daily temperature in a city is roughly stationary — it varies wildly day to day, but the long-run average for July is the same this decade as last, and an unusually hot day is followed, on average, by a cooler one. The technical phrase for "gets pulled back to its home" is **mean reversion**, and mean reversion is the single property a pairs trader is hunting for.

### What a "random walk" is, and why prices are one

Now untie the leash. A **random walk** is a series with no home. Each step, you add a fresh random shock to wherever you already are: today's value equals yesterday's value plus a coin-flip-sized nudge. Formally,

$$
x_t = x_{t-1} + \epsilon_t,
$$

where $x_t$ is today's level, $x_{t-1}$ is yesterday's, and $\epsilon_t$ is a fresh random shock with mean zero (sometimes up, sometimes down, no pattern). Here is the crucial part: because each shock is *added on top of* the running total and never decays, the shocks accumulate forever. There is no force pulling the series back to any level. A random walk that wanders to 200 has no more reason to come back to 100 than to keep going to 300. A drunk stumbling home with no sense of direction traces a random walk: each step is random, and over an hour he can end up anywhere.

Stock prices, to a very good first approximation, are random walks. This is not a coincidence — it is what an efficient market *should* look like. If everyone already knew a stock would rise tomorrow, they would buy it today, which pushes the price up today and erases the predictable move. What's left is the unpredictable part: a random walk. This is why the raw price of a single stock is famously almost impossible to forecast.

### The one technical term: a "unit root" and I(1) versus I(0)

The line $x_t = x_{t-1} + \epsilon_t$ has a hidden coefficient: the number multiplying $x_{t-1}$ is exactly $1$. That number is called the **root**, and when it equals one we say the series has a **unit root**. A unit root is the mathematical signature of "no home, wanders forever" — it is exactly what makes a random walk non-stationary. If that coefficient were less than one, say $0.9$, then each period the series would shrink 10% of the way back toward zero before getting nudged again, giving it a home: that would be stationary.

Quants use a compact shorthand for all this:

- A series is **I(0)** — "integrated of order zero" — if it is *already* stationary. The spread we trade is I(0). It has a home.
- A series is **I(1)** — "integrated of order one" — if it is not stationary, but its *changes* (today minus yesterday) are stationary. A random-walk price is I(1): the price wanders forever, but the daily *change* in price is just $\epsilon_t$, a stationary set of random shocks. "Integrated of order one" literally means "you have to difference it once — take changes — to make it stationary."

Almost every raw price series in finance is I(1). The entire trick of cointegration, as we will see, is finding a way to combine two I(1) series into something that comes out I(0).

To make the I(1)-versus-I(0) distinction stick, here is a small comparison table you can return to. It contrasts the wandering price you cannot trade with the stationary spread you can.

| Property | A raw price (I(1)) | A cointegrated spread (I(0)) |
| --- | --- | --- |
| Has a fixed mean it returns to | No — wanders forever | Yes — that is its "home" |
| Variance over time | Grows without bound | Stays bounded and stable |
| Forecastable from its own level | No — next move is a coin flip | Yes — far from mean means snap-back due |
| What you do with it | Cannot trade the level | Trade every stretch away from the mean |

Read the last row twice: the whole reason a quant cares about cointegration is that it manufactures the right-hand column out of two left-hand-column ingredients. You cannot trade a wandering price, but you *can* trade the bounded, mean-reverting spread that a cointegrating combination of two wandering prices produces.

![Pipeline from two price series through regression to a spread to a z-score trading signal](/imgs/blogs/cointegration-pairs-trading-math-for-quants-2.png)

The pipeline above is the road map for the whole post, and it is worth memorizing because every section below is one box in it. You start with two raw price series (both I(1), both wandering). You regress one on the other to find the right multiple — the **hedge ratio**. You subtract to form the **spread**. You test that the spread is genuinely stationary (I(0)). You standardize it into a **z-score**. And the z-score finally tells you when to enter and exit. Two wandering prices in, one clean trading signal out. We will now walk each box, starting with the trap that catches everyone who skips straight to "are these two prices correlated?"

## The spurious regression trap

Here is the mistake that has destroyed more naive trading strategies than any other. You take two stocks, you compute the correlation of their prices, you see a number like $0.95$, and you conclude they are deeply related and tradeable. You run a regression of one price on the other and the $R^2$ — the fraction of variation "explained" — comes back at $0.92$ with a t-statistic so large it looks like a sure thing. You feel like you have found gold.

You have found nothing. This is **spurious regression**, and it was first nailed down by the econometricians Clive Granger and Paul Newbold in 1974, in a paper that genuinely changed the field. Their discovery: *two completely independent random walks — series with no relationship whatsoever — will routinely produce a high correlation and a high-$R^2$ regression purely by chance.* The statistics that are supposed to detect a relationship break down completely when the inputs are I(1).

Why does this happen? Because both series are wandering, and over any finite stretch of time, two wanderers will *appear* to trend together or against each other simply because each one happens to be drifting somewhere. Correlation and regression assume the data has a stable mean to measure deviations from. A random walk has no stable mean, so the math measures a relationship that does not exist. The longer the sample, the *worse* the illusion gets — the spurious t-statistics grow without bound. This is the opposite of how statistics is supposed to behave, and it is why you can never test a trading relationship on raw price levels. If your intuition says "but the correlation is 0.95, surely that means something," the next worked example is designed to break that intuition cleanly.

#### Worked example: two random walks that fake a relationship

Let us manufacture the illusion with our own hands so you trust it. We build two series, $A$ and $B$, that are *guaranteed* to have nothing to do with each other, then we measure their "relationship."

Generate $A$ as a pure random walk: start at \$100, and each of 250 trading days (one year) add a random daily change drawn independently with mean \$0 and a standard deviation of \$1. Generate $B$ the exact same way, with its *own* independent random shocks — a different stream of coin flips entirely. The two share no shocks, no driver, no economic link. By construction they are independent.

Now compute the correlation of the two price *levels* over the year. In a typical draw of this experiment you will see a correlation around \$0.8\$ or higher in absolute value — and across many repetitions, the correlation is spread almost uniformly between $-1$ and $+1$, with values above $0.8$ appearing roughly a quarter of the time. Run the regression $A_t = \alpha + \beta B_t + \text{error}$ and you routinely get an $R^2$ above $0.7$ and a t-statistic on $\beta$ above $10$, which under the textbook rules would be "overwhelmingly significant." A trader looking only at these numbers would size up a \$1,000,000 position with total confidence.

Here is the tell that exposes the fraud: look at the regression *residuals* — the leftover errors after fitting the line. For a genuine relationship those residuals are stationary noise hugging zero. For two random walks they are themselves a wandering, non-stationary mess that never settles. The residual is the spread you would trade, and a spread that wanders is a spread that never reverts, which means every "mean-reversion" trade you place can run against you forever. **The one-sentence lesson: a high price correlation and a high $R^2$ prove nothing about two I(1) series; the only thing that matters is whether the leftover spread is stationary, and for unrelated random walks it never is.** This is exactly the trap detailed in the [covariance and correlation pitfalls](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews) post, here in its most expensive form.

## What cointegration actually is

So correlation is a liar on prices. What is the honest test? It is **cointegration**, and the definition is short enough to memorize: two series $x_t$ and $y_t$ are cointegrated if they are each I(1) — each wanders forever on its own — but there exists some number $\beta$ such that the combination

$$
z_t = y_t - \beta x_t
$$

is I(0), that is, *stationary*. The combination has a home even though the ingredients do not. The number $\beta$ is the **cointegrating coefficient**, and in trading it is the **hedge ratio**: it tells you how many units of $x$ to trade against one unit of $y$ so that the leftover, $z_t$, is the well-behaved, mean-reverting spread.

The everyday picture is a drunk walking her dog. The woman traces a random walk — she wanders, no home, fully I(1). The dog also wanders, I(1) on its own. But they are joined by a leash. So while *each* of them, tracked alone, looks like an aimless wanderer that could be anywhere, the *distance between them* — woman's position minus dog's position — can never exceed the leash and keeps getting pulled back. That distance is stationary. That distance is the spread. Cointegration is the statistical statement "there is a leash."

This is the deep reason correlation and cointegration are different things, and confusing them is the most common error in the field. Correlation is about whether two series move *together in the short run, step by step*. Cointegration is about whether two series are *tied together in the long run* so the gap between them stays bounded. They are genuinely independent properties: you can have either without the other.

![Matrix contrasting correlation and cointegration across daily co-movement, bounded gap, and tradeable spread](/imgs/blogs/cointegration-pairs-trading-math-for-quants-3.png)

The matrix above lays out all four combinations, and each corner is a real thing that happens in markets. Two stocks can be **correlated but not cointegrated**: they jiggle together day to day, but their gap drifts ever wider with no anchor — trading the gap is a trap because it never reverts. Two stocks can be **cointegrated but barely correlated** day to day: the gap is tightly tethered over months even though individual days look unrelated — this is the *best* kind of pair, because the tether is what you trade and the day-to-day noise is just opportunity. Most tradeable pairs are **both**. And most random pairs of stocks are **neither**. The single most important row is the second one: cointegration, not correlation, is the property that makes a spread tradeable, and you can have it without high correlation.

### The error-correction model: the engine of reversion

Cointegration comes with a profound bonus result, the **Granger representation theorem**, which says: *if two series are cointegrated, then there must exist an error-correction mechanism that pulls them back together — and vice versa.* The two ideas are mathematically inseparable. The model that writes this down is the **error-correction model**, or ECM, and it is the actual engine of mean reversion. For one of the series it reads:

$$
\Delta y_t = \gamma\, z_{t-1} + (\text{short-run terms}) + \epsilon_t,
$$

where $\Delta y_t$ is today's change in $y$, $z_{t-1} = y_{t-1} - \beta x_{t-1}$ is yesterday's spread (how far the leash was stretched), and $\gamma$ (gamma) is the **speed of adjustment** — the heart of the whole thing.

Read the equation in plain English: *today's change in $y$ depends on how stretched the spread was yesterday.* If $\gamma$ is negative, then a positive spread yesterday (the leash stretched one way) creates a negative change in $y$ today (a pull back). The size of $\gamma$ measures how *hard* the leash pulls. A $\gamma$ of $-0.5$ means the spread closes half of its gap each period — a fast, snappy reversion. A $\gamma$ of $-0.02$ means it claws back only 2% per period — a slow, patient pull. As we will see, $\gamma$ is exactly what determines the **half-life** of the trade: how long you will sit in a position before the spread comes home. The error-correction model is not abstract bookkeeping; it is the literal mathematical promise that your spread will revert, and a measurement of how fast.

There is a second, subtler payload in the full error-correction model that is worth naming, because it is what makes the framework so much more honest than a plain correlation. A complete ECM has *two* equations, one for each series — both $\Delta y_t$ and $\Delta x_t$ get their own line, and each can load on the lagged spread $z_{t-1}$ with its own speed. The relative sizes of those two speeds tell you *which* leg does the adjusting. If $y$'s speed is large and $x$'s is near zero, then when the spread stretches it is mostly $y$ that moves back toward $x$ — economically, $x$ is the "leader" and $y$ the "follower." That distinction matters in practice: the leg that adjusts is the one whose mispricing you are really betting on, and knowing which is which helps you reason about why the pair is cointegrated in the first place (often the more liquid or more fundamental name leads, and the thinner name catches up). You do not need to act on this to trade the spread, but it is the difference between treating the pair as a black box and understanding the machine.

## The Engle-Granger two-step test

We now have the definition; we need a procedure to *test* a real pair for it. The classic recipe, which won Clive Granger the 2003 Nobel Prize in economics (shared with Robert Engle), is the **Engle-Granger two-step test**. It is exactly as simple as its name.

![Stack of the Engle-Granger steps from regressing one series on the other to testing the residual for a unit root](/imgs/blogs/cointegration-pairs-trading-math-for-quants-4.png)

The stack above shows the two steps in order, and the whole test fits in two sentences. **Step one:** run a plain ordinary-least-squares regression of one price on the other, $y_t = \alpha + \beta x_t + z_t$, and keep the residuals $z_t = y_t - \alpha - \beta x_t$ — that is your candidate spread. **Step two:** test those residuals for a unit root using the **Augmented Dickey-Fuller (ADF) test**. If the ADF test *rejects* the unit root — concludes the residual is stationary — then the residual has a home, the spread mean-reverts, and the two prices are cointegrated. If it cannot reject, you have two wanderers and no tether, and you walk away.

### What the ADF test is actually doing

The ADF test deserves a plain-English unpacking, because "test for a unit root" sounds like a black box. The idea is to take the spread $z_t$ and ask: does it pull itself back toward its mean? You regress the *change* in the spread on its *previous level*:

$$
\Delta z_t = \rho\, z_{t-1} + (\text{lag terms}) + u_t.
$$

Here $\rho$ (rho) is the key number. If $\rho$ is significantly negative, then whenever the spread is above its mean (positive $z_{t-1}$), the next change $\Delta z_t$ tends to be negative — it gets pulled down — and vice versa. That is mean reversion, and it means stationary, cointegrated. If $\rho$ is statistically indistinguishable from zero, then the spread's level today tells you nothing about whether it will rise or fall next — it wanders, unit root, not cointegrated. The ADF test wraps this in a statistical procedure that accounts for noise and outputs a test statistic you compare against critical values. The more negative the statistic, the stronger the evidence of a real tether.

One subtlety that trips people up: because the spread in step two is itself *estimated* from a regression (not observed directly), the ordinary ADF critical values are too lenient. You must use special critical values (the Engle-Granger / MacKinnon tables) that are more demanding. Skip this and you will declare cointegration that is not there — a quiet but expensive bug.

#### Worked example: estimate the hedge ratio and test the spread

Let us do the real test on a concrete pair. Suppose we are looking at two refiner stocks — call them Refiner A and Refiner B — that buy the same crude oil and sell the same gasoline, so they share an economic driver. Over the past year we have daily closing prices. Step one: regress A on B.

The regression $A_t = \alpha + \beta B_t + z_t$ comes back with $\alpha = \$3.20$ and $\hat{\beta} = 1.45$. That hedge ratio of \$1.45\$ is the practical payload: it says one share of A behaves, in the long run, like \$1.45\$ shares of B plus a constant. So to build a market-neutral spread we go long 1 share of A and short 1.45 shares of B (or scale both to equal dollar amounts; we will get to sizing). The spread on any day is

$$
z_t = A_t - 3.20 - 1.45 \times B_t.
$$

Plug in a day where $A = \$58.00$ and $B = \$37.00$: $z_t = 58.00 - 3.20 - 1.45 \times 37.00 = 58.00 - 3.20 - 53.65 = \$1.15$. So the spread sits \$1.15 above its line that day.

Step two: feed the whole year of $z_t$ values into the ADF test. Say it returns a test statistic of $-3.9$. The Engle-Granger 5% critical value for a two-variable system is about $-3.34$. Because $-3.9$ is *more negative* than $-3.34$, we reject the unit root: **the spread is stationary, the pair is cointegrated, and we have a tradeable relationship.** Compute the spread's mean (say \$0.00 by construction after centering) and its standard deviation (say \$0.60). The one-sentence lesson: **the hedge ratio $\beta$ is the recipe for the spread, and the ADF test on that spread — not the correlation of the prices — is the only thing that tells you whether the pair is real.**

### When two series isn't enough: the Johansen test

Engle-Granger has two real limitations. First, it is asymmetric: regressing A on B can give a slightly different hedge ratio than regressing B on A, and you have to pick one. Second, and more important, it only handles *two* series at a time. But sometimes a tradeable relationship lives in a *basket* — three, four, five instruments whose weighted combination is stationary even though no single pair is. Think of three Treasury bonds of different maturities, or a stock versus a basket of its sector peers.

For these the standard tool is the **Johansen test**, developed by Søren Johansen in the late 1980s. Where Engle-Granger does two regressions, Johansen works inside the full vector error-correction model and uses an eigenvalue computation to answer two richer questions at once: *how many* independent cointegrating relationships exist among the $n$ series (the "rank"), and *what are* the weight vectors for each one. With three series there might be zero tethers (all independent wanderers), one tether (one stationary combination), or two. Johansen's two test statistics — the **trace statistic** and the **maximum-eigenvalue statistic** — each estimate the rank by stepping through hypotheses ("at most zero relationships," "at most one," and so on) and comparing against critical values.

You do not need to compute Johansen's eigenvalues by hand — every stats package does it. What you need to carry away is *when to reach for it*: any time you suspect the stationary relationship involves more than two instruments, Engle-Granger's pairwise view will miss it, and Johansen is the symmetric, multi-series test that finds the basket and its weights in one shot. For the rest of this post we stay in the clean two-asset world, because that is where the intuition and the dollars are clearest.

It helps to see the two tests side by side, because the choice between them is one a working quant makes constantly.

| Question | Engle-Granger | Johansen |
| --- | --- | --- |
| How many series? | Exactly two | Any number ($n \ge 2$) |
| Symmetric in the inputs? | No — pick which to regress | Yes — treats all series alike |
| How many tethers can it find? | At most one | Counts them (the "rank") |
| What it gives you | One hedge ratio | All tether weights at once |
| Difficulty | A regression plus one test | An eigenvalue computation |
| Best for | A clean two-stock pair | A basket: yield curve, sector group |

The honest summary: for a single pair, Engle-Granger is simpler, more transparent, and perfectly adequate — you can do it in a spreadsheet and *see* the spread. The moment your hypothesis grows to three or more legs, switch to Johansen, because the pairwise approach can completely miss a relationship that only appears in the right multi-asset combination.

## The trading rule: z-scores and bands

We have a stationary spread with a known mean and standard deviation. Now we turn it into trades. The conversion is the **z-score**, the most useful single number in mean-reversion trading. The z-score restates "how stretched is the spread right now" in units of standard deviations:

$$
z\text{-score}_t = \frac{z_t - \mu}{\sigma},
$$

where $z_t$ is today's spread value, $\mu$ is the spread's long-run mean, and $\sigma$ is its standard deviation. A z-score of $0$ means the spread is exactly at its home. A z-score of $+2$ means it is stretched two standard deviations above home — unusually far, and (for a stationary series) due to snap back. A z-score of $-2$ means stretched two standard deviations below. The z-score is dimensionless, so the same thresholds work for a \$0.60-wide spread or a \$6.00-wide spread.

The classic rule is almost insultingly simple:

- When the z-score rises to $+2$: the spread is too high, so **short the spread** (sell $y$, buy $\beta$ units of $x$), betting it falls back.
- When the z-score falls to $-2$: the spread is too low, so **long the spread** (buy $y$, sell $\beta$ units of $x$), betting it rises back.
- When the z-score returns to $0$: **close** the position and bank the move.
- Often a **stop** at $\pm 3$ or so: if the spread keeps stretching past 3 sigma instead of reverting, that is a warning the tether may have broken, and you cut the trade.

![Timeline of one round trip where the spread hits the plus-two band, the trade opens, the spread reverts to zero, and the trade closes for a profit](/imgs/blogs/cointegration-pairs-trading-math-for-quants-5.png)

The timeline above walks one complete round trip, and it is the rhythm of every pairs trade. The spread starts near its mean. Over a week or two it drifts up and finally pierces the $+2$ band — that is the entry trigger, and you put on the short-the-spread position. Then, because the spread is stationary, the error-correction pull does its work and the spread grinds back toward zero over the next couple of weeks. When it crosses zero, you close, and the distance the spread travelled — from $+2\sigma$ back to $0$ — is your gross profit, scaled by your position size. The art is in choosing the bands ($\pm 2$ is conventional but not sacred) and in knowing how long the round trip should take, which is the half-life we cover next.

### Why the bands are a real trade-off, not a convention

It is tempting to think tighter bands are strictly better — enter at $\pm 1$ instead of $\pm 2$ and you trade more often, so you make more money. That reasoning is wrong, and seeing why sharpens the whole strategy. Three forces fight each other. First, a wider band means each trade captures a *bigger* move (a 2-sigma reversion banks twice the dollars of a 1-sigma reversion) and a *cleaner* signal (a 2-sigma stretch is more likely to be a genuine dislocation than random noise). Second, a wider band means *fewer* trades, because the spread reaches $\pm 2$ far less often than $\pm 1$, so you sit on the sidelines more. Third — and this is the one beginners miss — every round trip pays the *same* fixed transaction cost regardless of how far the spread moved, so tiny trades at tight bands hand a larger fraction of their gross to the broker.

Put concretely: if a round trip costs \$1,000 in fees and slippage, a 2-sigma trade worth \$30,000 gross keeps 97% of it, while a 1-sigma trade worth \$15,000 gross keeps 93%, and a 0.5-sigma trade worth \$7,500 gross keeps only 87% — and the 0.5-sigma "signal" is half noise, so a chunk of those trades never revert at all. There is an interior optimum: wide enough that each trade clears costs with room to spare and the signal is real, narrow enough that you actually get trades. Where that optimum sits depends on the spread's volatility, its half-life, and your costs — which is exactly why a serious desk *solves* for the band rather than defaulting to $\pm 2$ out of habit. The conventional $\pm 2$ is a reasonable starting point, not a law.

#### Worked example: the dollar P&L of one round trip on a \$1,000,000 pair

Now the money. We trade the cointegrated Refiner A / Refiner B pair from before. The spread has mean \$0.00 and standard deviation $\sigma = \$0.60$. We run a **dollar-neutral** book of \$1,000,000: \$500,000 long one leg and \$500,000 short the other, so the net market exposure is roughly zero and we profit only from the spread, not from the market going up or down.

One day the z-score hits $+2$. That means the spread is $2 \times \$0.60 = \$1.20$ above its mean — leg A is expensive relative to leg B. We put on the trade: short \$500,000 of A and long \$500,000 of B (weighted by the hedge ratio so it is market-neutral). With A at \$58, \$500,000 buys us a short of about $500{,}000 / 58 \approx 8{,}620$ shares of A; with B at \$37, the long side is about $500{,}000 / 37 \approx 13{,}510$ shares of B, trimmed by the hedge ratio so the dollar betas match.

We wait. Three weeks later the spread reverts all the way to its mean — the z-score is back to $0$, meaning the spread moved \$1.20 in our favor. How much money is that? The spread is the per-unit gap, and we have roughly \$500,000 of exposure per \$1 of spread movement *relative to the spread's own scale*. The clean way to see it: a 2-sigma round trip captures $2\sigma$ of spread, and with a \$1,000,000 dollar-neutral book where the spread's standard deviation is about 1.5% of the position value, a 2-sigma move is about $2 \times 1.5\% \times \$1{,}000{,}000 = \$30{,}000$ gross. Subtract trading costs — say a round-trip cost of 5 basis points on \$1,000,000 each side, roughly \$1,000 total — and the net is about **\$29,000** on one round trip.

That is the engine. The one-sentence lesson: **each round trip from a 2-sigma stretch back to the mean converts the spread's standard deviation into dollars, and on a \$1,000,000 dollar-neutral pair that is roughly \$30,000 of gross edge per clean reversion — provided the spread actually reverts.** That last clause is everything, and it is what the half-life and the risk sections below are about.

## The half-life of mean reversion

A 2-sigma round trip is worth \$30,000 — but is that \$30,000 earned in three days or three months? That changes everything: a fast pair you can trade dozens of times a year, compounding the edge; a slow pair ties up capital for a quarter per trade and earns far less annualized. The number that answers "how long do I wait" is the **half-life of mean reversion**, and it falls straight out of the error-correction speed $\gamma$.

The cleanest way to estimate it is to model the spread as an **Ornstein-Uhlenbeck process** — a continuous-time leash — or, discretely, to regress the change in the spread on its lagged level:

$$
\Delta z_t = \gamma\, z_{t-1} + u_t.
$$

This is the same error-correction equation from before. The coefficient $\gamma$ is negative for a mean-reverting spread, and the half-life — the time for the spread to close *half* of any given gap to its mean — is

$$
\text{half-life} = \frac{\ln(2)}{-\gamma} = \frac{0.693}{-\gamma}.
$$

The everyday picture is a cup of hot coffee cooling toward room temperature. It does not snap to room temperature instantly; it closes half the gap in some fixed time (say 15 minutes), then half of *what's left* in another 15, and so on. That fixed "close half the gap" interval is the half-life. A spread with a short half-life is like coffee in a thin paper cup — it reverts fast. A long half-life is coffee in a thick thermos — it takes forever to come home.

#### Worked example: half-life, holding period, and sizing to a risk budget

We estimate the speed of adjustment on our refiner spread and get $\gamma = -0.10$. The half-life is then

$$
\text{half-life} = \frac{0.693}{0.10} = 6.93 \approx 7 \text{ days}.
$$

So the spread closes half of any gap in about 7 trading days. A round trip from $2\sigma$ back to roughly the mean is a little over two half-lives — call it **15 trading days**, about three calendar weeks. That matches the timeline figure above, and it tells us we can expect to turn this pair over maybe 12 to 15 times a year (allowing for gaps where the spread does not stretch). At \$29,000 net per round trip and, say, 12 clean trips, that is on the order of **\$350,000 a year** of gross edge from a single \$1,000,000 pair — before we account for the trips that fail, which we must.

Now sizing to a **risk budget**. Suppose our risk limit says: do not let any single pair lose more than \$25,000 on a bad trade. A bad trade is one where, instead of reverting from $+2\sigma$, the spread keeps stretching to our stop at $+3\sigma$ before we exit — a move of $1\sigma$ against us. With a \$1,000,000 position, $1\sigma$ of spread is about $1.5\% \times \$1{,}000{,}000 = \$15{,}000$ of loss, comfortably under the \$25,000 limit, so \$1,000,000 is a safe notional. If we wanted to lose at most \$15,000 at the stop, we would hold the size at \$1,000,000; if our risk limit were tighter at \$10,000, we would scale the position down to about \$667,000 so that the $1\sigma$ adverse move costs exactly \$10,000. The half-life also caps how much capital we *should* commit: a 7-day half-life means capital is recycled fast, so a given dollar of risk budget supports more annual trades than a sluggish 60-day pair would.

The one-sentence lesson: **the half-life converts the abstract speed of adjustment into a concrete holding period — about 7 days here — which sets both how many times a year you can earn the \$30,000 edge and how a fixed-dollar risk budget translates into position size.**

## How the whole strategy fits together

We have built every piece. Let us see the architecture as one map before we turn to the risks, because the risks all attack specific joints in this structure.

![Tree mapping statistical arbitrage down through stationarity and cointegration to the spread, z-score, and error-correction speed](/imgs/blogs/cointegration-pairs-trading-math-for-quants-6.png)

The tree above is the concept map of everything we have covered. At the root is statistical arbitrage — the broad family of strategies that bet on statistical relationships rather than fundamental views. It rests on two pillars. The first is **stationarity**: the whole game requires something that mean-reverts, and that something is identified through unit-root tests and the I(1)-versus-I(0) distinction. The second is **cointegration**, which is what produces a stationary thing — the hedged spread — out of two non-stationary prices, and which comes with the error-correction speed $\gamma$ that times the trades. The spread, the z-score rule, and the half-life are all downstream of these two pillars. If you understand stationarity and cointegration, every other box is just an application of them.

It is worth pausing on what makes this strategy structurally attractive, because it explains why stat-arb desks exist at major firms. A pairs trade is **market-neutral**: because you are long one leg and short the other in matched dollar amounts, a broad market crash that drags both legs down hurts you very little — the spread between them is what you own, and the spread does not care which way the whole market went. That is a genuinely different return stream from "buy stocks and hope," which is exactly why it diversifies a portfolio and why the edge survived for years. It is also why, when it fails, it tends to fail in a specific and brutal way, which is our final and most important topic. For more on how a signal like the z-score becomes a full strategy, see [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research).

## Why backtested pairs decay: the risks

Everything above assumes the cointegration is real and *stays* real. The single hardest truth in this field is that cointegration is not a law of nature — it is a statistical property of a particular sample, and it can vanish. When it does, every assumption above inverts: the spread that "always reverts" keeps widening, the z-score that "snaps back from $\pm 2$" sails through $3$, $4$, $5$, and the market-neutral trade that "can't lose to a crash" loses steadily and quietly. This is how pairs traders blow up, and it deserves real attention.

![Before and after panels contrasting a stable reverting pair with a broken pair whose mean shifts and spread drifts away](/imgs/blogs/cointegration-pairs-trading-math-for-quants-7.png)

The figure above is the difference between a good night's sleep and a margin call. On the left is a stable pair: the two companies still share the same business risk, the spread oscillates, the mean stays put, and the trades pay off. On the right is a **structural break**: a merger, a bankruptcy, a regulatory change, a new product that hits one company and not the other — and suddenly the economic leash that tied the two together is cut. The spread's mean shifts to a new level, or worse, the spread stops being stationary at all and just walks away. A trader who put on the position at $+2\sigma$ expecting reversion now watches the spread go to $+5\sigma$ and beyond, because there is no longer any force pulling it back. The position that was sized to lose at most \$15,000 at a $3\sigma$ stop instead bleeds through the stop in a market where the spread gaps overnight.

### The three ways a pair dies

There are three distinct failure modes, and a serious pairs trader monitors for all three.

**Structural change.** The cleanest break. Two airlines were cointegrated for a decade; then one announces it is acquiring the other. Overnight the relationship is no longer two independent-but-tethered companies — the target's price snaps to the deal price and the historical spread is meaningless. Mergers, spinoffs, major lawsuits, debt restructurings, and management blowups all sever the economic link that *caused* the cointegration. No amount of statistics on past data protects you, because the future is generated by a different process than the past. The defensive move is fundamental: know *why* each pair is cointegrated (same inputs, same customers, same regulatory regime) so you can spot when that reason disappears.

**Parameter drift.** The subtler killer. Even without a dramatic break, the hedge ratio $\beta$ and the spread's mean and standard deviation slowly wander as the two businesses evolve. A $\beta$ estimated on last year's data is stale this year; trading on a stale $\beta$ means your "market-neutral" spread is quietly accumulating market exposure, and your "mean" is the wrong target. This is why practitioners re-estimate the hedge ratio on a rolling window, or use a [Kalman filter](/blog/trading/math-for-quants/regression-ols-gls-regularized-math-for-quants) to let $\beta$ update dynamically rather than freezing it.

**Overfitting and crowding.** The statistical killer. If you scan thousands of stock pairs looking for the ones with the strongest historical cointegration, some will look cointegrated *purely by luck* — the spurious-regression trap returns, now wearing the disguise of a stationarity test. Test 5,000 pairs at the 5% significance level and you expect 250 false positives even if no pair is genuinely cointegrated. The pairs that backtest best are disproportionately the lucky ones, and luck does not repeat out of sample. On top of this, when a pair is *genuinely* good, other desks find it too; the more capital chases the same spread, the smaller and faster the reversion becomes, until the edge is competed away. This is the deep reason a beautifully backtested pair decays the moment real money touches it. The discipline that fights this — out-of-sample testing, multiple-testing correction, realistic costs — is the subject of [backtesting done right](/blog/trading/quantitative-finance/backtesting-done-right-quant-research).

#### Worked example: the dollar cost of a broken pair

Numbers make the danger concrete. We are short our refiner spread at $+2\sigma$ on a \$1,000,000 book, expecting \$30,000 of gross profit as it reverts. Instead, Refiner B announces it is being acquired at a 30% premium. B's price jumps, the spread — which is A minus 1.45 times B — lurches violently *against* us because we are long B... no, recall we are short the spread, meaning short A and long B; the takeover *helps* the B leg but the relationship is now broken and the spread no longer tracks anything.

Take the cleaner case where the break goes against us: the spread, instead of reverting from $+1.20$ (its 2-sigma level), keeps widening to $+3.60$ — a 6-sigma move — before we can react to a gap. That is a $4\sigma$ adverse move, and at roughly \$15,000 per sigma on a \$1,000,000 book, the loss is about $4 \times \$15{,}000 = \$60{,}000$ — double what we hoped to *make*, and four times our intended \$15,000 stop, because the spread gapped past the stop overnight when the news hit. A single broken pair can erase the profit from a dozen good round trips. The one-sentence lesson: **the math of cointegration tells you the edge when the tether holds, but the moment a structural break cuts the tether, the same position that was engineered to be safe becomes an unhedged directional bet, and the loss can dwarf any single trip's gain — which is why position limits, stops, and knowing *why* a pair is cointegrated matter more than the elegance of the test.**

## Common misconceptions

**"High correlation means a tradeable pair."** This is the original sin of pairs trading. Correlation is about short-run co-movement and is computed, fatally, on non-stationary prices where it is largely meaningless. Two unrelated random walks can show 0.9 correlation by pure chance. The property you need is cointegration — a stationary spread — and you can have strong cointegration with modest correlation and strong correlation with no cointegration at all. Always test the spread, never trust the price correlation.

**"A high $R^2$ in the price regression confirms the relationship."** No. With I(1) inputs, $R^2$ and t-statistics are spurious — they grow large for unrelated series and grow *larger* with more data, the opposite of honest statistics. The only valid check is a unit-root test (ADF) on the regression *residuals*, using the stricter Engle-Granger critical values, not the textbook ones.

**"Cointegration is permanent once established."** It is a property of a sample, not a law. Mergers, regulatory shifts, and slow business drift sever the economic link that caused the tether. A pair cointegrated for ten years can break in a day, and the break is exactly when your position is largest and most confident. Monitoring *why* the pair is cointegrated is as important as the statistics.

**"The trade is market-neutral, so it's low risk."** Market-neutral means insulated from broad market moves, not insulated from *spread* moves. Your entire risk is the spread, and when cointegration breaks the spread becomes an unbounded directional exposure. The 2007 "quant quake," when many stat-arb books that looked perfectly hedged lost double digits in days, is the canonical reminder that market-neutral is not risk-neutral.

**"Tighter entry bands always make more money."** Entering at $\pm 1$ instead of $\pm 2$ gives more trades but smaller, noisier moves with worse signal-to-noise, and each trade still pays the full round-trip cost. There is an optimal band that depends on the half-life, the spread's volatility, and transaction costs; tighter is not free, and below some threshold the costs eat the edge entirely.

**"You can find pairs by data-mining thousands of candidates."** Scanning thousands of pairs guarantees false positives — at the 5% level, 5,000 truly-unrelated pairs throw off about 250 that "pass" the cointegration test by luck. The best-backtesting pairs are disproportionately the lucky ones. Genuine pairs come from an economic hypothesis first (same sector, same inputs, same customers), with the statistical test as confirmation, not as a search engine.

## How it shows up in real markets

**The original Morgan Stanley stat-arb desk (mid-1980s).** Nunzio Tartaglia's group at Morgan Stanley is widely credited with industrializing pairs trading. They automated the scan for divergent pairs and the mechanical entry/exit, running hundreds of pairs simultaneously so that the law of large numbers smoothed out individual failures. The desk reportedly made tens of millions a year at its peak. The lesson that built the field: a single pair is a coin flip, but a diversified book of dozens of genuinely cointegrated pairs, each with positive expected reversion, is a business.

**Royal Dutch and Shell — the textbook pair.** Before their 2005 unification, Royal Dutch Petroleum and Shell Transport were two listings of effectively one company, with cash flows split in a fixed 60/40 ratio. By any economic logic their prices should have stayed in that fixed ratio — perfect cointegration with a known hedge ratio. Yet the spread between them wandered as much as 10-15% away from parity for *months* at a time, driven by index inclusion, tax treatment, and home-country buying. It was one of the cleanest real cointegrated pairs ever, and also a humbling lesson: even a near-mechanical tether can stretch far and stay stretched long enough to wipe out an undercapitalized trader before it reverts. The reversion was certain; the *timing* was not.

**The 2007 quant quake.** In early August 2007, a sudden deleveraging by one or more large quant funds forced rapid liquidation of stat-arb positions. Because so many funds held similar cointegration-based pairs, the forced selling pushed spreads *further* apart instead of letting them revert — every model said "the spread is 4 sigma stretched, buy it," and every fund's buying-into-the-divergence accelerated the divergence. Funds that were textbook market-neutral lost 10-30% in a few days, then most of it reverted the following week. The mechanism from this post in action: crowding turned a self-correcting spread into a self-reinforcing one, temporarily breaking the mean reversion the whole strategy depends on.

**Index-arbitrage and ETF-versus-basket.** A liquid, ongoing form of cointegration trading is keeping an ETF in line with the basket of stocks it holds. The ETF price and the net asset value of its holdings are cointegrated by *construction* — an authorized participant can create or redeem shares to capture any gap — so the spread is tightly stationary with a tiny half-life of minutes. Desks trade this all day for fractions of a percent, and it is the rare case where the cointegration is enforced by an explicit arbitrage mechanism rather than mere economic similarity, making it far more reliable than a stock pair.

**Treasury on-the-run versus off-the-run.** The most recently issued ("on-the-run") Treasury of a given maturity trades at a slightly richer price than nearly identical slightly-older ("off-the-run") issues, because of a liquidity premium. The spread is cointegrated and mean-reverting most of the time, and relative-value desks trade it. Long-Term Capital Management famously held enormous leveraged positions in exactly these convergence trades; in 1998 a flight to liquidity widened the spreads instead of converging them, and the leverage that magnified the tiny normal edge magnified the abnormal loss into a near-collapse of the firm. The lesson: cointegration spreads are small, so they get traded with leverage, and leverage turns a temporary break into a terminal one.

**Crypto and stablecoin pairs.** Modern crypto markets are full of cointegration trades: a stablecoin meant to track \$1.00 is cointegrated with \$1.00 by design, and the spread (the "peg deviation") is a classic mean-reversion trade — until the peg breaks, as TerraUSD's did in May 2022, when the spread went from a few basis points to a total loss within days. The pattern is identical to the equity case: a tight, profitable, reliable-looking spread that mean-reverts for a long time and then, on a structural break, does not.

## When this matters to you and further reading

If you ever build a trading strategy, this is the first place naive intuition will cost you money: you will look at two charts that move together, compute a correlation, and feel certain. Cointegration is the discipline that turns that feeling into a testable, falsifiable claim — and just as often tells you the relationship you were sure about is a mirage. Even outside trading, the spurious-regression trap shows up everywhere two trending series are compared: "ice cream sales correlate with drownings," "the more X grew, the more Y grew." Whenever you see a correlation between two things that both grow over time, the right reflex now is: *are these I(1), and is anything about their combination actually stationary?*

For the trader specifically, the practical takeaway is a sequence: hypothesize a pair from economics, regress to get the hedge ratio, test the *spread* (not the prices) for stationarity with the right critical values, estimate the half-life to set your holding period and sizing, trade the z-score with bands and a stop, and re-estimate constantly while watching for the structural break that ends every pair eventually. The edge is real — roughly \$30,000 per clean round trip on a \$1,000,000 pair in our example — but it is rented, not owned, and the rent comes due when the cointegration breaks.

To go deeper into the pieces: the [covariance and correlation pitfalls](/blog/trading/quantitative-finance/covariance-correlation-pitfalls-quant-interviews) post drills the spurious-correlation trap in its general form; the [regression deep-dive from OLS to GLS to regularized](/blog/trading/math-for-quants/regression-ols-gls-regularized-math-for-quants) post is where the hedge-ratio estimation and its instability live in full detail; [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) shows how a z-score becomes a complete, risk-managed strategy; and [backtesting done right](/blog/trading/quantitative-finance/backtesting-done-right-quant-research) is the antidote to the overfitting-and-crowding decay that quietly kills every data-mined pair. This is educational material, not investment advice — every spread that can revert for you can also break against you, and the dollar figures here are illustrations, not promises.

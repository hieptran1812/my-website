---
title: "The Kelly criterion and sequential betting: how much to bet"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch guide to bet sizing for quant trader interviews: why long-run growth lives in the geometric mean, how the Kelly criterion f* = edge/odds maximizes expected log-wealth, why overbetting ruins you even with an edge, and why desks run fractional Kelly -- with fully worked dollar examples and seven solved interview sizing problems."
tags:
  [
    "kelly-criterion",
    "bet-sizing",
    "position-sizing",
    "quant-interviews",
    "expected-value",
    "geometric-mean",
    "risk-of-ruin",
    "fractional-kelly",
    "quantitative-trading",
    "money-management",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** -- if you have an edge, the question that actually determines whether you get rich is not *whether* to bet but *how much*. The Kelly criterion answers it by maximizing the long-run growth rate of your bankroll.
>
> - Long-run wealth compounds, so what matters is the **geometric** mean of your returns, not the arithmetic mean. A +50% year and a -50% year average to 0% arithmetically but leave you down 25%.
> - The Kelly criterion maximizes **expected log-wealth**. For a simple even-money bet that you win with probability $p$, the optimal fraction of your bankroll to stake is $f^* = p - q = 2p - 1$. For general odds it is $f^* = \text{edge}/\text{odds}$.
> - Betting **too big** destroys you even *with* an edge: the growth rate is a downward parabola in the bet fraction that peaks at $f^*$, falls back to **zero at exactly twice Kelly**, and goes negative beyond. Past two-times Kelly, a positive edge becomes a guaranteed slow bleed to ruin.
> - Practitioners almost never bet full Kelly. **Half-Kelly** keeps about three-quarters of the growth for roughly half the variance and is far more forgiving when your estimate of $p$ is wrong.
> - The one number to remember: with a 60/40 even-money edge and a $\$10{,}000$ bankroll, full Kelly says stake exactly $\$2{,}000$ -- 20% -- per bet.

Here is a question a trading-firm interviewer might open with: *I'll flip a coin that lands heads 60% of the time. Every round you can bet any fraction of your money; if it's heads you win even money, if it's tails you lose your stake. You start with $\$10{,}000$ and we play 100 rounds. How much do you bet each round?*

Almost everyone's first instinct is wrong in one of two directions. The cautious answer -- "I'd bet small, maybe a few percent, to be safe" -- leaves enormous amounts of money on the table. The aggressive answer -- "It's a 60% edge, I'd bet most of it" -- *goes broke with near certainty*, even though the edge is real and large. The right answer is a specific number, $\$2{,}000$ a round, and it comes from a single idea: when wealth compounds, you should maximize its growth *rate*, and the bet size that does this is the Kelly fraction.

![Two traders with the identical 60 percent edge diverge entirely on bet size: betting the Kelly fraction f-star equals 0.20 maximizes long-run compounding, betting too small grows slowly, and betting past twice Kelly drives the bankroll to zero.](/imgs/blogs/kelly-criterion-sequential-betting-quant-interviews-1.png)

The diagram above is the mental model for the whole post. The edge is fixed -- both traders win 60% of the time -- but the *size* of the bet sends them to completely different fates. Bet too little and you grow slowly. Bet the Kelly fraction and you compound as fast as the edge allows. Bet too much and you grind your bankroll to nothing. The edge tells you to bet; the size decides whether you survive.

This is one of the most-loved topics in quant *trader* interviews at market-making and proprietary trading firms -- Jane Street, Optiver, SIG, IMC, Hudson River Trading, Jump, Citadel Securities -- because it sits exactly at the intersection of probability, intuition about risk, and the actual job of sizing positions. Interviewers can ask it as a clean coin-flip puzzle, and they can also push on the second-order ideas (why you'd bet *less* than Kelly, what happens at twice Kelly, how it connects to volatility targeting) to see how deep your understanding really goes.

We will build the whole thing from absolute zero. No finance or probability background is assumed; every term is defined the first time it appears, every claim is grounded in a worked dollar example you can check by hand, and the back half is framed explicitly for the interview room with five fully solved sizing problems. This is educational material about a mathematical idea, not investment advice.

## The building blocks: bankroll, odds, edge, and two kinds of average

Before we can size a bet we need four plain-English ideas nailed down.

Your **bankroll** is simply the total pool of money you are betting with -- the number that goes up when you win and down when you lose. In trading it is your trading capital or the equity in your account. The entire Kelly framework is about choosing what *fraction* of this bankroll to put at risk on each opportunity.

A bet's **odds** describe how much you win relative to how much you stake. We will use **net odds**, written $b$: if you stake $\$1$ and win, you receive your $\$1$ back *plus* $b$ dollars of profit. An **even-money** bet has $b = 1$ -- risk a dollar to make a dollar. A bet that pays **2-to-1** has $b = 2$ -- risk a dollar to make two. A coin flip in a casino, where you either double your stake or lose it, is even money.

The probability you win is $p$, and the probability you lose is $q = 1 - p$. (A *probability* is just a number between 0 and 1 measuring how often something happens in the long run; $p = 0.6$ means it happens 60% of the time.)

Your **edge** is your expected profit per dollar staked -- the average amount you make on a bet, weighted by how often each outcome happens. For a bet with net odds $b$, win probability $p$, and loss probability $q$, the edge is

$$\text{edge} = b \cdot p - q.$$

Here $b \cdot p$ is what you expect to *win* (the payoff $b$ times the chance $p$ of getting it) and $q$ is what you expect to *lose* (you forfeit your $\$1$ stake with probability $q$). For our 60/40 even-money coin, $b = 1$, $p = 0.6$, $q = 0.4$, so the edge is $1 \times 0.6 - 0.4 = 0.2$. You make, on average, **20 cents of profit per dollar staked**. That is a genuine, large edge. The whole question of this post is: given that 20-cent edge, what fraction of your bankroll should you risk?

It's worth slowing down on the word "edge," because it is the single quantity that decides whether you should bet *at all*. The edge is just the **expected value** of the bet per dollar staked, where *expected value* means the probability-weighted average of every possible outcome. If you stake $\$1$ on our coin, you end the round with either $\$2$ (probability 0.6) or $\$0$ (probability 0.4), so your expected ending value is $0.6 \times \$2 + 0.4 \times \$0 = \$1.20$ -- a 20-cent expected profit, matching the edge. When the edge is *negative*, the formula doesn't fail; it correctly returns a negative number, and the right Kelly bet is **zero** -- you don't bet at all, or if you can, you bet the other side. A roulette wheel, for instance, has a negative edge for the player on every bet: an American wheel pays 1-to-1 on red but red comes up only $\frac{18}{38} \approx 47.4\%$ of the time, so the player's edge is $1 \times 0.474 - 0.526 = -0.052$, a *negative* 5.2 cents per dollar. No bet size makes a negative-edge game grow; Kelly returns $f^* < 0$, which means "don't play." This is the first sanity check to run on any sizing problem: *compute the edge; if it isn't positive, the answer is don't bet.*

A quick note on terminology you'll hear in interviews. We've defined edge as expected profit per dollar of *stake*. People sometimes quote edge per dollar of *bankroll* or as an annualized return; the principle is identical, only the denominator changes. And the word "odds" is overloaded -- gamblers say "2-to-1" to mean the net odds $b = 2$ we use here, but the same situation can be quoted as "implied probability $\frac{1}{1+b} = \frac{1}{3}$." Always pin down whether a number is a *probability*, a *payoff ratio*, or a *return* before you plug it into Kelly; mixing them up is the most common arithmetic mistake under interview pressure.

### Why "average return" is a trap: the two means

Here we hit the single most important idea in the entire subject, and the one that interviewers most love to probe. There are two different ways to "average" a set of returns, and for repeated betting they give wildly different answers.

The **arithmetic mean** is the ordinary average you already know: add up the numbers and divide by how many there are. The arithmetic mean of $+50\%$ and $-50\%$ is $\frac{50\% + (-50\%)}{2} = 0\%$.

The **geometric mean** is the average that respects *compounding* -- the fact that next period's gain is applied to the money you have *after* this period's gain or loss. To compute it you turn each percentage return into a growth factor (a $+50\%$ return is a factor of $1.5$; a $-50\%$ return is a factor of $0.5$), multiply the factors together, and take the appropriate root. For two periods, the geometric mean growth factor is $\sqrt{1.5 \times 0.5} = \sqrt{0.75} \approx 0.866$, i.e. a geometric mean *return* of about $-13.4\%$ per period.

Those two numbers -- $0\%$ and $-13.4\%$ -- could not be more different in their meaning, and the difference is the heart of bet sizing.

![A +50% year then a -50% year averages to 0% arithmetically, but the geometric mean return is -13.4% per year and the realized wealth is only $75 of every $100 -- compounding makes the geometric mean, not the arithmetic mean, the number that governs long-run growth.](/imgs/blogs/kelly-criterion-sequential-betting-quant-interviews-2.png)

#### Worked example: the +50% / -50% trap

You start with $\$100$. In year one you earn $+50\%$, so you have $\$100 \times 1.5 = \$150$. In year two you lose $50\%$, so you have $\$150 \times 0.5 = \$75$. You did *not* break even. You ended with $\$75$ -- you lost a quarter of your money -- despite the arithmetic mean return being exactly $0\%$.

The arithmetic mean answers "what is the average of the percentage returns?" The geometric mean answers "what *actually happened to my money* after compounding?" When you bet the same way repeatedly and let winnings ride, **only the geometric mean matters**, because each period multiplies the last. The one-sentence intuition: *a sequence of returns compounds multiplicatively, so your long-run fate is the product of the growth factors, and the geometric mean is the only average that captures a product.*

This is why volatility is so costly to a compounding bankroll. The bigger the swings, the more the geometric mean falls below the arithmetic mean -- a relationship sometimes called *volatility drag*. A bet can have a positive arithmetic average return and still drive your geometric growth *negative* if you size it too large. That single fact is the engine behind everything that follows.

### Expected log-wealth: turning a product into a sum

There is a beautiful mathematical trick for working with products of growth factors: take logarithms. The **logarithm** (we'll use the natural log, written $\ln$) has the magic property that it turns multiplication into addition: $\ln(a \times b) = \ln a + \ln b$. So if your bankroll after $n$ bets is the starting amount times a product of growth factors,

$$W_n = W_0 \cdot G_1 \cdot G_2 \cdots G_n,$$

then taking logs turns it into a *sum*:

$$\ln W_n = \ln W_0 + \ln G_1 + \ln G_2 + \cdots + \ln G_n.$$

Here $W_n$ is your wealth after $n$ bets, $W_0$ is your starting bankroll, and each $G_i$ is the growth factor on bet $i$ (e.g. $1.20$ for a 20% gain, $0.80$ for a 20% loss). Because the log of your final wealth is a *sum* of the per-bet log-growths, the long-run average log-growth per bet -- the thing that determines how fast $\ln W_n$ climbs -- is just the expected value of $\ln G$ on a single bet. **Maximizing your long-run growth rate is exactly the same as maximizing your expected log-growth per bet.** That is the Kelly criterion, stated in one line: choose the bet size that maximizes $\mathbb{E}[\ln G]$, the expected logarithm of your wealth growth factor.

Why is the *average* of the per-bet log-growths the right thing to maximize, rather than, say, the average growth factor itself? Because of a deep result called the **law of large numbers**: when you add up many independent random quantities and divide by how many there are, the result converges to the expected value of a single one. The sum $\ln G_1 + \cdots + \ln G_n$ is exactly such a sum, so after many bets $\frac{1}{n}\ln(W_n / W_0)$ -- the realized average log-growth per bet -- converges almost surely to $\mathbb{E}[\ln G]$. In words: *with near certainty, your long-run compound growth rate equals the expected log-growth of a single bet, whatever luck does along the way.* That "almost surely" is what makes Kelly so powerful -- it's not a claim about averages over many parallel universes, it's a claim about what happens to *your* bankroll in *this* one, given enough bets. A strategy with higher $\mathbb{E}[\ln G]$ will, with probability approaching 1, eventually and permanently overtake any strategy with lower $\mathbb{E}[\ln G]$. That is the precise sense in which the Kelly bettor "wins" in the long run.

There's a subtlety worth naming because interviewers probe it: "long run" can be *very* long. Kelly's dominance is asymptotic, and over short horizons a more aggressive bettor can easily be ahead. If you're going to play only ten rounds and then stop, the analysis changes -- the geometric-growth argument relies on compounding over many periods. Most trading is effectively the long run (thousands of trades a year), which is why the framework fits; but a one-shot bet, or a fund with a hard one-year horizon and redemption risk, is a genuinely different optimization. Knowing *when* Kelly applies is as important as knowing the formula.

This is also the most common interview "gotcha." Kelly does **not** maximize your expected wealth -- that would tell you to bet everything every time, which is insane. It maximizes your expected *log*-wealth, which is the right objective precisely because of the compounding argument above. We'll come back to this distinction in the misconceptions section, because getting it right is what separates a memorized answer from real understanding.

## Deriving the Kelly fraction

Now we can actually compute the optimal bet size. We'll do the clean even-money case in full, then state the general formula.

You bet a fraction $f$ of your bankroll on an even-money bet ($b = 1$) that you win with probability $p$ and lose with probability $q = 1 - p$. After one bet, your bankroll has been multiplied by one of two growth factors:

- If you win (probability $p$): your money becomes $(1 + f)$ times what it was -- you gained $f$ of your bankroll.
- If you lose (probability $q$): your money becomes $(1 - f)$ times what it was -- you lost $f$ of your bankroll.

The expected log-growth per bet, which we call $g(f)$, is therefore the probability-weighted average of the two log growth factors:

$$g(f) = p \ln(1 + f) + q \ln(1 - f).$$

Each symbol: $f$ is the fraction of bankroll we stake (our choice), $p$ and $q$ are the win and loss probabilities, $\ln(1+f)$ is the log-growth if we win, and $\ln(1-f)$ is the log-growth (a negative number) if we lose. To find the $f$ that maximizes this, we use calculus: take the derivative of $g$ with respect to $f$ and set it to zero, because a smooth function is at its peak where its slope is flat.

![Deriving the even-money Kelly fraction: write expected log-wealth as p times ln(1+f) plus q times ln(1-f), differentiate, set the derivative to zero, and the optimum simplifies to f-star equals p minus q, which is 0.20 when p is 0.60.](/imgs/blogs/kelly-criterion-sequential-betting-quant-interviews-3.png)

The derivative is

$$g'(f) = \frac{p}{1 + f} - \frac{q}{1 - f}.$$

Setting $g'(f) = 0$ gives $\frac{p}{1+f} = \frac{q}{1-f}$. Cross-multiplying: $p(1 - f) = q(1 + f)$, so $p - pf = q + qf$, which rearranges to $p - q = pf + qf = f(p + q)$. Since $p + q = 1$, we land on the famous result:

$$\boxed{f^* = p - q = 2p - 1.}$$

The optimal fraction to bet on an even-money wager is simply your win probability minus your loss probability. (The second form, $2p - 1$, follows because $q = 1 - p$.) The asterisk in $f^*$ just denotes "the optimal value."

#### Worked example: the 60/40 coin and a $\$10{,}000$ bankroll

For our coin, $p = 0.6$ and $q = 0.4$, so

$$f^* = 0.6 - 0.4 = 0.2 = 20\%.$$

With a $\$10{,}000$ bankroll, full Kelly says bet exactly $0.2 \times \$10{,}000 = \$2{,}000$ on the first round. Notice the bet is a *fraction* of your bankroll, so the dollar amount changes every round: after a win you have $\$12{,}000$ and your next bet is $0.2 \times \$12{,}000 = \$2{,}400$; after a loss you have $\$8{,}000$ and your next bet is $0.2 \times \$8{,}000 = \$1{,}600$. You always risk the same *percentage*, never the same dollar amount. The one-sentence intuition: *Kelly bets a constant fraction of your current bankroll, so you automatically bet more when you're winning and pull back when you're losing -- the math builds in a survival mechanism.*

### The general formula: f\* = edge / odds

For a bet with general net odds $b$ (not just even money), the same calculus gives the general Kelly formula:

$$f^* = \frac{bp - q}{b} = \frac{\text{edge}}{\text{odds}}.$$

The numerator $bp - q$ is exactly the edge we defined earlier; dividing by the odds $b$ scales it correctly for how much each winning dollar pays. This one formula specializes to every betting situation an interviewer can hand you.

![The general Kelly formula f-star equals edge divided by odds specializes to every bet type: an even-money 60 percent edge gives 20 percent, a 2-to-1 payout at 40 percent gives 10 percent, a 5-to-1 payout at 25 percent gives 10 percent, and the general b-to-1 case gives (bp minus q) over b.](/imgs/blogs/kelly-criterion-sequential-betting-quant-interviews-9.png)

The matrix above runs the formula across several cases. Check the even-money row: $b = 1$, $p = 0.60$, edge $= 1 \times 0.6 - 0.4 = 0.20$, and $f^* = 0.20 / 1 = 20\%$ -- matching what we derived. Now a 2-to-1 payout where you win 40% of the time: $b = 2$, $p = 0.40$, $q = 0.60$, edge $= 2 \times 0.4 - 0.6 = 0.20$, and $f^* = 0.20 / 2 = 10\%$. The edge is the same 20 cents per dollar, but because each win pays double, you only need to stake *half* as much to capture it. Higher odds mean smaller optimal bets, which is deeply counterintuitive until you internalize that the formula is dividing edge by payoff.

This connects directly to ideas covered in our walkthroughs of [expected value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) and the [classic quant probability problem set](/blog/trading/quantitative-finance/classic-quant-probability-problems) -- Kelly is, at its core, an expected-value optimization, just over log-wealth instead of wealth.

### Deriving the general formula (for the curious)

The even-money derivation generalizes cleanly, and walking through it once is worth it because interviewers occasionally ask you to handle non-even odds from scratch. With net odds $b$, a winning bet of fraction $f$ multiplies your bankroll by $(1 + bf)$ -- you keep your money and gain $b$ for each unit staked -- while a loss multiplies it by $(1 - f)$. The expected log-growth is

$$g(f) = p \ln(1 + bf) + q \ln(1 - f).$$

Differentiate and set to zero: $g'(f) = \dfrac{bp}{1 + bf} - \dfrac{q}{1 - f} = 0$. Cross-multiplying gives $bp(1 - f) = q(1 + bf)$, so $bp - bpf = q + bqf$, hence $bp - q = bpf + bqf = bf(p + q) = bf$. Dividing by $b$ yields $f^* = \dfrac{bp - q}{b}$, exactly the edge-over-odds formula. Notice that when $b = 1$ this collapses to $f^* = p - q$, our even-money result. One formula, derived once, covers every discrete bet.

### A glance at continuous Kelly and multiple bets

Real trading rarely looks like discrete win-or-lose coins; positions move continuously and you often hold *many* at once. Two extensions matter.

First, the **continuous-time** version. When returns arrive smoothly rather than as discrete wins and losses, the growth-optimal fraction of capital to allocate is approximately

$$f^* \approx \frac{\mu}{\sigma^2},$$

where $\mu$ is the expected excess return of the strategy (above the risk-free rate) and $\sigma^2$ is its variance. The shape is identical to the discrete case -- it's still edge divided by a measure of "odds," where now the odds are the *variance* of the bet. This is the form that shows up directly in portfolio theory, and it is why a strategy's optimal leverage rises with its expected return and falls with the *square* of its volatility. Double the volatility of a strategy and the growth-optimal position quarters.

Second, **multiple simultaneous bets**. If you can place several independent bets at once, you don't simply size each one at its own Kelly fraction, because together they consume more of your risk budget than any one alone. The multi-bet Kelly solution accounts for the *correlations* between bets: independent edges can each be sized near their individual Kelly because their risks partly cancel, but highly correlated bets must be sized down sharply, since a bad day hits them all together. The practical upshot for a desk is that diversification *increases* the total fraction you can safely deploy -- ten uncorrelated 1%-edge bets support far more total exposure than one 10%-edge bet, because the law of large numbers smooths the combined outcome. This is the mathematical backbone of why funds prize uncorrelated alpha streams; it is explored from the allocation side in our note on [why a portfolio isn't 100% equities](/blog/trading/quantitative-finance/jpm-why-not-100-equities).

## Why over-betting destroys growth

Here is where Kelly stops being a formula and starts being a survival rule. The natural worry about the cautious answer ("bet small") is that you're leaving money on the table -- and you are. But the danger on the *other* side is far worse and far less intuitive: betting too much doesn't just reduce your growth, it can make a *positive*-edge bet lose money in the long run.

To see why, plot the growth rate $g(f)$ as a function of the bet fraction $f$, holding the edge fixed at our 60/40 coin.

![The long-run growth rate g(f) for a 60/40 even-money edge is a downward hump that rises to a maximum of +2.01% per bet at the Kelly fraction f-star = 0.20, falls back to exactly zero growth at twice Kelly (f = 0.40), and turns negative beyond -- so overbetting converts a real edge into a guaranteed bleed.](/imgs/blogs/kelly-criterion-sequential-betting-quant-interviews-4.png)

The curve tells the whole story. At $f = 0$ you bet nothing and grow at $0\%$ -- obviously. As you increase $f$, your growth rate climbs, reaching its maximum at $f^* = 0.20$. At that peak, $g(0.20) = 0.6\ln(1.20) + 0.4\ln(0.80) \approx 0.6(0.1823) + 0.4(-0.2231) \approx 0.0201$, a growth rate of about **2.01% per bet** in log terms. That is the fastest your bankroll can compound given this edge. Then -- and this is the crucial part -- the curve *comes back down*. Past the peak, betting more makes you grow *slower*, not faster.

### The twice-Kelly mirror: betting 2f\* gives zero growth

Where does the growth rate hit zero again? Exactly at **twice the Kelly fraction**. For our coin that is $f = 0.40$.

![Because the growth-rate curve is symmetric about its peak, betting at twice Kelly (f = 0.40) earns g = 0.00% -- the same zero long-run growth as betting nothing at all -- and any larger fraction makes a positive-edge bet shrink the bankroll.](/imgs/blogs/kelly-criterion-sequential-betting-quant-interviews-11.png)

#### Worked example: 2x Kelly turns a 20% edge into zero growth

Let's verify it. At $f = 0.40$, the growth rate is

$$g(0.40) = 0.6\ln(1.40) + 0.4\ln(0.60) \approx 0.6(0.3365) + 0.4(-0.5108) \approx 0.2019 - 0.2043 \approx 0.00.$$

It rounds to zero (it is exactly zero in the idealized symmetric case). Read that again: you have a genuine 20-cent-per-dollar edge, you bet aggressively at 40% of your bankroll each round, and your money does *not grow at all* in the long run. The wins and losses, compounded, exactly cancel. Bet any fraction *larger* than 40% and your growth rate goes negative -- your bankroll trends to zero with probability 1, despite a real edge on every single bet. The one-sentence intuition: *an edge is a budget for risk, and overbetting spends past the budget -- beyond twice Kelly, the volatility you take on costs more growth than the edge produces.*

This is the deepest and most surprising point in the whole subject, and a favorite interview follow-up. A positive expected value per bet is *necessary* but absolutely *not sufficient* for long-run growth. Size matters as much as edge.

### Watching it happen: three wealth paths

Abstract growth rates are one thing; seeing the dollars diverge is another. Here are three strategies on the *same* 60/40 edge over 100 bets, starting from $\$10{,}000$.

![Over 100 bets of a 60/40 edge from a $10,000 start, full Kelly (f=0.20) compounds to roughly $74,000, flat $100 betting crawls to about $12,000, and 2.5x Kelly (f=0.50) decays toward a few hundred dollars -- identical edge, opposite fates, decided entirely by bet size.](/imgs/blogs/kelly-criterion-sequential-betting-quant-interviews-5.png)

- **Full Kelly** ($f = 0.20$): grows at about 2.01% per bet. After 100 bets the *expected* log-wealth is $\ln(10{,}000) + 100 \times 0.0201 \approx 9.21 + 2.01 = 11.22$, i.e. roughly $e^{11.22} \approx \$74{,}000$. The bankroll compounds steeply.
- **Flat betting** ($\$100$ per round, never re-sized): the wins and losses add up roughly linearly. The expected profit per bet is $0.6(\$100) - 0.4(\$100) = \$20$, so over 100 bets you'd expect about $100 \times \$20 = \$2{,}000$ of profit, ending near $\$12{,}000$. Steady but slow, because you never let your edge compound.
- **Over-betting at 2.5x Kelly** ($f = 0.50$): the growth rate here is $0.6\ln(1.5) + 0.4\ln(0.5) \approx -0.034$ per bet -- *negative*. The bankroll decays exponentially, trending toward a few hundred dollars. Same edge as full Kelly. Opposite destination.

The gap between $\$74{,}000$ and $\$330$ is created entirely by the choice of bet size. Nothing about the edge changed.

## Fractional Kelly: why practitioners bet less than the formula says

If full Kelly maximizes growth, why does virtually every professional -- every poker pro, every quant fund -- deliberately bet *less* than full Kelly, often half or a quarter of it? The answer is the most practically important idea in the post, and it comes down to two words: **variance** and **uncertainty**.

### The growth/variance tradeoff is wildly asymmetric

Full Kelly is the growth-maximizing bet, but it is also a *brutally* volatile one. The reason you can shave the bet without losing much growth is that the growth curve is nearly flat at its peak (a parabola's top is rounded), while variance keeps climbing roughly with the *square* of the bet fraction. So pulling back from the peak costs you very little growth but buys you a large reduction in volatility.

![Scaling the bet down to a fraction c of full Kelly: growth g(c) is a rounded hump peaking at c = 1 (full Kelly), while variance rises roughly like c-squared, so half Kelly (c = 0.5) retains about 75% of the growth for only about half the variance -- the practitioner sweet spot.](/imgs/blogs/kelly-criterion-sequential-betting-quant-interviews-6.png)

Let $c$ be the fraction of full Kelly you actually bet ($c = 1$ is full Kelly, $c = 0.5$ is half Kelly). The growth, expressed as a fraction of the maximum, behaves like $c(2 - c)$, and the variance scales like $c^2$. Plug in $c = 0.5$:

- **Growth** $= 0.5 \times (2 - 0.5) = 0.5 \times 1.5 = 0.75$. Half-Kelly keeps **75% of the maximum growth rate**.
- **Variance** $\approx 0.5^2 = 0.25$ to $0.5$ of full Kelly's, depending on how you account for it -- roughly half.

#### Worked example: half-Kelly on the 60/40 coin

Full Kelly was $f^* = 20\%$, betting $\$2{,}000$ on a $\$10{,}000$ bankroll. Half-Kelly is $f = 10\%$, betting $\$1{,}000$. Its growth rate is $g(0.10) = 0.6\ln(1.10) + 0.4\ln(0.90) \approx 0.6(0.0953) + 0.4(-0.1054) \approx 0.0150$, about **1.50% per bet** versus full Kelly's 2.01%. You've given up roughly a quarter of your growth (1.50 vs 2.01 is about 75%). In exchange, your bet -- and therefore the size of your swings -- is *half*. The one-sentence intuition: *the top of the growth curve is flat but the variance curve is steep, so trading a sliver of growth for a big cut in volatility is almost always worth it.*

### Robustness to a wrong p is the real reason

The variance argument is the textbook one, but the reason desks *actually* run fractional Kelly is subtler and more important: **you never know your true edge.** The Kelly formula assumes you know $p$ exactly. In reality you *estimate* it, and your estimate has error. Here's the asymmetry that makes overbetting so dangerous: if you overestimate your edge and bet full Kelly on the inflated number, you can easily end up betting *past* the true twice-Kelly point -- where growth goes negative. Betting half-Kelly gives you a huge safety margin. Even if your real edge is only half what you thought, half-Kelly on your (wrong) estimate is still at most full Kelly on the truth, so you stay on the safe side of the peak.

![Full Kelly maximizes theoretical growth but runs 100% of the volatility, suffers routine 50% drawdowns, and overbets badly if your edge estimate is wrong; half Kelly keeps about 75% of the growth, halves the volatility, cuts drawdowns, and stays robust when p is mis-estimated.](/imgs/blogs/kelly-criterion-sequential-betting-quant-interviews-10.png)

The comparison above is why "half-Kelly" is close to a professional default. You sacrifice a quarter of your theoretical growth to gain robustness against the one thing you can't control -- the gap between your estimated edge and your true edge. In a world where your $p$ is uncertain, half-Kelly is often *higher* expected growth than full Kelly computed on a noisy estimate, because it protects you from the catastrophic overbetting tail.

## Risk of ruin and drawdowns under full Kelly

Even when you know $p$ exactly, full Kelly is a wild ride. We need to be precise about *how* wild, because "Kelly never goes fully bust" is true but deeply misleading.

Because Kelly bets a fraction of your bankroll, you can never lose *everything* on a finite number of bets -- after a loss you still have $(1-f)$ of your money, which is positive. So the literal *probability of ruin* (hitting exactly zero) is zero for fractional betting. But that is cold comfort, because the *drawdowns* -- the peak-to-trough drops in your bankroll along the way -- are enormous.

![The probability of suffering a 50% drawdown over a long horizon rises sharply with the bet fraction: at half Kelly it is about 25%, at full Kelly it is about 50% (a coin flip), and beyond Kelly it climbs toward certainty -- so pushing the bet fraction up trades growth for a steep increase in drawdown risk.](/imgs/blogs/kelly-criterion-sequential-betting-quant-interviews-7.png)

There is a clean theoretical result here that interviewers sometimes know: under **full Kelly**, the probability that your bankroll ever drops to a fraction $x$ of its starting value (before growing without bound) is approximately $x$ itself. So the probability of *ever halving* your bankroll is about $\frac{1}{2}$ -- a coin flip. The probability of ever dropping to a tenth is about $\frac{1}{10}$. These are not rare tail events; a 50% drawdown under full Kelly is something close to a routine experience over a long betting career.

Under **half Kelly**, the math improves dramatically: the probability of a drawdown to fraction $x$ becomes roughly $x^2$ (the exponent moves from 1 toward $1/c$). So the chance of ever halving your bankroll drops from about 50% to about $0.5^2 = 25\%$. Cutting the bet in half cuts the deep-drawdown probability far more than in half. This is the same asymmetry as before, viewed through the lens of pain rather than variance.

It's worth being concrete about why this matters beyond the math, because it's where theory meets human reality. A 50% drawdown is not just a number on a chart -- it is the experience of watching half your capital evaporate while still believing your edge is intact. For an individual it is psychologically brutal; for a fund it is often *terminal*, because investors redeem and risk limits trip well before the strategy gets its chance to recover. Full Kelly is "optimal" only for an agent with infinite patience, perfect knowledge of $p$, and no external constraints -- a creature that does not exist in real markets. The moment you add a finite career, a boss, redeeming investors, or a margin desk that can force you to sell at the worst possible time, the *practically* optimal bet drops below full Kelly. This is the unifying reason behind everything in the fractional-Kelly and drawdown sections: full Kelly maximizes a mathematical quantity, but real bettors optimize a messier objective that penalizes the deep, prolonged drawdowns full Kelly makes routine. Fractional Kelly is the rational answer to *that* objective, and it is why you will essentially never see a serious practitioner running the full formula. Another way to see it: full Kelly puts you exactly at the growth-maximizing edge of the cliff, where the curve is flat in growth but steep in risk; stepping back to half Kelly costs you almost nothing in growth and moves you to far safer ground.

### The fan of outcomes

One more picture makes the volatility visceral. Imagine simulating many independent runs of full-Kelly betting on the same edge and plotting all the bankroll paths together.

![Across many simulated full-Kelly runs from a $10,000 start, the median bankroll compounds steadily upward, but the spread of individual paths is enormous: a lucky run rockets up while an unlucky run halves early and recovers only slowly -- same edge, same strategy, vastly different experienced outcomes.](/imgs/blogs/kelly-criterion-sequential-betting-quant-interviews-8.png)

The median path climbs nicely -- that is the growth-maximization at work. But the *spread* is huge. Some runs rocket upward; others suffer brutal early drawdowns and crawl back over many bets. Every one of those paths has the identical edge and the identical strategy. The difference is pure luck in the order the wins and losses arrive. This dispersion is exactly what fractional Kelly is buying you out of: a narrower, more bearable fan of outcomes, at the cost of a slightly lower median.

## In the interview room

Now let's put it to work the way an interviewer will. Each problem below is solved in full, with the reasoning an interviewer wants to hear out loud. Define your terms, write the formula, plug in the numbers, and -- crucially -- sanity-check the answer.

#### Problem 1: the $\$10{,}000$ 60/40 coin, full solution

*You have $\$10{,}000$. A coin pays even money and lands in your favor 60% of the time. You can bet any fraction each round, repeated many times. How much do you bet, and how fast does your money grow?*

**Solution.** Even money means $b = 1$; $p = 0.6$, $q = 0.4$. The Kelly fraction is $f^* = p - q = 0.6 - 0.4 = 0.20$. On a $\$10{,}000$ bankroll, bet $0.20 \times \$10{,}000 = \$2{,}000$ the first round. The growth rate per bet is

$$g(0.20) = 0.6\ln(1.20) + 0.4\ln(0.80) \approx 0.0201,$$

about 2.01% per bet in log terms. Over 100 bets, expected log-wealth grows by $100 \times 0.0201 = 2.01$, multiplying the bankroll by $e^{2.01} \approx 7.5$ -- so roughly $\$75{,}000$ in expectation. **State the bet ($\$2{,}000$), the fraction (20%), and the growth rate (2% per bet)**; that trio is the complete answer.

#### Problem 2: a 2-to-1 payout with a 40% win rate

*A bet pays 2-to-1 and you win 40% of the time. What fraction do you bet?*

**Solution.** Net odds $b = 2$, $p = 0.40$, $q = 0.60$. First, confirm there's an edge: $\text{edge} = bp - q = 2(0.4) - 0.6 = 0.8 - 0.6 = 0.20$, a positive 20-cent edge per dollar -- worth betting. Now apply the general formula:

$$f^* = \frac{bp - q}{b} = \frac{0.20}{2} = 0.10 = 10\%.$$

On a $\$10{,}000$ bankroll that's a $\$1{,}000$ bet. The instructive point to voice: this bet has the *same* 20-cent edge as the 60/40 coin, yet Kelly says bet *half* as much, because each win pays double. **Higher odds, smaller bet** -- if you can articulate why, you've shown real understanding rather than a memorized formula.

#### Problem 3: flat betting versus Kelly, in dollars

*Compare betting a flat $\$1{,}000$ every round against Kelly betting on the 60/40 coin, over 50 rounds, starting from $\$10{,}000$.*

**Solution.** *Flat betting:* each round wins or loses exactly $\$1{,}000$, with expected profit $0.6(\$1{,}000) - 0.4(\$1{,}000) = \$200$ per round. Over 50 rounds, expected profit is $50 \times \$200 = \$10{,}000$, ending near $\$20{,}000$ -- the profit grows roughly *linearly*. *Kelly betting:* you bet 20% of the *current* bankroll, so winnings compound. Expected log-wealth grows by $50 \times 0.0201 \approx 1.005$, multiplying the bankroll by $e^{1.005} \approx 2.73$, ending near $\$27{,}000$ in expectation. The lesson to state: **flat betting profits add; Kelly betting profits multiply.** Over a long horizon the multiplicative strategy pulls far ahead -- but it also has bigger swings, which is the price of compounding.

#### Problem 4: the half-Kelly tradeoff, quantified

*Your colleague bets full Kelly on the 60/40 coin; you bet half Kelly. Quantify what you give up and what you gain.*

**Solution.** Full Kelly: $f = 0.20$, growth $\approx 2.01\%$ per bet, betting $\$2{,}000$ on $\$10{,}000$. Half Kelly: $f = 0.10$, growth $g(0.10) = 0.6\ln(1.10) + 0.4\ln(0.90) \approx 0.0150$, about $1.50\%$ per bet, betting $\$1{,}000$. *What you give up:* growth falls from 2.01% to 1.50%, about $\frac{1.50}{2.01} \approx 75\%$ of the maximum -- you forfeit a quarter of your growth rate. *What you gain:* your bet is half the size, so your swings are roughly half as large, and the probability of ever halving your bankroll falls from about 50% to about 25%. The closing line: **you pay a quarter of your growth to roughly halve your risk -- a trade most professionals take every time**, especially because it also protects against having over-estimated $p$.

#### Problem 5: "you have $\$10{,}000$ and this repeated bet -- how much per round?"

*A repeated opportunity: 55% chance to win, even money, you can size it freely. You have $\$10{,}000$. How much per round?*

**Solution.** Even money, $b = 1$, $p = 0.55$, $q = 0.45$. Full Kelly is $f^* = p - q = 0.55 - 0.45 = 0.10 = 10\%$, i.e. $\$1{,}000$ on the first round. But here's the senior move: *out loud, recommend less.* A 55% edge is a thin one, and in a real setting your estimate of that 55% is noisy. So you'd say: "Full Kelly is 10%, $\$1{,}000$, but I'd bet half-Kelly -- about 5%, or $\$500$ -- to stay robust to estimation error in my win rate and to keep drawdowns tolerable." That answer demonstrates you know the formula *and* know why practitioners discount it. Compute the growth rate too if asked: $g(0.10) = 0.55\ln(1.10) + 0.45\ln(0.90) \approx 0.55(0.0953) + 0.45(-0.1054) \approx 0.0050$, about 0.5% per bet -- thin edges grow slowly, another reason not to over-leverage them.

#### Problem 6: the negative-edge trap

*A coin lands heads only 45% of the time and pays even money. You can bet any fraction. What's your Kelly bet?*

**Solution.** This is a trick to see if you blindly apply the formula. Even money, $p = 0.45$, $q = 0.55$. The edge is $bp - q = 0.45 - 0.55 = -0.10$ -- *negative*. The formula gives $f^* = p - q = -0.10$, which you should read as: **don't bet** (a negative fraction means you'd want to bet the *other* side if you could). No positive bet size makes a negative-edge game grow; every fraction has negative long-run growth. The correct answer is "$\$0$ -- I wouldn't play this game." If the question allows you to take the *other* side of the coin, then you have a $+10\%$ edge betting tails, and Kelly says stake $10\%$ on tails. The discipline being tested: *always check the sign of the edge before sizing.*

#### Problem 7: re-sizing after your bankroll moves

*You started with $\$10{,}000$ and bet full Kelly (20%) on the 60/40 coin. You win three rounds in a row. How much do you bet on the fourth round, and why isn't it $\$2{,}000$?*

**Solution.** Kelly bets a fixed *fraction* of the *current* bankroll, so the dollar bet floats with your wealth. After round 1 ($\$2{,}000$ bet, won): bankroll $= \$12{,}000$. Round 2 bets $0.2 \times \$12{,}000 = \$2{,}400$; won, bankroll $= \$14{,}400$. Round 3 bets $0.2 \times \$14{,}400 = \$2{,}880$; won, bankroll $= \$17{,}280$. Round 4 bets $0.2 \times \$17{,}280 = \$3{,}456$. It's not $\$2{,}000$ because you re-size off the larger bankroll -- winning makes you bet *more* in dollars (but the same in percentage), and losing makes you bet less. The key sentence: **Kelly is self-scaling; you never re-bet a fixed dollar amount, you re-bet a fixed fraction**, which is precisely what prevents a losing streak from ruining you and lets a winning streak compound.

#### Bonus: the "bet everything?" trap

*If the bet has positive expected value, why not bet 100%?*

**Solution.** Because a single loss wipes you out. On our even-money coin, $f = 1$ means one tail -- which arrives 40% of the time -- takes your entire bankroll to zero, and you can never recover. Kelly maximizes expected *log*-wealth, and $\ln(0) = -\infty$: betting your whole bankroll on a bet you can lose carries an infinite penalty in log terms. The formula's $f^* = 20\%$ is precisely the size that balances growth against the catastrophe of ruin. Saying this clearly is often the whole point of the question. A useful follow-up the interviewer may want: even on a bet you can *never* lose your whole stake on -- say, you can lose at most 50% -- betting your full bankroll is still wrong, because the volatility drag from large swings drops your geometric growth below what a smaller, Kelly-sized bet achieves. Maximizing expected dollars and maximizing compound growth are different objectives, and only the second keeps you in the game.

## Common misconceptions

**"Kelly is aggressive."** Backwards. Kelly is the *growth-maximizing* bet, and it sits well short of all-in -- 20% on a strong 60/40 edge. What feels aggressive is *full* Kelly's volatility, which is why most people run half or quarter Kelly. The formula is better described as the boundary between disciplined growth and reckless overbetting: anything past it is strictly worse.

**"Kelly maximizes my expected wealth."** No -- and this is the single most-tested subtlety. Maximizing expected *wealth* tells you to bet everything every time (more money at stake means higher expected dollars), which guarantees eventual ruin. Kelly maximizes expected *log*-wealth, the right objective for a compounding bankroll. The whole reason logs appear is to convert the product of growth factors into a sum, whose long-run average is what you actually want to maximize.

**"A positive edge means I'll make money long-term, regardless of size."** False, and it's the most dangerous misconception in trading. Past twice the Kelly fraction, a positive-edge bet has *negative* long-run growth -- your bankroll trends to zero with probability 1, even though every individual bet has positive expected value. Edge is necessary but not sufficient; size is co-equal.

**"I should bet full Kelly because it's optimal."** Optimal only if you know $p$ exactly, which you never do. Full Kelly on an over-estimated edge can land you past true twice-Kelly, in negative-growth territory. Fractional Kelly isn't a fudge -- it's the rational response to estimation uncertainty, and on a noisy edge it often *beats* full Kelly in realized growth.

**"Bigger bets always grow faster."** True only up to $f^*$. Beyond the peak, bigger bets grow *slower*, and beyond twice $f^*$ they grow negatively. The growth-versus-size relationship is a hump, not a ramp -- internalizing that shape is most of the battle.

**"Kelly tells me to bet, so this opportunity is good."** Kelly tells you *how much* to bet given an edge; it says nothing about whether your edge estimate is right. Garbage in, garbage out: if your $p$ is wrong, Kelly will faithfully size a losing bet. The hard part of real trading is the edge estimate, not the sizing formula.

## How it shows up on a real trading desk

The coin-flip framing is a teaching device; the idea is load-bearing in real markets. Here's how the same mathematics drives professional position sizing.

![On a trading desk, a Kelly bet is the middle of a pipeline: an alpha signal yields an edge estimate, volatility and win probability are measured, the Kelly fraction is computed, a half-Kelly robustness haircut is applied, a leverage and risk cap clamps the size, and only then does it become the final dollar position.](/imgs/blogs/kelly-criterion-sequential-betting-quant-interviews-12.png)

**Position sizing.** Every systematic trading strategy must answer "how big?" for each trade, and Kelly is the canonical starting point. A strategy with an estimated edge and an estimated volatility implies a Kelly fraction of capital; that fraction, scaled down for safety, sets the dollar position. A desk with $\$10{,}000{,}000$ in capital and a signal whose Kelly fraction is 4% would size a $\$400{,}000$ position at full Kelly -- and likely $\$200{,}000$ at the half-Kelly they actually run.

**Volatility targeting.** Most quant funds don't compute Kelly fractions trade-by-trade; they target a *volatility* level for the whole portfolio -- say, 10% annualized. This is Kelly in disguise. For a continuous return stream, the growth-optimal leverage is approximately $\mu / \sigma^2$ -- the expected excess return divided by the variance -- which is the continuous-time Kelly fraction. Targeting constant volatility is mathematically close to betting a constant fraction of Kelly, which is why the two ideas live side by side on a desk.

**Leverage limits.** The twice-Kelly result is the rigorous reason desks impose hard leverage caps. Beyond a certain leverage, *more* exposure to a profitable strategy *lowers* its compound growth and explodes its drawdowns. Risk managers may not phrase it as "we're staying under twice Kelly," but that is the mathematical content of a leverage limit on a positive-edge book.

**Why funds run fractional Kelly.** Real edges are estimated from noisy, non-stationary data, and an edge that was real last year may have decayed. Funds therefore run a *fraction* of Kelly -- often a quarter to a half -- precisely to survive both the volatility of full Kelly and the very real chance that their estimated edge is too high. The 2008 and 2020 deleveraging episodes are cautionary tales of books that were, in effect, levered past their true Kelly point and forced to cut risk into falling markets. The asymmetric penalty for overbetting -- catastrophic, versus the mild penalty for underbetting -- is why the industry errs systematically toward caution.

**The Thorp connection.** This isn't just theory. The mathematician Ed Thorp, who first applied Kelly to blackjack card-counting and then ran the wildly successful hedge fund Princeton-Newport Partners, is the bridge from the formula to the trading desk. His funds sized positions using exactly these ideas -- a real edge, sized by a fraction of Kelly, with hard risk limits -- and compounded for decades with remarkably few losing periods. That track record is the empirical argument for everything above.

**Blackjack card-counting.** The cleanest real-world Kelly setting is the one where it was first applied. A card-counter in blackjack has a tiny, *fluctuating* edge -- often well under 1%, and only when the remaining deck is rich in high cards. Thorp's insight was to bet proportionally to that edge: a Kelly fraction of a small bankroll when the count is favorable, the table minimum when it isn't. Bet too big relative to the thin edge and a normal run of bad luck busts you out before the edge can express itself; bet Kelly (or, in practice, a fraction of it) and the edge compounds while the swings stay survivable. The lesson transfers directly to trading thin, intermittent alphas: *size proportional to the edge, and remember that a small edge supports only a small bet.*

**Sports and prediction-market bettors.** Professional sports bettors and prediction-market traders live by fractional Kelly because their edges come from probability estimates that are inherently noisy. A bettor who thinks a team should be a 55% favorite when the market prices 50% has an edge -- but if their model is even slightly miscalibrated, full Kelly overbets. The disciplined ones bet a quarter to a half Kelly, precisely because the dominant risk is *being wrong about $p$*, not the variance of a correctly-sized bet. It's the same robustness argument from the fractional-Kelly section, playing out with real money every weekend.

**The LTCM cautionary tale.** Long-Term Capital Management, the famous hedge fund that collapsed in 1998, is in part a story about effective overbetting. The fund ran enormous leverage on convergence trades that each had a real, positive expected edge -- but the leverage pushed the *effective* bet size past the point where the strategy's growth was robust, and the bets were far more correlated than the models assumed. When markets moved against them in a way the historical data hadn't priced, the levered, correlated book suffered drawdowns that a Kelly-disciplined sizing would have made far smaller. The mechanism from this post -- positive edge, but sized past the safe fraction, with hidden correlation collapsing the diversification benefit -- is exactly what turned a profitable strategy into an insolvency.

**2008 and 2020 forced deleveraging.** In both the 2008 financial crisis and the March 2020 COVID shock, levered funds were forced to cut positions into falling markets. Part of the dynamic is pure liquidity, but part is a Kelly story: books that were sized for normal-regime volatility found themselves, when volatility spiked, effectively betting a far larger *fraction* of their risk budget than intended -- because the same dollar position is a bigger bet when the asset is more volatile. The continuous-Kelly formula $f^* \approx \mu/\sigma^2$ makes this precise: when $\sigma$ jumps, the growth-optimal position *shrinks*, so a fund that doesn't cut is suddenly overbet relative to the new environment. Volatility targeting -- continuously re-sizing to hold $\sigma$ constant -- is the industry's institutionalized defense against exactly this, and it is Kelly logic wearing a risk-management hat.

## When this matters and where to go next

If you take one idea from this post into an interview -- or into any decision about sizing a risky, repeated bet -- let it be this: **edge and size are co-equal.** Finding an edge is only half the job; the other half is sizing it so that the volatility you take on doesn't eat the edge you found. The Kelly criterion is the precise statement of that balance, and the reason it shows up in every quant trader interview is that it tests whether you understand *compounding*, *expected value*, and *risk* all at once -- which is, not coincidentally, exactly what the job requires.

For the foundations underneath Kelly, our walkthrough of [expected value techniques for quant interviews](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) builds the expectation machinery from scratch, and the [classic quant probability problem set](/blog/trading/quantitative-finance/classic-quant-probability-problems) drills the probability reasoning you'll lean on. To go to the source: J. L. Kelly Jr.'s original 1956 paper, *A New Interpretation of Information Rate*, is the founding document; Ed Thorp's writings (and his memoir *A Man for All Markets*) show the ideas applied to real money in blackjack and on Wall Street; and William Poundstone's *Fortune's Formula* is the accessible, story-driven history of how a formula from information theory became the secret sizing rule of gamblers and quants alike. Read those in that order and you'll have travelled from the coin flip in the interview room all the way to the trading desk.

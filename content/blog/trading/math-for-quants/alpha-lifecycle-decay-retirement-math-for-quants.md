---
title: "The life of an alpha: crowding, decay, and knowing when to kill it"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "An alpha is not a discovery, it is a depreciating asset. It has a birth, a half-life set by how many other people found it, and a retirement that costs money whether you call it too early or too late. The retirement decision is harder than the discovery, and it is the senior's job."
tags: ["alpha-decay", "crowding", "signal-retirement", "half-life", "information-coefficient", "capacity", "portfolio-construction", "decision-under-uncertainty", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 19
---

> [!important]
> **TL;DR:** An alpha decays because capital arrives and trades the mispricing away. It stops decaying at the cost of running it, not at zero, and the date it crosses that floor is the only retirement date that means anything.
>
> - McLean and Pontiff (2016) measured the one hard anchor we have: across 97 published predictors, portfolio returns were **26% lower out-of-sample** and **58% lower post-publication**, and they attribute the 32-point gap to publication-informed trading.
> - Fitting a half-life to two years of monthly ICs is nearly worthless. On a signal genuinely halving every 30 months, the fitted slope carries **t = -0.739** and **11.5% power**. No exponential decay at all is detectable at the 5% level in a 24-month window.
> - Because capacity scales as the *square* of the alpha, a signal that halves loses **75%** of the capital it can support. Decay hits size before it hits Sharpe.
> - On a \$60m sleeve, retiring a year early costs \$292,600 and a year late costs \$243,200. But if the decay was never real, the early kill forfeits **\$1.68m a year**, which is why keeping is worth **+\$489,728** even at a 66% belief that the alpha is dying.
> - A 0.45-Sharpe strategy at correlation 0.10 is worth **\$1,712k** to the book against **\$431k** for a 0.90-Sharpe strategy at correlation 0.55. Retirement is a decision about the book, never about the strategy alone.

## Introduction

Every research group treats a new signal as a discovery. The correct accounting treatment is a depreciating asset. It is bought with research time, it throws off cash for a while, its value falls at a rate you do not control, and at some point the capital it occupies is worth more somewhere else. That last moment is the retirement date, and almost nobody in the industry can tell you when theirs was.

The reason is not laziness. It is that decay and noise look identical over any window a live strategy actually gives you. The companion post on [telling decay from bad luck](/blog/trading/math-for-quants/live-vs-backtest-divergence-math-for-quants) works through the detection problem for a single strategy that is underperforming right now. This post sits one layer above it: what decay *is*, why it happens mechanically, and how to make the retirement decision when the statistics will not decide it for you.

Start with the shape of the thing. An alpha does not fade to zero. It fades to the point where what is left no longer pays for the cost of capturing it plus whatever else that capital could be doing.

![Decay curve of gross alpha from 6.0% at launch falling toward a horizontal floor at 3.20%, the sum of 1.20% costs and a 2.00% hurdle, with the crossover marked at month 27](/imgs/blogs/alpha-lifecycle-decay-retirement-math-for-quants-1.webp)

## Foundations: four things to have straight

**The information coefficient (IC)** is the cross-sectional correlation between your forecast and the return that follows. For equity signals it lives between 0.02 and 0.06. An IC of 0.04 means your ranking explains about four percent of a standard deviation of next month's cross-sectional return, which sounds like nothing and is a good living.

**Exponential decay** is the working model. Write the strength at time $t$ as ${IC_0 \cdot 2^{-t/H}}$, where $H$ is the **half-life**: the number of months over which the signal loses half its strength. A 30-month half-life means that after two and a half years you have half the edge you started with, after five years a quarter.

**The cost floor.** Running the strategy costs money every year: commissions, spread, market impact, borrow on the short leg, the data subscription, the researcher. Call it 1.20% of the sleeve's capital annually. Below that, gross alpha is not income, it is a subsidy to your brokers.

**The hurdle.** Capital is not free even inside your own firm. If the next best use of a dollar earns 2.00% net, then a strategy earning 2.00% net is worth exactly nothing to keep. Add them and you get a floor of 3.20% of gross alpha, which is where that first figure's horizontal line sits. Everything in this post is about when the curve crosses it and what to do about the fact that you cannot see the crossing.

## Why alphas decay, mechanically

There is nothing mystical here. A signal works because a price is wrong. When more capital trades the same wrongness, the trading itself pushes the price toward fair value. The mispricing you were harvesting is consumed by the harvesting.

![Cyclic diagram showing mispricing attracting capital, capital creating buying pressure, price impact pushing price toward fair value, mispricing shrinking, and the cycle stopping at an equilibrium where net alpha equals the marginal investor's hurdle](/imgs/blogs/alpha-lifecycle-decay-retirement-math-for-quants-2.webp)

That loop has a fixed point, and it is not zero. Capital keeps arriving while the trade clears somebody's hurdle, and stops when it does not. The last dollar in earns the marginal investor's required return net of what it costs them to trade. So the equilibrium level of a public anomaly is roughly the cost of capturing it for the cheapest capable participant. Say that plainly in a meeting: **decay is the market working**, and a signal that never decays is a signal nobody else can find, which should worry you more than it comforts you.

The consequence people miss is about size, not Sharpe. Under the square-root impact law, the cost of building a position $Q$ against daily volume $V$ is about $\sigma\sqrt{Q/V}$. Set that equal to the alpha and the position the trade can support is

$$\frac{Q}{V} = \left(\frac{\alpha}{\sigma}\right)^{2}.$$

With a daily volatility of 1.8% and a 60 bps edge, ${0.0060/0.0180 = 0.3333}$ and ${0.3333^2 = 0.1111}$, so the trade saturates at 11.1% of a day's volume. Halve the edge to 30 bps and ${0.0030/0.0180 = 0.1667}$, ${0.1667^2 = 0.0278}$, a capacity of 2.78%. On a universe trading \$4bn a day that is \$444m of supportable position falling to \$111m. **Capacity is quadratic in alpha: one halving costs you three quarters of your size.** Decay reaches the size of the business long before it reaches the Sharpe ratio on the tear sheet.

## Publication decay: the one number that is measured

Most claims about decay rates in this industry are folklore. One is not.

McLean and Pontiff (2016) took 97 variables that peer-reviewed papers had shown to predict cross-sectional stock returns and compared their in-sample returns to what came afterward. Portfolio returns were **26% lower out-of-sample**, which is the period after the original sample ended but before the paper appeared, and **58% lower post-publication**. They read the 26% as an upper bound on the data-mining component, since it is the part that decays with no one having read anything, and attribute the remaining **32 points** to publication-informed trading. They also document post-publication increases in trading volume and short interest in the affected stocks, which is the crowding mechanism showing up in the data rather than in a story.

Note what that result is and is not. It is a level comparison across sample periods, not an estimated decay rate, and the paper does not report a half-life. Turning 58% into a half-life would be inventing a number the authors did not measure. What it does license is the prior: **the base rate for a published edge is that most of it goes away, and about half of the loss is other people trading it.**

Harvey, Liu and Zhu (2016) come at the same problem from the discovery end. They count at least 316 published factors and argue that with that much collective searching, the conventional ${t \gt 2.0}$ threshold is meaningless. Their multiple-testing model puts the hurdle for a newly discovered factor at ${t \gt 3.0}$. Read the two together and you get the lifecycle in one sentence: a large fraction of alphas were never as strong as their discovery t-statistic claimed, and the rest decay once found.

## What crowding looks like when you try to measure it

You will be asked to monitor crowding. Be honest about the instrument.

What you can actually observe: **13F overlap**, which covers US long positions of managers above \$100m, quarterly, with a 45-day lag, and misses every short and most derivatives. **Short interest**, published twice monthly by the exchanges with a lag of about a week, plus **days to cover**. The **securities lending fee** on the short leg, which is a price rather than a survey and therefore the fastest honest crowding signal you have. The **correlation of your returns with peer indices**, which rises when everyone holds the same book. And the one nobody uses enough: **your own impact**. If your fills are getting worse at constant size and constant volatility, somebody is in front of you.

What you cannot observe is the thing you want, which is the total capital in your trade. Every proxy above is a lagged, partial view of a quantity that moves in days. August 2007 is the standing lesson: over the 6th to the 9th, crowded quantitative equity strategies took simultaneous losses as one large book deleveraged into positions that many funds held at once, and a substantial part reversed on the 10th. Khandani and Lo (2007) documented it. No 13F filed in time to help anyone, and the funds involved did not know their positions were crowded until the crowd ran.

So treat crowding metrics as a slow prior, not a trigger. The trigger, if you want one, is your own transaction costs.

## Fitting the half-life, and what the fit is worth

Suppose you accept exponential decay and try to estimate $H$ from live data. Over a short window the exponential is close to linear, so regress the monthly IC on time and read the decay rate off the slope:

$$\widehat{H} = \frac{\ln 2 \cdot \hat{a}}{-\hat{b}}, \qquad \hat{b} = \text{slope},\ \hat{a} = \text{intercept}.$$

That is a ratio of two estimates, and the denominator is the noisy one. Here is what that does.

#### Worked example 1: two years of data, and a half-life you cannot pin down

A signal ranks 1,000 names monthly. The sampling standard deviation of a cross-sectional IC on ${N}$ names is about ${1/\sqrt{N}}$, so each monthly IC carries noise of roughly **0.032**. Truth, which you do not know: ${IC_0 = 0.040}$ and a 30-month half-life.

- Over 24 months, $t$ runs 1 to 24, so ${S_{xx} = \sum (t - \bar{t})^2 = 1{,}150}$ and ${\sqrt{1{,}150} = 33.91}$.
- Fit the true curve and the slope is **-0.00070** per month (unrounded, -0.000698) with intercept ${\hat{a} = 0.0391}$.
- The slope's standard error is ${0.032/33.91 = 0.000944}$, so ${0.000698/0.000944 = 0.739}$ and **t = -0.739**.
- Point estimate of the half-life: ${0.6931 \times 0.0391/0.000698 = 38.8}$ months, against a truth of 30.
- The 95% interval on the slope is ${-0.000698 \pm 1.96 \times 0.000944}$, which is **-0.00255 to +0.00115**. It contains zero, so the interval on the half-life runs from ${0.6931 \times 0.0391/0.00255 = 10.6}$ months to **no decay at all**, and is unbounded above.

![Three lines fanning out from an information coefficient of 0.039, the point estimate implying a 38.8-month half-life, the lower bound 10.6 months, and the upper bound rising, so the ninety-five percent interval includes no decay](/imgs/blogs/alpha-lifecycle-decay-retirement-math-for-quants-3.webp)

The intuition: a half-life is a ratio, and when the denominator is not distinguishable from zero the ratio has no upper bound. Quoting "half-life 38.8 months" without the interval is the single most common way a decay estimate misleads a risk meeting.

In money, the undetectable decay is not small. At ${IC = 0.040}$ the sleeve grosses 6.00% on \$60m, which is \$3.600m a year. By month 24 the IC is ${0.040 \times 0.574349 = 0.023}$, so gross alpha is ${6.00 \times 0.574349 = 3.446}$ percent. 3.446% of \$60m is \$2.068m, so \$1.532m a year has evaporated while the t-statistic sat at -0.739.

#### The null result, which is the real finding

The natural next question is how long you would have to wait. At this noise level the slope reaches ${|t| = 2}$ after **60 months**, five years, at which point the signal retains 25.0% of its original IC. You can prove the decay only after losing three quarters of the edge.

Worse, and this is the part that should change how you talk: **no exponential half-life whatsoever is detectable at the 5% level in a 24-month window here.** Sweeping every half-life from half a month to sixty, the largest attainable ${|t|}$ is **1.40**, at a half-life of 6.7 months, worth 28.8% power. A 30-month half-life gives 11.5% power, meaning that a correctly specified test detects it roughly one time in nine. And the 0.032 noise figure is optimistic, because names are not independent: if the factor structure leaves 150 effective independent bets rather than 1,000, the noise becomes ${1/\sqrt{150} = 0.0816}$, the t-statistic falls to -0.29, and power reaches 6.0% against a 5% test size. That is a coin flip wearing a lab coat.

The senior conclusion is not "measure harder". It is that the decay estimate is a prior with a wide posterior attached, and the retirement decision has to be built to work anyway.

## Retirement is an option, not a threshold

Since the statistics will not tell you when to stop, decide by comparing what the two errors cost.

#### Worked example 2: the crossover date, and a year on either side of it

The \$60m sleeve inside a \$500m book. Gross alpha 6.00% at launch, half-life 30 months, costs 1.20%, hurdle 2.00%. Net alpha equals the hurdle when gross equals the 3.20% floor:

$$2^{-T^{\ast}/30} = \frac{3.20}{6.00} = 0.5333 \quad\Longrightarrow\quad T^{\ast} = 30 \log_2 1.875 = 27.21 \text{ months.}$$

Integrating the gap between the curve and the floor over a year on each side, with ${30/\ln 2 = 43.281}$:

- **A year late** (retire at month 39.21): ${3.20 \times 12 - 6.00 \times 43.281 \times (0.533333 - 0.404191)}$, which is ${38.40 - 33.536 = 4.864}$ percent-months. 4.864% of \$60m spread over twelve months is **\$243,200**.
- **A year early** (retire at month 15.21): ${6.00 \times 43.281 \times (0.703738 - 0.533333) - 38.40}$, which is ${44.252 - 38.40 = 5.852}$ percent-months, or **\$292,600**.

So being early costs ${292{,}600/243{,}200 = 1.203}$ times being late. Nearly symmetric, and both are rounding errors on a \$500m book. If that were the whole calculation, retirement timing would not be worth a meeting.

It is not the whole calculation, because worked example 1 says the decay might not be real. If the alpha was flat all along, the early kill forfeits ${(6.00 - 1.20 - 2.00)}$ percent of \$60m, which is **\$1.680m every year** until somebody notices, or ${1{,}680{,}000/243{,}200 = 6.91}$ times the cost of a year's patience. **That is the asymmetry: the cost of being late is bounded by a shrinking gap, and the cost of being early is the entire remaining value of a live strategy.**

#### Worked example 3: the decision under a posterior, not a rule

It is month 24. Gross alpha is 3.446%, net is 2.246%, still above the 2.00% hurdle, and the crossover is three months away. Retire now or keep for a year?

Start with a prior. McLean and Pontiff justify pessimism, so put ${P(\text{decaying}) = 0.60}$, odds of 1.500. The evidence is the observed slope, with ${t = -0.739}$ (that chart rounds it to -0.74). Comparing "half-life 30 months" against "flat", the likelihood ratio is ${e^{t^2/2}}$, so ${0.739^2 = 0.546121}$ and the Bayes factor is **1.314**. Posterior odds ${1.500 \times 1.314 = 1.971}$, so ${1.971/2.971 = 0.6634}$.

Two years of live data moved the belief from 60.00% to **66.34%**. That is the honest yield of the entire monitoring exercise.

Now price the two branches of keeping for twelve more months:

- **If the decay is real**, the sleeve runs below the floor for part of the year: ${6.00 \times 43.281 \times (0.574349 - 0.435275) = 36.116}$ against ${3.20 \times 12 = 38.40}$, a shortfall of 2.284 percent-months, which on \$60m is **-\$114,200**.
- **If it is flat**, you keep 2.80% of \$60m, **+\$1,680,000**.

![Two by two matrix of keep or retire against decaying or flat, showing minus 114,200 dollars for keeping a decaying alpha, plus 1,680,000 dollars for keeping a flat one, and minus 1,680,000 dollars a year for retiring a live alpha](/imgs/blogs/alpha-lifecycle-decay-retirement-math-for-quants-4.webp)

${0.6634 \times (-114{,}200) + 0.3366 \times 1{,}680{,}000}$ gives **+\$489,728 a year**. Keeping wins comfortably, and it wins at a posterior that says the alpha is probably dying, because the downside is capped at \$114,200 while the upside is ${1{,}680{,}000/114{,}200 = 14.7}$ times larger.

The lever that flips this is not the posterior. It is the hurdle. Raise the next best use of the capital and the expected value of keeping falls to zero at about **2.82%**. **Retirement becomes urgent when capital is scarce, not when a t-statistic moves.** A shop with nothing to redeploy into should be slow to kill; a shop with a queue of researched strategies should be fast. The asymmetry also runs the other way through a fact nobody models: a retired strategy can be restarted from the same code in a week, while a team that killed a good alpha loses the conviction to run the next one at size.

## The portfolio view, which usually decides it

A strategy's standalone Sharpe is not what it is worth. What it is worth is what it adds to the book you already have, and adding an uncorrelated mediocrity beats adding a correlated star. The [post on combining weak alphas](/blog/trading/math-for-quants/combining-weak-alphas-math-for-quants) derives the machinery; the retirement consequence is one line. With an existing book of Sharpe $S_P$, a candidate earns a positive weight if and only if

$$S_{\text{new}} \gt \rho \, S_P,$$

and below that line the constrained optimum is a weight of zero. Retire it.

#### Worked example 4: the same alpha, kept and killed

A \$500m book targeting 10% annual volatility, so \$50m of vol, running at an IR of 1.40.

- **Strategy B**, Sharpe 0.90, correlation 0.55 with the book. Hurdle ${0.55 \times 1.40 = 0.770}$, so keep. The combined IR is ${\sqrt{(1.9600 + 0.8100 - 1.3860)/0.6975}}$; since ${1.384/0.6975 = 1.984229}$ and ${\sqrt{1.984229} = 1.408627}$, the gain of 0.008627 on \$50m of vol is **\$431k a year**. Its risk budget is ${0.130/0.905 = 0.1436}$ of the existing book's, about \$7.18m of vol.
- **B after one halving**, Sharpe 0.70. Now 0.70 is below the same 0.770 hurdle, the optimal weight is zero, and the contribution is **\$0**. A 0.70 Sharpe is a perfectly respectable strategy that is worth nothing to this particular book.
- **Strategy C**, Sharpe 0.45, correlation 0.10. Hurdle ${0.10 \times 1.40 = 0.140}$. ${2.0365/0.9900 = 2.057071}$ and ${\sqrt{2.057071} = 1.434249}$, so the gain of 0.034249 is **\$1,712k a year**, and its risk budget is ${0.310/1.355 = 0.2288}$ of the book's, about \$11.44m of vol.

![Scatter of standalone Sharpe against correlation with the existing book, with a diagonal hurdle line at Sharpe equals rho times 1.40, strategy B above it, B after decay below it, and strategy C well above it](/imgs/blogs/alpha-lifecycle-decay-retirement-math-for-quants-5.webp)

C is worth ${0.034249/0.008627 = 3.97}$ times what B is worth on half the standalone Sharpe. Read the hurdle line both ways: B dies the moment its correlation to the book reaches ${0.90/1.40 = 0.643}$ even with no decay at all, while C survives down to a Sharpe of 0.140. **The same decay can be fatal in one book and irrelevant in another, so retirement is never a property of the strategy.**

## The failure mode is organisational

Nobody is promoted for retiring a strategy. The discovery has an author; the retirement has a defendant. So the incentives run one way, dead alphas accumulate in the book, and each one quietly holds capital at the hurdle rate while consuming risk limits, monitoring attention and someone's on-call rotation.

Three practices fix most of it, and all three are the senior's to install. Write the retirement criterion into the launch document, before anyone is attached: the floor, the hurdle, and the review date. Record the counterfactual, so a retired strategy keeps running on paper and the firm eventually learns whether its kills were right. And make retirement a routine scheduled decision rather than an event, because a decision that only ever happens after a drawdown is a decision made by the drawdown.

## Common misconceptions

**"The signal stopped working."** Almost never. It decayed toward the cost of capturing it, which is what a functioning market does to a public edge, and at 3.446% gross with a 3.20% floor it is still working. It is simply no longer worth the capital.

**"Three bad months means it is dead."** A Sharpe-1.5 strategy has losing quarters routinely. Every drawdown you are about to retire on was in the backtest's own distribution; [the companion post](/blog/trading/math-for-quants/live-vs-backtest-divergence-math-for-quants) puts a number on how rarely three months distinguishes anything.

**"We fitted the half-life, it is 38.8 months."** That estimate came with an interval from 10.6 months to infinity. A half-life quoted without its interval is a story, not a measurement.

**"Sharpe held up, so there is no decay."** Capacity falls as the square of the alpha, so the first symptom is that the same trade at the same size costs more to execute. Watch the transaction costs, not the tear sheet.

**"Retire it at a Sharpe of 0.5."** A fixed threshold ignores both the correlation hurdle and the alternative use of the capital, which are the two things that actually determine the answer. Strategy C at 0.45 is worth four times strategy B at 0.90.

## Sources and further reading

- R. David McLean and Jeffrey Pontiff (2016), "Does Academic Research Destroy Stock Return Predictability?", *Journal of Finance* 71(1), 5-32. The 97 predictors, and the 26% / 58% decomposition.
- Campbell R. Harvey, Yan Liu and Heqing Zhu (2016), "... and the Cross-Section of Expected Returns", *Review of Financial Studies* 29(1), 5-68. The 316-factor count and the ${t \gt 3.0}$ hurdle.
- Richard Grinold and Ronald Kahn, *Active Portfolio Management*, 2nd edition, 1999. The fundamental law, the information horizon, and the marginal-contribution framing behind worked example 4.
- Amir Khandani and Andrew Lo (2007), "What Happened to the Quants in August 2007?", *Journal of Investment Management* 5(4). The crowding unwind.

The dollar figures in the worked examples are illustrative arithmetic on assumed inputs, not observed results from any fund. The two empirical anchors are the McLean and Pontiff percentages and the Harvey, Liu and Zhu factor count and hurdle.

## In the interview room and on the desk

The question arrives as "how do you know when a signal is dead?", and it is usually a screen for whether you confuse a performance threshold with a decision. The weak answer names a number: Sharpe below 0.5 for two quarters, or a drawdown past some multiple of backtested vol. It sounds disciplined and it is not, because it never asks whether the data could distinguish the two hypotheses.

The strong answer runs in four steps. **First, separate decay from noise**, and say out loud that you usually cannot: a 30-month half-life fitted to 24 months of monthly ICs carries t = -0.739 and 11.5% power, and no half-life at all clears the 5% level in that window. **Second, put an error bar on the half-life itself** and note that it is a ratio whose denominator straddles zero, so the interval runs from 10.6 months to no decay and is unbounded above. **Third, reframe retirement as a decision with asymmetric costs**: keeping a decayed alpha costs the bounded gap to the hurdle, \$114,200 on a \$60m sleeve, while killing a live one costs its whole remaining value, \$1.68m a year, which is why the expected value of waiting is positive at a 66.34% belief that the alpha is dying. **Fourth, say what actually flips it**: the hurdle. Retirement becomes urgent when the capital has somewhere better to go, and the strategy's correlation to the existing book decides that, not its standalone Sharpe.

The trap is the drawdown that the backtest always contained. A candidate who retires on it looks rigorous and has made the one error the whole framework exists to prevent, because the drawdown is evidence about luck and the retirement decision is about the hurdle. The related trap is quoting a fitted half-life with no interval, which is the same mistake wearing better clothes.

WorldQuant weights this most heavily, since a factory producing thousands of signals lives or dies on the retirement rule rather than the discovery rate. Citadel and Two Sigma press on the portfolio version: expect a follow-up asking what the same strategy is worth in a book that already holds three things correlated 0.5 to it, and the expected answer is zero.

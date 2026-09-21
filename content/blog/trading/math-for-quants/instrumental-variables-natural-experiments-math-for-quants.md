---
title: "Instrumental variables: borrowing randomness you do not have"
date: "2026-09-21"
publishDate: "2026-09-21"
description: "When the confounder cannot be measured, no set of controls closes the back door. An instrument is a variable that moves your treatment for reasons unrelated to the outcome, and it buys a causal estimate out of observational data. What it has to satisfy, how two-stage least squares uses it, and why the failure modes are quiet."
tags: ["instrumental-variables", "two-stage-least-squares", "exclusion-restriction", "weak-instruments", "local-average-treatment-effect", "natural-experiments", "causal-inference", "index-inclusion", "quant-research", "math-for-quants", "quant-interview", "quant-finance"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 20
---

> [!important]
> **TL;DR:** When the confounder is unmeasurable you cannot control your way to a causal number. You can sometimes borrow randomness instead: a variable that shoves your treatment around for reasons unconnected to the outcome.
>
> - An instrument $Z$ needs **relevance** ($Z$ moves $X$), **exclusion** ($Z$ touches $Y$ through $X$ and no other path) and **independence** ($Z$ is as-good-as-random). Only the first is testable. The other two are arguments you make in words, and that asymmetry is the whole craft.
> - Two-stage least squares keeps only the part of $X$ that $Z$ explains. With one instrument it is a single ratio: reduced form over first stage.
> - In the worked example OLS reads **-1.90%** per cent of quoted spread and IV reads **-0.67%**. On a \$50m position, a \$950,000 estimate against a \$335,000 one, of the same effect.
> - Weaken the first stage from 1.5 cents to 0.12 cents and the design returns **-1.29%, standard error 1.75**, an interval from -4.72% to +2.14%. The estimate slides back toward OLS and the interval swallows zero.
> - Size off the confounded coefficient and you book 52 bps a month against a 35 bps hurdle instead of 18 bps. On \$140m of gross that turns **\$2.86m a year of expected profit into \$2.86m a year of loss**, and about 30 years of P&L would be needed to notice.
> - You estimate the effect on **compliers**, the units the instrument moved. In markets that subgroup is often not the one holding your risk.

## When no control set can save you

[Causal inference for alpha research](/blog/trading/math-for-quants/causal-inference-alpha-research-math-for-quants) ended on the back-door criterion, which tells you exactly which controls close the paths running backwards from treatment into outcome. It has one requirement that quietly does all the work: those variables must be **measured**. When the common cause is something nobody records, the criterion still tells you what to do and you still cannot do it.

That is the normal case in markets. Whatever makes a market maker widen a quote also moves the price, and "what she inferred from the flow she just saw" appears in no vendor's schema. The confounder is real, first order, and unmeasurable.

![Causal diagram with an unmeasured confounder U above, arrows down to X the quoted spread in cents and Y the stock return, a bold arrow from X to Y labelled the effect you want, the instrument Z feeding into X with a first stage of plus 1.5 cents, and two red dashed arrows marking the assumed absent paths Z to Y and U to Z](/imgs/blogs/instrumental-variables-natural-experiments-math-for-quants-1.webp)

$U$ carries a dashed border because it is real and invisible, and both arrows out of it are alive, so the association between $X$ and $Y$ mixes the effect you want with the effect $U$ manufactures. So you come in from the side: $Z$ pushes $X$ around. The two red dashed arrows are the price of admission. You are asserting that $Z$ has no direct line to $Y$, and that nothing causing $Y$ also causes $Z$. Neither assertion is visible in the data.

## Foundations: what an instrument is

The structural equation is a statement about mechanism, not about fit:

$$
Y_i = \beta X_i + \gamma U_i + \varepsilon_i
$$

$\beta$ is the number you want: what happens to $Y$ if something reaches in and changes $X$ by one unit. Because the unmeasured $U$ also causes $X$, the treatment moves with the error term and least squares converges to $\beta$ plus a bias term that does not shrink with more data. A hundred million rows buys a precise estimate of the wrong number, which is worse than a noisy estimate of the right one, because precision is what makes a committee believe you.

An **instrument** is a variable $Z$ satisfying three conditions.

**Relevance**, $\operatorname{Cov}(Z, X) \neq 0$: the instrument moves the treatment. Both variables are observed, so this one is testable, and the whole weak-instrument literature is about how thin it can get before everything breaks.

**Exclusion**: $Z$ affects $Y$ only through $X$. Not testable with one instrument, because the data contain no version of the world where $Z$ moved and $X$ did not.

**Independence**, $\operatorname{Cov}(Z, U) = 0$: nothing that causes the outcome also causes the instrument. Sometimes nearly free, because a regulator assigned $Z$ by published rule. Usually argued for.

Now watch the three work. Take the covariance of $Z$ with both sides:

$$
\operatorname{Cov}(Z, Y) = \beta \operatorname{Cov}(Z, X) + \gamma \operatorname{Cov}(Z, U) + \operatorname{Cov}(Z, \varepsilon)
$$

Independence kills the second term. Exclusion is what let you write the structural equation with no $Z$ in it, and it kills the third. What survives divides cleanly:

$$
\beta = \frac{\operatorname{Cov}(Z, Y)}{\operatorname{Cov}(Z, X)}
$$

Each condition is visibly responsible for one thing. Relevance keeps the denominator away from zero, independence and exclusion empty the numerator of everything but the channel you want. Break one and it is a ratio of two covariances with no interpretation.

The everyday version: does carrying an umbrella keep you dry? Comparing carriers to everyone else is hopeless, because people carry one when they expect rain and expecting rain predicts arriving wet. Now suppose the building hands out free umbrellas on randomly chosen days. The giveaway moves umbrella-carrying, keeps you dry only by putting an umbrella in your hand, and was decided by a coin rather than by the sky. Note where the sceptic attacks: if giveaways run on days the lobby is already staffed for bad weather, exclusion is dead, and no amount of umbrella data will tell you so.

## Two-stage least squares, mechanically

The ratio above assumes one instrument, one treatment, no covariates. Two-stage least squares handles all three, and the name is literal. **Stage one** regresses the treatment on the instrument, $X_i = \pi_0 + \pi_1 Z_i + \eta_i$, and keeps the fitted values $\hat X_i$. **Stage two** regresses the outcome on those fitted values.

![Stacked bar of all variation in quoted spread, split into a green segment driven by the instrument worth plus 1.5 cents at F equals 115 and a larger red segment driven by U and everything else, with an arrow carrying only the green segment through both stages to an IV estimate of minus 0.67 percent, against a grey box showing OLS on all the variation at minus 1.90 percent](/imgs/blogs/instrumental-variables-natural-experiments-math-for-quants-2.webp)

That is the whole conceptual move. The variation in the spread comes from two places: some forced on the stock by an external rule, the rest from news and fundamentals, which is precisely the variation that also moves returns directly. OLS uses the whole bar. Stage one splits it and stage two keeps only the green segment, so you are discarding most of the sample's information about $X$. IV standard errors therefore always exceed OLS standard errors on the same data: you buy consistency with precision, at an exchange rate set by the first stage.

#### Worked example: the liquidity coefficient, OLS against IV

Does a wider quoted spread **cause** a lower price? All numbers below are illustrative arithmetic on assumed inputs, built so every step is checkable in one line.

Take 800 small-cap stocks over a two-year window. $X$ is the average quoted spread in cents, $Y$ the cumulative abnormal return in percent.

**The naive regression.** Pool all 800 and regress $Y$ on $X$. The slope is **-1.90% per cent of spread**. Read causally, one cent wider destroys 1.90% of market value: on a \$50m position, \$950,000.

**Why it is confounded.** A company whose prospects are deteriorating gets both a wider spread, because market makers who suspect informed flow pull back, and a falling price. That is $U \rightarrow X$ and $U \rightarrow Y$, and the variable is what those market makers inferred, which nobody records.

**The instrument.** 400 of the 800 were assigned to a regulatory pilot forcing 5-cent quoting increments; the other 400 were the control. $Z$ is test-group membership.

**The first stage.** Average spread 6.2 cents in the test group, 4.7 in the control:

$$
\hat\pi_1 = 6.2 - 4.7 = 1.5 \text{ cents}, \quad \text{SE} = 0.14, \quad t = 10.7
$$

**The reduced form**, the instrument's effect on the outcome directly. Test group -1.30%, control -0.30%:

$$
\hat\delta = -1.30 - (-0.30) = -1.00\%, \quad \text{SE} = 0.22
$$

**The IV estimate** is one over the other:

$$
\hat\beta_{\text{IV}} = \frac{-1.00}{1.5} = -0.67\% \text{ per cent of spread}
$$

with a standard error near ${0.22/1.5 = 0.15}$ and a t-statistic of 4.6. Running stage two literally gives the same number: $\hat X$ takes only the values 4.7 and 6.2, so the slope through the group means is ${(-1.30 + 0.30)/(6.2 - 4.7) = -0.667}$.

**What the gap is.** The confounded estimate is **2.8 times** the causal one. On that \$50m position, \$950,000 against \$335,000, and the \$615,000 difference is not liquidity at all. It is fundamentals wearing a liquidity costume.

*The intuition: OLS answered "what do wide-spread stocks do", IV answered "what happens when you widen a spread", and where bad news widens spreads those are different questions.*

## When relevance is thin: weak instruments

Relevance is testable, which makes it sound like the safe condition. It is not, because passing a test for "not exactly zero" is a long way from "strong enough to divide by". The estimator is a ratio whose denominator is the first stage, so when that is small two things go wrong at once.

**The variance explodes.** With one instrument the standard error is roughly $\text{SE}(\hat\delta)/|\hat\pi_1|$, so halving the first stage doubles it with no change in the reduced form.

**The estimate becomes biased toward OLS.** In finite samples, sampling noise in $Z$ correlates with the structural error by chance, and that accidental correlation acts exactly like a violation of independence, producing bias in the same direction as the OLS bias. A weak instrument hands you not a noisy version of the truth but a noisy version of the number you were escaping, and with no first stage at all 2SLS collapses onto OLS. Bound, Jaeger and Baker made this concrete in 1995 by rerunning a famous IV study with randomly generated instruments and getting results that looked entirely reasonable.

The diagnostic is the **first-stage F-statistic**, which with a single instrument is the square of the t. The folklore threshold $F > 10$ comes from Staiger and Stock in 1997: at an expected F of 10, 2SLS bias is around 10% of the OLS bias and a nominal 95% Wald interval has at least roughly 85% coverage, the model being that relative bias falls about like ${1/F}$. Treat it as folklore, because it is. Lee, McCrary, Moreira and Porter showed in 2022 that a genuine 5% t-test with one instrument needs $F > 104.7$, and that in a quarter of the specifications they examined across 61 published papers, correctly sized standard errors were at least 49% larger than the conventional ones. An F of 12 clears the folklore bar and fails the real one.

#### Worked example: the same design with a weak instrument

Change one thing. Suppose the pilot rule had carried enough exemptions, midpoint executions, retail price improvement, negotiated blocks, that the forced widening barely bit.

**First stage.** The test group's spread is now 0.12 cents wider than the control's, standard error 0.085, so $t = 1.41$ and $F = t^2 = 2.0$.

**Reduced form.** With little treatment delivered the outcome gap shrinks too: -0.155%, standard error 0.21.

**The estimate.**

$$
\hat\beta_{\text{IV}} = \frac{-0.155}{0.12} = -1.29\%, \quad \text{SE} \approx \frac{0.21}{0.12} = 1.75
$$

giving a 95% interval of roughly **-4.72% to +2.14%**.

![Two estimates with their intervals on a shared axis from minus 5 to plus 2 percent, with reference lines at OLS minus 1.90, strong IV minus 0.67 and zero; the strong row at F equals 115 is a short bar on minus 0.67, the weak row at F equals 2.0 is a long bar from minus 4.72 to plus 2.14 whose point at minus 1.29 has slid toward OLS and whose interval crosses zero](/imgs/blogs/instrumental-variables-natural-experiments-math-for-quants-3.webp)

**The point estimate moved toward OLS**, from -0.67 to -1.29, almost exactly halfway to the OLS value of -1.90. That is what ${1/F}$ predicts: the gap between the causal number and the confounded one is 1.23 percentage points, an F of 2 leaves roughly half the bias standing, and half of 1.23 taken off -0.67 is -1.29. You got not a wider version of the right answer but a partial retreat to the wrong one.

**The interval blew up**, from a standard error of 0.15 to 1.75, nearly twelve times wider. On a \$50m position it says one cent of spread costs somewhere between a \$2.36m loss and a \$1.07m gain. That is not a measurement, it is a shrug with a decimal point.

An F of 2.0 is impossible to miss. The dangerous cases are F of 12 or 25: respectable-looking, clearing the threshold everyone quotes, still materially biased with intervals that under-cover. This study's strong version had F of 115, which clears even the Lee bar.

*The intuition: the first stage is the exchange rate between your instrument and your treatment, and near zero you are converting at a rate you cannot measure.*

## You did not estimate the average effect

Suppose it all works. You still have not estimated the average effect of $X$ on $Y$ across your universe, only the effect on the units the instrument moved. Imbens and Angrist showed in 1994 that under **monotonicity**, which says the instrument pushes everyone the same direction or not at all, IV recovers the **local average treatment effect**: the average among **compliers**, whose treatment status changed because of the instrument.

![One horizontal stacked bar of the small-cap universe in the tick size pilot, split into compliers at 58 percent forced wider by the rule, always-wide at 22 percent already above 5 cents, and never-moved at 20 percent quoting tight through the exemptions, with an arrow into the complier segment reading your IV estimate describes this slice and no other](/imgs/blogs/instrumental-variables-natural-experiments-math-for-quants-4.webp)

In the pilot the groups are concrete. **Compliers** quoted inside 5 cents and got pushed wider, so they are the only stocks where the experiment did anything. **Always-wide** stocks already quoted above 5 cents, so the floor bound on nothing. **Never-moved** stocks kept quoting tight through the exemptions. **Defiers**, which would have tightened because the minimum increment widened, are assumed not to exist. The shares are illustrative, and you never see which group a given stock is in.

This bites harder in markets than in economics, because the complier subgroup is usually defined by exactly the characteristic that makes a name atypical. The stocks the tick rule moved quoted tight enough for a 5-cent floor to bind, meaning liquid small caps. If your book lives in the illiquid tail, you have a clean causal estimate for a population you do not own. Two honest researchers with two valid instruments for the same treatment can therefore report different numbers and both be right. That is not a replication failure, it is the estimand changing under you.

## Natural experiments in markets

Four families come up most. Notice that the condition under strain is almost never relevance.

**Index inclusion and deletion.** Index funds must buy, so relevance is mechanical. Shleifer (1986) and Harris and Gurel (1986) both found significant abnormal returns on announcement of an S&P 500 addition, Harris and Gurel measuring 3.13% with an offsetting -2.49% over the next 29 trading days. **Exclusion is the strain**: a committee chose the name, so inclusion carries information and simultaneously changes analyst coverage, options listing, borrow and institutional eligibility. The same literature also shows that an instrument's strength is not a constant. Greenwood and Sammon document the effect falling from 3.4% in the 1980s and 7.6% in the 1990s to 0.8% over 2010 to 2020, with deletions at -0.6%, as additions became predictable and arbitrageurs front-ran them. A design strongly identified in 1995 is weakly identified now.

**Tick-size pilots.** The SEC's pilot began in October 2016 and ran two years, moving roughly 1,200 small-capitalisation stocks into 5-cent quoting. **Independence is the strong condition**, since assignment came from a published rule. **Exclusion is the strain**: trade-at provisions moved order flow between venues, so the treatment bundles a spread change with a routing change.

**Regulatory thresholds.** Reconstitution ranks, filer-status cutoffs, capital-bucket boundaries. **Independence is the strain**, because firms manipulate whatever decides which side of a bright line they land on. The diagnostic is a density test around the cutoff.

**Weather for commodities.** **Independence is free**, since nobody manipulates the weather. **Exclusion is the strain, badly**: a hurricane that shuts a refinery also shuts trucking and moves risk appetite. Angrist, Graddy and Imbens used stormy weather at sea to instrument wholesale fish prices precisely because independence is unimpeachable there, and even so the argument in the paper is about exclusion.

## What the difference is worth on a real book

An estimate is not the deliverable. A position size is, and the arithmetic turning a coefficient into a size is linear, so a coefficient 2.8 times too large produces a book 2.8 times too large.

#### Worked example: sizing off the confounded coefficient

You run a \$500m equity market-neutral book. A researcher brings a signal: buy names whose spreads are about to tighten. Expected edge is the forecast spread change times the coefficient above. Illustrative arithmetic on assumed inputs throughout.

1. **Edge under the OLS coefficient.** At -1.90% per cent of spread, expected gross edge is **52 bps per month** on deployed gross.
2. **Edge under the IV coefficient.** The same spread forecast scaled by the causal coefficient: ${52 \times (0.67/1.90) = 18}$ bps per month.
3. **The cost hurdle.** The book turns over once a month and round-trip cost in these names is 35 bps of notional traded.
4. **The sizing decision.** 52 clears 35 comfortably, so the committee allocates **\$140m of gross**. Note that 18 does not clear the hurdle at all, so the correct allocation under the causal estimate is zero.
5. **Expected P&L as pitched.** ${(52 - 35) = 17}$ bps per month on \$140m is **\$238,000 a month**, about **\$2.86m a year**.
6. **Expected P&L as it really is.** ${(18 - 35) = -17}$ bps per month is **minus \$238,000 a month**, about **minus \$2.86m a year**.

The swing is **\$5.72m a year** on a \$500m fund, 1.14% of the book, from one coefficient in one regression.

Now the part that makes it dangerous rather than merely expensive. Say the strategy runs at 1.6% monthly volatility on that gross, about \$2.24m a month of standard deviation. You are losing \$238,000 a month against \$2.24m of noise, so rejecting "this strategy is flat" at a t-statistic of 2 needs

$$
\sqrt{n} = 2 \times \frac{2.24}{0.238} \approx 18.8, \quad n \approx 354 \text{ months}
$$

about **thirty years** of live trading. The P&L will never tell you. That is the [sample-complexity arithmetic](/blog/trading/math-for-quants/concentration-inequalities-sample-complexity-math-for-quants) pointed in the direction people find least comfortable: a losing strategy is also indistinguishable from flat, so it survives every review on the grounds that it is within noise. Only the research design could have caught it.

*The intuition: bad coefficients do not announce themselves in the P&L, because the P&L is too noisy to see them. They announce themselves in the identification argument or not at all.*

## Common misconceptions

**"Any variable correlated with my treatment works as an instrument."** Correlation with $X$ is one of three conditions and much the cheapest. A variable correlated with both $X$ and $U$ is worse than useless, because IV divides the violation by the first stage:

$$
\hat\beta_{\text{IV}} \;\xrightarrow{p}\; \beta + \frac{\gamma \operatorname{Cov}(Z, U)}{\operatorname{Cov}(Z, X)}
$$

Put numbers on it. Say the pilot's routing changes cost test-group stocks 0.05 percentage points of return directly: a tiny exclusion violation. With the strong first stage of 1.5 cents it adds ${0.05/1.5 = 0.033}$, 5% of -0.67, and you would never care. With the weak first stage of 0.12 cents it adds ${0.05/0.12 = 0.42}$, which is 63% of the estimate. **Weak and slightly invalid is not a small problem, it is a multiplicative disaster.**

**"A high F-statistic means my instrument is valid."** It means the instrument is relevant, and nothing else. Exclusion and independence leave no fingerprint in the data with one instrument and one endogenous variable, because no degrees of freedom remain to test anything with. Over-identification tests such as Sargan or Hansen's J only ask whether several instruments agree, and several instruments invalid in the same direction agree beautifully. Exclusion is not something you test, it is something you argue.

**"IV gives me the average treatment effect."** It gives a local average treatment effect on compliers under monotonicity. If the instrument moved the liquid names and your book holds the illiquid ones, your coefficient cleanly estimates something that does not describe your risk.

## Sources and further reading

- Joshua Angrist and Alan Krueger, "Does Compulsory School Attendance Affect Schooling and Earnings?", *Quarterly Journal of Economics* 106(4), 1991. Quarter of birth as an instrument.
- Joshua Angrist and Jörn-Steffen Pischke, *Mostly Harmless Econometrics*, Princeton University Press, 2009. Chapter 4 covers everything above.
- Guido Imbens and Joshua Angrist, "Identification and Estimation of Local Average Treatment Effects", *Econometrica* 62(2), 1994. Monotonicity and compliers.
- John Bound, David Jaeger and Regina Baker, "Problems with Instrumental Variables Estimation when the Correlation between the Instruments and the Endogenous Explanatory Variable is Weak", *Journal of the American Statistical Association* 90(430), 1995. The randomly-generated-instruments demonstration.
- Douglas Staiger and James Stock, "Instrumental Variables Regression with Weak Instruments", *Econometrica* 65(3), 1997, and James Stock and Motohiro Yogo, "Testing for Weak Instruments in Linear IV Regression", 2005. The F > 10 rule and its proper critical values.
- David Lee, Justin McCrary, Marcelo Moreira and Jack Porter, ["Valid t-ratio Inference for IV"](https://www.aeaweb.org/articles?id=10.1257%2Faer.20211063), *American Economic Review* 112(10), 2022, 3260-90. Why a true 5% test needs F > 104.7.
- Andrei Shleifer, "Do Demand Curves for Stocks Slope Down?", *Journal of Finance* 41(3), 1986, and Lawrence Harris and Eitan Gurel, "Price and Volume Effects Associated with Changes in the S&P 500 List", *Journal of Finance* 41(4), 1986, 815-29. The 3.13% announcement effect and its -2.49% reversal.
- Robin Greenwood and Marco Sammon, ["The Disappearing Index Effect"](https://www.nber.org/papers/w30748), NBER Working Paper 30748, 2022, published in the *Journal of Finance*, 2025. The decay to 0.8% over 2010 to 2020.
- Joshua Angrist, Kathryn Graddy and Guido Imbens, "The Interpretation of Instrumental Variables Estimators in Simultaneous Equations Models with an Application to the Demand for Fish", *Review of Economic Studies* 67(3), 2000. Weather as an instrument.
- [SEC Tick Size Pilot Program](https://www.sec.gov/data-research/tick-size-pilot-program) documentation, for the 2016 to 2018 pilot design.

Every number in the worked examples is illustrative arithmetic on assumed inputs. The empirical magnitudes attributed to published studies are the ones those studies report.

## In the interview room and on the desk

The question is almost never "explain instrumental variables". It arrives as a market question: *a stock gets added to the index and it pops 3%. Did the inclusion cause the move, or did it only predict it?* Two Sigma and Citadel both ask versions of this in research rounds, because it separates a candidate who has read about identification from one who can run it on a problem they have not seen.

A strong answer moves in a fixed order. **Name the confounder that makes the naive comparison useless.** Committees add companies that have grown and stabilised, so inclusion correlates with everything good that was already happening, and comparing added to non-added names measures the committee's taste as much as the flows. **Name the instrument and the variation you are using.** Reach for something rule-based rather than discretionary, a mechanical rank cutoff in a reconstitution or a pilot assignment, because the discretionary committee decision is the very thing you are escaping. **State the exclusion restriction out loud, as a sentence.** "I am assuming that crossing the rank threshold affects the price only through forced index demand, and not through analyst coverage, options listing, borrow supply or the certification value of membership." **Then say what would break it**: a density test around the cutoff for manipulation, a first-stage F to confirm the flows really moved, a placebo on names crossing the threshold in a year with no reconstitution.

The trap is treating exclusion as something you test. A candidate who offers "I would run an over-identification test to check the instrument is valid" has said something that sounds rigorous and is wrong, because the J-test asks whether several instruments agree, and instruments invalid in the same direction agree perfectly. The same trap in its other costume is quoting an F of 30 as though it settled the question: it settles relevance, and after Lee and co-authors it barely settles that. The candidate who says "this condition is not testable, here is my argument for it, and here is the observation that would make me abandon it" has shown what the question was asked to find.

Worth volunteering too: **the index effect has largely gone away**, with recent addition abnormal returns near 0.8%. Saying so shows you treat identification as a live property of a market rather than a fact about a paper.

This is educational material about research method, not investment advice.

---
title: "Model validation: what an independent reviewer actually checks"
date: "2026-09-28"
publishDate: "2026-09-28"
description: "Validation is not a compliance ritual and it is not re-running the researcher's backtest. It is one adversarial question, what would have to be true for this model to be wrong, and a small set of tests that answer it. The cheapest of them, benchmarking against a deliberately simple model, is the one most often skipped."
tags: ["model-validation", "model-risk", "sr-11-7", "governance", "backtest-overfitting", "benchmarking", "sensitivity-analysis", "deflated-sharpe-ratio", "quant-interview", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 21
---

> [!important]
> **TL;DR:** Validation is not reproduction. Re-running the researcher's code reproduces the researcher's bugs. An independent reviewer rebuilds the number from the specification and then asks what would have to be true for the model to be wrong.
>
> - The single most informative test is also the cheapest and the most often skipped: **benchmark against a deliberately simple model**. On a \$500m book, a gradient-boosted forecast beating a three-factor linear one by Sharpe 1.31 against 1.24 is worth **\$382k a year** after its own costs, 7.6 basis points of the book.
> - And you cannot prove even that. The two return streams are 0.93 correlated, so the t-statistic on the difference is **0.32**, and reaching ${t = 2}$ would take **114.3 years**. That null result is the finding.
> - **Sensitivity ranks the model risk.** One standard error on the alpha forecast moves annual net P&L by **\$18.9m**. One standard error on the impact coefficient moves it **\$0.65m**. The quarter of engineering went into the second one.
> - A validated model is validated **for a stated use**. An impact model calibrated on orders of 0.5% to 3% of average daily volume, applied at 12%, understates cost by **34.64 bps**, which is **\$6.24m a year** on a \$1.8bn program.
> - Governance exists because organisations forget, not because regulators are fussy. Tiering, severity, and a remediation date are the machinery that makes a finding survive the researcher's next promotion.

## The meeting you will be in

A researcher has a model. It is good. The backtest is clean, the code is tested, the Sharpe is 1.8, and they would like to run \$500m against it starting next month. You are the reviewer. You have two weeks, no stake in the outcome, and the same data they had.

The tempting move is to clone the repository, run the backtest, and confirm the number. Resist it. If you get 1.8 you have learned almost nothing, because a shared bug reproduces perfectly and a shared assumption reproduces even better. Every implementation error survives a re-run. So does every look-ahead leak in the data loader, every survivorship hole in the universe, and every point where the model was fitted to the same three years it is now being scored on.

What you are looking for is narrower and harder. Model risk is not the risk of a bug. It is the risk of a model that is **right on the data it saw and wrong on the data it will see**, and no amount of re-running finds that. The Federal Reserve's supervisory guidance, SR 11-7, puts it in one sentence: model risk is the potential for adverse consequences from decisions based on incorrect or misused model outputs, and it has two sources, a model that is wrong and a model used where it does not apply. Different failures, different tests.

![The three pillars of validation, all hanging off one adversarial question: conceptual soundness, ongoing monitoring including benchmarking, and outcomes analysis, producing findings with severity and a permitted use](/imgs/blogs/model-validation-governance-math-for-quants-1.webp)

That figure is the mental model for the whole post. One question at the top, three pillars under it, and an output that is not a yes or a no but a **permitted use with conditions attached**.

## Foundations: four terms and one definition

A **model**, in the regulatory sense the rest of this post uses, is any quantitative method that turns input data into an estimate a decision depends on. SR 11-7 splits it into three components that fail differently: the **information input**, the **processing** that applies the theory, and the **reporting** that turns the output into something a human acts on. A perfect equation fed stale data is a broken model. So is a perfect forecast reported without its error bar.

**Model risk** is the expected cost of acting on that estimate when it is wrong. It has a dollar size, and the first job of validation is to work out what that size is.

**Effective challenge** is SR 11-7's term for what makes validation work: critical analysis by people who are objective, competent, and senior enough to make someone change the model. All three matter. A reviewer who understands the maths but reports to the researcher's boss is not independent. A reviewer who is independent but cannot read the code is not competent. Both produce a signed document and no challenge.

**Tiering** decides how much of the above a given model gets. A model sizing a \$500m book is not reviewed like one that colours a dashboard. Materiality drives depth, and saying so explicitly is what stops a validation function spending its year on the easy models.

One conversion to carry: a **basis point** is one hundredth of a percent, 0.01%. On a \$500m book, one basis point is \$50,000 a year.

## Pillar one: conceptual soundness, which catches most of it

Before any test, the reviewer asks the researcher to state the mechanism in one sentence. Not the method, the mechanism: why should this number predict that number?

"Gradient boosting on 180 features" is a method. "Stocks whose short interest rises while their borrow cost stays flat are being shorted by people not paying up for urgency, and that is slower, better informed selling" is a mechanism. The first cannot be wrong, because it does not claim anything. The second can be wrong in three specific places, and now you know where to look.

Most models that fail fail here, visibly. The tell is usually one of four things. The mechanism requires an economic agent who does not exist. It is real but the data cannot see it, so the model fits a proxy with its own unrelated dynamics. It was real in the sample and has been arbitraged since. Or it is stated so loosely it would have explained the opposite result equally well, which is the one that gets through, because it is unfalsifiable rather than false.

This pillar also covers developmental evidence: why this functional form, this estimation window, this universe. The reviewer's question is not "is this choice correct" but "was this choice made or inherited". Inherited choices are where the assumptions nobody re-examined live.

## Pillar two: replicate from the specification, not the repository

Process verification asks whether the model does what the documentation says. The only way to answer it is to **rebuild the number from the specification**, in different code, ideally in a different language, from raw data you pulled yourself.

This is slower than cloning the repo, and it is the difference between validation and inspection. When your independent build matches to the fourth decimal, you have learned that the specification is complete enough for someone else to implement, which is a genuinely strong claim. When it does not match, the gap is the finding: either the document omits something the code does, or the code does something the document does not sanction. The second is more common and more interesting, because it is usually a fix applied once during debugging and never written down.

It is cheaper than it sounds. You rebuild only the path from raw data to the one number the decision depends on, and you can do it on a subsample, because a specification error of this kind shows up on 200 names as clearly as on 3,000.

## The benchmark test: what is the complexity actually worth?

Now the test that matters most and gets skipped most. Put the researcher's model beside a deliberately, almost insultingly simple alternative on exactly the same problem: same universe, same period, same cost model. Then ask what the difference is worth in money after the complexity has paid its own bills.

It is the most informative single test because it is the only one that prices the model against its real alternative. Every other test compares the model to being wrong. This one compares it to being simple, which is what you would do instead.

#### Worked example 1: a gradient-boosted model against three factors on a \$500m book

The numbers below are illustrative arithmetic on assumed inputs, sized to be the right order of magnitude for a liquid equity market-neutral book. They are not a claim about any real strategy.

The book is \$500m at 8% annualised volatility, so one unit of Sharpe ratio is worth ${0.08 \times \$500\text{m} = \$40\text{m}}$ a year of expected P&L. That conversion is the whole example.

- The **simple benchmark**, a three-factor linear forecast on value, momentum and quality, backtests at Sharpe **1.24**.
- The **candidate**, gradient boosting on 180 features, backtests at Sharpe **1.31**.

So the simple model delivers ${1.24 / 1.31 = 94.7\%}$ of the candidate's Sharpe. The gross uplift is ${(1.31 - 1.24) \times \$40\text{m} = \$2.800\text{m}}$ a year.

Now charge the complexity for what it consumes. The candidate's forecasts are noisier name by name, so it trades more: 68% of the book one way each month against the benchmark's 45%, which annualises to 8.16 turns against 5.4. That is ${(8.16 - 5.4) \times \$500\text{m} = \$1.380\text{bn}}$ of extra one-way notional a year, and at an all-in one-way cost of 11.0 bps, ${\$1{,}380\text{m} \times 0.0011 = \$1.518\text{m}}$.

Then charge it for upkeep: the feature pipeline, the vendor data the three-factor model does not need, and roughly 1.5 fully loaded researcher-years. Call it \$900k.

In millions of dollars a year:

$$2.800 - 1.518 - 0.900 = 0.382$$

![Waterfall showing a \$2.800m gross uplift reduced by \$1.518m of extra trading and \$0.900m of model upkeep to \$0.382m net, with the t-statistic on the difference at 0.32](/imgs/blogs/model-validation-governance-math-for-quants-2.webp)

**\$382k a year, or 7.6 basis points of the book.** That is what 180 features buy over three factors once they pay their own way. Not zero, and not obviously worth the operational risk of a model nobody in the room can explain in a sentence. That trade-off is now a decision someone can make with a number in front of them rather than an argument about elegance.

The intuition: a benchmark test does not ask whether the model works. It asks what you are paying for the part that is not simple.

## The null result: the benchmark test cannot tell you either

Here is the part that makes the example honest, and the part a weak validation report leaves out.

Is the 0.07 Sharpe difference real? The two models forecast the same names from overlapping information, so their return streams are highly correlated. Take ${\rho = 0.93}$ with both at 8% volatility. The difference series ${d = r_{\text{complex}} - r_{\text{simple}}}$ has volatility

$$\sigma_d = \sqrt{\sigma_c^2 + \sigma_s^2 - 2\rho\,\sigma_c\sigma_s} = 8\% \times \sqrt{2 - 2(0.93)} = 2.993\%$$

and an expected return of ${0.07 \times 8\% = 0.56\%}$ a year. So the difference has its own Sharpe ratio of ${0.56 / 2.993 = 0.1871}$, and over the three-year backtest its t-statistic is

$$t = 0.1871 \times \sqrt{3} = 0.324$$

A t-statistic of **0.32**. To reach ${t = 2}$ you would need ${(2 / 0.1871)^2 = 114.3}$ years of data. The correlation is doing you a large favour here by cancelling most of the shared volatility, and it is still hopeless.

So the correct finding is neither "the candidate is better" nor "the candidate is no better". It is: **on the available data these two models cannot be distinguished, and the candidate's measured advantage of \$382k a year is smaller than the error on the measurement.** Changing that needs either a statistic with more power, such as the cross-sectional information coefficient computed name by name rather than the portfolio P&L, or a live parallel run long enough to matter. That diagnostic ladder, and the arithmetic of how little live data can resolve, is the subject of [when live stops matching the backtest](/blog/trading/math-for-quants/live-vs-backtest-divergence-math-for-quants).

Writing that down is not a failure of the validation. It is the validation. A report that says "approved, the candidate outperforms" on ${t = 0.32}$ has told the committee something false.

## Outcomes analysis, and the ceiling a search puts on a backtest

Outcomes analysis compares what the model predicted against what happened. For a point forecast that means backtesting the forecast itself rather than the P&L. For a model that produces a **distribution**, which is what a risk model does, it is a bias test: if the model says 1% of days should breach the 99% loss level, count the breaches. That version, and the tests that catch a risk model systematically understating a factor, belong to [building a factor risk model you can defend](/blog/trading/math-for-quants/factor-risk-model-build-math-for-quants).

The reviewer's own contribution here is usually not to run the backtest again. It is to ask **how many were run before this one**.

This is the deepest published result in the area, from Bailey, Borwein, Lopez de Prado and Zhu. If a researcher tries ${N}$ configurations whose true Sharpe is zero, the expected **best** in-sample Sharpe is not zero, it is

$$E\!\left[\max_N \mathrm{SR}\right] \approx \frac{1}{\sqrt{y}}\left[(1-\gamma)\,\Phi^{-1}\!\left(1 - \tfrac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1 - \tfrac{1}{N e}\right)\right]$$

where ${y}$ is the backtest length in years, ${\Phi^{-1}}$ is the inverse standard normal, and ${\gamma \approx 0.5772}$ is the Euler-Mascheroni constant.

#### Worked example 2: the noise ceiling on a three-year backtest

Suppose the researcher tried 200 configurations, which is a modest number for a feature search. The bracketed term evaluates to **2.766**. Over a three-year backtest that is a noise ceiling of

$$\frac{2.766}{\sqrt{3}} = 1.60$$

An in-sample Sharpe of **1.60 is what you expect from pure noise** after 200 tries on three years of data. The researcher reported 1.8. The standard error of a Sharpe estimate is roughly ${\sqrt{(1 + \mathrm{SR}^2/2)/T}}$, which at ${\mathrm{SR} = 1.8}$ over three years is **0.93**. So 1.8 sits about a fifth of a standard error above the level noise alone would have produced.

Be fair about the objection: the 200 configurations are not independent, so the effective ${N}$ is smaller. Grant an aggressive discount and call it 50 independent tries. The bracketed term falls to 2.276 and the ceiling to ${2.276/\sqrt{3} = 1.31}$, still within one standard error of the reported number.

Two things follow, and the second is the useful one. The reported Sharpe is not evidence at the strength it appears to have, which is what the **deflated Sharpe ratio** formalises by discounting the observed statistic for the number of trials, the skew and the kurtosis. And you can run the formula backwards: to justify a Sharpe threshold of 1.0 after 200 trials you need ${2.766^2 = 7.65}$ years of backtest. That **minimum backtest length** is a number the reviewer can hand the researcher before the next project starts rather than after.

The intuition: a backtest is not a measurement of a strategy, it is the maximum of a search, and the maximum of a search has a distribution even when nothing works.

## Sensitivity: which input actually owns the model risk

Sensitivity analysis is often run as decoration, moving each input by an arbitrary 10% and tabulating the result. That tells you about the function, not about the model risk. The right perturbation is **one standard error of that input's own estimate**, because that is the amount by which the input actually might be wrong. Then rank the output moves. The top of that ranking is where the model risk lives, and it is usually not where the effort went.

#### Worked example 3: ranking the inputs on the same \$500m book

The decision is whether to allocate, so the output is expected annual net P&L. Central case: gross alpha of ${1.2 \times 8\% = 9.6\%}$ on \$500m, which is **\$48.0m**, less **\$2.97m** of trading cost (5.4 turns of \$500m at 11.0 bps) and **\$1.375m** of stock borrow (\$250m short at 55 bps), for ${\$48.0\text{m} - \$2.97\text{m} - \$1.375\text{m} = \$43.655\text{m}}$.

Now perturb each input by one standard error of its own estimate.

| Input | How it was estimated | One standard error | Swing in annual net P&L |
| --- | --- | --- | --- |
| Alpha forecast (mean IC) | 36 monthly ICs, mean 0.038, SD 0.09 | ${0.09/\sqrt{36} = 0.015}$, which is 39.5% of the mean | **\$18.9m** |
| Impact coefficient | fitted on the desk's own fills | 22% of the coefficient | **\$0.65m** |
| Stock borrow rate | broker quotes over the sample | 15 bps | **\$0.38m** |

![Tornado chart ranking the swing in annual net P&L: alpha forecast plus or minus \$18.9m, impact coefficient plus or minus \$0.65m, borrow rate plus or minus \$0.38m](/imgs/blogs/model-validation-governance-math-for-quants-3.webp)

The alpha input moves the answer **29 times** further than the next one (${\$18.9\text{m} / \$0.65\text{m} = 29.0}$). Carry it through and the plausible range of the decision is

In millions of dollars a year:

$$48.0 \times (1 \pm 0.395) - 4.345 \;\Rightarrow\; 24.7 \ \text{to} \ 62.6$$

a **2.53-fold spread** on the same book, driven entirely by an estimate the model treats as an input.

Here is the part the reviewer writes down. The desk spent the previous quarter refining the impact model. That work was worth, at the absolute limit of its own uncertainty, \$650k of resolution. The alpha forecast, whose error bar is \$18.9m wide, arrived as a given. The finding is not that the impact model is wrong. It is that the whole decision is a bet on an information coefficient with a 39.5% standard error, and should be sized so it survives the bottom of that range.

The intuition: sensitivity does not tell you whether the model is right. It tells you which number, if wrong, costs the most, and that is where the review time belongs.

## Limitations and use restrictions: validated **for what**

A validated model is validated for a stated use, on a stated input range, under stated conditions. The commonest expensive failure in the discipline is an entirely correct model used outside the regime it was tested on. Nothing is broken, nobody notices, and the number is simply wrong in a direction that costs money.

#### Worked example 4: an impact model outside its calibration range

A market-impact model estimates the cost of executing an order as

$$\mathcal{C} = c\,\sigma\sqrt{Q/V}$$

with ${c}$ a fitted constant, ${\sigma}$ the stock's daily volatility, ${Q}$ the order size and ${V}$ the average daily volume. Take ${c = 0.5}$ and ${\sigma = 2.0\%}$, and note the document's calibration range: parent orders of **0.5% to 3% of average daily volume**, which is what the desk's fill history contained.

At the top of that range, 3% of ADV, the model says ${0.5 \times 2.0\% \times \sqrt{0.03} = 17.32}$ bps. That number is trustworthy, because it sits inside the data that produced it.

A new strategy trades **12% of ADV** in its least liquid decile. Four times the participation, so the square root says cost rises by ${\sqrt{4} = 2}$: ${17.32 \times 2 = 34.64}$ bps.

But nothing in the fill history supports the square root out there, and the published estimates do not either. Almgren and co-authors measured an exponent nearer 0.6 than 0.5 on US equities, and the literature is fitted on ordinary participation rates. The honest alternative is that above roughly 5% of ADV, where you are a visible fraction of the day's flow, cost moves closer to **linear in participation**. Anchored on the same trustworthy 3% point, linear gives ${17.32 \times 4 = 69.28}$ bps.

$$69.28\ \text{bps} - 34.64\ \text{bps} = 34.64\ \text{bps}$$

![XY chart of execution cost against participation, showing the square-root extrapolation reaching 34.64 bps at 12% of ADV while a linear arm anchored at the calibrated 3% point reaches 69.28 bps, a 34.64 bps gap worth \$6.24m a year](/imgs/blogs/model-validation-governance-math-for-quants-4.webp)

On a \$1.8bn annual program in that bucket, the gap is ${0.003464 \times \$1{,}800\text{m} = \$6.24\text{m}}$ a year of cost the plan never budgeted. It is invisible in every backtest, because the backtest used the model.

The finding is not "the impact model is wrong". That figure makes the point: inside the calibrated range the model is fine, and rewriting it would be wasted work. The finding is **"there is no control preventing this model from pricing an order outside the range it was fitted on"**, severity high, remediation a hard bound in the optimiser plus an alert when participation exceeds 5%. A two-day fix worth \$6.24m a year, and the kind of thing only an independent reviewer finds, because the researcher has no reason to look at the edge of their own calibration set.

## The governance layer, without the cynicism

The paperwork exists for a reason that has nothing to do with regulators.

**Tiering by materiality.** Every model gets a tier based on what it decides and how much money rides on it. Tier 1 gets the full treatment and annual re-validation; tier 3 gets a documented sanity check. Without tiering, review effort distributes itself by how *interesting* each model is, which is uncorrelated with how much it can cost.

**Findings with severity.** A finding is a written statement of a specific weakness, rated, with a named owner and a date. The rating is what stops the conversation being about whose model it is. High means the model cannot be used for the proposed purpose until it is fixed. Medium means it can, with a compensating control such as a size cap. Low means fix it next cycle.

**Remediation tracking**, the part people find bureaucratic and the part that actually works. The impact-model finding above is worth \$6.24m a year only if somebody builds the bound. Six months later the researcher has moved teams and the reviewer is on another model, and the only thing that remembers is the tracker.

**A use statement.** The document ends with what the model is approved for and what it is not: instruments, size range, market conditions. That sentence makes the next reviewer's job possible, and it makes the failure in worked example 4 a control breach rather than an unlucky surprise.

SR 11-7 is the standard framing for this in the US, and the Bank of England's SS1/23 is the more recent UK counterpart. Neither is written for hedge funds. Both are the best available checklist for "what did we forget", which a research organisation faces whether or not anyone is examining it.

## Common misconceptions

**"Validation means reproducing the researcher's results."** Reproduction confirms the code is deterministic. It cannot find a shared assumption, and shared assumptions are where model risk lives. Rebuild from the specification, and treat a mismatch as information rather than an error to be reconciled away.

**"A more sophisticated model needs a more sophisticated validation."** The opposite. The more sophisticated the model, the harder the review should lean on the simple benchmark, because complexity's whole claim is that it beats simplicity and that claim is directly testable.

**"The backtest is out of sample, so overfitting is handled."** Only if the out-of-sample period was used once. If the researcher looked, adjusted, and looked again, it is in-sample with extra steps, and the noise ceiling in worked example 2 applies to the whole search.

**"Sensitivity analysis means moving everything 10%."** A 10% move on a precisely estimated input and on a barely estimated one are not comparable. Perturb by each input's own standard error or the ranking means nothing.

**"A validated model is safe."** It is safe **for the stated use**. The most expensive validated-model failures involve no modelling error at all, only a correct model asked a question it was never tested on.

## Sources and further reading

- Board of Governors of the Federal Reserve System, **SR 11-7, "Guidance on Model Risk Management"**, 4 April 2011, issued jointly with OCC Bulletin 2011-12. Source of the three-pillar framing, "effective challenge", and the input / processing / reporting decomposition.
- Prudential Regulation Authority (Bank of England), **SS1/23, "Model risk management principles for banks"**, May 2023. The more recent counterpart, with explicit tiering and findings management.
- D. H. Bailey, J. M. Borwein, M. Lopez de Prado and Q. J. Zhu, **"Pseudo-Mathematics and Financial Charlatanism: The Effects of Backtest Overfitting on Out-of-Sample Performance"**, Notices of the AMS 61(5), 458-471, 2014. The expected-maximum-Sharpe result and minimum backtest length.
- D. H. Bailey and M. Lopez de Prado, **"The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting, and Non-Normality"**, Journal of Portfolio Management 40(5), 94-107, 2014.
- M. Lopez de Prado, **Advances in Financial Machine Learning**, Wiley, 2018, chapters 11 and 12.
- V. K. Chopra and W. T. Ziemba, **"The Effect of Errors in Means, Variances, and Covariances on Optimal Portfolio Choice"**, Journal of Portfolio Management 19(2), 6-11, 1993. Why the expected-return vector dominates the sensitivity ranking.
- R. Almgren, C. Thum, E. Hauptmann and H. Li, **"Direct Estimation of Equity Market Impact"**, Risk, July 2005. The measured impact exponent and the participation ranges it was fitted on.

The dollar figures in the worked examples are illustrative arithmetic on assumed inputs, sized to be realistic for a liquid equity market-neutral book. They are not measurements of any real strategy.

## In the interview room and on the desk

The question arrives as "how would you validate this model?", usually attached to a one-paragraph description of something a researcher on the desk built. It is asked at Citadel and Two Sigma, at every bank-affiliated seat, and it is the entire job in any model-risk function.

The weak answer lists tests. Backtest, cross-validation, sensitivity, stress. It sounds thorough and it is empty: a list has no ordering and therefore no judgment in it, and the interviewer cannot tell whether you would find anything.

The strong answer starts one step earlier, with the adversarial question: **what would have to be true for this model to be wrong, and has anyone checked?** Then it moves in order. State the mechanism in one sentence, because most failures are visible there and never reach the maths. Rebuild the headline number from the specification rather than the repository, because a shared bug reproduces perfectly. Then, before anything sophisticated, **run the simple benchmark**: if three factors do 94.7% of what 180 features do, the complexity has to earn the remaining sliver against its own trading and upkeep costs, and it often does not. Only then outcomes analysis, with the number of configurations searched treated as an input to what the reported Sharpe is worth. Then sensitivity, perturbing each input by its own standard error so the ranking means something. And finish where the money is: what is this model approved for, and what stops it being used outside that.

Two things separate a senior answer. One is willingness to deliver a null result: "these two models cannot be distinguished on three years of data, here is the t-statistic, here is what would be needed". The other is that a senior expects to be on both sides of this, defending their own models on Tuesday and reviewing someone else's on Thursday, so the tone is never prosecutorial. The useful finding is usually a missing control, not a wrong equation.

The trap that makes a candidate look rigorous while being wrong is treating validation as reproduction. Cloning the repo, re-running the pipeline, and reporting agreement to four decimals is a lot of visible work that tests nothing the researcher had not already tested. Say out loud that you would not do it, and say why.

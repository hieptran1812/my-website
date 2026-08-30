---
title: "Causal inference for alpha research: what a regression coefficient does not mean"
date: "2026-08-30"
description: "A build-from-zero guide to confounders, colliders, mediators, DAGs and the back-door criterion, and why a backtest measures association while a tradeable edge is a claim about mechanism."
tags: ["causal-inference", "confounding", "collider-bias", "survivorship-bias", "directed-acyclic-graph", "back-door-criterion", "do-operator", "alpha-research", "regression", "quant-research", "math-for-quants", "quant-finance"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 22
---

> [!important]
> **TL;DR:** A backtest measures association. A tradeable edge is a claim about mechanism, and the tools that tell those two apart are causal, not statistical.
>
> - A regression coefficient is the sum of *every* open path between your signal and returns, not only the one you meant to measure. Closing the right paths is what turns it into an estimate of an edge.
> - Three graph shapes decide everything. A **fork** (a confounder) must be controlled for, a **chain** (a mediator) must not be, and a **collider** must never be touched. The same act has opposite consequences in each.
> - Survivorship bias is not a data-cleaning annoyance, it is collider conditioning. In the worked example below, deleting the funds that died manufactures a correlation of ${-0.50}$ between skill and luck inside a universe where they were independent by construction.
> - More controls is not closer to the truth. Pearl's back-door criterion names the specific set that closes the leaks, and one wrong extra control makes the answer worse while making the regression table look more rigorous.
> - The number to remember: an information coefficient of 0.03 can be entirely back door, because a signal correlated 0.60 with a liquidity factor that is itself correlated 0.05 with returns produces exactly ${0.60 \times 0.05 = 0.03}$ on its own.

## Introduction

Here is a conversation that happens in every research meeting. A junior researcher shows a signal. The backtest is clean, the information coefficient is positive, the Sharpe ratio survives transaction costs, and the out-of-sample window was held back honestly. The portfolio manager listens, nods, and asks one question: *why does this work?*

That question is not a formality, and it is not the PM being difficult. It is the only question that distinguishes a signal from an edge. A backtest tells you that the signal and the returns moved together in the past. It cannot tell you whether trading on the signal *makes* the returns happen. Those are different mathematical objects, and the difference is not a subtlety. It is the reason alphas decay, the reason a strategy that looked bulletproof for eight years dies in six weeks, and the reason the researcher who can name the mechanism gets to run capital.

![A signal X, a next-month return Y, and a liquidity factor L. On the left, conditioning on X leaves the back-door path through L open, so the coefficient equals the real edge plus the back-door leak. On the right, intervening on X deletes the arrows into X, the back door closes, and the coefficient equals the real edge only.](/imgs/blogs/causal-inference-alpha-research-math-for-quants-1.webp)

The figure above is the mental model for the whole post. Your signal `X` and next-month return `Y` are connected by more than one route. There is the **front door**, the path you believe in, the actual mechanism by which the signal earns money. And there is the **back door**, a path that runs backwards out of `X`, through some common cause such as a liquidity factor `L`, and forward into `Y`. Your regression coefficient does not know the difference. It adds them up. What a trade earns is only the front door, and the gap between the two is where alpha goes to die.

This post builds the machinery to see that gap. No prior exposure to causal inference is assumed. By the end you will be able to draw the graph for your own signal, name which variables belong in the regression, name which ones would quietly destroy it, and answer the PM's question with something better than "the backtest says so."

## Foundations: three different questions a number can answer

Start with vocabulary, defined from zero.

A **variable** here is anything you can measure per stock per day: a signal value, a return, a volatility, a sector label. An **association** between two variables means they tend to take high and low values together. Correlation is one way to measure association; a regression coefficient is another.

A **regression** fits a line. When you regress next-month return `Y` on signal `X`, you are choosing the number ${\beta}$ that makes ${\hat{Y} = \alpha + \beta X}$ fit best in a least-squares sense. Almost everything in quant research is a regression wearing a costume, so it is worth being precise about what that ${\beta}$ contains. If you want the mechanics rather than the interpretation, the companion post on [OLS, GLS and regularized regression](/blog/trading/math-for-quants/regression-ols-gls-regularized-math-for-quants) covers the estimation side.

**Conditioning** on a variable means restricting attention to observations that share its value: looking only at high-volatility days, or including the variable as a control in the regression so the coefficient is read "holding that variable fixed." Statisticians write this ${P(Y \mid X = x)}$, read as "the distribution of `Y` among the observations where `X` happened to equal `x`."

**Intervening** is different. It means reaching into the world and setting `X` to a value, rather than watching where it landed. Judea Pearl gave this its own notation, the do-operator: ${P(Y \mid \mathrm{do}(X = x))}$, read as "the distribution of `Y` if I *set* `X` to `x`." Trading is intervention. Backtesting is conditioning. The central fact of this post is that

$$
P(Y \mid X = x) \;\neq\; P(Y \mid \mathrm{do}(X = x))
$$

in general, and the whole discipline exists to say when they are equal and how to bridge them when they are not.

Pearl organizes this into three rungs of a ladder, in *Causality* (2nd edition, 2009) and more informally in *The Book of Why* (Pearl and Mackenzie, 2018):

| Rung | The question it answers | Notation | What can answer it |
| --- | --- | --- | --- |
| 1. Association | What do I see happening together? | ${P(Y \mid X)}$ | A backtest, a correlation, an unadjusted regression |
| 2. Intervention | What happens if I act? | ${P(Y \mid \mathrm{do}(X))}$ | A genuine experiment, or a graph plus the back-door criterion |
| 3. Counterfactual | What would have happened instead? | ${P(Y_x \mid X = x', Y = y')}$ | A full structural model of the system |

Data alone lives on rung 1. No amount of it climbs to rung 2 by itself. What lets you climb is an assumption about structure, and the standard way to write that assumption down is a **directed acyclic graph**, or DAG: a picture with one node per variable, an arrow from cause to effect, and no loops. The DAG is not estimated from the data. It is what you believe about the world, stated clearly enough to be argued with. That is a feature. A researcher who cannot draw the DAG for their own alpha does not yet know what they are claiming.

#### Worked example: does cutting risk cost money?

A fund records 1,000 trading days. On each day the risk desk either cut gross exposure or did not, and the day's P&L is recorded. The raw numbers, all hypothetical:

- 200 of the days were high-volatility days. Risk was cut on 160 of them, mean P&L minus \$40k; not cut on the other 40, mean P&L minus \$55k.
- 800 of the days were calm. Risk was cut on 40 of them, mean P&L plus \$20k; not cut on the other 760, mean P&L plus \$25k.

Every number below is in thousands of dollars. The naive comparison pools everything. Across all 200 cut days, mean P&L is ${(160 \times -40 + 40 \times 20)/200 = -5{,}600/200 = -28}$, so minus \$28k. Across all 800 not-cut days it is ${(40 \times -55 + 760 \times 25)/800 = 16{,}800/800 = +21}$, so plus \$21k. The difference is minus \$49k per day. Read literally: cutting risk costs the fund \$49,000 a day, so stop doing it.

That reading is wrong, and the graph says why. Volatility causes the risk desk to cut, and volatility causes bad P&L. It is a common cause of both, so a back-door path is open. Close it by comparing cut against not-cut *within* each volatility regime, then averaging with the regime's own frequency rather than the frequency it had among cut days. This is Pearl's adjustment formula:

$$
P(Y \mid \mathrm{do}(X = x)) = \sum_{z} P(Y \mid X = x,\, Z = z)\, P(Z = z)
$$

Applying it: ${0.2 \times (-40) + 0.8 \times (+20) = +8}$ for cutting, so plus \$8k, and ${0.2 \times (-55) + 0.8 \times (+25) = +9}$ for not cutting, so plus \$9k. The causal difference is minus \$1k per day, not minus \$49k. Cutting helps by \$15k on stormy days and costs \$5k on calm ones, and it nets out to roughly nothing.

**The intuition:** the same dataset supports a number that is 49 times too large, and the only thing separating the two answers is a claim about which variable caused which.

## Three structures, three opposite rules

Every DAG, however large, is built from three local shapes. Knowing what conditioning does in each is most of the working knowledge.

![Three graph structures side by side. A fork, where a confounder L causes both signal X and return Y, where conditioning on L blocks the spurious path. A chain, where signal X causes mediator M which causes return Y, where conditioning on M deletes the effect being measured. A collider, where signal X and return Y both cause C, where conditioning on C invents an association that was not there.](/imgs/blogs/causal-inference-alpha-research-math-for-quants-2.webp)

**The fork: a confounder.** `L` causes both `X` and `Y`, and there is no arrow between `X` and `Y` at all. They still move together, because they are both being pushed by the same hand. This is the classic spurious correlation, and conditioning on `L` **blocks** it. Confounders are the variables you were taught to control for, and controlling for them is the entire justification for the word "control."

**The chain: a mediator.** `X` causes `M`, and `M` causes `Y`. Here `M` is not a nuisance, it is the machinery. It is *how* the effect travels. Conditioning on `M` **deletes** the very effect you were trying to measure, because you have held fixed the channel through which the cause operates. A signal whose entire mechanism runs through institutional flow will show a coefficient of approximately zero the moment you add realized flow as a control, and a researcher who reads that as "the signal is fake" has thrown away a real alpha.

**The collider: a common effect.** `X` and `Y` both cause `C`, and they have no relationship with each other. Left alone, they are independent. Condition on `C` and they become dependent: the conditioning **invents** an association that was not there. This is the least intuitive of the three and by far the most dangerous, because the conditioning is usually invisible. Nobody writes `+ C` in the regression. Instead the sample was already filtered on `C` before the researcher ever saw it.

The shapes look almost identical on paper. Only the arrow directions differ, and the rules they imply are not merely different but opposite. A variable is not a confounder because of what it is called or what asset class it belongs to. It is a confounder because of where it sits relative to `X` and `Y` in a graph you had to draw yourself.

## The back-door criterion, in plain words

Pearl's back-door criterion turns the three shapes into a decision procedure. A path from `X` to `Y` is a **back-door path** if it starts with an arrow pointing *into* `X`. Those are the paths that carry information from `X`'s causes rather than to `Y`'s effects, and they are exactly the ones that leak into the coefficient without being edge.

A set of variables `Z` satisfies the back-door criterion for the pair `(X, Y)` when two conditions hold:

1. No variable in `Z` is a descendant of `X`, meaning nothing in `Z` is caused by `X` directly or indirectly.
2. `Z` blocks every back-door path from `X` to `Y`.

When such a `Z` exists and is measured, the adjustment formula above converts an observed association into a causal effect, and only then does the regression coefficient mean what the researcher wanted it to mean.

Two practical consequences follow. First, the criterion identifies a *set*, not a list of everything available. Adding variables beyond the sufficient set is not additional rigor. Second, the criterion is checkable only against a graph, which means the graph has to exist before the regression is run. In practice this is the discipline the criterion imposes: write down what you think causes your signal, before you look at what your signal predicts. It costs twenty minutes and it is the cheapest research improvement available.

## Good control, bad control

The reflex "control for more things to be safe" is the single most common way a careful researcher produces a worse number. Cinelli, Forney and Pearl catalogued this in "A Crash Course in Good and Bad Controls" (2022), and the taxonomy is worth memorizing.

![A table of control types. A confounder should be added and removes back-door bias. A cause of the return only should be added and shrinks standard errors. A cause of the signal only should be added only if needed, since it grows standard errors and amplifies remaining bias. A mediator should not be added because it strips the indirect effect. A collider should not be added because it creates an association from nothing. A descendant of the return should not be added because it leaks the answer back in.](/imgs/blogs/causal-inference-alpha-research-math-for-quants-3.webp)

Reading down the table: a **confounder** is the good case, and adding it removes the back-door bias. A variable that causes the **return only**, such as a well-established risk factor that your signal has nothing to do with, leaves the bias unchanged and shrinks the standard errors. That is free precision and you should take it. A variable that causes the **signal only** also leaves the bias unchanged, but it inflates the standard errors, and if any bias remains in the specification it amplifies it. Add that one only with a reason.

The bottom three are the traps. A **mediator** strips out the indirect effect and reports an edge smaller than the one you actually own. A **collider** fabricates an association out of nothing. A **descendant of the outcome**, meaning anything measured after the return that the return influenced, leaks the answer back into the right-hand side and produces a coefficient that is meaningless in a way no diagnostic will flag.

#### Worked example: controlling away your own alpha

A signal scores analyst estimate revisions. Its mechanism, as the researcher believes it, is that revisions attract institutional buying and that buying moves the price. Regressing next-day return on the signal alone gives a coefficient of 1.2 basis points per unit of signal. The researcher then adds realized institutional flow as a control, "to make sure the signal is not just a flow proxy." The coefficient falls to 0.4 basis points.

Flow is the mediator. It is the channel, not a confounder. The 1.2 bps is the total effect, the 0.4 bps is the direct effect that does not run through flow, and the 0.8 bps difference is the part of the mechanism the researcher just deleted from the estimate. Sizing the book on 0.4 instead of 1.2 means running the strategy at a third of the size the edge supports.

**The intuition:** a coefficient that collapses when you add a control is evidence about the graph, not automatically evidence against the signal.

## Worked example: a signal that is really a liquidity bet

This is the confounding case with numbers attached, and it is the most common way a backtest lies.

![The causal graph and the numbers for a hypothetical liquidity-confounded signal. A liquidity factor L correlates 0.60 with signal X and 0.05 with next-month return Y. The raw information coefficient between X and Y is 0.03, which is exactly the product, and the partial correlation given L is 0.00.](/imgs/blogs/causal-inference-alpha-research-math-for-quants-4.webp)

#### Worked example: the whole coefficient is the back door

The signal `X` is a five-day reversal score scaled by volume, computed across a universe of 500 stocks. The measured information coefficient, meaning the cross-sectional correlation between the signal and next-month return `Y`, is 0.03. That is a perfectly respectable number. Under the textbook fundamental law of active management (Grinold, 1989), an information ratio of roughly ${\mathrm{IC} \times \sqrt{\mathrm{breadth}}}$ with 500 names rebalanced monthly gives ${0.03 \times \sqrt{6{,}000} \approx 2.3}$, which is why anyone bothers. The law is an idealized upper bound and real implementations land far below it, but the point stands: 0.03 is worth chasing.

Now measure two more correlations. The signal correlates 0.60 with a liquidity factor `L`, which is unsurprising because it is volume-scaled. And the liquidity factor itself correlates 0.05 with next-month returns, which is the long-documented finding that illiquidity carries a premium (Amihud and Mendelson, 1986; Amihud, 2002). All three correlations here are hypothetical, chosen to make the arithmetic legible.

Multiply: ${0.60 \times 0.05 = 0.03}$. The back-door path alone reproduces the entire measured information coefficient.

To confirm it, compute the partial correlation of `X` and `Y` given `L`:

$$
\rho_{XY \cdot L} = \frac{\rho_{XY} - \rho_{XL}\,\rho_{YL}}{\sqrt{(1 - \rho_{XL}^2)(1 - \rho_{YL}^2)}}
$$

The numerator is ${0.03 - (0.60)(0.05) = 0.03 - 0.03 = 0}$. The denominator is ${\sqrt{(1 - 0.36)(1 - 0.0025)} = \sqrt{0.64 \times 0.9975} \approx 0.799}$, which is comfortably nonzero, so the partial correlation is 0.00. Holding liquidity fixed, the signal predicts nothing.

**The intuition:** the signal was never wrong, it was just redundant. It is a liquidity factor with extra steps, and it will be priced, crowded and charged for as one. If your risk model already carries the liquidity factor, this alpha contributes zero incremental information and pays full transaction costs for the privilege. The [eigendecomposition and PCA of a return covariance matrix](/blog/trading/math-for-quants/eigendecomposition-pca-returns-math-for-quants) is the standard machinery for finding which factors your signal is secretly loading on.

## Worked example: survivorship is collider conditioning

Every quant knows survivorship bias as a rule: use a point-in-time database, include the delisted names. Fewer know *why* it distorts things, and the why is what lets you spot the cases where no database vendor will save you.

![The collider structure of survivorship. Skill S and luck R both cause whether a fund still exists in 2026, labelled C. In the full universe of 1,000 funds the four skill-by-luck cells hold 250 each, and good luck is equally likely at both skill levels. Among the 750 survivors the low-skill bad-luck cell is empty, so good luck is 50 percent likely for high-skill funds and 100 percent likely for low-skill funds.](/imgs/blogs/causal-inference-alpha-research-math-for-quants-5.webp)

#### Worked example: manufacturing a correlation of ${-0.50}$

Take a hypothetical universe of 1,000 hedge funds launched at the same time. Classify each on two independent axes: skill `S`, either high or low, and luck `R`, either good or bad. By construction they are independent, so the four cells hold 250 funds each.

Now let survival depend on both. A fund closes only if it is both low-skill and unlucky. That is the whole model, and it is generous to the industry. The low-skill, bad-luck cell empties out. The database in 2026 contains 750 funds: 250 high-skill and lucky, 250 high-skill and unlucky, 250 low-skill and lucky, and 0 low-skill and unlucky.

Compute the conditional probabilities inside the surviving sample. Among the 500 high-skill survivors, 250 were lucky, which is 50%. Among the 250 low-skill survivors, all 250 were lucky, which is 100%. Skill and luck are now strongly related in the data. Quantify it with the phi coefficient, which is the correlation between two binary variables:

$$
\phi = \frac{ad - bc}{\sqrt{(a+b)(c+d)(a+c)(b+d)}}
$$

With ${a = 250}$ high-skill lucky, ${b = 250}$ high-skill unlucky, ${c = 250}$ low-skill lucky and ${d = 0}$, the numerator is ${(250)(0) - (250)(250) = -62{,}500}$ and the denominator is ${\sqrt{500 \times 250 \times 500 \times 250} = 125{,}000}$. So ${\phi = -0.50}$.

A correlation of ${-0.50}$ between skill and luck, in a universe where they were independent by construction, produced by nothing except deleting the rows where the fund had died. Survival is a **collider**: skill causes it and luck causes it, and looking only at survivors is conditioning on it.

**The intuition:** the bias is not in the data, it is in the act of selecting. Point-in-time databases fix the version of this problem you can see. The versions you cannot see are everywhere: a signal library containing only the alphas that passed review, a backtest window starting the year the strategy was built, an ADV filter that quietly selects on the same volume the signal uses. Every filter is a candidate collider. Empirical work on this in mutual funds put the survivorship inflation of measured average returns at roughly 0.5 to 1.5 percentage points a year depending on sample and period (Malkiel, 1995; Elton, Gruber and Blake, 1996), which is larger than most reported alphas.

## Common misconceptions

**"Out-of-sample performance proves the effect is causal."** It does not. Out-of-sample testing controls for overfitting, which is a different failure. If a confounder was present in-sample it is present out-of-sample too, because it is a feature of the world, not of the fitting procedure. A liquidity-driven signal will validate beautifully on held-out data right up until the liquidity premium changes sign. Out-of-sample validation and causal identification are orthogonal checks and you need both. The post on [purged cross-validation and the deflated Sharpe ratio](/blog/trading/quantitative-finance/overfitting-purged-cv-deflated-sharpe-quant-research) covers the other half properly.

**"More controls means a cleaner estimate."** Only for confounders. For mediators it destroys the estimate, for colliders it fabricates one, and for descendants of the return it makes the coefficient meaningless. A regression with 40 controls is not more careful than one with 4; it is simply 40 opportunities to have opened a path instead of closing one. There is no such thing as a control that is safe by default.

**"Granger causality is causality."** Granger's test (1969) asks whether past values of one series improve the forecast of another beyond that series' own past. It is a statement about predictive precedence in observed data, not about intervention. Two series driven by a common factor with different lags will pass it. The name is unfortunate and Granger himself was careful about the distinction; the practitioners quoting it usually are not.

**"A high t-statistic means the relationship is real."** The t-statistic measures how confident you are that the coefficient is not zero. It says nothing about what the coefficient *contains*. A back-door path estimated on 500 stocks over 20 years will produce a very confident estimate of a leak. Precision and identification are separate properties, and adding data buys you the first while doing nothing for the second. On the precision half, see [estimators, bias, variance and consistency](/blog/trading/math-for-quants/estimators-bias-variance-consistency-math-for-quants).

**"If the DAG is just my assumption, this is all subjective."** The DAG is an assumption, and that is the honest part. The alternative is not assumption-free, it is assumption-hidden: running a regression with a chosen control set already commits you to a graph, silently. Writing it down makes the commitment visible and therefore arguable. A colleague can point at an arrow and say they think it runs the other way, which is a productive conversation. Nobody can argue with an unstated belief.

## The checklist before you defend an alpha

Six questions, in order. If any answer is missing, the alpha is not ready.

1. **Draw the DAG.** One node per variable in the regression, plus every unmeasured common cause you can name. If you cannot draw it, you do not know what you are claiming.
2. **Name the mechanism.** In one sentence, without the words "the model finds": who is on the other side of the trade, and why do they keep taking it?
3. **List the back-door paths.** For each, either close it with a control or state why you believe it is small.
4. **Check every control for its role.** Confounder, mediator, collider or descendant of the outcome. Any variable measured after the return is guilty until proven otherwise.
5. **Find the hidden conditioning.** Every filter applied to the universe is a potential collider: survivorship, liquidity screens, data availability, and the review process that decided which signals got backtested at all.
6. **State the falsifier.** What observable would tell you the mechanism has stopped? Write it down now, while you have no position, and set the alert.

## In the interview room and on the desk

The question arrives in almost these words: *your signal works in the backtest, why do you believe it will keep working?* The follow-up is *what would make you kill this alpha?* Both are asked constantly in the research rounds at Two Sigma and Citadel, and both are asked in every committee meeting where a researcher defends a live book. They are the same question, which is whether you understand the mechanism or only the fit.

A strong answer has three parts, in this order. First, **name the mechanism**: who is on the other side, what constraint or behavior makes them trade against you, and why that persists. "Index funds must trade at the close, so the liquidity demand is price-insensitive and predictable" is a mechanism. "The model finds a pattern in the residuals" is not. Second, **name the confounder you controlled for and the one you deliberately did not**, with the reason. Saying "I controlled for the liquidity factor because it causes both my signal and returns, and I did not control for realized flow because flow is how the signal works, so controlling for it would have deleted two thirds of the effect" tells the interviewer you can tell a fork from a chain. Third, **state the observable that would falsify the thesis**: the borrow cost that would have to normalize, the flow that would have to reverse, the spread that would have to compress. An alpha you cannot falsify is an alpha you cannot risk-manage.

The trap is reaching for more controls. Under pressure the instinct is to demonstrate rigor by adding variables, and a candidate who says "I also controlled for realized volatility, subsequent turnover and whether the name was still in the index at year end" has just described conditioning on a mediator, a descendant of the outcome and a collider, in one breath. The coefficient will look cleaner. The answer is worse, and the person across the table will notice, because the whole point of the question was to see whether you know that a control set is chosen from a graph rather than accumulated for safety. The candidate who removes a control and explains why beats the candidate who adds five.

Two Sigma weights this heaviest of the large shops, given how much of its research process is framed around statistical learning and inference. Citadel probes it in the portfolio-construction and risk conversation, where an alpha has to survive a committee that will ask what it is really loading on. Any seat where you defend a book to someone who controls its capital will ask a version of it eventually.

## Sources and further reading

The causal machinery in this post is standard and well documented. Every claim
about the do-operator, the three graph shapes, and the back-door criterion traces
to these:

- Judea Pearl, *Causality: Models, Reasoning, and Inference*, 2nd edition,
  Cambridge University Press, 2009. The do-operator, d-separation, and the
  back-door criterion in their original form.
- Judea Pearl and Dana Mackenzie, *The Book of Why*, Basic Books, 2018. The same
  ladder without the measure-theoretic overhead; the fastest way in.
- Carlos Cinelli, Andrew Forney and Judea Pearl, "A Crash Course in Good and Bad
  Controls", *Sociological Methods & Research*, 2022. The paper behind the
  bad-control section: it enumerates the control variables that help, the ones
  that do nothing, and the ones that actively create bias.
- Joshua Angrist and Jörn-Steffen Pischke, *Mostly Harmless Econometrics*,
  Princeton University Press, 2009. The applied counterpart, and the standard
  reference for the identification strategies this post points to next.

Every number in this post appears inside a clearly labelled hypothetical worked
example. They are illustrative arithmetic chosen so the bias is visible in one
step, not measurements of any real fund or signal.

## Where to take this next

Causal reasoning does not stop at the back-door criterion. When no sufficient control set is measurable, the next tool is an instrument: a variable that moves your signal but touches returns through no other path, which is how index reconstitutions, regulatory changes and exchange rule changes get used as natural experiments. After that come event studies done properly, difference-in-differences, and synthetic controls, all of which are ways of building a counterfactual for a market event.

The reading list above is the shortest path into all of it. On the finance side, the practical companion to this post is [evaluating alpha signals with IC, Sharpe and turnover](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research), which measures the thing this post asks you to justify.

This is educational material about research method, not investment advice.

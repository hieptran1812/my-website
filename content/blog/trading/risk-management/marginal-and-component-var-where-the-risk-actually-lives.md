---
title: "Marginal and Component VaR: Where the Risk Actually Lives"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Total portfolio risk hides where it comes from; this builds marginal and component VaR from zero so you can see which position is really carrying the book — often a small-capital one — and trim the right name first."
tags: ["risk-management", "value-at-risk", "marginal-var", "component-var", "risk-decomposition", "position-sizing", "portfolio-construction", "diversification", "risk-budgeting"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **The one idea:** a single portfolio VaR number tells you how big the risk is but says nothing about *where* it comes from — and when you decompose it, the position carrying the most risk is frequently not the one holding the most money.
> - **Marginal VaR** answers "how much does total portfolio risk change if I add one more dollar to this position?" It is the right number for deciding what *not* to add to.
> - **Component VaR** answers "what share of the total risk is this position responsible for?" — and the beautiful fact is that the component VaRs **sum exactly to total VaR**, with no leftover, so it is an honest carve-up.
> - The surprise that makes this worth learning: a position with **10% of your capital can be the #1 risk contributor**, while a position with **30% of your capital adds almost no risk at all** (a real diversifier). Dollars and risk are different currencies.
> - This is the diagnostic that makes VaR, Expected Shortfall, and risk parity *actionable at the position level*: it tells you which single name to trim first to cut the most risk per dollar sold.
> - It only helps you survive if you act on it — rank by component VaR, trim the top marginal-VaR position, and set risk-contribution limits, not just dollar limits.

A risk manager I once worked alongside had a habit that annoyed every trader on the desk. When someone proudly showed off a "diversified" book — twelve names, no single position over 15% of capital, looked textbook clean — she would ignore the dollar weights entirely and ask one question: "If this whole book loses money tomorrow, which position do you think did the damage?" The trader would point at the biggest holding. She would run the decomposition, turn the screen around, and show that the real culprit was usually some small, twitchy, high-volatility name buried near the bottom of the position list — a name with 8% of the capital and 40% of the risk. The big "core" holding everyone worried about was often a placid diversifier carrying barely a tenth of its dollar share of the risk.

That gap — between where the money is and where the risk is — is the single most common blind spot in portfolio risk, and it is not a matter of opinion. It is arithmetic. A portfolio's total risk is a single number (its volatility, or its Value at Risk), and that number genuinely tells you how much you can lose. But it is a sum, and like any sum it hides its terms. Two books with the *same* total VaR can have completely different internal structures: one where the risk is spread evenly across ten names, and one where a single position is secretly responsible for half of it. The first book is robust; the second is a concentrated bet wearing a diversified costume. You cannot tell them apart from the headline number. You have to open it up.

![Bar chart of component VaR by position on a ten-million-dollar book, with a ten-percent-of-capital concentrated position contributing the largest share of total risk at about fifty-one percent](/imgs/blogs/marginal-and-component-var-where-the-risk-actually-lives-1.png)

Figure 1 is the whole post in one chart, and it is the picture to burn into memory. It is a real three-position book — a large-cap index fund with 60% of the capital, a government-bond fund with 30%, and a small concentrated single name with just 10% — sliced not by dollars but by **component VaR**, each position's share of the total risk. The 10%-of-capital concentrated name is the *tallest bar*. It contributes about 51% of the entire book's risk on a tenth of the money, edging out the index fund that holds six times as much capital. The bond fund, with 30% of the money, contributes under 1% of the risk. This post builds, from absolutely nothing, the machinery that produces that chart: what "marginal VaR" and "component VaR" mean, how to compute them by hand, why the slices sum exactly to the total, and — most usefully — how to use the decomposition to trim the *right* position first. We will do the full dollar arithmetic on a \$100,000 retail account and a \$10,000,000 book, and land on a concrete playbook for risk-budgeting at the position level.

This is the survival spine of the whole series seen at higher resolution. The first job of a trader or allocator is not to make money, it is to *not blow up*, because you can only compound if you are still in the game. And the fastest way to blow up while believing you are prudent is to be concentrated without knowing it — to think your risk is spread when one position is quietly carrying half the book. The traders who survive 2008, 2020, and 2024 are not the ones with the lowest headline VaR. They are the ones who knew, position by position, *where their risk actually lived*, so that when they needed to cut, they cut the thing that mattered instead of selling their diversifiers and keeping their landmines.

## Foundations: the building blocks of risk decomposition

Before we can decompose anything, we have to define every term precisely. If you size portfolios for a living, skim. If you do not, this is the floor everything stands on. We will build up from the single-asset notion of risk to the portfolio level, and then to the decomposition that is the point of the post.

**Volatility (the risk of one asset).** The **volatility** of an asset is the standard deviation of its returns — a number for how much its return bounces around its average, usually quoted per year. A broad stock index runs around 15% annual volatility; a government-bond fund maybe 6%; a single small, leveraged, or momentum-driven stock can run 50% to 90% or more. We write it σ (sigma). Volatility is symmetric (it treats up and down the same) and it is *not* the whole story of risk — a separate post, [volatility and why it is not risk](/blog/trading/risk-management/volatility-and-why-it-is-not-risk), draws that line carefully. But for *decomposing a portfolio*, volatility is the right working unit, because it adds up in a clean way that we will exploit.

**Value at Risk (VaR).** **Value at Risk** is volatility translated into a dollar loss with a confidence level attached. A "1-day 95% VaR of \$161,543" means: on a normal day, there is about a 5% chance the book loses *more* than \$161,543, and a 95% chance it loses less (or gains). It is a single number that answers "how bad is a bad-but-not-catastrophic day?" Under the simplest (parametric, normal) model, VaR is just a multiple of the portfolio's volatility: you take the daily volatility, multiply by a confidence factor (1.645 for 95%, 2.326 for 99%), and multiply by the book size. VaR famously lies about the *tail* — it says nothing about how bad the worst 5% of days are — and a companion post, [value at risk and exactly how VaR lies](/blog/trading/risk-management/value-at-risk-and-exactly-how-var-lies), is dedicated to that flaw. Here we take VaR as given and ask a different, orthogonal question: not "how much is the total?" but "where does it come from?"

**Weight.** The **weight** of a position, written w, is the fraction of your capital allocated to it. A book with \$6,000,000 in a \$10,000,000 portfolio has a weight of 0.60 in that position. Weights normally sum to 1 if you are fully invested; with leverage they can sum to more.

**Correlation.** **Correlation**, written ρ (rho), measures how two positions move together, on a scale from −1 to +1. Plus one means perfect lockstep, zero means unrelated, minus one means exact opposites. Correlation is the hinge on which this whole post swings, because a position's contribution to *portfolio* risk depends not just on its own volatility but on how it co-moves with everything else you hold. A high-volatility name that moves *with* the rest of the book amplifies risk; the same name moving *against* the book can actually reduce it.

**Portfolio volatility.** When you combine positions, the portfolio's volatility is *not* the weighted average of the individual volatilities — that is the entire magic of diversification. For a book of many assets, it is a quadratic form built from the **covariance matrix** Σ (Sigma), the table of every variance and covariance among the positions:

$$\sigma_p = \sqrt{w^\top \Sigma\, w}$$

You do not need to love that notation. What matters is the intuition: combining imperfectly correlated assets produces a portfolio volatility *less* than the weighted average of the parts. When correlations rise toward 1, that benefit shrinks and the formula collapses toward a simple weighted average — which is the failure mode that haunts every crisis case study at the end of this post.

**Marginal VaR — the heart of the decomposition.** Here is the first new idea, and it is the whole point. Suppose you nudge your holding of one position up by a tiny amount — one more dollar. How much does the *whole portfolio's* VaR change? That sensitivity is the position's **marginal VaR**. Formally, it is the partial derivative of portfolio VaR with respect to that position's size. Working in volatility units first, the marginal contribution of position *i* is the *i*-th row of the covariance matrix times the weight vector, divided by portfolio volatility:

$$\text{MCR}_i = \frac{(\Sigma\, w)_i}{\sigma_p}$$

and marginal VaR is that, scaled by the same confidence-and-time factor that turns volatility into VaR:

$$\text{Marginal VaR}_i = \frac{z}{\sqrt{252}} \cdot \frac{(\Sigma\, w)_i}{\sigma_p}$$

where z is the confidence factor (1.645 for 95%) and the √252 converts annual volatility to daily. The key reading: marginal VaR is "the extra VaR added by the *next* dollar into this position." It depends on the *whole portfolio*, not just the position in isolation, because the (Σw) term mixes in how the position co-moves with everything else.

**Component VaR — the slice that sums to the total.** Multiply a position's marginal VaR by the number of dollars actually in it (equivalently, multiply the marginal contribution by the weight) and you get the position's **component VaR** — the actual share of total VaR that this position is responsible for:

$$\text{Component VaR}_i = w_i \times \text{Marginal VaR}_i \times (\text{book size})$$

The fact that makes this the *right* decomposition, and not just one of many arbitrary ways to split a number, is that the component VaRs **sum exactly to total VaR**:

$$\sum_i \text{Component VaR}_i = \text{Total VaR}$$

This is not an approximation. It is an exact identity, a consequence of Euler's theorem: VaR (like volatility) is a *homogeneous function of degree one* in the position sizes — double every position and VaR doubles exactly — and Euler's theorem says any such function equals the sum of (each input × the function's sensitivity to that input). The sensitivity is the marginal VaR; the input is the dollar size; their product summed across positions reconstructs the total with no residual. That clean additivity is why practitioners decompose VaR and volatility, *not* variance (volatility squared), which leaves an uncounted interaction term. If someone shows you a "risk decomposition" whose pieces do not sum to the total, they used the wrong measure. (This is the same Euler decomposition that powers [risk parity sizing](/blog/trading/risk-management/risk-parity-sizing-equal-risk-not-equal-money); component VaR is risk parity's diagnostic twin — risk parity *sets* the risk shares, component VaR *measures* them.)

**Incremental VaR (the cousin you should not confuse with marginal VaR).** One more term, because people mix them up constantly. *Marginal* VaR is the derivative — the risk added by an *infinitesimal* next dollar, the local slope. **Incremental VaR** is the *finite* change: how much total VaR moves if you add (or remove) a specific, sizable chunk of a position — say a whole \$500,000 trade. For a tiny change the two agree, but for a large one they diverge, because adding a big block changes the portfolio's own composition and therefore changes every correlation-weighted term. The trimming experiment later in this post is an *incremental* VaR calculation (we sell a full \$500,000 and recompute); the per-dollar ranking is a *marginal* VaR calculation. The practical rule: use marginal VaR to *rank* what to trim, then use incremental VaR to size the *actual* trade, because the marginal rate at the start of a big sale is not the rate at the end. Marginal VaR is the price of the first dollar; incremental VaR is the bill for the whole order.

With volatility, VaR, weight, correlation, portfolio volatility, marginal VaR, component VaR, and incremental VaR defined, you have the entire toolkit. Now let us use it on a real book and watch the surprise fall out.

## The mismatch: money is not risk

Take a concrete \$10,000,000 book with three positions, chosen to look reasonable to anyone eyeballing it:

- **Position A — a large-cap index fund.** \$6,000,000, so **60% of capital**. Volatility σ = 14%. The "core" holding.
- **Position B — a government-bond fund.** \$3,000,000, **30% of capital**. Volatility σ = 6%. The "ballast."
- **Position C — a small, concentrated, high-flying single name** (think a leveraged momentum stock or a single-name swap). \$1,000,000, **10% of capital**. Volatility σ = 85%. The "small side bet."

And three honest correlation inputs: A and C move together strongly (ρ = 0.70 — the small name is a high-beta version of the market the index tracks); A and B are mildly negative (ρ = −0.20 — bonds hedge stocks in calm times); B and C are near zero (ρ = 0.10).

By dollar weight this book looks dominated by A: it is six times the size of C. Almost every trader, asked "where's your risk?", would point at the \$6,000,000 index position. Let us see whether the arithmetic agrees.

These inputs are not cherry-picked to manufacture a paradox; they describe a very ordinary book. The "core index plus a little bond ballast plus a spicy satellite" shape is how a huge fraction of real portfolios are actually built — a long-only equity base, some fixed income for diversification, and a small high-conviction position for upside. The volatilities are realistic (a broad index near 14%, a bond fund near 6%, a single concentrated or leveraged name easily 50–90%), and the correlations are the textbook ones (the satellite is a high-beta version of the market, the bond is a mild hedge). The surprise that falls out of these honest, unremarkable inputs is precisely the point: you do not need a contrived portfolio for the risk to live somewhere other than where the money is. You need an *ordinary* one.

![Side by side bar chart comparing each position's share of capital against its share of portfolio risk, showing the bond at thirty percent of capital but near zero risk and the concentrated name at ten percent of capital but the largest risk share](/imgs/blogs/marginal-and-component-var-where-the-risk-actually-lives-2.png)

Figure 2 is the headline of the whole post, and it puts the two currencies side by side. For each position, the left bar is its share of the *capital* and the right bar is its share of the *risk* (its component VaR). If money and risk were the same thing, every pair of bars would match. They do not, and the mismatches are dramatic. Position A holds 60% of the money but produces about 48% of the risk — close-ish, slightly *less* risk than its money share because its diversification with the bond pulls it down. Position B holds 30% of the money — a lot of capital — but produces essentially *zero* risk: under 1%. It is a genuine diversifier, soaking up capital and contributing almost nothing to the day-to-day P&L swings. And position C, the 10%-of-capital "side bet," produces about *51% of the total risk* — more than the position six times its size. The money says A is the big bet. The risk says C is. The decomposition is how you find that out before the market finds it out for you.

#### Worked example: portfolio VaR for the three-asset book

Let us compute the total first, so the components have something to sum to. We run the \$10,000,000 book at a **1-day 95% VaR**, so z = 1.645 and we convert annual volatility to daily by dividing by √252 ≈ 15.87.

**Step 1 — build the covariance terms.** With σ_A = 0.14, σ_B = 0.06, σ_C = 0.85 and the three correlations, the portfolio variance is the full quadratic form w⊤Σw with weights (0.60, 0.30, 0.10). Grinding through every variance and covariance term:

$$\sigma_p^2 = (0.60 \cdot 0.14)^2 + (0.30 \cdot 0.06)^2 + (0.10 \cdot 0.85)^2 + 2\big[\,\text{cross terms}\,\big]$$

The own-variance pieces are (0.084)² = 0.007056, (0.018)² = 0.000324, and (0.085)² = 0.007225. The cross terms (each 2·w_iw_jρ_ijσ_iσ_j): A–B = 2(0.60)(0.30)(−0.20)(0.14)(0.06) = −0.000605; A–C = 2(0.60)(0.10)(0.70)(0.14)(0.85) = +0.009996; B–C = 2(0.30)(0.10)(0.10)(0.06)(0.85) = +0.000306. Sum it all: **σ_p² ≈ 0.024302**, so σ_p ≈ **0.1559, i.e. about 15.6% per year**.

**Step 2 — turn annual volatility into a 1-day 95% VaR in dollars.** Daily volatility = 0.1559 / 15.87 = 0.00982, i.e. about 0.98% a day. Then:

$$\text{Total VaR} = 1.645 \times 0.00982 \times \$10{,}000{,}000 \approx \$161{,}543$$

So on a bad-but-ordinary day, this book stands to lose more than **\$161,543** about one day in twenty. That is the total. Now we crack it open.

*A book's total VaR is a real, hard dollar number — but it is a sum, and the whole game is finding out which positions the sum is made of.*

## Marginal VaR: which dollar costs the most risk

Component VaR tells you where the risk *is*. Marginal VaR tells you where *not to add*. They are two readings of the same decomposition, and you need both.

Marginal VaR is the derivative — the risk added by the *next* dollar. Concretely, if you were about to deploy fresh capital, marginal VaR ranks your positions by how much extra portfolio risk each one would soak up per dollar. The position with the highest marginal VaR is the one you should be most reluctant to add to, because every dollar you put there costs you the most risk. The position with the *lowest* (or negative) marginal VaR is where new money is cheapest in risk terms — and a negative marginal VaR means adding to that position would actually *lower* total portfolio risk, the mathematical signature of a true hedge.

![Horizontal bar chart of marginal VaR per one hundred thousand dollars added to each position, showing the concentrated name adding about eight thousand dollars of risk per dollar tranche versus thirteen hundred for the index and under forty for the bond](/imgs/blogs/marginal-and-component-var-where-the-risk-actually-lives-3.png)

Figure 3 shows marginal VaR for our three positions, expressed as the extra 1-day 95% VaR added by the *next* \$100,000 you put into each. The numbers are not close. The next \$100,000 into the concentrated name C adds about **\$8,227** of VaR. The next \$100,000 into the index A adds about **\$1,302**. The next \$100,000 into the bond B adds about **\$39** — almost nothing, because the bond barely co-moves with the rest of the book. Put plainly: a dollar into C is roughly 6.3 times as risky as a dollar into A and over 200 times as risky as a dollar into the bond. If you were deciding where to deploy new capital, this chart screams "anywhere but C." And if you are already over your risk budget, it screams "C is where the marginal risk is concentrated — trim there."

#### Worked example: marginal VaR position by position

Let us compute the marginal VaR for each position on the \$10,000,000 book, using MCR_i = (Σw)_i / σ_p and then scaling by z/√252.

**Step 1 — compute (Σw), the covariance matrix times the weights.** This is the row of co-movement-weighted exposures. Working it out for each position (in annual variance units):

- (Σw)_A = w_Aσ_A² + w_Bρ_AB σ_Aσ_B + w_Cρ_AC σ_Aσ_C = 0.60(0.0196) + 0.30(−0.20)(0.14)(0.06) + 0.10(0.70)(0.14)(0.85) ≈ **0.01959**
- (Σw)_B ≈ 0.30(0.0036) + 0.60(−0.20)(0.14)(0.06) + 0.10(0.10)(0.06)(0.85) ≈ **0.00058**
- (Σw)_C ≈ 0.10(0.7225) + 0.60(0.70)(0.14)(0.85) + 0.30(0.10)(0.06)(0.85) ≈ **0.12376**

**Step 2 — divide by σ_p to get MCR (marginal contribution in volatility units).** With σ_p = 0.1559: MCR_A = 0.01959 / 0.1559 ≈ **0.1256**, MCR_B ≈ 0.00058 / 0.1559 ≈ **0.0037**, MCR_C = 0.12376 / 0.1559 ≈ **0.7939**. Notice C's marginal contribution is more than six times A's, even before any dollar scaling — that is the high volatility and the 0.70 correlation with the market doing their work.

**Step 3 — scale to a per-\$100,000 marginal VaR.** Multiply each MCR by (z/√252) × \$100,000 = (1.645 / 15.87) × \$100,000 ≈ \$10,366 per unit of MCR:

- Position A: 0.1256 × \$10,366 ≈ **\$1,302 of VaR per \$100,000 added**
- Position B: 0.0037 × \$10,366 ≈ **\$39 per \$100,000 added**
- Position C: 0.7939 × \$10,366 ≈ **\$8,227 per \$100,000 added**

*Marginal VaR is the price tag on the next dollar: position C charges you \$8,227 of risk for every \$100,000 you add, the bond charges \$39 — so if you must add somewhere, you now know exactly where it is cheapest.*

### Why the next dollar's risk depends on the whole book

It is worth slowing down on the single most important and least intuitive fact in this whole framework: a position's marginal VaR is not a property of the position. It is a property of the position *and everything it sits next to*. The same \$1,000,000 holding can have a high marginal VaR in one book and a low — even negative — marginal VaR in another, with not a single share of the position itself changed. What changed is the company it keeps.

The mechanism is the (Σw) term. When you nudge position *i* up by a dollar, the portfolio's variance changes by an amount that has two parts: the dollar's *own* contribution (its variance, σ_i²) and its *interaction* with every other dollar already in the book (the covariance terms, σ_iσ_jρ_ij, summed over every other position *j*, weighted by how much of *j* you hold). For our concentrated name C, the own-variance part is large (σ_C = 85% is enormous), but the *interaction* part is what really inflates its marginal VaR: C is 70% correlated with the \$6,000,000 index position, so adding a dollar of C adds a dollar that tends to lose *on the same days* the big index position is losing. The market does not get a vote on whether your losses arrive politely one at a time; correlated positions deliver their losses *together*, which is precisely when a drawdown is most dangerous. Marginal VaR prices that synchronization. A name that loses when everything else loses is far riskier, at the margin, than its standalone volatility suggests — and a name that gains when everything else loses can have a marginal VaR *below* its standalone volatility, sometimes below zero.

This is also why "I'll just look at each position's standalone volatility" is a trap that feels rigorous but is not. Standalone volatility is blind to correlation. It would rank our three positions C, then A, then B by riskiness — which happens to get C right but for the wrong reason, and it would badly misprice a book full of correlated mid-vol names that each look tame alone but move as one. The classic version is a "diversified" sleeve of ten regional bank stocks: each one's standalone volatility looks moderate, but their pairwise correlations are near 0.8, so the sleeve behaves like one big position and its true component VaR is enormous. Standalone volatility says "ten moderate risks"; component VaR says "one giant risk." Only the decomposition that carries the full covariance matrix tells you the truth, because only it sees the correlations.

There is a clean geometric way to hold this. Total portfolio risk is a vector sum, not a scalar sum. Each position contributes a risk "vector" whose length is its dollar volatility and whose *direction* is set by its correlations. Positions pointing the same direction (high correlation) add up nearly head-to-tail, producing a long total vector — lots of risk. Positions pointing opposite directions (negative correlation) partly cancel, producing a short total vector — little risk. Marginal VaR is how much *longer the total vector gets* when you extend one position's vector by a unit. Extend a vector that already points along the total and the total grows a lot; extend one that points across the total and it barely grows; extend one that points *against* the total and it shrinks. That is the entire content of marginal VaR, and it is why the bond — pointing against the equity book — barely lengthens the total no matter how much of it you hold, while the concentrated name, pointing right along the equity book, lengthens it sharply with every dollar.

## Component VaR: the slices that add up

Marginal VaR is the *rate*; component VaR is the rate times the *quantity* — multiply each position's marginal contribution by the dollars actually sitting in it, and you get its component VaR, the share of total VaR it is genuinely responsible for. The miracle, again, is that these components sum to the total exactly.

![Stacked bar of the three component VaRs on the left equalling a single solid total VaR bar on the right, both reaching about one hundred sixty-one thousand dollars, showing the decomposition adds up with no residual](/imgs/blogs/marginal-and-component-var-where-the-risk-actually-lives-4.png)

Figure 4 makes the additivity literal. On the left is a single bar stacked from the three component VaRs — A's slice, B's slice, C's slice. On the right is the total VaR computed *directly* from the portfolio volatility, the \$161,543 we found above, with no decomposition involved at all. The two bars are the same height to the dollar. That is the Euler identity made visible: the parts reconstruct the whole. This is what licenses you to treat each position's component VaR as its *true* risk budget consumption. It is not a heuristic split or a rough attribution — it is an exact carve-up of the very same number your risk system reports as the headline VaR. You can hand each desk its component VaR and the numbers will tie out to the firm total.

#### Worked example: component VaR for the three-asset book

Now the payoff. Component VaR_i = w_i × MCR_i × σ_p-to-VaR scaling — but the cleanest way is component VaR_i = w_i × marginal VaR per dollar × book. Let us just multiply each position's MCR by its weight to get its risk-contribution share, then dollarize.

**Step 1 — risk contribution in volatility units, RC_i = w_i × MCR_i:**

- RC_A = 0.60 × 0.1256 = **0.07534**
- RC_B = 0.30 × 0.0037 = **0.00111**
- RC_C = 0.10 × 0.7939 = **0.07939**

**Step 2 — check they sum to σ_p.** 0.07534 + 0.00111 + 0.07939 = **0.15584 ≈ σ_p (0.1559)**. They do, to rounding. The decomposition is exact.

**Step 3 — convert each to a percentage of total risk.** Divide each RC by σ_p: A = 0.07534 / 0.1559 = **48.3%**, B = 0.00111 / 0.1559 = **0.7%**, C = 0.07939 / 0.1559 = **50.9%**.

**Step 4 — dollarize against the \$161,543 total VaR.** Each component VaR is its percentage times the total:

- Component VaR_A = 48.3% × \$161,543 ≈ **\$78,116**
- Component VaR_B = 0.7% × \$161,543 ≈ **\$1,161**
- Component VaR_C = 50.9% × \$161,543 ≈ **\$82,267**

Sum: \$78,116 + \$1,161 + \$82,267 = **\$161,544 ≈ total VaR**. The 10%-of-capital position C is the single largest risk contributor in the entire book, at \$82,267 — more than the \$78,116 from the index position holding six times the money. And the \$3,000,000 bond position, nearly a third of the capital, contributes \$1,161, well under 1% of the risk.

*Component VaR is the only honest answer to "how much of my risk does this position own?" — and on this book the answer ranks the positions in exactly the opposite order to their dollar size.*

The same exercise scales down cleanly to a retail book. Run the *identical weights* on a **\$100,000 account** — \$60,000 in A, \$30,000 in B, \$10,000 in C — and every percentage is unchanged because the decomposition is scale-free. The total 1-day 95% VaR becomes about **\$1,615**, and the component VaRs become roughly **\$781 for A, \$12 for B, and \$823 for C**. A retail trader with \$10,000 in one spicy name and \$60,000 in an index fund is, by component VaR, taking *more* risk in the \$10,000 name than in the index. The dollars hide it; the decomposition reveals it.

## Where you think the risk is vs. where it actually lives

The reason this matters is psychological as much as mathematical. Human risk intuition anchors on dollar size — the biggest number on the position blotter feels like the biggest risk. The decomposition systematically corrects that bias, and the correction is often a complete reordering.

![Before and after comparison showing positions ranked by dollar size on the left flagging the large index fund as the top worry, versus positions ranked by component VaR on the right flagging the small concentrated name as the real risk and the bond as a near-zero-risk diversifier](/imgs/blogs/marginal-and-component-var-where-the-risk-actually-lives-5.png)

Figure 5 lays the two mental models side by side. On the left is how you *assume* the risk is ranked when you eyeball the book by dollar size: the \$6,000,000 index at the top of the worry list, the \$3,000,000 bond second, the \$1,000,000 side bet dismissed last. On the right is how the component-VaR decomposition *actually* ranks it: the small concentrated name C at the top with 51% of the risk, the index A second at 48%, and the bond B last at under 1% — but for the opposite reason you assumed, because it is a genuine diversifier, not because it is small. The two orderings are nearly reversed. The position you would have ignored is the one carrying the book; the position you would have hedged against is the one quietly protecting you. Every risk failure that begins with "but we were diversified" is, at root, someone reading the left-hand list and believing it was the right-hand one.

This reordering is not a quirk of contrived inputs. It is the generic situation whenever a book mixes a high-volatility, market-correlated name with low-volatility diversifiers. The high-vol name's component VaR is amplified twice — once by its own large σ, and again by its positive correlation with the rest of the book, which means it tends to lose *when everything else is losing*, exactly when it hurts most. The bond's component VaR is suppressed twice — by its small σ and by its negative correlation, which means it tends to be *up* on the days the rest of the book is down. Component VaR captures both effects automatically, because the (Σw) term carries every co-movement. Standalone volatility cannot; it would tell you C is risky and B is safe in isolation, but it could never tell you that C owns half your book and B owns essentially none of it.

## Trimming the right position first

Here is where the decomposition pays for itself. Suppose your risk system flashes red: the book is over its VaR limit and you have to take risk down. You can sell \$500,000 of *something*. Which position? Dollar intuition says "sell some of the big one" — trim A. Marginal VaR says something very different.

![Bar chart of total portfolio VaR before and after selling five hundred thousand dollars from each candidate position, showing that trimming the high-marginal-VaR concentrated name cuts total VaR by about forty thousand dollars versus six thousand for the index and under two hundred for the bond](/imgs/blogs/marginal-and-component-var-where-the-risk-actually-lives-6.png)

Figure 6 runs the experiment. Starting from total VaR of \$161,543, we sell \$500,000 from each candidate in turn and recompute the total. Sell \$500,000 of the index A and total VaR drops to about \$155,066 — a \$6,477 reduction, roughly \$0.013 of VaR removed per dollar sold. Sell \$500,000 of the bond B and total VaR barely moves, falling to \$161,379 — a \$164 reduction, essentially nothing per dollar (in fact, trimming a diversifier can *raise* risk in some books, because you are removing protection). But sell \$500,000 of the concentrated name C — the highest-marginal-VaR position — and total VaR falls all the way to \$121,434, a **\$40,109** reduction, about \$0.080 of VaR removed per dollar sold. The *same* \$500,000 of selling removes more than six times as much risk when aimed at C instead of A, and over 200 times as much as aiming it at the bond. This is the entire practical value of the decomposition: it tells you that to cut risk efficiently you trim the highest-marginal-VaR name, not the biggest-dollar name.

#### Worked example: cutting \$500,000 of risk efficiently

Your \$10,000,000 book is over a \$150,000 VaR limit (it sits at \$161,543) and you need to get under it. You decide to sell \$500,000 of one position. Let us compare the three choices with explicit math.

**Option 1 — trim the biggest dollar position, A.** New weights (0.55, 0.30, 0.10) on a \$9,500,000 book (or equivalently, weight 0.55 of the original \$10,000,000-scaled exposure). Recomputing σ_p with A reduced gives a new total VaR of about **\$155,066**. Reduction: \$161,543 − \$155,066 = **\$6,477**. Per dollar sold: \$6,477 / \$500,000 ≈ **\$0.013**. You are still \$5,066 over the limit. Selling the big name barely helped.

**Option 2 — trim the bond, B.** New total VaR ≈ **\$161,379**. Reduction: just **\$164**. Per dollar: \$0.0003. You sold a diversifier and removed essentially no risk; you are still \$11,379 over the limit and you have *worse* diversification now.

**Option 3 — trim the concentrated name, C.** New total VaR ≈ **\$121,434**. Reduction: \$161,543 − \$121,434 = **\$40,109**. Per dollar sold: \$40,109 / \$500,000 ≈ **\$0.080**. One \$500,000 sale takes you from \$161,543 all the way down to \$121,434 — comfortably under the \$150,000 limit, with \$28,566 of headroom to spare.

The marginal-VaR ranking predicted this exactly: C had the highest marginal VaR (\$8,227 per \$100k), so each dollar sold from C removes the most risk. To get under the limit you needed *one* trade aimed at the right position, not a scattershot sale across the book.

*When you have to cut risk, sell the highest-marginal-VaR position first — the decomposition turns "reduce risk" from a vague instruction into a single, optimal trade.*

## The full picture in one view

It helps to see the whole decomposition for a book laid out at once — capital, marginal VaR, component VaR, and percentage share, position by position — because the *contrast* between the capital ranking and the risk ranking is the lesson.

![Two stacked panels for the three-asset book, the top panel showing capital in millions per position and the bottom panel showing component VaR per position with marginal VaR and percentage labels, making the big-capital small-risk and small-capital big-risk contrast explicit](/imgs/blogs/marginal-and-component-var-where-the-risk-actually-lives-7.png)

Figure 7 stacks the two rankings. The top panel is capital: A towers at \$6.0M, B at \$3.0M, C a stub at \$1.0M — the ordering your eye expects. The bottom panel is component VaR with the marginal VaR and percentage annotations: now C towers, A is second, and B has all but vanished. Reading the two panels together is the skill. The position that is biggest by money (A) is *second* by risk; the position that is smallest by money (C) is *first* by risk; and the position that is a third of the capital (B) is a rounding error of the risk. This is the table a real risk report should put in front of a portfolio manager — not "you have 60/30/10 in capital" (which everyone already knows) but "you have 48/1/51 in *risk*," because the second sentence is the one that changes a decision.

#### Worked example: a "diversified" retail book that is secretly one bet

A retail trader runs a **\$100,000 account** they describe as diversified: \$40,000 in a broad index ETF (σ = 15%), \$40,000 in a dividend-stock fund (σ = 16%), and \$20,000 in a single leveraged semiconductor name (σ = 80%) that is highly correlated (ρ ≈ 0.75) with both equity sleeves. "No position over 40%," they say. "I'm spread out." Let us decompose.

**Step 1 — the leveraged name's marginal contribution.** Because it is high-vol *and* highly correlated with the two equity sleeves it sits beside, its (Σw) entry is large: roughly 0.20 × 0.80² + (0.40 + 0.40) × 0.75 × 0.80 × 0.155 ≈ 0.128 + 0.074 ≈ **0.202** in variance units, far larger than either equity sleeve's ≈ 0.03. After dividing by σ_p (≈ 0.19 here), its marginal contribution dwarfs the others.

**Step 2 — the component VaR shares.** Crunching the full decomposition, the \$20,000 semiconductor name lands around **45–50% of the account's total VaR**, while the two \$40,000 equity sleeves split the rest. The position the trader called a "small 20% satellite" owns roughly half the risk of the entire account.

**Step 3 — the survival reading.** This account is not diversified; it is a leveraged-semiconductor bet with two index funds bolted on for comfort. On a bad day for chips, the \$20,000 sleeve drives the loss and the two equity sleeves — correlated with it — amplify rather than cushion it. The fix the decomposition points to is precise: cut the semiconductor sleeve (the highest marginal VaR), not the index funds.

*"No position over 20%" is a capital limit, and capital limits say nothing about risk — a 20% position can own 50% of your risk, which is why every limit in your book should be a risk-contribution limit too.*

## Decomposing by factor, not just by position

So far we have decomposed risk *position by position*. That is the right first move, and for many books it is enough. But there is a subtler and often more dangerous form of hidden concentration that a position-level decomposition can still miss: concentration in a shared *risk factor* that cuts across many positions. This is the failure mode behind the most expensive blow-ups, so it earns its own section.

A **risk factor** is a common driver that several positions all load on — the broad equity market, the level of interest rates, the price of oil, the strength of the dollar, a single funding currency like the yen. Two positions can look completely different on the blotter (a tech stock and an emerging-market bond, say) and yet both be, underneath, a bet on the same factor (global risk appetite). If you decompose only by position, each looks like a modest, independent slice of risk. But if you decompose by *factor*, you discover that ten "different" positions are really one big factor bet wearing ten name-tags, and the factor's component VaR is enormous.

The math is the same Euler decomposition, applied one level deeper. Instead of asking "how much VaR does each *position* contribute?", you map every position onto its factor exposures (its betas to the market, to rates, to the dollar) and ask "how much VaR does each *factor* contribute?" The factor component VaRs sum to the total exactly, just as the position ones do, because it is the same homogeneity argument. What the factor view surfaces that the position view hides is *correlated exposure through a common driver*. A book that is beautifully diversified by position can be catastrophically concentrated by factor — and the factor decomposition is the only thing that shows it before the factor moves.

#### Worked example: a position-diversified book that is one factor bet

A macro fund runs a **\$10,000,000 book** across eight positions: long Brazilian equities, long Mexican bonds, long South African rand, long Turkish lira carry, long an emerging-market equity ETF, and three more "uncorrelated" emerging-market trades. The position-level decomposition looks healthy — no single position over about 18% of component VaR, the textbook picture of diversification.

**Step 1 — find the shared factor.** Map each position onto its sensitivity to *global risk appetite* (proxy it with the inverse of the VIX, or with a broad risk-on/risk-off factor). Every one of the eight positions is a risk-on trade: they all rise when global risk appetite rises and fall when it falls. Their betas to the risk-on factor are all positive and large.

**Step 2 — decompose by factor.** When you compute factor component VaR, the risk-on factor alone accounts for roughly **75–85% of the total VaR**. The eight positions were never independent; they were eight expressions of one bet — "risk-on stays on." The position view showed eight ~12% slices; the factor view shows one ~80% slice.

**Step 3 — the survival reading.** On a risk-off day (a Fed surprise, a geopolitical shock), all eight positions fall *together*, because they share the factor. The "diversification" evaporates exactly when needed, and the book takes a loss as if it were a single 80%-of-risk position — which, by factor, it was. The position-level component VaR limit of "nothing over 18%" did not catch it; only a *factor-level* limit would.

*Position-level component VaR catches concentration in a single name; factor-level component VaR catches concentration in a single bet spread across many names — and the second kind is what turns a "diversified" book into a one-way wager in a crisis.*

This is the deeper reason the case studies below — the yen-carry unwind, LTCM's convergence trades — were blow-ups despite looking diversified by position. Their risk was concentrated in a factor (the yen funding leg, the flight-to-quality correlation regime) that only a factor decomposition would have surfaced. Position-level component VaR is necessary; factor-level component VaR is what separates the survivors. The full machinery for the covariance and factor models lives in [mean-variance and the efficient frontier](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants); here the takeaway is operational — decompose by factor, set factor-risk limits, and never trust a position blotter to tell you you are diversified.

## Common misconceptions

**"My biggest position is my biggest risk."** Usually false. Risk scales with *volatility and correlation*, not just dollars. In our book the \$6,000,000 index position (60% of capital) contributed 48% of the risk, while the \$1,000,000 concentrated name (10% of capital) contributed 51%. The biggest-dollar position is your biggest risk only when it also happens to be the highest-volatility, most-correlated name — which for a diversified core holding it usually is not.

**"If no single position is over 10% of capital, I'm diversified."** A capital limit is not a risk limit. A book can satisfy "nothing over 10%" and still have one of those 10% positions owning half the risk, if that position is volatile and correlated with the rest. Diversification is a property of *component VaR shares*, not dollar weights. The honest test is "is any position over, say, 25% of total component VaR?" — and a clean capital blotter routinely fails it.

**"Component VaRs are just an approximate attribution."** No — they are *exact*. The component VaRs sum to total VaR to the dollar, by Euler's theorem, because VaR is homogeneous of degree one in position sizes. In our book \$78,116 + \$1,161 + \$82,267 = \$161,544, equal to the directly-computed total VaR of \$161,543 up to rounding. This is a hard identity, not a heuristic. (Variance does *not* decompose cleanly, which is why you decompose volatility/VaR, not variance.)

**"To cut risk, sell a bit of everything."** Inefficient. Selling \$500,000 spread across our book would remove far less risk than \$500,000 aimed at the single highest-marginal-VaR position. The same \$500,000 removed \$40,109 of VaR from position C but only \$6,477 from A and \$164 from the bond. Marginal VaR tells you the one trade that does the most work; scattershot selling wastes liquidity and leaves your real risk concentration largely intact.

**"A diversifier with lots of capital is a big part of my risk."** Often the opposite. The \$3,000,000 bond position — 30% of the account — contributed \$1,161, under 1% of total VaR, because its low volatility and negative correlation suppress its component VaR twice over. Selling it to "reduce exposure" would barely move total VaR and would *worsen* diversification. Capital tied up in a true diversifier is not risk; it is the thing holding your risk down.

**"Marginal VaR is always positive."** Not necessarily. A position negatively correlated enough with the rest of the book can have a *negative* marginal VaR — adding to it *lowers* total portfolio risk. That is the mathematical fingerprint of a hedge. If your decomposition shows a position with negative marginal VaR, adding to it (within reason) is risk-reducing, and trimming it is risk-*increasing* — the exact reverse of the usual instinct.

## How it shows up in real markets

The decomposition is not an academic nicety; it is the difference between the funds that survived the last twenty years and the ones whose names are now cautionary tales. Every major blow-up in the series' data is, at heart, a story of a book whose component VaR was concentrated in one place that the headline numbers did not surface.

**Archegos, March 2021.** Archegos Capital Management held enormous concentrated single-stock exposure through total-return swaps, financed across several prime brokers — each of whom saw only *their* slice and none of whom could see the total. Reconstructed at the portfolio level, the risk was savagely concentrated: a handful of correlated single names owned almost all of the component VaR, and they were levered roughly 5x. When those names fell together, the component-VaR concentration became a real loss of **over \$10 billion to the banks**, with Credit Suisse alone losing about **\$5.5 billion**. The fatal blind spot was exactly the one this post is about — each counterparty looked at *their* dollar exposure and missed that the true risk lived in the concentrated, correlated whole. A component-VaR view of the aggregate book would have screamed concentration; the per-broker dollar views whispered nothing.

**Amaranth, September 2006.** Amaranth Advisors lost about **\$6.6 billion in a single week**, almost all of it from one concentrated, levered natural-gas calendar-spread bet in an illiquid market. By dollar exposure the book may have looked like a multi-strategy fund; by component VaR it was a single gas-spread position carrying nearly all the risk. When the spread moved against them, there was no diversification to cushion it because the diversification was an illusion of the capital blotter, not a property of the risk decomposition. One position owned the book.

**LTCM, August–September 1998.** Long-Term Capital Management held dozens of convergence trades that *looked* diversified by instrument and geography. But in the flight to quality after Russia's default, the correlations between those trades all went to 1 — and a book that decomposes into many small independent component VaRs in calm times collapses into one giant component VaR when everything co-moves. LTCM lost about **\$4.6 billion** in four months on roughly 25x leverage. The decomposition lesson is brutal and central: component VaR depends on the correlation regime, and the regime can flip exactly when you most need the diversification to hold. (The strategic version of this — many smart funds crowding the same convergence trade — is dissected in [the LTCM crowded-genius case study](/blog/trading/game-theory/case-study-ltcm-1998-the-crowded-genius-trade), and the correlation-failure mode itself in [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).)

**The yen-carry unwind, August 5, 2024.** A crowded funding-carry trade — borrow cheap yen, buy higher-yielding assets worldwide — looked diversified across dozens of long positions. But every leg shared the same hidden risk factor: the yen funding leg. By component VaR the book was one bet (short yen) in a hundred costumes. When the yen ripped higher, the shared factor dominated and the "diversified" book lost together; the Nikkei fell **−12.4% in a day**, its worst since 1987, and the VIX spiked intraday to about **65.7**. A factor-aware component-VaR decomposition would have shown the carry funding leg owning most of the risk despite holding none of the headline dollar weight. The positions were many; the risk was one.

**COVID dash-for-cash, March 2020.** It is worth one more, because it shows the decomposition failing in real time. Coming into February 2020, plenty of "diversified" books showed clean component-VaR breakdowns: equities here, credit there, a bit of gold, some carry. The decomposition, run on *calm-regime* correlations, said the risk was spread. Then the dash-for-cash hit and correlations across almost every risk asset went to 1 at once — stocks, credit, even gold sold off together as investors raised cash indiscriminately. A book whose component VaR was nicely distributed in January became, by late March, a book where one factor (the scramble for dollar liquidity) owned nearly all the risk. The S&P fell about **−34%** peak to trough in roughly a month and the VIX hit a record **82.69** close. The lesson is not that the decomposition was wrong; it is that *it was run on the wrong correlations*. Component VaR computed on calm correlations is a fair-weather instrument. The funds that survived were the ones who had also run it on crisis correlations and sized for the world where their diversification fails — which is the difference between a risk report that comforts you and one that protects you.

The thread through all five: the dollar blotter looked diversified, and the component-VaR decomposition — run honestly, on the *right* correlations and at the *factor* level — would have shown it was not. Survival, again, goes to whoever knew *where their risk actually lived* before the market told them.

## The risk-decomposition playbook

Concrete rules, in priority order. None of this matters unless you act on it.

1. **Decompose every book by component VaR, not dollar weight.** The capital blotter is the question, not the answer. Run the marginal-VaR / component-VaR decomposition on every portfolio and read the *risk* ranking. If your risk system only shows a single headline VaR, it is hiding the very thing you need to manage.

2. **Set risk-contribution limits, not just dollar limits.** "No position over 15% of capital" is necessary but wildly insufficient. Add a hard rule: **no single position over ~25–30% of total component VaR** (tighten for smaller books). A capital limit lets a 10% position own 50% of your risk; a component-VaR limit catches it. This is the position-level companion to [concentration and position limits](/blog/trading/risk-management/concentration-and-position-limits-the-one-trade-that-can-end-you).

3. **Rank candidates for new capital by marginal VaR, and never add to the top of that list.** Before deploying fresh money, sort positions by marginal VaR. New capital is cheapest (in risk) where marginal VaR is lowest, and *most* dangerous where it is highest. If you must add to a high-marginal-VaR name, do it knowing each dollar costs multiples of the risk it would cost elsewhere.

4. **When you have to cut risk, trim the highest-marginal-VaR position first.** This is the single highest-leverage risk action you can take. In our book, one \$500,000 sale aimed at C removed \$40,109 of VaR versus \$6,477 for the same sale aimed at A — six times the risk reduction per dollar. Do not scatter the cut; aim it.

5. **Watch for negative marginal VaR — those are your hedges, protect them.** A position with negative marginal VaR is reducing your total risk; trimming it *raises* risk. Identify your diversifiers explicitly and resist the instinct to sell them when you want to "reduce exposure." Selling a diversifier is usually the worst risk trade on the board.

6. **Re-run the decomposition when correlations move.** Component VaR depends on the whole covariance matrix, and correlations are a regime, not a constant. A position that is a diversifier in calm times can become a risk amplifier when correlations go to 1 — the LTCM, 2020, and 2024 pattern. Recompute on stressed (crisis-regime) correlations, not just calm ones, and size for the world where your diversification fails. ([Correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) makes this precise.)

7. **Tie it back to survival.** The point of all this is not elegance; it is to *not blow up*. You blow up when one position you did not know you owned the risk of moves against you. Component VaR is how you find that position before the market does. A book you have decomposed, limited by risk contribution, and stress-tested on crisis correlations is a book that can take a hit on any single name and stay in the game — and staying in the game is the only thing that lets your edge compound.

The deeper truth under the whole post is that a portfolio's risk is a *structure*, not a number. Total VaR is the top line, but the structure underneath — which positions own which slices of that line, and how those slices balloon when correlations rise — is what determines whether a bad day is a setback or a funeral. Marginal and component VaR are the instruments that let you read the structure. Used honestly, they turn the most dangerous sentence in trading — "but we were diversified" — into something you can check, and fix, before it costs you the account.

### Further reading

- [Value at Risk and exactly how VaR lies](/blog/trading/risk-management/value-at-risk-and-exactly-how-var-lies) — the headline number this post decomposes, and the tail it hides.
- [Risk parity sizing: equal risk is not equal money](/blog/trading/risk-management/risk-parity-sizing-equal-risk-not-equal-money) — the same Euler decomposition used to *set* risk shares rather than measure them.
- [Concentration and position limits: the one trade that can end you](/blog/trading/risk-management/concentration-and-position-limits-the-one-trade-that-can-end-you) — why a component-VaR limit beats a dollar limit, and the single position that ends accounts.
- [Mean-variance and the efficient frontier](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants) — the covariance-matrix machinery underneath portfolio volatility, derived in full.
- [Tail risk and extreme value theory](/blog/trading/math-for-quants/tail-risk-extreme-value-theory-math-for-quants) — what component VaR misses about the deep tail, and how to model it.

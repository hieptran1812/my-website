---
title: "Risk Parity Sizing: Equal Risk Is Not Equal Money"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A classic 60/40 portfolio looks balanced but is roughly 90% equity risk; this builds risk parity from first principles, shows why the safe asset gets levered, and is honest about the assumptions it can fail."
tags: ["risk-management", "risk-parity", "position-sizing", "portfolio-construction", "leverage", "diversification", "correlation", "volatility", "asset-allocation"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **The one idea:** allocating equal *money* across assets gives you wildly *unequal* risk — a classic 60/40 stock/bond portfolio puts 60% of the capital in stocks but more than 90% of the *risk* there too, so it is an equity bet wearing a balanced-portfolio costume.
> - **Risk parity** flips the question: instead of "how do I split the money?", ask "how do I split the *risk*?" — and then size so each asset contributes the same amount of risk.
> - Because the low-vol asset (bonds) contributes almost nothing per dollar, you must hold *more* of it — and to make it carry an equal risk share at a useful return, you **lever it above 100%** of capital.
> - The whole approach leans on three assumptions: **stable correlations, cheap and available leverage, and no shared shock**. In 2022 all three cracked at once — stocks and bonds fell together — and levered risk parity lost *more* than the 60/40 it was meant to beat.
> - This is sizing at the **portfolio** level: a discipline for splitting risk across positions, not a single-position stop. It only helps you survive if you respect the leverage and de-risk when correlations break.

In 2018, a quiet corner of the asset-management world ran a number that should be on every investor's wall. They took the most boring, "balanced," grandmother-safe portfolio in existence — 60% stocks, 40% bonds, the default allocation taught in every personal-finance book — and they did not ask the usual question, "how is the money split?" Everyone knows that answer: 60 and 40. They asked a different one: **how is the *risk* split?** How much of the portfolio's day-to-day jumpiness, its drawdowns, its sleepless nights, actually comes from each piece?

The answer was brutal. The 40% in bonds — nearly half the money — was contributing less than a tenth of the risk. The 60% in stocks was contributing the rest. By the only measure that matters when things go wrong, a "balanced" 60/40 portfolio is not balanced at all. It is an equity portfolio with a small bond ornament hanging off the side. When stocks have a bad year, your "diversified" 60/40 has a bad year, almost exactly as bad as if you'd held nothing but stocks, because the stocks were carrying all the risk the whole time. The bonds were along for the ride, contributing comfort but almost no protection in proportion to the capital you tied up in them.

![Two bar charts side by side, the left showing a 60/40 portfolio split by capital as 60 percent stocks and 40 percent bonds, the right showing the same portfolio split by risk contribution as 92 percent stocks and 8 percent bonds](/imgs/blogs/risk-parity-sizing-equal-risk-not-equal-money-1.png)

Figure 1 is the entire thesis of this post in one image, and it's the one to burn into memory. On the left is how a 60/40 splits the *money*: a tall stocks bar at 60% next to a bonds bar at 40% — looks reasonably even. On the right is how that exact same portfolio splits the *risk*: stocks tower at about 92%, bonds shrink to about 8%. Same portfolio. Same dollars. Completely different picture, because money and risk are not the same currency. This post builds, from absolutely nothing, the machinery that produces those numbers — what "risk contribution" means, how to compute it, and why one asset can soak up 92% of the risk on 60% of the money. Then it builds the cure, **risk parity**: a sizing method that equalizes the risk each asset contributes rather than the capital. We'll see why that forces you to lever the safe asset, do the full dollar arithmetic on a \$100,000 account and a \$10,000,000 book, and — most importantly — be ruthlessly honest about the three assumptions risk parity leans on and the regime that breaks all three at once.

This is the survival spine of the whole series, viewed through a new lens. The first job of a trader or an allocator is not to make money, it's to *not blow up*, because you can only compound if you're still in the game. And the fastest way to blow up while *thinking* you're being prudent is to believe you're diversified when you're not. A 60/40 holder who lived through 2008 thought they owned a balanced portfolio; they actually owned an equity portfolio that fell about 30% while they told themselves the bonds were protecting them. Risk parity is one answer to "what does *genuinely* balanced look like?" — and, just as importantly, a cautionary tale about how even a beautifully balanced risk budget can fail when the correlations it depends on betray it.

## Foundations: the building blocks of risk contribution

Before we can equalize risk, we have to define it precisely. Let's nail down every term from zero. If you allocate portfolios for a living, skim. If you don't, this section is the floor everything else stands on.

**Volatility (the risk of a single asset).** The **volatility** of an asset is the standard deviation of its returns — a measure of how much its return bounces around its average, usually quoted per year. A stock index might have a volatility of about 16% per year; a broad bond index, more like 6%. That single number, written σ (sigma), is our basic unit of "how risky is this thing on its own." It is symmetric (it treats up-moves and down-moves the same) and it is famously *not* the whole story of risk — a separate post, [volatility and why it is not risk](/blog/trading/risk-management/volatility-and-why-it-is-not-risk), draws that line carefully. But for *sizing a portfolio*, volatility is the right working unit, and that's the job here.

**Weight.** The **weight** of an asset, written w, is the fraction of your capital allocated to it. A 60/40 portfolio has w = 0.60 in stocks and w = 0.40 in bonds. Weights normally sum to 1 (you're fully invested), but — and this matters enormously for risk parity — once you allow *leverage*, the weights can sum to *more* than 1. A portfolio with 42% in stocks and 112% in bonds has weights summing to 154%; the extra 54% is borrowed money.

**Correlation.** **Correlation**, written ρ (rho), measures how two assets move together, on a scale from −1 to +1. A correlation of +1 means they move in perfect lockstep; 0 means their moves are unrelated; −1 means they move exactly opposite. The headline reason anyone holds bonds alongside stocks is that, *historically and in calm times*, stocks and bonds have had a low or even negative correlation — when stocks fall in a recession, bonds often rally as the central bank cuts rates, so the bonds cushion the stocks. Hold that thought, because the central failure mode of this entire post is that correlation is **not a constant** — it's a regime — and the regime can flip exactly when you need it not to. (That correlation is a regime, not a number, is its own topic in [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant).)

**Portfolio volatility.** When you combine assets, the portfolio's volatility is *not* the weighted average of the individual volatilities — that's the whole magic of diversification. For two assets it's:

$$\sigma_p = \sqrt{(w_s \sigma_s)^2 + (w_b \sigma_b)^2 + 2\, w_s w_b\, \rho\, \sigma_s \sigma_b}$$

The cross term, with ρ in it, is the diversification benefit: when correlation is low, that term is small and the combined volatility is *less* than the weighted average. When correlation goes to 1, the formula collapses to a simple weighted average and the diversification vanishes — which is precisely the disaster we'll build toward.

**Marginal contribution to risk.** Here is the new idea, and the heart of the whole post. Suppose you nudge your holding of one asset up by a tiny amount. How much does the *whole portfolio's* volatility change? That sensitivity is the asset's **marginal contribution to risk** (MCR). For asset *i*, it is the *i*-th row of the covariance matrix times the weight vector, divided by portfolio volatility:

$$\text{MCR}_i = \frac{(\Sigma\, w)_i}{\sigma_p}$$

where Σ (Sigma) is the covariance matrix — the table of every variance and covariance among the assets. You don't need to love that notation. What matters is the intuition: an asset's marginal contribution to risk depends not just on its *own* volatility but on how it *co-moves* with everything else you hold.

**Risk contribution.** Multiply the marginal contribution by the weight, and you get the asset's **risk contribution** (RC) — the actual share of total portfolio risk that asset is responsible for:

$$\text{RC}_i = w_i \times \text{MCR}_i$$

The beautiful fact that makes the whole framework work is that these risk contributions **sum exactly to the total portfolio volatility**:

$$\sum_i \text{RC}_i = \sigma_p$$

This is not an approximation — it's an exact decomposition (a consequence of Euler's theorem, because volatility is a function that scales linearly with the size of the book). It means we can carve the portfolio's total risk into clean, additive slices, one per asset, and ask: is this carving *equal*? In a 60/40, emphatically not. In risk parity, by construction, yes.

It's worth pausing on *why* this decomposition is the right one, because it's the engine of everything that follows. Volatility is what mathematicians call a **homogeneous function of degree one** in the weights: double every position and the portfolio's volatility doubles exactly. Euler's theorem says that for any such function, the function equals the sum of (each input × the function's sensitivity to that input). The sensitivity to a position *is* its marginal contribution to risk, and the position size *is* the weight, so weight × marginal contribution sums to total volatility. That's the whole reason risk contributions add up cleanly rather than leaving some uncounted "interaction" residue floating around. Variance — volatility *squared* — does **not** decompose so cleanly, which is exactly why practitioners decompose *volatility*, not variance, when they talk about risk budgets. If someone shows you a risk decomposition whose pieces don't sum to the total, they've used the wrong measure.

One more subtlety the formula hides: an asset's risk contribution depends on the *whole portfolio*, not just on the asset. The same bond position contributes a different amount of risk depending on what it's sitting next to, because the covariance term mixes in how it co-moves with every other holding. An asset that's negatively correlated with the rest of the book can even have a *negative* marginal contribution — adding more of it *reduces* total portfolio volatility. That's the mathematical signature of a true hedge, and it's why risk contribution, not standalone volatility, is the honest unit for sizing. A position isn't risky or safe in isolation; it's risky or safe *in context*.

With volatility, weight, correlation, portfolio volatility, marginal contribution, and risk contribution defined, you have the entire toolkit. Now let's use it to expose the 60/40.

## The 60/40 autopsy: where the risk really lives

Let's do the decomposition by hand, with real numbers, because the result is so counterintuitive that you should see every step. Take the textbook 60/40 with three honest inputs: stocks at σ = 16% volatility, bonds at σ = 6% volatility, and a calm-times correlation of ρ = 0.10 (stocks and bonds barely moving together — generous to the bonds-as-diversifier story).

#### Worked example: the 60/40 risk decomposition

You run a **\$100,000 account** as a 60/40: **\$60,000 in stocks, \$40,000 in bonds.** Let's find where the risk lives.

**Step 1 — the portfolio volatility.** Plug into the two-asset formula with w_s = 0.60, w_b = 0.40, σ_s = 0.16, σ_b = 0.06, ρ = 0.10:

$$\sigma_p = \sqrt{(0.60 \times 0.16)^2 + (0.40 \times 0.06)^2 + 2(0.60)(0.40)(0.10)(0.16)(0.06)}$$

The pieces: (0.096)² = 0.009216, (0.024)² = 0.000576, and the cross term = 2 × 0.60 × 0.40 × 0.10 × 0.16 × 0.06 = 0.000092. Sum = 0.009884, square root = **0.1013, i.e. about 10.1% per year**. So the whole portfolio wobbles about 10.1% a year.

**Step 2 — each asset's risk contribution.** The stocks' contribution works out to about **0.0933** (in volatility units) and the bonds' to about **0.0080**. They sum to 0.1013 — exactly the portfolio volatility, as promised. (If you want the mechanics: RC_stocks = w_s × [w_s σ_s² + w_b ρ σ_s σ_b] / σ_p, and similarly for bonds.)

**Step 3 — turn those into percentages.** Stocks' share of total risk = 0.0933 / 0.1013 = **92.1%**. Bonds' share = 0.0080 / 0.1013 = **7.9%**.

So a portfolio that put **60% of the money** in stocks put **92% of the risk** there. The 40% you allocated to bonds — \$40,000 of real capital, nearly half your account — is buying you under 8% of the portfolio's risk profile. In a bad year for stocks, that \$40,000 of bonds is not going to save you, because it was never carrying its share of the risk to begin with.

*A 60/40 portfolio is an equity portfolio with a rounding-error of bonds attached: the money looks balanced, the risk is anything but.*

This is the number that launched an entire investment philosophy. It is not a quirk of my chosen inputs — change the volatilities a bit and you still land somewhere north of 85% equity risk, because stocks are simply *two to three times as volatile* as bonds, and risk scales with volatility, so dollar-for-dollar the stocks dominate. The bonds in a 60/40 are doing almost nothing to balance the *risk*; they are mostly dragging down the *return* while contributing a sliver of stability. You're paying, in foregone return, for diversification you're not really getting.

There's a clean way to see why the imbalance is so extreme. Ignore correlation for a moment (it's small here) and the risk contributions are roughly proportional to (weight × volatility)² for each asset. For stocks that's (0.60 × 0.16)² = 0.096² ≈ 0.0092; for bonds (0.40 × 0.06)² = 0.024² ≈ 0.00058. The ratio is about **16 to 1** — and 16/(16+1) ≈ 94%, almost exactly the equity risk share we computed. The squaring is what makes it so lopsided: stocks have 4× the dollar-risk of bonds in a 60/40 (0.096 vs 0.024), and squaring 4× turns it into 16× the *risk contribution*. The 60/40's equity dominance isn't a modeling artifact; it's baked into the geometry of how variance adds up. Two assets whose dollar-volatilities differ by 4× will *always* have a wildly skewed risk split, no matter how reasonable the capital split looks.

This also explains a fact that surprises people: changing the *correlation* assumption barely moves the 60/40's risk split. Run it at ρ = 0 (perfect diversification) or ρ = 0.30 (stocks and bonds more linked) and the equity risk share still sits around 90%. The skew comes from the volatility gap, not the correlation, so you can't argue your way out of it by claiming stocks and bonds are "really" uncorrelated. The only lever that genuinely rebalances the risk is the *weights* — and that is exactly what risk parity reaches for.

Why does this matter for survival? Because the whole point of holding two assets is so that one can hold you up when the other falls. But a hedge that contributes 8% of your risk can absorb at most about 8% of your shock. When the stocks are down 30%, the bonds — even if they rally — move the needle on a portfolio that was 92% driven by those stocks only a little. You *thought* you bought insurance; you bought a token. The 60/40 holder's sense of safety is, to a large degree, an illusion produced by looking at the capital split instead of the risk split.

## Risk parity: equalize the risk, not the money

So here's the fix, and it's almost embarrassingly simple to state: instead of splitting the *capital* equally (or 60/40, or any capital rule), split the *risk* equally. Choose weights so that every asset contributes the **same** risk contribution. For two assets, that means:

$$\text{RC}_\text{stocks} = \text{RC}_\text{bonds}, \quad \text{i.e. each is } 50\% \text{ of total risk.}$$

This is called **risk parity** (sometimes "equal risk contribution," ERC). The name is the whole idea: *parity*, equality, *of risk*, not of money. Let's solve for the weights that achieve it with our two assets.

For two assets, when the correlation is modest, the risk-parity weights turn out to be almost exactly proportional to the *inverse* of each asset's volatility — you hold *more* of the calmer asset and *less* of the jumpy one, in inverse proportion to how much risk each unit brings. With σ_s = 16% and σ_b = 6%, the inverse-volatility weights are:

$$w_s = \frac{1/16}{1/16 + 1/6} \approx 0.273, \qquad w_b = \frac{1/6}{1/16 + 1/6} \approx 0.727$$

So the risk-parity split is roughly **27% stocks / 73% bonds** — almost the mirror image of the 60/40. You hold nearly three-quarters of your money in the boring bonds, because each dollar of bonds is so much quieter that it takes that much more of them to carry an equal share of the risk.

![Grouped bar chart comparing the risk share of stocks and bonds under a 60/40 portfolio versus a risk-parity portfolio, with the 60/40 bars at 92 percent and 8 percent and the risk-parity bars both at 50 percent on the equal-risk line](/imgs/blogs/risk-parity-sizing-equal-risk-not-equal-money-2.png)

Figure 2 shows the payoff. On the left, the 60/40's lopsided 92/8 risk split. On the right, the risk-parity weighting's clean 50/50: the bars land exactly on the dashed equal-risk line. We did nothing magical — we just re-weighted toward the calmer asset until each side contributed the same risk. Let's verify with dollars.

#### Worked example: building the unlevered risk-parity portfolio

Back to the **\$100,000 account.** We want each asset to contribute 50% of the risk.

**Step 1 — the weights.** From the inverse-volatility rule: 27.3% stocks, 72.7% bonds. In dollars that's **\$27,270 in stocks** and **\$72,730 in bonds.** (We've completely flipped the 60/40 — much more in bonds now.)

**Step 2 — check the risk split.** Compute the portfolio volatility with these new weights: σ_p ≈ 6.47% per year. Then each asset's risk contribution comes out to about 0.0324 in volatility units — *identical for both* — so each is 0.0324 / 0.0647 = **50%** of the total. Equal risk, achieved.

**Step 3 — notice the problem.** The portfolio volatility is now only **6.47%**, versus the 60/40's 10.1%. We made the portfolio *safer* — genuinely balanced — but also *sleepier*. Because we tilted hard into low-volatility bonds, the whole portfolio barely moves. And a portfolio that barely moves also barely *returns*. We bought balance, but at the cost of expected return. That is the precise problem leverage is about to solve.

*Equalizing the risk forces you to hold a pile of the low-vol asset, which mechanically shrinks the whole portfolio's volatility — and its return — down to the level of the calm asset.*

A quick but important caveat on that inverse-volatility shortcut. The rule "weight each asset in inverse proportion to its volatility" gives *exactly* equal risk contributions only in two cases: when there are just two assets, or when all the correlations are equal. With three or more assets and unequal correlations, inverse-volatility is a decent first guess but not the true risk-parity solution — an asset that's highly correlated with the rest of the book should be held *less* than inverse-vol suggests, because its risk "double-counts" with its neighbours. The genuine equal-risk-contribution weights require solving a small fixed-point problem: start from any weights, compute each asset's risk contribution, nudge the weights toward equalizing them, and iterate until they converge. It's a handful of lines of code and converges in a few dozen steps. The two-asset case in this post happens to land on the inverse-vol weights exactly, which is why the arithmetic stays clean — but don't carry the shortcut into a real multi-asset book without checking it.

This step is where most people's intuition first rebels. "You want me to put nearly three-quarters of my money in *bonds*? That's a retiree's portfolio, not a growth portfolio." Correct — the *unlevered* risk-parity portfolio is sleepy by design. It is balanced but timid. The genius and the danger of risk parity are both contained in the next move: rather than accept the timid return, you take this beautifully balanced little portfolio and you *scale the whole thing up* with leverage, until its risk — and its expected return — match whatever target you actually want.

## Leverage: how the safe asset gets levered up to carry its share

Here is the move that defines risk parity as a strategy rather than just a sizing observation. We have a balanced-but-sleepy portfolio with 6.47% volatility. We want, say, the same 10.1% volatility the 60/40 had — but with the *risk evenly split*, not piled 92% on stocks. The solution: borrow money and scale up *both* legs of the balanced portfolio by the same factor, preserving the 50/50 risk split while lifting the total risk to where we want it.

The leverage factor L is just the ratio of the volatility we want to the volatility we have:

$$L = \frac{\sigma_\text{target}}{\sigma_\text{unlevered}} = \frac{10.1\%}{6.47\%} \approx 1.55$$

Multiply both weights by 1.55. The stocks go from 27.3% to about **42%** of capital; the bonds go from 72.7% to about **112%** of capital. That bond figure is the whole controversy in one number: **112% of your capital in bonds means you have borrowed money to hold *more bonds than you have cash*.** Risk parity levers the *safe* asset, not the risky one — which feels backwards to most people until you remember *why*. Bonds contribute so little risk per dollar that to make them carry half the risk budget at a respectable return, you need a *lot* of them, more than your own capital. Leverage is not bolted on to chase return; it's the structural consequence of equalizing risk across assets with very different volatilities.

![Line chart showing stock exposure and bond exposure as a function of the portfolio volatility target, with bond exposure rising past 100 percent into a shaded leverage region and a marked point at 112 percent bonds matching the 60/40 ten point one percent volatility](/imgs/blogs/risk-parity-sizing-equal-risk-not-equal-money-3.png)

Figure 3 traces this directly. The horizontal axis is the volatility target you choose; the lines show the resulting stock and bond exposures. Notice the bond line crosses the 100% mark — into the amber leverage region — at a fairly modest vol target, and to match the 60/40's 10.1% volatility you need about **112% bond exposure** (the red dot). Push the target to 12% and you'd need about 135% bonds; aim to match a *pure equity* portfolio's 16% volatility and you'd be running about 180% bonds. The whole strategy lives or dies on access to cheap, reliable leverage — which is exactly the assumption that fails in a crisis. Hold that.

#### Worked example: levering the risk-parity book to a 10.1% vol target

Still the **\$100,000 account.** We have the balanced 27/73 mix at 6.47% vol. We want 10.1% vol with the risk still split 50/50.

**Step 1 — the leverage factor.** L = 10.1% / 6.47% = 1.545. We'll scale both legs by 1.545.

**Step 2 — the levered dollar exposures.** Stocks: \$27,270 × 1.545 = **\$42,100.** Bonds: \$72,730 × 1.545 = **\$112,400.** Total exposure = \$42,100 + \$112,400 = **\$154,500**, which is 154.5% of the \$100,000 account.

**Step 3 — the borrowing.** You only have \$100,000 of your own. To hold \$154,500 of assets you must **borrow \$54,500.** That borrowed money finances the extra bonds. You are now running at about 1.55× gross leverage.

**Step 4 — the result.** This levered portfolio has the same 10.1% volatility as the 60/40 — but the risk is split **50/50** between stocks and bonds, not 92/8. If stocks have a terrible year, you've capped their damage at half the portfolio's risk instead of nearly all of it. *That* is the trade risk parity offers: same total risk as a 60/40, but genuinely balanced, paid for with leverage.

*Leverage is the price of admission to a balanced risk budget at a useful return: you borrow to hold more of the quiet asset so it can finally pull its weight.*

There's a second-order effect of leverage that the simple "scale the vol up" story leaves out, and it sets a ceiling on how far you should push. Leverage scales your *arithmetic* return, but it does *not* scale your *compound* return one-for-one, because of a phenomenon called **volatility drag**: the more your portfolio bounces, the more the geometry of compounding eats into your realized growth. The growth rate of a levered book is roughly its levered arithmetic edge *minus* half its levered variance — so as you crank leverage L up, the gain term grows linearly but the drag term grows with L *squared*. Past a certain point, more leverage means *lower* compound growth, even before you account for the financing cost or the risk of a forced sale at the bottom. There is a growth-optimal leverage, it's finite, and it's lower than the maximum your broker will extend — the same math that governs [the Kelly criterion](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge). A risk-parity book run at maximum leverage is not maximizing anything except its odds of a margin call.

This is why serious risk-parity practitioners do not lever to match equity-like volatility (which Figure 3 showed takes ~250% gross). They typically run a moderate vol target — often in the 10–12% range, the ~155–185% gross zone — precisely so that the volatility drag stays small and a single bad month can't trigger the margin spiral. The leverage is a tool for *balance*, not for *maximum return*; the moment you treat it as a return amplifier, you've left risk management and entered the casino.

Figure 7 lays this whole build out in one picture, end to end.

![Horizontal bar chart with three rows showing the risk-parity capital allocation in dollars, the equal fifty-fifty risk contribution, and the levered exposures of forty two thousand dollars stocks and one hundred twelve thousand dollars bonds with fifty four thousand dollars borrowed marked at the hundred percent line](/imgs/blogs/risk-parity-sizing-equal-risk-not-equal-money-7.png)

Figure 7 is the complete recipe on the \$100,000 account: row one, the unlevered capital split in dollars (\$27,270 stocks / \$72,730 bonds); row two, the resulting equal 50/50 *risk* contribution; row three, the levered version that targets 10.1% volatility, where the bonds alone are \$112,400 and the gross book stretches to 155% of capital, with \$54,500 borrowed past the dashed 100% line. Read top to bottom, it's the entire method: equalize the risk, then lever the balanced whole up to the volatility you actually want.

## Scaling up: from two assets to a real multi-asset book

Two assets is the teaching case. Real risk-parity funds run stocks, government bonds, corporate credit, commodities, sometimes inflation-linked bonds and currencies — and the principle generalizes cleanly: choose weights so every asset (or every *risk bucket*) contributes the same risk. The arithmetic gets harder because correlations between many assets interact, so there's no clean closed-form inverse-volatility shortcut — you solve it numerically with a short iterative routine — but the *target* is unchanged: equal risk contributions across the lot.

Let's add a third asset class, commodities, with a higher volatility of about 20% and a moderate 0.30 correlation to stocks (commodities and equities both like a hot economy) but near-zero correlation to bonds. Watch what equal *capital* does to the risk split now.

![Stacked column chart for three asset classes stocks bonds and commodities, the first column equal one third capital each, the second column the resulting unequal risk split of forty five and fifty five percent, the third column the risk-parity capital weights of twenty sixty two and seventeen percent](/imgs/blogs/risk-parity-sizing-equal-risk-not-equal-money-4.png)

Figure 4 tells the three-asset story in three columns. The first column is equal *capital* — a third in each. The second column shows the *risk* that equal-capital split actually produces: about **40% stocks, 5% bonds, 55% commodities.** Equal money, wildly unequal risk again — the two high-vol assets (stocks and commodities) hog 95% of the risk while the bonds, a full third of the money, contribute a measly 5%. The third column is the **risk-parity capital weighting** that fixes it: roughly **20% stocks, 62% bonds, 17% commodities**, which produces an even 33/33/33 risk split. Same pattern as the two-asset case, scaled up: pile into the calm asset, trim the jumpy ones, and the risk evens out.

#### Worked example: a \$10,000,000 book in three asset classes

Now run it on the **\$10,000,000 book** — the institutional scale. We want equal risk from stocks, bonds, and commodities.

**Step 1 — the equal-capital trap.** Naively, a third each: \$3.33M stocks, \$3.33M bonds, \$3.33M commodities. Computing the risk split, that gives roughly **40% / 5% / 55%** — bonds, \$3.33M of capital, contribute almost nothing; commodities dominate. This is the 60/40 problem in triplicate.

**Step 2 — the risk-parity capital weights.** Solving for equal risk gives about 20.4% stocks, 62.4% bonds, 17.2% commodities. In dollars on the \$10M book (unlevered): **\$2,040,000 stocks, \$6,240,000 bonds, \$1,720,000 commodities.** Each now contributes exactly one-third of the risk.

**Step 3 — the vol and the leverage decision.** This unlevered mix is, again, low-vol — heavy in bonds — so its portfolio volatility lands well under our target. To run the \$10M book at, say, 10% volatility, you'd apply a leverage factor and scale all three legs up proportionally, borrowing the difference. If that factor were 1.5×, you'd be running \$15M of gross exposure on \$10M of capital, with the extra \$5M financed — and the *biggest single position in the book would be levered government bonds*, the asset everyone calls "safe."

*The more asset classes you add, the more the risk piles onto whichever ones are most volatile under equal capital — and the more aggressively risk parity has to overweight, and then lever, the calm ones to even it out.*

That last point in step 3 is the one to sit with. In a fully built risk-parity fund, the *largest* position by dollars is almost always **levered long-duration government bonds**. That is a profound bet — not on bonds being safe in isolation, but on bonds *diversifying* stocks and on the cost of leverage staying low. It is a bet that paid off spectacularly for thirty years, for a specific historical reason we have to confront honestly.

There's a structural refinement that real funds layer on top of plain equal-risk-contribution, and it's worth knowing because it changes the failure modes. Sophisticated risk-parity managers don't just equalize risk across *assets* — they equalize it across **risk factors** or **economic regimes**. The idea: don't ask "is each asset contributing equal volatility?" but "is each *economic environment* — growth-up, growth-down, inflation-up, inflation-down — equally hedged?" You hold assets that do well in each quadrant and size them so no single environment can sink you. This is the "all-weather" framing, and it's more robust than naive volatility parity because it targets the *driver* of correlation rather than the correlation itself. But it leans on the *same* leverage and the *same* assumption that the four environments are genuinely distinct — and an inflation shock that hits stocks and bonds together collapses two of the four quadrants into one, which is exactly the 2022 failure. The factor framing is better engineering; it is not an escape from the core dependency. (The allocation-seat version of this regime-by-regime construction is [all-weather and risk parity](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime).)

There's also a practical cost the clean math ignores: **turnover**. Risk parity is not a set-and-forget allocation. As volatilities and correlations drift, the equal-risk weights drift with them, so the book has to be rebalanced — and every rebalance pays spreads, commissions, and, on the levered legs, financing roll. In a volatile period the weights can move fast (vol spikes, so the high-vol asset's risk share jumps, so you trim it), generating turnover exactly when transaction costs are highest. A live risk-parity strategy is a continuous re-sizing machine, and the drag from that machinery is a real subtraction from the elegant backtest. The same tension shows up in its close cousin, [volatility targeting](/blog/trading/risk-management/volatility-targeting-sizing-by-risk-not-by-dollars), where chasing a fixed vol target generates its own turnover bill.

## The honest part: what risk parity assumes

Every elegant strategy hides its assumptions in plain sight, and risk parity's are unusually load-bearing. Three of them, and each one fails in exactly the regime where you most need the portfolio to hold together.

![Before-and-after diagram with three pillars risk parity assumes on the left, stable correlations, cheap available leverage, and no shared shock, each connected by an arrow to its failure mode on the right, correlation goes to one, leverage gets dear or pulled, and a shared shock hits all legs](/imgs/blogs/risk-parity-sizing-equal-risk-not-equal-money-5.png)

Figure 5 lays out the three pillars and how each one cracks. Let's take them one at a time.

**Assumption 1: correlations are stable (and stocks and bonds stay diversifiers).** The entire benefit of holding levered bonds alongside stocks comes from the assumption that when stocks fall, bonds rise — or at least don't fall with them. That low-to-negative stock-bond correlation is the foundation of the whole structure. But correlation is a *regime*, not a constant. For most of 1998–2020 the stock-bond correlation was negative: a disinflationary world where every growth scare brought rate cuts and a bond rally. When that flips positive — as it did, hard, in 2022 — the bonds stop cushioning the stocks and start falling *with* them. And because you've *levered* the bonds, a falling bond position doesn't just fail to help; it actively makes things worse. The diversification you sized around evaporates precisely when you need it. (This failure mode — the hedge vanishing in the crisis — is its own subject in [when correlation goes to one, the diversification that vanishes in a crisis](/blog/trading/risk-management/when-correlation-goes-to-one-the-diversification-that-vanishes-in-a-crisis).)

**Assumption 2: leverage is cheap and always available.** Risk parity *requires* leverage — usually well over 100% bond exposure — and it assumes you can finance that cheaply and roll it indefinitely. Two ways this breaks. First, the *cost*: when short rates rise, the borrow cost on your leverage rises, eating directly into returns. Risk parity was born and thrived in an era of near-zero short rates, where leverage was nearly free; in a high-rate world the financing drag is real money. Second, the *availability*: leverage runs on margin, and margin can be *called*. In a crisis, when your levered bond position is falling, your lender demands more collateral exactly when your portfolio can least afford it — forcing you to sell into the decline, which pushes prices down further, which triggers more calls. That is a margin spiral, and it is how leverage turns a bad day into a blow-up.

#### Worked example: the margin call on the levered book

Hold the levered book from before — **155% gross exposure** on the **\$100,000 account**, with **\$54,500 borrowed**. Now run a sharp bad week where both legs drop 8% together (a 2022-style shared shock in miniature).

**Step 1 — the equity hit.** Your gross exposure is \$154,500. An 8% drop across the book is 0.08 × \$154,500 = **−\$12,360.** That loss comes entirely out of *your* equity, not the lender's — the lender is owed \$54,500 regardless. Your equity falls from \$100,000 to about **\$87,640.**

**Step 2 — the leverage ratchets up.** Your gross exposure is now about \$142,140, but your equity is only \$87,640, so your leverage has *risen* from 1.55× to about **1.62×** — without you doing anything. This is the cruel arithmetic of leverage: losses automatically *increase* your leverage, because the debt is fixed while the equity shrinks. The position gets riskier exactly as it goes against you.

**Step 3 — the forced sale.** If your broker's maintenance requirement caps you at, say, 1.55× and you've drifted to 1.62×, you get a **margin call**: sell assets to bring leverage back down. You must liquidate into a falling market, at the worst prices, locking in the loss and shrinking the position that needs to recover. Each forced sale by you (and everyone else levered the same way) pushes prices down further, triggering the next call. That is the spiral.

*Leverage doesn't just amplify a loss — it amplifies your leverage, so a shared shock forces you to sell at the bottom, converting a paper drawdown into a realized, unrecoverable one.*

**Assumption 3: no shared shock hits every leg at once.** Risk parity's whole bet is that something is always rising to offset what's falling — different assets, different risks, different drivers. But some shocks hit *everything*. An inflation shock raises the discount rate on stocks *and* crushes bond prices *and* sometimes hits commodities too. A funding shock — a dash for cash — makes investors sell *whatever they can*, correlating everything to 1 on the way out the door. In those moments there is no offsetting leg; every position is red at once, and the leverage that was meant to amplify a balanced return amplifies a synchronized loss instead. (That correlations stampede to 1 in a liquidity crisis is the deeper risk story; the allocation-level version lives in [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).)

These aren't theoretical caveats. All three cracked simultaneously in 2022, and the result is the cleanest real-world stress test risk parity has ever faced.

## The failure mode: 2022, when stocks and bonds fell together

For most of risk parity's life, the stock-bond correlation was the strategy's best friend. From the late 1990s through 2020, when stocks got scared, bonds rallied — the central bank cut rates, bond prices rose, and the levered bond leg of a risk-parity book delivered exactly the offset it was designed to. The strategy looked like genius, and a great deal of that "genius" was really one enormous, decades-long bond bull market plus a reliably negative stock-bond correlation. Risk parity was, in a real sense, *born of* that regime.

Then 2022 happened. Inflation surged to multi-decade highs, central banks hiked rates at the fastest pace in a generation, and the discount-rate shock hit stocks and bonds *at the same time*. Stocks fell because higher rates lower the present value of future earnings. Bonds fell because their prices move inversely to yields, and yields rocketed. For the first time in a long while, the two main legs of a balanced portfolio went down *together* — and the stock-bond correlation, negative for a generation, flipped positive.

![Two panel chart, the left panel showing 2022 calendar returns of US stocks at minus eighteen percent, US bonds at minus thirteen percent, a 60/40 at minus sixteen percent, and levered risk parity at minus twenty three point five percent, the right panel showing the stock-bond correlation flipping from minus zero point three in the diversifier era to plus zero point five five in 2022](/imgs/blogs/risk-parity-sizing-equal-risk-not-equal-money-6.png)

Figure 6 shows the damage and the cause side by side. On the left, the 2022 calendar-year returns: US large-cap stocks fell about 18%, US aggregate bonds about 13%, so a 60/40 lost roughly 16% — a painful year, but bounded. The *levered* risk-parity book lost *more* — illustratively about −23% to −24% — because its biggest single position was levered bonds, and bonds were one of the things falling, with the borrowing cost adding insult. On the right is the why: the stock-bond correlation, which had sat negative through the diversifier era, flipped to clearly positive in 2022. The offset that risk parity sized around didn't just weaken; it reversed.

#### Worked example: the 2022 stress test on a levered book

Take the levered risk-parity exposures from before — about **42% stocks, 112% bonds** on a \$100,000 account (155% gross, \$54,500 borrowed) — and run 2022's returns through them.

**Step 1 — the gross asset returns.** Stocks at −18%: 0.42 × (−18%) = −7.6%. Bonds at −13%: 1.12 × (−13%) = −14.6%. Both legs red. Gross portfolio return ≈ **−22.2%.**

**Step 2 — the financing cost.** You borrowed \$54,500. At an average 2022 short rate of roughly 2.5%, the financing drag is 0.545 × 2.5% ≈ **−1.4%.** Net ≈ **−23.5%.**

**Step 3 — compare to the supposedly worse 60/40.** The plain 60/40 lost about −16% the same year. The "balanced," "diversified," "lower-risk" levered risk-parity portfolio lost about **−23.5%** — *seven and a half points more*. The leverage that was supposed to deliver a balanced return delivered an amplified, synchronized loss, because the diversification it depended on had vanished.

*Leverage is a multiplier, and a multiplier has no opinion about direction: when the diversification fails and both legs fall, the same leverage that smoothed the good years magnifies the bad one.*

This is the most important lesson in the post, and it ties straight back to the survival spine. A −23.5% year, by the [asymmetry of losses](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain), needs about a +31% gain to recover — versus the +19% needed to climb out of the 60/40's −16%. The strategy sold as *lower risk* produced a *deeper* hole in the one regime it wasn't built for. Risk parity didn't fail because the math was wrong; the risk decomposition was perfectly correct. It failed because the *inputs* — the correlations — were estimated from a regime that ended, and the leverage turned that estimation error into a real loss. That is the recurring tragedy of every model-driven sizing method: the model is exactly as trustworthy as its assumption that tomorrow looks like the data it was fit on.

There's a deeper, almost philosophical point hiding in 2022, and it's the one to take away. Risk parity's pitch was that it freed you from forecasting — you don't need to predict which assets will do well, you just balance the risk. But "balance the risk" *is itself a forecast*: it's a bet that the past covariance structure — how the assets co-moved — will persist. When that covariance structure broke, the strategy had no edge left to fall back on, because it had explicitly disclaimed any view on returns. A discretionary 60/40 holder in 2022 at least *knew* they were mostly long equities and could brace for an equity-style year; the risk-parity holder thought they owned a balanced book right up until both halves caved in. There's no free lunch in the literal sense the series uses that phrase: survival is the only free lunch, and survival here meant having a de-risk rule for the moment the correlations betrayed the model — not trusting the model to be right.

It's also worth being fair to the strategy: 2022 was a *single calendar year*, and over the full arc since the early 1990s, levered risk-parity approaches delivered competitive risk-adjusted returns with shallower equity-driven drawdowns than a 60/40 in the recessions that *did* feature flight-to-quality bond rallies (2000–02, 2008, 2020's first leg). One regime doesn't condemn a method any more than one decade anointed it. The correct conclusion isn't "risk parity is broken"; it's "risk parity, *run statically at high leverage*, is fragile to a correlation regime change — and the fix is a leverage cap and a de-risk trigger, not abandonment."

## Common misconceptions

**"Risk parity is the same thing as 60/40, just relabeled."** No — they are nearly opposite portfolios. A 60/40 holds 60% stocks; an unlevered risk-parity mix of the same two assets holds about **27%** stocks and **73%** bonds, and a levered one holds **112%** bonds on borrowed money. The 60/40 is 92% equity risk; risk parity is 50/50. They share a goal (balance) but the 60/40 only balances the *money*, which leaves the *risk* radically unbalanced.

**"Leveraging bonds is reckless — the safe move is to avoid leverage."** This conflates two different risks. The point of levering the bonds is to make a *more balanced* risk budget reach a useful return; the unlevered version is genuinely low-risk but also low-return. The real danger isn't leverage per se — it's leverage *plus a correlation that can flip*. A levered bond position is fine while bonds diversify stocks; it becomes dangerous the moment bonds start falling *with* stocks. The 2022 levered book lost about **−23.5%** versus the 60/40's **−16%** — not because leverage is inherently reckless, but because the diversification it leaned on disappeared.

**"Risk parity removes the need to forecast returns."** It removes the need to forecast *expected returns* — that's true and genuinely valuable, because nobody forecasts those well. But it quietly *replaces* that with a forecast of **risk and correlation**, which it assumes are stable and estimable. That's a softer forecast, but it is still a forecast, and 2022 showed it can be wrong in the worst way. You haven't escaped prediction; you've moved it from the return inputs to the covariance inputs.

**"Equal risk contribution means each asset has an equal chance of hurting you."** Not quite. Equal risk contribution means each asset contributes equally to the portfolio's *volatility* under the estimated covariances. It says nothing about *tail* behavior — fat tails, skew, and the tendency of correlations to converge in a crash. Two assets can have identical risk contributions in calm times and then both crater together in a panic, which is exactly the [fat-tail and correlation-to-one](/blog/trading/risk-management/fat-tails-and-the-normal-distribution-trap) problem that volatility-based risk parity is blind to.

**"If it blew up in 2022, the strategy is broken — abandon it."** Also wrong, in the other direction. One bad regime doesn't invalidate a sizing principle any more than one good decade validated it. The lesson of 2022 isn't "risk parity is bad"; it's "any sizing method built on estimated correlations needs a de-risk trigger for when those correlations break, and leverage needs a hard cap." The discipline survives; the naive, always-on, max-leverage version does not.

**"More leverage just means more return."** Leverage scales *both* return and risk, and it amplifies losses as readily as gains — a fact the continuous growth math makes precise (see [the Kelly criterion](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge)). Past the growth-optimal point, *more* leverage means *less* compound growth, because the volatility drag (and the financing cost, and the risk of a margin call at the bottom) overwhelms the extra expected return. There is an optimal leverage, and it is lower than "as much as the broker allows."

## How it shows up in real markets

**2008 — the regime risk parity was built for.** In the global financial crisis, stocks collapsed (the S&P 500 fell about 37% for the year) and government bonds *rallied hard* as the Fed slashed rates and investors fled to safety. A levered risk-parity book, heavy in those rallying bonds, came through 2008 far better than a 60/40 — its bond leg did exactly what the design intended. This is the win that built the strategy's reputation, and it's real: in a disinflationary recession where bonds diversify stocks, risk parity shines.

**2022 — the regime it wasn't built for.** As detailed above, the simultaneous stock and bond collapse (stocks ≈ −18%, aggregate bonds ≈ −13%) and the positive correlation flip turned the levered bond leg from a cushion into a second source of loss. Levered risk-parity strategies posted some of their worst years on record, underperforming even the plain 60/40 they were designed to beat. The mirror image of 2008: same strategy, opposite regime, opposite outcome.

**1998, LTCM — the cautionary tale about levered "diversified" books.** Long-Term Capital Management ran a heavily levered portfolio of convergence trades it believed were diversified — roughly **\$125 billion of assets on about \$4.7 billion of equity, about 25-to-1 leverage**, with gross derivative notionals near **\$1.25 trillion**. In the 1998 flight to quality, the "diversified" positions all moved against the fund *together* — correlations went to 1 — and the leverage that had smoothed years of returns vaporized about **\$4.6 billion** of capital in roughly four months, requiring a Fed-organized rescue. LTCM wasn't risk parity, but it is the canonical lesson risk parity must heed: *leverage plus a correlation assumption that fails is the classic blow-up*. (The strategic, crowded-trade angle on LTCM lives in [the LTCM case study](/blog/trading/game-theory/case-study-ltcm-1998-the-crowded-genius-trade).)

**2020 — the dash for cash.** In the COVID crash of February–March 2020, the VIX hit a record close of **82.69** and, for a few terrifying days in mid-March, *everything* sold off at once — stocks, corporate bonds, even gold and Treasuries briefly — as investors raised cash indiscriminately. Correlations went to 1 across the board. Even before the Fed's intervention restored order, those days were a live demonstration that "shared shock" is not a hypothetical: in a true liquidity panic, the diversification every multi-asset strategy relies on can blink out simultaneously, levered or not. Several levered risk-parity funds cut exposure sharply during those weeks — the right move, and a vindication of the de-risk discipline — but the funds that cut *fastest* fared best, which is the whole argument for a pre-committed trigger over a discretionary "let's wait and see."

**2024 — the yen-carry unwind.** On **5 August 2024**, a crowded funding-carry trade — borrowing cheaply in yen to buy higher-yielding assets elsewhere — unwound violently as the yen surged. The Nikkei fell about **−12.4%** in a single day, its worst since 1987, and the VIX spiked intraday toward **65.7**. It wasn't a risk-parity blow-up, but it's the same family of failure: a *levered, crowded* trade where everyone's positioning was correlated, so the exit was a stampede and the leverage forced synchronized selling. The lesson generalizes to any levered, balanced-looking book: your true risk isn't just your own correlations, it's whether *everyone else* sized the same way and will be forced through the same door at the same time. (That crowding dimension — who's on the other side when you need to de-lever — is the subject of [crowded trades and the exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game).)

## The risk parity playbook

Strip away the math and the history, and risk parity reduces to a handful of concrete, survival-first rules. Here is how to actually use it without letting it use you.

**Size by risk contribution, not by capital — always.** Whatever your assets, compute each one's risk contribution before you accept an allocation. If one position is contributing 80–90% of your portfolio risk on a fraction of the capital — the 60/40 signature — you are not diversified, you are concentrated and don't know it. *Measuring* the risk split is valuable even if you never run full risk parity; it tells you where your real exposure lives.

**Decide on leverage deliberately, and cap it hard.** The unlevered risk-parity portfolio is balanced but sleepy; leverage is how you reach a useful return. But leverage is also the thing that converts a failed correlation assumption into a blow-up. Set a *maximum* gross leverage (many practitioners cap risk-parity books well below the 2–3× that pure vol-targeting would suggest) and never let the vol-target math push you past it. Remember Figure 3: matching a 60/40's risk already takes ~155% gross; chasing equity-like returns takes ~250%, which is where margin spirals live.

**Stress-test against a shared-shock, correlation-to-1 regime — not just the historical covariance.** Before you size, ask: what happens to this book if the stock-bond correlation flips to +0.6 and stays there? If both legs fall 15% together? Run the 2022 scenario explicitly. If the levered version loses more than your maximum-drawdown limit in that stress, the leverage is too high regardless of what the calm-times covariance says.

**Define the de-risk trigger in advance.** Risk parity's fatal flaw is that it's *always on* with *static* correlation assumptions. Fix that with a pre-committed rule: if the realized stock-bond correlation turns persistently positive, or if realized portfolio volatility exceeds your target by some margin (vol-targeting and risk parity are natural partners — see [volatility targeting](/blog/trading/risk-management/volatility-targeting-sizing-by-risk-not-by-dollars)), cut the leverage. De-risking when the diversification breaks is the single most important discipline; it's the difference between a drawdown and a ruin.

**Treat the financing cost as a real, rising input.** In a near-zero-rate world, leverage was almost free and the math flattered the strategy. When short rates are high, the borrow cost on 50%+ of your book is a meaningful drag every single year — model it, and re-evaluate whether the levered version still beats the unlevered one after costs.

**Remember what this is *for*.** Risk parity is a portfolio-level *sizing discipline* — a way to split *risk* across positions so no single asset secretly dominates. It is not a single-position stop, and it is not an allocation thesis about which assets to own. For the allocation-and-regime view of the same idea — owning every regime through a balanced risk budget — see [all-weather and risk parity](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime); for the optimization theory behind balancing risk and return, see [the mean-variance efficient frontier](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants). Here, the whole point is the survival spine: a genuinely balanced risk budget keeps any one position from blowing up the book — *as long as you respect the leverage and de-risk when the correlations that hold it together start to fail*.

### Further reading

- [Volatility targeting: sizing by risk, not by dollars](/blog/trading/risk-management/volatility-targeting-sizing-by-risk-not-by-dollars) — the single-portfolio cousin of risk parity: scale total exposure to hit a target volatility, the natural de-risk mechanism for a risk-parity book.
- [When correlation goes to one: the diversification that vanishes in a crisis](/blog/trading/risk-management/when-correlation-goes-to-one-the-diversification-that-vanishes-in-a-crisis) — the failure mode that breaks risk parity, treated head-on as a risk problem.
- [The Kelly criterion: how much to bet when you have an edge](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge) — the growth-optimal leverage that caps how far you should scale a risk-parity book.
- [All-weather and risk parity: owning every regime](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime) — the allocation-and-regime view of the same strategy, from the asset-allocation seat.
- [The mean-variance efficient frontier](/blog/trading/math-for-quants/mean-variance-efficient-frontier-math-for-quants) — the optimization math underneath balancing risk against return.

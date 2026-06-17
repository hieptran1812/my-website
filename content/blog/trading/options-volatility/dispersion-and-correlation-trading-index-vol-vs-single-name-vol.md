---
title: "Dispersion and Correlation Trading: Index Vol vs Single-Name Vol"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Build correlation intuition from zero, learn why index vol is cheaper than single-name vol, and see how the dispersion trade harvests the correlation risk premium until correlations go to one."
tags: ["options", "volatility", "dispersion", "correlation", "index-vol", "variance-swap", "correlation-risk-premium", "vega", "skew", "tail-risk"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — An index is less volatile than its average component, and the size of that gap is, by definition, the correlation between the components; dispersion trades that gap, selling rich index vol and buying cheaper single-name vol so that you are, net of everything else, short implied correlation.
>
> - Index variance is a weighted sum of the component variances **plus** every pairwise covariance. Diversification cancels some of the single-name vol, and what survives is the correlation. When correlation is low, index vol is far below average single-name vol; when correlation goes to one, the gap vanishes and index vol equals the weighted-average single-name vol.
> - You can back the market's **implied correlation** out of the prices: take index implied vol, divide by the weighted-average single-name implied vol, square it. Roughly, implied correlation ≈ (index vol / average single-name vol)².
> - Index vol is **structurally rich** because institutions hedge the index, not 500 single names — that one-sided put demand bids index vol and skew, so implied correlation prints above the correlation that realizes. That gap is the **correlation risk premium**, a cousin of the variance risk premium.
> - The dispersion trade (short index vol, long single-name vol, vega-matched) harvests that premium for years, then loses badly in a crash when correlations snap to one, index vol explodes, and the short-correlation leg pays the claim. **Size for the crash, not the carry.**

A dispersion desk at a large bank ran the same trade for the better part of a decade. Every month they sold volatility on the S&P 500 index and bought volatility on a basket of the individual stocks inside it, sized so the two legs cancelled on the overall level of volatility. They were not betting that vol would rise or fall. They were betting on something subtler: that the stocks would keep moving on their own news — one up on earnings, another down on a downgrade, a third flat — so that the index, averaging all that idiosyncratic churn, would stay calmer than any single name. Month after month it worked. The index realized less volatility than the basket, the short-index leg printed more than the long single-name leg cost, and the desk booked a steady, almost boring profit. Risk managers called it carry.

Then the carry stopped being boring. In the autumn of 2008, and again in March 2020, the thing the trade was implicitly short — correlation — went vertical. Every stock fell at once. Banks, airlines, tech, staples: it did not matter what a company did, only that it was equity and equity was for sale. The diversification that had kept the index calm evaporated, because there was nothing left to diversify; the stocks were all the same trade. Index volatility, which lives off correlation, blew out far faster than single-name volatility. The short-index leg that had paid the rent for years took a loss that ate several years of carry in a few weeks. The desk had been collecting a premium for warehousing exactly this risk, and the bill came due.

That is dispersion trading, and to understand it you only need one idea, built from zero: **the volatility of a basket depends not just on how volatile its pieces are, but on how much they move together.** That "move together" is correlation, and the whole trade is a wager on it. Let us build the intuition before any of the desk machinery.

![Index vol against average single-name vol as correlation rises from zero to one, with the gap shaded as the diversification benefit](/imgs/blogs/dispersion-and-correlation-trading-index-vol-vs-single-name-vol-1.png)

## Foundations: why a basket is calmer than its pieces

Start with something you already feel in your gut, with no finance in it at all. You run a small delivery company with two drivers. On any given day each driver might be slow — traffic, a flat tyre, a wrong turn — so each one's delivery time is, on its own, quite variable. But if you average the two drivers' times, the average is steadier than either driver alone, because on a day one is slow the other is often fine. The averaging smooths out the bumps. That smoothing is **diversification**, and it is the most reliable free lunch in all of finance.

But notice the hidden assumption. The averaging only smooths the bumps *if the two drivers' bad days are not the same days.* If both drivers hit the same blizzard — one cause that slows them both at once — averaging buys you nothing; both are slow together, and the average is just as slow. The amount of smoothing you get depends entirely on **how much the drivers' problems are shared versus independent.** That shared-versus-independent split is the single idea behind this entire post. When the problems are independent, the average is calm. When the problems are common — a blizzard, a recession, a panic — the average is as wild as the parts. Correlation is simply the technical name for "how much are the problems shared."

Hold onto the blizzard. A stock index is a fleet of hundreds of drivers. Most days, each company has its own private weather: one beats earnings, another loses a lawsuit, a third announces a buyback. Those private events are independent, they partly cancel in the average, and the index glides along calmer than any single stock. But some days there is a blizzard for everyone — a rate shock, a credit event, a pandemic — and on those days nothing cancels, every stock skids together, and the index is suddenly as violent as its components. The whole trade we are building is a wager on how often, and how hard, the blizzard comes.

Now make it precise. A stock's **volatility** is the standard deviation of its returns, annualized, quoted in "vol points" (a stock at 30% vol typically moves about 30%/√252 ≈ 1.9% on a given day). An index like the S&P 500 is a weighted basket of stocks: its return is the weighted average of the component returns. The question dispersion lives on is: **given how volatile each component is, how volatile is the basket?** And the answer, you can already guess, is "it depends on how much they move together."

The answer is the single most important identity in this whole post. For a basket with weights `w_i` and component volatilities `σ_i`, the index variance (variance = vol squared) is:

```
sigma_index^2  =  sum_i  w_i^2 sigma_i^2     (the "own" variance of each name)
              +  sum_{i != j}  w_i w_j rho_ij sigma_i sigma_j   (every pairwise co-movement)
```

Read it slowly, because everything follows from it. The index variance is **not** just the weighted average of the component variances. It is that, **plus a sum over every pair of stocks of how much they move together** — and "how much they move together" is exactly the pairwise correlation `rho_ij` scaled by the two vols. The diagonal terms (`w_i² σ_i²`) are the parts of each stock's risk that nobody else shares. The off-diagonal terms (the covariances) are the shared, undiversifiable risk. The intuition here is that diversification cancels the diagonal but cannot touch the off-diagonal: you can average away the idiosyncratic wiggles, but you cannot average away the days when everything moves as one.

If you have met the covariance matrix in a linear-algebra context, this is exactly `wᵀ Σ w`, the variance of a weighted sum, written out in components. (See [the covariance matrix](/blog/trading/math-for-quants/covariance-matrix-linear-algebra-math-for-quants) for the matrix algebra; here we only need the intuition.)

Two consequences fall straight out, and they are the entire trade:

1. **If correlation is low, the off-diagonal terms are small, so index vol is far below the average single-name vol.** The basket diversifies hard. This is the calm-market state.
2. **If every correlation goes to one, every off-diagonal term is at its maximum, the variance identity collapses to a perfect square, and index vol equals the weighted-average single-name vol.** There is nothing left to diversify. This is the crash state.

Index vol therefore lives in a band: between (very low, set by the diagonal alone) when correlation is zero, and (the weighted-average single-name vol) when correlation is one. **Where in that band index vol sits IS the correlation.** That is not a metaphor or an approximation — it is the literal content of the identity. The gap between average single-name vol and index vol is the visible fingerprint of how correlated the components are, which is exactly what the cover chart above plots: as correlation climbs from 0 to 1, index vol climbs from the diversified floor right up to the average single-name line, and the amber gap that is the diversification benefit shrinks to nothing.

### The two-stock case, where you can see every term

The full identity over 500 names is a mouthful, so collapse it to two stocks, equal weight, and you can hold every term in your head.

#### Worked example: index vol of two 30%-vol stocks at different correlations

Two stocks, A and B, each with volatility `σ = 30%`, equal weight `w = 0.5`. The two-asset variance identity is:

```
sigma_index^2 = w^2 sigma_A^2 + w^2 sigma_B^2 + 2 w w rho sigma_A sigma_B
            = 0.25(0.30^2) + 0.25(0.30^2) + 2(0.5)(0.5) rho (0.30)(0.30)
            = 0.0225 + 0.0225 + 0.045 rho
            = 0.045 (1 + rho)
```

Now plug in correlations:

- **rho = 0 (independent):** `σ_index² = 0.045`, so `σ_index = √0.045 = 21.2%`. The two 30%-vol stocks combine into a 21.2%-vol index. You shed nearly **9 vol points** to diversification.
- **rho = 0.30 (typical calm):** `σ_index² = 0.045 × 1.30 = 0.0585`, so `σ_index = 24.2%`. Still well under 30%; you keep about **5.8 vol points** of diversification.
- **rho = 0.50:** `σ_index = √(0.045 × 1.5) = 26.0%`.
- **rho = 1.0 (crash):** `σ_index² = 0.045 × 2 = 0.090`, so `σ_index = √0.090 = 30.0%` — exactly the single-name vol. The diversification is gone.

Pause on the rho = 0 result, because it is the cleanest demonstration of the free lunch. Two stocks, each genuinely a 30%-vol asset on its own, combine into a 21.2%-vol index — nearly **nine vol points** vanish into thin air, paid for by nothing except the fact that the stocks are independent. No one gave up return; the basket simply averages two unrelated wiggle-streams and the wiggles partly cancel. That is the magic of diversification, and it is also exactly what the dispersion seller is short: when you sell index vol and buy single-name vol, you are *selling* those nine vanished vol points. As long as they keep vanishing (low correlation), you keep the difference. The day the stocks stop being independent, the nine points reappear in the index — and you are the one who has to produce them.

The takeaway in one line: at 30% correlation the index runs 5.8 vol points calmer than its components, and that 5.8-point cushion is the diversification you are long when you are short the index and long the names — it is the prize and the trap at once.

This same arithmetic, drawn as a curve, is the figure below: index variance climbing toward the corr=1 ceiling as the amber diversification benefit narrows.

![Two-stock index variance against correlation, climbing to the single-name ceiling as the diversification benefit shrinks to zero](/imgs/blogs/dispersion-and-correlation-trading-index-vol-vs-single-name-vol-4.png)

Notice what the figure says that the table of numbers does not. The benefit of diversification is not constant — it is **largest when correlation is lowest and shrinks fastest as correlation climbs toward one.** The amber gap is fat on the left and pinched closed on the right. This is the geometric heart of why dispersion is a tail trade: most of the time you sit in the comfortable middle, collecting the gap; but the gap you collect is exactly the thing that evaporates, all at once, on the right edge. You are short the very cushion that protects you, and the cushion is thinnest precisely when you would most want it.

### Why we measure in variance, not vol

One technical habit makes everything cleaner: **work in variance (vol squared), not vol.** Variance is additive in a way vol is not. The identity above is linear in the covariances; the standard deviation (vol) is the square root, which is non-linear and harder to reason about. Almost every serious dispersion book is run on **variance swaps** — instruments that pay the difference between realized variance and a strike, in vol-points-squared — precisely because variance is the natural, additive unit. We will price the trade in variance for that reason, and translate back to vol points for intuition. Keep this in your back pocket: **the index's variance is a clean linear function of correlation; its vol is the messier square root.** That non-linearity is where the crash convexity hides.

### From two stocks to five hundred: why large N sharpens the bet

The two-stock case is where the intuition lives, but the real S&P 500 has hundreds of names, and the count itself matters. Take the simplest honest generalization: `N` names, all with the same single-name vol `σ_s`, all equally weighted at `w = 1/N`, and all sharing the same average pairwise correlation `rho`. The index variance collapses to a single clean expression:

```
sigma_index^2 = rho * sigma_s^2  +  (1 - rho) * sigma_s^2 / N
```

The first term, `rho · σ_s²`, is the **systematic** piece — the shared, undiversifiable risk that every name carries. The second, `(1 − rho) · σ_s² / N`, is the **idiosyncratic** piece — the part each name owns alone, divided by `N` because averaging over many names crushes it.

Now watch what `N` does. With `N = 2` the idiosyncratic term is `(1 − rho)σ_s²/2`, a meaningful chunk. With `N = 50` it is `(1 − rho)σ_s²/50`, almost nothing. With `N = 500` it is essentially zero. **As the basket gets large, the idiosyncratic risk diversifies away entirely, and the index variance becomes `rho · σ_s²` — pure correlation times single-name variance.** This is the deep reason index vol is a near-clean read on correlation: in a big index there is nothing left *but* the correlation. The index vol is `σ_s · √rho`, almost exactly.

#### Worked example: the same index, two basket sizes

Single-name vol 30%, correlation 0.30. Compare a 2-name basket to a 50-name basket.

```
N = 2:   sigma_index^2 = 0.30 * 0.09  +  0.70 * 0.09 / 2   = 0.027 + 0.0315 = 0.0585  -> 24.2%
N = 50:  sigma_index^2 = 0.30 * 0.09  +  0.70 * 0.09 / 50  = 0.027 + 0.00126 = 0.0283 -> 16.8%
large N: sigma_index    ~  0.30 * sqrt(0.30) = 0.30 * 0.548 = 16.4%
```

With only two names the index runs 24.2% — still carrying a fat dollop of idiosyncratic risk. With fifty names it falls to 16.8%, and in the large-`N` limit it converges to 16.4%, almost entirely set by `σ_s · √rho`. The lesson: **the more names in the basket, the more cleanly index vol reflects correlation alone, and the more a dispersion trade is a pure correlation bet rather than a residual-idiosyncratic-risk bet.** This is why real dispersion is run on broad indices, not pairs.

## Implied correlation: reading correlation out of option prices

So far correlation has been a property of how stocks *actually* move — realized correlation. But options are forward-looking. An option's price embeds the market's forecast of future volatility: the **implied volatility**. If you know the implied vol of the index and the implied vols of all the components, you can run the variance identity *backwards* and solve for the one unknown left in it: the correlation the market is pricing. That number is **implied correlation**, and it is the heart of the trade.

Here is the back-out. Suppose, to keep it clean, that every name has the same correlation `rho` with every other name (the "average pairwise correlation" — a standard simplification, and the basis of CBOE's published implied-correlation indices like COR3M and the older ICJ). Then the index variance is:

```
sigma_index^2 = rho * (sum_i w_i sigma_i)^2  +  (1 - rho) * sum_i w_i^2 sigma_i^2
```

The first term is what the index variance *would* be if all the stocks were perfectly correlated (the square of the weighted-average vol); the second is the diversification haircut, which shrinks as `rho → 1`. Solve for `rho`:

```
rho_implied = ( sigma_index^2  -  sum_i w_i^2 sigma_i^2 )
            / ( (sum_i w_i sigma_i)^2  -  sum_i w_i^2 sigma_i^2 )
```

For a large index the diagonal term `Σ w_i² σ_i²` is tiny (each weight squared is minuscule when you have 500 names), so this collapses to a rule of thumb worth memorizing:

```
rho_implied  ~  ( sigma_index / weighted-average single-name sigma )^2
```

Implied correlation is, near enough, **the square of the ratio of index vol to average single-name vol.** All the option market is telling you, when it quotes index vol below single-name vol, is how correlated it thinks the stocks will be.

#### Worked example: backing implied correlation out of the quotes

The desk pulls quotes one calm morning. The index 1-month implied vol is **18%**. The weighted-average 1-month implied vol of the components is **28%**. Using the rule of thumb:

```
rho_implied  ~  (0.18 / 0.28)^2  =  (0.643)^2  =  0.413
```

The market is pricing about a **41% average pairwise correlation**. Now the calm deepens and index vol drops to **16%** while single-name vol holds at 28%:

```
rho_implied  ~  (0.16 / 0.28)^2  =  (0.571)^2  =  0.327
```

Implied correlation has fallen to **33%** — the index got cheaper *relative to its components*, which is the market saying "the stocks will move more on their own news." Now the regime flips: a macro scare lifts index vol to **25%** while single-name vol rises only to 30%:

```
rho_implied  ~  (0.25 / 0.30)^2  =  (0.833)^2  =  0.694
```

Implied correlation has jumped to **69%**. The index got expensive relative to its components, because the market now expects the stocks to move together. The one-line lesson: **implied correlation rises when index vol rises faster than single-name vol, and that ratio is something you can read straight off the screen.**

This is what a CBOE-style implied-correlation index publishes daily: a single number, derived exactly this way, summarizing how correlated the options market thinks the S&P 500's members will be over the next month or quarter. In calm regimes it might print in the 20–40% range; in a crisis it can leap toward 80–90%. Figure 3 sketches that path across the regimes of the last two decades.

![Implied correlation across regimes, low in calm markets and spiking toward one in the 2008 and 2020 crises](/imgs/blogs/dispersion-and-correlation-trading-index-vol-vs-single-name-vol-3.png)

There is one subtlety to flag before we move on. The clean back-out assumed every pair shares the same correlation and used the same maturity for index and names. Real desks must match maturities, deal with the fact that the index skew is steeper than the single-name skew (so "implied correlation" depends on which strikes you read it from), and handle the difference between the published CBOE construction and a bespoke basket. But the core fact survives all of it: **index vol below average single-name vol is a quote on correlation, and you can extract that quote.**

### Correlation skew: it is not one number

There is a refinement that pays off the moment you trade this for real. Implied correlation is not a single number — it depends on *which strikes* you read it from, because index skew and single-name skew have different shapes. Index puts (downside protection) trade at a much higher implied vol than index calls, thanks to the hedging demand we will dissect next. Single-name skew is flatter. When you back out implied correlation from *downside* strikes you get a higher number than from *at-the-money* or *upside* strikes — a phenomenon desks call **correlation skew.** It encodes a simple market belief: stocks are expected to correlate *more in a sell-off* than in a rally. The market does not think correlation is constant; it thinks correlation is itself state-dependent, rising exactly when prices fall.

This matters for the trade in two ways. First, *where* you read implied correlation changes whether the premium looks fat or thin, so a serious desk quotes implied correlation as a term-and-strike surface, not a scalar. Second, it tells you the market has *already priced* the corr-to-one tail to some degree — the downside implied correlation is elevated precisely because everyone knows a crash synchronizes everything. So the naive "implied correlation 0.40, realized usually 0.30, free 0.10" story overstates the edge: a chunk of that gap is fair compensation for the downside-correlation risk the skew is flagging. Reading correlation off the at-the-money strikes (where the premium is cleanest) and respecting that the wings are pricing the tail honestly is part of trading this without fooling yourself.

## The dispersion trade: getting short implied correlation

Now the trade itself. You have two observations:

1. **Implied correlation is a number you can read** off index vol versus single-name vol.
2. **Index vol is structurally rich** — it tends to price a higher correlation than actually realizes (we will see why in the next section). So implied correlation sits, on average, *above* realized correlation.

If something is persistently overpriced, you sell it. To get short implied correlation, you do two things at once:

- **Sell index vol** — short the S&P 500 straddle, or sell an index variance swap. You collect the rich index implied vol. This leg is **vega-negative** (it loses if vol rises) and, crucially, it is the part of your book that is short correlation, because index vol *is* a correlation bet.
- **Buy single-name vol** — buy straddles on the components, or buy single-name variance swaps. You pay the cheaper single-name implied vol. This leg is **vega-positive**.

Then you **vega-match** the two legs: size the single-name basket so that its total vega roughly equals the index vega you are short. The point of vega-matching is that if implied vol simply rises or falls *across the board* — index and names together — the two legs cancel and you are flat. You did not want to bet on the level of vol. **What you are left exposed to is the relationship between the two: correlation.** Sell expensive index vol, buy cheap single-name vol, neutralize the level, and the residual is a clean short position in implied correlation.

![Balance figure showing short index vol plus long single-name vol netting to a short implied correlation position after vega matching](/imgs/blogs/dispersion-and-correlation-trading-index-vol-vs-single-name-vol-2.png)

Why does this net to short correlation, mechanically? Recall index vol = f(single-name vols, correlation). You are short index vol and long single-name vol in equal vega. Movements in the single-name-vol level cancel. The only driver of index vol you have not offset is correlation. So you make money when correlation comes in **lower** than the implied level you sold (index realizes calmer than the basket implies → your short index leg wins by more than your long single-name leg loses), and you lose when correlation comes in **higher**.

What does the residual position actually *feel* like day to day? On a quiet day where one stock jumps on earnings and another sags on a downgrade, the single-name straddles you own light up — they are long gamma, so they profit from those individual moves — while the index barely twitches, so the index straddle you are short barely costs you. You collect on the long single-name legs and pay almost nothing on the short index leg: the trade prints. On a day where the *whole market* lurches in one direction, the index straddle you are short moves hard against you while the single-name straddles, all moving the same way, give back only their share — the index move is the *coordinated sum*, and you are short it. So the position is, viscerally, **long idiosyncratic moves and short coordinated moves.** That is short correlation stated in the language of the tape, and it is why a dispersion book quietly loves an active, stock-picker's market and dreads a macro-driven, risk-on/risk-off market where everything trades as one ticket.

This connects directly to the trade at the heart of this whole series. Just as the simple short-vol trade is a bet that [implied volatility exceeds the realized volatility that follows](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options), dispersion is a bet that **implied correlation exceeds the realized correlation that follows.** It is the same edge — selling something the market overpays for — applied to correlation instead of volatility. And like every "the market overpays" trade, the question is always: *overpays in exchange for what risk?* The answer determines whether you survive.

### Why index vol is structurally rich: the correlation risk premium

Here is the engine of the whole trade, and it is a market-structure fact, not a pricing-theory one. **Institutions hedge the index, not the individual names.** A pension fund, an insurer, a long-only manager who is nervous about a drawdown does not go and buy puts on each of its 300 holdings — that is operationally absurd and expensive. It buys S&P 500 puts. The whole book is hedged in one trade. So there is a large, persistent, price-insensitive, **one-sided demand for index downside protection** that has no equivalent on the single-name side. (No one is forced to buy puts on every individual stock the same way.)

That one-sided demand does two things, both of which you have already met in this series. It **bids up index implied vol** generally, and it **bids up the index put skew** — the steep premium on out-of-the-money index puts that you read about in [the volatility smile and skew](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more). Single-name skew is much flatter, because single names lack that forced hedging buyer. So index implied vol is rich relative to what the components imply, and the *form* of that richness is a **high implied correlation.** Dealers who absorb all that index-put demand are warehousing correlation risk, and they charge for it.

![Cause figure tracing index-put hedging demand to a bid in index vol and skew and a correlation risk premium](/imgs/blogs/dispersion-and-correlation-trading-index-vol-vs-single-name-vol-5.png)

The gap between implied correlation (what you sell) and realized correlation (what actually happens) is the **correlation risk premium**. It is the direct sibling of [the variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt): both are premia the market pays sellers for bearing a risk buyers desperately want to offload. The VRP pays you for being short volatility; the CRP pays you for being short correlation. Empirically the average S&P 500 1-month implied vol prints roughly **+3.7 vol points above the realized vol that follows** (≈19.5% implied vs ≈15.8% realized, the long-run gap behind the VRP). The correlation risk premium is the analogous, structurally positive gap between implied and realized correlation — and dispersion is the cleanest way to isolate and harvest it.

There is a clean way to hold the whole thing in one sentence: **a dispersion seller is renting out diversification.** The index-put hedgers want, above all, protection against the day diversification fails — the day everything falls together. They will overpay for it, because that is the day that hurts them most. The dispersion desk is the counterparty who *sells* that protection: it accepts the rich index vol (promising to pay when the index blows out) and offsets it with cheaper single-name vol, pocketing the spread. In the good months, diversification holds, the renter never files a claim, and the desk keeps the rent. In the crash, diversification fails, the claim comes in, and the desk pays. Framed this way the whole risk/reward is obvious: it is the economics of an insurer who has written hurricane policies along the coast. Most years, premiums in, nothing out. Then the hurricane. The skill is not in deciding whether to write the policy — the premium is real and positive — but in pricing it, diversifying the book, holding reinsurance for the big one, and never writing so much that a single storm is fatal.

#### Worked example: the correlation premium as a vol-points edge

Take the calm-morning quotes again: index implied vol 18%, weighted-average single-name implied vol 28%, so implied correlation ≈ 0.41. Suppose that over the month the names realize roughly the same 28% vol, but they move on their own news and the **realized correlation comes in at 0.30** rather than the 0.41 you sold. Using the same rule of thumb, the *realized* index vol is:

```
sigma_index_realized  ~  0.28 * sqrt(0.30)  =  0.28 * 0.548  =  15.3%
```

You sold the index straddle at an 18% implied vol; the index only realized about 15.3%. The short index leg captures roughly **18% − 15.3% = 2.7 vol points**, while your long single-name basket realizes about what you paid (28% ≈ 28%), so it is roughly flat. Net, you harvest about **2.7 vol points** of correlation premium on the index notional. That is the carry: small, steady, paid for as long as realized correlation undershoots implied. The intuition is that you are paid a few vol points a month to promise the market that, in a crash, you will hand back a multiple of that.

## The mechanics: how the trade is actually built

The clean "short index vol, long single-name vol" description hides the engineering, and the engineering is where most of the variance between a good dispersion book and a blown-up one lives. There are two ways to put the trade on, and they behave differently.

### Variance swaps versus straddles

**Variance swaps** are the purist's instrument. A variance swap pays you the difference between the realized variance of an underlying over a period and a fixed strike (the implied variance you traded at), in vol-points-squared, with no path-dependence and no need to delta-hedge. You short an index variance swap (collect index implied variance, pay realized) and buy a basket of single-name variance swaps. Because variance is additive, the trade's P&L is *exactly* the variance identity in action: short-index-variance P&L minus long-single-name-variance P&L equals a clean function of realized correlation. This is why the worked examples above are stated in variance terms — a variance swap literally settles in vol-points-squared. The downside is that single-name variance swaps are illiquid and OTC; many names simply do not have a tradeable variance swap, and the ones that do carry wide dealer markups.

**Straddles** (and strangles) are the practical, listed alternative. You sell an at-the-money index straddle and buy at-the-money straddles on the components. A straddle is long vol but also has delta, gamma, and theta, so this version of the trade is **not** a pure variance bet — it must be **delta-hedged** continuously, and its vol exposure decays as spot drifts away from the strikes (a straddle's vega is highest at the money and falls as the option moves in or out of the money). Run with straddles, dispersion is a daily delta-hedging operation, and your realized P&L depends on the *path*, not just the endpoint variance. Most desks use a blend: variance swaps where they trade, straddles or strangles where they must.

### Vega-weighting the basket

However you build it, the sizing rule is the same: **vega-match the legs at inception.** You compute the vega of the index leg (its sensitivity to a one-point change in implied vol) and size the single-name basket so the sum of the component vegas equals it. The point, again, is to be neutral to the *level* of vol. There is a subtlety in *how* you weight the single names: a "theoretical" dispersion weights each name by its index weight times its vol contribution; a "vega-flat" version weights to equalize vega. The choice changes how the residual behaves when single-name vols move relative to each other, and good desks tune it to the correlation view they actually hold rather than blindly equal-weighting.

#### Worked example: vega-matching a two-leg trade

Suppose your short index variance swap has a vega notional of **\$100,000 per vol point** (you lose \$100,000 for every point index vol realizes above strike, gain \$100,000 per point below). To be vega-neutral on the level, you buy single-name variance swaps whose vegas sum to **\$100,000 per vol point** as well — say five names at \$20,000 of vega each, chosen for liquidity and weight. Now if *all* vols rise by 3 points together (index and names), the short-index leg loses \$300,000 and the long single-name basket gains \$300,000: net zero, as designed. The trade only moves on the *spread* between index and basket vol — that is, on correlation. The intuition: vega-matching turns a two-legged vol position into a single-factor correlation position, but only at the instant you put it on; the match drifts the moment the market moves.

### The frictions that eat the edge

The clean math hides real costs that determine whether the premium is harvestable in practice:

- **Single-name liquidity.** You cannot trade options on all 500 S&P names; spreads on the small ones are wide. Desks trade a *proxy basket* of the most liquid 30–100 names, which introduces basket-vs-index tracking error: your long basket may not move like the index you are short.
- **Vega drift and re-balancing.** Vega-matching holds only at inception. As spots move and time passes, the index and single-name vegas drift apart and the book picks up an unintended long/short vol bias that must be re-balanced — at a cost in spreads each time.
- **Skew mismatch.** Index skew is steeper than single-name skew. A straddle-based dispersion trade is exposed to that, and a move can leave you short index downside (rich) and long single-name downside (cheaper) in a way that hurts when skew steepens. Many desks run the trade with variance swaps or carefully chosen strangles to manage this.
- **Dividends, corporate actions, and weights.** Single-name vols are affected by earnings dates and dividends that the index averages out; mergers and index reconstitutions change the basket under you.
- **Funding and balance sheet.** A large OTC variance-swap book consumes counterparty credit lines and balance sheet; the carry must clear those internal costs before it is profit.

None of these kill the trade, but they shave the premium and demand active management. The honest version: **the correlation risk premium is real, but a meaningful slice of it is the fee for doing this operationally hard thing well.**

## How it shows up in real markets

Theory is clean; the desk is messy. Here is how dispersion actually behaves, with the worked P&L and the two crashes that define the trade.

### The carry years, then the cliff

Consider a desk running large-N dispersion: a 50-name basket as a stand-in for the index, each name around 30% single-name vol, with the index straddle sold at an implied correlation of **0.40**. Using the homogeneous large-N identity, the implied index vol they sold is:

```
sigma_index_implied  =  0.30 * sqrt(0.40 + 0.60/50)  =  0.30 * sqrt(0.412)  =  19.3%
```

#### Worked example: dispersion P&L when correlation falls (the win)

The calm regime persists. Single-name vol realizes at 30% as priced, but the stocks move on their own news and **realized correlation comes in at 0.25**. The realized index vol is:

```
sigma_index_realized  =  0.30 * sqrt(0.25 + 0.75/50)  =  0.30 * sqrt(0.265)  =  15.4%
```

The desk sold the index at 19.3% implied and it realized 15.4%. The short-index leg wins about **3.8 vol points**; the long single-name legs realize what was paid, roughly flat. In variance terms — the unit a variance swap actually pays — the short-index leg captures `19.3² − 15.4² ≈ 372 − 238 = 134` vol-points-squared. A steady, repeatable harvest. Booked as carry, this is the trade that paid the desk's bonuses for years. The takeaway: **when realized correlation undershoots the implied correlation you sold, the index realizes calmer than its components and the short-index leg prints.**

#### Worked example: the same trade when correlation goes to one (the loss)

Now 2008, or March 2020. A macro shock hits; every stock is for sale at once; **realized correlation snaps to 0.95.** Single-name vol also rises, but the catastrophe is in the index leg, because index vol lives off correlation. With names now realizing, say, 30% still but correlation at 0.95, the realized index vol is:

```
sigma_index_realized  =  0.30 * sqrt(0.95 + 0.05/50)  =  0.30 * sqrt(0.951)  =  29.3%
```

The desk sold the index straddle at 19.3% implied and it realized **29.3%** — a **10-vol-point** loss on the short-index leg. In variance terms the damage is `19.3² − 29.3² ≈ 372 − 858 = −486` vol-points-squared. One bad month wipes out roughly the carry of *the entire calm period*, because the loss is the **square** of a large vol move while each month's carry was a few points. The single-name longs help a little — single-name vol also rose — but they do **not** offset the index leg, because the index blew out by *more* than the names did. That asymmetry is the trap: the very correlation that made the index calm in good times makes it explode in bad ones, and you were short exactly that. The one-line intuition: **dispersion's loss is convex in correlation while its gain is roughly linear, so the rare crash dwarfs the steady carry.**

![Dispersion P&L against realized correlation, a small win when correlation falls and a large convex loss when correlation approaches one](/imgs/blogs/dispersion-and-correlation-trading-index-vol-vs-single-name-vol-6.png)

Figure 6 draws this directly: a modest profit zone (green) as realized correlation comes in below the 0.40 sold, then a deep loss region (red) as correlation spikes toward one. The profit and loss are not symmetric around the breakeven; the downside runs far deeper than the upside, which is the signature of every short-premium trade.

### 2008 and 2020: when the off-diagonal swallowed everything

The two textbook blowups share one mechanism. In the 2008 global financial crisis, what had been a market of distinct sectors — financials, industrials, energy, tech — became a single risk asset. Everyone was deleveraging the same book at the same time, so every stock moved with every other. Realized correlation, normally lurking around 0.3–0.4, ran toward 0.8 and above. The index, which had been quietly absorbing offsetting single-name moves, now had no offsets to absorb, and its volatility exploded; the VIX closed near **80.86** at its November 2008 peak. In March 2020, the COVID shock did the same thing faster: a one-way, all-correlated liquidation across every name, the VIX closing at **82.69** on 16 March 2020 — the highest daily close on record. In both episodes the index's volatility blew out far more violently than the average single name's, because **the index's volatility is leveraged to correlation and correlation went to one.** Dispersion desks that had been short that correlation took their biggest losses of the cycle in those weeks.

It helps to see the asymmetry in the numbers the cover and P&L charts are built on. In a calm regime our 50-name, 30%-vol index sits near 16–19% — a 11-to-14-vol-point cushion below the components. The dispersion desk collects a slice of that cushion every month. But the cushion is finite: at correlation 1.0 it is *zero*, and the index vol equals the 30% single-name vol. So the most the short-index leg can ever lose to rising correlation is bounded by the move from ~19% implied to ~30% realized — about 11 vol points of vol, but **far more in variance terms**, because variance is the square. From 19.3² ≈ 372 to 29.3² ≈ 858 is a swing of 486 vol-points-squared, and a variance swap pays on variance. The carry was collected in vol points; the loss is paid in variance. That mismatch — linear-ish carry, convex loss — is the entire risk signature, and it is why no amount of calm-year profit makes the trade "safe."

There is a quieter cautionary tale in the years *between* the crashes, too. Through the long, calm post-2012 bull market, realized correlation drifted persistently low (stocks moving on idiosyncratic, single-stock stories), and dispersion was one of the best risk-adjusted trades on the Street — which drew in size and compressed the premium. By the time a synchronizing shock arrived (the February 2018 "Volmageddon" vol spike, the August 2024 yen-carry unwind that briefly closed the VIX at 38.57, and above all March 2020), the trade was crowded and the premium thin, so the same crash hurt the *late* entrants far more than the early ones. The lesson sits alongside every other short-premium trade in this series: **a structural premium that everyone has discovered is a thinner premium against the same fat tail.**

This is the same phenomenon the cross-asset world calls "[when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis)": the diversification you were relying on is precisely the thing that vanishes when you need it most. Dispersion is the purest options expression of that risk, because it is *explicitly, by construction* short correlation. The free lunch of [diversification](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch) and the bill for dispersion are two sides of one coin: you are paid in calm to give back the diversification you would have had in a crash.

### Why the single-name longs do not save you

The reflexive defence of a dispersion desk in a crash is "but I am *long* single-name vol, so when everything blows up my long leg pays." It is true but insufficient, and understanding why is the difference between sizing the trade right and being surprised. In a crash, single-name vol does rise — but the index vol rises *more*, because the index vol is the lever on correlation and correlation is the thing that exploded. Run the numbers from the crash example: single-name vol went from 30% to 30% (or even up a bit), while index vol went from a sold-implied 19.3% to a realized 29.3%. The long single-name leg gained little; the short index leg lost a full 10 vol points. The legs were vega-matched at inception against a *parallel* move in vol; a crash is the opposite of parallel — it is the index vol decoupling violently upward from the single-name vol. **The very thing the hedge was built to neutralize (the level) holds; the very thing it was built to harvest (the spread) detonates.** That is structural, not bad luck, and it is why dispersion needs an explicit tail hedge beyond the single-name longs.

## Common misconceptions

**"Dispersion is a market-neutral arbitrage — it's basically free money."** No. It is a **short-correlation, short-tail** position dressed up to look neutral. The vega-matching makes it neutral *to the level of vol*, not to correlation, and correlation is precisely the risk that pays you. You are not arbitraging; you are selling insurance against everything-moving-together. The "free money" of the carry years is the premium for the 2008/2020 claim. Concretely: years of +2 to +4 vol points of monthly carry can be erased by a single month where the index realizes 10 vol points over implied (a −486 vol-points-squared variance loss in our example). Free lunches don't have a −486 tail.

**"If I'm short index vol and long single-name vol, the legs cancel, so my risk is small."** They cancel on the *level* of vol, not on correlation, and the two legs respond to a crash very differently. When correlation goes to one, the index leg blows out by far more than the single-name leg, because the index's vol is leveraged to correlation while the single names' vol is not. In the worked crash, the index leg lost about 10 vol points while the single names were roughly flat-to-up. The legs do **not** cancel where it matters.

**"Implied correlation can be anything from −1 to 1, like a normal correlation."** In practice, implied correlation derived from index and single-name options is bounded and almost always **positive and well below 1 in calm markets** (often 0.2–0.45), pushing toward — but rarely all the way to — 1 in a crisis. It cannot exceed 1 in the homogeneous model (that would require index vol above the weighted-average single-name vol, which the diversification term forbids), and a *negative* average pairwise correlation across hundreds of equities essentially never happens; equities share too much common risk. Treat implied correlation as a number that lives roughly in [0.2, 0.95], not [−1, 1].

**"Selling index vol and selling single-name vol are the same short-premium trade."** They are opposite *in correlation*. Selling index vol is short correlation (you lose when correlation rises). To run dispersion you must be **long** single-name vol to neutralize the level and isolate the correlation bet. If you instead sold *both*, you would be doubly short vol — a different (and even more dangerous) trade, with no correlation hedge at all. The sign of the single-name leg is the whole point.

**"Index vol is rich because the index is riskier than the average stock."** Backwards. The index is *less* volatile than the average stock (diversification), and yet its vol is *expensive* relative to the components. The richness comes from one-sided hedging demand and the resulting correlation risk premium, not from the index being riskier. A cheaper-but-pricier asset is exactly the kind of structural quirk a desk trades.

## The playbook: how to trade dispersion (and survive it)

If you want to harvest the correlation risk premium, here is the position, the Greeks, the sizing, and the invalidation — the part of every post in this series that matters most.

**The position.** Sell index volatility (short the index straddle or, cleaner, short an index variance swap) and buy single-name volatility on a liquid proxy basket (long single-name straddles / variance swaps). **Vega-match the legs** so total vega ≈ 0 at inception. You are now short implied correlation: you collect the gap between the rich index implied vol and the cheaper single-name implied vol, and you keep it as long as realized correlation undershoots implied.

**The Greek profile.** Net vega ≈ 0 at entry (you're not betting on the vol level). Net **short correlation** is the live exposure. Because index vol is convex in correlation, your loss accelerates as correlation rises — you are effectively **short gamma in correlation space**, the same toxic curvature you met in [gamma, the Greek that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short), now one level up. You are also implicitly short the index skew you sold and long the flatter single-name skew, so monitor [the vol surface](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) on both legs, and watch your [vega and vol-of-vol](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) drift as spots and time move the match out of balance.

**Entry.** Put the trade on when implied correlation is **high** relative to its own history and to your forecast of realized correlation — i.e., when the index is especially rich versus its components and you expect the stocks to keep moving on idiosyncratic news (earnings season, sector rotation, low-macro-news regimes). Read the implied-correlation index off the screen; sell it when it's elevated, not when it's already cheap.

**Sizing — the only thing that keeps you in business.** Size for the crash, not the carry. Model the loss at correlation = 0.9–1.0 (our example: −10 vol points on the index leg, ≈ −486 vol-points-squared), decide how many of *those* you can survive, and size so a single corr-to-one event is a survivable drawdown, not a career-ending one. The steady carry will tempt you to size up; the convexity of the tail is exactly why you must not. A practical rule: never let the worst-case (corr → 1) loss exceed a small, pre-set fraction of capital, and remember that the loss scales with the square of the index vol move.

**Manage the tail explicitly.** Because the single-name longs do not fully offset the index blowout, keep a dedicated tail hedge: buy a cheap deep-OTM index put wing, or hold an explicit long-index-correlation position (a small long index variance position) sized to cap the corr → 1 loss. Think of this exactly the way the next post in the series frames [hedging a portfolio with protective puts, collars, and tail risk](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk): the wing costs carry, but it converts an unbounded convex loss into a known, budgeted cost.

**Invalidation.** Get out — or flip — when implied correlation is already *low* (no premium left to harvest, and the trade is now poorly compensated for the same tail), when realized correlation is trending up toward your implied entry (the regime is turning against you), or ahead of a known macro catalyst that could synchronize the market (a central-bank decision, a systemic-credit event). If the thesis was "the stocks will move on their own news" and the tape shows them increasingly moving as one, the thesis is broken; cut before correlation finishes going to one.

#### Worked example: sizing dispersion for the crash, not the carry

Put concrete numbers on the sizing rule. You have a **\$50 million** book and want the carry, but you refuse to let a corr-to-one event cost more than **10% (\$5 million)** of capital. From the worked P&L, a corr-to-one crash on a 30%-single-name, 0.40-implied index costs roughly **−486 vol-points-squared** of variance on the index leg per unit of variance notional. Suppose your chosen index variance-swap vega notional translates that into a loss of about **−10 vol points** on a \$X-per-vol-point line, so a crash costs roughly `X × (29.3 − 19.3) = 10X` dollars at the vol level (with the variance convexity making it worse, so treat 10 vol points as a floor on the pain). To cap the crash at \$5 million you need `10X ≤ \$5,000,000`, i.e. **X ≤ \$500,000 per vol point** of index notional. At that size, the calm-regime carry of ~2.7 vol points a month earns about `2.7 × \$500,000 ≈ \$1.35 million` a month gross — before frictions. The asymmetry is the whole point: you sized so the *crash* is a survivable \$5 million, and accepted that the *carry* is the modest residual. Sizing off the carry instead (chasing the \$1.35 million) would have let the crash run to a multiple of capital. The intuition: pick the size from the worst case you can survive, then take whatever carry that size happens to throw off — never the other way around.

![Decision figure separating the calm regime where dispersion harvests carry from the crash tail where correlation goes to one](/imgs/blogs/dispersion-and-correlation-trading-index-vol-vs-single-name-vol-7.png)

Figure 7 is the whole discipline on one page: in the calm regime you harvest the correlation premium but cap the line; in the crash regime you respect a convex, only-partly-hedged loss; and the rule that survives both is to **size for the crash and keep a wing on, never running dispersion naked into a macro event.** The desks that blew up in 2008 and 2020 were not wrong about the premium — it was real, and it had paid them for years. They were wrong about the size of the line relative to the tail they were short. The premium is the wage for warehousing the day everything moves together; collect it humbly, hedge the day it arrives, and dispersion is one of the most elegant trades in the volatility book.

Step back and the whole structure rhymes with everything else in this series. An option is a bet on volatility and time; you make money trading the gap between implied and realized, and you survive by managing your exposures. Dispersion is that exact spine applied one level deeper — to the *correlation* embedded inside index vol rather than the vol itself. The gap you trade is implied-correlation-minus-realized-correlation; the Greek you manage is your short position in correlation, which behaves like a short-gamma exposure that bites convexly when correlation runs to one. The reason the trade exists at all is a structural quirk you can see and quantify: index hedging demand makes index vol rich, index vol below single-name vol *is* implied correlation, and implied correlation sits above realized correlation often enough to pay. Master those three facts — the variance identity, the back-out, and the correlation risk premium — and you can read an index option chain and tell, at a glance, what the market is paying you to be short, and what it will charge you when the blizzard finally comes for everyone at once.

## Further reading & cross-links

- [Implied vs realized volatility: the trade at the heart of options](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options) — the parent idea: selling what the market overpays for; dispersion applies it to correlation.
- [The variance risk premium: why selling vol pays until it doesn't](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) — the correlation risk premium is its sibling; same structure, same tail.
- [The volatility smile and skew: why OTM puts cost more](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more) — why index-put demand bids the skew that makes index vol rich.
- [Vega: your exposure to implied volatility and the vol of vol](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) — the Greek you vega-match the two legs on.
- [Reading the vol surface like a trader: the 3D map of fear](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) — index vs single-name surfaces are where dispersion is read.
- [Gamma: the Greek that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) — the convexity intuition; dispersion is short gamma in correlation space.
- [Hedging a portfolio with options: protective puts, collars, and tail risk](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk) — how to put a wing on the corr → 1 tail.
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — the cross-asset view of the exact risk dispersion is short.
- [Correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch) — the free lunch you sell when you run dispersion.
- [The covariance matrix](/blog/trading/math-for-quants/covariance-matrix-linear-algebra-math-for-quants) — the linear algebra behind the index-variance identity.
- [Exotic derivatives](/blog/trading/quantitative-finance/exotic-derivatives) — correlation products and the pricing of correlation as an asset.

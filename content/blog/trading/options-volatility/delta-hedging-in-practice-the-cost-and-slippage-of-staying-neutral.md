---
title: "Delta Hedging in Practice: The Cost and Slippage of Staying Neutral"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why the textbook 'riskless' delta hedge leaks money in the real world — discrete rehedging error, transaction costs, and overnight gaps — and how to choose a rehedge policy that survives."
tags: ["options", "volatility", "delta-hedging", "options-greeks", "transaction-costs", "gamma", "rehedging", "market-making", "variance-risk-premium", "black-scholes"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — The textbook says a continuously delta-hedged option is "riskless." In the real world you hedge in discrete steps, pay a bid-ask spread on every share, and can't trade through an overnight gap — so the hedge *leaks*, and that leak has to be priced into the option.
>
> - **Discrete hedging leaves residual gamma P&L.** Hedging once a day instead of continuously turns a "riskless" position into one whose error has a standard deviation near \$1.74 on a \$2.45 option. The error's spread scales as roughly 1 over the square root of the number of rehedges — quadruple your rehedging and you roughly halve the error.
> - **Costs grow as you hedge more, error shrinks — so there's a sweet spot.** Transaction cost rises with frequency while hedging error falls, producing a U-shaped total-cost curve. The optimal policy is a *no-trade band* (Whalley-Wilmott): rehedge only when your delta drifts outside a tolerance, not on a fixed clock.
> - **You can't hedge a gap.** A delta-hedged *short* option is short gamma; a ±7% overnight gap costs roughly \$145-160 per contract on a 30-day at-the-money call no matter which way it jumps, because the gamma leak is `0.5 × gamma × (move)²` and you weren't there to rehedge through it.
> - **The one rule to remember:** the "riskless" hedge has a real, quantifiable cost — discretization noise plus spread plus gap risk — and that cost is exactly why implied volatility trades above realized. A market-maker who hedges has to charge for it.

## The desk that hedged perfectly and still lost

A volatility desk I'll describe sold a book of short-dated index options one quiet summer and did everything the model told them to do. Every leg was delta-hedged. The risk system updated the net delta in real time, and a junior trader's entire job was to keep it near zero: when the book's delta drifted, he traded the underlying future to flatten it, exactly as Black, Scholes, and Merton say you should. By the model, the position was replicated and riskless — the premium they collected was supposed to exactly fund the cost of dynamically hedging the options to expiry. For weeks it worked. The book bled a little theta into their pocket every day, the hedge stayed tight, and the P&L was a gentle upward drift.

Then two things happened that the textbook never mentioned. First, the *slippage*: every time the trader flattened delta, he paid the bid-ask spread on the future and, on busier days, a little market impact when he had to push size. None of that appears in the Black-Scholes replication argument, which assumes you trade for free. Across thousands of rebalances those pennies compounded into real money — a steady drag the model said should not exist. Second, the *gap*: one Friday they went home delta-neutral and content, short a stack of gamma. Over the weekend a macro shock hit, and the index gapped down 6% at Monday's open. There was no continuous path to hedge along — the price simply teleported from Friday's close to Monday's open with no trades in between. By the time the desk could touch the future, the short-gamma loss was already locked in. A book that was "riskless" by the model gave back two months of theta in a single morning, plus the cumulative slippage tax it had been paying all along.

This is the gap between the seminar and the trading floor. The replication argument is one of the most beautiful results in finance — it's *why* options have a price at all — but it rests on three assumptions that are all false in practice: you hedge *continuously*, you trade *for free*, and the price moves *without jumps*. Relax any one and the hedge stops being riskless and starts being a cost. This post is about that cost: how big it is, where it comes from, and how a real desk chooses a rehedging policy that bleeds as little as possible. It assumes you already know what [delta](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio) is, that [gamma is the curvature that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short), and that you've seen how a long-gamma position turns rehedging into a vol harvest in [gamma scalping](/blog/trading/options-volatility/gamma-scalping-turning-a-long-straddle-into-a-vol-harvest). Here we go deep on the *frictions* those posts set aside.

![Replicating-portfolio value versus option payoff at expiry under daily versus frequent delta hedging, showing the leak shrink as hedging gets more frequent](/imgs/blogs/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral-1.png)

The chart above is the whole story in one frame. The smooth gray line is the option's true payoff — what you owe at expiry. Each scatter is what a *replicating portfolio* (shares plus financing, rehedged dynamically) is actually worth at expiry across many simulated paths. When you hedge rarely (the wide red cloud), the replication misses the target by a lot: the cloud is fat, and that fatness is real money you either won or lost versus the model. When you hedge frequently (the tight blue cloud), the replication hugs the payoff and the leak nearly vanishes. The textbook lives in the limit where the cloud collapses to the line. You live in the cloud. Everything below is about measuring it and managing it.

## Foundations: what "delta hedging replicates the option" actually means

Before we can talk about why the hedge leaks, we have to be precise about what it's *supposed* to do. The claim at the heart of option pricing is bolder than most people realize, so let's build it from zero.

**Delta** is how much an option's price moves for a \$1 move in the underlying. A call with delta 0.53 gains about \$0.53 when the stock rises \$1. Read as a *hedge ratio*, delta tells you how many shares offset the option's directional risk: if you're short one call with delta 0.53 (on 100 shares), you buy 53 shares, and now a small move in the stock barely changes your combined value — the share gain offsets the option loss, or vice versa. That combined position is **delta-neutral**: it has, momentarily, no first-order exposure to the direction of the stock.

The deep claim is this: if you keep rebalancing those shares so the position stays delta-neutral at *every instant*, then the directional risk of the option is completely cancelled at all times, and what's left behind is deterministic — it depends only on volatility and time, not on which way the stock went. Black, Scholes, and Merton showed that this dynamically hedged portfolio earns exactly the risk-free rate, no more and no less, regardless of the stock's path. That is the **replication argument**: the option plus its delta hedge is, in the continuous-time limit with no frictions, a *riskless* bond. And if it's a riskless bond, no-arbitrage forces its cost to be the Black-Scholes price. The full derivation lives in [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) and the math of `dS` in [Itô's lemma](/blog/trading/math-for-quants/ito-integral-itos-lemma-math-for-quants); we won't repeat it. What matters here is the punchline: *the option's price is the cost of hedging it.* The premium you pay or collect is supposed to be exactly enough to fund (or be funded by) the dynamic hedge to expiry.

### Why continuous hedging is "riskless" — and why you can't do it

Here's the mechanism, because it's the key to understanding the leak. When you're delta-hedged, your *first-order* directional risk is zero. But the option still has **gamma** — its delta changes as the stock moves. Over a tiny time step `dt` in which the stock moves by `dS`, your hedged position's change in value is, to leading order:

`dP ≈ theta × dt + 0.5 × gamma × (dS)²`

The first term is theta — the option's time decay, which you pay (if long) or collect (if short) every instant. The second term is the gamma P&L: it depends on the *square* of the move, so it's always positive for a long option (positive gamma) and always negative for a short option (negative gamma). The miracle of Black-Scholes is that, on average, these two terms cancel. The expected `(dS)²` over a small step is `S² × sigma² × dt` — variance grows with time — so the expected gamma term is `0.5 × gamma × S² × sigma² × dt`, which is exactly minus the theta term. Decay paid out, variance harvested back, net zero. That's the replication, written as a daily balance.

> **The replication is a continuous tug-of-war between theta and gamma.** Every instant you pay theta and earn (or pay) a gamma P&L of `0.5 × gamma × (dS)²`. In the continuous limit with no frictions, they cancel exactly and the hedge is riskless.

#### Worked example: theta IS the rent on gamma

Take our running setup — a \$100 stock, a \$100-strike call, 20% annualized volatility, a 4% rate, and 30 days to expiry. From this series' Black-Scholes pricer:

- **Premium:** `bs_price(100, 100, 30/365, 0.04, 0.20, kind="call")` = **\$2.4513**.
- **Delta:** **0.5343** (you'd hold 53.4 shares per short call to be neutral).
- **Gamma:** **0.06932** — delta moves 0.069 per \$1.
- **Theta:** **−\$0.0436 per calendar day** (the long holder bleeds about four-tenths of a cent... no, **4.4 cents** a day; the short collects it).

Now watch theta and gamma cancel. The expected daily gamma P&L for the *short* position is `−0.5 × gamma × S² × sigma² / 365` (variance per day):

`−0.5 × 0.06932 × 100² × 0.20² / 365 = −\$0.038 per day`

Compare that to the short's daily theta credit of **+\$0.0436**. They nearly match (the small gap is the risk-free-rate drift term, which we've folded out). The short collects \$0.0436 of decay and pays back about \$0.038 of expected gamma — net a few tenths of a cent of edge per day if realized vol comes in exactly at 20%. The intuition: **theta is the rent you pay to be long gamma, and the variance you harvest is the rent you collect to be short it.** When realized vol equals implied, they're a wash. That equality is the entire engine, and it's also why the trade is so fragile to anything that breaks the cancellation — which is everything in the rest of this post.

The problem is the word *continuous*. The replication needs you to rebalance the hedge at every instant. You can't. You rebalance at discrete moments — every hour, every few cents of move, at the close — and between those moments your delta is wrong. The stock moves, your hedge doesn't follow until your next trade, and the gamma term stops cancelling cleanly. That residual is the **hedging error**, and it's the first and most fundamental leak.

## The discrete hedging error: the leak you can't avoid

Imagine you hedge only once a day instead of continuously. You set your delta at the open and leave it. Over the day the stock wanders, your delta drifts away from where it should be, and only at tomorrow's open do you reset it. During each day you were carrying a residual directional bet you didn't intend — sometimes it helped, sometimes it hurt — and the sum of those accidental bets over the option's life is your hedging error.

Crucially, this error is **path-dependent**: it depends on the specific zig-zag the stock took, not just where it ended up. Two paths that start at \$100 and end at \$100 can produce very different hedging P&L if one chopped wildly between rehedges and the other crept smoothly. That's the signature of leftover gamma: your hedged book still responds to the *square* of moves between rebalances, and squared moves are bigger when the path is jagged.

Because the error depends on the random path, it isn't a single number — it's a *distribution*. Run the same delta-hedged short option through thousands of simulated paths and you get a spread of outcomes. The center of that spread is roughly zero (when realized vol equals the implied vol you priced at, discrete hedging is *fair* on average — you don't systematically win or lose by hedging coarsely). But the *width* of the spread is the risk you've taken on, and it's far from zero.

#### Worked example: hedging once a day versus continuously

Simulate the short \$100 call above over its 30-day life, rebalancing the delta hedge a fixed number of times, with realized volatility equal to the 20% implied we priced at. Across 4,000 simulated paths, the profit-and-loss of the hedged position (premium collected, shares traded, option settled) comes out as:

| Rehedges over the life | Mean P&L | Std dev of P&L |
| --- | --- | --- |
| 1 (hold the open hedge) | +\$0.007 | **\$1.74** |
| 5 | −\$0.014 | \$0.84 |
| 21 (about daily) | −\$0.010 | \$0.44 |
| 252 (about hourly) | −\$0.001 | \$0.13 |

The mean sits at essentially zero in every row — discrete hedging doesn't bias your P&L when you've priced at the right vol. But look at the standard deviation. With a single hedge, the typical miss is **\$1.74** on a \$2.45 option — you might as well not have hedged the gamma at all; the "riskless" position has a one-sigma error of 71% of the premium. Hedge 21 times (roughly daily) and the typical miss falls to **\$0.44**, about 18% of premium. Hedge 252 times (roughly hourly) and it's **\$0.13**, around 5%. The leak shrinks, but it never reaches zero, because you never hedge truly continuously. The intuition: discrete hedging is fair on average but noisy in any single realization, and the noise is your residual gamma talking.

There's a clean law hiding in that table. Multiply each standard deviation by the square root of the number of rehedges:

`1.74 × √1 = 1.74`   `0.44 × √21 = 2.00`   `0.13 × √252 = 2.00`

The product is roughly *constant* — about 2.0 for this option. That's the famous **Boyle-Emanuel** result: the standard deviation of the discrete hedging error scales as 1 over the square root of the number of rehedges. Quadruple your rehedging frequency and you halve the error; do it 100× more often and the error falls 10×. The error never vanishes for any finite frequency — the Black-Scholes-Merton replication is an *asymptotic* statement that only becomes exact in the continuous limit you can't reach.

![Distribution of delta-hedged option P&L across many paths, widening as the rehedge spacing grows](/imgs/blogs/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral-2.png)

The chart above shows those distributions directly. Each curve is the P&L spread of the same delta-hedged option across thousands of paths, for a different rehedge frequency. The hourly-hedged curve (blue) is tall and narrow — tightly centered on zero, small misses. The daily curve (amber) is wider. The hedge-once curve (red) is a broad, flat mound — the same expected value, but enormous dispersion. Every curve is centered at the same place; what changes is how *confident* you can be that any single trade lands near it. The replication argument promised a delta function — a spike at zero with no width. Discreteness fattens it into these mounds, and the width is the price of not being continuous.

### Why the residual is gamma, not delta

It's worth being precise about *what* you're left holding between rehedges, because it tells you where the risk lives. After you set your delta to zero, your *first*-order exposure is gone. What remains is the *second*-order term: gamma. Between rehedges, your hedged book behaves like a small position in pure gamma — it gains on big moves in either direction (if long gamma) and loses on big moves in either direction (if short gamma), proportional to the square of the move. So:

- **A long-option book that hedges discretely has long residual gamma.** Between rehedges, every move — up or down — helps it a little, and rehedging banks that gain. This is exactly [gamma scalping](/blog/trading/options-volatility/gamma-scalping-turning-a-long-straddle-into-a-vol-harvest): the discrete hedge *is* the harvest. The hedging "error" is, for a long-gamma trader, the source of the P&L.
- **A short-option book that hedges discretely has short residual gamma.** Between rehedges, every move hurts it, and rehedging locks in the loss. The hedging error is a pure cost, and it's worst exactly when the market is most volatile.

This is the same coin as the [gamma post's](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) "long gamma is a blessing, short gamma is a time bomb," now expressed through the hedge. The discrete hedging error has *mean* zero but *sign* that matters: for the long-gamma trader the dispersion is upside they harvest; for the short-gamma trader it's a tax that's largest when realized vol is largest. The width of those mounds is symmetric, but who profits from the width is not.

### When the mean is *not* zero: realized versus implied

The tables above all assumed realized volatility came in at exactly the 20% implied we priced at — and that's why their means sat at zero. But that's the special case, not the rule. In the real world realized vol almost never equals the implied vol you traded, and the *moment* it differs, the hedging P&L develops a systematic mean, not just noise. The reason runs straight through the theta-gamma identity: you collected (or paid) decay at the *implied* vol, but you realized the gamma P&L at the *actual* vol of the path. The two no longer cancel.

This is the deepest point in the whole post, and it reframes delta hedging entirely: **a delta-hedged option is not a directional bet at all — it's a bet on realized versus implied volatility.** Hedge a *long* option and you win when the stock realizes *more* vol than you paid for (your harvested gamma P&L beats the theta you paid); hedge a *short* option and you win when the stock realizes *less* (the theta you collected beats the gamma P&L you paid back). Direction has been hedged out; what's left is pure vol.

#### Worked example: the systematic P&L when realized misses implied

Sell the same \$100 call at 20% implied (collect \$2.4513), delta-hedge it daily, and run two worlds across 4,000 paths each: one where the stock actually realizes **15%** vol, and one where it realizes **25%**.

- **Realized 15% (calm world):** the short collects theta priced at 20% but pays back gamma P&L sized to only 15% realized variance. The mismatch is in the seller's favor — across paths the mean P&L is roughly `+0.5 × (0.20² − 0.15²) × S² × gamma × T`, which works out to about **+\$0.50** per contract of *systematic* edge on top of the daily-hedging noise (the simulation lands near +\$0.58 with the convexity of the lognormal path). The seller wins because the option was "too expensive" for the calm that followed.
- **Realized 25% (stormy world):** now the short pays back gamma P&L sized to 25% realized variance against theta priced at only 20%. The same formula flips sign: a mean of roughly **−\$0.64** per contract (the simulation lands near −\$0.58). The seller loses because the option was "too cheap" for the storm.

The intuition: the discretization noise (the \$0.44 std at daily hedging) wobbles you around a *center*, and that center is set by the gap between realized and implied vol. Get the vol call right and the mean works for you; get it wrong and no amount of careful rehedging saves you — you'll bleed the realized-versus-implied gap regardless of how tightly you hedge. This is exactly the [implied-versus-realized trade](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options), and the structural tilt of implied above realized is the [variance risk premium](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options) that pays the disciplined seller on average.

## The rehedge policy: when, exactly, do you trade?

If hedging more often shrinks the error, why not hedge constantly? Because every trade costs money. The real decision a desk makes isn't "hedge or don't" — it's *what rule triggers a rehedge*, and that rule trades off error against cost. There are three families.

**Time-based: rehedge on a clock.** Every N hours, or at the close, you reset delta to neutral regardless of where the stock is. Simple, predictable, easy to automate. Its weakness: it trades on a schedule, not on need. If the stock is dead flat, you still rebalance and pay the spread for nothing; if the stock is screaming, you wait for the clock and carry a huge residual delta in between. Time-based hedging spends cost where there's no risk and skips trades where there is.

**Move-based (bands): rehedge on a threshold.** You define a tolerance — a *no-trade band* — around neutral, say "let net delta drift up to ±10 shares, then snap it back to zero." You only trade when the move is big enough to push you outside the band. This spends cost where the risk actually is (big moves) and saves it where there isn't (quiet drift). The catch is you must watch the position continuously to know when you've breached the band, and in a fast market you can be triggered repeatedly.

**Cost-aware (optimal bands): rehedge when the benefit beats the cost.** The sophisticated version sets the band *width* by balancing the marginal cost of trading against the marginal risk of not trading. Wider bands when costs are high or gamma is low; tighter bands when gamma is high (because then residual delta builds fast and a small move does real damage). This is the **Whalley-Wilmott** result, which we'll take lightly below.

#### Worked example: a move band versus a fixed interval on one path

Take a single simulated 30-day path for our \$100 stock that wanders between \$97.8 and \$103.6 and closes at \$101.4. Hedge the short call two ways and count the trades and the leftover (residual) delta, measured in shares per contract:

| Policy | Trades over the life | Average residual delta | Worst residual delta |
| --- | --- | --- | --- |
| Fixed interval, every 4 steps | 64 | 2.7 shares | 19.3 shares |
| Fixed interval, every 21 steps | 13 | 7.1 shares | 42.3 shares |
| Move band, ±5 shares | 61 | 1.6 shares | 4.9 shares |
| Move band, ±10 shares | 23 | 3.5 shares | 10.0 shares |

Compare the rows with similar trade counts. The ±5-share band and the every-4-steps interval both trade about 61-64 times — but the band holds the average residual delta to **1.6 shares** and *never* lets it exceed **5**, while the fixed interval averages **2.7 shares** and lets the residual blow out to **19 shares** between scheduled trades. For the same cost (same number of trades), the band controls your actual risk far better, because it spends its trades when the stock moves rather than when the clock ticks. And the worst-case residual — the size of the accidental directional bet you're carrying at the worst moment — is the number that hurts you in a fast market. The intuition: a band caps your risk directly; a clock caps it only on average, and "on average" is cold comfort the afternoon the stock runs.

![Residual delta over a path under a fixed-interval hedge versus a move-based band, with trade markers for each policy](/imgs/blogs/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral-3.png)

The chart above traces the same single path and shows the residual delta each policy carries through time. The fixed-interval line (amber) snaps back toward zero on each scheduled trade, then drifts — sometimes a long way, out past 30 and 40 shares late in the life when gamma is highest — until the next tick, with the tall excursions marking the moments it was carrying a big accidental directional bet. The move-band line (blue) stays far closer to neutral: the instant it would travel past the band edge, the trade triggers and pulls it back, so its excursions are a fraction of the clock's. The two policies place a comparable number of trade markers (the triangles along the axis), but they place them *differently* — the band clusters its trades where the path actually moves and skips the quiet stretches, while the clock spreads them evenly whether or not anything happened. That is the entire case for move-based hedging in one frame: a similar budget of trades, far tighter control of the risk you're actually carrying.

![The cost versus error trade-off showing frequent rehedging as high cost low error, rare rehedging as low cost high error, and the band as the sweet spot](/imgs/blogs/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral-4.png)

The diagram above lays out the trade-off as a decision. On the left, rehedging very often: your residual delta stays tiny, your hedging error is small — but you're paying the spread constantly, and transaction cost dominates. On the right, rehedging rarely: you barely pay any spread, but your residual delta wanders far and your hedging error balloons. Neither extreme is right. The sweet spot in the middle is a *band*: trade only when the move is big enough to matter, so you pay cost in proportion to risk. The band's width is the one dial that tunes the whole policy.

### The optimal band, lightly (Whalley-Wilmott)

You don't need the derivation to use the result, but the *shape* of the answer is worth carrying. Whalley and Wilmott showed that when you have a fixed proportional transaction cost (a bid-ask spread) and a tolerance for risk, the optimal hedging rule is not a clock at all — it's a no-trade band around the perfect Black-Scholes delta. You let your delta drift inside the band and do nothing; the moment it touches the edge, you trade just enough to bring it back to the edge (not all the way to center). The half-width of that band scales like:

`band half-width ∝ ( transaction-cost-rate × gamma² × S² / risk-aversion )^(1/3)`

Three readings of this formula tell you everything you need:

- **Higher transaction costs → wider band.** If trading is expensive, tolerate more residual delta before you pay to fix it. (The cube root means costs have to rise a lot to move the band much — costs matter, but with diminishing effect.)
- **Higher gamma → wider band, but...** the gamma appears squared *inside* the cube root, so net it pulls the band *narrower as gamma rises* in the practically relevant range — when your gamma is high (near expiry, at the money), residual delta builds fast, so you tighten the band and rehedge sooner. This is why desks hedge near-dated at-the-money books on a hair-trigger and let far-dated wings drift.
- **Lower risk-aversion → wider band.** If you can stomach more P&L noise, trade less and save the spread.

The practical takeaway is that the *right* policy is move-based with a width that you widen when costs are high and gamma is low, and tighten when gamma is high. A fixed clock can't do this — it's blind to both your costs and your curvature. This is why professional desks parametrize their hedgers by delta tolerance, not by a timer.

## The costs that bite: spread, impact, financing, dividends

The hedging error is the cost of being *discrete*. The transaction costs are the cost of trading not being *free*. There are four, and each chips at the "riskless" hedge.

**Bid-ask spread on the underlying.** Every time you rehedge you cross a spread — buy at the offer, sell at the bid. On a liquid index future or a large-cap stock that might be a cent or two; on a thin name it's wider. You pay roughly *half the spread* per share on each trade (the half-spread is the cost of immediacy). It seems tiny, but you pay it on *every* rehedge, and a tight-banded hedger trades a lot.

**Market impact.** When you trade size, you move the price against yourself — your buying pushes the offer up, your selling pushes the bid down. This is on top of the quoted spread, and it scales with how much you trade relative to available liquidity. A desk hedging a large book can't flatten 50,000 shares of delta without moving the tape, so its *effective* cost per share is worse than the screen spread. This is the [hidden tax of liquidity](/blog/trading/options-volatility/liquidity-bid-ask-spreads-and-getting-filled-the-hidden-tax), and it's why size itself is a cost.

**Financing and borrow.** Your hedge is a stock or futures position, and carrying it isn't free. If you're long shares to hedge a short call, you've tied up cash that could earn the risk-free rate (or you borrowed it and pay interest). If you're *short* shares to hedge a long put, you must *borrow* the stock to short it, and you pay a borrow fee — which on a hard-to-borrow name can be enormous and is the dominant cost. The Black-Scholes rate term `r` captures the financing of the *theoretical* hedge, but real borrow costs and the spread between your funding rate and the risk-free rate are extra drag the model glosses over.

**Dividends.** If the stock pays a dividend while you hold the hedge, the dividend changes the forward and therefore the delta. A long-share hedge *receives* the dividend; a short-share hedge *pays* it. Get the dividend forecast wrong and your hedge ratio is wrong, and an unexpected dividend (or a cut) repositions every delta in the book. Dividends are a known quantity for index hedging but a real source of error for single names.

These four costs compound in a way the single bid-ask line hides. Take the daily-hedged short call from the worked example: it traded about 150 shares over its life for \$1.50 of half-spread, but layer in even a fraction of a cent of market impact per share, a few days of financing on the 53-share hedge, and a forgotten ex-dividend date, and the realized drag can easily double to \$3 on a \$2.45 option. That is the practitioner's hard truth — the costs that look negligible per trade are the ones that quietly decide whether a hedged short-vol position clears its variance risk premium or bleeds out one penny at a time. A market-maker models all four explicitly before quoting, which is exactly why the screen price of an option is never the textbook Black-Scholes number.

#### Worked example: the transaction-cost bill grows with frequency

Stay with the short \$100 call. Each rehedge trades the *change* in delta — `|new delta − old delta|` shares. Summing that across the option's life over thousands of paths gives the average total turnover, and at a half-spread of 1 cent on a 100-share contract the transaction-cost bill is:

| Rehedges over the life | Total shares traded (per contract) | Transaction cost at 1¢ half-spread |
| --- | --- | --- |
| 1 | 50 | \$0.50 |
| 5 | 80 | \$0.80 |
| 21 (about daily) | 150 | \$1.50 |
| 252 (about hourly) | 502 | \$5.02 |

Two things stand out. First, the cost *rises* with frequency — the opposite direction from the hedging error. Hedge once and you trade 50 shares (set the hedge, unwind it) for 50 cents; hedge hourly and you trade 502 shares for **\$5.02**, which is more than *twice* the option's \$2.45 premium. A trader who naively hedges "as often as possible" to kill the error can spend the entire premium and then some on spread. Second, the turnover grows *sublinearly* — it scales roughly like the square root of the frequency (the share count goes 50 → 80 → 150 → 502 as rehedges go 1 → 5 → 21 → 252, a log-log slope near 0.42), because more-frequent rehedges each trade a smaller delta change. The intuition: error falls like 1/√N and cost rises like √N, so they pull in opposite directions and there's a frequency that minimizes their sum.

![Transaction cost rising with rehedge frequency and hedging error falling, summing to a U-shaped total cost curve](/imgs/blogs/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral-5.png)

The chart above is the punchline of the policy problem. The falling curve is the hedging-error cost — high when you hedge rarely, shrinking as you hedge more (the 1/√N law). The rising curve is the transaction cost — near zero when you hedge rarely, climbing as you trade more often (the √N law). Their sum is the U-shaped total-cost curve, and its minimum is the *optimal rehedge frequency*: hedge less than that and error dominates; hedge more and spread dominates. Where the bottom of the U sits depends on your spread, your gamma, and how much P&L noise you'll tolerate — which is exactly what the Whalley-Wilmott band formalizes. The model's "riskless, costless" hedge lives at the impossible left edge where both curves would have to be zero at once.

### Why "riskless" carries a real price — and where it shows up

Put the two pieces together and the conclusion is unavoidable: the textbook-riskless hedge has a *strictly positive real cost* — discretization noise you can't fully kill, plus the spread/impact/financing you pay to trade, plus gap risk we're about to meet. A market-maker who sells you an option and hedges it does not get a free risk-free bond; they get a position that costs them money to run. So they cannot sell the option at the frictionless Black-Scholes price. They have to charge *more* — enough to cover the expected hedging cost and a margin for the risk that the cost runs higher than expected.

That markup is exactly why **implied volatility trades above realized volatility** on average — the [variance risk premium](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options). The option seller, who is short gamma and pays the hedging cost, demands compensation; the buyer, who wants the convex payoff, pays it. The frictions in this post are a big part of *why* that premium exists: if hedging were truly free and continuous, the seller could replicate at the frictionless price and there'd be no reason for implied to exceed realized. The cost of staying neutral is the seller's cost of goods, and it gets passed into the option's price. We'll see the seller's full calculus from the other side in [how a market-maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade).

## The gap: the risk you cannot hedge at all

Everything so far assumed the stock moves along a continuous path — it might be jagged, but it never *teleports*. Between any two prices there's a sequence of trades you could have made. That assumption is the soul of the replication argument, and it's the one that fails most violently. Stocks gap. The price at Monday's open is not connected to Friday's close by any tradeable path — overnight, on a weekend, around an earnings release or a macro shock, the price jumps from one level to another with *no trades in between*. You cannot hedge through a gap, because there's nothing to hedge along. You're simply sitting at the old hedge when the new price prints.

For a short-gamma book — which is what a delta-hedged *short* option is — this is the catastrophe. Your hedge was calibrated to the old price. The stock gaps. Your delta is now badly wrong, and the move that revealed it already happened. You realize the full gamma loss `0.5 × gamma × (gap)²` instantly, with no chance to scalp it back in pieces. And because the loss goes with the *square* of the gap, a big gap hurts grotesquely more than a series of small moves adding to the same distance. This is the structural reason short-gamma books "work until they don't," and the precise mechanism behind the desk in the hook losing two months of theta over one weekend.

#### Worked example: the cost of a gap on a delta-hedged short call

You're short one \$100 call, delta-hedged at \$100 with 53.4 long shares, and the stock gaps overnight (treat the gap as instantaneous, so time and vol are essentially unchanged). The position's P&L per contract — short-option gain plus hedge-share gain — for gaps of various sizes:

| Overnight gap | New price | Short-option P&L | Hedge-share P&L | Net P&L | Gamma approx |
| --- | --- | --- | --- | --- | --- |
| −7% | \$93.00 | +\$214.70 | −\$374.00 | **−\$159.30** | −\$169.80 |
| −3% | \$97.00 | +\$129.00 | −\$160.30 | **−\$31.30** | −\$31.20 |
| +3% | \$103.00 | −\$190.10 | +\$160.30 | **−\$29.80** | −\$31.20 |
| +7% | \$107.00 | −\$518.40 | +\$374.00 | **−\$144.50** | −\$169.80 |

Read the −7% row. Your short call *gained* \$214.70 (the call fell in value as the stock dropped, good for the seller), but your 53.4 long hedge shares *lost* \$374.00, for a net loss of **\$159.30** per contract — more than 60× the option's \$2.45 premium, gone in one open. Now read the +7% row: the stock gapped *up*, your short call lost \$518.40, your hedge shares gained only \$374.00, and you're down **\$144.50**. The loss is large *in both directions* — that's negative gamma's fingerprint: a delta hedge protects you against small moves either way, but a gap blows past the hedge and the short gamma bites regardless of sign. Notice how well the simple `0.5 × gamma × (gap)²` approximation tracks the exact P&L (the last column): for the −3% gap it predicts −\$31.20 versus the actual −\$31.30. And notice the *convexity of the damage*: doubling the gap from 3% to roughly 7% (a bit more than 2×) multiplies the loss by about 5×, not 2×, because the loss goes with the square. The intuition: you can delta-hedge the drift, but you cannot hedge the jump, and the jump is where a short-gamma book dies.

![Profit and loss of a delta-hedged short call as the underlying gaps overnight, losing in both directions because the position is short gamma](/imgs/blogs/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral-6.png)

The chart above plots that net P&L against the size of the overnight gap. It's a downward-opening parabola pinned at zero for no gap: small gaps barely hurt (the delta hedge does its job locally), but the curve falls away faster and faster as the gap grows, symmetric around zero because the gamma loss doesn't care about direction. There is no gap size at which the hedged short option makes money on the jump — the best case is "no gap," and everything else is a loss. A long-gamma book sees the mirror image: an upward-opening parabola, profit in both directions, which is precisely why long gamma loves a gap and short gamma fears the weekend. This is the same asymmetry the [gamma post](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) calls the toxic short, now stated in the language of overnight risk.

### Overnight and weekend risk in practice

Gaps cluster at predictable times, and managing them is mostly about *when you choose to be short gamma*:

- **The overnight gap.** Markets close; news happens; the open reprices. You held a hedge for ~17.5 hours that you couldn't adjust. Short-gamma desks routinely *reduce* gamma into the close — buying back some optionality even at a loss — so the overnight gap they're exposed to is smaller. The cost of doing so is real (you pay the spread and give up some theta), and it's a direct trade-off against the gap risk.
- **The weekend gap.** Three calendar days of theta decay but also three days of accumulated risk with no chance to trade (barring futures, which trade limited hours). A short-vol book collects extra weekend theta — which is exactly the bait. The same Monday open that pays the weekend decay can also deliver the gap that erases a quarter.
- **The event gap.** Earnings, central-bank decisions, elections, data releases. Here the gap is *expected*, so it's priced — implied vol is elevated going in (the [expected move](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options)). You can't hedge the gap itself, but you can choose your gamma into the event with eyes open, knowing the jump is coming. The [event-volatility post](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush) covers the implied-versus-realized dynamics around scheduled events.

The unifying lesson: a delta hedge is a *local* tool. It neutralizes small, continuous moves. It does nothing about the discontinuous jump, and the only defenses against a gap are (a) carrying less gamma into gap-prone windows, or (b) hedging the gamma itself with *other options* — because, as [the net-Greeks dashboard](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard) makes explicit, gamma is hedged with options, never with shares. Shares hedge delta; they are powerless against the curvature that a gap exposes.

## Common misconceptions

**"Delta hedging makes an option riskless."** No — *continuous, frictionless* delta hedging makes it riskless, and you can do neither. In practice the hedge leaves three live risks: discretization error (the once-a-day hedge above had a \$1.74 standard deviation on a \$2.45 option), transaction costs (hourly hedging cost \$5.02, more than twice the premium), and gap risk (a 7% overnight gap cost \$145-160 per contract). The replication argument is the *reason options have a price*, not a recipe for a free lunch. A delta hedge converts a large directional risk into a smaller, second-order, frictional risk — it doesn't eliminate risk, it *transforms* it into something you can size.

**"Hedging more often is always better."** Only for the error, never for the cost. The hedging-error standard deviation falls as 1/√N, but transaction cost rises as ~√N, and total cost is a U-curve with a minimum at finite frequency. Hedging our example hourly drove the error to \$0.13 but ran a \$5.02 spread bill — worse, overall, than hedging daily for \$1.50 in spread and \$0.44 of error. The right frequency balances the two, which is precisely what a no-trade band does and a fixed clock cannot.

**"A delta-neutral book has no risk."** Delta-neutral means no *first-order* directional risk *right now*. It says nothing about gamma (your delta is changing as the stock moves), vega (your exposure to implied vol), or the gap. The desk in the hook was delta-neutral by the model every single day and still lost a fortune, because it was *short gamma* and a weekend gap blew past the delta hedge. "Neutral" is a snapshot of one Greek at one instant, not a state of safety — read all the Greeks, as [the dashboard post](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard) insists.

**"The hedging cost is small enough to ignore."** It's the size of the variance risk premium — which is to say, it's the *entire edge* in the trade. The 30-day implied vol on the S&P 500 has historically printed around 19.5 vol points against subsequent realized vol near 15.8 — a gap of about 3.7 points that option sellers earn and buyers pay. A meaningful chunk of that gap is compensation for exactly the frictions in this post: the seller's cost of running an imperfect, expensive, gap-exposed hedge. Ignore the cost and you've mispriced the option by the very amount that makes selling vol profitable.

**"Move-based and time-based hedging are basically the same if you tune the frequency."** They are not, and the band-versus-interval table proves it: for the same number of trades, the move band held residual delta to ±5 shares while the fixed interval let it drift to ±19. A clock spends its trades when time passes; a band spends them when *risk* appears. In a fast market those are wildly different, and the difference is exactly the residual delta you're carrying at the worst moment.

## How it shows up in real markets

**The August 2024 yen-carry gap.** On Friday, August 2, 2024, short-vol desks across the Street went home delta-neutral and short gamma, collecting weekend theta on a calm tape. Over the weekend the yen carry trade unwound; on Monday, August 5, equity indices gapped down hard at the open and the VIX spiked to **38.57** intraday — one of the largest single-day volatility shocks on record. There was no continuous path to hedge along: the move happened between Friday's close and Monday's open. Every delta-hedged short-gamma book realized its gamma loss in one print, exactly the −7%-gap row of our worked example scaled up to portfolio size. The desks that had *reduced* gamma into that Friday — paying the spread to buy back optionality — lost far less than the ones that held the full short-gamma position to collect the last of the weekend theta. The gap is not a tail you hedge; it's a tail you *avoid being short into*.

**The market-maker's daily grind.** A liquid options market-maker is, in aggregate, hedging a vast book delta-neutral with index futures, rebalancing thousands of times a day. Their P&L is the variance risk premium they collected (implied above realized) *minus* the slippage of all that rehedging *minus* whatever gaps caught them short gamma. On a calm day the premium beats the slippage and they make money on the grind. On a gap day the short gamma can erase weeks of grind in an hour. This is the day job behind [how a market-maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade): they don't have a view on direction; they have a view on whether the premium they're charging covers the cost of hedging it, gaps included. When implied is rich relative to expected realized *plus hedging costs*, they sell; when it's not, they don't, or they charge more.

**The single-name borrow trap.** A trader hedging a long put on a hard-to-borrow biotech discovers the financing cost is the dominant friction. To stay delta-neutral on the long put they must short the stock, which means borrowing it — and the borrow fee on a squeezed name can run tens of percent annualized, dwarfing the bid-ask spread. The Black-Scholes `r` term assumes you fund the hedge at the risk-free rate; the real hedge bleeds the borrow fee every day. This is why options on hard-to-borrow names carry distorted implied vols and skews — the *cost of hedging* is baked into the price, and the puts are expensive partly because shorting the hedge is expensive.

**The 0DTE gamma cliff.** Same-day-expiry options have enormous gamma in the final hours — recall from the [gamma post](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) that a one-day at-the-money option's gamma can be five times a 30-day's. A market-maker short 0DTE gamma must rehedge furiously as spot oscillates around the strike, and the bid-ask cost of all that rehedging is brutal — every wiggle through the strike triggers a buy-high-sell-low rehedge. The dealer charges for this in the option's price (0DTE premiums embed the expected hedging cost of that gamma), and on a choppy afternoon the realized rehedging cost can exceed even those rich premiums. The Whalley-Wilmott logic says the band should be *tight* here (high gamma narrows the band), which means many trades, which means the cost is structurally high — there's no clever band that makes hedging extreme gamma cheap.

## The playbook: how to hedge so the leak doesn't eat you

Everything above collapses into an operating routine. The goal is not a perfect hedge — that doesn't exist — but a *cheap enough* hedge whose residual risk you've sized and accepted in advance.

**Pick a band, not a clock.** Set a no-trade tolerance on net delta (e.g. "rehedge when delta drifts past ±X shares") rather than a fixed timer. The band spends your trades where the risk is — on big moves — and saves the spread on quiet drift. The band-versus-interval table showed that for the same trade count, a band holds residual delta several times tighter than a clock. Default to move-based hedging; use a clock only as a backstop (e.g. "and at least flatten by the close").

**Set the band width from cost and gamma.** Widen the band when your bid-ask/impact costs are high or your gamma is low; tighten it when gamma is high (near expiry, at the money), because then residual delta builds fast and a small move does real damage. This is the Whalley-Wilmott rule in plain language: the band half-width scales like `(cost-rate × gamma² × S² / risk-aversion)^(1/3)`, so high gamma pulls the band tighter and high costs push it wider, both with the gentle leverage of a cube root. A near-dated at-the-money book gets a hair-trigger; a far-dated wing gets a loose leash.

**Find your frequency at the bottom of the U.** Remember the two opposing laws: hedging error falls as 1/√N, transaction cost rises as ~√N. Don't hedge "as often as possible" (you'll spend the premium on spread) or "as rarely as possible" (you'll carry a \$1.74-sigma error on a \$2.45 option). Estimate both curves for *your* option and *your* spread, and hedge at the frequency that minimizes their sum. For a liquid 30-day at-the-money option with a 1-cent half-spread, that's typically somewhere around daily-to-a-few-times-daily, not hourly and not weekly.

**Price the cost into your vol.** If you're *selling* options and hedging, your breakeven is not the frictionless Black-Scholes price — it's that price *plus* your expected hedging cost (discretization noise budget + spread + financing + a gap buffer). Only sell when implied vol is rich enough to cover all of that with margin. If you're *buying* and hedging (gamma scalping), your breakeven realized vol must clear implied *plus* your own hedging costs — the slippage eats into the harvest. The whole [implied-versus-realized trade](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options) is fought on the *cost-adjusted* spread, not the raw one.

**Accept the residual gamma — and manage the gap separately.** A delta hedge will never kill your gamma; the discretization noise is irreducible at any finite frequency, so size the position small enough that the residual gamma P&L is a noise you can live with, not a risk that can ruin you. And treat the gap as its own line item: it is *not* hedgeable with shares. Your only defenses are to carry less gamma into gap-prone windows (close, weekend, events) and to hedge the gamma itself with other options when you must hold it. Before every weekend and every event, ask the [dashboard question](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard): how big is my worst overnight move, and can I survive it? If the answer scares you, you're too short gamma — reduce it, even at a cost.

**Watch the hedge evolve, not just its opening snapshot.** Because your delta is a function of spot, vol, and time, a book set neutral this morning drifts by afternoon — and a short-gamma book drifts *the wrong way* (it gets longer as the stock rises and shorter as it falls, leaning into every move). Re-read your net delta continuously against the band, not once a day. The difference between managing a hedge and being managed by it is whether you're watching the residual build before it breaches, or discovering it after a move you didn't trade through.

![The delta-hedging playbook as a decision flow: pick a band, price the cost into implied volatility, accept the residual gamma, and manage the gap separately](/imgs/blogs/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral-7.png)

The decision diagram above is the routine on one screen. Start with the position's gamma and your cost of trading; from those, set a band width (tight if gamma is high or costs are low, wide otherwise). Run the hedge against the band, not a clock. Price the expected hedging cost into the volatility at which you'll sell — or require realized to clear implied plus cost if you're buying. Size so the irreducible residual gamma is survivable. And carve out the gap as a separate risk you manage by reducing gamma into the dangerous windows, never by trading more shares. Follow the flow and the "riskless" hedge becomes what it really is: a cost you've measured, priced, and chosen to pay — which is the only honest way to stay neutral.

**The single habit to build:** before you put on any hedged option position, write down three numbers — your expected hedging cost (so you know the vol you need), your rehedge band (so you know when you'll trade), and your worst gap loss (so you know what you can't hedge). If you can't produce all three, you don't have a hedged position; you have a short-gamma bet you haven't priced. The desk in the hook had a beautiful real-time delta — and none of those three numbers. That's why it lost money staying perfectly, expensively, fatally neutral.

## Further reading & cross-links

- **[Delta: Direction, Exposure, and the Hedge Ratio](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio)** — what delta is, the hedge ratio, and why a delta-neutral book is only neutral for an instant.
- **[Gamma: The Greek That Bites — Curvature, Convexity, and the Toxic Short](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short)** — the residual you carry between rehedges, and why a short-gamma hedge bleeds on every swing and dies on a gap.
- **[Gamma Scalping: Turning a Long Straddle into a Vol Harvest](/blog/trading/options-volatility/gamma-scalping-turning-a-long-straddle-into-a-vol-harvest)** — the mirror image: for a long-gamma trader, the discrete hedging "error" is the source of the P&L.
- **[The Net Greeks of a Position: Building Your Risk Dashboard](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard)** — why gamma is hedged with options not shares, and how to read the whole book's risk before a gap finds it.
- **[Implied vs Realized Volatility: The Trade at the Heart of Options](/blog/trading/options-volatility/implied-vs-realized-volatility-the-trade-at-the-heart-of-options)** — why implied trades above realized, and why hedging cost is a big part of the reason.
- **[How an Options Market-Maker Thinks: The Other Side of Your Trade](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade)** — the dealer who hedges your trade and prices the cost of doing so into the option (forward reference).
- **[Liquidity, Bid-Ask Spreads, and Getting Filled: The Hidden Tax](/blog/trading/options-volatility/liquidity-bid-ask-spreads-and-getting-filled-the-hidden-tax)** — the spread and market impact you pay on every rehedge (forward reference).
- **[Black-Scholes](/blog/trading/quantitative-finance/black-scholes)** — the replication argument every claim here builds on, with the full derivation we deliberately did not repeat.
- **[Itô's Integral and Itô's Lemma](/blog/trading/math-for-quants/ito-integral-itos-lemma-math-for-quants)** — the math of `dS` and why the continuous hedge is exact only in the limit.
- **[Risk-Neutral Pricing and the Martingale Measure](/blog/trading/quantitative-finance/risk-neutral-pricing-martingale-measure-quant-interviews)** — the pricing-theory frame in which the hedged option is a riskless bond.

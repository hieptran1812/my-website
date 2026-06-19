---
title: "Position sizing for tail and political risk: the discipline that keeps you in the game"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why legal and geopolitical shocks are fat-tailed, how that breaks naive position sizing, and the discipline to size for ruin and hedge with convexity without bleeding the portfolio dry."
tags: ["risk-management", "position-sizing", "tail-risk", "geopolitics", "hedging", "options", "value-at-risk", "kelly-criterion", "convexity", "correlation"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Legal and geopolitical risks are fat-tailed (rare but huge), so they break the position sizing that works in calm markets; the discipline is to size for survival first and hedge with convexity, not to forecast the shock.
>
> - **Size for ruin, not for the average.** Cap any single tail-exposed position so a 50% loss is a survivable drawdown, not a near-fatal hole. On a \$1,000,000 book, that often means 20-25% per name, not 50%.
> - **VaR understates legal and geopolitical risk** because it is built on a thin-tailed model and stops measuring exactly where these shocks live — the deep left tail. Expected shortfall and stress scenarios go where VaR will not.
> - **Hedge with convexity.** A put bends: its loss is capped at the premium but its crash payoff accelerates. A linear short does not. Convexity is what pays you back many-fold in the one event that matters.
> - **The number to remember:** in a crisis, cross-asset correlations jump from roughly 0.2 toward 0.9. Diversification, the thing you were counting on, fails at the exact moment you need it.

On 24 February 2022, Russian forces crossed into Ukraine. Within hours, Brent crude jumped from \$96.84 to \$99.08 and kept climbing to \$127.98 by 8 March; European natural gas (TTF) was already elevated and would spike to EUR339/MWh by late August. The Russian rouble collapsed, roughly \$300bn of Russia's \$643bn in foreign reserves was frozen by Western governments, and entire categories of "safe" exposure — Russian sovereign bonds, blue-chip Russian equities, the rouble itself — went to something close to zero or became untradeable overnight. Funds that had sized those positions off a calm-market risk model, where Russian assets looked merely cheap and a little volatile, discovered that the real risk was not volatility. It was a legal and geopolitical regime change that rewrote the rules of ownership in a weekend.

This is the recurring shape of legal and geopolitical risk. It does not arrive as a gentle drift you can trade around. It arrives as a jump: a sanction that strands an asset, a court ruling that halves a stock, a tariff that reprices a supply chain, an invasion that closes a market. The Caldara-Iacoviello Geopolitical Risk index spiked to 512 after 9/11 — more than five times its long-run mean of 100 — and to 277 in February 2022. These are not the small, frequent wiggles that fill a normal distribution. They are the fat tail.

What makes these shocks uniquely dangerous to a portfolio is not just their size but their *structure*. An ordinary market drawdown is something your risk model has seen before in some form; it can be measured, stress-tested against history, hedged with instruments that track it. A legal or geopolitical shock can rewrite the very thing the model assumes is fixed — that you can sell what you own, that the contract will be honored, that the asset is yours. When the rules themselves are the variable, the loss can be 100%, it can be instantaneous, and it can hit an asset that showed no warning in its price history at all. That is a categorically harder risk to manage than mere volatility, and it demands a categorically more conservative approach to size.

Every other post in this series teaches you to *read* a rule-change early: how a regulation moves from proposed to final, how a sanction transmits to prices, how to trade a regulatory event. This post is the other half of the discipline — the part that keeps you solvent when you read the shock wrong, or right but too early, or when a shock arrives that nobody read at all. It is the risk-management spine of the whole series, and it rests on one uncomfortable fact: you cannot reliably forecast the timing of a fat-tailed shock, so you must instead **size and structure your book so that no single one of them can end you.**

![The tail-risk discipline showing a fat-tailed shock branching into a ruin path and a survival path](/imgs/blogs/position-sizing-for-tail-and-political-risk-1.png)

## Foundations: fat tails, ruin, and the tools to measure both

Before any sizing rule makes sense, we have to define the vocabulary precisely. Each of these terms is a load-bearing piece of the discipline, so we build them from zero.

### The normal distribution and why markets are not normal

A **distribution** is just a description of how likely each possible outcome is. The **normal distribution** — the famous bell curve — is the default model behind most textbook risk math. It has a convenient property: outcomes far from the average are *vanishingly* unlikely. A move of three standard deviations (three "sigma") has about a 0.1% chance on any given day; a five-sigma move is supposed to happen roughly once every 14,000 years.

Markets do not obey this. Five-sigma days happen every few years. The 1987 crash was a 20-plus-sigma event by the normal model's reckoning — which, under that model, should never have happened in the history of the universe, let alone on a Monday. The reason is that real return distributions are **fat-tailed** (also called *leptokurtic*): they have far more probability mass out in the extremes than a bell curve allows.

![Fat-tailed distribution with a heavier left tail than the normal distribution at the same peak](/imgs/blogs/position-sizing-for-tail-and-political-risk-2.png)

The figure above is the single most important one in this post. Both curves have the *same peak* — they agree about calm, everyday returns, which is why a normal model looks fine most of the time. The danger hides in the tails. Out at a "three-sigma" loss, the fat-tailed curve carries roughly six times the probability of the normal one. A model that assumes normality is not slightly wrong about crashes; it is wrong by multiples, and always in the direction that hurts you.

The technical name for "how fat is the tail" is **kurtosis** — the fourth statistical moment, which measures how much of a distribution's variance comes from rare extreme moves versus frequent small ones. A normal distribution has a kurtosis of 3; real equity-return distributions routinely show kurtosis of 6, 10, or higher, and the excess is concentrated in the tails. There is a second wrinkle: financial returns are also *negatively skewed* — the left tail is fatter than the right. Crashes are sharper and faster than rallies; markets "take the stairs up and the elevator down." For legal and geopolitical risk the skew is especially severe, because a rule-change that *destroys* value (a ban, a freeze, a block) is far more common and more violent than one that *creates* an equivalent windfall overnight.

**Why are legal and geopolitical shocks especially fat-tailed?** Because they are *discontinuous changes to the rules of the game*. A normal-ish return comes from many small independent buyers and sellers nudging a price; by the central-limit logic, the sum of many small independent nudges tends toward a bell curve. A sanction, a ruling, or an invasion violates both assumptions at once: it is not small, and it is not independent of everything else — it is a single switch that flips the regime for an entire asset, sector, or country simultaneously. Ownership becomes illegal, a merger is blocked, a border closes — and the price gaps to a new level with no trading in between. There is no smooth path from "tradeable" to "frozen." That gap is the fat tail made concrete, and it is why the calm-market statistics that fill the body of the distribution tell you almost nothing about the events that actually decide whether you survive.

This has a direct consequence for measurement: any statistic estimated from a calm sample will *systematically* understate fat-tailed risk, because the sample contains the body of the distribution and almost none of the tail. The longer the calm stretch you measure, the more confident — and the more wrong — your risk estimate becomes. That paradox, calm breeding false confidence, is the thread running through every failure case later in this post.

### Tail risk and the left tail

**Tail risk** is the risk of an outcome far from the average — and in practice we care almost entirely about the **left tail**, the region of large losses. (The right tail, large gains, is a happy surprise, not something you manage against.) The defining feature of tail risk is that it is *rare but enormous*: it contributes almost nothing to your day-to-day P/L and then, occasionally, contributes everything. A book can look brilliant for years precisely because it is quietly accumulating left-tail exposure that has not yet paid out — short volatility, oversized in a single political bet, long an asset whose legal status is a coin flip.

### Value-at-risk and where it fails

**Value-at-risk (VaR)** is the most common single number for downside risk. The 95% one-day VaR is the loss you expect to *not* exceed on 95 of 100 days. If your 95% daily VaR is \$1,000,000, then on a typical day you should lose less than that, and on the worst 5 days in 100 you lose more — but VaR says *nothing about how much more.*

That last clause is the whole problem. VaR is a *threshold*, not a *magnitude*. It tells you where the bad 5% begins and then goes silent about everything inside it. Worse, the standard ways to estimate VaR either assume a normal distribution (which, as we saw, is wrong about exactly this region) or extrapolate from recent history (which, in a calm stretch, contains no crash to learn from). So a calm market produces a low VaR, which invites larger positions, which makes the eventual tail event more lethal — VaR can actively *encourage* the exposure it is supposed to flag.

#### Worked example:

Take a \$10,000,000 book whose 99% one-day VaR is reported at \$200,000. A risk committee reads that as "in the worst 1% of days we lose about \$200,000" and signs off on the positions. Then a geopolitical shock hits and the book drops 9% — \$900,000 — in a single session. VaR was not "wrong" in a narrow sense: a loss beyond \$200,000 is exactly what the 1% tail is *supposed* to contain. But VaR never said the tail loss would be \$900,000 rather than \$250,000. The lesson: VaR tells you where the trapdoor is, not how deep the fall. For fat-tailed legal and geopolitical exposure, the fall is far deeper than the threshold implies.

### Expected shortfall: measuring inside the tail

The fix for VaR's silence is **expected shortfall (ES)**, also called conditional VaR or CVaR. Where VaR asks "what is the threshold of the worst 5%?", expected shortfall asks "*given* that we are in the worst 5%, what is the average loss?" It measures the magnitude inside the tail, not just its edge. For a normal distribution ES is only modestly larger than VaR; for a fat-tailed distribution it can be *multiples* larger, which is precisely why regulators (Basel's market-risk framework) moved bank capital requirements from VaR toward expected shortfall. For our purposes the takeaway is simpler: **measure with a tool that looks inside the tail, because that is where legal and geopolitical risk lives.**

Expected shortfall has a second virtue that matters for portfolios: it is a *coherent* risk measure, meaning (among other things) that the ES of a combined portfolio never exceeds the sum of the parts' ES — diversification can only help, never hurt, by this measure. VaR lacks that property; in pathological cases the VaR of a diversified book can be *larger* than the sum of the standalone VaRs, which makes it treacherous to aggregate across desks. None of this rescues either measure from the deeper problem — both are only as good as the distribution you feed them, and a calm sample feeds them a thin-tailed lie. That is why the serious practice is not to pick the perfect single number but to pair a statistical measure with explicit **stress scenarios**: "what does this book lose if a sanctions regime freezes our largest position to zero?" or "what if our four 'independent' positions all gap 30% together?" The scenario does not need a probability; it just needs to be survivable. Stress testing is the discipline of looking directly at the tail instead of estimating its shape from the body.

### Convexity: the shape that pays off in a crash

**Convexity** describes a payoff that *curves* rather than running in a straight line — specifically, one whose gains accelerate as the move gets larger. The defining convex instrument is the **option**. Buy a **put** (the right to sell at a fixed strike price) and your payoff is flat until the underlying falls below the strike, then rises faster and faster as the price drops. Your loss is capped at the premium you paid; your gain in a crash is unbounded and *accelerating*.

![Convex put payoff capped at premium versus a linear short payoff](/imgs/blogs/position-sizing-for-tail-and-political-risk-6.png)

The contrast with a **linear hedge** — shorting the underlying outright, or selling a future — is the point of the figure. The short's payoff is a straight diagonal: it gains one-for-one as the market falls and *loses one-for-one if the market rebounds.* The put bends. In the one scenario you are hedging against — a violent gap down — convexity is what turns a small premium into a many-fold payoff, and the capped downside is what lets you hold the hedge through the false alarms without it bleeding you to death on a rally. We will quantify exactly that trade-off below.

### The cost of carry: why hedges bleed

Convexity is not free. A put has a finite life and loses value every day it is held — this decay is called **theta**, and the steady drip of premium you pay to stay hedged is the **cost of carry** (or just *carry*). A hedge is, in effect, an insurance policy: you pay a premium continuously, and most of the time the house — the option seller — keeps it. The reason the premium exists at all is that the seller is taking the convex risk off your hands, and they price it to be paid for bearing it; on average, over many quiet periods, the seller wins. You are not buying a positive-expected-value trade. You are buying *shape* — a payoff that shows up precisely when the rest of your book is collapsing — and paying for the privilege.

Long-volatility positions and VIX products are even worse on this axis, because they bleed through the *roll* (the structural cost of rolling expiring exposure into longer-dated, usually pricier, contracts). The VIX futures curve is usually in *contango* — longer-dated contracts trade above the spot index — so a long-vol position that rolls from a cheaper near contract into a pricier far one loses money on the roll every single month the market stays calm. This is why long-VIX ETPs have lost the overwhelming majority of their value over the long run despite occasional violent spikes: the spikes are real, but the carry between them is relentless.

The central tension of all hedging is this: **the protection that pays the most in a crash is the most expensive to hold in the calm before it.** Manage that tension badly and you can lose more to years of premium bleed than the crash would ever have cost you. Manage it well — finance the convexity, budget the bleed, time the cheap windows — and you can hold a meaningful hedge for a fraction of the naive cost. The whole back half of this post is about managing it well.

### Correlation going to one

**Correlation** measures how much two assets move together, from -1 (perfect opposites) to +1 (lockstep). **Diversification** — spreading a book across many assets — works because in normal times those assets have low correlation: when one zigs, another zags, and the portfolio's swings are smaller than any single holding's. The catch, and it is the cruelest one in finance, is that **correlations are not stable**. In a crisis, they jump toward +1. Everyone sells everything at once to raise cash; the distinctions between assets that justified your diversification dissolve; the whole book becomes one trade.

![Before and after diagram of cross-asset correlation jumping from 0.2 to 0.9 in a crisis](/imgs/blogs/position-sizing-for-tail-and-political-risk-7.png)

This is why diversification, on its own, is *not* a tail-risk strategy. It dampens the ordinary volatility of normal markets and then evaporates in the extreme. A book of stocks, bonds, credit, and emerging-market assets might run at a comfortable 0.2 pairwise correlation for years, then snap to 0.9 in a single shock as the figure shows — and a risk model that was sized off the 0.2 number suddenly finds its losses stacking instead of offsetting. We explore this directly in [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis); for sizing, the operative fact is that you must assume your correlations will betray you when it counts.

The mechanism behind the correlation jump is worth understanding, because it tells you which "diversifiers" are real and which are fake. In a crisis, the dominant driver of every asset's price stops being its own fundamentals and becomes a single shared factor: the scramble for liquidity. Levered players hit margin calls and must sell whatever they can, regardless of quality; risk-parity and volatility-targeting strategies mechanically cut exposure across the board as volatility rises; redemptions force funds to liquidate their *most liquid* holdings (often the good ones) to meet outflows. All of this makes assets that have nothing fundamentally in common move together, because they are all being sold by the same desperate hands at the same moment. The assets that genuinely hold up are the ones a frightened world *buys* in a panic rather than sells — Treasuries (sometimes), the dollar, gold — which is exactly why those, and not a wider basket of risk assets, are the real diversifiers in the tail.

### Position sizing, fixed-fractional, and Kelly

**Position sizing** is the decision of *how much* to put into a given bet — and it matters far more than most beginners expect, often more than the entry price itself. The simplest disciplined approach is **fixed-fractional sizing**: risk a fixed fraction of your book on each position (say, never more than 2% of capital lost on any single trade if it hits its stop). This bounds how much any one mistake can cost.

The more aggressive framework is the **Kelly criterion**, which computes the bet size that maximizes the long-run growth rate of your capital given your edge and the odds. Kelly is mathematically optimal for growth — but it is *brutally* aggressive, it assumes you know your probabilities precisely (you don't), and it can prescribe enormous positions. Almost every serious practitioner uses **fractional Kelly** — a half, a quarter, or less of the full Kelly bet — precisely because the full bet's drawdowns are unbearable and its assumptions are fragile. We will work an example that shows why.

With the vocabulary in place, we can now build the discipline itself.

## Why VaR understates legal and geopolitical tail risk

It is worth dwelling on *why* the standard risk number is structurally blind to the risks this series is about, because the blindness is not a tuning error you can fix by picking a better confidence level.

First, **the model is thin-tailed.** Parametric VaR assumes returns are normal (or close to it). We have already seen that the normal model under-weights extreme losses by multiples. Pushing the confidence level from 95% to 99% does not help much, because the entire shape of the assumed distribution is wrong in the tail — you are reading a more extreme percentile off a curve that doesn't bend the way reality does.

Second, **the history contains no analog.** Historical-simulation VaR replays the recent past. But a novel sanction regime, a first-of-its-kind merger block, a war in a region that has been quiet for decades — these have no precedent in the lookback window. The 2022 weaponization of central-bank reserves was, in scale, unprecedented; no amount of pre-2022 history would have flagged the risk that a G20 central bank's reserves could be frozen. VaR cannot price a risk it has never seen.

Third, **legal and geopolitical risk is often binary and discrete**, not continuous. A court either blocks the merger or it doesn't. A regulator either grants the license or it doesn't. A binary outcome with a large gap is the antithesis of the smooth, small-step world VaR assumes. The right tool for a binary event is an expected-value calculation over the discrete outcomes (which we do in the playbook), not a percentile of a continuous curve.

Fourth, and most insidiously, **VaR is procyclical.** In calm times, realized volatility is low, so VaR is low, so the risk limit allows bigger positions — right as complacency is building and the next shock is incubating. The metric loosens the leash exactly when it should tighten it. This is not a hypothetical; it is a recurring mechanism in financial crises, where risk models gave their cleanest readings in the months before the blow-up.

There is a more general lesson hiding here, and it applies to *every* risk model, not just VaR. A model is a map of the territory, and the map is drawn from data — which means it can only show terrain that has already been walked. Fat-tailed legal and geopolitical shocks are, by their nature, the parts of the territory that are *not yet on the map*: the first sovereign reserve freeze, the first ban on a category of asset, the first conflict in a region that has been quiet for a generation. Any model that purports to assign a precise probability to such an event is overstating its own knowledge. The mature response is not to build a better model of the unknowable, but to *act as if your model is blind to the worst case* — to size as though the tail is fatter than any backtest shows, because it is. Models are for managing the ordinary; rules are for surviving the extraordinary.

The practical conclusion is not "abandon VaR" — it is a fine summary of *ordinary* risk. It is: **never let VaR be the binding constraint on a fat-tailed position.** For legal and geopolitical exposure, the binding constraint must be the ruin constraint, which we turn to now.

## Sizing for survival: the ruin constraint

The deepest idea in tail-risk management is also the simplest. It is not about maximizing returns. It is about **never going to zero**, because zero is absorbing — once you are ruined, no future edge can help you, since you have no capital left to deploy it. The mathematician's version is the *gambler's ruin* problem; the trader's version is a single rule:

> Size every tail-exposed position so that its worst plausible loss is a *survivable drawdown*, not a fatal one.

The mechanism that makes this non-negotiable is the asymmetry of recovery. A loss and the gain needed to undo it are not symmetric: lose 25% and you need +33% to get back to even; lose 50% and you need +100%; lose 80% and you need +400%. The deeper the hole, the more savagely the recovery math turns against you. This is why capping the *depth* of any single loss matters more than chasing the upside.

![Before and after sizing comparison showing a 50 percent position versus a 25 percent position in a tail shock](/imgs/blogs/position-sizing-for-tail-and-political-risk-4.png)

#### Worked example:

Take a \$1,000,000 book and a single position in a name with real legal or geopolitical exposure — say a stock facing a pending regulatory ruling that could halve it.

- **Oversized:** you put 50% of the book, \$500,000, into the name. The ruling goes against it and it falls 50%. Your loss is \$250,000. The book is now \$750,000 — a 25% drawdown, requiring a +33.3% gain just to recover.
- **Sized for the constraint:** you put 25% of the book, \$250,000, into the same name. The same 50% fall costs you \$125,000. The book is now \$875,000 — a 12.5% drawdown, requiring +14.3% to recover.

The position halved, but the *drawdown* and the *recovery burden* both more than halved relative to where the danger lies, because you stayed out of the steep part of the recovery curve. The first path is recoverable but painful; a couple more like it and the fund is in trouble. The second is a bruise. Same view, same shock — the sizing decided whether it was a dent or a disaster.

We can turn this into a formula. If your hard limit is that no single position may cause more than a *D* drawdown of the book, and the worst plausible tail loss on that position is a fraction *L* of the position's value, then the maximum weight *w* you may hold is:

```
w_max = D / L
```

#### Worked example:

Set a firm rule: no single tail-exposed name may cost the book more than 10% in its worst plausible case. You judge the worst plausible case for this name to be a 50% loss (*L* = 0.50). Then:

```
w_max = 0.10 / 0.50 = 0.20
```

You may hold at most 20% of the book — \$200,000 on a \$1,000,000 book — in that name. If a 50% shock then hits, you lose \$100,000, exactly your 10% drawdown budget and no more. The formula forces the *worst-case loss*, not the expected return, to be the thing you solve for first. Flip the inputs around and it also tells you what conviction is required: a position you would only size at 5% is one you have implicitly judged to carry a 200% "worst case" of book-budget — i.e., you should not hold it at all without a hedge.

The crucial discipline here is honesty about *L*. For an ordinary diversified equity, a 50% single-shock loss is pessimistic. For a single name facing a binary legal ruling, a sanctioned-asset risk, or a frontier-market political event, 50% is *optimistic* — the real tail can be 80% or a total loss (a frozen Russian ADR, a delisted Chinese name, a company whose key drug fails FDA approval). Set *L* to the genuinely-plausible bad case, not the comfortable one.

## The hedging toolkit: cost versus convexity

Sizing alone caps your loss by holding *less*. Hedging lets you hold a position *and* cap the loss — at the price of carry. Every hedge sits somewhere on a two-axis map: how *convex* is its payoff (how much it accelerates in a crash), and how *costly* is it to carry (how much it bleeds in the calm)? The art is matching the instrument to the threat and the budget.

![Matrix of hedging tools arranged by carry cost and convexity](/imgs/blogs/position-sizing-for-tail-and-political-risk-5.png)

Walk the toolkit, from most convex to least:

- **Outright out-of-the-money (OTM) puts** are the purest convex hedge. They pay off explosively in a crash and cap your loss at the premium — but that premium bleeds continuously, and a perpetually-held tail put can cost several percent of the protected notional per year. Best reserved for when protection is *cheap* (low implied volatility) or when a specific catastrophic event is in view. For the mechanics of option pricing, see [options theory](/blog/trading/quantitative-finance/options-theory).
- **Put spreads** (buy a closer put, sell a further-out one) keep most of the convexity while *financing* much of the premium with the sold leg. The trade-off is a *capped* payoff — you give up the deepest-crash protection in exchange for paying far less to carry. For most practitioners hedging a fat-tailed book, this is the workhorse: cheap enough to hold, convex enough to matter.
- **Long volatility / VIX products** profit when volatility spikes — and as the VIX-at-stress-events chart below shows, volatility *does* spike violently in exactly the events we fear. But these instruments suffer a relentless roll cost (the perma-bleed of holding futures up a steep curve), so they are excellent *tactical* hedges around a known catalyst and terrible *strategic* holds.
- **Gold** is the classic crisis ballast: it has low carry (no premium to bleed, just opportunity cost), it has historically held or gained value when financial assets crater, and it carries no counterparty. It is only a *partial* hedge — it does not reliably move one-for-one against equities — but its near-zero carry makes it cheap to hold permanently. We treat its role and sizing in depth in [gold's job in a portfolio](/blog/trading/gold/golds-job-in-a-portfolio-sizing-rebalancing-and-the-permanent-portfolio) and its crisis behavior in [the safe-haven trade](/blog/trading/gold/fear-and-the-safe-haven-trade-how-gold-behaves-in-a-crisis).
- **Defensives and cash** are the lowest-convexity, lowest-carry option. Holding cash, short-duration government bonds, or defensive equities (utilities, staples) does not *pay off* in a crash so much as it *fails to lose*. Cash is dry powder with zero bleed but zero crash kicker; its value is optionality — the ability to buy the dislocation when everyone else is forced to sell.

![VIX close at six stress events with a calm-market baseline of seventeen](/imgs/blogs/position-sizing-for-tail-and-political-risk-3.png)

The chart makes the case for why convex hedges work at all: in calm markets the VIX sits around 17, but it hit 37 in the 2018 "Volmageddon," 83 at the March 2020 COVID low, 30 on the Ukraine invasion, and 39 in the August 2024 yen-carry unwind. A hedge that pays off when volatility triples or quadruples is paying off precisely when your fat-tailed positions are bleeding — and the spikes are large enough that even a small, cheaply-carried convex position can offset a large book loss.

#### Worked example:

You hold a \$1,000,000 equity portfolio and worry about a geopolitical gap-down. You buy a three-month put struck 10% below the current level, costing 1.5% of notional — a premium of \$15,000.

- A geopolitical shock hits and the market falls 30%. Your portfolio loses \$300,000. But your put, struck at -10%, is now 20% in the money, paying intrinsic value of 20% of notional = \$200,000. After the \$15,000 premium, the put nets \$185,000.
- Your hedged loss is \$300,000 − \$185,000 = \$115,000, versus an unhedged \$300,000. For \$15,000 of premium you converted a \$300,000 hole into a \$115,000 bruise.

That is convexity at work: a 1.5% premium bought protection worth 20% of the book in the crash. The capped downside — you can never lose more than the \$15,000 — is what lets you hold the put through the months when the shock *doesn't* come.

## The cost-of-carry problem: paying for protection without bleeding out

Here is the trap that snares disciplined hedgers. Tail events are *rare*. If you hold a convex hedge permanently, you pay the premium every single quiet quarter — and the quiet quarters vastly outnumber the crashes. A perma-hedge can quietly cost you more than the disasters it prevents, turning prudent insurance into a chronic performance drag. The graveyard of "tail-risk funds" that bled to death waiting for a crash that came too late, or to someone else, is real.

#### Worked example:

Take a \$10,000,000 portfolio and a tail-hedge program that rolls three-month OTM puts continuously, costing roughly 3.5% of notional per year after accounting for roll-down — a bleed of \$350,000 per year.

- Over five *calm* years with no crash, the program spends \$1,750,000 and pays off nothing. That is a 17.5% cumulative drag on the book — a brutal headwind that, compounded, can swamp the manager's entire edge.
- Now suppose instead you run a **trigger-based** program: you only put the hedge on in the quarters your process flags as elevated-risk — roughly 25% of the time. The carry drops to about \$87,500 per year, saving \$262,500 annually, while still being protected during the windows when shocks actually cluster.

The arithmetic is stark: perma-hedging is a tax you pay forever; trigger-based hedging is a tax you pay only when the odds justify it. The catch, of course, is that triggers can miss — a shock can arrive in an "all-clear" quarter. The resolution is not to choose one or the other but to *layer* them, which we formalize in the playbook.

There are several ways to pay for protection without bleeding out:

1. **Finance the convexity.** Use put *spreads* rather than outright puts, selling a further-OTM put to fund most of the premium. You cap the payoff but slash the carry — usually the right trade for a strategic hedge.
2. **Budget the bleed.** Decide in advance what fraction of expected return you will spend on tail protection — a *tail-hedge budget* — and never exceed it. A common figure is on the order of 0.5-1% of the portfolio per year for a strategic tail overlay. Treat it as a fixed insurance line item, not a discretionary trade.
3. **Hedge only when it's cheap.** Implied volatility is mean-reverting; protection is far cheaper when the VIX is at 13 than at 30. Buying tail hedges when they are cheap (and *selling* some of the spike when fear is extreme) inverts the bleed — you accumulate convexity in the calm and monetize it in the panic.
4. **Use low-carry ballast for the always-on layer.** Gold and cash bleed almost nothing, so they can sit in the book permanently as a base layer of crisis resilience, with the expensive convex hedges reserved for the trigger windows.

## Common misconceptions

Three beliefs about tail risk are not just imprecise — they are actively dangerous, and each one is corrected by a number.

### "Diversification handles tail risk."

It does not. Diversification reduces the *ordinary* volatility of a portfolio by combining low-correlated assets, but its benefit is computed off normal-times correlations — and those correlations jump toward 1 in a crisis. A book diversified across stocks, credit, and emerging markets at 0.2 pairwise correlation might cut its everyday volatility by roughly 40% relative to a single holding. But when correlation snaps to 0.9 in a shock, that 40% benefit collapses toward zero: the legs that were supposed to offset each other all fall together. In 2008 and again in the March 2020 COVID crash, "diversified" portfolios fell almost as one because *everything correlated to the dash for cash.* Diversification is a volatility tool, not a tail tool. The tail tool is convex hedging plus the ruin constraint.

### "VaR captures the downside."

VaR captures the *threshold* of the downside, not its *depth* — and for fat-tailed risk the depth is what kills you. Recall the worked example: a book with a reported 99% one-day VaR of \$200,000 lost \$900,000 in a single shock. VaR was not violated in spirit — a loss beyond the threshold is what the tail is for — but the magnitude was 4.5 times the headline number, because VaR is silent about everything past the threshold and is usually estimated off a thin-tailed model that under-weights the very region in question. Use expected shortfall and explicit stress scenarios; never let the VaR number lull you into thinking the worst case is anywhere near it.

### "Hedging is free, or always worth it."

Hedging is *insurance*, and insurance has a premium that the seller keeps most of the time. A perma-hedge on a \$10,000,000 book can cost \$350,000 a year — \$1,750,000 over five calm years — and pay off nothing if the crash doesn't come on your watch. Hedging is worth it *when* the protection is cheap relative to the risk, *when* a specific catalyst is in view, or *as a budgeted, financed overlay*. It is a chronic loser when held indiscriminately and permanently at full size. The skill is not "hedge" versus "don't hedge"; it is *paying for convexity efficiently.*

## How it shows up in real markets

The textbook ideas are vivid in real episodes. But first, the raw fact that makes all of them possible: geopolitical risk does not build gradually, it *jumps*. The Caldara-Iacoviello Geopolitical Risk index, which tracks the share of newspaper coverage devoted to geopolitical tensions, sits near its long-run mean of 100 in normal times and then leaps to multiples of that on a shock — 512 after 9/11 (five times normal), 290 around the 2003 Iraq war, 277 in February 2022.

![Geopolitical Risk index spiking above its long-run mean at major shocks](/imgs/blogs/position-sizing-for-tail-and-political-risk-8.png)

The chart is the geopolitical analog of the VIX-spike chart: both show that the variable you are managing against is not a gentle drift you can lean into, but a step function that resets the world in days. A position sized for the GPR-100 world is the wrong size for the GPR-500 world — and you do not get advance notice of the transition. That single observation is why the discipline is *structural* (size and shape your book so it survives the jump) rather than *predictive* (forecast the jump). Now the episodes.

**A tail event that ruined an oversized position.** The classic modern case is Long-Term Capital Management in 1998. LTCM ran enormous leveraged convergence trades sized off a risk model that assumed near-normal returns and stable correlations. When Russia defaulted on its domestic debt in August 1998 — a sovereign-legal shock — correlations across its many "independent" trades jumped toward 1, the trades all moved against it at once, and the leverage that had amplified its gains amplified the losses into insolvency. The fund lost roughly 90% of its capital in weeks and required a Fed-orchestrated bailout. The lesson the survivors drew was exactly the ruin constraint: a position sized so its worst case is a survivable drawdown cannot be killed by a single fat-tailed shock, however well-modeled the calm looked. LTCM's risk model was not crude — it was built by Nobel laureates — and that is the chilling part. The sophistication of the model was no defense, because the model's *assumptions* (near-normal returns, stable correlations) were the thing the shock destroyed. A simpler manager with a hard rule that "no cluster of correlated trades may exceed X% of capital" would have survived where the geniuses did not.

More recently, the 2021 Archegos blow-up showed the same shape — concentrated, heavily-levered single-name swaps that worked beautifully until a few of the names gapped down together, wiping out the family office in days and inflicting more than \$10bn in combined losses on its prime brokers. And the 2022 Russia sanctions, with which this post opened, are the purest legal-tail case: there was no "volatility" in a frozen Russian ADR to warn you in advance; the asset went from liquid to legally untradeable in a single weekend. No position-sizing model built on price history could have flagged it, because the risk was not in the price series at all — it was in the possibility of a rule-change that the price series never contained. The only protection was to have sized the exposure, *ex ante*, small enough that a total loss was survivable. We trace the full mechanism of that episode in [the weaponization of finance](/blog/trading/law-and-geopolitics/the-2022-russia-sanctions-and-the-weaponization-of-finance).

**A convex hedge that paid off.** The COVID crash of March 2020 is the canonical case. The VIX rocketed from the teens to 82.7 on 16 March — the highest close on record. Funds holding convex protection (long puts, long volatility) saw those positions multiply many-fold in days, offsetting the collapse in their risk assets and, crucially, giving them dry powder to buy the dislocation near the lows. The asymmetry was the point: a small premium spent on out-of-the-money puts in the calm of January and February returned multiples of itself in March, because the payoff *accelerated* as the move grew — convexity doing exactly what it is for. One widely-discussed tail-risk fund reportedly returned several thousand percent on its hedging book in the first quarter of 2020, an absurd-sounding number that is simply what deep convexity does when the move is large enough: the further out-of-the-money the put, the cheaper it was to own and the more explosively it pays as the price blows through it.

The same dynamic played out in miniature in the August 2024 yen-carry unwind, when the VIX spiked to 38.6 intraday as a sudden Bank of Japan policy shift and a soft US jobs print forced a violent deleveraging — and convex hedges paid out before the market round-tripped within weeks. The brevity of that episode is itself instructive: a linear short held through August 2024 would have given back most of its crash gain on the rebound, while a put that was *sold* into the spike locked in the convex payoff and left the holder flat into the recovery. Knowing to *monetize* a convex hedge at the peak of fear is as much a part of the discipline as owning it in the first place.

**The carry cost of perma-hedging.** The flip side is the steady erosion suffered by always-on tail protection. Several well-known "tail-risk" strategies bled persistently through the long, calm bull markets of the 2010s, when the VIX sat near its 17 baseline for years at a stretch and the crash they were built for simply did not come. Investors who held them throughout paid the premium quarter after quarter and underperformed badly — then, when COVID finally hit, many had already capitulated and sold the hedges near the lows. The episode is the cautionary half of the convexity story: the same instrument that saved disciplined hedgers in March 2020 ruined the patience of those who held it indiscriminately for a decade.

The deeper trap is *behavioral*, not arithmetic. A perma-hedge that bleeds for years tests the holder's conviction precisely when the case for it feels weakest — after a long calm, the crash feels abstract and the premium feels like a certainty. So the hedge gets cut, or shrunk, or "optimized" right before it would have paid. This is why the surviving tail-risk programs are the ones that are *budgeted and automatic* rather than discretionary: a fixed annual hedge budget that the manager cannot talk themselves out of, deployed through financed structures cheap enough that the bleed never becomes unbearable. Convexity pays, but *only if you can afford to carry it until it does and have the discipline not to abandon it the quarter before* — which is why the financing, budgeting, and trigger discipline of the previous section is not optional.

These episodes also connect to the broader machinery of how geopolitical risk gets priced as a risk premium in the first place, which we develop in [geopolitics as a market factor](/blog/trading/law-and-geopolitics/geopolitics-as-a-market-factor-the-risk-premium-channel), and to how a hedge fund builds risk management into its very structure, covered in [risk management as a business function](/blog/trading/hedge-funds/risk-management-as-a-business-function).

## The playbook: sizing and hedging for fat-tailed risk

Here is the discipline as a concrete, repeatable process. It has four parts: size, hedge selection, the hedge budget, and the invalidation rules.

### 1. The sizing rule for tail-exposed positions

Before entry, do three things in order:

1. **Estimate the genuine worst case, *L*.** Not the comfortable case — the plausible catastrophe. For a binary legal ruling or a sanctionable asset, assume *L* is large (50-100%). Be honest; this number does the most work.
2. **Set the per-position drawdown budget, *D*.** Decide how much of the *book* you are willing to lose to any single name's tail. For a concentrated book this might be 10%; for a diversified one, 3-5%.
3. **Size at `w_max = D / L`.** Hold no more than that weight. A 50% worst case and a 10% drawdown budget cap you at 20% of book. A 100% worst case (total loss is plausible) and a 5% budget cap you at 5%.

Then layer the *aggregate* constraint: assume your positions' tail losses correlate to 1 in a crisis, and check that the *sum* of your tail-exposed weights, all hit at once, stays inside the book's total survivable drawdown. Diversification will not save you in the tail, so do not let it inflate your aggregate sizing. This is the step most often skipped — each position passes its individual sizing test, but the *cluster* of them, all exposed to the same kind of shock, would blow through the book's total drawdown budget if they moved together. And in a crisis they will.

#### Worked example:

You hold five separate names, each individually sized at the 10%-book worst-case limit using `w_max = D/L`. Individually, each is fine — no single one can cost more than 10%. But suppose all five share a common exposure: they are all emerging-market names sensitive to the same geopolitical fault line. In calm markets their pairwise correlation is 0.2, so a naive risk model treats the cluster as well-diversified and the combined worst case as far below the sum. Now a shock hits and the correlation jumps to 0.9. The five positions, each down toward its worst case, now lose *together*: if each is sized to a 10% book-drawdown worst case and they all hit at once, the book faces a 50% drawdown — catastrophic, and requiring a +100% gain to recover. The fix is to set an *aggregate* limit on any correlated cluster — say, no group of names exposed to the same shock may collectively risk more than 15-20% of the book in the tail — and to size *within* that cap, not just per-name. The per-name limit protects you from one mistake; the aggregate limit protects you from the correlation jump.

For *directional, asymmetric bets* — a political wager with a real edge — use fractional Kelly to size, never full Kelly.

#### Worked example:

You judge a pending court ruling will go your way with probability *p* = 40%; if it does, the position gains 60%; if it doesn't, it loses 25%. The expected value per dollar is 0.40 × 0.60 − 0.60 × 0.25 = +0.09, a healthy 9% edge. The Kelly fraction for a bet that wins a fraction *b* and loses a fraction *a* is:

```
f* = p/a - q/b = 0.40/0.25 - 0.60/0.60 = 1.60 - 1.00 = 0.60
```

Full Kelly says bet **60% of the book** on a single binary court ruling. That is insane — it ignores that your 40% probability is itself a guess, and a single adverse ruling at that size would inflict a 15% book drawdown on a position you were *supposed* to win. Quarter-Kelly cuts it to 15% of book (\$150,000 on a \$1,000,000 book), and even that should be checked against your ruin constraint. The point of fractional Kelly is humility: it shrinks the bet to absorb the error in your probability estimate, which for legal and geopolitical events is always larger than you think. The Kelly framework itself is developed further in [the Kelly criterion for sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews).

### 2. Hedge selection: cost versus convexity

Match the hedge to the threat and the budget:

- **Known catalyst, defined date** (an election, a ruling, a sanction deadline): buy *event-specific* convexity — short-dated puts or call/put structures expiring just after the event, sized to the binary. Buy when implied volatility has not yet bid up the event.
- **Strategic, always-on protection:** use *put spreads* (financed convexity) plus *low-carry ballast* (gold, cash). Reserve outright puts and long-vol for trigger windows.
- **Cheap-protection windows** (low VIX): accumulate convexity opportunistically; this is when insurance is on sale.
- **Avoid** perma-holding the most expensive convex instruments (outright far-OTM puts, long-VIX futures) at full size — their carry will exceed their expected payoff over any normal horizon.

#### Worked example:

Compare two hedges for a \$1,000,000 portfolio against a geopolitical gap-down.

- **Convex (put spread):** buy a 10%-OTM put, sell a 25%-OTM put, for a net premium of 0.6% = \$6,000. In a 30% crash it pays the 15% spread between the strikes = \$150,000, netting \$144,000. On a +20% *rebound*, it loses only the \$6,000 premium.
- **Linear (short):** short 30% of the portfolio notional (\$300,000) via futures. In the same 30% crash it gains 30% of \$300,000 = \$90,000. But on a +20% rebound it *loses* 20% of \$300,000 = \$60,000.

The put spread paid more in the crash (\$144,000 vs \$90,000) *and* lost vastly less on the rebound (\$6,000 vs \$60,000). That asymmetry — pay more when right, lose little when wrong — is the entire reason to prefer convex hedges over linear ones for tail protection, and it is why the capped payoff of the spread is a price worth paying for the bounded downside.

### 3. The tail-hedge budget

Treat tail protection as a fixed line item:

- **Set an annual budget** — commonly 0.5-1% of the portfolio for a strategic overlay — and do not exceed it. This is your insurance premium; size the hedges to fit it.
- **Layer the protection:** an always-on base of low-carry ballast (gold, cash) plus a trigger-activated layer of financed convexity. The base is cheap enough to hold forever; the convex layer goes on only in flagged windows.
- **Monetize spikes:** when fear is extreme and your convex hedges have paid off, *sell some of the gain* — harvesting the spike both funds future premium and avoids round-tripping the payoff.

### 4. What invalidates the discipline

Know the failure modes of your own framework:

- **Your *L* was too optimistic.** If a "worst case" you sized for 50% turns out to be a total loss (a freeze, a delisting, a fraud), your sizing was wrong at the root. Revisit *L* upward for anything with binary legal exposure.
- **The hedge has basis risk.** A hedge that doesn't track the thing you own (an index put against a concentrated single-name book) can fail to pay in your specific shock. Check that the hedge actually covers *your* exposure.
- **Carry exceeded the payoff for too long.** If your hedge budget is being blown by perma-bleed, your *selection* is wrong — move toward financed convexity and trigger-based activation.
- **You relied on diversification in the tail.** If your aggregate sizing assumed low correlations would hold, a crisis will breach it. Re-stress every aggregate limit at correlation = 1.
- **The regime changed.** If the legal or geopolitical environment shifts structurally — a new sanctions regime, a new antitrust posture, a new conflict — the *base rate* of tail events has risen, and your trigger and budget should reset higher. The post-2022 world, in which reserve freezes and weaponized finance are now established tools, carries a permanently higher legal-tail base rate than the world before it; sizing and budgets calibrated to the calmer 2010s are now too loose.
- **You let a winning position drift past its limit.** A position that rallies grows as a share of the book without any decision on your part — and silently breaches the sizing rule you set at entry. Rebalance back toward the limit on the way up; the most common way a disciplined sizer ends up oversized is by *winning* and never trimming.

A final note on temperament. Every rule here trades expected return for survival, and in any given calm year the disciplined book will look too cautious — under-sized, over-hedged, dragging on a benchmark that rewards the reckless. That is the cost of the strategy, and it is real. The payoff is asymmetric and lumpy: you give up a little in the many normal years and keep everything in the one abnormal one. Over a full cycle that includes its inevitable fat-tailed shock, the manager who was still solvent and still had dry powder to buy the dislocation compounds from a far higher base than the one who was forced to sell at the bottom. Survival is not a constraint on the strategy. Survival *is* the strategy.

The throughline is the opening thesis: you cannot forecast the timing of a fat-tailed legal or geopolitical shock, so you do not try. You size every position so that no single shock can ruin you, you hold convex protection you can afford to carry until it pays, and you assume your diversification will fail at the worst moment. Do that, and the rare-but-huge event that wrecks the over-sized and the over-confident becomes, for you, a survivable bruise — and the dry powder to buy what everyone else is forced to sell. That is what it means to still be in the game.

## Further reading & cross-links

Within this series:

- [Geopolitics as a market factor: the risk-premium channel](/blog/trading/law-and-geopolitics/geopolitics-as-a-market-factor-the-risk-premium-channel) — how geopolitical risk gets priced as a premium in the first place.
- [How to trade a regulatory event](/blog/trading/law-and-geopolitics/how-to-trade-a-regulatory-event) — sizing and structuring around a known legal catalyst.
- [Scenario analysis and war-gaming geopolitical events](/blog/trading/law-and-geopolitics/scenario-analysis-and-war-gaming-geopolitical-events) — building the worst-case *L* this post depends on.
- [The law, policy, and geopolitics playbook](/blog/trading/law-and-geopolitics/the-law-policy-and-geopolitics-playbook) — the series capstone that ties reading rule-changes to positioning.

Going deeper on the mechanisms:

- [Options theory](/blog/trading/quantitative-finance/options-theory) — how the convexity of a put is priced.
- [The Kelly criterion for sequential betting](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews) — why fractional Kelly, in depth.
- [Gold's job in a portfolio: sizing, rebalancing, and the permanent portfolio](/blog/trading/gold/golds-job-in-a-portfolio-sizing-rebalancing-and-the-permanent-portfolio) — sizing the low-carry ballast layer.
- [Fear and the safe-haven trade: how gold behaves in a crisis](/blog/trading/gold/fear-and-the-safe-haven-trade-how-gold-behaves-in-a-crisis) — gold's behavior when correlations go to one.
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — why diversification's correlations are regime-dependent.
- [Risk management as a business function](/blog/trading/hedge-funds/risk-management-as-a-business-function) — embedding the ruin constraint into a fund's structure.

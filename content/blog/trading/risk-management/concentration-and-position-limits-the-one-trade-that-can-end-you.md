---
title: "Concentration and Position Limits: The One Trade That Can End You"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Build the math of concentration from zero: why a single position can sink an otherwise great book, how to set single-name and single-theme limits, and how to size every name to a maximum-loss budget so no surprise can end you."
tags: ["risk-management", "concentration-risk", "position-limits", "position-sizing", "single-name-risk", "diversification", "max-loss-budget", "survival"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **One sentence:** No matter how good the rest of your book is, one position that is too big can end you — so the discipline of survival is capping how much any single name, or single theme, can ever cost the whole portfolio.
> - **A loss in one position hits the whole book in proportion to its weight.** A 20% position down 50% is a clean −10% to the entire portfolio; the size of the hit is just the weight times the move.
> - **Concentration is multiplicative, and that is what makes it dangerous.** Double the position weight and you double the book damage from the same surprise; the loss scales with how big you let the position get.
> - **A hard position limit converts a runaway worst case into a flat one.** Cap any single name at, say, 10% of the book and the very worst a single-name shock can do is bounded, no matter how tempting the trade looked.
> - **Size every name to a maximum-loss budget, not to a target return.** Decide the most you are willing to lose to one name first, divide by that name's worst-case move, and the largest position you may hold falls right out.
> - **Almost every famous blow-up is a concentration story.** Archegos and Amaranth were not killed by being wrong across a diversified book — they were killed by one concentrated, levered bet that the rest of the portfolio could not absorb.

There is a particular kind of trader who is right far more often than wrong, runs a thoughtful, well-researched book, manages risk on ninety-nine of a hundred positions — and still blows up. Not because the edge was fake. Not because the market was unfair. Because on the hundredth position, the one they were *most* sure about, they let the size run, and that one trade was big enough to take down everything the other ninety-nine had built. This is the most common way real money dies, and it has almost nothing to do with being wrong. It has to do with being *concentrated*.

The survival thesis of this whole series is that your first job is not to make money — it's to not blow up, because [you can only compound if you're still in the game](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain). Concentration is the single fastest route to violating that rule. A diversified book that loses on a few names can be unpleasant; a concentrated book that loses on its one dominant name can be terminal. The difference is not the quality of the analysis. It's the arithmetic of how much a single position is allowed to cost.

This post is about that arithmetic, and about the discipline that contains it. We are going to build the math of concentration from the ground up — how much a single position can drag on the whole portfolio, why a 20% position down 50% is exactly a −10% portfolio hit, what single-name and single-theme limits actually buy you, and how to size each position to a maximum-loss budget so that no surprise, however large, can end you. By the end you should be able to look at any position in your book and answer one question precisely: *if this one trade goes against me as badly as it plausibly can, how much of my whole portfolio does it cost?* If you can't answer that, you don't know your real risk — you know your hopes.

![One position one bad day portfolio loss versus single position weight for a fixed fifty percent adverse move marking five fifteen and thirty percent positions](/imgs/blogs/concentration-and-position-limits-the-one-trade-that-can-end-you-1.png)

Look at the figure above before reading on, because it contains the entire thesis on one line. The horizontal axis is how big a single position is, as a fraction of your whole portfolio. The vertical axis is how much that position costs the whole book if it drops by half. The line is dead straight, because the relationship is multiplication: a 5% position down 50% costs the book 2.5%, a 15% position costs 7.5%, and a 30% position costs a full 15% — past the drawdown threshold where most serious risk frameworks force you to act. The same 50% drop. The same bad luck. The only thing that changed was how much you let the position weigh. That straight line is the whole game, and the rest of this post is about staying on the safe end of it.

## Foundations: the building blocks of concentration risk

Before we can size or limit anything, we need a handful of terms defined from absolute zero. None of this assumes a finance background; every piece is just careful bookkeeping. Skip nothing — the entire playbook is built from these few definitions.

### What a "position" and a "book" actually are

A **position** is a single bet: you own (or are short) some amount of one thing — one stock, one bond, one currency pair, one commodity contract, one crypto token. The thing itself doesn't matter for this post; concentration risk is asset-class-agnostic. What matters is the *dollar value* of that bet relative to everything else you hold.

A **book** (or portfolio) is the collection of all your positions taken together. When people say "the book is up 2%" they mean the whole collection gained 2% of its total value. Your book is the unit of survival: you don't go broke because one position lost money, you go broke because the *book* lost too much. So every position has to be judged not on its own terms but by how much it can move the book.

### Position weight: the one number that drives everything

The **weight** of a position is its dollar value divided by the total value of the book. If your book is worth \$100,000 and you hold \$20,000 of one stock, that position's weight is \$20,000 / \$100,000 = 20%. Weight is the bridge between a single position's fortunes and the whole portfolio's, and it is the single most important number in this entire post. Hold that idea: *a position can only hurt the book in proportion to its weight.* A 2%-weight position simply cannot sink you, no matter how badly it does. A 50%-weight position can sink you almost by itself. Same analysis, same conviction, wildly different danger — and the only difference is the weight.

### How a position's loss flows into the book

Here is the mechanical heart of concentration, and it is just multiplication. If a position has weight **w** (as a fraction of the book) and the thing you own moves by **m** (a return, positive or negative), then the book moves by **w × m** because of that one position. A 10% position (w = 0.10) that drops 30% (m = −0.30) costs the book 0.10 × (−0.30) = −0.03, or −3%. That's it. There is no hidden complexity in how a single loss reaches the portfolio — it scales linearly with the weight. The danger of concentration is precisely that this multiplication has *no ceiling* until you impose one: as w grows, the same move m does proportionally more damage, all the way up to the point where one position can wipe the book.

### The recovery asymmetry, recalled

We need one fact from earlier in the series, because it's what makes a big single-name loss so much worse than it first looks. A drawdown of size **d** requires a *gain* of **d / (1 − d)** to climb back to even — losses and the gains needed to undo them are not symmetric. A −10% book drawdown needs +11.1% to recover; a −20% drawdown needs +25%; a −50% drawdown needs a brutal +100%. (The full derivation lives in [the asymmetry-of-losses post](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain).) This matters here because concentration produces *large* drawdowns from single events, and large drawdowns sit in the steep, near-unrecoverable part of that curve. A concentrated loss doesn't just hurt — it pushes you into the region of the recovery math where digging out is genuinely hard.

### Diversification, in one sentence

**Diversification** is spreading the book across many positions whose fortunes don't all move together, so that no single one can dominate the outcome. It is, as a [companion post in this series argues](/blog/trading/risk-management/diversification-the-only-free-lunch-and-when-it-works), the closest thing to a free lunch in finance — but only when the positions are actually independent. Concentration is its exact opposite: letting one position (or one cluster of positions that secretly move together) dominate. Through this whole post, concentration is the failure mode and diversification is the discipline that contains it. We are not going to re-derive the math of why uncorrelated bets shrink risk — that belongs to the diversification post and to [the cross-asset treatment of the free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch). Here we focus on the dangerous side: what happens when you *don't* diversify, and one name gets too big.

## The math of one position: how much a single name can cost you

Now we can make the cover figure precise and turn it into a tool. The claim is simple and worth stating as a one-line rule you can carry everywhere:

> **Book hit from one position = position weight × that position's move.**

Everything about concentration is contained in that line. The position's *move* is largely out of your control — you can research it, you can have an edge, but you cannot legislate that a stock won't gap down 50% on a fraud revelation or a missed earnings number or a sector-wide repricing. The position's *weight*, on the other hand, is entirely yours to choose. You decide how big to make it. So the entire discipline of concentration risk reduces to controlling the one factor you control — the weight — given that the other factor — the move — can sometimes be catastrophic.

Let's put real dollars on it, using the two accounts this series carries throughout: a **\$100,000 retail account** and a **\$10,000,000 book**.

#### Worked example: the same drop at three different weights

Start with the \$100,000 account and one stock that is about to drop 50% — a real, if unpleasant, single-day or single-event move, the kind a biotech suffers on a failed trial or a company suffers when a fraud is exposed. We'll size that one position three ways.

**Case A — a 5% position.** You hold \$5,000 of the stock (5% of \$100,000). It drops 50%, so it loses \$5,000 × 0.50 = \$2,500. The book goes from \$100,000 to \$97,500, a **−2.5%** book drawdown. Annoying, fully survivable, recovered with a +2.6% gain on the rest of the book.

**Case B — a 15% position.** You hold \$15,000 of the same stock. The 50% drop loses \$15,000 × 0.50 = \$7,500. The book falls to \$92,500, a **−7.5%** drawdown. That stings — it's most of a typical monthly risk budget gone to one name — but it's still recoverable.

**Case C — a 30% position.** You hold \$30,000. The 50% drop loses \$30,000 × 0.50 = \$15,000. The book falls to \$85,000, a **−15%** drawdown from a *single position*. By the recovery math, you now need +17.6% on the remaining book just to claw back to even, and you've blown clean through the 10% drawdown line that most disciplined traders treat as a hard stop.

The three cases differ only in the weight — 5%, 15%, 30% — and the damage scaled exactly with it: 2.5%, 7.5%, 15%. The conviction was identical; the stock was identical; the bad luck was identical. *The only thing you chose was how much of your survival to hand to one name, and that choice was the entire difference between a shrug and a crisis.*

### Why concentration is multiplicative, not additive

People underrate concentration because they think additively: "it's just one position, how bad can it be?" But the book hit is *multiplicative* in the weight. Going from a 10% position to a 20% position doesn't add a fixed amount of risk — it *doubles* the book damage from any given move. Going from 10% to 40% quadruples it. This is why a book can look fine on ninety-nine positions and be mortally exposed on the hundredth: the danger isn't spread evenly, it's concentrated exactly where the weight is concentrated. A single 40% position carries more book risk than the other sixty positions combined, if those sixty are each 1%.

This is also why your intuition about "I'm diversified, I hold thirty names" can be dead wrong. If twenty-nine of them are 1% each and one is 71%, you do not hold a diversified book — you hold one big bet with a rounding error of decoration around it. The *count* of positions tells you almost nothing; the *distribution of weight* tells you everything. A book's real concentration is about where the weight piles up, not how many tickers are on the screen.

There's a clean way to make "how concentrated am I, really" into a single number, borrowed from economics: the **effective number of bets**. Take each position's weight, square it, add up the squares across the book, and take the reciprocal. For a book of *n* equally-weighted positions, the squared weights are each (1/n)², there are *n* of them, the sum is 1/n, and the reciprocal is *n* — so an equal-weight book of thirty names has an effective number of thirty, exactly what you'd hope. But now take the lopsided book: one position at 71% and twenty-nine at 1%. The sum of squared weights is (0.71)² + 29 × (0.01)² = 0.504 + 0.003 = 0.507, and the reciprocal is about 1.97. That thirty-name book has an *effective* number of bets of roughly two. You are, for risk purposes, holding two positions, not thirty — and the formula sees through the decoration instantly. Squaring the weights is what does the work: it makes big positions count enormously and tiny ones count for almost nothing, which is precisely the right weighting for concentration risk, because a big position can sink you and a tiny one can't. Run this number on your own book occasionally; if it's far below your position count, your diversification is mostly cosmetic.

![Heatmap of portfolio impact as a grid of position weight times single name move with a minus ten percent book drawdown contour and a diagonal of pain](/imgs/blogs/concentration-and-position-limits-the-one-trade-that-can-end-you-2.png)

The figure above is the multiplication table of pain. Each cell is the book hit — position weight (vertical) times single-name move (horizontal). Read across the bottom row: even an 80% crash in a 5%-weight position only dents the book 4%. Now read up the left columns: a 40%-weight position taking a 30% move is a −12% book hit, and an 80% crash there is a portfolio-ending −32%. The black contour is the −10% book-drawdown line — the boundary between "bad week" and "real trouble." Everything above and to the right of it is the danger zone, and notice how you reach it: not by being more wrong, but by being more concentrated. A small position can survive an enormous move; a large position can be killed by a modest one. The diagonal of pain is where size and surprise meet.

#### Worked example: a 25% position gapping −40% on the big book

Now scale up to the **\$10,000,000 book** and a position you were genuinely confident in — so confident you let it grow to 25% of the book, \$2,500,000. Then it gaps down 40% overnight on news you didn't see coming (an earnings miss, a regulatory action, a failed deal — pick your poison; gaps don't ask permission).

The position loses \$2,500,000 × 0.40 = \$1,000,000. The book falls from \$10,000,000 to \$9,000,000 — a **−10%** book drawdown from one name, in one gap, while you were asleep. There was nothing you could do once the gap happened; a stop-loss order doesn't help you across a gap, because the price never traded at your stop — it opened far below it, so your fill is below your stop, not at it. The entire outcome was decided the moment you chose to make the position 25% rather than, say, 8%. At 8% — \$800,000 — the same 40% gap would have cost \$320,000, a −3.2% book hit: a bad day, not a crisis.

*The position's move was fate; the position's weight was a decision, and it was the decision that determined whether a bad night was survivable or not.*

This is the crux. You cannot stop a gap. You can absolutely stop a gap from being a 10% book event, by never letting a single name reach a weight where a plausible gap becomes a portfolio crisis. That control is what position limits are.

## Position limits: trading a runaway worst case for a flat one

A **position limit** is a hard rule that no single position may exceed some fraction of the book — pick a number, say 10%, and enforce it mechanically, regardless of how good the trade looks. It sounds almost too simple to matter. It is, in fact, one of the highest-leverage risk controls that exists, and the reason is geometric.

Without a limit, the worst case a single name can do to your book grows linearly with how big you let that name get — and the temptation to let your best idea get big is strong precisely when the idea looks best, which is often right before it blows up. With a limit, the worst case is *capped*. No matter how convinced you become, no matter how the position drifts up as it works, the most a single-name shock can ever cost you is bounded by the limit. You have converted an open-ended downside into a closed one, and you've done it before you knew which trade would betray you.

![Position size limits and the survival they buy showing worst case portfolio drawdown from one name capped at ten percent versus uncapped](/imgs/blogs/concentration-and-position-limits-the-one-trade-that-can-end-you-3.png)

The figure above shows what the limit buys, for a fixed −60% single-name shock. The red line is the uncapped world: the worst case rides straight up with the largest position you allow — let one name reach 80% of the book and a −60% shock there costs you 48% of everything. The blue line is the same world with a hard 10% single-name limit: it tracks the red line up to 10%, then goes *flat* — because once the cap binds, the largest position can't grow, so the worst case can't grow either. Past the cap, the worst a −60% shock can ever do is 10% × 60% = 6% of the book, forever, no matter how much you wish you'd bet more. The green shaded wedge between the two lines is the drawdown the limit *avoids* — it's the survival you bought by giving up the right to over-concentrate. That wedge is the entire value of the discipline, and it grows exactly in the region where over-concentration is most tempting.

#### Worked example: what a 10% single-name limit guarantees

Take the **\$10,000,000 book** with a hard rule: no single position above 10% of the book, ever. The largest any one name can be is \$1,000,000.

Now take the worst plausible single-name event — a −50% gap on your biggest holding. The most it can cost: \$1,000,000 × 0.50 = \$500,000, a **−5%** book hit. Even a once-in-a-decade −80% catastrophe on that name costs \$1,000,000 × 0.80 = \$800,000, an **−8%** book hit. Painful, but you are unambiguously still in business — the book is at \$9,200,000, you need a +8.7% recovery, and you trade tomorrow.

Compare the same events with *no* limit and a 40% position (\$4,000,000): the −50% gap costs \$2,000,000 (−20% book), and the −80% catastrophe costs \$3,200,000 (−32% book), which by the recovery math needs a +47% gain to undo — the kind of hole careers don't climb out of. The limit didn't change a single thing about your analysis or your edge. It changed the *worst case* from career-ending to survivable, and that is the only thing a limit is for.

*A position limit is a promise you make to your future self in calm weather, that you will not let any one bet become large enough to end you in a storm you can't see coming.*

### Single-name limits versus single-theme limits

Here is where naive position limits fail, and where real practitioners earn their keep. A single-name limit says "no one ticker above 10%." Fine. But suppose you hold ten different regional banks at 8% each. Every single one is *under* the single-name limit. Your book passes the position-limit check with room to spare. And yet you are 80% concentrated in *one bet* — "regional banks" — that will move almost as one in a banking stress event. You have obeyed the letter of the limit and violated its entire spirit.

This is **single-theme** (or single-factor) concentration, and it is the more dangerous cousin of single-name concentration precisely because it hides. Ten 8% positions *look* diversified — ten names, none over the limit — but if they share a common driver (the same sector, the same factor, the same macro sensitivity, the same liquidity source), they are functionally one position. When the theme breaks, they break together, and your "diversified" book takes the full hit of an 80% position. The math is identical to the single-name case, just with the weight summed across the cluster: the book hit is the *total theme weight* times the theme's move.

The discipline, therefore, is to set limits at two levels:
- a **single-name limit** (no one position above x% of the book), and
- a **single-theme limit** (no one cluster of correlated positions above y% of the book, where y is larger than x but still well short of "you bet the firm").

Identifying the themes is its own skill — it's the subject of [the factor-risk post in this series](/blog/trading/risk-management/factor-risk-and-the-hidden-bets-in-your-portfolio), which shows how a book that looks diversified by name can be one giant bet on rates, or growth, or liquidity. And when a theme *does* break, the correlations between its members snap toward 1 exactly when you need them apart — the failure mode covered in [the correlation-goes-to-one post](/blog/trading/risk-management/when-correlation-goes-to-one-the-diversification-that-vanishes-in-a-crisis) and, as an allocation topic, in [the cross-asset crisis-correlation post](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis). For our purposes the lesson is blunt: count your concentration by *theme*, not just by name, or you will think you're diversified right up to the day you find out you weren't.

![Before and after comparison of a diversified book of many small names versus a concentrated book with one dominant name and what a shock does to each](/imgs/blogs/concentration-and-position-limits-the-one-trade-that-can-end-you-4.png)

The figure above lays the two worlds side by side. On the left, the diversified book — twenty names at 5% each. One of them gaps −50%, which is only 5% × 50% = 2.5% of the book; the other nineteen absorb the rest, the dent is a −2.5%, and you survive to trade the next one. On the right, the concentrated book — one name at 40% and small change around it. The *same* name gaps the *same* −50%, but now it's 40% × 50% = 20% of the book; the hit blows past the 10% drawdown threshold, and that one surprise demands a +25% recovery just to get back to even. Identical shock, identical analysis. The left book treats it as weather; the right book treats it as an existential event. The difference is concentration, and nothing else.

## Sizing one name: Kelly's cap versus a hard concentration limit

So far we've talked about limits as ceilings you impose from above. But there's a second, deeper question: how big *should* a position be in the first place? There is a famous answer — the Kelly criterion — and it's worth seeing exactly where it agrees with a concentration limit and where it dangerously disagrees.

The **Kelly criterion** sizes a bet to maximize the long-run growth rate of your wealth, given an edge. For a continuous bet with an expected excess return (edge) of **μ** and a volatility of **σ**, the growth-optimal amount of capital to put into it is the Kelly leverage **L\* = μ / σ²**. (The full development is in [the Kelly post of this series](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge) and, for the sequential-betting derivation, in [the quant-interview Kelly post](/blog/trading/quantitative-finance/kelly-criterion-sequential-betting-quant-interviews).) The formula says something sensible: bet more when your edge is bigger, and bet less when the thing is more volatile. So far, so good.

The trouble is what Kelly says when you become *very confident* about a single name. As your estimated edge μ rises, L\* rises right along with it — Kelly will happily tell you to put 30%, 50%, 80% of your book into one name if you tell it the edge is large enough. And there's the trap: your estimate of μ is *noisy*. You don't actually know the edge; you have a guess, and guesses about single names are wrong all the time. Kelly, fed an overconfident edge estimate, will cheerfully recommend a position size that violates every survival instinct — because Kelly optimizes growth assuming your inputs are correct, and it has no separate term for "but what if you're wrong about this one and it gaps?"

![Kelly implied position cap versus a hard concentration limit showing two ways to size one name with the hard cap binding above a small edge](/imgs/blogs/concentration-and-position-limits-the-one-trade-that-can-end-you-5.png)

The figure above sizes one name two ways across a range of how good you think its edge is, with the name's volatility fixed at 35%. The amber line is full Kelly (μ / σ²); the dashed blue line is half-Kelly, the practitioner's standard haircut for the fact that you can't trust your own edge estimate. Both rise with confidence — and both cross the flat red line, which is a hard 10% single-name concentration limit. To the left, where the edge is small, everyone agrees: bet small. But past about a 1.2% estimated edge, full Kelly wants *more* than the 10% cap allows, and from there the hard limit binds and overrides Kelly entirely. The red-shaded region is where Kelly's confidence-driven sizing would over-concentrate you, and where the concentration limit simply says *no, not in one name, regardless of how good you think it is.*

This is the crucial reconciliation. Kelly is right about the *direction* — size up with edge, down with volatility — but Kelly has no notion of survival across the *specific* catastrophe of a single name blowing up, because it assumes your edge estimate is the truth and that returns are well-behaved. They aren't. A single name can fraud, can halt, can gap, can go to zero in ways the smooth μ-and-σ model never anticipated. The concentration limit is the survival override: it caps the *consequence* of being wrong about one name, independent of how confident the sizing model is. Practitioners run *both* — size by some fraction of Kelly to respect the edge, then *clip* the result at a hard concentration limit so that no single estimate, however bullish, can hand one name enough weight to end the book.

#### Worked example: sizing one name two ways

Take the **\$100,000 account** and one stock you're genuinely excited about. You estimate its edge at μ = 8% (expected annual excess return) and its volatility at σ = 35%.

**Kelly's answer.** Full Kelly leverage is L\* = μ / σ² = 0.08 / (0.35)² = 0.08 / 0.1225 = 0.653 — Kelly would put **65% of your account**, \$65,300, into this one name. Even half-Kelly, the cautious version, is 32.6%, or \$32,600. Both are wildly concentrated.

**The limit's answer.** Your hard single-name limit is 10% of the book — \$10,000, full stop.

So which wins? The limit, every time. You size at \$10,000, not \$32,600 or \$65,300. And here's why that's not just timid — it's correct: your "8% edge" is an *estimate*, and if you've overestimated it (which you have no way to rule out), Kelly's recommended size is catastrophic. Suppose the stock then gaps −50%. At Kelly's 65% size you'd lose \$65,300 × 0.50 = \$32,650, a **−32.7%** account drawdown from one name — a hole needing a +48.5% recovery. At the 10% limit you lose \$10,000 × 0.50 = \$5,000, a **−5%** drawdown you barely feel.

*Kelly tells you how much to bet if your edge estimate is exactly right; the concentration limit tells you how much you can afford to bet given that it might be badly wrong — and survival belongs to the second question.*

## Sizing to a maximum-loss budget: the rule that ties it all together

We now arrive at the most useful single technique in this post, the one that converts all the preceding intuition into a number you can compute for every position before you put it on. It flips the sizing question on its head. Instead of asking "how much capital should I allocate to this name?", you ask "how much am I willing to *lose* to this name?" — and then back out the size.

This is **sizing to a maximum-loss budget**, and the logic is almost embarrassingly simple. Decide, in advance and in calm weather, the most you are willing to lose to any single name — call it **R**, expressed as a fraction of the book (say, 2%). Then estimate that name's plausible worst-case adverse move — call it **g**, the gap or drawdown you'd suffer if it went badly wrong (say, 40%). For the loss to stay within budget, you need:

> position weight **w** × worst-case move **g** ≤ loss budget **R**

which rearranges to the rule:

> **w ≤ R / g.**

The largest position you may hold in a name is your loss budget divided by that name's worst-case move. That's the whole technique. And notice what it does automatically: it makes you hold *smaller* positions in *jumpier* names and *larger* positions in *calmer* ones, so that every position contributes the *same* worst-case loss to the book. You are no longer sizing by dollars or by gut — you're sizing by *risk*, with a guaranteed ceiling on what any one name can cost.

![Sizing a single name to a max loss budget showing the largest position weight that keeps worst case loss within one two and four percent of the book](/imgs/blogs/concentration-and-position-limits-the-one-trade-that-can-end-you-7.png)

The figure above plots the rule directly. The horizontal axis is the name's assumed worst-case move; the vertical axis is the largest position weight the budget allows. The three curves are three loss budgets — 1%, 2%, and 4% of the book. Read off the green (2%) curve: a calm name with a 20% worst-case gap earns up to a 10% position (because 2% / 20% = 10%), while a jumpy name with a 50% worst-case gap earns only a 4% position (2% / 50% = 4%). Same loss budget, very different sizes — because the jumpier name needs to be smaller to keep its worst-case contribution equal. The curves fall away steeply: as a name's tail gets fatter, the size it earns shrinks fast. This is the discipline that makes "equal risk, not equal dollars" concrete, and it's the practical engine behind [volatility-targeted sizing](/blog/trading/risk-management/volatility-targeting-sizing-by-risk-not-by-dollars), specialized to the single-name tail rather than everyday volatility.

#### Worked example: sizing three names to a 2% loss budget

Take the **\$10,000,000 book** and a 2% max-loss-per-name budget — so the most any single name may cost you is \$10,000,000 × 0.02 = \$200,000.

**A blue-chip with a 20% worst-case gap.** Max weight = R / g = 2% / 20% = 10%. So up to \$1,000,000 in this name. Check: a 20% adverse move on \$1,000,000 is exactly \$200,000 — right at budget.

**A volatile growth name with a 40% worst-case gap.** Max weight = 2% / 40% = 5%. So up to \$500,000. Check: a 40% move on \$500,000 is \$200,000 — at budget again.

**A speculative single-catalyst name with a 60% worst-case gap.** Max weight = 2% / 60% = 3.33%. So up to \$333,000. Check: a 60% move on \$333,000 is \$200,000 — at budget once more.

Three names, three very different sizes — \$1,000,000, \$500,000, \$333,000 — and yet every one of them, if its worst case hits, costs the book exactly \$200,000, a clean −2%. You've equalized the *risk* contribution, not the dollar size, and you've guaranteed that no single name can hand you more than a −2% book hit even in its worst plausible day. Stack ten such names and your maximum single-name damage is bounded at −2% each; you'd need many of them to go wrong at once to threaten the book, which is exactly the diversified situation a sane risk framework wants you in.

*The max-loss budget turns sizing from a guess into a calculation: you decide what you can afford to lose, you respect each name's real tail, and the position size that keeps you safe falls right out of the arithmetic.*

### Combining the budget with a hard cap

The max-loss budget and the hard concentration limit are not competitors — they work together, and the rule is to take whichever is *smaller* for any given name. The budget (w ≤ R / g) keeps the *expected worst-case loss* equal across names; the hard cap (w ≤ limit) prevents the budget from ever recommending a wildly large position in a name you've judged to be unusually calm. Suppose you have a name with a tiny 5% worst-case move and a 2% budget — the budget alone would allow w = 2% / 5% = 40% of the book in one name. That's absurd; "low volatility" is exactly the property that breaks worst, because calm names are where leverage and concentration pile up unnoticed before a regime change (the cautionary tale of [risk parity and the "safe" asset that gets levered](/blog/trading/risk-management/risk-parity-sizing-equal-risk-not-equal-money)). The hard cap is the backstop: *never above 10% in one name, even if the budget math says you could.* Final size = min(budget size, hard cap). Two independent checks, and the smaller one wins.

## Second-order effects: why concentration is worse than the multiplication suggests

The straight-line "weight times move" math is the first-order story, and it's already sobering. But concentration carries three second-order effects that make a large position *more* dangerous than its weight alone implies — and ignoring them is how people who "did the position-sizing math" still get surprised.

**Concentration interacts with liquidity, and the interaction is vicious.** A position's size relative to the *market it trades in* — not just relative to your book — determines whether you can exit it. A 10% position in a name where you are a small fraction of daily volume can be sold in an afternoon at roughly the screen price. The same 10% position in a name where you *are* a meaningful fraction of the daily volume cannot be sold without moving the price against yourself, so your realized loss exceeds your paper loss, sometimes badly. This is the trap that turned Amaranth's paper loss into a larger realized one: the size that maximized the gain on the way up was the same size that made exit impossible on the way down. Concentration and illiquidity compound — a big position in a thin market is doubly dangerous, because the very act of trying to reduce your concentration deepens your loss. The general principle is that you should size not only to your book but to your *exit*: never let a position grow so large relative to its market that you couldn't unwind it inside the window a crisis gives you.

**Concentration interacts with correlation, and correlation rises exactly when you need it not to.** We treated single-theme concentration as a static fact — ten correlated 8% names equal one 80% bet. But correlation isn't static; it's a regime, and in calm markets a cluster of names might genuinely move somewhat independently, making them *look* less concentrated than they are. Then a stress event arrives and the correlations within the cluster snap toward 1 — everything in the theme falls together — and the hidden concentration reveals itself at the worst possible moment. (Why this happens is the subject of the [correlation-goes-to-one post](/blog/trading/risk-management/when-correlation-goes-to-one-the-diversification-that-vanishes-in-a-crisis), and as a regime phenomenon, the [macro-correlations treatment of correlation as a regime](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant).) The practical implication is conservative: when estimating theme concentration, use *crisis* correlations, not calm-market ones. Assume the cluster moves as one, because in the event that matters, it will. A book that's diversified in calm correlations and concentrated in crisis correlations is concentrated, full stop — because crises are the only time the diversification was supposed to help.

**Concentration drifts, and it drifts toward your best ideas.** This is the quiet one. Your winners grow into your largest positions *by winning* — a 10% position that doubles while the rest of the book is flat is now an 18% position, and you never traded a share to get there. So concentration silently rebuilds itself out of your successes, and the position that has worked best — the one you're now most attached to and most reluctant to trim — is exactly the one that has crept past your limit. Left unmanaged, a disciplined entry process produces a concentrated book over time, purely through the mechanics of compounding winners.

#### Worked example: how a winner drifts past the limit

Take the **\$10,000,000 book** with a 10% single-name limit. You enter a position at exactly the cap: \$1,000,000, 10% of the book. Over six months it triples while the rest of the book is flat. The position is now worth \$3,000,000. The book is worth \$10,000,000 − \$1,000,000 + \$3,000,000 = \$12,000,000. The position's weight is \$3,000,000 / \$12,000,000 = **25%** — two and a half times your limit, without a single deliberate decision to concentrate. And now the worst-case math is dangerous again: a −50% gap on this beloved winner costs \$3,000,000 × 0.50 = \$1,500,000, a **−12.5%** book hit, well past your drawdown threshold. To get back inside the 10% limit you'd need to trim the position to 10% of \$12,000,000 = \$1,200,000 — selling \$1,800,000 of your best-performing name, which every instinct will resist.

*A position limit is not a thing you set once at entry; it's a thing you re-enforce after every move, because success itself is constantly trying to rebuild the concentration you started out avoiding.*

## Common misconceptions

**"I hold thirty positions, so I'm diversified."** The number of positions is nearly irrelevant; what matters is where the *weight* sits. Thirty names with one at 60% and twenty-nine at small change is a concentrated book with decoration. Check the *distribution* of weight, not the count of tickers. A book is as concentrated as its largest one or two positions, and a 60% position is a 60% bet no matter how many 1% names surround it.

**"A position limit costs me upside — I'd make more if I could let my winners run bigger."** Sometimes, yes — over a single trade, an uncapped winner beats a capped one. But survival is a multiplicative game played over many trades, and the cost of the *one* time your biggest, most-loved position gaps against you dwarfs the foregone upside on all the others. The limit gives up a little expected return on the body of the distribution to remove the tail that ends you, and by the recovery asymmetry, removing a −32% book event is worth far more than the trimmed upside it cost you to avoid it.

**"My worst position is only 8%, so I'm safe."** Not if you hold ten correlated 8% positions in the same theme. Single-name limits are necessary but not sufficient; ten names that move as one are a single 80% bet that passes every single-name check. You have to count concentration by *theme* and *factor*, not just by name, or hidden correlation will reassemble the concentration you thought you'd broken up.

**"A stop-loss protects me, so I can size bigger."** A stop-loss caps your loss only if the price actually trades at your stop on the way down. Across a gap — an overnight move, a halt-then-reopen, a limit-down session — the price never touches your stop, and you're filled far below it. Sizing is the only protection that works *before* the move; a stop is a hope that depends on liquidity being there when you need it, which in a crisis it won't be.

**"Concentration is fine if my conviction is high — that's how you make real money."** High conviction is exactly when concentration is most dangerous, because conviction is correlated across the market: the trades you're most sure of are usually the trades *everyone* is sure of, which makes them crowded, which makes their unwinds violent (the subject of [the crowded-trades exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game)). The graveyard is full of high-conviction concentrated bets. Conviction sizes *whether* you take a trade; it should never size *how much of your survival* you stake on it.

**"If I'm wrong about position size, I can fix it later."** You can fix it later only if "later" exists — and concentration's signature is that it removes "later." A 40% position that gaps −50% has already cost you 20% of the book before you can react; there is no later for the loss that already happened across the gap. The decision that mattered was made when you sized the position, in calm weather, days or weeks before the event. Sizing is a *pre-commitment*, not a thing you manage in the moment.

## How it shows up in real markets

The case for concentration discipline isn't theoretical. The most spectacular blow-ups in modern finance are, almost without exception, concentration stories — one bet, too big, that the rest of the book could not absorb. Two examples make the point with brutal clarity.

### Amaranth Advisors, September 2006 — one trade, \$6.6 billion

Amaranth was a multi-strategy hedge fund that, on paper, ran many books. In practice, by 2006 its fortunes had become dominated by one trader's concentrated, levered bet on **natural-gas calendar spreads** — wagers on the price difference between gas delivered in different months (specifically March versus April contracts). The position was enormous relative to the market it traded, and it was deeply concentrated in a single, illiquid theme.

When the spread moved against the fund in September 2006, the size that had made the position so profitable on the way up made it impossible to exit on the way down — every attempt to sell pushed the price further against the remaining position. Amaranth lost roughly **\$6.6 billion**, most of it in a single week, and collapsed (CFTC/Senate PSI report, 2007). The fund wasn't killed by being wrong across a diversified book of strategies. It was killed by *one* concentrated bet in an illiquid market that grew large enough that the rest of the firm couldn't absorb its loss — and large enough that it couldn't be unwound without crushing its own price. Concentration plus illiquidity is a particularly lethal combination, because the size that maximizes your gain is the same size that traps you when you need out.

#### Worked example: the arithmetic of an Amaranth-style concentration

Put the Amaranth lesson in our framework with the **\$10,000,000 book** (Amaranth was a thousand times bigger, but the arithmetic is scale-free). Suppose you let one illiquid theme grow to 50% of the book — \$5,000,000 — because it had been working beautifully and you kept adding. Then the theme moves 40% against you over a week. The loss is \$5,000,000 × 0.40 = \$2,000,000, a **−20%** book drawdown, which by the recovery math needs a +25% gain to undo. Now add the illiquidity tax: as you try to exit the \$5,000,000 position, your own selling moves the price, so you realize *more* than the 40% paper loss — say 50% by the time you're out. That's \$2,500,000 lost, a **−25%** book drawdown, needing +33% to recover. Had that same theme been capped at 10% of the book (\$1,000,000), the identical 50%-with-slippage move would have cost \$500,000 — a survivable **−5%**. *The limit doesn't just cap your loss; it keeps the position small enough that you can still get out, because a position you can't exit is a loss that keeps growing.*

### Archegos Capital Management, March 2021 — one concentrated book, levered, gone overnight

Archegos was a family office that built enormous, concentrated single-stock positions through **total-return swaps** — derivative contracts with prime brokers that gave it economic exposure to stocks without owning them outright, and crucially, with **leverage of around 5x or more**. Because the exposure was held through swaps spread across several banks, each individual prime broker could see only its own slice; *no single counterparty knew the total size of the bet* (Credit Suisse / Paul Weiss report, 2021). The concentration was real, it was levered, and it was hidden — even from the firms financing it.

When a few of Archegos's concentrated single-stock holdings fell in late March 2021, the leverage did its work. A drawdown in the underlying names that an unlevered holder would have shrugged off was multiplied roughly fivefold against the fund's equity, and that equity went through zero. Archegos was liquidated; its prime brokers, scrambling to dump the concentrated positions all at once, took aggregate losses of **more than \$10 billion**, with **Credit Suisse alone losing about \$5.5 billion** (bank disclosures). The fund's entire capital was erased in a matter of days. This is concentration and leverage in their purest, most lethal combination: a few names, financed at 5x, with no one seeing the whole picture.

![How leverage turns a single name drawdown into an overnight wipeout showing a concentrated five times levered book equity curve collapsing through zero](/imgs/blogs/concentration-and-position-limits-the-one-trade-that-can-end-you-6.png)

The figure above illustrates the mechanism — a seeded illustration, not a price claim, with the leverage and loss figures cited from the Archegos record. The blue line is an unlevered book that simply tracks a concentrated basket of names: it grinds up, then gives back a chunk in the bad stretch, and ends bruised but alive. The red line is the *same basket* held at 5x leverage. On the way up, leverage amplifies the gains into a euphoric run. Then the basket draws down — and at 5x, a basket move that an unlevered holder would survive multiplies straight through the fund's equity, which hits zero and triggers liquidation. *Gone overnight*, in the figure's words, exactly as Archegos was. The unlevered version of the identical bet survived the identical drawdown; leverage is what converted a survivable single-theme loss into a terminal one. (For the leverage arithmetic in full, see [leverage and the arithmetic of ruin](/blog/trading/risk-management/leverage-and-the-arithmetic-of-ruin); for the firm-level view of how funds like these die, [the hedge-fund failure taxonomy](/blog/trading/hedge-funds/how-hedge-funds-die-the-failure-taxonomy).)

The shared lesson of Amaranth and Archegos is the thesis of this post stated in blood: it was not a diversified book of bad bets that killed them, and it was not a failure of analysis in the usual sense. It was *one* concentrated position — gas spreads in one case, swap-financed single stocks in the other — sized large enough, and in Archegos's case levered enough, that the rest of the portfolio could not absorb its loss. Every other risk control was beside the point once the concentration was in place. The one trade was the whole story.

## The concentration playbook

Survival from concentration risk is not a feeling or a slogan; it's a small set of mechanical rules you set in calm weather and enforce without exception when the temptation to break them is strongest. Here is the concrete playbook.

**1. Set a hard single-name limit, and never override it.** Pick a maximum weight for any one position — for a concentrated book this might be 10%, for a more diversified mandate 5% or lower. Enforce it mechanically: when a winner drifts above the cap, you trim it back, no matter how much you love it. The limit's entire value is that it binds when you least want it to. A 10% single-name limit on a \$10,000,000 book means the worst a single-name catastrophe (−80%) can ever cost you is \$800,000, an −8% book hit — survivable by construction.

**2. Set a single-theme limit above the single-name limit, and count themes honestly.** No cluster of correlated positions — same sector, same factor, same macro driver, same liquidity source — above some larger fraction (say 20–25%). Identify the themes deliberately, because they hide: ten names under the single-name limit can be one bet over the theme limit. When in doubt about whether two positions share a theme, assume they do; correlations rise in stress, never fall.

**3. Size every name to a maximum-loss budget.** Before you put a position on, decide R, the most you'll lose to this name (say 2% of the book), estimate g, its plausible worst-case move, and set the size to w = R / g. This makes jumpy names small and calm names larger, equalizing each name's worst-case contribution. Then take the *smaller* of this budget size and your hard cap — the cap backstops the budget when a "calm" name's low g would otherwise justify a huge position.

**4. Make the worst-case move (g) honest — and fat.** The whole budget rule depends on g being a *realistic* tail, not an average day. A single name can gap 40%, 50%, even to zero on fraud, failed trials, halts, or sector repricings. Use a worst case that respects the asset's real tail, and remember that stops do not protect you across a gap — only the size you chose beforehand does.

**5. Watch leverage like a concentration multiplier.** Leverage doesn't just amplify returns; it amplifies concentration. A 10% position at 5x leverage behaves like a 50% position for loss purposes. Apply your limits to *leverage-adjusted* exposure, not just to the cash you put down — Archegos's 5x is exactly what turned a survivable single-theme drawdown into a wipeout.

**6. Re-check concentration after every big move, not just at entry.** Concentration drifts. Your best ideas grow into your biggest positions precisely by working — a 10% position that doubles while the rest is flat becomes an 18% position without you trading a share. Rebalance back toward your limits on a schedule, so that success doesn't quietly rebuild the concentration you started out avoiding.

**7. The one-question test.** For every position in your book, you should be able to answer instantly: *if this one name goes against me as badly as it plausibly can, how much of my whole portfolio does it cost?* If the answer is more than a single-digit percent for any one name, you are concentrated, and the playbook above is how you fix it before the market fixes it for you.

The deepest point is the one the whole series keeps returning to: you do not get to choose when the surprise comes or how large it is — but you do, entirely, get to choose how much of your survival rides on any single bet. Concentration is the discipline of making that choice deliberately, in advance, while you can still think clearly. The trader who survives is rarely the one who was right most often. It's the one who never let a single trade get big enough to end them — and who therefore was still in the game when the next opportunity came.

### Further reading

- [The asymmetry of losses: why a 50% loss needs a 100% gain](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) — why the large drawdowns concentration produces are so hard to recover from.
- [The gambler's ruin and bet sizing: the math of staying solvent](/blog/trading/risk-management/the-gamblers-ruin-and-bet-sizing-the-math-of-staying-solvent) — sizing every bet so the probability of touching zero stays below a budget you set.
- [Marginal and component VaR: where the risk actually lives](/blog/trading/risk-management/marginal-and-component-var-where-the-risk-actually-lives) — decomposing a book to find which position is quietly carrying the most risk.
- [How hedge funds die: the failure taxonomy](/blog/trading/hedge-funds/how-hedge-funds-die-the-failure-taxonomy) — the firm-level view of the concentration-and-leverage blow-ups from the GP seat.
- [Position sizing and risk of ruin in options trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) — the same survival discipline applied to the leverage and convexity of options.

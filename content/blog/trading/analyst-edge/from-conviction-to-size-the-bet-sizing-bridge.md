---
title: "From Conviction to Size: The Bet-Sizing Bridge"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Turn a view into a position: how risk-per-trade, the invalidation distance, fractional Kelly, volatility-targeting, and a portfolio-heat cap convert conviction into an exact number of shares at risk."
tags: ["analysis", "market-view", "position-sizing", "risk-per-trade", "kelly-criterion", "fractional-kelly", "volatility-targeting", "portfolio-heat", "invalidation", "conviction", "trading-process"]
category: "trading"
subcategory: "The Analyst's Edge"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A view is worthless until it is a position size. Sizing is the bridge from "I believe X" to "I have \$N at risk," and it is governed by edge and by risk, never by excitement.
>
> - **Risk-per-trade is the unit.** Decide what fraction of equity you will lose if you are wrong (often ~1%), and let the invalidation distance — not the share price, not your enthusiasm — set the share count.
> - **The Kelly criterion tells you the growth-maximizing fraction; you bet a fraction of it.** Full Kelly is brutally volatile and assumes you know your edge exactly. Half Kelly keeps roughly three-quarters of the growth at a fraction of the drawdown.
> - **Size by risk, not by dollars.** Volatility-targeting shrinks the high-vol position so every name contributes the same risk; a portfolio-heat cap stops the sum of open risks from quietly compounding into ruin.
> - The one rule: **a view you cannot turn into an exact number of shares at a defined risk is not a position yet** — it is a feeling.

Two analysts walk out of the same earnings call with the same view. The stock is at \$50, it just reported a clean beat with raised guidance, and both of them are convinced it runs to \$70 over the next quarter. Same information, same read, same conviction. One of them buys 250 shares — \$12,500 of stock, with a stop at \$46 that would cost him \$1,000 if he is wrong, which is exactly 1% of his \$100,000 account. The other one, *certain* this is the trade of the year, buys with both hands: he puts \$80,000 of his \$100,000 into it on margin, no stop, because why would you stop out of a trade you *know* is right?

For six weeks they look like geniuses together. The stock grinds to \$58, then \$63. The disciplined one is up \$3,250 and sleeping fine. The all-in one is up more than \$20,000 on paper and telling everyone who will listen. Then a competitor pre-announces, the whole sector gaps down 9% one morning, and the stock that was "going to \$70" is suddenly trading \$44. The disciplined analyst's stop triggered at \$46; he is out, down \$1,000, annoyed, and still has \$99,000 to trade tomorrow. The all-in analyst, leveraged and stopless, watches \$80,000 become \$60,000 in a single session, gets a margin call he cannot meet, and is forced to liquidate at the lows. The view was *right* about the company. It did not matter. He is done, and she is fine, and the only thing that differed between them was size.

This is the part of the craft that almost nobody teaches and almost everybody gets wrong. The entire rest of this series is about *forming* a view — reading the lenses, finding the variant perception, knowing what is priced in, quantifying it as expected value. This post is about what happens *after* you have the view: the bridge that turns conviction into a position. Because a view does not move your account. A position does. And the size of that position — not the brilliance of the thesis — is what decides whether a good view compounds your capital or a single bad run ends your career.

![From conviction to size pipeline through risk unit invalidation conviction volatility and heat cap](/imgs/blogs/from-conviction-to-size-the-bet-sizing-bridge-1.png)

The figure above is the whole post in one picture, and we will earn the right to trust each box. A view — your edge, expressed as expected value — enters on the left. It does not come out the other side as "a position" by magic. It passes through five gates: a fixed risk-per-trade unit, the invalidation distance that converts that risk into a share count, a conviction-tier multiplier, a volatility adjustment so every name contributes equal risk, and a portfolio-heat check that caps the sum of your open risks. What survives all five gates is the number that actually hits your account: dollars at risk. Skip any gate and you are the second analyst.

## Foundations: what position sizing actually is

Start with the words, because the whole discipline lives in getting two definitions straight.

**Position sizing** is the decision of *how much* of an asset to hold once you have decided to hold it at all. It is a separate question from *whether* to take the trade. Forming the view answers "do I buy this?" Sizing answers "how much?" — and the second question, it turns out, dominates your results far more than the first. A mediocre view sized well outlives a brilliant view sized badly, every time, because sizing controls the one thing that ends accounts: the magnitude of your losses.

**Risk-per-trade** is the amount of money you will lose if a specific trade goes against you and you exit at your predefined invalidation point. It is *not* the size of the position. If you buy \$12,500 of stock with a stop \$4 below your \$50 entry, your position is \$12,500 but your risk-per-trade is only \$1,000 — that is what you actually lose if the stop triggers. The distinction is the single most important idea in this post. Most people who blow up confuse "how much I bought" with "how much I can lose," and those two numbers are wildly different the moment a stop enters the picture.

That gives us the cleanest possible definition of sizing-by-risk versus sizing-by-dollars:

- **Sizing by dollars** means you decide the position's notional value — "I'll put \$10,000 into each idea" — and let whatever happens to your loss happen. The trouble is that an identical \$10,000 position can risk \$400 (if your invalidation is tight) or \$3,000 (if it is wide), so a "constant" dollar size produces wildly *inconstant* risk.
- **Sizing by risk** means you decide the loss first — "I will risk \$1,000 if I am wrong" — and then *solve for* the number of shares that produces exactly that loss given where your invalidation sits. The position's notional value falls out as a byproduct; the loss is what you control.

Professionals size by risk. Amateurs size by dollars (or worse, by enthusiasm), and the difference is most of the gap between the two analysts in the opening.

### Why sizing matters more than entry

Here is the uncomfortable truth that the opening dramatizes: your entry is a rounding error next to your size. A great entry on a position sized too large is a margin call waiting for a catalyst. A mediocre entry on a position sized correctly is a survivable mistake. The reason is the brutal arithmetic of drawdowns — losses and gains are not symmetric. Lose 50% of your account and you do not need a 50% gain to get back; you need a 100% gain, because you are now compounding from a smaller base. Lose 80% (the all-in analyst) and you need a 400% gain just to break even. Sizing is the only lever that controls how deep your worst run goes, and the depth of your worst run sets the height you must climb back. (This asymmetry is foundational; the deep version is in [the asymmetry of losses](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain).)

State the consequence in one line, the line every sizing decision is ultimately defending against: **risk of ruin** is the probability that a string of losses takes your account below the level where you can keep trading. Size small enough relative to your edge, and that probability is effectively zero no matter how unlucky you get; size too large, and a perfectly ordinary losing streak — the kind that happens to *everyone* — wipes you out. Sizing is the machinery that keeps risk of ruin near zero while still putting enough capital behind your edge to compound it. The full mathematics of survival lives in [risk management as the only free lunch](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine); here we build the bridge that produces a number.

Put the asymmetry into dollars so it stops being abstract. Start with \$100,000. A 10% drawdown leaves \$90,000, and you need an 11.1% gain to get back to even — annoying, recoverable. A 25% drawdown leaves \$75,000, requiring a 33.3% gain to recover — the climb is now meaningfully steeper than the fall. A 50% drawdown leaves \$50,000 and demands a 100% gain — you have to *double* your money just to undo a single bad period. And an 80% drawdown, the all-in analyst's fate, leaves \$20,000 and requires a 400% gain to get whole, which for most traders simply never happens; the account never recovers. Notice the shape: the recovery requirement grows *faster* than the loss. Each additional percent of drawdown costs you disproportionately more on the way back, which is exactly why the depth of your worst run — the thing sizing controls — matters far more than how often you win. A trader who never lets a drawdown exceed 15% is playing a fundamentally different, vastly more survivable game than one who occasionally takes a 50% hit, even if their average trade is identical.

### Where sizing sits in the analyst's process

It helps to locate sizing in the full arc of forming and acting on a view, because the bridge metaphor is precise about what feeds into it. Upstream of sizing is *everything this series has built*: reading the six lenses, finding the variant perception, mapping what is priced in, structuring the thesis, and quantifying it as an expected-value table with scenario probabilities and dollar payoffs. That entire process produces two outputs that sizing consumes — an **edge** (how favorable the bet is, which sets the Kelly ceiling and the conviction tier) and an **invalidation** (where the thesis is wrong, which sets the stop distance and therefore the share count). Sizing takes those two numbers and produces the third number, the only one the market reacts to: dollars at risk. Downstream of sizing is execution, monitoring, and updating — but none of that can begin until the position exists, and the position does not exist until it has a size. Sizing is the hinge between thinking and acting. Skip it, or do it by feel, and all the upstream analytical rigor is wasted, because a brilliantly reasoned view sized by adrenaline is, in its consequences, indistinguishable from a coin flip sized by adrenaline.

## Risk-per-trade: the unit everything is measured in

Every sizing system needs a unit, and the right unit is a fixed fraction of your current equity that you are willing to lose on any single trade. Call it your **R** — your risk-per-trade. A common professional choice is **1% of equity**. On a \$100,000 account, 1% is \$1,000. That \$1,000 is the most you will lose if any single trade hits its invalidation. Everything downstream is denominated in this unit.

Why a *fraction* of equity rather than a fixed dollar amount? Because it makes your sizing self-correcting. When you are winning and your equity grows, your 1% grows with it, so you press your edge harder with house money. When you are losing and your equity shrinks, your 1% shrinks too, so you automatically de-risk into a drawdown — the opposite of the doubling-down instinct that kills people. A \$100,000 account risks \$1,000 per trade; after a rough patch drops it to \$80,000, the same 1% rule risks only \$800. The percentage rule throttles you down exactly when you most need throttling.

How big should R be? That is a function of your edge and your stomach, and it is genuinely small — far smaller than beginners guess. At 1% per trade, even ten consecutive losses (an unpleasant but routine streak) costs you roughly 10% of your account, fully recoverable. At 5% per trade, those same ten losses cost you about 40%, and you are now in the territory where the recovery arithmetic gets vicious. At 10% per trade, a streak of ten losses is functionally fatal. Most retail traders die not from being wrong about direction but from sizing R at 5–10% and meeting an ordinary losing streak. Keep R at 0.5–2% and the streaks become survivable noise.

#### Worked example: turning a 1% risk budget into an exact share count

You have a \$100,000 account and you size at 1% per trade, so your risk-per-trade is **R = \$1,000**. You want to buy a stock trading at **\$50.00**. Your thesis breaks — your **invalidation** — if the stock closes below \$46.00, because that violates the base it has been building (defining invalidation upfront is its own discipline, covered in [what would change my mind](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront)). So your stop is \$4.00 below your entry. The invalidation distance is **\$4.00 per share**.

Now solve for shares. The position will lose (invalidation distance) × (shares) if it stops out, and you want that loss to equal exactly R:

$$\text{shares} = \frac{R}{\text{invalidation distance}} = \frac{\$1{,}000}{\$4.00} = 250 \text{ shares}$$

So you buy **250 shares**, a position worth 250 × \$50 = **\$12,500**. If the stock falls to \$46 and you exit, you lose 250 × \$4 = **\$1,000** — exactly your 1% R. Notice what you did *not* do: you did not start with "\$12,500 feels right" or "I'll buy 1,000 shares because I'm confident." You started with the loss you would accept (\$1,000) and the distance to invalidation (\$4), and the share count fell out as arithmetic. The size is a *consequence* of your risk budget and your stop, not a guess. That is the entire move — and once you internalize it, "how many shares?" stops being a feeling and becomes a division problem.

![Risk-per-trade and invalidation distance map to an exact share count on a fixed budget](/imgs/blogs/from-conviction-to-size-the-bet-sizing-bridge-2.png)

The table makes the dependency vivid. Hold the risk budget fixed at \$1,000 and the entry fixed at \$50, and watch what the invalidation distance alone does to the share count. A tight \$2 stop buys you 500 shares (\$25,000 notional); a \$4 stop buys 250 shares (\$12,500); a wide \$8 stop buys only 125 shares (\$6,250). Same risk every time — \$1,000 — but the position's *size* swings by 4×, driven entirely by where your invalidation sits. This is why "I always buy \$10,000 worth" is incoherent as a risk practice: it means your actual risk silently quadruples whenever your stop happens to be wide. The invalidation distance is the dial; the dollar risk is what you hold constant.

## The Kelly criterion: how much edge justifies

Risk-per-trade gives you a *unit*, but it does not tell you how many units to deploy or how that should scale with the strength of your edge. A genuine 60/40 edge should be backed harder than a marginal 52/48 edge. The framework that answers "how much should I bet given my edge" is the **Kelly criterion**, and it is worth understanding precisely — not because you will bet full Kelly (you should not), but because it tells you the *direction* and rough magnitude of how size should respond to edge.

Kelly answers a specific question: what fraction of your bankroll should you bet on a repeated favorable wager to maximize the long-run growth rate of your capital? Not to maximize expected profit on the next bet (that would say "bet everything," which guarantees eventual ruin), but to maximize the *compounded* growth over many bets. The answer balances two forces: bet too little and you leave growth on the table; bet too much and the volatility of your bankroll drags down its compounded return — and past a point, sends it toward zero.

For the cleanest case — a bet that either wins or loses, where you win `b` dollars per dollar risked with probability `p` and lose your dollar with probability `q = 1 − p` — the Kelly fraction is:

$$f^{*} = \frac{p \cdot b - q}{b} = \frac{\text{edge}}{\text{odds}}$$

Read it as *edge divided by odds*. The numerator `p·b − q` is your edge: the expected profit per dollar risked. The denominator `b` is the payoff odds. Divide the edge by the odds and you get the growth-maximizing fraction of your bankroll to put at risk. (The full derivation and the continuous version live in [the Kelly criterion](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge); we want the intuition and the number here.)

The formula has properties worth feeling. No edge (`p·b = q`) means `f* = 0` — Kelly says bet nothing on a fair game, which is correct. A bigger edge means a bigger fraction. And critically, *worse odds* (a bigger potential loss per dollar) means a *smaller* fraction even for the same win probability, because the downside is more punishing. Kelly is the mathematically honest statement of "size by edge and by risk."

There is a second, equally useful form of Kelly for continuous outcomes — for a position whose return is not a simple win-or-lose but a spread, with an expected excess return `μ` and a variance `σ²`, the growth-optimal fraction is approximately `f* = μ / σ²`. Read this one carefully, because it is the mathematical root of everything in the volatility-targeting section below: the optimal fraction is your expected edge *divided by your variance.* Two positions with the same expected edge but different volatility get different sizes — the more volatile one (bigger `σ²`) gets a *smaller* fraction, because its variance eats into compounded growth faster. That is vol-targeting, derived from first principles: size inversely to volatility because volatility is the denominator of the growth-optimal fraction. The discrete `edge ÷ odds` and the continuous `μ ÷ σ²` are the same idea wearing two outfits — both say size up for edge, size down for risk.

One more property matters for how Kelly connects to the risk-per-trade unit we started with. Kelly is expressed as a fraction of *bankroll*, while R is expressed as a fraction of bankroll you are willing to *lose*. They are different quantities — Kelly's fraction is the amount *deployed* in a binary bet where you can lose all of it, whereas R is the loss you cap via your invalidation. But they answer the same governing question from two directions: Kelly says "given your edge, here is the most you should rationally commit," and R says "regardless of edge, here is the most I will lose on any one trade." A disciplined trader uses Kelly to understand the *ceiling* their edge justifies and then sets R *below* even fractional Kelly, because the edge estimate is uncertain and survival is non-negotiable. When R and fractional Kelly disagree, you take the smaller. They are two governors on the same engine, and you obey whichever one binds first.

### Why full Kelly is too aggressive

Here is the catch that every practitioner learns, often the hard way: **full Kelly is far too aggressive to trade.** The fraction that maximizes long-run growth in theory produces gut-churning volatility in practice, and it rests on an assumption you can never satisfy — that you know your edge *exactly*. You don't. You estimate `p` and `b` from a finite, noisy history, and if you overestimate your edge even slightly, full Kelly tips you past the growth-maximizing point into the region where you are *destroying* capital while feeling aggressive.

Two facts make full Kelly impractical. First, the growth curve around the Kelly fraction is *flat on top* — betting somewhat less than full Kelly costs you very little growth. Second, the drawdown you suffer scales roughly linearly with how much you bet — bet half as much, suffer roughly half the drawdown. Put those together and you get the central practical result of the whole field: you can give up a little growth to buy a lot of calm.

![Kelly fraction versus long-run growth and drawdown showing half Kelly keeps most growth](/imgs/blogs/from-conviction-to-size-the-bet-sizing-bridge-3.png)

The chart shows it. The green curve is long-run growth as a function of how much you bet, expressed as a multiple of full Kelly. It peaks at exactly full Kelly (1.0 on the x-axis) and falls away on both sides — bet too little (left) and you grow slowly; bet *more* than full Kelly (right) and growth collapses, hitting zero at twice Kelly. The red line is the characteristic drawdown, climbing steadily with bet size. Now find half Kelly (0.5 on the x-axis): the green curve is still at about **75% of the maximum growth**, but the red drawdown line is at *half* the full-Kelly level. You surrender a quarter of your growth and you halve your worst-case pain. For a number you estimated from noisy data and have to live through emotionally, that is the trade every sane practitioner makes.

This is **fractional Kelly**: bet a fixed fraction — typically a half or a quarter — of the full Kelly amount. It is the single most important adjustment between the textbook and the trading desk. The deep treatment, including optimal-f and the case for quarter Kelly, is in [fractional Kelly and optimal-f](/blog/trading/risk-management/fractional-kelly-and-optimal-f-betting-less-to-sleep-at-night).

#### Worked example: full Kelly versus half Kelly on a measured edge

You have built a setup you have traded 200 times. It wins **55% of the time** (`p = 0.55`, so `q = 0.45`), and your winners are about **1.5×** the size of your losers — when you win you make 1.5 units, when you lose you lose 1 unit (`b = 1.5`). Plug into Kelly:

$$f^{*} = \frac{p \cdot b - q}{b} = \frac{(0.55)(1.5) - 0.45}{1.5} = \frac{0.825 - 0.45}{1.5} = \frac{0.375}{1.5} = 0.25$$

Full Kelly says bet **25% of your bankroll** on every instance of this setup. On a \$100,000 account, that is \$25,000 *at risk* per trade — not notional, at risk. Sit with that. A 25% loss of your account on a single trade if you are wrong, and you will be wrong 45% of the time. A run of three or four losers — entirely normal at a 55% win rate — would carve 60–70% out of your account. The math says full Kelly maximizes growth, and it is correct, but no human being can trade through those swings without panicking and abandoning the system at the worst moment.

So you bet **half Kelly: 12.5%** — still aggressive at \$12,500 at risk, and most traders would quarter it further. But watch the relationship to where we started: this setup's *full*-Kelly fraction (25%) is the ceiling that your conviction can justify, and your risk-per-trade unit (1% = \$1,000) sits far below even quarter Kelly. The Kelly fraction is not the size you trade; it is the *upper bound* your edge permits, and your actual R lives well underneath it precisely because your edge estimate is uncertain. Kelly tells you which direction to lean; fractional Kelly and a fixed R keep you alive long enough to collect on it.

## Volatility-targeting: equal risk, not equal dollars

The invalidation method sizes a single trade off its stop distance. But how do you size *across* positions that have nothing to do with stops — a sleepy utility versus a volatile biotech, held without a hard stop? Equal dollar amounts will not do it, because a \$20,000 position in a stock that moves 1% a day and a \$20,000 position in a stock that moves 5% a day carry completely different risk. The second one is contributing five times the daily dollar swing of the first. Your "diversified" book is secretly a concentrated bet on the volatile name.

The fix is **volatility-targeting**: size each position *inversely* to its volatility so that every position contributes the same dollar risk to the portfolio. Decide the daily dollar risk you want from a position — say \$200 — and then solve for the notional that produces it:

$$\text{notional} = \frac{\text{target daily dollar risk}}{\text{daily volatility (\%)}}$$

A low-volatility name needs a *large* notional to reach \$200 of daily risk; a high-volatility name needs a *small* one. The riskier the asset, the smaller the position — which is exactly backwards from how excitement sizes (excitement piles into the name that moves the most). Volatility-targeting is the portfolio-level expression of "size by risk, not by dollars," and it is what lets you compare a bond position and a tech position on the same axis. The mechanism in depth is in [volatility-targeting](/blog/trading/risk-management/volatility-targeting-sizing-by-risk-not-by-dollars).

#### Worked example: vol-targeting a low-vol and a high-vol name to equal risk

You want two positions, and you want each to contribute about **\$200 of daily risk** — roughly a \$200 typical day's move on the position. Name A is a low-volatility name with a **1% daily volatility**. Name B is a high-flyer with a **5% daily volatility**. The naive approach gives each \$20,000:

- Name A at \$20,000: daily dollar risk = 1% × \$20,000 = **\$200.** Fine.
- Name B at \$20,000: daily dollar risk = 5% × \$20,000 = **\$1,000.** Five times too much.

So a "balanced" \$20,000-and-\$20,000 book is not balanced at all — Name B dominates the portfolio's daily P&L by 5 to 1. Vol-target instead:

$$\text{notional}_A = \frac{\$200}{1\%} = \$20{,}000, \qquad \text{notional}_B = \frac{\$200}{5\%} = \$4{,}000$$

You buy \$20,000 of the calm name and only **\$4,000** of the wild one. Now each contributes \$200 of daily risk, and your book is genuinely balanced — a 1% adverse day in either name costs you the same \$200. The high-vol name got a *fifth* of the dollars precisely because it carries five times the per-dollar risk.

![Volatility-targeting gives the high-vol name a smaller position so each contributes equal risk](/imgs/blogs/from-conviction-to-size-the-bet-sizing-bridge-4.png)

The bars make the asymmetry impossible to miss. Under equal-dollar sizing (red), the high-vol name's daily risk towers at \$1,000 against the low-vol name's \$200 — the red bars are the *same notional* producing wildly different risk. Under vol-targeting (green), both sit exactly on the \$200 target line because the high-vol name's notional was cut to \$4,000. The visual claim is the whole idea: equalize *risk*, and the dollar sizes will be unequal — the riskier asset gets the smaller position, every time.

## Portfolio heat: the cap on the sum of your risks

You can size every individual trade perfectly at 1% R and still blow up — if you have thirty of them on at once. Each position is a small, controlled risk in isolation, but risk *adds up* across open positions, and the sum is what actually exposes your account. The sum of all your open risks has a name: **portfolio heat**. It is the total amount you would lose if every open position simultaneously hit its invalidation. Heat is the portfolio-level version of risk-per-trade, and it needs its own cap.

Why does it need a cap separate from per-trade R? Because correlated positions tend to hit their stops *together*. The thirty 1%-R positions feel like thirty independent coin flips, but in a real market drawdown they are not independent — they are thirty bets on "risk assets go up," and when the sector or the whole market rolls over, they all stop out in the same week. Thirty times 1% is 30% of your account gone in a correlated flush. Portfolio heat is the discipline that says: *I do not care how good each individual trade looks; the sum of what I can lose right now is capped.*

A common cap is **6% total heat** — you will hold open risk summing to no more than 6% of equity at any time. At 1% R per trade, that is six full-size positions. Want a seventh? Something has to give: you close or trim an existing position to make room, or you size the new one smaller so the sum stays under 6%, or you pass. The cap forces the portfolio to compete for a fixed risk budget, which is exactly the discipline that keeps a string of correlated losers from compounding into a catastrophe.

There is a subtlety in how heat *evolves* as positions work. When a position moves in your favor and you raise its stop to lock in profit — trailing the invalidation up behind the price — its risk-if-stopped *shrinks*, and may even go negative once the stop is above your entry (a stop-out now banks a gain, not a loss). That freed-up heat is real risk budget you can redeploy. A position that started at 1% R and has run far enough that its trailed stop guarantees a profit contributes *zero* heat to the cap — it can no longer lose you money — so it does not count against your six slots. This is the mechanism that lets a trending book hold more than six names: winners that have de-risked themselves stop consuming the budget, freeing room for new ideas without raising your true exposure. Heat is not a static count of positions; it is the live sum of what you can still lose, and it breathes as your stops move.

![Portfolio heat builds as cumulative open risk and hits the six percent cap at the sixth position](/imgs/blogs/from-conviction-to-size-the-bet-sizing-bridge-5.png)

The chart shows the heat building position by position. Each open trade adds its 1% to the cumulative open risk: position one takes you to 1%, position two to 2%, and so on, the bars cascading rightward like a waterfall. By position six you are exactly at the 6% cap (the dashed red line). The seventh position — drawn in red, breaching the line — is the one the cap stops: you cannot add it at full size without taking your total open risk above the budget you swore to defend. Either it does not go on, or it goes on small, or something else comes off. The cap is not a suggestion; it is the wall that a correlated drawdown cannot push you through.

#### Worked example: portfolio heat caps the sixth position

It is a good week and you keep finding setups. Your \$100,000 account sizes at 1% R, and you cap portfolio heat at 6%. You put on five trades, each risking \$1,000 if it stops out:

| Position | Risk if stopped | Cumulative heat |
|---|---|---|
| 1 | \$1,000 | \$1,000 (1%) |
| 2 | \$1,000 | \$2,000 (2%) |
| 3 | \$1,000 | \$3,000 (3%) |
| 4 | \$1,000 | \$4,000 (4%) |
| 5 | \$1,000 | \$5,000 (5%) |

Now a sixth idea appears, and it is a good one. At full 1% R it would add \$1,000, taking heat to \$6,000 — exactly the 6% cap. You can take it at full size. But suppose a *seventh* idea shows up the same day. Full size would push heat to \$7,000, which is 7% — over your cap. The rule binds: you cannot add the seventh at full size. Your options are concrete. You can trim an existing position to free up risk budget — close half of position three, recovering \$500 of heat, and size the seventh at \$500. You can size the seventh down to whatever keeps total heat at \$6,000 (if you are already at the cap, that is zero — pass). Or you can decide the seventh is better than one you hold, close the weaker one entirely, and put the new one in its place. What you may *not* do is wave the cap away because you like all seven. The single worst day of your trading life is the one where all six (or seven) of those correlated longs gap down together — and the cap is the only thing standing between a bad day and a ruinous one.

## Scaling size across conviction tiers

We have a unit (R), a share-count rule (invalidation distance), a ceiling on aggressiveness (fractional Kelly), a cross-position equalizer (vol-targeting), and a portfolio cap (heat). One piece remains: not every view deserves the same size. A trade you would stake your reputation on and a trade you are taking mostly to stay engaged are not the same bet, and your sizing should say so — *through a disciplined multiplier, not through your mood in the moment.*

The clean way to do this is **conviction tiers**: predefine two or three sizes and assign each trade to a tier based on how strong the edge is, then never deviate. For example:

- **Full size (1.0× R):** a high-conviction trade — your thesis is clear, the catalyst is identified, the invalidation is tight and well-defined, and the expected value is strongly positive. Risk the full 1%.
- **Half size (0.5× R):** a medium-conviction trade — the thesis is reasonable but you are less sure of the timing or the catalyst, or the setup is good but not your best. Risk 0.5%.
- **Quarter or starter size (0.25× R):** a probe — you want exposure to a developing thesis but the evidence is still thin, or you are establishing a position you intend to add to as it confirms. Risk 0.25%.

The tiers connect directly to the conviction you measured upstream (the discipline of measuring how sure you really are is in [measuring conviction](/blog/trading/analyst-edge/measuring-conviction-how-sure-are-you-really)). A 70% probability-of-success view with a clean catalyst earns full size; a 55% view you are taking on partial information earns half. The tier multiplier is how the *strength* of the view enters the size — which is the legitimate version of "size by how much I like it." The illegitimate version sizes by how *excited* you feel; the disciplined version sizes by how strong the *edge* is, measured before the trade, in cold blood. The discipline is in defining the tiers in advance so that excitement cannot promote a half-size idea to a full-size position in the heat of the moment.

#### Worked example: a conviction tier multiplier on the same setup

Your base R is 1% of \$100,000 = \$1,000, and your stock is at \$50 with a \$4 invalidation distance, so full size is 250 shares (as computed earlier). Now run the same setup through three conviction tiers:

- **High conviction (1.0× R):** risk \$1,000 → 1,000 / 4 = **250 shares**, \$12,500 notional.
- **Medium conviction (0.5× R):** risk \$500 → 500 / 4 = **125 shares**, \$6,250 notional.
- **Probe (0.25× R):** risk \$250 → 250 / 4 = **62 shares** (round down), \$3,100 notional.

Same stock, same entry, same stop — three different sizes, each a deliberate function of how strong you judged the edge to be *before* you clicked. The probe lets you participate in a developing thesis while risking a quarter of a percent; if it confirms, you can scale toward full size as the evidence builds. The tier is the dial that translates the *strength* of your conviction into dollars, on a scale you set in advance and refuse to override in the moment. That last clause is the whole discipline: the tiers exist so that the size is decided by the analysis, not by the adrenaline.

## Sizing errors that quietly compound

Most sizing damage does not come from a single dramatic blow-up; it comes from a handful of small, repeated errors that each feel reasonable in the moment and compound over a career. Three of them are worth naming precisely, because they are the ways even disciplined people drift back into sizing by excitement.

**Sizing by share price instead of by risk.** This error hides behind a feeling of prudence. A trader avoids a \$400 stock because "that's a lot per share" and piles into a \$5 stock because "I can buy a lot of shares." But the share price tells you nothing about risk. The \$400 stock with a tight \$4 invalidation (1% away) risks far less per dollar deployed than the \$5 stock with a \$2 invalidation (40% away). Run the share-count math from the risk down — always shares = R ÷ invalidation distance — and the share price disappears from the decision, as it should. Sizing from the share price up is how people end up with a "small" 200-share position that happens to be risking 8% of their account because the stop was wide and the price was high.

**Doubling down — averaging into a loser.** A position moves against you toward your invalidation, and the temptation is to add more "at a better price." Sometimes scaling into a planned position is legitimate — *if it was the plan from the start and the total risk still respects R.* But doubling down on a loser to "lower your average" is, in risk terms, the most dangerous move on this list: you are *increasing* your risk-per-trade on the exact position the market is telling you is going wrong, and you are doing it because of an emotional refusal to be wrong rather than because your edge grew. The arithmetic is merciless. A position sized at 1% R that you double at the halfway-to-stop point is now risking close to 2% R, and if it keeps going (as losers tend to), the loss you take is double the loss you signed up for, on the trade you were already losing. Define the full intended size *before* entry; if you want to scale in, build the scaling into the original R so the total never exceeds your unit. Never add to a loser to feel better about it.

**Ignoring correlation between positions.** This is the error that turns a portfolio of well-sized trades into a single oversized bet. Each position is a clean 1% R, but if six of them are long the same sector — or, more insidiously, long six *different* things that all happen to be the same macro bet on "rates fall and risk assets rise" — then your true risk is not six independent 1% bets. It is one 6% bet wearing six costumes. When the regime turns, they do not stop out one at a time over weeks; they all stop out in the same session, and your "diversified" 6% of heat becomes a single 6% loss in a day. The fix is to size against *correlated* heat: when positions move together, treat their combined risk as the concentrated bet it is, and count it against the cap accordingly. A book of genuinely uncorrelated 1% bets is robust; a book of correlated 1% bets is a leveraged directional position you did not realize you put on.

What unites all three errors is the same root cause as the opening disaster: each one lets something other than measured edge and defined risk set the size. The share-price error lets the quoted price set it; doubling down lets the refusal to be wrong set it; the correlation error lets the appearance of diversification hide the real bet. Sizing by edge and by risk is not one rule — it is the discipline of refusing, in every one of these moments, to let anything else into the decision.

## Common misconceptions

**"I size by how much I like the trade."** This is the most common and the most dangerous, because it sounds like conviction-based sizing — which is legitimate — but it is actually mood-based sizing, which is ruinous. The difference is *when* the judgment happens. Conviction sizing assigns a tier based on the edge you measured before the trade, in cold blood, and holds you to it. Mood sizing lets the excitement of the moment — the FOMO of a stock running, the rage of a position that went against you — set the size in real time. The first is a system; the second is a tell that you are about to do something you will regret. If your size is bigger because the chart "feels" hot rather than because the edge you measured is larger, you are sizing by excitement, and excitement is uncorrelated with edge.

**"The Kelly criterion tells me to bet 25%, so I bet 25%."** Full Kelly is a theoretical ceiling derived under an assumption you cannot meet — that you know your edge exactly. You estimate your edge from a finite, noisy sample, and the cost of *overestimating* it and betting full Kelly is catastrophic, while the cost of betting half Kelly when you were right is a modest haircut on growth. The asymmetry is decisive: the downside of betting too much (ruin) dwarfs the downside of betting too little (slightly slower compounding). Bet a fraction — half or a quarter — of whatever Kelly says, always. Anyone trading full Kelly is one bad estimate away from a drawdown they will not psychologically survive.

**"A bigger position is a better trade."** Size is not a vote of confidence that improves the trade's odds; it is an *amplifier* of whatever the trade actually is. A bigger position on a positive-EV setup makes a good thing bigger and a bad run worse. A bigger position does not make you more right — it makes the consequences of being right *or* wrong larger. The all-in analyst in the opening was not more right than the disciplined one; he had a bigger position on the same view, and the bigger position is precisely what destroyed him when the view, briefly, went the wrong way. Confidence belongs in your tier assignment, sized in advance; it does not belong in an ad-hoc decision to "really lean into this one."

**"Sizing is just risk management — it's separate from forming the view."** Sizing *is* the view, expressed in the only language the market understands: dollars at risk. A view without a size is an opinion. The strength of your conviction is not some abstract quality that lives in your head; it is *defined* by how much you are willing to risk on it. If you say you are "highly confident" but you size the trade at a quarter percent, you are not highly confident — your size is telling the truth your words are hiding. Sizing is the moment your view becomes accountable, the point where "I believe X" turns into "I have \$N riding on X," and that is not separate from analysis; it is the *completion* of analysis.

**"I size by the share price."** A surprising number of people anchor on the share count or the share price — "I bought 100 shares" or "I never put more than \$50 a share into anything" — neither of which has anything to do with risk. A hundred shares of a \$10 stock with a \$5 stop risks \$500; a hundred shares of a \$10 stock with a \$1 stop risks \$100. Same share count, fifth the risk. The share price is irrelevant to sizing; what matters is the dollar risk, which is shares times invalidation distance. Always size from the risk down to the shares, never from the share price up.

## How it plays out in real markets

The sizing rules are not academic. Watch them decide outcomes in episodes you remember.

**The COVID crash, March 2020.** Markets fell roughly 34% from the February 19 peak to the March 23 low in about a month — one of the fastest crashes in history. Consider two traders who were both, reasonably, long going into it. The first sized every position at 1% R with hard invalidations and capped portfolio heat at 6%; when the cascade began, his stops triggered across the book, and the *maximum* his open positions could cost him was his 6% heat — he took a bad week, raised cash, and lived to buy the March lows. The second was running concentrated, stopless, "long-term conviction" positions sized at 15–20% each, because he *knew* the businesses were great. He was right about the businesses. It did not save him from a 50%+ drawdown and, if he was on any leverage, forced liquidation at the worst possible prices. The market did not punish the second trader for being wrong about companies; it punished him for sizing as if a 34% crash could not happen to him. Heat caps and per-trade R are precisely the machinery that bounds your loss in the month you did not see coming.

**The 2018 volatility spike — "Volmageddon," February 5, 2018.** The VIX more than doubled in a single session and a popular short-volatility product (XIV) lost about 96% of its value overnight and was liquidated. The people wiped out were not, mostly, wrong about the *strategy* — selling volatility is a real, persistent edge most of the time. They were wrong about *size*. Selling vol is a strategy whose returns look like picking up small, steady gains until a tail event hands you a catastrophic loss — the textbook profile that full Kelly and equal-dollar sizing handle worst. Put it in dollars: a trader with \$100,000 who had put \$50,000 into the short-vol product because it had returned steadily for two years lost about \$48,000 in one night — nearly half the account, on a single position. The same trader sizing that exposure at 1% R, capping the loss at \$1,000, took a rounding-error hit and woke up to fight another day. Anyone who had sized that exposure at a fractional-Kelly fraction of their bankroll, or vol-targeted it so the position shrank as volatility's *potential* to spike grew, survived the night with a bruise. Anyone who sized it big because it had "always worked" was gone by morning. The edge was real; the sizing was fatal — and the difference between the \$48,000 loss and the \$1,000 loss was nothing but the position size on an identical view.

**The 2022 rate-hike grind.** Through 2022 the S&P 500 fell about 19% on the year as the Fed hiked aggressively, but the path mattered as much as the destination: a series of vicious bear-market rallies and fresh-low selloffs that stopped traders out repeatedly, in both directions. This is the environment where *correlation* between positions quietly kills you. A book of "diversified" long-equity positions in 2022 was not diversified — every name was the same bet on "rates stay low and risk assets rise," and they all stopped out together on the same hot CPI prints. A trader who tracked portfolio heat saw the correlation and held total open risk under the cap; a trader who counted thirty "independent" 1%-R positions discovered, on a single 4%-down day, that thirty correlated stops is a 30% drawdown. The lesson 2022 hammered home: heat must be measured against *correlated* risk, not the naive sum, because the market collects on correlation exactly when you can least afford it.

In all three episodes the entry was almost beside the point. What separated the survivors from the casualties was whether their size — per trade and across the portfolio — was bounded by a rule they set in calm and refused to override in panic.

## The playbook

Here is the routine that turns a view into a position. Run it in order, every time, before you click. The sequence is the same one drawn in the figure at the top of this post — view in, dollars at risk out, five gates in between.

1. **Set your risk-per-trade unit (R).** Decide what fraction of *current* equity you will lose if this trade hits its invalidation. Default to 1%. On a \$100,000 account, R = \$1,000. Recompute R from current equity each time so you de-risk automatically into drawdowns and press into gains. This is the unit; everything below is measured in it.

2. **Convert the invalidation distance into a share count.** Find the price at which your thesis is wrong (your invalidation, defined *before* the trade, never moved looser afterward). Measure the distance from entry to that point. Then:
   $$\text{shares} = \frac{R}{\text{invalidation distance}}$$
   The share count is arithmetic, not a guess. A wider stop means fewer shares for the same risk; the dollar risk stays pinned at R.

3. **Apply the conviction-tier multiplier.** Assign the trade to a predefined tier — full (1.0× R), half (0.5× R), or probe (0.25× R) — based on the *edge you measured*, not how excited you feel. Multiply R by the tier before computing shares. The tiers are set in advance and not negotiable in the moment; that is what keeps strength-of-view in the size and mood out of it.

4. **Apply the volatility adjustment across positions.** When you are sizing without a hard stop, or comparing positions of very different volatility, target equal *risk* rather than equal dollars: notional = (target daily dollar risk) ÷ (daily volatility). The high-vol name gets the smaller position. This makes every position contribute the same risk, so your book is balanced by risk, not by accident of price.

5. **Check portfolio heat against the cap.** Sum the risk-if-stopped across *all* open positions, including the new one. If the total exceeds your cap (default 6% of equity), the trade does not go on at full size — trim an existing position, size the new one down, or pass. And size against *correlated* heat: count positions that would stop out together as the concentrated bet they really are, not as independent risks.

Run those five steps and the output is a specific, defensible number of shares at a known dollar risk — a position, not a feeling. The discipline that makes it work is captured in the checklist card below; print it, tape it to your monitor, and run it before every trade until it is automatic.

![The five step bet-sizing checklist card from risk unit to portfolio heat cap](/imgs/blogs/from-conviction-to-size-the-bet-sizing-bridge-7.png)

The card is the post in operating form. Step one fixes the unit. Step two turns the invalidation distance into shares. Step three scales for conviction. Step four equalizes risk across names. Step five caps the sum. The first three steps are blue and green because they are the constructive path — building the position up from your edge. The last two are amber and red because they are the *governors* — the volatility adjustment and the heat cap exist to stop you, to shrink a position or block a trade when the risk is quietly too large. A complete sizing decision touches all five.

One closing reframe, because it is the thing to carry out of here. Sizing is not the unglamorous administrative step that happens after the real work of forming a view. Sizing *is* where the view becomes real. Everything upstream — the lenses, the variant perception, the priced-in analysis, the expected-value table — produces an opinion, and an opinion moves nothing. The position moves your account, and the size of the position is the exact measure of how much that opinion is worth in dollars. The two analysts in the opening had the identical view. One sized by edge and by risk and is still trading; one sized by conviction and excitement and is not. Size by edge and by risk, never by excitement — and a good view will compound your capital instead of ending your career. The next post takes this one step further, into the structure of trades whose *payoff* is asymmetric — the high-conviction bets where sizing and asymmetry compound each other (covered in [asymmetry and the art of the high-conviction bet](/blog/trading/analyst-edge/asymmetry-and-the-art-of-the-high-conviction-bet)).

![Sizing by dollars versus sizing by risk shown as a before and after comparison](/imgs/blogs/from-conviction-to-size-the-bet-sizing-bridge-6.png)

The final figure compares the two worlds side by side. On the left, sizing by dollars: buy \$10,000 of everything, and the loss per trade swings 6× depending on whether the stop is tight or wide — your "constant" size produces wildly inconstant risk. On the right, sizing by risk: risk \$1,000 on everything, and the *share count* flexes (2,500 shares of the tight-stop name, 400 of the wide-stop name) while the loss per trade stays pinned at \$1,000. The left column is what amateurs do and the right column is what professionals do, and the entire difference is which number you hold constant: the dollars you put in, or the dollars you can lose. Hold the loss constant, and let everything else flex around it. That is the bridge.

## Further reading & cross-links

**Within this series — the view you are sizing:**

- [Measuring conviction: how sure are you, really?](/blog/trading/analyst-edge/measuring-conviction-how-sure-are-you-really) — the conviction that feeds the tier multiplier in step three.
- [Expected value: the only math a view really needs](/blog/trading/analyst-edge/expected-value-the-only-math-a-view-really-needs) — the EV that justifies taking the trade before you size it.
- [Thinking in probabilities, not predictions](/blog/trading/analyst-edge/thinking-in-probabilities-not-predictions) — why every view is a distribution, which is what makes sizing necessary.
- [What would change my mind? Defining invalidation upfront](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront) — the invalidation distance that converts risk into a share count in step two.
- [Base, bull, and bear: building three scenarios](/blog/trading/analyst-edge/base-bull-and-bear-building-three-scenarios) — the scenarios whose payoffs and probabilities set your edge.
- [Asymmetry and the art of the high-conviction bet](/blog/trading/analyst-edge/asymmetry-and-the-art-of-the-high-conviction-bet) — where sizing meets convex payoffs (forward).

**Out to the risk-management deep math:**

- [Position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) — the practitioner's view of the share-count and Kelly mechanics.
- [The Kelly criterion: how much to bet when you have an edge](/blog/trading/risk-management/the-kelly-criterion-how-much-to-bet-when-you-have-an-edge) — the full derivation behind the edge-over-odds fraction.
- [Fractional Kelly and optimal-f: betting less to sleep at night](/blog/trading/risk-management/fractional-kelly-and-optimal-f-betting-less-to-sleep-at-night) — why half and quarter Kelly are the practical choices.
- [Volatility-targeting: sizing by risk, not by dollars](/blog/trading/risk-management/volatility-targeting-sizing-by-risk-not-by-dollars) — the cross-position equalizer in step four, in depth.
- [Risk management: the only free lunch — survival as a compounding engine](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine) — why bounded risk is the precondition for compounding at all.
- [The asymmetry of losses: why a 50% loss needs a 100% gain](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) — the drawdown arithmetic that makes small R non-negotiable.

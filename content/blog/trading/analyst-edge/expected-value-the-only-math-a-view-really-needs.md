---
title: "Expected Value: The Only Math a View Really Needs"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Turn any probabilistic market view into a take-or-pass decision with one calculation: expected value, the sum of probability times payoff minus costs."
tags: ["analysis", "market-view", "expected-value", "probability", "payoff", "asymmetry", "win-rate", "trading-process", "position-sizing", "convexity", "costs"]
category: "trading"
subcategory: "The Analyst's Edge"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — You do not need stochastic calculus to trade well. You need expected value: the sum of (probability × payoff) across every outcome, minus costs. It is the single calculation that turns a probabilistic view into a decision.
>
> - **EV is the number, win rate is vanity.** An 80%-win-rate strategy can lose money if the 20% of losses are large enough; a 30%-win trade can be excellent if the payoff is convex.
> - **Take the trade only when EV is positive after costs**, then size it by the edge. EV is the first filter; sizing handles ruin.
> - **The EV table is the decision artifact**: scenarios down the side, probability × payoff across, sum to gross EV, subtract costs, read off take-or-pass.
> - The one rule: **a view you cannot put into an EV table is not a tradeable view yet** — and EV is only as good as the probabilities you feed it.

A trader you know is up at the bar after the close, and he is talking. He has, he tells you, an 80% win rate. Eight trades out of every ten close green. He shows you his journal on his phone and the green ticks really are there — a long, satisfying column of small wins, day after day. He sells volatility into earnings, fades stretched moves, scalps the open. Most days, most trades, he is right. And yet — you happen to know, because you clear at the same prop shop — he is *down on the year.* Not flat. Down. A man who is right four times out of five, bleeding money.

This is not a paradox, and it is not bad luck. It is arithmetic, and it is the single most important arithmetic in trading. His 80% of winners are small — he takes \$300 here, \$400 there, clipping premium and fading noise. His 20% of losers are not small. When the volatility he sold explodes, or the stretched move keeps stretching, he loses \$3,000, \$5,000, occasionally more. Run the numbers and the column of green ticks is a mirage: eight wins of \$400 is \$3,200, and two losses of \$2,500 is \$5,000, so across every ten trades he is down \$1,800 *while being right 80% of the time.* His win rate is a vanity metric. The number that actually governs his account is one he never computes.

That number is **expected value.** It is the average dollar outcome of a trade if you could run it many times — the probability-weighted sum of everything that can happen, net of what it costs to put on. It is the only math a view really needs, in the precise sense that it is the calculation that *converts* a view into a decision. Forming a market view is the subject of this entire series — reading the lenses, finding the variant perception, knowing what is priced in. But a view, however brilliant, is inert until you can answer one question: *given this view, do I take the trade, and how much do I bet?* Expected value is the bridge from "I believe X" to "so I do Y," and the rest of this post builds it from zero.

![Each scenario probability times its dollar payoff sums to one expected value minus costs](/imgs/blogs/expected-value-the-only-math-a-view-really-needs-1.png)

The figure above is the whole post in one picture, and we will spend the rest of it earning the right to trust that picture. You list the scenarios your view implies. You attach a probability and a dollar payoff to each. You multiply and sum to get the gross expected value. You subtract the costs — commissions, the spread, slippage, financing. And the single number that survives at the end is the decision: positive after costs, you take it and size it; negative, you pass, no matter how good the story sounded. The 80%-win-rate trader at the bar skipped this calculation his whole career. Let's make sure you never do.

## Foundations: what expected value actually is

Start with the word itself, because it is slightly misleading. The "expected value" of a trade is *not* the value you expect to get on any single trade — in fact, it is frequently a number you will *never* get on any single trade. It is the **long-run average** outcome: if you could clone this exact trade, with these exact odds, and run it a thousand times, expected value is the average dollar result per run. The law of large numbers does the rest — over enough independent repetitions, your realized average converges on the expected value. One trade is a coin flip; a thousand trades is the EV.

Here is the precise definition we will use for the whole post. For a trade with a set of possible outcomes, where outcome $i$ happens with probability $p_i$ and produces a dollar payoff $x_i$ (positive for a gain, negative for a loss), the **expected value** is:

$$\text{EV} = \sum_i p_i \, x_i = p_1 x_1 + p_2 x_2 + \dots + p_n x_n$$

Read it left to right: for each thing that can happen, multiply how likely it is by how much money it makes or loses, and add up all those products. That sum is the EV. Two rules make it honest. First, **the probabilities must sum to 1** (100%) — you have to account for everything that can happen, including "nothing happens." Second, **the payoffs are in dollars on a specific position** — not percentages, not "points," but the actual P&L on the actual size you would trade, because that is what hits your account.

A worked instance fixes the idea. Suppose you flip a fair coin. Heads, you win \$10; tails, you lose \$6. The probabilities are 50% and 50%. The EV is $0.5 \times \$10 + 0.5 \times (-\$6) = \$5 - \$3 = \$2$. That \$2 is the EV per flip. You will never *actually* win \$2 on any flip — you win \$10 or lose \$6, never \$2 — but if you flip a thousand times, you will be up roughly \$2,000, give or take. The \$2 is not a prediction about the next flip; it is the gravitational center the average is pulled toward over many flips. This distinction — EV as the long-run average, not the next outcome — is the source of half the confusion about the concept, and we will return to it in the misconceptions section.

### Probability times payoff, scenario by scenario

A real trade is not a coin flip with two outcomes; it is a *spread* of outcomes. So the working version of EV is built from **scenarios.** A scenario is a coherent story about what happens — the bull case, the base case, the bear case — each with its own probability and its own dollar payoff. You do not need a continuous probability distribution; three to five discrete scenarios that span the realistic range are enough to make a good decision, and they are vastly easier to reason about. The art is in choosing scenarios that are *mutually exclusive* (they don't overlap) and *collectively exhaustive* (together they cover everything that can plausibly happen, so the probabilities sum to 100%).

Take a concrete position. You are long a stock into an event — say you buy \$25,000 of shares because you believe a product launch will go well. You sketch three scenarios. In the **bull** case (the launch is a hit and the stock rallies), you make \$8,000. In the **base** case (it's fine, the stock drifts up modestly), you make \$1,500. In the **bear** case (the launch disappoints and the stock sells off), you lose \$5,000. Those are the payoffs. Now the probabilities, which are *your view* — this is where the analysis you did upstream gets expressed as numbers. You think the bull case has a 35% chance, the base case 40%, and the bear case 25%. They sum to 100%, good.

Multiply and sum: $0.35 \times \$8{,}000 + 0.40 \times \$1{,}500 + 0.25 \times (-\$5{,}000) = \$2{,}800 + \$600 - \$1{,}250 = \$2{,}150$. The gross expected value of this trade is **+\$2,150.** That is the average dollars you make, before costs, every time you put on a trade with these odds and these payoffs. The trade is positive-EV. We have not yet subtracted costs or talked about how much to bet — but we have already done the one thing that matters most, which is to turn a vague "I think the launch goes well" into a hard number that says *this is worth doing.*

### Including costs: the EV that actually matters

The \$2,150 above is the *gross* EV, and gross EV is a fantasy. No trade is free. To get the EV that actually lands in your account, you subtract the **frictions**: the commission you pay to your broker, the **bid-ask spread** you cross to get filled (you buy at the offer and sell at the bid, and the gap between them is a cost you pay twice, on the way in and the way out), the **slippage** between the price you saw and the price you got (the market moves while your order works, especially in size), the financing cost if you are using leverage or holding overnight, and any borrow fee or tax. Call the total cost $C$. The honest expected value is:

$$\text{EV}_{\text{net}} = \left(\sum_i p_i \, x_i\right) - C$$

For the launch trade, suppose the round-trip cost — commission plus half-spread in and out plus a little slippage on a \$25,000 position — comes to \$150. Then $\text{EV}_{\text{net}} = \$2{,}150 - \$150 = \$2{,}000$. Still strongly positive. **This is the number you trade on.** Costs look small on a single good trade and they are easy to wave away, but they are not small in aggregate, and on thin-edge trades they are frequently the entire game. We will see a worked example later where a \$300 gross edge becomes a *loss* once costs are subtracted — the difference between a strategy that compounds and one that bleeds is often nothing but the cost line.

![Probability times dollar payoff per row sums to gross EV then costs net it out](/imgs/blogs/expected-value-the-only-math-a-view-really-needs-2.png)

The table above is the single most important artifact in this entire post, and it is worth pausing on its structure because you will draw it for the rest of your trading life. Down the left are the scenarios. Across the top are probability, dollar payoff, and the *contribution* — which is just probability times payoff for that row. The contributions sum to the gross EV. The bottom row nets out costs and reads off the answer. Notice what the table forces you to do: it makes you write down the bear case (you cannot leave the row blank), assign it an honest probability (the rows must sum to 100%), and price the actual dollar loss (not a hand-wave). Most bad trades die in this table, killed not by the math but by the act of being *forced to write the downside down.*

### Positive EV, negative EV, and the decision rule

Now the decision rule, which is almost embarrassingly simple given how much work went into the table:

- **If $\text{EV}_{\text{net}} > 0$: the trade is worth taking.** On average, repeated, it makes money. How much you bet is a separate question (sizing — we get there), but the trade earns its place in your book.
- **If $\text{EV}_{\text{net}} \le 0$: pass.** No matter how compelling the narrative, how high the win probability, or how much you want it — if the probability-weighted dollars come out non-positive after costs, repeating this trade is a slow path to zero.

That is the entire decision criterion. Everything else in this post is about (a) computing the table honestly and (b) understanding why this rule survives contact with the things that *look* like they should override it — high win rates, scary downsides, low win probabilities. The rule does not care about any of those directly. It cares about one number, and the number already contains all of them.

### EV as expectancy: win rate is one term, not the answer

There is a second way to write the EV of a trade that makes the relationship to win rate explicit, and it is worth keeping in your head because it shows exactly where win rate lives in the formula — as *one term among several*, never as the answer. Collapse the scenarios into two buckets, "win" and "lose," and the EV becomes:

$$\text{EV} = (W \times A_W) - (L \times A_L) - C$$

where $W$ is the win probability (the win rate), $A_W$ is the average win in dollars, $L$ is the loss probability ($L = 1 - W$), $A_L$ is the average loss in dollars, and $C$ is costs. Traders call this **expectancy** — the expected dollars per trade. Stare at it and the bar-stool error becomes obvious: win rate $W$ is one of *four* numbers, and it is multiplied by the average win, not standing alone. You can push $W$ all the way to 0.80 and still get a negative number if $A_L$ is large enough — which is precisely what happens when you take small wins and let losses run. The two trades to come make this vivid, but the algebra already tells you the punchline: **you cannot read profitability off the win rate, because the win rate is only one of the four terms that determine it.** The other three — your average win, your average loss, and your costs — are doing at least as much work, and a strategy that improves $W$ by shrinking $A_W$ or inflating $A_L$ has made itself *worse* while looking better on the only metric the loser at the bar was tracking.

### EV versus the most-likely outcome

Here is the subtlety that trips up nearly everyone, and it is worth its own subsection because confusing the two is a classic, expensive error. **Expected value is not the most-likely outcome.** They are different quantities and they frequently disagree.

In the launch trade, the *most-likely* single outcome is the base case — it has the highest probability, 40%, so if you had to bet on one scenario, you'd bet "modest drift up, make \$1,500." But the *expected value* is \$2,000, which is higher than the most-likely payoff of \$1,500, because the fat 35% bull case pulls the average up. The EV is a blend of all scenarios weighted by probability; the mode is just the single tallest bar. A trade can have a positive EV while its most-likely outcome is a *loss* — a lottery-like structure where you most often lose a little and rarely win a lot. And a trade can have a negative EV while its most-likely outcome is a *win* — which, as it happens, is exactly the structure that destroyed our 80%-win-rate friend at the bar. He optimized for the most-likely outcome (a small win) and ignored the EV (negative, because the rare losses were huge). Trade the EV, not the mode.

## Computing a trade's EV in practice

The foundations give you the formula. Practice is about doing it well under real conditions — choosing good scenarios, attaching defensible probabilities, pricing payoffs in real dollars, and not fooling yourself. Let's build a full worked example end to end on a position size you might actually run.

#### Worked example: the full EV table on a \$25,000 position

You have done your work on a mid-cap stock trading at \$50. Your view: an upcoming guidance update is more likely to surprise positively than the market expects. You decide the expression is to buy \$25,000 of stock — 500 shares at \$50 — and hold through the update, then exit. You build the table.

You define three scenarios and, crucially, you define them by *price target*, then convert to dollars on your 500 shares:

- **Bull** — guidance is strong, stock to \$66 (+\$16/share). Payoff: $500 \times \$16 = +\$8{,}000$. Your probability: **35%.**
- **Base** — guidance is fine, stock to \$53 (+\$3/share). Payoff: $500 \times \$3 = +\$1{,}500$. Your probability: **40%.**
- **Bear** — guidance disappoints, stock to \$40 (−\$10/share). Payoff: $500 \times (-\$10) = -\$5{,}000$. Your probability: **25%.**

The probabilities sum to 100%. Now the contributions:

$$0.35 \times \$8{,}000 = +\$2{,}800$$
$$0.40 \times \$1{,}500 = +\$600$$
$$0.25 \times (-\$5{,}000) = -\$1{,}250$$

Gross EV $= \$2{,}800 + \$600 - \$1{,}250 = +\$2{,}150$. Costs on a \$25,000 round trip — commission, crossing the spread twice, a touch of slippage — run about \$150. Net EV $= \$2{,}150 - \$150 = +\$2{,}000$. The decision: **take it.** The trade returns, on average, +\$2,000 per \$25,000 deployed — an 8% expected return on the position before you even think about how the rest of your book is allocated. The intuition: a positive net EV means that repeating this exact bet many times grows the account, so this view has earned a place in the book.

Notice three things this example teaches that the formula alone does not. First, **the payoffs came from price targets, not from gut feel** — you anchored each scenario to a specific level and let the share count do the arithmetic, which keeps you honest. Second, **the bear case is in the table at full size** — \$5,000 of real loss, 25% of the time, staring back at you; a lot of "obvious" trades stop looking obvious once the bear row is filled in. Third, **the EV of +\$2,000 is a number you will never actually realize on the trade itself** — you'll make \$8,000 or \$1,500 or lose \$5,000 — but it is the right number to *decide* on, because you are going to make many such decisions over a career.

### Why a 35%-win trade can be excellent: asymmetry and convexity

Now we attack the deepest and most counterintuitive idea in the whole subject, the one that separates traders who understand EV from those who merely recite it. **A trade can win only a minority of the time and still be one of the best trades you will ever make** — provided the payoff is *asymmetric*: small, frequent losses and large, rare wins. This is the structure of buying a cheap option, of a venture investment, of a breakout trade, of any bet where the downside is capped and the upside is open-ended. The word for it is **convexity** — your payoff curves upward, so the wins are disproportionately larger than the losses.

The reason win rate misleads here is that it throws away the *size* of the outcomes and keeps only their *sign*. A 35% win rate tells you that 35% of the time the trade is green — but it says nothing about whether the green is \$200 or \$20,000, or whether the red is \$200 or \$20,000. EV keeps the sizes. And once you keep the sizes, a low-win-rate, high-payoff structure can dominate.

#### Worked example: a 30%-win trade that is strongly +EV

You are looking at an option-like, asymmetric setup on a \$25,000 risk budget. The structure: you risk a defined amount, and most of the time you lose some or all of it, but in the rare scenario where your thesis hits, the payoff is many multiples of the risk. You map four outcomes:

- **Total loss** — thesis wrong, the position goes to near-zero. Payoff: **−\$2,500.** Probability: **45%.**
- **Partial loss** — thesis early or partially wrong, you cut for a smaller loss. Payoff: **−\$1,000.** Probability: **25%.**
- **Small gain** — thesis works modestly. Payoff: **+\$1,500.** Probability: **10%.**
- **Big win** — thesis fully hits, convex payoff. Payoff: **+\$18,000.** Probability: **20%.**

The win rate here is the small gain plus the big win — **30%.** Seven times out of ten, this trade loses money. Every instinct trained by the vanity of win rate screams *gambling, stay away.* Now compute the EV:

$$0.45 \times (-\$2{,}500) = -\$1{,}125$$
$$0.25 \times (-\$1{,}000) = -\$250$$
$$0.10 \times (\$1{,}500) = +\$150$$
$$0.20 \times (\$18{,}000) = +\$3{,}600$$

Sum: $-\$1{,}125 - \$250 + \$150 + \$3{,}600 = +\$2{,}375$. The expected value is **+\$2,375** — even *higher* than the "safe" launch trade, on the same capital, with less than half the win rate. The single \$18,000 outcome, at 20%, contributes +\$3,600 all by itself, which is more than enough to pay for all the losing scenarios with room to spare. The intuition: when the wins are far larger than the losses, you can be wrong most of the time and still come out far ahead — convexity, not frequency, is what pays.

![A 30 percent win asymmetric trade can be strongly positive EV](/imgs/blogs/expected-value-the-only-math-a-view-really-needs-4.png)

The chart makes the mechanism visible. The two red bars on the left are the losses — frequent (45% and 25%) but small. The two green bars on the right are the wins — rare (10% and 20%) but the rightmost one is enormous. The dashed blue line is the EV at +\$2,375, sitting well above zero. The bar chart and the win rate tell opposite stories: the win rate counts the bars (mostly red, "bad trade") while the EV weights the bars by both probability *and* size (the one tall green bar dominates, "great trade"). Whenever a setup has this shape — capped downside, open-ended upside — distrust your win-rate instinct and run the EV. This is why traders who buy cheap convexity look "wrong" most days and compound anyway, and it is the seed of an entire approach we develop later in [asymmetry and the art of the high-conviction bet](/blog/trading/analyst-edge/asymmetry-and-the-art-of-the-high-conviction-bet).

### EV of holding versus exiting: the decision repeats

EV is not a one-time calculation you do before entering and then forget. It is a *live* number that you should recompute at every decision point, because the most common version of the question is not "should I enter?" but "should I stay in?" Once you are in a position, the relevant comparison is the EV of *holding* versus the EV of *exiting now.* And the entry price is irrelevant to that comparison — a hard, unnatural truth.

Here is the logic. Suppose you are in the launch trade and the stock has already run from \$50 to \$60 ahead of the update. You are up \$5,000 on paper. The question "should I hold through the update?" has nothing to do with the fact that you bought at \$50. It depends only on the EV *from here*: given the stock is now at \$60, what are the scenarios and probabilities for the update, and what is the EV of holding versus the certain \$5,000 you lock in by selling now? If the remaining EV of holding is positive and large, you hold; if the run-up has eaten most of the upside so the remaining EV is negative (you are now risking the \$5,000 gain against a small remaining edge), you sell. The price you paid is a **sunk cost** — it is information about your past, not about the future distribution of the position, and EV is forward-looking by construction. Recomputing EV at each decision point is what frees you from the two great position-management errors: holding losers because "I'm down and want to get back to even," and selling winners because "I'm up and don't want to give it back." Both anchor to the entry price. EV anchors to the future.

Put dollars on it so the discipline is concrete. The stock is at \$60, you hold 500 shares, and the update is still ahead. From \$60, your scenarios are: bull (still good news beyond the run-up, stock to \$66, +\$3,000 more) at 25%; base (the run-up was the move, stock flat at \$60, \$0) at 45%; bear (any stumble unwinds the rally, stock back to \$52, −\$4,000 from here) at 30%. The EV of *holding* from \$60 is $0.25 \times \$3{,}000 + 0.45 \times \$0 + 0.30 \times (-\$4{,}000) = \$750 + \$0 - \$1{,}200 = -\$450$. The EV of *selling now* is the certain \$5,000 gain you bank, which from a forward perspective is \$0 of additional risk and \$0 of additional EV — you simply keep what you have. Holding has a *negative* forward EV of −\$450; selling has a forward EV of \$0. So you sell, lock the \$5,000, and walk. Crucially, this decision is identical whether you bought at \$50 (up \$5,000) or at \$58 (up \$1,000) or at \$62 (down \$1,000) — the entry price never entered the calculation. The forward distribution from \$60 is all that mattered, and it said the rally had priced in the good news and left you holding the downside. A trader anchored to "I'm up \$5,000, let it ride" holds a negative-EV position out of greed; a trader anchored to "I bought at \$62 and won't sell at a loss" holds it out of stubbornness. The EV calculation ignores both feelings and reads off the same answer: sell.

### EV with fat tails: when the rare scenario dominates

The EV formula sums over scenarios, which quietly assumes you have *included the right scenarios.* The most dangerous failure of EV in practice is not bad arithmetic — it is a missing row. Real market returns have **fat tails**: extreme outcomes happen far more often than a tidy bell curve predicts, and they are larger when they happen. The 2020 COVID crash, the 2018 volatility spike, the 2008 collapse — these are not three-sigma curiosities that a careful analyst can ignore; they are the events that determine the long-run EV of entire strategies, because their payoffs are so large that even a tiny probability dominates the sum.

This is precisely the trap our 80%-win-rate trader fell into. Selling volatility has a payoff structure that is the mirror image of the asymmetric trade above: frequent small wins (you collect premium) and rare enormous losses (the tail event). If you build the EV table and *leave out the tail row* — if you model only "calm market, collect premium" and "mild move, give a little back" — the EV looks fabulous and the strategy looks like free money. Add the row that says "1% of the time, a tail event costs you \$40,000," and the same strategy is wildly negative-EV. The arithmetic did not change; the table got honest. When you build an EV table for any strategy with a capped upside and an open downside — selling options, carry trades, picking up pennies in front of a steamroller — you *must* include a fat-tail row with a brutally large loss and a small-but-not-zero probability, or your EV is a comforting fiction. We treat the mechanics of this in depth via the cross-link to [fat tails and the normal-distribution trap](/blog/trading/risk-management/fat-tails-and-the-normal-distribution-trap); here the point is narrow and load-bearing: **EV is a sum over scenarios, and a sum is only as complete as its terms.**

### EV is necessary but not sufficient: variance and ruin

Now the honest limitation, stated plainly so no one accuses EV of claiming too much. **Positive EV is necessary for a good trade, but it is not sufficient.** EV tells you the average outcome over many repetitions. It says *nothing* about the path — the volatility, the drawdowns, the order in which the wins and losses arrive. And the path can kill you before the average ever shows up.

The killer is **ruin.** EV's promise — that your realized average converges on the EV over many trades — is only redeemable *if you survive to make many trades.* If a single bad outcome wipes you out, the long run never arrives. The asymmetric trade above has a glorious +\$2,375 EV, but it also has a 45% chance of losing \$2,500 on each attempt; if you put your entire \$25,000 account into one such trade, a string of three total losses in a row (perfectly possible at 45% each) does serious damage, and over-betting it can mean you are gone before the +20% big-win scenario ever shows up. This is why EV is the *first* filter, not the only one. EV decides *whether* to take a trade; **position sizing** decides *how much* so that no single outcome, and no plausible run of bad outcomes, can take you out of the game. The two are a matched pair, and we hand the baton to sizing in [from conviction to size: the bet-sizing bridge](/blog/trading/analyst-edge/from-conviction-to-size-the-bet-sizing-bridge), which builds on the classic framework in [position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion). For now, internalize the division of labor: *EV is the green light; sizing is the speed limit.* You need both, and they answer different questions.

### Garbage in, garbage out: the probabilities are the hard part

There is a temptation, once you have the EV machinery, to treat it as an oracle — feed in numbers, read out a decision, done. But the machinery is trivial; the inputs are everything. **EV is exactly as good as the probabilities and payoffs you feed it.** The payoffs are usually the easy part — they come from price targets and position size, and they are checkable. The probabilities are where the real skill, and the real self-deception, live.

Where do the probabilities come from? They are *your view, quantified.* They are the output of all the upstream work this series is about — reading the lenses, mapping the consensus, finding the variant perception, knowing what is priced in. If you assign the bull case 35% because you have a genuine, well-reasoned edge that the market is underpricing the upside, your EV is a real estimate. If you assign it 35% because you are excited about the trade and 35% felt good, your EV is a number-shaped feeling. The format is identical; the content is the difference between analysis and astrology. Two disciplines protect you. First, **calibrate** — track whether the things you call 70% actually happen 70% of the time, and if your 70%s come in at 50%, your probabilities are inflated and so is every EV you compute. Second, **stress the inputs** — recompute the EV with pessimistic probabilities and see if the decision survives; if the trade is only positive-EV under your most optimistic guesses, it is not really positive-EV, it is wishful. We develop the discipline of working in probabilities properly in [thinking in probabilities, not predictions](/blog/trading/analyst-edge/thinking-in-probabilities-not-predictions); the warning here is blunt: the EV table launders your guesses into a confident-looking number, so the table is only as trustworthy as your honesty about what goes into it.

## Common misconceptions

The concept of expected value is simple to state and genuinely hard to internalize, because several intuitions that feel airtight are wrong. Each of these has cost real traders real money.

**"A high win rate means I'm profitable."** This is the bar-stool error from the opening, and it is the most common and most expensive misconception in trading. Win rate counts how *often* you win; it discards how *much.* Profit is win rate *times average win* minus loss rate *times average loss* — the sizes are half the equation and win rate ignores them entirely. An 80%-win strategy with \$300 wins and \$2,500 losses is a money-loser; a 35%-win strategy with \$8,000 wins and \$2,000 losses is a money-printer. Win rate is comfortable because being right *feels* like winning, and a long column of green ticks is emotionally satisfying. But the account does not care how often you were right. It cares about the dollar-weighted average, which is the EV.

![Win rate is not expected value across illustrative trading styles](/imgs/blogs/expected-value-the-only-math-a-view-really-needs-3.png)

The scatter makes the divorce between win rate and EV total. Each bubble is a trading style; the horizontal axis is win rate, the vertical axis is EV per trade, and the bubble size grows with the win/loss asymmetry. There is *no* relationship between left-right (win rate) and up-down (EV): the 85%-win "premium seller" sits deep in negative-EV territory, while the 30%-win "event asymmetry" style sits at the top, strongly positive. Green bubbles (positive EV) and red bubbles (negative EV) appear at high and low win rates alike. If win rate predicted EV, the bubbles would line up along a diagonal; instead they scatter, because win rate and EV are measuring different things. This single picture should permanently inoculate you against the bar-stool boast. The deeper treatment of why win rate is a deceptive statistic lives in [expectancy: why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies), which is the same idea from the systematic-trading angle.

**"EV is the most-likely outcome."** No — EV is the probability-weighted average across *all* outcomes, and it routinely differs from the single most-likely one. The most-likely outcome is the mode (the tallest bar); the EV is the balance point of the whole distribution. In a convex trade, the most-likely outcome is a small loss while the EV is a large gain. In a "picking up pennies" trade, the most-likely outcome is a small win while the EV is a large loss. People who plan around the most-likely outcome are systematically blindsided by the tails that dominate the average. Decide on the EV; the mode is just one bar in the chart.

**"Small-probability trades are gambling."** This conflates probability with value, and it is the error that keeps cautious traders out of the best risk-reward setups in the market. A 20%-chance trade is "gambling" only if the payoff doesn't justify the odds. If you risk \$1 to make \$20 at 20% odds, the EV is $0.20 \times \$20 - 0.80 \times \$1 = \$4 - \$0.80 = +\$3.20$ per dollar risked — that is not gambling, that is one of the best bets you will ever find, and the casino would never offer it. Conversely, a 95%-chance trade where you risk \$20 to make \$1 has an EV of $0.95 \times \$1 - 0.05 \times \$20 = \$0.95 - \$1.00 = -\$0.05$ — that is the actual gamble, dressed up as a sure thing. Probability alone is meaningless. Probability times payoff is the whole game. The low-probability convex trade is often the *more* disciplined choice, not the reckless one.

**"EV ignores risk, so it's useless for real trading."** EV does not *ignore* risk — it just doesn't handle *all* of risk by itself, which is a different and fair point. EV is the **first filter**: it tells you whether a trade is worth doing on average. Risk of ruin — the path, the variance, the catastrophic single outcome — is handled by **sizing**, the second filter, which decides how much to bet so the trade can't take you out. The criticism that "EV ignores risk" is really an argument for *using EV plus sizing together*, which is exactly the correct workflow. Throwing out EV because it doesn't do sizing's job is like throwing out a thermometer because it doesn't cook the meal. Use EV to decide *whether*; use sizing to decide *how much*. The full survival argument — that staying in the game is itself a compounding engine — is made in [risk management: the only free lunch](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine).

**"If I just win more often, the rest takes care of itself."** This is the win-rate fallacy in motivational clothing, and it pushes traders toward exactly the wrong adjustments. Chasing a higher win rate usually means taking profits earlier (smaller wins) and giving losers more room (larger losses) — both of which *lower* your EV while *raising* your win rate, the precise trade our bar-stool friend made. The drive to "be right more often" is an emotional need, not a financial strategy. The financial strategy is to maximize EV, which frequently means *accepting* a lower win rate in exchange for a far better payoff ratio.

**"EV only matters if I do the trade many times — this is a one-off."** This objection feels reasonable: EV is a long-run average, and if you genuinely place this exact bet only once, the average never gets a chance to assert itself. But it rests on a mistake about what "many trades" means. You are not going to make this *identical* trade a thousand times — but you are going to make *a thousand decisions*, each one a different EV calculation, over a career. The relevant ensemble is not "this trade repeated" but "every take-or-pass decision you ever make." If you consistently take positive-EV decisions and pass on negative-EV ones, the law of large numbers operates across your *portfolio of decisions over time*, even though no single decision is ever repeated. The trader who reasons "this one's a one-off, so I'll go with my gut" and abandons EV on the very trades that feel special has simply found a license to make negative-EV bets whenever emotion runs high — which is exactly when discipline matters most. There is one true exception, and it points back to sizing rather than away from EV: if a single outcome can *ruin you*, then the one-off nature genuinely matters, because you won't survive to play the next decision. But the answer to "this could ruin me" is never "ignore EV" — it is "take the positive-EV trade at a size small enough that the bad outcome can't end your career." EV decides direction; size handles the one-off catastrophe.

## How it plays out in real markets

The arithmetic is clean in a worked example; the lesson is sharper when you watch it decide real money. Three scenarios, each a version of a thing that happens constantly.

### An EV table flipping an "obvious" trade to a pass

A company you follow is the clear leader in a hot end-market. Everyone loves it, the chart is beautiful, and earnings are in two days. The "obvious" trade is to buy it into the print — great company, great momentum, what could go wrong. Before clicking buy on \$25,000 of stock, you build the table, and the act of building it does the work.

You are forced to confront what is *priced in.* The stock has already rallied 20% into the quarter; the bar is high, and a good number is expected. So your bull case isn't "the stock pops on good earnings" — it is "the stock pops on earnings *better than the already-high expectations*," which is a much narrower event. You estimate: bull (genuine upside surprise, stock +12%, +\$3,000) at 25%; base (good-but-priced-in, stock flat to −3%, −\$500) at 45%; bear (any disappointment against a high bar, stock −10%, −\$2,500) at 30%. Contributions: $0.25 \times \$3{,}000 = +\$750$; $0.45 \times (-\$500) = -\$225$; $0.30 \times (-\$2{,}500) = -\$750$. Gross EV $= \$750 - \$225 - \$750 = -\$225$. After costs, more negative still. The "obvious" trade is **negative-EV** — not because the company is bad, but because the good news is already in the price and the asymmetry runs against you (limited upside surprise, real downside on any stumble). The table flipped a trade that *felt* like a buy into a clear pass. This is the priced-in problem made quantitative, and it connects directly to [what's priced in: the question behind every trade](/blog/trading/analyst-edge/whats-priced-in-the-question-behind-every-trade) — EV is how "priced in" becomes a number you can act on.

The macro version of this is just as instructive. Through 2022, every monthly U.S. CPI print became a binary event for equities, and the "obvious" trade was to position for the direction you thought inflation would go. But the directional view was almost never the EV-positive trade, because the market had *already* moved hard into each print on the consensus forecast. On the November 10, 2022 release, headline CPI came in at 7.7% against a 7.9% consensus — a downside surprise of just two-tenths — and the S&P 500 rallied roughly 5.5% in a single session, one of its biggest up days of the year. A trader who built an EV table the day before would have seen the structure clearly: the market was positioned short and braced for a hot number, so the *surprise* (actual minus expected), not the level, was what carried the payoff, and the convexity was enormous to the downside-surprise side because positioning was so one-way. The level of inflation — still a punishing 7.7% — was almost irrelevant to the trade. The EV lived entirely in the gap between the print and what was priced, and in the asymmetry created by crowded positioning. The same number can be a buy or a sell depending only on what the price already assumed, which is why "I think inflation will fall" is a forecast, not a trade, until you run it through an EV table against the consensus.

### The asymmetric trade nobody wanted, in March 2020

Rewind to late February 2020. COVID is spreading, equity markets are still near highs, and volatility is cheap because the consensus is "it's a flu, markets shrug it off." A trader runs an EV table on a small position in deep-out-of-the-money index puts — the kind of trade that loses a little almost every day and is "obviously" a waste of premium. The structure is brutally asymmetric: total loss (puts expire worthless, the dominant outcome) at, say, 85%, costing −\$2,000; and a tail scenario (a real crash, puts go up 10×) at 15%, paying +\$20,000.

Run it: $0.85 \times (-\$2{,}000) + 0.15 \times (\$20{,}000) = -\$1{,}700 + \$3{,}000 = +\$1{,}300$. Positive EV, *despite* an 85% chance of losing money, *because* the tail payoff is enormous and the trader's probability of the tail (15%) was higher than the market's (priced near zero). When the crash came in March — the S&P fell roughly 34% in 23 trading days, the fastest bear market in history — the 15% scenario hit and the convex payoff dwarfed the months of small premium bleeds. The trade looked insane to a win-rate thinker and was a clear take to an EV thinker. The general lesson outlives the specific episode: when the consensus underprices a tail, a low-win-rate convex bet on that tail is exactly where positive EV hides.

### Costs and slippage eating a "good" systematic edge

A quant backtests a mean-reversion strategy. On paper, it has a small but consistent edge — an average gross EV of about +\$300 per trade on a \$25,000 notional. The equity curve in the backtest is smooth and lovely. Then the strategy goes live and bleeds. The culprit is not the signal; it is the cost line the backtest under-counted. On a real \$25,000 position the strategy crosses the bid-ask spread (a fast-mean-reversion signal trades into liquidity-taking, so it pays the spread), suffers slippage because it tries to fill quickly while the price is still moving, pays commission, and on overnight holds eats financing and borrow.

#### Worked example: a \$300 gross edge becoming a loss after costs

Take the strategy's \$300 gross EV per trade on the \$25,000 position and subtract the realistic frictions one at a time:

- Commission, round trip: **−\$18.**
- Bid-ask spread crossed twice (in and out), the largest cost for a fast signal: **−\$95.**
- Slippage from filling into a moving market in size: **−\$140.**
- Financing on the overnight leverage and platform fees: **−\$65.**
- Borrow cost on the short side and taxes on short-term gains, amortized: **−\$77.**

Total costs: $\$18 + \$95 + \$140 + \$65 + \$77 = \$395$. Net EV $= \$300 - \$395 = -\$95$ **per trade.** The strategy that looked like a +\$300 winner is a −\$95 *loser* once the real world charges its tolls. Run a thousand trades and the backtest says +\$300,000 while the live account says −\$95,000 — a \$395,000 gap that is *entirely* the cost line the table made visible. The intuition: on a thin-edge strategy, costs are not a footnote, they are the whole P&L, and an EV table that omits them is describing a market that does not exist.

![Costs and slippage turn a positive gross EV into a negative net EV](/imgs/blogs/expected-value-the-only-math-a-view-really-needs-5.png)

The waterfall shows the bleed step by step. The tall blue bar on the left is the +\$300 gross edge — the number the backtest celebrated. Each amber step down is a real cost: commission, spread, slippage, financing, borrow. By the time all five are subtracted, the bar has crossed below zero into the red −\$95 net EV on the right. The visual lesson is that no single cost is the villain — the spread alone, or slippage alone, leaves the trade marginally positive — but *summed*, the frictions exceed the entire gross edge. This is why edge-thin strategies live or die on execution, and why "it works in the backtest" is the beginning of the analysis, not the end. The high-frequency end of this — that your edge has to clear the cost of crossing to a market maker who is on the other side of every fill — is exactly the dynamic in [how an options market maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade).

### Two trades, ranked by EV instead of by win rate

The cleanest real-market lesson is comparative. You have capital for one of two trades on a \$25,000 position. **Trade A** is the "obvious" one — a high-probability mean-reversion setup with a 78% win rate, the kind of trade that *feels* safe and produces a satisfying string of green. **Trade B** is the "scary" one — a 35%-win-rate breakout/event trade that loses small most of the time and occasionally wins big. Every instinct says take A. The EV says otherwise.

#### Worked example: A looks safe, B is the trade

**Trade A** — 78% win rate. Win: +\$400 (you clip a small mean-reversion move). Loss: −\$2,500 (when mean reversion fails, the move keeps going and you eat a real loss). EV $= 0.78 \times \$400 + 0.22 \times (-\$2{,}500) = \$312 - \$550 = -\$238$. After ~\$37 of costs on the position, net EV ≈ **−\$275.** A *negative-EV* trade despite winning 78% of the time, because the rare 22% loss is more than six times the typical win.

**Trade B** — 35% win rate. Win: +\$8,000 (the breakout runs). Loss: −\$2,000 (it fails and you cut). EV $= 0.35 \times \$8{,}000 + 0.65 \times (-\$2{,}000) = \$2{,}800 - \$1{,}300 = +\$1{,}500$. After ~\$100 of costs on a wider-spread name, net EV ≈ **+\$1,900** when you account for the higher achievable payoff in the convex case. A strongly *positive-EV* trade despite winning only 35% of the time, because the wins are four times the losses and outweigh the lower hit rate.

Ranked by win rate: A (78%) beats B (35%) — take A. Ranked by EV: B (+\$1,900) beats A (−\$275) — take B, and don't touch A. The EV ranking is the correct one. The intuition: win rate told you which trade *feels* safer, and EV told you which one *makes money* — and they pointed in opposite directions, which is exactly when getting the math right matters most.

![Rank trades by expected value not by win rate](/imgs/blogs/expected-value-the-only-math-a-view-really-needs-6.png)

The dual-axis bar chart drives it home. The lavender bars are win rate: Trade A towers over Trade B, 78% to 35%. The right-hand bars are EV: Trade A is a red, *negative* −\$275 while Trade B is a green, *positive* +\$1,900. The two metrics rank the trades in *opposite order.* If you trade by the lavender bars, you systematically pick the comfortable, losing trade and skip the uncomfortable, winning one. If you trade by the EV bars, you take the money. Every working trader faces some version of this chart constantly — the "obvious safe" trade against the "scary good" one — and the discipline is to let the EV bars, not the win-rate bars, make the call.

## The playbook

Everything above collapses into one repeatable artifact and a short checklist you run on every view before it becomes a position. The artifact is the **EV table**, and producing it is the act of turning analysis into a decision.

**The five-step EV decision process:**

1. **List the scenarios.** Write out the bull, base, and bear cases your view implies (three to five, mutually exclusive, collectively exhaustive). For any strategy with a capped upside, *force in a fat-tail row* — the rare, brutal loss — or your table is lying. The cases must cover everything that can plausibly happen.

2. **Assign probabilities.** Attach a probability to each scenario, reflecting *your view* after the upstream work. The probabilities must sum to 100%. This is the hard part and the honest part — these numbers are your edge, quantified, and they are exactly as good as your calibration. If a number is just a feeling, label it as such and stress it.

3. **Price each payoff in dollars.** Convert each scenario to a dollar P&L on your actual position size, ideally via price targets times share or contract count, not gut feel. State the stake explicitly (a \$25,000 position, a \$2,500 risk) so the math is real money, not a vague feeling about size.

4. **Build the table and net out costs.** Multiply probability × payoff for each row, sum to gross EV, then subtract the *full* cost stack: commission, bid-ask spread (paid twice), slippage, financing, borrow, tax. The number that survives is $\text{EV}_{\text{net}}$.

5. **Decide and size.** If $\text{EV}_{\text{net}} > 0$, take the trade and *then* size it by the edge so no single outcome — and no plausible run of bad ones — can ruin you. If $\text{EV}_{\text{net}} \le 0$, pass, no matter how good the story felt. Win rate does not enter the decision at any step.

![Estimate probabilities and payoffs build the table net out costs then take and size or pass](/imgs/blogs/expected-value-the-only-math-a-view-really-needs-7.png)

The decision card above is the process as a single flow. Steps 1–3 on the left feed the EV table in the center; the table nets out costs and hits the gate; the gate routes to PASS (no edge, garbage in) or to TAKE-and-size (positive edge, survive the tails). Tape it to your monitor. The discipline it enforces is the entire point: it makes you write the bear case down, assign it an honest probability, price the real dollar loss, and let *one number* — not the strength of your conviction, not the beauty of the chart, not the win rate of the setup — decide whether the trade lives.

A few practitioner notes to keep the table honest:

- **The bear row is the most valuable row.** Most bad trades die the moment you are forced to write down the actual dollar loss and assign it an honest probability. If you find yourself flinching from filling in the bear case, that flinch is information.
- **Recompute at every decision point.** EV is not a one-time entry gate. The question "should I hold?" is just "is the EV of holding from here positive versus exiting now?" — and your entry price is irrelevant to it. Sunk cost is past; EV is forward-looking.
- **Stress the probabilities, not just the payoffs.** A trade that is only positive-EV under your most optimistic probabilities is not really positive-EV. Recompute with pessimistic inputs; if the decision flips easily, the edge is thin or imaginary.
- **EV decides whether; sizing decides how much.** The two are a matched pair. A positive EV is a green light, not a green light to bet the account. Survival is the precondition for the long-run average to ever arrive.

You do not need stochastic calculus to trade well. You need to be able to draw this table, fill it in honestly, and obey the number it produces — every time, including the times the number tells you to pass on the trade you most want to take. That is expected value, and it is the only math a view really needs to become a decision.

## Further reading & cross-links

**Within this series — The Analyst's Edge:**

- [What's priced in: the question behind every trade](/blog/trading/analyst-edge/whats-priced-in-the-question-behind-every-trade) — the upstream skill that tells you whether your scenarios are surprises or already in the price; EV is how "priced in" becomes a number.
- [Thinking in probabilities, not predictions](/blog/trading/analyst-edge/thinking-in-probabilities-not-predictions) — where the probabilities in your EV table come from, and how to keep them calibrated.
- [Decision trees for event-driven views](/blog/trading/analyst-edge/decision-trees-for-event-driven-views) — when outcomes branch in sequence, the EV table grows into a tree.
- [From conviction to size: the bet-sizing bridge](/blog/trading/analyst-edge/from-conviction-to-size-the-bet-sizing-bridge) — the second filter that EV hands off to: how much to bet given the edge.
- [Asymmetry and the art of the high-conviction bet](/blog/trading/analyst-edge/asymmetry-and-the-art-of-the-high-conviction-bet) — going deep on why convex, low-win-rate trades are where the best EV often lives.
- [Base, bull and bear: building three scenarios](/blog/trading/analyst-edge/base-bull-and-bear-building-three-scenarios) — how to construct the scenario rows that feed the EV table.

**Going deeper on the mechanisms (other series):**

- [Expectancy: why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) — the same win-rate-versus-EV lesson from the systematic-trading angle.
- [Position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) — the formal framework for turning an edge into a bet size.
- [Risk management: the only free lunch — survival as a compounding engine](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine) — why surviving long enough for the EV to arrive is itself the strategy.
- [Fat tails and the normal-distribution trap](/blog/trading/risk-management/fat-tails-and-the-normal-distribution-trap) — why your EV table must include the rare, brutal row.

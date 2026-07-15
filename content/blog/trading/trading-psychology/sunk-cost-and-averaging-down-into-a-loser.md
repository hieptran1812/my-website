---
title: "Sunk Cost and Averaging Down: Throwing Good Money After Bad"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "A practitioner's deep dive into why adding to a losing trade wrecks accounts — the sunk-cost fallacy behind it, the clean test that separates a planned scale-in from a reactive martingale, the arithmetic that fools you, and the drill that stops it."
tags: ["sunk-cost-fallacy", "averaging-down", "martingale", "trading-psychology", "risk-management", "behavioral-finance", "position-sizing", "loss-aversion"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — Adding to a loser is only a disaster when it is *reactive*. The same act, done as a *planned, budgeted* scale-in, can be sound. The entire skill is telling the two apart before you click.
>
> - **The sunk-cost fallacy** makes you honor money already spent: in Arkes & Blumer's 1985 study, 54% of people chose a trip they expected to enjoy *less*, purely because it had cost more.
> - **The test**: a planned scale-in has an intact thesis, a total size budgeted *before* entry, and an invalidation price. A reactive double-down has a broken thesis, no size cap, and is driven by the pain of being down.
> - **The arithmetic trap**: averaging down lowers your *average price* — which feels like progress — while your *dollars at risk* and your loss balloon far faster.
> - **The martingale has ruin baked in**: doubling to "win it back" needs ${2^n - 1}$ units after $n$ losses, so a finite bankroll meets an unbounded losing run — and on a fair coin the whole scheme nets exactly zero.
> - **The one drill**: before every add, ask "would I OPEN this position, fresh, at this size, today?" If the answer is no, don't add — and cut the moment your invalidation is hit.

You have almost certainly done this, and it probably felt responsible at the time.

A stock you own is down 20%. The loss sits there in red, and it stings. Then a thought arrives, dressed up as discipline: *it's even cheaper now — I loved it at $100, I should love it at $80.* You buy more. Your average cost drops, the position needs a smaller bounce to break even, and for a moment you feel clever and brave instead of wrong. Then it goes to $60, and the same voice says the same thing, only louder. By $40 you are not investing anymore. You are trying to win back the money you have already lost, and you are betting the account to do it.

This is the oldest way to blow up in markets, and it has a specific engine: the *sunk-cost fallacy* — the human habit of letting money you can never get back drive a decision that should only look forward. The dangerous part is that the healthy version of adding to a position looks almost identical from the outside. A disciplined trader scaling into a well-planned idea and a panicking trader martingaling into a falling knife both "buy more as it drops." The diagram below is the mental model this whole article tours: one underwater position, two roads out of it, and only one of them survives.

![A tree diagram: one underwater position of −$2,000 forks into a PLANNED scale-in (thesis intact, size budgeted, invalidation set) leading to a controlled outcome, and a REACTIVE double-down (thesis broken, no cap) leading to ruin risk.](/imgs/blogs/sunk-cost-and-averaging-down-into-a-loser-1.webp)

The left branch is a business decision made in advance. The right branch is an emotional reaction made in the moment. They can start from the exact same price and the exact same red number on the screen. What separates them is not the *action* — both add — but everything that was decided *before* the action: whether the reason to own it still holds, whether the size was budgeted, and whether there is a line at which you admit you are wrong. The rest of this piece is about seeing that fork clearly enough to take the left road every time.

This is educational, not financial advice. The numbers inside the *worked examples* are round and hypothetical so you can do them in your head; every number attributed to a *study, a person, or a market event* is real and sourced at the end.

## Foundations: how the sunk-cost fallacy actually works

You do not need any finance background for this section. You only need to notice a bias you already have, because it runs your everyday life long before it touches your trading account.

### What a "sunk cost" is, and why it should be invisible

A **sunk cost** is money (or time, or effort) you have already spent and cannot get back, no matter what you do next. The ticket you already bought. The five years you already put into a career. The $2,000 you are already down on a trade. It is *sunk* because no future decision can un-spend it.

Here is the uncomfortable rule that all of economics agrees on and almost no human obeys: **a sunk cost should have zero weight in a forward-looking decision.** When you decide what to do *next*, the only things that matter are the costs and benefits that lie *ahead* of you. What you paid to get here is gone in every scenario, so it cannot favor one choice over another. If you would not buy more of a stock at $80 as a fresh idea today, the fact that you paid $100 for your existing shares is not a reason to buy more — it is just a wound you are trying to soothe.

That is easy to say and brutally hard to feel, because your brain is doing something else entirely. Instead of measuring the trade forward, it anchors on the price you paid and treats "getting back to breakeven" as the goal. Three separate forces do the pulling, and it helps to see them named.

![A graph diagram: a neutral 'sunk cost −$2,000' box plus three force boxes (loss aversion, hurts about 2.25x; commitment and consistency; waste-avoidance) all feed into a central decision node 'buy MORE here, fresh, today?' which leads to a forward loss of −$6,000.](/imgs/blogs/sunk-cost-and-averaging-down-into-a-loser-2.webp)

The picture is the mechanism: the −$2,000 in the top-left is *gone*, yet three arrows drag your decision back toward it. Let me name each force, because you will feel all three the next time you are underwater.

- **Loss aversion.** A loss hurts far more than an equal gain feels good — by the best measurement, about $\lambda \approx 2.25$ times as much (Tversky & Kahneman, 1992). Realizing a loss by selling *closes the wound* and makes the pain permanent, so you will do almost anything — including throwing more money at it — to keep the loss "on paper" and reversible. This is the deep root of the whole family of mistakes; it gets its own full treatment in [loss aversion and the disposition effect](/blog/trading/trading-psychology/loss-aversion-and-the-disposition-effect).
- **Commitment and consistency.** Once you have publicly or privately committed to a view — "this is a great company," "the trade is working" — your brain fights to stay consistent with it. Adding more *is* consistency in action: it says "I still believe," and it feels better than the self-contradiction of selling.
- **Waste-avoidance.** Humans are trained from childhood not to waste. Arkes and Blumer, whose 1985 paper *The Psychology of Sunk Cost* named the effect, argued the whole bias is powered by "the desire not to appear wasteful." Selling a loser feels like *admitting the money was wasted*; holding — or adding — lets you pretend the story is not over.

### The evidence that this is real (and not just your problem)

The sunk-cost fallacy is one of the most reliably reproduced findings in decision science, and the classic demonstrations have nothing to do with markets — which is exactly why they are so damning.

In one of Arkes and Blumer's experiments, people were asked to suppose they had bought a $100 ticket for a ski weekend in Michigan, then later bought a $50 ticket for a ski weekend in Wisconsin they expected to enjoy *more*. When the trips turned out to fall on the same weekend and neither ticket could be refunded, which do you take? Rationally, you take the trip you will enjoy more — the money is gone either way. But **54% chose the more expensive Michigan trip they expected to enjoy less**, purely because it cost more (Arkes & Blumer, 1985). The extra $50 of sunk cost literally bought them a worse weekend.

In a companion field study, the same researchers sold theater season tickets at full price or with a random discount. Those who paid full price attended noticeably more plays early in the season — roughly four versus a little over three in the first half — even though every ticket holder had the identical right to attend. The bigger the sunk cost, the harder people worked to "use it up," even against their own convenience.

Scale that instinct up to a government and you get the **Concorde fallacy**. Britain and France kept pouring money into the supersonic Concorde jet for years after it was clear the plane would never be commercially profitable, precisely *because* so much had already been spent — the sunk cost became the argument for continuing (Arkes & Ayton, 1999). The term is now shorthand for exactly the trade you are tempted to make: honoring the past investment instead of judging the future one. Concorde flew its first commercial route in 1976 and was finally retired in 2003, decades of losses later.

### Escalation of commitment: the trap is worse when it was your idea

There is a crucial wrinkle that turns the sunk-cost fallacy from a minor accounting error into a career-ender, and it has a name: **escalation of commitment**. The psychologist Barry Staw demonstrated it in a 1976 study with the memorable title *Knee-Deep in the Big Muddy*. He gave business students a role-play in which an earlier investment decision had turned out badly, then asked how much *more* they would commit. The finding that matters for you: people committed the **most** additional money to a failing course of action precisely when *they* had been personally responsible for the original decision. Owning the mistake made them throw more at it, not less.

That is the difference between an abstract bias and the thing that empties your account. It is not just that the loss hurts; it is that *your ego is now attached to being right about the trade you chose*. Cutting it does not only realize the loss — it confesses that your judgment was wrong, out loud, to yourself. Adding more is a way of insisting you were right all along, and the more public or costly the original call, the harder you double down to defend it. Every disaster in the case studies below has this signature: not a stranger's bad position, but the trader's *own* prized idea, defended past the point of ruin because admitting the error felt worse than the loss.

One clean way to keep this straight is to separate a **sunk cost** (money already spent, which should be ignored) from an **opportunity cost** (what else that same capital could earn from here, which is the thing you *should* weigh). Every dollar you add to a loser to defend your average is a dollar that is not compounding in your best current idea. The sunk cost is a ghost; the opportunity cost is real, and it is the one you keep ignoring.

#### Worked example: the $2,000 that isn't there

Let me make the "sunk costs are invisible" rule concrete with numbers you can hold in your head.

Suppose you bought 100 shares of a stock at $100, so you put in $10,000. It falls to $80. You are down $2,000. Now the only real question in front of you is: *at $80, is this a buy?*

Watch what happens when we strip the history out. Two traders — call them A and B — are both looking at the same stock at $80 right now.

- Trader A owns 100 shares she bought at $100 (down $2,000).
- Trader B owns nothing; he just has $8,000 of cash and is deciding whether to buy 100 shares at $80.

If buying 100 shares at $80 is a good forward bet, it is good for *both* of them — and A should add exactly as much as B would buy fresh. If it is a bad forward bet, it is bad for both — and A should *not* add just because she is down, any more than B should buy something he does not want. Her $2,000 loss is real, but it is *identical* in every future she can choose, so it cannot tilt the decision. The correct add size is whatever a clear-eyed person with no position would buy here, and not one share more.

**The takeaway:** the price you paid is a fact about the past; the only honest input to the next trade is what you would do if you were flat right now.

## The one distinction that matters: planned scale-in vs reactive double-down

Everything in this article hinges on a single fork, because "don't add to losers" is bad advice — plenty of great trades are *built* by adding as price improves against your entry. The problem is never the act of adding. It is *why* and *how much*.

![A before/after comparison: the left column (REACTIVE double-down, the trap) lists trigger = the pain of being down, thesis broken, size = whatever cuts the average with no cap, exit = none; the right column (PLANNED scale-in, the discipline) lists trigger = a pre-set price rung, thesis intact and re-verified, size = budgeted total with a hard cap, exit = invalidation set.](/imgs/blogs/sunk-cost-and-averaging-down-into-a-loser-3.webp)

Read the two columns side by side and the difference stops being fuzzy. A **planned scale-in** was written *before* you entered: you decided in advance that you would add at specific lower prices, you sized the *entire* campaign up front so the adds were part of the plan rather than a surprise, and you set a price at which the thesis is wrong and you are out. A **reactive double-down** is written *by the pain* in the moment: the trigger is the red number, the thesis has quietly broken or you have stopped checking it, the size is "whatever it takes to get my average down," and there is no exit because admitting the exit means admitting the loss.

Here is the same fork as a checklist you can actually use at the screen.

| Question | Planned scale-in | Reactive double-down |
|---|---|---|
| When was the add decided? | Before entry, in writing | Right now, in reaction to the loss |
| Is the original thesis intact? | Yes, and re-verified | Broken, or no longer being checked |
| Is total size capped in advance? | Yes — a hard budget | No — as big as the pain demands |
| Is there an invalidation price? | Yes — sell if it's hit | No — "it has to come back" |
| What's the emotion driving it? | A plan executing | The wish to erase a loss |

There is a clean, one-line field test that collapses all five rows into a single question, and it is the most valuable sentence in this article: **"Would I OPEN this position, fresh, at this size, at this price, today — if I had no position and no loss?"** If yes, your add is a scale-in and the sunk cost is irrelevant. If no, your add is a double-down and you are honoring a wound. Say it out loud before every add.

> A scale-in is a decision you already made. A double-down is a decision the loss is making for you.

#### Worked example: two traders, same falling price, opposite outcomes

Suppose two traders both buy 100 shares at $100 ($10,000 in), and the stock falls to $80. Both are down $2,000. Now the roads diverge.

**Trader P (planned).** Before entering, P wrote a plan: "Core idea is intact down to $70; I'll add one more 100-share rung at $80 and one at $72, total position capped at 300 shares / roughly $25,000, and if it closes below $70 the thesis is dead and I'm flat." At $80, P adds 100 shares (thesis still valid), now owns 200 at an average of $90, with $18,000 committed. If it keeps falling and closes at $69, P sells all 200 for about $13,800 and books a loss of roughly $4,200 — painful, bounded, survivable, and exactly the size P signed up for.

**Trader R (reactive).** R had no plan. At $80 the loss hurts, so R buys 100 more to "average down." At $60 it hurts more, so R buys 200. At $40, terrified and hoping to be made whole on the next bounce, R buys 400. R now owns 800 shares, $46,000 committed, an average cost of $57.50, and an unrealized loss of $14,000 — seven times the loss R started with. There was never an exit, so at $30 R is down about $22,000 and finally capitulates at the worst possible price.

Same stock, same entry, same $2,000 starting loss. P risked what P chose to risk; R let the loss choose, and it chose ruin. **The lesson:** the plan caps the damage in advance, and the absence of a plan lets the damage cap *you*.

## Averaging down: the arithmetic that fools you

The reason reactive averaging down feels so reasonable is that it improves the one number your eye is drawn to — the average cost — while quietly wrecking the numbers that actually determine your survival. Let me show you the trap in a table of real arithmetic.

![A grid table with five columns (Add @ price, Shares held, Avg cost, Cash committed, Unrealized P&L) and four rows: $100/100/$100/$10,000/$0; then $80/200/$90/$18,000/−$2,000; then $60/400/$75/$30,000/−$6,000; then $40/800/$57.50/$46,000/−$14,000. The average-cost column falls while cash committed and the loss grow far faster.](/imgs/blogs/sunk-cost-and-averaging-down-into-a-loser-4.webp)

#### Worked example: the average falls, the dollars at risk explode

Follow the rows. You start with 100 shares at $100 — $10,000 committed, no loss yet. Then you martingale the adds, doubling each time:

- Price hits **$80**: you add 100 shares. Now 200 shares, average cost $90, **$18,000** committed, down **$2,000**.
- Price hits **$60**: you add 200 shares. Now 400 shares, average cost $75, **$30,000** committed, down **$6,000**.
- Price hits **$40**: you add 400 shares. Now 800 shares, average cost **$57.50**, **$46,000** committed, down **$14,000**.

Look at what your eye is being sold versus what is actually happening. The **average cost** slides from $100 to $90 to $75 to $57.50 — a lovely, reassuring descent that feels like you are "buying the bargain" and lowering the bar to break even. But the **cash committed** has ballooned from $10,000 to $46,000, and the **unrealized loss** has gone from $0 to $14,000 — seven times worse — even though the stock only fell from $100 to $40. You did not reduce your risk by averaging down. You *multiplied* it, and the falling average cost was the anesthetic that let you do it.

There is a second, meaner effect hiding in the same table. Because your position keeps growing, each further drop hurts more in dollars than the last, even though the *percentage* moves are getting smaller. From $100 to $80 (a 20% drop) you lost $2,000 on 100 shares. From $40 to $32 — also a 20% drop — you would lose $6,400 on 800 shares. **The takeaway:** averaging down turns a shrinking price into a growing dollar loss, because you are holding more and more of the thing that is falling.

This is the exact opposite of the discipline good traders describe as "add to winners, not losers." When you add to a *winning* position that is moving your way, your growing size rides a rising price. When you add to a *losing* position, your growing size rides a falling one — you concentrate the most capital into the trade precisely as the market is telling you, in the only language it has, that you are wrong.

#### Worked example: the same idea, as a capped anti-martingale ladder

The fix is not "never add lower" — it is to invert the size schedule and cap the total *before* you start. Suppose the same idea, but this time you plan it. You decide up front: total position capped at 400 shares, and I will add *smaller* each time, not larger. Your written ladder is 200 shares at $100, then 100 more at $85, then 100 more at $70 — and a hard rule that a close below $65 kills the thesis and I sell everything.

Run it. You buy 200 at $100 ($20,000). It falls to $85; you add 100 ($8,500) — now 300 shares, average $95, $28,500 committed. It falls to $70; you add your last 100 ($7,000) — now 400 shares, average $88.75, $35,500 committed, and *you are done adding no matter what happens next*. If it then closes at $64, you sell all 400 for $25,600 and take a loss of about $9,900 — real, but a number you chose and can name in advance. Compare that to the doubling ladder, where the same $100-to-$40 path put $46,000 at risk and a $14,000-and-climbing loss with no floor. Same instinct to buy weakness; one is a budgeted campaign, the other is a runaway. **The takeaway:** a scale-in is defined by its *cap and its floor*, not by the fact that it buys lower — decreasing size and a written invalidation are what make "adding down" survivable.

## The martingale delusion: why "double down to get it back" has ruin baked in

The purest, most dangerous form of reactive averaging down is the **martingale** — the old gambler's system of doubling your bet after every loss so that the eventual win recovers everything plus one unit. It is seductive because the logic seems airtight: as long as you win *eventually*, you come out ahead. The flaw is not in the logic. It is in the word "eventually," which quietly assumes you have an infinite bankroll and infinite time. You have neither, and the gap between "eventually" and "before I run out of money" is where accounts die.

![A bar chart titled 'Throwing good money after bad: the martingale wall'. Bars for cumulative dollars at risk after each consecutive loss rise $1, $3, $7, $15, $31, $63 (blue, under a dashed red 'Bankroll wall = $100' line) and then $127 (red) which smashes through the wall on the seventh loss. Annotations note the bet doubles each loss and seven losses in a row is about 1 in 128.](/imgs/blogs/sunk-cost-and-averaging-down-into-a-loser-5.webp)

The picture shows the whole tragedy on one axis. Each loss forces the next bet to double, so the *cumulative* amount you have to risk to claw back a single unit after $n$ straight losses is ${2^n - 1}$: one, three, seven, fifteen, thirty-one, sixty-three, one hundred twenty-seven. The bars look harmless for a while and then go vertical, which is exactly how it feels: manageable, manageable, manageable, *gone*.

#### Worked example: the martingale ruin sequence

Suppose you have a $100 bankroll and you run a martingale with a $1 base bet on a coin-flip game. You bet $1; if you lose, you bet $2; then $4, $8, $16, $32. You can fund six doublings, because $1 + $2 + $4 + $8 + $16 + $32 = $63, which fits inside $100. Lose all six in a row and you are down $63, with $37 left — and the system now demands a *seventh* bet of $64, which you cannot place. To even attempt recovery you would need $127 committed against a $100 account. That is the wall in the picture.

How likely is that? On a fair coin, six losses in a row is a 1-in-64 event (about 1.6%); seven in a row is 1-in-128 (about 0.8%). Tiny — which is the whole trap. Ninety-eight-plus percent of the time, the martingale hands you a small, satisfying win and whispers that it works. But you do not run it once. You run it over and over, and a 1-in-64 catastrophe that you face hundreds of times is not a tail risk anymore; it is a *scheduled* event. When it arrives, it does not just dent you — it takes everything the small wins gave you and the rest of your stake besides.

And here is the part that surprises people: even in the best possible case — a perfectly fair coin with no house edge — the martingale does not make money. Its expected value is exactly zero. Work it out: with probability 63/64 you win $1, and with probability 1/64 you lose $63, so the expectation is $\frac{63}{64}(+\$1) + \frac{1}{64}(-\$63) = 0$. All the martingale does is *reshape* your outcomes — it trades a lot of small wins for one enormous loss of the same total size. It converts a boring, zero-sum coin-flip into a slot machine that pays out reliably and then, once, keeps the whole jar. **The lesson:** doubling to recover does not change your expectation, it just hides your risk in the tail where you cannot see it until it detonates.

### Why markets are worse than the coin flip

Everything above assumed a *fair coin* — independent flips, 50/50 odds, no drift. Real losing trades are worse than that on every count, which is why market martingales blow up faster than casino ones.

- **The odds are not 50/50 — they are against you.** A position falling on a broken thesis is not a coin at equilibrium; it is a thing with *negative drift*. The market's persistent verdict is that it is worth less, and averaging down bets against that verdict repeatedly.
- **Losing runs are longer and correlated.** Coin flips are independent, so long streaks are rare. Prices *trend* — a stock going down tends to keep going down, and a "streak" of losses in a downtrend is not seven independent 1-in-2 events; it is one sustained move. The rare tail in the coin model is the *common* case in a bear market.
- **Gaps skip the ladder.** Casinos let you place every doubled bet. Markets gap — a stock can open 40% lower on an earnings miss or a fraud headline, blowing through every rung of your ladder at once and handing you the maximum loss with no chance to "stop after the next one."
- **Leverage adds a margin call.** With borrowed money, you do not even get to choose when you are done. The broker liquidates you at the bottom, on its schedule, converting a paper loss into a realized one at the worst tick.

The gap point deserves a number, because it is the one that kills the martingale fastest. Suppose you have been averaging down a $50 stock, and after several adds you hold 1,000 shares at an average of $44, planning to add 1,000 more if it dips to $38 and "stop there." Overnight the company pre-announces a bad quarter and the stock opens at $22 — it never traded at $38, $34, or $30 for you to add into; it *jumped* the whole ladder. Your existing 1,000 shares are now worth $22,000 against $44,000 of cost — a $22,000 loss in one print — and the "disciplined" add you promised yourself would have simply put more money into the same hole a second before it deepened. The martingale assumes a continuous price you can average into calmly; markets deliver discontinuous prices that hand you the maximum loss with no rung to stop on. **The takeaway:** the doubling plan quietly depends on the market never gapping, and markets gap exactly when your thesis is breaking.

Put those together and the martingale in markets is not a fair coin with a hidden tail — it is a *biased* coin with a fatter, more frequent tail and a broker standing by to end the game for you. That is why the single most reliable way to turn a bad trade into a fatal one is to keep doubling into it.

## What it looks like at the screen

Biases are easier to catch when you know their *tells* — the specific, physical things you do when the reactive double-down has you. If you have traded for any length of time, this passage will read like a transcript of a bad afternoon.

You are refreshing the position more often than the thesis could possibly have changed — every few minutes, then every few seconds — because you are not monitoring an investment, you are watching a wound. Your internal monologue has shifted from the company or the setup to the *breakeven price*: you catch yourself calculating "if it just gets back to $92 I'm out," a number that has nothing to do with what the thing is worth and everything to do with what you paid. You open the order ticket to add, and you notice — if you are honest — that the add is *bigger* than your original position, and you size it not from a risk number but from "how much would it take to get my average to a level that doesn't hurt."

You stop reading news that might be bad and start hunting for anyone, anywhere, who agrees with you — a classic slide into [confirmation and motivated reasoning](/blog/trading/trading-psychology/confirmation-bias-and-motivated-reasoning). You move your mental stop lower "just this once," then again. You find yourself *not* writing the add in your journal, or logging it vaguely, because part of you already knows it would not survive being written down honestly. And the feeling is not the calm of executing a plan — it is a hot mix of hope and dread, the specific physiology of gambling to get even. When you feel that cocktail, you are not scaling in. You are chasing, and the market does not owe you your money back.

The single most useful habit here is to treat those tells as a *fire alarm*, not a footnote. The instant you notice yourself computing a breakeven price, sizing an add from pain, or hiding a trade from your own journal, stop touching the mouse. The tell is the signal; the drill in the last section is what you do about it.

## Common misconceptions

**"Averaging down is what value investors do — Buffett buys more when it drops."** Great investors *do* add to declining positions — but only as *planned scale-ins* into a thesis they have re-verified, sized inside a total budget, with the humility to be wrong. The difference between Buffett buying more of a business whose fundamentals are intact and a panicking trader martingaling a chart is the entire subject of this article. Copying the *action* without the *thesis, the size discipline, and the invalidation* is cargo-cult investing.

**"A lower average cost means I'm taking less risk."** The average cost is the most misleading number on your screen. Lowering it does not reduce your risk; as the arithmetic showed, your *dollars at risk* and your *dollar loss* grow far faster than your average falls. Risk lives in the size of the position and the dollars exposed, not in a per-share average that only feels better.

**"It's not a loss until I sell."** It is a loss the moment the price moves against you; selling only *records* it. Treating unrealized losses as somehow less real is precisely the loss-aversion trick that keeps you holding and adding. The market marks your position to the current price whether you look or not.

**"If I don't add, I'm wasting the conviction that got me in."** The conviction that got you in was a forecast, and forecasts get updated by new information — like the price falling and the thesis weakening. Refusing to update because you already committed is the commitment-and-consistency force doing your thinking for you. Deciding what *would* change your mind, in advance, is the antidote; it is worth reading [what would change my mind: defining invalidation upfront](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront) as a companion to this piece.

**"The odds of a long losing streak are tiny, so I'm safe."** They are tiny *per attempt* and near-certain *over many attempts*, and in trending markets they are not even tiny per attempt. A strategy whose survival depends on never hitting a bad streak will hit one, because you will run it enough times to guarantee it.

**"I'll just add smaller amounts so it's not a martingale."** Adding *smaller* each time (an anti-martingale ladder) is genuinely safer than doubling, and if it is planned and capped, it can be a legitimate scale-in. But if it is still *reactive* — triggered by pain, with a broken thesis and no invalidation — you have only slowed the bleeding, not stopped it. The size schedule matters, but the trigger and the exit matter more.

**"Isn't this just dollar-cost averaging, which everyone recommends?"** No — and the difference is the whole point. Dollar-cost averaging means investing a *fixed amount on a fixed schedule* into a *diversified* holding (usually a broad index), regardless of price, as a way to remove timing and emotion. It has a plan, a cap (your regular contribution), and no single-name thesis that can "break." Reactive averaging down is the opposite on every axis: it is triggered by the pain of a *specific* position falling, it is unbudgeted, and it concentrates more capital into one thing that may be failing. Same words, "buying as it drops," entirely different act.

**"Big institutions with real risk systems can't fall for something this basic."** They fall for it hardest, because escalation of commitment scales with how much prestige is attached to the original call. Barings, Legg Mason, and JPMorgan's Chief Investment Office were run by celebrated professionals with sophisticated risk tools — and each turned a losing position into a catastrophe by adding to it rather than admitting it was wrong. The bias is not a knowledge gap you can educate away; it is an ego reflex, and a multibillion-dollar desk feels it exactly like a retail trader down $600 does.

## How it shows up in real markets

The abstract danger becomes visceral when you watch it end real careers. These are not obscure cautionary tales; they are among the most studied blowups in finance, and every one of them is, at its core, a sunk-cost double-down.

### 1. Nick Leeson and Barings Bank — the account called 88888

Nick Leeson was a young trader running the futures desk for Barings, Britain's oldest merchant bank, in Singapore in the early 1990s. What began as a small hidden loss — an error account, numbered 88888, originally opened to bury a junior colleague's mistake of around £20,000 — became a machine for doubling down. As his bets on the Japanese Nikkei index went wrong, Leeson did not cut them; he hid the losses in 88888 and *increased* the positions, betting bigger to win it all back, the martingale logic writ enormous.

Then the world gapped through his ladder. On 17 January 1995, the Kobe earthquake sent the Nikkei tumbling, and Leeson's giant long position — built by averaging into weakness — cratered. He doubled again, trying to push the index back up single-handedly, and failed. By the time it unraveled, the losses reached roughly **£827 million (about US$1.4 billion)** — more than the entire bank's capital. Barings, founded in 1762, collapsed at the end of February 1995 and was sold to the Dutch bank ING for a symbolic **£1**. Leeson was sentenced to six and a half years in a Singapore prison. Every dollar of that loss was, mechanically, the same trade you make when you buy more of a $40 stock because you are down from $100: honoring the money already lost by risking more to get it back.

### 2. Bill Miller and the Legg Mason Value Trust

Not every averaging-down catastrophe comes from a rogue. Bill Miller was, for a while, the most celebrated fund manager in America: his Legg Mason Value Trust beat the S&P 500 for **fifteen straight calendar years, 1991 through 2005** — a streak so improbable it made him a legend. His stated philosophy was that a lower price on a stock he liked simply made it a better bargain, so he bought more as prices fell. For fifteen years, that discipline of buying weakness looked like genius.

![A timeline of Bill Miller's Legg Mason Value Trust: 1991–2005, beats the S&P 500 fifteen years running; 2007, adds to Bear Stearns, Countrywide, Freddie Mac as they fall; March 2008, Bear Stearns held near $30 and rescued at $2 a share days later; September 2008, Freddie Mac wiped out in the government takeover; December 2008, fund about −55% on the year with assets falling from $16.5 billion to $4.3 billion.](/imgs/blogs/sunk-cost-and-averaging-down-into-a-loser-6.webp)

Then came the financial crisis, and the same instinct that built the streak destroyed it. As financial stocks fell in 2007 and 2008, Miller added to them — Bear Stearns, Countrywide, Freddie Mac, and others — treating each new low as a bigger bargain. But these were not temporarily cheap; their businesses were failing. Miller reportedly held Bear Stearns near $30 a share just before it was rescued in a fire sale at $2 in March 2008, and stayed with Freddie Mac until the government wiped out its shareholders in the September 2008 conservatorship. The Value Trust fell roughly **55% in 2008**, far worse than the S&P 500's own brutal decline of about 37% that year, and assets in the fund collapsed from around **$16.5 billion to $4.3 billion** through losses and redemptions. The lesson is the sharp one: averaging down works right up until the thesis is actually broken, and the discipline that makes you great in normal markets is the exact thing that ruins you when a price is falling for a real reason. Avoiding it requires an invalidation you will actually honor — see [the cognitive bias map for traders](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders) for how this bias interlocks with the others.

### 3. JPMorgan's "London Whale" — a hedge that became a doubling machine

In 2012, a desk inside JPMorgan's Chief Investment Office — nominally there to *hedge* the bank's risk — built an enormous position in synthetic credit derivatives. The trader at the center, Bruno Iksil, earned the nickname "the London Whale" for the sheer size of the bets. When the position began losing money in the spring of 2012, the desk did not cut it. It did what the whole of this article warns against: it *added*, increasing the size to try to overwhelm the market and force the trade back to breakeven. A flawed internal risk model even emboldened the escalation by understating how much was at stake.

The market did not cooperate. By the time JPMorgan unwound the position, the loss was roughly **$6.2 billion** — from a unit whose entire job was to *reduce* risk (2012 JPMorgan Chase trading loss). The head of the CIO, Ina Drew, stepped down, and the bank later paid around **$920 million** in regulatory fines. What makes the Whale so instructive is that these were not naive retail traders or a lone rogue; they were sophisticated professionals at the largest US bank, and the mechanism that undid them was the same primitive one: a losing position, defended by adding to it, because backing down meant admitting the original call was wrong. Escalation of commitment does not care how smart you are.

### 4. The retail "grid bot" and the revenge trade

You do not need a billion dollars to run this play; the retail version is everywhere. In FX and crypto, automated "grid" and "martingale" bots are sold as passive-income machines: they add to a losing position at fixed intervals and close the whole basket for a small profit on any bounce. For months, the equity curve is a smooth line up and to the right — exactly the martingale's many-small-wins signature — and buyers conclude they have found free money. Then a trend runs the wrong way without a bounce, the grid keeps adding into it, the account hits a margin call at the worst point, and months of gains vanish in a day. The pattern is identical to Leeson's and Miller's, just automated and smaller: a finite account meets an unbounded losing run.

The individual version is the **revenge trade** — you take a loss, feel the sting, and immediately put on a bigger position in the same name to "get it back this session." It is a martingale with a size of one step and a bankroll of your composure, and it ends the same way: the pain sizes the trade, there is no plan, and one bad move takes back everything the small recoveries earned. If you have ever traded angry, you have run a martingale on your own account.

## The drill: a protocol you can run in ten seconds

Insight does not survive contact with a red P&L; only a *pre-committed rule* does. So the fix is not "be more disciplined" — it is a small, mechanical gate that every proposed add must pass before it becomes an order. Run it out loud. If any answer is *no*, you do not add; you reduce or you cut.

![A decision-flow graph: a proposed add to a losing position runs through four questions in sequence — (1) would I open this fresh at this size today? (2) is the original thesis still intact and re-verified? (3) is this rung inside my pre-set total-size budget? (4) is price still on the right side of invalidation? — where all four Yes leads to 'add one planned rung' and any single No routes to 'do not add, consider cutting'.](/imgs/blogs/sunk-cost-and-averaging-down-into-a-loser-7.webp)

The gauntlet has four questions, in order, and a single *no* anywhere sends you to the exit rather than to more size.

1. **Would I open this position, fresh, at this size, at this price, today?** The one-line field test from earlier. If a flat trader with cash would not buy here, you are honoring a sunk cost, not making a trade.
2. **Is the original thesis still intact, and have I just re-verified it?** Not "do I still hope," but "has the reason I bought — the earnings power, the setup, the catalyst — actually survived the new information the falling price is carrying?" If you have to squint to keep the thesis alive, it is dead.
3. **Is this rung inside the total-size budget I set before entry?** Every position gets a *maximum* size decided when you are calm, before the first share. If the add would push you past that cap, the answer is no by definition — the cap exists precisely to overrule the version of you that is in pain.
4. **Is price still on the right side of my invalidation?** You drew a line in advance where the thesis is wrong. If price is past it, you are not adding, you are exiting. The invalidation is not a suggestion; it is the whole reason you get to have conviction on the way down.

Around that gate, build three standing rules and write them where you will see them at the moment of temptation:

- **Pre-write the scale-in ladder before you enter, with a hard total-size cap.** Decide the exact prices you would add at, the exact size at each, and the *maximum* total position — all before the first fill. An add that is not on the pre-written ladder does not happen. A common, sane structure is *anti-martingale*: add the same or *smaller* amounts as it falls, never larger, so a continued decline cannot force an exploding bet.
- **Ban all adds once the thesis breaks or the invalidation is hit.** These are not "sell some" moments; they are "you are done" moments. The whole point of setting the line in advance is to remove the in-the-moment negotiation, which you will always lose.
- **When you catch a screen tell, the default action is reduce, not add.** If you notice yourself computing a breakeven price or sizing from pain (the passage above), treat it as an automatic signal to *trim* risk while you re-decide with a cooler head — the opposite of your impulse, which is exactly why it works.

### Budget the whole campaign from your risk, not your hope

The reason reactive adds explode is that they are sized from the *loss* ("how much to fix my average") instead of from a *risk budget* ("how much am I willing to lose on this idea, total"). Flip that around and the size cap writes itself. Decide, before entry, the most you will lose on the entire position if your invalidation is hit — call it your risk unit, or **1R**. Then your ladder is just the set of entries whose combined loss-to-invalidation equals 1R. The adds are not a rescue; they are pre-committed slices of a fixed budget.

#### Worked example: sizing a scale-in from a fixed risk budget

Suppose your account is $100,000 and you cap risk at 1% — so 1R is **$1,000**, the most you will lose on this trade. Your thesis is invalid on a close below $46. You want to build a position around $50 with one lower add. From $50 your loss-to-invalidation is $4 a share; from $48 it is $2 a share. To keep total risk at $1,000, you might buy 150 shares at $50 (risking $600 to the $46 line) and, *if* the thesis is still intact, 200 shares at $48 (risking $400 more). Total risk: $600 + $400 = $1,000 — exactly 1R — across 350 shares for about $17,100 committed. Now the "add" at $48 is not a panicked doubling; it is the second, pre-planned tranche of a budget you set while calm, and a close below $46 sells all 350 for a loss you already agreed to. **The takeaway:** when the position's *total* loss-at-invalidation is fixed in advance, there is no room for a reactive add — the budget is already spent, and the market cannot goad you into risking more than 1R.

Suppose you own 200 shares of a stock at an average of $50 ($10,000 committed), it drops to $38, and every cell in your body wants to buy 400 more to drag your average toward $42. Run the gate.

*Question 1: would I open a 600-share position here, fresh, at $38?* Be honest — you would not; a clean-slate you would want to see the trend stabilize first. That single *no* ends it: no add. But suppose you had a plan. Your pre-written ladder said "core 200 at $50, add 100 at $40 if the thesis holds, total cap 300 shares, invalidation on a close below $37." *Question 2:* thesis intact? You check — earnings guidance was reaffirmed, nothing structural broke, yes. *Question 3:* is the add inside the cap? Adding 100 takes you to 300, exactly the cap — yes, but only 100, not 400. *Question 4:* is $38 on the right side of the $37 invalidation? Yes, barely. So you add **100** shares — a planned rung — not 400, and you know that a close below $37 means you sell all 300 for a bounded, pre-agreed loss.

**The takeaway:** the same red screen produces a $15,200 reckless double-down or a small, capped, pre-planned rung depending entirely on whether you ran the gate — and the gate takes ten seconds.

## When this matters to you

This is not an exotic professional's problem; it is the single most common way ordinary investors turn a manageable mistake into a life-altering one. It shows up when you keep buying a sliding stock because you "already own it," when you refuse to sell a losing fund because selling makes the loss real, when you add to a leveraged crypto position to avoid liquidation, or when you take one more angry trade to end the day green. In every case the mechanism is identical: a sunk cost you cannot recover is quietly running a decision that should only look forward.

The good news is that this bias is unusually *fixable*, because the fix is mechanical rather than emotional. You will never talk yourself out of the pain of a loss in the moment — but you do not have to. You only have to decide, while you are calm, three things per position: the reason you own it, the most you will ever risk on it, and the price at which you admit you are wrong. Write them down before you enter, and let those three numbers overrule the version of you that is staring at red. The sunk cost will still tug; it always does. The difference between the traders who survive and the ones who become case studies is not that the tug is weaker — it is that they decided, in advance, not to answer it.

## Sources & further reading

- Hal R. Arkes & Catherine Blumer, "The Psychology of Sunk Cost," *Organizational Behavior and Human Decision Processes*, 35(1), 1985 — the theater-ticket and ski-trip experiments and the 54% finding. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0749597885900494)
- Hal R. Arkes & Peter Ayton, "The sunk cost and Concorde effects: Are humans less rational than lower animals?," *Psychological Bulletin*, 125(5), 1999 — origin and framing of the Concorde fallacy. [APA PsycNet](https://psycnet.apa.org/record/1999-11440-006)
- Barry M. Staw, "Knee-Deep in the Big Muddy: A Study of Escalating Commitment to a Chosen Course of Action," *Organizational Behavior and Human Performance*, 16, 1976 — escalation of commitment and the role of personal responsibility. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/0030507376900052)
- Amos Tversky & Daniel Kahneman, "Advances in Prospect Theory: Cumulative Representation of Uncertainty," *Journal of Risk and Uncertainty*, 5, 1992 — the loss-aversion coefficient $\lambda \approx 2.25$.
- Nick Leeson and the collapse of Barings Bank (1995): losses of roughly £827 million and the £1 sale to ING. [Britannica: Bankruptcy of Barings Bank](https://www.britannica.com/event/bankruptcy-of-Barings-Bank) · [Nick Leeson (Wikipedia)](https://en.wikipedia.org/wiki/Nick_Leeson) · Stephen Brown, "Doubling: Nick Leeson's Trading Strategy," NYU Stern.
- Bill Miller and the Legg Mason Value Trust: the fifteen-year streak (1991–2005), the 2008 drawdown of roughly 55%, and the fall in assets from about $16.5 billion to $4.3 billion. [Forbes: End of an Era](https://www.forbes.com/sites/steveschaefer/2011/11/17/end-of-an-era-bill-miller-to-give-up-reins-at-legg-mason-value-trust/) · [Institutional Investor: Bill Miller in the Wilderness](https://www.institutionalinvestor.com/article/2bswi2n30990nntbscc1s/corner-office/bill-miller-in-the-wilderness-and-loving-it)
- S&P 500 2008 total return of roughly −37% for the comparison to the Value Trust's decline.
- The 2012 JPMorgan "London Whale" trading loss of roughly $6.2 billion in the Chief Investment Office, and the roughly $920 million in fines. [2012 JPMorgan Chase trading loss (Wikipedia)](https://en.wikipedia.org/wiki/2012_JPMorgan_Chase_trading_loss) · [CFA Institute: Understanding the Hedge That Wasn't](https://blogs.cfainstitute.org/investor/2012/05/17/jpmorgan-chase-and-the-london-whale-understanding-the-hedge-that-wasnt/)
- Related on this blog: [loss aversion and the disposition effect](/blog/trading/trading-psychology/loss-aversion-and-the-disposition-effect) · [what would change my mind: defining invalidation upfront](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront) · [the cognitive bias map for traders](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders) · [confirmation bias and motivated reasoning](/blog/trading/trading-psychology/confirmation-bias-and-motivated-reasoning)

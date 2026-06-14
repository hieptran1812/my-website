---
title: "Market-making games: making tight markets under uncertainty"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch guide to the signature quant-trader interview: quote a two-sided market on an unknown quantity, read what every fill tells you, manage inventory, and keep your composure -- with five fully solved market-making games."
tags: ["quant-interviews", "market-making", "bid-ask-spread", "adverse-selection", "expected-value", "inventory-risk", "trading-games", "quantitative-trading", "glosten-milgrom", "two-sided-market"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A market-making interview game asks you to quote a two-sided price on an unknown number, then trade against a sharp interviewer who is reading you for mistakes. It tests expected-value reasoning, adverse-selection awareness, and composure far more than arithmetic.
>
> - **Make a market** means name a *bid* (where you will buy) and an *ask* (where you will sell) at the same time. The gap between them is the *spread*; the midpoint is your *mid*, your best guess at fair value.
> - **Center your quote on your expected value (EV)** and set the **width by your uncertainty**: a confident estimate quotes tight, a foggy one quotes wide.
> - **Every fill is information.** If the interviewer *lifts* your ask (buys from you), they think it is cheap, so nudge your fair value *up*. If they *hit* your bid (sell to you), nudge it *down*.
> - **Adverse selection is the whole game**: you only trade when someone wants the other side, and the sharp ones only want it when you are wrong. Your spread has to be wide enough that the *edge* you earn from random flow covers the *loss* you take to informed flow.
> - The number to remember: a completed round-trip — buy the bid, sell the ask — earns you **the full spread**, so each single fill banks **half the spread** relative to mid. On a $6.50 / $7.50 market, that is **$0.50 per fill**.

## A game you can lose in the first ten seconds

Here is a question that has ended more quant-trader interviews than any brain-teaser about pirates or light bulbs: *"Make me a market on the number of windows in this building."*

You have never counted the windows. You have no idea. And that is exactly the point. The interviewer is not testing whether you know the answer — nobody does. They are testing whether, under that uncertainty, you can name a *bid* and an *ask*, defend the width between them, and then keep your head when they start trading against you and asking you to tighten up.

It is the closest thing an interview has to sitting on a real trading desk. A market maker's entire job is to quote prices on things whose true value is uncertain, get picked off occasionally by people who know more than they do, and still come out ahead by collecting a small edge from everyone else. The game compresses that whole job into five minutes across a table.

![Bid is where you can sell, ask is where you can buy, the spread is the gap, and the mid is fair value.](/imgs/blogs/market-making-games-quant-interviews-1.png)

The diagram above is the mental model for everything that follows. A *market* is two prices at once: a lower one you are willing to buy at (the bid) and a higher one you are willing to sell at (the ask). The distance between them is your spread, and it is where your profit lives. The invisible point halfway between is the mid — your honest guess at what the thing is actually worth. Master how those three numbers move and you have mastered the game.

This post builds the whole thing from zero. We will define every term the first time it appears, work each idea with small dollar examples you can do in your head, then put it all together in an "In the interview room" section with five games solved end to end — including a sequential one where the interviewer trades against you repeatedly and you have to track your running profit and loss. By the end you should be able to walk into a Jane Street, Optiver, SIG, IMC, Jump, HRT, or Citadel Securities room and play the game instead of freezing.

A quick note before we start: this is educational. It explains how market makers think and how interviews test it. It is not financial advice, and the "trades" here are paper trades in a game, not a recommendation to do anything with real money.

## Foundations: what "make a market" actually means

Before any technique, we need the vocabulary cold. A market maker is someone who stands ready to *both* buy and sell a thing at all times, quoting a price for each. If you have ever exchanged currency at an airport kiosk, you have traded against a market maker: the board shows one rate to buy euros and a worse rate to sell them back, and the kiosk pockets the difference. That difference is the spread, and it is the same idea whether the thing being quoted is euros, a stock, or the number of windows in a building.

### Bid, ask, spread, mid — the four numbers

Let us pin down the four words that do all the work. Imagine the thing you are quoting is worth, in truth, some number you do not know. You name two prices:

- The **bid** is the price at which *you will buy*. It is the most you are willing to pay. Someone who wants to sell can *hit* your bid and you take the thing off their hands.
- The **ask** (also called the *offer*) is the price at which *you will sell*. It is the least you are willing to accept. Someone who wants to buy can *lift* your ask and you hand the thing over.
- The **spread** is `ask − bid`, the gap between the two. It is the cushion that makes the business work. Quote `$6.50` bid and `$7.50` ask and your spread is `$1.00`.
- The **mid** (midpoint) is `(bid + ask) / 2`, the price exactly halfway between. With a `$6.50 / $7.50` market the mid is `$7.00`. The mid is your stated guess at fair value; the spread is your margin of safety around that guess.

Notice the bid is *always* below the ask. If you ever quote a bid above your ask, you have made a *crossed* market — you are simultaneously offering to buy high and sell low, which lets the interviewer buy from you cheap and sell to you dear in one motion and book free money. Crossing your own market in an interview is an instant fail, so the very first sanity check on any quote is: *is my bid below my ask?*

### "Make a market" — the instruction and the rules

When the interviewer says "make a market on X," they are asking for exactly those two numbers: your bid and your ask on the unknown quantity X. The standard rules of the game, which good interviewers state and bad ones assume, are:

1. **You quote both sides.** You do not get to pick whether you are a buyer or a seller. You must be willing to do either at your quoted prices.
2. **The interviewer chooses.** After you quote, *they* decide whether to buy at your ask, sell at your bid, or pass. You are committed to honoring whichever they pick.
3. **They can trade more than once.** In a sequential game they will trade, watch you re-quote, and trade again — often on the same side — to see whether you update correctly.
4. **The truth gets revealed at the end**, and your profit and loss (P&L) is computed against it. If you bought at `$6.50` and the true value turns out to be `$7.00`, you made `$0.50` on that unit. If you sold at `$7.50` and the truth was `$7.00`, you made `$0.50` there too.

The trader's word for being on the wrong end of a trade is being *picked off* or *run over*. The trader's word for the steady small profit you collect from people who trade with you for reasons unrelated to value is *edge*. The entire game is a tug-of-war between edge and getting picked off, and the spread is the dial you set to balance them.

### Why a spread exists at all

A beginner's instinct is to quote a single number — "I think it's seven, so my market is seven." But a single price is not a market, it is a sitting duck. If you quote `$7.00` flat as both your bid and your ask, the interviewer simply buys from you whenever the truth is above `$7.00` and sells to you whenever it is below, and you lose every single time information moves against you while never collecting anything for the privilege. The spread is what you charge for standing in the middle and being willing to trade either way. No spread, no business.

With the foundations in place, the rest of the post is four moves — quote around EV, widen for uncertainty, read your fills, and skew for inventory — plus the games that combine them. It helps to see those moves as a *loop* rather than a checklist, because in a real game they repeat every time the interviewer trades.

![Making a market is a repeated loop: form an expected value, quote a two-sided price, get filled, update your belief and inventory, then re-quote.](/imgs/blogs/market-making-games-quant-interviews-2.png)

The figure above is the rhythm of the whole game. You start by forming an EV — your fair value. You wrap a spread around it whose width reflects your risk. The interviewer trades, which both hands you a position and reveals information. You fold that information and that inventory back into a new fair value and a new skew. Then you re-quote, and the loop runs again. A candidate who treats the game as "name one market and stop" has only walked the first two boxes; the interview lives in the *update* and *re-quote* steps, where the same person who quoted a sensible opening market either adapts gracefully or falls apart. Every technique section below is really one box of this loop examined up close.

## Quoting around your expected value

The first move is choosing where to center your market. The answer is your **expected value** (EV) — the probability-weighted average of all the outcomes you think are possible. Your mid should sit at your EV, because the mid is your fair-value guess and EV is the formal name for "the average of what I think will happen."

> [!info]
> **Expected value, defined.** If a quantity can take values $x_1, x_2, \dots$ with probabilities $p_1, p_2, \dots$, its expected value is $\text{EV} = \sum_i p_i \, x_i$. It is the long-run average if you could repeat the situation many times. For the sum of two fair dice, every face is equally likely, the symmetry sends the average to the middle, and $\text{EV} = 7$.

### Worked example: the sum of two dice

Take the cleanest possible game. The interviewer rolls two fair six-sided dice behind a screen and asks you to make a market on their *sum*. The sum can be anything from 2 to 12, but not with equal probability — there is exactly one way to roll a 2 (snake eyes) but six ways to roll a 7 (1+6, 2+5, 3+4, 4+3, 5+2, 6+1).

![The sum of two dice peaks at 7, which is both the most likely value and the mean, so a fair mid sits at $7.00.](/imgs/blogs/market-making-games-quant-interviews-8.png)

The distribution is a perfect triangle, peaking at 7. Because it is symmetric around 7, the expected value is exactly 7:

$$\text{EV} = \frac{2 + 3\cdot2 + 4\cdot3 + 5\cdot4 + 6\cdot5 + 7\cdot6 + 8\cdot5 + 9\cdot4 + 10\cdot3 + 11\cdot2 + 12}{36} = \frac{252}{36} = 7.$$

So your mid is `$7.00`. (We will treat each "point" of the sum as one dollar, which is the convention in these games — a market "on the sum" is really a market on a contract that pays out the sum in dollars.) You do not yet know how *wide* to go, but you know the *center*: a fair, honest market on the sum of two dice is built around `$7.00`. Quote `$6.50 / $7.50` and your mid is `$7.00`, dead on the EV.

The one-sentence intuition: **center your market on your expected value, because the mid is a price and EV is the average price the thing is worth.**

### When EV is not the obvious middle

The dice are friendly because symmetry hands you the EV. Most game quantities are not symmetric, and you have to *estimate* the EV by bracketing. Asked for the number of countries in the world, you might reason: "It is definitely more than 100 and definitely fewer than 300; I have a vague memory it is just under 200." That lands your EV around 195. Asked for the population of a city you have never visited, you anchor on cities you do know and interpolate. The skill the interview is probing is not whether your EV is exactly right — it is whether you can produce a *defensible* EV fast and quote around it without flinching.

A common trap: people anchor on the first number that pops into their head and quote around *that* instead of around a thought-out EV. If your gut says "175 countries" but a moment's reasoning says "the UN has around 193 members, plus a few non-members, call it 195," quote around 195. Spend the three seconds to reason; do not quote your reflex.

## How wide should the spread be?

Centering is half the quote. The other half is the width, and width is governed by a single principle: **the more uncertain you are about fair value, the wider you should quote.** A tight spread is a statement of confidence; a wide spread is a statement of doubt. Quoting tight on something you are clueless about is the single most expensive mistake in the game, because it invites everyone who knows more than you to trade against you at near-fair prices.

![Center every quote on EV and set its width by uncertainty: a confident dice estimate quotes tight, an unsure country estimate quotes wide.](/imgs/blogs/market-making-games-quant-interviews-3.png)

The figure shows the same recipe applied twice. On the dice sum you are confident — the EV is provably 7 and the whole distribution lives between 2 and 12 — so a `$1.00`-wide market (`$6.50 / $7.50`) is reasonable. On the number of countries you are genuinely unsure, so you quote `170 / 220`, a width of 50 around your EV of 195. Both quotes use the identical construction: **midpoint = EV, half-width = how wrong you might plausibly be.**

### The spread is a risk dial

Think of your half-spread — half the gap between bid and ask — as a buffer against being wrong. If your EV is `$7.00` and your true uncertainty is "I might be off by about `$0.50` in either direction," then a half-spread of `$0.50` (a `$1.00` total spread) means that even when someone trades against you because they know the truth, they only take you for about the amount you were already unsure by. Quote a half-spread of `$0.05` on that same `$0.50` of uncertainty and every informed trade takes you for ten times your cushion.

![Wider uncertainty about fair value pushes the spread you should quote wider, from about a dollar on a dice sum to tens of dollars on a private guess.](/imgs/blogs/market-making-games-quant-interviews-4.png)

The relationship is monotone: as the uncertainty in fair value rises — from a dice sum, to the total of three drawn cards, to a city's population, to the number of countries, to a quantity only you have a private guess about — the spread you should quote rises with it. There is no fixed "right" width; there is only "wide enough that your edge survives the people who know more than you," which we will make precise in a moment.

### Worked example: sizing a dice market

Suppose you decide on `$6.50 / $7.50` for the two-dice sum. Is `$1.00` too wide, too tight, or about right? Reason it through. The standard deviation of a two-dice sum is about 2.4 — that is the typical distance of the actual roll from 7. But you are not exposed to the *whole* roll's variation, because you are quoting around the *mean*, which you know exactly. Your risk is not "how far might the roll be from 7" but "how badly can a counterparty who sees the roll pick me off." Since the interviewer in the basic game does *not* see the roll either (they roll behind a screen and the truth is random), there is no informed trader at all, and a tight market is actually safe. The `$1.00` spread is, if anything, generous.

Contrast that with a game where the interviewer *does* know the answer — say, "make a market on the closing price of a stock you do not follow, and I have today's chart in front of me." Now there *is* an informed trader on the other side, and the same `$1.00` width could be far too tight. **The right width depends on who is on the other side as much as on your own uncertainty.** That insight is the bridge to the next section.

#### Worked example: how tight is too tight

Let us put a number on "too tight." Suppose you are quoting a quantity where you genuinely might be off by `$5` in either direction — your honest one-sided uncertainty is `$5`. If you quote a spread of `$10` (half-spread `$5`), then even a counterparty who knows the exact truth can only pick you off by about your half-spread: they lift your ask when the truth is just above it, but your ask already sits `$5` above your mid, so their edge is small. Now shrink the spread to `$2` (half-spread `$1`) on that same `$5` of uncertainty. An informed trader who knows the truth is `$4` above your mid will lift your ask — which sits only `$1` above mid — and pocket roughly `$3` per unit at your expense. You have cut your spread by `$8` and handed most of that `$8` straight to anyone who knows more than you. The arithmetic is brutal and one-directional: **tightening below your true uncertainty does not win you more good trades, it only enlarges the loss on every bad one.** That is why the reflex "tighten to look confident" is so dangerous, and why a disciplined trader ties width to uncertainty and refuses to go below it.

## Getting hit or lifted: what a fill tells you

Here is where novices and traders split. To a novice, a fill is just a transaction — the interviewer traded, the round is over, move on. To a trader, a fill is *information*, and often the most valuable information in the whole game. The reason is uncomfortable but central: **you only ever trade when someone wants the other side, and the people most eager to take the other side are precisely the ones who think you are wrong.**

### The two ways you get filled

There are exactly two things the interviewer can do to your quote:

- They can **lift your ask** — buy from you at your higher price. They do this when they think the thing is worth *more* than your ask. The trader's phrase is "you got lifted" or "they lifted your offer."
- They can **hit your bid** — sell to you at your lower price. They do this when they think the thing is worth *less* than your bid. The phrase is "you got hit."

Both leave you holding a position. Get lifted and you are now *short* — you have sold something you have to deliver, and you profit if its value falls. Get hit and you are now *long* — you own something, and you profit if its value rises. The position itself is risk, which we will handle in the next section. Right now we care about the *signal*.

![Getting lifted on your ask is evidence the true value sits above your mid, so you revise your fair value up toward where the trade happened.](/imgs/blogs/market-making-games-quant-interviews-5.png)

### Worked example: revise toward the trade

You quote `$6.50 / $7.50`, mid `$7.00`, on some quantity. The interviewer lifts your ask — they buy at `$7.50`. What just happened? Someone willing to pay `$7.50` for the thing thinks it is worth *at least* `$7.50`, probably more. Your `$7.00` mid was, in their view, too low. The honest response is to **revise your fair value up.** If you nudge your mid from `$7.00` to `$7.40` and re-quote `$6.90 / $7.90`, you have done two things at once: moved toward the information the trade revealed, and positioned yourself to buy the thing back lower (your new bid `$6.90` is still below the `$7.50` you sold at, so unwinding is profitable if the value really is near your old mid).

The symmetric case: get *hit* on your bid (someone sells to you at `$6.50`) and you should revise your mid *down*, because the seller thought `$6.50` was rich. The rule, stated once and worth memorizing: **always revise your estimate toward the side that traded.** Lifted → up. Hit → down.

The one-sentence intuition: **a fill is a vote on your price by someone who chose to act on it, so move your fair value toward where they traded.**

### Adverse selection: the reason this matters so much

Now the deep version. If the only people trading with you were *informed* — people who know the true value — you would lose on every fill, because they would only ever buy when your ask is too low and only ever sell when your bid is too high. You would be the patsy in the room. The reason market making is a viable business is that *most* flow is **not** informed: it is *noise*, people trading for reasons that have nothing to do with whether your price is right.

> [!info]
> **Noise traders vs informed traders.** A *noise trader* trades for a reason unrelated to value — they need cash, they are rebalancing, they are hedging something else, or in the game, they are a random buyer or seller. They are equally likely to hit your bid or lift your ask, so over many trades they pay you the spread and their direction averages out. An *informed trader* trades *because* your price is wrong, and only on the side that is wrong, so they systematically cost you money. The term for the fact that your fills are skewed toward the informed side is **adverse selection** — the trades you get "selected" into are adversely chosen against you.

![Noise traders fill both sides and pay you the spread, while informed traders only lift your ask when it is too low, so the spread must cover the informed loss.](/imgs/blogs/market-making-games-quant-interviews-6.png)

The figure traces both paths from your single two-sided quote. Noise traders fan out to both sides over time, you earn roughly half the spread on each, and their random direction nets to a profit for you. Informed traders only take the side where you are mispriced, so they hand you an expected loss every time they trade. The whole craft of setting a spread is making sure the green edge from noise outweighs the red loss to information. We will put numbers on that balance shortly.

## Inventory: skewing your quotes after a fill

A fill does not just give you information — it gives you a *position*, and a position is risk you did not choose. If you have bought (you are long) and the value then falls, you lose. If you have sold (you are short) and the value rises, you lose. A market maker does not want to *hold* positions; they want to *cycle* them — buy, then sell, then buy again, collecting the spread each round-trip. So after a fill, beyond updating your mid for information, you also **skew your quote to lean toward getting back to flat.**

### What skewing means

Skewing is shifting *both* your bid and your ask in the same direction without necessarily changing the spread width. If you are long (you bought and want to sell), you skew *down*: lower both prices so your ask becomes more attractive to buyers (inviting someone to take your inventory) and your bid becomes less attractive to sellers (discouraging you from buying even more). If you are short, you skew *up* by the same logic.

![After buying a lot you are long, so you shade both your bid and ask down to make selling easy and buying more unattractive.](/imgs/blogs/market-making-games-quant-interviews-7.png)

### Worked example: skewing after buying inventory

You are flat and quoting `$6.50 / $7.50` (mid `$7.00`, spread `$1.00`). The interviewer hits your bid — you buy one lot at `$6.50`. You are now long one lot, and you would like to sell it. So you skew your whole quote down by, say, `$0.10`: new market `$6.40 / $7.40`, mid `$6.90`, *same* `$1.00` spread.

Look at what this accomplishes. Your ask dropped from `$7.50` to `$7.40`, so it is now closer to the action and more likely to attract a buyer who takes your long off your hands. Your bid dropped from `$6.50` to `$6.40`, so you are less eager to buy *more* and add to a position you are trying to shed. You have not changed how much edge you charge — the spread is still `$1.00` — you have only *recentered* it to tilt the odds toward unwinding. And note the two motives can point in *opposite* directions: information from a *hit* says revise the mid down (the seller thought it cheap), and inventory from being *long* also says skew down. When both point the same way, skew confidently. When they conflict — say a fill that is informationally bullish but leaves you long — you net the two effects and move by the difference.

The one-sentence intuition: **after a fill, shift your whole quote toward the price that returns you to flat, because holding inventory is unchosen risk you want to recycle.**

### How much to skew

The size of the skew scales with how much the position scares you. A one-lot position in a thing you understand well warrants a gentle nudge. A position that is large relative to the limits you are allowed to carry warrants an aggressive shift — sometimes so aggressive that you make your bid *worse than fair* just to avoid buying more, effectively telling the market "I really do not want to add here." Real desks formalize this with an *inventory penalty*: the more inventory you carry, the more you shade your quotes, so that the cost of accumulating a dangerous position is baked into the prices you show.

## Information events: updating your mid when news arrives

So far the only new information has come from *your own fills*. But in richer games — and in real markets every second — information also arrives from *outside*: the interviewer reveals a card, announces a fact, or says "by the way, one of the dice is a 6." When that happens you must re-price *immediately*, before you quote again, because a quote built on stale information is a quote built to be picked off.

### Worked example: a card is revealed

Suppose the game is "make a market on the total of three cards drawn from a deck," where each card counts its face value (ace = 1, ..., king = 13). Before any card is shown, each card has an expected value of 7 (the average of 1 through 13), so the total has EV `3 × 7 = 21`. You quote around `$21`.

Now the interviewer flips the first card face up: it is a king, worth 13. The math instantly changes. Two unknown cards remain, each still worth 7 on average, so the new EV is `13 + 7 + 7 = 27`. Your mid must jump from `$21` to `$27` *before you re-quote*. If you sleepily re-quote around `$21` after seeing a king, the interviewer lifts your ask all day and you bleed `$6` a unit. The discipline is: **news first, quote second.** Re-price on every piece of revealed information, then build your spread around the *updated* mid.

### Worked example: news that changes your uncertainty, not just your mid

Sometimes news does not move your EV much but slashes your *uncertainty*, and that should *tighten* your spread. Imagine you are quoting a quantity you were very unsure about, with a wide spread to match. The interviewer then tells you something that pins the answer down to a narrow range — your EV barely moves, but your uncertainty collapses. The correct response is to *narrow* your spread, because you no longer need the wide cushion. Width tracks uncertainty, so when uncertainty drops, width should drop too. A trader who keeps quoting wide after the fog lifts is leaving edge on the table; a trader who keeps quoting tight after fog *rolls in* is begging to be run over.

## The interviewer's tricks (and how to hold your composure)

The interviewer is not a passive price-taker. They are an active adversary whose job is to find the seam in your reasoning. Knowing their playbook is half the battle.

**Trick 1: trading against you repeatedly on the same side.** They lift your ask, you re-quote, they lift it again, and again. A panicking candidate thinks "they keep buying, I must be way too low" and skews up aggressively each time — and then the truth comes out near the original mid and they have sold a big short at terrible prices. The composed response: *some* update is warranted (repeated lifts are repeated evidence), but each successive lift at a price *you* keep raising is weaker evidence than the first. Update, but with diminishing steps, and never let repeated trades stampede you off a fair value you can defend. Also watch your inventory: three lifts means you are short three lots, and the inventory skew alone should be pulling you up — do not double-count the same signal as both information and inventory.

**Trick 2: demanding you tighten.** "That's a terrible market, way too wide. Tighten it up." This is a test of *conviction*, not a genuine request. If your wide spread is justified by real uncertainty, you tighten only modestly and explain why: "I'll come in to `170 / 215`, but I'm staying wide because I genuinely don't know this number to within five." If you cave and quote `193 / 197` on something you cannot estimate to within fifty, you have just told the interviewer you will fold under pressure — which is the one trait that gets a trader fired. Hold a spread you can defend; do not hold one out of spite, and do not collapse one out of fear.

**Trick 3: the asymmetric reveal.** They give you one piece of information and watch whether you update by the right amount. Reveal a king in the three-card game and the correct jump is exactly `+6` (from EV 21 to 27). Jump `+13` (overreacting, as if the whole total were the king) or `+0` (ignoring it) and you have shown you cannot integrate news into a price. Compute the update explicitly; do not eyeball it.

**Trick 4: the crossed-market trap.** They trade in a way that, if you skew sloppily, leads you to quote a bid above your ask. Always re-check after every skew that your bid is still below your ask. A crossed market in front of an interviewer is the fastest way to end the conversation.

The meta-lesson across all four: **the game rewards composure and correct updating, not cleverness or speed.** A candidate who quotes a defensible market, updates sensibly, and never panics beats a candidate who computes faster but folds.

## In the interview room: five games solved end to end

Now we put every piece together. Here are five market-making games of increasing difficulty, each solved the way you would actually play it across the table. Numbers are in dollars throughout, treating one "point" of the quantity as one dollar.

### Game 1: make a market on the sum of two dice

**Setup.** The interviewer rolls two fair dice behind a screen. Make a market on the sum.

**Your move.** The EV is 7 by symmetry (worked above), and in the basic version the interviewer cannot see the roll either, so there is no informed trader. You quote a moderate market: **`$6.50 / $7.50`**, mid `$7.00`, spread `$1.00`.

**The interviewer trades.** They lift your ask — they buy at `$7.50`. (Maybe they are just testing you; the roll is still random.) You are now short one lot at `$7.50`.

**P&L when the truth is revealed.** The dice come up 4 and 3, summing to 7. You sold at `$7.50` something worth `$7.00`, so you made `$7.50 − $7.00 = $0.50`. Had they hit your bid instead, buying at `$6.50`... wait, *you* would have bought at `$6.50` something worth `$7.00`, again making `$0.50`. **Either fill, against a fair mid, banks you half the spread.** That is the single most important number in market making, so let us draw it.

![A completed round-trip earns the full spread, so each single fill banks half the spread relative to the mid.](/imgs/blogs/market-making-games-quant-interviews-10.png)

The figure makes the arithmetic concrete. Buying at the `$6.50` bid is `$0.50` of edge versus the `$7.00` mid; selling at the `$7.50` ask is another `$0.50` of edge versus the mid. A full round-trip — buy then sell — captures both halves, the entire `$1.00` spread. A single fill captures one half, `$0.50`. So **your edge per round-trip is the full spread, and your edge per fill is half the spread.** Memorize that and you can compute your expected profit in any of these games in your head.

**The lesson.** On a symmetric, no-information game, any fill at your fair mid is pure edge. The dice game is the "hello world" of market making precisely because it strips away adverse selection and lets you feel the spread as profit.

### Game 2: make a market on the number of countries in the world

**Setup.** "How many countries are there in the world? Make me a market."

**Your move.** You do not know the exact figure, but you can bracket it. The UN has 193 member states; add a couple of widely-recognized non-members and you land around 195. Your uncertainty is real — you might be off by ten or fifteen — so you quote *wide*: **`170 / 220`**, mid `195`, spread `50`. If asked to justify the width, you say plainly: "I don't know this to within five, so I'm quoting fifty wide; tighten it and you're just picking my pocket."

**The interviewer trades.** They lift your ask at `220`. You are short at `220`. They are signaling they think the number is above 220 — but is it? You revise your EV up *modestly*, say from 195 to 200, because one trade is weak evidence and 220 is already well above any defensible estimate. You re-quote `175 / 225`, both for the information and to skew up off your short.

**The interviewer demands you tighten.** "Come on, fifty wide is a joke, make it five wide." You hold firm: "I'll tighten to `185 / 215`, thirty wide, but I won't go to five — I genuinely don't know the answer that precisely, and a tight market here is just a free option for you." 

**P&L.** The truth is revealed as 195. You sold at `220` something worth `195`, so you *made* `220 − 195 = $25` on that lot. The interviewer who lifted you was wrong — they paid `220` for something worth `195`. Your wide spread *protected* you: a tight `193 / 197` market would have meant selling at `197`, making only `$2`; the wide market let you sell `$25` rich.

**The lesson.** Width is your friend on genuinely uncertain quantities. The interviewer pushing you to tighten is testing conviction, not asking sincerely. Tighten modestly if at all, and never below the width your uncertainty justifies.

### Game 3: make a market on the total of three drawn cards

**Setup.** Three cards are drawn from a standard deck; each counts its face value (ace = 1, ..., king = 13). Make a market on the total.

**Your move.** Each card averages `(1 + 2 + ... + 13) / 13 = 7`, so three cards total EV `21`. The spread of possible totals is `3` (three aces) to `39` (three kings), but the bulk sits near 21. Your uncertainty is moderate — wider than dice, narrower than countries. You quote **`$18 / $24`**, mid `$21`, spread `$6`.

**The information event.** The interviewer flips the first card: a king, worth 13. Re-price *before* quoting: two cards remain at EV 7 each, so the new total EV is `13 + 7 + 7 = 27`. Your mid jumps from `$21` to `$27`. You re-quote **`$24 / $30`**, mid `$27`. (Your uncertainty also shrank slightly — one of three unknowns is now known — so you could even tighten to `$25 / $29`.)

**The interviewer trades.** They hit your bid at `$24` — they sell to you at `$24`. You are now long at `$24`. Their selling hints they think the total is *below* `$24`; you revise your mid down a touch, to `$26.50`, and skew down for your long, re-quoting `$23.50 / $29.50`.

**P&L.** The remaining two cards come up a 5 and a 9, so the true total is `13 + 5 + 9 = 27`. You bought at `$24` something worth `$27`, making `27 − 24 = $3`. Note the seller who hit you at `$24` was *wrong* — they sold something worth `$27` for `$24` — so here the "informed" trade was actually noise, and you collected. That is the texture of real play: not every aggressive trade is informed, and your job is to update *probabilistically*, not to assume every counterparty is a genius.

**The lesson.** Re-price on every revealed card *before* re-quoting; integrate the news exactly (`+6` for the king, not `+13` or `+0`); then treat each subsequent fill as evidence to weigh, not gospel to obey.

### Game 4: the sequential dice game with running P&L

**Setup.** This is the game that separates candidates. The interviewer will make a market — *you* quote, *they* trade, you re-quote, they trade again — and at the end you must state your total P&L against the revealed truth. It is the two-dice sum, EV `7`, but now played as a multi-round battle.

![Each trade moves your inventory and your fair value; both branches of this sequential dice game settle at +$0.50 P&L.](/imgs/blogs/market-making-games-quant-interviews-9.png)

**Round 1.** You quote **`$6.50 / $7.50`**, mid `$7.00`. The interviewer lifts your ask at `$7.50`. You are short one lot at `$7.50`, and you have collected `$0.50` of theoretical edge (you sold `$0.50` above fair). You skew down for your short — wait, you are *short*, so you skew *up* to discourage selling more and encourage buying back. Re-quote **`$6.60 / $7.60`**, mid `$7.10`. (A short wants the price to fall, but to *unwind* a short you must buy, so you raise your bid to invite a seller to trade with you — the skew direction for a short is *up*.)

**Round 2.** From the new market `$6.60 / $7.60`, the interviewer now hits your bid at `$6.60` — they sell to you at `$6.60`. You buy one lot at `$6.60`. This *closes* your short: you sold one at `$7.50` and bought one back at `$6.60`, locking in `$7.50 − $6.60 = $0.90` on that round-trip. You are now flat (short one, then long one, nets to zero).

**The reveal.** The dice sum to 7. Let us total your P&L two ways to be sure. By round-trip: you sold at `$7.50`, bought at `$6.60`, profit `$0.90`. By mark-to-truth: the lot you sold at `$7.50` is worth `$7.00`, a `+$0.50` gain; the lot you bought at `$6.60` is worth `$7.00`, a `+$0.40` gain; total `$0.90`. **Both methods agree: `+$0.90`.** You played two rounds, captured edge on both fills because both happened at prices favorable to fair value, and ended flat with a clean profit.

The diagram shows a slightly simpler variant where each branch ends at `+$0.50` — the point is the same: **track your inventory and your fair value after every single trade, and your running P&L is just the sum of edge captured minus any losses to information.** In a sequential game, the candidates who fail are the ones who lose track of whether they are long or short after the third trade. Write it down if you have to: position, average price, mid, after every fill.

**The lesson.** A sequential game is bookkeeping plus the four moves. After each trade, update three things — your position (long/short and how much), your fair value (toward the trade), and your skew (toward flat) — and your P&L takes care of itself. Composure here means never losing the thread of your own inventory.

### Game 5: make a market when the interviewer knows the answer

**Setup.** The hardest version. "I'm looking at the exact figure right now. Make me a market on the number of employees at this company." Now there *is* an informed trader — and it is the person you are quoting to.

**Your move.** This changes everything about width. Against a fully-informed counterparty, *any* market you quote will be picked off: they buy when your ask is below the truth and sell when your bid is above it, and they pass otherwise, so you lose on every trade that happens. The only defense is a spread wide enough that even your worst-case error is smaller than your half-spread. If your EV is `5,000` employees with an honest uncertainty of `±2,000`, you must quote at least `3,000 / 7,000` — and even then a perfectly-informed adversary picks the side where you are wrong. 

The sophisticated answer acknowledges this out loud: "Since you know the answer and I don't, you'll pick me off whichever side you trade, so I have to quote wide enough that your information edge is capped. I'll show `3,000 / 7,000`, and I'd want it wider if anything." This is exactly the Glosten-Milgrom insight (defined in the next section) playing out in miniature: **the spread must be wide enough to cover the expected loss to the informed trader.**

**P&L intuition.** Suppose the truth is `6,000`. The informed interviewer lifts your `7,000` ask? No — your ask is `7,000`, the truth is `6,000`, so they would be *overpaying*; they hit your `3,000` bid instead, selling you something worth `6,000` for... no, they sell to you at `3,000`, and you *buy* at `3,000` something worth `6,000`, making `$3,000`. So they do *not* trade that side. With truth `6,000` inside your `3,000 / 7,000` quote, the informed trader has no profitable trade and *passes* — which is the whole point of quoting wide enough to straddle the truth. **A spread that brackets the true value with margin starves the informed trader of edge.** If instead the truth were `8,000`, outside your quote, they lift your `7,000` ask and you lose `$1,000` — the price of having quoted too tight relative to where the truth turned out to be.

**The lesson.** When the other side is informed, width is not a comfort, it is survival. Quote wide enough to bracket plausible truth, say so explicitly, and accept that against perfect information your best outcome is *no trade*, not a profitable one.

## Common misconceptions

**"Just quote your point estimate with zero width."** This is the cardinal sin. A single price is not a market; it is a free option you hand to everyone smarter than you. Quote `$7.00 / $7.00` on the dice sum and the interviewer buys from you every time the roll beats 7 and sells every time it misses, and you have no spread to cushion the losses. You must quote *two* prices with a gap.

**"A fill is just a transaction; it tells me nothing."** A fill is the highest-information event in the game. Someone *chose* to trade at your price, on the side they preferred, which is evidence about where they think fair value is. Ignoring that signal — re-quoting the same market after getting lifted three times — is how you end up massively short at terrible prices. Always update toward the side that traded.

**"Widen the spread to punish an aggressive interviewer."** Widening out of spite is as wrong as tightening out of fear. Your spread should reflect *risk* — your uncertainty and the chance of informed flow — not your emotions. If the interviewer trades against you and nothing about your uncertainty has changed, your spread should barely move (you skew for inventory, but the *width* stays tied to risk). Widening because you are annoyed, or tightening because you are intimidated, both signal that your prices are driven by feelings rather than expected value.

**"The mid is the average of my bid and ask, so I set it last."** Backwards. The mid is your *fair value*, the thing you compute *first* from your EV, and the bid and ask are placed symmetrically around it by your chosen width. Setting bid and ask first and backing into a mid means you are not actually pricing the underlying quantity — you are just naming numbers.

**"Getting picked off means I made a mistake."** Not necessarily. Adverse selection guarantees that *some* of your fills will be to informed traders who knew more than you — that is the cost of doing business. A good market maker loses on those trades *by design* and more than makes it back on the noise flow. The mistake is not getting picked off occasionally; the mistake is quoting so tight that the informed losses swamp the noise edge.

**"Skewing changes my spread."** Skewing shifts both quotes *together* and leaves the spread width unchanged — it recenters your market to manage inventory. Changing the spread *width* is a separate decision driven by uncertainty and informed-flow risk. Conflating the two leads to messy quotes; keep "how wide" (risk) and "centered where" (inventory + information) as independent dials.

## How it shows up on a real trading desk

The interview game is not a toy abstraction — it is a faithful, scaled-down model of what a market-making desk does millions of times a day. Here is how each move maps to the real thing.

![Every move the interview game tests maps directly to a market-making decision the desk makes thousands of times a second.](/imgs/blogs/market-making-games-quant-interviews-12.png)

**Real bid-ask spreads.** On a liquid stock like Apple, the spread on the exchange might be a single penny — say `$190.00` bid, `$190.01` ask — because thousands of market makers compete and the underlying is easy to price. On an illiquid small-cap or a complex option, the spread might be several percent of the price, because uncertainty and adverse-selection risk are far higher. The exact same principle from the game — *wider when more uncertain* — sets those real spreads. A market maker quoting a thinly-traded name is playing Game 2 (countries) with real money; one quoting a megacap is playing Game 1 (dice) at penny widths.

**Capturing edge.** A desk that quotes a `$190.00 / $190.01` market and trades both sides through the day earns roughly half the penny spread per fill — `$0.005` a share — but does it across hundreds of millions of shares. That is the round-trip edge from Game 1, industrialized. The business is not about being right on direction; it is about cycling inventory and collecting half-spreads at enormous volume. A firm like Citadel Securities or Virtu reportedly makes money on the overwhelming majority of trading *days* precisely because the law of large numbers turns a tiny per-fill edge into a reliable income when repeated billions of times.

**Adverse selection in the wild.** When a desk's fills suddenly skew to one side — they keep getting lifted and never hit — that is informed flow arriving, exactly like the interviewer lifting your ask three times. Real desks monitor this constantly; a sustained one-sided fill pattern means someone knows something, and the response is identical to the game: revise the mid toward the flow and widen or pull quotes if the flow looks *toxic* (the desk's word for systematically informed counterparties). The 2010 "Flash Crash" was, in part, market makers widening and pulling quotes en masse when order flow became impossible to price — the real-world version of refusing to make a tight market in a fog.

**The Glosten-Milgrom intuition.** In 1985, economists Lawrence Glosten and Paul Milgrom published a model that is, essentially, Game 5 written as math. Their insight: even a market maker with *zero* costs and *zero* profit target must quote a positive spread, purely because some fraction of their counterparties are informed. The spread is set so that the expected loss to informed traders exactly equals the expected gain from noise traders — the market maker breaks even on average, and the spread is the price of not knowing who is informed.

![Set the spread so the green edge collected from noise flow at least covers the red expected loss to informed flow.](/imgs/blogs/market-making-games-quant-interviews-11.png)

The balance is concrete. Suppose for every 10 trades you face, 9 are noise (paying you `$0.50` of edge each, for `+$4.50`) and 1 is informed (costing you `$4.00` because they only trade when you are badly wrong). Your net is `$4.50 − $4.00 = +$0.50` per 10 trades — the spread is just wide enough to survive. Tighten the spread and the noise edge per trade falls below `$0.50`; soon `9 × (edge) < $4.00` and you lose money on the whole book. *That* is why spreads exist, and it is the rigorous version of the gut feeling the interview is testing: **the spread is set so the expected dollar loss to informed flow is at most the dollar edge collected from noise flow.**

**Inventory limits and skewing.** Every real desk runs hard *position limits* — the maximum long or short it will carry in any name — and an inventory-skew model that shades quotes more aggressively as the position grows, exactly like Game 4. A desk that is long 100,000 shares of a stock will show a slightly lower bid and ask than its fair-value model suggests, leaning to sell, just as you skewed down `$0.10` after buying one lot. When the position hits the limit, the desk stops quoting one side entirely — the industrial version of making your bid "worse than fair" to refuse more inventory. Risk managers enforce these limits because an unmanaged inventory is how a market-making desk turns into a directional bet, which is precisely what it is not supposed to be.

**Composure as a hireable trait.** Finally, the reason firms run these games in interviews at all: the desk version of "the interviewer trades against you repeatedly and demands you tighten" happens for real, every volatile afternoon, when flow turns one-sided and a junior trader must decide whether to update, widen, or pull. Firms want to see, before they hand you the keys, that you update *correctly* and *calmly* under pressure rather than panicking. The game is a five-minute audition for a behavior the desk needs every single day.

## When this matters and where to go next

If you are interviewing for a quant *trader* or market-making seat — Jane Street, Optiver, SIG, IMC, HRT, Jump, Citadel Securities, and their peers all run versions of this — the market-making game is very likely the round that decides you. The good news is that it rewards a small, learnable set of moves: center on EV, widen for uncertainty, read every fill, skew for inventory, re-price on news, and never panic. Practice them out loud with a friend who will trade against you and demand you tighten; the muscle you are building is *quoting and updating under live pressure*, which no amount of silent reading produces.

The deeper reason to understand this, even if you never sit a trading desk, is that market making is how prices actually get *made* in the real economy. Every time you buy a stock, exchange currency, or take the other side of an option, an automated descendant of the player in this game quoted you a bid and an ask, collected a half-spread, and managed the inventory you just handed it. Understanding the game is understanding the machinery underneath every price you have ever paid.

To go further, the natural next steps build out the probability and pricing toolkit the games lean on. The [expected-value techniques](/blog/trading/quantitative-finance/expected-value-techniques-quant-interviews) post sharpens the EV reasoning that sets your mid. [Conditional probability and Bayes](/blog/trading/quantitative-finance/conditional-probability-bayes-quant-interviews) formalizes the "update toward the trade" instinct into the math of updating beliefs on evidence — exactly what a fill is. The [classic quant probability problem set](/blog/trading/quantitative-finance/classic-quant-probability-problems) drills the distributions (like the two-dice triangle) that show up as game quantities. And for how real instruments get priced once you are past the games, [options theory](/blog/trading/quantitative-finance/options-theory) and the [volatility surface](/blog/trading/quantitative-finance/volatility-surface) show where the spread-setting logic goes when the thing you are quoting is a derivative rather than a dice roll.

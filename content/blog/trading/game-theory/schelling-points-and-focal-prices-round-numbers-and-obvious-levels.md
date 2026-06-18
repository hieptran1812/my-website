---
title: "Schelling Points and Focal Prices: Round Numbers and Obvious Levels"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Round numbers, prior highs, and big option strikes act as price levels because everyone watches them at once, which makes them self-fulfilling coordination points rather than forces with any intrinsic pull."
tags: ["game-theory", "trading", "schelling-point", "focal-point", "coordination-game", "round-numbers", "support-and-resistance", "option-pinning", "behavioral-finance", "market-microstructure"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A focal price is a Nash equilibrium that everyone lands on not because the logic forces it but because it is the *obvious* one to pick — and in markets the obvious price is the round number, the prior high, the 200-day average, the big option strike. These levels "work" because everyone watches them, so everyone's orders cluster there, so the level reacts — which is exactly why everyone watches it. They are self-fulfilling, not fundamental.
>
> - A **Schelling point** (focal point) is the choice people converge on when they must coordinate *without communicating* — selected by salience and shared expectation, not by any payoff advantage. Most strangers told to meet in New York with no time agreed will pick noon at a famous landmark, because each expects the others to pick the obvious one.
> - In a **coordination game** there are multiple stable equilibria and game theory alone cannot pick between them; salience breaks the tie. \$100 and \$98.40 are both equilibria for where to put your orders — everyone goes to \$100 because it is the one each trader expects every other trader to expect.
> - A level **holds until it breaks decisively**, then the coordination flips and the old support becomes resistance. The break is not a force giving way; it is a shared belief switching states.
> - The one rule to remember: a focal level is held up by *attention*, a fundamental level by *value* — so the focal level can vanish the instant attention moves, with **no support below it**.

In December 2017 Bitcoin ran at \$19,000 and then collapsed. Two years later, climbing back, it spent weeks grinding against \$10,000 like a swimmer hitting a glass ceiling — touch, retreat, touch, retreat. There is nothing in the blockchain, nothing in the hash rate, nothing in the supply schedule that says \$10,000 is special. It is special only because it is *round*. Every trader, every headline, every exchange dashboard fixated on the same five-digit number, and so every stop-loss, every take-profit, every "I'll buy the dip at ten grand" limit order piled up around it. The level mattered because everyone agreed it mattered.

Now flip the question. Suppose I tell you to meet a stranger in New York City tomorrow, but I never tell you *where* or *when*. You have to show up at the same place and time as someone you have never met and cannot call. Most people, faced with this puzzle, pick noon at the information booth in Grand Central Terminal. Not because noon is correct — there is no correct — but because it is the *obvious* choice, the one you expect the other person to expect you to pick. This is Thomas Schelling's famous thought experiment, and the answer he found in 1960 has a name: a **focal point**. The diagram below is the mental model for this whole post — a round price is a focal point, and a focal point is a coordination trap that closes on itself.

![Round number becomes a self-fulfilling focal price feedback loop](/imgs/blogs/schelling-points-and-focal-prices-round-numbers-and-obvious-levels-1.png)

The market version of "noon at Grand Central" is "\$100." When a stock approaches a round number, a cryptocurrency approaches \$100,000, an index approaches 5,000, traders who have never spoken and never will, who hold opposite views and opposite positions, nonetheless converge their attention and their orders on the same price. That convergence is what *makes* the level a level. This post builds the idea from zero: what a Schelling point is, why coordination games have many equilibria and salience picks one, why round numbers and obvious levels become the market's focal points, why orders cluster there, why a level holds until it suddenly does not, how to tell a focal level from a fundamentally-justified one, and — because this is a trading series — how you actually play around them without becoming the person whose stop got hunted.

## Foundations: coordination without communication

Start with the most basic distinction in game theory, because everything here depends on it. Some games are about *conflict* — what is good for me is bad for you, and we are pulling in opposite directions. Poker is like this; so is a short squeeze, where the shorts and the longs want exactly opposite outcomes. Other games are about *coordination* — we both want the same thing, namely to end up doing the *same* thing as each other, and the only problem is making sure we pick the same one. Two people walking toward each other in a hallway both want to not collide; they just need to agree, without speaking, on who steps left and who steps right.

Let me define the terms precisely, because we will use them all post.

- A **player** is anyone whose choice affects the outcome — here, every trader deciding where to rest an order.
- A **strategy** is a complete plan of action: "I will put my buy limit at \$100" is a strategy.
- A **payoff** is what you get from a combination of everyone's choices — a fill, a profit, a missed trade.
- A **Nash equilibrium** (named for John Nash) is a combination of strategies where *no single player can do better by changing their own choice alone*, given what everyone else is doing. It is a stable resting point: a configuration nobody wants to unilaterally walk away from. We built this idea in depth in [Nash Equilibrium, Best Response, and the Price as a Truce](/blog/trading/game-theory/nash-equilibrium-best-response-and-the-price-as-a-truce); here we only need the one-line version.

Now the crucial fact that this entire post hangs on: **a coordination game has more than one Nash equilibrium.** In the hallway, "you-left, me-right" is an equilibrium (neither of us wants to switch and cause a collision), and so is "you-right, me-left." Both are perfectly stable. Game theory, by itself, *cannot tell you which one you will land in.* The math is silent. Both are equally valid solutions.

This silence is a genuine problem, and it is the problem Schelling set out to solve. If two rational players both know the full game, both know the other is rational, both know that both know — and there are still two equally good answers — how do they actually coordinate? Pure logic deadlocks. You need something *outside* the logic to break the tie. Schelling's answer: people use **salience**. They pick the option that *stands out*, the one that is psychologically prominent, the one each player expects the other to find obvious. He called such an option a **focal point**, and in his honor we now call it a **Schelling point**.

The deep move here is that a Schelling point works *because of shared expectation, not because of any payoff advantage.* You go to Grand Central at noon not because you prefer Grand Central — you might hate it — but because you expect the stranger to go there, because you expect them to expect you to go there, because it is the obvious one. The focal point is selected by a tower of mutual expectation, the same "I-know-that-you-know-that-I-know" structure we unpack in [Common Knowledge and "I Know That You Know That I Know"](/blog/trading/game-theory/common-knowledge-and-i-know-that-you-know-that-i-know). Salience is just the thing that lets everyone's expectations rendezvous on the same answer without a single word being exchanged.

Three properties of Schelling points matter for markets, and we will lean on each one:

1. **Salience is about prominence, not optimality.** The focal choice is rarely the *best* choice; it is the *obvious* one. \$100 is not the "right" price for anything. It is just round.
2. **Focal points are self-reinforcing.** The more people expect a point to be focal, the more focal it becomes, because each person's expectation that others will gather there is precisely the reason they gather there too.
3. **Focal points are fragile in a specific way.** They hold only as long as the shared expectation holds. If the obvious choice *stops* being obvious — if attention shifts — the coordination collapses, often suddenly, because everyone re-coordinates somewhere else at once.

Hold those three properties. They explain why support and resistance "works," why round numbers attract orders, why levels hold until they break decisively, and why the break is so violent when it comes.

### What Schelling actually found

It is worth pausing on the evidence, because the focal-point idea sounds almost too neat to be true, and Schelling tested it. In his 1960 book *The Strategy of Conflict* he posed coordination puzzles to ordinary people and recorded their answers. The New York meeting puzzle is the famous one — a large majority picked Grand Central, and of those, an overwhelming majority picked noon, even though any time and place would have been logically "valid." He asked people to name "heads or tails" hoping to match a partner: about 86% said heads. He asked them to pick a positive number to match a stranger: a huge share said "1." He asked them to split \$100 with a partner where they only get the money if their two proposed splits add up to exactly \$100, with no communication: the dominant answer was 50/50, even though "60/40 in my favor" would have paid more *if* the partner happened to offer the matching 40/60.

That last one is the deepest, because it shows salience beating self-interest. A 50/50 split is not the *greedy* answer — you would rather have 60 — but it is the *focal* answer, the one each player expects the other to expect, the unique split that is "obviously fair." People sacrifice the better-for-me equilibrium to land on the one they can actually reach without talking. The lesson for markets is exact: traders gravitate to \$100 not because \$100 is the price they would individually choose, but because it is the one each can count on the others choosing. Coordination beats optimization when you cannot communicate, and in a market of millions of anonymous strangers, *you can never communicate.*

### The tower of expectations, in one paragraph

Here is the mechanism stated as carefully as I can. You go to the focal point because you expect others to. You expect others to because *they* expect others to. They expect others to because — and this regress has to bottom out somewhere — the point is *manifestly, obviously* the one a reasonable stranger would pick. Salience is what gives the regress a floor: a round number is so plainly the obvious choice that everyone can stop reasoning at "it's obvious" and trust that everyone else stopped there too. This is precisely the [common-knowledge](/blog/trading/game-theory/common-knowledge-and-i-know-that-you-know-that-i-know) structure — not just "I know the level," but "I know that you know that I know the level" — and it is why a *public, visible* number (one printed on every screen, computable identically by everyone) becomes focal while a private one never can. A level you alone noticed is worthless; a level *everyone* noticed *and everyone knows everyone noticed* is a Schelling point.

## The coordination game where salience picks the winner

Let me make the abstract concrete with the actual game traders are playing, with real payoffs you can compute. Two traders — call them You and Them — each have to decide *which price level to rest their orders at*. To keep it clean, give each one two choices: cluster at the round number \$100, or cluster at some unround level, say \$98.40 (a price with no special property at all). Neither can talk to the other. The payoff structure is the heart of coordination: **you do well only if you pick the same level as the other trader**, because a level only "works" — only produces a bounce you can trade — if enough orders gather there to actually move price.

The matrix below shows the payoffs. I built it with the series' `nash_2x2` solver so the equilibria are computed, not asserted: the row player (You) earns 10 if you both pick \$100, 8 if you both pick \$98.40, and 0 if you split (no cluster forms, the level does nothing, your order sits there ignored). The column player (Them) faces the mirror image.

![Coordination game payoff matrix where both diagonal cells are equilibria](/imgs/blogs/schelling-points-and-focal-prices-round-numbers-and-obvious-levels-2.png)

Run the solver on this and it returns **two pure Nash equilibria** — the top-left cell (both at \$100, payoff 10/10) and the bottom-right cell (both at \$98.40, payoff 8/8). Both are stable: in either one, if you alone deviate, you drop to 0. Game theory says both are valid solutions and *stops there.* It has no way to prefer 10 over 8 as the prediction, because each is a self-consistent resting point. (There is also a messy mixed equilibrium, which we will compute in a moment, and it is the *worst* outcome of all.)

So which one do real traders land in? The \$100 one — every time — and not because its payoff is higher (though here it is; that is a convenience, not the reason). They land there because \$100 is the **focal point**. It is round, it is the number every headline prints, it is the price each trader expects every other trader to be watching. The salience of \$100 is what selects the equilibrium that the math left undetermined. This is Schelling's insight in its purest market form: *the round number does not win because it is correct; it wins because it is obvious, and obviousness is what lets a crowd coordinate without talking.*

#### Worked example: the coordination loss of *not* having a focal point

Suppose there were no obvious level — two equally plausible prices, \$100 and \$98.40, and no salience to break the tie. Now the traders have nothing to anchor on, so each one *randomizes*: this is the mixed equilibrium the solver finds. With these payoffs, each trader picks \$100 with probability 4/9 ≈ 44.4% and \$98.40 with probability 5/9 ≈ 55.6% (the odd level gets *more* weight because its payoff is lower, which is exactly the indifference condition that makes mixing an equilibrium). What is the expected payoff when both randomize like this?

You coordinate at \$100 only when *both* of you happen to pick it: probability 0.444 × 0.444 ≈ 0.198, worth 10 → contributes about \$1.98. You coordinate at \$98.40 when both pick it: 0.556 × 0.556 ≈ 0.309, worth 8 → contributes about \$2.47. The rest of the time (about 49% of the time) you split and get nothing. Total expected payoff ≈ \$1.98 + \$2.47 ≈ **\$4.44 per trader.**

Compare: when a focal point *does* exist and pulls everyone to \$100, each trader earns **\$10.** The Schelling point is worth 10 − 4.44 ≈ **\$5.56 of pure coordination value per trader** — value created out of nothing but a shared, salient number. The intuition: a focal point is not a constraint on what you can do; it is a free coordination device that lets a crowd agree on a level it could never have negotiated.

That \$5.56 is, in a real sense, *why round-number support and resistance pays.* The level concentrates liquidity and attention that would otherwise scatter, and concentrated attention is tradeable.

## Why round numbers, specifically

So far "salient" has been hand-waved. Let me be precise about *which* prices become focal, because the answer is both more specific and more useful than "round numbers."

A price becomes focal when it satisfies two conditions. First, it must be **easy to name and recall** — a number a person can hold in their head, repeat in a sentence, type into an order ticket without checking. \$100 qualifies; \$98.4137 does not. Second, and more importantly, it must be **common knowledge that it is salient** — not just obvious to *you*, but obvious that it is obvious to *everyone*, so that you can expect the whole crowd to converge there. Round numbers nail both. They are the universal default, the price a stranger would guess, the market's "noon at Grand Central."

But round numbers are not the only focal prices. Anything that is both memorable and publicly visible becomes a candidate. The four big families are below.

![Grid of focal levels traders watch round numbers prior highs moving average option strikes](/imgs/blogs/schelling-points-and-focal-prices-round-numbers-and-obvious-levels-4.png)

- **Round numbers.** \$100, \$1,000, \$50, Bitcoin \$100,000, the S&P at 5,000, gold at \$2,000, EUR/USD at 1.10. The roundest numbers (whole thousands, whole hundreds) are the strongest; half-round levels (\$50, \$2,500) are next; quarter levels weaker still. The more zeros, the more focal.
- **Prior highs and lows.** The all-time high, last week's low, the level a stock "double-topped" at. These are focal because the chart *printed* them — they are a shared, visible historical fact every trader can see on the same screen. An all-time high is the most-watched number a stock has.
- **The 200-day moving average** (and the 50-day). A moving average is the average price over the last N days; the 200-day is the canonical "is this a bull or bear market" line. It is focal not because the math is magic but because *every charting app draws the same line in the same place* — it is genuinely common knowledge, computable identically by everyone.
- **Big option strikes.** A *strike* is the price at which an option can be exercised. When a huge number of contracts sit at one strike, that price becomes a magnet into expiry — the famous "max-pain" or "pinning" effect, which we will dissect with dealer hedging below.

Notice the common thread: every focal level is *public and shared.* You can compute the 200-day average yourself and get the same number I get. You can see the all-time high on your chart and know I see the identical figure. That shared visibility is what lets the crowd's expectations rendezvous — it is the market's substitute for being able to phone the stranger and agree on a meeting spot.

#### Worked example: how much "rounder" makes a level more focal

Round-number clustering is measurable, and the effect scales with roundness. In a stock trading near \$100, suppose the *baseline* resting depth at an ordinary, unround tick (say \$99.40) is 8 units of orders. Empirically, depth at the whole-dollar level (\$100.00) runs many multiples of that, the half-dollar (\$100.50) a smaller multiple, and the quarter-dollar levels smaller still. Putting rough numbers on it: if the whole-dollar level carries ~100 units, the half-dollar ~48, the quarter-dollars ~22, and ordinary ticks ~8–11, then the whole-dollar tick holds roughly **100 ÷ 9 ≈ 11×** the depth of an ordinary tick.

The chart below shows this shape: tall spikes of resting orders at \$99, \$100, \$101, thin grass everywhere in between.

![Bar chart of resting order depth spiking at whole-dollar prices](/imgs/blogs/schelling-points-and-focal-prices-round-numbers-and-obvious-levels-3.png)

Why does it matter that the whole-dollar tick holds 11× the orders? Because that is *liquidity you can predict will be there.* If you want to buy a large position with minimal slippage, you lean on the round number — there is a wall of resting sell limits at \$100 to fill against. If you are short and want a logical place to cover, \$100 is where the bids are thick. The clustering is self-reinforcing: traders put orders at \$100 *because* they expect liquidity there, and the liquidity is there *because* traders put orders at \$100. The intuition: the round number is not a price with special physics — it is a Schelling point for *where to leave your orders*, and a crowd that all leaves orders in the same place builds a real wall.

This clustering is documented, not folklore. Carol Osler's work on currency markets (2003) found that take-profit and stop-loss orders cluster heavily at round numbers, and that price reversals concentrate at those levels precisely as a result. Studies of equity limit-order books (Bhattacharya, Holden, and Jacobsen, 2012, among others) find the same round-number "barrier" effect. The orders genuinely pile up where the chart above shows them piling up.

### Why humans round in the first place

There is a layer beneath the coordination story worth naming, because it explains *why* round numbers are the universal default rather than some arbitrary convention. Human beings are bad at precision and good at landmarks. When you decide to take profits on a winning trade, you do not compute that your optimal exit is \$103.47; you think "I'll get out around a hundred and three" and you round to a number you can hold in your head — \$103, or more likely the cleaner \$105 or \$100. This is *cognitive economy*: round numbers are cheaper to store, recall, and communicate. Behavioral researchers call the tendency the "round-number heuristic," and it shows up everywhere from salary negotiations (we ask for \$100,000, not \$98,750) to speed limits to the prices on a menu.

Now stack that individual habit across a million traders. Each one independently rounds their target to the nearest clean number. Because they all round to the *same* clean numbers — there is only one \$100 — their independent, uncoordinated decisions *pile up at the same prices.* No trader is trying to coordinate with any other; they are each just being lazy in the same direction. And that accidental alignment is what creates the cluster. The round number is focal not because traders agreed to make it focal, but because the universal human habit of rounding *guarantees* they will land on it together. Salience here is not a choice; it is a property of how minds store numbers. The market's focal points are, at bottom, the fingerprints of human cognitive limits stamped onto the order book.

## Why support and resistance "works" — and the honest caveat

Here is where the focal-point lens does something that pure technical analysis cannot: it explains *why* a horizontal line on a chart can matter, while also drawing a hard boundary around how much it can matter.

The traditional story is that support is a price where "buyers step in" and resistance a price where "sellers appear," as if the level exerts some force. The honest, game-theoretic story is different and better: a level is support because *orders cluster there*, and orders cluster there because *it is a focal point everyone watches.* There is no force. There is a coordination of expectations that concentrates real buy and sell orders at a memorable price, and that concentration produces the bounces that, after the fact, look like the level "holding."

This means support and resistance is partly **self-fulfilling.** Traders watch \$100, so they put buy orders just above it and stop-losses just below it; the buy orders create the bounce; the bounce "confirms" that \$100 is support; the confirmation draws more traders to watch \$100. Round and round. The level works because people believe it works and act on the belief — exactly the reflexive loop we develop in [Reflexivity: Markets That Watch Themselves](/blog/trading/game-theory/reflexivity-markets-that-watch-themselves), and exactly the mechanism the honest technical-analysis treatment lays out in [Support and Resistance: Why Price Levels Exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist).

But — and this is the caveat that separates a thoughtful trader from a chart mystic — **self-fulfilling does not mean reliable.** A focal level holds only as long as the coordination holds. It is held up by attention, and attention is a fickle, fast-moving thing. The level has no anchor in value, no reason it *must* hold. It holds until enough people decide it won't, at which point it gives way completely, because the same coordination that built it now works in reverse.

#### Worked example: the math of why a focal level is not "strong"

People talk about a level being "strong support." Let me show why that phrase is misleading with the order-cluster numbers. Say \$100 has 100 units of resting buy orders (our 11× cluster) versus 8 at an ordinary tick. That sounds like a fortress. But suppose a seller arrives with 250 units to dump — a large institutional liquidation. The 100 units of bids at \$100 absorb the first 100; the next 150 units have *nothing focal to hit* until the next round number, because the in-between ticks hold only ~8 each.

So the price does not gently sink — it *air-pockets.* It fills the \$100 wall, then falls through the thin grass below it in a rush, because there is no liquidity between focal levels to slow it down. The 11× concentration that made \$100 look strong is exactly what makes the move *below* \$100 so violent: all the liquidity was at the level, none of it just beneath. A focal level is not a thick cushion all the way down; it is a thin shelf with a cliff behind it. The intuition: clustered liquidity is a trap-door, not a floor — it holds your weight right up until it doesn't, and then there is nothing underneath.

This is the single most important practical fact about focal levels, and we will return to it in the playbook: **the strength of a focal level and the violence of its break are the same phenomenon.**

## How a level holds, then breaks decisively

Watch a focal level over time and you see a characteristic pattern: the price tests the level repeatedly and bounces, tests and bounces, building the appearance of an impregnable floor — and then one test slices straight through and keeps going. The level did not "weaken." The *coordination flipped.*

![Price path holding a focal level on repeated tests then breaking decisively below it](/imgs/blogs/schelling-points-and-focal-prices-round-numbers-and-obvious-levels-5.png)

The chart above traces the life cycle. In the green phase, every dip into \$100 meets the cluster of buy orders, bounces, and confirms the level. Each bounce is a small coordination success: the buyers who gathered there got their fill and their bounce, the sellers who faded the level got their reversal. The level is a *stable equilibrium*, and everyone is at their best response — buyers buy the dip, sellers sell the rally, and the truce holds at \$100.

Then comes the red phase. Something tips the balance — a wave of selling larger than the cluster can absorb, a piece of news, a big liquidation, or just the slow exhaustion of buyers who have already bought. Price closes *decisively* below \$100, not by a tick but by enough that it is unambiguous. And now the coordination inverts. Everyone who was using \$100 as "the floor" re-coordinates: the buyers who were leaning on it pull their bids and flip to sellers; the stop-losses resting below \$100 trigger and add to the selling; the traders who fade levels now fade \$100 *from below* as resistance. The same focal number that organized the buying now organizes the selling. Support has become resistance — what chartists call a "role reversal," and what game theory calls *the crowd re-coordinating on the other side of the level.*

The word "decisively" is doing real work. A one-tick poke below \$100 does not flip the coordination, because everyone knows a one-tick poke is noise and expects everyone else to know it too — so the focal point survives. It takes a move large enough that it becomes *common knowledge that the level broke*, large enough that each trader expects every other trader to now treat \$100 as broken. The break has to be salient itself. This is why traders wait for a "confirmed" break (a close beyond the level, a retest that fails) before trusting it: they are waiting for the new coordination to become common knowledge, not for any physical confirmation.

#### Worked example: the break-and-retest, in numbers

Here is the cleanest tradeable pattern around a focal level, with concrete prices. A stock has held \$100 as support for three weeks — call it four clean tests and bounces. On the fourth-and-a-half test, it closes at \$98.50, decisively below. The break is now common knowledge: \$100 has flipped from floor to ceiling.

A few sessions later the stock drifts back *up* to \$99.80 — a "retest" of the broken level from below. What happens? The traders who got trapped long at \$100 and are now underwater are desperate to exit "at breakeven," so they sell into \$100. The traders who shorted the break add to their position on the bounce. The result: the retest of \$99.80–\$100 gets sold, and the stock rolls back over toward \$97. A trader who *shorted the retest* at \$99.80 with a stop at \$100.60 (just above the now-resistance level) and a target of \$97 risked \$0.80 to make \$2.80 — a **3.5-to-1 reward-to-risk** trade, with a clean invalidation: if the stock reclaims \$100.60, the coordination flipped *back* and the short is wrong.

The intuition: the break-and-retest is so reliable-looking precisely because it is two coordination events stacked — first the crowd agrees the level broke, then the trapped longs and fresh shorts agree to sell the bounce. You are not trading a line on a chart; you are trading the predictable behavior of two groups re-coordinating around a number everyone is watching.

## Focal level versus fundamental level

This is the distinction that keeps you from getting killed: telling a focal level apart from a fundamentally-justified one. They look identical on a chart — both are horizontal lines that price respects — but they are held up by completely different things and they fail in completely different ways.

![Before and after comparison of a focal level held by attention versus a fundamental level held by value](/imgs/blogs/schelling-points-and-focal-prices-round-numbers-and-obvious-levels-6.png)

A **focal level** is held up by *attention.* \$100 is support because everyone watches \$100. There is no buyer who *needs* to own the stock at \$100 rather than \$99.50 for any reason connected to its value — they are there only because the number is salient. The level works as long as the coordination holds, and the moment attention shifts, it can evaporate. And crucially, *there is nothing below it* — once \$100 breaks, the next focal level might be \$90, with thin liquidity all the way down, because the support was never about value.

A **fundamental level** is held up by *value.* Suppose a profitable company trades at \$100, and at \$80 its dividend yield, its earnings multiple, or its breakup value becomes so attractive that real long-term buyers — pension funds, value investors, the company itself buying back stock — step in *because they want to own the asset at that price.* That \$80 support is not a coordination trick; it is genuine demand anchored to cash flows. It holds because buyers have a *reason* independent of what other traders are doing, and if price overshoots below it, the same value pulls it back up. A fundamental level re-emerges; a focal level vanishes.

The practical test is to ask: **"Who is buying here, and why?"** If the answer is "traders, because the number is round / it is the prior low / it is the 200-day," you have a focal level — tradeable but fragile, with a cliff behind it. If the answer is "real owners, because the asset is cheap relative to its cash flows," you have a fundamental level — sturdier, and likely to be defended on a retest. Most chart levels are focal. A few are fundamental. The ones that are *both* — a round number that also happens to be a value level — are the strongest of all, because attention and value point the same way.

There is a second, sharper way to read the distinction: ask *what happens after a break.* A focal level, once broken, gives you no information about where price goes next except "to the next focal level," because the support was never about value — it was about a number, and that number is now behind you. A fundamental level, once broken, *gets cheaper*, which means the very buyers who were defending it have more reason to buy, not less, so an overshoot below it tends to be bought back. Watch the behavior on the *retest from below*: a focal level that broke will be sold on the retest (trapped longs exiting, role reversal), while a fundamental level that briefly broke will often be *reclaimed* (value buyers stepping back in). The retest is the market telling you which kind of level you were standing on all along — and that information is worth waiting for before you commit size.

#### Worked example: distinguishing the two at Bitcoin \$100,000 versus a stock at book value

Bitcoin at \$100,000 is a *pure* focal level. There is no cash flow, no book value, no dividend yield that says \$100,000 is fair — Bitcoin's "value" is itself a coordination of beliefs. So \$100,000 is support or resistance *only* because it is the roundest, most-watched number in the asset's history. When it broke above \$100,000 in late 2024, the move was explosive precisely because the level was pure attention: once the coordination flipped from "ceiling" to "floor," there was nothing fundamental to slow it. Pure focal levels give the cleanest break-and-go moves *and* the nastiest air-pockets.

Contrast a boring industrial company trading at \$30 with a book value (net assets per share) of \$28 and a 5% dividend. If it falls to \$28, real buyers — value funds, the company's own buyback — have a *reason* to step in: they are buying \$1 of assets for \$1 and collecting a rising yield. That \$28 is a fundamental level. It can still break in a panic, but it tends to be *re-bought*, because the value did not disappear; if anything it got cheaper. Same-looking line on the chart; opposite character. The intuition: ask whether the buyers at the level are there for the *number* or for the *asset* — the answer tells you whether you are standing on a shelf or on solid ground.

## Pinning: when a big option strike becomes the focal price

The option market produces the most mechanical focal-price effect there is, and it is worth its own section because the coordination there is enforced by *hedging math*, not just attention. It is called **pinning**, and the level it pins to is the **max-pain strike** — the strike price where the largest number of option contracts expire worthless, which is also, usually, the strike with the most open interest.

Here is the chain, built from the bottom. An *option* is a contract to buy (call) or sell (put) at a fixed *strike* price. The dealers who sell options to the public are left holding the other side, and they neutralize their risk by *delta-hedging* — buying or selling the underlying stock to stay market-neutral. The key fact: when dealers are **long gamma** (which they are when the public has net *bought* options at a strike), their hedging is *stabilizing.* As the stock rises toward the strike, their hedge requires them to *sell* stock; as it falls toward the strike, their hedge requires them to *buy.* They are mechanically leaning against every move — selling strength, buying weakness — and that pins the price toward the strike. We unpack the full dealer mechanics in [Dealer Gamma, Charm, and Vanna: How Options Flows Move the Spot](/blog/trading/options-volatility/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot); here the point is just that the pin is a *coordination enforced by arbitrage*, the firmest kind.

![Price coiling toward a high open interest option strike into expiry](/imgs/blogs/schelling-points-and-focal-prices-round-numbers-and-obvious-levels-7.png)

The chart shows the signature shape: into expiry, the price *coils* — its range tightens — and gravitates toward the big strike, here \$50. Early in the week the stock wanders freely; as Friday's expiry approaches and the dealers' gamma grows, their hedging tightens its grip, realized volatility collapses, and the stock gets "pinned" near \$50 at the close. This is not a metaphor for attention; it is a genuine, measurable price-suppression effect that strengthens as time decay accelerates.

#### Worked example: why the pin tightens into the close

Take a stock at \$50.30 on the morning of monthly expiry, with enormous open interest at the \$50 strike, mostly bought by the public (dealers long gamma). The dealers' aggregate hedge requires them to hold a position whose *delta* — sensitivity to the stock — changes fast near the strike. Suppose collectively they must sell 5,000 shares for every \$0.10 the stock rises above \$50 and buy 5,000 shares for every \$0.10 it falls below.

Now the stock ticks up to \$50.40: dealers mechanically sell ~20,000 shares to re-hedge, pushing it back down. It dips to \$49.90: dealers buy ~25,000 shares, pushing it back up. Every move toward the edges is met by hedging that shoves it back to \$50. As expiry nears, gamma rises (the option's delta swings ever more sharply near the strike), so the required hedge per tick *grows* — the spring gets stiffer — and the realized range collapses from maybe \$1.50 on Monday to \$0.20 in the final hour. The stock closes at \$50.02, pinned.

What is the magnitude in the real world? Ni, Pearson, and Poteshman (2005) studied U.S. equity options and found that on expiration Fridays, stocks with listed options cluster at strike prices far more than chance would predict — the pinning effect moved expiration-day returns by an economically meaningful amount and was strongest in the names with the most open interest. The intuition: a heavily-traded strike is a focal price with *teeth* — the coordination is enforced by dealers who have no choice but to defend it, right up until the options expire and the spell breaks.

When do pins fail? When dealers are *short* gamma (the public net *sold* options at that strike) — then the hedging is destabilizing and amplifies moves instead of damping them, and instead of a pin you get an acceleration *away* from the strike. The sign of dealer gamma flips the focal strike from a magnet into a launchpad. Knowing which regime you are in is the whole game.

## Common misconceptions

**"Round numbers have some intrinsic technical power."** No. \$100 has no physics. Its power is entirely borrowed from the fact that everyone watches it — it is a coordination device, not a force. Strip away the shared attention — an asset that, hypothetically, no human ever looks at — and the round number would mean exactly nothing. The level is real because the watching is real; that is the whole mechanism, and the diagram of the self-fulfilling loop above is the complete explanation.

**"If the level is strong, it will hold."** This confuses the *appearance* of strength (thick clustered liquidity) with *durability* (an anchor in value). As the air-pocket worked example showed, the thick liquidity at a focal level is exactly what makes its break violent — all the orders are *at* the level, none just below it. A focal level is most dangerous when it looks strongest, because that is when the most traders are leaning on a shelf with a cliff behind it.

**"Support and resistance is either magic or nonsense."** Both extremes are wrong. It is neither a mystical force nor pure superstition; it is a *self-fulfilling coordination* among traders watching the same focal prices. That makes it genuinely real (orders do cluster, bounces do happen) but only conditionally reliable (it holds only while the coordination holds). The right stance is "real but fragile, and only as a focal point — never as a force."

**"I'll know in advance when the level is going to break."** You won't, and the coordination math says why. The break happens when shared belief flips, and belief flips are nearly discontinuous — the level holds, holds, holds, then goes, because everyone re-coordinates at roughly the same moment. There is no smooth weakening to read in advance. What you *can* do is define your invalidation at the level and let the break tell you, rather than predict it. This is the same "I can't time the stampede, so I pre-commit my exit" logic that governs every coordination trade, from [bank runs](/blog/trading/game-theory/bank-runs-as-coordination-games-diamond-dybvig-and-svb) to crowded-trade unwinds.

**"Pinning means I can sell options at the big strike for free money."** Dangerous. Pinning works when dealers are *long* gamma; if they are short gamma at that strike, the same setup amplifies moves and your sold options blow up. And even a long-gamma pin can shatter on news. The pin is a *conditional* focal point — conditional on the gamma sign and on no exogenous shock — not a guarantee.

## How it shows up in real markets

**Bitcoin and the \$10,000 / \$100,000 levels.** Bitcoin is the purest laboratory for focal prices because it has *no* fundamentals to anchor on — its value is entirely a coordination of belief. So its support and resistance are almost perfectly round. The \$10,000 level acted as a ceiling for much of 2018–2020, getting tested repeatedly before finally breaking. \$100,000, the roundest milestone in the asset's history, capped the market through 2024 and became a self-fulfilling battle line; when it finally broke above in December 2024 the move was explosive, exactly as a pure focal-level break should be — once the coordination flipped, nothing fundamental stood in the way.

**The S&P 500 at 5,000.** When the S&P first approached 5,000 in early 2024, the round number drew enormous attention — every financial headline counted down to it, options activity concentrated around it, and the index paused there before pushing through. There is nothing in corporate earnings that makes 5,000 special; it is special because it is round and because the entire market agreed to watch it. The countdown coverage *was* the coordination mechanism, making 5,000 common knowledge as a focal price.

**Round-number clustering in foreign exchange.** Carol Osler's research on currency order flow (2003) is the canonical evidence: take-profit orders cluster at round numbers, stop-loss orders cluster just *beyond* round numbers, and exchange rates reverse at round numbers more often than chance allows. The mechanism is exactly the focal-point story — dealers and traders place their orders at the obvious numbers, the orders cluster, and the clusters produce the reversals. The clustering is so reliable that it has been a documented feature of FX microstructure for two decades.

**Option pinning on expiration Fridays.** Ni, Pearson, and Poteshman (2005) showed that U.S. stocks with listed options close *at* their strike prices on expiration Fridays far more often than chance predicts, with the effect concentrated in names with high open interest — the fingerprint of dealer gamma pinning. Traders watch the max-pain strike into monthly and weekly expiries precisely because it acts as a focal magnet; the largest strikes in heavily-traded names (think the round strikes in mega-cap tech) regularly pin into the close.

**The all-time high as a focal level.** When an index or a leading stock makes a new all-time high and then pulls back, the old high becomes a battle line on the retest, because it is the single most-watched number the asset has — everyone can see the same prior peak on the same chart. Breakout traders buy a decisive move above it; faders sell the first touch. The all-time high is focal not because of roundness but because the chart *printed* it and made it common knowledge.

**The IPO price.** When a company goes public at, say, \$30, that price becomes an instant focal level — it is the one number every participant knows and every headline cites ("trading above/below its IPO price"). Insiders, early investors, and traders all anchor on it, so the stock often gravitates to, and battles around, the offer price for weeks. It is a focal level created not by roundness or history but by a single, universally-known reference number.

**The unemployment rate and policy "lines in the sand."** Focal points are not only price levels. When a central bank says it will act "if inflation stays above 2%," that 2% becomes a coordination point for the entire bond market — everyone watches the same threshold, so everyone trades around it, and the number acquires power far beyond its arbitrary origin. The 2% target is a round, salient, publicly-announced line that lets millions of participants coordinate their expectations of policy. It is the macro cousin of \$100 support, and we trace how such public lines move markets in [Buy the Rumor, Sell the News: Public Signals and the Fed](/blog/trading/game-theory/buy-the-rumor-sell-the-news-public-signals-and-the-fed).

#### Worked example: trading the all-time-high breakout

Put numbers on the all-time-high case. A leading stock has an all-time high of \$200, set eighteen months ago, and has spent the last few weeks grinding up toward it. \$200 is doubly focal — it is both round *and* the historic peak, so attention and memory point at the same number. Here is the coordination on each side: trapped buyers from the last failed attempt at \$200 are waiting to "sell at breakeven," forming a wall of supply; breakout traders are waiting *above* \$200 to buy a decisive new high; short sellers are leaning on the level expecting another rejection.

The stock pushes to \$200 and stalls — the trapped sellers and the shorts win the first battle, and it pulls back to \$192. Then it tries again and this time *closes at \$203*, a decisive new all-time high. Now the coordination flips hard: the trapped sellers are gone (they sold into the first test), the shorts are forced to cover (buying), and the breakout crowd piles in (buying), all at once. With the overhead supply cleared and *no chart history above \$200* — the air above an all-time high is the emptiest air in the market, because nobody is trapped up there waiting to sell — the stock runs to \$215 with little resistance.

A trader who bought the decisive close above \$200 at \$203, with a stop back below \$200 (at \$199, where a failure would prove the breakout false) and a first target near \$215, risked \$4 to make \$12 — a **3-to-1 trade** with a clean invalidation: the breakout is wrong only if price falls back inside the old range. The intuition: an all-time-high break is the cleanest focal-level flip there is, because above the high there is *no* prior coordination to fight — just open sky.

## The playbook: how to play it

You now have the full game. Here is how to actually trade focal prices without becoming the liquidity everyone else is hunting.

**Who is on the other side?** At a focal level, the other side is *the crowd coordinating on the same number.* The buyers leaning on round-number support, the stops resting just beyond it, the dealers hedging a big strike, the breakout traders waiting above the prior high. Your edge is not predicting the level — everyone sees it — but understanding *what each group will do* when it is tested, and being one step ahead of the re-coordination. This is the same one-level-deeper reasoning as the [Keynesian beauty contest](/blog/trading/game-theory/the-keynesian-beauty-contest-and-level-k-thinking): don't ask where the level is, ask what everyone *else* watching the level will do, and act before they finish doing it.

**Expect a reaction *at* the level — fade or follow the cluster.** The single most reliable thing about a focal level is that *something happens there* — a bounce, a pause, a spike in volume. So plan for a reaction. The two clean stances:

- **Fade the level** (bet it holds): buy into round-number support / sell into round-number resistance, with a tight stop *just beyond* the level. You are betting the coordination holds. Small risk (the stop is close), modest reward (a bounce back into the range). Win rate high, but each win small — and the rare loss is the air-pocket break, which can be large, so size for it.
- **Follow the break** (bet it flips): wait for a *decisive* close beyond the level, then trade in the break's direction — ideally on the **break-and-retest**, entering on the failed retest of the now-flipped level (the 3.5-to-1 setup from the worked example). You are betting the coordination inverted. Lower win rate (many breaks are fakeouts), but the wins are large because, once a focal level flips, there is little liquidity in the new direction.

You cannot do both at once on the same test, so decide in advance which side of the coordination you are betting on, and let price tell you which one you are in.

#### Worked example: sizing the fade so the air-pocket can't ruin you

The fade looks like easy money — high win rate, small stop — which is exactly the trap. Let me size it honestly. Say you fade round-number support at \$100, buying with a stop at \$99 (a \$1 risk), targeting a bounce to \$103 (a \$3 reward). On paper that is 3-to-1, and the bounce comes, say, 70% of the time. But the 30% of the time it fails, the air-pocket means you may not get out at \$99 — a decisive break can gap or slide straight through the thin liquidity below the level, and you fill at \$97 instead, a \$3 loss, not \$1.

Now do the expected value. Wins: 0.70 × \$3 = +\$2.10. Losses: 0.30 × \$3 (the real, slippage-inflated loss) = −\$0.90. Net EV ≈ +\$1.20 per share — still positive, but *half* of what the naive 0.70 × \$3 − 0.30 × \$1 = +\$1.80 calculation suggested, because the air-pocket roughly tripled your assumed loss. The lesson for sizing: assume your stop on a focal-level fade will slip *past* the level, not fill *at* it, and size the position so that the slipped loss — not the textbook loss — is still a survivable fraction of your capital. The intuition: the clustered liquidity that makes the fade tempting is the same liquidity vacuum that makes its rare failure expensive, so you must size for the gap, not the stop.

**Respect the air-pocket.** Because liquidity clusters *at* focal levels and thins *between* them, a confirmed break can run fast to the next focal level with nothing to slow it. If you fade a level and it breaks, *get out immediately* — do not "give it room," because the room below is empty. The stop just beyond the level is not optional; it is the entire risk management of the trade.

**For pins, check the gamma sign first.** Around a big option strike into expiry, the strike is a magnet *only if dealers are long gamma.* If they are short gamma, it is a repellent. Never sell premium at a big strike assuming a pin without knowing which regime you are in — the destabilizing case is exactly the one that ruins you.

**Know whether you are on a shelf or on solid ground.** Before you lean on any level, ask the focal-vs-fundamental question: are the buyers here for the *number* or for the *asset*? A focal level can vanish; a fundamental level gets re-bought. Size larger and hold longer at fundamental levels; treat focal levels as fast, tactical, stop-protected trades. A level that is *both* round and fundamentally cheap is the highest-conviction setup there is.

**The invalidation is the level itself.** This is the gift of trading focal prices: the level *is* your line in the sand. If you fade support at \$100, you are wrong the instant it closes decisively below \$100 — clean, unambiguous, pre-defined. If you short a flipped level at \$99.80, you are wrong if it reclaims \$100.60. You never have to guess your exit; the coordination point that drew you in is the same point that tells you when the crowd has re-coordinated against you.

The one rule, restated: **a focal level is held up by attention, not value.** Trade it for the reaction you can reliably expect at the number, protect yourself against the break you cannot predict, and never mistake a crowd watching a round number for a floor under your feet. The level works because everyone watches it — which is exactly why it stops working the moment they look away.

## Further reading & cross-links

- [Nash Equilibrium, Best Response, and the Price as a Truce](/blog/trading/game-theory/nash-equilibrium-best-response-and-the-price-as-a-truce) — the equilibrium concept this whole post selects *between*; a coordination game is a game with multiple Nash equilibria, and a focal point is the one salience picks.
- [Common Knowledge and "I Know That You Know That I Know"](/blog/trading/game-theory/common-knowledge-and-i-know-that-you-know-that-i-know) — why a focal point requires not just that a level be obvious to you, but that its obviousness be common knowledge, so the whole crowd's expectations rendezvous.
- [The Keynesian Beauty Contest and Level-k Thinking](/blog/trading/game-theory/the-keynesian-beauty-contest-and-level-k-thinking) — the one-level-deeper reasoning that turns "where is the level" into "what will everyone watching the level do," which is the actual edge.
- [Support and Resistance: Why Price Levels Exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist) — the honest technical-analysis treatment of levels as zones, not lines, and the same self-fulfilling mechanism from the chartist's side.
- [Dealer Gamma, Charm, and Vanna: How Options Flows Move the Spot](/blog/trading/options-volatility/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot) — the full hedging mechanics behind option pinning, and how the sign of dealer gamma turns a strike from a magnet into a launchpad.

*This is educational material about market mechanisms and game theory, not individualized financial advice. Focal levels are tradeable but fragile; every level that can be faded can also break violently against you, so size and stop accordingly.*

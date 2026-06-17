---
title: "Nash Equilibrium, Best Response, and the Price as a Truce"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "The market price is a Nash equilibrium — a truce where the marginal buyer and seller are both at their best response, which is exactly what people mean when they say it is already priced in."
tags: ["game-theory", "trading", "nash-equilibrium", "best-response", "dominant-strategy", "market-microstructure", "price-discovery", "mixed-strategy", "equilibrium"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — The market price is a Nash equilibrium: a truce where the marginal buyer and the marginal seller are both already at their best response, so neither can profitably move. "It's priced in" is not a forecast — it is an equilibrium statement about beliefs.
>
> - A **best response** is your most profitable action given what everyone else is doing; a **Nash equilibrium** is a set of strategies where every player is at a best response at once, so no one can gain by deviating alone.
> - A **dominant strategy** is best no matter what others do; deleting strategies that are never best (**iterated dominance**) can sometimes solve a game outright.
> - Some games have **no pure equilibrium** (matching pennies) and the only stable play is a **mixed** one — be 50/50 or be read; others have **two** equilibria (coordination games), so stability does not pick a winner.
> - The one rule to remember: equilibrium means **stable**, not **fair** and not **efficient** — a price can be a perfectly good Nash equilibrium and still be a terrible deal for someone.

A trader I know once shorted a stock the morning of an earnings beat. The company had crushed every number — revenue, margins, guidance — and the stock dropped four percent in the first ten minutes. He was furious. "The numbers were great. How does it go *down*?" The answer is the most important sentence in trading, and almost no one says it out loud: the good news was already in the price. The market had spent three weeks bidding the stock up *in anticipation* of a beat, and when the beat arrived exactly as expected, there was nothing left to buy on. The people who wanted in were already in. The price had settled into a kind of truce — a level where the last buyer and the last seller had both decided they had no better move — and the only thing left to move it was a surprise, which a perfectly-expected beat is not.

That truce has a precise name in mathematics: a **Nash equilibrium**. It is the single most useful idea in this whole series, and it is the one that takes "who's on the other side?" from a slogan to a tool. Once you can see the price as an equilibrium, a dozen market sayings that sounded like folklore — "buy the rumor, sell the news," "it's priced in," "the market has already discounted that," "you're the sucker if you don't know who is" — turn into statements you can actually reason about and sometimes profit from.

This post builds the machinery from absolute zero. We will define a *best response*, a *dominant strategy*, *iterated dominance*, and finally the Nash equilibrium itself — every single one with a trading example and, where there is a number to compute, a number computed from a real 2×2 game (never hand-faked). Then we turn the whole thing on the market: the price is the equilibrium, "priced in" is the equilibrium of beliefs, and the reason this matters is that *your edge only exists where the equilibrium is wrong or hasn't formed yet*.

![A two by two payoff matrix with the Nash equilibrium cell highlighted in green](/imgs/blogs/nash-equilibrium-best-response-and-the-price-as-a-truce-1.png)

The matrix above is the mental model for the entire post. Two traders each pick "aggressive" or "patient." Each cell lists both players' payoffs. The green cell — top-left — is the Nash equilibrium: the one cell where, given what the other player is doing, *neither* wants to switch. Every other cell has at least one player itching to move. By the end of this post you will be able to find that cell in any small game, see why the market price sits in exactly such a cell, and know the three ways the picture gets more interesting (no equilibrium, two equilibria, and equilibria that are stable but unfair).

## Foundations: best response, dominance, and the Nash equilibrium from zero

Let me define the four building blocks before we use them. A *game*, in the technical sense, is just three things: the **players** (who is acting), each player's **strategies** (the actions available to them), and the **payoffs** (how much each player gets for every combination of actions). That is it. A trade is a game: the players are you and your counterparty, the strategies are "buy / sell / wait" or "bid \$100 / bid \$101," and the payoff is the profit or loss each of you walks away with. The whole field of game theory is the study of what rational players do in games like this — and the spine of this series is that *a trade is a strategic game against an adaptive opponent, not a bet against a coin*.

### Best response

A **best response** is your most profitable action, *given a specific choice by everyone else*. The "given" is the whole point. You do not pick a best response in a vacuum; you pick it against an assumed move by the other side. If I tell you the seller across from you has decided to sit at an ask of \$100 and never budge, your best response — your most profitable move — is well-defined: buy at \$100 if the stock is worth more than \$100 to you, and walk away otherwise. Change the seller's behavior and your best response changes.

Notice what a best response is *not*. It is not the best outcome you could dream of (you would love the seller to give you the stock for free; that is not on the menu). It is the best *you* can do, holding the others fixed. This conditional-on-others structure is what separates game theory from ordinary decision-making. In a one-player decision — should I take an umbrella? — you optimize against nature, which does not care what you do. In a game, you optimize against people who are optimizing against you, and *they will move when you move*.

### Dominant strategy

Sometimes a strategy is a best response *no matter what the others do*. We call that a **dominant strategy**, and it is a gift, because it lets you stop guessing what the other side will do. If action A pays you more than action B in every single column of the payoff matrix — against every move the opponent could make — then A *strictly dominates* B, and a rational player simply never plays B.

Here is the trading version. Suppose you are deciding whether to put a stop-loss on a position. Strategy A is "set a sensible stop"; strategy B is "no stop, hope." If a sudden 30% gap-down is possible, A protects you; if no gap comes, A costs you almost nothing (a slightly worse fill if you are stopped on noise). There is essentially no state of the world where "no stop, hope" beats "sensible stop" by enough to justify the tail risk — so for most traders, having a stop *dominates* not having one. You do not need to forecast the market to make that choice; dominance made it for you. (We will see in a moment that dominance is rarer than people think — most real choices are *not* dominant, which is exactly why the game gets hard.)

There is a sharp version and a soft version of this. **Strict dominance** means A beats B in *every* column by a real margin; a rational player will *never* play a strictly dominated strategy, full stop, and you can delete it with no hesitation. **Weak dominance** means A is at least as good as B everywhere and strictly better in at least one column — A is never worse, but in some states they tie. Weakly dominated strategies are trickier: a player *might* still play one (it never hurts against the right opponent), so deleting them can quietly discard equilibria. For the worked examples in this post we will stick to strict dominance, where the deletions are clean and the logic is airtight. The distinction matters at the desk because most "obvious" trades — hedge a known exposure, take a riskless arbitrage — are only *weakly* dominant (they tie when nothing happens, win when it does), which is precisely why they are crowded but not automatic.

### Iterated dominance

If you delete every strategy that is dominated — that is never anyone's best move — you sometimes change *other* players' best responses, because some of the moves they were guarding against can no longer happen. Then you can delete more. This round-by-round deletion is **iterated elimination of dominated strategies**, and when it shrinks the game down to a single cell, that cell is the answer.

The classic everyday version is the "guess 2/3 of the average" game (the *beauty contest*, which gets its own post in this series). For now, hold the idea: dominance is a ladder. You knock out the obviously-bad moves; that makes some of the opponent's moves obviously-bad; you knock those out too; and the game collapses.

### Nash equilibrium

Now the headline. A **Nash equilibrium** is a combination of strategies — one for each player — with the property that *every player is simultaneously playing a best response to everyone else's choice*. Equivalently: no single player can do better by unilaterally changing their move, holding everyone else fixed. It is a **mutual best response**, a configuration where everybody is already doing the best they can given what everyone else is doing, so the situation is *stable*: left alone, no one walks away from it.

That word — stable — is the entire concept, and it is what makes equilibrium the right model for a market price. A price that is not an equilibrium has someone with a profitable move (a buyer who would gladly pay more, a seller who would happily sell lower) — and the moment they make that move, the price changes. The price stops moving exactly when no one has a profitable unilateral move left. That is a Nash equilibrium, and it is what a *price* is.

One subtlety to flag now, because it trips up almost everyone: a Nash equilibrium says no one can profit by deviating *alone*. It says nothing about whether players could *all* do better by deviating *together*. The prisoner's dilemma (its own post in this series) is the canonical case: both prisoners confessing is the Nash equilibrium — neither can improve by staying silent alone — yet both would be better off if they could both stay silent. Equilibrium is about unilateral stability, not collective optimality. Hold that thought; it is why "equilibrium" and "good outcome" are different words.

It is also worth being precise about what we are *assuming* when we say players reach an equilibrium. The Nash concept assumes each player is rational (picks a best response), knows their own payoffs, and — this is the strong part — correctly anticipates the others' equilibrium strategies. That last assumption is heroic in real markets, where participants have different information, different models, and different amounts of attention. So treat "the market is in equilibrium" not as a law of nature but as a *tendency*: prices are pulled toward the level where no one has a profitable move, fast when participants are sophisticated and information is shared, slowly (or never) when they are not. The gap between the tidy equilibrium and the messy reality is, quite literally, where trading profits come from — and most of this series is about reading that gap. The places where the equilibrium assumption holds tightest (deep, liquid, heavily-arbitraged markets like front-month S&P futures) are the places with the least edge; the places where it holds loosest (illiquid small-caps, freshly-listed tokens, stressed markets where forced sellers dominate) are where the gaps live.

A pure-strategy Nash equilibrium is one where each player picks one definite action. A **mixed-strategy** equilibrium is one where players randomize over actions with specific probabilities — and as we will see, some games have *only* a mixed equilibrium and no pure one at all. Let me now make every one of these definitions earn its keep with numbers.

## Best response, made concrete: who moves and who is stuck

The cleanest way to *find* a Nash equilibrium in a small game is the best-response method: for each thing the opponent might do, mark your best reply; for each thing you might do, mark the opponent's best reply; the cells where both marks land on the same square are the equilibria. A cell is a Nash equilibrium if and only if it is a best response for *both* players at once.

Let me walk the cover game. Two traders — call them You and Them — each choose Aggressive (cross the spread, take liquidity, move fast) or Patient (post a resting order, wait). The payoffs (in some abstract units of profit) are:

| | Them: Aggressive | Them: Patient |
|---|---|---|
| **You: Aggressive** | You 3, Them 3 | You 1, Them 0 |
| **You: Patient** | You 0, Them 1 | You 2, Them 2 |

To find your best responses: if Them plays Aggressive (left column), you get 3 from Aggressive vs 0 from Patient — so Aggressive is your best reply. If Them plays Patient (right column), you get 1 from Aggressive vs 2 from Patient — so Patient is your best reply. Your best response depends on what they do. Now Them: if You play Aggressive (top row), they get 3 vs 1, so Aggressive. If You play Patient (bottom row), they get 1 vs 2, so Patient. The cell where both players are best-responding is **(Aggressive, Aggressive)** at the top-left — 3 and 3 — the green cell. Neither can profitably move: if you switched to Patient you would drop from 3 to 0. That is the truce.

#### Worked example: checking that no one wants to deviate

Let me verify the equilibrium the way the definition demands — by testing every possible unilateral deviation and showing each one loses. Start at the Nash cell, (Aggressive, Aggressive), where you each earn \$3.

- You deviate to Patient (Them stays Aggressive): you move from the top-left cell to the bottom-left cell. Your payoff falls from \$3 to \$0. A loss of \$3. You won't do it.
- Them deviates to Patient (You stay Aggressive): they move from top-left to top-right. Their payoff falls from \$3 to \$0. A loss of \$3. They won't do it either.

Both deviations lose, so the cell is stable — it is a Nash equilibrium. Now contrast the *bottom-right* cell, (Patient, Patient), where you each earn \$2. Is that an equilibrium? Test it: if you deviate to Aggressive while Them stays Patient, you move from bottom-right (\$2) to top-right (\$1) — wait, that is a loss, so maybe it is stable? Check the other player too: if Them deviates to Aggressive while You stay Patient, they move from bottom-right (\$2) to bottom-left, earning \$1 — also a loss. So (Patient, Patient) — earning \$2 each — *is also stable*: it is a second Nash equilibrium. The intuition: a game can have more than one truce, and "stable" does not mean "the best one."

That worked example revealed something the cover figure hinted at: this particular game has two pure equilibria, not one — we will return to multiple equilibria shortly. First, the case where the equilibrium is a single crossing point, which is the picture most people carry in their heads.

![Best response curves for matching pennies crossing at the mixed Nash equilibrium](/imgs/blogs/nash-equilibrium-best-response-and-the-price-as-a-truce-2.png)

The chart above plots **best-response curves** for a different game — matching pennies, which we will define in a moment — where each player picks not an action but a *probability*. Your best response (blue) to their probability mix `q` jumps from "always Tails" to "always Heads" at `q = 0.5`; their best response (lavender, dashed) to your mix `p` jumps the other way. The two curves cross at one point: `p = q = 0.50`, the red dot. *That crossing is the Nash equilibrium.* This is the deep geometric picture of Nash: equilibria are where best-response curves intersect, where each player's optimal reply is consistent with the other's. John Nash's famous theorem is that for any finite game, at least one such crossing always exists — possibly only in mixed strategies, but it always exists. That guarantee is why we can always *talk* about "the equilibrium" of a market.

## The market price is the equilibrium

Now the payoff for all this machinery. The price of anything that trades — a stock, a bond, a barrel of oil, a token — is a Nash equilibrium. Here is the argument, slowly.

At any instant there is a population of potential buyers, each with a private maximum they would pay (their *reservation value*), and a population of potential sellers, each with a private minimum they would accept. Order them: the buyers from most eager (highest value) down, the sellers from most desperate-to-sell (lowest acceptable price) up. The buyers' values, plotted against quantity, slope *down* (the demand curve — each additional buyer values it less). The sellers' minimums slope *up* (the supply curve — each additional seller needs more). They cross at one price. That crossing is the **clearing price**, and it is an equilibrium in the exact Nash sense.

Why is it Nash? Consider the *marginal* buyer — the last buyer who transacts, the one whose value is just barely above the price. Their best response is to buy: paying the clearing price for something they value slightly more is a (tiny) gain. Could they do better by *deviating* — by bidding lower? No: at a lower bid, no seller will hand over the unit (everyone left wants more than that), so they get nothing instead of a small gain. Deviating loses. The same holds for the marginal seller: selling at the clearing price beats holding for more (no buyer will pay more) and beats accepting less (why would they?). Every participant is at a best response. *No one can profitably move.* That is the definition of Nash equilibrium, applied to a price.

![Supply and demand curves crossing at the clearing price with deviations that lose](/imgs/blogs/nash-equilibrium-best-response-and-the-price-as-a-truce-3.png)

The chart above is the price-as-truce. Demand (blue) is what the marginal buyer will pay; supply (lavender) is what the marginal seller needs. They cross at the clearing price — here \$53 — the green point. The two red marks show deviations and why they fail: a seller who asks \$65 finds the buyers walk (no fill), and a buyer who bids \$42 finds the sellers walk (no fill). Both deviations forfeit the trade. The price sits where it sits *because* every unilateral move away from it is a losing move. The price is a truce, not a fact about the asset's "true" value — it is a fact about where the marginal players' best responses meet.

#### Worked example: solving for the clearing price

Let me compute the equilibrium price the way the figure does, with explicit demand and supply schedules, so you can see it is just arithmetic — a best-response crossing in disguise.

Suppose the marginal buyer's value falls linearly with quantity: at quantity `q`, a buyer will pay `\$100 − 0.7q`. And the marginal seller's cost rises with quantity: at quantity `q`, a seller needs `\$20 + 0.5q`. The market clears where the two are equal — where the next unit's buyer value exactly matches the next unit's seller cost:

```
100 - 0.7q = 20 + 0.5q
80 = 1.2q
q* = 66.67 units
p* = 100 - 0.7 * 66.67 = 53.33
```

So the clearing price is \$53.33 on about 67 units. Now test that it is an equilibrium. The 67th buyer values the unit at \$53.33 and pays \$53.33 — a break-even best response (they are the *marginal* buyer; everyone before them got a deal). The 68th would-be buyer values it below \$53.33, so they correctly do *not* buy — buying would be a loss for them, and not-buying is their best response. If a seller tries to hold out for \$60, the marginal buyer (value \$53.33) walks, and so do all the buyers behind them, so the holdout sells *nothing* — a worse outcome than \$53.33. Every player is at a best response; the price is stable. The intuition: the clearing price is simply the level at which the marginal buyer and marginal seller stop having a reason to move.

That is the whole secret of "it's priced in." When a stock has fully absorbed an expected earnings beat, the price has already moved to the level where the marginal buyer (who believes in the beat) and the marginal seller (who is happy to lock in the gain) meet. The beat arriving on schedule gives no one a new reason to move — so the price does not move, or even drifts down as the people who bought "for the beat" take profits. My furious friend was not wrong about the fundamentals; he was wrong about the equilibrium. He was trading against a price that had already reached its truce.

#### Worked example: what a forced seller does to the equilibrium

Let me show why the *interesting* prices are the ones knocked *off* equilibrium, because that is where an edge actually lives. Keep the same demand and supply: buyers pay `\$100 − 0.7q`, sellers need `\$20 + 0.5q`, clearing at \$53.33 on 67 units. Now a fund holding 30 units gets a margin call and *must* sell all 30 today, regardless of price — it is not choosing a best response, a constraint is choosing for it. That 30-unit dump shifts the supply curve right by 30 at any price the fund will accept (which, being forced, is "whatever clears"). The new clearing condition pits the same buyers against 30 extra forced units on top of the willing sellers:

```
Forced sale pushes price down to clear the extra 30 units.
Buyers will absorb 30 more units only if the price drops:
new clearing price falls from $53.33 toward roughly $42
(the marginal buyer for the 97th unit values it near $42).
```

The price gaps down to around \$42 — not because the asset got worse, but because one player was off their best response. Here is the edge: *you* are not forced. Your best response to a temporarily-depressed \$42 price for something worth \$53 in normal equilibrium is to buy, hold, and wait for the forced flow to clear and the price to snap back to its truce. You made \$11 a unit not by forecasting the asset — its value never changed — but by being the willing buyer on the other side of someone who had no choice. The intuition: edges are not where the equilibrium *is*, they are where someone has been knocked *out* of it and you can take the other side of their constraint.

## Solving a game by elimination: when dominance does the work

Best-response hunting always works, but sometimes there is a shortcut so clean it feels like cheating: if some strategies are *dominated* — never a best reply against anything — you can delete them, and the deletion can cascade. Let me show iterated elimination on a market game.

![Iterated elimination of dominated strategies shown as before and after](/imgs/blogs/nash-equilibrium-best-response-and-the-price-as-a-truce-4.png)

The diagram above shows the idea as before-and-after. On the left, the full game: you can panic-sell or hold; they can overpay or wait. Suppose panic-selling is worse than holding against *every* move the other side makes — then panic-sell is dominated, and you delete it. Once it is gone, suppose overpaying is now worse than waiting for the counterparty in every remaining case — delete it too. On the right, what is left: one cell, hold-versus-wait, which is the unique Nash equilibrium, found without ever guessing what the opponent would do.

#### Worked example: solving a dominance game with computed equilibria

Let me put real numbers on it and confirm the equilibrium against the verified solver. Here is a 2×2 where the row player (You) has a strictly dominant strategy. Your payoffs are `A`, the column player's (Them) are `B`:

```
You (rows):     A = [[3, 2],
                     [1, 0]]
Them (cols):    B = [[3, 1],
                     [2, 0]]
```

Read your matrix `A`. If Them plays the left column, you get 3 (top) vs 1 (bottom) — top wins. If Them plays the right column, you get 2 (top) vs 0 (bottom) — top wins again. Your top row beats your bottom row in *both* columns, so the top row **strictly dominates** the bottom: you play it no matter what. Delete the bottom row. Now Them knows you will play top. In Them's matrix `B`, along your top row, the left column pays them 3 vs the right column's 1 — left wins. So Them plays the left column. The surviving cell is top-left: **(0, 0)** in zero-indexed terms, paying You \$3 and Them \$3. Running this through the verified solver:

```
>>> import data_gametheory as gt
>>> gt.nash_2x2([[3,2],[1,0]], [[3,1],[2,0]])
{'pure': [(0, 0)], 'mixed': None}
```

The solver confirms exactly one pure Nash equilibrium at `(0, 0)` — the top-left cell — and no mixed one. Iterated dominance and the solver agree: when a dominant strategy exists, you do not need to outguess anyone, the game solves itself. The intuition: dominance is the rare luxury of a move so good you can play it blind — and the rarity is the lesson, because most market choices are *not* dominant.

Why does this matter for trading? Because the times when a market has a *dominant* move are the times the "trade" is barely a game at all — and those are exactly the times everyone takes the same side and the move is already gone by the time you see it. Stops-when-tail-risk-exists, hedging-a-known-exposure, taking-free-money-in-an-arbitrage: these are dominant, so everyone does them, so the edge is competed to zero. The interesting money is in the games *without* a dominant strategy — where your best move genuinely depends on reading the other side, which is the whole rest of this series.

## When there is no pure equilibrium: mixed strategies

Here is a game with a shocking property: it has *no* equilibrium in definite actions at all. It is called **matching pennies**, and it is the mathematical heart of every situation where being predictable gets you exploited — a market maker reading your order flow, a poker opponent picking off your tell, an algorithm front-running your routine.

The setup: two players each secretly choose Heads or Tails. You (the *matcher*) win if the choices match; Them (the *mismatcher*) wins if they differ. Payoffs of +1 to the winner, −1 to the loser:

| | Them: Heads | Them: Tails |
|---|---|---|
| **You: Heads** | You +1, Them −1 | You −1, Them +1 |
| **You: Tails** | You −1, Them +1 | You +1, Them −1 |

Try to find a pure Nash equilibrium by best-response hunting. Suppose you both play Heads (top-left): Them is losing (−1), so Them wants to switch to Tails. Now you are at top-right and losing, so *you* want to switch to Tails. Now bottom-right, Them is losing, so Them switches to Heads. Now bottom-left, you are losing, switch to Heads — and we are back where we started. The best responses chase each other around the matrix forever; *no cell is a mutual best response.* There is no pure Nash equilibrium.

So what is stable? The answer is to randomize. If you play Heads with probability `p` and Tails with `1 − p`, and you pick `p` so that Them is *indifferent* between Heads and Tails — earning the same expected payoff either way — then Them has no reason to favor one over the other, and your randomization is "safe." Solving for that `p` (and the symmetric `q` for Them) is what the solver does:

```
>>> gt.nash_2x2([[1,-1],[-1,1]], [[-1,1],[1,-1]])
{'pure': [], 'mixed': (0.5, 0.5)}
```

No pure equilibria, and one **mixed** equilibrium: `p = q = 0.5`. Both players flip a fair coin. At 50/50, you cannot be exploited — whatever Them does, your expected payoff is zero, and theirs is too. The randomization itself is the equilibrium.

![Stacked bars showing the fifty fifty mixed Nash and an exploitable predictable player](/imgs/blogs/nash-equilibrium-best-response-and-the-price-as-a-truce-5.png)

The bars above show why this is the only safe play. You and Them at the Nash mix are each exactly 50/50 (the blue/amber split). The third bar is a *predictable* player who leans 80% Heads. The instant a player is not 50/50, the opponent's best response is no longer indifferent — against an 80%-Heads player, you should lean Tails to exploit them, and you will, on average, take their money. In a market, "predictable" is the deadliest word there is: a fund that always rebalances on the last day of the quarter, a stop cluster sitting at an obvious round number, an algorithm that always replenishes its bid at the same depth — each one is a non-50/50 player, and the other side will find the pattern and lean against it. (The full strategic logic of unpredictability gets its own post: [mixed strategies and the value of being unpredictable](/blog/trading/game-theory/mixed-strategies-and-the-value-of-being-unpredictable).)

#### Worked example: the cost of being predictable

Let me put a dollar figure on predictability. You play matching pennies for \$10 a round against an opponent. At the Nash mix — you 50/50, them 50/50 — your expected profit per round is exactly \$0 (it is a zero-sum game; neither side can win in the long run). Now suppose you get lazy and play Heads 80% of the time, and a sharp opponent notices.

The mismatcher's best response to your 80%-Heads is to play Tails always (so they mismatch your Heads, which is the likely outcome). Compute your expected payoff per round:

```
P(you Heads) = 0.80, P(you Tails) = 0.20
Opponent always plays Tails.
- You Heads (0.80) vs their Tails -> mismatch -> you LOSE $10
- You Tails (0.20) vs their Tails -> match    -> you WIN  $10
E[your payoff] = 0.80 * (-10) + 0.20 * (+10) = -8 + 2 = -$6 per round
```

You lose \$6 every round, on average, *purely because you were readable*. Over 100 rounds that is \$600 handed to a sharper player — not because they forecast better, but because they read your pattern and you handed them the other side. The intuition: in a game with no pure equilibrium, predictability is not a style choice, it is a leak, and the size of the leak is exactly how far you are from 50/50.

This is the matching-pennies structure behind why a market maker can earn a living off uninformed flow: see [how an options market maker thinks about the other side of your trade](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade). The dealer does not need to forecast the stock; they need you to be predictable enough that, on average, they get the better of the spread.

## When there are two equilibria: coordination games

The opposite problem to "no equilibrium" is "too many." Some games have *multiple* Nash equilibria, and the theory itself cannot tell you which one will occur. This is the **coordination game**, and it is everywhere in markets: which exchange has the liquidity, which stablecoin people trust, whether a bank run happens, which "story" the market decides to believe.

![A coordination game matrix with two Nash equilibria on the diagonal](/imgs/blogs/nash-equilibrium-best-response-and-the-price-as-a-truce-6.png)

The matrix above is the simplest coordination game. Two traders each pick a venue, A or B, to route an order. If they pick the *same* venue, both get filled (a win); if they split, they miss each other (zero). The payoffs:

| | Them: Venue A | Them: Venue B |
|---|---|---|
| **You: Venue A** | You 2, Them 2 | You 0, Them 0 |
| **You: Venue B** | You 0, Them 0 | You 1, Them 1 |

Both "matching" cells are Nash equilibria. (A, A) pays 2 each and (B, B) pays 1 each — and both are stable, because in either one, deviating alone means you go to the empty venue and get 0. Let me confirm with the solver:

```
>>> gt.nash_2x2([[2,0],[0,1]], [[2,0],[0,1]])
{'pure': [(0, 0), (1, 1)], 'mixed': (0.333, 0.333)}
```

Two pure equilibria — `(0,0)` is (A, A) and `(1,1)` is (B, B) — plus a mixed one. The unsettling part: equilibrium theory says *both* (A, A) and (B, B) are valid, stable outcomes. Even though everyone would prefer (A, A) — it pays more — there is nothing in the *game* that forces it. If everyone *believes* the action is on Venue B, then routing to B is each person's best response, and B becomes the equilibrium that holds, even though it is the worse one for everybody. The equilibrium that occurs is selected by **beliefs and history** — what people expect everyone else to do — not by the payoffs alone.

#### Worked example: the mixed equilibrium of a coordination game

There is also a third, fragile equilibrium hiding in that coordination game — a mixed one — and it is worth computing because it shows how bad a failure to coordinate can be. The solver returned `mixed = (0.333, 0.333)`: each player goes to Venue A with probability 1/3 and Venue B with 2/3. Let me verify it makes the other player indifferent, and compute the payoff.

If you go to A with probability 1/3, then Them's expected payoff from also going to A is `(1/3 × 2) + (2/3 × 0) = 0.667`, and from going to B is `(1/3 × 0) + (2/3 × 1) = 0.667`. Equal — so Them is indifferent, which is exactly the condition for a mixed equilibrium. Now compute the *expected* payoff at this mixed equilibrium. The chance you both land on A is `1/3 × 1/3 = 1/9` (worth 2), and both on B is `2/3 × 2/3 = 4/9` (worth 1):

```
E[payoff] = (1/9)*2 + (4/9)*1 + (4/9)*0 = 2/9 + 4/9 + 0 = 6/9 = 0.667
```

So at the mixed equilibrium each trader earns just \$0.67 — far worse than the \$2 of the good pure equilibrium or even the \$1 of the bad one, because more than half the time the two of you simply miss each other and get nothing. The intuition: when players cannot coordinate on *which* equilibrium to play, the mathematics of randomizing leaves everyone worse off than any of the orderly outcomes — coordination failure is expensive.

This is why liquidity is "sticky" and why incumbents are so hard to dislodge: an exchange or a stablecoin that everyone already uses is a self-fulfilling equilibrium, and a rival can be *better* on every dimension yet fail to attract anyone, because no individual wants to be the first to switch to the empty venue. It is also the skeleton of a bank run, which we will meet again: "everyone withdraws" and "no one withdraws" are both equilibria of the same game, and which one happens depends entirely on what each depositor believes the others will do.

## "Priced in" is an equilibrium of beliefs

We can now state the deepest version of the price-as-equilibrium idea. When traders say something is "priced in," they are making a statement about an equilibrium — but not of *actions*, of *beliefs*. The price is a fixed point: a level at which everyone's best response to it, given their beliefs about value, leaves the price unchanged.

![Pipeline showing how trading on the price value gap closes until priced in](/imgs/blogs/nash-equilibrium-best-response-and-the-price-as-a-truce-7.png)

The flow above is the belief equilibrium. Each trader has a private estimate of what the asset is worth. They look at the price and act on the *gap*: buy if they think it is worth more than the price, sell if less. That trading moves the price — buying lifts it, selling presses it down. As the price moves, the easy gaps close: the people who thought it was cheap have bought (lifting the price toward their value), the people who thought it was rich have sold. The process stops when the *marginal* trader is indifferent — when the price has moved to a level where the last person to act has no profitable gap left. At that fixed point, the price is "priced in": it has absorbed everyone's beliefs, and no one has a profitable unilateral move. That is a Nash equilibrium of beliefs.

This reframes a whole vocabulary. "The market has discounted the rate cut" means the price already sits at the equilibrium that assumes the cut — so the cut arriving changes nothing, and only a *surprise* (a different cut, or a hawkish tone) can move the price, because only a surprise gives someone a new profitable move. "Buy the rumor, sell the news" is the same statement: the rumor moves the price to the priced-in equilibrium *before* the news, so when the news confirms the rumor there is nothing left to buy on, and the people who bought the rumor sell into the confirmation. (Event-reaction trading lives entirely in this space; the mechanics of *why* confirmed news so often fails to move a market are an event-trading subject.)

#### Worked example: how much of the news was priced in

Let me make "priced in" a number. A biotech stock trades at \$40. The market believes there is a 70% chance an FDA approval comes through, which would make the stock worth \$50, and a 30% chance of rejection, which would make it worth \$20. Is \$40 an equilibrium price?

```
Expected value given beliefs = 0.70 * $50 + 0.30 * $20
                            = $35 + $6 = $41
```

The belief-weighted value is \$41, just above the \$40 price — so there is a tiny gap, and marginal buyers will nudge the price up toward \$41, where it settles. Now the approval is announced. The naive trader thinks: "approval means \$50, the stock is at \$41, easy 22% gain!" But watch what actually happens. The market had *already* priced 70% odds of exactly this. The move from the priced-in \$41 to the post-approval reality is not \$41 → \$50 (that would be the whole surprise); it is the *removal of the 30% rejection risk*. The stock jumps to \$50 — a gain of \$9, or 22% — but only the part that was *not* already priced gets paid to a fresh buyer. Anyone who bought at \$41 expecting the news captured the surprise; anyone buying *after* the announcement at \$50 captures nothing, because at \$50 the news is now priced in. Had the market priced 95% approval odds, the stock would have sat at `0.95 × 50 + 0.05 × 20 = $48.50` beforehand, and the approval would have moved it only \$1.50 — almost nothing, because almost all of it was priced in. The intuition: the tradeable move is never the size of the news, it is the size of the *surprise* relative to the equilibrium that already existed.

## Common misconceptions

**"Nash equilibrium means the fair or efficient outcome."** No. Equilibrium means *stable* — no one can profitably deviate alone — and that is all it means. The prisoner's dilemma equilibrium is jointly terrible for both players; the bad coordination equilibrium pays everyone less than the good one; a market can clear at a price that ruins one side. A monopolist's profit-maximizing price is a Nash equilibrium and it is neither fair to buyers nor efficient for society. Stable ≠ fair ≠ efficient. Conflating them is the single most common error people make with this idea, and it leads straight to "the market price must be right," which is a much stronger (and often false) claim than "the market price is stable."

**"If a price is an equilibrium, it can't be wrong."** A price is an equilibrium of *current beliefs and current participants*. If the beliefs are wrong — if everyone is mis-estimating value the same way — the equilibrium price is wrong too, and it stays "stable" right up until the beliefs change. Equilibrium is about consistency among the players, not correctness about the world. Bubbles are equilibria: at the top of a mania, the price is a perfectly good Nash equilibrium given that everyone believes the buyer-behind-them will pay more. It is stable until the belief breaks. (See [the prisoner's dilemma in markets, why everyone sells at once](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once) for what the break looks like.)

**"I just need to forecast better than the market."** This misunderstands what you are trading against. The price is not a forecast you can beat with a better forecast; it is an *equilibrium of everyone's forecasts*. Your edge does not come from being right about the asset — it comes from being right about a *gap between the equilibrium and reality* that the other participants have not closed, and from knowing why they cannot close it (they are forced sellers, they are uninformed, they are constrained, they are not paying attention). "Who is on the other side, and why are they leaving money on the table?" is the right question. "What will the stock do?" is not.

**"In a game with no pure equilibrium I should just pick whichever action seems best."** In matching-pennies-type games, *any* deterministic choice is exploitable — that is the whole content of "no pure equilibrium." Picking "whichever seems best" makes you predictable, and predictable is the one thing that costs money (we computed \$6 a round above). The equilibrium *is* the randomization. If your market behavior has an exploitable pattern — same rebalance day, same stop level, same replenish depth — you are off equilibrium and someone is leaning against you.

**"Multiple equilibria means the model is broken or useless."** The opposite. Multiplicity is the model correctly telling you that *beliefs and coordination*, not fundamentals, decide the outcome — which is genuinely true for liquidity, for which stablecoin survives, for whether a run happens. When a game has two equilibria, the useful question is not "which is correct?" but "which one do people expect, and what could flip the expectation?" That is exactly the question a trader watching a fragile peg or a crowded venue should be asking.

## How it shows up in real markets

**Earnings "sell the news," repeatedly.** The pattern my friend got caught by is one of the most documented in markets. Across many studies of post-earnings behavior, stocks that beat expectations frequently *fall* on the day, because the run-up into the report had already moved the price to the priced-in equilibrium. Netflix in mid-2018 is a textbook case: the stock had rallied hard into Q2 earnings, the company beat on profit, and the stock fell roughly 14% the next day — not because the results were bad, but because subscriber adds slightly missed the *expectations embedded in the price*. The surprise, not the news, moved the equilibrium. The lesson: before any scheduled event, ask "what is already priced in?" — the gap between that and the print is the only thing you can trade.

**The Fed and the priced-in cut.** Central-bank meetings are the purest "equilibrium of beliefs" market there is. By the time of a rate decision, the futures market has assigned probabilities to each outcome, and the spot market has moved to the belief-weighted equilibrium. When the Fed cut 50 basis points in September 2024, the move had been telegraphed and largely priced for weeks; markets had spent the run-up adjusting, so the *decision itself* produced a muted reaction relative to the size of the cut. What moves markets on Fed day is the *surprise* in the statement and the press conference — the part the equilibrium did not already contain. ("Don't fight the Fed" is, in this language, "don't deviate from the equilibrium the central bank is anchoring.")

**Stablecoin runs as coordination equilibria.** The collapse of TerraUSD in May 2022 is a coordination game that flipped equilibria in days. The "peg holds" equilibrium — where no one redeems because everyone believes the peg holds — and the "everyone runs" equilibrium both existed for the same system. A few large redemptions shifted beliefs; once enough holders expected others to run, running became each holder's best response, and roughly \$18 billion of UST value evaporated as the bad equilibrium took hold. Nothing about the *fundamentals* changed in those hours — what changed was the belief about what everyone else would do, which is precisely what selects between equilibria in a coordination game.

**Liquidity stickiness and the venue that won't die.** Why does so much equity volume still concentrate on a handful of exchanges and dark pools even when rivals offer better fees? Because routing is a coordination game: liquidity attracts liquidity, and a venue everyone already uses is a self-fulfilling equilibrium. New venues routinely launch with better technology and lower cost and still struggle, because no individual trader wants to route to the empty book first. The famous historical case is the persistence of the NYSE floor and the slow, decades-long migration to electronic venues — coordination equilibria move at the speed of changing collective belief, not at the speed of the better mousetrap.

**The crowded trade and the simultaneous exit.** When a position becomes a consensus equilibrium — everyone long the same momentum names, everyone short the same volatility — the price is stable *as long as no one moves first*. But the equilibrium is fragile, because each participant's best response if others start selling is to sell faster (a coordination problem with a bad equilibrium lurking). The August 2007 "quant quake," when many statistical-arbitrage funds held near-identical positions and a forced deleveraging by one triggered a cascade, is the canonical example: a stable, profitable equilibrium that flipped into a violent unwind the moment beliefs about "who sells first" shifted. The proprietary-trading view of reading the other side and sizing for these flips is the SIG/poker mindset: [the SIG / Susquehanna playbook on poker, game theory, and EV](/blog/trading/quant-careers/sig-susquehanna-playbook-poker-game-theory-and-ev).

**The auction and the winner's curse.** Every Treasury auction, every IPO allocation, every takeover bid is a game whose equilibrium bidding strategy is *below* your honest valuation — because winning means you bid more than everyone else, which is itself bad news about the asset's value. The equilibrium price in a common-value auction reflects this shading. A bidder who ignores it and bids their full estimate is off-equilibrium, wins too often, and overpays — the winner's curse. The equilibrium is not "bid what you think it's worth"; it is "bid what you think it's worth *conditional on winning*," which is a strictly lower, strategically-shaded number.

**Short squeezes as a coordination equilibrium flipping the other way.** The GameStop episode of January 2021 is the coordination game run in reverse — and a vivid demonstration that "equilibrium" does not mean "fundamentally justified." The stock ran from around \$20 to an intraday high near \$483 in a few weeks, not because anyone's estimate of the business changed by 24×, but because a self-reinforcing equilibrium formed among buyers: each one's best response, *given that everyone else was buying and shorts were being forced to cover*, was to keep buying. Short sellers were the forced players here — margin calls and borrow costs pushed them to buy back at any price, which lifted the price further, which forced more covering. For a while, "keep buying" was a genuine Nash equilibrium of beliefs, perfectly stable as long as everyone believed everyone else would hold. Then the belief broke — brokers restricted buying, the coordination cracked — and the price collapsed back toward the old equilibrium just as fast. The lesson is the one from the misconceptions section made violent: a price can be a textbook-stable equilibrium and still be wildly disconnected from value, because equilibrium is about consistency of beliefs, not correctness about the world.

**Index reconstitution as predictable, exploitable flow.** When a stock is added to or dropped from a major index, every passive fund tracking that index *must* trade it on the reconstitution date — a perfectly predictable, forced flow. This is the predictable-player problem at industrial scale: the index funds are not choosing a best response, they are mechanically buying or selling on a known date, and faster players front-run the flow, pushing the price up before the funds buy (and down before they sell). Studies of S&P 500 additions have documented price run-ups of several percent into the effective date followed by partial reversals afterward — the off-equilibrium footprint of a flow everyone could see coming. The funds accept this cost as the price of mechanical tracking; the lesson for you is that *predictable forced flow is the most reliable off-equilibrium signal there is*, precisely because the player creating it has no discretion to randomize.

## The playbook: how to trade against an equilibrium

You now have the lens. Here is how to actually use it at the desk.

**Who is on the other side, and what game are they playing?** Before any trade, name the counterparty and their payoff. A market maker is playing matching pennies with your order flow — they profit if you are predictable, they are indifferent if you are random. A forced seller (margin call, fund redemption, index reconstitution) is *not* at a best response — they are deviating from what they'd freely choose because a constraint is forcing them — and that is where your edge lives: you are buying from someone who has no choice. A fellow speculator in a crowded trade is in a coordination game with you, and the question is who blinks first. Identify the game and you identify whether there is money on the table.

**Your edge is a gap in the equilibrium, not a better forecast.** The price already contains the consensus forecast. You do not make money by forecasting the asset better; you make money by finding a place where the equilibrium is *wrong or hasn't formed* — a forced flow that pushed the price off its truce, a piece of information not yet incorporated, a coordination equilibrium about to flip — and by knowing *why* the other participants can't or won't close the gap. If you cannot articulate who is leaving money on the table and why, you do not have an edge; you have a forecast, and the market already has a better one.

**Respect "priced in" — trade the surprise, not the news.** Before any scheduled catalyst (earnings, Fed, data, a vote), write down what the price implies is expected. The tradeable move is the *difference* between the print and the priced-in expectation, and it can be the opposite sign of the news (good news, stock falls). If you cannot state what is priced in, you are not trading the event, you are gambling on a coin you can't see.

**Don't be the predictable player.** If your own behavior has an exploitable pattern — a fixed rebalance date, a stop at the round number everyone uses, an order size you always show — you are a non-50/50 player and the other side will lean against you. Mix your execution, vary your timing, hide your size. The cost of predictability is real and computable, as we saw: \$6 a round when you should be earning \$0.

**Know which equilibrium you are in, and what flips it.** In a coordination situation — a crowded trade, a fragile peg, a liquidity venue — the price is stable until beliefs shift, and then it moves violently. Your invalidation is not a price level, it is a *belief change*: the first large redemption, the first major holder to exit, the first headline that makes everyone reconsider who sells first. Size for the flip, because the move from one equilibrium to the other is not gradual.

**The invalidation and the honesty.** This is educational, not advice. The hard truth of equilibrium thinking is that *most* of the time, the price is an equilibrium and you have no edge — the gap is closed, the surprise is symmetric, you are the marginal trader who breaks even. Edges are the exception: specific moments when a specific counterparty is off their best response for a specific, nameable reason. If you trade as though every price is wrong, you are the predictable sucker funding the people who only trade when the equilibrium is genuinely broken. The discipline is to wait for those moments and to size up when the other side has no choice — and to do nothing, gladly, the rest of the time.

## Further reading & cross-links

- [The trade is a game: why markets are strategic, not random](/blog/trading/game-theory/the-trade-is-a-game-why-markets-are-strategic-not-random) — the foundation of this whole series: every trade has a specific counterparty, so your P&L is their P&L flipped.
- [The prisoner's dilemma in markets: why everyone sells at once](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once) — the canonical case where the Nash equilibrium is bad for everyone, and what the cascade looks like.
- [Mixed strategies and the value of being unpredictable](/blog/trading/game-theory/mixed-strategies-and-the-value-of-being-unpredictable) — the full strategic logic of the matching-pennies result, and how to be genuinely random in execution.
- [Common knowledge and "I know that you know that I know"](/blog/trading/game-theory/common-knowledge-and-i-know-that-you-know-that-i-know) — what it takes for an equilibrium of beliefs to actually form, and how it breaks.
- [How an options market maker thinks about the other side of your trade](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) — the dealer's view of the matching-pennies game they play with your flow.
- [The SIG / Susquehanna playbook: poker, game theory, and EV](/blog/trading/quant-careers/sig-susquehanna-playbook-poker-game-theory-and-ev) — how a prop firm turns "find the off-equilibrium player" into a hiring filter and a business.

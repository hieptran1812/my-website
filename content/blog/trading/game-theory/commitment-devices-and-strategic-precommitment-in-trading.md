---
title: "Commitment Devices and Strategic Precommitment in Trading"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "How deliberately destroying your own options can make a threat credible against opponents and bind your disciplined self against your panicked one."
tags: ["game-theory", "trading", "commitment-device", "precommitment", "schelling", "stop-loss", "discipline", "credibility", "behavioral-finance", "risk-management"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Sometimes you gain power by *reducing* your own options: credibly tying your own hands changes how the other side plays against you, and how you play against your own future self.
>
> - A threat or a price floor only moves an opponent if it is **credible** — irreversible or expensive to undo. An empty threat (cheap talk you can quietly abandon) is correctly ignored.
> - A sunk **precommitment** can flip the entire equilibrium: in our entry game it changes the unique Nash outcome from "rival enters, you accommodate" to "rival stays out, you win uncontested" — without a shot fired.
> - A **stop-loss** is a commitment device against your future self. On a position down -8%, a disciplined stop has expected value of \$-8% versus \$-13.85% for "hold and hope," so the rule saves about 5.85 percentage points per trade.
> - Commitment beats flexibility when **you** are the main risk (you are predictably tempted to deviate); flexibility wins when new information should change the call — and a visible stop can be hunted.
> - The one rule: **a commitment is only worth what it costs to break.** If breaking it is free, the other side knows it, and so does the panicked version of you.

In 1519, Hernán Cortés landed on the coast of Mexico with about 500 men, facing an empire of millions. According to the legend that has outlived the messier history, he ordered his own ships scuttled. His soldiers now had no way home. Retreat was not merely discouraged — it was *impossible*. Whether the ships were burned or run aground, the strategic point is the one Thomas Schelling, the economist who won a Nobel Prize for thinking about exactly this, would later make famous: by destroying his own escape route, Cortés changed the game. His men, with no exit, would fight with the desperation of the cornered. And his enemies, reasoning about a foe who literally could not flee, had to weigh a far more expensive fight.

This is a genuine paradox, and it runs against every instinct we have about freedom and choice. We are taught that more options are always better — that keeping your powder dry, staying nimble, never getting locked in, is the mark of a sophisticated player. And often it is. But in a *strategic* setting — where your outcome depends not on nature but on what another reasoning agent decides to do — the opposite can be true. **You can make yourself stronger by making yourself less free.** Removing one of your own options can change what your opponent expects, and that change in expectation is worth more than the option you gave up.

The chart below is the mental model for the whole post. On the left, you keep every option open: you announce "I will not sell below \$100," but everyone (including you) knows you can still bail at \$95 to cut a loss — so your threat is cheap talk and the level gets attacked. On the right, you visibly burn the escape route — a resting buy wall, funds you cannot withdraw, a public irreversible floor — and now the threat is real, so the other side yields. The trick is not power in the usual sense. It is the *credible removal* of your own ability to back down.

![Before and after of tying your own hands to make a threat credible and the opponent yield](/imgs/blogs/commitment-devices-and-strategic-precommitment-in-trading-1.png)

There are two layers to this idea, and a trader needs both. The first is **strategic precommitment against others**: the burning bridges, the doomsday device, the price floor, the poison pill, the capacity that signals "I will fight." The second — quieter, more personal, and for most retail traders far more valuable — is **precommitment against your own behavioral weakness**: the stop-loss that binds the disciplined you against the panicked you, the written plan, the position limit, the auto-invest, the locked-up funds. Both are the same trick. Both work only when the commitment is credible. And both can backfire, because the very visibility that makes a commitment credible to others also makes it a target. By the end of this post you will be able to tell, for any rule you are considering, whether it will actually change the game in your favor or whether it is just a story you are telling yourself.

## Foundations: commitment, credibility, and tying your own hands from zero

Let us build the vocabulary from nothing, because the words sound like ordinary English but mean something precise here.

A **game**, in the technical sense, is any situation where your best move depends on what someone else does, and their best move depends on what you do. A *trade* is a game in exactly this sense: every dollar you make is a dollar a specific counterparty loses or pays for a hedge, so your edge is never a pure forecast about nature — it is knowing who is on the other side and reasoning one step deeper than they do. (We build that foundation in [The trade is a game](/blog/trading/game-theory/the-trade-is-a-game-why-markets-are-strategic-not-random) and [Who is on the other side of your trade](/blog/trading/game-theory/who-is-on-the-other-side-of-your-trade).)

A **strategy** is a complete plan of action — what you will do in every situation you might face. A **payoff** is what each player gets from each combination of strategies, usually money or utility. An **equilibrium** — specifically a *Nash equilibrium*, named after John Nash — is a combination of strategies where no player can do better by changing their own move alone, given what everyone else is doing. It is the resting point of the game: the place where everyone's plan is a best response to everyone else's. We treat the equilibrium as the prediction for how a game actually plays out.

Now the new word. A **commitment** is a move that *restricts your own future choices* — it takes some action off your own menu, on purpose. The thing you use to do it is a **commitment device**: any mechanism that makes a future action either impossible or so costly that you (or the other side) can be sure you will not take it. Burning the bridge is a commitment device. So is a contract with a penalty clause, a bet you have publicly staked your reputation on, a broker-side stop order that fires automatically, and a savings account you cannot touch without a fee.

Here is the part that feels backwards. In a one-player decision against nature — say, deciding whether to carry an umbrella — having *more* options is never worse. You can always ignore an option you do not want. But in a *game*, your options are information the other player uses. If the other side knows you *can* back down, they will plan around the possibility that you *will*. Removing the option removes their hope. **The value of a commitment is not in the action it lets you take; it is in the action it credibly tells the other side you will take, which changes the action they take.**

That word — *credibly* — is the whole ballgame. A threat or a promise only changes behavior if the other side believes it. And they will only believe it if backing down is genuinely costly or impossible for you. An ordinary threat — "I'll cut my position if it hits \$92, so don't push me there" — is what game theorists call **cheap talk**: words that cost nothing to say and nothing to abandon. The other side knows that when the moment comes, your incentive will be to *not* follow through (selling into a forced low is painful), so they discount the threat to zero. A **credible commitment** is the opposite: you have arranged things so that *not* following through is the costly path. You have changed your own future incentives.

To make this concrete, the technical term for the kind of cost that makes a commitment stick is a **sunk cost** — money or effort already spent that you cannot recover. Sunk costs are the villain of most decision-making advice ("ignore sunk costs!"), but in commitment they are the hero. Cortés's scuttled ships were a sunk cost: the option to sail home was gone, and *because it was gone*, his men's incentive to fight was no longer in question. The very irreversibility that decision theory tells you to ignore is what makes a strategic commitment believable.

One last definition before we go deep. **Precommitment** is just commitment made *in advance* — you bind your future self at a moment when you are calm and rational, so that the binding holds later, at a moment when you might not be. Ulysses had himself tied to the mast before the Sirens sang, precisely because he knew the song would make him want to steer the ship onto the rocks. The trade equivalent is writing down your exit *before* you are in the position and your money is on the line, then setting it as an order so the calm you overrules the panicked you. We will spend the second half of the post on exactly this self-directed kind, because it is where most traders actually win or lose.

## How a precommitment flips the equilibrium: the entry-deterrence game

The cleanest way to see commitment change a game is the classic **entry-deterrence** game, which we can map directly onto a market. Picture an incumbent — a large holder defending a price level, or a firm that owns a market — and a **rival** deciding whether to enter (attack the level, launch a competing product, push price through). This is a sequential interaction, but we can read it as a payoff matrix and find its equilibrium with the same `nash_2x2` machinery we used in [Nash equilibrium, best response, and the price as a truce](/blog/trading/game-theory/nash-equilibrium-best-response-and-the-price-as-a-truce).

The incumbent has two strategies: **accommodate** (share the space, retreat gracefully) or **fight** (defend hard — flood supply, cut price, absorb losses to make the rival bleed). The rival has two: **stay out** or **enter**. The matrix below shows the payoffs in two worlds — before the incumbent can commit, and after.

![Payoff matrix showing how a credible precommitment flips the Nash equilibrium from rival enters to rival stays out](/imgs/blogs/commitment-devices-and-strategic-precommitment-in-trading-2.png)

Read the matrix as the equilibrium engine for the whole post: the same game, played two ways, lands on opposite outcomes purely because of one sunk commitment. Let us compute it.

#### Worked example: the equilibrium before commitment (the empty threat)

In the *before* world, fighting is genuinely costly to the incumbent — defending a level by flooding supply or undercutting price burns real money. The payoffs (incumbent first, rival second) are:

- Accommodate, rival stays out: incumbent **10**, rival **0**.
- Accommodate, rival enters: incumbent **5**, rival **+4**.
- Fight, rival stays out: incumbent **8**, rival **0** (idle defensive capacity still costs something).
- Fight, rival enters: incumbent **1**, rival **-2** (a price war hurts both, the incumbent most).

Feed this to the solver:

```
import data_gametheory as gt
A = [[10, 5], [8, 1]]      # incumbent payoffs (row 0 = accommodate, row 1 = fight)
B = [[0, 4], [0, -2]]      # rival payoffs   (col 0 = stay out,  col 1 = enter)
print(gt.nash_2x2(A, B))
>>> {'pure': [(0, 1)], 'mixed': None}
```

The unique Nash equilibrium is **(0, 1)** — the incumbent accommodates and the rival enters. Why? Look at the incumbent's choice *if the rival enters*: accommodating pays 5, fighting pays only 1. Accommodating is the better response. The rival, reasoning one step deeper, knows this: "If I enter, the incumbent's best move is to accommodate, which earns me +4. So I enter." The incumbent's threat to *fight* is an empty threat — cheap talk — because the moment the rival actually enters, fighting is the incumbent's *own* worst option. A rational rival ignores a threat the threatener would not want to carry out.

This is the central frustration. The incumbent *would love* to deter entry. Saying "enter and I will fight you to the ground" would, if believed, keep the rival out and earn the incumbent 10 instead of 5. But it is not believed, because everyone can see that once entry happens, fighting hurts the incumbent more than accommodating. The intuition: a threat you would not want to execute is worth nothing, no matter how loudly you shout it.

#### Worked example: the equilibrium after a credible commitment (the threat becomes real)

Now the incumbent does something clever *before* the rival decides. It sinks money into irreversible capacity — extra inventory, a pre-funded defense fund, a published rule it cannot legally walk back. This **changes its own future payoffs**. Two things shift: idle accommodation now wastes that sunk capacity (so accommodating pays less), and *fighting* now uses capacity that is already paid for (so fighting is cheap, even profitable). The new payoffs:

- Accommodate, rival stays out: incumbent **8**, rival **0**.
- Accommodate, rival enters: incumbent **3**, rival **+4**.
- Fight, rival stays out: incumbent **10**, rival **0**.
- Fight, rival enters: incumbent **5**, rival **-2**.

```
A2 = [[8, 3], [10, 5]]     # capacity sunk: fighting now dominates accommodating
B2 = [[0, 4], [0, -2]]     # rival's payoffs unchanged
print(gt.nash_2x2(A2, B2))
>>> {'pure': [(1, 0)], 'mixed': None}
```

The unique equilibrium has flipped to **(1, 0)** — the incumbent fights and the rival stays out. Now *fighting* is the incumbent's best response in **both** columns (10 > 8 if the rival stays out, 5 > 3 if the rival enters). Fighting is a *dominant* strategy. The rival, reasoning backward again: "If I enter, the incumbent will fight — that is now its best move — and I earn -2. If I stay out, I earn 0. So I stay out." The rival yields **without the incumbent ever firing a shot.** The price war that the matrix prices at (5, -2) never happens; the game ends in the top-left-ish corner where the rival simply declines to attack.

Notice the magic. The incumbent made itself *worse off in some cells* — accommodating now pays 8 instead of 10 — and that self-imposed handicap is exactly what made it *better off overall*, because it earns 10 (fight, stay out) instead of the old 5 (accommodate, enter). The intuition: by spending money to take its own escape option off the table, the incumbent turned an empty threat into a credible one, and a credible threat doesn't need to be used — it just needs to be believed.

This is Schelling's paradox in one matrix. The power came from *reducing* options, and the proof is that the equilibrium moved. Every real commitment device below — the price floor, the poison pill, the stop-loss — is a way of buying that flip.

### Why the order of moves matters: the first-mover and the meta-move

There is a structural reason commitment works, and it has to do with the *sequence* of decisions. In a game where both sides move at the same time, neither can condition on the other — they guess. But if one side can move *first* and that move is *visible and irreversible*, the second mover is forced to respond to a fact rather than a possibility. Commitment is precisely the art of converting a simultaneous game (where your threat is a guess) into a sequential one (where your threat is a fait accompli the other side must navigate around).

This is why a commitment is best understood not as an ordinary move but as a **meta-move** — a move about your own future moves. When the incumbent sinks capacity, it is not playing "fight" yet; it is restructuring the game so that "fight" becomes its own best response *later*. The opponent then solves the *new* game, not the old one. Schelling's phrase for this was that "the power to constrain an adversary may depend on the power to bind oneself." You are not changing what you *can* do in some abstract sense — you are changing what you *will* do, and making that change visible, so the other side's calculation runs through your new incentives instead of your old ones.

There is a price for moving first, though, and a good player weighs it. Moving first reveals information and forfeits flexibility — you commit *before* you have learned what the second mover would have done. In games of pure conflict where surprise matters, moving first can be a *disadvantage* (it tells the opponent your hand). The commitment trick only pays when the *deterrent value* of your visible, irreversible move exceeds the *option value* of waiting to see what the opponent does. That trade-off — deterrence now versus information later — is the same flexibility-versus-commitment tension we formalize at the end of the post, and it is why commitment is a scalpel, not a hammer.

### Capacity precommitment: a worked dollar version

The abstract "sink some capacity" deserves a concrete number, because capacity precommitment is one of the most common real commitment devices — a firm builds a factory bigger than it currently needs, or a market maker posts deeper size than today's flow requires, specifically to *signal* that it will defend its share.

#### Worked example: capacity as a credible "I will flood supply" threat

Suppose you are a large liquidity provider on an asset, and a rival is deciding whether to muscle into your spread. Each unit of the asset you can supply at competitive prices costs you about \$2 to provision (capital, risk, infrastructure). Today you can credibly supply **1,000 units** per session. Your threat — "enter my market and I will flood it with cheap supply until your margins vanish" — costs you `1,000 × \$2 = \$2,000` to execute, and your accommodating profit if you *don't* fight is, say, \$5,000. So fighting costs you `\$2,000` in provisioning *and* sacrifices spread, netting maybe \$1,000 — far worse than the \$5,000 of accommodating. The threat is empty; the rival enters.

Now you **pre-build capacity**: you sink \$2,000 *up front* into infrastructure that lets you supply **3,000 units** at near-zero marginal cost. That \$2,000 is now a sunk cost — gone whether or not you fight. After the rival enters, your fight cost is no longer \$2,000; it is roughly \$0 marginal, because the capacity is already paid for. Flooding 3,000 units of cheap supply now *crushes* the rival's economics while costing you almost nothing extra. Fighting becomes your best response, the rival computes this in advance, and stays out. You have converted a \$2,000 upfront expense into a credible deterrent that earns you the uncontested \$5,000 — a clear win. The intuition: pre-built capacity is a burned bridge made of money, and the rival, seeing the cost is already sunk, knows the flood is real.

This is exactly the logic behind a market maker quoting deeper size than the current flow requires, or a producer holding spare capacity it visibly *could* dump. The capacity is not there to be used in the average case — it is there so the threat to use it is credible, which means it usually never has to be used at all. (The economics of how many players a market like this supports, and why capacity choices drive the equilibrium price, are the [Cournot competition](/blog/trading/game-theory/cartels-collusion-and-the-cournot-game-from-opec-to-algorithms) story.)

## The stop-loss as a commitment device against your future self

Now we turn the telescope around. The entry game was about binding an *opponent's* behavior by changing your incentives. The far more common trading use of precommitment is binding *your own* behavior — specifically, binding the disciplined, rational version of you that exists when you are calm against the frightened, hopeful version that shows up when a trade goes wrong.

The behavioral pattern that makes this necessary has a name: the **disposition effect**. It is one of the most robustly documented biases in all of finance — the tendency to sell winners too early (to "lock in" a gain and feel smart) and hold losers too long (because selling at a loss means admitting you were wrong, and because the position "might come back"). Researchers Hersh Shefrin and Meir Statman named it in 1985, and Terrance Odean's 1998 study of 10,000 brokerage accounts found investors were roughly 50% more likely to realize a gain than a loss — even though, on those same accounts, the winners they sold went on to *outperform* the losers they kept. The disposition effect is, quite literally, selling your good trades to fund your bad ones.

A **stop-loss** is the commitment device that defeats it. A stop-loss is a pre-set order: "if the price falls to \$X, sell automatically." You place it *before* the position moves against you, while you can still think clearly, and — crucially — you place it *at the broker*, so it fires without asking your permission. That last detail is what makes it a real commitment rather than cheap talk. A stop "in your head" is an empty threat to yourself: when price approaches it, the same disposition effect that made you hold will whisper "just give it a little more room." A resting order at the exchange has burned that bridge. The decision has been moved out of the panicked moment.

Let us price exactly how much that bridge is worth.

![Bar chart of expected value of a disciplined stop versus holding losers, computed from the expected value model](/imgs/blogs/commitment-devices-and-strategic-precommitment-in-trading-3.png)

#### Worked example: the expected value of a disciplined stop vs. holding the loser

Picture a position that has moved against you to **-8%**. You face a fork. Path one, the disciplined stop, takes the 8% loss now — it is certain. Path two, "hold and hope," is the disposition-effect path: you refuse to sell, and the position's future splits into branches. Suppose, realistically, that holding a broken position resolves like this:

- **55%** of the time it bounces partway and you eventually exit around **-2%** (this is the seductive branch — "see, it came back, good thing I held").
- **30%** of the time it keeps bleeding and you finally bail near **-20%**.
- **15%** of the time it gaps or cascades — a liquidation spiral, a bad earnings night, a [stop-hunt that turns into a cascade](/blog/trading/game-theory/stop-hunts-liquidation-cascades-and-the-predator) — and you are out at **-45%** with no liquidity to escape sooner.

Compute the expected value of holding with `expected_value`, which takes a list of (probability, payoff) pairs:

```
import data_gametheory as gt
hold = [(0.55, -2.0), (0.30, -20.0), (0.15, -45.0)]
print(gt.expected_value(hold))
>>> -13.85
```

So the expected return of "hold and hope" is **\$-13.85%**, versus a certain **\$-8%** for the disciplined stop. On a \$10,000 position that is an expected loss of about **\$-1,385** for holding against **\$-800** for stopping out — the stop *saves* roughly **\$585 per trade**, or about 5.85 percentage points, on average.

Read the bar chart against this math: the green bar (stop) sits at \$-8%, the red bar (hold) at \$-13.85%, and the gap between them is the value of the commitment device. The crucial, counterintuitive point: the stop wins *even though* it "always loses" \$8% and the hold path "wins" (gets back near even) 55% of the time. The disposition effect feels good 55% of the time and is a disaster on the tails. The intuition: a stop-loss is not a prediction that you are wrong; it is insurance against the 15% branch that turns a bad trade into a ruinous one.

There is a deeper reason the stop matters beyond this single-trade EV, and it has to do with the *shape* of the loss distribution. A held loser is **negatively skewed**: most of the time it is a small annoyance, and rarely it is catastrophic. Catastrophic single losses are what end trading careers, because of the brutal arithmetic of recovery — a 45% loss requires an 82% gain just to break even, while an 8% loss needs only an 8.7% gain. We treat the math of how loss size compounds into ruin in [Position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion). The stop is the device that chops off the left tail.

## Rules vs. discretion: same market, opposite outcome

The stop-loss is one rule. The broader principle is **rules-based trading**: deciding your actions in advance and binding yourself to follow them, versus **discretion** — deciding in the moment, "using judgment." The case for rules is precisely the commitment case: in the moment, your judgment is contaminated by the very emotions you are trying to manage. The disciplined plan is the calm you binding the panicked you.

To see how much this can matter, hold the *market* fixed and change only the exit rule.

![Line chart of rules versus discretion equity curves on the same trade sequence ending far apart](/imgs/blogs/commitment-devices-and-strategic-precommitment-in-trading-7.png)

#### Worked example: rules vs. discretion on the identical trade sequence

Take 20 trades. The market hands both strategies the *same* sequence of outcomes — a 45% base rate of "this trade goes your way." The only difference is how each strategy exits:

- **Rules:** let winners run to **+15%**, cap every loss at **-8%** (a precommitted stop). Per-trade expected value: `0.45 × 15 + 0.55 × (−8) = 6.75 − 4.40 = +2.35%`. A positive edge.
- **Discretion:** snatch winners early at **+6%** (the disposition effect's "lock in the gain"), and hold losers to **-18%** (the disposition effect's "give it room"). Per-trade EV: `0.45 × 6 + 0.55 × (−18) = 2.70 − 9.90 = −7.20%`. A negative edge.

Now compound those over the same 20-trade draw (a fixed reproducible sequence in which 10 of the 20 trades win, right on the 45% rate). Starting both accounts at 100:

- The **rules** account ends at about **175.7** — up roughly 76%.
- The **discretion** account ends at about **24.6** — down roughly 75%.

The trader faced the *identical market* and the *identical winners and losers*. The only thing that differed was the exit rule — let winners run and cut losers, versus cut winners and let losers run. That single behavioral choice was the difference between compounding up 76% and bleeding down to a quarter of the starting stake. The intuition: your edge does not live in picking trades; it lives in the asymmetry of your exits, and a commitment device is how you guarantee that asymmetry survives contact with your own emotions.

This is why systematic and quantitative funds exist at all. A fully systematic strategy is the ultimate commitment device: the rules are coded, the orders are sent by a machine, and there is no in-the-moment human to override them when fear or greed hits. The whole architecture is built to make deviation impossible — which is exactly the property that made Cortés's men fight. We discuss the prop-trading culture of betting an *edge* mechanically rather than a *feeling* in [the SIG poker-and-EV playbook](/blog/trading/quant-careers/sig-susquehanna-playbook-poker-game-theory-and-ev).

## What makes a commitment credible — and what makes it cheap talk

Everything so far depends on one hinge: **credibility**. A commitment that the other side (or your future self) does not believe is worthless. So how do you tell a real commitment from an empty one? The test is mechanical, and the grid below lays it out.

![Grid comparing credible commitment against empty cheap talk across cost to back down visibility and opponent response](/imgs/blogs/commitment-devices-and-strategic-precommitment-in-trading-4.png)

Read each row of the grid as a credibility test you can apply to any rule you are about to make. There are three properties that separate a real commitment from a story:

**1. Backing down must be costly or impossible.** This is the core. If you can quietly reverse the commitment at no cost, it is cheap talk. The scuttled ship, the funded defense, the broker-side stop order, the locked vault — each makes *not* following through the expensive path. The test: ask "what does it cost me to break this?" If the answer is "nothing," neither the market nor your future self will respect it.

**2. It must be observable.** A commitment the other side cannot see cannot change their behavior, because they cannot condition on it. A central bank that *privately* decides to defend a currency peg accomplishes nothing; it must *publicly* commit, so that speculators see the line and price it in. A resting buy wall on the order book is visible; a plan to "step in if it dips" is not. For your own discipline, writing the rule down and setting the order makes it observable *to you* in the weak moment — you cannot pretend the rule was something else.

**3. It must be irreversible at the moment it matters.** Timing is everything. The commitment has to bind *before* the temptation arrives and stay bound *through* it. Ulysses tied himself to the mast before the Sirens, and told the crew to ignore his later orders to be untied. A stop set at the broker is irreversible-enough in the panic moment; a mental stop is not, because the you who set it and the you who could cancel it are the same person at the same keyboard.

#### Worked example: a price floor that is credible vs. one that is not

Suppose you hold a large position and want the market to believe you will defend the \$100 level. Compare two ways to "commit":

- **Cheap talk:** you post on social media, "I am a diamond-handed holder, I will never sell below \$100." Cost to abandon: zero. If price hits \$98 and falling, your private incentive is to sell at \$98 rather than ride it to \$80. The market knows this, so it discounts your floor entirely and probes straight through \$100. Your announced floor moved nothing.
- **Credible commitment:** you place a genuine resting buy order — say, **5,000 units at \$100** — that is visible on the book, and you have the settled cash behind it so it *will* fill. Now the cost of "abandoning" the floor is that someone hits your bid and you actually buy. The commitment executes itself. A seller looking to push price below \$100 must first eat through 5,000 units of real demand. The math is concrete: if the resting order is 5,000 units and the typical seller's clip is 1,500 units, an attacker must absorb your \$100 wall with more than three average sell orders before price can break — a real, visible cost, not a tweet.

The difference between the two is not conviction or tone. It is whether breaking the commitment costs the committer anything. The intuition: credibility is not a feeling you project; it is a cost you have actually sunk, and the other side can see the size of it.

A subtle corollary, and the dark side of this whole idea: the *same* visibility that makes your commitment credible also makes it a **target**. A resting \$100 buy wall tells predators precisely where the liquidity is. A cluster of stop orders just below an obvious round number tells them exactly where a cascade of forced selling waits. We turn to that next, because a commitment device you cannot defend is a map you have handed your opponent.

## When commitment helps and when flexibility is worth more

Commitment is not a free lunch. Every option you destroy is an option you no longer have — and sometimes that option was the one you needed. The art is knowing which regime you are in. The chart below splits the world.

![Before and after comparison of when commitment wins versus when flexibility and optionality win](/imgs/blogs/commitment-devices-and-strategic-precommitment-in-trading-5.png)

Consider the two cases the chart contrasts. **Commitment wins when *you* are the main risk** — when the plan is sound and the danger is your own predictable tendency to deviate from it. If you *know* you will hold losers, a hard stop is pure gain: you are binding a self you do not trust in the moment. The world is stable enough that the right action is knowable in advance, so locking it in costs you nothing real and saves you from yourself.

**Flexibility wins when the *world* is the main risk** — when genuine new information is going to arrive that *should* change your decision, and committing early means committing in ignorance. This is the value of **optionality**: the right, but not the obligation, to act later when you know more. If a major data release is hours away and could legitimately reverse your thesis, a rigid rule set in advance throws away the value of waiting to learn. Here, keeping your hands free is the sophisticated move, and binding them is just stubbornness wearing the costume of discipline.

There is a second, sharper reason flexibility sometimes wins, and it is specific to markets: **a fixed, visible commitment can be hunted.** A stop-loss is a commitment device against your own behavior — but it is *also* a resting sell order that a predator can see (or infer from where the obvious levels are). Clusters of stops sit just below round numbers and recent lows, because that is where everyone's "obvious" invalidation lives. A large player can push price into that cluster precisely *to trigger the cascade* — your committed sell order becomes the fuel that drives price down to where they want to buy, after which it reverses and leaves you stopped out at the worst possible price. This is the [stop-hunt](/blog/trading/game-theory/stop-hunts-liquidation-cascades-and-the-predator), and it is the reason a commitment device is never purely defensive. The obviousness that makes a [round number a focal point](/blog/trading/game-theory/schelling-points-and-focal-prices-round-numbers-and-obvious-levels) is the same obviousness that makes the stops behind it a target.

#### Worked example: the cost of a hunted stop vs. the cost of no stop

Suppose your real invalidation — the price at which your thesis is genuinely wrong — is \$92, but the *obvious* level (just under a round \$95 and a visible prior low) is \$94.50, and that is where you and everyone else parks the stop. Two failure modes:

- **No stop at all (pure flexibility):** you keep the option to decide later. But the disposition effect means that when price hits \$90 you "give it room," and you ride it to the -20% disaster branch from our earlier example. Expected cost: the \$-13.85% hold-and-hope EV.
- **An obvious stop at \$94.50 that gets hunted:** a predator drives price to \$94.40 to trigger the cluster, you sell, and price snaps back to \$98. You ate a roughly **-5.5%** loss (entry near \$100 to \$94.50) on a thesis that was *never actually invalidated*, then watched it recover without you.

The fix is not to abandon the stop — that re-opens the disposition-effect disaster. It is to make the commitment *less obvious and better placed*: set the stop at your *real* invalidation (\$92, below the hunt zone) and in a *size* the market cannot easily see, or use a wider mental rule combined with a hard *catastrophe* stop far below. The intuition: the goal is to bind your panicked self without painting a target your opponents can aim at — credible to you, invisible to them.

So the decision rule is: **commit against your own predictable weakness, stay flexible against genuine uncertainty, and never place a commitment where a predator can use it against you.** Most retail traders get this exactly backwards — rigid where the world is changing (refusing to update a thesis on real news because "the plan said hold") and flexible where their own behavior is the risk (no hard stop, "I'll use my judgment"). Flip both.

## The burning-bridges logic, formalized

We opened with Cortés. Now that we have the vocabulary, here is the burning-bridges logic as a clean strategic sequence — the purest illustration of gaining power by destroying your own options.

![Pipeline of the burning bridges logic where destroying retreat makes the threat credible and the enemy withdraws](/imgs/blogs/commitment-devices-and-strategic-precommitment-in-trading-6.png)

Read the pipeline left to right as a chain of strategic reasoning. A defender starts with two options — fight or retreat. As long as *retreat* is available, an attacker reasons: "If I press hard, the defender will probably flee rather than die, so attacking is cheap." That is the empty-threat world. Now the defender **burns the bridge**: retreat is destroyed, a sunk and irreversible act. Only fighting remains, so the defender's threat to fight to the end becomes *credible* — not because the defender is braver, but because the alternative no longer exists. The attacker, reasoning backward from a foe who cannot flee, sees that the fight will be maximally costly and *withdraws*. The defender wins by reducing its own options, and — the punchline of every good commitment — never has to fire a shot. The equilibrium moved off the battlefield entirely.

This is structurally identical to the entry-deterrence matrix from earlier: removing your own preferred-but-weak option (retreat / accommodate) makes your strong option (fight) credible, which deters the opponent. The doomsday device of Cold War strategy is the same idea taken to its terrifying limit — an automated retaliation you *cannot call off* makes deterrence perfectly credible precisely because no human can lose their nerve. And the market versions are everywhere: a poison pill that *automatically* dilutes a hostile acquirer makes "we will fight a takeover" credible without the board having to decide in the moment; a currency floor backed by *unlimited* printing makes "we will defend the peg" credible because the central bank has pre-committed to act regardless of how it feels at the time.

The same logic, pointed inward, is the systematic strategy: by coding the rules and removing the human override, you burn the bridge of discretion. The panicked you cannot retreat into "just this once I'll override the stop," because there is no you in the loop to do it.

## Common misconceptions

**"More options are always better, so committing is always a mistake."** This is true in a one-player decision against nature and false in a game. Against nature, you can always ignore an option you don't want, so extra options are free. In a game, your options are information the other side uses to plan against you; removing an option can change their behavior in your favor by more than the option was worth. The entry-deterrence matrix proves it numerically: the incumbent made itself worse off in some cells (accommodating fell from 10 to 8) and ended up better off overall (10 instead of 5), because the self-handicap flipped the equilibrium.

**"A stop-loss just locks in losses; real conviction means holding through the dip."** A stop-loss does take a certain small loss — that is the point. But the comparison is not "small loss vs. zero"; it is "small certain loss vs. the expected value of holding," and we computed that holding a broken position is worth \$-13.85% against the stop's \$-8%. The 55%-of-the-time bounce that "rewards" holding is exactly what makes the disposition effect feel good while it slowly bleeds you, because the 15% tail is catastrophic and a 45% loss needs an 82% gain just to recover. Conviction is for sizing the entry, not for refusing to admit the thesis is broken.

**"A threat is a threat — saying it loudly makes it credible."** Volume is not credibility. A threat is credible only if carrying it out is in your interest *at the moment you'd have to carry it out* (or if you've removed your ability to not carry it out). The incumbent's shouted "I'll fight you" was worth nothing because, the instant the rival entered, fighting was the incumbent's own worst option. Credibility comes from a sunk cost the other side can see — a funded defense, a resting order, a binding contract — not from tone.

**"If I just decide firmly enough, my mental stop will hold."** The you who sets the mental stop and the you who could cancel it are the same person at the same keyboard, and the cancel happens at the worst possible moment — when fear and the disposition effect are loudest. A mental stop is an empty threat to yourself for the same reason the incumbent's shout was empty: in the moment of truth, your incentive is to *not* follow through. Only moving the decision out of the moment — a broker-side order, a coded rule — makes the commitment real.

**"Systematic trading removes judgment, which is obviously worse than a smart human deciding."** It removes *in-the-moment* judgment, and that is a feature, not a bug, precisely because in-the-moment judgment is contaminated by exactly the emotions you most need to control. The judgment hasn't vanished; it has been moved to the calm research phase where the rules were designed and tested. The rules-vs-discretion chart shows the same market handing one path +76% and the other -75% on identical trades — the difference was entirely the *removal* of in-the-moment overrides.

**"Commitment is always defensive — it can only protect, never cost me."** A commitment is a double-edged sword by construction: the same irreversibility that makes it credible is what makes it dangerous when the world surprises you. The Swiss franc floor protected exporters for three years and then bankrupted brokers in a single morning; leverage commits you to a thesis and then forces you to abandon it at the worst price; a visible stop defends your discipline and simultaneously paints a target. There is no commitment that is pure upside. The honest question is never "does this protect me?" but "what does this cost me in the world where I am wrong?" — and if you cannot answer that, you have not understood the device you just armed.

## How it shows up in real markets

**Currency floors: the Swiss National Bank, 2011–2015.** In September 2011, with the franc soaring as a safe haven and crushing Swiss exporters, the SNB announced it would enforce a *minimum* exchange rate of 1.20 francs per euro and would defend it by "buying foreign currency in unlimited quantities." That last phrase is the commitment device — the word *unlimited* removed the SNB's own option to give up, making the floor credible. For three and a half years it held, and speculators stopped fighting it because a central bank that can print its own currency without limit genuinely cannot be overwhelmed on the sell-franc side. Then on January 15, 2015, the SNB *abandoned* the floor without warning. The franc instantly surged about 30% against the euro in minutes; brokers like Alpari UK and several FX shops were bankrupted, and retail accounts were wiped out and pushed negative. The lesson cuts both ways: a credible commitment is enormously powerful while it holds, and *catastrophic* the instant it breaks — because everyone had committed *their own* positions on the assumption that the SNB would not.

**Poison pills.** A poison pill (formally a "shareholder rights plan") is a corporate commitment device: a pre-arranged rule that, *if* any hostile buyer accumulates past a trigger (often 10–20% of shares), automatically lets all *other* shareholders buy new stock cheap, massively diluting the raider. The power is that it is automatic and pre-committed — the board does not have to find its nerve in the heat of a bid; the dilution just happens. In April 2022, Twitter's board adopted a poison pill within days of Elon Musk's roughly \$43 billion offer, capping any accumulation above 15%. The pill did not "stop" the deal — pills rarely block a determined buyer outright — but it forced the bid to the *negotiating table* on the board's terms, which is exactly what a credible commitment is supposed to do: change the other side's strategy.

**The Cortés of crypto: locked liquidity and time-locked vaults.** In decentralized finance, a project founder can credibly commit not to "rug" (drain the pool and vanish) by *locking* the liquidity in a smart contract for a fixed period — the funds are provably untouchable, on-chain, for everyone to verify. This is a pure burning-bridges device: the founder destroys their own option to exit-scam, and *because* the option is verifiably gone, investors are willing to participate. The credibility is mechanical, not reputational — the contract code is the sunk cost. The flip side is that the same on-chain visibility that proves the lock also shows every observer exactly when it expires.

**Stop-hunts and the visibility curse.** Because stops cluster at obvious levels — just under round numbers, prior lows, moving averages — they form a visible pool of forced-selling fuel. In crypto especially, where liquidation levels of leveraged positions are partly inferable, price routinely wicks down into a cluster, triggers a cascade of forced sells, and snaps back, leaving committed traders stopped out at the bottom of a move their thesis never justified. The May 2021 and subsequent leverage flushes are full of these: a relatively small push into a dense liquidation band cascades into outsized moves as each triggered stop becomes the next one's trigger. The commitment device (your stop) became the predator's ammunition. The defense is not to drop the stop but to place it at your *real* invalidation rather than the *obvious* one, and to keep its size from being readable.

**The Fed and "forward guidance."** Central-bank credibility is a commitment story at the macro scale. When the Federal Reserve commits to a path — "rates will stay near zero until inflation is sustainably at target" — the commitment only works if the market believes the Fed will not blink. A central bank that has *built* credibility over decades (by following through on painful tightening before) can move long-term rates today merely by announcing intent, because its threat is credible. One that has cried wolf cannot. This is why "don't fight the Fed" is a commitment-credibility statement: you are betting the central bank's threat is more credible than your position. We unpack the public-signal mechanics in [Buy the rumor, sell the news](/blog/trading/game-theory/buy-the-rumor-sell-the-news-public-signals-and-the-fed).

**The commitment you didn't choose: forced precommitment by leverage.** Not every commitment is a clever choice. Leverage is a commitment device imposed *on* you — when you borrow to trade, you have pre-committed to sell at your liquidation price whether you want to or not, because the lender's automated margin engine will do it for you. Long-Term Capital Management in 1998 is the textbook disaster: a fund run by Nobel laureates, levered roughly 25-to-1 (and far more counting derivatives), found that its size *was* its commitment — it was too big to exit quietly, so when its trades moved against it, every attempt to reduce risk pushed prices further against itself. The Federal Reserve organized a roughly \$3.6 billion rescue by a consortium of banks in September 1998, not out of charity but because LTCM's forced unwind threatened the whole system. The lesson for a trader: leverage *removes* your option to wait out a drawdown — it is a burned bridge you did not mean to burn, binding the future you to sell at exactly the moment selling is worst. Position limits are the voluntary commitment device that stops the involuntary one from ever triggering.

**The doomsday limit and why no one builds it.** The logically purest commitment device is the doomsday machine: an automated, un-recallable retaliation that makes a threat perfectly credible because no human can lose their nerve. Cold War strategists studied it precisely because it is the *most* credible deterrent imaginable — and concluded no one should ever build one, because a credible commitment that fires on a *false* signal is a catastrophe with no off-switch. The market echo is the fully automated trading rule with no human override: maximally committed, and therefore maximally dangerous if the world does something the rule's designers never modeled. The 2010 "flash crash" and the 2012 Knight Capital incident — where a software error fired \$7 billion of unintended orders in 45 minutes and cost the firm about \$440 million, nearly destroying it overnight — are reminders that a commitment device with no human in the loop will execute its mistake with the same perfect discipline it executes its plan. The art is committing hard enough to bind your behavior, but never so hard that a false signal becomes unstoppable.

## The playbook / How to play it

Here is how to actually use commitment and precommitment as a trader, layer by layer.

**Know which game you are in.** First decide whether your risk is *you* or *the world*. If you are predictably tempted to deviate — hold losers, cut winners, oversize after a loss, revenge-trade — your main opponent is your own future self, and a commitment device is pure gain. If genuine new information is coming that should change your call, your main risk is the world, and rigid commitment will make you commit in ignorance. Most traders misdiagnose this: rigid on theses (refusing to update on real news) and loose on behavior (no hard stop). Flip it.

**Build commitment devices against your own weakness.** The toolkit, in rough order of how strongly each binds:

- **Broker-side stop orders**, not mental stops — moved out of the panic moment, fired by the exchange.
- **A written trading plan** with entry, invalidation, and size *decided before* the trade, so the calm you overrules the panicked you. Co-locate the plan with the position so you cannot pretend the rule was different.
- **Hard position and leverage limits** — a maximum size per trade and per book, so a single conviction cannot blow up the account.
- **Rules-based / systematic execution** — code the rules and let a machine send the orders; the ultimate burned bridge, because there is no in-the-moment human to override.
- **Auto-investing and locked funds** for the savings side — automatic contributions and accounts you cannot raid impulsively bind the long-term you against the bored, fearful one.

**Make every commitment actually credible.** Apply the three-part test before trusting any rule: (1) Is backing down costly or impossible? A mental stop fails; a resting order passes. (2) Is it observable in the weak moment, so you cannot pretend otherwise? Write it down; set the order. (3) Is it irreversible *when it matters* — bound before the temptation, held through it? If a rule fails any of the three, it is cheap talk, and the panicked you (or the market) will treat it as such.

**Use precommitment against opponents only when the cost is real.** If you want a level believed — a floor you'll defend, a bid you'll hold — back it with a genuine resting order and the settled cash to fill it, not a tweet. A visible, funded commitment changes how others trade against the level; an announcement does not. And size it so the cost of breaking through is a real number an attacker must pay (our \$100 wall of 5,000 units against 1,500-unit clips), because that visible cost *is* the credibility.

**Defend your commitments from the visibility curse.** The same visibility that makes a stop credible makes it a target. Place stops at your *real* invalidation, below the obvious hunt zone, not at the round-number level where everyone's stops cluster. Keep size unreadable. Consider a wider behavioral rule plus a hard catastrophe-stop far below, so a predator cannot cheaply trigger you out of a thesis that was never broken. The goal: credible to you, invisible to them.

**The invalidation of this whole approach.** Commitment is a tool, not a religion. It *helps* when your plan is sound and you are the deviation risk; it *hurts* when the plan itself is wrong and the rule keeps you in a losing structure, or when real news arrives that should change the call and the rule forbids updating. A commitment device cannot fix a bad thesis — it can only stop a good process from being sabotaged by a panicked moment. If you find your rules forcing you into trades you'd never take fresh, the commitment has become the disease, not the cure. Bind your behavior; never bind your judgment about whether the world has actually changed.

The deepest lesson is Schelling's, and it is genuinely strange: in a world of strategic opponents — including the strategic opponent that is your own future self — freedom is not always strength. Sometimes the strongest move on the board is to take a move *off* your own board, visibly and irreversibly, so that everyone (you included) knows the version of you who would have flinched is no longer in the room. The trader who can do that on purpose, and who knows exactly when *not* to, is playing the game one level deeper than the one who keeps every option open and loses to their own hand.

*This is educational, not individualized financial advice. Commitment devices and stops manage risk; they do not eliminate it, and any position that can profit can also lose.*

## Further reading & cross-links

- [The prisoner's dilemma in markets: why everyone sells at once](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once) — the coordination failure that commitment devices are often trying to escape.
- [Mixed strategies and the value of being unpredictable](/blog/trading/game-theory/mixed-strategies-and-the-value-of-being-unpredictable) — the opposite reflex: when *not* committing to a fixed move is the edge, so a hunter cannot read you.
- [Stop-hunts, liquidation cascades, and the predator](/blog/trading/game-theory/stop-hunts-liquidation-cascades-and-the-predator) — the dark side of a visible commitment, where your stop becomes someone else's ammunition.
- [Schelling points and focal prices: round numbers and obvious levels](/blog/trading/game-theory/schelling-points-and-focal-prices-round-numbers-and-obvious-levels) — why obvious levels gather both commitments and the predators who hunt them.
- [Position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) — the recovery arithmetic that makes cutting the left tail worth a small certain loss.

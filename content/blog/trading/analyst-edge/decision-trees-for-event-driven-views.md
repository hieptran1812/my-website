---
title: "Decision Trees for Event-Driven Views"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Turn any scheduled catalyst into a decision tree: branch the outcomes, price the move, assign odds, pre-commit one action per branch, roll back the EV, and decide before the number prints."
tags: ["analysis", "market-view", "decision-tree", "event-driven", "expected-value", "fomc", "earnings", "expected-move", "pre-commitment", "trading-process"]
category: "trading"
subcategory: "The Analyst's Edge"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Around a scheduled event the outcome space branches. A decision tree maps each branch — outcome times probability times market reaction times your pre-planned action — and rolls the expected value back from the leaves to a single number you decide on before the print.
>
> - A tree forces you to **pre-decide your reaction in every branch**, so when the number lands you execute a plan instead of inventing one under adrenaline.
> - Build it in five columns: the event, the outcomes, your probabilities, the market reaction per outcome, and the action you will take. Then roll the EV back from the leaves.
> - **Price the move first.** The options straddle hands you the one-sigma move the market has already paid for; your tree only adds value where your branch odds differ from what's priced.
> - The one rule: **the most likely branch is not the plan — every branch is the plan.** You trade the whole tree's EV, and you commit the action for each leaf in writing before the event.

It is 1:58 p.m. on an FOMC day. Two traders hold the same position: a long in the 2-year Treasury note, betting that the Fed is closer to cutting than the market thinks. The decision drops at 2:00 sharp.

The first trader is "waiting to see the number." He has a view — dovish — but no plan for what he does if the statement comes out hawkish, or exactly in line, or more dovish than even he expects. At 2:00:00 the headline crosses, the 2-year yield gaps twelve basis points *higher* in five seconds, his screen turns red, and his pulse spikes. Now he is making the single most important decision of his week — hold, double down, or bail — in a state of acute physiological stress, with the price moving against him every second he hesitates. He freezes for ten seconds, then panics out at the worst tick, locking a loss that his own thesis said was a 30% scenario he should have been *prepared* for.

The second trader is bored. She drew her decision tree the night before. She knows that a hawkish surprise sends the note down about 1.5%, that her stop sits there, and that her pre-committed action is to cut the position cleanly and not re-enter today. She knows the in-line case is a near-non-event where she trims into any relief pop. She knows the dovish case is where she adds. When the hawkish print lands, she does not decide anything — she *executes* the decision she already made, calmly, in the first second. The number didn't surprise her into a mistake, because she had already lived through all three numbers on paper. That difference — a plan made cold versus a plan made hot — is most of the edge in event-driven trading, and a decision tree is the tool that manufactures it.

![An event branches into outcomes each with a probability a market reaction and a pre-planned payoff](/imgs/blogs/decision-trees-for-event-driven-views-1.png)

This is the whole idea on one page. The event is on the left. It fans into the outcomes the world can produce — hawkish, in line, dovish — each tagged with the probability *you* assign. Each outcome maps to a market reaction, and each reaction maps to a payoff for your specific position and the action you pre-committed to take. The job of this post is to teach you to build that tree for any scheduled catalyst, to price the move into it, to roll the expected value back from the leaves to a single go/no-go number, and — most importantly — to use it to pre-decide your reactions so you never again make a five-figure decision in the ten seconds after a print.

## Foundations: what a decision tree actually is

A **decision tree** is a diagram of a decision that unfolds in stages, where some stages are *your choices* and some are *the world's outcomes*. It is the oldest tool in formal decision analysis, and it is built from three kinds of object.

A **node** is a point in the diagram. There are two types. A **chance node** is a point where the *world* decides — the event prints, and one of several outcomes occurs, each with a probability. A **decision node** is a point where *you* decide — you choose an action, and you control which branch you take. The opening figure has one event (a chance node: the FOMC prints hawkish, in line, or dovish) and, hanging off each outcome, a decision (what you do in response).

A **branch** is a line out of a node. Out of a chance node, each branch is one possible outcome of the world, and the branches carry probabilities that sum to 100%. Out of a decision node, each branch is one action you could take, and you pick exactly one.

A **leaf** is the end of a path — the far right of the tree — where you write down the **payoff**: the dollar profit or loss your position ends with if the world and your choices followed that exact path. The payoff is the whole point. Everything to the left exists to get you to a number at every leaf.

The **event outcome space** is just the complete list of things the event can do, carved into branches that are *mutually exclusive* (only one happens) and *collectively exhaustive* (one of them definitely happens, so the probabilities sum to 100%). For a rate decision the natural carve is hawkish / in line / dovish. For a clinical trial it is approved / rejected. For earnings it is beat / miss against the number the price is discounting. Carving the outcome space well is half the skill, and we will spend real time on it.

A **conditional reaction** is how the market moves *given* a particular outcome. The phrase "given that" is doing heavy lifting: the note doesn't have one reaction, it has a reaction *conditional on hawkish*, a different reaction *conditional on in line*, and another *conditional on dovish*. Your job is to estimate each conditional reaction separately, because they are genuinely different numbers, and the average of them tells you nothing useful by itself.

Finally, **EV roll-back** (also called *folding back* or *backward induction*) is the arithmetic that collapses the whole tree into one number. You start at the leaves, where the payoffs live. At each chance node you compute the **expected value** — the probability-weighted average of the payoffs of its branches. At each decision node you take the *best* branch, because you get to choose. You roll these values leftward, node by node, until the root carries a single number: the EV of the entire tree, which is the EV of the trade. We give expected value its own deep treatment in [Expected value: the only math a view really needs](/blog/trading/analyst-edge/expected-value-the-only-math-a-view-really-needs); here we use it as the engine that turns a tree into a decision.

The distinction between chance nodes and decision nodes is worth pinning down hard, because mixing them up is the most common modeling error. At a **chance node** you have *no control* — you can only assign probabilities to the branches and live with whatever the world picks. The FOMC printing hawkish, in line, or dovish is a chance node: you do not get to choose which the Fed does. At a **decision node** you have *full control* — you pick exactly one branch, and the rule for valuing a decision node is therefore different. At a chance node you take the probability-weighted *average*; at a decision node you take the *maximum*, because you will obviously choose the most valuable action available to you. A useful way to keep them straight while drawing: draw chance nodes as circles and decision nodes as squares, the convention from formal decision analysis. The order matters too — in event trading the usual shape is a decision (do I put the trade on? hold or flatten?) *before* a chance node (the print) and another decision *after* (what do I do given the print). That decision-chance-decision sandwich is the skeleton of nearly every event trade.

The reason this matters in dollars: a chance node is risk you are *exposed to*, and a decision node is a lever you *control*. Good event trading is largely about adding decision nodes that cap the bad branches of chance nodes — a stop is a decision node bolted onto the downside of a chance node, converting an open-ended loss into a known, bounded one. When you find a trade with terrible EV, the fix is often not to abandon it but to insert a decision node (a stop, a hedge, a partial trim) that lops off the worst leaf. The tree shows you exactly which leaf is dragging the EV, so you know precisely where to bolt on the lever.

### Why pre-deciding beats reacting live

The deepest reason to build a tree has nothing to do with the arithmetic. It is about *when* you make the decision.

When you "wait to see the number," you are choosing to make your most consequential decision at the single worst possible moment: the instant of maximum surprise, maximum price velocity, and maximum adrenaline. Your heart rate is up, the loss (or gain) is being marked against your account in real time, and the part of your brain that does careful probabilistic reasoning has been hijacked by the part that handles physical threats. Every study of decision-making under acute stress points the same way — you get narrower, more loss-averse, more prone to freezing and to impulsive flight. This is not a character flaw you can train away; it is how the nervous system works. The full mechanics of how this wrecks execution are laid out in [trading psychology and the execution gap](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap).

A decision tree moves the decision *backward in time*, to a moment when you are calm, the position is not yet moving, and you can reason about all the branches at once. You pre-decide: "if hawkish, I cut at the stop; if in line, I trim a quarter into strength; if dovish, I add at the first pullback." When the print lands, there is no decision left to make. You read which branch the world chose and execute the action you already wrote down. The live moment becomes pure execution, not analysis — and execution is something a stressed nervous system can do fine.

That is the entire value proposition: a decision tree is a machine for relocating your decisions from the hot moment to a cold one. The expected-value arithmetic is real and useful, but the pre-commitment is what saves you the five-figure mistakes.

## Building the tree: five columns from event to payoff

Here is the construction, column by column, in the order you fill them in. We will build a concrete FOMC tree as we go and carry the same numbers all the way to the EV.

**Column 1 — the event.** Write the catalyst and your current position. Be specific about timing and size, because the tree is for *this* position, not the abstract market. Ours: *the 2:00 p.m. FOMC decision, and I am long \$20,000 of the 2-year note.*

**Column 2 — the outcomes.** Carve the outcome space into mutually exclusive, collectively exhaustive branches. The number of branches is a judgment call: too few and you miss a path that matters; too many and the tree becomes unmanageable and the probabilities get noisy. For most scheduled macro events, three branches — a hawkish/bad tail, the consensus middle, and a dovish/good tail — is the sweet spot. Ours: *hawkish surprise, in line, dovish surprise.*

**Column 3 — your probabilities.** Assign a probability to each branch, summing to 100%. These are *your* odds after doing your work, not the market's. The gap between your odds and the market's priced odds is literally where your edge lives, and we will make that gap explicit in a moment. Ours: *30% hawkish, 50% in line, 20% dovish.*

**Column 4 — the market reaction, conditional on each outcome.** For each branch, estimate how far and which way the price moves *given* that outcome. This is where you use the asset's reaction function — how sensitive it is to a surprise of this kind — covered mechanically in [the reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently). Ours, for the 2-year note: *hawkish → −1.5%, in line → +0.2% relief drift, dovish → +1.8%.*

**Column 5 — your pre-planned action and the payoff.** For each branch decide, in advance, what you will do, and compute the dollar payoff that results. The action shapes the payoff — a stop that cuts the hawkish branch at −1.5% caps that loss; riding the dovish branch captures its full +1.8%. Ours, on the \$20,000 position: *hawkish → cut on the stop, −\$300; in line → hold or trim, +\$40; dovish → add and ride, +\$360.*

That is a complete tree. Five columns, three branches, a number at every leaf. The opening figure is exactly this tree drawn out. Now we make it pay.

### Carving the outcome space without lying to yourself

The branches in column 2 are where most trees go wrong, and the errors are subtle enough that they slip past you unless you know to look. There are three traps.

The first is **non-exhaustive branches** — leaving out an outcome that can actually happen. The classic miss is forgetting the "nothing happens / in line" branch and modeling only the bull and bear cases, which inflates your EV because you've dropped the most likely, lowest-payoff outcome. The fix is the discipline of the phrase *collectively exhaustive*: after you list your branches, ask "is there a number the event could print that doesn't fall into any of these?" For a continuous event the answer is always yes unless you have a middle branch, so a continuous event needs at least three. The probabilities must sum to exactly 100% — if they sum to 90%, you have a missing 10% branch hiding somewhere, and that branch usually contains a payoff you didn't want to think about.

The second is **overlapping branches** — outcomes that aren't mutually exclusive, so a single real-world result fires two branches at once. This corrupts the EV because you double-count. The cure is to define branches by a *single, observable threshold* on one variable: for CPI, "below 2.9%," "2.9% to 3.1%," "above 3.1%" — three branches that carve the number line cleanly, with no gaps and no overlaps. If you find yourself writing a branch like "hawkish *and* the dollar rallies," you have two variables tangled together; split them into a two-stage tree (the rate decision, then the dollar reaction conditional on it) rather than one muddled branch.

The third, and most dangerous, is **motivated carving** — drawing the branches to flatter the trade you already want to do. You want to be long, so you draw a fat 60% bull branch and a thin 15% bear branch and call it analysis. The tell is that your probabilities always seem to favor the position you came in wanting. The defense is to carve the branches and assign reaction sizes *before* you look at your own odds, then read the priced odds, and only then write your probabilities — and to write down *why* yours differ from the market's in a single sentence per branch. If you can't articulate why the market is wrong about a branch, use the market's number. The decision journal habit from [structuring a thesis](/blog/trading/analyst-edge/structuring-a-thesis-claim-evidence-and-catalyst) applies directly: every probability you write should have a one-line reason attached.

### Rolling the EV back from the leaves

With payoffs at the leaves and probabilities on the branches, the expected value of the trade is the probability-weighted sum of the leaf payoffs:

$$\text{EV} = \sum_{\text{branches}} P(\text{branch}) \times \text{Payoff}(\text{branch})$$

For our FOMC tree:

$$\text{EV} = 0.30 \times (-\$300) + 0.50 \times (+\$40) + 0.20 \times (+\$360)$$

$$\text{EV} = -\$90 + \$20 + \$72 = +\$2$$

![Each branch contributes odds times payoff and the contributions sum to the tree expected value](/imgs/blogs/decision-trees-for-event-driven-views-2.png)

The chart rolls the EV back visually: each branch contributes its probability times its payoff — the hawkish branch drags −\$90, the in-line branch adds a token +\$20, the dovish branch adds +\$72 — and the three stack into a tree EV of barely +\$2. That number is the punchline of the whole construction. *This trade, sized and structured exactly this way, has an expected value of about two dollars.* It is essentially a coin flip with a slight positive lean. Before you built the tree, "I'm dovish, I'll be long into the Fed" felt like a real trade. After you built it, you can see it is almost EV-neutral — the small dovish edge is nearly cancelled by the fat hawkish tail. That is not a reason to feel bad; it is the tree doing its job, which is to *tell you the truth about the trade before you put it on.*

The roll-back is more interesting when there is a *decision* node in the tree, not just a chance node. Suppose, instead of being committed to holding through the print, you have the option to *take the position off before 2:00 p.m.* and avoid the event entirely. Now the tree has a decision node at the root: branch A is "hold through the print," whose EV we just computed as +\$2; branch B is "flatten before the event," whose payoff is \$0 (you are out, you capture nothing and risk nothing). At the decision node you take the better branch. Since +\$2 barely beats \$0 — and carries real variance — a sensible trader might fold this particular tree and stand aside. The roll-back didn't just value the trade; it valued *the option to skip it*, which is itself a position.

### Waiting for the print is a position

That last point deserves its own heading, because it is the subtlest idea in event trading and most people get it backwards.

Doing nothing is not the absence of a decision. **Standing aside through an event is an active position with its own EV — usually \$0 — that every other branch must beat.** When you flatten before a catalyst, you are choosing the \$0 leaf on purpose. That is the correct move whenever the tree's EV from being involved doesn't clear \$0 by enough to pay you for the variance and the stress. Our FOMC tree's +\$2 EV does not clear that bar; the disciplined read is "no edge, stand aside, redeploy when the dust settles."

This reframes the most common event-trading question. People ask "should I be long or short into the number?" The better question is "is being *involved at all* worth more than being *flat*?" The flat branch is the benchmark. A view only earns a position when its tree EV beats the flat \$0 leaf by a margin that compensates you for the risk — and a great many scheduled events fail that test, which is exactly why experienced traders sit out far more catalysts than beginners expect. The tree makes the flat option a first-class branch you have to actively beat, instead of an afterthought.

There is an even subtler form of optionality the tree exposes: the value of *waiting until after the print to act*. Sometimes the highest-EV branch is not "long into the event" or "flat through the event" but "flat into the event, then position aggressively once the branch is known." This is the fade-the-overshoot trade: events routinely overshoot in the first minutes and mean-revert over the next hour, so the trader who stands aside through the print and then steps in once the move has overextended is harvesting the reversion. Drawn as a tree, this is a decision node placed *after* the chance node — you let the world resolve the uncertainty for free, then make your decision with strictly more information and no exposure to the gap. Whenever the post-print decision node has higher EV than the pre-print one, the discipline is brutal and simple: do nothing until 2:00:01, then act. The opportunity cost of waiting is usually small; the value of deciding with the outcome already known is often large. Beginners can't stand the idea of "missing the move," so they pre-position; professionals know the move you join *after* the uncertainty resolves is frequently the better-EV move.

### Pricing the expected move into the tree

Where do the reaction sizes in column 4 come from? You do not have to guess them. For any liquid asset with options, **the market has already priced the move for you**, and you can read it straight off the options.

The **expected move** (also called the *implied* or *priced* move) is the one-standard-deviation move the options market has priced into the event. The quick way to read it: take the at-the-money **straddle** — buying both the call and the put at the current price — for the expiry that spans the event. Its total premium, as a percent of the underlying, is roughly the one-sigma move the market expects by that expiry. If a stock trades at \$200 and the straddle that covers earnings costs \$16, the priced move is about \$16, or 8%. The market is saying: *we expect this thing to move about 8% on the print, one way or the other.* The full mechanics of how a market maker derives and hedges that number — and why it sits on the other side of your trade — are in [how an options market maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade).

![A priced one-sigma move with branch cuts at minus and plus two percent](/imgs/blogs/decision-trees-for-event-driven-views-5.png)

The chart shows how the priced move feeds the tree. The blue curve is the distribution the straddle implies — a one-sigma move of about ±4% for our FOMC example. Your tree carves that distribution into branches: the hawkish branch is the left region (a move worse than −2%), the in-line branch is the priced cone in the middle (−2% to +2%), and the dovish branch is the right region (better than +2%). The straddle hands you the *size* of the move; your tree decides *where to cut the branches* and *how likely each is*. This is the right division of labor: let the options market price the magnitude — that is its specialty and it is very good at it — and spend your effort on the thing you might know better, which is the *direction* and the *odds*.

The expected move also gives you a discipline check. If your tree's branch reactions are wildly bigger than the priced move, you are assuming a move the options market would happily sell you cheaply — a sign you are over-excited. If they are much smaller, you may be ignoring tail risk the market is paying up to hedge. Anchor your column-4 reactions to the priced move, then adjust for your specific view.

### Binary events versus continuous surprises

Not all events branch the same way, and the shape of the outcome space changes how you build the tree.

A **binary event** has, essentially, two outcomes with a gap and little in between. A drug gets FDA approval or it doesn't. A company wins the lawsuit or loses. An election goes one way or the other. A merger closes or breaks. The defining feature is *discontinuity*: there is no "slightly approved." The stock gaps to one of two prices and the middle is empty. For these, a two-branch tree is the honest representation, and the whole game is the two probabilities and the two payoffs. The expected move from the straddle will be large — options on binary events are expensive precisely because the underlying is going to jump — and the priced odds are readable from the option skew or, for events like elections, from prediction markets.

A **continuous event** produces a number on a spectrum, and the reaction scales with the surprise. A CPI print can come in at 2.9% or 3.1% or 3.4%, and the bond reaction is roughly proportional to how far the actual lands from what was expected — this is the surprise framework, and the sensitivity is the *beta to the surprise*, covered in [the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises). For continuous events you *could* draw dozens of branches, but you don't need to. Three branches — a meaningful downside surprise, a near-consensus middle, and a meaningful upside surprise — capture the decision-relevant structure. You are not trying to model the full distribution; you are trying to pre-decide your action in the three cases that would actually change what you do.

The practical tell: if the event can produce a smooth range of numbers, it's continuous, carve it into a low/mid/high three-branch tree. If it produces one of two discrete states, it's binary, use two branches and focus all your energy on the two probabilities. We will work a full example of each.

The shape of the event also dictates *how you size and protect* the trade, which is why getting the classification right matters beyond the diagram. For a continuous event, the reaction is bounded and roughly proportional to the surprise, so a stop is a real tool — you can place it at a level the price has to *pass through* on its way down, and it will usually fill near where you set it. Your downside is genuinely capped at the stop, and the tree's hawkish-branch payoff reflects that cap. For a binary event, there is no "passing through" — the price *gaps* from one state to the other overnight or in a single tick, and a stop set at −5% does you no good when the stock opens −60%. There is no orderly exit on a binary gap, so the only protection is *size*: the entire position is at risk, full stop, and you control your loss solely by how much you commit up front. This is why the binary worked examples lean on a sizing rule while the continuous ones lean on a stop. Misclassify a binary as continuous — "I'll just put a stop on it" — and you discover, the hard way, that the stop was a fiction the moment the gap printed.

### Using the tree to pre-commit reactions

The five-column tree gives you the EV. The *reaction plan* turns the tree into something you can execute under fire. Take the action column and rewrite it as a lookup table: **a trigger you can see on the tape, mapped to one action you will take, decided in advance.**

![A grid mapping each branch to a tape trigger and one pre-committed action](/imgs/blogs/decision-trees-for-event-driven-views-4.png)

The grid is the FOMC tree's action column made operational. Each row is a branch. The middle column is the *trigger* — the specific, observable thing on your screen that tells you which branch the world chose: the 2-year yield jumping +12 bps in five minutes means hawkish; a move under 0.4% either way means in line; the yield dropping 15 bps with a softer statement means dovish. The right column is the *one action* you pre-committed: cut on the stop and don't re-enter; hold the core and trim a quarter; add 50% at the first pullback and trail the stop.

The discipline this enforces is enormous. You are not deciding what hawkish *means* at 2:00:01 — you decided that last night, and you tied it to a number (+12 bps) you can read in a glance. You are not deciding whether to hold or fold — the row tells you. The tape shows you which branch fired, you find the row, you execute the action. Adrenaline cannot corrupt a lookup. This is the same principle as a surgeon's checklist or a pilot's emergency card: when the stakes are high and the time is short, you do not want to be reasoning from first principles, you want to be reading off a plan you reasoned out when you were calm.

The triggers are the part people underspecify, and a vague trigger is almost as bad as no plan. "If it's hawkish" is not a trigger — it requires a live judgment about what counts as hawkish, made in the exact moment you can't judge well. "If the 2-year yield is up more than 12 bps thirty seconds after the print" is a trigger: it is a number on a screen, true or false, no interpretation required. Tie every branch to an observable, quantitative threshold you could hand to a stranger and have them execute correctly. The same goes for the actions — "manage the position" is not an action; "sell 50 of my 100 shares at the open" is. The test for a good reaction plan is whether someone else, handed your card, could trade your account through the event without calling you. If they'd have to ask "what do you mean by hawkish?" or "how much should I sell?", the plan isn't finished. This is also why the plan must be *written*, not held in your head: a plan in your head gets quietly renegotiated under stress ("well, it's only up 10 bps, that's basically in line, I'll hold"), whereas a plan on paper with a hard number resists that rationalization. The written card is a commitment device against your own future self, who will be a worse decision-maker than you are right now.

### Combining the tree with the consensus map

There is one more layer that turns the tree from a personal worksheet into an actual edge. The probabilities in column 3 are *yours*. But the market has its own probabilities for each branch, baked into the price — that is what [What's priced in: the question behind every trade](/blog/trading/analyst-edge/whats-priced-in-the-question-behind-every-trade) and [mapping the consensus](/blog/trading/analyst-edge/mapping-the-consensus-what-does-the-market-already-believe) are about. You can read the market's branch odds off fed funds futures, the option skew, and prediction markets, as detailed in [consensus expectations and priced-in](/blog/trading/event-trading/consensus-expectations-and-priced-in).

![Bar chart comparing market-priced branch odds with your own probabilities](/imgs/blogs/decision-trees-for-event-driven-views-6.png)

Lay your odds next to the market's and the trade jumps out. The lavender bars are the priced-in probabilities — the market's odds, read off futures and skew: 38% hawkish, 50% in line, 12% dovish. The blue bars are your probabilities: 30% / 50% / 20%. You are *underweight the hawkish branch* (you think it's less likely than priced) and *overweight the dovish branch* (you give it 20% versus the market's 12%). That eight-point dovish gap is your variant view — the place where the market and you genuinely disagree. **You do not trade the tree; you trade the gap between your tree and the priced tree.** If your odds matched the market's exactly, your tree's EV would be roughly zero by construction — there would be nothing to trade. The EV you computed earlier (+\$2) is small precisely because your edge over the consensus on this particular event is small. When the gap is wide — when you give a branch double the odds the market does — the tree's EV against the priced tree gets large, and that is when you size up.

There is a clean way to make this rigorous: compute the EV of your tree *twice*, once with your probabilities and once with the priced probabilities, and take the difference. The priced tree's EV is, by definition, roughly zero — the market sets the price so that holding the asset into the event is a fair bet given the market's own odds. Your tree's EV against the *same payoffs* but *your* odds is what you actually expect to earn. The gap between the two is your edge in dollars, and it is the only number that should determine your size. A trader who sizes off the absolute payoffs ("the dovish branch pays +\$360, that's exciting") is sizing off the market's information, not his own. A trader who sizes off the probability gap is sizing off the only thing he might know better than the crowd.

### Multi-stage trees: when one event triggers another

Real catalysts often cascade. The Fed decision at 2:00 is followed by the chair's press conference at 2:30, and the press conference can *reverse* the initial reaction — a hawkish statement softened by dovish Q&A is a regular occurrence. A single chance node cannot capture this; you need a **two-stage tree**, where each first-stage branch sprouts its own second-stage chance node.

The structure is: the statement prints (stage one: hawkish / in line / dovish), and *then*, conditional on the statement, the presser lands (stage two: the chair walks it back, confirms it, or doubles down). Each leaf is now a *path* through both stages, and its probability is the *product* of the two branch probabilities along the path. If the statement is hawkish (30%) and, given a hawkish statement, the chair walks it back 40% of the time, the path "hawkish statement, dovish presser" has probability 0.30 × 0.40 = 12%, and its payoff might be much smaller than the raw hawkish leaf because the walk-back rescues the position.

You roll a multi-stage tree back the same way, just in two passes: first collapse each second-stage chance node into the EV at the end of its first-stage branch, then collapse the first-stage node using those EVs as the leaf payoffs. The arithmetic is identical — probability times payoff, summed — applied recursively from the rightmost leaves inward. The payoff is that a two-stage tree captures the *real* structure of an event whose reaction isn't settled until the second shoe drops, and it stops you from cutting a position on the initial print that the presser was about to rescue. The pre-committed action for a two-stage event is correspondingly richer: "if hawkish statement, *wait for the presser* before acting; if the chair walks it back, hold; if the chair confirms, then cut." The decision node moves to *after* stage one, because the information you need arrives in stage two.

Do not over-build these. Two stages is usually the most a human can reason about cleanly under event pressure, and beyond that the probabilities get too noisy to be worth the extra branches. If you find yourself drawing a four-stage tree, you are modeling, not trading — collapse the later stages into a single "everything after the first hour" branch and move on.

## Common misconceptions

**"Decision trees are overkill for trading — they're for textbooks."** This is the most common dismissal and it is exactly backwards. The tree is not the slow part; *deciding under fire* is the slow, expensive part, and the tree is what removes it. The actual construction takes ten minutes the night before an event you were going to trade anyway. Ten minutes of cold reasoning to avoid a ten-second hot mistake that costs four or five figures is one of the best trades in the business. The traders who think trees are overkill are usually the ones freezing at 2:00:01.

**"Just react to the print — you'll have more information then."** You will have *one* more piece of information — the headline number — and you will have it at the moment you are least equipped to use it. The print tells you which branch fired; it does not tell you what to do, and "what to do" is a decision best made calmly with all branches in view. The trader who reacts live is not using more information; he is using the same information at a worse time, filtered through a stress response. The tree lets you incorporate the print instantly *because* you already mapped every possible print to an action.

**"The most likely branch is the plan."** No — *every* branch is the plan. This error kills accounts. You assign 50% to the in-line case, mentally plan for it, and treat the 30% hawkish tail as something that "probably won't happen." Then it happens — three times out of ten, by your own numbers — and you have no plan, so you panic. The whole point of the tree is that you trade the *entire* distribution. The 30% branch is not unlikely enough to ignore; it is likely enough that you must have a written, pre-committed action for it. Plan every leaf, especially the ones you hope won't happen.

**"I'll know what to do when it happens — I've done this before."** You won't, and experience makes this worse, not better, because it breeds false confidence. The version of you that built the tree is calm, rested, and reasoning clearly. The version of you at 2:00:01, staring at a position gapping against you, is a different person with a hijacked prefrontal cortex. Do not trust that person with a five-figure decision. The whole architecture of pre-commitment exists because the live-you is unreliable in exactly the moment that matters. Write the plan down so live-you only has to read, not think.

**"A tree needs precise probabilities, and I can't forecast the Fed to the percent."** You don't need precision; you need the *gap* to the market. Your "30% hawkish" is not a claim to three-decimal accuracy. It is a claim that the market's priced 38% is too high. Even rough odds — "the market is overpricing the hawkish tail, I'll mark it down ten points" — are enough to find the trade, because the trade is the *difference* between your odds and the priced odds, and differences survive a lot of imprecision in the levels.

**"Once I've built the tree, I should follow its EV mechanically every time."** The tree is a decision aid, not an autopilot, and there is one input it cannot check: whether your *probabilities* are any good. A tree fed garbage odds produces a confident, precise, garbage EV. The defense is to track your event calls over time — write down the branch odds before each event, the actual outcome after, and periodically check whether your "30% hawkish" branches really did hit about 30% of the time. This is calibration, and it is the only thing that tells you your trees are worth following. A trader whose branch probabilities are well-calibrated can lean on the EV; a trader who has never checked is just dressing up a hunch in arithmetic. The tree disciplines your *reaction*; calibration disciplines your *inputs*, and you need both.

## How it plays out in real markets

Let us put real dollars through three complete trees: a full FOMC tree, an earnings binary, and an expected-move straddle sized to a real account. Then we close with the example that matters most — a pre-committed reaction saving a position from a panic exit.

#### Worked example: the full FOMC tree on a \$20,000 note position

You are long **\$20,000** of the 2-year Treasury note into a 2:00 p.m. FOMC decision, because you believe the Fed is more likely to signal cuts than the market is pricing. The night before, you build the tree.

**Outcomes and your odds:** hawkish surprise 30%, in line 50%, dovish surprise 20%.

**Priced-in odds** (read off fed funds futures and the option skew): hawkish 38%, in line 50%, dovish 12%. Your variant view is the dovish branch — you give it 20% versus a priced 12%, an eight-point overweight — and a corresponding underweight of the hawkish tail.

**Conditional reactions** on the 2-year note: hawkish → −1.5%, in line → +0.2%, dovish → +1.8%.

**Pre-committed actions and payoffs** on the \$20,000 position:

- *Hawkish (30%):* cut on the stop at −1.5%. Payoff = −0.015 × \$20,000 = **−\$300**.
- *In line (50%):* hold the core, trim a quarter into the relief drift. Payoff ≈ +0.002 × \$20,000 = **+\$40**.
- *Dovish (20%):* add at the first pullback and ride the move. Payoff = +0.018 × \$20,000 = **+\$360**.

**Roll the EV back from the leaves:**

$$\text{EV} = 0.30(-\$300) + 0.50(+\$40) + 0.20(+\$360) = -\$90 + \$20 + \$72 = +\$2$$

The trade's expected value is about **+\$2** on \$20,000 of risk. The dovish edge you have over the consensus is real, but it is nearly cancelled by the fat hawkish tail you are still exposed to. Compared to the flat \$0 leaf, +\$2 of EV does not pay you for the variance. The tree's verdict: *your view is correct but too small to trade as a naked long.* The disciplined move is to stand aside on the outright — or, better, express the dovish variant view in a way that pays only if you're right, such as a position that profits from the dovish tail without the symmetric hawkish bleed. The lesson: the tree didn't tell you your view was wrong; it told you the *trade structure* didn't capture the view efficiently. *A correct view in an inefficient structure is a coin flip.*

#### Worked example: an earnings binary tree on a \$20,000 long

You hold **\$20,000** of a high-multiple software stock — 100 shares at \$200 — into earnings after the close. This is a binary event: the stock will gap one way or the other on whether it beats or misses the whisper number the price is discounting.

![A two-leaf earnings tree with beat and miss branches and pre-planned actions](/imgs/blogs/decision-trees-for-event-driven-views-3.png)

The tree has two branches off the event. **Beat the whisper: 55% (your odds).** On a beat, the stock gaps up about +12%, and your pre-committed action is to *sell half at the open* to lock the gain and let the rest run — locking roughly **+\$2,400** on the 100 shares (a 12% move on \$20,000 is +\$2,400, and selling half secures the bulk of it while leaving upside). **Miss the whisper: 45%.** On a miss, the stock gaps down about −15% and your stop is *jumped* — there is no orderly exit on a gap — so your pre-committed action is to *exit on the open and accept the loss*, no averaging down, no "waiting for the bounce." That's roughly **−\$3,000** (−15% on \$20,000).

Roll back the EV:

$$\text{EV} = 0.55(+\$2,400) + 0.45(-\$3,000) = +\$1,320 - \$1,350 = -\$30$$

The EV is about **−\$30** — slightly negative. Even though you think the beat is *more likely* than the miss (55% vs 45%), the miss *hurts more* than the beat helps (−\$3,000 vs +\$2,400), because the downside gap is bigger than the upside gap, which is typical for crowded high-multiple names. The most-likely branch is the beat, but the most-likely branch is not the plan — the plan is the whole tree, and the whole tree says holding into this print is a small negative-EV bet. *When the bad branch's payoff is larger than the good branch's, a coin-flip-favorable probability is not enough — you have to weight by dollars, not by odds alone.* The pre-committed actions here are doing critical work: the "exit on the open, no averaging" rule on the miss branch is what keeps the loss at −\$3,000 instead of the −\$6,000 it becomes when a panicked holder doubles down into a falling knife.

#### Worked example: the expected-move straddle decision sized to \$25,000

Now suppose you have *no directional view* — you genuinely don't know if the software stock beats or misses — but you suspect the move will be **bigger than the options market is pricing**. This is a volatility view, not a direction view, and the natural expression is to *buy* the straddle.

You have **\$25,000** to deploy. The stock is at \$200. The at-the-money straddle covering earnings — one call plus one put at the \$200 strike — costs **\$16** in total premium, so the market's priced one-sigma move is \$16, or **8%**. To break even, the stock must move *more than \$16 in either direction* by expiry; anything inside ±\$16 and you lose part or all of the premium.

Build the tree for *buying* the straddle. You size it at \$16,000 of premium (80 contracts' worth on \$25,000 of capital, keeping dry powder), and you assign your odds to three branches based on your view that the real move is larger than priced:

- *Big move, beyond ±12% (your odds 35%):* the straddle is deep in the money. A 12% move is \$24 against a \$16 cost, so you net roughly +\$8 per share — about **+\$8,000** on the position.
- *Move between ±8% and ±12% (40%):* you finish modestly in the money, netting roughly +\$2 per share — about **+\$2,000**.
- *Small move, inside ±8% (25%):* the straddle expires near worthless; you lose most of the \$16,000 premium — about **−\$13,000**.

Roll back:

$$\text{EV} = 0.35(+\$8,000) + 0.40(+\$2,000) + 0.25(-\$13,000)$$

$$\text{EV} = +\$2,800 + \$800 - \$3,250 = +\$350$$

The straddle's EV is about **+\$350** on \$16,000 of premium at risk — a small positive edge that exists *only because* you believe the realized move will beat the priced 8%. If your branch odds had matched the market's pricing, the EV would be near zero (that's what "priced in" means — the straddle is fairly priced to the market's distribution). The trade is alive solely on your variant volatility view. *Buying a straddle is a bet that the world moves more than the options market paid for — your tree has to assign enough probability to the big-move branches to overcome the brutal small-move leg, or the trade is just paying the market's vol premium.*

#### Worked example: a binary FDA catalyst sized to \$10,000

A small biotech reports its phase-3 trial result on a known date. This is the purest binary in markets: the drug works or it doesn't, the stock either roughly *doubles* or roughly *halves*, and the middle is empty. You have a view — you think the trial succeeds — and you want to put **\$10,000** behind it.

Build the two-branch tree. **Success (your odds 45%):** the stock gaps up about +100%, doubling your position to roughly +\$10,000 of profit. **Failure (55%):** the stock gaps down about −60% as the drug fails, a loss of about −\$6,000. Note the asymmetry of your odds — you think success is *less* likely than failure (45% vs 55%), because most phase-3 trials in this class fail, even ones you like.

Roll the EV back:

$$\text{EV} = 0.45(+\$10,000) + 0.55(-\$6,000) = +\$4,500 - \$3,300 = +\$1,200$$

The EV is about **+\$1,200** on \$10,000 — a strong positive edge, *despite* success being the minority outcome, because the success payoff (+100%) dwarfs the failure payoff (−60%). This is the mirror image of the earnings example: there, a favorable probability lost to an unfavorable payoff asymmetry; here, an *unfavorable* probability wins because the payoff asymmetry runs the other way. The tree forces you to weigh both, and the EV is the only honest summary. The critical pre-commitment for a binary like this is a *sizing* rule, not a stop — there is no stop on a 60% overnight gap, so the entire \$10,000 is genuinely at risk, and the only lever you control is the size. The discipline is: never put more behind a binary than you can lose entirely, because the failure branch will sometimes fire no matter how good your edge, and the only way to survive a string of them is to have sized each one to a survivable fraction. The math of why a sequence of large all-or-nothing bets ruins you even at positive EV is the Kelly story in [position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion). *On a binary, the tree tells you whether to play; the Kelly fraction tells you how much, and "the whole account" is never the answer no matter how good the EV looks.*

#### Worked example: a pre-committed reaction saving a \$20,000 position from a panic exit

This is the example that justifies the entire exercise. Return to the FOMC tree and the \$20,000 note position, but now compare two traders who hold the *identical* position into a hawkish print.

**Trader A has no tree.** The hawkish statement crosses at 2:00:00. The 2-year yield gaps +12 bps; his note position drops through −1.5% and keeps sliding toward −2.2% in the chaotic first two minutes as the move overshoots. He has no pre-decided action. He freezes, watches it tick to −2.2% (a **−\$440** mark), then panics and dumps the whole position at the very worst tick, locking **−\$440**. Worse, twenty minutes later the overshoot mean-reverts and the note settles back to −1.5% — exactly where his stop *would* have been — but he is already out at the bottom. His undisciplined reaction cost him an extra **\$140** beyond the planned loss, and the panic primes him to revenge-trade the rest of the afternoon.

**Trader B has the tree.** Her pre-committed action for the hawkish branch is a hard stop at −1.5% and no re-entry today. The same +12 bps prints; her stop fills at −1.5% in the first seconds, locking exactly **−\$300** — the loss her tree already accounted for as a 30%-probability cost. She feels nothing she didn't already feel last night when she wrote the number down. She is flat, calm, and done for the day.

The pre-committed reaction saved Trader B **\$140** on this single event versus Trader A's panic exit — and that understates it, because it ignores the revenge trades the panic spawns and the compounding cost of repeating this dozens of times a year. *The hawkish branch was always going to cost you something; the tree's job is to make sure it costs you the planned −\$300, not the panicked −\$440 plus whatever the tilt does to the rest of your day.* The math of why that small recurring leak is so destructive to compounding is in [risk management: survival as a compounding engine](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine).

## The playbook

Here is the repeatable process. Run it the night before any scheduled catalyst you are considering trading. It takes about ten minutes and it is the same six steps every time.

![Six steps drawing the tree pricing the move assigning odds committing actions rolling EV and deciding](/imgs/blogs/decision-trees-for-event-driven-views-7.png)

1. **Draw the tree.** Write the event and your current (or intended) position with its dollar size. Carve the outcome space into mutually exclusive, collectively exhaustive branches — two for a binary event, three (low / mid / high) for a continuous surprise. List every outcome the event can actually print.

2. **Price the move.** Read the expected move off the at-the-money straddle covering the event: total premium as a percent of the underlying is the priced one-sigma move. Use it to anchor the *size* of your branch reactions, so you are not inventing moves the options market would sell you cheaply.

3. **Assign your probabilities.** Put your own odds on each branch, summing to 100%. Then read the *priced* odds off futures, skew, or prediction markets, and lay them side by side. The gap between your odds and the priced odds is your variant view — and it is the only thing you are actually trading.

4. **Pre-commit one action per branch.** Turn the action column into a lookup table: a specific, observable trigger on the tape mapped to one action, written down. Include a stop level and a "no re-entry / no averaging" rule where it belongs. The flat branch — doing nothing — is a real branch with a \$0 payoff; write it in.

5. **Roll the EV back.** Compute the probability-weighted payoff across the leaves. At any decision node (hold vs flatten), take the better branch. The root carries the trade's EV — a single number.

6. **Decide and size.** Trade only if the tree's EV beats the flat \$0 leaf by enough to pay for the variance, *and* only because your odds differ from the priced odds. If the EV is a coin flip, stand aside or restructure the trade so it captures your variant view efficiently. When you do trade, size to the conviction the gap justifies — and then, when the number prints, execute the row. Do not think. Read the branch, find the action, do it.

That is the entire method. The arithmetic gives you an honest EV and stops you from trading coin flips you mistook for edges. The pre-commitment relocates your decision from the hot moment to the cold one, so the print becomes execution instead of analysis. Build the tree before the event, and you will be the bored trader at 1:58, not the panicked one at 2:00:01. The discipline of writing down what would make you act — including what would make you wrong — is the same discipline behind [defining your invalidation upfront](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront), and it is the throughline of forming any accountable view: decide cold, commit in writing, and let the market only choose the branch, never the plan.

## Further reading & cross-links

- [What's priced in: the question behind every trade](/blog/trading/analyst-edge/whats-priced-in-the-question-behind-every-trade) — the question your tree's probabilities must beat.
- [Mapping the consensus: what does the market already believe](/blog/trading/analyst-edge/mapping-the-consensus-what-does-the-market-already-believe) — how to read the priced odds you compare your tree against.
- [Expected value: the only math a view really needs](/blog/trading/analyst-edge/expected-value-the-only-math-a-view-really-needs) — the EV engine that rolls a tree back to one number.
- [Thinking in probabilities, not predictions](/blog/trading/analyst-edge/thinking-in-probabilities-not-predictions) — why you assign odds to every branch instead of betting the most likely one.
- [What would change my mind: defining invalidation upfront](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront) — the pre-committed stop, generalized to any view.
- [Consensus expectations and priced-in](/blog/trading/event-trading/consensus-expectations-and-priced-in) — where to read the market's branch probabilities.
- [The reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently) — how to estimate the conditional reactions in column 4.
- [How an options market maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) — who sells you the straddle and how they price the expected move.
- [The surprise, not the level: betas to data surprises](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) — the sensitivity behind continuous-event branch reactions.
- [Trading psychology and the execution gap](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap) — why live-you can't be trusted with the decision the tree pre-makes.
- [Risk management: survival as a compounding engine](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine) — why the small recurring panic-exit leak is so destructive.

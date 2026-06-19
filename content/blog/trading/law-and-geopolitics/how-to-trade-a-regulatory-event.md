---
title: "How to Trade a Regulatory Event, End to End"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The full, repeatable seven-step playbook for trading a legal or policy catalyst — spot it, measure what is priced in, form a differentiated view, choose the structure, size it for the asymmetry, set the invalidation, and manage the drift afterward."
tags: ["regulation", "event-trading", "priced-in", "implied-move", "options", "position-sizing", "kelly", "binary-event", "pdufa", "expected-value", "trading"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Trading a regulatory event is not a single bet on the outcome you expect; it is a seven-step process that turns a legal catalyst into a sized, hedged position with a written invalidation at every step.
>
> - The whole game is **surprise versus priced-in**. The market has already discounted the *expected* ruling; you only earn money on the gap between what happens and what was already in the price.
> - Measure the priced-in baseline three ways: the pre-event **run-up**, the option-implied move from the **straddle**, and the **consensus** probability. Your edge is your own probability *minus* the market's.
> - Express a binary catalyst with **options** so the loss is capped at the premium while the upside stays open — then size it with a **Kelly-lite** fraction, not by conviction.
> - The one number to remember: into the January 2024 US spot-bitcoin-ETF approval, bitcoin rose about **+72%**, then *fell* about **15%** in the nine days after the rule changed. If you "traded the outcome you expected," you lost on the best news in the asset's history.

On 11 January 2024, the United States Securities and Exchange Commission — the federal agency that polices stock and securities markets — let the first US *spot bitcoin exchange-traded funds* begin trading. An exchange-traded fund (ETF) is a fund whose shares trade on a stock exchange like an ordinary stock; a *spot* bitcoin ETF holds real bitcoin, so one share is a regulated, brokerage-account way to own the coin. For a decade the SEC had said no. By any plain reading of the headline this was the single most bullish regulatory event in bitcoin's history.

Bitcoin fell. Over the next nine trading days it dropped from about \$46,640 to roughly \$39,530 — down about 15% — on exactly the news it had wanted for ten years. To anyone watching the news ticker this looked insane. To anyone watching the *price*, it was the most ordinary thing in the world: bitcoin had already climbed about 72% from its mid-October low into the approval, as the rumor of approval hardened into near-certainty. The good news was already *in the price*. There was nothing left to buy, only profit to take.

That single episode contains the entire discipline this post teaches. A statute, a rule, a court ruling, a tariff, a sanction, or a drug approval does not move a price when it is *announced*; it moves the price continuously, from the first credible rumor, as the market revises the *probability* and the *magnitude* of the change. By the time the gavel falls, most of the move is usually behind you. The practitioner's job is not to predict the outcome — it is to measure how much is *already* priced, form a *differentiated* view of the part that isn't, express it in a structure with the right shape, size it for the asymmetry, and write down what would prove the whole thesis wrong. This post is the end-to-end playbook for doing exactly that, walked through one event from first rumor to final exit.

![Seven step pipeline for trading a regulatory event from identify to manage the drift](/imgs/blogs/how-to-trade-a-regulatory-event-1.png)

## Foundations: the event lifecycle, priced-in, and the asymmetric payoff

Before the seven steps, four building blocks, each defined from zero: how an event reaches the market (the lifecycle), what "priced-in" means and how you measure it, the difference between a direction trade and a volatility trade, and why a capped-loss structure changes everything about sizing.

### The event lifecycle: rumor, proposal, decision, enforcement

A legal or policy event is almost never a single instant. It is a *process* with four phases, and the price reacts in each one:

- **Rumor.** A leak, a lobbyist's read, a judge's pointed question at oral argument, a regulator's speech. Nothing is official, but the *probability* of the eventual outcome shifts — and the price shifts with it. This is where the largest, slowest part of the move usually happens.
- **Proposal.** The draft rule is published (an agency posts a *Notice of Proposed Rulemaking*, a bill is introduced, a target date like a drug-review deadline is set). The outcome is now *probable* and dated, so positioning accelerates into the date.
- **Decision.** The gavel falls: the rule is finalized, the court rules, the agency approves or rejects. The price gaps — but it gaps only on the *surprise*, the difference between the actual outcome and the baseline that was already priced.
- **Enforcement.** The rule starts to bite in cash flows: a tariff is collected, a fine is paid, a drug is sold, a bank's capital ratio tightens. Here a slow, fundamental *drift* can continue for months as the real effect shows up in earnings.

The single most important fact about this lifecycle is *where the money is*: the run-up from rumor to decision usually captures most of the move, and the decision itself prices only what was not already expected.

![Timeline of the event lifecycle from rumor to enforcement with the trade at each stage](/imgs/blogs/how-to-trade-a-regulatory-event-8.png)

### Binary versus continuous events

Events come in two shapes, and the shape dictates the structure you use.

A **binary event** has a small set of discrete outcomes on a known date: a drug is approved or it is not; a merger clears antitrust review or it is blocked; a court rules for the plaintiff or the defendant. The price often *jumps* — a large gap in one direction or the other — and the magnitude of the jump is roughly symmetric in probability but asymmetric in payoff. A pharmaceutical company awaiting a *PDUFA date* — the Prescription Drug User Fee Act deadline by which the US Food and Drug Administration must decide on a new drug — is the textbook binary: approve and the stock might jump 40%; reject and it might fall 60%.

A **continuous event** reprices gradually because the outcome is a matter of *degree*, not yes/no: a tariff that ratchets up over many rounds, a regulation whose stringency is set by a dial, an interest-rate path. The price drifts rather than gaps, and the trade is about the *path* and the *terminal level*, not a single date.

The classification matters because it tells you whether to buy a lottery-ticket structure (a binary) or to express a directional view on a level (a continuous trend). Get this wrong and you will buy expensive options for a slow grind, or sell cheap optionality right before a cliff.

### Priced-in, and the three ways to measure it

**Priced-in** is the part of the eventual move that the market has *already* discounted into today's price. Because markets price the *expected* outcome continuously, the price you see is a probability-weighted average over every outcome. The new information at the decision is only the *surprise* — actual minus expected. There are three independent ways to estimate the baseline, and you want all three to triangulate:

1. **The run-up.** Compare the price now to its level before the catalyst was on anyone's radar. A name that has already rallied 30% into a ruling has a lot of good news priced; a name that has not moved has little. The run-up is the crudest gauge but the most visible.
2. **The option-implied move.** The options market quotes, in dollars, how big a move it expects *over the event*. The price of a *straddle* — buying a call and a put at the same strike and expiry — is roughly what the market will pay for the event's uncertainty, and it backs out to an expected move size. This is the cleanest, most quantitative read on "how big a jump is priced."
3. **Consensus.** The explicit or implied *probability* the crowd assigns to each outcome — analyst approval odds, prediction-market prices, merger-arb spreads. This is the baseline your own view must beat.

The reason all three measures matter is that each one can be wrong in a different way, and you want them to *triangulate*. A run-up can be driven by a crowded momentum trade that has nothing to do with the event; an implied move can be distorted by a single large hedger buying protection; a consensus number can be stale or anchored to an old data point. When all three agree, you have a robust read on the baseline. When they diverge, the divergence itself is information: either you have found a genuine mispricing, or you are missing something the other participants can see. Either way, you do not trade until you can explain the gap.

The formal statement of "you trade the surprise, not the outcome" is simple algebra. Let the current price `P` be the probability-weighted average of the outcomes:

```
P = p_up * V_up + (1 - p_up) * V_down
```

where `p_up` is the priced probability of the bullish outcome, `V_up` the value if it happens, and `V_down` the value if it does not. When the event resolves, the price jumps to whichever `V` actually occurs. The size of that jump is not `V_up - P_before` in some absolute sense the headline suggests — it is the difference between the *realized* outcome and the *expected* value that was already in `P`. If the market had priced a 90% chance of the bullish outcome and it happens, the jump up is *small*, because almost all of `V_up` was already in `P`. If the market had priced 30% and it happens, the jump is *large*. The headline ("approved!") is identical in both cases; the trade is completely different. That is the entire reason a bullish outcome can produce a falling price: the run-up had pushed `p_up` so high that the realized `V_up` was *below* the price the optimism had already created.

![Price path split into a run up that is priced in and an announcement gap that is the surprise](/imgs/blogs/how-to-trade-a-regulatory-event-2.png)

### Direction versus volatility, and the asymmetric payoff

There are two fundamentally different trades you can put on around an event.

A **direction trade** bets on *which way* the surprise goes: you think the actual outcome will be more bullish (or more bearish) than the priced-in baseline, so you go long (or short), or buy calls (or puts).

A **volatility trade** bets on *how big* the move will be, regardless of direction: you think the realized jump will be larger (or smaller) than the option-implied move, so you buy (or sell) the straddle. You can have no view on direction at all and still have an edge if you think the market has *mispriced the size* of the move.

The reason options dominate event trading is the **asymmetric payoff**. A long call costs a fixed premium; below the strike you lose only that premium, but above the breakeven your gain grows without limit. That convex shape — capped downside, open upside — is exactly the shape you want over a binary catalyst whose downside outcome you cannot rule out. The asymmetry is also what makes naive sizing dangerous: a structure that can return five times its cost will tempt you to bet too much, and the one time the binary goes against you, an oversized position ends the account. That tension — convex payoff, capped per-trade loss, but ruinous if oversized — is what *position sizing* exists to manage.

There is one more piece of machinery to define, because it is where event traders most often get fleeced: **implied volatility** and the **event premium**. Implied volatility is the volatility number that, plugged into an option-pricing model, reproduces the option's market price; it is the market's forward-looking estimate of how much the underlying will move, expressed as an annualized percentage. Into a known event, implied volatility *rises*, because everyone knows a jump is coming and bids up the options. That elevated level is the **event premium** — the extra cost baked into the options purely because the date is approaching. The danger is the **volatility crush**: the instant the event resolves, the uncertainty is gone, implied volatility collapses, and every option loses the event premium *regardless of direction*. A trader who buys an at-the-money straddle the day before a decision can be *right about the direction* and still lose money, because the move was smaller than the inflated implied move and the crush ate the premium. This is why the implied move from Step 2 is the number that matters: you are not betting that the stock moves, you are betting that it moves *more (or less) than the premium you paid or collected*.

Why insist on a seven-step *process* at all, rather than a sharp read and a quick trade? Because regulatory events are precisely the place where a hunch goes wrong in ways that feel right. The headline is vivid and the outcome feels obvious, so the brain skips straight to "this is bullish, buy" — exactly the reflex that bought bitcoin at the top. A written process forces the three questions the reflex skips: *how much is already priced* (Step 2), *what is my edge over that baseline* (Step 3), and *what would prove me wrong* (the invalidation at every step). Each step also has a clean failure mode, so when a trade goes wrong you can locate *which* step broke — was the classification wrong, the priced-in estimate off, the edge illusory, the structure mispriced, the size too big, the exit undefined, or the drift misjudged? — and fix that step rather than abandoning the whole approach. A process is not bureaucracy; it is how you turn a one-off lucky call into a repeatable, auditable edge.

With those four blocks in hand, here is the process.

## Step 1 — Identify and classify the catalyst

The first job is to *find* the event early and *classify* it, because the classification picks your tools.

**Identify.** Build a catalyst calendar the way an equity trader builds an earnings calendar: agency rulemaking dockets and effective dates, court argument and decision dates, Supreme Court terms, central-bank meeting dates, drug-review (PDUFA) dates, merger review clocks, election dates, OPEC meetings, tariff-review deadlines. The earlier you spot a catalyst in its lifecycle, the more of the run-up you can capture and the cheaper the optionality.

**Classify** along three axes:

- **Binary or continuous?** A PDUFA date or a single court ruling is binary; a multi-round tariff escalation or a rate-hiking cycle is continuous.
- **What reprices?** Trace the transmission chain from the rule to the cash flow to the security. A drug approval reprices the issuer's future revenue; a tariff reprices the importer's margin and the exporter's volume; a bank-capital rule reprices the dealer's return on equity. Name the specific instrument that moves and *why*.
- **How fast and how cleanly?** A clean binary with a fixed date is tradeable with dated options; a messy, litigated, multi-year rule is a slow fundamental drift better expressed with the underlying.

For the in-depth chain from rule to cash flow to price, the mechanism is laid out in [how a rule becomes a price](/blog/trading/law-and-geopolitics/how-a-rule-becomes-a-price-expectations-drift-and-repricing); this post assumes that chain and focuses on the trade.

> **The invalidation at Step 1:** if you cannot name the specific cash flow that the rule changes and the specific instrument that prices it, you do not have a trade — you have a headline. Stop here.

**Our running example.** Take a mid-cap biotech, call it *NovaRx*, trading at \$40 per share with a market value of \$2 billion. It has one drug, *Catalyx*, awaiting an FDA decision on a fixed PDUFA date 60 days out. This is a clean **binary**: approve and the drug's projected sales make the company worth far more; the *Complete Response Letter* (a rejection) and the company is worth little more than its cash. What reprices: the entire enterprise value, gated on one yes/no decision on one date. This is the cleanest catalyst there is, which is why we will walk it the whole way.

## Step 2 — Estimate the priced-in baseline

You cannot know your edge until you know what the market has already discounted. Measure it three ways and reconcile.

#### Worked example: measuring priced-in from the run-up

NovaRx traded at \$28 four months ago, before its strong Phase III readout put approval on the radar. It now trades at \$40. The run-up is:

```
run-up = (40 - 28) / 28 = 0.4286 = +42.9%
```

The stock is already up about 43% on the *anticipation* of approval. That is a lot of priced-in optimism — the market is not pricing a coin flip; it is leaning toward approval. To sanity-check how much is priced, compare against the two terminal scenarios you will refine in Step 3: if approval is worth \$70 and rejection is worth \$15, then a price of \$40 sits:

```
implied weight on approval ~ (40 - 15) / (70 - 15) = 25 / 55 = 0.4545 = ~45%
```

The run-up alone implies the market is pricing roughly a **45% chance of approval** (ignoring discounting and risk premia). The takeaway: a 43% run-up is not "the good news is all in" — it is the market pricing a slightly-below-even bet, leaving real room for a surprise in either direction.

#### Worked example: the option-implied move from the straddle

Now read the options market. With NovaRx at \$40 and the PDUFA date inside the option's life, the at-the-money straddle expiring just after the date is quoted at \$13.00 (a \$40-strike call at \$6.60 plus a \$40-strike put at \$6.40). The straddle price is roughly what the market will pay for the event's uncertainty, so the **implied move** is approximately:

```
implied move ~ straddle price / spot = 13.00 / 40 = 0.325 = +/- 32.5%
```

The options market is pricing a move of about **plus or minus \$13**, i.e. a jump to roughly \$53 on approval or a drop to roughly \$27 on rejection. Compare that to your terminal scenarios (\$70 and \$15): the market's implied move is *smaller* than your scenarios imply, which is the first hint that the options may be *cheap* relative to your view — a clue we will use when we choose the structure. The core idea: the straddle is the market's own dollar estimate of the surprise, and your job is to decide whether that estimate is too big or too small.

![VIX at six stress events showing the volatility regime that sets event option prices](/imgs/blogs/how-to-trade-a-regulatory-event-7.png)

The level of broad volatility matters here too: when the VIX — the index that measures expected stock-market volatility — is elevated, every option is more expensive, so the same view costs more to express. The straddle you buy into a calm tape at a VIX of 14 is a very different price than the same straddle when the VIX is at 30. Read the regime before you price the trade; the mechanics of how events move volatility are covered in the [event-trading series](/blog/trading/event-trading).

**Consensus.** Finally, gather the crowd's explicit probability: sell-side analysts put approval odds at about 55%, and a prediction market on the approval trades at 0.52. So three readings — run-up (~45%), straddle (implies a move smaller than your scenarios), consensus (~52–55%) — cluster around a market that prices approval as a near-coin-flip, leaning slightly positive.

#### Worked example: backing a probability out of a spread

The cleanest way to read consensus is to invert a price into a probability, the same arithmetic merger-arbitrage desks use every day. In a takeover, a target trades *below* the agreed deal price because there is some chance the deal breaks (on antitrust, on financing, on a regulator's veto), and the discount measures that risk. Apply the identical logic to NovaRx. The stock at \$40 sits between the approval value (\$70) and the rejection value (\$15). Solving for the implied approval probability `p`:

```
40 = p * 70 + (1 - p) * 15
40 = 70p + 15 - 15p
25 = 55p
p  = 25 / 55 = 0.4545 = ~45%
```

The spread between today's price and the two terminal values implies the market is pricing approval at about **45%**. Notice this is the *same* number the run-up gave — which is the point: the run-up and the spread are two views of the same quantity. Now compare it to the prediction market at 52% and the analysts at 55%, and you see a small internal disagreement (the equity is pricing slightly *lower* odds than the survey-based measures). That gap is worth understanding before you trade: perhaps the equity holders fear the downside more (they price a fatter left tail), or perhaps the prediction market is thin and stale. The takeaway: any event with two values and a current price contains an *implied probability* you can extract with one line of algebra — always extract it, because it is the precise baseline your own view must beat.

> **The invalidation at Step 2:** if the three measures disagree wildly — say the run-up implies 45% but the prediction market is at 80% — do not trade until you understand why. A divergence is either your edge or your blind spot, and you must know which.

## Step 3 — Form a differentiated view

Here is where an *edge* lives. The market prices approval at roughly 52%. To have a trade, you need a *differentiated* view — your own probability that is meaningfully different from the market's, grounded in something the market is underweighting: the strength of the trial data, the FDA advisory committee's vote, the precedent of similar drugs, the regulatory track record of the agency for this class.

Your edge is not "I think it gets approved." The market thinks that too, at 52%. Your edge is the *gap* between your probability and the market's.

Where does a legitimate gap come from? In regulatory and legal events, differentiated views tend to come from three honest sources, and it is worth being precise about which one you are claiming, because each has a different shelf life and a different legal boundary.

- **The mosaic.** No single piece is material non-public information, but assembling many public pieces — the advisory-committee transcript, the agency's recent approval pattern for the drug class, the label language in a comparable approval, the company's manufacturing-inspection history — into a clearer read than the crowd has bothered to build. This is the *mosaic theory*, and it is the legal, repeatable edge: you out-work the consensus on public information. The boundary is bright — the moment a single piece is material and non-public, trading on it is insider trading, not research.
- **Base rates.** The crowd systematically over- and under-reacts to event categories. Most geopolitical shocks fade; most well-telegraphed approvals are priced; first-cycle FDA approval rates for a given division are knowable historical frequencies. Anchoring your probability to the *base rate* for the category, rather than to the vivid story in the headline, is an edge precisely because most participants anchor to the story.
- **Structure.** Sometimes the edge is not about the outcome at all but about *how the market is positioned*: a crowded long that must unwind on any disappointment, a forced hedger inflating implied volatility, an index rebalance that mechanically buys or sells regardless of fundamentals. A structural edge can be real even when you have no special view on the ruling itself.

For NovaRx, your edge is a mosaic edge: the 9-to-2 advisory-committee vote, the clean safety profile, and the agency's three-of-four approval rate for this class are all public, but few have assembled them into a single number. That is what lets you carry a 70% when the market carries 52%.

#### Worked example: your probability versus the market, and the expected value

Suppose your homework — the advisory committee voted 9-to-2 in favor, the safety profile is clean, the agency has approved three of the last four drugs in this class — leads you to assign a **70% probability of approval**, versus the market's **52%**. Your terminal scenarios are \$70 on approval and \$15 on rejection. The *fair value* under each set of probabilities:

```
market fair value  = 0.52 * 70 + 0.48 * 15 = 36.4 + 7.2  = $43.6
your fair value     = 0.70 * 70 + 0.30 * 15 = 49.0 + 4.5  = $53.5
```

The stock trades at \$40. By the market's own probabilities it is already a touch cheap (\$43.60 fair); by *your* probabilities it is worth \$53.50 — an edge of about **\$13.50 per share**, or roughly 34% upside to your fair value. The expected value of simply buying the stock at \$40, under your view:

```
EV(long stock) = 0.70 * (70 - 40) + 0.30 * (15 - 40)
               = 0.70 * 30 + 0.30 * (-25)
               = 21.0 - 7.5 = +$13.50 per share
```

A positive expected value of \$13.50 per share on a \$40 stock. The core idea: you are not paid for being right about approval; you are paid for the gap between your 70% and the market's 52%, applied to a payoff the market has only partly discounted.

> **The invalidation at Step 3:** write down, *before* you trade, what would move your probability. If the FDA issues a surprise safety signal, or a competing drug fails on the same mechanism, your 70% is wrong and the thesis is dead. A view you cannot invalidate is a hope, not a thesis.

## Step 4 — Choose the structure

You have a differentiated, positive-EV view. Now pick the *shape* of the position. The choice falls out of two questions: do you have a directional edge, and how binary is the event?

![Decision tree for choosing a direction trade or a volatility trade by edge and event type](/imgs/blogs/how-to-trade-a-regulatory-event-4.png)

- **Directional edge, binary event → long calls or a call spread.** You think approval is more likely than priced (70% vs 52%), and the downside outcome is real and large. Buying the stock exposes you to the full \$25 drop on rejection. A *long call* caps your loss at the premium while keeping the upside — exactly the convex shape a binary deserves. A *call spread* (buy a lower-strike call, sell a higher-strike call) cuts the cost further if you have a target ceiling.
- **Directional edge, continuous event → long or short the underlying, or a spread.** A slow tariff grind does not need expensive dated options; express it in the underlying or a longer-dated spread.
- **No directional edge but the implied move looks wrong → a volatility trade.** If you have no view on *which way* but you think the realized jump will be *bigger* than the \$13 the straddle prices, buy the straddle. If you think the move will be *smaller* than priced, sell it (defined-risk, via an iron condor) and collect the premium as the event passes.

For our NovaRx trade: directional edge (70% vs 52%), clean binary, real downside. The structure is a **long call**. Recall from Step 2 that your scenarios imply a bigger move than the options price — so the calls are relatively cheap for your view, which reinforces buying optionality rather than selling it.

It is worth weighing the realistic menu, because the same view can be expressed several ways and the choice is mostly about cost and the shape of the payoff:

- **Outright long call.** Maximum convexity and uncapped upside, but you pay the full event premium, and if the move is smaller than priced you can lose to the volatility crush even on a correct direction. Best when you think the move will be *larger* than the implied move and you want the full tail.
- **Call spread (buy a lower strike, sell a higher strike).** You finance part of the premium by selling away the upside above the higher strike. If your fair value is \$53.50, selling a \$60 call you do not expect to need cuts your cost and your breakeven, at the price of capping the win. On NovaRx, a \$45/\$60 call spread might cost \$2.50 instead of \$4.00 — a better risk-reward if you do not believe the \$70 ceiling. The spread also partly neutralizes the volatility crush, because the short leg loses event premium too.
- **Risk reversal (buy a call, sell a put).** Cheapest directional exposure — the put you sell funds the call you buy — but you re-introduce the uncapped *downside* you were trying to avoid, so it is only for a view where you would happily own the stock on a dip. On a true binary with a 60% rejection-drop, a risk reversal is dangerous.
- **Selling defined-risk volatility (iron condor / short strangle with wings).** The opposite trade: if you thought the implied move was *too big* — the \$13 straddle pricing more jump than you expect — you sell it inside defined-risk wings and harvest the event premium as it crushes. You make money if the stock stays *between* the wings, and your loss is capped by the long wings you buy.

The decision tree above is the quick filter; this menu is the fine adjustment once the branch is chosen.

#### Worked example: the asymmetric payoff of the chosen structure

You buy the \$105-equivalent structure — but let us use round per-share numbers that match the figure. Take a call struck at \$45 (just above spot) for a premium of \$4 per share. (We use the same shape as the generic payoff figure: strike \$105, premium \$4, in the figure's units; the logic is identical at any strike.) The payoff at expiry:

- **Rejection (stock to \$15):** the call expires worthless. Loss = the \$4 premium. That is the *entire* downside, versus a \$25 loss on the stock.
- **Approval (stock to \$70):** intrinsic value = \$70 − \$45 = \$25; minus the \$4 premium = **+\$21 per share**.

```
breakeven      = strike + premium = 45 + 4 = $49
max loss       = premium          = -$4 per share
upside at $70  = (70 - 45) - 4    = +$21 per share
payoff ratio   = 21 / 4           = 5.25 to 1
```

The structure risks \$4 to make \$21 — a 5.25-to-1 payoff. That convexity is why a binary catalyst is almost always expressed with options rather than the underlying: you keep the upside and you cap the loss at a number you choose in advance.

![Asymmetric payoff of a long call with capped loss at the premium and open upside](/imgs/blogs/how-to-trade-a-regulatory-event-5.png)

> **The invalidation at Step 4:** if the option you need is so expensive that your positive-EV view turns negative after paying the premium and the spread, the structure is wrong, not the view. Re-price with a spread, a different expiry, or pass.

## Step 5 — Size it for the asymmetric payoff and the tail

This is the step that separates survivors from blow-ups. The convex payoff that makes the trade attractive is exactly what makes oversizing lethal: a 5.25-to-1 winner feels like free money until the binary goes against you and a too-large position is gone. Size by formula, not by conviction.

The **Kelly criterion** gives the bet fraction that maximizes long-run growth. For a binary bet with win probability `p`, loss probability `q = 1 − p`, and payoff odds `b` (you win `b` dollars per \$1 risked), the optimal fraction of your bankroll is:

```
f* = (b * p - q) / b
```

Practitioners almost never bet *full* Kelly, because Kelly assumes you *know* `p` exactly. You do not — your 70% is an estimate. Overestimating the edge by a little makes full Kelly bet far too much, so the standard discipline is **fractional Kelly**: bet a quarter to a half of `f*`. Fractional Kelly sacrifices a little growth for a large reduction in the chance of a deep drawdown.

![Kelly sizing curve where the bet fraction rises with edge and is cut to a fraction of full Kelly](/imgs/blogs/how-to-trade-a-regulatory-event-6.png)

#### Worked example: Kelly-lite sizing on a real book

Your view is `p = 0.70`, so `q = 0.30`. The option's payoff odds are `b = 5.25` (win \$5.25 per \$1 of premium risked, from the 21-to-4 ratio). Full Kelly:

```
f* = (b * p - q) / b
   = (5.25 * 0.70 - 0.30) / 5.25
   = (3.675 - 0.30) / 5.25
   = 3.375 / 5.25
   = 0.643 = 64.3% of the book
```

Full Kelly says bet 64% of your account on one binary — which is insane for a single drug decision where your `p` is a guess. Cut to **half Kelly**:

```
half Kelly = 0.643 / 2 = 0.321 = ~32%
```

Still aggressive for one binary. A seasoned event trader caps any single binary at a hard ceiling — say **5% of the book at risk** — well *inside* even half Kelly, precisely because the inputs are uncertain and the loss is total if it misses. On a \$1,000,000 book:

```
premium at risk     = 5% * $1,000,000 = $50,000
contracts (100 sh)  = $50,000 / ($4 * 100) = 125 contracts
shares controlled   = 125 * 100 = 12,500 shares
```

You buy 125 calls, risking \$50,000 in premium. The two outcomes:

```
approval (stock to $70):  125 * 100 * $21 win = +$262,500   (+26.25% of book)
rejection (call to zero): premium lost        = -$50,000    (-5.00% of book)
```

If approval hits, the payoff is a 5.25x return on the \$50,000 risked, or about +26% on the whole book; if it misses, you lose the \$50,000, which is −5% of the book — survivable, repeatable. The core idea: Kelly tells you the *most* you could bet; your job is to bet a fraction of that so that the one binary that goes wrong is a bruise, not a funeral. The deeper treatment of sizing for fat tails is in [position sizing for tail and political risk](/blog/trading/law-and-geopolitics/position-sizing-for-tail-and-political-risk).

> **The invalidation at Step 5:** if the size that your edge justifies is larger than the size you can lose without changing your behavior, the size is wrong. Cap by the loss you can take calmly, not by the win you are hoping for.

## Step 6 — Set the exit and the invalidation

A trade without a written exit is a position you will manage with emotion. Decide all three exits *before* the event:

- **The target.** Where do you take profit? For a binary, the natural target is your approval fair value (\$53.50) or the option's intrinsic value at that price. Decide whether you sell into the gap (capture the jump, avoid the post-event fade) or hold for a fundamental drift.
- **The stop / invalidation.** For a long option the maximum loss is already the premium, so the "stop" is structural — you cannot lose more than \$50,000. But the *thesis* invalidation is separate: if before the date the FDA posts a negative briefing document, or an advisory committee votes against, your 70% is wrong and you should cut even though the option still has time value.
- **The time stop.** Options decay. If the date slips or the catalyst is delayed, theta bleeds your premium. Decide in advance how long you will hold a dead catalyst.

The crucial discipline here is that **the trade does not end at the announcement.** This is the most common error, addressed head-on below. The decision is the *middle* of the trade, not the end: what happens in the hours and days *after* the gavel — the drift or the fade — is often where the real money (or the real giveback) lives.

#### Worked example: sell into the gap or hold for the drift

The exit is itself an expected-value calculation. Suppose at the decision NovaRx gaps to \$56 and your calls are worth \$11. You face a choice for each contract: sell now and bank \$11, or hold for a possible drift to \$64 (calls worth \$19) against the risk of a fade back to \$50 (calls worth \$5). Assign the post-event probabilities your drift diagnostics support — say 60% drift, 40% fade:

```
EV(hold) = 0.60 * $19 + 0.40 * $5 = 11.4 + 2.0 = $13.40 per contract-share
EV(sell) = $11.00 per contract-share  (certain)
```

Holding has a higher expected value (\$13.40 vs \$11.00), but it is *uncertain* and you have already won. The professional answer is rarely all-or-nothing: you **sell enough to lock the certain win and de-risk the book, and hold the rest for the positive-EV drift.** Splitting the position in half:

```
sell half (certain):  0.5 * $11.00 = $5.50 per share banked
hold half (expected): 0.5 * $13.40 = $6.70 per share expected
blended exit value  = 5.50 + 6.70 = $12.20 per share
all-sell value      =               $11.00 per share
```

The blended exit (\$12.20) beats selling everything (\$11.00) while carrying far less variance than holding everything. The core idea: after a binary resolves in your favor, the question is no longer "am I right" but "how much certain profit do I trade for how much extra expected value" — and scaling out is how you answer it without betting the win you already have.

> **The invalidation at Step 6:** if you have not written down the price, the news, and the date at which you will exit, you have not finished building the trade. Do it before the event, when you are calm.

## Step 7 — Manage the drift and the fade after the decision

The decision lands. Now the price does one of two things, and knowing which is its own edge.

A **drift** is when the price keeps moving in the direction of the surprise for days or weeks after the event. This happens when the rule is *complex and slow* — the market under-reacts on the day because the full cash-flow impact takes time to model. A bank-capital rule, a complicated tax change, a multi-part antitrust remedy: these often drift, because the market needs time to digest second-order effects.

A **fade** is when the price reverses after the event because the move *over-reacted* or, more often, because the good news was *fully anticipated* and the run-up unwinds. This is "buy the rumor, sell the news" in its purest form — and it is exactly what bitcoin did after the ETF approval.

How do you tell, in advance, which one you are about to get? There is no certainty, but a few diagnostics tilt the odds:

- **Size of the run-up.** A large pre-event run-up (bitcoin's +72%) is fade fuel: the marginal buyer is already long, and the event removes the reason to hold. A small or absent run-up leaves room to drift.
- **Complexity of the rule.** Simple, fully-understood outcomes (a yes/no approval that everyone modeled) fade; complex, slow-to-model rules (bank capital, multi-part remedies, intricate tax changes) drift, because the market cannot price the full cash-flow impact on day one.
- **Positioning and flows.** Crowded, leveraged longs into the event predict a fade as they unwind; under-owned names with skeptical sell-side coverage predict a drift as the doubters capitulate over weeks.
- **The gap versus the implied move.** If the realized gap *exceeds* the implied move, the surprise was genuine and a drift is more likely; if the gap *undershoots* the implied move, the event was over-priced and a fade (the volatility crush plus profit-taking) dominates.

For NovaRx, the run-up was a moderate 43%, the catalyst is a *slow* one (drug sales ramp over quarters and analysts upgrade over weeks), and the realized gap will sit near the implied move — so the balance tilts toward a *drift*, which is exactly how we manage it in the P&L below.

![Bitcoin run-up into the 2024 ETF approval and the fade in the nine days after](/imgs/blogs/how-to-trade-a-regulatory-event-3.png)

The bitcoin chart is the cleanest illustration of the fade in the data. The good news was so well telegraphed — a multi-month run-up of about 72% — that the approval itself was a sell signal: there was no marginal buyer left, leveraged longs took profit, and the price fell about 15% before resuming its longer trend. If you had "traded the outcome you expected" by buying spot bitcoin *on* the approval, you bought the exact local top.

#### Worked example: the full end-to-end P&L across the decision and the drift

Bring NovaRx home. You bought 125 calls at \$4 (risking \$50,000) when the stock was \$40, with your 70%-vs-52% edge. The PDUFA decision is **approval**, and the stock gaps to \$56 on the day (a +40% jump — close to your scenario but not all the way to \$70, because the market re-rates toward your fair value rather than the rosy ceiling).

```
call value at $56  = (56 - 45) intrinsic = $11 per share
position value     = 125 * 100 * $11 = $137,500
premium paid       = $50,000
P&L on the gap     = 137,500 - 50,000 = +$87,500  (+8.75% of the book)
```

Now Step 7. Because a drug approval is a *slow* catalyst — sales ramp over quarters, analysts upgrade over weeks — this is a *drift* setup, not a fade. You sell half the position into the gap to lock in the certain part, and hold half for the drift:

```
sell 62 contracts at $11   = 62 * 100 * 11   = $68,200 booked
hold 63 contracts          = 63 * 100 * (cost basis $4) = $25,200 still at risk
```

Over the next three weeks the stock drifts to \$64 as the Street upgrades. You exit the remaining 63 calls at \$19 intrinsic (\$64 − \$45):

```
remaining 63 contracts at $19 = 63 * 100 * 19 = $119,700
```

Total proceeds and P&L on the whole trade:

```
proceeds  = $68,200 (sold into gap) + $119,700 (sold on drift) = $187,900
cost      = $50,000 premium
net P&L   = 187,900 - 50,000 = +$137,900
return    = 137,900 / 50,000 = +275.8% on capital at risk
book P&L  = +$137,900 / $1,000,000 = +13.79% on the whole book
```

A binary you sized at 5% of the book, expressed with a capped-loss structure, returned about 13.8% on the entire account — and the *worst case* the whole time was a 5% loss. The core idea of the whole playbook: you did not get paid for predicting approval; you got paid for the gap between your probability and the market's, expressed in a structure whose downside you fixed in advance and whose upside you managed across the decision *and* the drift.

### What can go wrong in execution

The clean arithmetic above hides a layer of friction that turns paper edges into real losses if you ignore it. Five execution risks deserve a place in your pre-trade checklist:

- **Liquidity and slippage.** Single-name event options are often wide and thin. A straddle quoted at \$13.00 mid might cost \$13.60 to buy and fetch \$12.40 to sell — a \$1.20 round-trip spread, almost 10% of the premium, before you are right about anything. Always price your edge *after* realistic transaction costs; a 34% edge that survives a 10% spread is a trade, one that does not is a mirage.
- **The volatility crush, again.** It bears repeating because it is the most common way a directionally-correct event trade loses money. If you buy the inflated event premium and the move undershoots the implied move, the crush eats you even on a correct call. Spreads (which sell some premium back) partly defend against it.
- **Gapping through your strike.** A binary can gap *past* the level where your structure pays best, or settle exactly at a strike (pin risk), leaving an assigned position you did not want. For multi-leg structures, know what you are left holding under each settlement.
- **Date risk.** Catalysts slip. A PDUFA date can be extended, a ruling delayed, a vote postponed. Every extra day of delay bleeds theta from a long option. Size and expiry should both carry a margin for a delay.
- **Borrow and assignment.** Short-leg structures (the put you sold in a risk reversal, the call you sold in a spread) carry assignment risk and, for shorts of hard-to-borrow names, a borrow cost that can swamp a thin edge. Defined-risk structures bound this; naked shorts do not.

None of these change the seven steps; they change the *numbers* inside them. The discipline is to run the whole playbook with realistic costs, not mid-market fantasies — an edge that only exists at the mid is not an edge.

## Common misconceptions

**"Trade the outcome you expect."** This is the deadliest mistake, and the bitcoin ETF is the proof. The outcome was approval — the single most bullish event possible — and the price *fell 15%* afterward, because the approval was already in the price after a 72% run-up. You are never paid for the outcome; you are paid for the *surprise versus what was priced*. A bullish outcome that is fully anticipated is a sell. Always trade your probability *minus* the market's, never the raw outcome.

**"Bigger conviction means bigger size."** Conviction is not a sizing input — edge and uncertainty are. In the NovaRx trade, full Kelly on a 70% view said bet 64% of the book; betting that on one drug decision, where the 70% is itself a guess, is how accounts die. The correct size was 5% — well inside *half* Kelly — precisely *because* the input was uncertain. The more binary and the more estimate-dependent the edge, the *smaller* the fraction, no matter how confident you feel. Size is set by the loss you can survive, not the win you are hoping for.

**"The trade ends at the announcement."** It does not. The decision is the middle of the trade. Roughly half the skill is in Step 7 — reading whether the post-event price will *drift* (slow, complex rules: under-reaction, hold or add) or *fade* (hyped, fully-priced events: over-reaction, take profit or reverse). In the NovaRx example, selling half into the gap and holding half for the drift turned a +8.75% day into a +13.8% trade. In the bitcoin example, *not* recognizing the fade would have turned a winning thesis into a 15% loss. The announcement is a checkpoint, not a finish line.

**"Options are always the right tool for an event."** Options are right for a *binary* with real downside, where you want capped loss and open upside. For a slow *continuous* event — a multi-year tariff grind — dated options bleed theta while you wait, and the underlying or a long-dated spread is cheaper. Match the structure to the event's shape, not to a habit.

## How it shows up in real markets

The NovaRx walk-through is a composite, but every step maps to real, dated episodes you can study.

**The fade — bitcoin and the 2024 spot-ETF approval.** Bitcoin ran from about \$27,150 in mid-October 2023 to \$46,640 by the approval on 10 January 2024 — about +72% — then fell to roughly \$39,530 over the next nine trading days, about −15%. The most bullish regulatory headline in the asset's history was a local top, because the run-up had already priced it. This is Step 7's fade in its purest form, and the cleanest single illustration that you trade the surprise, not the outcome.

**The drift — slow, complex rules.** When a rule is intricate enough that the market cannot fully model it on the day, the price keeps moving as the cash-flow impact is digested. Post-2008 bank regulation reshaped dealer balance-sheet economics over *years*, not on any single headline; the repricing of bank return-on-equity was a multi-quarter drift, not a one-day gap. The lesson: complex rules under-react on day one and reward patience.

**The volatility regime sets the price of protection.** Whether you can express an event cheaply depends on the broad volatility level when you put the trade on. The VIX sat near 13 for much of calm 2024 but spiked to 38.6 on 5 August 2024 in the yen-carry unwind and reached 82.7 in the March 2020 COVID crash. The identical event structure costs multiples more to buy when the VIX is elevated — so the same view is a different trade in a different regime. Read the regime first; the [event-trading series](/blog/trading/event-trading) covers how specific catalysts move volatility.

**Binary events with fixed dates.** PDUFA drug-decision dates, merger antitrust-review deadlines, and major court ruling dates are the cleanest binaries on the calendar — known dates, discrete outcomes, large gaps. They are where the options-structured, Kelly-sized version of this playbook applies most directly, and where the discipline of trading the *surprise* (the advisory-committee vote, the merger-arb spread's implied probability) versus the *priced-in baseline* pays off most clearly.

**The continuous case — the 2018-19 US-China trade war.** Not every catalyst is a one-date binary, and the playbook bends to fit. The trade war was a *continuous* event: the US average tariff on Chinese imports climbed in steps from about 3.1% in early 2018 to roughly 21% by late 2019 before settling near 19.3%, across many separate announcements and threats. There was no single date and no single gap; instead, each escalation round repriced the affected sectors — soybeans, semiconductors, retailers with China-heavy supply chains — a little more. The playbook still applies, but the structure changes: rather than buying dated options for one cliff, you express a *directional view on the level* in the underlying or a long-dated spread, you re-estimate priced-in after each round (because the baseline keeps moving), and your invalidation is a de-escalation headline rather than a single ruling. The continuous case is a reminder that Step 1's classification — binary versus continuous — is the fork that sets the whole rest of the process.

## How to trade it: the playbook

The whole process, as a checklist with the invalidation at each step. This is the deliverable.

1. **Identify and classify.** Build a catalyst calendar; for each event, decide binary vs continuous and name the exact cash flow and instrument that reprice. *Invalidation:* if you cannot name the cash flow, it is a headline, not a trade.
2. **Estimate priced-in.** Triangulate the baseline three ways — the run-up, the option-implied move from the straddle, and consensus probability. *Invalidation:* if the three disagree wildly, find out why before you trade.
3. **Form a differentiated view.** Compute your probability and your fair value; your edge is your probability *minus* the market's, and the trade exists only if the expected value is positive. *Invalidation:* write down, in advance, the news that would move your probability.
4. **Choose the structure.** Directional edge plus a binary with real downside → long calls or a call spread; no directional edge but a mispriced move → buy or sell the straddle; a slow continuous event → the underlying or a long-dated spread. *Invalidation:* if the premium turns your positive EV negative, re-price or pass.
5. **Size it.** Compute full Kelly, then bet a quarter to a half of it, capped by a hard per-event ceiling (e.g. 5% of the book) because the inputs are uncertain and the binary loss is total. *Invalidation:* if the justified size is larger than the loss you can take calmly, the size is wrong.
6. **Set the exit and invalidation.** Write the target, the thesis kill-switch, and the time stop *before* the event. *Invalidation:* an unwritten exit means the trade is not finished.
7. **Manage the drift and fade.** After the decision, decide whether the setup drifts (complex, slow rules — hold or add) or fades (hyped, fully-priced events — take profit). *Invalidation:* treating the announcement as the finish line is itself the error.

The one rule that ties the seven together: you are paid for the *gap between your view and the priced-in baseline*, expressed in a structure whose downside you fix in advance and whose upside you manage across the decision and the days after. Predicting the outcome is not the job; pricing the surprise is.

Run the loop enough times and the seven steps become second nature, but the discipline never gets easier in the moment — the headline always *feels* like a reason to act, the convex payoff always *feels* like free money, and the win you have already booked always *feels* like a reason to hold for more. The process exists precisely to override those feelings with arithmetic: an implied probability you extract instead of a story you tell, a Kelly fraction you compute instead of a conviction you feel, an invalidation you wrote down when you were calm instead of one you improvise when you are not. The traders who survive a long run of regulatory events are not the ones who call the most outcomes correctly; they are the ones who, win or lose, can point to exactly which step did its job and which one needs work. That auditability — not any single prediction — is the durable edge in trading the rules.

## Further reading & cross-links

- [How a rule becomes a price: expectations, the drift, and the repricing](/blog/trading/law-and-geopolitics/how-a-rule-becomes-a-price-expectations-drift-and-repricing) — the event-study toolkit and the run-up / gap / drift mechanics this playbook operationalizes.
- [Building a legal and geopolitical risk dashboard](/blog/trading/law-and-geopolitics/building-a-legal-and-geopolitical-risk-dashboard) — the monitoring stack that feeds Step 1: dockets, calendars, the volatility and risk indices.
- [Position sizing for tail and political risk](/blog/trading/law-and-geopolitics/position-sizing-for-tail-and-political-risk) — the deeper treatment of fractional Kelly, hedging, and convexity behind Step 5.
- [The law, policy, and geopolitics playbook](/blog/trading/law-and-geopolitics/the-law-policy-and-geopolitics-playbook) — the capstone that pulls the whole transmission spine together.
- [The event-trading series](/blog/trading/event-trading) — how specific catalysts (CPI, FOMC, earnings) move price and volatility across assets.
- [The quantitative-finance series](/blog/trading/quantitative-finance) — options pricing, the straddle, implied volatility, and the hedging mechanics behind every structure here.

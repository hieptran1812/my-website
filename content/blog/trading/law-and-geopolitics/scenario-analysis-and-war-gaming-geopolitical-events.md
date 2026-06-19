---
title: "Scenario analysis and war-gaming geopolitical events"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "You cannot predict geopolitics, but you can war-game it: build a probability tree, anchor the odds in base rates, compute the expected value across branches, map the second-order effects, and pre-plan a trade for each outcome before the event happens."
tags: ["geopolitics", "scenario-analysis", "war-gaming", "probability-tree", "base-rates", "expected-value", "second-order-effects", "tail-risk", "decision-making", "risk-management"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — You cannot forecast geopolitics, but you can war-game it: lay out the branches, attach base-rate odds, price the asset payoff in each branch, compute the expected value across the whole tree, and pre-plan a trade for every outcome before the event fires. That turns paralysis into a position.
>
> - The edge is not prediction; it is **structure**. The person who has written down "if A, I do X; if B, I do Y; if C, I do Z" trades the news in seconds while everyone else is reading the headline and panicking.
> - **Anchor the odds in base rates, not the vivid story.** A Strait-of-Hormuz "closure" headline feels like a 40 percent event; the historical record says it is closer to a single-digit-percent event. Same news, opposite trade.
> - **The first effect is rarely the one that matters most.** An oil shock is not really an oil trade — it is an inflation-then-rates-then-equities trade. Map the chain three links deep before you size anything.
> - The one number to remember: a book sized so it is **positive expected value in every branch** of the tree cannot be wrong-footed by the outcome — only by mis-estimating the odds, which is a far smaller, far more controllable error.

In the third week of a Gulf crisis, the wire flashes a single line: *military forces have closed the Strait of Hormuz to commercial shipping.* Roughly a fifth of the world's seaborne oil moves through that twenty-one-mile chokepoint. Brent futures gap higher in the overnight session. Equity index futures crater. Gold jumps. The phone lights up. You have about ninety seconds before the open to decide what to do with real money.

There are two kinds of investors in that ninety seconds. The first is reading the headline for the first time, heart rate climbing, trying to reason from a blank page about a situation that is genuinely terrifying and genuinely uncertain — and, almost always, freezing or panic-selling at the worst possible tick. The second pulls up a document they wrote three weeks ago, when the crisis was just a buildup and their pulse was normal. The document has three branches, a probability on each, an asset payoff for each, and a single line under each branch that says exactly what to do. They are not predicting anything. They are executing a plan. The difference between those two investors is not intelligence or information. It is whether they war-gamed the event before it happened.

This post is about that document and how to build it. It is deliberately analytical and politically neutral — we are studying how to reason and position under geopolitical uncertainty, not advocating any policy, party, or outcome. The thesis is simple and, once you internalize it, freeing: **you cannot predict geopolitics, but you can war-game it.** You can enumerate the branches, anchor their probabilities in the historical base rate rather than the day's narrative, compute what each branch does to your assets, calculate the expected value across the whole tree, and pre-plan a trade for each branch so that whatever happens, you already know your move. The figure below is the entire process on one page; the rest of the post fills in each box.

![Diagram of the war-gaming loop from mapping branches to base-rate odds to payoffs to expected value to pre-planned trades to updating](/imgs/blogs/scenario-analysis-and-war-gaming-geopolitical-events-1.png)

## Foundations: scenario analysis and the tools to do it from zero

Start from the ground. **Scenario analysis** is the discipline of laying out the distinct ways the future could unfold, attaching a probability to each, and working out what each one does to the thing you care about — here, your portfolio. It is the opposite of **forecasting**, which tries to name *the* single thing that will happen. Forecasting says "the strait will not close." Scenario analysis says "there is roughly a 7 percent chance it closes, a 28 percent chance of a brief disruption, and a 65 percent chance of de-escalation, and here is what my book does in each." The forecaster is right or wrong. The scenario analyst is never exactly right, because they never committed to a single outcome — and that is the point, because in geopolitics nobody is reliably right, and pretending otherwise is how money gets lost.

Why does forecasting fail so badly here? Because geopolitical outcomes are driven by the decisions of a handful of people under conditions of secrecy, bluff, and accident. There is no equation for whether a leader escalates or backs down. The political scientist Philip Tetlock spent two decades scoring expert geopolitical predictions and found that the average expert performed barely better than chance, and worse than simple statistical baselines. The honest response to that finding is not to find a better forecaster. It is to stop forecasting and start war-gaming.

The core tool of war-gaming is the **probability tree** (also called a decision tree). It is a diagram that starts from an event and branches into its possible outcomes, with a probability on each branch and a consequence at the end of each branch. The probabilities on the branches leaving any single node must sum to 100 percent, because the branches are meant to be **mutually exclusive** (only one can happen) and **collectively exhaustive** (together they cover every possibility). That "sum to 100 percent" discipline is quietly powerful: it forces you to account for the boring outcome — de-escalation, nothing-happens — which the panicking investor forgets exists.

Each branch needs a probability, and this is where most people go wrong, so it gets its own foundation. A **base rate** is the historical frequency of an outcome across many similar past situations — the "outside view," in Daniel Kahneman's phrase. The **inside view**, by contrast, builds a probability from the specific, vivid details of *this* case: this leader, this fleet, this threat. The inside view feels more informed and is almost always more wrong, because the vivid details hijack your judgment and crowd out the boring statistical truth that most scary buildups fizzle. The single most important habit in war-gaming is to start every probability from the base rate and adjust *cautiously* from there, rather than starting from the narrative.

Once each branch has a probability and a payoff, you compute the **expected value (EV)**: the probability-weighted average of the payoffs, `EV = Σ (probability × payoff)` summed across every branch. Expected value is the number that tells you whether a position is worth taking *on average*, across all the ways the future could go. A trade can lose in the most likely branch and still be a great trade if its payoff in the unlikely branches is large enough — that is the whole logic of a hedge, and EV is how you see it.

It is worth being precise about what EV does and does not tell you. EV is a *long-run average* — it is the number you would converge to if you could replay the same event thousands of times. You cannot replay a geopolitical event thousands of times; it happens once. So EV is not a promise about this particular outcome; it is a measure of whether the *structure* of the position is sound. A positive-EV position can still lose money on the single realization you actually live through. That is why EV is necessary but not sufficient: you also need to know that the worst branch will not bankrupt you before the long-run average has a chance to assert itself. War-gaming therefore tracks two numbers per strategy — the EV (is it positive on average?) and the worst-branch loss (can I survive the bad draw?) — and a position is only worth holding when the first is positive *and* the second is survivable. A strategy with a glorious EV that wipes you out 5 percent of the time is not a strategy; it is a slow-motion accident.

Two more tools complete the kit. **Conditional probability** lets the tree grow past one level: the probability of B *given that* A has happened, written `P(B | A)`. A closure raises the conditional probability of a recession, which raises the conditional probability of rate cuts — the tree branches again at each node. The arithmetic of a multi-level tree is just multiplication along each path: the probability of reaching a particular leaf is the product of the conditional probabilities of every branch you took to get there. If a closure has a 7 percent probability, and *given* a closure there is a 50 percent chance of a recession, then the joint probability of "closure and recession" is `0.07 × 0.50 = 0.035`, or 3.5 percent. This is how a two-level tree stays internally consistent — the joint probabilities of all the leaves still sum to 100 percent, because at every node the conditional branches sum to 100 percent. The reason most war-games stop at one level is not that deeper trees are wrong but that each extra level multiplies in another probability you cannot estimate honestly, and false precision compounds faster than insight.

And the **pre-mortem**, a technique from psychologist Gary Klein: before you put the trade on, assume it has already lost money, then write down every reason why. The pre-mortem inverts the usual planning question — instead of "what could go right," it forces "it went wrong; explain how" — and that inversion is what surfaces the branch you forgot and the assumption you smuggled in. It works because people are far better at generating reasons for an outcome that is stated as fact than at brainstorming risks in the abstract. We will build each of these tools concretely below; the loop figure above shows how they chain together.

One last distinction that the whole post turns on: the difference between a **hedge** and a **bet**. A bet is a position you put on because you expect it to make money — you think the odds are in your favor. A hedge is a position you put on because you expect it to *lose* money in the most likely branch, but to pay off enormously in the branch that would otherwise wreck the rest of your portfolio. A hedge is insurance; you are happy when it "loses," the way you are happy when your house does not burn down despite paying the premium. Confusing the two — treating a hedge like a bet and cutting it because it is bleeding, or treating a bet like a hedge and refusing to size it — is one of the most common and expensive errors in event trading. War-gaming makes the distinction explicit, because the tree shows you exactly which branch each leg is there to cover.

## Building a probability tree for a geopolitical event

Take the Strait of Hormuz scare as the running example, because it is concrete, it has a clear transmission to assets, and the curated oil data lets us ground the payoffs in real numbers. The event is a military crisis in the Gulf that *might* interrupt oil shipping. Step one is not to guess what happens — it is to enumerate the mutually exclusive, collectively exhaustive branches.

A clean three-branch tree for this event looks like this. **Branch one: de-escalation.** The crisis cools, shipping continues, oil gives back its fear premium. **Branch two: brief disruption.** Some tankers are delayed or a limited skirmish bumps insurance and freight rates, oil spikes for weeks then normalizes, but the strait stays effectively open. **Branch three: real closure.** The strait is genuinely shut to commercial traffic for a sustained period, choking roughly a fifth of seaborne oil. These three are exclusive (only one obtains) and exhaustive (every realistic path falls into one of them). You can split them finer, but three is usually the right resolution — enough to capture the structure, few enough that you can actually assign honest probabilities.

![Diagram of a three-branch probability tree for a Hormuz scare with de-escalation brief disruption and closure branches and oil and equity payoffs](/imgs/blogs/scenario-analysis-and-war-gaming-geopolitical-events-2.png)

Now attach a payoff to each branch — what the relevant assets do *if that branch happens*. This is where you need to think like the market, not like a news reader. In the de-escalation branch, the fear premium that built up during the crisis unwinds: oil falls (say −8 percent from the elevated crisis level), and equities tick up (+1 percent) as the cloud lifts. In the brief-disruption branch, oil spikes hard but temporarily (+20 percent), and equities wobble (−3 percent) on the inflation scare. In the real-closure branch, oil rockets (+60 percent or more — a genuine supply shock to a fifth of the world's oil is not a small number), and equities fall sharply (−12 percent) as the market prices the inflation-and-recession risk. Those payoff figures are the second number on each leaf of the tree, and they are what make the scenario *tradable* rather than merely interesting.

Notice what the tree has already done for you. It has separated the *probability* question (how likely is each branch?) from the *payoff* question (what happens to my assets in each?). These are completely different skills, and conflating them is a classic error: people who are sure a closure is unlikely also tend to assume the payoff if it *did* happen would be modest, because the whole scenario feels remote. The tree forces you to price the closure payoff at its true, terrifying size *regardless* of how unlikely you think it is — which is exactly the information you need to decide whether to carry a cheap hedge against it.

#### Worked example: pricing a probability tree into an expected dollar value

Take a \$1,000,000 equity portfolio and ask what the Hormuz tree implies for it over the next month, using the branch payoffs above applied to the equity leg only. The branches and their base-rate probabilities are: de-escalation 65 percent (equities +1 percent), brief disruption 28 percent (equities −3 percent), real closure 7 percent (equities −12 percent).

Compute the expected equity move as the probability-weighted average:

```
de-escalation : 0.65 × (+1%)  = +0.65%
brief disrupt : 0.28 × (-3%)  = -0.84%
real closure  : 0.07 × (-12%) = -0.84%
expected move = 0.65 - 0.84 - 0.84 = -1.03%
```

So the tree says the *expected* move in your equity over the month is about −1.03 percent, or **−\$10,300** on the \$1,000,000 book. That is a small number — and that smallness is the first real insight. The market, on average, barely moves on a Gulf scare, because the most likely branch (de-escalation) is benign and roughly offsets the unlikely-but-severe branches. The investor who dumps the whole portfolio on the headline is trading as if the −12 percent closure branch were certain; the tree says the *expected* damage is a rounding error. **The expected loss from a geopolitical scare is usually tiny; the panic is what is expensive.**

## Anchoring probabilities in base rates, not the vivid narrative

The whole tree rests on those branch probabilities, and the branch probabilities are where the discipline lives. Here is the trap: the closure branch comes with a vivid, cinematic, terrifying story — warships, fire, a fifth of the world's oil cut off — and vividness masquerades as probability in the human mind. Kahneman called this the **availability heuristic**: the easier an outcome is to call to mind, the more likely it feels, regardless of its actual frequency. A headline-driven investor, reasoning from the vivid inside view, will set the closure probability at something like 40 percent, because the story is so easy to recall and so emotionally loud.

![Diagram comparing headline-driven closure odds at 40 percent against base-rate-anchored closure odds near 7 percent and the opposite trades they imply](/imgs/blogs/scenario-analysis-and-war-gaming-geopolitical-events-4.png)

The base rate tells a different story. The Strait of Hormuz has been the object of crises, threats, and tanker incidents repeatedly across decades — the Tanker War of the 1980s, periodic seizures, recurring threats during sanctions standoffs — and across all of those buildups, a sustained, genuine closure to commercial shipping has essentially never materialized for any meaningful duration. The outside view, counting how often a closure scare actually became a closure, lands the probability in the low single digits, not at 40 percent. We use 7 percent in the tree as a deliberately *generous* base-rate estimate — generous because the cost of underweighting a true tail is high, so we round the historical near-zero up to a respectful single digit rather than dismissing it.

To see how much this matters, watch what the Geopolitical Risk index does around real crises. The index, built by economists Dario Caldara and Matteo Iacoviello, counts how often major newspapers mention war and military tension, normalized so 100 is the long-run average. It is a measure of *attention*, not of outcomes — and attention is exactly what the vivid narrative inflates.

![Line chart of the Geopolitical Risk index spiking at 9 11 Iraq Crimea Ukraine and Gaza then reverting toward the long run mean of 100](/imgs/blogs/scenario-analysis-and-war-gaming-geopolitical-events-5.png)

Two features of that chart are the base-rate lesson in one view. First, every spike is real and large — 9/11 pushed the index above 500, five times its mean. Second, **every spike fades.** The index reverts toward 100 over months. Attention is acute and temporary. If the closure probability tracked the attention spike, it would be a 40-percent event at the peak; but the attention always recedes precisely because the feared outcome almost never arrives. The base rate is the calm fact underneath the loud headline, and anchoring to it is the single highest-value habit in the whole process.

#### Worked example: how the base rate flips the trade

Two investors war-game the same Hormuz scare on the same \$1,000,000 equity book. Investor H uses the headline-driven probabilities; investor B uses base-rate-anchored ones. The payoffs are identical (de-escalation +1 percent, brief disruption −3 percent, real closure −12 percent); only the probabilities differ.

Investor H, anchored to the vivid story: closure 40 percent, brief disruption 35 percent, de-escalation 25 percent.

```
H expected move = 0.25(+1%) + 0.35(-3%) + 0.40(-12%)
                = +0.25% - 1.05% - 4.80% = -5.60%  -> -$56,000
```

Investor B, anchored to the base rate: closure 7 percent, brief disruption 28 percent, de-escalation 65 percent (from the worked example above).

```
B expected move = 0.65(+1%) + 0.28(-3%) + 0.07(-12%)
                = +0.65% - 0.84% - 0.84% = -1.03%  -> -$10,300
```

Same event, same payoffs, but investor H computes an expected loss of −\$56,000 and slashes the portfolio; investor B computes −\$10,300, holds the core, and buys a cheap closure hedge with the saved capital. When the scare de-escalates — the base-rate-likely outcome — H has locked in real losses selling the bottom while B is roughly flat and still invested. **The probabilities, not the payoffs, are where the money is made or lost, and the probabilities come from the base rate.**

## The expected-value calculation across branches

So far the EV has covered a single asset (equities). The real power of war-gaming shows up when you compute expected value across *multiple* assets and a *whole position*, because that is what tells you whether your planned trades are collectively worth doing. The mechanics are the same — probability times payoff, summed — but now the payoff in each branch is the dollar result of your *entire book*, not one leg.

Lay it out as a matrix: branches down the rows, your positions across the columns, and the dollar payoff of each position in each branch in the cells. The bottom-right of the matrix — each branch's row summed across all your positions, then weighted by the branch probability — is the EV of the whole strategy. This matrix is the single most useful artifact in war-gaming, because it lets you *engineer* the payoffs: you adjust position sizes until the book has the EV profile you want, branch by branch.

![Matrix of expected value across three branches showing long oil long gold and equity put spread payoffs per branch and the net book result](/imgs/blogs/scenario-analysis-and-war-gaming-geopolitical-events-7.png)

Read the matrix the way you would read a stress test. Each row is a possible world; each cell is what one of your positions does in that world; the rightmost column is your total profit and loss in that world. The de-escalation row is negative — your hedges bleed when nothing happens, which is exactly what hedges do. The disruption and closure rows are strongly positive, because the hedges pay off when they are needed. The question war-gaming answers is not "which row will happen" — you do not know — but "is the probability-weighted sum of these rows positive, and can I survive the worst row?" If yes to both, you have a position worth holding regardless of what the world does next.

The deep point here is that **a position can have positive expected value even though it loses money in the most likely branch.** That is not a paradox; it is the definition of insurance and the reason hedges exist. The de-escalation branch is the most likely (65 percent), and the book loses money there — but it loses a small, known amount, while it gains a large amount in the unlikely-but-severe branches, and the probability-weighted sum comes out positive. An investor who cannot tolerate "losing in the likely case" will never carry a good hedge, and will get destroyed the one time the tail arrives.

The matrix also makes a second, subtler property visible: the *shape* of the payoffs across branches, not just their probability-weighted sum. Two books can have identical EV but completely different shapes. One book might earn a steady small profit in every branch (a flat payoff); another might lose a little in the common branch and earn a fortune in the rare one (a convex payoff). For a single, once-in-a-decade geopolitical event, the convex shape is usually the right one, because the convex book is the one that *protects the rest of your portfolio* in the branch that would otherwise hurt the most. The flat book makes you feel good more often but leaves you naked to the tail. Reading the matrix column by column — not just the bottom-line EV — is how you choose the shape that fits the rest of your holdings. If your core portfolio is long equities, you specifically want a war-game book that is convex *in the same branches where your core gets hurt*, so the two offset.

A practical note on building the matrix: keep the positions to a handful of clean, liquid legs whose payoffs you can actually estimate. A matrix with twelve positions and forty cells looks sophisticated and is mostly invented numbers — you cannot honestly price a dozen instruments across three branches. Three to four legs, each with a clear mechanism (oil for the supply shock, gold for the haven bid, a put spread for the equity drawdown, optionally a rates payer for the second-order move), is the sweet spot: enough to shape the payoff, few enough that every cell is a number you can defend.

#### Worked example: the expected value of the whole war-gamed book

Use the matrix above with the base-rate probabilities (de-escalation 65 percent, disruption 28 percent, closure 7 percent) and the net-book payoffs in the rightmost column (de-escalation −\$16,000, disruption +\$36,000, closure +\$142,000). The book is: long oil call options, long gold, and an equity put spread — a structure built to pay in a supply shock.

```
EV = 0.65 × (-$16,000)  = -$10,400
   + 0.28 × (+$36,000)  = +$10,080
   + 0.07 × (+$142,000) = +$9,940
   -------------------------------------
   EV = -10,400 + 10,080 + 9,940 = +$9,620
```

The whole war-gamed book has an expected value of **+\$9,620** even though its single most likely outcome is a \$16,000 loss. Note also that the closure branch — a 7 percent event — contributes nearly as much to the EV (+\$9,940) as the 28-percent disruption branch (+\$10,080), because its payoff is so large. That is the signature of a well-built tail position: a low-probability branch can be a major EV contributor when its payoff is convex. **War-gaming lets you assemble a book that is positive on average and devastating to no branch — the opposite of betting the portfolio on a forecast.**

## Mapping the second-order effects

Here is the mistake that separates amateurs from professionals: stopping at the first effect. A Hormuz closure is "an oil trade" only to someone who quits thinking after one step. The professional traces the chain three or four links deep, because **the first effect is rarely the one that matters most for your actual portfolio.** Most investors do not hold oil directly; they hold equities, bonds, and cash, and those get hit by the *downstream* effects, not the oil price itself.

Trace it. A closure cuts supply, so oil spikes — that is the first effect. Oil is a pervasive input, so the spike passes through into the consumer price index: gasoline, shipping, plastics, food all cost more, and headline inflation jumps — the second effect. Higher inflation forces the central bank's hand: it cannot cut rates into an inflation spike, so it holds rates higher for longer, or even hikes — the third effect. Higher-for-longer rates raise the discount rate on every future cash flow, which compresses equity valuations, *and* they widen the interest-rate differential that supports the dollar — the fourth effect. Meanwhile, the fear itself drives a separate flight-to-safety bid into gold and Treasuries. The portfolio damage lives in those third- and fourth-order effects — the rates-driven equity selloff — not in the oil price that started it all.

![Diagram of a second-order-effects cascade from a Hormuz closure through oil and inflation to rates and then to equities the dollar and gold](/imgs/blogs/scenario-analysis-and-war-gaming-geopolitical-events-3.png)

The cascade map is worth drawing explicitly for any event you war-game, because it reveals trades that the first-order view completely misses. The first-order view says "buy oil." The cascade view says "buy oil, *and* buy a payer option on rates because the central bank gets trapped, *and* buy gold for the haven-plus-inflation double driver, *and* be short the long-duration growth stocks that get hit twice — once by the discount rate and once by the demand hit." The second-order trades are often better risk-reward than the first-order one, because the crowd is all piled into the obvious oil trade while the rates-and-equities consequence is still under-priced in the first hours.

There is a structural reason the second-order trade tends to be cheaper. The first-order link — supply shock to oil price — is so obvious that it is priced almost instantly; the futures gap on the open *is* the first effect, already done. The further down the chain you go, the more steps the market has to reason through, the more participants drop off (an equity manager may not be thinking about the central bank's reaction function at all), and the longer the repricing takes. The third- and fourth-order effects unfold over days and weeks, not seconds, which is exactly the window a war-gamer who mapped the chain in advance can act in. Speed of pricing decays with distance from the headline, and your edge lives in that decay.

A useful discipline is to ask, at each link, *who has to act for the next effect to happen, and what is their incentive?* The oil-to-inflation link happens mechanically (energy is an input cost). The inflation-to-rates link requires the central bank to choose to hold or hike — and its mandate gives it little choice, which is what makes that link reliable rather than speculative; the deeper mechanics are in [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates). The rates-to-equities link happens through the discount-rate channel and through the demand hit. Naming the actor at each link tells you which links are near-certain (mechanical), which are likely (incentive-aligned), and which are genuinely uncertain (a discretionary human choice) — and you size the corresponding trades accordingly: large on the mechanical links, smaller and optional on the discretionary ones.

History rhymes here. The clearest real example of an oil-supply shock cascading is the 2022 Russia invasion of Ukraine, which threatened European energy supply. Brent crude was the first effect.

![Area chart of Brent crude spiking above 120 dollars after the 2022 invasion then round tripping back near pre war levels by year end](/imgs/blogs/scenario-analysis-and-war-gaming-geopolitical-events-6.png)

Brent jumped from roughly \$79 at the start of 2022 to a peak near \$128 in early March — a textbook first-order supply-fear spike. But the trade that defined 2022 was not oil; it was the *second-order* effect. The energy-driven inflation surge forced the most aggressive central-bank tightening cycle in four decades, and *that* — not the oil price — crushed both stocks and bonds simultaneously, producing one of the worst years on record for a standard balanced portfolio. An investor who war-gamed the invasion and stopped at "oil up" caught a spike that round-tripped by year-end (oil ended 2022 near where it started). An investor who traced the cascade to "inflation up, rates up, long-duration assets down" caught the move that actually mattered and lasted all year. **The second-order effect was the trade; the first-order effect was a head-fake.**

#### Worked example: pricing a second-order chain through to a bond move

War-game a supply shock that adds 1.5 percentage points to expected inflation, and price what that does to a 10-year Treasury bond — a pure second-order trade, because the bond has nothing to do with oil directly. Suppose the market responds by repricing the 10-year yield up by 1.0 percentage point (100 basis points), from 4.0 percent to 5.0 percent, as it prices "higher for longer."

A bond's price moves opposite to its yield, scaled by its **duration** — the approximate percentage price change for a 1-percentage-point yield move. A 10-year Treasury has a duration of roughly 8.5 years. So:

```
price change  approx  -duration × yield change
              = -8.5 × (+1.0%) = -8.5%
on a $1,000,000 bond position: -8.5% × $1,000,000 = -$85,000
```

The bondholder loses about **\$85,000** from an event that, on the surface, was "about oil" and never touched a barrel. A war-gamer who held that bond and traced the cascade would have bought a hedge against the rate move — a payer swaption or a short Treasury-futures position — sized to offset that \$85,000, turning a blindside into a planned, neutralized exposure. **The asset that takes the biggest hit is often three links down the chain from the headline, which is exactly why you map the chain before the event, not during it.**

## Pre-positioning a trade per branch

Now the payoff of the whole exercise: the pre-planned trade. The entire value of war-gaming is that it lets you decide *in advance, while calm,* what you will do in each branch — so that when the event fires and everyone else is reasoning from a blank page in a panic, you are simply executing a written plan. This is the difference between a position and a paralysis.

For each branch of the tree, write a single, concrete, executable line. Not "be cautious" or "watch closely" — an actual order. For the Hormuz tree: **de-escalation branch** — "let the closure hedges expire worthless, redeploy the freed premium into equities at the lower volatility." **Brief-disruption branch** — "take profit on half the oil calls into the spike, hold gold, roll the equity put spread down." **Real-closure branch** — "the closure hedges are now the core position; do not chase oil higher, instead add to the rates-payer and gold legs because the second-order inflation trade has the most room left." Each line names the instrument, the direction, and the trigger. Three weeks before the event, with a clear head, you can write good versions of these. Ninety seconds after the headline, you cannot.

This is also where the **hedge-versus-bet** distinction becomes operational. The closure hedges are a *hedge*: you expect them to lose money in the likely branch, and you have pre-decided to let them bleed and expire rather than panic-cutting them the moment they go red. Writing "let them expire worthless" into the de-escalation plan *in advance* is what stops you from doing the single most common hedge-destroying mistake — abandoning the insurance right before the fire. The pre-plan protects you from yourself as much as from the market.

There is a behavioral reason the *written* plan matters so much, beyond mere organization. In the calm of the buildup, you are reasoning with what psychologists call your deliberate, slow system; in the ninety seconds after the headline, you are flooded with adrenaline and reasoning with your fast, fear-driven system. These are, for practical purposes, two different decision-makers, and the panicked one makes systematically worse choices — it overweights the vivid tail, it cuts hedges, it sells at the lows. The pre-written plan is a message from your calm self to your panicked self, and the entire value of writing it down is that it lets the better decision-maker bind the worse one. This is the same logic as a pilot's checklist or a ship's standing orders: the time to decide what to do in an emergency is emphatically *not* during the emergency. A war-game with no written branch trades is barely a war-game at all — it is analysis that evaporates the moment it is needed.

A second operational point: pre-plan the *exits* as well as the entries. For each branch, the plan should say not only what to put on but when to take it off — "take profit on half the oil calls if Brent is up more than 25 percent," "cut the equity short if the index recovers its pre-event level." Geopolitical moves often round-trip (the Brent chart is the canonical example: up 60 percent, then back to flat by year-end), so a trade with no exit rule can give back its entire gain. The branch plan that names both the entry and the exit is the one that actually captures the move rather than watching it come and go.

#### Worked example: sizing the branch trades for positive EV across the tree

Build the book so it is positive expected value across the whole tree, on a \$1,000,000 portfolio with a risk budget of \$20,000 (the most you will spend on hedges). Three legs: oil calls, gold, and an equity put spread. Size them so the de-escalation bleed stays inside the budget while the closure payoff is large.

```
leg sizing (premium spent):
  oil calls         : $8,000 premium  -> de-escal -$8k, disrupt +$30k, closure +$90k
  gold (long)       : $7,000 at risk   -> de-escal -$3k, disrupt +$4k,  closure +$12k
  equity put spread : $5,000 premium  -> de-escal -$5k, disrupt +$2k,  closure +$40k
  total premium at risk = $20,000  (inside the budget)

branch P&L (net of the three legs):
  de-escalation -$16,000 | brief disruption +$36,000 | real closure +$142,000

EV = 0.65(-16,000) + 0.28(+36,000) + 0.07(+142,000)
   = -10,400 + 10,080 + 9,940 = +$9,620
```

The maximum loss is the \$16,000 de-escalation bleed — comfortably inside the \$20,000 budget and 1.6 percent of the portfolio — while the book is +\$9,620 in expectation and pays \$142,000 in the tail that would otherwise wreck the un-hedged portfolio. **Correct sizing is what makes a war-game real: the tree tells you the payoffs, but position sizing is what makes the whole portfolio positive-EV and survivable in every branch at once.** For the deeper mechanics of sizing tail and political exposures, see [position sizing for tail and political risk](/blog/trading/law-and-geopolitics/position-sizing-for-tail-and-political-risk).

## Updating as information arrives

A war-game is not a one-time document; it is a living estimate that you revise as news arrives. This is the Bayesian habit, stripped of the math: when a new piece of evidence comes in, ask "which branch does this make more or less likely?" and shift the probabilities accordingly — then re-derive the trade from the new probabilities. You are not throwing out the tree; you are updating the weights on its branches.

Concretely: a diplomatic back-channel opens — that is evidence for de-escalation, so raise its probability and trim the hedges. A second navy moves into the region — that is evidence for escalation, so raise the disruption and closure probabilities and add to the hedges. The key discipline is to update *the probabilities*, not to abandon the framework and start reasoning from scratch every time a headline crosses. Each headline is a small Bayesian nudge to existing weights, not a reason to tear up the plan. The investor who re-derives their entire worldview from every news alert is, in effect, forecasting again — and forecasting loses.

There is a subtle trap in updating, and it is the mirror image of anchoring too hard: **over-updating on vivid news.** A dramatic but low-information headline (a fiery speech, a provocative statement) feels like a big update but usually carries little actual information about whether the strait closes. A boring but high-information signal (tanker insurance rates, ship-tracking data showing vessels rerouting, the actual flow of oil through the strait) deserves a large update but lands quietly. War-gamers learn to weight updates by their *information content*, not their emotional volume — to move the probabilities a lot on the boring tanker-rerouting data and barely at all on the loud speech.

The honest version of Bayesian updating also requires you to specify, *before* the event, what evidence would move you and by how much. Decide in the calm window: "if war-risk insurance triples, I raise the closure probability to roughly 18 percent; if tankers physically reroute, to roughly 25 percent; if a vessel is actually struck, higher still." Writing the update rules in advance does the same job as writing the trades in advance — it stops the panicked, fast-thinking version of you from either freezing (failing to update on real information) or thrashing (re-deriving everything on every headline). It also guards against the most insidious updating error, *confirmation drift*: the tendency, once you have a position on, to interpret every new headline as supporting your existing trade. Pre-committing to what would change your mind, and by how much, is the antidote. A war-game without pre-specified update triggers tends to calcify into a forecast in disguise — you stop weighing the branches and start defending the one you are positioned for.

#### Worked example: a Bayesian-style update shifting the trade

Start with the base-rate prior: closure 7 percent. Then a high-information signal arrives — ship-tracking data shows commercial tankers have begun rerouting around the strait, and war-risk insurance premiums on Gulf shipping have tripled. This is real behavior by people with money on the line, so it deserves a meaningful update. Suppose you revise the closure probability from 7 percent up to 18 percent, the disruption branch from 28 to 40 percent, and de-escalation down from 65 to 42 percent.

Re-run the equity-leg EV with the updated weights (payoffs unchanged: +1 percent, −3 percent, −12 percent):

```
prior  EV = 0.65(+1%) + 0.28(-3%) + 0.07(-12%) = -1.03%  -> -$10,300
posterior EV = 0.42(+1%) + 0.40(-3%) + 0.18(-12%)
             = +0.42% - 1.20% - 2.16% = -2.94%  -> -$29,400
```

The expected equity damage nearly triples, from −\$10,300 to −\$29,400, because real money is now rerouting away from the strait — a genuine information update, not a headline. The pre-plan responds automatically: the higher closure and disruption weights mean you add to the oil and rates hedges and trim equity exposure, and you do it calmly because you decided the *rule* in advance. **Update the branch weights on real information, re-derive the trade from the new weights, and never tear up the tree just because a headline is loud.**

## Avoiding the over-precision trap

A warning that runs against the grain of everything above: do not mistake the precision of the arithmetic for precision in the world. The tree produces clean numbers — EV of +\$9,620, closure probability of 7 percent — and the danger is believing those numbers to three significant figures. They are not that good. The closure probability is not 7.0 percent; it is "low single digits, call it roughly 7." The payoffs are not exact; they are reasoned estimates. The framework is a tool for *structured thinking*, not a crystal ball with a decimal point.

The discipline is to use ranges and round numbers, and to test whether your conclusion survives reasonable wiggle in the inputs. If the book is positive-EV at a 7 percent closure probability but flips negative at 5 percent, your strategy is too fragile to the one number you are least sure about — that is a red flag, not a green light. A robust war-game produces a conclusion that holds across the plausible range of every input: positive-EV whether closure is 4 percent or 10 percent, survivable whether the closure payoff is −10 percent or −15 percent. **If your conclusion depends on a probability estimate being exactly right, you do not have a conclusion; you have a guess dressed in arithmetic.**

This is also why the three-branch resolution is usually right. You *could* build a fifteen-branch tree with conditional sub-branches and joint probabilities, and it would look impressively rigorous — but every extra branch is another number you cannot estimate, and the false precision compounds. The skill is to make the tree exactly as detailed as your honest knowledge supports, and no more. A simple tree with honest probabilities beats an elaborate tree with invented ones every time.

A good habit for catching over-precision is to run the whole war-game twice — once with your central probabilities, and once with a deliberately pessimistic and a deliberately optimistic set, each shifted by a few percentage points. If the trade is positive-EV and survivable across all three runs, you have a robust conclusion. If it flips between them, you have learned the most valuable thing the exercise can teach you: that your view is not actually a view, it is a coin flip wearing a spreadsheet, and the correct position size is small or zero. The point of the arithmetic is not to produce a precise answer; it is to reveal how sensitive your answer is to the things you do not know. A war-game that always concludes "do the trade" regardless of the inputs is not analyzing anything — it is rationalizing a position you had already decided to take.

## Common misconceptions

**"Scenario analysis is just guessing with extra steps."** No — guessing produces a single number with no structure; scenario analysis produces a probability-weighted distribution anchored in base rates and tested against second-order effects, and crucially it produces a *position* that is positive-EV across branches. The difference is measurable. In the worked examples, the base-rate war-gamer computed a −\$10,300 expected loss and held; the headline-driven guesser computed −\$56,000 and sold the bottom. Same event, and the structured process saved roughly \$45,000 of avoidable loss in the likely branch. That gap is not luck; it is the base-rate anchor doing its job. Guessing has no such anchor.

**"When you are unsure, assign 50/50."** This is one of the most expensive habits in the entire field, and it is exactly backwards. "Unsure" does not mean "equally likely" — it means "I should use the base rate." A Hormuz closure and a non-closure are not 50/50 just because you personally cannot predict which; the historical record says closure is a low-single-digit event and non-closure is the overwhelming default. Assigning 50/50 to a strait closure would put its probability at roughly seven times the base-rate estimate, and as the second worked example showed, inflating the closure weight from 7 percent to 40 percent moved the computed expected loss from −\$10,300 to −\$56,000. "I don't know, so 50/50" is how you end up sizing a panic. The correct response to uncertainty is the outside view, not the coin flip.

**"The most vivid scenario is the most likely."** The vividness of a scenario is *negatively* correlated with its probability as often as not — the catastrophic outcomes are precisely the ones that are both easy to recall and historically rare. The closure branch is the most vivid (warships, fire, a fifth of the world's oil) and the *least* likely (7 percent) in our tree; the de-escalation branch is the most boring and the most likely (65 percent). The Geopolitical Risk index chart shows this directly: attention spikes to five times its mean on the vivid events, then reverts every single time, because the feared outcome almost never arrives. If you let vividness set your probabilities, you will systematically overweight every tail and trade in a state of permanent overreaction.

**"A hedge that loses money was a mistake."** A hedge is *supposed* to lose money in the likely branch — that is the premium you pay for protection, exactly like a fire-insurance policy that "loses" every year your house does not burn down. In the worked book, the hedges cost \$16,000 in the 65-percent de-escalation branch, and that bleed is not a mistake; it is the price of the +\$142,000 payoff in the closure branch and the +\$9,620 positive EV across the tree. The mistake would be cutting the hedge the moment it goes red — which is why the pre-plan writes "let it expire worthless" into the de-escalation branch in advance. Judge a hedge by its EV across the whole tree, never by its profit and loss in a single realized branch.

## How it shows up in real markets

**The 2022 Ukraine invasion: the war-gamed branch trade that paid.** Through January and February 2022, the buildup was visible for weeks — exactly the calm window in which to war-game. A disciplined analyst's tree had branches for "no invasion" (the consensus base case at the time), "limited incursion," and "full invasion." The full-invasion branch carried a specific, pre-planned trade: long energy, long defense, long gold, short long-duration equities and bonds for the inflation-then-rates cascade. When the invasion came on February 24, the pre-planned book executed itself. Brent ran from \$79 to \$128 (the chart above), and — more importantly — the *second-order* inflation-and-rates trade defined the entire year, with the aggressive Fed tightening cycle crushing the standard balanced portfolio. The war-gamer who had written the cascade trade in advance was positioned for the move that mattered; the forecaster who "knew" Russia would not invade was caught flat. For the full mechanics of how conflict reprices each asset, see [war and markets: how conflict prices into assets](/blog/trading/law-and-geopolitics/war-and-markets-how-conflict-prices-into-assets).

**The second-order effect that mattered more than the first.** The most instructive part of 2022 is that the obvious first-order trade — long oil on the supply shock — was a *round-trip*. Brent peaked near \$128 in March and ended the year near \$86, roughly where it started; an investor who put on only the oil trade and held it made very little for the full year. The trade that actually paid all year was the second-order one: the energy-inflation surge forcing the fastest rate-hiking cycle in four decades, which transmitted into a brutal repricing of both equities and bonds. The cascade map — oil to inflation to rates to long-duration assets — was the difference between a trade that head-faked and a trade that lasted. This is the single most reliable lesson of war-gaming real events: **trace the chain past the first link, because the crowd piles into the first effect while the larger, more durable move is two or three links down.**

**Taiwan and the supply-chain branch.** A different shape of geopolitical war-game centers on Taiwan, where the asset at risk is not oil but the world's advanced-semiconductor supply, roughly 90 percent of leading-edge chip fabrication. A war-game here has a very fat tail: the closure branch (a blockade or conflict cutting off TSMC) is low-probability but would be a global supply shock with no near-term substitute, repricing the entire technology complex and triggering a scramble for alternative capacity. The pre-planned tail trade — long the handful of non-Taiwan foundry alternatives and equipment makers, hedged against a broad tech selloff — is a textbook convex hedge: small bleed in the likely (status-quo) branch, enormous payoff in the tail. The mechanics of why this chokepoint is so consequential are in [Taiwan, semiconductors, and the most important supply chain on Earth](/blog/trading/law-and-geopolitics/taiwan-semiconductors-and-the-most-important-supply-chain-on-earth).

**The repeated base-rate fade.** The least dramatic but most lucrative pattern in war-gaming is the one that recurs every few months: a geopolitical scare spikes the Geopolitical Risk index, equities sell off, the headlines scream the catastrophic branch — and then, as the base rate predicted, the event de-escalates and the selloff reverses. Look again at the GPR chart: every one of those spikes was a moment when the catastrophic branch felt imminent, and every one reverted toward the mean of 100. The war-gamer who anchored on the base rate faded the panic each time — buying the equity selloff, selling the volatility spike, letting the tail hedge expire — and was paid for it in the overwhelming majority of cases, while carrying a cheap hedge for the rare time the base rate breaks. This is not a forecast that "it will de-escalate"; it is a position that is positive-EV *because* de-escalation is the base-rate default and the panic systematically over-prices the tail. The edge is not predicting which scare fizzles; it is knowing that *most* scares fizzle and sizing for that distribution. The pattern is reliable enough that it underpins much of how the political calendar is traded; see [elections and political risk: trading the political calendar](/blog/trading/law-and-geopolitics/elections-and-political-risk-trading-the-political-calendar) for the scheduled-event version of the same fade.

## The playbook: how to war-game a geopolitical event

Here is the full process, in the order you run it. The loop figure at the top of this post is this checklist in diagram form.

**1. Frame the branches.** Write down three to five mutually exclusive, collectively exhaustive outcomes for the event. Always include the boring "nothing happens / de-escalation" branch — it is usually the most likely and the one panic forgets. Resist the urge to over-split; three branches with honest probabilities beat fifteen with invented ones.

**2. Anchor the probabilities in base rates.** For each branch, start from the historical frequency of that kind of outcome (the outside view), then adjust *cautiously* for the specifics. Never start from the vivid narrative. Never default to 50/50 when unsure — "unsure" means "use the base rate." Write the probabilities as round numbers and ranges, not false-precision decimals.

**3. Price the payoffs.** For each branch, estimate what each relevant asset does *if that branch happens* — independently of how likely the branch is. Price the scary branch's payoff at its true, full size even if you think it is unlikely; that is the information you need to value a hedge.

**4. Map the second-order effects.** Trace the cascade three or four links deep: first effect to who reacts to next effect. Find the trade that lives downstream (the rates-and-equities move behind the oil shock), because the crowd is in the first-order trade and the second-order one is under-priced.

**5. Build the EV matrix.** Branches down, positions across, dollar payoff in each cell, probability-weighted net at the bottom. Engineer the position sizes until the book is positive-EV across the tree and survivable in the worst branch.

**6. Pre-plan the branch trades.** Write one concrete, executable order per branch — instrument, direction, trigger — while you are calm. Decide in advance to let hedges expire in the benign branch so panic cannot make you cut your insurance at the worst moment.

**7. Run the pre-mortem.** Assume the trade lost money and write down why: did the vivid story inflate a tail? Is each probability anchored to a real base rate? Did you miss a second-order effect? Is each leg a hedge or a bet, and sized accordingly? What single piece of news invalidates the whole view?

![Diagram of a pre-mortem checklist grid with six questions on narrative inflation base rates second-order effects hedge versus bet invalidation and false precision](/imgs/blogs/scenario-analysis-and-war-gaming-geopolitical-events-8.png)

**8. Update on information, not noise.** As news arrives, nudge the branch probabilities by the *information content* of the news, not its emotional volume — large updates on tanker-rerouting and insurance data, tiny updates on speeches. Re-derive the trade from the new weights. Never tear up the tree.

**What invalidates the war-game.** Three things. First, if the base rate itself has structurally changed — a genuinely new weapon, alliance, or capability that makes the historical frequency irrelevant — then the outside view is broken and you must rebuild from the new regime. Second, if your conclusion flips on a small wiggle in your least-certain probability, the strategy is too fragile and should not be sized up. Third, if you find yourself re-forecasting — committing to a single outcome and abandoning the tree — you have left war-gaming and gone back to guessing, which is the thing the whole process exists to prevent. The discipline is to stay in the tree: branches, base rates, payoffs, EV, pre-planned trades, and disciplined updates. That is how you turn an unforecastable event into a positive-expected-value position, and paralysis into a plan.

For how to fold political and tail risk into your overall risk budget and sizing, see [position sizing for tail and political risk](/blog/trading/law-and-geopolitics/position-sizing-for-tail-and-political-risk), and for the full synthesis of the law-and-geopolitics toolkit, [the law, policy, and geopolitics playbook](/blog/trading/law-and-geopolitics/the-law-policy-and-geopolitics-playbook).

## Further reading & cross-links

- [War and markets: how conflict prices into assets](/blog/trading/law-and-geopolitics/war-and-markets-how-conflict-prices-into-assets) — the asset-by-asset reaction map and base rates that the war-game's payoffs are built on.
- [Taiwan, semiconductors, and the most important supply chain on Earth](/blog/trading/law-and-geopolitics/taiwan-semiconductors-and-the-most-important-supply-chain-on-earth) — a fat-tail supply-chain scenario to war-game.
- [Position sizing for tail and political risk](/blog/trading/law-and-geopolitics/position-sizing-for-tail-and-political-risk) — how to size the pre-planned branch trades within a risk budget.
- [The law, policy, and geopolitics playbook](/blog/trading/law-and-geopolitics/the-law-policy-and-geopolitics-playbook) — the capstone synthesis of the whole toolkit.
- [Cross-asset allocation and rebalancing](/blog/trading/cross-asset/building-a-cross-asset-allocation-and-rebalancing) — how the branch trades fit into a full multi-asset book.
- [A gold playbook: reading the macro signals and building a position](/blog/trading/gold/a-gold-playbook-reading-the-macro-signals-and-building-a-position) — the haven-plus-inflation leg that recurs in nearly every geopolitical war-game.

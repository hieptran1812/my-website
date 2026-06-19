---
title: "Base, Bull, and Bear: Building Three Scenarios"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How to replace a fragile point forecast with three weighted scenarios — base, bull, and bear — each with a driver, a target, a probability, and a path, then collapse them into a probability-weighted target and an asymmetry read you can size and grade."
tags: ["analysis", "market-view", "scenario-analysis", "base-bull-bear", "probability-weighting", "expected-value", "asymmetry", "risk-reward", "position-sizing", "calibration", "re-weighting", "process"]
category: "trading"
subcategory: "The Analyst's Edge"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A single point forecast is fragile because it hides everything that could go the other way. Professionals build **three scenarios** — base, bull, bear — each with a **driver**, a **target**, a **probability**, and a **path**, then collapse them into a probability-weighted target and an asymmetry read.
>
> - A view without a bear case cannot be sized. The bear case is not pessimism; it is the worst-case loss budget the whole position is built around.
> - The three scenarios must differ by the **state of one driver**, not by arbitrary price levels. If your bull and bear are just base ±10%, you have one scenario painted three colors.
> - The **probability-weighted target** is one number you can compare to what's priced. The **bull/bear skew** is what tells you whether the trade is worth more than its expected value suggests.
> - The one rule to remember: **never carry a target without the probability beside it and the bear case under it.** A target alone is a wish; a target with a weight and a downside is a position.

## "And if you're wrong?"

It is a Tuesday research meeting and a junior analyst is pitching a semiconductor name trading around \$100. The pitch is fluent. The end market is inflecting, the new product cycle is underappreciated, channel checks are good. "I think it goes to \$120," he finishes, with the quiet confidence of someone who has done the work. The portfolio manager nods slowly, then asks the only question that matters: *"And if you're wrong?"*

Silence. Not because the analyst hasn't thought about it — he has, vaguely, in the back of his mind, the way you know a road has a cliff somewhere off to the side without ever measuring how far down it goes. But he has no number. He cannot say *where* the stock goes if the cycle disappoints, or *how likely* that is, or *what he'd see first* on the way down. He has a target. He does not have a view. And a view you cannot describe being wrong about is a view you cannot size, because the PM has nothing to anchor a stop or a position weight to. The \$120 target, however well-researched, is unsizeable. It dies in the meeting.

The fix is not more conviction in the \$120. The fix is to stop pretending the future is a single number. The stock does not "go to \$120." It goes to one of many places, and \$120 is merely the analyst's guess at the center of a cloud he never drew. The discipline this post teaches is drawing that cloud deliberately — collapsing the infinite fan of futures into three named, weighted scenarios you can actually reason about: a **base** case (what you most expect), a **bull** case (what happens if your thesis works better than expected), and a **bear** case (what happens if it breaks). Three targets, three probabilities, three paths. From those nine numbers fall everything the PM was missing: a probability-weighted target, a risk-reward read, and a loss budget to size against.

![One price today fans into three weighted price paths labeled base, bull, and bear](/imgs/blogs/base-bull-and-bear-building-three-scenarios-1.png)

This sits downstream of the work earlier in the series. You have already structured a thesis with a claim, evidence, and a catalyst — [structuring a thesis: claim, evidence, and catalyst](/blog/trading/analyst-edge/structuring-a-thesis-claim-evidence-and-catalyst) — and you have mapped what the market already believes — [what's priced in: the question behind every trade](/blog/trading/analyst-edge/whats-priced-in-the-question-behind-every-trade). The scenario set is how you turn that structured thesis into a *distribution* — a quantified, gradeable object — instead of a single brave number. It is the bridge between having a view and being able to bet on it.

## Foundations: scenarios, weights, and why a distribution beats a point

Before building anything, we define the machinery precisely, because most scenario work fails not in the math but in sloppy definitions of what each piece is supposed to be.

### Scenario analysis, and what a "scenario" actually is

**Scenario analysis** is the practice of describing the future as a small set of distinct, internally-coherent stories — each with its own outcome and its own probability — rather than as a single forecast. A **scenario** is not a price level. It is a *world*: a specific state of the one or two variables that drive the outcome, plus the price that world implies. "The stock goes to \$140" is not a scenario. "The Monterrey plant ramps faster than guided, segment margin prints 23% instead of 20%, and the stock re-rates to \$140" is a scenario — it names the driver state (fast ramp), the consequence (higher margin), and the price (\$140). The price is the *last* thing in a scenario, derived from the world, not asserted on its own.

The convention is three of them. **Base case**: the single most likely world — your central expectation, what you'd bet on if forced to pick one. **Bull case**: the world where your thesis works *better* than the base — the driver breaks favorably, the upside surprise lands. **Bear case**: the world where the thesis breaks — the driver disappoints, and the position loses money. Three is not magic; it is the smallest set that captures the center, the upside tail, and the downside tail without drowning you in detail. (We will see later why four or seven scenarios are usually worse, not better.)

### Probability weighting, and why the weights must sum to one

Each scenario carries a **probability** — your honest estimate of how likely that world is. The three probabilities must **sum to 1.0**, because the three scenarios are meant to *partition* the future: between them, they cover everything that can happen, so the total likelihood is 100%. If your base, bull, and bear sum to 0.9, you have a 10% hole — some world you haven't named — and your weighted math is wrong. If they sum to 1.1, you have double-counted. The sum-to-one constraint is not bookkeeping pedantry; it is the discipline that forces you to ask "what *else* could happen?" until the unaccounted mass is gone.

These are subjective probabilities — your degree of belief — not frequencies you measured. That is fine and unavoidable; every forward view is a subjective probability. What matters is that they are *calibrated*: when you say 55%, the thing should happen about 55% of the time across many such calls. We will return to calibration, because it is the skill that separates a useful weight from a number you made up to feel rigorous.

### Scenario versus guess: why the distribution beats the point

Here is the core reason this whole apparatus exists. A point forecast — "\$120" — is a single number that throws away all the information about *how confident you are* and *how the outcomes spread*. Two analysts can both forecast \$120 and mean wildly different things: one means "almost certainly \$118–\$122," the other means "probably \$120 but it could be \$80 or \$160." Those are completely different trades — one is a high-conviction tight bet, the other a wide low-conviction punt — and the point forecast cannot tell them apart. The scenario set *can*: it shows you the spread, the skew, and the downside explicitly.

There is a second, deeper reason rooted in how non-linear payoffs work. If the relationship between the driver and your P&L is curved — and in markets it almost always is, because of stops, options, leverage, and the simple fact that a 50% loss needs a 100% gain to recover ([the asymmetry of losses: why a 50% loss needs a 100% gain](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain)) — then the expected outcome is *not* the outcome at the expected input. Plugging your single best-guess driver into the model gives a different, usually too-optimistic number than averaging the P&L across the full set of scenarios. This is a consequence of Jensen's inequality, and the practical upshot is blunt: **you must average over outcomes, not forecast the average.** The scenario set is how you do that with three points instead of a full simulation.

A third reason is purely behavioral, and it may be the most important in practice. A point forecast is psychologically *sticky* in a way a distribution is not. Once you have said "\$120," you have anchored yourself to a single number, and every subsequent piece of information gets filtered through the question "does this support \$120?" — a frame that quietly recruits confirmation bias. A scenario set defends against this by making the alternatives explicit and *pre-committed*: the bear case is already written down, already weighted, already waiting, so when bad news arrives it has a home to go to rather than being argued away. The act of having named the bear before you needed it is what lets you act on it without ego when it arrives. A point forecaster has to *admit they were wrong* to respond to bad news; a scenario thinker merely *re-weights*, which is emotionally cheap and therefore actually happens. The distribution doesn't just produce better math — it produces better behavior under pressure, which is where most edge is won and lost.

The point forecast, in short, is a distribution that someone collapsed to its mean and then forgot was ever a distribution. Scenario analysis simply refuses to forget.

### Calibration: the skill that makes a probability worth anything

A probability you assign is only useful if it is **calibrated** — meaning that, over many predictions, your stated odds match the realized frequencies. If you label fifty different events "70% likely" and roughly thirty-five of them happen, you are well calibrated and your 70% means something a risk manager can size against. If only twenty happen, you are overconfident: your "70%" is really a 40%, and every position you built on it was too big. Calibration is not about being right on any single call — you will be wrong on 30% of your 70% calls *by design* — it is about your *numbers* being honest across many calls.

This matters enormously for scenario weights, because the weights are the load-bearing inputs to the weighted target and the sizing decision. A miscalibrated analyst who systematically assigns bull cases 35% when their true frequency is 20% will systematically over-target and over-size, and no amount of good stock-picking rescues a process whose probabilities are inflated. The way you find out whether you are calibrated is the same way you find out anything in this craft: you write the weights down, you wait for resolution, and you score yourself. The standard score is the **Brier score** — the average squared error between your stated probability and the 0/1 outcome — and lower is better. You do not need to compute it formally to benefit; the mere act of recording "I said bull 25%" and later checking "did the bull-ish worlds happen about a quarter of the time across my last forty calls?" is most of the value. The series treats the broader habit of grading yourself in its work on the decision journal, but the seed is here: a scenario weight is a falsifiable claim, and an analyst who never scores their weights is flying blind on the most important input they produce.

The most common calibration failure in scenario work is a specific one: the **base case is too wide and too confident at once.** Analysts love a base case at 70% because it feels like conviction, but a 70% base case implies you are *quite sure* which world you're in — and for most market questions, with a genuine fork in front of you, you are not. A more honest base case for a real, uncertain catalyst is often 45–55%, with meaningful mass on both the bull and the bear. If your base case is routinely 70%+, you are probably either ignoring real uncertainty or, worse, defining your scenarios so loosely that the base "world" quietly absorbs most of the bull and bear too.

## Building the three scenarios

A scenario set is built one scenario at a time, and each scenario is built from the same four parts: the **driver** that defines it, the **target** it implies, the **probability** you assign it, and the **path/trigger** that would put you in that world. Get these four right for each of base, bull, and bear and the rest is arithmetic.

![Scenario table for Meridian with driver, target, probability, and path columns for bull, base, and bear](/imgs/blogs/base-bull-and-bear-building-three-scenarios-2.png)

We'll build a complete worked set for a single name and carry it through the whole post. The name is **Meridian Industrial**, a fictional mid-cap that just opened a new plant in Monterrey. It trades at \$100. The thesis: the plant ramps, segment margin expands, and free cash flow inflects. The whole scenario set hangs off one driver — *how fast the plant ramps* — which is exactly what makes it a good set.

### The driver: what actually defines each scenario

The single most important rule in scenario construction: **the three scenarios must differ by the state of one driver, not by three arbitrary price levels.** The driver is the variable that, more than any other, determines which world you end up in. For Meridian it is plant utilization, which flows through to segment margin and then to the stock. Name it explicitly, then define each scenario as a *state* of that driver:

- **Bear**: ramp stalls, utilization stays under 55%, margin holds at the old ~16%, demand for the new line disappoints.
- **Base**: ramp proceeds on plan, utilization reaches ~75%, margin lands at the guided 20%.
- **Bull**: ramp beats plan, utilization passes 90%, margin prints 23% as fixed-cost absorption surprises.

Notice that the *price targets fall out of these worlds* — they are not the starting point. The bull is \$140 because a 23% margin on the segment, run through the same multiple, produces roughly that. The bear is \$80 because flat margin plus a de-rating on a broken growth story produces roughly that. This is the discipline that keeps your bull and bear from being lazy ±10% bands: each one is anchored to a *different state of the world*, so the gap between them reflects genuine uncertainty about the driver, not a number you picked to look balanced.

![Scenario driver tree branching from plant utilization into bull, base, and bear targets](/imgs/blogs/base-bull-and-bear-building-three-scenarios-6.png)

The tree above shows the structure: one master driver at the top, three states of it in the middle, and the price each state implies at the bottom. When you can draw your scenario set this way — one driver, three branches, three targets — you know the set is coherent. When you can't, when the branches need different drivers to reach different prices, that is a signal your scenarios are not really alternatives to each other; they are three separate theses pretending to be one. Pick the dominant driver and let it define the fork.

#### Worked example: building the targets from the driver

Meridian's industrial segment does \$2.00 of revenue per share, and the stock trades at \$100. In the **base** world, segment margin is 20%, so segment profit is \$0.40 per share; apply the sector's normal ~12.5× multiple to that profit stream (alongside the rest of the business) and the math supports roughly **\$115** — a +15% move. In the **bull** world, margin prints 23%, lifting segment profit to \$0.46 per share, and a faster-growing, de-risked franchise earns a richer multiple, supporting **\$140** — a +40% move. In the **bear** world, margin stays at 16% (\$0.32 per share) *and* the growth story breaks, so the multiple compresses too; the two effects compound down to roughly **\$80** — a −20% move. Each target is the *output* of a margin assumption, not an input — the takeaway is that a target you cannot derive from a driver is a target you cannot defend when the driver moves.

### The target: a price, with a magnitude

The **target** is the price the scenario's world implies. State it as a level (\$140) and as a magnitude from spot (+40%), because the magnitude is what you'll weight and compare. The base target is your central estimate; the bull and bear targets are the prices in their respective worlds. Be honest about magnitudes — the most common failure is a bull case that is timid (because naming a big number feels arrogant) and a bear case that is shallow (because naming a big loss feels uncomfortable). Both timidities destroy the value of the exercise, because the entire point is to capture the real spread.

### The probability: how likely is that world

Assign each scenario a probability, and make the three **sum to 1.0**. For Meridian: base **0.55**, bull **0.25**, bear **0.20**. The base gets the most weight because it is the most likely single world. The bull and bear split the rest. There is no formula that hands you these numbers — they are your calibrated judgment — but there are disciplines that make them honest:

- **Start from base rates, then adjust.** How often do plant ramps of this type hit guidance, beat it, or stall? If the historical base rate of "ramp beats guidance" for comparable projects is ~25%, your bull probability should start near 25% and only move on specific evidence about *this* plant.
- **Force the sum to one out loud.** Write base = 0.55, bull = 0.25, and notice that bear must be 0.20. If 0.20 feels too high for how bad the bear is, you have learned something: either the bear isn't that bad, or you are underweighting it because it's unpleasant.
- **Avoid the false-precision trap and the round-number trap together.** Don't write 0.5273 (false precision you don't have), but also don't always default to 0.6/0.2/0.2 (round numbers that mean "I didn't think"). Probabilities in 5% increments are honest about the resolution of your judgment.

### The path and trigger: how you'd get there and what you'd see first

The fourth part is the **path** — the sequence of observable events that would put you in each world — and the **trigger**, the specific marker that would first tell you a given scenario is playing out. For Meridian's bull case, the path runs through the Q2 print (early margin upside) and confirms at the Q3 print (the ramp at scale). For the bear, the path is an order-book deterioration and a Q3 margin that prints flat. The path matters because it converts the scenario from a static guess into a *thing you can monitor*: it tells you exactly what to watch and, crucially, the *order* you'd see it in, so you can update before the price has fully moved.

The path is also where you catch a scenario that is *internally incoherent* — one whose price target doesn't actually follow from its driver state. A useful test is to walk the path forward and ask, at each step, "is this how the market would actually re-price?" If your bull case says \$140 but the only path to \$140 requires the multiple to expand *and* margins to beat *and* a buyback to be announced — three independent things — then it isn't a 25% scenario; it is a conjunction of three favorable events whose joint probability is far lower than any one of them. Decomposing the path exposes these hidden conjunctions. A clean scenario reaches its target through *one* dominant mechanism; a scenario that needs three things to all go right is secretly a low-probability tail you've mislabeled as a credible bull.

### The fourth bucket: the world you didn't name

Three scenarios are the working convention, but the sum-to-one constraint forces an uncomfortable question: what about the world that fits *none* of your three? The plant doesn't ramp slowly or quickly — a fire shuts it for two quarters. The macro fork resolves, but a credit event nobody modeled hits first. These are the **unaccounted tails**, and the honest move is not to add a fourth and fifth scenario for every disaster you can name — that way lies the seven-scenario mush we'll criticize shortly. The honest move is to *acknowledge the residual mass explicitly* and fold it into the bear case as a small haircut, or to carry a thin "other/tail" probability and assign it a deliberately conservative (low) price.

For Meridian, the three named worlds — fast ramp, on-plan, stall — are exhaustive *of the ramp driver*, which is what makes them a clean partition. But they assume the plant exists and operates; a genuine left-tail event (a recall, a fraud, a key-customer loss) lives outside that partition. The disciplined treatment is to size the position so that even a scenario *worse than your stated bear* doesn't ruin you — which is precisely why sizing is governed by the bear-case loss budget *plus* a margin of safety, never by the bear case taken as a hard floor. The bear case is your *modeled* worst case; the real worst case is always somewhat worse, and your position size has to survive that gap. This connects to the broader point that real return distributions have fatter tails than any three-bucket model captures — [fat tails and the normal-distribution trap](/blog/trading/risk-management/fat-tails-and-the-normal-distribution-trap) — so the bear case is a floor on your *thinking*, not a floor on your *losses*.

### Stress-testing the bull and the bear before you trust them

Once the three scenarios are drafted, attack the two tails before the market does. The bull and bear are where motivated reasoning hides: you want the bull to be big (it justifies the trade) and you want the bear to be shallow (it makes the trade feel safe), and both pulls are toward a flattering, useless set. Two stress tests counter this directly.

First, the **pre-mortem on the bear**: assume the bear case has happened, the stock is at \$80, and write the post-mortem explaining why — *before* you trade. If the explanation is easy to write ("orders that looked firm were one customer who pulled them"), the bear is realistic and probably under-weighted. If you struggle to write a credible bear story, either your bear is too shallow or you don't understand the downside well enough to be in the trade. Second, the **devil's-advocate on the bull**: hand your bull case to the most skeptical colleague you have and let them argue it down. The bull case that survives a hostile read at 25% is worth more than the one you assigned 25% in the comfort of your own head. Both tests do the same job — they pull the tails back toward the truth, which is the only thing that makes the weighted target and the asymmetry honest.

## Collapsing the set: the probability-weighted target and the asymmetry

Once you have three targets and three probabilities, two numbers fall out, and they do different jobs.

### The probability-weighted target

The **probability-weighted target** (the expected target price) is the sum of each target times its probability. It is the single number you compare against what the market has priced. For Meridian:

> weighted target = 0.55 × \$115 + 0.25 × \$140 + 0.20 × \$80
> = \$63.25 + \$35.00 + \$16.00 = **\$114.25**

So your one-number expectation is \$114.25 against a spot of \$100 — a +14.25% expected move. That is the number you set beside the market's implied expectation to find your edge.

![Bar chart of bull, base, and bear price targets with a weighted target reference line](/imgs/blogs/base-bull-and-bear-building-three-scenarios-3.png)

The chart makes the weighting visible: each bar is a target, the dashed line is the \$114.25 weighted target, and the inner contribution figures (\$63.25 from base, \$35.00 from bull, \$16.00 from bear) show that the base case does most of the work but the bull and bear meaningfully pull the average. Note that the weighted target (\$114.25) sits *below* the base case (\$115) — the bear case is dragging it down by more than the bull lifts it, which is itself a piece of information about the trade.

#### Worked example: the dollar EV on a \$20,000 position

Suppose you buy **\$20,000** of Meridian at \$100 — that is **200 shares**. The expected value of the position is the weighted target minus the entry, times shares:

> \$ EV = (weighted target − entry) × shares
> = (\$114.25 − \$100.00) × 200
> = \$14.25 × 200 = **+\$2,850**

Equivalently, weight the dollar P&L of each scenario directly: base earns (\$115 − \$100) × 200 = +\$3,000 at 0.55 (+\$1,650 contribution); bull earns +\$8,000 at 0.25 (+\$2,000); bear loses (\$80 − \$100) × 200 = −\$4,000 at 0.20 (−\$800). Sum: \$1,650 + \$2,000 − \$800 = **+\$2,850**. The two methods agree, as they must — the lesson is that your \$20,000 has a positive expected value of \$2,850, but it also has a real \$4,000 loss living inside the bear case, and you now know both numbers instead of only the happy one.

### The asymmetry: why two trades with the same EV are not the same trade

The weighted target hides something a position-sizer desperately needs to see: the **asymmetry** — how the upside compares to the downside. Two scenario sets can produce the same +14% weighted target while being completely different trades. One might have a base \$115 / bull \$118 / bear \$110 (a tight, symmetric cluster) and another a base \$115 / bull \$160 / bear \$60 (a wide, fat-tailed spread). Same center, radically different risk. The **skew** — whether the spread is wider to the upside (good) or the downside (bad) — is the second number you read off every scenario set.

The cleanest way to see asymmetry is to look at the *shape* of the per-scenario payoff: a setup with a shallow bear and a deep bull is the one you want, because it loses a little when wrong and wins a lot when right. Let me make this concrete with a different name.

![Bar chart of asymmetric Helios position P&L with shallow bear loss and deep bull gain](/imgs/blogs/base-bull-and-bear-building-three-scenarios-4.png)

#### Worked example: an asymmetric setup with a shallow bear and a deep bull

Consider **Helios**, trading at \$50, where you put on **\$20,000** — that is **400 shares**. The scenario set is asymmetric: bear **\$45** (a shallow −\$5/share) at **0.20**, base **\$55** (+\$5/share) at **0.50**, bull **\$80** (a deep +\$30/share) at **0.30**. The per-scenario P&L on 400 shares is bear −\$2,000, base +\$2,000, bull +\$12,000. The expected value:

> \$ EV = 0.20 × (−\$2,000) + 0.50 × (+\$2,000) + 0.30 × (+\$12,000)
> = −\$400 + \$1,000 + \$3,600 = **+\$4,200**

Now read the *asymmetry*: your worst case loses \$2,000, your best case makes \$12,000 — a **6:1** ratio of best-case gain to worst-case loss. Even the base case makes money. This is a setup where the bear is a scratch and the bull is a windfall; the chart shows the deep green bull bar towering over the shallow red bear bar, with the EV line sitting comfortably positive. The intuition to keep is that the \$4,200 EV undersells this trade — the shape, where being wrong barely costs you, is worth more than the average number alone suggests, and that shape is exactly what point forecasts destroy.

The general principle connects straight to options and convexity: you want trades where the *distribution* is skewed in your favor, not just where the mean is positive. A trade with a +\$4,200 EV and a 6:1 best-to-worst ratio is far more attractive than a trade with the same EV and a 1:1 ratio, because the second one is one bad print away from a large loss. This is the same asymmetry an options market maker prices every day on the other side of your trade — [how an options market maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) — and it is why the bear case is the most important of the three scenarios even though it is the one nobody wants to write.

### The weighted target only matters against what's priced

The weighted target is not an edge by itself — it is an edge *only relative to what the market already expects*. This is the spine of the whole series: your view is worth money in proportion to how far it sits from the consensus, not from spot. A +14% weighted target on Meridian is a strong trade if the market is priced for +2%, an *unremarkable* one if the market is already priced for +14%, and a *short* if the market is priced for +25% and you think the bull is less likely than it's discounting. The number you build with the scenario set is only the first half of the comparison; the second half is the market's own implied scenario set, which you reconstruct from price, positioning, and option skew — [what's priced in: the question behind every trade](/blog/trading/analyst-edge/whats-priced-in-the-question-behind-every-trade).

This reframes the entire exercise. You are not really forecasting Meridian's price; you are forecasting the *gap* between your weighted target and the market's. And the most powerful version of the scenario set is the one where you and the market agree on the three worlds but disagree on the *weights* — you think the bull is 25% and the market is pricing it at 10%. That is a pure variant-perception trade on probability, and it is far more defensible than disagreeing on the targets themselves, because targets are arithmetic the whole street can replicate while weights are judgment the street can get wrong.

#### Worked example: same targets, different weights from the market

Suppose the market's implied scenario set for Meridian is bull 0.10, base 0.55, bear 0.35 — it agrees with your three worlds and your three targets but is far more pessimistic on the ramp. The market's implied weighted target is 0.10 × \$140 + 0.55 × \$115 + 0.35 × \$80 = \$14.00 + \$63.25 + \$28.00 = **\$105.25**. Yours is \$114.25. The **gap is \$9.00 per share**, or +9% — and *that* gap, not your raw +14% target, is your edge. On a \$20,000 position (200 shares) the gap is worth (\$114.25 − \$105.25) × 200 = **+\$1,800** of expected value attributable to your variant view on the weights. The lesson is that an analyst who reports "my target is \$114" has said almost nothing; an analyst who reports "I think the bull is 25% versus the market's 10%, worth \$9 a share" has located the edge precisely.

## The markers: what shifts the probability mass

A scenario set is not a static prediction; it is a *living* estimate that should update as evidence arrives. The mechanism for updating is the set of **markers** — the specific, pre-committed observables that, when they print, shift probability mass from one scenario to another. Defining them in advance is what turns "I changed my mind" from a vibe into a disciplined re-weighting.

![Grid of markers across fundamental, macro, and positioning domains that shift mass toward bull or bear](/imgs/blogs/base-bull-and-bear-building-three-scenarios-5.png)

The grid organizes Meridian's markers by domain (fundamental, macro/sector, positioning) and direction (which way each shifts the weights). A Q2 margin that beats 21% shifts mass toward the bull; a falling backlog shifts mass toward the bear. The capex cycle re-accelerating helps the bull; a PMI rolling under 50 helps the bear. Crowded shorts capitulating into weakness is a bull marker; longs already at max position is a bear marker (it caps the marginal buyer — a positioning read covered in [reading flows and positioning](/blog/trading/analyst-edge/reading-flows-and-positioning-the-tell-most-analysts-miss)). The discipline is to **write these markers down before the prints**, because pre-committed markers update your weights mechanically, while undefined markers let you rationalize whatever happened into "still on track."

The reason this matters so much: without pre-committed markers, every new data point gets absorbed into your existing view (you see what you expected to see), and your scenario weights never actually move. With them, a marker firing is a *contract you signed with yourself* to move mass in a specific direction. This is the operational version of variant perception staying honest — [variant perception: where real edge comes from](/blog/trading/analyst-edge/variant-perception-where-real-edge-comes-from) — because it forces your view to respond to the world rather than the other way around.

### Re-weighting as evidence arrives

When a marker fires, you move probability mass — and because the weights sum to one, lifting one scenario must lower the others. This is Bayesian updating done with three buckets instead of a continuous distribution: the prior is your current weights, the marker is the new evidence, and the posterior is the re-weighted set. You do not need to run Bayes' theorem formally on every print; the three-bucket version is a disciplined approximation that captures the essential move — strong evidence for one world *must* come at the expense of the others, and the size of the shift should scale with how surprising and how decisive the evidence is.

How much should a given marker move the weights? The honest answer is "as much as it changes your belief about the driver, and no more." A small, ambiguous data point — one channel check, one analyst note — should move mass by a few points at most; a decisive, hard data point — the actual margin in the actual filing — can move it by twenty or thirty. The error to avoid in both directions is the *sticky-weights* failure (refusing to update because you're anchored to your original thesis) and the *whipsaw* failure (lurching the weights on every noisy tick). The marker discipline guards against both: because you pre-committed which observables matter and roughly how much, a noisy tick that isn't on your marker list moves nothing, while a marker that *is* on your list moves the pre-agreed amount. Your weights respond to signal and ignore noise, which is the entire point of separating signal from noise in the first place — [building your information diet: signal versus noise](/blog/trading/analyst-edge/building-your-information-diet-signal-versus-noise).

![Before and after weight bars for bull, base, and bear shifting after the Q2 margin print](/imgs/blogs/base-bull-and-bear-building-three-scenarios-7.png)

The before/after above shows what a single hot Q2 print does. Before Q2, the weights are the originals: bull 0.25, base 0.55, bear 0.20. Q2 segment margin prints 21.5% — clearly hot, a bull marker firing. You move mass: the bull rises to 0.45, the base falls to 0.45, the bear collapses to 0.10 (the strong print makes the "ramp stalls" world much less likely). The weights still sum to 1.0. And the weighted target moves with them.

#### Worked example: the dollar impact of re-weighting

Re-run Meridian's weighted target with the new weights after the hot Q2 print:

> new weighted target = 0.45 × \$140 + 0.45 × \$115 + 0.10 × \$80
> = \$63.00 + \$51.75 + \$8.00 = **\$122.75**

The weighted target jumped from \$114.25 to \$122.75 — a +\$8.50 lift — *without any change to the targets themselves.* Only the probabilities moved. On your 200-share position, the expected P&L rises from +\$2,850 to (\$122.75 − \$100) × 200 = **+\$4,550**, and the bear-case loss budget shrinks because the bear now carries only 10% weight. The lesson is that most of your edge after a catalyst comes from re-weighting, not from re-forecasting — the targets were already right; the print told you which world you were heading into, so you let the position run and your expected value compounded with the new information.

## Common misconceptions

Scenario analysis is simple enough to describe and easy enough to do badly that the same four mistakes recur constantly. Each one quietly destroys the value of the exercise.

### "The base case is the forecast"

The most common error is treating the base case as *the* answer and the bull and bear as decorative footnotes. The base case is **one of three**, and the number you actually act on is the *probability-weighted* target, which is usually not the base. For Meridian, the base is \$115 but the weighted target is \$114.25 — the bear drags it below the base. If you size to the base case and ignore the weighting, you systematically over-position, because you're betting on the single most likely world as if it were certain. The base case is the center of mass of your *most likely* scenario, not the center of mass of the *distribution* — and you trade the distribution.

### "Bull and bear are just base ±10%"

The lazy scenario set takes the base, adds 10% for the bull, subtracts 10% for the bear, and calls it analysis. This produces three numbers that share one driver state and one story — it is a single scenario wearing three hats. A real bull and bear are anchored to *different states of the driver* and therefore usually have *asymmetric* magnitudes: Meridian's bull is +40% and its bear is −20%, because the upside (margin surprise plus re-rating) and the downside (margin disappointment plus de-rating) are genuinely different sizes. If your bull and bear are mirror images, you have almost certainly not thought about the driver — you've just drawn error bars around a point forecast and relabeled them.

### "Scenarios are academic — the market does what it does"

Some traders dismiss the whole exercise as ivory-tower busywork: the future is one path, you'll know it when you see it, why pretend to assign probabilities to worlds that won't happen? This confuses *forecasting* with *decision-making under uncertainty*. You are not predicting which scenario happens; you are sizing a bet across all of them. The probabilities are inputs to a sizing decision you have to make *today*, before the future resolves. A trader who refuses to quantify the bear case isn't being a realist — they're being a realist who can't tell you how big to bet or where to stop, which is the entire job. The scenario set is the most practical document in the book precisely because it converts an uncertain future into a single actionable position.

### "More scenarios are better"

If three scenarios are good, surely seven are more rigorous? No — and this is the subtlest mistake. More scenarios feel more thorough but degrade the analysis in three ways. First, the probabilities become meaningless: you cannot calibrate a 7-way partition (is this world 11% or 14%?) the way you can calibrate "more likely than not / decent shot / tail risk." Second, the scenarios stop being *distinct* — seven scenarios always include several that differ only trivially, which means you're double-counting some part of the distribution. Third, it hides the asymmetry, which is the whole point: three scenarios put the upside and downside tails in sharp relief, while seven smear them into a gradient you can't read. The skill is *collapsing* the infinite future into the three points that carry the most decision-relevant information — center, upside tail, downside tail — not enumerating possibilities for their own sake.

### "The weighted target is the price target I publish"

A final, quieter error: treating the probability-weighted target as a precise price prediction to be defended to two decimal places. The weighted target is a *summary statistic of your belief distribution*, useful for comparing against what's priced and for computing EV — it is not a claim that the stock will trade at exactly \$114.25. The stock will end up at one of your scenario prices (or near one), almost never at the weighted average, because the average of \$80, \$115, and \$140 is a number none of the three worlds produces. Confusing the weighted target with a forecast leads to two failures: you defend a number that was never meant to be a forecast, and you forget that the *real* outcome will be one of the corners, where the bear-case loss actually lives. Hold the weighted target as what it is — a decision input — and keep the three corners always in view, because that is where your money actually goes.

## How it plays out in real markets

The three-scenario discipline is not a toy. It is how desks frame macro forks, single-name catalysts, and the re-weighting that follows a data point. Here are three real-shaped episodes.

### A single name into an earnings catalyst

Return to Meridian, but now treat it as a live trade into the Q3 print. Before Q3, your weights are the re-weighted post-Q2 set: bull 0.45, base 0.45, bear 0.10, weighted target \$122.75. The Q3 print is the *catalyst* — the dated event that resolves the driver at scale (a concept this series treats at length in the forward post on [catalysts and timing](/blog/trading/analyst-edge/catalysts-and-timing-why-cheap-can-stay-cheap-for-years)). You hold your \$20,000 position into it with a clear map: if Q3 margin confirms above 22%, you're in the bull world and the stock works toward \$140; if it prints in-line around 20%, the base holds and the stock drifts toward \$115; if it prints flat at 16%, the bear fires and you exit into \$80. The value of having drawn this in advance is that you don't have to *think* during the volatile post-print tape — you've already decided what each outcome means and what you'll do. The scenario set is your pre-committed reaction function.

This pre-commitment is worth dwelling on, because it is where scenario analysis pays off most concretely in real time. The moment a print hits, the tape is at its most violent and your judgment is at its most compromised — the stock gaps, the narrative flips, and every instinct screams to react to the price rather than the fundamentals. An analyst without a scenario map *reads the print off the price*: the stock is up 8%, so the print must have been good, so I should add. That is backwards, and it is how people buy the top of a relief rally and sell the bottom of an over-reaction. An analyst *with* a map reads the price off the print: Q3 margin came in at 22.4%, which is my bull confirmation, so I expected a move toward \$140, and an 8% pop to \$133 is *less* than my bull target implies — I hold or add, I don't fade. The map inverts the causal arrow from price→interpretation back to fundamentals→interpretation, which is the only direction that preserves edge. It also tells you when the market has *over*-reacted relative to your worlds: if a merely in-line base-case print sends the stock to \$135, that is above your base target of \$115 and you trim into strength, because the price has run past the world the print actually delivered. None of this is possible without having written the three worlds and their prices down beforehand.

### An index around a macro fork: 2022's hard-versus-soft landing

Through 2022, the entire equity market traded around a single binary driver: would the Fed's hiking cycle produce a *soft landing* (inflation falls, growth holds, the Fed pivots) or a *hard landing* (inflation falls only because a recession crushes demand)? A disciplined macro analyst in mid-2022 with the S&P near 3,900 might have framed it as three scenarios on that one driver. **Base** (soft-ish landing, prob ~0.45): inflation cools gradually, earnings hold, multiple stabilizes, index ~4,300 by year-end. **Bull** (clean soft landing, prob ~0.20): inflation falls fast, Fed signals a pause, multiple re-rates, index ~4,700. **Bear** (hard landing, prob ~0.35): inflation stays sticky, the Fed over-tightens, earnings estimates get cut, index ~3,300. The weighted target works out near 4,000 — barely above spot — which correctly captured the *truth of late 2022*: the market was roughly fairly-valued for the distribution of outcomes, with a fat and unpleasant bear tail. An analyst running this set would have sized small and watched the inflation prints as markers, exactly as the CPI surprises that autumn re-weighted the whole complex.

#### Worked example: the 2022 macro set as a \$20,000 index position

Put **\$20,000** into an S&P index proxy at the 3,900 level. Translate the three index targets into returns from 3,900: bull +20.5% (to 4,700), base +10.3% (to 4,300), bear −15.4% (to 3,300). The dollar P&L on \$20,000: bull +\$4,100, base +\$2,060, bear −\$3,080. The expected value:

> \$ EV = 0.20 × (+\$4,100) + 0.45 × (+\$2,060) + 0.35 × (−\$3,080)
> = +\$820 + \$927 − \$1,078 = **+\$669**

A +\$669 EV on \$20,000 is a +3.3% expected return — thin, with a −\$3,080 bear case carrying 35% weight. The asymmetry is *unfavorable* here: the bear is nearly as deep as the bull and far more likely than the bull. The takeaway is that the scenario math correctly said "this is a small, tactical position, not a conviction bet" — and a trader who'd skipped the bear case and anchored on the +10% base would have over-sized into one of the worst index years in a generation.

What makes the 2022 episode such a clean teaching case is that the *driver was genuinely binary and genuinely unknown*. Through that summer and autumn, every macro desk on the street was running some version of this set, and the disagreement was almost entirely about the *weights* — how much mass to put on the hard landing. Bears put the hard-landing probability at 50%+ and were positioned defensively; bulls put it at 20% and were leaning long into the bottom. The price action that year was the market's collective weights sloshing back and forth between the two tails with every inflation and labor print, which is exactly why the index was so volatile around a roughly flat full-year path in the second half: the *targets* (where each world led) were broadly agreed, but the *probabilities* lurched on every data point. An analyst who had drawn the three worlds explicitly experienced 2022 as a series of legible re-weightings; an analyst with only a point forecast experienced it as chaos. The difference between those two experiences is the entire argument for this discipline.

### A re-weight after a single CPI print

In November 2022, the October CPI report printed softer than expected — a genuine downside surprise to inflation. For the macro set above, this was a major bull marker firing: it raised the odds of the soft-landing and clean-soft-landing worlds and cut the hard-landing odds. A disciplined analyst would have re-weighted, perhaps to bull 0.30, base 0.50, bear 0.20, lifting the weighted target toward ~4,250 and the EV materially. The market did exactly this in miniature — the S&P rallied over 5% in a single session as the *whole market re-weighted its scenario set simultaneously*. The lesson, beyond the specific numbers, is that big single-day moves are usually not new information about the targets; they are the market collectively shifting probability mass between scenarios it already knew about. If you'd written your markers down, you re-weighted *with* the move instead of being shocked by it — and the size of the move scales with how *surprising* the print was relative to what was priced, which is the betas-to-surprises mechanism covered in [the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises).

## The playbook

Here is the repeatable process — the three-scenario worksheet you run on every view before you size it. Work down the columns for each of base, bull, and bear, then read off the two summary numbers at the bottom.

**1. Name the one driver.** Before anything else, identify the single variable that most determines the outcome (plant utilization, the landing scenario, a drug trial readout, a Fed pivot). If you can't name one dominant driver, your scenarios will be incoherent — go back and find it. Every scenario will be a *state* of this driver.

**2. Fill the worksheet — one row per scenario, four columns each:**

- **Driver state** — what value or condition the driver takes in this world (e.g. "utilization > 90%").
- **Target** — the price that driver state implies, derived from it, stated as a level and a magnitude (e.g. "\$140, +40%").
- **Probability** — your calibrated odds for this world, starting from base rates. The three **must sum to 1.0**.
- **Path / markers** — the observable sequence that puts you in this world, and the specific marker that would first confirm it.

**3. Pressure-test the set before you trust it.** Three quick checks:

- *Sum check*: do the probabilities sum to exactly 1.0? If not, you've left a world unnamed — find it.
- *Distinctness check*: are the three scenarios genuinely different driver states, or is your bull/bear just base ±10%? If the latter, you have one scenario, not three.
- *Discomfort check*: does the bear case make you uncomfortable? If it doesn't, it's probably too shallow. The bear should describe a real loss you'd hate to take.

**4. Collapse to the two summary numbers:**

- **Probability-weighted target** = Σ (target × probability). This is the number you compare against what's priced to find your edge.
- **Asymmetry read** = the ratio of best-case gain to worst-case loss, and which way the spread skews. A shallow bear and a deep bull is a trade worth more than its EV; a deep bear and a shallow bull is a trade worth less.

**5. Size to the bear, not the base.** The position size is governed by the *worst-case loss budget*, which lives in the bear case — never by the bull's upside.

The reason sizing keys off the bear rather than the EV is worth stating plainly, because it is the single most violated rule in the craft. Expected value tells you *whether* to take the trade; it does not tell you *how big*. A trade with a glorious +\$4,200 EV can still ruin you if you size it to the upside and the bear hits at a position so large the loss is unrecoverable — and recovery is asymmetric, so a loss that's too big doesn't just hurt the trade, it impairs the whole book's ability to compound. Sizing therefore runs off the downside: pick a loss budget you can absorb without impairing the book, look up the bear-case percentage loss, and divide. The EV and the asymmetry tell you this trade is *worth doing*; the bear case tells you *how much of it you can afford*. Conflating the two — sizing to the EV or, worse, to the bull — is how analysts with genuinely good views still blow up. A good view, sized to the bull, is just a slower way to lose all your money.

#### Worked example: sizing Meridian to the bear's loss budget

You decide that this single name may cost you at most **\$2,000** if the bear hits — that is your loss budget for the position (a budget set by your overall risk framework — [risk management: survival as a compounding engine](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine)). The bear target is \$80, a −20% move from \$100. So the maximum position is the size at which a −20% move costs \$2,000:

> max position = loss budget ÷ bear-case % loss
> = \$2,000 ÷ 0.20 = **\$10,000**

That is 100 shares, not the 200 the earlier examples assumed — the bear case just *halved* your position. The bull's +40% and the +\$2,850 weighted EV did not size the trade; the bear's −20% and your \$2,000 loss budget did. The discipline to internalize is that the bear case is not the pessimistic footnote — it is the single most operationally important scenario, because it sets the size of every position you take. A view without a bear case isn't cautious; it's unsizeable, which is where this post began, in the silence after "and if you're wrong?"

The three-scenario worksheet is the answer to that question, made permanent. Run it on every view, and you will never again stand silent in a meeting — you'll have the bear case, its probability, and its loss budget ready, which is the difference between a target and a position. The next posts in the series take the two summary numbers this worksheet produces and go deeper: [thinking in probabilities, not predictions](/blog/trading/analyst-edge/thinking-in-probabilities-not-predictions) sharpens the weights, [expected value: the only math a view really needs](/blog/trading/analyst-edge/expected-value-the-only-math-a-view-really-needs) formalizes the EV, and [from conviction to size: the bet-sizing bridge](/blog/trading/analyst-edge/from-conviction-to-size-the-bet-sizing-bridge) turns the asymmetry and the loss budget into a position weight.

## Further reading & cross-links

Within this series:

- [The analyst's edge: why forming a view is the real job](/blog/trading/analyst-edge/the-analysts-edge-why-forming-a-view-is-the-real-job) — why the meta-skill of forming a view, not the inputs, is the job.
- [Structuring a thesis: claim, evidence, and catalyst](/blog/trading/analyst-edge/structuring-a-thesis-claim-evidence-and-catalyst) — the upstream step: the structured thesis a scenario set quantifies.
- [What's priced in: the question behind every trade](/blog/trading/analyst-edge/whats-priced-in-the-question-behind-every-trade) — the benchmark you compare the weighted target against.
- [Variant perception: where real edge comes from](/blog/trading/analyst-edge/variant-perception-where-real-edge-comes-from) — the markers keep your variant view honest.
- [Reading flows and positioning](/blog/trading/analyst-edge/reading-flows-and-positioning-the-tell-most-analysts-miss) — where the positioning markers come from.
- [Thinking in probabilities, not predictions](/blog/trading/analyst-edge/thinking-in-probabilities-not-predictions) — the deeper treatment of the weights.
- [Expected value: the only math a view really needs](/blog/trading/analyst-edge/expected-value-the-only-math-a-view-really-needs) — formalizes the probability-weighted EV.
- [Catalysts and timing: why cheap can stay cheap for years](/blog/trading/analyst-edge/catalysts-and-timing-why-cheap-can-stay-cheap-for-years) — the dated event that resolves the driver.
- [From conviction to size: the bet-sizing bridge](/blog/trading/analyst-edge/from-conviction-to-size-the-bet-sizing-bridge) — turning the asymmetry and loss budget into a position weight.

Out to the rest of the blog:

- [The surprise, not the level: betas to data surprises](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) — why the size of a re-weight scales with how surprising the print was versus what's priced.
- [The asymmetry of losses: why a 50% loss needs a 100% gain](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) — the convexity that makes the bear case the load-bearing scenario.
- [How an options market maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) — the same skew you read off your scenario set, priced by the other side.
- [Risk management: survival as a compounding engine](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine) — where the loss budget that sizes the position comes from.

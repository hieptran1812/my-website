---
title: "Stress-Testing Your Thesis with a Pre-Mortem"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Before you put a trade on, kill it in your head: run a pre-mortem that assumes the trade already failed and asks why, steelman the bear case, find the single thing most likely to break the thesis, and turn those findings into invalidation levels and a smaller, safer size."
tags: ["analysis", "market-view", "pre-mortem", "risk", "confirmation-bias", "steelman", "base-rates", "invalidation", "position-sizing", "decision-process", "overconfidence", "single-point-of-failure"]
category: "trading"
subcategory: "The Analyst's Edge"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Before you put a trade on, run a **pre-mortem**: assume the trade already failed, list every way it could have died, rank those causes by probability times dollar impact, steelman the bear, isolate the single point of failure, and convert that into an invalidation level and a smaller size. The risks confirmation bias hides are the ones that take your money.
>
> - "I've considered the risks" is not a pre-mortem. A pre-mortem is the specific act of *assuming failure as a fact* and reasoning backward — that prospective-hindsight framing surfaces failure modes your forward-looking optimism never names.
> - Steelman the bear: argue the other side better than a real bear could. The size that survives the strongest possible counter-case is the honest size; everything above it is conviction you haven't earned.
> - The **outside view** beats the inside view. Your felt 90% is almost always the base rate plus overconfidence; correcting a 90% to a 65% can flip a trade from "obvious" to "barely worth it."
> - The one rule to remember: **find the single thing most likely to break the thesis, turn it into a written invalidation level, and size so that being wrong about it costs you a survivable amount.**

## The trade he could have killed on day one

It is mid-September and a portfolio manager — call him Daniel — is as certain as he has ever been about a name. NorthStar, a mid-cap industrial, is going to re-rate hard into its Q3 print. The story is clean: a new plant in Monterrey is ramping, the segment margin is structurally higher than the 18% the Street still carries in its models, and when that margin first prints in late October, analysts will have to raise estimates and the stock will follow. He has done the work. He has the supplier purchase-order data. He has management's "full utilization by Q3" guidance. He builds a \$20,000 position — his full intended size for a high-conviction idea — around \$50 a share, targeting \$70.

Six weeks later the trade is down 30%. Not because the thesis was crazy. Because one assumption he never stress-tested quietly broke. NorthStar's single largest customer — a fact buried in the 10-K's concentration disclosure that Daniel had skimmed and dismissed — cut its order book in early October on its own demand weakness. The plant that was supposed to run at full utilization ran at 70%. The margin ramp that was the whole thesis slipped a quarter. The stock didn't wait for the print; it leaked lower as the order cut became visible in channel data, and Daniel, anchored to his \$70 target, added on the way down. By the time the Q3 margin printed flat, he was out \$6,000 he had never expected to lose on this name.

Here is the uncomfortable part: every single piece of what went wrong was knowable on the day he put the trade on. The customer concentration was in the filing. The dependence of the entire thesis on one operational assumption — the ramp — was structural and obvious the moment you wrote the thesis down. A disciplined exercise that *assumed the trade had already failed and asked why* would have surfaced the order-cut scenario, ranked it as the single most dangerous failure mode, and either cut his size or set a stop that got him out at \$45 instead of riding it to \$35. He didn't do that exercise. He did the opposite: he gathered the evidence that confirmed his view and waved off the evidence that didn't. This post is about the exercise he skipped.

![The pre-mortem turns a thesis into a sized, falsifiable position in seven stages](/imgs/blogs/stress-testing-your-thesis-with-a-pre-mortem-1.png)

A note on where this sits. Earlier in this series we built the thesis itself — [structuring a thesis: claim, evidence, and catalyst](/blog/trading/analyst-edge/structuring-a-thesis-claim-evidence-and-catalyst) — and we found the gap between what's priced and what you believe — [variant perception: where real edge comes from](/blog/trading/analyst-edge/variant-perception-where-real-edge-comes-from). Now you have a thesis. The pre-mortem is the step that comes *before you size it*: the deliberate act of trying to kill your own idea while it is still cheap to kill, so the market doesn't kill it later at full price.

## Foundations: the pre-mortem, the steelman, and the two views

Before we run anything, we need four terms defined precisely, because the entire method rests on them and each is routinely misused. A pre-mortem done with the wrong definition is just a worry list with a fancy name.

### The pre-mortem

A **pre-mortem** is a technique introduced by the psychologist Gary Klein. A *post*-mortem examines a failure after it has happened to learn from it. A pre-mortem runs the same examination *before* the decision is final, by assuming the failure as a settled fact. You do not ask "what could go wrong?" — that is a generic risk question your optimism easily deflects. You assert: "It is six weeks from now. This trade is a disaster. It is down 30%. I am writing the explanation of why." Then you write the explanation.

The reason this specific framing matters — and why it is not just "thinking about risks" with extra steps — is a cognitive effect Klein's collaborators documented called **prospective hindsight**. When people are asked to explain a future event as though it had already occurred, they generate roughly 30% more concrete, specific reasons than when asked whether the event *might* occur. The grammar does the work. "Why might this fail?" invites you to defend your idea. "It failed; explain why" forces you to prosecute it. The same brain that cannot find the holes in a thesis it loves can find them effortlessly the moment failure is stipulated as already true.

Why does the grammar matter so much? Because "might it fail?" is a *yes/no* question, and a committed advocate answers it "probably not" and stops thinking. "It failed; why?" is an *open* question that presupposes the failure, so the brain has no escape hatch into "probably not" — it must produce content, and the content it produces is the failure modes. The presupposition is the trick. It's the same reason a leading question ("when did you stop cutting corners on this thesis?") extracts more than a neutral one ("did you cut any corners?"). You are leading yourself, on purpose, toward the answers your optimism would otherwise suppress. This is why the pre-mortem outperforms a generic risk checklist: a checklist asks you to *evaluate* a fixed list of dangers, which your bias does badly; the pre-mortem asks you to *generate* the list from a stipulated failure, which your bias, redirected, does well.

There is a second reason the pre-mortem works in a *team* setting that is worth knowing even if you trade alone. In a normal "any concerns?" meeting, dissent is socially expensive — voicing a risk after the PM has championed the trade reads as disloyalty or doubt about the PM's judgment, so the people who see the problem stay quiet. The pre-mortem inverts the social incentive: once everyone is asked to explain a *stipulated* failure, naming a risk is no longer dissent, it's just playing the assigned game. The quiet skeptic who'd never have raised their hand will now happily explain exactly how the trade blew up, because the framing gave them permission. If you run a book with analysts under you, this alone is worth the ten minutes.

### The steelman

A **steelman** is the opposite of a strawman. A strawman is the weakest version of an opposing argument, built so you can knock it down and feel right. A steelman is the *strongest* version of the opposing argument — stronger, ideally, than any actual opponent has bothered to make — constructed by you, on purpose, so that if you can still hold your view after facing it, your conviction is earned. For a trade, the opposing argument is the bear case (if you're long) or the bull case (if you're short). To steelman it you ask: *what does the smartest possible bear know that I am ignoring or rationalizing away?* Not the dumb bear's argument. The one that would worry you if a person you respected made it over dinner.

### The inside view and the outside view

This pair, from Daniel Kahneman and Amos Tversky, is the hinge of the whole post. The **inside view** is the forecast you build from the specifics of *this particular* case: NorthStar's plant, its guidance, its margin path, your read of the catalyst. It feels rigorous because it is detailed. The **outside view** ignores the specifics and asks instead: *how often do cases in this reference class actually work out?* How often does a "high-conviction catalyst re-rating into an earnings print" trade actually pay, across all the times you and people like you have put one on? The outside view is a **base rate** — the unconditional frequency of an outcome in a class of similar events.

The deep finding is that the inside view is systematically overconfident and the outside view is systematically more accurate, because the inside view's wealth of specific detail crowds out the one number that matters most: the historical hit rate of trades that felt exactly this good. Every failed trade also had a detailed, compelling inside view at the time. The base rate already prices that in.

The classic illustration is the planning fallacy: ask people how long their own project will take and they give you an optimistic number built from the specifics ("the model trains in two days, then a week to write up"); ask how long *similar* projects have actually taken and the honest number is two to three times longer. The detail of your own plan feels like information, but it is mostly a story, and stories systematically omit the friction, the delays, and the unforeseen that the base rate has already absorbed. Trading has its own version: "this catalyst is so clean, the gap is so obvious" is the inside view, and it feels like an edge, but the right prior is "catalyst trades like this one pay X% of the time," and X is lower than the cleanliness of any single story suggests.

The hard part is that the outside view feels *lazy* and the inside view feels *rigorous* — exactly backwards from their accuracy. Treating a brilliant, hand-built thesis as "just another catalyst trade with a 60% base rate" feels like throwing away your work. It isn't. The base rate is the foundation; your specific analysis is a *small, earned adjustment* on top of it. The error is never "I used the base rate"; the error is always "I started from my felt conviction and never went looking for the reference class at all."

### Why you cannot see your own thesis's weaknesses

The reason all of this requires a *deliberate ritual* rather than just being careful is **confirmation bias**: the well-documented tendency to seek, notice, weight, and remember evidence that supports a belief you already hold, while overlooking or discounting evidence that contradicts it. Once Daniel decided NorthStar was a buy, his information-gathering quietly reorganized itself around that conclusion. The supplier purchase orders (confirming) felt vivid and important. The customer-concentration disclosure (disconfirming) felt like boilerplate and got skimmed. He wasn't being dishonest — confirmation bias operates below awareness, which is exactly why it's dangerous. You cannot will yourself to be unbiased about a thesis you love, any more than you can tickle yourself. The pre-mortem works because it does not rely on you being unbiased; it changes the *question* so that your bias is pointed at finding holes instead of defending the view. (The deeper treatment of falling in love with a thesis gets its own post later in this series — [confirmation bias and the thesis you fall in love with](/blog/trading/analyst-edge/confirmation-bias-and-the-thesis-you-fall-in-love-with).)

With those four terms in hand, we can build the actual process.

## Running the pre-mortem

The pre-mortem is a fixed sequence. The discipline is in doing all of it, in order, every time — because the steps you skip are the ones your conviction wants you to skip.

### Step 1 — Assume failure as a fact

Write the sentence, literally, before anything else: *"It is [date six weeks out]. This trade is down 30%. Here is the full story of why."* The date and the loss number are not optional flavor; they are what triggers the prospective-hindsight effect. A vague "let me think about the downside" does not move your brain into explanation mode. A stipulated, dated, specific disaster does. Some analysts make it visceral on purpose — they write the email they'd have to send to their boss explaining the loss. The more real the failure feels, the more failure modes surface.

The output of step 1 is not a list yet. It is a posture. You have stopped being the trade's advocate and become its coroner.

Two details make the posture stick. First, set the loss figure *above* your stop, not at it. If you only stipulate "it hit my stop and I lost \$2,500," your brain explains a clean, expected outcome and generates little. Stipulating "it's down 30%, far past where I thought I'd be out" forces you to explain how your stop *failed* — how you talked yourself out of it, how a gap jumped it, how you added on the way down — and those mechanisms (the ways your discipline breaks, not just the ways the thesis breaks) are exactly what kills real trades. The biggest losses almost never come from the thesis being wrong; they come from the thesis being wrong *and* the trader refusing to act on the invalidation. The pre-mortem should rehearse both failures.

Second, do it before you're committed, and do it in writing. A pre-mortem run in your head, after you've already decided to put the trade on, is theater — you'll generate two token risks and move on. The exercise has to happen at the moment of *maximum reversibility*, before the position exists, when killing the idea costs nothing but a few minutes. Once the trade is on, sunk-cost reasoning and ego make an honest pre-mortem nearly impossible. Write it down, too, because writing is slower than thinking and slowness is the point: it's much harder to wave off "the customer cuts orders and the plant runs at 70%" once it's a sentence on the page than while it's a half-formed thought you can let evaporate.

### Step 2 — List every way it failed

Now, from that posture, write every distinct cause of the failure you can generate. Distinct is the operative word — you want *independent* failure modes, not ten restatements of "the market went down." For Daniel's NorthStar long, a good list looks like this:

- The plant ramp slipped a quarter (operational execution).
- A major customer cut orders, so the plant ran below full utilization regardless of execution (demand).
- Consensus already carried the higher margin, so the print was a non-event (no variant perception left — the gap was already closed).
- The whole industrial sector de-rated on a macro shock unrelated to NorthStar.
- A fraud or accounting restatement (low probability, high impact tail).
- The thesis was simply too early — right, but the catalyst slipped past the horizon and he got bored or stopped out before it paid.

Six distinct, independent ways to lose. Notice that some of these have *nothing to do with whether the thesis is correct*: a sector de-rate or a "too early" outcome can sink a trade whose central claim was right. That is the point of listing them — your inside view only protects you against being wrong about the thing you focused on.

A useful structure for generating a *complete* list is to walk the four families of failure, because most people generate two of them and miss the other two:

- **The thesis is wrong.** The core claim is simply false — the ramp doesn't ramp, the product flops, the merger breaks. This is the family analysts list first and most completely, because it's the one they were already thinking about.
- **The thesis is right but already priced.** You were correct about the fact and made no money because the market knew it too. The variant perception you thought you had wasn't there. This family is invisible from the inside view, because your work *feels* original even when the desk across the street did the same work.
- **The thesis is right but the timing is wrong.** Correct call, wrong clock — the catalyst slips past your horizon, you get stopped out or bored or margin-called before the thesis pays. "Right but early" is indistinguishable from "wrong" on your P&L.
- **Something outside the thesis breaks it.** A macro shock, a sector de-rate, a liquidity event, a position-crowding unwind — none of which has anything to do with your specific company. Your beautifully-reasoned single-name thesis gets run over by something you weren't even analyzing.

Walking all four families is the difference between a list of three obvious things and a list of six to eight real ones, and it's where the prospective-hindsight lift earns its keep. The third and fourth families — already-priced and externally-broken — are precisely the ones confirmation bias hides, because they don't engage your thesis at all; they sit in your blind spot by construction. (The "already priced" family is so central to the craft that it has its own treatment — the question of what the market already believes is the spine of the whole series.)

### Step 3 — Rank by probability times impact

A raw list is not actionable, because not all failures are equal and you have a finite budget of attention, hedging, and size reduction. Rank each cause by **probability times impact**: how likely is it, and if it happens, how much does it cost in dollars? This is the same expected-value logic that governs the whole craft, applied to the downside. A high-probability, low-impact cause (a small earnings miss that dings you 5%) is a manageable annoyance. A low-probability, high-impact cause (fraud) is a tail you note and largely accept. The dangerous quadrant is **high probability and high impact** — and that is exactly where Daniel's customer-concentration risk lived, and exactly where his confirmation bias had filed it under "boilerplate."

![Rank pre-mortem causes by probability times impact, not by how scary they feel](/imgs/blogs/stress-testing-your-thesis-with-a-pre-mortem-2.png)

The matrix sorts your worries into four actions. Bottom-left (low probability, low impact): ignore it; spending attention here is theater. Top-left (high probability, low impact): watch it, but it won't break you. Bottom-right (low probability, high impact): note it, accept it, and make sure no single one of these can ruin you — this is the tail-risk corner. Top-right (high probability, high impact): this is the kill zone, and it is where every dollar of your risk-reduction budget should go first. The single most common pre-mortem failure is ranking by how *scary* a cause feels rather than by probability times impact — fraud feels terrifying and gets all the attention, while the boring, likely, expensive failure mode gets waved through.

The probability estimates don't have to be precise — you are not pretending to a false decimal-point accuracy. The matrix works even with crude buckets (low ≈ under 15%, medium ≈ 15–40%, high ≈ over 40%) because its job is to *sort*, not to measure. What matters is the relative ranking, and the relative ranking is usually robust to large errors in the individual numbers. If the customer-cut risk is "medium-to-high probability, high impact" and the fraud risk is "low probability, high impact," you don't need to know whether the customer cut is 25% or 35% to know it ranks above the fraud — and that ordering is what tells you where to spend your attention.

One subtlety that trips people: the impact you score is the impact *to your position*, in dollars, not the impact to the company or the world. A cause that would be catastrophic for NorthStar's long-term value but resolves over three years is *low* impact to a four-month catalyst trade, because you'll be out long before it bites. Conversely, a cause that's a minor blip for the business — a one-quarter inventory build — can be high impact to *your* trade if it lands right on your catalyst date and spooks the print. Score the matrix from your position's clock and your position's dollars, not from a generic view of how bad the news is.

#### Worked example: the pre-mortem surfaces a \$6,000 downside not in the base case

Daniel's base case sized the trade off a symmetric-feeling bet: \$20,000 position at \$50, target \$70 (+40%, +\$8,000), and a vague mental stop "if it breaks." He had never quantified the *specific* downside of his top-ranked failure mode. The pre-mortem forces it.

The top-ranked cause is the customer order cut, which (he now estimates honestly) has maybe a 25% probability — this customer is large, the end-market is softening, and the dependence is structural. If it happens, the plant runs at ~70% not 100%, the margin ramp slips, and the stock — which the market will mark down as soon as channel data leaks the order cut — likely trades to around \$35 before the print even arrives. That is a −30% move: −\$6,000 on the \$20,000 position.

So the pre-mortem has converted a worry into a number. The trade is not "+\$8,000 if right, some-vague-loss if wrong." It is: **+\$8,000 with ~60% probability if the ramp holds, versus a specific −\$6,000 with ~25% probability from the single highest-ranked failure he had previously filed as boilerplate.** Run the rough expected value: (0.60 × \$8,000) − (0.25 × \$6,000) − (0.15 × \$2,000 for the assorted smaller failures) ≈ \$4,800 − \$1,500 − \$300 = +\$3,000. Still positive, but a long way from the "obvious layup" the inside view sold him. **The pre-mortem didn't kill the trade — it priced the \$6,000 downside that the base case had quietly left out, which is the difference between a sized bet and a hope.**

### Step 4 — Steelman the bear

Ranking your own failure list is necessary but not sufficient, because your list is still *your* list — confirmation bias shaped which causes occurred to you in the first place. Step 4 imports an adversary. Construct the strongest possible bear case: not the lazy short-seller's tweet, but the argument a sharp analyst who is short the name would make if they were trying to change your mind and they were good at it.

The test for a real steelman is whether it makes you uncomfortable. If your "bear case" is a set of points you can immediately dismiss, you've built a strawman and learned nothing. A real steelman for NorthStar might be: *"Your entire thesis is one operational assumption — the ramp — and ramps slip more often than they hit; you're paying full price for execution risk you're calling a sure thing. The customer concentration means you don't even control the variable that matters. And the 'consensus carries the old margin' edge is thinner than you think, because the smart money on the desk has read the same purchase-order data you have."* That argument should sting, because parts of it are right.

![Steelmanning the bear case shrinks an overconfident position to a survivable one](/imgs/blogs/stress-testing-your-thesis-with-a-pre-mortem-3.png)

What you do with a steelman is not necessarily abandon the trade. It is to let the strongest counter-case set your size. If, after honestly facing the best bear argument, you still believe the thesis but you now see that two of the bear's points are genuinely uncertain, your conviction should fall — and size follows conviction. The size that survives the steelman is the honest size. Everything above it was conviction you hadn't earned because you'd never argued the other side properly.

There is a practical technique for building a steelman that's hard to cheat: go find the actual bear. Read the short report, the bearish sell-side note, the skeptical thread — written by someone with money on the other side, who has every incentive to find the holes you missed. Then take their best point and make it *better* than they did. The short-seller has done a chunk of your disconfirming research for free; your job is to refuse the comfort of dismissing them as "just talking their book" (everyone is always talking their book, including you) and instead ask what they'd have to be *right* about for you to lose. If you can't find a real bear, that's itself a signal — either the trade is genuinely uncontested (rare, and worth interrogating: why does no one disagree?), or you haven't looked hard enough.

A second test separates a real steelman from a fake one: after you've built the bear case, can you state the *specific observation* that would make it correct? "The customer cuts orders, which shows up as channel weakness in the next four weeks" is a steelman with a checkable claim. "Sentiment could turn against the name" is not — it's a vague gesture you can never be proven wrong about, which means it can never cut your size in a disciplined way. A steelman that can't be turned into an observable is a steelman you'll rationalize away the moment the trade gets uncomfortable.

#### Worked example: a steelman cuts size from \$20,000 to \$12,000

Before the steelman, Daniel's intended size was \$20,000 — his standard high-conviction allocation, justified by a felt conviction around 85%. He runs the steelman above and two points land. First, the thesis really does rest on a single operational assumption (the ramp), and he cannot honestly claim ramps hit on schedule 85% of the time — across the industrials he's followed, maybe 60% do. Second, the customer concentration means the variable he most needs (utilization) is partly out of the company's hands.

Honest conviction, post-steelman, drops from 85% to about 60%. He sizes proportionally to conviction (a simplification of the conviction-to-size mapping the series covers elsewhere): roughly, intended-risk × conviction-adjustment. His full-size \$20,000 was calibrated for ~85% conviction; at 60% the proportional size is \$20,000 × (60/85) ≈ \$14,000, and he rounds down to **\$12,000** to leave a margin for the fact that he's probably *still* a little overconfident. The trade goes on at \$12,000, not \$20,000.

Now look at what that did to the downside. The same −30% customer-cut scenario costs −\$3,600 on a \$12,000 position instead of −\$6,000 on \$20,000. **The steelman didn't change the thesis or the catalyst — it cut the size by 40%, which cut the worst-case loss by \$2,400, purchased entirely with the discomfort of arguing the other side honestly.** The thesis you can defend against the best bear is the one you size to.

### Step 5 — Find the single point of failure

A thesis is a chain of assumptions, and a chain breaks at its weakest link. The **single point of failure** (SPOF) is the one assumption whose failure *alone* invalidates the entire thesis, no matter what else is true. Most assumptions in a thesis are robust — they can wobble and the thesis survives. A few are load-bearing. Exactly one is usually *the* load-bearing one, and naming it is the highest-leverage thing the pre-mortem produces, because it tells you precisely what to monitor and where to put your stop.

![The single point of failure is the one assumption that alone kills the whole thesis](/imgs/blogs/stress-testing-your-thesis-with-a-pre-mortem-4.png)

Decompose the thesis into its assumptions and ask of each: *if this one thing is false and everything else is true, is the thesis dead?* For NorthStar: if demand softens but the plant still ramps and consensus still carries the old margin, the thesis survives, just delayed — not a SPOF. If consensus turns out to already carry the new margin but the ramp hits and demand holds, you make less but the direction is intact — not a SPOF. But if **the ramp slips** — if the Q3 margin prints flat — then the catalyst is a non-event, the variant perception evaporates, and the thesis is simply dead, regardless of demand or consensus positioning. The ramp is the SPOF.

Finding the SPOF is clarifying in a way nothing else in the process is, because it collapses a fuzzy cloud of risks into a single, observable, monitorable thing. You are no longer worried about "the trade going wrong." You are watching one variable: does the Q3 segment margin print at or above the level the ramp implies? That is a yes/no question with a date attached, which means it can become an invalidation level and a stop.

A trap to avoid: do not confuse the SPOF with the *most painful* failure or the *most likely* failure. The SPOF is the failure that, on its own, ends the thesis. The most likely failure (a small earnings wobble) might only delay it; the most painful failure (fraud) might be a tail you've already accepted. The SPOF is defined by *sufficiency*, not by probability or pain: is this one thing, by itself, enough to kill the thesis? A useful way to test a candidate is to assume it's false and everything else goes your way — if you'd still hold the trade, it wasn't the SPOF; if the trade is dead, it is.

Some theses have more than one SPOF, and that is a warning sign, not a richer thesis. If three different independent assumptions each, alone, would kill the trade, then the trade's survival requires all three to hold, and the probability of that is the *product* of their individual probabilities — three 80% assumptions chained give you 0.8 × 0.8 × 0.8 ≈ 51%, a coin flip, even though each link felt safe. The pre-mortem catches this multiplication that the inside view hides: a thesis that "feels 80% likely" but secretly depends on three independent 80% links is really a 50/50, and should be sized like one or restructured to remove the fragile links.

#### Worked example: the SPOF becomes a \$2,500 stop

The SPOF is the ramp, and the observable that resolves it is the Q3 segment margin. But Daniel can't only wait for the print — the stock will move on leaked channel data before late October, and a SPOF you can only check at the very end gives the loss room to run. So he translates the SPOF into a *price-based* tripwire that proxies it: the level below which the market is clearly pricing a ramp failure.

He reasons: the thesis is built off \$50 with a \$45 invalidation (the level below which the bullish read is broken on the chart and on the channel data he's watching). The distance from entry to invalidation is \$50 − \$45 = \$5 a share, or 10%. He wants the *dollar* loss at that stop to be a survivable, pre-decided number — he chooses \$2,500, which on his book is about 1% of capital and well inside his per-trade risk budget. Working backward: to lose only \$2,500 at a 10% adverse move, the position must be \$2,500 / 0.10 = \$25,000 of notional... but his steelmanned size is \$12,000, at which a 10% move is only −\$1,200. So the \$2,500 isn't the constraint — the steelman already made him smaller than his risk budget required. He keeps the \$12,000 size and sets the hard stop at \$45, knowing the realized risk (\$1,200) sits comfortably under his \$2,500 cap.

**The SPOF gave him a single thing to watch (Q3 margin / the \$45 proxy), the invalidation level gave him an exit that doesn't depend on his in-the-moment judgment, and the dollar cap confirmed the trade fits his book — so the worst case is a known, survivable \$1,200 to \$2,500, not the open-ended −\$6,000 he originally risked.** A loss you defined before the trade is a cost of doing business; a loss you discover after the fact is a wound.

### Step 6 — The outside view: check your estimate against the base rate

The last analytical step is the one that humbles every analyst, which is why it's easiest to skip. You have an inside-view conviction — say 85%, or now 60% after the steelman. The outside view asks: across *all* the trades you've ever put on that felt like this one — high-conviction catalyst plays into an earnings print — how many actually paid? That historical hit rate is your base rate, and it is almost always lower than your felt conviction, because every losing trade also felt convincing at the time.

![Your estimate versus the base rate: the overconfidence the outside view corrects](/imgs/blogs/stress-testing-your-thesis-with-a-pre-mortem-5.png)

The chart shows the pattern across setup types: the felt estimate (inside view, blue) sits well above the historical base rate (outside view, amber) for every class of trade. The gap is overconfidence, and it is remarkably stable. The discipline is to anchor on the base rate and then make *small, defensible* adjustments for what is genuinely special about this case — not to start from your felt number and adjust down grudgingly. Start outside, adjust inside, sparingly.

#### Worked example: a base-rate adjustment drops a "90%" to 65% and rewrites the EV

Suppose Daniel, in a moment of full confidence before the pre-mortem, would have put his probability of the thesis paying at 90%. He pulls his own trade journal — the outside view in its purest form — and finds that across the last 30 catalyst-re-rating trades he's taken, 20 worked. His personal base rate for this class is 20/30 ≈ 67%, call it 65%. Even granting that NorthStar's edge (the proprietary purchase-order data) is real and special, the honest number is the base rate plus a *small* premium, not his felt 90%. He settles on 65%.

Watch what that does to the expected value, holding the \$12,000 steelmanned size, +40% target (+\$4,800) and the −30% SPOF downside (−\$3,600):

- At the felt **90%**: EV = (0.90 × \$4,800) − (0.10 × \$3,600) = \$4,320 − \$360 = **+\$3,960**. A near-certain winner.
- At the base-rate-anchored **65%**: EV = (0.65 × \$4,800) − (0.35 × \$3,600) = \$3,120 − \$1,260 = **+\$1,860**.

The trade is still positive-EV at 65% — it survives the outside view, which is genuinely good news and a reason to keep it. But the EV is less than half what the inside view promised, and the probability of *losing* is 35%, not 10%. **The base-rate adjustment didn't change a single fact about NorthStar; it corrected a 25-point overconfidence error that would have led Daniel to over-size and to be blindsided by a loss he'd mentally assigned a 10% chance when the real chance was 35%.** Calibrate your conviction to your own history, not your current enthusiasm.

### Step 7 — Actively search for disconfirming evidence

The pre-mortem isn't only a pre-trade ritual; it sets a *standing instruction* for the life of the trade. Confirmation bias doesn't stop once you're in — it gets worse, because now you're emotionally and financially committed. So the final step is to commit, in writing, to actively seek disconfirming evidence on a schedule. For the NorthStar long, that means: each week, deliberately look for signs the ramp is slipping or the customer is cutting, and weight a piece of disconfirming data *more* heavily than a piece of confirming data, precisely because your bias under-weights it. The decision journal entry reads: "I will exit if Q3 margin prints below 19%, if peers fall 15% on no NorthStar news, or if I see two consecutive weeks of weakening channel data — and I will look for those things on purpose." The trader who only notices confirming data discovers the disconfirming data when it's already in the price.

The deeper move here is to flip the *burden of proof* once you're in the trade. Naturally, your bias treats the thesis as innocent until proven guilty: you demand overwhelming evidence to exit, while accepting flimsy evidence to stay. Invert it deliberately. Treat the thesis as guilty until proven innocent — make the trade re-earn its place in the book each week against the disconfirming data you went looking for. A single solid piece of disconfirming evidence (the customer's own guidance softens) should outweigh three pieces of confirming evidence (the stock ticked up, a bull note came out, the chart looks good), because you already over-weight the confirming ones for free.

#### Worked example: weighting disconfirming evidence with a Bayesian nudge

Three weeks into the NorthStar long, Daniel gets one disconfirming data point: the key customer's own quarterly guidance comes in soft, hinting at order cuts. His confirmation bias wants to file it under "macro noise, not specific to us" and move on. The pre-mortem's standing instruction forces him to price it instead.

Before the data point, his pre-mortem probability that the ramp holds (the thesis pays) sat at 65%. The customer-cut scenario is exactly his top-ranked SPOF-adjacent risk, and this is a direct read on it. He doesn't need formal Bayes to do the update honestly: a soft print from *the* customer he was worried about is strong evidence in the failure direction. He cuts his "ramp holds" estimate from 65% to roughly 45% — now below a coin flip. Re-run the EV on the \$12,000 position (+\$4,800 if right, −\$3,600 from the SPOF downside): at 45%, EV = (0.45 × \$4,800) − (0.55 × \$3,600) = \$2,160 − \$1,980 = **+\$180** — essentially zero. The trade no longer earns its risk. He cuts the position in half to \$6,000 on the data point alone, locking in a far smaller loss than the −\$3,600 he'd face if he waited for the print to confirm what the customer's guidance already told him. **The disconfirming data point didn't prove the thesis dead — it dropped the EV from +\$1,860 to roughly breakeven, which is exactly the signal to take risk off before the crowd reads the same guidance and the gap closes against you.** A trade that no longer clears its own EV bar is a trade you shrink, whatever your original conviction was.

## Common misconceptions

The pre-mortem is simple to describe and easy to fake. These are the ways analysts convince themselves they've done it when they haven't.

### "I've considered the risks"

This is the big one. *Considering* risks is a passive, forward-looking, optimism-friendly activity that confirmation bias sails right through. "Sure, the ramp could slip, but I think it'll be fine" is considering a risk and then dismissing it in the same breath. A pre-mortem is structurally different: it stipulates the failure as *already true* and forces a written explanation, which is what triggers prospective hindsight and the 30% lift in specific failure modes. Vague consideration produces a vague list you ignore. A real pre-mortem produces a ranked list, a named SPOF, and a number. If you didn't write "it's down 30%, here's why" and then write the ranked causes, you considered the risks — you did not run a pre-mortem.

### "A stop-loss replaces the pre-mortem"

A stop-loss is the *output* of a good pre-mortem, not a substitute for one. A stop placed without a pre-mortem is placed at a level chosen by chart aesthetics or by how much you're willing to lose — neither of which has anything to do with what would actually prove your thesis wrong. The whole value of the SPOF step is that it tells you *where* the invalidation level belongs: at the price (or the data point) that signals the load-bearing assumption has broken. A stop at \$45 because "that's the 10% I'll risk" is arbitrary; a stop at \$45 because "below there, the market is clearly pricing a ramp failure, which is my SPOF" is a stop with a thesis behind it. The pre-mortem makes the stop meaningful, and the stop is only one of its outputs — the others (size, the watch-list of disconfirming data, the decision to take the trade smaller or not at all) the stop can't replace.

There's also a failure mode unique to thinking the stop is enough: the *data-point* invalidation that a price stop can't capture. Some theses die not at a price but at an event — a margin print, a guidance update, a regulatory ruling — and the stock can sit perfectly still right up until that event detonates it. A price stop never triggers because the price never moved; then the event hits and you gap straight through it. The pre-mortem's SPOF step is what catches this: it tells you the invalidation is the Q3 margin number, not a chart level, so you write "exit if margin prints below 19%" *as well as* the price stop. A trader who relies on the stop alone is protected against the slow bleed and defenseless against the cliff, and the cliffs are where the account-ending losses live.

### "The bear case is for bears"

The belief that you only need to understand the other side if you're *on* the other side is precisely backwards. The bear case is most valuable to the bull, because it's the map of how the bull loses money. The smartest bear isn't your enemy; they're a free research analyst who's already done the work of finding your thesis's weak points. Refusing to steelman the bear because "I'm long, why would I argue the short?" is refusing the single cheapest source of disconfirming evidence available. The market is, at root, the other side of your trade — and there's a whole discipline in understanding who's on it ([how an options market maker thinks: the other side of your trade](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade)). The bear case is your trade's other side made explicit.

### "My edge means I'm probably right"

Edge is real and it's the reason to take the trade at all — but edge is a *small* tilt on top of a base rate, not a license to ignore the base rate. A genuine informational edge might move your hit rate from a 55% base rate to 62%; it does not move it to 90%. The analyst who reasons "I have an edge, therefore I'm probably right" is double-counting: they're using the edge to justify a felt conviction that already exceeds what the edge could possibly deliver. The outside view is the corrective. Anchor on the base rate, add a defensible premium for your specific edge, and notice how rarely that premium is more than 5 to 10 points. The edge is the reason the trade is positive-EV; it is not the reason you can skip the pre-mortem.

### "A pre-mortem is just being negative"

A pre-mortem is the opposite of pessimism — it's how confident people stay in the game long enough to compound. Pessimism says "this might fail, so don't do it." A pre-mortem says "this might fail in these specific, ranked ways, so do it *this size* with *this stop* and *this watch-list*." It's a tool for taking risk intelligently, not for avoiding it. The analyst who skips it isn't braver; they're just blind to a downside they'll meet later at a worse price.

## How it plays out in real markets

The method lands harder against real episodes where the failure mode was nameable in advance, and the people who named it survived while the people who didn't got hurt.

### A pre-mortem surfacing a downside not in the base case — the 2018 short-vol blowup

Through 2017 and into early 2018, a popular trade was being short volatility — selling VIX futures or holding inverse-VIX products like XIV — collecting steady premium as realized volatility stayed historically low. The inside view was seductive: vol is mean-reverting, it's been low for a year, the carry is excellent. The base case for most of these traders was "small steady gains, occasional small drawdown." What a pre-mortem would have stipulated — "it's some morning and this position is down 90% overnight; why?" — was the structural SPOF that the base case omitted: these products had to *buy* VIX futures to rebalance after a spike, and a large enough single-day spike would force buying that fed on itself. On February 5, 2018, the VIX more than doubled in a day, XIV lost about 90% of its value after hours, and the product was liquidated. The loss wasn't in anyone's base case because the base case was built from the inside view — "vol stays low" — rather than from the stipulated failure. The few who'd run the pre-mortem and named the rebalancing SPOF had either sized tiny or weren't in it. The disaster was nameable; most just hadn't named it.

### A steelman that cut size — the 2020 COVID drawdown

In February 2020, plenty of analysts saw the early COVID news and concluded the market would shrug it off, as it had with SARS, Ebola, and other scares — a clean inside-view analogy. The steelman of *that* view was available and uncomfortable: "this one is different because it's already spreading uncontained in multiple countries, the response will be economic shutdowns not just travel bans, and the analogy to prior contained outbreaks is exactly the kind of base-rate-feeling reasoning that breaks when the case is genuinely novel." An analyst who steelmanned the bear didn't have to call the crash perfectly to benefit; they just had to let the strongest counter-case cut their long exposure from full size to half. When the S&P fell roughly 34% in five weeks, the steelman-cut position lost half as much as the full-conviction one — and the analyst who'd cut had dry powder to redeploy near the March bottom. The steelman didn't have to be *right* to be worth it; it had to be strong enough to right-size the bet.

### A base-rate adjustment to overconfidence — the perennial earnings-beat trade

Every quarter, analysts build high-conviction theses that a company will beat earnings and the stock will pop, and every quarter a chunk of them are blindsided by a "good print, stock down" reaction. The inside view says "numbers will be great, so the stock goes up" — and is frequently right about the numbers and wrong about the stock. The base rate that the outside view supplies is sobering: even when a company beats, the stock falls a meaningful fraction of the time, because the beat was already priced in (the variant perception was gone), or guidance disappointed, or positioning was offside. An analyst anchored to "I'm 90% sure they beat *and* the stock rises" who checks the base rate — "how often does my 'beat-and-rally' call actually rally?" — and finds it's closer to 60%, sizes the trade for a coin-flip-plus, not a sure thing, and isn't wiped out by the good-print-stock-down outcome. The mechanism of why the same number moves prices differently is its own subject ([the surprise, not the level: betas to data surprises](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises)); the pre-mortem discipline is simply to let the base rate set the size.

### A single point of failure ignored — the 2022 "Fed pivot" trades

Through much of 2022, a recurring trade was to position for a Federal Reserve "pivot" — to buy risk assets ahead of each CPI print or FOMC meeting on the inside-view thesis that inflation had peaked and the Fed was about to soften. The thesis was often half-right: inflation *was* rolling over in places, and the underlying read on the economy was reasonable. But the trades repeatedly got run over, most memorably around the hotter-than-expected CPI prints, because the SPOF was never honestly named. The single point of failure for a "pivot" trade wasn't whether inflation was peaking — it was whether *the next data print* would let the Fed pivot, and that print was a near-coin-flip the trader didn't control. A pre-mortem that stipulated "it's the morning after CPI and this long is down hard; why?" would have surfaced the obvious answer — "CPI came in hot and the pivot got pushed out another quarter" — ranked it as a high-probability, high-impact kill-zone risk, and either sized the trade to survive a hot print or waited until after the data. The traders who got hurt weren't wrong about the destination; they were wrong to bet the position on a single data point they'd never isolated as the SPOF. The thesis (inflation peaks, Fed eases eventually) was sound; the *expression* (full size into a coin-flip print) was the failure the pre-mortem would have caught.

## The playbook

Here is the ritual, reduced to a card you run on every thesis before it becomes a position. It takes ten minutes. The trades that survive all six steps at full size are rare, and that scarcity is the point.

![The pre-mortem ritual: six questions to ask before any trade goes on](/imgs/blogs/stress-testing-your-thesis-with-a-pre-mortem-7.png)

1. **Assume failure.** Write it literally: *"It is [date]. This trade is down 30%. Here is why."* Make it specific and dated — that grammar is what triggers prospective hindsight. Do not write "what could go wrong."
2. **List the causes.** From the coroner's posture, write every *distinct, independent* way the trade died. Push past the obvious ones; the prospective-hindsight lift shows up in the fourth and fifth cause, not the first. Include the failures that don't depend on your thesis being wrong (sector de-rate, "too early").
3. **Rank by probability times impact.** Score each cause by its likelihood and its dollar cost. Put your risk-reduction budget on the top-right kill zone (high probability, high impact) — not on whatever feels scariest. Quantify the top cause's downside in dollars; that number belongs in your sizing math.
4. **Steelman the bear.** Build the strongest possible opposing case — stronger than any real opponent's — and check that it makes you uncomfortable. If it doesn't, you built a strawman. Let the strongest surviving doubts cut your conviction, and let conviction set your size.
5. **Find the single point of failure.** Ask of each assumption: *if this alone is false and all else is true, is the thesis dead?* The one that kills it is your SPOF. It is now the single thing you monitor.
6. **Set invalidation and size to it.** Turn the SPOF into a written, observable invalidation level (a price, a data print, a date). Translate the top-ranked downside and the base-rate-adjusted probability into a size whose worst case is a pre-decided, survivable dollar number. Then commit, in writing, to actively hunt disconfirming evidence for the life of the trade.

![Every pre-mortem finding becomes a concrete invalidation level you can monitor](/imgs/blogs/stress-testing-your-thesis-with-a-pre-mortem-6.png)

A closing note on *when* to run it hardest, because this is the counterintuitive part that separates the discipline from the slogan. The natural instinct is to pre-mortem the trades you're unsure about and skip the ones you're certain about — after all, if you're sure, what's to examine? This is exactly backwards. Certainty is not evidence the trade is safe; it's evidence your confirmation bias has done its most thorough work, scrubbing the disconfirming data from your awareness until nothing remains to make you doubt. The trades you're proudest of, the ones that feel like layups, the ones where you catch yourself thinking "this is the easiest money I'll make all year" — those are precisely the trades where the hidden SPOF is largest, because your enthusiasm has buried it deepest. Daniel didn't lose \$6,000 on a trade he was nervous about; he lost it on the one he was *certain* of. Treat your own certainty as the alarm bell that triggers the pre-mortem, not as the excuse to skip it.

The throughline of the series holds here as everywhere: a view that cannot be falsified, sized, and updated is a horoscope, not analysis. The pre-mortem is the step that makes a thesis falsifiable *before* the market does it for you — it names the SPOF that would prove you wrong, prices the downside the inside view hid, and anchors your confidence to your own track record instead of your enthusiasm. The next time you're certain about a trade, that certainty is the signal to run the pre-mortem hardest, because the trades you're surest about are the ones whose risks your confirmation bias has hidden most thoroughly. Kill it in your head first. What survives, you size.

## Further reading & cross-links

Within this series:

- [Structuring a thesis: claim, evidence, and catalyst](/blog/trading/analyst-edge/structuring-a-thesis-claim-evidence-and-catalyst) — the construction manual for the thesis you then stress-test here.
- [Variant perception: where real edge comes from](/blog/trading/analyst-edge/variant-perception-where-real-edge-comes-from) — finding the gap between what's priced and what you believe, before you pressure-test it.
- [Base, bull, and bear: building three scenarios](/blog/trading/analyst-edge/base-bull-and-bear-building-three-scenarios) — the pre-mortem's ranked failure list feeds directly into a structured bear scenario.
- [What would change my mind: defining invalidation upfront](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront) — turning the SPOF into a formal, written kill switch.
- [Measuring conviction: how sure are you really?](/blog/trading/analyst-edge/measuring-conviction-how-sure-are-you-really) — the base-rate-anchored probability that the pre-mortem produces, made rigorous.
- [Confirmation bias and the thesis you fall in love with](/blog/trading/analyst-edge/confirmation-bias-and-the-thesis-you-fall-in-love-with) — the bias the pre-mortem is engineered to defeat.

Going deeper on the mechanisms this post leans on:

- [The asymmetry of losses: why a 50% loss needs a 100% gain](/blog/trading/risk-management/the-asymmetry-of-losses-why-a-50-percent-loss-needs-a-100-percent-gain) — why the downside the pre-mortem prices matters so much more than the upside.
- [Fat tails and the normal distribution trap](/blog/trading/risk-management/fat-tails-and-the-normal-distribution-trap) — why the low-probability, high-impact corner of the matrix is fatter than your intuition assumes.
- [How an options market maker thinks: the other side of your trade](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) — the steelman made literal: who is actually on the other side.

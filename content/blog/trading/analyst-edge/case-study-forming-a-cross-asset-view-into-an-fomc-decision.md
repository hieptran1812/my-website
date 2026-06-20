---
title: "Case Study: Forming a Cross-Asset View Into an FOMC Decision"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A full end-to-end walkthrough of the whole analyst pipeline on one FOMC decision: information diet, six lenses, what's priced, variant, thesis, scenarios, decision tree, EV, size, cross-asset expression, the note, invalidation, live monitoring, the Bayesian update, and the journal entry."
tags: ["analysis", "market-view", "fomc", "cross-asset", "decision-tree", "expected-value", "bet-sizing", "variant-perception", "event-trading", "case-study"]
category: "trading"
subcategory: "The Analyst's Edge"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — This is the whole series run on one real-style setup. We take a single FOMC decision that is 48 hours out and walk it through every stage of the pipeline — from the information diet to the post-event journal entry — so you can see the machine work start to finish across rates, the dollar, gold, and equities.
>
> - The cut is **priced**; the edge is in the **2026 path**, not the level. We do not get paid for being right that the Fed eases — we get paid only for the gap between our path and the market-implied path.
> - Every stage is a tool you have already met. Here they run in sequence on one decision: lenses synthesize a read, the priced path defines the consensus, the variant defines the edge, a decision tree prices each branch, EV decides go/no-go, conviction sets size, and a cross-asset expression decides *which instrument* owns the view.
> - The expression matters as much as the view. The same dovish thesis can be a long in the 2-year, a curve steepener, a short dollar, or long gold — and they have wildly different payoffs, costs, and ways to be wrong.
> - The one rule: **decide everything cold, before 2:00 p.m.** When the statement crosses, you should be executing a written plan, not inventing one under adrenaline — and after the print you update like a Bayesian and grade the *process*, not the P/L.

It is Monday afternoon. The Federal Open Market Committee meets Wednesday at 2:00 p.m. Eastern. The financial press has already written the headline: a 25 basis point cut, the third of the cycle, taking the target range to 4.25-4.50%. Fed funds futures put the odds of that cut at 96%. The strategists have published their previews. The desks are bored. Everyone "knows" what is going to happen, and the consensus among traders is that this meeting is a non-event — the cut is in the price, so there is nothing to do.

That boredom is the tell. When a roomful of smart people all agree a catalyst is settled, two things are true at once: the *obvious* part really is priced, so there is no money in it — and the *non-obvious* part, the part nobody is hedging, is exactly where an edge can hide. A meeting that is "fully priced" on the headline is almost never fully priced on everything that headline implies: the pace of the next four cuts, the language of the statement, the new dot plot, the tone of the press conference. The headline is a decoy. The real decision is happening one layer down, and most of the desk is not even looking there.

So we walk in as an analyst with a blank page. We are not going to ask "will they cut?" — that question is answered and worthless. We are going to run the entire pipeline this series has built, on this one decision, and see whether there is a tradeable view hiding under the boredom. This post is the applied capstone: every earlier post taught one stage in isolation, and here we run all of them in order on a single FOMC meeting so you can watch the whole machine turn.

![The whole analyst pipeline runs on one FOMC decision from information diet to post-event review](/imgs/blogs/case-study-forming-a-cross-asset-view-into-an-fomc-decision-1.png)

That pipeline is the spine of the entire post. The information diet feeds the six-lens read; the lenses synthesize into a read of what is priced; the gap between the priced path and your path becomes the variant; the variant compresses to a one-sentence thesis; the thesis fans into base, bull, and bear scenarios; the scenarios become a decision tree whose EV rolls back to one number; the EV and your conviction set the size; the size meets a cross-asset expression; the whole thing lands on a one-page note with an invalidation; you monitor it live through the statement and presser; and afterward you update like a Bayesian and write the journal entry. We will walk every box, in order, on this FOMC.

## Foundations: the setup and the pipeline we are about to run

Before the work starts, two things need defining: the *scene* (the specific FOMC setup, what is known, and what the market has already priced) and the *pipeline* (the sequence of stages, each of which has its own deep-dive post that we will cross-link rather than re-derive).

### The scene: what is known on Monday

Here is the board as it sits 48 hours out. These are illustrative figures chosen to be realistic, not a forecast of any specific meeting — the point is the *process*, and the numbers are the worked example we carry to the end.

- **The policy rate** is currently 4.50-4.75%. This is the third meeting of an easing cycle that began two meetings ago.
- **The priced cut.** Fed funds futures imply a 96% probability of a 25 bp cut to 4.25-4.50% on Wednesday. This is as close to certain as the market ever gets.
- **The priced path.** Beyond Wednesday, the futures strip discounts a roughly continuous sequence of cuts — about one every meeting — taking the target down to near 3.25% by the middle of next year.
- **The dot plot.** This meeting is a "dots" meeting: the Committee publishes its updated Summary of Economic Projections, including each member's projection for the rate path. The market expects the 2026 median dot to show three more cuts.
- **The data backdrop.** Core services inflation has been sticky in the last two prints, running a touch above the level that would let the Fed cut freely. The labor market is softening, but slowly — no cracks, just a gentle cooling. Credit spreads are tight; equities are near highs; there is no recession in the price.
- **Positioning.** Speculative accounts are short the dollar and long the front end of the curve — the market is *already* leaning dovish. Gold ETFs have been accumulating. This matters enormously and we will come back to it.
- **The options.** The expected move the rates and equity options are pricing for the 24 hours around the meeting is modest — the market is treating Wednesday as low-risk.

That is the full information set. Now the pipeline.

### The pipeline: fourteen stages, each a tool you have already met

The discipline of this series is that no stage is invented here. Each is the subject of its own post, and the job of this case study is to chain them. Here is the sequence, with the question each stage answers:

1. **The information diet** — what do I read for this meeting, and what do I ignore? ([building your information diet](/blog/trading/analyst-edge/building-your-information-diet-signal-versus-noise))
2. **The six lenses** — what does each independent reading say? ([the six lenses](/blog/trading/analyst-edge/the-six-lenses-a-framework-for-reading-any-market))
3. **What's priced** — what does the market already believe? ([what's priced in](/blog/trading/analyst-edge/whats-priced-in-the-question-behind-every-trade), [mapping the consensus](/blog/trading/analyst-edge/mapping-the-consensus-what-does-the-market-already-believe))
4. **The variant** — where, specifically, do I differ from the price? ([variant perception](/blog/trading/analyst-edge/variant-perception-where-real-edge-comes-from))
5. **The thesis** — can I say it in one falsifiable sentence?
6. **Scenarios** — base, bull, bear, with probabilities. ([base, bull, and bear](/blog/trading/analyst-edge/base-bull-and-bear-building-three-scenarios))
7. **The decision tree + EV** — branch the outcomes, price each move, roll the EV back. ([decision trees](/blog/trading/analyst-edge/decision-trees-for-event-driven-views), [expected value](/blog/trading/analyst-edge/expected-value-the-only-math-a-view-really-needs))
8. **Conviction → size** — how sure am I, and how much do I bet? ([from conviction to size](/blog/trading/analyst-edge/from-conviction-to-size-the-bet-sizing-bridge))
9. **Expression** — which instrument carries the view? ([choosing the instrument](/blog/trading/analyst-edge/choosing-the-instrument-to-express-your-thesis))
10. **The note** — the one-page write-up. ([the trade-idea note](/blog/trading/analyst-edge/the-trade-idea-note-a-one-page-template))
11. **Invalidation** — what would change my mind, defined upfront? ([what would change my mind](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront))
12. **Live monitoring** — what do I watch through the statement and presser? ([monitoring a live thesis](/blog/trading/analyst-edge/monitoring-a-live-thesis-building-your-watch-dashboard))
13. **The Bayesian update** — how do I re-weight after the print? ([updating like a Bayesian](/blog/trading/analyst-edge/updating-on-new-information-thinking-like-a-bayesian))
14. **The journal entry** — what do I record so this compounds? ([the decision journal](/blog/trading/analyst-edge/the-decision-journal-the-habit-that-compounds))

Fourteen stages, one decision. Let us run it.

## Stage 1: the information diet for this meeting

The first discipline is filtering. Into an FOMC there is a firehose of input — sell-side previews, dot-plot guesses, twelve different economists on the tape, social media takes, the last six inflation prints, the last six jobs reports, oil, the dollar, every Fed speaker's last speech. Most of it is noise. The information diet is the decision about what you actually let into your read.

For *this* meeting, the signal-versus-noise call is sharp. The signal is the small set of inputs that could actually change the *path* — not the cut, which is settled, but the trajectory beyond it:

- **The last two core services CPI prints**, because sticky services inflation is the single thing that could make the Committee cut slower than priced.
- **The last jobs report's wage component**, for the same reason: wage growth feeds services inflation.
- **The prior dot plot**, as the baseline the new one will be compared against.
- **The Chair's last two public comments on the *pace* of cuts**, because the presser will echo or revise them.

Everything else — the cut probability, the equity tape, the daily oil wiggle, the strategist previews that all say "25 and done for now" — is noise *for this view*, because it tells you only what is already priced. The diet's job is to stop you from anchoring on the consensus by drowning in consensus inputs. We covered the general construction in [building your information diet: signal versus noise](/blog/trading/analyst-edge/building-your-information-diet-signal-versus-noise); the meeting-specific application is simply: *read the four things that bear on the pace, and mute the rest.*

There is a specific failure the diet protects against into an FOMC, and it is worth naming because almost everyone falls into it. The volume of *consensus* input — the previews, the cut-probability dashboards, the strategist notes that all say the same thing — is so overwhelming that reading it feels like doing work, when in fact it is the opposite of work. Every hour you spend absorbing "the cut is 96% priced" anchors you harder on the consensus and makes it psychologically harder to hold a variant. The more consensus you consume, the more the variant feels reckless, because you have spent the week marinating in reasons the consensus is right. This is why the diet is a *subtractive* discipline first: the single most valuable thing you can do into a crowded meeting is stop reading the inputs that only confirm what is priced, and protect the small space in your head where a contrarian read can survive. The four signal inputs are not chosen because they are the *most information* — they are chosen because they bear on the *non-priced* question (the pace), and everything else is muted precisely because it would crowd that question out.

The output of this stage is not a view yet. It is a clean desk: four inputs in front of you, and a deliberate decision to ignore the noise that would only tell you what you already know.

## Stage 2: the six-lens read

With the inputs filtered, you read the market through the six lenses. The discipline here — laid out in full in [the six lenses](/blog/trading/analyst-edge/the-six-lenses-a-framework-for-reading-any-market) — is that no single lens decides; the call is the *synthesis*, and the most valuable information is where the lenses *disagree*. Here is each lens pointed at this FOMC.

![The six lenses each read the same FOMC meeting through a different feed and point toward one call](/imgs/blogs/case-study-forming-a-cross-asset-view-into-an-fomc-decision-2.png)

The grid is the whole read on one page. Walking it lens by lens:

- **Macro — the tide.** Inflation is cooling but services are sticky; the labor market is softening slowly. A cut is justified. But "justified to cut" and "justified to cut at every meeting" are different claims, and the macro lens is genuinely ambiguous on the *pace*. Call: cut yes, pace open.
- **Fundamental — the worth.** Real yields are still high, credit spreads are tight, there is no recession priced. This is a *normalization* cut, not an *emergency* cut. The fundamental backdrop argues for a slow, deliberate ease — not the near-continuous path the futures discount.
- **Technical — the path.** The 2-year yield is ranging; the dollar index sits near support; nothing is trending. The price is coiled, waiting for the catalyst. The technical lens has no opinion yet — it is telling you the move hasn't started.
- **Flows — who is buying.** Real-money accounts are long duration; gold ETFs are accumulating. The dovish trade is already *owned*. Flows are a warning: the easy money has been made by people who got long ahead of you.
- **Positioning — who is committed.** Speculative accounts are short the dollar and long the front end. This is the most important lens for this meeting: the *pain trade is hawkish*. If the Fed disappoints the dovish crowd even slightly, there is a crowded, one-sided book that has to unwind, and the move would be violent because everyone is on the same side. The mechanics of why a crowded book makes the surprise asymmetric are in [positioning and the pain trade](/blog/trading/event-trading/positioning-and-the-pain-trade).
- **Sentiment — the mood.** Consensus calls it a sure cut; desks are bored; nobody is hedging a surprise. Complacency. An unexpected outcome is under-hedged, which amplifies its market impact.

The synthesis: **five of six lenses say "dovish is justified but crowded."** The macro and fundamental lenses support easing but argue the *pace* is slower than priced. The flows, positioning, and sentiment lenses all flash the same warning — the dovish trade is owned, the hawkish surprise is the pain trade, and complacency means any deviation hits hard. The technical lens abstains. That is a coherent read, and it points somewhere specific: *the market is too confident in a fast, smooth easing path.*

Notice what the synthesis did. No single lens gave us a trade. Macro alone would have said "be long, they're cutting" — which is the consensus and worthless. It was the *combination* of a slow-pace fundamental read with a crowded-dovish positioning read that produced the actual insight. When the lenses agree you get conviction; when they disagree you get caution; and when most of them point at the same non-consensus risk, you get a candidate edge.

## Stage 3: what's priced in

Now the central question of the entire series: **what does the market already believe?** You cannot have an edge until you know the consensus you are betting against, because the consensus is what you get paid (or charged) relative to. There are three things priced into this meeting, and you read each off a different instrument. The full taxonomy of where each consensus lives is in [mapping the consensus](/blog/trading/analyst-edge/mapping-the-consensus-what-does-the-market-already-believe); the foundational logic of pricing relative to the consensus is in [what's priced in: the question behind every trade](/blog/trading/analyst-edge/whats-priced-in-the-question-behind-every-trade).

**1. The fed funds path.** Fed funds futures give you the market-implied trajectory of the policy rate, meeting by meeting. Wednesday's cut is priced at 96%. Beyond that, the strip discounts roughly one cut per meeting down to near 3.25% by mid-next-year. This is the line you must beat — not "will they cut" but "is this *pace* right?"

**2. The dot-plot expectation.** The market expects the Committee's published 2026 median dot to show three more cuts after this one. The dots are the Fed's own projection; the gap between the dots and the futures strip is itself a tradeable signal, but the relevant point here is the *expectation* the market has built about what the dots will say.

The dot plot deserves a moment of care, because it is the single most important release on Wednesday for *our specific variant* and it is widely misread. The dots are not a promise — they are each Committee member's best guess of where the appropriate policy rate sits at the end of each year, plotted as anonymous markers, and the median dot is the one the market fixates on. Two subtleties make the dots tradeable. First, the *median* can move even when most members do not change their view: if just two members on a knife-edge shift from "three cuts" to "two cuts," the median can drop a whole cut, which is a discontinuous jump in the headline the market reads. That makes the dots prone to surprising in exactly the direction our variant predicts — a small, plausible shift in a couple of members produces a hawkish-looking median revision. Second, the dots and the futures strip are measuring different things: the dots are the Committee's projection *under its own central forecast*, while the strip is the market's probability-weighted expectation *across all paths*, including recession scenarios where the Fed cuts fast. When the strip prices more cuts than the dots show, the market is partly pricing a recession-insurance path the Committee itself is not projecting. Our variant — that the pace is too dovish — is in effect a bet that the dots are closer to right than the strip, which is a cleaner and more defensible disagreement than "I think the market is wrong about everything."

**3. The options expected move.** The options market has priced how big a move it expects around the meeting. This is the number that lets you size the event risk, and it deserves a worked example because it is the most directly tradeable read of "what's priced" you will find.

#### Worked example: the options expected move sizing the event risk on a \$25,000 position

You are going to risk capital on this meeting, and the first question is: *how big is the event the market is pricing?* The options answer it. Suppose the relevant rates-linked instrument (use a 2-year-note proxy or a rate ETF) trades at \$100, and the at-the-money straddle — one call plus one put, both struck at \$100 and expiring just after the meeting — costs \$0.95. The straddle premium, as a percent of the underlying, is roughly the one-sigma move the market has priced:

$$\text{expected move} \approx \frac{\text{straddle premium}}{\text{underlying}} = \frac{\$0.95}{\$100} = 0.95\%$$

So the market is pricing about a **±0.95% move** in this instrument over the meeting window. That is a small expected move — confirmation that the desk treats this as a near-non-event.

Now translate it to dollars on a position. You are budgeting \$25,000 of risk capital to this idea. If you held the underlying outright in that size, the *one-sigma* dollar swing the market is pricing is:

$$\$25{,}000 \times 0.95\% = \$237.50$$

A one-sigma move costs or makes you about \$238. But events have fat tails — the realized move on a surprise is routinely two to three sigma. A two-sigma adverse move would be roughly \$475, a three-sigma roughly \$713, on a \$25,000 outright position. The expected move is the market's *low* estimate of the risk; the tail is what actually hurts. The discipline this hands you: size so that even a three-sigma adverse move is survivable, and *never* let the position be so large that the priced one-sigma move alone is uncomfortable. Put plainly, the options market is selling you a measuring stick for the event, and you size your bet against the tail of that stick, not its center.

The mechanics of how the options market maker on the other side derives and hedges that expected move are in [how an options market maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade), and the general logic of consensus expectations being priced into an event is in [consensus expectations and priced-in](/blog/trading/event-trading/consensus-expectations-and-priced-in).

## Stage 4: the variant perception

You now know the consensus precisely: a 96% cut, a smooth path to ~3.25%, an expected three more dot-plot cuts in 2026, and a tiny expected move. The variant perception is the single place where you differ — and if you cannot name it in one sentence, you do not have a trade. This is the heart of the series, covered in [variant perception: where real edge comes from](/blog/trading/analyst-edge/variant-perception-where-real-edge-comes-from).

Your six-lens read said the market is too confident in a *fast, smooth* easing path. So here is the variant: **the cut is right, but the pace beyond it is too aggressive. Sticky services inflation will force the Committee to pause after this cut, and the 2026 dots will be more hawkish than the market expects.** You are not betting against the cut. You are betting against the *pace*.

![The market-implied fed funds path runs smoothly to 3.25 percent while my path pauses and bottoms near 3.75 percent](/imgs/blogs/case-study-forming-a-cross-asset-view-into-an-fomc-decision-3.png)

The chart is the variant made visible. The blue step line is the **market-implied path** — Wednesday's cut, then a cut at nearly every meeting down toward 3.25%. The amber dashed line is **your path** — the same cut Wednesday, then a *pause*, with the bottom near 3.75% instead of 3.25%. The two lines agree on Wednesday and agree on direction (rates are falling). They diverge on *pace*, and by the fifth forward meeting the gap is about 50 basis points. That gap — and only that gap — is the trade. If your path is right, you make money even though you agree with the consensus that the Fed is cutting, because the cuts you doubt are already in the price.

This is the discipline the variant enforces, and it is the one most traders skip. "I think the Fed is dovish, I'll be long bonds" is *not a variant* — it is the consensus, and you would be paying the spread to own what is already priced. The variant has to be a specific, falsifiable disagreement with a specific, priced number. Ours is: *the 2026 path is ~50 bp too low; the market is pricing a pause that won't be skipped.* The reason this is an edge rather than a guess is the six-lens synthesis behind it — the fundamental lens (normalization not emergency) plus the positioning lens (the dovish book is crowded, so the pain trade runs your way).

## Stage 5: the one-sentence thesis

A thesis is a forecast plus a reason plus a catalyst, compressed until it fits on one line. We built the anatomy in [thesis vs forecast](/blog/trading/analyst-edge/thesis-vs-forecast-the-anatomy-of-a-tradeable-view); here is the FOMC thesis:

> **"The cut is priced, but the slow-and-steady 2026 path is not — sticky services inflation will force a pause the market hasn't discounted, so the front end can reprice higher relative to the long end into a steeper curve."**

One sentence. It names the forecast (a pause), the reason (sticky services inflation), and the catalyst (this meeting's dots and presser). It is falsifiable — if the dots show no pause and the curve flattens after the presser, the thesis is wrong. And it points at a specific instrument relationship (front end versus long end), which we will turn into an expression in stage 9. A thesis you can't falsify is a horoscope; this one tells you exactly what would prove it wrong.

## Stage 6: base, bull, and bear scenarios

A single thesis is too brittle to trade. You build three scenarios — the base case (what you most expect), the bull case (your thesis works even better than expected), and the bear case (you are wrong) — each with a probability that sums to 100%. The construction is in [base, bull, and bear](/blog/trading/analyst-edge/base-bull-and-bear-building-three-scenarios). For this FOMC, the natural carve maps to the three things the Committee can do with the path:

- **Bear case for us (25%) — hawkish.** The Fed cuts but the dots and presser lean *more* hawkish than even we expect, or they signal the cut is the last for a while in language that the dovish-positioned market reads as a shock. The crowded long-front-end book unwinds violently. Our steepener loses, because a hawkish surprise *flattens* the curve in the short run (the front end sells off faster than the long end). This is the pain-trade scenario the positioning lens warned about — and crucially, it hurts us.
- **Base case (45%) — in line.** The Fed cuts, the dots show the expected three 2026 cuts, the presser is balanced. Small moves. The curve steepens gently as the front end drifts down with the cut while the long end holds. Our thesis is mildly right — the market doesn't fully price our pause, but it leans our way over the following weeks as data confirms.
- **Bull case for us (30%) — dovish-on-pace-revision.** The Fed cuts but the *dots* reveal a pause (a 2026 median above what's priced), or the Chair explicitly flags sticky services inflation as a reason to slow down. This is our variant printing on the screen. The front end repriced higher relative to the long end, the curve steepens hard, and the crowded dovish book is forced to capitulate in our direction.

Note something subtle and important: this is *not* the textbook hawkish/in-line/dovish carve where dovish is always the good outcome. Our thesis is about the *pace*, so our "bull case" is the Fed signaling a *slower* pace — which is the hawkish-on-pace outcome. We have to define the branches around *our variant*, not around the generic direction of the decision, or we will mis-map the payoffs. The 25/45/30 split reflects the six-lens read: the crowded positioning makes the slow-pace outcome more likely than the market thinks (hence 30% on our bull, above the priced odds), but the hawkish *shock* scenario is a real 25% tail because a crowded book can overshoot in either direction.

## Stage 7: the decision tree and the EV

Scenarios with probabilities become a decision tree: for each branch, you estimate the market reaction, decide your pre-committed action, compute the dollar payoff at the leaf, and roll the expected value back to a single number. This is the engine that turns a view into a go/no-go decision, built in full in [decision trees for event-driven views](/blog/trading/analyst-edge/decision-trees-for-event-driven-views) with the EV math in [expected value: the only math a view really needs](/blog/trading/analyst-edge/expected-value-the-only-math-a-view-really-needs).

![The FOMC decision branches into hawkish in line and dovish each with odds a reaction and a pre-planned payoff](/imgs/blogs/case-study-forming-a-cross-asset-view-into-an-fomc-decision-4.png)

The tree lays out the whole decision. The event is on the left with our \$25,000 risk budget. It fans into the three branches with our probabilities (25% hawkish, 45% in line, 30% dovish-on-pace). Each branch maps to a cross-asset reaction, and each reaction maps to a dollar payoff on our chosen expression (a steepener, which we settle on in stage 9) with a pre-committed action. The payoffs reflect a position structured so the pre-committed stop caps the hawkish branch while the in-line and bull branches are allowed to run.

#### Worked example: the decision-tree EV across the three branches

Here is the roll-back, with every number explicit, on the \$25,000 risk budget. The payoffs assume a steepener sized so that a hawkish surprise is cut on a pre-committed stop at a \$1,250 loss, the in-line branch is held for a modest \$900 gain as the curve drifts steeper, and the dovish-on-pace branch is added to and ridden for a \$2,600 gain:

$$\text{EV} = P_{\text{hawk}} \times \text{Payoff}_{\text{hawk}} + P_{\text{base}} \times \text{Payoff}_{\text{base}} + P_{\text{bull}} \times \text{Payoff}_{\text{bull}}$$

$$\text{EV} = 0.25 \times (-\$1{,}250) + 0.45 \times (+\$900) + 0.30 \times (+\$2{,}600)$$

$$\text{EV} = -\$312.50 + \$405.00 + \$780.00 = +\$872.50$$

The tree rolls back to about **+\$873** of expected value. Compare that to the flat \$0 leaf — standing aside through the meeting. The trade clears the flat leaf by a healthy margin: +\$873 of EV for \$1,250 of pre-committed downside is a reward-to-risk on the EV of roughly 0.7, and the structure caps the worst case at a known, survivable number. That is a trade worth doing. (If the EV had come out near \$0, the disciplined call would be to stand aside — *being flat is itself a position with \$0 EV that every branch must beat.*) What the tree really does is convert a vague "I think the pace is too dovish" into a single dollar number you can compare against doing nothing, and only the comparison tells you whether the view is worth capital.

The decisive feature of the tree is the *pre-committed action* in each branch. We are not deciding live whether to cut, hold, or add — we have decided it now, cold, on Monday. When the print lands Wednesday at 2:00:00, we read which branch the world chose and execute. The decision has already been made; the live moment is pure execution.

## Stage 8: conviction to size

EV tells you the trade is positive; conviction and size tell you *how much* to bet. The bridge from "how sure am I" to "how many dollars" is in [from conviction to size](/blog/trading/analyst-edge/from-conviction-to-size-the-bet-sizing-bridge) and [measuring conviction](/blog/trading/analyst-edge/measuring-conviction-how-sure-are-you-really). The mapping for this trade has two pulls in opposite directions.

On the *high-conviction* side: the six-lens read is coherent, the variant is specific and falsifiable, and the positioning lens gives the trade an extra edge (the pain trade runs our way). On the *low-conviction* side: this is a **binary catalyst** — the outcome resolves in a single instant on Wednesday, with a fat-tailed surprise distribution, and our variant is partially *crowded* (the dovish book is owned, so some of our edge is shared). Binary catalysts and crowded variants both argue for *smaller* size, because you cannot scale in or out around the event and you are not the only one holding the view.

The resolution: **a half unit, not a full unit.** A full unit in this book risks 1% of capital; a binary catalyst with a crowded variant gets the binary discount — half of that. On a \$250,000 book, a full unit is \$2,500 of risk; the half unit is \$1,250, which is exactly the pre-committed stop loss in the hawkish branch of the tree. The size and the stop are the same number on purpose: the most you can lose is the half-unit risk budget, set before entry. This is the discipline that keeps a binary from blowing a hole in the book — you size the bet so the worst pre-committed leaf is a number you chose, not a number the market chose for you. The asymmetry logic behind sizing a high-conviction-but-binary bet smaller is in [asymmetry and the art of the high-conviction bet](/blog/trading/analyst-edge/asymmetry-and-the-art-of-the-high-conviction-bet).

## Stage 9: the cross-asset expression

This is the stage that separates the case study from every single-asset post in the series, and it is where most of the craft lives. You have a view — *the 2026 pace is too dovish* — and a size — *half a unit, \$1,250 of risk*. But a view is not a position until you choose the *instrument* that carries it, and the same view can be expressed in at least four different assets, each with a different payoff, cost, and way to be wrong. The framework for choosing is in [choosing the instrument to express your thesis](/blog/trading/analyst-edge/choosing-the-instrument-to-express-your-thesis); the cross-asset reaction logic — why the same decision moves four assets differently — is in [the surprise, not the level: betas to data surprises](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises).

![The same FOMC decision moves rates the dollar gold and equities in four different ways per branch](/imgs/blogs/case-study-forming-a-cross-asset-view-into-an-fomc-decision-5.png)

The reaction map shows why the choice matters. The same decision moves four assets in four different ways across the three branches. A hawkish surprise sends the 2-year yield up 12 bp, the dollar up 0.8%, gold down 1.5%, and equities down 1.2%. A dovish-on-pace outcome (our bull) does roughly the opposite. The reactions are *correlated but not identical* — gold and equities both react to real yields, but equities also carry a growth signal that muddies the read, and the dollar reacts to the *relative* rate path versus other central banks. The expression you pick decides which of these reactions you own. Here is how the four candidates stack up for *our specific variant*:

- **Long the 2-year note (outright).** Cleanest direct expression of "the front end can fall." But it is a *directional duration* bet — you make money if rates fall *at all*, which is the consensus. It does not isolate the *pace* view; you'd be long the part everyone already owns.
- **A 2s10s curve steepener (long the 2-year, short the 10-year, DV01-neutral).** This isolates the *relative* move of the front end versus the long end — which is exactly our variant. If the Fed signals a pause, the front end reprices higher relative to the long end and the curve steepens; we profit *without* taking an outright directional rate bet. This expresses the variant and hedges out the consensus part. The general logic of expressing a view without a directional bet is in [relative value](/blog/trading/analyst-edge/relative-value-expressing-a-view-without-a-directional-bet).
- **Short the dollar.** A dovish path weakens the dollar, but the dollar's move depends on *relative* policy versus the ECB and others, so it is a noisy, second-order expression of a Fed-pace view. Too much else can move it.
- **Long gold.** A slower real-yield decline is *bad* for gold, so if our variant (a pause) is right, gold should *fall* — meaning the gold expression of our view is to be *short* or *flat*, not long. This is a trap: a trader who is "dovish, so long gold" has not noticed that our specific variant (hawkish-on-pace) cuts the other way for gold.

A word on *why* the steepener is the right shape, because it is the part of the curve mechanics that trips people up. There are two distinct horizons in our thesis. In the *immediate* reaction to a hawkish-on-pace signal, the front end sells off (yields up) because the market pulls cuts *out* of the near-term path, while the long end barely moves because long rates are anchored by expectations of the terminal rate and inflation, not the next few meetings. That makes the curve *steepen* between the 2-year and the 10-year — the 2-year yield rises toward the 10-year. So the trade that profits from "the front end reprices higher relative to the long end" is to be *long* the 2-year (positioned for the 2-year price, which falls as its yield rises, to be hedged) and *short* the 10-year, sized DV01-neutral so the two legs have equal sensitivity to a parallel move in rates. The steepener nets out the part of the move that is just "rates went up a bit everywhere" — the consensus directional component — and keeps only the *relative* repricing of the front end versus the long end, which is precisely our variant. If we had instead bought the 2-year outright, a chunk of any gain would simply be the priced cut delivering, and we would be paid for the consensus rather than the disagreement. The steepener is the surgical instrument; the outright is the blunt one.

#### Worked example: cross-asset position sizing across rates, gold, and an equity expression

The \$1,250 risk budget has to be sized differently in each instrument because each has a different reaction size and a different stop distance. Walk three candidate expressions on the same \$1,250 of risk:

- **Steepener (chosen).** Stop is set where the curve flattens 10 bp against us (the hawkish branch). If the curve moving 10 bp against us costs \$1,250, then each basis point of curve move is worth \$125, so the DV01 of the steepener is sized to **\$125 per bp**. In the bull branch (curve steepens ~21 bp our way), the gross is about \$2,600 — matching the tree's dovish leaf.
- **Outright long 2-year.** To risk the same \$1,250, with a hawkish stop at the 2-year selling off 12 bp, you'd size to about **\$104 per bp** of 2-year yield. But this expression *also* pays you for the consensus cut — so a chunk of any gain is just the priced move, not the variant. Same risk, dirtier signal.
- **Long gold (rejected as wrong-signed).** If you mistakenly went long gold with \$1,250 of risk and a stop at gold down 1.5% (the hawkish branch), you'd hold roughly \$1,250 / 0.015 = **\$83,000 notional** of gold. But our variant says gold should *fall* if we're right — so this expression loses in our bull case. The position is sized correctly and *pointed the wrong way*. The sizing math cannot save an expression that contradicts the view.

The chosen expression is the **2s10s steepener at \$125 per bp**, risking \$1,250, because it isolates the pace variant and nets out the consensus directional move. Picking the instrument is really a second view layered on the first: you are choosing not just *what* will happen but *which price relationship most purely captures the part you actually disagree with the market about* — and a perfectly correct thesis expressed in the wrong instrument is still a losing trade.

## Stage 10: the filled trade-idea note

Everything above now collapses onto one page. The trade-idea note is the contract you write with your future self *before* entry — thesis, priced-in-versus-variant, evidence, catalyst, expression, size, invalidation, markers to watch, and conviction tier, all on a single dated page. The template is in [the trade-idea note](/blog/trading/analyst-edge/the-trade-idea-note-a-one-page-template).

![The completed FOMC trade fills all nine note fields from thesis through conviction on one dated page](/imgs/blogs/case-study-forming-a-cross-asset-view-into-an-fomc-decision-6.png)

The note is the whole pipeline made portable. Read across the nine fields and you can reconstruct every stage: the one-sentence thesis (stage 5), the priced-versus-variant gap (stages 3-4), the evidence (the six-lens read, stage 2), the catalyst and timeline (the meeting itself), the expression (stage 9), the size and dollar risk (stage 8), the invalidation (stage 11, next), the markers to watch (stage 12), and the conviction tier (medium — real but crowded — hence the half unit). The EV check sits at the bottom: the tree rolls to +\$873, it clears the flat leaf, the trade goes on.

The discipline of the note is that there are no blank fields. An empty invalidation field means *do not enter* — you have no idea what would prove you wrong. An empty variant field means *no edge* — you are about to trade the consensus. And the note is written *before* 2:00 p.m., when you are calm, so that every reaction is pre-committed and nothing is decided in the hot moment. We argued the longer-form version of this — the investment memo — in [the investment memo](/blog/trading/analyst-edge/the-investment-memo-arguing-your-view-on-paper); the note is its one-page, event-driven cousin.

## Stage 11: the invalidation

The single most skipped step, and the one that separates analysis from hope. Before you put the trade on, you define — in writing — exactly what would prove you wrong, so that when it happens you act instead of rationalize. The discipline is in [what would change my mind: defining invalidation upfront](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront).

For this steepener, the invalidation is precise: **the thesis is wrong if the dots show no 2026 pause (a median at or below the priced path), or if the 2s10s curve *flattens* 10 bp against us after the presser.** Either event says the market is right about the pace and we are not. The flattening level is also our hard stop — the \$1,250 risk number from stage 8. Invalidation and stop are linked but distinct: the stop is the price-based exit (curve flattens 10 bp, cut the position); the invalidation is the *thesis*-based exit (dots show no pause, the reason for the trade is gone even if the price hasn't moved yet).

This matters because the two can fire at different times. If the dots come out with no pause but the curve hasn't moved yet, the *thesis* is already invalidated even though the stop hasn't been hit — and the disciplined action is to cut anyway, because you are now holding a position whose reason has evaporated. The hardest version of this call — telling whether the thesis is genuinely broken or you are just looking at noise — is the subject of [thesis broken or just noise](/blog/trading/analyst-edge/thesis-broken-or-just-noise-the-hardest-call-you-make); here the dot plot is a clean, unambiguous signal, so the invalidation is easy to read.

## Stage 12: live monitoring through the statement and presser

At 1:55 p.m. Wednesday the work is done. The note is written, the size is set, the steepener is on at \$125 per bp, the stop sits where the curve flattens 10 bp, and every branch has a pre-committed action. Now you build the watch dashboard for the live event — the small set of things you will actually look at as the decision unfolds. The construction of a live watch dashboard is in [monitoring a live thesis](/blog/trading/analyst-edge/monitoring-a-live-thesis-building-your-watch-dashboard).

The FOMC unfolds in three acts, and each maps to a marker on the dashboard:

1. **2:00:00 — the statement.** The headline cut crosses (priced, ignore it). The first thing that matters is any change in the *language* about the pace. You are watching the 2s10s curve on the tape: does it steepen (our way) or flatten (against us)?
2. **2:00:01 — the dots.** The Summary of Economic Projections drops with the statement. The single number on your dashboard is the **2026 median dot**. If it shows a pause (median above the priced path), the variant is printing. If it shows the expected three cuts, the base case. If it shows *more* cuts, the bear tail.
3. **2:30 — the press conference.** The Chair speaks. You are listening for one thing: the *tone on the pace*. Does the Chair flag sticky services inflation as a reason to go slowly (our way), or emphasize the easing trajectory (against us)? The presser routinely moves the market more than the statement, and it is where the curve makes its real move.

The monitoring discipline is brutal and simple: you do not re-decide anything. You read each marker, match it to the branch it belongs to, and execute the pre-committed action for that branch. The dashboard exists to tell you *which branch the world chose*, not to invite you to form a new view at 2:01 p.m. The mechanics of how the same FOMC outcome moves different assets differently — and why the curve, not the headline, carries the signal — are in [the reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently).

## Common misconceptions: the mistakes traders make into an FOMC

Before we see how it plays out, the four errors that wreck FOMC trades — each the mirror image of a stage we just walked.

**Mistake 1: trading the level, not the surprise.** The most common error is positioning on the *direction* of the decision — "they're cutting, be long" — when the cut is already in the price. You are paid only for the gap between the outcome and what was priced, never for the priced part itself. The fix is stage 4: name a specific variant against a specific priced number. If your "view" is the consensus, you have no trade, only a spread to pay.

**Mistake 2: no pre-committed reactions.** Walking into 2:00 p.m. with a view but no plan for each branch means making your most consequential decision at the moment of maximum adrenaline and price velocity — exactly when your reasoning is most compromised. The fix is stage 7: the decision tree pre-decides your action in every branch, so the live moment is pure execution.

**Mistake 3: narrative-driven positioning.** Building the trade around a story ("the Fed has pivoted, this is the start of a big easing cycle") rather than a falsifiable, priced disagreement. Narratives feel like analysis but carry no invalidation — there is no specific event that would make you admit you are wrong, so you hold the loser and add to it. The fix is stages 5 and 11: a one-sentence falsifiable thesis with a written invalidation.

**Mistake 4: oversizing a binary.** Treating a fat-tailed, single-instant catalyst like a normal trade you can scale in and out of. You cannot manage a binary in real time — it resolves in one tick — so a full-sized position can blow a hole in the book on the tail. The fix is stage 8: the binary discount, sizing the bet at a half unit so the worst pre-committed leaf is a number you chose. The broader logic of why oversizing destroys a compounding book is in [risk management: the only free lunch](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine).

A fifth, quieter mistake threads through all four: **falling in love with the thesis** and reading every incoming tick as confirmation. The defense is the written note and the pre-committed invalidation — you decide what would prove you wrong *before* you have an emotional stake in being right.

## How it plays out in real markets

Wednesday, 2:00:00 p.m. The statement crosses. The 25 bp cut is there, as priced — a non-event. But the language on the pace has shifted: the Committee adds a line noting that "the Committee will be attentive to the persistence of services inflation," softer on the trajectory than the prior statement.

2:00:01 — the dots. The 2026 median dot shows **two** more cuts, not the three the market expected. The Committee is signaling a pause. *This is the bull branch printing on the screen.* The front end reprices higher relative to the long end almost instantly — the 2-year yield jumps as the market pulls forward its re-pricing of the pace, while the 10-year barely moves. The 2s10s curve steepens 9 basis points in the first ninety seconds.

This is the dovish-on-pace branch from our tree — our 30% bull case. We pre-committed to *add* on this branch and ride it. There is no decision to make at 2:01; we read the dots, match them to the bull branch, and execute the add we wrote down on Monday. The crowded long-front-end book the positioning lens flagged is now unwinding in our direction — the speculative accounts who were short the dollar and long the front end are caught, and their capitulation steepens the curve further.

2:30 — the press conference. The Chair explicitly flags sticky services inflation as a reason to slow the pace of cuts. This confirms the statement and the dots. The curve extends its steepening to a total of about 21 bp by 3:00 p.m. Our steepener, sized at \$125 per bp, is up roughly \$2,600 gross on the bull branch — exactly the dovish leaf of the tree.

The contrast with the bored desk is the whole lesson. The trader who walked in thinking "the cut is priced, nothing to do" had no position and missed it. The trader who was "dovish, so long gold" was *long the wrong asset* — gold fell about 1.5% as real yields backed up on the hawkish-on-pace signal, so being directionally "dovish" still lost money in the wrong instrument. The trader who ran the pipeline had the *specific* variant (the pace, not the level), the *right* expression (the steepener, not gold or an outright), and a *pre-committed action* (add on the bull branch), and executed it calmly while the rest of the desk scrambled to form a view in real time.

Two caveats keep this honest. First, this is the branch that *hit* — a 30% branch. Had the dots shown three or four cuts (the 70% of outcomes that were not our bull case), the steepener would have been flat or stopped out at the pre-committed \$1,250, and that would have been a *good* trade with a bad outcome, because the process was sound and the EV was positive. We judge the decision, not the result. Second, the move was clean because the dots gave an unambiguous signal; many FOMCs are murkier, the statement and presser conflict, and the curve whips both ways before settling. The pipeline still holds — you just hold tighter to the pre-committed actions when the live read is noisy.

## Stage 13: the post-event Bayesian update

The print is information, and information should move your view by a specific, proportional amount — not flip it to certainty, and not leave it unchanged. After the dovish-on-pace outcome, you update like a Bayesian: the evidence was diagnostic (the dots are the Committee's own projection, hard to fake), so it moves your variant probability meaningfully. The mechanics are in [updating on new information: thinking like a Bayesian](/blog/trading/analyst-edge/updating-on-new-information-thinking-like-a-bayesian).

![After the dovish print the posterior odds rise the position is re-sized and the decision is graded on process](/imgs/blogs/case-study-forming-a-cross-asset-view-into-an-fomc-decision-7.png)

The before-and-after shows the update. Before the print, your prior on the variant — the pace is slower than priced — was about 55%. The dots confirmed a pause, which is strong, diagnostic evidence in favor. Your posterior rises to roughly 72%. The variant is now *more* likely than it was, because the Committee just put its own projection behind it. And because the posterior is higher, the position is re-sized: the binary catalyst has passed, so the binary discount no longer applies, and you can add the steepener up toward a full unit on the post-print pullback — the catalyst risk that justified the half unit is gone, replaced by a multi-week thesis that the data will either confirm or break.

#### Worked example: the post-event Bayesian update re-sizing the book

Make the re-size explicit. Before the meeting, the half-unit steepener risked \$1,250 (0.5% of the \$250,000 book) because of the binary discount. The print confirmed the variant, raising your posterior from 55% to 72% and — critically — *removing the binary risk*, since the single-instant catalyst is now behind you and the remaining risk is the slower, manageable risk of the pace thesis playing out over weeks.

The book can now hold the full unit: 1% of capital, or **\$2,500 of risk**. You already have \$1,250 on. So you add a second \$1,250 of steepener risk on the post-print pullback, bringing the position to the full \$2,500 unit:

$$\text{new risk} = \$2{,}500_{\text{full unit}} - \$1{,}250_{\text{already on}} = \$1{,}250 \text{ added}$$

At \$125 per bp, that doubles the DV01 of the position to **\$250 per bp**. The stop on the larger position moves to where a 10 bp curve flattening costs the full \$2,500. You are not chasing the move with adrenaline; you are *systematically* re-sizing because the posterior rose and the binary risk resolved, exactly as the Bayesian update prescribes. The print did two distinct things at once — it raised the probability your view is right *and* it removed the event risk that capped your size — and both changes pull in the same direction, so the disciplined response is to add, sized to the new posterior, not to the old fear.

The general principle of managing the position around the view — when to add, cut, or exit as the thesis evolves — is in [when to add, cut, or exit](/blog/trading/analyst-edge/when-to-add-cut-or-exit-managing-the-position-around-the-view).

## Stage 14: the decision-journal entry

The last stage, and the one that makes the whole pipeline compound. You write down the decision — not the P/L, the *decision* — so that over hundreds of trades you can grade your *process* independent of any single outcome. The habit is in [the decision journal](/blog/trading/analyst-edge/the-decision-journal-the-habit-that-compounds).

The entry for this trade, written the same evening:

- **What I believed and why.** The cut was priced; the 2026 pace was too dovish; sticky services inflation would force a pause. Supported by five of six lenses and a crowded dovish positioning that made the pain trade run my way.
- **What I priced.** Smooth path to 3.25%, three 2026 dots, tiny expected move.
- **My variant and its odds.** Pace ~50 bp too low; 30% on the bull (slow-pace) branch versus the lower priced odds.
- **The expression and why.** 2s10s steepener at \$125 per bp, half unit (\$1,250 risk), because it isolated the pace variant and netted out the consensus directional move — *not* gold, which my own variant pointed the wrong way for.
- **The outcome.** Bull branch hit; dots showed a pause; curve steepened 21 bp; +\$2,600 gross; added to a full unit on the posterior update.
- **Process grade: A.** The tree was right, the size was disciplined (half unit on a binary), the expression was clean, and I executed the pre-committed action calmly with no adrenaline-driven decision. The good outcome was *not* the reason for the A — the process earned it, and the same process would have earned an A even on a stopped-out hawkish branch.

That last line is the entire philosophy of the series compressed to a sentence. We judge decisions, not results — because over a career the process is the only thing you control, and a sound process with a 30% branch that happened to hit is worth exactly as much, in learning, as a sound process with a 25% branch that happened to stop out. The full argument for grading process over outcome is in [process versus outcome](/blog/trading/analyst-edge/process-versus-outcome-judging-decisions-not-results), and the scorekeeping that lets you check whether your probabilities are any good over time is in [calibration: keeping score on your own forecasts](/blog/trading/analyst-edge/calibration-keeping-score-on-your-own-forecasts).

## The playbook: the FOMC-day process checklist

The whole pipeline, compressed to an event-day routine you can run on any scheduled catalyst — an FOMC, a CPI print, an earnings report. Run it in order, finish every step *before* the print, and execute mechanically when the number lands.

**The week before (cold, calm, blank page):**

1. **Filter the diet.** Identify the four-to-six inputs that bear on the *non-priced* part of the decision (here, the pace). Mute the rest.
2. **Run the six lenses.** Read macro, fundamental, technical, flows, positioning, sentiment. Write the synthesis in one line. Note where they disagree — that is where the edge hides.
3. **Read what's priced.** Pull the implied path (futures), the dot-plot expectation, and the options expected move. Write the consensus down as specific numbers.
4. **Name the variant.** State, in one sentence, where you differ from a specific priced number. If you can't, there is no trade — stand aside.
5. **Write the one-sentence thesis.** Forecast + reason + catalyst, falsifiable.
6. **Build three scenarios** with probabilities summing to 100%, carved around your variant, not the generic direction.
7. **Draw the decision tree.** Estimate each branch's reaction, pre-commit an action, compute the dollar payoff, roll the EV back. Compare to the flat \$0 leaf — does it clear?
8. **Map conviction to size.** Apply the binary discount on a single-instant catalyst. Set the dollar risk; make the stop equal the size.
9. **Choose the expression.** List the cross-asset candidates; pick the instrument that *isolates your variant* and nets out the consensus. Check the sign — make sure the asset moves *your* way in *your* bull branch.
10. **Fill the one-page note.** Nine fields, dated, no blanks. Write the EV check at the bottom.
11. **Define the invalidation.** In writing: the thesis-based exit (the reason is gone) and the price-based stop. Make them precise.

**The day of (execution only, no new analysis):**

12. **Set up the dashboard.** The two or three markers you will actually watch (here: the curve on the tape, the 2026 dot, the presser tone on pace).
13. **Execute the branch.** When the print lands, read which branch the world chose, match it to your tree, and execute the pre-committed action. Decide *nothing* new in the hot moment.

**The evening after (compound the learning):**

14. **Update like a Bayesian.** Move your posterior by the diagnostic weight of the print. Re-size: if the binary risk resolved and the posterior rose, add toward a full unit.
15. **Write the journal entry.** Record the decision and grade the *process*, not the P/L. A sound process on a branch that stopped out still earns the A.

That is the machine, end to end, on one decision. The cut was priced and boring; the edge was in the pace, the expression, and the discipline of deciding cold. Run this routine on every catalyst and two things happen over a career: you stop making five-figure decisions under adrenaline, and your journal slowly tells you — honestly, trade by trade — whether your process actually has an edge. That feedback loop, not any single FOMC, is the analyst's edge.

## Further reading & cross-links

This case study chains the entire series. To go deep on any single stage:

- **Information & reading the tape:** [building your information diet](/blog/trading/analyst-edge/building-your-information-diet-signal-versus-noise) · [the six lenses](/blog/trading/analyst-edge/the-six-lenses-a-framework-for-reading-any-market) · [mapping the consensus](/blog/trading/analyst-edge/mapping-the-consensus-what-does-the-market-already-believe)
- **Finding the edge:** [what's priced in](/blog/trading/analyst-edge/whats-priced-in-the-question-behind-every-trade) · [variant perception](/blog/trading/analyst-edge/variant-perception-where-real-edge-comes-from) · [thesis vs forecast](/blog/trading/analyst-edge/thesis-vs-forecast-the-anatomy-of-a-tradeable-view)
- **Quantifying it:** [base, bull, and bear](/blog/trading/analyst-edge/base-bull-and-bear-building-three-scenarios) · [decision trees](/blog/trading/analyst-edge/decision-trees-for-event-driven-views) · [expected value](/blog/trading/analyst-edge/expected-value-the-only-math-a-view-really-needs)
- **Sizing & expressing it:** [from conviction to size](/blog/trading/analyst-edge/from-conviction-to-size-the-bet-sizing-bridge) · [asymmetry and the high-conviction bet](/blog/trading/analyst-edge/asymmetry-and-the-art-of-the-high-conviction-bet) · [choosing the instrument](/blog/trading/analyst-edge/choosing-the-instrument-to-express-your-thesis) · [relative value](/blog/trading/analyst-edge/relative-value-expressing-a-view-without-a-directional-bet)
- **Writing it down & managing it:** [the trade-idea note](/blog/trading/analyst-edge/the-trade-idea-note-a-one-page-template) · [what would change my mind](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront) · [monitoring a live thesis](/blog/trading/analyst-edge/monitoring-a-live-thesis-building-your-watch-dashboard) · [when to add, cut, or exit](/blog/trading/analyst-edge/when-to-add-cut-or-exit-managing-the-position-around-the-view)
- **Closing the loop:** [updating like a Bayesian](/blog/trading/analyst-edge/updating-on-new-information-thinking-like-a-bayesian) · [the decision journal](/blog/trading/analyst-edge/the-decision-journal-the-habit-that-compounds) · [process versus outcome](/blog/trading/analyst-edge/process-versus-outcome-judging-decisions-not-results) · [calibration](/blog/trading/analyst-edge/calibration-keeping-score-on-your-own-forecasts)
- **The mechanisms (out of series):** [consensus expectations and priced-in](/blog/trading/event-trading/consensus-expectations-and-priced-in) · [the reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently) · [positioning and the pain trade](/blog/trading/event-trading/positioning-and-the-pain-trade) · [the surprise, not the level](/blog/trading/macro-correlations/the-surprise-not-the-level-betas-to-data-surprises) · [how an options market maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) · [risk management: the only free lunch](/blog/trading/risk-management/risk-management-the-only-free-lunch-survival-as-a-compounding-engine)

---
title: "Confirmation Bias and Motivated Reasoning: How Your Thesis Defends Itself"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "How a held position quietly rewrites what you notice, remember, and weigh — the science of confirmation bias and motivated reasoning, the P&L it destroys, and a mechanical drill to break the loop."
tags: ["trading-psychology", "confirmation-bias", "motivated-reasoning", "behavioral-finance", "cognitive-bias", "wason-selection-task", "kunda", "belief-entrenchment", "falsification", "risk-management", "decision-making"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — The moment you take a position, your mind stops *testing* the thesis and starts *defending* it. You seek the news that agrees, discount the news that doesn't, and grow more certain as the evidence gets more mixed. The thesis literally defends itself.
>
> - **Confirmation bias** (Peter Wason, 1960) is testing to *confirm* rather than to *falsify*; **motivated reasoning** (Ziva Kunda, 1990) is reasoning toward the conclusion you *want* and then building the justification. Fused, they produce a self-sealing belief.
> - The mechanism is **asymmetric weighting**: confirming evidence gets near-full weight and sizes you *up*; disconfirming evidence gets discounted to a fraction and explained away. A useful stand-in is roughly five to one.
> - Challenging a shaky thesis often *strengthens* it — **biased assimilation and attitude polarization** (Lord, Ross & Lepper, 1979) — which is why "the market will prove me right" feels truer the longer it doesn't.
> - The P&L signature is unmistakable: you add only on confirming news, so a small loss compounds into a large one. Bill Miller's fund fell **55%** in 2008 riding financials down; Bill Ackman lost close to **\$1 billion** defending a public short for five years.
> - The one habit to build: before you add to a winner, write the three strongest reasons you are wrong and the exact price that would prove it. **If no price could prove you wrong, that is the signal to stop.**

Here is a small experiment you can run on yourself. Think of the last position you held with real conviction. Now try to remember the single best argument *against* it — the strongest bear case if you were long, the strongest bull case if you were short. Not a strawman. The one that would have genuinely scared you.

Most traders can't do it. They can recite fifteen reasons the trade was good and draw a blank on the reasons it was bad — not because the bad reasons didn't exist, but because their mind quietly stopped storing them the day the position went on. That is not a memory failure. It is the position doing exactly what a position does: changing what you perceive, what you recall, and what you believe, all in the direction of the trade you already have.

This article is about the two mental machines that make that happen, and how they combine into something more dangerous than either alone: a thesis that defends itself. The diagram below is the mental model the whole piece is built on. Read it as a loop, not a list.

![The confirmation loop: you take a position, seek evidence you are right, find it and dismiss what disagrees, and grow more convinced, which sends you back around](/imgs/blogs/confirmation-bias-and-motivated-reasoning-1.webp)

You take a position. Now you have a stake in being right, so you go looking for evidence that you *are* right — you open the bullish feeds, you follow the accounts that agree, you read the earnings call for the good lines. You find that evidence, because in a noisy market you can always find *some*, and you dismiss what disagrees as "noise." Your conviction rises. So you add, and you hold longer. And a trader with more conviction and a bigger position seeks confirmation even harder. Each trip around the loop tightens it. The center box is the punchline: after enough laps, the position has changed what you perceive and remember, and the bar for any evidence that could talk you out of it has quietly risen out of reach.

The goal here is not to make you feel bad about a universal feature of the human mind. It's to make the loop *visible* — to give you the science of why it runs, the exact places it touches your P&L, and a concrete drill that jams it before it costs you. We'll build every idea from zero, ground each one in a worked example with real numbers, and end with four market episodes where this exact loop, at scale, blew up funds you've heard of.

## Foundations: the building blocks

You need no psychology background for this. We'll define the four ideas the rest of the article stands on — confirmation bias, falsification, motivated reasoning, and belief entrenchment — one at a time, starting from a card game a psychologist invented in 1960 that still predicts how you'll trade.

### What confirmation bias actually is

*Confirmation bias* is the tendency to seek, interpret, and remember information in a way that favors what you already believe. The cleanest definition in the literature comes from Raymond Nickerson's landmark 1998 review, which called it "the seeking or interpreting of evidence in ways that are partial to existing beliefs, expectations, or a hypothesis in hand." Nickerson's title is worth remembering because it's the whole warning in six words: *Confirmation Bias: A Ubiquitous Phenomenon in Many Guises.* It is not a rare glitch that afflicts weak minds. It is a default setting of every mind, dressed up in a hundred different costumes.

The word doing the heavy lifting is *seeking*. Confirmation bias is not mainly about lying to yourself once the facts are in. It operates one step earlier, on *which facts you go and get*. A biased mind and an unbiased mind can look at the same world and end up with different databases, because they searched it differently.

It helps to understand *why* the mind is built this way, because the bias is not a random defect — it's a feature that usually earns its keep. Two forces install it. The first is cognitive economy: testing every belief against every possible disconfirmation would be paralyzing, so the mind defaults to treating a working belief as true and looking for reasons to keep using it. That's efficient in a stable world where your beliefs are mostly right. The second is social and motivational: humans evolved to defend commitments and stay consistent, because a person who flip-flops on every contrary signal is neither trustworthy nor persuasive. Holding your ground and marshaling evidence for your side was, for most of human history, a *social advantage*. The catch is that markets invert the payoff. They reward exactly the trait these instincts suppress — cheap, fast surrender of a wrong belief — and punish the consistency and commitment that served your ancestors. Confirmation bias isn't you being irrational; it's a well-adapted mind running in the one environment specifically designed to exploit its defaults.

The founding demonstration is a deceptively simple task designed by the British psychologist Peter Wason.

### The 2-4-6 task: testing to confirm hides the rule

In 1960, Wason published a paper with a title that is itself a thesis: "On the Failure to Eliminate Hypotheses in a Conceptual Task." He sat people down and told them he had a rule in mind that generated three-number sequences. He gave them one example that fit the rule: **2, 4, 6.** Their job was to discover the rule by proposing their own triples. For each triple they offered, Wason would say only "yes, that fits the rule" or "no, it doesn't." When they were confident, they announced the rule.

Almost everyone did the same thing. Looking at 2-4-6, they formed a hypothesis — "even numbers going up by two" — and then proposed triples *designed to get a yes*: 8, 10, 12. Yes. 20, 22, 24. Yes. 100, 102, 104. Yes. After a handful of confirmations, they announced their rule with confidence.

And they were usually wrong. The actual rule was **any three numbers in ascending order.** "2, 4, 6" fits it, but so does "1, 2, 3" and "5, 40, 900." The only way to *discover* the real rule is to propose a triple you expect to *fail* — to try 2, 4, 5, or even 6, 4, 2, and watch what happens. A triple that breaks your hypothesis but still gets a "yes" is worth a hundred that confirm it, because it's the only kind that teaches you something.

![Wason's 2-4-6 task: people propose triples they expect to pass and announce the wrong rule; the triples that could fail reveal the true rule](/imgs/blogs/confirmation-bias-and-motivated-reasoning-2.webp)

The figure lays the two strategies side by side. On the left is what most people do — a chain of confirmations (8-10-12, 20-22-24) leading to a confident, wrong answer. On the right is the falsifying strategy: try 2, 4, 5, which *breaks* the "+2" idea but still returns "yes," and the true rule ("any ascending numbers") suddenly comes into view. In Wason's original run, only about **one in five** subjects — 6 of 29 — found the rule cleanly on the first announcement. The rest confirmed themselves into an error.

Wason later built an even more famous version, the *selection task* (mid-1960s), where people must decide which of four cards to turn over to test a conditional rule like "if a card shows an even number on one face, its opposite face is red." The logically correct answer is to check the cases that could *break* the rule. Across decades of studies, typically **under 20%** of people pick the right cards on the abstract version — and on the classic form, often **under 10%**. The failure is not arithmetic. It's that people instinctively look for confirmations and never think to look for the disconfirmation that would actually settle the question.

> The only test that teaches you anything is one that could say "no." A test you already know will say "yes" is not evidence — it's a ritual.

Now translate the game into your P&L. Your trading thesis *is* a rule you've inferred from a few examples: "this stock goes up when rates fall," "this breakout pattern works," "management always beats guidance." And the 2-4-6 result says you will instinctively test that rule by looking for cases that confirm it — the days it went up when rates fell — and almost never propose the trade or scan the data that could prove it false. You will confirm your way into a rule that isn't real.

#### Worked example: your thesis as a 2-4-6 game

Suppose you believe: "Stock ABC rallies whenever the sector ETF is green." You've seen it happen three times. To *confirm* it, you do what comes naturally: on the next three green-ETF days you check ABC, and it's up all three. Six for six. You size up, certain you've found an edge.

But you never ran the disconfirming test — the one that could say "no." How does ABC behave on the days the ETF is *red*? If you'd checked, you might have found it's up 60% of those days too, because ABC just drifts up in a bull market regardless of the ETF. Your "edge" was ascending numbers all along: the sector had nothing to do with it. The confirming evidence was real and it was worthless, because you only ever collected the kind that couldn't fail. **The intuition: confirming instances feel like proof and carry almost no information; the trade that could have proven you wrong is the only one that would have taught you anything.**

### Motivated reasoning: reasoning toward the answer you want

Confirmation bias explains how a *neutral* observer with a hypothesis goes astray. But traders are not neutral. Once money is on the line, you don't just have a hypothesis — you have a *wish*. You want the thesis to be true, because your account depends on it. That wish bends the reasoning, and the psychologist who mapped how is Ziva Kunda.

Kunda's 1990 paper "The Case for Motivated Reasoning" (one of the most-cited papers in social psychology, with well over 9,000 citations) drew a careful line. She argued that motivation doesn't let you believe *anything* you want — you can't simply decide the stock will go up and feel it. Instead, motivation works through the *cognitive machinery*: it biases which memories you access, which rules of inference you apply, and which evidence you count as good, all in service of a conclusion you can then defend as reasonable. In her framing, we reason toward a desired conclusion but are "constrained by our ability to construct seemingly reasonable justifications." You need a story you could tell a skeptic. Motivated reasoning is the search for that story — and the search is rigged.

The tell is a double standard for evidence, which Kunda and others captured in a famous question: people apply the test **"Can I believe this?"** to evidence they *like*, and **"Must I believe this?"** to evidence they *don't*. Good news needs only to be *possible* to be accepted. Bad news must be *airtight* to get past the gate, and almost nothing about a market is airtight, so bad news rarely makes it through.

Confirmation bias and motivated reasoning are cousins, and it's worth keeping them straight:

| | Confirmation bias | Motivated reasoning |
|---|---|---|
| **The driver** | A hypothesis you're holding | A conclusion you *want* to be true |
| **The mechanism** | Seek/interpret evidence to fit the belief | Bias memory and inference toward the wish |
| **The tell** | "Look for the yes, skip the no" | "Can I believe the good, must I believe the bad" |
| **In trading** | You read only the bullish takes | You *need* it up because you're long and levered |
| **The source** | Wason (1960); Nickerson (1998) | Kunda (1990) |

In a real trade they arrive fused. You hold a position (confirmation bias has a hypothesis to protect) *and* you need it to work (motivated reasoning has a wish to serve). The combination is what makes the loop in the mental-model figure so hard to break: one machine curates the inputs, the other launders the reasoning, and the output is a conclusion that feels like sober analysis and is actually the position talking.

### Belief entrenchment: why challenges backfire

The last building block is the most counterintuitive, and the most expensive. You'd think that showing a person evidence against their belief would soften it. Often it does the opposite.

In 1979, Charles Lord, Lee Ross, and Mark Lepper ran a study that became a classic. They took 48 people with strong prior views on capital punishment — half in favor, half against — and showed *all of them the same two fictional research studies*: one suggesting the death penalty deters crime, one suggesting it doesn't. Identical packets. If evidence moved beliefs the way we imagine, both groups should have converged a little toward the middle.

The reverse happened. Each side rated the study that *agreed* with them as well-designed and convincing, and picked apart the study that *disagreed* as methodologically flawed — the same designs, judged oppositely depending on which conclusion they supported. This is *biased assimilation*: the identical evidence is absorbed differently by different priors. And then something worse: when asked afterward, both groups reported being *more* extreme in their original view than before. Mixed evidence had *polarized* them. Lord, Ross and Lepper called it *attitude polarization*, and it means a belief can grow stronger from the very information that should have weakened it.

For a trader, this is the mechanism behind the phrase "the market will prove me right." Every ambiguous tape, every mixed print, gets assimilated as support — the good parts confirm you, the bad parts are dismissed as noise or manipulation — so a stream of genuinely balanced evidence leaves you *more* convinced, not less. The longer a bad position argues with you, the more certain you can become that it's about to work. We'll put a chart on that later; for now, just hold the shape: challenges don't automatically weaken a thesis. Often they entrench it.

That is the whole toolkit — confirmation bias, falsification, motivated reasoning, and entrenchment. Everything below is these four ideas colliding with a live position.

## The four screens: how a position rewrites what you see

Confirmation bias doesn't strike once. It operates at every stage of processing a piece of information, and a held position tilts each stage the same direction. Think of your mind as a series of screens the world's data passes through on its way to becoming a decision — and picture the position as a thumb on the scale at every screen.

![Confirmation bias operates at every screen of one decision: perception, memory, judgment, and action, each tilted the same way, locking in an entrenched thesis](/imgs/blogs/confirmation-bias-and-motivated-reasoning-3.webp)

Follow the figure through the four screens.

**Perception** is the first screen: what you even notice. Before you can misweigh a data point, it has to enter your awareness, and a position changes what does. When you're long, the bullish headline jumps off the page and the bearish one slides by unread. You're not lying — you genuinely *saw* less of the disconfirming data, because attention is a spotlight and the position aims it. This is where confirmation bias is cheapest and most invisible: the evidence against you was on the screen and you never registered it.

**Memory** is the second screen: what you recall. Your brain is not a hard drive; it's a storyteller that reconstructs the past to fit the present. Hold a winning thesis and you'll vividly recall the times this setup worked and quietly lose the times it didn't. The database of experience you draw on to size the next trade has been silently edited toward your current view. Ask a losing trader for their track record on a given pattern and you'll get the highlight reel, not the tape.

**Judgment** is the third screen and the crowded one: how you weigh what survived the first two. This is where motivated reasoning does its work — the "can I believe it / must I believe it" double standard. Confirming data gets waved through at near-full weight; disconfirming data has to clear a bar so high that almost nothing does. Later we'll put a rough number on the asymmetry — call it five to one.

**Action** is the fourth screen: what you actually do. The tilt at the first three screens funnels into a lopsided behavior — you *add* on agreement and *freeze* on doubt. Confirming news makes you size up; disconfirming news makes you do nothing, or rationalize, rather than cut. The asymmetry of *belief* becomes an asymmetry of *position*, which is where it finally touches money.

The rightmost box is the output: an **entrenched** thesis, with the bar to change your mind ratcheted higher than when you started. The four screens don't just distort one decision; their output feeds back — a more entrenched view tilts the next round of perception even harder. That feedback is the loop from the first figure, drawn as a pipeline.

The important move here is to stop treating "confirmation bias" as one thing you can will away and start seeing it as four separate leaks, each with its own patch. You can't un-bias perception by trying harder to be fair — but you *can* mechanically force yourself to read the bear case (patch the perception screen), keep a written trade journal so memory can't edit itself (patch the memory screen), and pre-commit weights and exits (patch the judgment and action screens). We'll build those patches into a drill at the end. First, the screen that costs the most: judgment.

## Motivated reasoning at the judgment screen: the asymmetry engine

The judgment screen is where the position turns evidence into a decision, and its defining feature is a double standard. To *see* the double standard, put a number on it. Imagine every new data point about your position gets assigned a "weight" — how much it moves your conviction and, through it, your sizing. A fair mind assigns weight by the *quality* of the evidence, the same scale for good news and bad. A held position does not.

![You apply roughly five times the weight to news that agrees: confirming evidence near full weight, disconfirming evidence discounted, versus a fair mind's equal weighting](/imgs/blogs/confirmation-bias-and-motivated-reasoning-5.webp)

The figure is a rough model, not a measured constant — I've drawn confirming evidence at near-full weight, disconfirming evidence at about a fifth of that, and a fair mind weighting either at the same middling level set by evidence quality. The exact ratio isn't the point; the *asymmetry* is. Motivated reasoning means agreement is cheap to accept and disagreement is expensive, and that gap — call it roughly five to one — is the engine that sizes you up on confirming news and leaves you paralyzed on the disconfirming news that should be shrinking the position.

Watch what the asymmetry does to a perfectly balanced stream of information.

#### Worked example: the asymmetric-weight ledger

Suppose over a month your long position throws off ten news items, and they're genuinely balanced: five support your thesis, five contradict it. A fair reading nets to zero — five for, five against, no reason to grow more bullish. You should be exactly as convinced at month-end as month-start.

Now run the same ten items through the five-to-one weighting. The five confirming items land at full weight: 5 × 1.0 = +5.0 units of bullishness. The five disconfirming items get discounted to a fifth: 5 × 0.2 = 1.0 unit of bearishness. Net: +5.0 − 1.0 = **+4.0**, strongly bullish. The identical, balanced evidence that *should* have left you neutral has instead made you materially more convinced — and because conviction drives sizing, you've probably added. **The intuition: motivated reasoning doesn't require biased information; it manufactures a bullish conclusion out of perfectly neutral information by weighting the two halves differently.**

That single mechanism — different weights for agreement and disagreement — reappears at every stage of handling a fact. It's worth seeing all at once, because it's the anatomy of the whole bias.

![The asymmetry: confirming and disconfirming evidence handled by opposite rules at seeking, interpreting, remembering, weighting, and acting](/imgs/blogs/confirmation-bias-and-motivated-reasoning-8.webp)

Read the matrix row by row. When you're **seeking**, you hunt out confirming evidence — you curate a feed of bulls when you're long — and you never go looking for the bear case. When you're **interpreting**, confirming news is taken at face value while disconfirming news gets relabeled "noise," "the market's wrong," or "manipulation." When you're **remembering**, the confirming episodes stay vivid and rehearsed while the disconfirming ones fade. When you're **weighting**, agreement gets full weight and you size up; disagreement gets a fifth and is explained away. And when you're **acting**, confirmation means add and hold, while disconfirmation means freeze — or, worse, *move the goalposts* so the disconfirming fact no longer counts ("I'm a long-term investor now"). Five stages, one asymmetry, applied consistently. That consistency is why the output feels so coherent: every screen agreed, because every screen was tilted by the same thumb.

The reason this is so hard to catch in the moment is that none of it feels like bias. Each individual step feels like *judgment*. Of course you discount the bearish tweet — it's from a permabear with an axe to grind. Of course you weight the strong earnings — the numbers are the numbers. Each rationalization is locally reasonable. It's only when you notice that *every* call went the same way — that you have never once, in this position, weighted a bearish fact heavily — that the pattern gives itself away.

## Two traders, one tape: how priors flip the read

If confirmation bias and motivated reasoning are real, they make a sharp, testable prediction: two traders looking at the *identical* piece of news should reach *opposite* conclusions if they hold opposite positions — and each should feel completely objective. That is exactly what happens, every earnings season, on every ambiguous print.

![Two traders read the same report in opposite directions because of the prior each held, and the P&L follows the prior, not the news](/imgs/blogs/confirmation-bias-and-motivated-reasoning-4.webp)

The figure traces both paths from one headline. Follow the fork.

#### Worked example: same report, opposite trades

Stock XYZ trades at \$50. It reports: revenue up 8% year-over-year, but management cuts full-year guidance by about 5%. Genuinely mixed — a good quarter and a cautious outlook.

**Trader A is already long** 1,000 shares from \$50. Motivated reasoning goes to work: the +8% "is the story," the guidance cut "is just management being conservative so they can beat later." A reads the print as confirmation, and adds — buys 1,000 more at \$52. Average cost is now \$51 on 2,000 shares.

**Trader B is already short** 1,000 shares from \$50. The same print reads inverted: the guidance cut "is the tell — management sees weakness," the +8% "is backward-looking, the market trades the future." B reads it as confirmation too, and adds — sells 1,000 more short at \$52. Average cost \$51 on 2,000 shares short.

Same headline. Opposite trades. Now the tape resolves: over the next few weeks the guidance cut wins the argument and XYZ drifts to \$44.

- **Trader A** is long 2,000 at an average of \$51. At \$44 that's a loss of (\$51 − \$44) × 2,000 = **−\$14,000.** Had A *not* added on the confirming read — just held the original 1,000 from \$50 — the loss would have been (\$50 − \$44) × 1,000 = **−\$6,000.** Sizing up on confirmation more than doubled the damage.
- **Trader B** is short 2,000 at an average of \$51. At \$44 that's a gain of (\$51 − \$44) × 2,000 = **+\$14,000.**

**The intuition: the report didn't decide the P&L — the prior did. The same objective facts became a reason to add for both traders, and the only difference in outcome was which position each already owned.** The market is a machine for showing you evidence that confirms whatever you already think, because it is always emitting evidence in both directions and you are always choosing which half to weight.

Notice the specific damage in Trader A's path: the loss didn't come from being wrong about XYZ. Being wrong cost \$6,000. The extra \$8,000 came from *adding on confirming news* — from letting the confirmation loop turn a small, survivable mistake into a large one. This is the single most expensive habit the bias produces, and it deserves its own name: **you only ever size up in the direction of your existing conviction.** Good news is a reason to add. Bad news is never a reason to trim; it's a reason to explain. So positions only grow when they're winning the argument in your head, which is precisely when they're most dangerous.

## Belief entrenchment: why the losing trade feels most certain

Here's the cruelest turn. You might hope that as a position moves against you and the disconfirming evidence piles up, the sheer weight of it would eventually break through. Lord, Ross and Lepper's 1979 result says the opposite can happen: mixed evidence *polarizes*. The more the market argues with you, the more certain you can become.

![More mixed evidence, more conviction: a biased mind polarizes upward while a fair mind converges to what the evidence supports](/imgs/blogs/confirmation-bias-and-motivated-reasoning-6.webp)

The chart contrasts two minds fed the *same* stream of balanced evidence — half supporting the thesis, half contradicting it — as it accumulates left to right. The **fair mind** converges: with each new mixed data point it settles toward what the evidence actually supports, hovering near the middle, staying anchored in the "evidence-anchored zone" where doubt stays healthy. The **biased mind** does the opposite. Through biased assimilation, it counts the supporting half and discounts the contradicting half, so every new *batch* of mixed evidence nudges conviction *up*. The line climbs into the "entrenchment zone," where conviction has outrun the evidence entirely.

This is why a trader can be *most* certain at the exact moment they should be least — deep in a losing position, having "done the work," having "looked at all the arguments." They did look at all the arguments. They assimilated each one through the five-to-one filter, and every lap of the confirmation loop moved them further into the entrenchment zone. The conviction is real. It's just manufactured, and it's inversely related to the truth.

#### Worked example: the account that got more sure as it got more wrong

Imagine a trader long a stock at \$100 with a \$20,000 position. The stock falls to \$90; along the way, ten pieces of news arrive, five bad and five good. A fair mind would net them near zero and, seeing the price action, at least entertain that the thesis is broken. The entrenched trader assimilates the five good ones at full weight and the five bad ones at a fifth — nets to strongly bullish, exactly as in the ledger example — and concludes the \$10 drop is a *gift*. So they average down, buying \$20,000 more at \$90. Now they hold \$40,000 with an average cost near \$95, and *more* conviction than at \$100, because they've "seen the evidence and it supports the thesis." The stock falls to \$80. The loss is now (\$95 − \$80) / \$95 ≈ 16% on \$40,000 ≈ **\$6,400** — larger in dollars than the entire original position was ever down, and held with more certainty than ever. **The intuition: entrenchment makes averaging down feel like conviction when it's really the confirmation loop converting disconfirming evidence into fuel.**

Averaging down is not always wrong — a genuine value investor with a falsifiable thesis and a real edge can do it deliberately. The problem is that entrenchment produces the *feeling* of that disciplined conviction without any of the underlying discipline. It feels identical from the inside. The only way to tell them apart is from the outside — with a rule written before the position moved. Which is the whole point of the drill.

## What it looks like at the screen

Everything above is the theory. Here is what the confirmation loop actually feels like in real time, at the desk, because you will not catch it by remembering a definition — you'll catch it by recognizing the *behavior* while you're doing it.

You're long, and the position is green. You refresh the chart every few minutes, and it feels good, and you notice you keep going back to the *same* three bullish accounts on your feed — the ones who are also long. You've stopped reading the analyst who downgraded it last week; you scrolled past their latest note without opening it, and if asked, you'd say it "wasn't worth reading." You open the earnings transcript and your eye goes straight to the revenue beat; you skim the section on margin compression. When a bearish headline crosses, you feel a small spike of irritation and you immediately reach for the counterargument — before you've even finished reading the headline, you're composing the rebuttal. That reflex, the pre-loaded rebuttal, is motivated reasoning caught in the act.

Now the position turns red. Watch what changes. You stop refreshing as often — there's a specific reluctance to look at the tab, a physical flinch away from the number. When you do look, you find yourself widening the timeframe on the chart: the 5-minute looked ugly, so you flip to the daily, then the weekly, until you find the chart on which the trade still looks fine. That zoom-out is a tell. You start saying "I'm a longer-term holder anyway" about a position you put on as a swing trade — the goalposts moving in real time. You seek out the one bull who's still loud and you mute or unfollow the bears, calling it "cleaning up my feed." You explain the price action rather than respond to it: "it's just a shakeout," "weak hands getting flushed," "the market's wrong here, they'll figure it out." You catch yourself arguing *with the tape* — literally narrating why the price is incorrect.

And the physical signatures are consistent enough to use as an alarm. A pre-loaded rebuttal to bad news before you've read it. A widening timeframe. A shrinking, friendlier feed. The phrase "the market is wrong." A story for every red candle and no story you'd accept for closing the trade. When you notice two or three of those at once, you are not analyzing — you are defending. The tell is not the content of any single thought; it's the *direction*. Every thought is going the same way. A mind actually weighing evidence produces thoughts that cut both ways. A mind in the loop produces a stream that all points at "I'm right, hold." When your internal monologue has that uniform direction, stop, because the analysis is already over — the position is doing the thinking.

## The disconfirmation drill

You cannot delete confirmation bias. It's not a bug you patch once; it's the default mode of a mind that evolved to defend commitments, and it reboots with every new position. What you *can* do is build a mechanical process that forces the disconfirming step the mind will never take on its own. The 2-4-6 lesson is the whole strategy in one line: **you have to deliberately propose the test that could say "no,"** because you will never do it by instinct.

Here is the drill. Run it before you add to any winning position — the exact moment the loop is strongest and the urge to size up feels most like insight.

![The disconfirmation drill: six fixed steps before you add to a winner, turning an unfalsifiable thesis into a bounded, pre-committed risk](/imgs/blogs/confirmation-bias-and-motivated-reasoning-7.webp)

The steps, in order.

**1. Write the three strongest reasons you are WRONG.** Not weak strawmen you can knock down — the three arguments that would genuinely scare you, written as if by the smartest bear you know. If you can't produce three real ones, you haven't done the work; you've done the confirmation loop. This single step patches the perception and memory screens by *forcing* the disconfirming evidence into awareness, where the position has been keeping it out.

**2. Name the exact price or datum that proves you wrong.** This is the load-bearing step, and it's a direct application of falsification. What specific, observable event would tell you the thesis is broken — a price level, an earnings miss below a number, a guidance cut, a spread blowing past a threshold? Write it down as a concrete trigger, not a feeling. "If it closes below \$48" is a falsifier. "If it starts looking weak" is not.

**3. Appoint a red-team bear on your own thesis.** Assign someone — a colleague, a friend who trades, or a disciplined version of yourself in a separate journal entry — the explicit job of arguing the *other* side, hard. The value isn't the debate; it's that a red-teamer isn't protected by *your* motivated reasoning. They'll weight your disconfirming evidence at full weight, because they have no position to defend. If you trade alone, this is why a written journal beats a mental note: the version of you writing the bear case tomorrow morning, flat and calm, is a different, more honest analyst than the version holding the position now.

**4. Pre-commit the invalidation exit in writing.** Turn step 2 into an order. The price that would prove you wrong becomes your stop, entered or at least written and dated *before* you add. Pre-commitment is the whole trick: you're using the calm, fair-minded version of yourself to bind the future, entrenched version who will want to move the goalposts. A written invalidation level is a message from the you who could still think straight.

**5 & 6. Then — and only then — size the add, or stand down.** With the bear case forced into view, a falsifier named, a red-teamer's objections heard, and an exit pre-committed, you've converted an unfalsifiable, self-defending thesis into a bounded risk. Now you can add. Or you'll discover, running the drill, that you *can't* name a price that would prove you wrong — and that discovery is the most valuable output of all. A thesis with no possible falsifier isn't a strong conviction; it's the confirmation loop wearing conviction's clothes. When there's no price that could change your mind, the correct size is not "bigger." It's zero.

#### Worked example: the pre-mortem that saves \$12,000

Go back to Trader A, long 1,000 shares of XYZ from \$50, tempted to add at \$52 after the mixed report. Suppose A runs the drill first. Step 1 forces the bear case onto the page: "the guidance cut signals real demand weakness the +8% doesn't capture." Step 2 demands a falsifier, and A writes it: "I'm wrong if XYZ closes below \$48 on the guidance news." Step 4 turns that into a pre-committed stop at \$48, written before any add.

Running step 5, A can't honestly argue that piling on more risk *improves* the setup — so A stands down: no add, just the original 1,000 shares with the stop in place. Now the tape does what it did before — XYZ heads toward \$44. But this time, at \$48, the pre-committed stop fires. A is out at \$48 with a loss of (\$50 − \$48) × 1,000 = **−\$2,000**, instead of the **−\$14,000** from sizing up on confirmation and riding to \$44. **The intuition: the drill doesn't make you right about XYZ — it makes being wrong cost \$2,000 instead of \$14,000, by pre-committing the disconfirming exit before entrenchment could veto it.** That \$12,000 gap is the entire edge of trading with a written invalidation over trading with a defended thesis.

One caution about the red-team step, because it interacts with the entrenchment result. If you hand your thesis to a bear and then spend the conversation *rebutting* them, you've just run biased assimilation with extra steps — you'll come away more convinced, exactly as Lord, Ross and Lepper predicted. The drill only works if you red-team to *update*, not to win. The test isn't "can I answer the bear?" It's "did the bear give me a reason to lower my size or tighten my stop?" If the honest answer is never, every time, across every position, that's not because you're always right. It's the loop.

## Common misconceptions

**"I'm too experienced / too smart to fall for confirmation bias."** Backwards. Expertise gives you *more* raw material to build a "seemingly reasonable justification," which is exactly what Kunda showed motivated reasoning needs. Smarter, more knowledgeable people are often *better* at rationalizing, because they can generate more sophisticated counterarguments to disconfirming evidence. The bias isn't a function of IQ; it's a function of having a position. Anyone with a stake has the wish, and the wish does the bending.

**"Doing more research protects me."** Only if the research is *disconfirming*. More research through the confirmation loop just means more confirming evidence collected and more disconfirming evidence explained away — it deepens the entrenchment. The 2-4-6 task is the proof: the subjects who tested the most triples weren't the ones who found the rule; the ones who found it were the ones who tested a *falsifying* triple. Volume of research is not the variable. Direction of research is.

**"If I just stay objective and look at both sides, I'm fine."** You cannot introspect your way out, because the bias operates on perception and memory *before* the evidence reaches the part of you that's trying to be fair. By the time you sit down to weigh "both sides," the sides have already been curated — you're weighing the confirming evidence you noticed and remembered against the disconfirming evidence that survived the filter, which isn't both sides at all. Objectivity as an *intention* fails. Objectivity as a *procedure* — a written falsifier, a pre-committed exit — is the only kind that works.

**"Confirmation bias just means I ignore evidence against me."** It's broader and sneakier than ignoring. You often *engage* disconfirming evidence — you read the bear note, you consider the risk — and then assimilate it in a way that leaves you *more* convinced, per attitude polarization. The dangerous version isn't the trader who won't look at the bear case. It's the one who looks at it, argues it down, and walks away more certain. Engagement is not immunity.

**"Cutting losses fixes this."** A stop-loss helps, but only if you set it *before* entrenchment sets in and then don't move it. The bias's signature move is to *widen* the stop as the position moves against you — "I'll give it more room" — which is the goalpost-moving from the matrix, dressed as flexibility. The discipline isn't having a stop; it's honoring a stop you wrote when you could still think clearly. An invalidation level you're willing to move is not an invalidation level. It's a suggestion the loop will overrule.

## How it shows up in real markets

Four episodes, each a case study in the confirmation loop running at a scale you can see in the P&L. Real names, real dates, real numbers.

### 1. Bill Miller and the value trap that ate a 15-year streak

Bill Miller ran Legg Mason Value Trust to one of the most famous records in fund history: his flagship beat the S&P 500 for **fifteen consecutive calendar years, 1991 through 2005** — a streak so improbable it's often quoted at odds of roughly one in 2.3 million. He was, by any measure, an elite analyst, which is exactly what makes what came next instructive.

Into 2007–2008, Miller loaded up on financials as they fell — Bear Stearns, Citigroup, AIG, Freddie Mac — reading each leg down as a cheaper entry into a sound franchise, the classic value-investor thesis. It's the entrenchment chart come to life: each new piece of bad news about the banks was assimilated as *more* reason to buy the "bargain," and the falling price was evidence of a gift, not a warning. Reporting from the period describes him holding Freddie Mac even as colleagues urged him to sell, right up until the government nationalization in September 2008 wiped the equity out. By year-end 2008, Value Trust had fallen about **55%** while the S&P 500 fell roughly **37%** — nearly 18 percentage points worse than the market it had beaten for a decade and a half. The assets Miller managed collapsed from a pre-crisis peak reported around **\$77 billion** to roughly **\$20 billion** as performance and redemptions compounded. The lesson isn't that Miller was a bad investor — he was a great one. It's that a great analyst with an entrenched thesis and no falsifier is *more* dangerous than a mediocre one, because the sophistication makes the rationalizations more convincing, right up to the nationalization.

### 2. Bill Ackman's Herbalife short: a public thesis you can't abandon

In December 2012, Bill Ackman announced a roughly **\$1 billion** short against Herbalife at around \$45 a share, calling it a pyramid scheme he expected to go to zero — and, crucially, he made the case *publicly*, in a three-hour presentation, staking his reputation on it. That public commitment is a motivated-reasoning accelerant: now the thesis isn't just a position, it's an identity to defend, and abandoning it means admitting you were wrong in front of everyone.

What followed was a five-year confirmation loop. Carl Icahn took the other side, building a large long stake; the stock rose instead of falling. Ackman poured resources into building the disconfirming-to-everyone-else case that Herbalife was a fraud — more research, deeper conviction, exactly the pattern where volume of work entrenches rather than corrects. He held the losing short through years of the stock grinding *up*. He finally exited on **February 28, 2018**, after the shares had roughly doubled from where he'd bet against them, with reported losses approaching **\$1 billion**. The tell throughout was that no price seemed to be his falsifier — there was always another reason the market was "wrong" and vindication was near. A thesis with no price that can prove it wrong is the confirmation loop in its purest form, and going public had welded the goalposts in place.

### 3. Wirecard: when the crowd attacks the disconfirming evidence

The Wirecard collapse is confirmation bias operating on a whole ecosystem — bulls, regulators, and auditors — against a steady stream of disconfirming evidence that was *available for years*. The German payments company was a market darling, once valued near **€24 billion** and briefly worth more than Deutsche Bank.

Disconfirming evidence arrived early and loudly. Short-sellers and the Financial Times published detailed allegations of accounting fraud; the 2016 "Zatarra" report laid out specifics. The response is the case study: rather than weight the disconfirming evidence, the believers *attacked its source*. Germany's regulator BaFin opened an investigation into the short-sellers and briefly banned shorting the stock — treating the people supplying the falsifying evidence as the problem, a textbook "it's manipulation" dismissal from the interpreting row of the asymmetry matrix. The believers assimilated every red flag as an attack to be repelled. It held until it didn't: in April 2020 KPMG's special audit could not verify large chunks of profit; in June 2020 EY refused to sign the accounts because it could not confirm the existence of **€1.9 billion** in cash — money that did not exist. The stock collapsed to near zero within days. The disconfirming evidence had been sitting in plain sight for four years; the loop had simply relabeled it as slander.

### 4. GameStop and Melvin Capital: entrenchment meets a squeeze

In January 2021, Melvin Capital held a large short position in GameStop, thesis: a declining mall retailer worth far less than its price. A wave of retail buying, coordinated on social media, sent the stock from under \$20 to an intraday high of **\$483** on **January 28, 2021**. As the short went catastrophically against them, the position had to be defended and financed rather than simply reconsidered — the fund reportedly lost about **53%** (on the order of **\$6.8 billion**) in January alone, and took a **\$2.75 billion** capital injection from Citadel and Point72 to keep going. Melvin closed in 2022.

Both sides of GameStop were riddled with confirmation loops — the retail longs assimilated every red candle as a "shakeout by the hedgies," a mirror image of the shorts' certainty — which is the tidiest possible illustration that the bias isn't about being a bull or a bear. It's about having a position and needing it to work. The GameStop tape emitted enough ambiguous evidence for *everyone* to confirm whatever they already believed, and the people who blew up were the ones with the biggest positions and the least willingness to name a price that would prove them wrong.

## When this matters to you

You will not remember the definitions of biased assimilation and motivated reasoning when it counts. That's fine. What you need to carry is smaller and more useful: the *feeling* of the loop, and one habit that jams it.

The feeling is the uniform direction of your own thoughts. When every consideration about a position points the same way — when you have a rebuttal ready for every piece of bad news and no price you'd accept as proof you're wrong — you are not analyzing anymore. You're defending. The content of the thoughts will feel like sharp analysis; only the *direction* gives it away. A mind weighing evidence produces thoughts that cut both ways. A mind in the loop produces a one-way stream.

The habit is the falsifier. Before you add to a winner — the highest-risk moment, when sizing up feels most like insight — write down the exact price or datum that would prove you wrong, and pre-commit the exit at that level. If you can name it, you've converted a self-defending thesis into a bounded bet. If you *can't* name it, you've found the loop, and the honest size is zero. That one question — "what price proves me wrong?" — is worth more than any amount of research, because research runs through the loop and the falsifier cuts across it. It's the disconfirming triple in the 2-4-6 game: the one test you'll never propose by instinct and the only one that teaches you anything.

None of this is investment advice; it's a description of how a normal mind behaves around a position, and a procedure for behaving a little less like it. The mechanism that makes you a confident holder of a losing trade is the same one that let Bill Miller ride the banks down and kept Bill Ackman short for five years — it doesn't spare the professionals, and it won't spare you. But it's mechanical, which means the countermeasure can be mechanical too. You can't out-think the loop. You can pre-commit around it.

If you want the wider map of how this bias chains into the others — how a distorted perception hands off to a distorted memory and a distorted judgment — start with [the cognitive bias map for traders](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders). For the discipline of naming your falsifier *before* you enter, the companion piece is [what would change my mind: defining invalidation upfront](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront). And because confirmation bias is often downstream of falling in love with a story, pair this with [narrative addiction: when a good story beats the data](/blog/trading/analyst-edge/narrative-addiction-when-a-good-story-beats-the-data) and the broader case for [why your brain is bad at markets](/blog/trading/trading-psychology/why-your-brain-is-bad-at-markets).

## Sources & further reading

- Wason, P. C. (1960). "On the Failure to Eliminate Hypotheses in a Conceptual Task." *Quarterly Journal of Experimental Psychology*, 12(3), 129–140. — The original 2-4-6 experiment; about one in five subjects (6 of 29) found the rule cleanly.
- Wason, P. C. (1966–1968). The *selection task* (four-card problem), introduced in "Reasoning" and developed in Wason & Shapiro (1971). — Abstract-version success rates are typically under 20%, and often under 10% on the classic form.
- Kunda, Z. (1990). "The Case for Motivated Reasoning." *Psychological Bulletin*, 108(3), 480–498. — The foundational account of reasoning toward desired conclusions, constrained by the ability to justify them.
- Nickerson, R. S. (1998). "Confirmation Bias: A Ubiquitous Phenomenon in Many Guises." *Review of General Psychology*, 2(2), 175–220. — The definitive review and the source of the working definition used here.
- Lord, C. G., Ross, L., & Lepper, M. R. (1979). "Biased Assimilation and Attitude Polarization: The Effects of Prior Theories on Subsequently Considered Evidence." *Journal of Personality and Social Psychology*, 37(11), 2098–2109. — The capital-punishment study behind the entrenchment chart.
- Tversky, A., & Kahneman, D. (1992). "Advances in Prospect Theory: Cumulative Representation of Uncertainty." — The 2.25× loss-aversion estimate referenced in the series.
- Barber, B. M., & Odean, T. (2000). "Trading Is Hazardous to Your Wealth." *Journal of Finance*. — The 11.4% vs 17.9% gap for the most active traders (66,465 households, 1991–1996).
- Bill Miller / Legg Mason Value Trust: contemporaneous reporting (Institutional Investor; *Money*; Wikipedia, "Bill Miller (investor)") on the 1991–2005 streak, the ~55% 2008 drawdown vs the S&P's ~37%, and the collapse in assets under management.
- Bill Ackman / Herbalife: CNN Money and *Fortune* reporting (Feb–Mar 2018) on the ~\$1 billion short announced December 2012 at ~\$45, the five-year hold, and the February 28, 2018 exit near reported losses approaching \$1 billion.
- Wirecard: *Financial Times* investigations and the "Zatarra" report (2016); KPMG special audit (April 2020); EY's refusal to sign over €1.9 billion in missing cash (June 2020); Wikipedia, "Wirecard scandal."
- GameStop / Melvin Capital: CNBC and CNN reporting (January 31, 2021) on Melvin's ~53% (~\$6.8 billion) January loss, the \$2.75 billion Citadel/Point72 injection, and GameStop's \$483 intraday high on January 28, 2021.
- On this blog: [the cognitive bias map for traders](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders), [why your brain is bad at markets](/blog/trading/trading-psychology/why-your-brain-is-bad-at-markets), [what would change my mind: defining invalidation upfront](/blog/trading/analyst-edge/what-would-change-my-mind-defining-invalidation-upfront), and [narrative addiction: when a good story beats the data](/blog/trading/analyst-edge/narrative-addiction-when-a-good-story-beats-the-data).

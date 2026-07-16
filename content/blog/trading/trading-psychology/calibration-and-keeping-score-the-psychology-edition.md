---
title: "Calibration: Keeping Score on Your Own Confidence"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "Confidence is an emotion, and like every emotion it can lie — the fix is to score it. This is the psychology of calibration: why your 90%-sure calls only land ~65% of the time, how that gap quietly turns a winning book into a losing one, and the decision-journal habit that trains the feeling to stop lying."
tags:
  [
    "trading-psychology",
    "calibration",
    "overconfidence",
    "brier-score",
    "decision-journal",
    "probability",
    "base-rates",
    "superforecasting",
    "behavioral-finance",
    "expected-value",
    "position-sizing",
    "metacognition",
  ]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 38
---

> [!important]
> **TL;DR** — Confidence is a *feeling*, and like every feeling it can lie. Calibration is the discipline of checking whether the feeling is telling the truth: do the calls you feel 70% sure about actually happen about 70% of the time?
>
> - Most traders are systematically **overconfident**: the things they call 90% likely land closer to 65%. That is not a character flaw, it is the default wiring — the feeling of certainty is built out of fluency, vividness, and recency, none of which track how often you are actually right.
> - You cannot fix a feeling by feeling harder about it. The only known cure is to **score it externally**: state a probability, write it down, wait for the outcome, and grade yourself — the calibration curve and the Brier score turn a private hunch into a public number.
> - The gap is expensive in dollars, not just pride. A book that *looks* +EV because you feel 90% sure can be quietly **−EV** once your real hit rate is 65%. Garbage probabilities in, garbage sizing out.
> - Calibration is **trainable**, and we know this because some professionals have it: weather forecasters, whose "70% chance of rain" really does rain about 70% of the time, and Philip Tetlock's superforecasters — both groups get their probabilities scored, repeatedly. Pundits, who never get scored, stay confident and wrong.
> - The habit that fixes it is boring and cheap: a **calibration log**. State a number on every call, score the log monthly, and widen your ranges and anchor to base rates until your confidence stops lying to you.

Ask a room of traders how their year went and you will hear a story. Ask them what probability they assigned to each of the trades in that story *before* they put them on, and the room goes quiet. The story is vivid; the probabilities were never written down. And that gap — between the confident narration and the un-scored forecast underneath it — is where most trading psychology quietly goes wrong.

Here is the uncomfortable thing this article is about. The *feeling* of being sure and the *fact* of being right run on two different circuits. One is a fast, warm, automatic emotion your brain generates in a fraction of a second. The other is a slow, cold statistical property of your track record that you can only see by keeping count. They are supposed to line up. In most people, most of the time, they do not — and the feeling runs ahead of the fact.

Calibration is the name for how well those two circuits agree. A perfectly calibrated trader is one whose confidence is honest: when they say 70%, it happens 70% of the time; when they say 90%, it happens 90% of the time. This has a companion post on the mechanics — [Calibration: Keeping Score on Your Own Forecasts](/blog/trading/analyst-edge/calibration-keeping-score-on-your-own-forecasts) in the Analyst's Edge series works through the curve, the Brier score, and the sizing math in detail. This post is about the *psychology*: why the feeling lies in the first place, what it looks like at the screen, and the one habit that trains it to stop.

![The calibration loop: a feeling of certainty only becomes accurate once you state it as a number and let outcomes grade it.](/imgs/blogs/calibration-and-keeping-score-the-psychology-edition-1.webp)

The diagram above is the whole article in one picture. On the top path, a feeling of certainty gets committed to a number ("I'm 70% sure"), the market resolves it, you score whether your 70% calls actually hit around 70%, and you recalibrate the next estimate. On the bottom path — the one almost everyone is on by default — you skip the number, so the feeling is never scored, and therefore never corrected. Nothing about your brain fixes this on its own. The correction only exists if you build the loop.

## Foundations: what calibration actually is

Before we can talk about why confidence lies, we need three ideas defined from zero: what calibration is (and what it is *not*), how you draw it, and how you grade it with a single number. If you have read the companion mechanics post, skim this; if this is your first pass, read slowly, because everything later leans on it.

### Calibration is not accuracy

Start with a distinction that trips up almost everyone. **Accuracy** is "how often are you right?" **Calibration** is "when you say a probability, does it match how often that thing happens?" They are not the same, and the difference is the whole game.

A weather forecaster who says "70% chance of rain" and is rained on 70% of those days is *perfectly calibrated* — even though it stayed dry on 30% of those days, which by the accuracy standard looks like being "wrong" almost a third of the time. Calibration does not ask you to be right every time. It asks you to be **honest about how sure you are**. A 70% call that fails is not a miscalibration; a 70% call that *succeeds 95% of the time* is — because you were needlessly timid and should have said 95%.

There is a third property worth naming so you do not confuse it with the first two. **Resolution** (sometimes called discrimination) is how far your probabilities spread away from the base rate — a forecaster who says "50%" to everything is perfectly calibrated in the long run and completely useless, because they never commit. Good forecasting needs both: calibration (your numbers are honest) *and* resolution (your numbers are decisive). A calibration log fixes the first without which the second is just confident noise.

| Property | The question it answers | The failure it catches |
|---|---|---|
| Accuracy | How often am I right? | Being wrong a lot |
| Calibration | Do my 70% calls happen ~70% of the time? | Confidence that does not match reality |
| Resolution | Do I commit, or hug 50% on everything? | Useless hedging that is technically "calibrated" |

The reason calibration, not accuracy, is the psychologically load-bearing one: you can be wrong on a specific trade and still be a perfectly honest forecaster, and you can be *right* on a specific trade and still be dangerously overconfident. Outcomes are noisy; calibration is the signal underneath the noise. (If that distinction interests you, [process versus outcome and the trap of resulting](/blog/trading/trading-psychology/process-versus-outcome-and-the-trap-of-resulting) is the sibling post on judging decisions by their quality rather than their result.)

### The calibration curve

Now picture how you would *draw* calibration. Put your stated confidence on the horizontal axis and your realized hit rate on the vertical axis. If you were perfectly honest, every point would sit on the 45-degree diagonal: 70% stated, 70% realized; 90% stated, 90% realized. Overconfidence shows up as points that **sag below** the diagonal — you said 90% but only 65% happened. Underconfidence bows above it.

![The calibration curve: an overconfident trader's points sag below the diagonal, and the vertical gap is the size of the lie.](/imgs/blogs/calibration-and-keeping-score-the-psychology-edition-2.webp)

This is the single most important picture in the whole subject, so spend a moment with it. The dashed diagonal is perfect calibration. The red curve is a typical overconfident trader: at the low end their confidence is roughly honest, but as they get more sure, reality falls further and further behind. By the time they say 90%, the truth is around 65% — that vertical red gap is 25 points of pure overconfidence. The three boxes along the bottom read straight off the curve: their 70% calls hit ~60%, their 80% calls hit ~64%, their 90% calls hit ~65%. The higher the confidence, the bigger the lie. That top-heavy shape — fine at the bottom, collapsing at the top — is the fingerprint of overconfidence, and it is astonishingly common.

### The Brier score

The curve is a picture; sometimes you want a single number you can track month over month. That number is the **Brier score**, invented by the meteorologist Glenn Brier in 1950 for exactly this purpose — grading probabilistic weather forecasts (Brier, 1950). For a set of yes/no calls it is just the average squared error between what you said and what happened:

$$
\text{Brier} = \frac{1}{N}\sum_{i=1}^{N}\left(p_i - o_i\right)^2
$$

where p is the probability you stated on a call, o is the outcome (1 if it happened, 0 if it did not), and N is the number of calls. The score runs from **0 (perfect)** to **1 (as wrong as possible)**. Say "90%" and be right, and you contribute (0.9 - 1)² = 0.01 — tiny. Say "90%" and be wrong, and you contribute (0.9 - 0)² = 0.81 — enormous. The squaring is the whole point: it punishes confident mistakes far more harshly than hedged ones. A single loud, sure, wrong call costs you as much as eighty-one quiet honest ones.

We will compute a real Brier score in a moment. For now, hold onto the intuition: the Brier score is your confidence's report card, and overconfidence is the fastest way to flunk it.

#### Worked example: a single 70% bucket

Let's ground all three ideas in the simplest possible case before we scale up. Suppose over a quarter you make ten trades that, at the moment of entry, you honestly rated as "70% likely to work." That is one confidence bucket. Now you wait, and the outcomes land: 6 of the 10 worked, 4 did not.

Are you calibrated? Your stated confidence in this bucket was 70%; your realized hit rate was 6/10 = 60%. You are overconfident by 10 points — small, but real. What did it cost your Brier score? Each winner contributed (0.7 - 1)² = 0.09; each loser contributed (0.7 - 0)² = 0.49. Total = 6 × 0.09 + 4 × 0.49 = 0.54 + 1.96 = 2.50, averaged over 10 calls = **0.25**. For comparison, if you had honestly said 60% on all ten, your Brier would have been 6 × (0.6 - 1)² + 4 × (0.6 - 0)² = 6(0.16) + 4(0.36) = 0.96 + 1.44 = 2.40, averaged = **0.24** — very slightly better, because your stated numbers matched reality.

The intuition: even a modest 10-point overconfidence is measurable, and the only way you would ever have seen it is by writing the 70% down before the outcomes arrived.

### Confidence is a frequency wearing the costume of a feeling

Here is the conceptual pivot the rest of the article turns on. In the math above, a probability is a **frequency** — a fact about how often something happens across many repetitions. But inside your head, a probability does not arrive as a frequency. It arrives as a *feeling*: a warm, immediate sense of "yeah, this one's good." Your brain hands you the feeling and lets you assume it came from the frequency. Usually it did not. That substitution — feeling standing in for frequency — is the source of every calibration error in this article, and it is where the psychology begins.

## Confidence is an emotion, and emotions lie

We do not usually think of confidence as an emotion. It feels more like a readout — a little gauge reporting how good the evidence is. But neuroscience and decades of judgment research point the other way: the *feeling* of certainty is generated fast, automatically, and largely independently of the actual quality of your evidence. It is closer to a mood than to a measurement. And a mood can be manipulated.

The philosopher-neurologist Robert Burton called this "the feeling of knowing" — the sensation of certainty that attaches itself to a belief the way warmth attaches to a good memory. Crucially, that sensation can fire even when the belief is wrong, and it can fail to fire even when the belief is right. Your brain does not have a truth detector. It has a *fluency* detector, a *vividness* detector, a *recency* detector — and it reports their combined output as "confidence."

![The four confidence inflators: fluency, vividness, recency, and coherence all pump up the feeling of certainty without adding any accuracy.](/imgs/blogs/calibration-and-keeping-score-the-psychology-edition-3.webp)

Think of the feeling of certainty as a gauge with four inputs wired into it, none of which measure probability. The matrix above lays them out. **Fluency**: an idea that is easy to process feels more true — a clean chart, a thesis that "just makes sense," a setup you have seen a hundred times. Ease of recall is not probability of truth, but your brain treats them as the same. **Vividness**: a dramatic, memorable scenario feels more likely than a boring one, so you overweight the headline outcome and underweight the dull base case. **Recency**: your last trade colors your next one — win and you feel invincible, lose and you feel snake-bitten — even though one sample tells you almost nothing about the base rate. **Coherence**: a tidy story where all the pieces fit feels certain, so the "can't-lose" setup with a beautiful narrative gets sized like a sure thing. Every one of these makes the *feeling* stronger; not one of them makes the *forecast* more accurate.

This is why you cannot introspect your way to calibration. Feeling more sure does not make you more right; it usually just means one of those four dials got turned up. The gauge is not broken — it is working exactly as designed, reporting fluency and vividness and recency. It was simply never designed to report probability, and no amount of staring at it harder will change what it measures.

> The feeling of certainty is not evidence about the world. It is evidence about how your memory happened to retrieve the idea.

### The proof that your confidence lies

You do not have to take this on faith — psychologists have measured it, repeatedly, with a demonstration you can run on yourself. In the classic version, from Marc Alpert and Howard Raiffa (1982), people are asked to give a range for some unknown quantity — the length of a river, the number of eggs produced in a country in a year — wide enough that they are "98% sure" the true answer falls inside it. If everyone were honest, the true value would land outside the range only 2% of the time. Instead it lands outside roughly 40% of the time: their supposedly 98% ranges behaved like 60% ranges. The intervals were far too narrow, which is overconfidence in its purest form — an inability to make "I don't know" wide enough.

The finance-flavored version is even more pointed. J. Edward Russo and Paul Schoemaker gave more than 1,000 managers a ten-question quiz — dates, distances, financial figures — and asked for ranges they were "90% confident" contained each answer (Russo & Schoemaker, *Decision Traps*, 1989). A well-calibrated person should miss about one of ten. Most managers missed four to seven — a "surprise rate" of 40% to 70% where it should have been 10% — and fewer than 1% got nine or ten right. These were smart, senior, numerate people. Their confidence was not tracking their knowledge; it was tracking the feeling.

This matters for traders because a price forecast is exactly one of those intervals. "The stock will be between 95 and 115 by Friday, and I'm 90% sure" is a 90% confidence range, and if you are like the managers in the study, it is far too narrow — the real 90% range might be 85 to 125. A too-narrow range is not a harmless quirk; it is the input that makes you set stops too tight, size too big, and get "surprised" by moves that were never actually surprising.

#### Worked example: the range that was never 90%

Suppose you forecast that a stock, trading at $100, will close between $95 and $115 in a month, and you state 90% confidence in that range — a $20-wide window. You size a position as if a move outside that window is a 1-in-10 event. Over the next year you make 20 such forecasts. If you were calibrated, the price should land outside your stated range about 2 times out of 20.

Instead, it lands outside 8 times — a 40% surprise rate, right in line with the research. Each of those 8 "surprises" hit a position sized for a 10% tail, so each one hurt roughly four times more than you had budgeted for. Your ranges were never 90% ranges; they were 60% ranges wearing a 90% label, and the eight tail events were the invoice. Had you widened each range to an honest $85 to $125, most of the same eight moves would have stayed inside, your stops would have survived the noise, and your sizing would have matched the real odds.

The intuition: overconfidence is not only an inflated point estimate — it is a range that is too narrow, and a range that is too narrow turns ordinary volatility into a stream of "surprises" you keep paying for.

### What it looks like at the screen

Overconfidence does not announce itself. It does not feel like a bias — biases never do, from the inside. It feels like *clarity*. So you have to learn to recognize it by its behavioral tells, the things you actually do at the screen when the confidence gauge has run ahead of the facts. Here is what miscalibrated confidence looks like in real time:

- **You size up "just this once."** The position is 2–3x your normal risk because this one is "different," "a lock," "the cleanest setup I've seen all month." The size is the tell. Your normal size is calibrated to your normal edge; the oversize is your feeling overriding your process.
- **You skip the checklist.** The pre-trade routine feels unnecessary because the trade is *obviously* good. Skipping the process is the surest sign the process was about to catch something.
- **You stop looking for the other side.** You cannot name what would make the trade wrong, and you do not want to — the question feels like a buzzkill. A calibrated trader can always state their kill criteria; an overconfident one is annoyed by the question.
- **You feel a flush of certainty right after a win.** Two or three winners in a row and the gauge is pinned. That is recency, not skill, and it is exactly when your sizing quietly creeps up. ([Recency, availability, and the tyranny of the last trade](/blog/trading/trading-psychology/recency-availability-and-the-tyranny-of-the-last-trade) is the full treatment.)
- **You argue with the tape.** The position goes against you and instead of asking "what did I get wrong?" you explain why the market is wrong. The strength of the feeling has become a reason to disbelieve reality.
- **Your stops get "mental."** A calibrated bet has a pre-committed exit; a sure thing does not need one, because it's a sure thing. The disappearing stop is overconfidence made visible.

None of these feel like errors while you are doing them. They feel like conviction. The only reliable way to tell conviction from calibration apart is to have written the number down beforehand — because then you can check.

## Measuring the gap: the calibration curve and the Brier score

If the feeling cannot be trusted and cannot be introspected, the only move left is to measure it from the outside. The method is mechanical and it is the same one the [companion mechanics post](/blog/trading/analyst-edge/calibration-keeping-score-on-your-own-forecasts) walks through in full: **bucket, plot, score.** Take your last N journaled calls, group them by the confidence you stated, compute the realized hit rate inside each bucket, and compare the two. The gap between what you said and what happened *is* your miscalibration, made visible for the first time.

![What you feel versus what the scoreboard says: a 90% feeling that wins ~65% of the time hides a 25-point overconfidence gap.](/imgs/blogs/calibration-and-keeping-score-the-psychology-edition-4.webp)

The figure above is what the measurement reveals. On the left is the feeling: "90% sure," which should mean 9 of 10 win; it feels like easy money, so you bet big. On the right is the scoreboard: those calls actually win about 65% of the time, meaning roughly 1 in 3 of your "locks" is a loser, and the 25-point gap between the two columns is your overconfidence — not as an insult, but as a *number* you can now shrink. The whole value of measuring is that it converts a vague unease ("am I too cocky?") into an exact quantity ("I am 25 points overconfident at the top of my confidence range").

#### Worked example: bucketing 50 calls into a curve

Let's build a real curve. Suppose you kept a log for six months and made 50 trades, each tagged with a stated confidence. You group them:

- The **60% bucket**: 10 calls, 6 worked → realized 60%. On the diagonal. Honest.
- The **70% bucket**: 15 calls, 9 worked → realized 60%. Ten points low.
- The **80% bucket**: 15 calls, 10 worked → realized 67%. Thirteen points low.
- The **90% bucket**: 10 calls, 6 worked → realized 60%. Thirty points low.

Plot those four points and you get exactly the sagging shape from the curve figure: fine at the bottom, collapsing at the top. Notice the pattern — your 60% calls are honest, but everything you feel *strongly* about is inflated, and the inflation grows with the confidence. That is the signature of overconfidence, and it means your problem is not "I'm wrong a lot," it is "I'm most wrong exactly when I feel most sure."

The intuition: the curve does not just tell you *that* you are overconfident — it tells you *where*, and for almost everyone the damage is concentrated at the high-confidence end where the sizing is biggest.

#### Worked example: computing the Brier score on ten calls

Now the single-number version. Take ten specific calls with their stated probabilities and outcomes (1 = worked, 0 = did not):

| Call | Stated p | Outcome | Squared error |
|---|---|---|---|
| 1 | 0.90 | 1 | 0.01 |
| 2 | 0.90 | 0 | 0.81 |
| 3 | 0.80 | 1 | 0.04 |
| 4 | 0.80 | 0 | 0.64 |
| 5 | 0.70 | 1 | 0.09 |
| 6 | 0.70 | 1 | 0.09 |
| 7 | 0.60 | 0 | 0.36 |
| 8 | 0.60 | 1 | 0.16 |
| 9 | 0.55 | 1 | 0.20 |
| 10 | 0.50 | 0 | 0.25 |

Sum of the squared errors = 0.01 + 0.81 + 0.04 + 0.64 + 0.09 + 0.09 + 0.36 + 0.16 + 0.20 + 0.25 = 2.65. Divide by 10: **Brier = 0.265.** Notice what dominated: the two confident misses (calls 2 and 4) contributed 0.81 + 0.64 = 1.45 — more than half the total error came from two trades where you were loud and wrong. That is the squaring doing its job. If you had dialed those two 90%/80% calls down to an honest 65%, their contribution would have fallen from 1.45 to (0.65)² + (0.65)² = 0.42 + 0.42 = 0.845, dragging your whole Brier from 0.265 down to about 0.205.

The intuition: your Brier score is not hurt by being wrong — it is hurt by being *sure* and wrong, which is precisely the failure mode overconfidence produces.

## The dollar cost: when 90% is really 65%

So far this could read like a self-improvement exercise — keep score, feel humbler, be a better person. It is not. Miscalibration is expensive in cash, and the mechanism is worth seeing clearly, because it is the reason a calibration log is a risk-management tool and not a journaling hobby.

Your stated probability is not a decoration. It is the *input* to two decisions that determine whether you make money: whether to take the trade at all (is it positive expected value?) and how much to bet (sizing). Feed those decisions an inflated probability and both come out wrong — you take trades you should skip, and you bet too much on the ones you take. The math is unforgiving because expected value is *linear* in probability: a few points of overconfidence do not cost you a few points of return, they can flip the sign of the whole enterprise.

![When 90% is really 65%: at a 1:4 payoff you need 80% wins to break even, so a 90% feeling that is really 65% flips the book from profit to loss.](/imgs/blogs/calibration-and-keeping-score-the-psychology-edition-5.webp)

Conceptually, the figure above is the trap in one frame. It shows a trade with an asymmetric payoff — the kind of "high win-rate" setup overconfident traders love, where you win small and lose big. Because the loser is four times the winner, you need to win 80% of the time just to break even. If you *feel* 90% sure, you think each trade is worth +$50 of expected value (the green bar) and you happily size it up. But if your real hit rate is 65%, each trade is actually worth −$75 (the red bar). The book you believed was a money machine is bleeding, and nothing on the screen tells you — because the loss is hidden inside the gap between your 90% feeling and your 65% reality.

#### Worked example: an overconfident book turning +EV into −EV

Let's put real numbers on it. You sell premium — say you collect $100 when the trade works and lose $400 when it doesn't (a 1:4 reward-to-risk, typical of a short-volatility or "picking up nickels" strategy). The break-even win rate is where the expected value is zero:

$$
\text{EV} = p \times 100 - (1-p) \times 400 = 0 \;\Rightarrow\; 500p = 400 \;\Rightarrow\; p = 80\%.
$$

You need to win 80% of the time to break even. Now, you *feel* 90% sure on these, so you compute your expected value as 0.90 × 100 - 0.10 × 400 = 90 - 40 = +$50 per trade. On 200 trades a year, that is a believed +$10,000. So you run the strategy in size, maybe $2,000 of risk per trade.

But your calibration log — which you kept — reveals your real hit rate on these is 65%, not 90%. The true expected value is 0.65 × 100 - 0.35 × 400 = 65 - 140 = −$75 per trade. Across 200 trades that is **−$15,000 a year**, not +$10,000. The 25-point calibration gap did not shave your returns; it reversed them, a $25,000 swing on the year, and every extra dollar of size you added because you "felt sure" made the bleeding faster.

The intuition: at asymmetric payoffs, the difference between feeling 90% and being 65% is not the difference between a great year and a good one — it is the difference between a winning book and a blown-up one.

#### Worked example: widening the range to get the sizing right

The fix is not to stop trading the strategy — it might be genuinely good at an honest probability. The fix is to feed your decisions the *calibrated* number. Suppose you do the humbling work, anchor to the base rate, and conclude your honest hit rate on this setup is 78% — below the 80% break-even. Now the expected value reads 0.78 × 100 - 0.22 × 400 = 78 - 88 = −$10 per trade. Barely negative. You either pass, or you renegotiate the trade structure until the payoff justifies the honest odds — collect $120 instead of $100, and now 0.78 × 120 - 0.22 × 400 = 93.6 - 88 = +$5.60, thin but positive.

The intuition: calibration is not about being less confident for its own sake — it is about feeding your sizing model a number that is true, so the trades you keep are the ones that actually pay.

If you want the emotional-regulation angle on why honest sizing is also the thing that keeps you sane through a drawdown, [position sizing as emotional regulation](/blog/trading/trading-psychology/position-sizing-as-emotional-regulation) is the companion piece.

## Calibration is a trainable skill

Here is the good news, and it is genuinely good. Calibration is not fixed. It is a skill, and like any skill it improves with **scored, timely feedback** — the two words that matter most in this whole article. The evidence that it is trainable comes from looking at who has it and who does not.

![Who is well-calibrated, and why: the well-calibrated all get their probabilities scored against outcomes, repeatedly; the miscalibrated keep no receipt.](/imgs/blogs/calibration-and-keeping-score-the-psychology-edition-6.webp)

The matrix above sorts four kinds of forecaster by the one variable that predicts calibration: do they get scored? Consider each case, because the contrast is the lesson.

**Weather forecasters** are the gold standard, and it is not close. When a National Weather Service forecaster says "70% chance of rain," it rains on close to 70% of those days — their probability-of-precipitation forecasts are famously well-calibrated (Murphy & Winkler, 1977). Why them? Because they get the single cleanest feedback loop in all of forecasting: they make a probabilistic call every single day, and every single day nature grades it, publicly, with a Brier score. Decades of that loop grinds the overconfidence out. It is no accident that the Brier score was invented *for weather* — the field that scores itself is the field that gets calibrated.

**Superforecasters** are the proof that ordinary people can be trained into it. In the forecasting tournament run by the U.S. intelligence community's research arm (IARPA) from 2011, Philip Tetlock's Good Judgment Project fielded teams of talented amateurs against professional analysts. Over four years and roughly 500 questions, the best forecasters — the "superforecasters" — were about **30% more accurate than intelligence analysts who had access to classified information** (Tetlock & Gardner, *Superforecasting*, 2015). The project won every season of the tournament, beating competing university teams by margins of 35–72%. What made them super was not IQ or secret data; it was method — they stated explicit probabilities, tracked them, updated in small steps, and got scored. The habit *is* the talent.

**Pundits** are the control group, and they fail. In Tetlock's earlier 20-year study, published as *Expert Political Judgment* (2005), 284 experts made over 82,000 forecasts — and the average expert was, in his famous phrase, "roughly as accurate as a dart-throwing chimpanzee." Worse, the most confident, most famous experts — the "hedgehogs" who explain everything through one big idea — were the *least* accurate, and the more certain they sounded on television, the worse their record. They never got scored, so they never got calibrated; the reward for a pundit is confidence, not accuracy, and confidence is exactly what they optimized.

**The typical trader** sits in the pundit's chair without realizing it. No log, no scored probabilities, no receipt — just a vivid memory of the good calls and a merciful fog over the bad ones. The confidence gauge runs free because nothing ever pushes back on it. The entire difference between this row and the weather-forecaster row is a habit, and the habit is a calibration log.

## Why the market is the worst place to be miscalibrated

If scored, timely feedback is what builds calibration, then markets are almost engineered to prevent it. Three features of trading conspire to keep the confidence gauge uncorrected — which is exactly why the discipline has to be imposed by hand rather than absorbed from experience.

**The feedback is slow and noisy.** The weather forecaster learns tomorrow whether "70% rain" was honest, and learns it cleanly, because a day either has rain or it does not. A trader might wait weeks for a thesis to resolve, and even then the outcome is contaminated by luck. A great decision can lose and a terrible one can win, so any single result teaches you almost nothing about whether your probability was right. Where the forecaster gets one clean data point a day, the trader gets one noisy data point a month — and noise drowns the signal you are trying to learn from. This is why you cannot calibrate from memory: the signal is too faint to hear without writing every call down and aggregating over dozens of them.

**The scoring is self-serving.** When a trade wins, you file it under skill; when it loses, you file it under bad luck, a bad headline, a rigged tape. This is the self-attribution bias, and it is calibration poison, because it means every outcome — win *or* lose — gets read as confirmation that your process was sound. Wins prove you are good; losses prove the market was unfair. A gauge that reads all feedback as vindication never moves. The calibration log defeats this by fixing the probability in writing *before* the outcome, so your after-the-fact story cannot quietly rewrite what you actually believed. (This is the same machinery as [hindsight bias and the story you tell yourself later](/blog/trading/trading-psychology/hindsight-bias-and-the-story-you-tell-yourself-later).)

**The stakes amplify the error.** In a psychology quiz, overconfidence costs you a wrong answer. In a market, it costs you money, and it does so through sizing — the more sure you feel, the bigger you bet, so your miscalibration is largest exactly where your position is largest. The overconfident trader is not making small errors on small bets and large errors on large bets at random; they are systematically most wrong on the trades they have sized biggest, because the same feeling that inflated the confidence also inflated the size. Leverage does not just multiply your returns; it multiplies your calibration error.

Put those three together and the outlook is bleak: delayed, noisy feedback that you interpret in your own favor, attached to bets that get bigger as your confidence gets less reliable. No wonder the untracked trader ends up in the pundit's chair. The market will never spontaneously calibrate you the way the sky calibrates a meteorologist — the feedback is too slow, too noisy, and too flattering. You have to build the scoring loop yourself, which is the entire point of the drill that follows.

## The drill: the calibration log

Everything above converges on one intervention, and it is almost insultingly simple. You cannot fix a feeling by feeling differently about it; you fix it by building the scoring loop the feeling never had. That loop is a **calibration log** — a written record where you state a probability before the outcome, and score yourself after.

![The calibration-log protocol: state a number on every call, score it monthly, and let the gap tell you how much to widen the next estimate.](/imgs/blogs/calibration-and-keeping-score-the-psychology-edition-7.webp)

The protocol above is the entire drill, and each step is doing psychological work. **State a probability on every call.** Not "I like this one" — a number, 55% or 75% or 90%. The act of committing to a number is what drags the vague feeling into a form that *can* be scored, and it is the step almost everyone skips. **Log it before the outcome.** This is critical: written down, timestamped, before you know what happens — because your memory will otherwise quietly rewrite what you "really" thought (that is [hindsight bias](/blog/trading/trading-psychology/hindsight-bias-and-the-story-you-tell-yourself-later), and it will erase your calibration data if you let it). **Wait for the market to resolve.** **Bucket by confidence and score monthly** — build the curve, compute the Brier. **Read the gap.** And then the correction: **widen your ranges and anchor to base rates.**

Those last two moves are how you actually shrink the gap, so they deserve their own picture.

![Building a calibrated probability from the base rate: anchor to how often this kind of trade actually works, adjust a little for today, then widen when you are unsure.](/imgs/blogs/calibration-and-keeping-score-the-psychology-edition-8.webp)

Conceptually, a calibrated probability is built in three moves, shown in the tree above — and it is almost the exact opposite of how the feeling builds one. The feeling starts with a vivid story about *this* trade and reasons outward. The calibrated estimate starts from the outside. **First, the base rate**: how often does *this kind* of trade actually work, across all the times you or the market have seen it? Start from that frequency, not from today's feeling. **Second, adjust for this specific setup** — but move a little, not a lot; the specific always feels more special than it is. **Third, widen for what you don't know**: when you are genuinely unsure, drag the number back toward 50%, because false precision is the overconfident brain's favorite trick. Anchoring to base rates is the antidote to vividness and coherence; widening is the antidote to fluency and false precision.

#### Worked example: one month of the log, before and after

Make it concrete. In month one you log 20 calls without much discipline, tag most of them 85–90% because they all "look great," and at month-end your realized hit rate in that top bucket is 62%. Your Brier for the month comes out to 0.28. The gap is a screaming 25 points, concentrated at the top. That is your baseline.

For month two you change exactly one thing: before tagging each trade, you write down the base rate first ("setups like this work about 60% of the time"), then adjust only a few points for today's specifics, then widen if you are unsure. Your tags now cluster around 60–70% instead of 85–90%. At month-end, your realized hit rate is still about 62% — your *trading* did not change — but now your stated confidence matches it. Your Brier drops from 0.28 to about 0.23, and, more importantly, the trades you sized up are no longer the ones you were most wrong about.

The intuition: you did not get better at trading in a month — you got honest about your odds, and honesty alone improved the score and de-risked the sizing.

A few practical notes that make the drill survivable:

- **Score monthly, not daily.** Any single outcome is noise; calibration is a property of dozens of calls. Checking daily just feeds the recency bias you are trying to escape.
- **Log the probability, not a prediction.** "It will go up" cannot be scored. "70% it closes above 100 by Friday" can.
- **Keep it to one line per trade.** A calibration log that takes five minutes per entry gets abandoned. Confidence, direction, kill-criterion, done.
- **Separate calibration from being right.** A month where your 70% calls hit 70% is a *win*, even if you lost money on variance. You are grading the honesty of the number, not the outcome of the trade. This is the same discipline as [thinking in bets](/blog/trading/trading-psychology/thinking-in-bets-probabilistic-decision-making): judge the quality of the wager, not the result of the flip.

## Common misconceptions

**"I'm calibrated because I'm right most of the time."** Being right often is accuracy, not calibration. You can win 65% of your trades and still be badly overconfident if you *called* them 90%. The whole point is that the two come apart, and the gap between them is where the money leaks. High accuracy with inflated confidence is the most dangerous combination, because the wins keep the gauge pinned while the sizing quietly destroys you.

**"Overconfidence means being wrong."** No — overconfidence means being *more sure than you should be*, whatever the outcome. A 90% call that happens to work was still overconfident if that class of call only lands 65% of the time. Judging calibration one trade at a time is a category error; it is a property of a distribution, not an event.

**"If I just feel more humble, I'll be better calibrated."** Calibration is not a mood you can adopt. Feeling humble on Monday does nothing to the number you'll assign under pressure on Friday. The only thing that moves calibration is scored feedback — the log, the curve, the Brier. Sentiment is not a substitute for a scoreboard.

**"Confidence and competence go together, so my confidence is informative."** They are only loosely correlated, and the correlation gets *weaker* exactly where it matters — at the high-confidence end, which is where the overconfidence figure sags most. Worse, in unfamiliar or fast-changing markets, confidence and competence can decouple entirely; the feeling of certainty is often strongest precisely when you understand a situation least. (See [overconfidence and the illusion of control](/blog/trading/trading-psychology/overconfidence-and-the-illusion-of-control) for the full mechanism.)

**"Widening my ranges just means I'm wishy-washy."** Widening is not hedging everything to 50% — that would kill your resolution and make you useless. It is pulling your *dishonest* 90% back to an honest 70% while leaving your genuinely strong calls strong. The goal is numbers that are both honest and decisive, not numbers that are all mush.

**"Superforecasters are just smarter than me."** The Good Judgment Project's whole finding is that they are not — they are *methodical*. They state probabilities, track them, update in small steps, and get scored. Every one of those is a habit you can copy, and the copying is the training.

## How it shows up in real markets

Calibration is not an abstract idea that lives in a psychology lab. It shows up, with real numbers, in the professions that either keep score or refuse to.

### The weather forecast on your phone

The most calibrated forecast most people ever see is the one they trust least. When the National Weather Service says "70% chance of rain," it rains about 70% of the time — a level of honesty about uncertainty that would make most traders weep. The reason, again, is the feedback loop: a probabilistic call every day, graded every day, for decades, on a Brier score (Murphy & Winkler, 1977). The lesson for a trader is not "watch the weather." It is that a well-calibrated forecaster is *built*, not born, and the thing that builds them is the boring daily scoring loop you have been avoiding. The weather app is a monument to what scored feedback does to overconfidence.

### The Good Judgment Project

In 2011 the U.S. intelligence community ran a four-year tournament to find out who could actually forecast world events. Philip Tetlock's Good Judgment Project entered teams of ordinary volunteers — a retired computer programmer, a pharmacist, a filmmaker — and those volunteers, the superforecasters, beat professional intelligence analysts with access to classified intercepts by roughly 30% on accuracy, and beat rival research teams by 35–72% (Tetlock & Gardner, *Superforecasting*, 2015). Across the tournament, participants made over a million forecasts on around 500 questions. The superforecasters had no secret information and no special genius; what they had was the calibration habit — explicit probabilities, tracked, updated in small increments, and scored. It is the single best piece of evidence that the thing this article asks you to do actually works.

### Tetlock's pundits and the dart-throwing chimp

The mirror image is Tetlock's earlier 20-year study of political and economic experts, published in 2005. Across 284 experts and more than 82,000 predictions, the average expert forecast was about as accurate as chance — "a dart-throwing chimpanzee," in the line that made the study famous (Tetlock, *Expert Political Judgment*, 2005). And the twist that matters for traders: the most confident, most media-friendly experts, the "hedgehogs" with one grand unifying theory, were the *least* accurate of all. They were never scored, so their confidence was never disciplined by reality — and the market for pundits rewarded certainty over accuracy, so certainty is what they supplied. Every trader who narrates a confident macro thesis on social media and never logs a probability is running the hedgehog's playbook.

### The overconfident premium-seller

Consider the trader who spent 2017 selling volatility — collecting small, steady premiums, winning month after month, and *feeling* more certain with every green month (that is recency turning the gauge up). The strategy has an asymmetric payoff: many small wins, rare enormous losses, exactly the 1:4-style shape from the worked example. A calibrated seller would have known the true probability of a volatility spike was low but the cost of one was ruinous, and sized accordingly. The overconfident seller, whose string of wins had convinced them the probability was near zero, sized as if it were. When volatility spiked violently in early 2018, the un-hedged short-vol crowd took losses that erased years of premium in days. The math was the −$75 book all along; the winning months just hid it behind a feeling.

### The analyst who was "right 90% of the time"

A final, quieter case, of a type every desk has seen. An analyst builds a reputation for being "right about 90% of the time" — and they genuinely are, on the small, high-probability calls they favor. The trouble is the sizing: because they *feel* 90% confident on everything, they bet the same large size on the occasional low-probability call as on the safe ones. Their calibration curve, if anyone drew it, would be honest at the bottom and collapsing at the top — and it is the top, where the confidence is inflated and the size is biggest, that eventually delivers the loss that eats the year. Being right 90% of the time is worth nothing if your confidence is flat at 90% while your real odds slide from 90% down to 60%. The reputation was accuracy; the risk was miscalibration.

## When this matters to you

If you take one thing from all of this, make it the reframe: **your confidence is not data about the market — it is data about how your brain retrieved the idea.** The warm certainty you feel about a trade is real, but it is measuring fluency and vividness and how your last trade went, not the probability that this one works. Treating the feeling as a probability is the single most expensive habit in trading, because it silently corrupts the two decisions that make you money — whether to take the trade and how much to bet.

The fix is available to anyone and costs almost nothing: a written line before each trade with a number on it, and one hour a month scoring the log. That is the entire gap between the weather forecaster and the pundit, between the superforecaster and the dart-throwing chimp. You do not need to be smarter. You need to keep the receipt.

Start tonight. Before your next trade, write down a probability — an actual number — and the one thing that would make you wrong. In a month, bucket what you wrote and see whether your 70s happened 70% of the time. It will be humbling. It is supposed to be. Humbling is what calibration feels like from the inside, and it is the only feeling in this article you can trust — because it is the one that comes *after* the score, not before it.

None of this is individual financial advice; it is a method for making your own judgment honest. What you do with an honest probability is still up to you. But at least it will be honest.

## Sources & further reading

The research behind the headline claims:

- **Brier, G. W. (1950).** "Verification of Forecasts Expressed in Terms of Probability." *Monthly Weather Review*, 78(1), 1–3. The original Brier score — the report card for probabilistic forecasts, invented for weather.
- **Murphy, A. H., & Winkler, R. L. (1977).** "Reliability of Subjective Probability Forecasts of Precipitation and Temperature." *Journal of the Royal Statistical Society, Series C*. The evidence that weather forecasters are well-calibrated, and the origin of the reliability (calibration) diagram. [Publisher link](https://rss.onlinelibrary.wiley.com/doi/abs/10.2307/2346866).
- **Tetlock, P. E. (2005).** *Expert Political Judgment: How Good Is It? How Can We Know?* Princeton University Press. The 20-year study of 284 experts and 82,000+ forecasts; the dart-throwing chimpanzee; hedgehogs versus foxes. [Overview](https://braverangels.org/library/resources/expert-political-judgment-how-good-is-it-how-can-we-know/).
- **Tetlock, P. E., & Gardner, D. (2015).** *Superforecasting: The Art and Science of Prediction.* Crown. The Good Judgment Project, the IARPA tournament, and the finding that trained amateurs beat classified-access analysts by ~30%. [The Good Judgment Project (overview)](https://en.wikipedia.org/wiki/The_Good_Judgment_Project).
- **Alpert, M., & Raiffa, H. (1982).** "A progress report on the training of probability assessors." In Kahneman, Slovic & Tversky (eds.), *Judgment under Uncertainty: Heuristics and Biases.* The classic finding that people's 98% confidence intervals contain the truth only ~60% of the time — overprecision in one experiment, replicated hundreds of times.
- **Russo, J. E., & Schoemaker, P. J. H. (1989).** *Decision Traps.* The 10-question overconfidence quiz: across 1,000+ managers, most got 4–7 of 10 "90% confident" ranges wrong — a surprise rate of 40–70% where it should have been 10%.

Companion posts on this blog:

- [Calibration: Keeping Score on Your Own Forecasts](/blog/trading/analyst-edge/calibration-keeping-score-on-your-own-forecasts) — the full mechanics: the curve, the Brier decomposition, and the sizing math, in the Analyst's Edge series.
- [Overconfidence and the Illusion of Control](/blog/trading/trading-psychology/overconfidence-and-the-illusion-of-control) — the three faces of overconfidence and the P&L damage they do.
- [Thinking in Bets: Probabilistic Decision-Making](/blog/trading/trading-psychology/thinking-in-bets-probabilistic-decision-making) — treating every decision as a wager, and separating decision quality from outcome.
- [Process versus Outcome and the Trap of Resulting](/blog/trading/trading-psychology/process-versus-outcome-and-the-trap-of-resulting) — why a good decision can lose and a bad one can win, and how to judge yourself anyway.
- [The Trading Journal That Actually Changes Behavior](/blog/trading/trading-psychology/the-trading-journal-that-actually-changes-behavior) — how to build the log that makes the calibration drill stick.

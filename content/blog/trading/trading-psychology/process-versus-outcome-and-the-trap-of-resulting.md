---
title: "Process vs. Outcome: Why Judging Trades by Their Results Will Ruin You"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "In a game ruled by luck, grading your trades by their profit-and-loss reinforces lucky mistakes and punishes sound decisions. Here is the science of 'resulting,' why it quietly destroys traders, and the exact drill that fixes it."
tags: ["trading-psychology", "decision-making", "process-vs-outcome", "resulting", "outcome-bias", "expected-value", "luck-vs-skill", "decision-journal", "annie-duke", "behavioral-finance"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — In a probabilistic game, the result of a single trade tells you almost nothing about the quality of the decision behind it. Grading trades by their profit-and-loss is a broken measuring stick called *resulting*, and it corrupts the one thing that makes traders better: learning.
>
> - **Resulting** is the error of judging a decision by how it turned out. A good (+EV) decision can lose and a bad one can win, because short-run outcomes are dominated by luck, not skill.
> - Resulting poisons the feedback loop: you reinforce the lucky bad habit and abandon the unlucky good process — the two most expensive mistakes a trader can make.
> - The fix is a 2x2: grade **decision quality** (your process) on one axis and **outcome** (the P&L) on the other, and only ever act on the process axis.
> - **The number to remember:** a bet that wins just 40% of the time at 3-to-1 odds is solidly positive-expectancy, yet it *loses 6 times out of every 10* — six chances to "learn" the wrong lesson from a trade you should keep making.
> - **The drill:** grade the process, not the P&L. Keep a decision journal (thesis, odds, size rationale, invalidation) written *before* the trade, and review it before you ever look at the result.

On February 1, 2015, with 26 seconds left in Super Bowl XLIX, the Seattle Seahawks had the ball second-and-goal on the New England Patriots' 1-yard line. One yard from a second straight championship. They had the most punishing goal-line running back in football, Marshawn Lynch, standing in the backfield. Coach Pete Carroll called a pass. Russell Wilson threw a slant, an undrafted rookie named Malcolm Butler jumped the route and intercepted it, and the Seahawks lost. Within minutes it was being called the worst play call in Super Bowl history.

Here is the uncomfortable question that opens Annie Duke's book *Thinking in Bets*, and the question this entire article is built around: **was it actually a bad call?** That season, no NFL quarterback had thrown an interception on a pass from the opponent's 1-yard line — not one, all year. The odds of that specific disaster were tiny. The call had a defensible logic: throw once to freeze the clock, keep two running downs in reserve. The decision was reasonable. The *result* was catastrophic. And almost everyone confused the two.

That confusion has a name — **resulting** — and if you trade, it is quietly grading your report card every single day, usually wrong. The grid below is the mental model for the whole piece: the trap is that we grade the columns (what happened) when the only thing we control is the rows (how we decided).

![The decision-quality by outcome grid: good and bad process crossed with win and loss, with the two off-diagonal "trap" cells shaded amber.](/imgs/blogs/process-versus-outcome-and-the-trap-of-resulting-1.webp)

Read the grid top to bottom, not left to right. The two amber cells are where careers are made and unmade. **Bad beat** (good process, bad luck) is where a disciplined trader talks herself out of a winning system after a run of unlucky losses. **Dumb luck** (bad process, good luck) is where a reckless gambler mistakes a jackpot for genius and doubles down until the tail catches up. Both cells feel, in the moment, exactly like the diagonal cells next to them. That is the whole problem. Let's build the tools to tell them apart, from the ground up.

## Foundations: the building blocks of a broken measuring stick

You do not need any finance background for this section. We are going to define, from zero, the handful of ideas that make resulting so dangerous: what a "decision" even is when luck is involved, and the specific mental machinery that turns a random result into a false lesson. This is where the science lives.

### A decision is a bet, and a bet has an expected value

Start with the most basic reframe in all of decision-making: **every trade is a bet.** You are risking a known amount to win an uncertain amount, with uncertain odds. That is the definition of a bet, whether the ticket says "Seahawks -1" or "long 200 shares of AAPL."

Once something is a bet, we can measure its quality *before* we know how it turns out, using **expected value** — abbreviated **EV**. Expected value is just the average result you would get if you could make the same bet thousands of times. You compute it by weighting each possible outcome by its probability:

$$\text{EV} = p \times W - (1 - p) \times L$$

where `p` is your probability of winning, `W` is the amount you win when you win, and `L` is the amount you lose when you lose. A bet with positive EV (**+EV**) makes money on average; a bet with negative EV (**-EV**) loses money on average. The whole job of a trader is to find and repeat +EV bets. Notice what EV does *not* depend on: the result of any one trial. EV is a property of the decision, computed from the odds and payoffs, and it is fixed the moment you place the bet.

#### Worked example: the coin that wins 40% of the time and still prints money

Suppose I offer you a bet: risk \$100 on a coin that only comes up heads 40% of the time, but when it does, I pay you \$300; when it comes up tails, you lose your \$100. Most people's gut says "40%? That's a loser, pass." Let's compute it.

- Probability of winning: `p` = 0.40
- Amount won: `W` = \$300
- Amount lost: `L` = \$100

$$\text{EV} = 0.40 \times \$300 - 0.60 \times \$100 = \$120 - \$60 = +\$60$$

Every time you take this bet, you make \$60 on average. It is one of the best bets you will ever be offered. And yet — read this slowly — **it loses 60% of the time.** Six times out of ten you hand over \$100 and feel like an idiot. If you graded this bet by its results, you would quit it after two or three losses, right before the math bailed you out. The single most important sentence in this article: *a +EV decision that loses is still the right decision, and quitting it is the actual mistake.*

That gap — between a decision's quality and its result — is the entire territory of resulting. Now let's meet the three pieces of mental machinery that make us fall into it.

### Outcome bias: the lab has already proven you do this

In 1988, two researchers, Jonathan Baron and John Hershey, ran a now-classic set of experiments published in the *Journal of Personality and Social Psychology* ("Outcome Bias in Decision Evaluation"). They gave people descriptions of decisions made under uncertainty — some medical, some monetary gambles — and asked them to rate the *quality of the thinking*. The twist: different groups saw the same decision paired with different outcomes.

The finding was stark and it has been replicated for over 35 years. People rated the identical decision as *better thinking, made by a more competent person,* when it happened to work out — even when they were explicitly told to ignore the outcome, and even when they said they had ignored it. This is **outcome bias**: the human tendency to grade a decision by its result rather than by the information available when it was made. It is not a quirk of gamblers or novices. It is a default setting of the human mind, demonstrated in a controlled lab, again and again. When you catch yourself thinking "that was a dumb trade" the moment it loses, you are not being rigorous — you are exhibiting a bias with a citation.

### Hindsight bias: the story that rewrites itself

Outcome bias has a twin. In 1975, the psychologist Baruch Fischhoff demonstrated **hindsight bias** — the "I knew it all along" effect. Once people learn how something turned out, they systematically misremember how predictable it *was* beforehand. The 20% chance that came true gets remembered as "obvious." The interception that was a 2%-ish event gets remembered as "everyone could see that coming."

Hindsight bias is the accomplice that makes outcome bias feel justified. Resulting says "it lost, so it was a bad decision." Hindsight bias supplies the fake evidence: "and I could tell it would." Together they build an airtight, and completely false, case against a decision that was fine. For a trader, this is the mechanism behind every "I *knew* I should have sold" — a memory that did not exist until the price moved.

### Why noisy rewards breed superstition

Here is the deepest and most damaging piece, and it comes from the study of learning itself. Over a century ago, Edward Thorndike proposed the **law of effect**: behaviors followed by satisfying results get repeated; behaviors followed by unpleasant results get dropped. It is the engine of all learning by trial and error, and it works beautifully — *when the result is actually caused by the behavior.*

But what happens when the result is mostly random? In 1948, B.F. Skinner published a short, eerie paper in the *Journal of Experimental Psychology* titled "'Superstition' in the Pigeon." He put hungry pigeons in a box and delivered food on a fixed timer — completely independent of anything the bird did. The pigeons, of course, happened to be *doing something* each time the food arrived. And the law of effect did its work: a bird that was turning counter-clockwise when the pellet dropped started turning counter-clockwise more; another developed a head-tossing ritual; another a pendulum sway. Each pigeon had "learned" a superstition, because a few accidental pairings of a random reward with a behavior were enough to lock it in. Skinner himself drew the analogy to human rituals for changing one's luck at cards.

This is the science of what resulting does to a trader. When your reward — the P&L — is dominated by luck, the law of effect faithfully reinforces whatever you *happened* to be doing when you got paid. Held through the news against your plan and got lucky? Your brain logs "holding through news works." Sized up recklessly and hit? "Big size, big win." You are the pigeon, and the market is the fixed timer. The tell is that the reward is tracking the noise, not your process.

![Before-and-after comparison: when reward tracks the noisy outcome it trains a degrading superstition; when reward tracks the process it trains a compounding skill.](/imgs/blogs/process-versus-outcome-and-the-trap-of-resulting-3.webp)

The escape, shown on the right, is the whole point of this article: you have to move the reward off the outcome and onto the process. You have to grade the *decision*, so that the thing being reinforced is the thing you actually control. Everything that follows is machinery for doing that.

## 1. What "resulting" actually is

The word comes from professional poker, where distinguishing a good decision from a good result is a survival skill. **Annie Duke** — who won the 2004 World Series of Poker Tournament of Champions, beating Phil Hellmuth heads-up for a \$2 million top prize, with career live tournament earnings north of \$4 million — named and popularized the concept in her 2018 book *Thinking in Bets*. Her definition is exactly as blunt as it sounds:

> Resulting is the tendency to equate the quality of a decision with the quality of its outcome.

A poker player who shoves all-in as a 70/30 favorite and loses did not make a bad decision; they got unlucky 30% of the time, which is *supposed to happen 30% of the time.* A player who calls a hopeless bluff-catcher and happens to be right did not make a good decision; they got lucky. Duke's entire argument is that in any game where luck plays a role — poker, investing, business, football play-calling — the outcome of a single hand is a hopelessly noisy signal of decision quality, and building your learning on it is like calibrating a scale by weighing a feather in a windstorm.

The Pete Carroll call is her opening example precisely because it is so clean. Strip away the result and the decision is defensible; add the result back and the entire sports world graded it an F. The gap between those two grades *is* resulting, and it is measured in the amber cells of our grid.

### How the corruption actually spreads

Naming the bias is easy. Understanding the exact path by which it degrades a trader is what lets you interrupt it. Follow the arrows below: variance — the random scatter of outcomes around their average — sits *in between* your decision and your result, like a slot machine bolted onto the output. By the time you see a win or a loss, the signal from your decision quality has been drowned in noise.

![The resulting feedback loop: a decision passes through variance to a win or loss, which is graded by P&L, which reinforces or abandons the underlying habit, corrupting the next decision.](/imgs/blogs/process-versus-outcome-and-the-trap-of-resulting-2.webp)

Trace the loop. A decision of *some* quality goes in. Variance scrambles it into a win or a loss. Then resulting grades the trade by that win or loss. And here is the kill shot: that grade feeds straight back into your habits. A win reinforces whatever you did; a loss makes you abandon it — *regardless of whether the underlying decision was good or bad.* The next decision is now shaped by the last result, which was mostly luck. Do this a few hundred times and your process is no longer the product of your thinking; it is the product of your recent variance. That is not learning. It is the pigeon's dance.

#### Worked example: two identical decisions, two opposite grades

You and a friend independently spot the same setup and take the same +EV trade — the \$300-to-win, \$100-to-lose, 40%-hit bet from earlier, but sized up: each of you risks \$1,000 to win \$3,000.

$$\text{EV} = 0.40 \times \$3{,}000 - 0.60 \times \$1{,}000 = \$1{,}200 - \$600 = +\$600$$

Same decision, same +\$600 expected value, placed at the same time. The coin lands. You win \$3,000; your friend loses \$1,000. At the debrief, resulting hands out report cards: you are "sharp," your friend "should have known better." But the *decisions were identical.* The only difference between a genius and an idiot that day was a 40/60 coin. If you take that debrief seriously, you will both learn the wrong thing — you to trust a hot hand, your friend to distrust a sound process. *When two identical decisions get opposite grades, the grader is broken, not the decisions.*

## 2. Why a good decision loses and a bad one wins

To take resulting seriously you have to sit with an idea that feels almost offensive the first time: **over any short stretch, your results are mostly luck, even if your process is excellent.** Not "partly." Mostly. This is not cynicism; it is arithmetic, and once you see it you cannot unsee it.

The reason is **variance**. A +EV strategy makes money *on average*, but any individual sequence of trades scatters wildly around that average. Winning streaks and losing streaks are not signs that your edge turned on or off — they are the expected texture of a random process with an edge. A coin that is 55% weighted toward heads will still, routinely, come up tails five, six, seven times in a row. Not because it stopped being weighted. Because that is what randomness looks like up close.

![Bar chart of a positive-expectancy strategy's running P&L over 20 trades: red bars below zero for the first fourteen trades, reaching a 300-dollar drawdown, then green bars recovering to plus 200 dollars.](/imgs/blogs/process-versus-outcome-and-the-trap-of-resulting-4.webp)

Look at what a genuinely good strategy can do to you. The chart above is a strategy with a real, positive edge — it wins 55% of the time, making \$100 when it wins and losing \$100 when it loses, for an expected value of `0.55 × $100 − 0.45 × $100 = +$10` per trade. It *should* make money. And over these twenty trades it does: it ends up \$200 in profit, exactly its expectation. But watch the path. It goes underwater on the very first trade and *stays at or below breakeven for its first fourteen trades*, digging a \$300 hole along the way. Fourteen consecutive trades in which a trader running this exact system has every emotional reason to conclude "this is broken" — and every one of those conclusions would be resulting.

#### Worked example: the +EV strategy that spent 14 trades under water

Let's put numbers on the pain. Our strategy is +\$10 EV per trade. Over the first 14 trades, its *expected* profit is `14 × $10 = +$140`. Its *actual* profit at the low point (after trade 5) was **−\$300**. The gap between expectation (+\$140) and a plausible unlucky reality (−\$300) is \$440 — and nothing about the strategy changed. That gap is pure variance.

How long can a good strategy stay underwater? A useful rule of thumb: a strategy's worst drawdown tends to be *larger* than beginners ever expect, and periods spent below the previous high-water mark are measured in dozens of trades, not two or three. A trader who funds an account with \$1,000, runs a real edge, and quits at the \$700 mark — down \$300, "clearly not working" — just paid full price for the drawdown and left before collecting the edge that the drawdown was the cost of. *The market makes you pay for your edge up front, in the currency of drawdown, and resulting is what convinces you to stop paying right before it pays you back.*

### What this costs, and when it breaks

The practical cost is brutal and specific: **the traders who abandon good systems are disproportionately the ones who abandon them at the bottom of a drawdown** — that is, at the exact moment the system is cheapest to keep and most painful to hold. Resulting doesn't just make you quit; it makes you quit at the worst possible time, because the worst variance produces the strongest false signal. The only defense is to have decided, in advance and in writing, how much drawdown your process is *expected* to produce, so that a normal bad streak cannot masquerade as a broken system.

## 3. Skill vs. luck in a small sample

If short-run results are mostly luck, the obvious question is: how long is "short"? How many trades before your track record actually reflects your skill instead of your variance? The answer is: far more than you think, and the math is unforgiving.

Michael Mauboussin, in *The Success Equation*, frames every competitive activity on a **luck-skill continuum** — roulette at the pure-luck end, chess near the pure-skill end, and investing sitting uncomfortably close to the luck end over short horizons. His key insight, the **paradox of skill**, is counterintuitive: as the *players* in a field get more uniformly skilled, luck becomes a *bigger* determiner of who wins, not smaller. When everyone is excellent and closely matched, the residual differences in outcomes are increasingly down to chance. Modern markets, full of well-resourced professionals, are exactly this kind of arena. His formal condition is simple: luck overwhelms skill in the short run whenever the variance of luck is larger than the variance of skill — which, over a handful of trades, it almost always is.

There is a companion idea from Amos Tversky and Daniel Kahneman's 1971 paper, "Belief in the Law of Small Numbers": people wrongly expect small samples to look like the population they came from. We see a 6-trade winning streak and infer a real edge; we see a 6-trade losing streak and infer a broken one. Both inferences are statistically illiterate. Small samples are *supposed* to be streaky and misleading. Treating a short run as representative is the mathematical heart of resulting.

![Matrix comparing 10, 100, and 1,000 trades: the luck band shrinks from plus-or-minus 63 dollars to 20 to 6 dollars per trade against a fixed 20-dollar edge, so skill only becomes visible at 1,000 trades.](/imgs/blogs/process-versus-outcome-and-the-trap-of-resulting-5.webp)

The grid quantifies how slowly the fog lifts. The tool is the **standard error** — how far your *measured* average P&L is likely to sit from your *true* average, purely by chance. It shrinks with the square root of the number of trades:

$$\text{SE} = \frac{\sigma}{\sqrt{N}}$$

where the Greek letter sigma (`σ`) is the standard deviation — the typical size of a single trade's swing — and `N` is your number of trades. The square root is the villain here: to cut your uncertainty in half, you need *four times* as many trades. To cut it to a tenth, a hundred times as many.

#### Worked example: how many trades before you can trust the edge

Take a different strategy that makes \$20 per trade on average, with a typical single-trade swing (`σ`) of \$200. That is a genuinely strong edge. How long until it is visible above the noise?

- After **10 trades**: `SE = $200 / √10 = ±$63` per trade. Your measured average could easily be anywhere from −\$43 to +\$83 per trade. The \$20 edge is invisible; a losing sample is entirely normal.
- After **100 trades**: `SE = $200 / √100 = ±$20` per trade. The noise band now equals the edge itself. It is a coin flip whether your track record even looks profitable.
- After **1,000 trades**: `SE = $200 / √1,000 = ±$6` per trade. Now the \$20 edge stands clear of the \$6 noise band. Skill is finally legible.

To get the edge a comfortable *two* standard errors clear of zero — a rough bar for "this probably isn't luck" — you need `N` such that `$20 ≥ 2 × $200 / √N`, which solves to `N ≥ 400` trades. *Four hundred trades to be reasonably sure a strong edge is real — and most traders draw sweeping conclusions from their last five.* This is exactly the terrain of [keeping a calibration scorecard on your own forecasts](/blog/trading/analyst-edge/calibration-keeping-score-on-your-own-forecasts): the only way to know your edge is real is to accumulate a large, honestly-graded sample, not to trust the last handful of results.

## 4. How resulting quietly destroys traders: the two wrong lessons

Now we can assemble the pieces into the mechanism that actually ends trading careers. Resulting is dangerous in *both* directions — and the two directions are mirror images that share one root cause. Grading by outcome teaches the unlucky good trader to shrink her edge, and the lucky bad trader to grow his risk. Neither of them learns the thing that actually happened.

![Graph of two traders branching from one lucky short-run outcome: the good trader's plus-EV trade loses and she tightens into a minus-EV process; the reckless trader's minus-EV trade wins and he doubles his size.](/imgs/blogs/process-versus-outcome-and-the-trap-of-resulting-7.webp)

**Trader A** did everything right. She took a sound, +\$60-EV trade, sized appropriately. It lost \$1,000 — a completely ordinary outcome for a 40%-hit strategy, since it *loses 60% of the time.* But resulting whispers "your system is broken," and hindsight bias supplies the fake memory of warning signs. So she "fixes" it: she tightens her stops, promising herself she'll never take a \$1,000 hit again. **Trader B** did everything wrong. He sold a naked lottery ticket — a wildly -EV bet — and it happened to pay. Resulting shouts "free money," and he doubles his size. Watch how both of them, guided by the same broken instrument, walk directly into a worse process.

#### Worked example: the reckless bet that "worked"

Here is Trader B's trade in detail. He sells a far-out-of-the-money option — call it a bet where he collects \$200 in premium if nothing dramatic happens, which is *most* of the time, but he loses \$5,000 in the rare crash. Let's say the crash happens 10% of the time.

| Outcome | Probability | P&L | Contribution to EV |
| --- | --- | --- | --- |
| Quiet market (collect premium) | 90% | +\$200 | +\$180 |
| Crash (position blows up) | 10% | −\$5,000 | −\$500 |
| **Expected value** | 100% | | **−\$320** |

$$\text{EV} = 0.90 \times (+\$200) + 0.10 \times (-\$5{,}000) = +\$180 - \$500 = -\$320$$

This bet loses \$320 *every single time you take it*, on average. It is a terrible trade. But 90% of the time it pays \$200 and feels like the easiest money in the world.

![Bar chart of the reckless trade's expected value: a plus-180-dollar win-leg bar, a minus-500-dollar loss-leg bar, and a net minus-320-dollar expected-value bar, all against a zero baseline.](/imgs/blogs/process-versus-outcome-and-the-trap-of-resulting-6.webp)

The chart makes the trap visible. The win leg (green) contributes +\$180 to the average; the loss leg (red) subtracts \$500; the net is −\$320. Trader B collected his \$200, saw the green, and never saw the red, because the red only shows up in the sample once every ten tries. He "learned" a lesson that the math says is exactly backwards. *A bet that wins 90% of the time can still be one of the worst trades you can make, and the win rate is the disguise.*

#### Worked example: Trader A tightens her stop and kills her edge

Trader A's mistake is subtler and, honestly, more common among *good* traders, because it wears the costume of discipline. Her original trade had a +\$60 edge: win 40% of the time for +\$300, lose 60% for −\$100. After one unlucky loss, she tightens her stop to "protect herself." But a tighter stop does two things she didn't model. First, it gets tagged by ordinary noise before her thesis can play out, so her win rate drops from 40% to 30%. Second, out of fear she now takes profits early, cutting her average winner from \$300 to \$200. Recompute:

$$\text{EV}_{\text{tight}} = 0.30 \times \$200 - 0.70 \times \$100 = \$60 - \$70 = -\$10$$

She has converted a **+\$60 winner into a −\$10 loser** — a \$70-per-trade swing in the wrong direction — and she did it in the name of risk management, feeling *more* disciplined the whole time. This is resulting at its most insidious: it doesn't always make you reckless. Sometimes it makes you carefully, methodically, sand your edge down to nothing. *The most expensive tuition a good trader pays is "fixing" a process that was never broken, on the evidence of one unlucky trade.* When you feel the urge to overhaul your system after a loss, that is exactly the moment to ask whether you are looking at a broken thesis or [just noise](/blog/trading/analyst-edge/thesis-broken-or-just-noise-the-hardest-call-you-make).

## What it looks like at the screen

Theory is clean. Resulting, in the wild, is a set of physical and emotional tells that hit you in real time, before the rational part of your brain has any say. Learn to recognize the tells and you get a half-second of warning — which is sometimes all you need.

**After a lucky win on a rule-break**, the feeling is a warm, expansive certainty. You broke your own rule — held past your target, sized up on impulse, chased an entry — and it paid. There is a dopamine hit and, right behind it, a story: *"I'm reading this market perfectly right now."* You feel the urge to size up the next one. You feel a subtle reluctance to write the trade down honestly, because some part of you knows the journal will call it what it was. The tell is euphoria plus the impulse to *repeat the specific thing you did*. That impulse is the pigeon starting its dance.

**After an unlucky loss on a good trade**, the feeling is hot and contemptuous — usually aimed at your own system. *"This setup is garbage. This whole approach doesn't work."* You feel the urge to tear up the playbook mid-session. Your hand moves to tighten the next stop by half, or to shrink the next position to a third of its size, or — the quiet one — to simply *not take* the next perfectly valid signal, because you "don't trust it right now." Then, often, the revenge trade: an oversized, unplanned position to "make it back," which is resulting inverted, the losing pigeon flailing for the pellet. The tell is contempt plus the impulse to *change a process you validated an hour ago.*

The through-line in both states is the same: **an outcome you just witnessed is trying to rewrite a process you decided on with a clear head.** The physical cue — the flush of certainty or the flush of disgust — is your signal to freeze the keyboard. Not to trade on the feeling. Not against it either. Just to stop, and defer the judgment to the review, when the variance has been separated back out from the decision. The screen is where resulting happens; the journal is where it gets undone.

## Common misconceptions

**"If I'm losing money, I'm obviously doing something wrong."** Not obviously. Over any short window, a +EV trader loses money routinely — our example strategy was underwater for fourteen straight trades while being genuinely profitable. Losing money is *evidence*, but it is weak, noisy evidence over small samples, and treating it as proof is the core error. The right question is never "am I losing?" but "is my process still sound?" — a question the P&L cannot answer on its own.

**"Good traders win most of the time."** Plenty of excellent, highly profitable strategies win *less than half* their trades — trend-following systems famously win around 35-40% of the time and make everything on the size of the winners. Win rate and profitability are different quantities. Judging a trader by their win rate is resulting dressed up as analysis.

**"A big winner proves the trade was smart."** A big winner proves the trade *won*. Trader B's \$200 win came from a −\$320-EV bet. The size or presence of a profit tells you about the outcome, never about the decision. This is the single most expensive misconception in trading, because it is how gamblers get "confirmed."

**"Process over outcome means outcomes don't matter."** They matter enormously — *in aggregate, over a large sample.* Four hundred trades of P&L is real signal about your edge; four is noise. Process-over-outcome is not a denial that results matter; it is a claim about the *sample size* at which results become trustworthy. You judge single decisions by their process and you judge your process by its results — over hundreds of trades, not one.

**"Keeping a journal is just paperwork."** The journal is not a diary; it is the physical mechanism that separates decision quality from outcome. Without a pre-trade record of your thesis and odds, you have no way to grade the decision except through the result — which means you have no defense against resulting at all. The paperwork *is* the intervention.

## The drill: grade the process, not the P&L

Everything above is diagnosis. Here is the treatment — a concrete, repeatable protocol that moves your reward off the outcome and onto the process, so the law of effect starts reinforcing the right thing. It has three parts: a journal you fill in *before* the trade, a scorecard you fill in *after* but before you look at the P&L, and a weekly review that sorts everything into the 2x2.

### Part 1: the decision journal (written before the trade)

Before you put on a trade, you write down four things. This is non-negotiable and it takes ninety seconds. The reason it must be *before* is that a thesis written after the fact is contaminated by the result — that is hindsight bias, and it makes the whole exercise worthless.

![The decision-journal template as a table: rows for thesis, odds/EV, size rationale, and invalidation to write before the trade, plus a process grade after, with a worked example in each cell.](/imgs/blogs/process-versus-outcome-and-the-trap-of-resulting-8.webp)

The four fields, shown in the template above:

- **Thesis** — the setup, and specifically *why you believe it is +EV.* "AAPL breaking out on rising volume with the trend intact" beats "feels like it'll go up."
- **Odds / EV** — your honest estimate of the win probability and the payoff, netted for costs. Even a rough `55% × 2R − 45% × 1R = +0.65R` (where **R** is the amount you're risking — your unit of risk) forces you to confront whether the bet is actually positive. If you can't state an edge, you don't have a trade; you have a hope.
- **Size rationale** — how much you're risking and why *this* much. "0.5% of the book; medium conviction." Size is a decision too, and it gets resulted just as hard as entries.
- **Invalidation** — the specific price or fact that would prove the thesis wrong. "A close back below the breakout level." This is your pre-committed answer to "when do I get out," decided while you're calm.

### Part 2: the decision-quality scorecard (after the trade, before the P&L)

When the trade closes, before you tally the dollars, you grade the *decision* on a simple A-F scale, judged only against what you knew at the time. Did you follow your own plan? Was the thesis sound on the information available? Was the size right? Was the exit the one you pre-committed to, or an improvised panic? The grade has nothing to do with whether it won.

| Process grade | What it means | Example |
| --- | --- | --- |
| **A** | Flawless: sound thesis, correct size, followed the plan exactly | Took the A+ setup at planned size, exited at invalidation |
| **B** | Good decision, minor execution slip | Sound trade, but sized a touch large or entered a bit early |
| **C** | Defensible but sloppy | Right idea, hesitated, took a worse entry than planned |
| **D** | Poor decision that happened regardless of result | Off-plan entry, no clear invalidation, wrong size |
| **F** | Reckless: no edge, no plan, pure gambling | Revenge trade, or a -EV bet taken on a feeling |

Here is the discipline that makes it work: **an A trade that lost is still an A. An F trade that won is still an F.** You are grading the row of the 2x2, never the column. A trader who strings together a month of A and B grades is doing their job perfectly, *even in a losing month* — because over a large sample, good grades and dollars converge, and the grades show up first.

### Part 3: the weekly review (sort into the 2x2)

Once a week, go through every trade and drop it into one of the four cells of the opening grid: good-process/win, good-process/loss, bad-process/win, bad-process/loss. Then — and this is the entire trick — **you act only on the process axis:**

- **Good process, any outcome** → keep the process. Repeat it. Do not touch it because it lost.
- **Bad process, any outcome** → fix the process. Change it. Do *not* keep it because it won.

The wins in the "bad process" column are your most dangerous trades, because they are the ones tempting you to institutionalize a mistake. The losses in the "good process" column are your least dangerous, because they cost money but teach nothing — they are just the tuition on your edge. Reviewing on the process axis is how you make the law of effect finally start reinforcing decisions instead of luck. It pairs naturally with running a [pre-mortem on your thesis](/blog/trading/analyst-edge/stress-testing-your-thesis-with-a-pre-mortem) before the trade, so that your invalidation and your odds are stress-tested while you're still calm.

## How it shows up in real markets

Resulting is not a poker curiosity. It is one of the most reliably destructive forces in real trading and, more broadly, in every domain where luck and skill mix. Five episodes, each a case study in grading the outcome instead of the decision.

### 1. The call heard around the world

We opened with it, so close the loop. Pete Carroll's second-and-goal pass in Super Bowl XLIX (February 1, 2015) was graded by an entire planet as the worst decision in football history — because it was intercepted. Strip the result: the pass carried a tiny historical interception rate, and it preserved two running downs. Reasonable people can debate whether it was the *best* call, but the near-universal verdict of "catastrophically stupid" was pure resulting, driven entirely by a low-probability outcome that happened to land. It is the canonical example precisely because the decision and the result are so easy to separate once someone points at the gap. Annie Duke built a bestseller on that gap.

### 2. The blow-up artist

Victor Niederhoffer was, by the mid-1990s, one of the most celebrated hedge fund managers alive — a Harvard-trained statistician, a former George Soros collaborator, with a reported track record of roughly 30%+ annual returns over many years. His strategy leaned heavily on selling out-of-the-money put options on stock indices: collecting steady premium in exchange for taking on rare, catastrophic downside — the exact -EV-tail shape of Trader B's trade, run at institutional scale. For years it printed money, and the money "confirmed" his genius. On October 27, 1997, the Dow Jones Industrial Average fell 554 points — about 7% — in a single day. The puts he had sold detonated all at once; margin calls wiped out his funds, reportedly consuming more than \$100 million and his personal savings in a matter of days. The bitter epilogue: the market recovered within weeks and many of those puts expired worthless. The strategy's fatal flaw was never visible in the results — until the one time it was visible in all of them at once. (Niederhoffer, remarkably, rebuilt and blew up a *second* time in 2007 — a reminder of how hard the lesson is to actually learn.)

### 3. The amateur who "proved" anyone can win

In 2003, an accountant named Chris Moneymaker — an amateur who had qualified through a \$39 online satellite — won the World Series of Poker Main Event and its \$2.5 million first prize. It set off the "poker boom": hundreds of thousands of amateurs concluded that if a nobody could win the biggest event in poker, they had a real shot too. Most of them were resulting on a grand scale. One amateur winning a single, enormously high-variance tournament is exactly what you'd expect to happen *occasionally* by luck; it is close to zero evidence that any given amateur has an edge over professionals. The poker economy for the next decade was substantially funded by players who mistook the visibility of one lucky outcome for evidence about their own skill.

### 4. The lottery-ticket bull market

In the retail options and meme-stock frenzy of early 2021, a cohort of new traders bought far-out-of-the-money call options and heavily shorted-squeezed names and, for a while, got spectacularly rich. Screenshots of five- and six-figure gains circulated widely, and the lesson the crowd drew was "this is easy; I have a knack for this." Most of those gains came from -EV lottery bets that happened, in an extraordinary window, to pay — the mirror image of selling tails, but resulting all the same. When the variance mean-reverted, as it always does, a large share of those gains (and often the original stakes) went back to the market. The traders who kept the money were disproportionately the ones who graded the *process* — "this was a lucky -EV bet" — and stopped, rather than the ones who graded the outcome and doubled down.

### 5. Taleb's lucky fool

Nassim Taleb built his 2001 book *Fooled by Randomness* around a character who is now a fixture of trading psychology: the trader who runs a secretly -EV strategy — picking up nickels in front of a steamroller — enjoys a multi-year lucky run, and is celebrated (by himself and everyone around him) as a genius, right up until the tail arrives and erases everything. Taleb's point is that in a world with enough participants, some -EV strategies will *always* have a lucky survivor at any given moment, and that survivor will be the loudest voice in the room, offering the market exactly the wrong lesson. The "lucky fool" is not a strawman; he is the natural, statistically-guaranteed product of a large population of traders being graded by their outcomes.

## When this matters to you

If you trade — or run a business, or make any repeated decision under uncertainty — resulting is grading you constantly, and its default verdict is wrong in the two most expensive ways possible: it talks you out of good processes on unlucky days and into bad ones on lucky days. You cannot feel your way out of it, because the bias operates upstream of feeling; the euphoria and the contempt at the screen are the bias, not a reaction to it. The only reliable defense is structural: decide in writing before you know the result, grade the decision before you look at the P&L, and act only on the process axis of the 2x2.

None of this is investment advice, and none of it guarantees a profit — a good process can and will lose over any short stretch; that is the whole point. What a process-first discipline buys you is not the elimination of bad outcomes but the preservation of good decisions, so that your edge — if you have one — survives long enough in a large enough sample to actually pay you. In a game where luck runs the short term and skill only runs the long one, the trader who keeps grading the decision is the one still standing when the long term finally arrives.

The next step in this series looks at the flip side of the same coin: [separating luck from skill in your own track record](/blog/trading/analyst-edge/calibration-keeping-score-on-your-own-forecasts), so you can tell whether the edge you're protecting is real in the first place.

## Sources & further reading

- **Annie Duke, *Thinking in Bets* (2018)** — the book that named and popularized "resulting"; the Pete Carroll example opens it. Duke's WSOP record (2004 Tournament of Champions, \$2 million top prize; ~\$4.3 million career live earnings) is documented at [WSOP.com](https://www.wsop.com/players/annie-duke/) and [Wikipedia](https://en.wikipedia.org/wiki/Annie_Duke).
- **Jonathan Baron & John C. Hershey, "Outcome Bias in Decision Evaluation," *Journal of Personality and Social Psychology* 54(4), 569–579 (1988)** — the foundational lab demonstration of outcome bias, replicated repeatedly since. [Original PDF](https://bear.warrington.ufl.edu/brenner/mar7588/Papers/baron-hershey-jpsp1988.pdf).
- **B.F. Skinner, "'Superstition' in the Pigeon," *Journal of Experimental Psychology* 38, 168–172 (1948)** — accidental (non-contingent) reinforcement produces ritual behavior; the mechanism behind trading superstition. [Full text](https://psychclassics.yorku.ca/Skinner/Pigeon/).
- **Baruch Fischhoff, "Hindsight ≠ Foresight" (1975)** — the classic demonstration of hindsight bias, the accomplice that makes outcome bias feel justified.
- **Amos Tversky & Daniel Kahneman, "Belief in the Law of Small Numbers," *Psychological Bulletin* 76(2), 105–110 (1971)** — why we wrongly read small, streaky samples as representative.
- **Michael J. Mauboussin, *The Success Equation* (2012)** — the luck-skill continuum and the paradox of skill; a short summary lives at [Farnam Street](https://fs.blog/untangling-skill-and-luck/).
- **Nassim Nicholas Taleb, *Fooled by Randomness* (2001)** — survivorship, the "lucky fool," and hidden -EV tail strategies.
- **John Cassidy, "The Blow-Up Artist," *The New Yorker* (2007)** and the [Victor Niederhoffer profile](https://en.wikipedia.org/wiki/Victor_Niederhoffer) — the October 27, 1997 blow-up (Dow −554 points, ~7%) and the mechanics of selling tail risk.
- Sibling posts on this blog: [Calibration: keeping score on your own forecasts](/blog/trading/analyst-edge/calibration-keeping-score-on-your-own-forecasts), [Thesis broken or just noise?](/blog/trading/analyst-edge/thesis-broken-or-just-noise-the-hardest-call-you-make), and [Stress-testing your thesis with a pre-mortem](/blog/trading/analyst-edge/stress-testing-your-thesis-with-a-pre-mortem).

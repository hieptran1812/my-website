---
title: "Expected Value and Why Single Outcomes Lie"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "The result of one trade tells you almost nothing about whether it was a good decision. Here is expected value from first principles, why a +EV bet can lose and a -EV bet can win, and the exact drill that stops single outcomes from lying to you."
tags: ["trading-psychology", "expected-value", "probability", "law-of-large-numbers", "variance", "risk-of-ruin", "position-sizing", "ergodicity", "decision-making", "behavioral-finance"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — A single trade's result is one noisy sample from a wide distribution of outcomes. The quality of a decision lives in its *expected value* — the probability-weighted average over many repetitions — and one result can point the opposite way from the truth.
>
> - **Expected value (EV)** is the average payoff you would collect if you could make the same bet thousands of times: `EV = P(win) x avg_win - P(loss) x avg_loss`. It is fixed the moment you place the bet and does not depend on how any one trial turns out.
> - Two traps flow from this: a **+EV bet that loses** (you were unlucky, and still correct) and a **-EV bet that wins** (you were lucky, and still wrong). The outcome and the decision are different objects.
> - The **law of large numbers** is the only thing that ever separates skill from luck, and it works slowly: your measurement noise shrinks with the square root of the number of trades, so it can take *hundreds* of trades before a real edge is visible above the variance.
> - **The number to remember:** a bet that wins 95\% of the time can still be one of the worst trades you can make. Collecting \$50 nineteen times out of twenty is worthless if the twentieth time costs you \$1,200 — the net is -\$12.50 every single trade.
> - **The drill:** compute EV *before* every trade, size so a bad streak can never ruin you, and judge yourself on the aggregate of a hundred trades — never on this one.

You put on a trade. It loses. Everything in you wants to call it a mistake.

Now suppose a colleague, on the same afternoon, breaks every rule in his own playbook — chases a breakout with no plan, sizes triple his normal risk, holds through a news event he swore he would avoid — and walks away with a five-figure gain. The room congratulates him. Was he *right*?

Here is the uncomfortable truth this article is built on: **neither the loss nor the win tells you what you think it tells you.** In a game where luck runs the short term, the result of a single trade is almost pure noise about the quality of the decision behind it. Your losing trade may have been the correct bet, made again and again; his winning trade may have been a slow-motion catastrophe that happened, this once, to pay. The only measure that separates a good decision from a bad one is *expected value*, and expected value is a property of the bet, not of any one result. The diagram below is the mental model for the whole piece.

![A histogram of the possible outcomes of a single trade, with the expected value marked as the mean of the distribution and one far-from-average bar highlighted as the single result you actually saw.](/imgs/blogs/expected-value-and-why-single-outcomes-lie-1.webp)

Read it slowly. Every trade you take is a single draw from a distribution of outcomes you never get to see in full. The one bar you land on is loud — it hits your account, your mood, your ego. But the *stable* thing, the thing that actually describes the bet, is the mean of the whole distribution: its expected value. The highlighted red bar is a real, ordinary outcome, and it sits far from the mean. When you grade the bet by that one bar, you are letting a sample of one overrule the truth. Let's build the tools to stop doing that, from the ground up.

## Foundations: the building blocks of expected value

You need no finance or math background for this section. We are going to define, from zero, the handful of ideas that make single outcomes so misleading: what a "bet" is, what expected value means, and why one result is almost worthless as evidence. Everything after this section builds on these.

### Every trade is a bet

Start with the most useful reframe in all of trading: **a trade is a bet.** You are risking a known amount to win an uncertain amount, with uncertain odds. That is the definition of a bet, whether the ticket reads "long 200 shares" or "red on the roulette wheel." A *bet* is not a dirty word here and it does not mean gambling recklessly — it means any decision with a stake, an uncertain outcome, and a set of probabilities. Investing, market-making, and running a casino are all just portfolios of bets with different odds.

Once you accept that a trade is a bet, a powerful thing follows: you can measure its quality *before you know how it turns out*. You do not have to wait for the result. You judge it by the odds and the payoffs, which you can estimate the moment you place it.

### Expected value: the probability-weighted average

**Expected value** — abbreviated **EV** — is the average result you would get if you could make the same bet thousands of times. You compute it by weighting each possible outcome by its probability and adding them up. In its simplest two-outcome form:

$$\text{EV} = P(\text{win}) \times \text{avg win} - P(\text{loss}) \times \text{avg loss}$$

Here `P(win)` is your probability of winning, `avg win` is what you make when you win, `P(loss)` is your probability of losing, and `avg loss` is what you give up when you lose. A bet whose EV is positive (a **+EV** bet, read "plus-EV") makes money on average; a bet whose EV is negative (a **-EV** bet, "minus-EV") loses money on average. A *basis point* of edge, a coin weighted a hair toward you, a casino's tiny house advantage — all of them are just small positive EVs, repeated.

Notice the two words doing the heavy lifting: *average* and *many times*. EV says nothing about what happens on the next trade. It is the center of the distribution, not the next draw from it. The entire job of a serious trader is to find bets with positive expected value and repeat them enough times for the average to show up.

#### Worked example: the coin that wins 40% and still prints money

Suppose someone offers you a bet. You risk \$100 on a coin that comes up heads only 40\% of the time. When it lands heads, they pay you \$300; when it lands tails, you lose your \$100. Most people's gut says "40\%? That's a loser — pass." Let's actually compute it.

- Probability of winning: `P(win) = 0.40`
- Amount won on a win: `avg win = $300`
- Probability of losing: `P(loss) = 0.60`
- Amount lost on a loss: `avg loss = $100`

$$\text{EV} = 0.40 \times \$300 - 0.60 \times \$100 = \$120 - \$60 = +\$60$$

The figure below lays the same calculation out as a table — each outcome's payoff multiplied by its probability, summed down the last column.

![An expected-value table: the win row (40 percent, plus 300 dollars, plus 120 dollars), the loss row (60 percent, minus 100 dollars, minus 60 dollars), and the expected-value row summing to plus 60 dollars per trade.](/imgs/blogs/expected-value-and-why-single-outcomes-lie-2.webp)

Every time you take this bet, you make \$60 on average. It is one of the best bets you will ever be offered. And yet — read this twice — **it loses 60\% of the time.** Six times out of ten you hand over \$100 and feel like a fool. If you graded this bet by its result, you would abandon it after two or three losses, right before the math bailed you out. *A positive-EV bet that loses is still the correct bet, and quitting it is the actual error.*

### Stating your edge in units of risk

Traders often rewrite expected value in units of *risk* rather than dollars, because it makes edges comparable across positions of different sizes. Call the amount you risk on a trade — the distance from your entry to your stop — one unit of risk, or **1R**. A trade that can make three times what it risks is a `+3R` winner; a trade stopped out loses `-1R`. The coin above, in these terms, risks \$100 to make \$300, so it is a `+3R` bet won 40\% of the time. Its *expectancy per unit of risk* is:

$$\text{expectancy} = 0.40 \times (+3R) - 0.60 \times (-1R) = +1.2R - 0.6R = +0.6R$$

Every trade of this kind earns, on average, six-tenths of what it risks. This is the same +\$60 as before (0.6 times the \$100 risk), just expressed in a unit that lets you compare a tiny position to a huge one on the same scale. When a trader says "my system has a positive expectancy," this is the number they mean — and notice, again, that you can compute it entirely from the odds and payoffs, before a single trade is placed. If your estimated expectancy is not positive, no amount of good execution can save the bet; you are simply repeating a mistake with discipline.

### The law of large numbers: why one trade tells you almost nothing

Here is the deepest idea in the whole article, and it is the reason single outcomes lie. The **law of large numbers** is a theorem of probability that says: as you repeat a random bet more and more times, the *average* result you actually observe converges toward the expected value. Ten flips of a fair coin might give you seven heads; ten thousand flips will give you very close to five thousand. The average settles down as the count goes up. Skill — your edge, your EV — only becomes *visible* through this convergence, and only over many, many reps.

The flip side is what matters for your sanity: over a *small* number of trades, the average has not converged yet, so your observed results are dominated by luck, not edge. One trade is a sample of one. It tells you almost nothing about your EV, the same way one coin flip tells you almost nothing about whether a coin is fair. This is not a figure of speech; it is arithmetic we will make precise later with the standard error. For now, hold the core sentence:

> Your edge lives in the average over many trades. Any single trade is a sample of one, and a sample of one is mostly noise.

Amos Tversky and Daniel Kahneman gave this human failure a name in their 1971 paper "Belief in the Law of Small Numbers": people wrongly expect small samples to look like the population they came from. We see a five-trade winning streak and infer a real edge; we see a five-trade losing streak and infer a broken system. Both inferences are statistically illiterate. Small samples are *supposed* to be streaky and misleading. Treating a short run as representative is the mathematical heart of every mistake in this article.

## 1. The two ways a single outcome lies

Because the outcome and the decision are different objects, a single result can disagree with the truth in exactly two ways. Both feel, in the moment, like clear feedback. Both are lies. The grid below is the map.

![A two-by-two grid crossing the bet's expected value (good plus-EV versus bad minus-EV) against the single outcome (you won versus you lost), with the two off-diagonal cells shaded amber as the lies.](/imgs/blogs/expected-value-and-why-single-outcomes-lie-3.webp)

Read the rows, not the columns. The rows are what you control — whether the bet was +EV or -EV. The columns are what luck hands you — a win or a loss. On the green and red diagonal, luck agrees with the bet: a good bet won, or a bad bet lost, and the outcome tells the truth. On the two amber cells, luck and the bet disagree, and the single outcome lies to your face.

**The first lie is the +EV bet that loses.** You made the right bet — positive expected value, correctly sized — and it lost, because a +EV bet with a 60\% loss rate is *supposed* to lose most of the time. The result screams "your system is broken." It is not. You were still right. The trap here is quiet and it mostly catches *good* traders: they take a sound bet, lose, and "fix" a process that was never broken. Quitting is the mistake.

**The second lie is the -EV bet that wins.** Your colleague from the opening sold the trading equivalent of a lottery ticket — a wildly negative-EV bet — and it paid. The result shouts "free money, do it again." It is not free money; it is a bad bet that happened to land on its rare good outcome. The trap here catches *reckless* traders: they take a terrible bet, win, and institutionalize the mistake, sizing up until the tail finally arrives. Repeating is the mistake.

The two lies are mirror images, and they share one root cause: **variance**, the random scatter of outcomes around their average, sits between your decision and your result like a slot machine bolted onto the output. By the time you see a win or a loss, the signal from your decision quality has been buried in noise. If you want to separate the two, you cannot trust your eyes on a single trade — you have to know the shape of the noise. That is the next three sections.

Why is this so hard? Because your brain is a learning machine built for a world where results usually *are* caused by actions. Touch a hot stove, feel pain, learn not to touch it — the feedback is instant and honest, and a single trial is enough. That machinery, refined over millions of years, is superb when the outcome reliably follows the action. It is catastrophic when the outcome is mostly luck, because it will faithfully attach a "lesson" to whatever you happened to be doing when the reward or punishment landed. Win on a rule-break and the machinery encodes "rule-breaking works"; lose on a sound bet and it encodes "this setup is bad." You are not being weak or undisciplined when a single outcome rewires you — you are running the default firmware of a mind that evolved for a world with far less randomness than a market. Knowing that is the first defense: the feeling of "I just learned something" after one trade is precisely the signal to distrust.

This is the same territory that [thinking in bets](/blog/trading/trading-psychology/thinking-in-bets-probabilistic-decision-making) and [the trap of resulting](/blog/trading/trading-psychology/process-versus-outcome-and-the-trap-of-resulting) map from the decision-quality side; here we are coming at it from the mathematics of the outcome itself.

## 2. Variance is not the enemy of expected value

To take single outcomes seriously you have to sit with an idea that feels almost offensive the first time you meet it: **over any short stretch, your results are mostly luck, even when your process is excellent.** Not partly. Mostly. This is not cynicism — it is what a random process with an edge actually looks like up close.

The reason is variance. A +EV strategy makes money *on average*, but any individual run of trades scatters wildly around that average. Winning streaks and losing streaks are not signals that your edge switched on or off — they are the expected texture of the process. A coin weighted 55\% toward heads will still, routinely, come up tails five, six, seven times in a row. Not because it stopped being weighted. Because that is simply what randomness looks like before the law of large numbers has done its slow work.

![A running-equity chart of a positive-expectancy strategy over 100 trades: the actual path digs a 300-dollar drawdown and stays under water for its first 30 trades before recovering to end at plus 1,000 dollars, exactly the straight expectation line.](/imgs/blogs/expected-value-and-why-single-outcomes-lie-4.webp)

Look at what a genuinely good strategy can do to you. The chart shows a strategy with a real, positive edge: it wins 55\% of the time, making \$100 on a win and losing \$100 on a loss, for an expected value of `0.55 x $100 - 0.45 x $100 = +$10` per trade. It *should* make money, and over these hundred trades it does — it ends \$1,000 in profit, exactly its expectation. But watch the path. It goes under water almost immediately and *stays at or below breakeven for its first thirty-odd trades*, digging a \$300 hole. Thirty consecutive trades during which a trader running this exact, profitable system has every emotional reason to conclude "this is broken." Every one of those conclusions would be a lie told by a single stretch of outcomes.

#### Worked example: the +EV strategy that spent 30 trades under water

Let's put numbers on the pain. Our strategy is worth +\$10 of EV per trade. After its first 30 trades, its *expected* profit is `30 x $10 = +$300`. Its *actual* profit at the low point of the chart is about **-\$300**. The gap between what the edge predicts (+\$300) and a perfectly ordinary unlucky reality (-\$300) is \$600 — and nothing about the strategy changed to produce it. That entire \$600 swing is pure variance.

How long can a good strategy stay under water? Longer than beginners ever expect. Drawdowns — stretches spent below the previous high-water mark — are routinely measured in dozens of trades, not two or three. A trader who funds an account with \$1,000, runs a genuine +\$10 edge, and quits at the \$700 mark — down \$300, "clearly not working" — has just paid full price for the drawdown and walked away one step before collecting the edge the drawdown was the cost of. *The market makes you pay for your edge up front, in the currency of drawdown, and a single bad stretch is exactly what convinces you to stop paying right before it pays you back.*

The practical lesson is not "ignore your results." It is that a losing stretch is *weak evidence*, and you have to know in advance how much drawdown your +EV process is expected to produce, so a normal bad run cannot masquerade as a broken system.

## 3. How many reps before the edge is real

If short-run results are mostly luck, the obvious question is: how many trades before your track record actually reflects your skill instead of your variance? The answer is: far more than you think, and the math is unforgiving.

The tool is the **standard error** — a measure of how far your *observed* average P&L is likely to sit from your *true* average, purely by chance. It shrinks with the square root of the number of trades:

$$\text{SE} = \frac{\sigma}{\sqrt{N}}$$

The Greek letter sigma (`sigma`) is the *standard deviation* — the typical size of a single trade's swing, a plain measure of how much one trade bounces around. `N` is the number of trades. The square root is the villain: to cut your uncertainty in half, you need *four times* as many trades; to cut it to a tenth, a hundred times as many. The edge you are trying to detect is fixed, but the fog of noise only lifts at the pace of the square root.

![A chart of the noise band shrinking as trade count grows: at 10 trades the plus-or-minus 158-dollar band swamps a fixed 50-dollar edge, at 100 trades the band is plus-or-minus 50, and only by 400 trades does the band clear zero so the edge is real.](/imgs/blogs/expected-value-and-why-single-outcomes-lie-6.webp)

The chart quantifies how slowly the fog lifts for a genuinely strong strategy. Each bar is where your *measured* average P&L is likely to land by luck alone, around a true edge that never moves.

#### Worked example: 400 trades to trust a strong edge

Take a strategy that makes \$50 per trade on average, with a typical single-trade swing (`sigma`) of \$500. That is a genuinely strong edge. How long until it is visible above the noise?

- After **10 trades**: `SE = $500 / sqrt(10) = ±$158` per trade. Your measured average could easily sit anywhere from -\$108 to +\$208 per trade. The \$50 edge is invisible; a losing sample is entirely normal.
- After **100 trades**: `SE = $500 / sqrt(100) = ±$50` per trade. The noise band now equals the edge itself. It is roughly a coin flip whether your track record even looks profitable.
- After **400 trades**: `SE = $500 / sqrt(400) = ±$25` per trade. Now the \$50 edge stands two standard errors clear of zero — the rough bar for "this probably isn't luck."
- After **1,000 trades**: `SE = $500 / sqrt(1000) = ±$16` per trade. The edge is unmistakable.

To get the edge a comfortable two standard errors clear of zero, you need `N` such that `$50` is at least `2 x $500 / sqrt(N)`, which solves to `N ≥ 400` trades. *Four hundred trades to be reasonably sure a strong edge is real — and most traders draw sweeping conclusions from their last five.* Separating [luck from skill](/blog/trading/trading-psychology/luck-versus-skill-how-randomness-fools-you) is not a matter of willpower or intuition; it is a matter of sample size, and the sample you need is much larger than the one your emotions are reacting to.

## 4. The minus-EV "win," dissected

We have spent three sections on the first lie — the good bet that loses. The second lie, the bad bet that wins, deserves its own dissection, because it is the more seductive of the two and the one that ends accounts outright rather than quietly.

The most dangerous bets in markets are the ones that win *almost all the time*. A high win rate is the perfect disguise for a negative expected value, because the rare, catastrophic loss shows up so infrequently that you can collect the comfortable little wins for months and conclude you have found free money. You have found a slow-motion accident.

#### Worked example: the lottery ticket that pays

Say you sell a far-out-of-the-money option, or take any bet shaped like it: you collect \$50 whenever the market stays quiet, which is most of the time, but you lose \$1,200 in the rare crash. Suppose the crash happens 5\% of the time.

| Outcome | Probability | P&L | Contribution to EV |
| --- | --- | --- | --- |
| Quiet market (collect premium) | 95\% | +\$50 | +\$47.50 |
| Crash (position blows up) | 5\% | -\$1,200 | -\$60.00 |
| **Expected value** | 100\% | | **-\$12.50** |

$$\text{EV} = 0.95 \times (+\$50) + 0.05 \times (-\$1{,}200) = +\$47.50 - \$60 = -\$12.50$$

This bet loses \$12.50 *every single time you take it*, on average. It is a bad trade. But 95\% of the time it pays \$50 and feels like the easiest money in the world. The figure below shows why the feeling is a lie.

![A bar decomposition of the negative-EV bet: the win leg contributes plus 47.50 dollars, the rare loss leg subtracts 60 dollars, and the net expected value is minus 12.50 dollars per trade despite a 95 percent win rate.](/imgs/blogs/expected-value-and-why-single-outcomes-lie-7.webp)

The green bar — the win leg — contributes +\$47.50 to the average, and it is the only part you feel on nineteen trades out of twenty. The red bar — the loss leg — subtracts \$60, and you only see it once every twenty tries. The net is negative, but the loss that makes it negative is *invisible in any short sample*. A trader running this bet collects his \$50, sees only green, and "learns" a lesson the math says is exactly backwards. *A bet that wins 95\% of the time can still be one of the worst trades you can make, and the win rate is the disguise.*

The mirror image is just as common on the retail side: buying the far-out-of-the-money lottery ticket instead of selling it. Most of those tickets expire worthless — a steady drip of small -EV losses — but the rare one that pays off, splashed across a screenshot, convinces a crowd that the game is easy. Same math, opposite direction, same lie.

## 5. The casino's real secret is the law of large numbers

If you want to see expected value and the law of large numbers working together in the wild, look at a casino. Casinos are the purest EV machines humans have ever built, and their entire business model is a lesson in why single outcomes lie.

On an American roulette wheel there are 38 pockets — the numbers 1 through 36 plus a zero and a double-zero — but a winning straight-up bet pays only 35 to 1, as if there were 36 pockets. That gap is the whole edge. It works out to a house advantage of about **5.26\%**: for every dollar wagered, the casino expects to keep just over five cents. The European wheel drops the double-zero, leaving 37 pockets, and the edge falls to about **2.70\%**. These are small numbers — barely an edge at all on any single spin.

![A timeline of a casino's edge across increasing numbers of spins: a single spin can be a 350-dollar win or a 100-dollar loss dominated by luck, 100 spins still swing wildly, but by a million spins the house keeps almost exactly 5.26 percent like clockwork.](/imgs/blogs/expected-value-and-why-single-outcomes-lie-5.webp)

Here is the point the diagram makes. On a single spin, that 5.26\% edge is completely invisible — you might win \$350 and feel like a genius, or lose \$100 and feel robbed, and luck utterly dominates the tiny edge. Over a hundred spins the house's take still swings all over the place. But the casino does not play a hundred spins. It plays *millions* of spins a day across all its tables, and across millions of reps the law of large numbers turns that tiny 5.26\% from a hope into a certainty. The house's average result converges, ironclad, onto the edge. The casino never needs to win any particular spin. It needs only to keep the odds slightly in its favor and play enough times — and it has arranged its whole existence so that it always plays enough times.

That is the deepest lesson of this article, wearing a bow tie. The casino is +EV and it *still loses individual spins constantly* — it just does not care, because it has internalized that the single outcome is noise and the edge is destiny. A gambler at the same table, playing a handful of spins, experiences the exact opposite: for him the noise *is* the whole experience, and the edge never gets a chance to show up. Same wheel, same math, two completely different relationships with variance — and the difference is entirely the number of reps.

Ed Thorp, the mathematician who wrote *Beat the Dealer* in 1962, flipped this logic around: he found that by counting cards in blackjack, a player could turn the house's small edge into a small edge *for the player* — on the order of a percent or so. A percent is nothing on one hand. But Thorp understood that a small +EV, repeated relentlessly, is a fortune, and he won thousands in a weekend proving it. He was simply being the house.

## 6. The crowd's average is not your path

There is one more twist, and it is subtle enough that it is worth handling carefully, because it changes what "positive expected value" even means for *you*, personally, over time. So far we have treated EV as the thing you want to maximize. Usually that is right. But there is a class of bets where the expected value is positive and taking them repeatedly still ruins you — and understanding why is the bridge from EV to position sizing.

The distinction is between two different averages. The **ensemble average** is what happens across *many people* each taking the bet once — a snapshot across a crowd. The **time average** is what happens to *one person* taking the bet over and over — a single path through time. For most everyday bets these two averages agree, and we never notice the difference. But when your bets are *multiplicative* — when each result multiplies your bankroll rather than adding a fixed amount — they can point in opposite directions. This idea, worked out rigorously by the physicist Ole Peters and colleagues, is called ergodicity economics, and the cleanest illustration is a single coin.

![Two diverging lines from the same coin: the ensemble average across many players grows about 5 percent per round toward 432 dollars, while a single player's own compounding path decays about 5 percent per round toward roughly 20 dollars, near ruin.](/imgs/blogs/expected-value-and-why-single-outcomes-lie-8.webp)

#### Worked example: the +50/-40 coin that everyone should refuse

Start with \$100. You flip a coin. On heads, your wealth grows by 50\%; on tails, it shrinks by 40\%. Should you play, over and over?

The ensemble average says yes, enthusiastically. Averaging one heads and one tails: `0.5 x (+50%) + 0.5 x (-40%) = +5%` expected growth per round. Across a thousand people each playing once, the average wealth climbs. Positive EV, clear as day.

Now trace *your own* path through time. Heads then tails does not leave you flat — it compounds: `$100 x 1.5 = $150`, then `$150 x 0.6 = $90`. You are down to \$90 after one win and one loss. Every heads-tails pair, in any order, multiplies your wealth by `1.5 x 0.6 = 0.9` — a 10\% loss per pair. The number that governs *your* fate is not the arithmetic average (+5\%) but the compound growth rate per round, which works out to about **-5\%**. The ensemble average grows; your own path decays toward zero. In Peters' simulations, a crowd's *average* wealth soars while the *typical* individual — the median player — is driven to near-ruin. The green line is the crowd's mean; the orange line is you.

*The mean of a multiplicative bet is not your fate; the thing that compounds is, and a bet can be positive on average across a crowd yet negative for the one person who has to live through every round.* This is why position sizing exists. A bet with genuine positive EV can still ruin you if you size it so large that a normal losing streak multiplies your account down to nothing before the edge can compound in your favor. The fix is to keep each bet small relative to your bankroll, so your path stays additive enough for the law of large numbers to work *for* you instead of the risk of ruin working against you. Expected value tells you *which* bets to take; sizing tells you *how much* to bet so you survive long enough to collect the EV — a theme we return to in [position sizing as emotional regulation](/blog/trading/trading-psychology/position-sizing-as-emotional-regulation).

## What it looks like at the screen

Theory is clean. In the wild, the two lies arrive as physical and emotional states that hit you before the rational part of your brain gets a vote. Learn the tells and you buy yourself a half-second of warning, which is sometimes all you need to not act.

**After a -EV bet that won**, the feeling is a warm, expansive certainty. You broke a rule — sized up on impulse, chased an entry, sold the tail — and it paid. There is a dopamine hit and, right behind it, a story: *"I'm reading this market perfectly right now."* You feel the pull to size up the next one, to take the same shortcut again, and a quiet reluctance to write the trade down honestly, because some part of you suspects the journal will call it what it was. The tell is euphoria fused to the impulse to *repeat the specific thing you just did.* That impulse is the lucky fool being born.

**After a +EV bet that lost**, the feeling is hot and contemptuous, usually aimed at your own system. *"This setup is garbage. This whole approach doesn't work."* You feel the urge to tear up the playbook mid-session — to tighten the next stop by half, to shrink the next position to a third, or the quiet one, to simply *not take* the next perfectly valid signal because you "don't trust it right now." Then, often, the revenge trade: an oversized, unplanned position to make it back, which is the same error inverted. The tell is contempt fused to the impulse to *change a process you validated an hour ago.*

The through-line in both states is identical: **a single outcome you just witnessed is trying to overrule an expected value you estimated with a clear head.** The physical cue — the flush of certainty or the flush of disgust — is your signal to freeze, not to trade on the feeling and not against it either. Just stop, and defer the judgment to a moment when the variance has been separated back out from the decision. The screen is where single outcomes do their lying; a pre-committed process is where the lie gets caught.

## Common misconceptions

**"If I'm losing money, I'm obviously doing something wrong."** Not obviously. Over any short window, a +EV trader loses money routinely — our example strategy was under water for thirty straight trades while being genuinely profitable. Losing money is *evidence*, but it is weak, noisy evidence over small samples, and treating it as proof is the core error. The right question is never "am I losing?" but "is my expected value still positive?" — a question a short run of P&L cannot answer.

**"A big winner proves the trade was smart."** A big winner proves the trade *won*. The -EV lottery bet from Section 4 pays \$50 nineteen times out of twenty. The size or presence of a profit tells you about the outcome, never about the decision. This is the single most expensive misconception in trading, because it is exactly how a reckless gambler gets "confirmed" by luck.

**"Good traders win most of their trades."** Plenty of excellent, highly profitable strategies win *less than half* the time. Trend-following systems famously win only around 30 to 40\% of their trades and make everything on the size of the rare big winners. Win rate and profitability are different quantities; a bet's EV can be strongly positive with a low win rate, or strongly negative with a 95\% win rate. Judging a trader by win rate alone is letting the outcome lie to you in aggregate.

**"Expected value means I should take every +EV bet at any size."** No — this is where ergodicity bites. A +EV bet sized too large can still ruin you, because your wealth compounds multiplicatively and a bad streak can multiply you down to nothing before the edge compounds up. EV tells you *which* bets are worth taking; it does not tell you *how much* to bet. That is the separate, equally important job of position sizing.

**"With enough analysis I can know the result of the next trade."** You cannot, and chasing that certainty is a category error. The next trade is a single draw from a distribution; even a perfect read of your edge leaves the individual outcome uncertain. What analysis buys you is a better *estimate of the EV* — a distribution slightly more in your favor — not knowledge of the next draw. Trading is the management of a distribution, not the prediction of a point.

## The drill: compute EV before, judge the aggregate after

Everything above is diagnosis. Here is the treatment — a concrete, repeatable protocol that keeps single outcomes from doing your thinking for you. It has four parts, and none of them requires you to be right on any particular trade.

### Part 1: state the EV before you click

Before you put on a trade, you write down three numbers and combine them:

$$\text{EV} = P(\text{win}) \times \text{avg win} - P(\text{loss}) \times \text{avg loss}$$

Your honest estimate of the win probability, the average win, and the average loss — netted for costs. Even a rough `0.4 x $300 - 0.6 x $100 = +$60` forces you to confront whether the bet is actually positive. **If you cannot state the three numbers, you do not have a trade — you have a hope.** This single habit is the whole discipline in miniature: it moves your attention off "will this one work?" and onto "is this bet worth repeating?"

### Part 2: size so a bad streak can never ruin you

Once you know the bet is +EV, decide how much to risk — and here you respect ergodicity. Keep each trade small enough relative to your account that a normal losing streak, the kind Section 2 showed is completely ordinary, cannot compound you into a hole you can't climb out of. Risking a small fraction of your bankroll per trade keeps your path additive enough for the law of large numbers to work in your favor. The exact fraction is its own topic, but the rule of thumb is blunt: if a plausible run of losses would end you, you are betting too big, no matter how positive the EV.

### Part 3: judge the process on the aggregate, never on this one

When a trade closes, you are allowed to record the result — but you are *not* allowed to grade your decision on it. The decision is graded on the EV you estimated and whether you followed your own plan, judged only against what you knew at the time. A +EV trade that lost was a good trade. A -EV trade that won was a bad trade. You batch the grading: once every hundred trades or so, you look at whether your realized average is converging toward the EV you estimated. That is the only sample size at which the P&L becomes trustworthy signal about your edge.

### Part 4: pre-commit a drawdown budget

Because a +EV strategy produces ugly drawdowns as a matter of course, decide *in advance and in writing* how much drawdown your process is expected to produce. When a normal bad stretch arrives — and it will — a pre-committed budget lets you tell the difference between "this is the tuition on my edge" and "my edge is actually gone." Without that number written down beforehand, every drawdown looks like a broken system, and you will quit good processes at exactly the bottom, the most expensive possible moment.

> Compute the EV before the trade. Size so you survive the variance. Judge yourself on a hundred trades, never on this one.

## How it shows up in real markets

Expected value and the law of large numbers are not classroom theory. They are the machinery running underneath some of the most famous winners and blowups in finance. Six cases, each a lesson in trusting the aggregate over the single outcome.

### 1. The casino, the purest EV machine

We built Section 5 around it, so close the loop. A casino runs a tiny edge — about 5.26\% on American roulette, 2.70\% on the European wheel — and loses individual spins all night long without the faintest concern, because it has arranged to play millions of reps and let the law of large numbers convert that edge into near-certain profit. The casino is the ideal to aspire to and the warning at once: it wins by being utterly indifferent to single outcomes and utterly disciplined about repeating a small +EV. The gambler across the table, playing a few dozen spins, lives entirely inside the noise and never reaches the reps where the math would matter — which is the whole point of the house's design.

### 2. Ed Thorp becomes the house

In 1962, Ed Thorp published *Beat the Dealer* and proved mathematically that card counting could shift blackjack's small edge from the casino to the player. The per-hand edge was tiny — roughly a percent. On any single hand it was invisible, and Thorp lost plenty of hands. But he grasped that a small positive EV repeated relentlessly is a fortune, sat down, and won around \$11,000 over a weekend in the early trials. He then took the identical logic to Wall Street and ran one of the first quantitative hedge funds. Thorp's entire career is a monument to one idea: find a small +EV, then get enough reps that the single outcomes stop mattering.

### 3. Trend-following wins by losing most of the time

Managed-futures trend-following strategies are one of the longest-running, best-documented edges in markets, and they win only about 30 to 40\% of their individual trades. If you graded a trend-follower by any single trade, or even any single quarter, you would fire them — losing months and losing quarters are built into the approach, because the strategy takes many small, quick losses while waiting for the rare, enormous winners that pay for all of them and then some. The edge is entirely in the aggregate: a payoff structure where the average winner dwarfs the average loser, repeated across hundreds of trades and dozens of markets. Trend-followers survive precisely because they refuse to let a losing stretch — the +EV bet that loses, over and over — talk them out of a process with a real positive expectancy.

### 4. Powerball, the -EV bet that still mints winners

A Powerball ticket costs \$2, and the odds of hitting the jackpot are about 1 in 292 million. Lotteries are engineered to pay back only 50 to 60\% of ticket revenue as prizes, which puts the expected value of a typical \$2 ticket somewhere around negative \$0.90 to negative \$1.00 — you lose roughly half your money per ticket, on average. And yet, every few weeks, someone wins hundreds of millions of dollars. The winner is real; the game is still one of the worst bets on earth. Every jackpot is a -EV bet that landed on its one-in-hundreds-of-millions good outcome — the most extreme version of the second lie, dressed up in a press conference. That a single outcome was spectacular tells you nothing about whether the bet was ever worth making.

### 5. Renaissance's Medallion and the 50.75% edge

The Medallion fund at Renaissance Technologies, run by the mathematician Jim Simons, is widely regarded as the most successful trading operation in history — reportedly returning more than 66\% per year before fees (about 39\% after) from 1988 to 2018, according to Gregory Zuckerman's book *The Man Who Solved the Market*. How? Not by being right on big, dramatic calls. Robert Mercer, one of its leaders, reportedly said Medallion was right on only about **50.75\%** of its trades — barely better than a coin flip. But, he added, they were right *"50.75\% of the time... over millions of trades,"* and that razor-thin edge, repeated at enormous scale, was enough to make billions. Medallion is the casino's business model taken to its mathematical limit: a tiny +EV plus a staggering number of reps. It is the single most powerful real-world proof that the edge lives in the aggregate, not the individual trade.

### 6. Taleb's lucky fool

Nassim Taleb built his 2001 book *Fooled by Randomness* around a character who is now a fixture of trading psychology: the trader running a secretly negative-EV strategy — picking up nickels in front of a steamroller, like the option-seller in Section 4 — who enjoys a multi-year lucky run and is celebrated, by himself and everyone around him, as a genius, right up until the tail arrives and erases everything. Taleb's point is that in a large population of traders, some -EV strategies will *always* have a lucky survivor at any given moment, and that survivor will be the loudest voice in the room, offering the market exactly the wrong lesson. The lucky fool is not a strawman; he is the statistically guaranteed product of a crowd being judged by its single outcomes.

## When this matters to you

If you trade — or run a business, or make any repeated decision under uncertainty — single outcomes are grading you constantly, and their verdict is wrong in the two most expensive ways possible: they talk you out of good bets on unlucky days and into bad ones on lucky days. You cannot feel your way past this, because the euphoria and the contempt at the screen *are* the lie, not a reaction to it. The only reliable defense is structural: estimate the expected value before you know the result, size so the variance can't ruin you, and judge yourself on the aggregate of many trades rather than the memory of the last one.

None of this is investment advice, and none of it promises a profit — a +EV process can and will lose over any short stretch, which is the entire theme of this article. What an EV-first discipline buys you is not the elimination of bad outcomes but the preservation of good decisions, so that your edge — if you have one — survives enough repetitions to actually pay you. In a game where luck runs the short term and the law of large numbers runs the long one, the trader who keeps computing the expected value and keeps collecting reps is the one still standing when the long term finally arrives. The single outcome will always be loud. Your job is to stop letting it lie to you.

## Sources & further reading

- **Wizard of Odds, "Roulette" basics** — the house edge of 5.26\% (American, double-zero) and 2.70\% (European, single-zero) and how the 35-to-1 payout on 38 pockets produces it. [wizardofodds.com](https://wizardofodds.com/games/roulette/basics/).
- **Powerball official odds** — jackpot odds of roughly 1 in 292,201,338 on a \$2 ticket; lotteries return roughly 50 to 60\% of revenue as prizes, giving a typical ticket an expected value near -\$0.90 to -\$1.00. See [Powerball](https://www.powerball.com/) and any lottery expected-value calculator.
- **Ole Peters, "The ergodicity problem in economics," *Nature Physics* (2019)** and [ergodicityeconomics.com](https://ergodicityeconomics.com/) — the +50\%/-40\% coin, the ensemble-versus-time-average distinction, and why a positive-EV multiplicative bet can still ruin the individual who repeats it.
- **Edward O. Thorp, *Beat the Dealer* (1962)** — the mathematics of turning a small house edge into a small player edge via card counting, and the origin of "be the house." [edwardothorp.com](https://www.edwardothorp.com/books/beat-the-dealer/); background at [Wikipedia](https://en.wikipedia.org/wiki/Edward_O._Thorp).
- **Gregory Zuckerman, *The Man Who Solved the Market* (2019)** — the Medallion fund's reported >66\% gross / ~39\% net annual returns (1988-2018) and Robert Mercer's "right 50.75\% of the time... over millions of trades." Summarized at [Wikipedia: Renaissance Technologies](https://en.wikipedia.org/wiki/Renaissance_Technologies).
- **Amos Tversky & Daniel Kahneman, "Belief in the Law of Small Numbers," *Psychological Bulletin* 76(2), 105-110 (1971)** — why we wrongly read small, streaky samples as representative of the underlying edge.
- **Nassim Nicholas Taleb, *Fooled by Randomness* (2001)** — survivorship, the "lucky fool," and hidden negative-EV tail strategies that look brilliant until the tail arrives.
- Sibling posts on this blog: [Thinking in bets: probabilistic decision-making](/blog/trading/trading-psychology/thinking-in-bets-probabilistic-decision-making), [Process versus outcome and the trap of resulting](/blog/trading/trading-psychology/process-versus-outcome-and-the-trap-of-resulting), [Luck versus skill: how randomness fools you](/blog/trading/trading-psychology/luck-versus-skill-how-randomness-fools-you), and [Position sizing as emotional regulation](/blog/trading/trading-psychology/position-sizing-as-emotional-regulation).

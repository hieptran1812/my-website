---
title: "From Discretionary to Systematic: Writing Rules and Measuring Your Real Edge"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Most traders read the chart and decide in the moment, which makes their edge impossible to measure. This is the honest path from a discretionary feel to explicit, testable rules, the journal that records every trade, and the statistics that tell you whether you have a real edge or just a story."
tags:
  [
    "trading-rules",
    "trading-journal",
    "expectancy",
    "discretionary-trading",
    "systematic-trading",
    "r-multiple",
    "edge-measurement",
    "standard-error",
    "execution",
    "technical-analysis",
    "risk-management",
    "trading-process",
  ]
category: "trading"
subcategory: "Technical Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** -- you cannot improve what you do not measure, and you cannot measure a discretionary trader, because a discretionary trader changes the rules on every trade. The path to consistency is turning a feel into explicit, testable rules, journaling every trade by those rules, and computing the resulting statistics so your numbers actually mean something.
>
> - A **discretionary** trader "reads the chart" and decides in the moment. The trouble is not that judgment is bad; it is that an undefined process produces an *incoherent sample* -- 100 trades that were 100 different strategies -- and you cannot compute an edge on a moving target.
> - **Writing a setup as rules** means converting each part of the feel into one *if-this-then-that* clause: a bias condition, a level or zone, a trigger, an invalidation, a target, and a size. The test of a good rule is that two people reading the same chart would take the identical trade.
> - The **journal is your forward backtest**. A backtest tells you how rules would have done on old data; the journal tells you how *you* are doing on live data, which is the only number that pays. Record setup, entry, stop, target, size, the **R-multiple** result, the reason, and the mistake.
> - From your own journal you compute the things that matter: **win rate**, average win and loss in R, **expectancy** (the average R you make per trade), the **equity curve**, and the **standard error** that tells you how many trades it takes before your edge is real and not luck. A 30-trade record showing +0.35R per trade is *encouraging*, not *proven*.
> - The goal is not to delete all judgment. It is to find your place on the **discretionary-to-systematic spectrum** -- usually *rule-assisted discretion*, where the repeatable parts are rule-based so your statistics are honest, and judgment is reserved for the few things that genuinely need it.

A trader sits down at the end of a good month, up nicely, and tries to answer one simple question: *what exactly did I do?* They scroll back through their charts. Some trades were breakout buys. Some were pullback buys that look almost identical but were really the opposite idea. One was a short, taken because "it just felt toppy." A couple were revenge trades after a loss. The stops were in different places relative to the structure each time. The position sizes wandered with their confidence. By the time they have looked at twenty trades they realize they cannot describe their own strategy in a sentence, because there *was* no single strategy. There were twenty improvised decisions that happened to net out green this month.

That trader has a problem that no indicator, no pattern, and no new chart setup will fix. They have no way to know whether they have an edge. They cannot improve, because there is nothing fixed to improve. They cannot trust the green month, because they cannot tell skill from a lucky run of a process they never wrote down. This post is about the one move that breaks that trap: turning a discretionary *feel* into explicit *rules*, writing those rules into a journal, and reading the statistics that come out the other side. It is the least glamorous skill in trading and very close to the most important, because it is the skill that turns "I think this works" into "here is the number."

![The discretionary-to-systematic spectrum shown as three stages from a fully discretionary trader whose rules change every trade and cannot be measured, through rule-assisted discretion where the repeatable parts are fixed and expectancy becomes meaningful, to a fully systematic process that is fully measurable but brittle to regime shifts.](/imgs/blogs/from-discretionary-to-systematic-1.png)

The diagram above is the mental model for the whole post. On the left is the fully discretionary trader, who decides everything in the moment and therefore has no coherent sample to measure. On the right is the fully systematic trader, who has coded every decision and can measure everything but has handed the wheel to a machine that does not know when the regime has changed. The honest destination for almost everyone is in the middle: *rule-assisted discretion*, where the repeatable parts -- entry, stop, target, size -- are written as rules so the statistics mean something, and judgment is spent only where it actually adds value. Everything that follows is about how to get from the left of that picture to the middle, and how to read the numbers once you are there.

A note before we start: this is educational. It explains a *process* for measuring your own trading honestly. It is not advice to trade anything, and turning your trading into rules will not by itself make a losing method profitable -- it will, however, finally let you *see* whether the method loses, which is the necessary first step to fixing it.

## Foundations: why discretion can't be measured

Before we can write a single rule, we need to be precise about why the discretionary approach is so hard to measure. The reason is not that discretionary traders are undisciplined or that judgment is worthless. Some of the best traders alive are highly discretionary. The reason is statistical, and it is worth getting exactly right.

### What "discretionary" actually means

A **discretionary** trader makes each decision -- whether to enter, where to put the stop, how big to go, when to exit -- by judgment in the moment, looking at the current chart. There is no fixed procedure; there is an experienced human applying intuition to a fresh situation. The opposite pole is **systematic** (also called *mechanical* or *rules-based*): every decision follows from a written rule, and given the same chart the process produces the same trade every time.

Most real traders are somewhere in between, but the *default* starting point for nearly everyone is heavily discretionary, because that is how human pattern recognition works. You look at a chart, something clicks, and you act. The click feels like knowledge. Sometimes it is. The problem is what the click does to your ability to *measure*.

### A sample is only coherent if the trades are the same kind of bet

Here is the statistical heart of it. To say a strategy "has an edge" is to make a claim about a population of trades: *if I take this kind of trade many times, the average result is positive.* To estimate that average, you collect a **sample** -- a set of actual trades -- and compute the average over the sample. This only works if every trade in the sample is drawn from the *same population*: the same kind of bet, taken the same way, for the same reason.

A discretionary trader violates this on every trade. Trade 1 was a breakout. Trade 2 was a pullback. Trade 3 was a "feels toppy" short. Trade 4 was the same chart as trade 2 but taken with a wider stop because they were nervous. These are not 100 samples of one strategy. They are 100 samples of 100 different strategies, with sample size *one* each. You can average the numbers -- nothing stops you typing them into a spreadsheet -- but the average answers no question, because there is no single strategy whose edge it estimates. The technical phrase is that the sample is *not identically distributed*: the trades do not come from a common distribution, so the law of large numbers, which is what makes averages meaningful, simply does not apply.

This is why a discretionary trader's track record, however green, is so hard to trust or repeat. As [the expectancy post explains in detail](/blog/trading/technical-analysis/expectancy-why-win-rate-lies), expectancy -- the average money you make per trade -- is the number that decides whether a method makes money. But expectancy is an average *over a strategy*, and a discretionary trader has no fixed strategy to average over. You are trying to compute the average of a moving target.

### The moving-target problem, made concrete

Imagine you flip a coin 30 times and it comes up heads 18 times. You might reasonably ask whether the coin is biased. The question is answerable because every flip was the *same* coin under the *same* conditions: a coherent sample. Now imagine that on each flip you secretly swapped in a different coin, sometimes a fair one, sometimes a weighted one, sometimes a two-headed one, and you do not remember which was which. Eighteen heads out of thirty now tells you essentially nothing about any single coin. That is the discretionary trader's sample. Each trade is a different coin.

The fix is not to stop using judgment. The fix is to *hold the coin fixed* -- to define one repeatable trade precisely enough that every instance is the same bet -- so that the sample becomes coherent and the average finally means something. That defined, repeatable trade is exactly what a fully-specified **setup** is. We built one end to end in [the high-probability setup post](/blog/trading/technical-analysis/building-one-high-probability-setup): a setup is not a secret pattern but a complete, written plan -- bias, level, confluence, trigger, invalidation, target, size -- such that the same chart always produces the same trade. The rest of this post is about treating that written setup as the unit of measurement, and building the journal and statistics around it.

### Why this matters before we touch a single rule

It is tempting to skip straight to "here are the rules to write." But if you do not internalize *why* discretion can't be measured, you will write rules and then quietly break them, because in the moment the click will feel more trustworthy than the rule. The discipline to follow a written rule comes from understanding that the rule is not a cage -- it is the thing that makes your numbers real. Break the rule and you are back to swapping coins; your statistics dissolve, and you are flying blind again, no matter how good the click felt.

## Writing a setup as rules

The core move is to take a vague impression -- "this looks like a good long here" -- and decompose it into a small set of *if-this-then-that* clauses, each precise enough to be checked mechanically. We are not trying to capture everything you know. We are trying to capture the *repeatable* part: the conditions that, when present, define this particular trade.

![Six if-this-then-that rules in sequence: a bias condition selecting long-only above the 50-day average, a level being the demand zone, a trigger being a fifteen-minute bullish engulfing close, an invalidation at the stop defining one R, a target at resistance giving two R, and a fixed one-percent position size of forty shares.](/imgs/blogs/from-discretionary-to-systematic-2.png)

The diagram above shows the six clauses, and the figure is the template for every rule set you will ever write. Each box is one *if-this-then-that* condition. Read together, they fully specify a trade: given a chart, you can mechanically determine whether this is a trade, exactly where you get in, exactly where you get out if you are wrong, exactly where you take profit, and exactly how many shares you buy. Let us walk each clause and why it has to be there.

### The bias condition: which direction are we even allowed to trade?

The first clause filters direction. "IF the daily chart closes above its 50-day moving average, THEN I take long trades only." A **moving average** is just the average closing price over the last N days, which smooths the noise so you can see the prevailing trend; trading only in the direction of the higher-timeframe trend is one of the oldest edges in the book. The point of writing it as a rule is that it removes a decision you would otherwise make by feel. Without the rule, on a red day you might short because it "feels weak"; with the rule, you simply do not, because the condition for shorting is not met.

### The level or zone: where, specifically, are we looking?

The second clause fixes *location*. "IF price is inside the \$100-\$101 demand zone, THEN this chart is a candidate." A **demand zone** is a price area where buyers previously stepped in aggressively, leaving a visible base before a strong move up; it is a place where a bounce is more likely than at a random price. The rule turns "near support" -- which is vague, because *near* could mean anything -- into a bounded range you can check with a ruler. Either price is in the \$100-\$101 box or it is not.

### The trigger: what specific event lets us pull the trigger?

The third clause is the *timing* event, and it is the one beginners most often leave implicit. "IF a 15-minute bullish engulfing candle closes inside the zone, THEN enter at its close." An **engulfing candle** is a candle whose body completely covers the prior candle's body; a bullish one closing in a demand zone is a small, visible sign that buyers have taken control right where we expected them to. Without a trigger rule, you enter "when it looks ready," which is not a rule at all -- it is the click again. The trigger converts a candidate into an actual entry at a *specific, recordable* price.

### The invalidation: where are we provably wrong?

The fourth clause defines the **stop** -- the price at which the trade idea is wrong and you exit for a loss. "Stop at \$98.50, below the zone and the prior swing low." This single clause does two jobs. First, it caps your loss: you know before you enter the worst case. Second, it *defines your unit of risk*. The distance from entry (\$101) to stop (\$98.50) is \$2.50 per share, and that \$2.50 is what we call **1R** -- one unit of risk. As [the expectancy post lays out](/blog/trading/technical-analysis/expectancy-why-win-rate-lies), measuring trades in R-multiples is what lets you compare a \$2.50-risk trade to a \$25-risk trade on equal footing: a full stop-out is always -1R, regardless of the dollar amount. No invalidation rule means no 1R, and no 1R means no coherent statistics.

### The target and the size: reward and exposure

The fifth clause sets the **target** -- where you take profit -- and with it the reward-to-risk ratio. "Exit at \$106 resistance, which is +\$5 per share, or +2R; if the nearest sensible target is less than 2R away, skip the trade." The sixth clause sets **position size**: "risk 1% of a \$10,000 account, which is \$100; \$100 of risk divided by \$2.50 of risk per share is 40 shares." With size fixed by a rule, 1R becomes a constant *dollar* amount -- \$100 -- so a +2R winner is always +\$200 and a -1R loser is always -\$100, and your equity curve in R-multiples translates cleanly into dollars.

### The test of a good rule: two readers, one trade

How do you know whether a clause is precise enough? Use the *two-reader test*: if you handed the chart and the rule to two competent people, would they take the identical trade -- same entry, same stop, same size? "Buy when it looks strong" fails: two readers will disagree on what "strong" means. "Enter on the close of a 15-minute bullish engulfing inside the \$100-\$101 zone" passes: two readers will mark the same candle and the same price. Every clause in your rule set should pass this test. Where it cannot -- and some genuinely cannot, like "is this news environment too dangerous to trade?" -- that is precisely the judgment you keep, and we will return to it when we discuss the spectrum.

#### Worked example: a vague "I buy support bounces" turned into a 6-line rule set

Let us do the actual conversion. A trader describes their method as: *"I buy bounces off support in an uptrend when it looks like buyers are stepping in, and I sell into resistance."* That sentence is a feel, not a rule. A second reader cannot reproduce a single trade from it. Here is the same idea rewritten as six clauses, each passing the two-reader test:

1. **Bias.** IF the daily close is above the 50-day moving average, THEN longs only. (Defines "in an uptrend.")
2. **Level.** IF price trades into a marked demand zone -- a prior base from which price rallied at least 3R -- THEN it is a candidate. (Defines "support.")
3. **Trigger.** IF a 15-minute bullish engulfing candle closes inside the zone, THEN enter at its close. (Defines "buyers stepping in.")
4. **Invalidation.** Stop at the low of the zone minus 0.2R of buffer. Entry minus stop = 1R. (Defines the risk unit.)
5. **Target.** Exit at the nearest prior resistance level; require it to be at least 2R away or skip the trade. (Defines "sell into resistance" and enforces a minimum reward.)
6. **Size.** Risk a fixed 1% of account equity per trade; shares = (1% of equity) / (entry - stop). (Fixes exposure so 1R is a constant dollar amount.)

Notice what happened. The trader did not learn a new method. They wrote down the one they already had, in a form a second reader could execute identically. That is the entire move. The one-sentence intuition: *a rule set does not add to your edge; it makes your existing edge measurable by fixing the trade so every instance is the same bet.*

## The journal: your forward backtest

Once you have rules, you need a record of every trade taken against them. That record is the **journal**, and it is the single most undervalued tool in trading. To see why, contrast it with its more famous cousin, the backtest.

A **backtest** runs your rules over *historical* data and reports how they would have performed. It is genuinely useful -- we cover its many traps in [the backtesting post](/blog/trading/technical-analysis/backtesting-without-fooling-yourself) -- but it has a fatal limitation as a measure of *your* edge: it measures the rules in a vacuum, not you executing them. A backtest does not include the trade you skipped because you were scared, the stop you widened because you "knew" it would come back, the winner you cut early because you could not stand the heat, or the revenge trade you took after a loss. The backtest is the strategy's potential. The journal is your *realized* result.

That is why the right way to think about the journal is as a **forward backtest**: instead of testing the rules on old data, you are testing them on new, live data, one trade at a time, *with you in the loop.* It is the only honest record of your actual edge, because it is the only record that includes everything you actually did.

![A journal matrix with one column per trade and rows for setup, entry, stop, target, size, R-multiple result, reason, and mistake; the winner shows plus two R and the loser shows minus one R, both recorded in identical fields so the log can later be averaged into win rate, average R, and expectancy.](/imgs/blogs/from-discretionary-to-systematic-3.png)

### What to record, and why each field earns its place

The diagram above shows the columns. Every trade gets one row. The fields are not arbitrary; each one is needed either to *reconstruct* the trade or to *compute* a statistic later.

- **Setup / rule ID.** Which of your defined setups was this? You may run more than one. Tagging the setup lets you later compute expectancy *per setup* and discover that setup A is your money-maker and setup B is a leak.
- **Entry, stop, target.** The three prices. Entry minus stop is 1R; this is what makes the R-multiple computable. Without the stop recorded, you cannot express the result in R, and the whole statistical apparatus collapses.
- **Size.** Shares or contracts, and the resulting dollar risk. This ties R back to dollars and lets you check that you actually sized by the rule.
- **R-multiple result.** The outcome expressed in R: +2R, -1R, +0.4R, -0.3R. This is the field you will average. Record it as the *actual* result, not the planned one -- if you exited at +1.2R instead of the +2R target, you write +1.2R, because the gap between planned and actual is exactly the execution information you are after.
- **Reason for entry.** One line: which conditions were present. This catches the trades you took that did *not* actually meet your rules -- the ones where the reason reads "looked good" instead of naming the trigger. Those are your discipline leaks, and you cannot find them if you do not write the reason.
- **Mistake (if any).** The most valuable field and the one most often left blank. "Moved my stop." "Entered before the trigger fired." "Doubled size on a hunch." "None -- clean rule trade." This field is where the journal stops being a scoreboard and becomes a coaching tool.

### A screenshot is worth a thousand remembered trades

Add one more thing the table cannot hold: a **screenshot** of the chart at entry, marked with your level, trigger, and stop. Memory is not just imperfect; it is *actively misleading*, because it rewrites losing trades into "I knew that was bad" and winning trades into "I called that perfectly." The screenshot freezes what you actually saw, so that when you review the trade a month later you are reviewing the real decision, not a flattering reconstruction. The screenshot plus the written reason is what makes the journal a *forward backtest* rather than a *forward fairy tale.*

### Why the journal, and only the journal, is the honest record

A backtest can be curve-fit. A demo account does not carry real emotional weight, so your execution there is unrealistically clean. A vague memory of "I'm up this month" tells you nothing about *how* or *whether it repeats*. The journal is the only record that is simultaneously (a) about real money, with the real psychology that distorts execution, (b) tagged by rule so the sample is coherent, and (c) detailed enough to compute statistics from. It is, quite literally, the dataset of you. Everything in the next section is computed from it.

### Beating the friction that kills journaling

The reason most traders do not have a usable journal is not that they disagree with any of the above; it is *friction*. A journal that takes ten minutes per trade to fill out will be abandoned inside a week, and an abandoned journal measures nothing. So the practical craft of journaling is making the record cheap enough to actually keep while still capturing the fields the statistics need. A few principles make this work. First, **log at the moment of entry, not at the end of the day** -- the moment you place the trade, you already know the setup, entry, stop, target, and size, so write those five fields and the screenshot in under a minute while they are certain; come back only to fill the R-result and the mistake when the trade closes. Logging from end-of-day memory reintroduces exactly the rewrite problem the journal exists to defeat.

Second, **the R-result and the mistake are the two fields you must never skip, even when you skip everything else.** If you are too busy to log a full row, log those two: the outcome in R and the one-line honest note on what, if anything, you did wrong. Those are the fields the statistics and the leak analysis depend on, and they are the fields memory corrupts fastest. Third, **standardize the setup IDs** so that tagging is a one-token choice, not an essay -- "DZ-long" for the demand-zone long, "BO-cont" for the breakout continuation -- because a free-text setup field is both slow to write and impossible to group by later. The whole point of the journal is that at the end of a quarter you can filter to one setup ID and compute its expectancy; that only works if the IDs are clean. A journal you keep imperfectly for three hundred trades is worth infinitely more than a perfect template you abandon after ten.

## Measuring your real edge from your own trades

Now we turn the journal into numbers. The beautiful thing about a journal of rule-tagged trades expressed in R-multiples is that the statistics fall out with grade-school arithmetic. You do not need a quant background. You need to be able to count, add, and divide.

### Win rate: the split, and only the split

The **win rate** is the fraction of trades that made money: wins divided by total trades. If you took 30 trades and 13 were profitable, your win rate is 13/30 = 43%. Win rate is the first number everyone computes and the most over-weighted. As [the expectancy post argues at length](/blog/trading/technical-analysis/expectancy-why-win-rate-lies), win rate is *only the split* -- how often you win -- and tells you nothing about how much you win when you win or lose when you lose. A 90% win rate can lose money and a 40% win rate can be a goldmine. So compute it, but do not stop there.

### Average win and average loss, in R

Next, average your winners and your losers *separately*, in R. Suppose your 13 winners average +2.0R and your 17 losers average -0.91R (most were full -1R stop-outs, a few were small -0.5R scratches). These two numbers, together with the win rate, are everything you need.

### Expectancy: the average R you make per trade

**Expectancy** is the average R-multiple across *all* trades, winners and losers together. It is the single number that says whether the method makes money. The formula is:

$$E[R] = p \times W - (1-p) \times L$$

where $p$ is the win rate, $W$ is the average win in R, and $L$ is the average loss in R (as a positive number). Plugging in our journal:

$$E[R] = 0.43 \times 2.0 - 0.57 \times 0.91 = 0.87 - 0.52 = +0.35\text{R}$$

So each trade is worth, on average, +0.35R. With 1R fixed at \$100, that is +\$35 of expected profit per trade. The number is positive, which is the whole game: a positive expectancy means the method makes money as you repeat it; a negative one means it bleeds no matter how often it wins.

![A bar chart decomposing expectancy into a green win-leg bar of plus zero point eight seven R, a red loss-leg bar of minus zero point five two R, and a green expectancy bar of plus zero point three five R, with the formula expectancy equals win rate times average win minus loss rate times average loss shown beneath.](/imgs/blogs/from-discretionary-to-systematic-5.png)

The diagram above is the expectancy calculation drawn as bars. The green win-leg bar (+0.87R) is what the winners contribute; the red loss-leg bar (-0.52R) is what the losers take away; the difference, +0.35R, is the expectancy. The picture makes the structure of an edge legible at a glance: an edge is just the win leg being taller than the loss leg, and *both* the win rate and the payoff feed into the heights. This is why you cannot judge a method by win rate alone -- win rate only sets how much of the win leg you get; the payoff sets how tall it is.

### The equity curve: watching the edge compound (and drawdown)

If you plot the *cumulative* R after each trade -- start at 0, add +2 after a winner, subtract 1 after a loser, and so on -- you get the **equity curve**: a running picture of your account in R-units over the sequence of trades. A positive-expectancy method produces an equity curve that *drifts upward* over many trades, but never in a straight line. It climbs, gives some back, climbs again. The give-backs are **drawdowns** -- peak-to-trough declines -- and they are completely normal even for a real edge.

![An equity curve over thirty journaled trades rising from zero R to plus ten point five R, built from short green up-segments for winners and red down-segments for losers, with a labeled drawdown around trade thirteen where the curve retraces from about plus five R back toward plus two point five R before resuming its climb to a net plus ten point five R.](/imgs/blogs/from-discretionary-to-systematic-4.png)

The diagram above is the equity curve of exactly the 30-trade journal we have been computing. It rises from 0 to +10.5R, which is precisely 30 trades times +0.35R expectancy. The labeled dip around trade 13 is a real drawdown: the curve runs up to about +5R, then gives back to about +2.5R over a losing cluster, then resumes climbing. If you had stopped trading or abandoned the method at the bottom of that dip, you would have walked away from a positive edge at the worst possible moment. The equity curve is the picture that teaches you that drawdowns are weather, not climate change -- *as long as the underlying expectancy is genuinely positive,* which is the question the next subsection finally confronts.

There is a second, less obvious thing the equity curve lets you do: it lets you *watch for a regime change in your own results.* A positive-expectancy method does not just drift up; it drifts up at a roughly steady *slope*, because the slope is the expectancy. If the curve has been climbing at +0.35R per trade for two hundred trades and then flattens or turns down and *stays* down for fifty trades, that is information -- either the market regime that gave the edge has changed, or your execution has drifted, or you have unknowingly started trading a slightly different setup. The equity curve cannot tell you *which* of those happened, but it is the alarm that tells you to go look. This is precisely the alarm a pure mechanical system lacks the judgment to act on and a rule-assisted trader can: the curve flattens, the trader investigates, and they either confirm the regime has turned and stand down, or find the execution drift and correct it. Read this way, the equity curve is not a scoreboard you check for fun; it is a control chart for the only process you run.

### The standard error: how many trades before the edge is real?

Here is the hard truth that 30 trades cannot escape. An expectancy of +0.35R computed over 30 trades is *encouraging*, but it is not *proof*, because the estimate has a large margin of error. The relevant concept is the **standard error**: roughly, how much your *estimate* of the expectancy would jump around if you re-ran the same edge over a fresh batch of 30 trades. The standard error of an average shrinks with the square root of the number of trades -- to halve your uncertainty, you need *four times* as many trades.

For a method like ours, where individual trade results scatter by roughly 1.4R around the mean, the standard error of the expectancy estimate is about $1.4 / \sqrt{n}$. A 95% confidence band is roughly $\pm 1.96$ standard errors. Let us see what that does to our +0.35R estimate as the sample grows.

![A funnel chart showing the 95 percent confidence band around an estimated expectancy of plus zero point three five R shrinking as sample size grows: at thirty trades the red band of plus or minus zero point five R straddles zero, at one hundred trades the band is plus or minus zero point two seven R, at three hundred trades the amber band of plus or minus zero point one six R clears zero, and at one thousand trades the green band of plus or minus zero point zero nine R is tight around the true value.](/imgs/blogs/from-discretionary-to-systematic-6.png)

The diagram above is the funnel. At **n = 30**, the 95% band is about ±0.50R, so the true expectancy could plausibly be anywhere from -0.15R (a losing method) to +0.85R (a great one). The band straddles zero. Your +0.35R is real, or it is luck, and 30 trades genuinely cannot tell you which. At **n = 100**, the band tightens to about ±0.27R -- better, but its lower edge is still near zero. At **n = 300**, the band is about ±0.16R and finally clears zero: now the data says the edge is more likely real than not. At **n = 1,000**, the band is ±0.09R and you can trust the number. This is why every honest treatment of a track record asks first *how many trades*, and why a signal seller's screenshot of a 25-trade run should move you not at all. The math is the same one [the expectancy post derives](/blog/trading/technical-analysis/expectancy-why-win-rate-lies): small samples cannot distinguish a real edge from a coin.

#### Worked example: a 30-trade journal computed end to end

Let us do the full computation on the journal, so you can reproduce it on your own. The 30 R-multiple results, in order, are:

```
+2, -1, +2, -1, -1, +2, +2, -1, -0.5, +2,
-1, -1, -1, +2, +2, -0.5, -1, +2, +2, -1,
-1, +2, -0.5, +2, -1, -1, +2, +2, -1, -1
```

Step 1 -- **win rate.** Count the positive results: there are 13. Win rate = 13/30 = **43%**.

Step 2 -- **average win.** All 13 winners are +2R, so average win W = **+2.0R**.

Step 3 -- **average loss.** There are 17 losers: 14 at -1R and 3 at -0.5R. Total loss = 14(1) + 3(0.5) = 15.5R over 17 trades, so average loss L = 15.5/17 = **0.91R**.

Step 4 -- **expectancy.** $E = 0.43 \times 2.0 - 0.57 \times 0.91 = 0.87 - 0.52 = $ **+0.35R**. Over 30 trades that is 30 x 0.35 = +10.5R, which matches the equity curve's endpoint, and at \$100 per R it is +\$1,050.

Step 5 -- **is 30 trades enough?** Standard error $\approx 1.4 / \sqrt{30} = 1.4 / 5.48 = 0.256$R. The 95% band is $\pm 1.96 \times 0.256 = \pm 0.50$R, so the true expectancy is somewhere in roughly [-0.15R, +0.85R]. **The band includes zero, so 30 trades is not enough to prove the edge.** It is enough to say "promising, keep going and keep logging." You would need on the order of 300 trades before the band clears zero. The one-sentence intuition: *your own journal gives you the exact number, and the same journal tells you not to trust that number yet.*

### Separating setup-quality from execution-quality

There is a second, subtler measurement the journal makes possible, and it is the one that most accelerates improvement. Your realized edge has two ingredients: the quality of the *setup* (do the rules, executed perfectly, have positive expectancy?) and the quality of your *execution* (do you actually execute the rules?). These are different, and the journal can separate them.

To measure setup quality, compute the expectancy you *would* have gotten if every trade had been executed exactly to the rules -- entered at the trigger, stopped at the defined stop, exited at the target. To measure your realized edge, use the *actual* results. The gap between them is your **execution leak**: the R you are losing not because the method is bad but because you are not following it. We devote a full worked example to this below, and the psychology of why this gap exists is the subject of [the execution-gap post](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap). For now the point is structural: a single journal, kept honestly, lets you ask "is my *strategy* broken or is my *discipline* broken?" -- and those two diagnoses lead to completely different fixes.

#### Worked example: two traders take the same chart

Consider the same chart and the same defined setup handed to two traders. Trader R follows the written rules. Trader D trades the chart by feel. Run the identical setup ten times for each, on ten similar charts.

Trader R's ten results, because the trade is fixed, cluster tightly: six clean -1R losses and four clean +2R wins, for a total of 4(2) - 6(1) = +2R over ten trades, +0.2R per trade, with every individual result being exactly -1R or +2R.

Trader D, on the *same* ten charts, improvises. On chart 1 they enter early and get a worse price, turning a +2R winner into +1.3R. On chart 2 they widen the stop "just this once" and a -1R loss becomes -1.8R. On chart 3 they cut a winner at +0.6R out of fear. On chart 4 they skip the trade entirely after a loss and miss a +2R. On chart 5 they double size on a hunch and a -1R loss becomes -2R. Their ten results scatter from -2R to +1.6R with no two alike. The *total* might even come out similar to Trader R's, but the **variance** -- the trade-to-trade scatter -- is far larger, and that matters for two reasons. First, larger variance means deeper drawdowns, which means more chances to quit or to blow up before the edge compounds. Second, and more importantly, Trader D's scattered sample is *incoherent* -- it is ten different trades -- so Trader D cannot compute a trustworthy expectancy at all, while Trader R can.

![A before-and-after comparison of two traders on the same chart: on the red no-rules side the entry, stop, size, and exit are all decided by feel and results scatter from minus one point eight R to plus three R with no pattern, while on the green rules side the entry, stop, size, and exit are fixed and results cluster at a clean plus two R or minus one R that can be measured.](/imgs/blogs/from-discretionary-to-systematic-8.png)

The diagram above contrasts the two. The red column is the discretionary trader: every input -- entry, stop, size, exit -- is decided in the moment, so outcomes scatter with no pattern and cannot be measured. The green column is the rules trader: every input is fixed, so outcomes cluster at a clean +2R or -1R that *can* be averaged into a real expectancy. The one-sentence intuition: *rules do not just change your average; they collapse your variance, which both shrinks your drawdowns and makes your statistics trustworthy.*

#### Worked example: finding the leak

Now the most actionable computation in the whole post. A trader has been logging diligently for 100 trades. They tag each trade two ways: the *planned* R (what the rules would have produced) and the *actual* R (what they got). Averaging the planned column, the setup is worth **+0.5R per trade** -- a genuinely good edge. Averaging the actual column, their realized result is only **+0.2R per trade**. The setup is fine. So where did 0.3R per trade go?

They sort the trades by the gap between planned and actual and read the "mistake" field. The pattern jumps out: 0.18R of the leak comes from *cutting winners early* (exiting at +1.2R when the target was +2R, on about a third of their winning trades); 0.08R comes from *skipping valid setups* after a losing day (a missed +2R here and there); and 0.04R comes from *oversizing losers* on revenge trades. The setup is not the problem. The *execution* is leaking 60% of the edge.

![A waterfall bar chart showing a blue setup-quality bar at plus zero point five R, a red execution-leak bar removing zero point three R through early exits, skipped entries, and oversized losers, and a green realized-edge bar at plus zero point two R, with a note that sixty percent of the edge leaks in execution rather than in the setup.](/imgs/blogs/from-discretionary-to-systematic-7.png)

The diagram above is that leak drawn as a waterfall. The blue bar is the setup on paper (+0.5R). The red bar is the execution leak (-0.3R), itemized into early exits, skipped entries, and oversized losers. The green bar is what actually reached the account (+0.2R). The lesson is one that no amount of searching for a "better setup" would ever have surfaced: this trader does not need a new strategy; they need to *follow the one they have*. The fix is to stop cutting winners early -- which is a psychology problem, addressed by [the execution-gap post](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap), not by any chart pattern. Without a journal that separates planned from actual, this trader would have spent months hunting for a better entry trigger while bleeding 60% of a perfectly good edge out the exit. The one-sentence intuition: *the journal does not just tell you whether you have an edge; it tells you whether your problem is the strategy or yourself, and those need opposite fixes.*

## The discretionary-systematic spectrum

We have been building toward a defined, rules-based process, which might sound like an argument for full automation. It is not. The honest position is that there is a *spectrum*, and the right place to live on it depends on the trader, the strategy, and what genuinely can and cannot be reduced to a rule.

### The three stations

At the **fully discretionary** end, every decision is judgment in the moment. This is where most traders start and where, as we have seen, measurement is impossible. It is not that discretionary trading cannot work -- it demonstrably can in skilled hands -- but it cannot be *measured*, *taught*, *scaled*, or *trusted* without the operator, and it is brutally hard for a developing trader because there is no fixed thing to refine.

In the middle is **rule-assisted discretion**. The repeatable parts of the trade -- the setup definition, the entry trigger, the stop, the target, the size -- are written as hard rules. Judgment is reserved for a small number of genuinely judgment-shaped decisions: *is the broader context favorable enough to take signals at all today? Is this news environment too dangerous? Does this particular instance, while it meets the rules, have a reason to skip that the rules cannot capture?* The statistics are coherent because the trade is fixed; the human adds value on the margin where rules are weak. This is where the large majority of serious traders should live, and where the worked examples in this post quietly assumed we were.

At the **fully systematic** end, every decision is coded, including the context filter. There is no human in the loop on a trade-by-trade basis. This is fully measurable -- you can backtest it exactly and your live results should match your backtest, which is itself a powerful diagnostic. But it is *brittle*: a system that was tuned on one market regime can keep mechanically taking trades long after the regime that gave it an edge has gone, and it has no judgment to notice. A discretionary or rule-assisted trader feels the regime change in their results and pulls back; a pure system does not, until the drawdown forces a human to intervene anyway.

### Where most traders should live, and what to automate

The practical guidance is: **automate the parts that are mechanical and error-prone, keep judgment for the parts that genuinely need a human, and write everything down either way.** Position sizing should be a rule -- there is no judgment that improves on "risk a fixed fraction," and doing it by feel is how accounts blow up. Stop placement should be a rule, because moving stops by feel is the single most common execution leak. The entry trigger should be a rule, because "it looked ready" is unmeasurable. What can reasonably stay discretionary is the *context filter* -- the go/no-go on whether to be active at all, and the occasional well-justified skip of a setup that technically qualifies but sits in front of a major news event or a structurally broken market.

The key insight is that *the parts you keep discretionary should be the parts you have decided not to measure.* If a decision is judgment-based, you accept that you cannot compute its expectancy, so you keep it small and infrequent, and you make sure the *measurable* core of your process -- the part your statistics describe -- is the rule-based part. That way your numbers describe the bulk of your edge honestly, and the discretionary layer is a small, acknowledged overlay rather than a black hole that swallows your ability to measure anything.

### The cost of premature automation

There is a strong temptation, once you have rules, to code them up and let the computer trade them. Resist doing this too early. Premature automation has real costs. First, you automate a process you have not yet verified has an edge -- you need the journal to confirm positive expectancy *first*, over enough trades, before it is worth automating. Second, automation removes the very feedback loop -- *watching your own execution* -- that the journal exists to provide; you stop seeing your leaks because the machine has no leaks, but you also stop learning. Third, a coded system needs robust handling of the messy edge cases (gaps, halts, partial fills, data errors) that a human handles instinctively, and getting that wrong turns a +0.35R edge into a -2R disaster on the one chaotic day that matters. The right sequence is: write rules, journal them by hand, confirm the edge over hundreds of trades, *and only then* consider automating the parts that are truly mechanical -- keeping a human on the context filter unless you have a specific, tested reason to remove them.

## Common misconceptions

> [!warning]
> The beliefs below feel reasonable and are common among developing traders. Each one quietly undermines the ability to measure or improve, which is the whole point of writing rules in the first place.

**"Systematic means no judgment."** This is the most common misreading. Going systematic does not mean becoming a robot or distrusting your own perception. It means *fixing the repeatable parts* so your statistics are coherent, and being deliberate about the *few* places where judgment genuinely adds value. The best traders are not the ones with zero judgment; they are the ones who know exactly which decisions are rule-based and which are judgment-based, and who never let the two blur. Rule-assisted discretion -- the middle of the spectrum -- is a sophisticated position, not a compromise.

**"I'll remember my trades without a journal."** You will not, and worse, you will *misremember* them in a flattering direction. Human memory reconstructs rather than records: it smooths losing trades into "I knew that was risky," inflates the size of winners, forgets the trades you skipped, and quietly rewrites your stop as having been "about there" when it was actually moved. The execution leak in our worked example is invisible to memory by construction, because the trades you remember best are the ones with a clean story, and leaks are precisely the messy, story-less in-between. Only a written, timestamped, screenshot-backed record is immune to this rewrite.

**"Thirty trades proves my system."** It does not, and this is not a matter of opinion but of arithmetic. As the funnel showed, the 95% confidence band on a +0.35R edge over 30 trades is about ±0.50R -- wide enough that the true expectancy could be negative. Thirty trades is enough to be *encouraged* and to *keep logging*; it is not enough to be *confident*. Roughly 300 trades is where the band on a moderate edge clears zero. Acting as if 30 trades is proof leads to two symmetric errors: abandoning a real edge after an unlucky 30, and trusting a fake edge after a lucky 30.

**"I should automate everything immediately."** Automation is the *last* step, not the first, and only for the parts that are genuinely mechanical. Before you automate you must (a) confirm positive expectancy over hundreds of journaled trades, (b) have handled every messy edge case the live market throws, and (c) accept that you are giving up the execution feedback loop that teaches you to be a better trader. Automating an unverified strategy just lets you lose money faster and more consistently, and automating away your own learning early in your development is a serious mistake even when the rules are good.

**"A bigger win rate means a better system."** Win rate is the split, not the edge. A method can push its win rate up by taking tiny profits and giving losers room, and destroy its expectancy in the process. The journal computes the number that actually matters -- expectancy -- and a disciplined trader tracks that, not the win rate, as the headline figure. [The expectancy post](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) is the full treatment of why this is so.

## How it shows up in real markets

The mechanics above are not academic. The move from feel to rules, and the journal that measures it, is visible across the spectrum of real trading operations. The scenarios below are representative composites of widely-documented patterns rather than claims about specific named firms; treat them as illustrations of the mechanism, and note that the specifics of any one desk are private and change over time. As-of this writing in mid-2026, the *structure* of how serious trading operations enforce documented process has been stable for decades, even as the specific tools change.

### The discretionary trader who couldn't scale until they wrote rules

A common arc: a trader has a genuinely good discretionary feel and is profitable, but cannot grow. They cannot trade larger size because they do not trust the process enough to risk more on it. They cannot trade more markets because their feel is tied to the few charts they watch all day. And they cannot take a vacation, because the edge lives entirely in their head and stops when they step away. The breakthrough is always the same: they sit down and *write the rules* for the trades they were already taking by feel. The act of writing forces them to discover that 80% of their edge came from three repeatable setups they could now define precisely, and 20% came from genuine judgment they kept as an overlay. Once the 80% was written, it could be journaled, measured, trusted, sized up, and -- crucially -- explained to a risk manager or a partner. The feel did not get worse; it got *legible*, and legibility is what scales.

### A journal revealing a hidden execution leak

The execution-leak worked example is drawn from one of the most common real discoveries traders make when they first journal honestly. A trader convinced their *strategy* is failing -- they keep changing entry triggers, hunting for a better signal -- finally logs planned-versus-actual for a few hundred trades and finds the setup was positive all along; the leak was entirely in the exits. The realization reframes everything: the months spent searching for a better entry were wasted, because the entry was never the problem. This is so common that experienced traders treat "I need a better setup" with suspicion until the journal has ruled out an execution leak first. The diagnostic order matters: measure your execution before you blame your strategy.

### Prop firms requiring documented rules

Proprietary trading firms -- whether the modern remote "evaluation" firms that fund traders who pass a test, or traditional desks that allocate capital to in-house traders -- almost universally require a *documented, rule-based process* before they will risk real capital on a trader. The reason is exactly the one in this post: an undocumented discretionary process cannot be risk-managed, because the firm cannot bound what the trader will do. A written rule set with defined risk per trade, a maximum drawdown, and a journal that proves the trader follows it is what lets a firm allocate capital at all. The firm is, in effect, forcing the trader to do what this post recommends, because the firm's own survival depends on the process being measurable. As-of 2026, the specific rules and risk limits vary widely across firms and change frequently, but the requirement of *documented, journaled process* is essentially universal.

### The rule-assisted discretionary desk

Many of the most durable trading operations sit squarely in the middle of the spectrum, by design. The setups are defined and the risk is rule-bound -- position sizing, stops, and maximum exposure are non-negotiable rules enforced by the risk system, not the trader. But the *go/no-go* on whether conditions favor the strategy today, and the occasional override to stand down before a major scheduled event, stay with experienced humans. This is rule-assisted discretion at an institutional scale: the measurable core is rule-based so the desk's edge can be tracked and risk-managed, and the human judgment is confined to the few decisions where a person genuinely outperforms a fixed rule -- principally the recognition that the market regime has shifted and the edge has gone quiet. The desk keeps the journal because the journal is what tells them, statistically, whether the strategy is still working or whether the regime has turned, and that is a question no amount of in-the-moment feel can answer honestly.

## When this matters to you and further reading

This post is the hinge of the whole "Technical Analysis, Honestly" series, because it is where everything else becomes *measurable*. All the pattern-reading and indicator knowledge in the world is just opinion until it is written as a rule, journaled as a trade, and computed into an expectancy. The honest core, one more time: you cannot improve what you do not measure; a discretionary process cannot be measured because it is a different bet every trade; the fix is to write the repeatable parts as rules so the sample becomes coherent; the journal is the only honest record of your live edge; and the statistics that come out the other side -- win rate, average R, expectancy, the equity curve, and the standard error -- tell you not just whether you have an edge but whether your problem is your strategy or your discipline, which need opposite fixes.

Where this touches your trading directly: the next time you have a green month, do not celebrate the number -- ask whether you can *describe what you did in a sentence a second reader could execute.* If you cannot, that is the work. Write the rules for the trades you are already taking by feel. Start a journal with the eight columns from the figure, including the two everyone skips -- the reason and the mistake. Log thirty trades and compute your expectancy, then remind yourself that thirty is encouraging, not proof, and keep going to three hundred. And separate, ruthlessly, the question of whether your *setup* has an edge from whether your *execution* is following it, because the most expensive mistake in this entire process is hunting for a better strategy while bleeding a perfectly good one out the exit.

For the next steps in the series: the statistical backbone of everything here -- expectancy, R-multiples, breakeven win rate, variance, and risk of ruin -- is developed in full in [Expectancy: why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies). The concrete construction of a single rule-based setup, end to end through a winner, a loser, and a skip, is in [Building one high-probability setup](/blog/trading/technical-analysis/building-one-high-probability-setup). The reason your *backtest* and your *journal* can disagree, and how to test rules on historical data without fooling yourself, is the subject of [Backtesting without fooling yourself](/blog/trading/technical-analysis/backtesting-without-fooling-yourself). And the psychology of *why* the execution leak exists -- why you cut winners early, widen stops, and skip valid trades even when you know the rules -- is the heart of [Trading psychology and the execution gap](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap). Rules make your edge measurable; psychology decides whether you actually follow them; and the journal is the bridge that lets you see both at once.

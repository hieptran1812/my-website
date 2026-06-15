---
title: "Your Trading Plan: Assembling the Whole Series into One Page"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "The capstone of the series: how to turn everything about technical analysis into one written artifact -- a trading plan and a one-page pre-trade checklist. A plan is not a prediction engine; it is a discipline engine that fixes what you trade, your setups, your risk, the gate every trade must pass, and the journal that measures your real edge."
tags:
  [
    "trading-plan",
    "pre-trade-checklist",
    "risk-management",
    "position-sizing",
    "expectancy",
    "trading-discipline",
    "trading-journal",
    "trading-psychology",
    "process-goals",
    "technical-analysis",
    "systematic-trading",
    "trading-edge",
  ]
category: "trading"
subcategory: "Technical Analysis"
author: "Hiep Tran"
featured: true
readTime: 46
---

> [!important]
> **TL;DR** -- a trading plan is the single artifact that turns the whole series into action. It is not a prediction engine; it is a *discipline engine*. It states what you trade and when, defines your setups as rules, fixes your risk, gives you a checklist that must pass before every trade, and commits you to a journal and a review cadence so your edge can be measured and improved.
>
> - The thesis the entire series has built toward: technical analysis is a way of placing **probabilistic, risk-managed, measured** bets -- never a way of predicting the future. The plan is how that thesis becomes a daily practice instead of a slogan.
> - A complete plan fits on **one page** as **six parts**: (1) what and when, (2) the setup as rules, (3) risk and position sizing, (4) the pre-trade checklist, (5) the journal and review, (6) the psychology contract. Each part is a written rule and a fixed number you commit to *before* the trade.
> - The **pre-trade checklist** is the signature artifact: eight yes/no gates. Any single NO means no trade. It removes in-the-moment discretion from the entry, which is exactly where emotion does its damage.
> - The edge is **small, fragile, and statistical** -- a good plan might be worth **+0.30R per trade**, a fraction of the 1R you risk each time. The plan is what lets that thin math express itself across a large sample while keeping you alive through the variance.
> - A plan's job is to **survive** the drawdowns (a realistic path to +30R over 100 trades can dip to **-8R** along the way) and to **measure** the gap between what the plan can pay and what you actually keep -- so a behavioral leak gets fixed by behavior, not by tearing up a working strategy.

Imagine two traders sitting at the same desk, watching the same chart, trading the same market. One of them, before the session, opened a one-page document and read it: *I trade these four symbols, in this session, in this regime; my setup is this; I risk this much per trade and I stop for the day if I lose this much; before I click buy I run these eight checks; I log every trade and review on Sunday.* The other opened the chart and waited to "see what the market gives me." Over a hundred trades, with identical analysis and identical signals, the first trader will almost certainly end the quarter ahead of the second -- not because they predicted better, but because the document did the one thing the human brain refuses to do under pressure: it pre-decided. This post is about how to write that document.

This is the last post in the series, and it is where everything we have built becomes a single artifact. Across thirty-three posts we have argued, from many angles, one honest thesis: technical analysis is not a forecasting tool. It is a way of placing bets that are *probabilistic* (you trade odds, never certainties), *risk-managed* (you decide what you can lose before you can lose it), and *measured* (you prove your edge on a sample instead of assuming it). A trading plan is the place where those three commitments stop being ideas and become a routine you can actually run. It is not a crystal ball. It is a discipline engine.

![A trading plan on one page shown as a matrix of six parts -- what and when, the setup, risk and sizing, the pre-trade checklist, journal and review, the psychology contract -- with each part reduced to a written rule and a concrete number.](/imgs/blogs/building-your-trading-plan-capstone-1.png)

The matrix above is the whole post in one picture. Each row is one part of the plan; each part is reduced from a vague intention ("manage risk") into a written rule ("risk 0.5% per trade; stop trading at -2% on the day") and a concrete number you commit to in advance. By the end of this post you will be able to fill in every cell of that grid for your own trading. We will build it part by part, ground each part in a worked example with real dollar figures, and then hand you the one-page checklist that ties it together.

A note before we start, the same one we have made all series: this is educational, not advice. It explains the mechanics of building a disciplined process, so you can recognize what a plan is and why it works. It is not a recommendation to trade anything, and certainly not to trade more. Every method that can make money can lose it, and the entire point of a plan is to decide -- calmly, in advance -- how much.

## Foundations: a plan is a discipline engine, not a crystal ball

Before we build the plan, we need to be precise about what it is *for*. The instinct most beginners bring to trading is that the goal is to **predict** -- to look at the chart and figure out what the market will do next. A plan, in that frame, would be a list of predictions: "the market will go up, so I will buy." That frame is wrong, and getting it wrong is the single most expensive misunderstanding in trading. We have to clear it out of the way first, because everything in the plan follows from replacing it.

### What technical analysis actually is

Let us restate the *bản chất* -- the essential nature -- of technical analysis, because the plan is built on top of it. A chart is a record of every transaction: price over time, and how much traded at each price. Technical analysis is the practice of reading that record for **repeatable, statistically-favorable situations** -- places where, historically, the next move has been a little more likely to go one way than the other. The key words are "a little more likely." You are never reading a certainty off the chart. You are reading *odds*. (The full honest case for and against this is in [what technical analysis really is](/blog/trading/technical-analysis/what-technical-analysis-really-is); here we just need the conclusion: the chart is evidence, not a forecast.)

This is why the plan cannot be a list of predictions. You do not know what the market will do next, and neither does anyone else. What you can know is that *a certain situation, taken many times, has a positive average outcome*. A plan operationalizes that: it says "when this situation appears, I take this action, at this risk, every time" -- and then it lets the average play out over a large number of trades. The plan does not predict the next trade. It harvests the average across the next hundred.

### The edge, and why it must be measured

The word that makes this concrete is **edge**. An edge is a positive *expectancy*: the average dollar (or R) outcome of one trade, counting wins and losses together, is greater than zero. We measure trades in **R-multiples** to strip out position size: **1R is the amount you risk on a trade** -- the distance from your entry to your protective exit (your *stop-loss*, a pre-committed price at which you bail to cap the loss), times the number of shares. A full stop-out is **-1R**; a trade that makes twice your risk is **+2R**. Now every trade speaks the same units regardless of how big the position was.

Expectancy, in R-multiples, is:

$$E[R] = p \cdot W - (1-p) \cdot 1$$

where $p$ is the win rate (the fraction of trades that win), $W$ is the average size of a winner in R, and $1-p$ is the loss rate (we take losers at a clean -1R). If $E[R]$ is positive, the system makes money on average; if it is negative, it bleeds no matter how often it wins. That single sign is what "having an edge" means. (The full derivation, the breakeven win rate $\frac{1}{1+R}$, and why a 40% system can crush a 70% one are in [why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies). The plan assumes that math and asks the next question: how do you actually *collect* it?)

Here is the property that makes a plan necessary. Expectancy is an *average over many trades*. A system with $E[R] = +0.30$ does not make +0.30R on the next trade -- the next trade is a clean -1R loss, or a +4R win, or whatever the individual outcome happens to be. The +0.30R only appears when you average over a large number of trades. Over 100 trades the system is worth about +30R; over 1,000 trades, about +300R. The edge is a wage paid per trade, and the number of trades is your hours -- but each individual hour can pay anything, and many of them pay a loss. **The edge is real, but it is invisible over any short stretch.** To collect it you have to keep taking trades through long, painful stretches where variance is winning and the edge is nowhere to be seen.

That is the whole reason a plan exists. The math rewards a machine that takes a thousand identical trades without caring about any of them. You are a human being who feels every one. The plan is the bridge between the two: it pre-commits you to behave like the machine, so the math the machine would have earned can actually land in your account.

### Why the edge is thin, fragile, and statistical

One more foundational fact, and it is the one that disciplines everything else. A realistic, honestly-measured retail edge is **small**. A genuinely good discretionary technical system might earn +0.20R to +0.40R per trade after costs. That is a fraction of the 1R you risk on every single trade. The implication is brutal and clarifying at once: because the edge is thin, *small deviations destroy it*. Skip a few winners, widen a few stops, oversize after a hot streak, and you have given back more than the edge was ever going to pay. **A thin edge is exactly as fragile as it is valuable.** (This is the core of [trading psychology and the execution gap](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap): the gap between the edge you have and the edge you realize is usually bigger than the edge itself.)

And the edge is *statistical* -- it lives in a sample, not in any single trade, and it is submerged in **variance**, the random scatter of outcomes around the average. Even a fixed positive edge produces wildly different short-run results purely by chance. A 45%-win system will routinely throw five or six losers in a row. The human mind, a relentless pattern-finder, reads that streak as "the edge is broken" and overrides the plan -- which is precisely how the edge gets lost. The plan's job is to keep you taking the trades through the noise, sized small enough that no streak can ruin you, until the sample is large enough for the average to show up.

So: the edge is probabilistic (you trade odds), it must be measured (you prove it on a sample), it is thin (so it is fragile), and it is statistical (so it only appears over a large number of trades). A trading plan is the single artifact that honors all four of those facts at once. It is not a crystal ball. It is the discipline engine that lets a small, fragile, statistical edge express itself across a large sample while keeping you alive through the variance. The rest of this post builds it, part by part.

![A graph showing the six parts of a trading plan -- what and when, the setup, risk and size, the pre-trade checklist, journal and review, and the psychology contract -- all converging on one pre-trade decision node that branches to either go and place the trade or skip and wait.](/imgs/blogs/building-your-trading-plan-capstone-2.png)

The graph above is the mechanism the plan implements. Six components feed into a single pre-trade decision, and that decision has exactly two outcomes: place the trade at fixed risk, or skip it and wait for the next valid signal. Notice what this does to the moment of the trade. By the time you are looking at a live candle, every component has already been decided. The only thing left is to check whether the rules are satisfied and to act -- not to predict, not to deliberate, not to feel your way in. The plan front-loads all the thinking into a calm moment and leaves the heated moment with nothing to do but execute.

## Part 1 -- what you trade and when

The first part of the plan answers the most basic question, the one most traders never write down: *what, exactly, are you trading, and when?* Leaving this vague is the first leak. A trader who "trades whatever looks good" is, by definition, trading an incoherent sample -- a few stocks here, a crypto pair there, a forex scalp at 3 a.m. -- and an incoherent sample cannot be measured. Part 1 fixes the universe so the rest of the plan has something to operate on.

### Markets and instruments

Pick a small, specific set of instruments and write them down. "US large-cap equities and the S&P 500 ETF" is a universe. "Stocks" is not. The reason to keep it small is that every market has its own *personality* -- its typical volatility, its session rhythm, how cleanly it respects technical levels, its costs. A setup that works on the slow, mean-reverting behavior of a blue-chip stock can die on a trending, momentum-driven crypto pair. By restricting to a handful of instruments you actually understand, you make your sample coherent and your edge measurable.

The honest tradeoff: a narrow universe means fewer signals, which means it takes longer to build the large sample the edge needs. That is fine. A coherent sample of 100 trades on four instruments teaches you something; an incoherent sample of 100 trades across forty instruments teaches you almost nothing.

### Timeframes and the regime

Two more decisions live here. The first is **timeframe**: which chart you read for the *bias* (the broad direction you are willing to trade) and which you read for the *entry* (the precise moment you act). The series argues hard for trading the lower timeframe in the direction of the higher one -- you set your bias on, say, the daily chart, and you only take entries on the 15-minute chart that agree with it. This is [multi-timeframe analysis](/blog/trading/technical-analysis/multi-timeframe-analysis): the higher timeframe tells you which way the current is flowing, and you only paddle with it.

The second is the **regime** -- the kind of market behavior you are trading. Markets alternate between *trending* (price moves persistently in one direction, so buying strength and selling weakness pays) and *mean-reverting* or *ranging* (price oscillates around a level, so fading extremes pays). The same setup wins in one regime and dies in the other; this is the central lesson of [trend-following versus mean-reversion](/blog/trading/technical-analysis/trend-following-vs-mean-reversion). Your plan must say which regime your setup is built for, and -- crucially -- include a simple, objective test for whether the current market is in that regime, so you can sit out when it is not.

#### Worked example: a filled-out Part 1 for a specific trader

Let us make this concrete with a specific trader profile we will carry through the whole post. Call her a part-time swing trader with a \$50,000 account who can watch the market for two hours around the New York open. Her Part 1 reads:

- **Instruments:** SPY (the S&P 500 ETF) plus three large-cap US stocks she follows closely. Four symbols total.
- **Timeframes:** Bias on the **daily** chart; entries on the **15-minute** chart.
- **Session:** The first two hours of the New York session only (9:30--11:30 a.m. ET). She does not trade the close, the overnight, or any other session.
- **Regime:** She trades **pullbacks in an uptrend** -- a trend-following setup. Her objective regime test: price is above its rising 50-day moving average on the daily chart. If a symbol is below a falling 50-day average, she does not look for longs in it that week.

Notice that none of this is a prediction. It is a fence. It says, in advance and in writing, *these are the only situations I will even consider.* Everything outside the fence -- a hot tip on a small-cap, a crypto pump at midnight, a short in a downtrend she has not studied -- is automatically out, not because she judged it bad in the moment, but because she pre-decided it was off the menu. The one-sentence intuition: **the first job of a plan is to shrink the universe of things you can do to the small set you have actually measured an edge on.**

## Part 2 -- your setups as rules

Part 2 is the heart of the plan: the **setup**, fully specified as rules. A setup is not a feeling that "this looks good." It is a single object with six pre-defined ingredients, and the whole object is built before any money is at risk. We dedicated a full post to constructing one end to end -- [building one high-probability setup](/blog/trading/technical-analysis/building-one-high-probability-setup) -- and the plan simply pins that object down in writing. The six ingredients are bias, level, confluence, trigger, invalidation, and target.

### The six ingredients

**Bias** is the direction you are willing to trade, set from the higher timeframe (Part 1's regime). In our trader's case: long only, in a daily uptrend.

**Level** is the specific price or zone where you are interested -- a support or demand zone, a prior swing low, a moving average. You mark it in advance. Our trader buys pullbacks into a *demand zone*, a price area where buyers have previously stepped in with force; say the \$100--\$101 area on one of her stocks.

**Confluence** is the stack of independent reasons the level matters. One reason is a guess; several independent reasons stacked at the same price is an edge. The math of why is precise and worth internalizing: if each factor independently raises the probability the level holds, stacking *independent* factors multiplies the odds in your favor. This is the argument of [confluence: stacking independent factors](/blog/trading/technical-analysis/confluence-stacking-independent-factors). The word "independent" is load-bearing -- five indicators that are all just smoothed price are *one* factor wearing five hats, not five factors. Our trader requires at least four genuinely independent factors at the level: the daily uptrend, the demand zone, a round number (\$100), and a volume signature.

**Trigger** is the precise event on the entry timeframe that tells you to act -- a specific candlestick pattern, a break of a micro-level. Without a trigger, "price is at my level" is an invitation to catch a falling knife. Our trader's trigger is a *bullish engulfing* candle that closes on the 15-minute chart inside the zone.

**Invalidation** is the price at which the idea is *wrong* -- where you place your stop-loss. It is defined by structure, not by how much you are willing to lose: it sits at the price beyond which the setup no longer makes sense (here, below the demand zone, at \$98.50). The dollar risk follows from the stop, not the other way around.

**Target** is where you intend to take profit -- typically the next structural level (here, resistance at \$111). The distance from entry to target, divided by the distance from entry to stop, is your **reward-to-risk ratio**. The series treats stops and targets as a single object designed together, because the ratio between them is what makes the expectancy work; see [risk, reward, and expectancy in practice](/blog/trading/technical-analysis/risk-reward-and-expectancy-in-practice).

### Why the setup must be written as rules

The reason to write all six down -- in literal sentences -- is that a written rule can be checked, and a checked rule can be measured. The moment a setup lives only in your head, it changes shape from trade to trade: a little looser when you are eager, a little stricter after a loss. That drift makes your sample incoherent, which makes your expectancy unmeasurable, which means you are flying blind. Writing the setup as rules is the first step in moving [from discretionary to systematic](/blog/trading/technical-analysis/from-discretionary-to-systematic) -- not necessarily all the way to a fully-automated system, but far enough that the repeatable parts are fixed and only genuine context-judgment is left to you.

#### Worked example: the setup as six written rules, with the numbers

Here is our trader's setup, written exactly as it appears in her plan, with the numbers from the \$100--\$101 demand-zone trade:

1. **Bias:** Long only, when price is above the rising 50-day average on the daily chart.
2. **Level:** A pre-marked demand zone; this trade, \$100--\$101.
3. **Confluence:** Require at least four independent factors. This trade has four: daily uptrend, demand zone, the round number \$100, and a volume cluster on the original rally from the zone.
4. **Trigger:** A 15-minute bullish engulfing candle that closes inside the zone. Entry on the close, here \$101.
5. **Invalidation (stop):** \$98.50, below the zone and the prior swing low. Risk per share = \$101 - \$98.50 = **\$2.50**. That is 1R.
6. **Target:** \$111, the next daily resistance. Reward per share = \$111 - \$101 = \$10. Reward-to-risk = \$10 / \$2.50 = **4:1**.

Every number here was decided *before* the entry. When the engulfing candle closes at \$101, our trader does not deliberate -- she has already defined the entry, the stop, and the target. The one-sentence intuition: **a setup written as six rules turns "this looks good" into a checkable, repeatable, measurable object, which is the only kind of object an edge can live in.**

## Part 3 -- risk and position sizing

Part 3 is where the plan keeps you alive. Parts 1 and 2 define *what* you trade; Part 3 defines *how much*, and it is the part that separates traders who survive from traders who blow up. The central, non-negotiable idea: **risk is decided before the trade, as a fixed fraction of the account, and it does not change because you feel confident.**

### Per-trade risk

The foundation of position sizing is **fractional risk**: you risk the same small percentage of your account on every trade. Most professional retail risk frameworks land between 0.25% and 1% per trade. Our trader uses **0.5%**. On her \$50,000 account, 0.5% is **\$250** -- that is her 1R, in dollars, on every trade.

Now position size falls right out of the stop. The rule is:

$$\text{position size} = \frac{\text{dollar risk}}{\text{stop distance per share}}$$

For the demand-zone trade: dollar risk is \$250, stop distance is \$2.50 per share, so she buys \$250 / \$2.50 = **100 shares**. Notice the elegance: the position size *automatically* adjusts to the stop. A trade with a tight \$1 stop would let her buy 250 shares; a trade with a wide \$5 stop would limit her to 50. Either way, if the stop is hit she loses exactly \$250 -- her fixed 1R. The risk is constant even though the share count and the dollar exposure are not. (Why this fixed-fraction approach, and why the full Kelly bet is too aggressive for real trading, is worked out in [position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion).)

### Loss caps and max open risk

Per-trade risk is not enough on its own, because trades cluster. A bad day can be five losers in a row, and five times 0.5% is 2.5% -- recoverable, but if you keep pressing it becomes 5%, 8%, a real hole. So the plan adds **circuit breakers**:

- **Max daily loss:** stop trading for the day at **-2%** (four full stop-outs). Close the platform. The decision is pre-made so you cannot argue with it at -1.8%.
- **Max weekly loss:** stop trading for the week at **-5%**. A week this bad is a signal to step back and review, not to trade your way out.
- **Max open risk:** the total risk across all open positions at once is capped (say **1.5%**, three concurrent trades). This stops a "diversified" book from secretly being one big correlated bet that all stops out together.

These caps feel restrictive, and that is the point. Their entire job is to make sure that a normal bad streak -- which *will* happen, by variance alone -- never becomes an account-ending event. They are the difference between a -8R drawdown you trade through and a -40R drawdown you do not come back from.

#### Worked example: sizing three trades and hitting the daily cap

Walk through a realistic morning for our trader, account \$50,000, 0.5% (\$250) per trade.

- **Trade A:** demand zone, entry \$101, stop \$98.50 (\$2.50 risk). Size = \$250 / \$2.50 = 100 shares. It stops out: **-\$250 (-1R)**.
- **Trade B:** a tighter setup on SPY, entry \$500, stop \$498 (\$2.00 risk). Size = \$250 / \$2.00 = 125 shares. It stops out: **-\$250 (-1R)**.
- **Trade C:** entry \$80, stop \$76 (\$4.00 risk -- a wider, more volatile name). Size = \$250 / \$4.00 = 62 shares (rounded down). It stops out: **-\$250 (-1R)**.

Three losers, each a clean -1R, total **-\$750**, which is -1.5% of the account. She is now one loss away from her -2% daily cap. A fourth setup appears, and it looks great. The plan's answer is unambiguous: she has 0.5% of room left, so she may take *one* more trade at full size, and if it loses she is done for the day -- platform closed, no exceptions. Notice that the three different stop distances produced three different share counts (100, 125, 62) but identical dollar losses (\$250 each). The one-sentence intuition: **fixed-fraction sizing makes every loss the same size in dollars no matter how the trade is shaped, and loss caps make sure a normal cluster of losses can never become a catastrophe.**

## Part 4 -- the pre-trade checklist

We now reach the signature artifact of the entire plan: the **pre-trade checklist**. Parts 1 through 3 defined the rules; the checklist is the *gate* that enforces them at the one moment that matters most -- the instant before you click buy. It is a one-page, if-this-then-that list of yes/no questions, and the rule is absolute: **pass every gate, or skip the trade.** No partial credit, no "close enough," no overriding it because the chart "feels" right.

![A pipeline of eight pre-trade checklist gates -- regime, level, confluence, trigger, risk, reward, limits, and trader state -- each a single yes or no question, flowing into a final node where all eight yes results means place the trade.](/imgs/blogs/building-your-trading-plan-capstone-3.png)

The pipeline above is the checklist itself. Read it left to right: each box is one question with a yes/no answer and no opinion in it. A NO anywhere ends the trade idea on the spot. This is the most important figure in the post, so let us be exact about why a *checklist* -- a tool from surgery and aviation, not from finance -- is the right shape for this job.

### Why a checklist, and why yes/no

The power of a checklist is that it converts judgment into verification. In the moment of a trade, your judgment is compromised: you are excited, or afraid, or impatient, and those states quietly relax your standards. A checklist does not ask you to judge; it asks you to *verify* a condition you defined earlier, when you were calm. "Is price at a pre-marked demand zone?" has a yes/no answer that emotion cannot bend. By replacing a hundred small in-the-moment judgments with eight pre-defined verifications, the checklist moves the actual decision-making back to the calm moment where it belongs.

Each question must be genuinely binary. "Does the setup look strong?" is not a checklist item -- it is a feeling in disguise, and it will pass whenever you want it to. "Are there at least four independent confluence factors at this level?" is a checklist item, because you can count them and the count does not care how you feel. Writing the gates as countable, yes/no conditions is the entire discipline.

### The eight gates

Here is our trader's full checklist, the version that fits on one page taped to her monitor:

1. **Regime:** Is the daily trend up (above a rising 50-day average) for this long? *NO -> skip.*
2. **Level:** Is price at a pre-marked demand zone I identified before this session? *NO -> skip.*
3. **Confluence:** Are there at least four independent factors stacked at this level? *NO -> skip.*
4. **Trigger:** Has my specific 15-minute trigger (a bullish engulfing close) actually fired? *NO -> wait, do not anticipate.*
5. **Risk:** Is the stop placed at structural invalidation, and does the position size make the risk exactly 0.5%? *NO -> fix it or skip.*
6. **Reward:** Is the reward-to-risk to the nearest real target at least 2:1? *NO -> skip.*
7. **Limits:** Am I below my daily and weekly loss caps, and below max open risk? *NO -> stop trading.*
8. **State:** Am I calm -- not on tilt, not revenge-trading, not forcing a trade out of boredom? *NO -> walk away.*

Gates 1--4 verify the setup is present. Gates 5--6 verify the risk and reward are right. Gate 7 verifies you are allowed to trade at all today. Gate 8 verifies *you* are fit to trade -- the only subjective gate, and the one most traders refuse to write down. We will return to it in Part 6.

#### Worked example: the checklist applied to a real candidate (pass) and a tempting one (skip)

This is the discipline in action. Two trade ideas appear; the checklist decides both.

![A before-and-after comparison: on the left a candidate that passes all eight checklist gates and is taken at half a percent risk, on the right a tempting candidate that fails the regime and reward gates and is skipped despite looking attractive.](/imgs/blogs/building-your-trading-plan-capstone-4.png)

The figure above shows the two side by side. **Candidate 1 (the pass):** price has pulled back into the pre-marked \$100--\$101 demand zone in a clear daily uptrend. She counts the confluence: uptrend, zone, round number, volume cluster -- four, gate cleared. The 15-minute engulfing candle closes at \$101. Stop at \$98.50 makes risk \$2.50/share; at 100 shares that is \$250, exactly 0.5%. The target at \$111 is \$10 away, a 4:1 reward-to-risk, well above the 2:1 floor. She is below her loss caps and she is calm. **All eight gates: YES.** She places the trade, sets the stop and target, and -- this is the part people skip -- leaves it alone.

**Candidate 2 (the skip):** a different stock has just printed a huge green 15-minute candle. It is exciting; it feels like easy money is walking away. She runs the gates. Gate 1, regime: the daily trend on this name is *down*, below a falling 50-day average. That is already a NO for a long. But suppose she is tempted and keeps going. Gate 3, confluence: she can find only two reasons -- "a big candle" and "a hunch" -- not four independent factors. Another NO. Gate 6, reward: the nearest real resistance is only \$3 away against a \$2.50 stop, a 1.2:1 ratio, below her floor. A third NO. The trade fails three separate gates. The plan's verdict is to skip it, and skip it she does -- not because she predicted it would lose (it might well have won; that is the point), but because *it was never a trade her measured edge applies to.* The one-sentence intuition: **the checklist lets you skip a tempting trade with a clear conscience, because the reason to skip is a failed rule, not a forecast you have to defend.**

## Part 5 -- the journal and the review

A plan that you follow but never measure is half a plan. Part 5 closes the loop: the **journal** records every trade so it becomes data, and the **review** turns that data into your real, realized expectancy -- the number that tells you whether the plan is working and where it is leaking. This is the part that converts trading from a series of disconnected gambles into a measurable, improvable process.

![A graph of the review loop: take the trade by the plan, log it in R, build a sample of at least thirty trades, measure realized expectancy, diagnose the gap as a plan problem or an execution leak, then improve one rule at month-end.](/imgs/blogs/building-your-trading-plan-capstone-6.png)

The graph above is the loop the journal and review run. The plan turns trades into a measurable sample; the review computes realized expectancy and splits the gap into a plan problem or an execution leak; only then, at month-end, does one rule change, and the loop repeats. Each pass through it makes your edge a little more measured and your plan a little more honest.

### What to record

The journal must record enough to reconstruct the trade and to grade your process. For every trade, log:

- **The setup:** which symbol, the level, the confluence factors present.
- **Entry, stop, target, exit** in price, and the **outcome in R** (the only outcome unit that lets you compare trades).
- **A screenshot** of the chart at entry, marked up. This is the single highest-value habit in the journal -- it lets you re-examine your read after the emotion has drained away.
- **Rule-followed: yes/no.** Did you take the trade exactly as the plan defined it, or did you deviate? This one binary field is what makes the execution gap measurable.
- **Your emotional state** at entry and during the trade, in a word or two. Over time, patterns appear: the trades you took "to make back" a loss perform differently from the ones you took calmly.

### Computing your real expectancy

With a sample logged in R, your realized expectancy is just the average:

$$E[R]_{\text{realized}} = \frac{1}{n}\sum_{i=1}^{n} R_i$$

where $R_i$ is the R-outcome of trade $i$ and $n$ is the number of trades. Sum your R-multiples, divide by the number of trades, and you have the number that actually matters: what one trade pays you, on average, in your real account.

The critical discipline here is **sample size**. A handful of trades tells you nothing -- the standard error of an average outcome shrinks only as $1/\sqrt{n}$, so a 10-trade or 20-trade record is dominated by variance. Do not draw any conclusion about whether the plan works until you have **at least 30 trades**, and prefer 50 or 100 before making changes. (The statistics of why small samples lie, and how to backtest a plan honestly without fooling yourself with look-ahead bias and overfitting, are in [backtesting without fooling yourself](/blog/trading/technical-analysis/backtesting-without-fooling-yourself).)

### The review cadence and improving the plan

The review has two rhythms. The **weekly review** (our trader does it Sunday) is about *process*: did I follow the plan? It reads through the week's journal entries, checks the "rule-followed" column, and notes any deviation. It does not change the plan -- it grades the execution. The **monthly review** is about *the plan itself*: with a month of trades in hand, it computes realized expectancy, compares it to the plan's expected expectancy, and -- if a genuine, sample-supported problem appears -- makes **exactly one change**.

The "exactly one change" rule is deliberate. If you change five rules at once and the next month is better, you have no idea which change helped, and you have re-set your sample to zero. Disciplined improvement is one variable at a time, measured over a fresh sample. The plan evolves slowly, on evidence, at month-end -- never in the middle of a drawdown, and never on the strength of the last three trades.

#### Worked example: the plan's expected results, the equity path, and the drawdown

Before we diagnose leaks, we have to know what *success* looks like, so we can recognize it (and so we are not spooked by an ordinary drawdown). Let us compute the plan's expected results.

![A chart of the plan's expected equity path: a straight expected line rising to plus thirty R over one hundred trades, and a jagged realized path that dips into a roughly minus eight R drawdown near trade forty before recovering toward plus twenty-four R, with the drawdown labeled as normal variance.](/imgs/blogs/building-your-trading-plan-capstone-5.png)

The chart above is the plan's expected results. The straight line is the expected +30R; the jagged green-and-orange path is one realistic outcome that dips into a drawdown of about -8R near trade 40 before recovering. Let us do the arithmetic. Our trader's setup, measured over her sample, wins 45% of the time, with winners averaging +2R (she often takes partial profits before the full 4:1) and losers at -1R. Her expectancy is:

$$E[R] = 0.45 \times 2 - 0.55 \times 1 = 0.90 - 0.55 = +0.35\text{R per trade}$$

Round it to **+0.30R** to be conservative after costs. Over 100 trades, the expected result is +0.30 × 100 = **+30R**. In dollars, at \$250 per R, that is +30 × \$250 = **+\$7,500**, a 15% return on the \$50,000 account -- earned across roughly a year of part-time trading. That is a genuinely good, genuinely realistic result, and notice how *unglamorous* it is: a thin per-trade edge, harvested slowly, over a large sample.

Now the part that breaks most traders. The expected +30R is a straight line, but the *realized* path is jagged, and it includes a real drawdown. With a 45% win rate, a run of five or six losses in a row is not just possible -- it is *likely* somewhere in 100 trades. In the figure, an early loss streak drags the account to about **-8R** (a \$2,000 drawdown, 4% of the account) before the edge reasserts itself and the path climbs toward +24R by trade 100. That -8R drawdown is not a sign the plan is broken. It is exactly what a +0.30R edge looks like while it is being collected. The whole reason the risk rules in Part 3 are so conservative is to make that -8R drawdown *survivable* -- both financially (it is only 4% of the account) and emotionally (it is shallow enough to trade through). The one-sentence intuition: **a working plan does not produce a smooth rising line; it produces a jagged path with real drawdowns, and the plan's job is to make those drawdowns survivable so you are still trading when the edge shows up.**

#### Worked example: the review finds an execution leak, and the fix

Here is the review doing its most valuable work. Three months in, our trader sits down with a sample of 90 trades. She computes two numbers. First, on the trades where her journal says **rule-followed: yes**, the realized expectancy is **+0.30R** -- the plan is working exactly as designed. Second, across *all* trades including the deviations, her realized expectancy is only **+0.15R**. The gap between them is **0.15R per trade**, and it is not a plan problem -- the plan is fine. It is an **execution leak**.

![A bar chart of the execution-leak review: a tall blue bar showing the plan edge of plus zero point three R, a red bar showing the minus zero point one five R leak from skipped winners and moved stops, and a green bar showing the plus zero point one five R realized edge that is half of what the plan can pay.](/imgs/blogs/building-your-trading-plan-capstone-8.png)

The figure above prices the leak. The plan can pay **+0.30R** per trade (the blue bar) -- over 100 trades, +30R, or **+\$7,500**. But the execution leak costs **-0.15R** per trade (the red bar): she has been skipping valid signals after losses and, on a few trades, widening her stop "to give it room." Those deviations subtract 0.15R from every trade on average. What lands in her account is **+0.15R** (the green bar) -- over 100 trades, +15R, or **+\$3,750**. Half her edge is gone, and it has nothing to do with the market or the strategy. (This is precisely the execution gap from [why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) and the psychology post: the gap between the edge you have and the edge you keep, and for most traders it is bigger than the edge itself.)

The fix follows directly from the diagnosis, and it is behavioral, not strategic. She does **not** change the setup -- it earns +0.30R when she follows it. Instead she addresses the two specific leaks: she adds a hard rule that she takes *every* valid signal, including (especially) the ones right after a loss, because variance loads those with winners; and she makes her stop literally un-widenable by entering it as a hard order at the moment of the trade. The one-sentence intuition: **the review's superpower is separating a plan problem from an execution leak, because the two demand opposite fixes -- a plan problem means change the rules, but an execution leak means change your behavior and leave a working plan alone.**

## Part 6 -- the psychology contract

The last part of the plan is the one most traders skip, and it is the one that protects all the others. Parts 1 through 5 assume you will follow the rules. Part 6 -- the **psychology contract** -- is a set of pre-commitments that make following the rules possible when you are emotional, which is exactly when the rules matter most and feel least bearable. This is the operational core of [trading psychology and the execution gap](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap), written as commitments rather than as warnings.

### Pre-commitments against tilt

**Tilt** is the state -- borrowed from poker -- where a loss (or a win) knocks you off your disciplined process and you start trading emotionally: revenge-trading to "make it back," oversizing because you are angry, forcing trades out of frustration. You cannot reason your way out of tilt in the moment, because tilt *is* the impairment of your reasoning. The only defense is a pre-commitment made while calm. Our trader's contract includes:

- **The two-loss walk:** after two consecutive losses, she stands up and walks away from the screen for at least ten minutes. Not optional, not negotiable -- a physical interruption that breaks the spiral before it starts.
- **No revenge trade:** the trade taken immediately to recover a loss is forbidden by name. If she notices the *urge* to "make it back," that urge is itself the signal to stop for the day.
- **Zero overrides:** she does not override the checklist. If a trade fails a gate, it does not get taken, full stop. Overrides are logged as rule-breaks even when they win, because a winning override trains the worst possible habit.

### Sizing to stay calm

There is a quieter psychological lever that does more than any willpower technique: **size small enough to stay calm.** If a single trade's risk is large enough to make your heart race, you will manage it emotionally -- you will cut the winner early to relieve the anxiety, or freeze on a loser. The 0.5% sizing from Part 3 is not only a survival rule; it is a *psychological* rule. A \$250 risk on a \$50,000 account is small enough that no single outcome matters, which is exactly the detachment that lets you follow the plan. If you find yourself unable to follow your own rules, the first thing to try is not more discipline -- it is smaller size.

### Process goals over outcome goals

The final commitment reframes what counts as a good day. An **outcome goal** ("make \$500 today") is poison, because the outcome of any single day is mostly variance and not under your control -- chasing it pushes you to overtrade and override. A **process goal** ("follow my plan on every trade today; take every valid signal at the right size; skip everything that fails a gate") is entirely under your control, and it is the thing that actually produces the outcome over time. Our trader grades her day on process, not on profit. A day where she followed the plan perfectly and still lost \$500 to variance is a *good day*. A day where she made \$500 by overriding her checklist is a *bad day* that happened to be lucky -- and luck is the most dangerous teacher in trading.

#### Worked example: the psychology contract catching a tilt spiral

Trace a hard morning. Our trader takes Trade A (the demand-zone setup); it stops out, -\$250. She takes Trade B, a clean second setup; it also stops out, -\$250. She is now down \$500 (1% of the account, well within limits) and she feels it -- the urge to immediately take a third trade to "get it back" is strong and physical. Here is where the contract earns its place. Her two-loss rule fires: she stands up and walks away from the screen for ten minutes. When she comes back, a setup is forming, but it is marginal -- only three confluence factors. The calm version of her runs the checklist; gate 3 is a NO. She skips it. Had she stayed at the screen, tilted, she would almost certainly have forced that marginal trade at oversize "to make it all back" -- and the journal shows that her tilt-trades historically run at *negative* expectancy, the exact opposite of her edge. The one-sentence intuition: **the psychology contract does not try to make you feel calm; it pre-commits you to specific physical actions -- walk away, do not override, size small -- that hold the line for you precisely when you cannot hold it yourself.**

## The whole series in one thesis

We have now assembled the entire plan. Step back and see what it is. Every post in this series -- how a chart is born, why levels exist, what candlesticks really say, the indicator trap, expectancy, position sizing, the execution gap -- was building toward one thesis, and the plan is where that thesis becomes a thing you do.

![A tree of the series thesis: the root says TA places probabilistic, risk-managed, measured bets and never predicts; three branches are probabilistic (setups as odds), risk-managed (sizing and loss caps keep a thin edge alive), and measured (journal in R, realized expectancy, improve one rule a month).](/imgs/blogs/building-your-trading-plan-capstone-7.png)

The tree above is the entire series collapsed to its root claim and three pillars. The root is the thesis; the three branches carry the specific practices from the relevant posts. Read it as the contents page of everything we have done, organized not by topic but by the three commitments the plan makes.

### The three pillars, restated honestly

**Probabilistic.** Technical analysis reads the chart for odds, never for certainty. Every level, trend, and pattern is *evidence* that shifts the probability of the next move -- it is never a forecast. The plan honors this by defining setups as odds-favorable situations you take repeatedly, not as predictions you defend. You will be wrong on individual trades constantly, and that is fine, because being right on individual trades was never the goal.

**Risk-managed.** Because the edge is thin and the outcomes are random in the short run, survival is the precondition for everything. The plan fixes risk before the trade -- a small fixed fraction per trade, hard loss caps, sized to keep you calm -- so that no single trade and no normal losing streak can take you out. A thin +0.30R edge is worthless if one bad trade can blow up the account before the edge has 100 trades to express itself.

**Measured.** You do not *assume* you have an edge; you *measure* it. The journal turns trades into a sample, the review computes your realized expectancy, and the comparison between what the plan can pay and what you actually keep tells you whether to fix the strategy or fix your behavior. An unmeasured edge is a belief; a measured edge is a fact you can improve.

The plan is the single artifact that holds all three pillars at once. It makes the bets probabilistic (setups as rules), keeps them risk-managed (fixed sizing and caps), and renders them measured (the journal and review). That is the whole series, on one page.

### The honest closing message

Here is the honest part, the part the marketing never tells you. **The edge is small, fragile, and statistical.** A realistic, well-built retail technical edge is a fraction of an R per trade. It is fragile enough that a handful of emotional deviations can erase it. It is statistical enough that you will not see it cleanly for dozens of trades, and you will spend much of that time in drawdowns that feel like proof you are failing. There is no version of this where you predict the market and the money is easy. There is only a thin, real edge, and a plan disciplined enough to let the thin edge express itself across a large sample while keeping you alive through the variance.

That is not a disappointing conclusion. It is a *liberating* one. It means the job is not to be a genius forecaster -- it is to be a disciplined operator of a measured process. That is a skill you can actually build, with a plan you can actually write, on a page you can actually tape to your monitor. The market does not reward prediction. It rewards the discipline to keep placing the same probabilistic, risk-managed, measured bet, over and over, long after it stopped being exciting.

### What to study next

The plan is the destination of this series, but it is the *start* of the real work. From here the highest-value study is not another indicator or pattern -- it is **deliberate practice** of the plan you just wrote: take a small sample of trades, journal every one honestly, review them, and improve one rule. Re-read the foundations -- [what technical analysis really is](/blog/trading/technical-analysis/what-technical-analysis-really-is) and [why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) -- because the plan only makes sense on top of them, and they reward a second reading once you have a plan to anchor them to. And keep the most uncomfortable post close: [the execution gap](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap), because the gap between your plan and your results will always be the highest-return thing you can work on.

## Common misconceptions

A plan is simple to describe and hard to internalize, and the difficulty shows up as a set of stubborn misconceptions. Each one quietly undermines the plan; each correction is the *why*.

**"A plan predicts the market."** This is the root error, and every other misconception grows from it. A plan does not say what the market will do; it says what *you* will do when certain situations appear. It is a procedure, not a prophecy. If you find yourself updating your plan based on what you think the market will do next week, you have confused the two -- the plan is built from *measured tendencies over many trades*, not from a forecast of the next one. The market is going to do what it does; the plan is the only part of the situation you control.

**"More rules is a better plan."** Beginners, having learned that rules are good, conclude that more rules must be better, and bury their setup under fifteen conditions and nine indicators. This is a trap for two reasons. First, every added rule shrinks the number of qualifying trades, and a setup that triggers twice a year can never build the sample its edge needs. Second, the more conditions you require, the more likely you are *overfitting* -- piling on rules that explain the past beautifully and predict the future not at all. A good plan is *parsimonious*: a small number of genuinely independent, genuinely predictive rules. The skill is not adding rules; it is finding the few that carry the edge and ruthlessly cutting the rest.

**"You can skip the journal."** The journal is the most-skipped part of the plan and the least skippable. Without it, you have no idea what your realized expectancy is, which means you cannot tell a plan problem from an execution leak, which means you will "fix" the wrong thing -- tearing up a working strategy because of a behavioral leak, or grinding away at a behavioral leak that is actually a dead edge. Trading without a journal is trading blind: you feel like you are learning, but you have no measured feedback, so you are mostly just accumulating anecdotes and confirmation bias. The journal is what makes the edge a fact instead of a feeling.

**"The plan removes all judgment."** The opposite error from "more rules," and just as wrong. A plan does not turn you into a robot with no decisions to make; it *relocates* your judgment. The judgment that used to happen in the heated moment -- "should I take this? is it good enough?" -- gets moved to the calm moment of writing and reviewing the plan, where judgment is reliable. In the live moment, the plan removes the *kind* of judgment that emotion corrupts (impulsive go/no-go calls) and leaves the kind that genuine skill improves (reading context, recognizing your setup, assessing regime). You still judge; you just judge the right things at the right time.

**"A plan guarantees profit."** It does not, and any plan that claims to is a fraud. A plan with a genuinely positive expectancy, followed perfectly, still loses on roughly half its individual trades and still spends real time in drawdown. What the plan guarantees is not profit on any trade or in any month -- it is that *if* you have a real edge, you will actually collect it instead of leaking it away, and that no normal bad streak will end you before you do. Profit is the expected result of a disciplined process plus a real edge plus a large enough sample. The plan supplies the discipline; it cannot manufacture the edge or shrink the variance.

## How it shows up in real markets

The trading plan is not an academic nicety -- it is the operational core of how serious money is actually run, and the difference between traders who survive and traders who do not. Here are concrete, named ways the plan shows up in real markets.

**Prop firms require a documented plan before they fund anyone.** Proprietary trading firms and the modern "funded trader" challenge companies -- the ones that give you a large notional account if you pass an evaluation -- almost universally require, or strongly enforce, a written, rule-based trading plan, and they build the rules into the platform itself. The evaluation imposes a hard daily loss limit, a maximum total drawdown, and often a minimum number of trading days. Those are not arbitrary hurdles; they are exactly Part 3 of the plan -- per-trade risk, daily and weekly loss caps -- made into pass/fail conditions. The firms learned, by watching thousands of traders blow up, that the single best predictor of survival is not the strategy but the risk discipline. They encode the loss caps because they know that a trader without them will, on a bad day, give back a month of gains. The whole funded-trader industry is, in effect, a bet that a documented risk plan separates the survivors from the casualties -- and they price that bet in real money.

**The trader who only turned consistent after writing one.** This is the most common arc in trading, and it is worth stating plainly because it is so universal. A trader spends a year or two losing or breaking even, jumping between strategies, convinced the problem is that they have not yet found the *right* setup. They take more courses, buy more indicators, switch markets. Nothing works. Then -- often out of exhaustion -- they do the boring thing: they write down one setup, fix their risk at a small fixed fraction, build a checklist, and start journaling. And within a few months, with the *same* analytical skill they always had, they turn consistent. What changed was never the edge; it was the discipline that let the edge through. The plan did not make them a better analyst. It made them an operator instead of a gambler. This pattern is so reliable that experienced traders will tell a struggling beginner the same thing every time: stop looking for a better setup, and go write a plan for the one you have.

**The difference between a plan and a watchlist.** Many traders believe they "have a plan" when what they actually have is a *watchlist* -- a set of symbols and levels they are watching. A watchlist is useful, but it is not a plan, and confusing the two is a quiet failure. A watchlist says *what* you are looking at; a plan says what you will *do*, at what *risk*, under what *conditions*, and how you will *measure* the result. A watchlist is the input to Part 1 and Part 2; it is not Parts 3 through 6. You can tell the difference with one question: when a watched level is hit, does your "plan" tell you the exact entry, stop, size, and the gate the trade must pass -- or does it just tell you to "watch closely"? If it is the latter, you have a watchlist, and the in-the-moment decisions are still being made by your emotions.

**Why funds and desks run on process, not calls.** Inside professional asset managers and trading desks, the structure that governs everything is *process*, not prediction. A portfolio manager does not get to "feel" their way into a position; they operate inside a risk framework -- position limits, drawdown limits, sizing rules, a mandate that defines what they trade -- and they are reviewed on whether they followed the process, not just on the monthly P&L. Risk managers exist specifically to enforce the loss caps and sizing rules that the traders, left alone in a hot moment, would breach. Performance is attributed and measured trade by trade so that a good year is not mistaken for skill if it was variance, and a bad month is not mistaken for a broken strategy if the process was sound. This is, at institutional scale, exactly the six-part plan: a defined universe, defined setups, fixed risk, a gate, measurement, and a psychology layer (the risk manager) that holds the line when the human cannot. The reason the whole industry is built this way is the same reason your plan is: prediction does not survive contact with markets, but a disciplined, measured, risk-managed process does.

## When this matters to you / further reading

This matters the moment you place real money on a chart -- which is to say, before your first live trade, not after your first blown account. The single highest-leverage thing you can do, more than any indicator or pattern, is to write the one-page plan: your instruments and session, your setup as six rules, your fixed risk and loss caps, your eight-gate checklist, your journal format, and your psychology contract. It will fit on one page. Tape it where you can see it. Then take a small sample of trades, journal every one, and review them -- and resist, with everything you have, the urge to change the plan before you have the 30-plus trades that make its results mean anything.

This is the last post in the series, so let us close by pointing both backward and forward. Backward: the plan only makes sense on top of the foundations, and they reward re-reading now that you have a plan to anchor them to -- start again with [what technical analysis really is](/blog/trading/technical-analysis/what-technical-analysis-really-is) and the math of an edge in [why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies). Build the setup carefully with [building one high-probability setup](/blog/trading/technical-analysis/building-one-high-probability-setup) and size it with [position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion). Move it toward rules with [from discretionary to systematic](/blog/trading/technical-analysis/from-discretionary-to-systematic).

Forward: the plan is not the finish line; it is the starting block for **deliberate practice**. The real work -- the work that separates the people who read about trading from the people who can actually do it -- is the unglamorous loop of taking trades by the plan, journaling them honestly, reviewing them, and improving one rule at a time. Keep [the execution gap](/blog/trading/technical-analysis/trading-psychology-and-the-execution-gap) close, because closing that gap will always be the highest-return project you have. The edge is small, fragile, and statistical. The plan is what lets the math express itself across a large sample while keeping you alive through the variance. That is the whole series, and now it is one page you can write.

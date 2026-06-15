---
title: "Confluence: The Math of Why Stacking Independent Factors Raises Your Edge"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "High-probability setups are not magic, they are arithmetic: when independent factors from different evidence families agree at one price, their probabilities multiply into a real edge, while redundant signals only echo the first read."
tags:
  [
    "confluence",
    "technical-analysis",
    "probability",
    "independence",
    "trading-edge",
    "expectancy",
    "support-and-resistance",
    "momentum",
    "trading-setup",
    "risk-reward",
    "trading-psychology",
    "high-probability-setup",
  ]
category: "trading"
subcategory: "Technical Analysis"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** -- a "high-probability setup" is not a feeling and not a buzzword. It is the place on a chart where several *independent* reasons to expect the same move all point at the same price. The reason confluence works is pure probability: independent agreements multiply, while redundant agreements merely echo.
>
> - **Confluence means factors agreeing at one price.** When a prior swing low, a round number, a higher-timeframe uptrend, and a fresh trigger candle all land at \$100, you have four reasons instead of one. The trade is not stronger because the chart looks busy; it is stronger because four pieces of *different* evidence happen to agree.
> - **Independence is the whole game.** Two genuinely independent 60% factors do not give you 60% -- they give you about 69%. Three give about 77%, four about 84%. The combined odds climb because each factor adds evidence the others did not already contain.
> - **Redundant confluence is a lie.** Three momentum oscillators (RSI, stochastic, Williams %R) read the same recent price, correlate around 0.9, and so they are one witness saying the same thing three times. Stacking them keeps you near 61%, not 94%. This is the [indicator trap](/blog/trading/technical-analysis/the-indicator-trap) restated as probability.
> - **There are roughly five independent families:** structure and levels, trend and bias, momentum, volume, and timeframe alignment. Honest confluence takes *one* read from each family, never five from one.
> - **Score before you trade.** Pre-define the factors and the level, count how many genuinely independent ones agree, and only then decide. A higher hit rate at the same reward-to-risk lifts [expectancy](/blog/trading/technical-analysis/expectancy-why-win-rate-lies): in the worked example, 60% to 77% at a fixed 2-to-1 turns +0.80R per trade into +1.31R -- an extra +0.51R, or +51R over a hundred trades, for free.

A trader circles a level on the chart and says the setup is "high probability." Ask why and you get a list: the RSI is oversold, the stochastic is oversold, the price is near a round number, the MACD is about to cross, the Williams %R is oversold, and it "just looks good." Six reasons. Six confirmations. Surely six is better than one.

It depends entirely on whether those six reasons are *independent*, and most of the time they are not. Three of them -- RSI, stochastic, Williams %R -- are the same momentum read in three costumes, all computed from the same recent prices, all correlated around 0.9. Counting them as three separate confirmations is like asking the same witness the same question three times and treating the three identical answers as three independent corroborations. They are not. They are one witness, repeated.

This post is the honest math of confluence, and it is the constructive other half of the indicator-trap argument. There we showed that *redundant* signals do not confirm each other; here we show what *real* confirmation looks like and exactly how much it is worth. The short version is a single idea from probability: **when independent factors agree, their odds multiply; when redundant factors agree, the odds barely move.** A high-probability setup is the place where genuinely different kinds of evidence -- a structural level, a trend bias, a momentum read, a trigger -- all point at the same price. The picture below is the whole post in one frame.

![Four arrows labeled structure, trend bias, momentum, and trigger all pointing at a single price level of one hundred dollars, where their agreement combines into an edge that rises above any single factor and produces a high-probability long entry](/imgs/blogs/confluence-stacking-independent-factors-1.png)

The diagram above is the mental model for everything that follows. Four reasons, drawn from four different *types* of evidence, all point at the same \$100 level. Structure says \$100 is a prior swing low and a round number. Trend bias says the higher timeframe is in an uptrend, so you want to buy the dip. Momentum says RSI is at 32, leaving oversold. The trigger is a bullish engulfing candle right at the level. Because these four reads come from different evidence types, their agreement is not automatic -- and that is exactly why it matters. When evidence that *could* have disagreed instead agrees, the combined edge rises above any single factor, and you get a real high-probability setup: a long entry near \$100. The rest of this post makes that picture precise, with formulas and dollars.

One disclaimer, stated once. This is educational. It explains the probability mechanics of confluence so you can read any "high-probability setup" claim -- including your own -- honestly. It is not advice to trade anything, and it is certainly not advice to trade more. Every method that can make money can lose it, and we will be specific about how.

## Foundations: what confluence means

Before we can do any math, we need a precise, shared definition. The word "confluence" gets thrown around as a vibe -- "lots of things lining up" -- and the vibe is what gets traders into trouble. We will build the term from zero.

### A factor, and what it claims

A **factor** is a single, specific reason to expect a particular move from a particular price. "There is a prior swing low at \$100" is a factor: it claims that buyers previously defended \$100, so they may defend it again. "The daily chart is making higher highs and higher lows" is a factor: it claims the larger trend is up, so dips are more likely to be bought than sold. "RSI is at 32" is a factor: it claims momentum has stretched to the downside and may be due to revert. Each factor, on its own, is a probabilistic statement -- *this price is somewhat more likely to bounce than a random price would be*.

Notice the word *somewhat*. No single factor is decisive. A swing low gets broken all the time. Uptrends end. Oversold gets more oversold. A reasonable, honest number for a single decent factor's hit rate -- the fraction of times the expected move actually happens -- is around 55% to 65%. Throughout this post we will use **60%** as a round, defensible value for "one good factor." If you think your factors are better than that, the argument only gets stronger; if you think they are worse, the conclusions hold with smaller numbers. The point is never the exact percentage. The point is what happens when several of them agree.

### Confluence, defined

**Confluence** is the situation where two or more factors point at the *same price* (or a tight price zone) at the *same time*. The word comes from rivers: a confluence is where two streams meet and become one larger flow. On a chart, it is where two or more independent reasons to expect a move converge on one level.

The critical qualifier in that definition is "independent," and it is the qualifier almost everyone drops. Confluence is only worth something when the factors that agree are *different kinds of evidence* -- when each one could have disagreed but did not. Two factors that are really the same factor wearing two hats are not confluence. They are one factor, double-counted. The entire difference between a real high-probability setup and an expensive illusion lives in that distinction, so the rest of this section makes "independent" precise.

Two factors are **independent** when knowing one tells you nothing about whether the other is true. A structural level at \$100 and a daily uptrend are roughly independent: the fact that \$100 was a prior swing low does not tell you which way the 200-day average is sloping, and vice versa. They draw on different data and could easily disagree -- plenty of swing lows sit inside downtrends, and plenty of uptrends have no nearby level. So when they *do* agree, the agreement carries information.

Two factors are **redundant** (the technical word is *correlated*, or in the extreme *collinear*) when knowing one almost guarantees the other. RSI oversold and stochastic oversold are redundant: both are arithmetic transforms of the same recent closes, so if one is oversold the other almost certainly is too. They cannot easily disagree, so when they agree, the agreement tells you nothing you did not already know from the first one. (We unpacked exactly *why* these indicators are correlated -- they are deterministic functions of the same OHLC data -- in the [indicator trap](/blog/trading/technical-analysis/the-indicator-trap). Here we only need the consequence.)

Hold onto this sentence, because it is the thesis in one line: **independent agreements multiply your odds; redundant agreements echo the first read.** The next two sections turn that sentence into arithmetic.

### The witness analogy, made precise

The cleanest way to feel why independence matters is the courtroom analogy, and we will lean on it throughout the post, so let us make it exact. Imagine a crime and three witnesses. If three *different* people, who were standing in three *different* places, who do not know each other, all independently describe the suspect the same way, their agreement is powerful -- because for all three to coincide by chance is unlikely. Each one could have seen something different, and the fact that they did not is the evidence. That is **independent** corroboration.

Now imagine instead that you take *one* witness and ask her the same question three times, or you read her single written statement aloud three times. You have three "confirmations," but they all trace back to one observation. If that one observation was wrong, all three are wrong together; if it was right, you learned nothing on the second and third reading. That is **redundant** corroboration, and it is worth exactly as much as the single underlying observation -- no more. This is precisely the situation with technical factors: a structural level, a trend read, and a volume read are three witnesses who saw *different* things -- price location, the slope of the larger trend, and participation -- while RSI, stochastic, and Williams %R are one witness, recent price momentum, read aloud three times.

## The probability of independent factors

Now we make "the odds multiply" exact. We will use the cleanest possible model first, name its assumption honestly, and then show what changes when the assumption is only approximately true.

### The setup as a probability question

Pose the trade as a question: *what is the probability that this long works -- that price bounces from the level and reaches the target before the stop?* Call that probability $p$. A single factor gives you a base estimate; here $p = 0.60$. Confluence asks: if a *second* independent factor also points the same way, what is the new probability that the trade works?

To answer, we need one definition from probability. Two events $A$ and $B$ are **independent** when the probability that both happen equals the product of their individual probabilities:

$$P(A \text{ and } B) = P(A)\,P(B).$$

That little equation is the engine of confluence, but we have to apply it to the right events, or we will get nonsense. We are *not* multiplying "RSI oversold" by "price at a level." We are combining two factors that each shift our estimate of the *same* underlying question -- *does the trade work?* -- and the right tool for that is updating odds, not multiplying raw hit rates. Let us do it the careful way.

### Updating in odds: the honest version

The clean way to combine independent evidence is to work in **odds** rather than probabilities. The odds of an event with probability $p$ are $\frac{p}{1-p}$. A 60% factor has odds of $\frac{0.60}{0.40} = 1.5$, written 1.5-to-1 -- the trade is 1.5 times as likely to work as to fail, given that one factor.

Each additional *independent* factor that agrees multiplies the odds by a **likelihood ratio** -- roughly, "how much more often does this factor fire before a winner than before a loser?" If each of our 60% factors carries a likelihood ratio of about 1.5 (a modest, honest amount of evidence), then each independent agreement multiplies the running odds by 1.5:

- One factor: odds $= 1.5{:}1$, which is $p = \frac{1.5}{1+1.5} = 0.60$.
- Two independent factors: odds $= 1.5 \times 1.5 = 2.25{:}1$, which is $p = \frac{2.25}{3.25} = 0.69$.
- Three independent factors: odds $= 1.5 \times 1.5 \times 1.5 = 3.375{:}1$, rounded to about $3.4{:}1$, which is $p = \frac{3.375}{4.375} = 0.77$.
- Four independent factors: odds $\approx 5.06{:}1$, which is $p \approx 0.84$.

So two genuinely independent 60% factors give you about **69%**, three give about **77%**, and four give about **84%**. The combined edge rises with every independent agreement, and it rises fast at first because each factor is multiplying, not adding. The figure below plots exactly this climb.

![A bar chart of combined probability that the trade works, showing sixty percent for one independent factor, sixty-nine percent for two, seventy-seven percent for three, and eighty-four percent for four, with a separate bar for three redundant oscillators stuck near sixty-one percent](/imgs/blogs/confluence-stacking-independent-factors-2.png)

Read the green ladder left to right: one independent factor at 60%, two at 69%, three at 77%, four at 84%. Each step up is the same multiply -- the odds get scaled by 1.5 again -- but because we are converting back to probability, the *percentage* gains shrink as we climb (60 to 69 is nine points, 77 to 84 is only seven). That diminishing return is real and it is the reason five or six factors stop helping: by then you are near the ceiling, and any "extra" factor you add is almost always redundant anyway. Now look at the lone bar on the right, the one labeled three redundant oscillators: it sits at about **61%**, barely above the single-factor 60%. Three correlated momentum reads, stacked, gave you almost nothing. That bar is the entire argument of the next section, previewed.

#### Worked example: two independent factors beat either one alone

You are watching a stock drift down toward \$100. You have two factors, and you have done the honest work of checking that they are independent.

- **Factor 1 -- structure.** \$100 is a prior swing low and a round number. On its own, dips into this kind of level bounce roughly 60% of the time for you. Hit rate: $p_1 = 0.60$, odds $1.5{:}1$.
- **Factor 2 -- trend bias.** The daily chart is making higher highs and higher lows, an uptrend, so dips are more likely to be bought. On its own, buying dips in an established uptrend works roughly 60% of the time for you. Hit rate: $p_2 = 0.60$, odds $1.5{:}1$.

These two are independent: the location of a swing low says nothing about the slope of the 200-day average, and an uptrend says nothing about whether there happens to be a level nearby. Either could have been true without the other. So when both agree at \$100, you multiply the odds: $1.5 \times 1.5 = 2.25{:}1$, which is a hit rate of $\frac{2.25}{3.25} \approx 0.69$, or **69%**.

Sixty-nine percent is meaningfully better than the 60% you would have from *either* factor alone. You did not invent information; you collected a second, genuinely different witness, and the two agreeing is more convincing than one. That nine-point lift, for the cost of checking one more independent thing, is what confluence is actually selling. The rest of the post is about not getting cheated when you buy it.

### The assumption, named honestly

The multiply-the-odds rule assumes the factors are *conditionally independent* given the outcome -- loosely, that once you know whether the trade is going to win, the factors do not give each other extra information. Real factors are rarely perfectly independent, and pretending they are is how people get to absurd numbers like 94% from three oscillators. The fix is not to abandon the rule; it is to *be honest about how much each factor is really independent* and to refuse to count redundant ones. We treat the full conditional-independence machinery -- the exact formula, when it holds, and how partial correlation eats into the lift -- in the companion note on [joint and conditional independence](/blog/trading/math-for-quants/joint-conditional-independence-math-for-quants). For trading, the practical rule is simpler: *only multiply across factors that come from different evidence families*, which is what the next two sections are about.

### How sensitive is the lift to the base rate?

A fair objection: the whole ladder depends on assuming each factor is a 60% factor, and you do not actually know your factors are exactly 60%. So how much does the conclusion depend on that number? Reassuringly little, because the *mechanism* -- multiplying odds -- works at any base rate. Run it at a weaker 55% factor (odds $1.22{:}1$): two independent ones give odds $1.49{:}1$, a hit rate of about 60%, and three give about 64%. Run it at a stronger 65% factor (odds $1.86{:}1$): two give odds $3.45{:}1$, about 78%, and three give about 86%. In every case the same thing happens -- each independent agreement lifts the combined probability, and the lift is largest when you start from a middling base rate where there is the most room to move.

The base rate where confluence helps *most* is right around 50%, because that is where each factor is least decisive on its own and the multiplying does the most work. Near the extremes -- a factor that is already 90% reliable, or one that is barely 50% -- there is less to gain, either because you are near the ceiling or because the factor is too weak to move much even when stacked. This is a useful sanity check on your own factors: if a single read already wins 85% of the time, you do not need confluence, and you should be suspicious that you have measured the win rate honestly. The sweet spot for confluence is exactly the situation most traders are in: several individually-mediocre 55%-to-65% factors that, *if independent*, combine into something genuinely good. The "if independent" is doing all the work, which is why the next section is the one that matters most.

## Why redundant confluence is a lie

We have the optimistic case: independent factors multiply. Now the trap. The most common way traders fool themselves is by stacking factors that are not independent at all -- usually three or four momentum indicators -- and then multiplying their hit rates as if they were. The math they imagine and the math that is true are very far apart.

### What the redundant trader imagines

Suppose you have RSI oversold, stochastic oversold, and Williams %R oversold, and you treat each as an independent 60% factor. The naive multiplication of *failure* probabilities goes: the chance all three are wrong is $0.40 \times 0.40 \times 0.40 = 0.064$, so the chance at least one is right -- and the trade works -- is $1 - 0.064 = 0.936$, about **94%**. The trader feels enormous confidence. Five things, or three things, "all line up," and 94% is nearly a sure thing.

It is a fantasy, and the reason is that the three oscillators are not independent. They are all computed from the same recent closing prices. RSI, stochastic, and Williams %R correlate around **0.9** with each other -- when one is oversold, the others are oversold almost by construction. The naive calculation multiplied three numbers that are really one number. It is the probability equivalent of weighing yourself three times on the same scale and adding the readings.

### What is actually true

When factors are correlated, the second and third add almost no information *conditional on the first*. If RSI being oversold already told you most of what stochastic oversold would tell you, then learning stochastic is oversold barely updates your odds. In the extreme of perfect correlation (the indicators are identical), the second and third factors update your odds by a likelihood ratio of essentially 1 -- multiply by 1 and nothing changes. With correlation around 0.9, you get a tiny residual lift and no more. Three redundant 60% oscillators leave you at roughly **61%**, not 94%. The figure makes the contrast literal.

![A side-by-side comparison with redundant confluence on the left, where RSI stochastic and Williams percent R all read the same recent price and net out to one factor near sixty percent rather than ninety-four, and independent confluence on the right, where a level a higher-timeframe trend and a volume read are three different witnesses lifting the combined edge to about seventy-seven percent](/imgs/blogs/confluence-stacking-independent-factors-4.png)

The left stack is the lie. RSI says oversold, built from recent price -- one momentum view. Stochastic oversold reads the same recent price and restates the same view. Williams %R oversold is the same price again, correlation about 0.9, no lift. The net, honestly accounted, is still one factor: roughly 60% holds, *not* the 94% the naive multiplication promised. The right stack is the real thing. A level at \$100 (prior swing plus round number) answers *where*, at $p \approx 0.60$. A higher-timeframe uptrend answers *which way*, at $p \approx 0.60$, from different data. A volume read -- buyers stepping in -- answers *with what conviction*, from different data again. Three witnesses who saw different things, and their independent agreement lifts the combined edge to about **77%**. Same number of "confirmations" on each side. One side is worth nothing extra; the other is worth seventeen points of hit rate. The only difference is independence.

#### Worked example: three correlated momentum reads add no lift

You are looking at the same \$100 area and you want to feel confident, so you load three oscillators.

- **RSI** is at 32 -- oversold. You count it: $p \approx 0.60$, odds $1.5{:}1$.
- **Stochastic** is also oversold. You want to multiply: $1.5 \times 1.5 = 2.25{:}1$, implying 69%.
- **Williams %R** is also oversold. You multiply again: $2.25 \times 1.5 = 3.375{:}1$, implying 77%.

Stop. That 77% is a fiction, because the second and third factors are not independent of the first. All three are functions of the same recent closes; their pairwise correlation is about 0.9. The correct likelihood ratio for the second oscillator, *given* the first, is not 1.5 -- it is barely above 1.0, because stochastic told you almost nothing RSI had not already told you. Same for the third. So the honest combined odds stay near $1.5{:}1$ with a sliver of residual lift, and the true hit rate is about **61%**, not 77% and certainly not 94%.

You had one momentum factor. You looked at it through three lenses and convinced yourself you had three. The chart looked more confirmed; the trade was exactly as risky as before. This is why "more indicators" feels safer and is not -- and it is the single most expensive mistake in retail confluence.

#### The general rule for honest stacking

The figure that closes the post states the rule as a ladder, and it is worth lifting out here. Independent factors update the odds by 1.5 each step -- 1.5:1, then 2.25:1, then about 3.4:1 -- climbing 60 to 69 to 77 to 84 percent. Redundant factors update by about 1.0 each step -- the odds barely move, ~61%, ~62%, ~62% -- because each new oscillator is "three views, one witness." And beyond four genuinely independent factors, adding a fifth or sixth is almost always curve-fitting, not edge: you have run out of independent families, so anything extra is either redundant or noise you have hand-picked to fit the trade you already wanted. Independence is the rule, and four is roughly the natural ceiling.

### A quick test for whether two factors are really independent

You do not need to compute correlations to tell independence from redundancy at the table; you need one question: *could these two factors have disagreed, given the same chart?* Ask it of any pair.

- Structure (\$100 is a swing low) and trend (the daily is up): could disagree easily -- swing lows happen in downtrends all the time, and uptrends often have no nearby level. **Independent.** Count both.
- RSI oversold and stochastic oversold: could they have disagreed on this chart? Almost never -- both are functions of the same recent closes, so when one is oversold the other is too. **Redundant.** Count once.
- A round number (\$100) and a prior swing low that happens to be *at* \$100: partly overlapping -- round numbers often *become* swing lows because traders defend them, so these two share a cause. **Partly redundant.** Count as roughly 1.5 factors, not 2, and lean toward the conservative side.

The "could it have disagreed" test is the field version of the conditional-independence math. If two factors essentially always move together, their agreement is automatic and therefore uninformative; if they could plausibly have pointed different ways and instead agree, the agreement is evidence. Run that test on every factor before you add it to the score, and most of the redundant confluence you would otherwise stack falls away on its own.

## The independent factor families

If the secret is independence, the practical question is: *which factors are actually independent of each other?* You cannot eyeball it. The reliable way is to group factors into **families** that draw on genuinely different information, and then take *one* read from each family. Two factors from different families are roughly independent; two from the same family are redundant. Here are the families.

![Five labeled boxes for the independent factor families, structure or level for where price is, trend or bias for which way, momentum for how fast using one indicator not three, volume for how much conviction, and timeframe alignment for whether the timeframes agree, with a note that stacking three oscillators stays inside one family](/imgs/blogs/confluence-stacking-independent-factors-3.png)

The figure lays out the five families, and the discipline it encodes is "one read per family." Take them one at a time.

### Structure and levels -- *where*

This family answers **where** price is in its own history. It is raw price geometry: swing highs and swing lows, prior support and resistance, round numbers, prior consolidation edges, the high or low of a significant bar. A factor here claims that a specific price matters because the market has reacted to it before. It draws directly on price *location*, not on any smoothed transform. (Why levels have memory at all -- the order-flow and psychological reasons buyers and sellers cluster at the same prices -- is its own subject, covered in [why levels exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist).) One read: "\$100 is a prior swing low and a round number." That is the structure factor for the whole post.

### Trend and bias -- *which way*

This family answers **which way** the larger flow is going. It is the slope of the 200-day moving average, or simply whether the higher timeframe is making higher highs and higher lows (up) or lower highs and lower lows (down). A factor here claims that trades aligned with the trend are more likely to work than trades against it. This is *different information* from a level: a level tells you where price might react; the trend tells you which direction the reaction is more likely to resolve. One read: "the daily is in an uptrend, so favor longs."

### Momentum -- *how fast*

This family answers **how fast** price is moving and whether it has stretched. RSI, MACD, stochastics -- pick **one**, not three. This is the family where redundancy hides, because the popular oscillators are near-clones of each other. The entire indicator-trap problem lives inside this single family: stacking three momentum reads keeps you inside one family and adds almost nothing. One read: "RSI is at 32, leaving oversold." That is the momentum factor -- and it is *one* factor no matter how many oscillators you open.

### Volume -- *how much conviction*

This family answers **how much conviction** is behind the move. A volume spike on the bounce, price reclaiming VWAP (the volume-weighted average price), a turn in on-balance volume (OBV). Volume is genuinely different data from price geometry and from momentum oscillators -- it measures *participation*, how many shares actually traded, not just where price went. That independence is exactly why a volume read is a strong addition to a confluence: it can disagree with price, and when it agrees, the agreement is informative. One read: "volume spiked as buyers stepped in at the level."

### Timeframe alignment -- *do the timeframes agree*

This family answers whether **the timeframes agree**. The classic pattern is a daily uptrend providing the bias and a one-hour chart providing the precise entry trigger at the level. When the higher timeframe and the entry timeframe point the same way, you have alignment; when they fight, you do not. This is a meta-factor -- it is partly a combination of the others applied across scales -- so treat it as a confirmation that the picture is coherent rather than as a fully separate fifth witness. One read: "daily trend and one-hour entry agree."

A note on the trigger. The **trigger** -- the bullish engulfing candle at \$100 in our running example -- is not a sixth family; it is the timing event that turns a *zone of interest* into an *entry*. The four families tell you the area is worth trading; the trigger tells you *now*. We count it as a factor in the score (it is real evidence that buyers showed up at the level), but its main job is to convert a high-probability area into an actual entry with a defined stop.

### Why these families are roughly independent

The families are not an arbitrary list; they are chosen because they draw on *structurally different* information, which is exactly what makes their agreement multiply. It is worth seeing why, because once you understand the *why*, you can judge a new factor you have never met before -- you just ask which family it belongs to and whether you already have a read from that family.

Structure draws on price *location* in the raw series -- where, historically, the market reacted. Trend draws on the *direction and slope* of the larger flow -- a different feature of the same series, smoothed over a long window, so it captures the big picture rather than the local point. Momentum draws on the *rate of change* over a short window -- a derivative-like quantity that says how fast, not where or which way. Volume draws on an *entirely different data series* -- how many shares traded, which price geometry cannot see at all. Timeframe alignment draws on *scale* -- whether the same picture holds when you zoom out. Location, direction, speed, participation, scale: five different questions about the market, answered from different slices of data. That is why a factor from each can genuinely disagree with a factor from another, and why their agreement is informative.

Contrast that with the inside of the momentum family. RSI, stochastic, and Williams %R all answer the *same* question -- how fast, how stretched -- from the *same* short window of the *same* price series. They are not five different questions; they are one question asked five ways. Staying inside one family is the structural definition of redundancy, and crossing between families is the structural definition of independence. When you are unsure whether a new indicator adds anything, do not study its formula -- ask which of the five questions it answers, and whether you already have an answer to that question on the chart.

## Scoring a setup honestly

Knowing the families is not enough, because the human brain is very good at finding reasons for a decision it has already made. The defense is a *score* you compute on a fixed checklist, *before* you commit, so the count is evidence rather than a story. Here is the procedure.

### The checklist

Write down the five families. For the setup in front of you, mark each family as agreeing, neutral, or disagreeing, and take *at most one read per family*. Then count the agreements. That count -- not the busyness of the chart -- is your score. A few rules make the score honest:

1. **One read per family.** Three oversold oscillators count as one momentum agreement, period. If you find yourself listing RSI, stochastic, *and* Williams %R as three points, you are inflating the score.
2. **Independence over quantity.** A two-factor setup from two different families beats a five-factor setup that is really one family in five costumes. Quality of independence, not raw count.
3. **Pre-define the level and the factors before the trade.** Decide what you are looking for *before* price gets there. We will see in a moment exactly what goes wrong when you decide first and count second.
4. **Disagreements count too.** If the trend family disagrees -- you want to buy a dip but the daily is in a downtrend -- that is a *minus*, not a thing to ignore. A high score with a major family screaming the other way is not high probability.

### The confirmation-bias trap

The reason the "before" in rule 3 is in italics is that scoring after you have decided is worse than not scoring at all -- it launders a hunch into the appearance of evidence. The figure walks the failure step by step.

![A four-step flow of the confirmation-bias trap, first you want in because price is moving and the decision is made first, second you hunt for reasons by opening six indicators and keeping the three that agree, third you feel confident that five things line up although they were chosen to fit the decision, and fourth the fix is to pre-define the factors and the level before the trade then score then decide](/imgs/blogs/confluence-stacking-independent-factors-7.png)

Step one: you *want in*. Price is moving, there is fear of missing out, and the decision is effectively made before any analysis. Step two: you hunt for reasons -- open six indicators, keep the three that happen to agree, quietly ignore the two that do not. Step three: you feel confident because "five things line up," but they did not line up on their own; you selected the ones that fit the decision and discarded the ones that did not. That is not confluence, it is *curve-fitting a justification*. Step four is the fix and it is the whole point of scoring: list the factors and the level **before** the trade, score against that fixed list, and *then* decide. If the score is high on a list you wrote in advance, the agreement is real. If you only found the factors after you wanted in, the score is a story you told yourself.

This is also why the score has to be a small, fixed checklist. A long, flexible list of possible factors is an invitation to confirmation bias -- there is always *some* indicator that agrees if you are allowed to keep searching. Five families, one read each, defined in advance, is small enough to be honest.

### What score is "enough"?

Traders always want a threshold, so here is an honest one with its caveats. A score of **2 out of 5 independent families** is a marginal setup -- it lifts you to about 69%, which is real but thin, and you should size it small or pass. A score of **3** is a solid setup at about 77%, the workhorse number, and it is where most of the edge that confluence offers actually lives. A score of **4** is excellent at about 84%, but rare, and you should be suspicious if you score 4-out-of-5 often -- it usually means you are over-counting redundant factors or shopping for agreement. A score of **5** is essentially never genuinely independent, because the timeframe family overlaps with the others, so treat a claimed 5 as a 4 and move on.

The threshold is not the whole decision, though, because reward-to-risk and the disagreement of any single family both override the count. A 3-out-of-5 setup with a 3-to-1 reward-to-risk and no family disagreeing is far better than a 4-out-of-5 setup where the trend family is screaming the other way and the reward-to-risk is 1-to-1. The score is a *probability* input; you still have to combine it with the *payoff* input, which is the entire lesson of expectancy. Think of the score as setting the hit rate and the entry/stop/target as setting the R-multiple, and remember that you need both to be good before you have a trade worth taking.

## Worked examples: scoring a real setup

We have the families and the scoring rule. Now put them on a real chart with real numbers, entry, stop, and target. The numbers below are the ones in the figure; carry them exactly.

### The setup

Price has been trending up on the daily and is now pulling back toward \$100. You wrote your checklist before price arrived. As the pullback completes, you mark the families.

![A price-versus-time chart of a four-factor confluence long at one hundred dollars, with a confluence level zone from ninety-nine fifty to one hundred fifty, an entry at one hundred dollars fifty, a stop at ninety-eight equal to one R of two dollars fifty, and a target at one hundred eight for a three R trade, annotated with a bullish engulfing trigger candle and a daily higher-highs-higher-lows uptrend, scored four out of four](/imgs/blogs/confluence-stacking-independent-factors-5.png)

#### Worked example: a scored four-factor confluence long

Read the score off the chart, one family at a time:

- **Structure (agrees).** \$100 is a prior swing low *and* a round number. The confluence zone is the band \$99.50 to \$100.50 where these line up. *+1.*
- **Trend bias (agrees).** The daily is making higher highs and higher lows -- a clean uptrend -- so a dip into support is a buy-the-dip, with the larger flow behind you. *+1.*
- **Momentum (agrees).** RSI is at 32, leaving oversold as price turns up at the level. One momentum read, counted once. *+1.*
- **Trigger / timeframe (agrees).** A bullish engulfing candle prints right at the level on the entry timeframe, confirming the daily bias on the lower timeframe. *+1.*

**Score: 4 out of 4**, across four genuinely different families. By the probability ladder, four independent ~60% factors put the combined hit rate near **84%** -- though for the trade plan below we will use the more conservative three-independent-factor number, 77%, and treat the trigger as the timing event rather than a fully separate witness. Now the trade itself, straight off the figure:

- **Entry:** \$100.50, just as the engulfing candle confirms the bounce.
- **Stop:** \$98, below the level and the swing low. The distance from entry to stop is $100.50 - 98.00 = \$2.50$. That distance *is your 1R* -- the unit of risk for this trade, \$2.50 per share.
- **Target:** \$108. The distance from entry to target is $108.00 - 100.50 = \$7.50$, which is exactly three times the \$2.50 risk. So the target is **+3R**.

The reward-to-risk on this trade is **3-to-1**: you are risking \$2.50 to make \$7.50. (For the expectancy comparison in the next example we will hold reward-to-risk at a flat 2-to-1, which is the more conservative and more comparable number; this specific trade happened to offer 3-to-1.) The score told you the *probability* side -- four independent reasons agree, so the bounce is more likely than not. The entry, stop, and target give you the *payoff* side. You need both, and the next example shows why.

### From hit rate to expectancy

A high hit rate is only worth something if you convert it into money, and the bridge is **expectancy** -- the average R you make per trade, counting wins and losses together. We built expectancy from scratch in [why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies); here we just plug confluence into it. The formula, with $-1$R losers and a reward-to-risk of $W$ (so winners pay $+W$R):

$$E[R] = p \cdot W - (1-p) \cdot 1.$$

![A bar chart of expectancy per trade in R-multiples at a fixed two-to-one reward-to-risk, showing plus zero point eight zero R for a single factor at sixty percent hit rate and plus one point three one R for three independent factors at seventy-seven percent hit rate, a lift of plus zero point five one R per trade or plus fifty-one R over a hundred trades](/imgs/blogs/confluence-stacking-independent-factors-6.png)

#### Worked example: confluence lifts expectancy at the same reward-to-risk

Hold reward-to-risk fixed at **2-to-1** ($W = 2$, winners pay +2R, losers cost -1R) and change only the hit rate, the way confluence does.

- **Single factor, hit rate 60%.** $E[R] = 0.60 \times 2 - 0.40 \times 1 = 1.20 - 0.40 = +0.80\text{R}$ per trade.
- **Three independent factors, hit rate 77%.** $E[R] = 0.77 \times 2 - 0.23 \times 1 = 1.54 - 0.23 = +1.31\text{R}$ per trade.

The reward-to-risk did not change. The only thing that changed is that confluence raised the hit rate from 60% to 77% by adding two genuinely independent factors. That lifted expectancy from **+0.80R to +1.31R** -- an extra **+0.51R per trade**. Over a hundred trades, that is **+51R of additional edge**, collected from the same setups at the same reward-to-risk, purely by demanding that independent factors agree before you commit.

That is the dollars-and-cents case for confluence, and it is why the probability math is the point and not a decoration. Win rate alone does not pay; expectancy pays; and the clean, low-cost way to raise expectancy without touching your stops or targets is to be more selective -- to take the trade only where independent evidence stacks up. Selectivity *is* the edge. The figure's note says it plainly: +0.51R per trade, +51R over a hundred trades, from the same reward-to-risk.

There is a second, quieter benefit hiding in that 60%-to-77% jump, and it is about survival rather than expectation. A higher hit rate does not just raise the average; it *shortens the losing streaks*. The probability of a run of, say, six straight losers is $(1-p)^6$, which is $0.40^6 \approx 0.0041$ at a 60% hit rate but only $0.23^6 \approx 0.00015$ at 77% -- roughly twenty-seven times rarer. Fewer and shorter drawdowns mean you are far less likely to hit your risk-of-ruin threshold or, just as important, your *psychological* ruin threshold -- the point where a string of losses makes you abandon a perfectly good system. So confluence buys you two things at once: a higher average outcome *and* a smoother path to it. The smoother path is what lets you stay in the seat long enough for the higher average to compound, which is the whole reason expectancy matters in the first place.

## Common misconceptions

The math is simple once stated, but the intuitions it overturns are sticky. Here are the ones that cost the most.

### "More confirmations are always better"

No -- *more independent* confirmations are better, and you run out of independence fast. The fifth and sixth factors are almost always either redundant (another oscillator) or hand-picked to fit a decision you already made. Past four genuinely independent families, adding factors does not raise your edge; it raises your *confidence* without raising your odds, which is the most dangerous combination in trading. The right number of factors is "as many independent families as honestly agree," and that is rarely more than four.

### "If three indicators agree, the odds multiply to over 90%"

This is the central error and the reason this post exists. Multiplying hit rates is only valid for *independent* factors. Three momentum oscillators correlate around 0.9, so they are one factor, and one factor stays near 60%. The 94% is arithmetic applied to the wrong inputs. Whenever you catch yourself multiplying, stop and ask: *are these factors from different families, or the same family in different costumes?* If the same family, do not multiply -- count it once.

### "Confluence means a crowded chart"

The opposite. A crowded chart is usually five reads from one or two families, which is redundancy dressed as diligence. A clean confluence chart has *few* marks -- one level, one trend note, one momentum read, one volume note -- precisely because each one comes from a different family. If your chart needs eight indicators to "see" the setup, you are looking at the [indicator trap](/blog/trading/technical-analysis/the-indicator-trap), not confluence.

### "A high score guarantees the trade works"

It does not, and the language matters. Four independent factors put the hit rate near 84%, which means the trade *still fails about one time in six*. Confluence shifts the *probability*; it does not deliver certainty. This is why the stop is non-negotiable: even a beautifully scored 4-out-of-4 setup loses sometimes, and the only thing that keeps a losing high-probability trade from becoming a catastrophe is the predefined stop and a position sized so that a -1R outcome is survivable. High probability is not the same as no risk.

### "I can find confluence on any chart if I look hard enough"

Yes, you can -- and that is the disease, not the cure. If you allow yourself an unlimited, flexible list of possible factors and search until enough of them agree, you will *always* find a setup, because noise is plentiful. That is the confirmation-bias trap from earlier. The discipline that prevents it is the fixed, pre-defined checklist of five families: you mark them as they are, you do not go shopping for extra factors, and you accept the score you get.

### "Independence is binary -- factors either are or are not independent"

In practice it is a spectrum, and the families are a rough guide, not a guarantee. A swing low and a round number are *somewhat* correlated (round numbers often *become* swing lows), so two structural reads at the same price are not two fully independent factors. The honest move is to be conservative: when two factors share a family or clearly overlap, count them as roughly one, not two. Under-counting independence is a cheap mistake; over-counting it is the expensive one.

### "Confluence works because the level is special"

It is tempting to think the magic is in the level itself -- that \$100 has some intrinsic power. It does not. Confluence works because *several independent observers agree*, and the level is just the place where they happen to agree. The same logic would apply to any price the evidence converged on; \$100 is not special, the *agreement* is. This matters because it stops you from fetishizing particular levels or numbers and refocuses you on the only thing that carries information: whether genuinely different kinds of evidence point the same way. A round number with nothing else going for it is a weak factor. A round number where a swing low, an uptrend, and a volume surge also live is a high-probability setup -- and it would be exactly as high-probability at \$73.40 if the same four independent reads happened to land there.

### "If a setup loses, the confluence was wrong"

This is the most damaging psychological error, because it makes you abandon a good process after a normal loss. A 4-out-of-5 setup at 84% *is supposed to lose about one time in six*. When it does, the confluence was not wrong; the unlikely-but-expected outcome simply happened. Probability is a long-run statement, and any single trade can land on the wrong side of even excellent odds. The way to evaluate a confluence process is over dozens of trades -- does your 3-factor bucket actually hit near 77%? -- not over the last one. Judging the method by the most recent result is how traders talk themselves out of an edge that was working the whole time. Keep the score, log the outcomes, and let the sample, not the last trade, tell you whether your factors are as independent and as reliable as you claimed.

## How it shows up in real markets

Confluence is not a chart-pattern curiosity; it is how serious participants actually think about location, and it shows up across asset classes. A few examples, with the usual caveat that all live numbers are *as of* mid-2026 and are illustrative, not predictions.

### Round numbers as magnet levels

Major round numbers -- the S&P 500 at 5,000 or 6,000, Bitcoin at \$100,000, gold at \$2,000 -- repeatedly act as structural levels because orders cluster there: stops, limit orders, and options strikes pile up at round figures. When a round number *also* coincides with a prior swing or a moving average, you have structure-plus-structure or structure-plus-trend confluence, which is exactly the independent-family stacking this post describes. Bitcoin's behavior around \$100,000 through 2024 and 2025 is a textbook case of a round number that became a battleground precisely because so many independent reasons pointed at it at once. (As of mid-2026; round-number levels are well documented historically but never guaranteed to hold.)

### Moving averages plus prior support in equities

A very common institutional-style setup is a pullback in a strong stock to where the **200-day moving average** (a trend-family read) sits *on top of* a prior support level (a structure-family read). Because the trend filter and the price level are different families, their coincidence is genuine confluence, and these spots tend to attract buyers. You can see this on countless large-cap charts -- a multi-year uptrend dipping to the 200-day where an old breakout level also sits. It is not magic; it is two independent families agreeing on *where* and *which way*. (Illustrative; the 200-day is widely watched but is not a guaranteed floor, and strong downtrends slice through it routinely.)

### VWAP and the open in intraday futures

Intraday traders in index futures (ES, NQ) lean heavily on **VWAP** -- the volume-weighted average price -- as a volume-family level, and on the session open and prior-day high/low as structure. When price returns to VWAP *and* a prior-day level *while* the higher-timeframe trend is intact, that is three families converging intraday, and reactions there are reliably sharper than at a random price. The volume family is doing real, independent work here: VWAP encodes *where the average participant is positioned*, which raw price geometry does not. (As of mid-2026; intraday behavior is regime-dependent and can change with volatility and liquidity.)

### Fibonacci confluence, honestly

**Fibonacci retracements** (the 38.2%, 50%, 61.8% levels of a prior swing) are popular, and the honest read is that they matter mostly when they *coincide with a real structural level* -- a prior swing low, a round number, a moving average. A Fib level alone is a weak, self-referential factor; a Fib level *plus* a prior support *plus* the trend is genuine confluence because the latter two are independent of the Fib geometry. This is a useful test of the whole framework: a single soft factor is noise, but its coincidence with independent hard factors is signal. (Illustrative; Fibonacci levels have no mechanical force and "work" largely through self-fulfilling attention plus genuine confluence.)

### Options strikes and dealer positioning

In equity index and single-name options, large open interest at a strike (and the dealer hedging that comes with it) can pin or repel price near that strike, especially into expiration. When a heavily-traded strike lines up with a technical level, you again have two independent families -- one from the options market's positioning, one from price structure -- agreeing on a price. Practitioners watch these confluences around monthly and quarterly expirations. (As of mid-2026; dealer-positioning effects are real but estimated indirectly and vary with how options are actually hedged.)

The thread through all four: confluence in the wild is always *different kinds of evidence agreeing on a price*. Round numbers and prior swings (structure), moving-average slopes (trend), VWAP and open interest (volume and positioning), trigger candles (timing). Where independent families converge, reactions are sharper -- not because the chart is busy, but because independence is doing arithmetic on your behalf.

## When this matters to you and further reading

Confluence is the constructive half of a single, two-sided truth about technical analysis. The destructive half is the [indicator trap](/blog/trading/technical-analysis/the-indicator-trap): stacking correlated indicators *lowers* your edge because they echo. The constructive half is this post: stacking *independent* factors *raises* your edge because they multiply. Both halves are the same probability fact -- independence is the thing that determines whether agreement carries information -- seen from two directions.

If you take one operational habit from all of this, make it the **score on a pre-defined checklist**. Before price reaches your level, write down the five families -- structure, trend, momentum, volume, timeframe -- and decide what would count as each one agreeing. When price arrives, mark them honestly, count *one read per family*, and let the count decide whether you act. That single discipline prevents the confirmation-bias trap, kills the urge to stack five oscillators, and turns "this looks high probability" into a number you can defend. The closing figure is the rule in one frame.

![A summary ladder contrasting independent factors that climb from sixty to sixty-nine to seventy-seven to eighty-four percent as the odds multiply by one and a half each step, against redundant oscillators that stay stuck near sixty percent because three views are one witness, with the note that five or more factors add curve-fitting not edge](/imgs/blogs/confluence-stacking-independent-factors-8.png)

The green ladder is independence at work: 60% to 69% to 77% to 84%, the odds scaling 1.5:1, 2.25:1, 3.4:1 as each genuinely different factor agrees. The flat gray steps are redundancy: ~61%, ~62%, ~62%, because three correlated oscillators are one witness wearing three faces. And the warning at the end -- *5+ adds curve-fitting, not edge* -- is the discipline that keeps you from over-counting. Independence is the rule. When you have it, four factors are plenty; when you do not, no number of confirmations will save you.

Where to go next, in the order that builds the argument:

- **[Why levels exist](/blog/trading/technical-analysis/support-and-resistance-why-levels-exist)** -- the order-flow and psychological reasons price has memory at specific prices, which is the foundation of the structure family.
- **[The indicator trap](/blog/trading/technical-analysis/the-indicator-trap)** -- the destructive half: why correlated indicators echo instead of confirm, and why a crowded chart is a confession.
- **[Expectancy: why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies)** -- the money side: how a higher hit rate at the same reward-to-risk converts into a real, compounding edge.
- **[Building one high-probability setup](/blog/trading/technical-analysis/building-one-high-probability-setup)** -- turning this framework into a single, repeatable, fully-specified setup you can actually trade and track.
- **[Joint and conditional independence](/blog/trading/math-for-quants/joint-conditional-independence-math-for-quants)** -- the formal probability underneath the multiply-the-odds rule, including exactly when it holds and how correlation eats the lift.

One last time, plainly: this is educational, not advice. Confluence raises probabilities; it does not remove risk. Even a 4-out-of-4 setup loses about one time in six, every popular level eventually breaks, and the only thing standing between a high-probability loss and a blown account is a stop you placed in advance and a size you can survive. Do the probability honestly, count only independent evidence, and let the stop handle the times the odds simply do not land.

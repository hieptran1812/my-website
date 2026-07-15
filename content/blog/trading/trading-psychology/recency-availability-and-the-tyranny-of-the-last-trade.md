---
title: "Recency, Availability, and the Tyranny of the Last Trade"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "Your brain judges the odds of the future by how easily the recent past comes to mind, and it lets the result of your last trade size your next one. Here is the science of availability and recency, the dollar cost of both, and the exact drill that breaks the spell."
tags: ["trading-psychology", "recency-bias", "availability-heuristic", "behavioral-finance", "position-sizing", "revenge-trading", "base-rates", "extrapolation", "risk-management", "decision-making"]
category: "trading"
subcategory: "Trading Psychology"
author: "Hiep Tran"
featured: true
readTime: 32
---

> [!important]
> **TL;DR** — Your brain estimates how likely something is by how easily an example comes to mind, and nothing comes to mind more easily than what just happened. That single shortcut makes you extrapolate the last trend to infinity, over-insure right after a crash, and let your previous trade set the size of your next one.
>
> - **Availability** (Tversky & Kahneman, 1973): we judge probability by ease of recall, so vivid and recent events feel far more likely than their true frequency. **Recency** is the special case of overweighting the single latest data point.
> - **The tyranny of the last trade:** letting the outcome of trade N set the size and aggression of trade N+1 — revenge-sizing after a loss, freezing after a loss, or pressing recklessly after a win. The setup did not change; only your memory did.
> - Recency is expensive in three specific ways: it makes you buy protection at the top of a volatility spike, it makes you extrapolate at exactly the wrong moment, and it lets one doubling-down streak vaporize a genuinely winning edge.
> - **The number to remember:** in March 2009 the S&P 500 closed at its bear-market low of 676.53; by February 2020 it closed at 3,386.15, roughly five times higher. The traders who "learned their lesson" from 2008 and stayed out are the clearest picture of recency turned into a decade-long mistake.
> - **The drill:** replace "what just happened?" with "what is the long-run frequency?" (the base-rate anchor), and run a fixed reset ritual between trades so trade N+1 inherits none of trade N's emotion or P&L.

On the morning of January 17, 1995, a magnitude-6.9 earthquake struck Kobe, Japan. Half a world of financial consequences flowed from it, but one man felt it more than most: a 28-year-old trader named Nick Leeson, sitting on a pile of hidden bets that the Nikkei would hold steady. It did not hold steady. And here is the part that matters for us — Leeson did not cut. He did what the last several months of small recoveries had taught him to do. He doubled. Every time the position lost, he bet larger that it would come back, because in his recent experience it always had. Six weeks later [Barings Bank](https://www.britannica.com/event/bankruptcy-of-Barings-Bank), the oldest merchant bank in Britain, was insolvent, buried under roughly £827 million (about US\$1.4 billion) of losses hidden in an account numbered 88888.

Leeson is an extreme, criminal case. But the engine that drove him is the most ordinary thing in the world, and it is almost certainly running in your head right now while you trade. It is the tendency to let the most recent, most vivid thing you experienced dictate what you believe will happen next and how hard you should bet on it. Psychologists call the general version the *availability heuristic*; traders feel the sharpest version of it as what I call the tyranny of the last trade. The grid below is the mental model for this whole piece: one closed trade, two possible futures, and a single fork that decides which one you get.

![Two paths lead out of every closed trade: a red path where you react to the result and your edge leaks away, and a green path where you reset first and it compounds; the fork is whether you reset before re-entering.](/imgs/blogs/recency-availability-and-the-tyranny-of-the-last-trade-1.webp)

Read it top to bottom. The moment a trade closes, the profit or loss is a fact — it is in the account, it is over. What happens next is entirely inside your head. On the red path, you let the outcome set your mood, the mood distorts your sense of the odds, the distorted odds corrupt the size of your next bet, and a perfectly good system slowly leaks money through sizing noise. On the green path, you run a short reset, re-anchor on the long-run frequency, size the next trade exactly as the system says, and the edge compounds. The amber box in the middle is the whole secret: the setup you are about to trade is identical either way. The only thing that changed between the two futures is your memory of the last one. Let us build the tools to stay on the green path, starting from zero.

## Foundations: the building blocks of a brain that overweights the last thing it saw

You need no finance or psychology background for this section. We are going to define, from scratch, the three ideas that make recency so dangerous: the mental shortcut that swaps ease-of-recall for real probability, the special case where the newest data point drowns out everything before it, and the specific way both of those reach into your position size. This is where the science lives.

### A "base rate" is the long-run frequency, and it is what you should be estimating

Start with the single most useful word in this article: *base rate*. A base rate is simply the long-run frequency of something — how often it actually happens across a large number of trials. The base rate of a fair coin landing heads is 50%. The base rate of a single-session drop of 5% or more in a major stock index is tiny: far below one percent of all trading days, a handful of occurrences per decade. The base rate of your favorite setup working is whatever your journal or your backtest says it is across hundreds of instances, not across the last three.

Every question that matters in trading is secretly a base-rate question. "Will this bounce?" means "what fraction of times, historically, has this kind of setup bounced?" "Is a crash coming?" means "how often, per year, do crashes of this size occur?" The correct, boring, profitable answer is almost always a number you could look up. The problem is that your brain refuses to look it up. It reaches for something faster instead.

### The availability heuristic: judging odds by what springs to mind

In 1973, Amos Tversky and Daniel Kahneman published a paper with a deceptively plain title, ["Availability: A heuristic for judging frequency and probability."](https://www.sciencedirect.com/science/article/abs/pii/0010028573900339) Their claim was that when people are asked how likely or how frequent something is, they do not compute a base rate. Instead they run a shortcut: they ask themselves how easily they can bring examples to mind, and they read "easy to recall" as "common" and "hard to recall" as "rare."

Their most famous demonstration was almost silly in its simplicity. They asked people whether more English words begin with the letter K or have K as their third letter. Most people said words *starting* with K are more common — because it is easy to summon king, kite, kitchen, and hard to summon words like acknowledge or ask where the K hides in the third slot. In reality, across the letters they tested, there are roughly three times as many words with the letter in the third position. The mind confused "easy to retrieve" with "more numerous," and it was wrong by a factor of three.

The picture below is that shortcut, laid out as the five-step assembly line your brain actually runs.

![The availability heuristic in five steps: a vivid event is encoded strongly, retrieved easily, and its ease of recall is read as high frequency, so a single recent crash gets sized and hedged as if it were imminent.](/imgs/blogs/recency-availability-and-the-tyranny-of-the-last-trade-2.webp)

A vivid event — say, a day you watched your screen bleed 5% — gets encoded hard because emotion tags memories for keeping. Because it is encoded hard, it comes back instantly the next time you ask "how likely is a crash?" Because it comes back instantly, your brain reads it as frequent. And because it feels frequent, you size and hedge as though it is about to happen again. Every arrow in that chain is a substitution of "how does this feel to recall?" for "how often does this actually occur?" Nowhere in the chain does anyone consult the base rate.

### Recency: the special case where the newest point shouts loudest

Recency bias is availability aimed at time. Of all the things that are easy to recall, the easiest is the thing that happened most recently. So the single most recent observation gets a vote out of all proportion to what it deserves. Psychologists first pinned this down with the *serial position effect* — Bennet Murdock's 1962 experiments showed that when people memorize a list, they remember the last few items far better than the middle (the "recency" portion of the curve). The last thing in is the first thing out.

For a trader, the observation that just landed is the last trade, the last tick, the last three sessions. The chart below shows what recency does to the weight you place on a sequence of observations.

![A bar chart of the weight the mind assigns to ten past observations: nearly flat and small for the older bars but towering for the most recent, versus a dashed base-rate line that would give every observation an equal vote.](/imgs/blogs/recency-availability-and-the-tyranny-of-the-last-trade-3.webp)

The honest way to weigh ten equally relevant observations is the dashed line: each gets one-tenth of the vote, the base rate. What recency actually does is the red bar on the right: the "now" observation towers over everything before it, often carrying several times the weight of the older data, while observations from a month ago are quietly discarded. You are not looking at the distribution of outcomes; you are looking at the last outcome with a magnifying glass.

### The tyranny of the last trade: when recency sets your position size

Now we arrive at the specific way this reaches into your P&L. It is one thing to misjudge a probability. It is another, more expensive thing, to let the outcome of your previous trade decide the *size* of your next one. This is the tyranny of the last trade, and it takes three forms depending on how the last trade felt.

![A three-by-three grid: after a loss you feel angry (revenge, oversize) or shaken (timid, undersize); after a win you feel euphoric (house money, oversize); and in every case the disciplined move in the green column is the same fixed 1% risk, unchanged.](/imgs/blogs/recency-availability-and-the-tyranny-of-the-last-trade-4.webp)

After a loss, you are either angry, which makes you want to size up and win it back right now, or you are shaken, which makes you size down and skip the very next setup even when it is an A+. After a win, you are euphoric, playing with "house money," and you press the next bet harder than the plan allows. Notice the green column: whatever the last trade did, the correct move is identical — risk the same fixed fraction your system prescribes. Every distorted urge points away from that column. The tyranny is that the outcome of a trade that is *already over* is allowed to reach forward and change a trade whose odds it has nothing to do with.

#### Worked example: the felt-probability tax

Suppose you run a \$50,000 account. Yesterday the market fell 5% and you watched it happen in real time. Today your gut says another big down day is maybe 30% likely, so you buy one-month index puts as a hedge for \$600. The trouble is the base rate: single-session drops of 5% or more happen on far below 1% of trading days. Your gut's 30% is off by more than an order of magnitude.

If you let that feeling ride and re-buy the hedge every month the fear is loud, that is \$600 twelve times, or \$7,200 a year. On a \$50,000 account, \$7,200 is a 14.4% annual drag — paid to insure against an event whose real per-session odds are a fraction of one percent. You may still want *some* tail hedge; that is a legitimate strategy. But sizing it off yesterday's screen instead of the base rate turns prudent insurance into a recurring tax on your returns.

The one-sentence intuition: recency does not just feel bad, it quietly bills you every month you let it price your fear.

## 1. Availability in the wild: why the vivid crowds out the true

The reason availability is so hard to beat is that it is usually *useful*. In the world our brains evolved for, things that were easy to recall really were more common and more dangerous — the watering hole where a friend was taken by a crocodile genuinely was riskier. Ease of recall was a decent proxy for frequency when your data came from a small, personally-experienced sample. Markets break that proxy in two ways at once: the sample you personally experience is minuscule compared to the true distribution, and the media hands you vivid examples that have nothing to do with base rates.

Think about what is *available* to a trader after a crash. The memory is fresh, emotional, and rehearsed — you have told the story of that Tuesday to three people. Every financial news outlet is running crash retrospectives. Your feed is full of people who "called it." The base rate of another crash next month has not moved at all, but its availability has gone through the roof, so your felt probability of it soars. The reverse is just as damaging: after a long calm stretch, crashes become hard to recall, availability collapses, and your felt probability of danger sinks to zero right when leverage and complacency are highest. Availability makes you most afraid after the fall and least afraid before it — precisely backwards.

This is also why disasters are systematically mispriced right after they occur. Immediately following a plane crash, people drive instead of fly and die on the roads at higher rates; immediately following a market crash, people hoard cash and hedges and miss the recovery. The vividness of the last event overwrites the statistics of the long run. A trader's entire edge, in a sense, is the discipline to keep consulting the statistics while everyone else is consulting their most recent nightmare.

## 2. Recency and extrapolation: drawing the trend line to infinity

The most profitable-looking and most dangerous face of recency is *extrapolation* — taking the recent trend and mentally extending it forward forever. If the last several months went up, the future goes up; if the last several months went down, the future goes down. Extrapolation feels like pattern recognition. Often it is just recency in a nicer suit.

We are not guessing about this. In 2014, Robin Greenwood and Andrei Shleifer published ["Expectations of Returns and Expected Returns"](https://academic.oup.com/rfs/article-abstract/27/3/714/1580705) in the *Review of Financial Studies*. They gathered six independent measures of investor expectations of future stock returns — surveys of individuals, newsletters, CFOs, and more — spanning 1963 to 2011. The finding was stark and consistent across all six: investor expectations of future returns are strongly *positively* correlated with past returns and with the current level of the market. People are most optimistic after prices have already risen and most pessimistic after they have already fallen.

Here is the sting. Those same expectations are *negatively* correlated with the returns the market actually goes on to deliver. In plain English: the moments when the crowd is most sure the good times will continue are, on average, the moments just before they do not. Extrapolation is not merely unreliable; it is systematically anti-correlated with reality at the extremes. The recency-driven investor buys the most exposure at the top of the feeling and the least at the bottom.

#### Worked example: the base rate versus the last three observations

Suppose you trade a pullback-to-support setup that "always bounces." It has bounced the last three times you watched it, and that streak is screaming at you to load up. Now go to the boring number. Your journal has 200 instances of this exact setup. It bounced 116 times and failed 84 times — a 58% bounce rate. When it bounces, you make about \$400. When it fails, it fails hard, and you lose about \$1,200.

Compute the expected value per instance. A 58% chance of +\$400 contributes +\$232. A 42% chance of -\$1,200 contributes -\$504. Net expected value: \$232 minus \$504 equals -\$272 per instance. The setup that "always bounces" is, across 200 real instances, a losing trade — it bleeds \$272 every time you take it, on average. The last three bounces made it feel like a near-certainty; the base rate reveals a negative-expectancy trap where the rare failures are large enough to swamp the frequent small wins.

The one-sentence intuition: three bounces is a story, 200 instances is a number, and only the number pays.

## 3. The tyranny of the last trade, mechanism by mechanism

We defined the three faces of last-trade tyranny in the foundations. Now let us look at the machinery under each, because each has a named psychological driver you can learn to catch.

**Revenge after a loss** runs on *loss aversion*. In their 1979 work on prospect theory, Kahneman and Tversky found that losses hurt roughly twice as much as equivalent gains feel good — the pain-to-pleasure ratio is about two to one. A \$1,000 loss does not feel like the mirror image of a \$1,000 gain; it feels like roughly a \$2,000 gain's worth of pain. That asymmetry creates an overwhelming urge to erase the loss immediately, and the fastest apparent route is to bet bigger on the next thing. This is Leeson's engine, and it is a martingale — a doubling strategy that mathematically guarantees ruin if the streak runs long enough, which streaks always eventually do.

**Timidity after a loss** is the same loss aversion pointed inward. Having just felt the pain, you shrink from the very next opportunity to feel it again, so you skip the setup or size it down to nothing. The cruelty here is subtle: a real edge earns its money on a minority of large winners, and if you systematically shrink your size right after a loss, you will be smallest exactly when the next big winner arrives, because winners are not scheduled around your losses.

**Pressing after a win** runs on the *house money effect*, documented by Richard Thaler and Eric Johnson in 1990. After a win, people treat the profit as "the casino's money" rather than their own, and take wilder risks with it than they ever would with their starting capital. The win also produces a jolt of overconfidence — you feel like you have figured the market out — and the two combine into a fat, unplanned bet placed at the moment your judgment is most compromised.

There is a fourth, quieter cousin worth naming: the *disposition effect*, identified by Hersh Shefrin and Meir Statman in 1985 and confirmed in retail brokerage data. It is the tendency to sell winners too early and hold losers too long — to "take the sure gain" and "give the loser a chance to come back." It is recency and loss aversion braided together, and it is the single most common way ordinary investors turn a portfolio of decent picks into a portfolio of their worst ones.

#### Worked example: same setup, three sizes

Suppose your account is \$100,000 and your system risks 1% per trade — \$1,000 at your defined stop. Trade N just lost. The next signal that fires is an A+ setup identical to dozens you have traded before; its odds have nothing to do with the trade that just closed. Watch how the last result tries to resize it.

- **Revenge:** you risk \$3,000 (3%) to "make it back fast." If this one also loses — and independent setups do lose back-to-back all the time — you are down \$4,000 across the two trades instead of \$2,000. You doubled your drawdown to soothe a feeling.
- **Timidity:** you risk \$300 (0.3%) because you are gun-shy. If this is the +3R winner your edge depends on, you make \$900 instead of \$3,000. You shrank the exact trade your system needs to be full-sized.
- **Discipline:** you risk \$1,000, unchanged. The setup's expected value is whatever it always was, and you collect your fair share of it.

The one-sentence intuition: the market did not remember your last trade, so do not let your position size remember it either.

## 4. Same edge, opposite outcomes: how sizing noise kills a winner

Here is the fact that should frighten a systematic trader most: you can have a genuinely positive edge and still go broke, purely from letting recency drive your size. The edge lives in the odds of each trade; the ruin lives in the sizing rule laid over the top of them. Get the sizing rule wrong and it does not matter how good the signals are.

![Two equity curves from the same 20 signals and the same 55% edge: the base-rate sizer with constant 1% risk compounds steadily upward, while the recency sizer who doubles after losses climbs briefly then falls off a cliff when one losing streak arrives.](/imgs/blogs/recency-availability-and-the-tyranny-of-the-last-trade-5.webp)

Both traders in that picture take the identical twenty signals with the identical edge. The green trader risks a constant 1% and rides the edge to a steadily higher account. The red trader sizes off the last result — doubling after a loss to get it back, halving after a win out of timidity — and for a while it even looks clever, because the doubling recovers small losing runs and the account drifts up with the crowd. Then one ordinary losing streak arrives, the doubling turns a survivable dip into a cliff, and the account that started even with the winner ends at a fraction of it.

#### Worked example: the doubling cliff

Take the representative run drawn above, where the disciplined account ends near \$118,000 and the recency account ends near \$42,000, and let us isolate exactly where the cliff comes from. The killer is a single losing streak. In twenty trades at a 55% win rate, a run of four straight losses is not rare — it shows up more often than most traders expect. Now apply the doubling rule to that run, starting from a \$1,000 base risk:

- Loss 1: risk \$1,000, down \$1,000.
- Loss 2: double to \$2,000, down \$3,000 total.
- Loss 3: double to \$4,000, down \$7,000 total.
- Loss 4: double to \$8,000, down \$15,000 total.

Four losses under the doubling rule cost \$15,000. The same four losses at constant 1% risk cost \$4,000. The doubling trader lost nearly four times as much from the identical streak, and did it while holding a positive edge. The edge was earning pennies per trade; the doubling was risking the whole jar on every recovery attempt.

The one-sentence intuition: a real edge plus a doubling rule is a time bomb, because the edge earns slowly while the doubling bets the account on never having a normal losing streak.

## 5. Buying insurance at the worst possible price

One of the cleanest and most costly signatures of recency shows up in how traders buy protection. The price of downside insurance — the premium on put options, summarized market-wide by the VIX volatility index — is not constant. It is cheap when markets are calm and nobody wants it, and it explodes to its most expensive exactly after a crash, when recency has convinced everyone that another one is imminent.

![A timeline of the cost of protection through a crash: the VIX is a flat, cheap line in the calm, spikes above 80 at the panic, then decays; the annotations mark that insurance is cheapest when unwanted and most expensive right after the risk it hedges has largely passed.](/imgs/blogs/recency-availability-and-the-tyranny-of-the-last-trade-6.webp)

The two record highs on that chart are real. The VIX closed at 80.86 on November 20, 2008, and then set an all-time closing record of 82.69 on March 16, 2020 ([Cboe VIX all-time highs](https://www.macroption.com/vix-all-time-high/)). Both peaks came *after* the market had already fallen hard — the protection was at its most expensive precisely when much of the damage it insures against was already done. In the calm before, when the VIX sat in the mid-teens and protection was cheap, almost nobody wanted it, because no recent crash was available to make the danger feel real.

#### Worked example: the same hedge at VIX 15 and VIX 80

Suppose two traders each want about three months of downside protection on a \$100,000 portfolio, bought with index puts. Option premiums scale with implied volatility, so the price of the very same protection moves with the VIX.

- **Trader Calm** buys when the VIX is around 15. The premium runs on the order of \$1,500 for the quarter.
- **Trader Scared** waits until after the crash, when the VIX is around 80. The same protection now costs on the order of \$8,000 — roughly five times as much, because implied volatility, and therefore option premium, is roughly five times higher.

Same insurance, roughly 5x the price, and Trader Scared is buying it after most of the move it protects against has already happened. Trader Scared is not managing risk; Trader Scared is paying a premium for the feeling of having done something, at the exact moment the feeling is most expensive.

The one-sentence intuition: fear is the most expensive time to buy the very thing that fear is selling.

## 6. Availability cascades: when the whole market overweights the last thing

So far this has been about the individual. But recency does not stay in one skull — it spreads, and when it spreads it becomes a market force. In 1999, Timur Kuran and Cass Sunstein described the mechanism in ["Availability Cascades and Risk Regulation"](https://chicagounbound.uchicago.edu/journal_articles/8308/). An availability cascade is a self-reinforcing loop: an expressed perception (say, "a crash is coming") triggers a chain reaction in which the perception becomes more plausible simply because it is being repeated, which makes more people repeat it, which makes it more available still.

Two forces drive the cascade. The first is informational — you take other people's apparent beliefs as evidence, so if everyone is talking about the crash, you conclude they must know something. The second is reputational — you go along with the prevailing view to avoid looking foolish or out of step. Social feeds and financial media are availability-cascade machines by design: they amplify whatever just happened, because whatever just happened is what generates clicks and engagement. The last event is not merely available in your own memory; it is being broadcast back at you from every screen, manufacturing an artificial base rate out of pure repetition.

For a trader, this is why sentiment reaches its extremes exactly when it is least useful. The cascade tops out when the last move has been repeated so many times that no other future seems possible — which, per Greenwood and Shleifer, is roughly when the future is about to be different. Recognizing a cascade for what it is — availability wearing the costume of consensus — is one of the most valuable skills a trader can build, because it lets you fade the crowd at precisely the moments the crowd is most confidently wrong.

## What it looks like at the screen

Theory is easy to nod along to and hard to catch in the act. So here is what recency and last-trade tyranny actually feel like in real time, in the first person, so you can recognize the tells while they are happening rather than in the post-mortem.

You take a loss, and before the ticket even clears you feel your hand reaching to size up the next entry — not because the next setup is better, but because you want the red number gone *now*. That is revenge, and the giveaway is that your finger moved before your analysis did.

You take a loss and the opposite happens: the next A+ setup prints, clean and obvious, and you sit on your hands. You "want to see it confirm." You are not being patient; you are flinching, and the tell is that you would have taken this exact setup without hesitation two trades ago.

You catch yourself refreshing the chart of the one thing that just moved hard, over and over, while eight other instruments sit ignored. Your attention has been captured by availability — the mover is vivid, so it feels important, so it eats your focus even when the opportunity is elsewhere.

A phrase forms in your head, fully assembled, with total confidence: *"it always bounces here,"* or *"this thing only goes up,"* or *"we're definitely gapping down tomorrow."* Any sentence with "always," "never," or "definitely" built from the last few sessions is recency talking. The base rate never speaks in absolutes.

You widen a stop because the last trade would have worked if you had "just given it room," or you tighten one because the last trade gave back an open profit. You are re-fighting the last war, adjusting this trade's risk to what would have fixed the previous trade — a trade that is over and shares no P&L with this one.

And the biggest one: you glance at your daily P&L and feel it reach for the mouse. Green, and you feel entitled to a bigger swing. Red, and you feel you owe the account a comeback. The moment your account balance is an input to your next position size, the tyranny of the last trade has you, and the number on the screen is trading you instead of the other way around.

## Common misconceptions

**"Weighting recent data more heavily is just being adaptive."** Adaptivity is a principled update — you change your estimate by a defensible amount when new evidence arrives, in proportion to how much that evidence should move a base rate built from many observations. Recency is the opposite: it lets a single new data point, n equals one, overwrite a distribution built from hundreds. Real adaptation nudges the base rate; recency throws it away.

**"After three losses in a row, I'm due for a win."** This is the gambler's fallacy — the belief that independent events owe you a correction. If your trades are independent, three losses tell you nothing about the fourth. Its identical twin is the hot-hand version: "I've won three, I'm on fire, press it." Both take a recent streak and invent a trend from randomness. Gilovich, Vallone, and Tversky's 1985 study of basketball shooting showed how readily people see "hot hands" in what is statistically just noise.

**"Buying protection after a scare is responsible risk management."** Timing is everything, and this timing is backwards. You are buying insurance at the top of the price spike, after the risk it hedges has largely already materialized. Responsible risk management buys protection when it is cheap and unwanted — which, by construction, is exactly when no recent event makes it feel necessary.

**"The trend is my friend, so extrapolating it is the smart move."** Trends are real and trend-following can work, but naive extrapolation is not trend-following; it is assuming the trend is permanent. Greenwood and Shleifer showed that crowd expectations built by extrapolation are highest right before returns are lowest. Following a trend with a plan for the turn is a strategy; extrapolating it to infinity is recency with a chart.

**"One revenge trade won't hurt me."** It only takes one, if it lands on the wrong streak. The doubling worked example showed a single four-loss run turning a \$4,000 drawdown into \$15,000. Nick Leeson also thought each doubling would be the one that worked. The problem with betting the account on never having a normal losing streak is that normal losing streaks are, by definition, normal.

**"If I feel it this strongly, it must be likely."** The strength of a feeling tracks the vividness and recency of the memory behind it, not the probability of the event. A crash you lived through last week feels far more likely than one you read about in a table, even if the table says they happen at the same rate. Availability is precisely the error of reading emotional intensity as statistical frequency.

## How it shows up in real markets

### 1. The permabears who sat out the recovery

The clearest monument to recency in a generation is the cohort of investors who, having been scarred by 2008, stayed defensive through the entire recovery that followed. The scale of what they missed is a matter of record. The S&P 500 closed at its bear-market low of 676.53 on March 9, 2009 (it had touched an intraday 666.79 on March 6), down about 57% from its October 2007 peak of 1,565.15 ([S&P 500 closing milestones](https://en.wikipedia.org/wiki/Closing_milestones_of_the_S%26P_500)). From that low it climbed, with interruptions, to a closing high of 3,386.15 on February 19, 2020 — roughly five times the low, over about eleven years. The 2008 crash was so vivid and so available that for many it overwrote the base rate that markets recover, and the availability of the last disaster cost them the entire next expansion. Being right about the crash and wrong about everything after it is recency in its purest form.

### 2. The volatility spike nobody could buy cheaply

The put-buying panics of 2008 and 2020 are the market pricing recency in real time. Demand for downside protection surged to its maximum after the declines were largely over, driving the VIX to its two highest closes on record — 80.86 on November 20, 2008, and 82.69 on March 16, 2020. Tail-hedging strategies that had been quietly, cheaply in place before the fall paid off enormously; the crowd that rushed in afterward bought the same protection at five to six times the calm-market price. The lesson repeats every cycle: the insurance is cheap exactly when the recent past makes it feel unnecessary, and dear exactly when the recent past makes it feel essential.

### 3. Extrapolation at the top

The Greenwood-Shleifer finding is not a mere theory; it has a body count of famous tops. Near the dot-com peak in early 2000 and again near the 2007 high, surveyed investor optimism about future returns reached extremes — right before two of the worst drawdowns in modern history. The crowd extrapolated the boom, grew most confident at the moment of maximum danger, and was most exposed when the turn came. Recency at a top does not feel like a bias; it feels like obvious common sense that the thing that has been working will keep working.

### 4. Nick Leeson and the death of Barings

Return to where we started. Leeson's account 88888 was a machine for exactly the tyranny of the last trade: every loss triggered a larger bet that it would come back, because his recent experience of small recoveries had taught him that doubling worked. The Kobe earthquake of January 1995 sent the Nikkei through the floor while he was doubling into it, and the losses reached roughly £827 million — twice the bank's available capital. Barings, founded in 1762, was declared insolvent on February 26, 1995. One trader, one doubling rule, one refusal to let a losing streak be a losing streak, and a 233-year-old institution was gone. The tyranny of the last trade does not usually cost you a bank. But it is the same mechanism, scaled down to your account, running every day.

## The drill: the base-rate anchor and the between-trades reset

Naming a bias does not cure it. Recency is not a knowledge problem — you can know all of this cold and still feel your hand reach to size up after a loss. It is a habit problem, and it yields only to a habit. Here is the protocol, in two parts.

**Part one: the base-rate anchor.** Before every trade, and especially when a setup feels like a certainty, force one question to the front: *what is the long-run frequency here?* Not "what just happened," not "what does this feel like" — what does the base rate say across many instances? Concretely:

- Keep the number where you can see it. For each setup you trade, write its historical win rate and its average win and average loss at the top of your notes, from your journal or a backtest of at least a few dozen instances. When "it always bounces" shows up, you are looking at "58%, +\$400 / -\$1,200" instead.
- Ask "N of how many?" When a recent event is driving your conviction, say out loud how many observations it is built on. "The last three bounced" becomes "three out of the two hundred I have on record." Spoken that way, the streak shrinks back to its real size.
- Pre-commit your size. Decide the position size from the system *before* you enter, based on the setup's base rate and your fixed risk fraction, and treat it as locked. A size chosen before the trade cannot be edited by the trade before it.

**Part two: the between-trades reset.** The tyranny of the last trade lives in the seconds between closing one position and opening the next. Put a ritual in that gap so the emotional and P&L residue of trade N cannot leak into trade N+1.

![The between-trades reset as a five-step pipeline: close the trade, log the outcome and look away, do a 60-second physical reset, ask the base rate, then size from the system regardless of what the last trade did.](/imgs/blogs/recency-availability-and-the-tyranny-of-the-last-trade-7.webp)

Run the five steps in order, every time. Close the trade and accept that the P&L is now a fact in the past. Log the outcome in your journal, then deliberately stop staring at the number. Do a genuine physical reset — stand up, breathe, sixty seconds off the screen — because the urge to revenge-trade rides on a physiological arousal state that fades if you let it. Ask the base-rate question about the next setup. Then size from the system, risking your fixed fraction whatever trade N just did, so that trade N+1 inherits nothing from trade N but the capital in the account.

If you want a single sentence to tape to your monitor, make it this one, and read it after every close:

> The last trade is over. Its outcome is not evidence about the next one. Size from the base rate, not from the memory.

That is the whole discipline. It will not make the urges disappear — you will still feel the reach for a bigger size after a loss and the itch to press after a win. It just puts a fixed, boring procedure between the urge and the order, and over hundreds of trades that procedure is worth more than any signal.

## When this matters to you

You do not have to be a professional trading a seven-figure book for this to touch your money. Recency runs your 401(k) contributions when you pull back from stocks after a scary quarter and pile in after a great year. It runs your decision to buy travel insurance the week after a friend's flight was cancelled and skip it the rest of the year. It runs the urge to chase whatever asset just tripled and to swear off whatever just halved. Anywhere you are estimating how likely something is, your brain is quietly substituting how easily you can remember it — and the last thing that happened is always the easiest to remember.

The fix is not to become a robot who ignores new information. Genuinely new evidence *should* update your view. The fix is to update by the right amount: to weigh the last observation as one vote among the base rate's many, rather than as the only vote that counts. Ask "what is the long-run frequency?" before "what just happened?" Put a reset between your trades. And when a feeling of certainty arrives built entirely from the recent past, treat that certainty itself as the signal to slow down.

This is educational, not individualized advice — your situation, risk tolerance, and strategy are yours to decide. But the underlying skill transfers everywhere: the trader who can keep consulting the base rate while the crowd consults its most recent nightmare is holding the one edge that never stops working. For more on why grading yourself by outcomes reinforces exactly these mistakes, see [process versus outcome and the trap of resulting](/blog/trading/trading-psychology/process-versus-outcome-and-the-trap-of-resulting); for the emotions that power the revenge-and-timidity cycle, see [fear, greed, hope, and regret](/blog/trading/trading-psychology/fear-greed-hope-and-regret-the-four-emotions); and for where recency sits among the other traps, see [the cognitive bias map for traders](/blog/trading/trading-psychology/the-cognitive-bias-map-for-traders).

## Sources & further reading

- Amos Tversky and Daniel Kahneman, ["Availability: A heuristic for judging frequency and probability,"](https://www.sciencedirect.com/science/article/abs/pii/0010028573900339) *Cognitive Psychology* 5(2), 1973 — the founding paper on the availability heuristic.
- Daniel Kahneman and Amos Tversky, "Prospect Theory: An Analysis of Decision under Risk," *Econometrica* 47(2), 1979 — loss aversion and the roughly two-to-one pain-to-pleasure ratio.
- Bennet B. Murdock, "The serial position effect of free recall," *Journal of Experimental Psychology* 64(5), 1962 — the recency effect in memory.
- Richard Thaler and Eric Johnson, "Gambling with the House Money and Trying to Break Even," *Management Science* 36(6), 1990 — the house-money effect.
- Hersh Shefrin and Meir Statman, "The Disposition to Sell Winners Too Early and Ride Losers Too Long," *Journal of Finance* 40(3), 1985 — the disposition effect.
- Thomas Gilovich, Robert Vallone, and Amos Tversky, "The Hot Hand in Basketball," *Cognitive Psychology* 17(3), 1985 — seeing streaks in randomness.
- Robin Greenwood and Andrei Shleifer, ["Expectations of Returns and Expected Returns,"](https://academic.oup.com/rfs/article-abstract/27/3/714/1580705) *Review of Financial Studies* 27(3), 2014 — extrapolative expectations and their negative correlation with future returns.
- Timur Kuran and Cass Sunstein, ["Availability Cascades and Risk Regulation,"](https://chicagounbound.uchicago.edu/journal_articles/8308/) *Stanford Law Review* 51(4), 1999 — how recency spreads socially.
- ["Closing milestones of the S&P 500,"](https://en.wikipedia.org/wiki/Closing_milestones_of_the_S%26P_500) Wikipedia — the 676.53 low (March 9, 2009) and 3,386.15 high (February 19, 2020).
- ["VIX all-time highs,"](https://www.macroption.com/vix-all-time-high/) Macroption — the record closes of 80.86 (November 20, 2008) and 82.69 (March 16, 2020).
- ["Bankruptcy of Barings Bank,"](https://www.britannica.com/event/bankruptcy-of-Barings-Bank) Britannica — Nick Leeson, account 88888, and the roughly £827 million collapse.

---
title: "Consensus, Expectations, and 'Priced In': How the Market Sets Its Baseline"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "How the market builds a consensus from surveys, futures and options before any release, why only the surprise versus that consensus moves price, and how to tell when something is fully priced in."
tags: ["event-trading", "macro", "consensus", "priced-in", "fed-funds-futures", "cme-fedwatch", "expected-move", "economic-surprise-index", "fomc", "cpi"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Before any macro release, the market builds a *consensus* — the outcome everyone expects — and trades to it. That consensus is what "priced in" means, and once it is in the price, the event itself barely moves anything. Only the **surprise** — the gap between the actual number and the consensus — moves price.
>
> - Consensus is assembled from four readable sources: **economist surveys** (the Bloomberg/Reuters median and range), **market-implied probabilities** (Fed funds futures, read off as CME FedWatch odds), the **options-implied expected move** (the at-the-money straddle), and the informal **whisper number**.
> - The reaction map is simple: actual = consensus → near-zero move; actual ≠ consensus → repricing, and the size of the move scales with the size of the surprise (and the regime).
> - The trade: if an outcome is *fully priced* (futures show 90%+ one way), there is no edge in betting on it — you fade the knee-jerk or trade the second-order detail. If it is *not* priced, the surprise is the whole move and you ride the repricing.
> - The one number to remember: in December 2015 the first Fed hike in nine years was **~100% priced** by Fed funds futures — and the S&P 500 *rose* on the decision. The decision was a non-event; the path guidance was the trade.

## A meeting everyone already knew the answer to

On 16 December 2015, the Federal Reserve raised interest rates for the first time in nearly a decade. After seven years pinned at zero, after the 2008 crisis and the long crawl back, the central bank finally lifted its target range by a quarter point. By any ordinary intuition this should have been a seismic event — the end of the easy-money era, the moment the punch bowl started to drain. Markets crash on news like that, right?

They did not. The S&P 500 *rose* about 1.5% on the day of the decision. Treasury yields barely twitched. The dollar was roughly flat. A historic policy turn, and the tape shrugged. Why?

Because by the morning of the meeting, Fed funds futures — the contracts that let traders bet on the future path of the Fed's policy rate — implied something close to a **100% probability** that the hike was coming. Every desk on Wall Street had read the same futures curve, heard the same speeches, and adjusted their books weeks in advance. The hike was, in the trader's phrase, *fully priced in*. There was no surprise left in the actual decision to move anything. The few things that *did* move — a small relief rally in stocks — came from the *guidance*: Janet Yellen signalled a gradual, data-dependent path of future hikes rather than a fast march higher, and that nuance was the only part of the event that wasn't already in the price.

This is the single most important and most counterintuitive idea in event trading, and it is the entire subject of this post. Markets do not react to *news*. They react to news *minus what they already expected*. The expected part is in the price before the bell. Only the difference — the surprise — has anywhere to go. Master that, and half the mistakes new traders make around economic releases simply disappear. This post is the foundation the rest of the series builds on; for the calendar of which releases matter and when, see [the macro calendar](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi).

![How surveys, futures and options fuse into a consensus the price already holds, with the surprise moving it](/imgs/blogs/consensus-expectations-and-priced-in-1.png)

## Foundations: what "priced in" really means

Let us slow all the way down and define every word, because the whole edifice rests on these few terms.

A **forecast** is one person's guess at what a number will be. An economist at a bank publishes a forecast for next month's inflation reading. So does a hedge fund analyst, so does a model. There are dozens, hundreds of these forecasts floating around for any major release.

The **consensus** is the aggregate of all those forecasts — usually the *median*. When a financial data provider says "economists expect CPI of +0.3% month-on-month," they are reporting the median of a poll of professional forecasters. The consensus is the market's collective best guess at the answer *before the answer is known*. It is a single number (the median) plus, if you look closely, a *range* (the highest and lowest forecasts) that tells you how much disagreement there is.

**Expectations**, more broadly, are everything the market believes about an upcoming outcome — not just the headline number but the probability of each possible result. For a Fed meeting, the expectation isn't a single number; it's a probability distribution: "70% chance of a hold, 30% chance of a cut." Expectations live in prices, not in surveys, and we'll spend most of this post learning to read them out of market instruments.

**Priced in** (or "in the price") is the key phrase. A piece of information is *priced in* when the market has already adjusted asset prices to reflect it. If everyone expects a rate cut and has already bought the bonds and stocks that benefit from a cut, then when the cut actually happens, there is nothing left to buy — the move already occurred during the days and weeks of *anticipation*. The decision itself is a non-event. "It's priced in" is short for "the market already expected this and already moved for it."

The **surprise** is the gap between what actually happened and the consensus:

```
surprise = actual − consensus
```

A positive surprise on inflation means the number came in *hotter* (higher) than expected. A negative surprise means it came in *cooler*. The sign and size of the surprise — not the level of the number itself — is what drives the reaction. An 8.3% inflation print is terrifying in the abstract, but if the consensus was 8.3%, the market does almost nothing. If the consensus was 8.1% and the actual was 8.3%, that *+0.2 percentage-point surprise* is the trade.

Finally, the **whisper number** is the unofficial expectation that floats around trading desks — the number the market is *really* positioned for, which can differ from the published survey consensus. We'll come back to it; for now, just know that the "true" consensus the price reflects is sometimes a notch away from the headline poll.

It helps to nail down *why* expectation lives in price at all. Markets are forward-looking auctions. The price of a stock or a bond is not a record of what has happened; it is the market's current best estimate of what *will* happen, discounted to today. So when a future event is highly likely, that likelihood is already embedded in today's price — buyers have bid the price up (or down) to reflect it. A release confirms or refutes that estimate. Confirmation adds no information (the estimate was right) and the price holds; refutation adds information (the estimate was wrong) and the price jumps to a new estimate. This is why a "good" number can sell off and a "bad" number can rally: the price already contained an estimate, and what matters is whether reality beat or missed that estimate, not whether reality was objectively good or bad.

A useful way to keep this straight is to separate three distinct quantities that beginners constantly conflate: the **level** (the raw number — 8.3% inflation), the **consensus** (what the market expected — 8.1%), and the **surprise** (the gap — +0.2pp). The level is what makes headlines. The consensus is what's in the price. The surprise is what trades. A whole career of event-trading mistakes comes from reacting to the level when only the surprise matters. If you take one habit away from this post, make it this: every time a number crosses the wire, your first instinct should not be "is this number high or low?" but "is this number higher or lower than what everyone expected?"

The structure of the rest of this post follows the four ways the market builds and reveals its consensus — surveys, futures, options, and surprise indices — and then turns to the practical question every event trader asks before a release: *is this priced in, and what does that mean for my position?*

## 1. Survey consensus: who polls, the median, and the range

The most visible form of consensus is the economist survey. The big financial data vendors — Bloomberg, Reuters, Dow Jones, FactSet — each poll a panel of professional forecasters ahead of every major release. For a US Consumer Price Index report, dozens of bank and research-house economists submit their forecasts for headline and core inflation. The vendor publishes the **median** as "the consensus" and usually shows the **range** (the high and low estimates) and sometimes the full distribution.

Two numbers matter here, and most beginners only look at one.

The **median** is the level the market is benchmarking against. When a CPI report crosses the wire, the very first thing the algorithms and traders do is compare the actual headline to the survey median. Above the median = hot surprise; below = cool surprise. The headline reaction is keyed off this comparison in the first milliseconds.

The **range** tells you how *contested* the number is, and that controls how *big* the reaction can be. If 60 economists all cluster between +0.2% and +0.4%, the market is confident and a print inside that band barely moves things. But if forecasts are scattered from −0.1% to +0.6%, nobody really knows, the consensus is fragile, and an out-of-range print can unleash a violent move because positioning was uncertain to begin with. A wide range is a coiled spring; a tight range is a settled question.

There is a subtlety worth internalising. The survey median is a *forecast of the data*, but the price reflects a forecast of *how the data will affect policy and assets*. These usually point the same way, but not always. A CPI number can land exactly on the survey median and still move markets if the *internals* (the components beneath the headline) tell a different story than the headline — say, if shelter inflation cooled but the headline was propped up by a one-off energy spike. The survey is the headline benchmark; the reaction function is richer than that, which is why we cross-link the deeper mechanics in [the macro calendar post](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi).

### The whisper number: when the real consensus differs from the poll

The survey median has a known weakness: it can be *stale* and it can be *gamed*. Stale, because economists submit forecasts a few days before the release, and the market keeps learning right up to the morning of the print — a fresh data point or a leaked detail can shift the *real* expectation after the survey closed. Gamed, because some forecasters anchor conservatively to avoid looking foolish, so the published median can lag where the smart money actually sits.

The **whisper number** is the market's informal answer to this — the level traders are *really* positioned for, passed around desks, financial Twitter, and prediction markets, rather than the official poll. The whisper can differ from the survey median by a meaningful margin, and when it does, the *whisper* is closer to what's in the price. The practical consequence: a print can "beat the survey consensus" and *still sell off*, because it missed the higher whisper. Earnings season is full of this — a company beats the published analyst consensus and the stock drops, because everyone had quietly expected an even bigger beat. The same dynamic operates on macro prints when sentiment runs ahead of the formal poll.

For an event trader, the takeaway is that "the consensus" is not a single canonical number you can read off one screen. It is the *price-implied* expectation, and your job is to triangulate it from the survey median, the futures-implied odds, the options-implied move, and the whisper — and to weight the market-based signals (futures, options) more heavily than the survey when they disagree, because those reflect real money at risk rather than analysts' opinions.

#### Worked example: beating the survey but missing the whisper

The published CPI survey median is +0.3% month-on-month, but the whisper on desks is a cooler +0.2% — traders have quietly positioned for a soft print after a run of disinflation headlines. The actual prints +0.25%. Against the *survey* (+0.3%) this is a cool, risk-on result; against the *whisper* (+0.2%) it is a hot disappointment. You hold a \$20,000 long into the release expecting the survey-relative rally, but the market sells off −0.8% because it had priced the whisper: −0.8% × \$20,000 = **−\$160**. You read the headline consensus correctly and still lost money, because the price reflected the whisper, not the poll. **The number that matters is the one the price is actually leaning on — find it before you size.**

#### Worked example: a survey-aligned print does almost nothing

Suppose the CPI survey median is +0.3% month-on-month and the actual prints +0.3%. You hold a \$50,000 S&P 500 position into the release. The number lands *on consensus*: surprise = +0.3% − 0.3% = 0.0pp. The market has nothing to reprice, so the index might wobble ±0.1% in the first minute and settle flat. On your \$50,000, a 0.1% wobble is just \$50 either way — noise. Now say the actual instead prints +0.5%, a +0.2pp hot surprise in an inflation-fearful regime, and the S&P falls 1.5% on the day: that is a −1.5% × \$50,000 = −\$750 hit. The release is the *same event*; the only difference is whether the number matched the consensus. **The level of inflation didn't change your P&L — the surprise versus consensus did.**

## 2. Market-implied consensus: Fed funds futures and CME FedWatch

Surveys tell you what economists *say*. Markets tell you what traders are actually *betting*, with real money, and that is a stronger signal. The cleanest example is the expectation for the Fed's policy rate, which the market expresses through **Fed funds futures**.

A Fed funds futures contract settles based on the average effective federal funds rate over a calendar month. Without drowning in the mechanics: the price of the contract for a given month implies the *average policy rate the market expects for that month*. Because Fed meetings happen on known dates, you can back out, from the futures prices around a meeting, the market's implied probability that the Fed will hold, cut, or hike at that specific meeting. This is exactly what the CME's **FedWatch** tool does — it reads the futures and presents the implied odds as a simple bar: e.g. "85% probability of a hold, 15% probability of a 25bp cut" at the next meeting.

This is the single most useful gauge an event trader has, because it converts a fuzzy expectation into a hard, tradeable probability. When a commentator says "the cut is priced in," they usually mean FedWatch shows a high probability (say 90%+) of a cut. When they say "a hike is a coin flip," FedWatch is near 50/50.

How does a futures price become a probability? The arithmetic is more approachable than it sounds. Suppose the current policy rate is 5.00% and the next meeting could either hold (rate stays 5.00%) or cut 25bp (rate goes to 4.75%). The Fed funds futures for the meeting month imply an *average* expected rate — say 4.92%. The market's implied rate is a probability-weighted blend of the two outcomes: `4.92% = p × 4.75% + (1 − p) × 5.00%`, where `p` is the probability of a cut. Solving, the implied rate sits 0.08% below the no-cut level out of a 0.25% total possible move, so `p = 0.08 / 0.25 ≈ 32%`. FedWatch does exactly this (with month-fraction adjustments for the meeting date) across every possible outcome and reports the resulting odds. You don't have to do the algebra yourself — but knowing that the probability is *just the futures-implied rate's position between the possible outcomes* demystifies the whole tool. The bar is not a poll; it is real money in the futures market, translated into odds.

![CME FedWatch probability bar showing 85 percent hold flipping to 70 percent cut after a surprise](/imgs/blogs/consensus-expectations-and-priced-in-4.png)

Here is the crucial reading skill. **A high probability means low reaction-potential for that outcome and high reaction-potential for the other one.** If the market is 85% priced for a hold, then an actual hold delivers a tiny surprise (the market was *mostly* right) and prices barely move. But a *cut* — the 15% outcome — would be a genuine shock, and the repricing would be violent, because everyone positioned for the hold has to scramble. Probability and payoff are mirror images: the more certain an outcome, the less it pays if it happens and the more it costs if it doesn't.

This is why you must always read FedWatch as a *distribution*, not a prediction. "85% hold" is not "the Fed will hold." It is "if the Fed holds, ~nothing happens; if the Fed cuts, brace for a big move." Your trade is structured around the *deviation*, not the base case. For the deeper mechanics of how the Fed actually sets the rate the futures are pricing, see [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates), and for the path-pricing logic across a whole cycle, [terminal rate and rate-cut cycles](/blog/trading/macro-trading/terminal-rate-and-rate-cut-cycles-pricing-the-path).

The Fed funds path of 2019–2024 is a perfect illustration of consensus-building in action. Every step the Fed took — the 2019 mini-cuts, the COVID crash to zero, the furious 2022 hiking cycle, the first cut in 2024 — was *anticipated* by the futures curve before it happened. The market wasn't surprised by the destination; it spent every meeting cycle revising its estimate of when the next step would come and by how much.

![Step chart of the Fed funds target upper bound from 2019 to 2024 with liftoff and first-cut annotated](/imgs/blogs/consensus-expectations-and-priced-in-3.png)

#### Worked example: reading FedWatch on a \$50,000 equity book

You run a \$50,000 equity book into an FOMC meeting. FedWatch shows **85% hold, 15% cut**. Two scenarios:

- **The priced-in hold happens.** Surprise is small (the market was 85% right), so equities barely react — call it a 0.0–0.2% move. On \$50,000 that is roughly \$0 to \$100. You earned almost nothing for holding through the event; the outcome was already in the price.
- **The unpriced cut happens.** This is the 15% tail. A surprise easing in a market positioned for a hold could rally equities 1.5–2.0% as rate-cut beneficiaries get repriced. A +1.8% move on \$50,000 is +1.8% × \$50,000 = +\$900. The *same instrument*, the *same \$50,000*, but the payoff lives entirely in the surprise.

**The lesson in dollars: betting \$50,000 on the 85% outcome earns you ~\$0; the money is in correctly anticipating the 15% surprise — or in not being run over by it.**

### From probability to position: the priced-in ladder

Here is what "pricing in" actually looks like over time, and why the event itself is so often a let-down. Expectations don't snap from 0% to 100% on the day of the meeting. They *climb*, meeting by meeting, data point by data point, and the position is built on the way up.

![Priced-in ladder showing the market pre-positioning as probability climbs from twenty to ninety percent](/imgs/blogs/consensus-expectations-and-priced-in-2.png)

Say the market starts the cycle expecting the Fed to hold. The implied probability of a cut is 20%. Then the labour market softens, inflation cools, and over a few weeks the cut probability climbs to 50%, then 70%, then 90% the day before the meeting. At *each* of those steps, traders who became more confident of the cut bought the bonds and rate-sensitive stocks that benefit. By the time the meeting arrives and the cut is delivered, **the buying already happened** — spread across the rungs of the ladder. The decision pays almost nothing because the move was the *sum of the anticipatory steps*, not the event itself.

This is why veteran traders say "buy the rumour, sell the fact." The "rumour" is the rising probability; the "fact" is the confirmed event. Most of the money is made (or lost) on the climb, and the confirmation is frequently an anticlimax — or even a reversal, as positioned traders take profits the moment the uncertainty resolves.

## 3. Options-implied consensus: the expected move

Surveys and futures tell you the market's *central* expectation. Options tell you something different and complementary: **how big a move the market expects**, in either direction, around the event. This is the *expected move*, and it is read out of option prices.

An option is a contract whose value depends on volatility — on how much the underlying is expected to move. Around a scheduled event (a CPI print, an FOMC decision, an earnings report), option prices bake in an *event premium*: the market charges more for options because it knows a jump is coming. The simplest way to extract the market's expected move is to price the **at-the-money (ATM) straddle** — buying both a call and a put struck at the current price. The cost of that straddle is, roughly, the market's estimate of how far the underlying will travel by expiry, up or down.

The rule of thumb: **expected move ≈ ATM straddle price**, expressed as a percentage of the underlying. If the S&P 500 is at 5000 and the ATM straddle expiring just after the event costs \$60, the market is pricing an expected move of about 60 / 5000 = **1.2%** in either direction. (A more formal estimate uses implied volatility and time to expiry, expected move ≈ S × IV × √(t/365), but the straddle shortcut is what traders actually eyeball.) This is a preview of a much deeper topic — the volatility ramp into an event and the "vol crush" after — which gets its own treatment in the volatility post of this series and, for the underlying theory, in [options theory](/blog/trading/quantitative-finance/options-theory) and the [volatility surface](/blog/trading/quantitative-finance/volatility-surface).

Why does the expected move matter for our "priced in" question? Because it tells you what the market considers a *normal* reaction versus a *surprise*. If the expected move is ±1.2% and the actual reaction is +0.4%, the event under-delivered relative to what options priced — the "surprise" was small and the options buyers lost money on vol crush. If the actual reaction is +3.0%, it blew through the expected move — a genuine surprise that the options market under-estimated. The expected move is the consensus on *magnitude*; the actual range traversed is the *surprise on magnitude*.

There is a second, subtler reason the expected move matters: it has its own life cycle around the event, and that life cycle is itself a "priced in" story. In the days leading up to a scheduled release, implied volatility *ramps* — option buyers bid up protection because they know a jump is coming, and the event premium builds into the price. Then, the instant the number is out and the uncertainty resolves, implied volatility *collapses* — the so-called **vol crush**. This is why simply buying options before an event is not a free lunch even if you correctly predict a big move: you paid the inflated, ramped-up premium, and unless the *actual* move exceeds the *priced* expected move, the vol crush eats your profit. The expected move is, in this sense, the "priced in" amount of volatility — and to make money owning the straddle, the event has to surprise *beyond* it. A trader who buys a \$60 straddle and gets a move worth exactly \$60 of intrinsic value breaks even before costs; the edge requires a *surprise on magnitude*, not merely a move. This vol-ramp-then-crush pattern is the entire subject of the volatility post in this series.

#### Worked example: the options-implied expected move in dollars

The S&P 500 sits at 5000 and the ATM straddle expiring just after CPI costs \$60. Expected move = 60 / 5000 = **1.2%**, so the options market is pricing roughly ±1.2% around the print.

Translate that to a position. You hold \$5,000 of an S&P 500 index fund. A ±1.2% expected move means the options market expects your position to swing by about ±1.2% × \$5,000 = **±\$60** on the event. If the actual reaction is a tame +0.4%, your position gains 0.4% × \$5,000 = +\$20 — *less* than the expected move, meaning the event was a relative dud. If CPI surprises hot and the S&P drops 2.0%, you lose 2.0% × \$5,000 = −\$100, blowing past the −\$60 the options priced. **The expected move is the market's pre-paid estimate of your event-day swing; a reaction inside it is "as priced," a reaction outside it is the real surprise.** (For context, the same method on Bitcoin — spot 60,000, straddle \$2,400 — implies a far larger ±4.0% expected move, because crypto is structurally more volatile.)

## 4. The surprise = actual − consensus, and why only it moves price

We now arrive at the central mechanism. Let us state it as plainly as possible: **on a scheduled release, the price reaction is driven by the surprise, not the absolute number.** The consensus is already in the price; the surprise is the new information.

The cleanest empirical demonstration uses CPI days, the most reaction-heavy releases of the past few years. Look at three real prints and the S&P 500's same-day move:

- **13 September 2022** — August CPI came in at 8.3% versus an 8.1% consensus. A +0.2pp *hot* surprise. The S&P 500 fell **−4.32%**, its worst day since June 2020. The Nasdaq fell −5.16%, Bitcoin dropped roughly −9.4%, and the dollar jumped +1.4%.
- **10 November 2022** — October CPI printed 7.7% versus 7.9% consensus. A −0.2pp *cool* surprise. The S&P 500 rose **+5.54%**, one of its best days of the cycle. The Nasdaq surged +7.35%, the 10-year Treasury yield fell 28 basis points, and the dollar dropped −2.1%.
- **14 November 2023** — October CPI printed 3.2% versus 3.3% consensus. A −0.1pp *mild cool* surprise. The S&P rose +1.91%, and the rate-sensitive Russell 2000 small-cap index jumped +5.44%.

Notice what *isn't* driving these moves. The 8.3% print (hot) crushed stocks; the 7.7% print (cool) rocketed them. But 7.7% is *still very high inflation*. If the market reacted to the *level* of inflation, both should have been bad. Instead the +5.5% rally happened on a number that was, in absolute terms, alarming — because it came in *below consensus*. The sign of the surprise, not the level of the number, set the direction. The size of the surprise set the magnitude.

![Scatter of CPI surprise versus S&P 500 same-day move showing hot surprises fall and cool surprises rally](/imgs/blogs/consensus-expectations-and-priced-in-5.png)

The scatter makes it unmistakable. Plot the surprise on the horizontal axis (hot = positive, cool = negative) and the S&P's same-day move on the vertical axis, and the points line up: hot surprises in the lower-right (stocks down), cool surprises in the upper-left (stocks up). The relationship runs through the *surprise*, not the level. This is the empirical heart of event trading.

There is a deeper layer to "surprise" that separates intermediate traders from beginners: the surprise that matters is not always the *headline* surprise. Every major release has **internals** — the components beneath the top-line number — and the market sometimes reprices on those even when the headline lands on consensus. A CPI report has core versus headline, goods versus services, shelter, used cars, and the "supercore" services-ex-housing measure the Fed watches most closely. A jobs report has the headline payrolls number, but also the unemployment rate, average hourly earnings, the participation rate, and revisions to prior months. The headline can match consensus while an internal blows out — a benign headline CPI with a scorching shelter component, or an in-line payrolls number with a shock jump in wage growth — and the market trades the internal surprise. This is why two prints with identical headline surprises can produce different reactions: the *composition* of the surprise carries information the single number doesn't. When you read "the consensus," remember the market is benchmarking *several* numbers against *several* expectations simultaneously, and the dominant surprise is whichever component the current regime cares about most.

One vital caveat, which the series hammers everywhere: the *sign* of the relationship depends on the **regime**. In 2022–2023 the market was terrified of inflation, so a hot surprise was unambiguously bad-news-is-bad (it meant more Fed tightening). In a different regime — say a growth-scare recession where the fear is deflation and a hard landing — a *hot* inflation print could be read as reassuring (the economy isn't collapsing) and stocks could *rise* on it. The surprise is the input; the *reaction function* converts it into a sign, and the reaction function changes with the regime. The full treatment of how the Fed's reaction function maps inflation onto policy is in [inflation and the Fed reaction function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot).

![Cross-asset same-day reaction to the cool October 2022 CPI across stocks, bonds, the dollar and gold](/imgs/blogs/consensus-expectations-and-priced-in-6.png)

The cross-asset bars from that November 2022 cool print show the full risk-on cascade a single downside surprise can trigger: every risk asset up hard (S&P +5.5%, Nasdaq +7.3%, Dow +3.7%, Bitcoin ~+10%, gold +2.8%) and the safe-haven dollar down −2.1%. One number, below consensus by two-tenths of a percentage point, and a coordinated repricing across every asset class. That is the power of a surprise hitting a market that was leaning the other way.

#### Worked example: the cool Oct-2022 CPI on a long position

You hold a \$30,000 long S&P 500 position into the 10 November 2022 CPI release. The number prints cool — 7.7% versus a 7.9% consensus, a −0.2pp downside surprise — and the index rallies +5.54% on the day. Your gain: +5.54% × \$30,000 = **+\$1,662** in a single session. Had you instead held that \$30,000 long into the *hot* September print (−4.32%), you'd have lost −4.32% × \$30,000 = **−\$1,296**. Same position, same instrument, two months apart; the only thing that flipped your P&L by nearly \$3,000 was the sign of the surprise versus consensus. **You weren't betting on inflation; you were betting on inflation relative to what the market already expected.**

#### Worked example: the bond-market side of the same surprise

That cool November print didn't just move stocks — it slammed Treasury yields lower as the market repriced the Fed path. The 10-year yield fell about 28 basis points on the day. Suppose you hold a \$500,000 position in 10-year Treasuries with a DV01 (dollar value of a 1bp move) of roughly \$430 per basis point. When yields *fall*, bond prices *rise*, so a −28bp move is a gain. Your profit: 28 bp × \$430/bp = **+\$12,040** on the \$500,000 position. A two-tenths-of-a-point inflation surprise translated into over twelve thousand dollars of bond P&L because the surprise repriced the entire expected rate path. **In fixed income, the surprise moves the yield, and the yield times your DV01 is the money — the level of CPI never enters the arithmetic.**

## 5. Economic surprise indices: a regime gauge

Individual surprises move price on the day. But there is a higher-order question every macro trader wants answered: *is the data, on the whole, coming in better or worse than expected lately?* That is what an **economic surprise index** measures, and the most famous is the **Citi Economic Surprise Index (CESI)**.

The construction is exactly what its name says. For a basket of major US economic releases, the index tracks whether each one beat or missed its consensus, weights and smooths them, and produces a single line. When the index is **positive and rising**, data has been *systematically beating* expectations — the economy is outperforming what economists penciled in. When it is **negative and falling**, data is *systematically missing* — reality is undershooting the consensus.

Why does this matter for "priced in"? Because consensus is not static — it *adapts*. After a string of upside surprises, economists raise their forecasts, the bar gets higher, and it becomes harder for the *next* print to surprise to the upside. After a string of downside misses, forecasts get cut, the bar drops, and an in-line or slightly-better print can now clear an easy hurdle and rally. The surprise index is therefore a gauge of *how high the bar currently is*. A roaring-hot surprise index means the consensus has caught up to a strong economy, and the easy upside surprises are behind you. A deeply negative one means expectations are washed out and the risk skews to the upside.

Practically, traders use the surprise index as a *mean-reversion and regime tool*. Extreme positive readings tend to fade (the bar gets too high to keep beating); extreme negative readings tend to recover (the bar gets too low to keep missing). It tells you which way the *consensus itself* is likely to drift, which front-runs the surprises of the coming weeks. It pairs naturally with the broader risk-rotation logic in [risk-on, risk-off: how money rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates).

The surprise index also clarifies one of the most confusing market phrases: *"good news is bad news."* In a regime where the market's dominant fear is the central bank tightening too much, a *positive* surprise — strong jobs, hot inflation — pushes the surprise index up and the market *down*, because better data means a more aggressive Fed. In that world, traders literally root for weak data: a downside surprise on payrolls rallies stocks because it pulls forward rate cuts. The surprise index, read alongside the regime, tells you which sign the market currently wants. When the index is climbing and stocks are falling with it, you are in a "good-news-is-bad-news" regime and a strong upcoming print is a *risk to your longs*, not a comfort. When the index is climbing and stocks rise with it, you are in a "good-news-is-good-news" growth regime and the same strong print is a tailwind. Same data, opposite trade — and the surprise index plus the regime is how you tell which world you're in before the number lands.

#### Worked example: the bar moving against you

You are long a \$40,000 basket of cyclical stocks because the economy has been printing strong. The Citi surprise index has been deeply positive for two months — data has crushed expectations. But that very strength means economists have *raised* their forecasts: the consensus bar is now high. The next round of releases comes in *solid but merely in-line* with the now-elevated consensus. Surprise ≈ 0, so there's no fresh upside fuel, and the basket drifts down −2.5% as the "good news is fully expected" reality sets in: −2.5% × \$40,000 = **−\$1,000**. The economy didn't weaken — the *bar* rose to meet it, and an in-line print against a high bar is effectively a disappointment. **A positive surprise index is a warning that future upside surprises are getting expensive to come by.**

## 6. When is something "fully priced," and the asymmetric payoff it creates

We have used the phrase "fully priced" loosely; let us make it precise, because the answer is the difference between a good event trade and a bad one. An outcome is *fully priced* when the market's implied probability for it is near certainty — say 90% or more in the futures, a tight survey range clustered on that outcome, and a small options-implied expected move (because if the outcome is near-certain, there's little volatility to price). When all three agree on one outcome, the base case is in the price and the event has almost no power to move things *in the expected direction*.

This creates a profoundly **asymmetric payoff**, and it is the most important practical idea in the post. When an outcome is 90% priced:

- If the priced outcome happens (the 90% case), you make almost nothing — the move already occurred during the anticipation. Reward ≈ 0.
- If the *un-priced* outcome happens (the 10% tail), the move is violent, because the entire market has to reposition from the wrong base case. Reward (or loss) is large.

So betting *on* a fully-priced outcome is a terrible trade: you risk a large adverse move (the tail) to win a tiny favourable one (the base case). It is the textbook bad bet — picking up pennies in front of a steamroller. Conversely, the un-priced tail is where the asymmetry favours you: if you have a *genuine, well-founded* reason to expect the 10% outcome, the payoff for being right is enormous relative to the cost of being wrong, because the market has under-priced your scenario.

This asymmetry is why professional event traders are obsessed with *positioning* and *what's priced* rather than with *what they think will happen*. Two traders can agree the Fed will hold — but if one notices the market is only 60% priced for a hold while the other assumes it's a done deal, the first one sees an opportunity the second is blind to. The edge is never in the forecast; it is in the gap between your forecast and the market's price.

#### Worked example: the asymmetric payoff of a fully-priced event

A meeting is 90% priced for a hold. You hold a \$25,000 equity position. Consider the two ways to play it.

- **Bet on the hold (the base case).** If the hold happens (90% likely), the move is ~0.1% — about +\$25 on \$25,000. If the surprise cut happens (10%), equities jump +1.8% — but you weren't positioned for it, so you merely don't lose; your flat book gains the market 1.8% × \$25,000 = +\$450 only if you happened to be long. Betting *specifically on the hold* via, say, selling volatility, nets you the small premium and risks the tail.
- **Bet on the un-priced cut (the tail).** You'd need a real reason — but if you're right, the +1.8% move on \$25,000 is +\$450, won at long odds, against a small cost of being wrong (the position bleeds modestly if the hold happens as expected).

The point in dollars: the priced outcome pays \$25, the un-priced outcome pays \$450 — an 18-to-1 asymmetry. **You never get paid for agreeing with the consensus; you get paid for correctly disagreeing with it.**

## How it reacted: real episodes

Theory is cheap. Here are dated episodes where the "priced in" mechanism is the entire story.

### December 2015: the fully-priced first hike

We opened with it, so let us close the loop. The first Fed hike of the post-2008 era, 16 December 2015, was ~100% priced by Fed funds futures going in. The decision moved the S&P *up* ~1.5% — a relief rally — because the only un-priced element was the *guidance*, which signalled a gradual path. The lesson, drawn cleanly: when the decision is fully priced, the decision is a non-event and the *communication* is the trade. This is precisely why the FOMC press conference and the dot plot matter more than the rate decision itself — covered in depth in [trading the FOMC statement, presser and dot plot](/blog/trading/macro-trading/trading-the-fomc-statement-presser-dot-plot).

### December 2018: priced decision, un-priced guidance, market breaks

On 19 December 2018, the Fed hiked 25bp to a 2.25–2.50% range. The hike itself was largely expected. But the *guidance* — the dot plot still penciling in two more hikes for 2019, and Chair Powell's "auto-pilot" comment about balance-sheet runoff — was far more hawkish than a fragile market wanted. The S&P reversed to close **−1.54%** on the day, and the un-priced hawkishness fed a brutal sell-off: the index fell about −19.8% from its September high to the Christmas-Eve low (the 24 December session alone was −2.71%). Here the *decision* was priced and the *guidance* was the shock — the mirror image of December 2015. The surprise was in the path, not the move.

### March 2022: the relief rally on the first hike of the cycle

On 16 March 2022, the Fed began its hiking cycle with a +25bp move to a 0.50% upper bound. By any abstract logic, the start of an aggressive tightening campaign should be bad for stocks. Instead the S&P rose **+2.24%** and the Nasdaq +3.77% on the day. Why? The hike was fully priced — Fed funds futures had it nailed — and the *removal of uncertainty* plus a not-more-hawkish-than-feared message produced a relief rally. "Buy the fact" in action: the market had sold off into the anticipated tightening, and the confirmation let positioned shorts cover.

### September 2024: a 50bp cut, and a muted day

On 18 September 2024, the Fed delivered its first cut of the easing cycle — and made it a *jumbo* 50bp cut to a 5.00% upper bound. A bigger-than-the-minimum cut, yet the S&P closed essentially flat (**−0.29%**) on the day before rallying the next session. The 50bp size carried two opposing messages already largely in the price: it was dovish (more easing) but also slightly worrying (why does the Fed feel the need to move fast?). With the cut and even its size substantially anticipated by futures, the decision day was muted; the trend resumed once the market digested the guidance. Again: the priced part did little; the residual interpretation did the rest.

### November 2022: the un-priced cool CPI that ripped

The counter-case — where the surprise was genuinely large and the reaction was enormous — is the 10 November 2022 cool CPI we dissected above. Here the market was *not* positioned for a downside surprise; it was braced for sticky-high inflation. The −0.2pp miss was a real shock, and the S&P's +5.54% (with the cross-asset cascade in the bars above) shows what happens when a surprise lands on a market leaning the wrong way. The difference between this and the FOMC episodes is precisely *positioning*: the FOMC outcomes were priced, so they did little; the CPI miss was un-priced, so it did everything.

### Vietnam 2022: a priced policy defence and an un-priced flow

The consensus-and-surprise logic is not a US-only phenomenon — it governs every market with a forward-looking auction, including Vietnam's VN-Index. Through autumn 2022, the State Bank of Vietnam (SBV) raised its refinancing rate from 4.0% to 6.0% in two moves (23 September and 25 October) to defend the dong as the dollar surged. Those hikes were largely *anticipated* — the SBV signalled the defence as the dollar index climbed toward its September peak, so the rate moves themselves were partly priced. What was *not* fully priced was the depth of the de-leveraging and foreign-flow rout that followed: the VN-Index, which had peaked near 1,528 in January 2022, fell roughly −39% to a trough of **911 on 15 November 2022** before recovering to close the year near 1,007. The lesson maps cleanly onto the US episodes: the *policy decision* (the rate hikes) was anticipated and did less than a beginner would guess; the *un-priced* margin-call cascade and forced selling were the violent part. As the dong stabilised, the SBV reversed course in 2023, cutting the refinancing rate three times (to 4.5% by June) — cuts the recovering market had begun to anticipate, so the index had already started climbing off the lows before the cuts were confirmed. The full mechanics of SBV policy are in [Vietnam monetary policy: the State Bank, the dong and the credit ceiling](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling), and the margin cycle that amplifies these moves is in [liquidity and the margin cycle in Vietnam](/blog/trading/vietnam-stocks/liquidity-and-the-margin-cycle-vietnam); the point here is universal — *the priced part of any policy event does little, and the surprise does everything, in Hanoi exactly as in New York.*

#### Worked example: the Vietnam trough in dollars

Suppose a foreign investor held a \$15,000 VN-Index position from the January 2022 peak (~1,528) into the 15 November 2022 trough (911). The index fell (911 − 1,528) / 1,528 = −40.4%, so the position dropped by −40.4% × \$15,000 = **−\$6,055** to about \$8,945. Almost none of that loss came on the days the SBV *announced* its rate hikes (those were anticipated); the bulk came on the un-priced forced-selling sessions as margin calls and foreign outflows cascaded. **Even in a frontier market, your P&L tracks the surprises, not the headline policy moves everyone saw coming.**

## Common misconceptions

**"A rate cut is bullish."** Only if it wasn't already priced. By the time a cut is 90% priced into Fed funds futures, the rally that "a cut is bullish" predicts has already happened on the climb up the priced-in ladder. The September 2024 50bp cut was *more* than a standard cut and the S&P still closed −0.29% on the day. The bullishness lived in the weeks of anticipation, not the announcement. The decision pays for the *unexpected* part of itself only.

**"Bad economic news crashes stocks."** Only relative to consensus, and only in a regime where the market reads bad news as bad. The 7.7% CPI in November 2022 was, in absolute terms, dreadful inflation — and stocks rose +5.54% because it was *better than feared*. Conversely, in a soft-landing regime, genuinely bad growth data can rally stocks if it pulls forward expected rate cuts. The number's level is not the signal; the surprise versus consensus, filtered through the regime, is.

**"If I know the number will be strong, I should buy."** Not if everyone else knows it too — because then it's priced, and there's nothing to capture. Knowing the *outcome* is worthless; only knowing the outcome *differently than the consensus* has value. The edge is in the deviation, not the direction. If your forecast equals the consensus, you have no trade.

**"A big number means a big move."** Magnitude of the *number* is not magnitude of the *move*. An 8.3% print produced a −4.32% S&P day because it beat consensus by 0.2pp; an in-line 8.3% print would have produced near-nothing. The move scales with the *surprise*, which the options market pre-estimates as the expected move. A headline can be huge and the surprise tiny.

**"Priced in means the price won't move."** It means the *priced outcome* won't move it. The price will still move violently if the *un-priced* outcome occurs (the 15% tail in our FedWatch example) or if a second-order detail — the guidance, the internals, the dot plot — surprises. "Priced in" is a statement about the base case, not a guarantee of calm.

## The playbook: trading what's priced vs what's not

Everything above reduces to one pre-event question and a branching set of actions. Before any scheduled release, you ask: **is the outcome already priced in?**

![Decision tree for whether an outcome is priced in and whether to fade the knee-jerk or ride the repricing](/imgs/blogs/consensus-expectations-and-priced-in-7.png)

**Step 1 — Read the consensus.** Pull the survey median *and* range (how contested is it?), check FedWatch or the relevant futures-implied probability (how one-sided is it?), and note the options-implied expected move (how big a swing is priced?). These three together tell you the base case and how confidently the market holds it.

**Step 2 — Decide if it's priced.** If futures/surveys show the outcome ~fully priced (say 90%+ one way, a tight survey range, and the expected move is small), the base-case outcome is *in the price*. If the odds are split, the range is wide, or your own well-founded view differs materially from consensus, it is *not* fully priced.

**Step 3a — If it's priced:** there is no edge in betting on the base case. Two productive moves remain. First, **fade the knee-jerk** — when an in-line print produces a reflexive spike, that spike often reverts within minutes because there was no real new information. Second, **trade the second-order detail** — the press conference, the dot plot, the internals beneath the headline, the guidance. In December 2015, December 2018, March 2022 and September 2024, the *decision* was priced and the *communication* was the entire trade. Size *small* into a priced event, because the reward for the base case is near-zero; size up only if a genuine surprise materialises.

**Step 3b — If it's not priced:** the surprise versus consensus is the whole move, and you **ride the repricing**. This is where positioning ahead of a contested print, or pressing a surprise as it lands, has real reward. Size for the surprise, and — critically — **set your invalidation**: if the print comes in-line (no surprise), your thesis is wrong and you cut, because an in-line print against a not-fully-priced setup simply means you mis-read the consensus, not that a move is coming.

**Sizing and risk around the event.** The expected move is your risk yardstick. If options price a ±1.2% expected move and you cannot stomach a 1.2% adverse swing on your position size, you are too big for the event. A common discipline: size so that a *two-times-expected-move* adverse surprise is a survivable loss, not a blow-up — because surprises that blow through the expected move are exactly the ones that happen on the prints nobody priced. Around the most violent events (a carry unwind, a CPI shock), correlations converge and "diversified" books move together, so the practical exposure is larger than it looks; the cross-asset mechanics of that are in [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).

#### Worked example: sizing to the expected move

The S&P is at 5000, the ATM straddle into CPI is \$60, so the expected move is ±1.2%. You decide a *survivable* event-day loss for your account is \$1,000, and you want to survive even a *two-times* expected move (±2.4%) adverse surprise. Solve for position size: \$1,000 = 2.4% × position, so position = \$1,000 / 0.024 = **\$41,667**. If instead you sized a \$100,000 book into the same event, a 2.4% adverse surprise would cost 2.4% × \$100,000 = **−\$2,400** — more than double your survivable loss, meaning you're roughly 2.4× too large for the event. **The expected move converts your risk tolerance into a hard position cap; ignore it and a single surprise can take a chunk you never agreed to risk.**

### A pre-event checklist

Reduce the whole framework to a routine you run before every scheduled release:

1. **What is the consensus?** Pull the survey median *and range*. A tight range = settled; a wide range = a coiled spring with bigger reaction potential.
2. **What is the market-implied probability?** For Fed-sensitive events, read FedWatch or the relevant futures-implied odds. Is it one-sided (90%+) or split (near 50/50)?
3. **What is the expected move?** Eyeball the ATM straddle as a percentage of the underlying. That's the market's pre-paid estimate of the event-day swing and your risk yardstick.
4. **Is there a whisper?** Does desk chatter, prediction-market pricing, or the recent data trend put the *real* expectation away from the published survey? Weight the market-based signals over the poll.
5. **Which internals matter this regime?** Decide in advance which component (core, shelter, wages, the unemployment rate) the market will trade if the headline lands on consensus.
6. **Priced or not?** If the base case is fully priced, plan to fade the knee-jerk or trade the second-order detail, and size *small*. If it's not priced and your view differs from consensus, plan to ride the repricing, size for the surprise, and set the in-line print as your invalidation.
7. **What's my invalidation?** Write down, before the print, the outcome that means you were wrong — usually an in-line number when you expected a surprise — and the level at which you cut.

Run that list and you will never again react to the *level* of a number, only to its surprise versus what the market already held. That single shift — from "is the number good or bad?" to "is the number better or worse than priced?" — is the entire edge this post exists to give you.

The whole discipline, distilled: *find what the market expects, ask whether it's priced, and only bet where your view differs from the consensus.* The number is never the trade. The gap between the number and what everyone already expected — that is the trade.

## Further reading and cross-links

- [The macro calendar: CPI, NFP, FOMC, PMI](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi) — which releases matter, when they hit, and what the market watches in each.
- [Trading the FOMC: statement, presser, dot plot](/blog/trading/macro-trading/trading-the-fomc-statement-presser-dot-plot) — why the communication, not the decision, is so often the trade when the rate move is priced.
- [Terminal rate and rate-cut cycles: pricing the path](/blog/trading/macro-trading/terminal-rate-and-rate-cut-cycles-pricing-the-path) — how the market prices a whole sequence of meetings, not just the next one.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the policy mechanism behind the rate that Fed funds futures are pricing.
- [Inflation and the Fed reaction function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot) — how a CPI surprise maps onto policy and therefore onto the sign of the reaction.
- [Options theory](/blog/trading/quantitative-finance/options-theory) and the [volatility surface](/blog/trading/quantitative-finance/volatility-surface) — the machinery behind the options-implied expected move.

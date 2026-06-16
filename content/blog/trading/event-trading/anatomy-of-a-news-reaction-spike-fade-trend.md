---
title: "Anatomy of a News Reaction: The Spike, the Fade, and the Trend"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A macro release does not produce one move — it produces a sequence: an instant knee-jerk spike, then either a fade as the overshoot reverts or a trend as the data confirms a new regime. This is how to read which one you are in."
tags: ["event-trading", "macro", "market-microstructure", "knee-jerk", "fade", "trend", "liquidity", "cpi", "nfp", "carry-unwind", "order-flow", "trading"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A macro release never produces a single move; it produces a **sequence**: an instant knee-jerk *spike* the moment the number prints, then either a *fade* (the overshoot mean-reverts) or a *trend* (the data confirms or changes the regime and price runs all session). The trader's job is to react to the **reaction**, not the number.
>
> - **What happens:** at 8:30:00.000 algorithms reprice off the single headline number in milliseconds; seconds later humans read the *internals* and either confirm or reverse it; over hours-to-days the move settles at a new fair value.
> - **The cross-asset map:** the first tick hits every risk asset at once — on Aug 5 2024 the Nikkei fell **−12.40%**, the S&P **−3.00%**, the Nasdaq **−3.43%** and Bitcoin **−15%** in the same session — but a knee-jerk that overshoots can fully reverse: the Nikkei rebounded **+10.23%** the next day.
> - **The trade:** a move sticks when the print **changes the regime or confirms the trend** (ride it) and fades when it only **overshoots stretched positioning** (mean-revert it, with tight stops).
> - **The one number to remember:** the first move is the knee-jerk; the *second* move — when humans read the details — is usually the more honest one.

On the morning of November 14 2023, the US October CPI report hit the tape at 8:30 a.m. New York time. The headline came in at 3.2% year-over-year against a 3.3% consensus, and core inflation printed 4.0% against 4.1% expected. A one-tenth-of-a-point miss on each — a cool print. In the first second, before a single human had read past the top line, the S&P 500 futures jumped, the dollar dropped, and two-year Treasury yields fell. By the close, the S&P 500 was up **+1.91%**, the rate-sensitive Russell 2000 small-cap index was up a remarkable **+5.44%**, and the 10-year yield had dropped **19 basis points**. The move that started in the first millisecond ran all session. That was a *trend*.

Now rewind to a different kind of morning — the kind every news trader has been burned by. A release prints a number that *looks* good on the headline. Algorithms buy it instantly. The chart spikes green. Retail traders pile in, certain they have caught the move early. Then, sixty seconds later, the institutional desks finish reading the *internals* — the revisions to last month, the sticky core components, the detail that contradicts the headline — and the whole move reverses. The spike becomes a fade. Everyone who bought the first tick is now underwater, stopped out by the very move they thought they were front-running. This is the **headline-vs-internals trap**, and it is the single most expensive mistake in event trading.

These two stories are not different events. They are the same event caught at different points in its life. Every macro release — every CPI print, jobs report, central-bank decision — produces not one move but a *sequence*: a pre-event drift, an instant spike, then a fork into a fade or a trend, then a settle. Learning to read that sequence — to tell, in the seconds after a print, whether you are watching a fade or a trend — is the core skill of trading the news. This post is the anatomy lesson.

![Schematic timeline of the four phases of a news reaction: pre-event drift, instant spike, fade or trend, and settle](/imgs/blogs/anatomy-of-a-news-reaction-spike-fade-trend-1.png)

## Foundations: the four phases of a reaction

Before we can trade the reaction, we have to name its parts. Markets do not lurch randomly when a number drops; they move through a recognizable choreography, and each phase is driven by a different set of participants doing a different thing. Let us build it from zero.

If you want the *why* behind which numbers move markets and how the calendar is structured, the companion piece in the macro series — [the macro calendar of CPI, NFP, FOMC and PMI](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi) — lays out the mechanism. Here we are zoomed all the way in on the *minutes* around a single print.

### The starting point: markets trade the surprise, not the number

The most important idea in event trading, and the one beginners most often miss, is that **price already contains the consensus**. A CPI number does not move markets because inflation is 3.2%; it moves markets because 3.2% is *different from what was expected* (3.3%). The expected number is already baked into every quote on the screen before the release. Only the **surprise** — the gap between the actual figure and the consensus forecast — is new information, and only new information moves price.

Where does the consensus come from? Before every scheduled release, economists at banks, research shops, and data providers publish their forecasts, and the median of those forecasts becomes the headline "consensus" or "expected" figure that every trader watches. The market then *positions* around that number: if the consensus is +175,000 jobs, the price of every related asset already reflects a world in which +175,000 jobs are about to be reported. There is also a softer, unpublished number traders call the "whisper" — the consensus *behind* the consensus, the figure desks actually expect after seeing the morning's other data — and price often reflects the whisper more than the official forecast. The practical point is the same: by the time the number prints, expectations are already in the price.

Say the entire market expects the jobs report to show +175,000 new jobs. Traders have positioned for +175,000. The price reflects +175,000. When the actual number prints +175,000, *nothing happens* — there is no surprise, no new information, no reason to reprice. The number can be huge and the reaction can be zero. Conversely, a tiny one-tenth-of-a-point miss on a number everyone was watching can crater an index, because the *surprise* — not the level — is what trades.

There is a second layer that makes this even sharper: the *sign* of the reaction depends on the **regime**. The same hot CPI surprise that crashed stocks in 2022 — when the market's overriding fear was inflation and an aggressive Fed — might *rally* stocks in a different regime where the market is more worried about growth than prices. "Good news is bad news" and "bad news is good news" are not paradoxes; they are statements about which fear is dominant. A strong jobs report is good for the economy but bad for stocks when the market fears the Fed will keep rates higher for longer in response. So before you can predict the reaction, you have to know *what the market is afraid of right now* — and that is a property of the regime, not the number. (The macro series develops this idea fully in [inflation and the Fed reaction function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot).)

Hold both ideas in your head, because together they explain the entire choreography that follows: the knee-jerk spike is the market repricing the surprise, the *sign* of that repricing is set by the regime, and how big the spike is depends on how big the surprise is relative to what was already positioned.

### Phase 1 — the pre-event drift (positioning)

In the minutes and hours before a scheduled release, two things happen. First, traders who have a view *position* for it — buying or selling in anticipation — which creates a slow drift in the direction of the consensus bet. Second, and more importantly, market makers and liquidity providers begin to *withdraw*. Nobody wants to be holding a fat resting order on the book when a number that could move price 2% in a millisecond is about to drop. So the order book thins. We will return to this thinning — the **liquidity vacuum** — because it is where a lot of the damage gets done.

The practical consequence: by 8:29:55, five seconds before the print, the market is quiet, thinly traded, and *coiled*. The drift has often pushed price slightly in the "expected" direction, which sets up a nasty trap if the surprise comes in the other way.

There is also an *implied-volatility* dimension to the pre-event phase. Options that expire after the release carry an inflated price, because everyone knows a large move is possible and the people selling that insurance demand to be paid for the risk. This shows up as elevated implied volatility into the event — the options market is literally quoting the *expected move*. The closer the release, the more this event premium dominates the option price. We will use the expected move later for sizing; for now, just note that the market is telling you, in advance and in dollars, roughly how big it thinks the spike will be. When implied volatility is high going into a print, the market is braced for a large surprise; when it is low, the market is complacent and an off-consensus number will move price *more* than the same surprise would in a braced market.

### Phase 2 — the instant repricing (the knee-jerk spike)

At exactly 8:30:00.000, the number hits the wire. Within **milliseconds** — faster than any human can read, let alone think — algorithms parse the single headline figure and fire orders. This is the **knee-jerk**: the instant, reflexive repricing of the surprise. It is mechanical, it is fast, and it is driven entirely by the *one number* the algorithms are programmed to read. The spike can be violent: a few tenths of a percent in stock indices, dozens of basis points in rates, multiple percent in crypto, all in under a second.

The crucial thing to understand is *who* is trading here. It is not humans. It is execution algorithms reacting to a parsed data feed. They are fast but *shallow* — they read the headline number and nothing else. That shallowness is precisely what sets up the second move.

### Phase 3 — the fade or the trend

Seconds to minutes after the spike, the humans catch up. Analysts and discretionary traders open the actual report and read the **internals**: the components, the revisions to prior months, the core figures that strip out volatile items, the details the algorithms never saw. Now one of two things happens.

If the internals *confirm* the headline — if the cool CPI is cool all the way down, with soft core and downward revisions — the move **trends**: it follows through, sometimes accelerating, as the slower money piles in behind the fast money and the market re-rates to a new regime. This is **follow-through**.

If the internals *contradict* the headline — if the "cool" print had a sticky core, hot shelter costs, and an upward revision to last month — the move **fades**: it reverses, often completely, as the knee-jerk is recognized as an *overshoot* and traders unwind it. The fade can also happen with no contradiction at all, purely because the spike ran into stretched positioning and there is no one left to keep pushing. Either way, the result is a **whipsaw**: a sharp move one way followed by a sharp move back, stopping out everyone who chased the first tick.

### Phase 4 — the settle

Over the following hours and days, price anchors at a new fair value consistent with what the market now believes the data means for policy and growth. The intraday noise dies down; the **volatility crush** sets in; and if the print genuinely changed the regime, the *real* trend — the multi-day, multi-week repricing — begins. The biggest moves in markets are not the knee-jerk; they are the regime shifts that the knee-jerk merely announces.

The volatility crush deserves a word, because it catches out option buyers constantly. Recall that implied volatility was inflated going into the release to price the *uncertainty* of the number. The instant the number prints, that uncertainty resolves — the market now knows the figure — and the event premium evaporates. Implied volatility collapses. This means an option bought for the event can *lose money even when you got the direction right*, because the volatility crush wipes out more value than the move adds. A trader who bought a call into a CPI print, watched the index rally, and still lost money has met the volatility crush. The settle phase is when this premium bleeds out; it is also why selling options *before* an event (collecting the inflated premium) and buying them back after (when the premium has crushed) is a recognized event-trading style — one with its own fat-tail risk if the move is bigger than the premium implied.

So the four phases each have their own protagonist and their own danger: the drift is the positioners and the thinning book; the spike is the algorithms and the vacuum; the fork is the humans reading internals; the settle is the volatility crush and the start of the real trend. Hold the whole sequence in mind and the chaos of a release morning resolves into a process you can read.

#### Worked example: the surprise, not the level, sets the move

Suppose you hold a \$30,000 long position in an S&P 500 ETF into a CPI release. The market expects 3.3%; you are positioned for a cool print. Two scenarios:

- CPI prints exactly **3.3%** — no surprise. The index closes flat, your \$30,000 is unchanged: **\$0** P&L. The number was big news on TV and a non-event for your account.
- CPI prints **3.2%** (a cool surprise) and the S&P closes **+1.91%**, as it did on 2023-11-14. Your \$30,000 × 1.91% = **+\$573**.
- CPI prints **3.5%** (a hot surprise) and the index reacts like the hot Sep 2022 print, closing roughly **−4.32%**. Your \$30,000 × (−4.32%) = **−\$1,296**.

The lesson in dollars: your P&L is driven by the *surprise relative to consensus*, not by the headline level — and the downside on a hot surprise (−\$1,296) dwarfs the upside on a symmetric cool one (+\$573), because fear reprices faster than greed.

## Who moves the first tick

Let us zoom into the first second and ask the question that demystifies the whole spike: *who is actually trading at 8:30:00.000, and what are they reacting to?*

The answer is **execution algorithms reading a structured data feed**. The major data vendors — and the exchanges and wire services that distribute economic releases — publish the headline number in a machine-readable format the instant the embargo lifts. Algorithmic trading systems subscribe to these feeds, parse the headline figure, compare it to a pre-loaded consensus, compute the surprise, and fire pre-staged orders — all within microseconds to single-digit milliseconds. No human is in the loop. By the time a person has finished reading the first sentence of the report, the first leg of the move is already over.

This is why the spike is so fast and so *mechanical*. The algorithms are not making a nuanced judgment about what the print means for the economy. They are doing something much dumber and much faster: "headline below consensus → buy the index, sell the dollar; headline above consensus → do the opposite." They read *one number*. That is their entire model in the first 50 milliseconds.

It is worth being precise about the technology, because it explains both the speed and the shallowness. There are firms whose entire edge is *latency* — being the first to receive, parse, and act on a release. They co-locate their servers in the same data centers as the exchanges, license the machine-readable feed directly, and optimize their parsing code to extract the headline number in microseconds. For these players, being one millisecond faster than the next firm is worth real money, because the first orders into the thin post-release book get the best fills. This is a genuine arms race, and an ordinary trader has *no chance* of competing in it. The lesson is not to try to be faster; it is to recognize that the first tick belongs to a game you are not playing, and to position yourself for the *second* move, where reading and judgment — things humans are still better at than a number-parsing bot — actually matter.

Note also what the algorithms are *not* doing. They are not reading the full release PDF. They are not weighing the revision to last month against the headline. They are not asking whether the composition of the jobs number is healthy. They are not consulting the regime. They are pattern-matching one number to one rule. Everything that requires *understanding* the release — and understanding is where the durable edge lives — happens later, and slower, and is done by people. That division of labor between the fast-but-shallow first move and the slow-but-deep second move is the structural fact that the entire spike-fade-trend sequence rests on.

Two consequences fall out of this, and they are the key to everything:

**First, the first tick is shallow.** Because the algorithms read only the headline, the knee-jerk reflects only the headline surprise — not the internals, not the revisions, not the context. If the headline is misleading (a cool top line hiding a hot core), the first move is *wrong*, and it will get corrected the moment humans read the detail. The first move is fast, but speed is not accuracy.

**Second, the first tick is reflexive and crowded.** Every algorithm is reading the same number and reacting the same way at the same instant. They all buy, or they all sell, simultaneously. That synchronized rush is what produces the violent spike — and, because they are all on the same side, it is also what sets up the snap-back when there is no one left to push.

### The headline-vs-internals second move

Here is the trap drawn out. The headline prints, the algorithms buy it, the chart spikes green — and then the humans open the report.

![Two-step before and after: algos buy the headline number, then humans read the internals and reverse the move](/imgs/blogs/anatomy-of-a-news-reaction-spike-fade-trend-2.png)

A real economic release is not one number. The CPI report has a headline figure, a *core* figure (stripping food and energy), dozens of component breakdowns (shelter, used cars, medical services, airfares), *and* revisions to the prior one or two months. The jobs report has the headline payrolls number, the unemployment rate, average hourly earnings, the labor-force participation rate, *and* revisions that can erase or double the headline. The headline is just the cover of the book; the internals are the story.

Discretionary desks know this. So the first thing a human does after the spike is *check the internals against the headline*. Three things they look for:

- **Revisions.** A jobs report showing +200,000 new jobs looks strong — until you see that last month was revised *down* by 80,000. The net new information is far weaker than the headline. The first tick bought the +200,000; the second move sells the revision.
- **Core vs headline.** A CPI headline can be dragged down by a one-month plunge in energy prices while *core* inflation — the part the Fed actually targets — stays sticky. The algorithms bought the cool headline; the humans sell the hot core.
- **Composition.** A payrolls number propped up by part-time and government hiring, with full-time private jobs falling, is weaker than it looks. The headline beat is hollow.

When the internals confirm the headline, the second move *adds to* the first — that is a trend forming. When the internals contradict the headline, the second move *reverses* the first — that is the fade. And because the algorithms moved price to a level justified only by the headline, the reversal can be larger than the initial spike: the market has to give back the knee-jerk *and* reprice for the bad internals.

Let us walk through a concrete jobs-report trap, because it is the canonical version of this. The Employment Situation report — the "jobs report" — drops on the first Friday of the month at 8:30 a.m. Eastern. The headline is *nonfarm payrolls*, the number of new jobs created. Suppose it prints +250,000 against a +175,000 consensus — a big beat. The algorithms read +250,000, see a strong economy, and in a "good-news-is-bad-news" inflation regime they *sell* stocks and *buy* the dollar, because a strong economy means the Fed stays tight. The chart spikes down. Now the humans open the report and find three things: last month's +200,000 was revised *down* to +120,000 (erasing 80,000 jobs of apparent strength); the entire beat came from part-time and government jobs while full-time private payrolls *fell*; and average hourly earnings — the wage-inflation read the Fed actually cares about — came in *soft*. Suddenly the picture flips. The labor market is weaker than the headline, and wage pressure is cooling, which is *dovish*. The initial sell-off reverses; stocks rally and the dollar gives back its spike. Everyone who shorted the headline beat is now stopped out. The headline said "hot"; the internals said "cooling"; and the second move — the honest one — traded the internals.

This is why professional desks have a junior trader or an analyst whose entire job in the first sixty seconds is to *read the internals out loud* — payrolls, revisions, the unemployment rate, participation, average hourly earnings, the composition — while the senior trader decides whether the knee-jerk confirms or contradicts. The structure of the desk mirrors the structure of the reaction: someone fast for the headline, someone careful for the internals.

#### Worked example: fading a knee-jerk overshoot

You have studied a CPI print: the headline came in cool and the index spiked up, but you read the internals in real time and see a hot core and an upward revision to last month. You judge it an overshoot and **short \$20,000** of the index ETF at the top of the spike, expecting a fade.

- If you are right and the move mean-reverts **1%**: you cover at \$20,000 × (1 − 0.01) ≈ \$19,800, booking **+\$200**.
- If you are wrong and the headline was honest, the move *trends* against you **3%**: the index runs to \$20,000 × 1.03 = \$20,600 and you lose **−\$600**.
- Your reward-to-risk on this fade is +\$200 vs −\$600 — a 1-to-3 *against* you per unit of move.

The lesson in dollars: fades are high-conviction, low-margin trades — the −\$600 trend-against-you outcome is three times the +\$200 win, so a fade only makes sense with a *tight stop* (cut it the instant follow-through appears) and a clear internals-based reason. Fading "because it spiked" with no read on the internals is how accounts die.

## The liquidity vacuum at the release

We have talked about *who* trades the first tick. Now we have to talk about the *conditions* they trade into — because the single most underappreciated feature of a news reaction is that the market is at its **thinnest** exactly when the move is at its biggest.

Recall Phase 1: in the seconds before a scheduled release, market makers pull their resting orders off the book. They do this for a simple reason — *adverse selection*. A market maker earns the spread by posting a bid and an ask and getting hit on both. But if a number is about to drop that could move price 1% in a millisecond, any resting order left on the book is a sitting duck: it will get filled an instant before the market gaps, leaving the maker holding a position that is immediately deep underwater. So they pull their quotes. The book empties out.

![Order-book depth before, during, and after a release showing the liquidity vacuum and slippage](/imgs/blogs/anatomy-of-a-news-reaction-spike-fade-trend-4.png)

The result is a **liquidity vacuum**: for the first fraction of a second after the print, there is almost no resting depth to absorb the flood of market orders the algorithms are firing. With nothing to absorb them, those orders *walk the book* — they consume every thin remaining level and push price far further than the same volume would on a normal afternoon. The bid-ask spread, which might be a single tick in calm conditions, can blow out ten-fold. This is why the knee-jerk spike is so violent: it is a synchronized flood of orders hitting a near-empty book.

For a trader, the practical danger is **slippage**: the gap between the price you saw and the price you actually got. If you fire a market order into the vacuum at 8:30:00, you are not trading against a deep book at the screen price; you are trading against whatever scraps of liquidity remain, at progressively worse levels. The fill you get can be meaningfully worse than the last print you saw.

#### Worked example: slippage on a market order into the vacuum

You decide to buy \$25,000 of an index future the instant a number prints, firing a market order at 8:30:00.000 into the liquidity vacuum.

- On a normal afternoon, a \$25,000 order moves the price negligibly — slippage is a few cents, call it **\$5**.
- Into the vacuum, your order walks a near-empty book and fills **0.4% worse** than the price you saw: \$25,000 × 0.4% = **−\$100** of slippage, gone before the trade even has a chance to work.
- If you had instead *waited eight seconds* for the book to refill and used a limit order, you might have paid **\$10** of slippage — saving roughly **\$90** on this one trade.

The lesson in dollars: trading *into* the vacuum costs you the slippage (−\$100) on top of whatever the trade does; the discipline of waiting for liquidity to return — or using limit orders — is worth real money on every event trade you place.

The vacuum also explains a deeper truth about why reactions overshoot. Because there is no depth to absorb the first flood, price moves *too far* — further than the new information actually justifies — simply because there was nothing in the way. Once depth returns (within seconds, as makers re-post quotes now that the uncertainty has resolved), price often retraces part of that mechanical overshoot. That retracement is the *beginning* of the fade. The vacuum does not just make the spike violent; it manufactures part of the overshoot that the fade then corrects.

There is a feedback loop here that makes the overshoot worse, and it is worth understanding because it is the engine of the most violent reactions. Many traders place **stop-loss orders** — automatic sell orders that trigger if price falls to a certain level — to limit their losses. When the knee-jerk spike pushes price into a cluster of stops, those stops fire as market orders, which push price further, which triggers *more* stops, which pushes price further still. This is a **stop cascade**, and into a thin post-release book it is devastating: a modest surprise can snowball into a violent move purely through the mechanical chain-reaction of stops firing into a vacuum. The same logic runs in reverse for forced *buy-ins* (short sellers stopped out). None of this reflects new information about the economy; it is pure microstructure. And because it is mechanical rather than informational, the move it produces is exactly the kind that *fades* — once the stops are exhausted, there is no fundamental reason for price to stay at the extreme, and it snaps back.

This is why the relationship between the *size* of the surprise and the *size* of the move is so unreliable in the first seconds. A small surprise into a thin book sitting on top of a dense cluster of stops can produce a bigger spike than a large surprise into a deep book with no stops nearby. The spike measures liquidity and positioning as much as it measures information. That is the deepest reason "a big move" does not mean "big news" — and the deepest reason the first tick is not to be trusted.

A useful mental rule: the *quality* of the liquidity tells you which phase you are in. While the spread is blown out and the book is thin, you are in the vacuum and the overshoot — do not chase. As the spread tightens back toward normal and depth returns, the market is telling you the repricing is settling, and *that* is when a real read on fade-vs-trend becomes possible. Traders who watch the order book, not just the price, can often see the vacuum refill in real time and use it as their cue to act.

## Fade vs trend: what makes a move stick

We now have the whole machine: the surprise drives the spike, algorithms fire into a vacuum and overshoot, humans read the internals, and the move either reverses (fade) or follows through (trend). The trader's entire edge comes down to one question, answered in the seconds after the print: **am I watching a fade or a trend?**

![Decision tree for distinguishing a fade from a trend after a news spike](/imgs/blogs/anatomy-of-a-news-reaction-spike-fade-trend-6.png)

A move **trends** — it sticks and follows through — when at least one of two things is true:

- **The print changed the regime.** This is the big one. If the data is surprising enough to alter the market's belief about the *path of policy* — about what the Fed will do — then the repricing is not noise; it is a genuine re-rating that will play out over days. A CPI print that convinces the market the Fed will pivot from hiking to cutting does not just spike; it changes the discount rate for every asset on Earth, and that move trends. (The mechanism of how inflation feeds the Fed's reaction function is laid out in [inflation and the Fed reaction function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot).)
- **The internals confirm the headline.** When the cool headline is cool all the way down — soft core, downward revisions, broad-based disinflation — the second wave of (human) flow pushes the same direction as the first wave of (algo) flow. Confirmation begets follow-through.

A move **fades** — it reverses — when:

- **It only overshot stretched positioning.** If everyone was already leaning the same way going into the print, the surprise can trigger a violent move *that immediately runs out of fuel*, because there is no one left to keep pushing and plenty of profit-takers ready to fade it. The move was about *positioning*, not information. (Reading the build-up of that positioning — via the COT report and dealer hedging — is its own skill, covered in [following the flows](/blog/trading/macro-trading/following-the-flows-positioning-cot-dealer-hedging).)
- **The internals contradict the headline.** As we saw: the algorithms bought a misleading top line; the humans correct it; the move reverses.
- **There is no follow-through volume.** The single most practical tell. If the spike is not accompanied by *sustained* volume in the same direction — if the move comes on the initial flood and then volume dries up — it is a fade. A trend recruits new buyers (or sellers) wave after wave; a fade exhausts the first wave and stalls.

The honest truth is that you cannot know for certain in the first second. But you can read the tells in the right *order*: did the print plausibly change the policy regime? do the internals confirm or contradict? is volume sustaining the move or drying up? The more "yes-it-confirms-and-changes-the-regime" answers, the more you lean trend; the more "it-only-overshot-positioning-and-volume-died" answers, the more you lean fade.

#### Worked example: the regime-change print that trended

On 2022-11-10, the October CPI printed 7.7% against 7.9% expected — a cool surprise that, crucially, came when the market desperately wanted evidence that the Fed could stop hiking. This was not noise; it was a *regime* signal. The internals confirmed it. The result was follow-through all session:

- The S&P 500 closed **+5.54%** — its best day in over two years.
- The 10-year Treasury yield fell **−28 basis points** and the dollar fell **−2.1%**, confirming the move across assets (a fade would show stocks up but yields and the dollar disagreeing).
- A \$25,000 long S&P position made \$25,000 × 5.54% = **+\$1,385** on the day, and because it was a regime shift, the move *continued* for weeks rather than fading by the close.

The lesson in dollars: when a print changes the policy regime *and* the internals and cross-asset moves all agree, you are in a trend — the \$1,385 day was the *start*, not the peak, and the trade was to ride it, not fade it.

![S and P 500 same-day move on three CPI surprises showing which moves stuck](/imgs/blogs/anatomy-of-a-news-reaction-spike-fade-trend-7.png)

Compare the three CPI days above. The hot Sep 2022 print (−4.32%) and the cool Nov 2022 print (+5.54%) were both *regime-relevant* — they changed the market's belief about the Fed's path — so they produced enormous moves that trended. The milder Oct 2023 cool print (+1.91%) was a smaller, less regime-shifting surprise, and produced a more modest, more fade-able reaction. The magnitude of the move is itself a clue: the biggest moves are usually the ones that genuinely changed the regime, and those are the ones to ride.

## The multi-day after-life of a regime-shifting print

The spike-fade-trend sequence we have built so far plays out in seconds to hours. But the most valuable thing a release can do is not produce an intraday move at all — it is to *change the regime*, and a regime change has an after-life that runs for days and weeks. Learning to tell a one-day reaction from a regime change is what separates a scalper from a position trader, and it is where the real money in event trading is made.

A regime change is a shift in what the market believes about the *path of policy*. Most prints, even surprising ones, do not change the regime; they are one more data point consistent with the existing belief. But occasionally a print is decisive enough to flip the market's whole framework — from "the Fed will keep hiking" to "the Fed is done" to "the Fed will start cutting." When that happens, the intraday move is just the opening announcement of a repricing that has to ripple through *every* asset, because the discount rate that prices everything has changed.

Why does this trend rather than fade? Because a regime change is not about positioning being stretched; it is about *new, durable information* that re-rates the future. After the cool November 2022 CPI, the market did not just have one good day and revert. It entered a multi-week rally, because the print had genuinely shifted the consensus on the Fed's terminal rate, and that shift had to be priced into bonds, then into rate-sensitive stocks, then into the dollar, then into emerging markets and crypto — wave after wave of repositioning by money that moves slowly. The slow money does not pile in within milliseconds; it reallocates over days as portfolio managers digest the new regime and rotate. That slow reallocation *is* the trend.

The tells of a regime-changing print, as opposed to a one-day blip, are worth memorizing:

- **The bond market leads and agrees.** Bonds price the path of policy directly. When a print moves the 2-year yield substantially — the maturity most sensitive to the expected policy path — and that move *holds* into the close and the next session, the regime has shifted. The +18bp 2-year move on the hot Sep 2022 CPI was the bond market saying "higher for longer," and it stuck. A one-day equity blip with no durable 2-year move is not a regime change.
- **The move broadens.** A regime change shows up everywhere — rates, the dollar, credit spreads, the most rate-sensitive equity sectors — not just in the index. The 2023-11-14 cool print sent the *Russell 2000* up +5.44%, far more than the S&P's +1.91%, precisely because small caps are the most rate-sensitive corner of the equity market and a regime shift toward lower rates helps them most. When the rate-sensitive assets move *more* than the broad index, you are watching a regime trade.
- **The move persists into the next sessions.** This is the simplest tell of all, and it requires patience: a regime change does not give back the move overnight. If the reaction holds the next day and the day after, it was information; if it round-trips, it was a positioning blip.

#### Worked example: catching the trend instead of the spike

After a cool, regime-shifting CPI you skip the violent first-tick spike and instead build a \$40,000 position once the bond market confirms (the 2-year yield has fallen and *held* for an hour) and the move has broadened to the rate-sensitive Russell 2000.

- You enter a touch later and worse than the spike-chasers — say you give up the first **1.5%** of the move by waiting for confirmation: on \$40,000 that is roughly **\$600** of "missed" first-tick gain.
- But because it is a genuine regime change, the trend runs **+6%** over the next two weeks as the slow money reallocates: \$40,000 × 6% = **+\$2,400**.
- Net, you captured \$2,400 of a regime trend by giving up \$600 of first-tick noise — and you never had to win the millisecond race against the latency bots.

The lesson in dollars: the regime trend (+\$2,400 over weeks) dwarfs the first-tick spike you skipped (−\$600 missed), so waiting for confirmation and trading the *after-life* of a regime print is both safer and more profitable than chasing the knee-jerk.

## How it reacted: real episodes

Theory is cheap. Let us put real dated numbers on the spike-fade-trend sequence with two episodes that every news trader should know cold.

### August 5 2024: the knee-jerk panic that reversed

This is the cleanest example of a knee-jerk overshoot reversing that modern markets have produced. The setup: the Bank of Japan hiked rates on July 31 2024, and a weak US jobs report on August 2 (payrolls +114,000 vs ~175,000 expected, unemployment up to 4.3%) triggered fears of a US recession. Together they detonated the *yen carry trade* — a massive, leveraged position where traders had borrowed cheap yen to buy higher-yielding assets worldwide. (The full mechanism of how leverage breaks in a carry unwind is laid out in [carry-trade unwinds: 1998, 2008, 2024](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks).)

When the carry trade unwound, it forced synchronized selling across every asset at once — the spike hit everything simultaneously, exactly as the liquidity-vacuum-and-synchronized-flow model predicts.

![Cross-asset same-day moves on August 5 2024 showing the spike hit every asset at once](/imgs/blogs/anatomy-of-a-news-reaction-spike-fade-trend-5.png)

On August 5 2024:

- The **Nikkei 225 fell −12.40%** — its worst single day since the Black Monday crash of 1987.
- The **S&P 500 fell −3.00%** and the **Nasdaq −3.43%**.
- **Bitcoin fell −15%**, behaving exactly like the high-beta risk asset it is when liquidity vanishes.
- The **VIX** — the market's fear gauge — spiked to an intraday **65.73**, a level seen only in genuine crises, from a close of 23.4 just three days earlier.

That was the knee-jerk: a synchronized, leverage-driven panic that overshot massively because forced sellers do not care about price and there was no depth to absorb them. (Why correlations between unrelated assets all snap to 1 in exactly these moments is the subject of [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).)

And then it reversed. The next session, August 6 2024, the **Nikkei rebounded +10.23%**. The panic had overshot; once the forced selling exhausted itself and bargain-hunters stepped in, the move faded hard. Anyone who sold the bottom of the August 5 spike got run over the next morning.

![Nikkei 225 daily move on August 5 and 6 2024 showing a one-day panic that reversed](/imgs/blogs/anatomy-of-a-news-reaction-spike-fade-trend-3.png)

This was a *fade*, not a trend, because the move was driven by **forced positioning unwind**, not by a durable change in the economic regime. Once the leverage was flushed, there was no fundamental reason for assets to stay at the panic lows, and they snapped back. The tell was there in real time: the move was a one-day liquidation, volume was deleveraging rather than fresh conviction, and nothing about the medium-term US growth or earnings picture had actually broken.

#### Worked example: holding the Nikkei panic through the reversal

You hold a \$15,000 long Nikkei position into August 5 2024 and, crucially, you judge the crash to be a forced-positioning overshoot rather than a regime change, so you *hold* through it.

- August 5: the Nikkei falls **−12.40%**. Your \$15,000 × (−12.40%) = **−\$1,860** on paper. Brutal.
- August 6: the Nikkei rebounds **+10.23%**. On the now-\$13,140 position, +10.23% = **+\$1,344**.
- Net over the two days: −\$1,860 + \$1,344 = **−\$516**, a ~3.4% drawdown — versus locking in the full **−\$1,860** if you had panic-sold at the August 5 close.

The lesson in dollars: correctly diagnosing the crash as a *fade* (forced unwind, not regime change) turned a −\$1,860 realized loss into a −\$516 round-trip — the read on *why* the move happened was worth roughly \$1,344 of recovered capital.

#### Worked example: Bitcoin in the August 2024 vacuum

Crypto is the purest expression of the liquidity-vacuum effect because it trades 24/7 with thinner depth than equity index futures and a much higher retail-leverage component. (Why crypto behaves as a macro-liquidity asset is covered in [crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset).)

- You hold **\$8,000 of Bitcoin** into the August 5 cascade. BTC falls **−15%**: \$8,000 × (−15%) = **−\$1,200**.
- Worse, if you were using **3× leverage** on that \$8,000, the −15% move becomes a −45% hit to your equity: −\$3,600 — and you would likely have been *liquidated* before the rebound, locking in the loss with zero participation in the recovery.
- Unleveraged and held, the position recovered much of the loss over the following sessions as the panic faded.

The lesson in dollars: in a liquidity vacuum, leverage is the difference between a painful but survivable −\$1,200 and a position-ending −\$3,600 liquidation — the vacuum punishes leverage precisely when the overshoot is largest.

### A CPI day that trended all session

Contrast the August 2024 *fade* with a genuine *trend*. On 2022-09-13, the August CPI printed 8.3% against 8.1% expected — a *hot* surprise, and a regime-relevant one: it told the market the Fed would have to hike harder and longer than hoped. There was no contradicting internal to rescue it; core inflation was hot too (6.3% vs 6.1%). The result was a move that ran one direction all session:

- The **S&P 500 fell −4.32%** — its worst day since June 2020.
- The **Nasdaq fell −5.16%** and **Bitcoin −9.4%**, with the dollar *up* +1.4% — every asset agreeing on a risk-off, higher-for-longer regime.
- The **2-year Treasury yield rose +18 basis points**, the bond market repricing a more aggressive Fed.

This was a trend, not a fade, because the print *changed the regime*: the internals confirmed the hot headline, the cross-asset moves all agreed (stocks down, dollar up, yields up — the coherent signature of a real repricing rather than a positioning blip), and the selling sustained through the close rather than exhausting on the first flood. A trader who shorted the spike and *held* — recognizing a regime change, not an overshoot — was rewarded; a trader who tried to fade it (buy the dip on the theory that "the first move is the overreaction") got run over all day.

### A regime that trended for a year: Vietnam in 2022

The spike-fade-trend frame works on any timescale, not just the intraday. Zoom out to the slowest possible reaction — a regime that re-rated an entire market over a year — and Vietnam in 2022 is the textbook case. As US inflation forced the Fed to hike aggressively, the dollar surged and the Vietnamese dong came under pressure. The State Bank of Vietnam (SBV) responded by hiking its refinancing rate from 4.0% to 5.0% on September 23 and to 6.0% on October 25 2022 — +200 basis points in a month to defend the currency. (The mechanism of how the SBV manages the dong and the credit ceiling is laid out in [Vietnam's monetary policy](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling).)

For the equity market this was an unambiguous regime change, and it trended for the entire year. The VN-Index, which had peaked near 1,528 in January 2022, ground down through every relief rally to a trough of **911 on November 15 2022** — a drawdown of roughly **−39%** — before closing the year at 1,007. Each individual SBV announcement and CPI print produced its own spike-fade-trend reaction, but the *dominant* signal was the regime: higher rates, a defended currency, foreign selling, and a falling index. A trader who treated each bounce as a fade-able overshoot within a downtrend regime was right far more often than one who treated each bounce as a new uptrend. The regime is the trend; the daily reactions are the texture inside it.

Note the cross-asset coherence even here: as US rates rose, the dollar strengthened, the dong weakened (USD/VND climbed from ~22,790 at end-2021 toward ~23,633 at end-2022), foreign investors pulled money from the index, and the SBV hiked into the pressure. Every piece agreed — the signature of a real regime, not a blip. (How foreign flows move the VN-Index is covered in [foreign flows, ETFs and the index effect in Vietnam](/blog/trading/vietnam-stocks/foreign-flows-etfs-and-the-index-effect-vietnam).)

#### Worked example: fading bounces inside a downtrend regime

You trade a \$10,000 book on the VN-Index through the 2022 downtrend regime and use the regime read to bias every reaction.

- You correctly treat a relief bounce as a fade within the downtrend and short \$10,000 of an index proxy; the bounce rolls over **−3%** as the regime reasserts: **+\$300**.
- A trader who instead bought that same bounce as a "new uptrend," ignoring the regime, eats the −3% rollover: **−\$300** — a \$600 swing between the two reads on the same move.
- Over a −39% peak-to-trough year, repeatedly siding with the regime rather than against it is the difference between compounding the downtrend and being chopped up by it.

The lesson in dollars: when a policy regime is set (SBV hiking, dollar strong, foreigners selling), the regime *is* the trade — fade the counter-trend bounces, and the \$600 swing per bounce compounds in your favor across the whole year.

## Common misconceptions

**"The first move is the right move."** This is the single most expensive myth in event trading. The first move is the *fastest* move, not the right one — it is algorithms reacting to a single headline number with no read on the internals. On 2024-08-05 the first move took the Nikkei to **−12.40%**; the right move, with hindsight, was the **+10.23%** reversal the next day. The first tick is information, but it is shallow information; the second move, when humans read the detail, is usually the more honest one.

**"A big number means a big move."** No — a big *surprise* means a big move. A blowout jobs report that matches an already-blowout consensus produces nothing, because the surprise is zero. Conversely, the one-tenth-of-a-point CPI miss on 2023-11-14 — a tiny number — drove the Russell 2000 up **+5.44%**, because the surprise was meaningful relative to a market positioned the other way. Trade the surprise, not the headline.

**"Fading the spike is easy money."** Sometimes the fade is the trade — but it is high-risk. As the worked example showed, fading a \$20,000 overshoot that reverts 1% makes +\$200, while the same fade against a move that *trends* 3% loses −\$600 — a 1-to-3 risk against you. Fade only when you have a real reason (contradicting internals, exhausted volume, pure positioning overshoot) and always with a tight stop. Fading every spike on the theory that "spikes overshoot" will eventually meet a regime-change print that trends straight through your stop.

**"You can get filled at the screen price."** Not in the liquidity vacuum. The book is at its thinnest exactly when you most want to trade, so a market order into the 8:30:00 print can slip **0.4% or more** — −\$100 on a \$25,000 order before the trade even works. The screen price five seconds before the print is fiction the moment the number drops.

**"Crypto is decoupled from macro events."** The August 2024 cascade took Bitcoin down **−15%** in the same session a US jobs report and a BoJ hike detonated the carry trade. Crypto is one of the *highest-beta* risk assets there is; when liquidity vanishes and risk comes off, it moves *more* than stocks, not less. It reacts to the same macro reaction function as everything else — often more violently.

## The playbook: trading the spike, the fade, and the trend

Here is the operational map — the if-then logic for actually trading a release rather than getting whipsawed by it.

**Before the print — set the scenarios, not a single bet.** Know the consensus. Know what is already *positioned* (is everyone leaning one way? then beware the overshoot-and-fade). Write down, in advance, what you will do in each scenario: hot surprise, cool surprise, in-line. The single biggest source of event-trading losses is improvising in the chaos of the first ten seconds. Decide before, execute after. For the structure of which releases matter and when, lean on [the macro calendar](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi); for how the FOMC specifically is read, see [trading the FOMC statement, presser, and dot plot](/blog/trading/macro-trading/trading-the-fomc-statement-presser-dot-plot).

**At the print — do not trade the first tick.** This is the discipline that separates professionals from the people feeding them. The first tick is the liquidity vacuum: thinnest book, worst slippage, shallowest information (headline only). Let the algorithms have it. You are not faster than them and you should not pretend to be. Wait for the book to refill and for the *internals* to be read.

**Seconds after — diagnose fade vs trend.** Run the checklist in order:

- *Did the print plausibly change the policy regime?* If yes, lean trend.
- *Do the internals confirm the headline (core, revisions, composition)?* Confirm → trend; contradict → fade.
- *Do the cross-asset moves agree?* Stocks up, yields down, dollar down all together is a coherent regime signal (trend); stocks up while yields and the dollar disagree is incoherent (likely a fade or a head-fake). The broader cross-asset framework is in [risk-on, risk-off: how money rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates).
- *Is volume sustaining the move or drying up?* Sustaining → trend; drying up → fade.

**Position to the diagnosis, with the right risk.**

- If you read a **trend** (regime change + confirmation + agreement + sustained volume): enter *with* the move after the first tick, size for a multi-hour-to-multi-day hold, and place your stop on the *other side of the pre-event range* (a trend should not give back the whole move). The reward is the regime re-rating — the +5.54% S&P day that was the start, not the end.
- If you read a **fade** (overshoot + contradicting internals + exhausted volume + pure positioning): enter *against* the spike near its extreme, but with a **tight stop** just beyond the spike high/low, because if you are wrong it is a trend and it will run. Fades are low-margin: the worked example's +\$200 win against a −\$600 loss means you need a high hit-rate and ruthless stops.
- If you **cannot tell**: stand aside. There is another release next week. The trades you skip in ambiguity are the cheapest risk management there is.

**Size for the volatility, not the conviction.** The expected move around a major release is large — index options price in roughly ±1.2% for an S&P event and ±4% for a Bitcoin event in the curated examples. Size your position so that the *expected* event move is survivable, not so that the move you *hope for* is maximized. A position that is correct on direction but too large to hold through the whipsaw is a losing position.

#### Worked example: sizing a trend trade vs a fade trade

You have a \$50,000 trading account and a 1%-of-account risk budget per trade (**\$500** max loss).

- **Trend trade** (high conviction, wide stop): you enter long after the first tick with a stop on the far side of the pre-event range, say a **3%** stop. To risk \$500 at a 3% stop, your position is \$500 / 0.03 = **\$16,667**. If the regime-change trend runs +5% over the next two days, you make \$16,667 × 5% = **+\$833**.
- **Fade trade** (lower conviction, tight stop): you short the spike with a **1%** stop just beyond the high. To risk the same \$500 at a 1% stop, your position is \$500 / 0.01 = **\$50,000** — but that is your whole account, so you cap it and take a smaller \$25,000 position, risking only \$250. If the 1% fade works, you make \$25,000 × 1% = **+\$250**.

The lesson in dollars: the *stop distance* sets the position size, not your excitement — the wide-stop trend trade can be larger and aims for the +\$833 regime move, while the tight-stop fade is necessarily smaller and grinds out +\$250, and both keep your loss capped at the same risk budget.

The thread running through all of it: **react to the reaction, not the number.** The number is just the trigger. The trade is in reading the sequence it sets off — the spike, then the fade or the trend — and having the discipline to let the first tick go, diagnose what kind of move you are in, and size to survive being wrong. Do that, and the news stops being a thing that happens *to* your account and becomes a thing you trade.

## Further reading and cross-links

- [Risk-on, risk-off: how money rotates between assets](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates) — the cross-asset framework for reading whether the internals and the dollar and yields *agree* with a stock move (the coherence test for trend vs fade).
- [Carry-trade unwinds: 1998, 2008, 2024 — when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) — the full mechanism behind the August 2024 cascade and why forced-positioning moves fade rather than trend.
- [Following the flows: positioning, COT, and dealer hedging](/blog/trading/macro-trading/following-the-flows-positioning-cot-dealer-hedging) — how to read the pre-event positioning that determines whether a surprise overshoots-and-fades or trends.
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — why every asset spikes together in the liquidity vacuum, as Bitcoin, the Nikkei, and the S&P all did on August 5 2024.
- [The macro calendar: CPI, NFP, FOMC, PMI](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi) — which releases matter, when they drop, and what the consensus represents.
- [Inflation and the Fed reaction function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot) — how a CPI print changes the policy regime, which is what separates a trend from a fade.

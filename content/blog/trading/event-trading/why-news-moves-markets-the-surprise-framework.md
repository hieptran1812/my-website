---
title: "Why News Moves Markets: The Surprise Framework"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Markets trade expectations, not numbers — the price already contains the consensus, and only the surprise (actual minus consensus) moves it on release. This is the mental model the whole series is built on."
tags: ["event-trading", "macro", "cpi", "surprise", "consensus", "reaction-function", "cross-asset", "stocks", "bonds", "crypto", "volatility"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Markets trade *expectations*, not numbers. The price already contains the consensus forecast, so only the **surprise** — the gap between what actually prints and what was expected — moves price on release. The reaction is the trade, not the number.
>
> - A macro release (CPI, jobs, a Fed decision) is a scheduled number the whole market has already forecast. The forecast — the **consensus** — is baked into the price *before* the print lands.
> - On release, every asset reprices off the **surprise = actual − consensus**, not the headline level. One print hits stocks, bonds, the dollar, gold and crypto at the same instant — a single number, a cross-asset shockwave.
> - The same surprise can rally stocks in one regime and crash them in another. The **reaction function** decides the sign and size. Your job is to trade the *reaction*, with a plan for each surprise scenario and a clear invalidation.
> - The one number to remember: on 13 September 2022, August CPI came in just **+0.2pp** above consensus — and the S&P 500 fell **−4.32%**, its worst day since June 2020.

On the morning of 13 September 2022, the U.S. Bureau of Labor Statistics released the August Consumer Price Index. Inflation came in at 8.3% year-over-year. The consensus forecast — the average of professional economists' predictions, compiled by the data wires the night before — was 8.1%. So the actual number missed the forecast by a grand total of 0.2 percentage points. Two tenths of one percent hotter than expected. A rounding error to most people.

The S&P 500 fell **−4.32%** that day. The Nasdaq dropped **−5.16%**. Bitcoin, which trades 24 hours a day and supposedly answers to no one, fell roughly **−9.4%**. It was the worst single session for U.S. stocks since June 2020, the depths of the pandemic crash. Trillions of dollars in market value evaporated over a 0.2-point miss on a number that was already known to be high.

Here is the puzzle this whole series exists to answer: *why did such a small miss cause such an enormous move?* If inflation was already running at 8%, and everyone knew it, why did one more decimal place detonate the tape? The answer is the single most important idea in event-driven trading, and once you see it, you can never un-see it: **the market had already priced the consensus.** The 8.1% everyone expected was *already in the price*. What moved markets was not the 8.3% — it was the *surprise*, the 0.2-point gap between what printed and what was priced. Markets do not trade numbers. They trade the *difference* between reality and expectation.

![Pipeline showing expectations priced in then release then surprise then repricing across assets](/imgs/blogs/why-news-moves-markets-the-surprise-framework-1.png)

This is the founding post of *Trading the News: How Markets React to Macro Events*. Over the series we will take every major scheduled release apart — CPI, the jobs report, the Fed's rate decisions, central-bank meetings around the world, and Vietnam's own SBV and VN-Index machinery — and answer the practical question each time: *when this number hits the tape, how do stocks, bonds, the dollar, gold and crypto move, and how do you trade it?* But none of that makes sense until you internalize the one idea on which all of it rests. So this post does one job, slowly and completely: it teaches you the **surprise framework**. By the end you will understand why a 0.2-point miss crashed the market, why the *same* miss could rally it in a different year, and why professionals say *the reaction is the trade, not the number.*

## Foundations: how a number becomes a market move

Before we go deep, we need a shared vocabulary. Every term here gets used in every subsequent post, so we define each one from zero, with an everyday picture, before any math. If you have no finance background, this section is the load-bearing wall — read it carefully.

### A scheduled release is a number with an audience that has already guessed

A **macro release** is an official economic statistic, published on a known date and time, that tells the market something about the economy: how fast prices are rising (CPI), how many people got hired last month (the jobs report, or "non-farm payrolls" / NFP), what interest rate the central bank just set (the Fed decision), and so on. The defining feature for a trader is that these are *scheduled*. The market knows the August CPI report comes out at 8:30 a.m. Eastern on a specific Tuesday weeks in advance. There is a clock ticking toward it.

Because the date is known, the number is *anticipated*. And because it is anticipated, an entire forecasting industry exists to guess it. Banks, research shops and economists each publish an estimate. The data wires (Bloomberg, Reuters) collect those estimates and compute the average. That average is the **consensus** — the market's official best guess for what the number will be.

Say the weather forecast says tomorrow will be 30°C. You dress for 30°C, the city's air-conditioning load is planned for 30°C, ice-cream vendors stock up for 30°C. If tomorrow *is* 30°C, nothing about your day changes — the forecast was right, you were already prepared. The only thing that disrupts your day is if it comes in at 38°C or 18°C. Markets work exactly the same way. The consensus is the forecast, and the market has already "dressed" for it — positioned its portfolios for that number. A print that *matches* consensus changes nothing, because everyone was already prepared. Only a *deviation from the forecast* forces anyone to do anything.

### "Priced in" means the expectation is already in the number on your screen

When traders say a number is **priced in**, they mean the market has already moved as if that number were true. If everyone expects CPI to be 8.1%, then in the days before the release, traders have already sold the assets that 8.1% inflation hurts and bought the ones it helps. The S&P 500 level you see on the screen the morning of the release *already reflects* 8.1% inflation. There is nothing left to react to if 8.1% is what prints.

This is the most counterintuitive idea for newcomers, so sit with it: **the current price is not a blank slate waiting for news. It is a running tally of every expectation the market holds.** The price is the market's collective forecast, expressed in dollars. News only moves the price to the extent it *changes* that forecast.

How does the expectation physically get *into* the price? Through the actions of thousands of traders in the days and weeks before the release. A fund that believes inflation will print at 8.1% does not sit on its hands waiting for the number; it positions *now* — it sells the rate-sensitive growth stocks and the long-dated bonds that 8% inflation will hurt, and it does so today, at today's price, before the report. Every participant who holds a view acts on it ahead of time, and the aggregate of all that pre-positioning is the price you see on the screen the morning of the release. By the time 8:30 a.m. arrives, the 8.1% expectation has been bought and sold a million times over. It is, in the most literal sense, *already in the number on your screen.* This is also why markets are sometimes called "discounting machines": they discount — pull forward into today's price — everything the crowd expects to happen tomorrow.

A corollary that trips up beginners: **the price can move a lot in the days before a release and barely move on the release itself**, if the pre-positioning was aggressive. The market can sell off all week in dread of a hot CPI print, then *rally* when the print lands merely in-line — because the feared surprise didn't materialize and the pre-positioned shorts cover. The "news" was the absence of bad news. We will see this pattern repeatedly; it is pure surprise framework, and it is invisible to anyone watching only the headline number.

### The surprise is the only part that is new information

The **surprise** is the difference between what actually prints and what was priced in:

> **surprise = actual − consensus**

A *positive* surprise on inflation means the number came in *hotter* (higher) than expected. A *negative* surprise means *cooler* (lower). For August 2022 CPI: actual 8.3% − consensus 8.1% = **+0.2pp surprise** (a hot surprise). For a jobs report, a positive surprise means more jobs were created than expected.

The surprise is the *only* genuinely new information in the release. The consensus was already known and already priced. The surprise is the part the market had not seen, and therefore the part it must now react to. This is why the *magnitude of the move* tracks the *size of the surprise*, not the *level of the number*. An 8.3% inflation print sounds catastrophic in absolute terms, but if the consensus had been 8.3% it would have moved markets very little — the catastrophe was already in the price. The crash came from the 0.2-point gap.

Whether a surprise is "good" or "bad" for an asset is a separate question we will spend the rest of the series on. For now, just hold the mechanical definition: surprise is actual minus consensus, and surprise is what is new.

### Beat, miss, and the knee-jerk versus the reaction function

Two more pairs of terms. First, **beat** and **miss**. These come from the corporate-earnings world but apply everywhere: a release "beats" if it comes in stronger than consensus and "misses" if it comes in weaker. (Careful: for inflation, a "hot" number is technically a *beat* on the headline figure but is usually *bad* for risk assets — strength and goodness are not the same thing, which is the whole reason the reaction function matters.)

Second, **knee-jerk** versus **reaction function**. The **knee-jerk** is the instant, reflexive first move in the seconds after the print — algorithms fire on the headline number before any human has read the details. The **reaction function** is the deeper logic that decides whether the move *sticks*, *reverses*, or *extends* — it is the market's rule for translating "this kind of surprise, in this regime" into "this sign and size of move." A hot inflation surprise in 2022 (when the Fed was hiking aggressively and the market feared more hikes) was deeply risk-off. The same hot surprise in a year when inflation is not the dominant worry might barely register. The number is the input; the reaction function is the machine that turns it into a price move.

That is the entire foundation. A scheduled release carries a consensus that is already priced in; the surprise is actual minus consensus and is the only new information; and the reaction function decides what the surprise does to each asset. Everything below is an elaboration of those three sentences.

## The surprise, made precise: actual minus consensus

Let's nail down the surprise with a real worked example, because the framework lives or dies on this one subtraction.

For the August 2022 CPI report, the headline year-over-year inflation rate printed at 8.3%. The consensus, as collected by the wires the prior evening, was 8.1%. The surprise on the headline was therefore +0.2 percentage points (pp) — hot. But there was a second, arguably more important surprise hiding underneath. **Core CPI** — the index stripped of volatile food and energy prices, which the Fed watches because it reflects the *underlying* trend — printed at 6.3% against a consensus of 6.1%. Another +0.2pp hot surprise, and on the number that matters most for Fed policy.

So the market woke up that morning positioned for inflation that was high but *finally cooling*. Gasoline prices had fallen all summer, and the comfortable story was that the inflation problem was rolling over and the Fed could soon stop hiking. The print said the opposite: core inflation, the sticky underlying kind, was *still accelerating*. The surprise was small in points but enormous in *meaning* — it killed the "inflation is rolling over" narrative that the entire summer rally had been built on.

#### Worked example: turning a percentage move into real money

Percentages are abstract; money is not. Suppose you held a simple S&P 500 index position worth **\$25,000** going into that print — say, through an index fund or ETF. The index fell −4.32% on the day. Your loss:

- Position value: **\$25,000**
- Same-day move: −4.32%
- Dollar change: \$25,000 × (−0.0432) = **−\$1,080**

You lost **\$1,080** in a single session, on a position you might have held for years, because a number missed its forecast by two tenths of a point. That is the surprise framework with a dollar sign attached: the move was not about the 8.3% level, it was about the 0.2-point gap — and that gap cost a \$25,000 holder \$1,080.

The lesson is not "CPI is dangerous." The lesson is that the *price already reflected the consensus*, so the only thing that could hurt or help you was the *deviation* from it. If CPI had printed exactly 8.1%, that \$25,000 would very likely have been roughly flat on the day. Same scary 8% inflation level, no surprise, no move.

### Headline versus internals: not all of a release is equally surprising

A release is rarely a single number — it is a packet of numbers, and the market does not weight them equally. CPI ships a headline (all prices), a core figure (ex-food-and-energy), and a breakdown by category (shelter, used cars, airfares, medical care). The jobs report ships the headline payrolls count, the unemployment rate, average hourly earnings, and revisions to prior months. The market's reaction depends on *which* part surprised and *how much that part matters in the current regime.*

In September 2022, the headline and core both surprised hot by +0.2pp, but it was the *core* surprise that did the damage, because core is what the Fed steers by — gasoline prices (in the headline but not core) are volatile and the Fed looks through them. A hot headline driven entirely by a one-off energy spike, with core in-line, would have produced a far smaller reaction. So the trader's surprise is not one subtraction but several, weighted by relevance: *which line item surprised, and does the reaction function care about that line item right now?*

This is why the knee-jerk and the considered move so often diverge. Algorithms fire on the headline in the first heartbeat; humans then read the internals and decide whether the headline surprise was "real" (driven by the sticky components the Fed cares about) or "noise" (driven by a volatile one-off). When the headline and the internals disagree, the fade is violent — the spike traded one number, the retracement trades the other.

#### Worked example: a headline-versus-core split in dollars

Suppose a CPI report prints with a hot headline surprise but an in-line core, and the knee-jerk spikes the S&P down −0.8% in the first seconds, then fades back to −0.2% once traders see core was fine. A trader who shorted **\$50,000** of S&P exposure on the spike and covered at the close captured:

- Entry on the −0.8% spike, exit near the −0.2% close → captured roughly the **0.6%** retracement on the fade
- \$50,000 × 0.6% = **+\$300** for the fade trader

Meanwhile a "buy-and-hold" investor with the same **\$50,000** simply rode the −0.2% close: \$50,000 × (−0.2%) = **−\$100**. Same report, same asset, opposite outcomes — because one trader read the internals and the other read only the headline. The number was identical for both; the *reaction* was the trade.

## Why only the surprise matters

Here is the cleanest way to prove the framework: hold the *direction* of the surprise up against the *direction* of the market's move, across multiple episodes, and watch them line up. If markets traded the *level* of inflation, then every high-inflation print would crash stocks. They don't. What predicts the move is the *sign of the surprise*.

![Scatter of CPI surprise in percentage points against S&P 500 same-day return](/imgs/blogs/why-news-moves-markets-the-surprise-framework-2.png)

Look at the three CPI days plotted above. On 13 September 2022, a +0.2pp *hot* surprise (a positive number on the horizontal axis) sent the S&P 500 down −4.32%. On 10 November 2022, a −0.2pp *cool* surprise (negative on the axis) sent it *up* +5.54%. On 14 November 2023, a milder −0.1pp cool surprise produced a +1.91% gain. Three points, and they trace a clean upward-to-the-left, downward-to-the-right line: **cool surprises (left) rallied stocks; hot surprises (right) crashed them.** The horizontal position — the surprise — explains the vertical position — the move. The *level* of inflation on those three days (8.3%, 7.7%, 3.2%) does not; it fell steadily across all three while the moves alternated in sign.

This is the core empirical claim of the whole series, and it is why a trader watching a release does not ask "is this number high or low?" but "is this number *higher or lower than expected*?" The first question is about the economy. The second is about the *price*, and price is all a trader can trade.

### Why the magnitude tracks the surprise, not the level

There is a clean logic to *why* the size of the move scales with the size of the surprise. The price is the present value of a long stream of future expectations. A release updates that stream only by the amount it changes the forecast. A print exactly at consensus changes the forecast by zero, so it moves the price by zero (in principle). A print that misses consensus by a little changes the forecast by a little. A print that misses by a lot — or, more importantly, *changes the narrative* about the future path — changes the forecast by a lot, and the price gaps accordingly.

September 2022 is the perfect illustration of why the *point size* of a surprise undersells its *informational size*. The headline missed by only 0.2pp, but the surprise carried a much larger payload: it killed the "peak inflation" thesis, which in turn changed the market's entire expected *path* of future rate hikes — not just the next meeting but the whole next year. The reaction priced not the 0.2 points but the cascade of revisions the 0.2 points triggered. This is why a small surprise can produce an outsized move: the surprise is the *trigger*, but what reprices is the entire chain of expectations it disturbs.

### The efficient-market intuition, stated plainly

Underneath all of this is a famous idea — the **efficient-market hypothesis** — which says that prices already reflect all available information. You do not need to believe markets are *perfectly* efficient (they clearly are not) to use its one practical consequence: **if information is public and anticipated, it is probably already in the price, so it cannot be your edge.** The CPI consensus is the most public information on earth; trading on "inflation is high" is trading on something every participant already knows and has already acted on. The only tradable information in a scheduled release is the part nobody knew in advance — the surprise — and even that, the fastest algorithms price in milliseconds. Your edge, if you have one, is in the *reaction*: reading the regime, the positioning, and the spike-fade-trend dynamics better than the crowd. The number is everyone's; the reaction is where skill lives.

### The consensus is a moving target — and that is a trap

There is a subtle wrinkle that catches beginners. The consensus is not the only expectation in the price. Markets also build a **"whisper number"** — an unofficial expectation that drifts from the published consensus in the hours before a release, driven by recent data, positioning, and chatter. Sometimes the official consensus says 8.1% but the market is quietly braced for 8.4%, because a hot producer-price print landed two days earlier. In that case an 8.3% actual is a *cool* surprise relative to the *whisper*, even though it is a *hot* surprise relative to the *published consensus* — and the market can rally on a number that "beat" the official forecast.

You do not need to master the whisper number yet. Just hold the deeper point it teaches: **the relevant benchmark is whatever the price has actually discounted, which is not always the printed consensus.** When a market moves "the wrong way" on a release — rallies on a hot inflation print, say — the usual explanation is that the price had braced for something even worse, so the actual was a relief. The framework still holds; you just had the wrong number for "expected." A later post in this series is devoted entirely to reading what is *really* priced in, beyond the headline consensus.

## The three-layer model: expectation, surprise, reaction function

So far we have two layers: the **expectation** (priced in) and the **surprise** (the new bit). But the surprise alone does not tell you the *sign* of the move. A hot inflation surprise is bad for stocks in 2022 but, as we will see, could be neutral or even good in other regimes. We need a third layer between the surprise and the price: the **reaction function**.

![Layered graph from expectation to hot or cool surprise to reaction function to risk-on or risk-off move](/imgs/blogs/why-news-moves-markets-the-surprise-framework-3.png)

Read the figure top to bottom. The **expectation** is the consensus, already in the price. The release resolves into a **surprise** — either hot (actual above consensus) or cool (actual below). That surprise feeds the **reaction function**: the market's regime-dependent rule for what the surprise *means*. And only at the bottom does it become a **move** — risk-on (stocks up, yields down) or risk-off (stocks down, yields up). The crucial insight is that the same surprise enters the top of the reaction function and can exit the bottom with *either* sign, depending on the regime the function is set to.

In 2022, the reaction function was: *inflation is the enemy; the Fed will hike until it breaks; therefore any hot inflation surprise means more hikes, higher rates, lower asset valuations — sell everything.* This is the famous **"good news is bad news"** regime, where strong economic data is bad for stocks because it implies tighter policy. Flip the regime — say the economy is the worry and the Fed is on the reader's side, cutting rates to support growth — and the function inverts: now a *weak* data surprise is bad (recession fear) and the market hangs on every sign of *strength*. Same data, opposite reaction, because the function in the middle changed.

We will not fully derive the reaction function here — that is its own dedicated post, and the mechanics of *how* the Fed translates inflation into rate decisions are covered in depth in the macro series on [the Fed's reaction function and the dot plot](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot) and in [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates). For now, just hold the structure: **number → surprise → reaction function → move.** When you find yourself surprised that a "good" number tanked the market, the answer is always in the reaction function: in *this* regime, that surprise meant something bad.

#### Worked example: the same surprise, opposite money

Hold the structure with dollars. Take the identical event — a CPI print +0.2pp hotter than consensus — landing in two different regimes, on the same **\$25,000** S&P 500 position:

- **2022 regime (inflation is the enemy):** hot surprise → fear of more Fed hikes → risk-off. The S&P falls ~4%. Your \$25,000 → about **−\$1,000**.
- **A soft-landing regime (growth is the worry, inflation tame):** a +0.2pp hot surprise on a *low* base might barely register, or even reassure markets that demand is healthy. The S&P could be roughly flat to slightly up — call it +0.5%, or about **+\$125** on your \$25,000.

The surprise is byte-for-byte identical; the dollar outcome flips from a four-figure loss to a small gain. **You cannot trade the surprise without knowing the regime** — that is why the reaction function is a layer, not a footnote.

## One print, every market: the cross-asset shockwave

A macro release does not hit one asset. It hits *all* of them, in the same instant, because they are all priced off the same underlying variables — interest rates, growth expectations, and the dollar. When a hot CPI print raises the market's expected path of interest rates, that single change ripples through every asset class at once.

![Bar chart of cross-asset same-day moves to the hot August 2022 CPI print](/imgs/blogs/why-news-moves-markets-the-surprise-framework-4.png)

Here is the full cross-asset reaction to that 13 September 2022 print. The S&P 500 fell −4.32% and the Nasdaq −5.16% (tech stocks are "long-duration" — their value sits in far-future profits, which higher rates discount more heavily, so they fall hardest). The Dow fell −3.94%. Bitcoin, the supposed inflation hedge and uncorrelated asset, fell about −9.4% — in a risk-off panic, crypto trades like the most speculative tech stock there is, not like gold. The U.S. dollar (DXY) *rose* +1.4%, because higher U.S. rates make dollar-denominated assets more attractive and pull global capital in. Gold barely moved, −0.4%. And though they are not on this particular bar set, Treasury yields jumped — the 2-year yield, the maturity most sensitive to Fed policy, rose +18 basis points (a basis point is one hundredth of a percentage point) as the market priced in a higher rate path.

That is the shape of a risk-off shockwave: **stocks down, crypto down hardest, dollar up, bonds sold (yields up), gold roughly flat.** One number, six markets, one synchronized lurch. This cross-asset pattern — which assets move together and which move opposite — is the heart of the **risk-on / risk-off** framework, covered in depth in the macro post on [how money rotates between risk-on and risk-off](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates). A dedicated post later in this series traces exactly *how* one print transmits from the rate market out to every other asset; for now, the takeaway is that you must watch the whole board, not just the asset you trade.

### The transmission chain: from one number to every asset

The reason a single CPI print hits everything at once is that it enters through one door — the **expected path of interest rates** — and that path is the input to every other asset's valuation. Trace it:

1. **The surprise updates the rate path.** A hot inflation surprise makes the market price more (or higher-for-longer) rate hikes. The 2-year Treasury yield, the maturity most sensitive to near-term Fed policy, moves first and fastest — +18bp on the September 2022 print.
2. **Higher rates re-discount stocks.** A stock is worth the present value of its future profits. Discounting those future profits at a higher rate makes them worth less *today* — and the further in the future the profits sit, the more a higher discount rate bites. That is why the Nasdaq (long-duration, profit-far-in-the-future tech) fell −5.16% versus the Dow's −3.94%: same surprise, but the long-duration assets reprice harder.
3. **Higher U.S. rates pull capital into the dollar.** Money chases yield; higher U.S. rates make dollar deposits and Treasuries more attractive, so global capital rotates into dollars and the DXY rose +1.4%. A stronger dollar is, in turn, a headwind for commodities and emerging-market assets priced in dollars.
4. **Risk-off liquidation hits the highest-beta assets hardest.** When the rate shock turns sentiment risk-off, leveraged and speculative positions get cut first. Bitcoin, the highest-beta liquid asset most retail traders hold, fell −9.4% — not because anything changed about Bitcoin, but because it sits at the far end of the risk spectrum and gets sold first in a scramble for safety.

One surprise, one rate-path update, four knock-on repricings — all in the same minute. This is why an event trader watches the *whole* board: the 2-year yield often tells you the sign of the surprise before the stock index has finished gapping, and the dollar confirms the risk-off read.

### The Vietnam angle: the same shockwave, transmitted abroad

U.S. macro surprises do not stop at the U.S. border — they radiate to every market that prices off the dollar and global risk appetite, including Vietnam's VN-Index. When a hot U.S. CPI print drives the dollar up and global risk-off, the pressure shows up in Vietnam through two channels: **foreign capital flows** (global funds pull money out of emerging markets like Vietnam and back into the strengthening dollar) and **the currency** (the State Bank of Vietnam, the SBV, must defend the dong against a rising dollar, which can force it to tighten). In the autumn of 2022 — the same window as the hot-CPI shock — the SBV raised its refinancing rate from 4.0% to 6.0% in two moves to defend the dong, and the VN-Index fell to a trough of 911 in mid-November, down roughly 39% from its January 2022 peak near 1,528. The U.S. surprise framework, transmitted through the dollar and foreign flows, reached all the way to Ho Chi Minh City. Dedicated Vietnam posts later in this series trace the SBV reaction function and the foreign-flow channel in detail; the mechanism behind it is developed in the macro and finance series. The point here: **a U.S. release is a global event, and the surprise framework is the lens for all of it.**

#### Worked example: the cross-asset hit in dollars

Suppose that morning you held three positions: **\$25,000** in an S&P 500 fund, **\$10,000** in Bitcoin, and **\$5,000** in a gold position. The print lands hot. Your book:

- S&P 500: \$25,000 × (−4.32%) = **−\$1,080**
- Bitcoin: \$10,000 × (−9.4%) = **−\$940**
- Gold: \$5,000 × (−0.4%) = **−\$20**

Total: about **−\$2,040** across the book. Notice the Bitcoin position, less than half the size of the stock position in dollars, lost almost as much — because crypto's *reaction* to a risk-off surprise was more than twice as violent. The intuition: in a cross-asset shock, your real risk is not the dollar size of each position but the *size times the reaction*. A "diversified" book of stocks plus crypto was not diversified at all that day — both legs were just different-sized bets on the same surprise.

## The anatomy of the move: spike, fade, trend

We have been talking about "the same-day move" as if it were one thing. It is not. The reaction to a release unfolds in stages, and a trader who understands the stages sees opportunities (and traps) that the close-to-close number hides.

![Timeline of the reaction in three stages: knee-jerk spike, fade, trend](/imgs/blogs/why-news-moves-markets-the-surprise-framework-5.png)

In the first **0–2 seconds** after 8:30:00 a.m., the **knee-jerk spike** fires. Algorithms parse the headline number off the wire and trade it instantly, faster than any human can read. If the surprise is hot, futures gap down violently in the first heartbeat. This first move is pure reflex on the *headline* — it has not digested the internals, the revisions, or the context.

Over the next **2 to 30 minutes**, the **fade** often happens. Humans and slower models read past the headline: maybe the hot CPI was driven by a one-off jump in airfares, maybe last month was revised down, maybe the whisper number was even worse and this is a relief. The initial overshoot retraces some or all of the way. The knee-jerk move and the considered move frequently disagree, and the gap between them is where a lot of money changes hands.

Then over **hours to days**, the **trend** sets in — or doesn't. If the surprise genuinely changed the regime (as the September 2022 core-inflation print did), the move *extends*: the repricing of the entire Fed path plays out over the following sessions, and the close-to-close −4.32% was just day one of a larger leg lower. If the surprise was noise, the trend never materializes and price drifts back toward where it started.

This three-stage anatomy — **spike, fade, trend** — is why professionals insist *the reaction is the trade, not the number.* Two traders can see the identical CPI print; the one who shorted the knee-jerk spike and the one who bought the fade can *both* make money, and the one who simply "knew inflation was high" can lose. The number is the same for everyone. The edge is in reading the reaction. An entire post in this series dissects the spike-fade-trend mechanics and how to trade each stage; here we just plant the flag that the move has structure.

### Volatility before and after: the vol ramp and the vol crush

There is a second clock running into every scheduled release: the **implied volatility** clock. Implied volatility is the market's expectation of how much price will move, embedded in the price of options. Because a release is a known source of large potential movement, options that expire just after it get *more expensive* in the days before — the market pays up for protection and for bets on a big move. This is the **volatility ramp**: implied vol grinds higher into the event as uncertainty peaks.

Then the number prints, the uncertainty resolves, and implied volatility *collapses* — the famous **vol crush**. The event is now in the past; there is no longer a known shock to price, so the options that were expensive an hour ago are suddenly cheap. This matters enormously for anyone trading options around events: you can be *right about the direction* and still lose money, because the vol crush eroded your option's value faster than the underlying moved in your favor. The expected move we will use in the playbook comes straight from this — it is the market's pre-event vol, priced into the straddle. The deep mechanics of how volatility is priced live in the quant-finance series; here the takeaway is that *uncertainty itself is an asset that rises into an event and falls after it.*

## How it reacted: real episodes

Enough theory. Let's run the framework over three real, dated CPI sessions and watch it explain each one. These are among the most documented sessions in modern markets, and together they make the case that the *surprise*, filtered through the *regime*, predicts the move.

![Bar chart of S&P 500 same-day move on three CPI days, hot versus cool](/imgs/blogs/why-news-moves-markets-the-surprise-framework-6.png)

### Episode 1 — 13 September 2022: the hot print that broke the tape

August CPI: headline 8.3% vs 8.1% consensus (+0.2pp hot), core 6.3% vs 6.1% (+0.2pp hot). Regime: the Fed was mid-way through the most aggressive hiking cycle in 40 years, and the market's entire hope was that inflation had peaked. The surprise was small in points but it *invalidated the peak-inflation thesis* — core was still rising. The reaction function ("hot inflation → more hikes → sell risk") fired at full force.

Result: **S&P 500 −4.32%, Nasdaq −5.16%, Bitcoin ≈ −9.4%, dollar +1.4%, 2-year yield +18bp.** Worst S&P day since June 2020. The framework's verdict: a hot surprise in a "good-news-is-bad" regime, on the most policy-relevant number, produced a textbook risk-off cross-asset shockwave. The size of the move came from the *meaning* of the surprise (peak-inflation thesis dead), not from the 0.2 points themselves.

### Episode 2 — 10 November 2022: the cool print that ignited a rally

October CPI: headline 7.7% vs 7.9% consensus (−0.2pp cool), core 6.3% vs 6.5% (−0.2pp cool). The mirror image of September. Inflation was *still* running near 8% — by absolute level, still alarming — but it printed *cooler than expected*, and on the core number the market most cared about. In the same 2022 regime, a cool surprise meant *fewer* future hikes, which is risk-on rocket fuel.

Result: **S&P 500 +5.54%, Nasdaq +7.35%, 10-year yield −28bp, dollar −2.1%, gold +2.8%.** One of the best S&P days of the entire era. Note that the inflation *level* (7.7%) was barely lower than September's 8.3% — by level, both days were "high inflation." But the *surprises* had opposite signs, and so did the moves. This single pair of days is the cleanest possible proof that **markets trade the surprise, not the level.**

#### Worked example: the cool-CPI rally in dollars

Take the same **\$25,000** S&P 500 position through 10 November 2022:

- Position value: **\$25,000**
- Same-day move: +5.54%
- Dollar change: \$25,000 × (+0.0554) = **+\$1,385**

A −0.2pp cool surprise put **+\$1,385** into that account in one session — almost the exact mirror of the −\$1,080 the +0.2pp hot surprise had taken out two months earlier. Same position, same asset, near-identical inflation *level*, opposite surprise, opposite dollar outcome. If you remember one comparison from this entire post, make it this one: **−\$1,080 on the hot print, +\$1,385 on the cool print — the level was the same, the surprise was not.**

### Episode 3 — 14 November 2023: the cool print and the rate-sensitive winners

October 2023 CPI: headline 3.2% vs 3.3% consensus (−0.1pp cool), core 4.0% vs 4.1% (−0.1pp cool). By now inflation had fallen to the low 3s, and the regime had softened toward "the Fed is nearly done." A mild cool surprise reinforced the "hiking is over" story.

Result: **S&P 500 +1.91%, Nasdaq +2.37%, 10-year yield −19bp** — and most strikingly, the **Russell 2000** (small-cap stocks) jumped **+5.44%**. Why did small caps fly while the S&P gained a comparatively modest +1.91%? Because small companies carry more floating-rate debt and are more sensitive to the cost of borrowing — they are the most "rate-sensitive" corner of the equity market. When a cool inflation surprise pulled the expected rate path down, the assets most punished by high rates rallied hardest. This is the cross-asset framework at finer resolution: the surprise didn't just move "stocks," it moved the *most rate-sensitive* stocks most, exactly as the reaction function predicts.

### Episode 4 — the Fed decision: when the surprise is in the guidance, not the rate

CPI is data; the Fed decision is a *choice*. But the framework is identical — the market prices the expected decision in advance, and only the surprise moves price. The twist is that the rate change itself is usually *not* the surprise (the market almost always knows what the Fed will do at this meeting); the surprise lives in the **guidance** — the projections, the "dot plot," and the press-conference tone that signal the *future* path.

Two episodes make the point. On 16 March 2022, the Fed delivered its first rate hike of the cycle (+25bp). A first hike sounds like bad news, yet the S&P 500 *rose* +2.24% — because the hike was fully expected and the guidance removed uncertainty; the relief of clarity outweighed the hike itself. Then on 19 December 2018, the Fed hiked +25bp again as expected, but the *dot plot* still penciled in two more hikes for 2019 — more hawkish than a market already worried about slowing growth wanted. The S&P, up earlier in the session, reversed to close **−1.54%**, and that hawkish guidance helped extend a Q4 2018 drawdown that ran to roughly −19.8% from the September high, including a −2.71% plunge on Christmas Eve. Same kind of action (+25bp), opposite reactions — because the *surprise was in the guidance*, and the guidance changed the expected path. The mechanics of trading the Fed's statement, presser and dot plot are a dedicated macro post; the lesson here is that for a decision event, you locate the surprise in the *forward guidance*, not the headline action.

### Episode 5 — the jobs report and the day the surprise cascaded globally

The jobs report (non-farm payrolls) is the other 8:30 a.m. heavyweight, and it shows both the "good news is bad" regime *and* how a surprise can cascade across the globe. On 3 February 2023, payrolls printed +517,000 against a consensus near +187,000 — a colossal upside surprise. In the inflation-fighting regime, a labor market that hot meant the Fed would stay restrictive, so the S&P *fell* −1.04% and the 2-year yield jumped +18bp. Strong economy, weak stocks — the reaction function inverted "good" into "bad."

Eighteen months later the regime had flipped. On 2 August 2024, payrolls came in *weak* at +114,000 against ~175,000 expected, and the unemployment rate ticked up to 4.3%, tripping a recession-warning rule. Now the worry was growth, not inflation, so a *weak* surprise was the bad one: the S&P fell −1.84% and the 2-year yield *dropped* −28bp as the market raced to price rate *cuts*. That weak print, colliding with a Bank of Japan rate hike days earlier, detonated a global carry-trade unwind the following Monday (5 August 2024): the Nikkei fell **−12.4%** (its worst day since 1987), the S&P −3.0%, Bitcoin about −15%, and the VIX volatility index spiked intraday to 65.73 from around 23 a few sessions before. One weak U.S. jobs surprise, transmitted through leverage and the yen, became a worldwide risk-off cascade. The deep mechanics of carry unwinds are a macro post of their own; the framework lesson is that the *same release* (NFP) flips which surprise is "bad" when the regime flips — and that surprises can chain across markets when leverage is involved.

### The bonds in dollars

Each of these days moved bonds too, and bonds are where the surprise framework is most precise, because their price is mechanically tied to yields. When yields rise, bond prices fall, and vice versa. The September 2022 hot print pushed the 2-year Treasury yield up +18bp.

#### Worked example: an 18bp yield jump on a bond position

The standard way to translate a yield move into money is **DV01** — the dollar value of a one-basis-point change in yield. For a 2-year Treasury note, DV01 is roughly **\$190 per basis point per \$1,000,000 of face value** (a 2-year note has a duration near 2 years, and DV01 ≈ face × duration × 0.0001 ≈ \$1,000,000 × 1.9 × 0.0001 ≈ \$190). On the +18bp move:

- Position: **\$1,000,000** face value of the 2-year note
- DV01: ≈ **\$190 per basis point**
- Yield change: +18bp
- Mark-to-market change: \$190 × 18 = **≈ \$3,420**, and because yields *rose*, the bond's price *fell* — so a holder of the note lost about **−\$3,420**

A +0.2pp inflation surprise, transmitted into a +18bp jump in the 2-year yield, cost the holder of a \$1,000,000 2-year note about **−\$3,420** mark-to-market. The bond didn't default, nothing changed about its coupons — the *repricing* of the rate path alone moved its value, because a hot surprise means the market now expects higher rates, and higher rates mean lower bond prices. That is the surprise framework expressed in the one market where the arithmetic is exact.

## Common misconceptions

The framework is simple, but it cuts against intuition, so beginners reliably make the same handful of mistakes. Each is corrected with a number.

**Misconception 1: "Good economic news is good for stocks."** Not necessarily — it depends on the regime. On 3 February 2023, the January jobs report printed +517,000 jobs against a consensus near +187,000 — a *massive* upside surprise, supposedly "great news" for the economy. The S&P 500 *fell* −1.04% and the 2-year yield jumped +18bp. Why? In the inflation-fighting regime, a red-hot labor market meant the Fed would keep rates higher for longer. Good news for Main Street, bad news for asset prices. Strength is not goodness; the reaction function decides.

**Misconception 2: "A big number means a big move."** The *level* is nearly irrelevant; the *surprise* is what moves price. October 2022 inflation at 7.7% was, by any historical standard, a frightening level — yet the S&P rose +5.54% that day, because 7.7% was *cooler than the 7.9% expected*. A scary level with a benign surprise is a benign day for the market. The market is not reacting to the economy; it is reacting to the gap versus its forecast.

**Misconception 3: "Crypto is uncorrelated and a hedge against inflation."** On the day inflation printed *hottest relative to expectations* in 2022 — 13 September — Bitcoin fell about −9.4%, *more* than the Nasdaq's −5.16%. Far from hedging the inflation shock, crypto amplified it, trading like the highest-beta risk asset on the board. In a risk-off surprise, correlations converge toward one and the "diversifier" sells off with everything else. The macro post on [crypto as a macro liquidity asset](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates) develops why.

**Misconception 4: "If I know the number before the crowd, I'll win."** Even if you somehow knew the print in advance, you would *still* need to know what was priced in to predict the move — and you'd need to guess the knee-jerk-versus-fade dynamics. Markets have rallied on hot prints (relief versus an even-worse whisper) and sold off on in-line ones (positioning unwinds). Knowing the *number* is not knowing the *reaction*, and the reaction is the trade.

**Misconception 5: "The first move tells you the direction."** The knee-jerk spike is the *least* informed move of the day — algorithms reacting to the headline in the first two seconds, before anyone reads the internals or revisions. On many releases the spike fully reverses within thirty minutes once humans digest the details. Trading the headline blindly in the first heartbeat is how newcomers get run over by the fade.

**Misconception 6: "I can place a tight stop and be safe."** Around a release, a stop-loss order is not a guarantee — it is a *request* to sell at the next available price, and during the gap in the first seconds there may be no buyers at your stop level. A stop set 0.5% away can fill 2% away if the surprise gaps the market through it instantly. This is **gap risk**, and it is why event positions must be *sized smaller* than normal: you are not protected by your stop the way you are in a calm market, so you must be protected by your *size*. On the September 2022 print, the S&P gapped several percent in seconds — any stop inside that gap filled far worse than its level.

## The playbook: how to trade a release

The point of the framework is not to admire it — it is to trade it. Here is the repeatable loop that turns "a number is coming out" into a sized, disciplined trade with a clear invalidation. Every post in this series ends with a version of this playbook, specialized to its event.

![Pipeline of the event-trader's loop from prep to review](/imgs/blogs/why-news-moves-markets-the-surprise-framework-7.png)

**Step 1 — Prep.** Mark the release on your calendar with its exact date and time (CPI and jobs land at 8:30 a.m. ET; the Fed decision at 2:00 p.m. ET). Write down the **consensus** for the headline *and* the key internals (for CPI, that's core; for jobs, the unemployment rate). You cannot define a surprise without first writing down the expectation. The macro post on [the macro calendar — CPI, NFP, FOMC, PMI](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi) is the reference for what releases matter and when.

**Step 2 — Read what's priced.** Beyond the consensus number, gauge how big a move the market expects, and which way it's leaning. The cleanest read of "how much surprise is priced" comes from the options market via the **expected move** — roughly the price of an at-the-money straddle (a bet that pays off if price moves far in *either* direction). If S&P options imply a ±1.2% move into a CPI print, the market is braced for a fair-sized surprise; a print that lands in-line should produce *less* than that, and a position sized for ±1.2% that gets only ±0.3% can be a fade opportunity. A later post details how to read the expected move; the deeper options machinery lives in the quant-finance series.

**Step 3 — Map the surprise scenarios.** Before the print, write the if-then table. *If hot (actual > consensus): risk-off — stocks down, dollar up, yields up, crypto down hardest. If in-line: little net move, fade any knee-jerk overshoot. If cool: risk-on — stocks up, yields down, rate-sensitive small caps and crypto up most.* Crucially, **confirm the regime first** — in a growth-scare regime the signs flip, and a weak number becomes the bad one. You are not predicting the number; you are pre-deciding your response to each possible surprise.

**Step 4 — Anticipate the reaction.** Decide *which stage you are trading.* Are you fading the knee-jerk spike (betting the first move overshoots and retraces)? Are you trading the trend (betting a regime-changing surprise extends over days)? These are different trades with different time horizons and different risks. The single most important discipline: **do not confuse the number with the reaction.** A hot print does not automatically mean "short" — it means "the reaction function, in this regime, points risk-off, and now I watch whether the spike holds or fades."

**Step 5 — Put on the trade, sized and invalidated.** Position size around an event should be *smaller* than your normal size, because the gap risk is large — releases routinely move markets several percent in seconds, through any stop you've placed. Define your **invalidation** in advance: the price or condition that proves your read wrong and gets you out. For a "fade the hot-print spike" trade, invalidation might be "if the S&P makes a new low 30 minutes after the print, I'm wrong about the fade — exit." Never trade an event without a pre-written exit; the volatility is too fast to think clearly inside.

#### Worked example: sizing a release trade by the expected move

Suppose the S&P is at 5,000 and the options market prices an at-the-money straddle into the CPI print at **\$60**, implying an expected move of \$60 ÷ 5,000 = **±1.2%**. You want to risk **\$500** on a fade-the-spike trade. If the knee-jerk move is the full ±1.2% — about **\$60 per index point of exposure** — and you set your invalidation one expected-move beyond entry, then your per-unit risk is roughly the \$60 expected move. To cap your loss at \$500, you size so that a one-expected-move adverse swing costs \$500: a position of about **\$500 ÷ 1.2% ≈ \$41,700** notional. Risk the math, not the conviction: the **\$500** you're willing to lose, divided by the **±1.2%** the market says is normal, *is* your position size.

**Step 6 — Review.** After the dust settles, log it: what was the consensus, what printed, what was the surprise, how did each asset react, did the spike hold or fade, was your regime read right, did you follow your plan? The reaction function shifts over time — the "good news is bad" regime of 2022 is not permanent — and the only way to keep your priors current is to review every event you traded. Over dozens of releases, the loop compounds into a genuine edge: you stop being surprised by surprises.

### The two disciplines that separate winners from the run-over

Two habits do most of the work, and both are simple to state and hard to keep. The first: **define the surprise before the print, never after.** Write the consensus and the if-then scenario map *in advance*, when you are calm. After the number lands, the spike is firing, the chat is screaming, and your judgment is at its worst — that is exactly the moment to *execute* a pre-written plan, not to *form* one. A trader who decides what each surprise means before 8:30 is reacting to a checklist; a trader who decides at 8:30:01 is reacting to adrenaline.

The second: **trade the reaction, not the number, and respect that you might be wrong about the regime.** The single most expensive event-trading mistake is to be *correct about the data and wrong about the reaction* — to short a hot print into a market that had braced for worse and rallies in relief. Your invalidation level is the admission, written in advance, that your regime read could be off. When price violates it, you are not "wrong about inflation" — you are wrong about *what the market was positioned for*, which is a different and more humbling thing. The traders who survive events are the ones who hold that distinction.

#### Worked example: why smaller size is the real protection

Suppose your normal position is **\$50,000** and you'd normally risk **\$1,000** (2%) on a trade with a tight stop. Around a release where the expected move is ±1.2% and gap risk means your stop could fill at 2.5%, the *effective* risk on that \$50,000 is \$50,000 × 2.5% = **\$1,250** — already over your limit, and that's the *good* case. Halve the size to **\$25,000**, and the same 2.5% gap costs \$25,000 × 2.5% = **\$625** — back inside your risk budget with room for the gap. The fix for gap risk is not a tighter stop (the gap jumps it); it is a smaller position. You cannot control the surprise; you can control the dollars you expose to it.

That is the framework and the loop. Markets trade expectations; the price already contains the consensus; only the surprise moves it; the reaction function decides the sign and size; and the reaction — spike, fade, or trend — is the trade, not the number. Every subsequent post in this series takes one event and runs exactly this machinery over it, with the real cross-asset numbers attached.

## Further reading & cross-links

This post set up the mental model; the rest of the series and the sibling macro series fill in the mechanism behind each layer.

- [The macro calendar: CPI, NFP, FOMC, PMI](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi) — which releases matter, when they land, and what each one measures. Your prep-step reference.
- [Inflation and the Fed's reaction function: the dot plot](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot) — the deep mechanics of how the Fed turns inflation into rate decisions, which is the engine behind the "reaction function" layer.
- [Risk-on, risk-off: how money rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates) — the cross-asset map that explains why one print sends stocks, the dollar, gold and crypto in coordinated directions.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the foundational explainer on the policy lever that every U.S. macro release ultimately moves through.

Coming next in this series: a dedicated post on **reading what is really priced in** (beyond the headline consensus — the whisper number, positioning, and the expected move), a post on **the reaction function** and the regimes that flip the sign of every surprise, and a post on **cross-asset transmission** — how a single rate-market repricing radiates out to every other market in seconds. With the surprise framework in hand, all three will read as elaborations of one idea: the reaction is the trade, not the number.

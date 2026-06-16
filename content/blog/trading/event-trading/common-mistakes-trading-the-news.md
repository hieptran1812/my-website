---
title: "Common Mistakes Trading the News (and How to Avoid Them)"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "The same handful of errors drains most event-traders' accounts: trading the number instead of the reaction, ignoring what's priced, fading the spike too early, ignoring positioning, the revisions trap, over-trading tier-3 data, wrong size, and buying options into the vol crush. Each mistake with a real episode and a concrete fix."
tags: ["event-trading", "macro", "common-mistakes", "priced-in", "spike-fade-trend", "positioning", "revisions", "vol-crush", "position-sizing", "cpi", "nfp", "fomc"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Most event-traders don't lose money because they read the economy wrong. They lose because they make the *same eight execution errors* even when their macro call is right. Trade the **reaction**, not the number.
>
> - The eight mistakes group into three families: **reading errors** (trading the number, ignoring what's priced, the revisions trap), **timing errors** (fading the spike too early, ignoring positioning), and **sizing/instrument errors** (over-trading tier-3 data, wrong size, buying options into the vol crush).
> - The deadliest is the first: you can call a cool CPI print perfectly, go long — and still lose, because the rally was already priced and you bought the spike.
> - Every mistake has one concrete fix, and they all reduce to a single rule.
> - The one number to remember: a correct *cool-CPI* call paid **+5.54%** in Nov-2022 (unpriced) but only **+1.91%** a year later (mostly priced). Same direction, 2.9× the money.

Here is a true and very common story. A trader spends the morning of a CPI release doing real work. They model the components, they read the regional Fed surveys, they conclude — correctly — that the print will come in *cool*, below consensus, a friendly number for stocks and bonds. The clock hits 8:30 a.m. Eastern. The number drops. It is cool, exactly as predicted. The S&P futures spike up half a percent in three seconds. The trader, vindicated, hits the buy button at the top of the spike.

Twenty minutes later they are down money. The pop has faded, the index is drifting lower on profit-taking, and by the close the "great call" has turned into a small loss. The macro read was *right*. The trade was *wrong*. Nothing about the economy betrayed them — their own execution did.

This post is the autopsy. After enough of these, you notice the same handful of errors recurring across every trader's blown-up account, and they have almost nothing to do with predicting the economy. They are mistakes of *reading the reaction*, of *timing*, and of *sizing and instrument choice*. Below is the full catalogue — eight mistakes, each with a real dated episode that shows it in the wild and one concrete fix. They are the hard-won lessons that this whole series has been building toward.

![The eight news-trading mistakes mapped to their fixes, grouped by reading timing and sizing](/imgs/blogs/common-mistakes-trading-the-news-1.png)

## Foundations: why these mistakes happen

Before the catalogue, it helps to name the recurring traps, because every one of the eight mistakes is a failure to internalize one of these foundational facts about how markets process news. If you already know this series, skim it; if you're new, read it slowly, because everything downstream depends on it.

**The reaction is not the number.** A macro release is a single number — say, CPI inflation came in at 3.2% year-over-year. But your profit and loss does not come from the number. It comes from how *price moves* in response to the number. Those are different things, and conflating them is the single most expensive error in this whole domain. A "good" number can produce a falling market; a "bad" number can produce a rallying one. Your job is to predict the *reaction*, and the number is only one input into that.

**Priced-in.** Markets are forward-looking. By the time a release hits the tape, the price already contains the market's best guess of what that release will say — the *consensus*. If the actual number matches the consensus, there is, in principle, nothing new to react to, and the move is small or non-existent. Only the **surprise** — the gap between the actual number and the consensus — is genuinely new information, and only the surprise moves price. A cool print that everyone expected to be cool is not bullish news; it's old news. This is the deepest idea in the series, and the [consensus and priced-in deep-dive](/blog/trading/event-trading/consensus-expectations-and-priced-in) builds it from first principles.

**The spike, the fade, and the trend.** A news reaction is not one move; it's a sequence. First comes the **knee-jerk spike** — algorithms and fast money slam the number into price within milliseconds. Then one of two things happens: the spike **fades** (reverts, because it overshot or because the print was actually in-line) or it becomes a **trend** (extends, because the surprise was real and slower money piles in over hours). Knowing which of these you're in is the whole timing game, and the [anatomy of a reaction spike, fade, and trend](/blog/trading/event-trading/anatomy-of-a-news-reaction-spike-fade-trend) maps it in detail.

**Positioning and the pain trade.** Markets don't move to where the news points; they move to where it *hurts the most people*. If everyone is already short bonds going into a jobs report, even a hawkish number can spark a vicious short-covering rally because there's no one left to sell. The market's path is shaped by who is offside and forced to unwind. This is positioning, and the [positioning and pain-trade deep-dive](/blog/trading/event-trading/positioning-and-the-pain-trade) is the reference.

**The revisions trap.** Many headline numbers — payrolls, GDP, retail sales — are *first estimates* built from incomplete surveys. They get revised, sometimes massively, in later releases. A trade built on a first print that gets revised away is a trade built on noise that no longer exists.

**Tier-3 noise.** Not all releases matter. A small subset — CPI, the jobs report, FOMC decisions, PCE — are *tier-1* events that genuinely move the tape. Most of the economic calendar is *tier-3* noise: factory orders, the labor-cost index, regional surveys nobody trades. Trading them is paying transaction costs for almost no edge.

**The vol crush.** Options prices contain an *implied volatility* premium that swells before a scheduled event (because the outcome is uncertain) and collapses the instant the event passes (because the uncertainty is resolved). Buy an option into that swell and even a correct directional call can lose money as the premium deflates. The [event-volatility and vol-crush deep-dive](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush) covers the mechanics.

**Cross-asset, not single-asset.** One more foundation runs through every mistake: a macro print does not hit one market, it hits *all of them at once*, and the moves are linked. A hot CPI doesn't just sell stocks; it pushes the dollar up, Treasury yields up, gold down, and Bitcoin down, all in the same minute, because they're all expressing the same idea — "the Fed will be tighter." A trader who watches only their one instrument misses the confirmation (or the contradiction) sitting in the other four. On the hot August 2022 CPI, the S&P fell −4.32%, the Nasdaq −5.16%, Bitcoin roughly −9.4%, and the dollar rose +1.4% — a coherent risk-off cascade. When the cross-asset moves *agree*, the reaction is real and likely to trend; when they *diverge* — stocks down but the dollar also down — something is off, and the move is more likely to fade. Reading the whole board is part of reading the reaction, and the [cross-asset transmission post](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market) maps how one print propagates.

Every one of the eight mistakes below is a failure to respect one of these eight facts. Let's take them in order, grouped by family.

## Reading errors

The first family is about *misreading what the news means for price*. These are the most insidious because the trader who makes them often has a correct view of the economy — and that correctness gives them false confidence.

### Mistake 1: trading the number, not the reaction

This is the original sin of news trading, and it's the story from the opening. The trader sees a number, classifies it as "good" or "bad" using economic logic, and assumes price will follow that classification. Cool inflation is good for stocks, so buy. Strong jobs are good for the economy, so buy. The number's *sign* becomes the trade's direction.

The problem is that price has already incorporated the *expected* number, and it often reacts to the *details* and the *positioning* rather than the headline's sign. A cool print that everyone expected does not rally the market; it confirms what was already priced. Worse, in some regimes the sign is *inverted*: in 2022, strong economic data was *bad* for stocks, because it meant the Fed would hike more. A trader using naive "good number = buy" logic was systematically wrong for an entire year.

![A bullish cool CPI print that still fell because the good news was already priced in](/imgs/blogs/common-mistakes-trading-the-news-2.png)

Look at the figure. The left column is what the trader *sees*: a cool print, a correct read, a long position. The right column is what *happens*: the cool print was the consensus, there was no surprise to fuel a rally, the knee-jerk pop fades, and the trader is left long a move that already finished. The arrow from "correct read" to "loss" is the whole tragedy of this mistake.

**The real episode.** Compare two cool CPI prints. On 10 November 2022, the October CPI came in at 7.7% versus 7.9% expected — a genuine downside surprise — and the S&P 500 rallied **+5.54%**, its best day in over two years. A year later, on 14 November 2023, the October CPI came in at 3.2% versus 3.3% expected — also a cool print, also below consensus — but the S&P rose only **+1.91%**. Same direction, far smaller move. Why? Because by late 2023, a cooling inflation path was the *base case*; it was largely priced. The surprise in 2022 was a regime-shift; the surprise in 2023 was a confirmation. A trader who treated both as "cool print = big rally" sized the 2023 trade four times too large for the move it actually produced.

#### Worked example: right call, wrong trade

Say you correctly predicted a cool CPI print and went long \$25,000 of an S&P 500 ETF at the spike. You were betting on the 2022-style move: a +5.54% day on \$25,000 is +\$1,385. That's the trade you thought you put on. But the print was already priced — it was the 2023-style confirmation — and the index actually drifted to close just +0.3% off your entry. Your real result: +0.3% on \$25,000 = +\$75. You were *right about the number* and earned \$75 against the +\$1,385 you expected. The intuition: the surprise, not the number, is the fuel — and a priced-in number has almost no surprise left to pay you.

Why does the sign flip across regimes? Because the market cares about the number only insofar as it changes the *path of policy*, and that linkage changes with the environment. In a low-inflation, growth-scarce regime, strong economic data is unambiguously good: more growth, higher earnings, no inflation worry, so stocks rise. In a high-inflation regime like 2022, the same strong data is bad: it means the central bank must tighten harder to cool the economy, which raises the discount rate on every asset and threatens a recession — so stocks fall on good news. The number didn't change meaning; the *reaction function* — the rule translating data into the policy path into asset prices — flipped. A trader who hasn't identified which regime they're in is guessing at the sign of their own trade, and a 50/50 guess on direction is not an edge.

This is why the same trader can be a genius for a year and a fool the next without changing anything about their process: the regime changed under them and their fixed "good number = buy" rule went from right to wrong. The durable skill is not memorizing the current mapping but *always re-deriving it* from the current policy concern. Ask: what is the central bank most afraid of right now — inflation or recession? That fear determines whether good news helps or hurts, and it can change in a single quarter.

**The fix.** Stop classifying the number as "good" or "bad." Instead, ask: *what is the surprise versus consensus, and what is the regime's reaction function?* Trade the reaction you expect, not the headline's sign. If the print is in-line with consensus, the correct trade is often *no trade* — or a fade of the knee-jerk, not a chase of it. The [reaction-function deep-dive](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently) is the tool for figuring out which way the same number cuts in the current regime.

### Mistake 2: ignoring what's already priced

Mistake 1's twin. Where Mistake 1 ignores *direction* (trading the sign), Mistake 2 ignores *magnitude* (trading a move that has already happened). The market doesn't wait for event day to move; it moves *into* the event as the probability of an outcome firms up. By the time a rate cut is 95% priced in the futures, almost the entire move associated with that cut has already occurred — in the days and weeks *before* the meeting. Buying on the announcement is buying the tail end of a move that started long ago.

This is why "buy the rumor, sell the news" is a cliché: the rumor *is* the move, and the news is the exhaustion. The classic trap is a trader who is correct that the Fed will cut, waits for the cut to be confirmed, buys risk assets on the confirmation, and then watches the market *fall* because everyone who wanted to buy the cut already did so weeks ago, and now they're taking profits.

**The real episode.** On 18 September 2024, the Fed delivered its first cut of the cycle — a larger-than-usual 50 basis points. By any naive logic, a jumbo cut is a gift to risk assets. Yet the S&P 500 *fell* −0.29% on the day. The cut had been heavily debated and largely priced; the 50bp size was a mild surprise, but the market had spent weeks rallying on the *anticipation* of easing. The announcement itself was an anticlimax. A trader who bought the cut on the day, ignoring how much was already in the price, bought the top of a multi-week move.

#### Worked example: buying a priced-in move

You buy \$40,000 of equity-index exposure on the day of a "dovish" Fed decision, certain the cut means a rally. But the market rallied 4% over the three weeks *before* the meeting as the cut got priced. On the day, with the move exhausted, the index slips −0.29% (the real Sep-2024 number). Your result: −0.29% on \$40,000 = −\$116. Meanwhile the trader who bought three weeks earlier, when the cut was only 40% priced, captured most of that +4% = +\$1,600 on the same \$40,000. The intuition: the money is in the *repricing*, and the repricing happens before the event, not on it.

There's a subtler version of this mistake that catches even experienced traders: confusing *what is priced* with *what is obvious*. An outcome can be obvious — everyone agrees the Fed will cut, the economy is clearly slowing — and yet the market may not have *fully* priced the magnitude or the speed. The skill is distinguishing "everyone knows this will happen" (which may still be only 60% priced) from "the curve has it 95% in." The first leaves room for a move; the second does not. Reading the futures curve, not the headlines, is the only way to tell them apart, because the curve is where the money actually sits.

This is also where emerging markets differ from the U.S. in an instructive way. In a deep, liquid market like U.S. rates, outcomes get priced quickly and efficiently, so the "already priced" trap is severe — there's little slack. In a market like Vietnam's, where the State Bank of Vietnam's rate decisions are less continuously priced and foreign-flow dynamics dominate, a "known" policy move can still produce a real reaction because the market is thinner and slower to discount. The same mistake — ignoring what's priced — is less punishing where the priced-in mechanism is weaker, but the discipline of *checking* is identical. The [SBV and VN-Index reaction post](/blog/trading/event-trading/vietnam-events-sbv-vn-cpi-and-how-vn-index-reacts) covers how Vietnam's policy events transmit differently.

**The fix.** Before any event, read the market-implied probability — Fed funds futures via CME FedWatch for the Fed, the options-implied expected move for a data release. Ask: *how much of this outcome is already in the price?* If it's mostly priced, the event is a non-event for direction, and your edge (if any) is in the second-order detail — the press conference tone, the dot plot, the internals — not the headline. The [expected-move and consensus tooling](/blog/trading/event-trading/consensus-expectations-and-priced-in) shows exactly how to read what's priced.

### Mistake 5: the revisions trap

(We'll cover Mistakes 3 and 4 under timing; this one belongs with the reading errors.) Several of the most-watched releases are *first estimates*, and first estimates are noisy. Nonfarm payrolls, GDP, and retail sales are all built from incomplete survey responses and get revised in subsequent months — sometimes by amounts larger than the original surprise that moved the market. A trader who builds a thesis on the first print can find that the very data point underpinning the trade has been quietly erased a month later.

This matters most for the jobs report, where revisions have historically been large and one-directional for stretches at a time. A run of "strong" headlines that all get revised lower paints a false picture of a hot labor market, and the traders who pressed the "strong economy" trade on each headline get whipsawed when the revisions reveal the trend was softer all along.

![A strong NFP headline of +250k revised down by 80k the next month trapping the chaser](/imgs/blogs/common-mistakes-trading-the-news-5.png)

The figure walks it through. This month: a +250k headline lands above the +175k consensus, the trader chases the "strong economy," shorts bonds — but the first print is just an estimate. Next month: the revision cuts it by 80k to a soft +170k, the thesis evaporates, and the book bleeds unwinding a position built on a number that no longer exists.

**The real episode.** Through 2024, U.S. payroll figures were repeatedly revised lower, and the annual benchmark revision in August 2024 cut the prior twelve months of job growth by hundreds of thousands. Traders who had treated each strong monthly headline as confirmation of an overheating economy — and positioned for higher-for-longer rates — were caught when the revised picture showed a labor market that had been cooling faster than the first prints suggested. The market's repricing toward cuts was, in part, a correction of trades built on numbers that got revised away.

#### Worked example: trading a number that gets revised away

You see a +250k payrolls headline, decide the economy is overheating, and put on a \$20,000 short-bond position expecting yields to rise. For a few days it works. Then the next report revises that +250k down by 80k to +170k — barely above the prior trend — and the whole "hot economy" narrative reverses. Your book gives back −1.5% as yields fall: −1.5% on \$20,000 = −\$300. The intuition: you can't build a durable trade on a first estimate that the statisticians themselves don't yet trust — the revision is the real number, and it arrives a month too late for your entry.

**The fix.** Treat first prints of revision-prone series as *low-confidence signals*, especially the headline payroll number. Weight the parts of the report that revise less (the unemployment rate, derived from a separate and steadier survey) and the broader labor mosaic (jobless claims, JOLTS) over any single headline. If you must trade the first print, size it down and set a tight invalidation, because the ground under it may shift. Don't marry a thesis to a number that hasn't settled.

## Timing errors

The second family is about *when* you act. Here the trader may read the reaction correctly — they know the direction — but they enter at the wrong moment and get punished anyway.

### Mistake 3: fading the spike too early

This is the mirror image of Mistake 1's chaser. The fader has learned, correctly, that knee-jerk spikes often revert — and then over-applies the lesson, shorting *every* spike on the assumption it will fade. The trouble is that some spikes are not noise; they are the *first leg of a genuine trend*. Fading the front of a real trend means standing in front of a freight train, and the loss can be enormous because a trend leg, by definition, keeps going.

The fader's mistake is impatience. They see the spike, decide it's overdone, and short into it *before the market has revealed whether it's a fade or a trend*. If they'd waited even a few minutes — to see whether the move stalls and reverts (fade) or consolidates and extends (trend) — they'd have the information they need. Instead they pick the top of what turns out to be the bottom third of a much larger move.

![Price path schematic showing a short entered at the knee-jerk spike run over by the trend leg](/imgs/blogs/common-mistakes-trading-the-news-4.png)

The schematic shows the trap. The price gaps up on the print (knee-jerk spike, leg 1), pulls back slightly (leg 2 — this is where the fader gets seduced into thinking "see, it's reverting"), and then launches into the real trend (leg 3). The fader shorted near the top of leg 1, the trend leg blows through their stop, and they're run over for a multi-percent loss on a move they were *directionally* trying to play correctly.

**The real episode.** On 13 September 2022, the hot August CPI print started a relentless one-way day: the S&P 500 fell −4.32% and the Nasdaq −5.16% in a near-uninterrupted slide. There was barely any fade to short into — the knee-jerk *was* the trend. A bear who shorted the open had a great day; but a contrarian who tried to fade the initial down-spike, betting on a bounce, got steamrolled all session. When the surprise is large and the regime is clear, the first move is the trend, and fading it is fighting the tape.

#### Worked example: fading into a trend leg

Convinced the post-print spike will revert, you short \$15,000 of an index future right at the knee-jerk high. But this is a real surprise, and the move is a trend, not a fade. Price grinds against you and you cover after a −3% run: −3% on \$15,000 = −\$450. Had you waited for the trend to *fail* — to stall and roll over — before fading, you'd have shorted from a higher level into an actual reversal. The intuition: the knee-jerk is not the top; fade the *second* leg after the trend shows it can't extend, never the first.

**The fix.** Never fade the first move. Let the knee-jerk settle and watch for confirmation: if the spike stalls, gives back its gains, and the order flow turns, *then* fade the second leg. If instead the move consolidates and extends, you're in a trend — either trade with it or stand aside. The discipline is to require the market to *prove* it's a fade before you bet on one. The [spike-fade-trend anatomy](/blog/trading/event-trading/anatomy-of-a-news-reaction-spike-fade-trend) is the playbook for reading which regime you're in.

### Mistake 4: ignoring positioning and the pain trade

The market is a crowd, and crowds get lopsided. When everyone is leaning the same way — all short bonds, all long the dollar, all crowded into the same carry trade — the market becomes fragile in the *opposite* direction, because the next move has to come from the people who are offside being forced out. A trader who reads only the fundamentals, ignoring *who is positioned how*, repeatedly gets surprised by violent moves that "make no sense" given the news — but make perfect sense given the positioning.

The classic version: a number comes out that, on fundamentals, should push the dollar up. But the entire fast-money community is *already* long the dollar going in. There's no one left to buy, and the slightest disappointment triggers a stampede for the exits — the dollar *falls* on dollar-positive news, because the pain trade is down. The news was a catalyst, not a cause; the cause was the crowded positioning.

**The real episode.** The yen-carry unwind of early August 2024 is the textbook case. For years, traders had borrowed cheap yen to buy higher-yielding assets everywhere — a massively crowded, one-directional trade. When the Bank of Japan hiked on 31 July and a weak U.S. jobs report landed on 2 August, the carry trade began to unwind, and because *everyone* was on the same side, the unwind fed on itself. On 5 August the Nikkei fell **−12.4%** (its worst day since 1987), the VIX spiked to an intraday **65.73**, the S&P fell **−3.0%**, and Bitcoin dropped roughly **−15%**. None of these moves were proportionate to the news; they were proportionate to the *positioning*. The [carry-unwind history in the macro series](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) traces how leverage turns a small catalyst into a cascade.

#### Worked example: getting caught in a positioning unwind

You're long a crowded carry-style trade — say \$30,000 of risk exposure funded by a low-yield short — going into a central-bank meeting, reasoning purely from the rate differential. The differential logic is fine, but the trade is packed and the meeting triggers an unwind. On the cascade day your book takes the kind of hit the broad market took: −3.0% on \$30,000 = −\$900, and that's *before* any leverage. A 2× levered version would have lost −\$1,800 on the same move. The intuition: in a crowded trade, the fundamentals stop mattering at the margin — the exit door is the only thing that matters, and everyone reaches it at once.

The positioning lesson is sharpest in bonds, where the move is measurable to the penny through DV01 — the dollar change in a position's value per one-basis-point move in yield. On the blowout +517k payrolls of 3 February 2023, the 2-year Treasury yield jumped +18bp in a single session, far more than the headline alone justified, because the bond market was positioned for *softer* data and the strong number forced a violent repricing — a positioning-driven overshoot, not a pure fundamental move.

#### Worked example: the positioning overshoot in bonds

Suppose you are long \$1,000,000 face of the 2-year Treasury (a DV01 of roughly \$20 per basis point) going into a jobs report, comfortable because "the economy is slowing." The market is crowded long bonds on that same view. Then payrolls blow out by +517k and the 2-year yield jumps +18bp as the crowd is forced to sell. Your loss: 18bp × \$20/bp = −\$360 on the position, in minutes. The move was larger than the surprise warranted precisely *because* everyone was offside the same way. The intuition: in a crowded trade, the surprise that hurts is amplified by the stampede for the exit — the positioning, not the data, sets the size of the move.

**The fix.** Always ask *who is offside* before an event. Watch the positioning signals — Commitment of Traders data, dealer hedging flows, the consensus among fast money — and respect the asymmetry they create. When a trade is crowded, the surprise that pays is the one that forces the crowd to puke, not the one the fundamentals point to. The [positioning and pain-trade deep-dive](/blog/trading/event-trading/positioning-and-the-pain-trade) and the macro series' [flows and positioning post](/blog/trading/macro-trading/following-the-flows-positioning-cot-dealer-hedging) are the references for reading the crowd.

## Sizing and instrument errors

The third family is about *how much* and *with what*. A trader can read the reaction right and time it right, and still blow up because they sized wrong or chose the wrong instrument. These are the errors of the otherwise-skilled trader, and they are brutal because they convert good analysis into bad outcomes.

### Mistake 6: over-trading tier-3 data

The economic calendar is enormous, and most of it is noise. There's a small set of releases that genuinely move markets — the jobs report, CPI, FOMC decisions, PCE, the big PMI surveys — and a vast tail of *tier-3* releases that produce nothing tradeable: factory orders, wholesale inventories, the employment-cost index, second-tier regional surveys. The over-trader treats every line on the calendar as an opportunity, takes a position into each one, and slowly bleeds out through transaction costs and small whipsaws while waiting for an edge that the tier-3 print was never going to provide.

This is the death-by-a-thousand-cuts mistake. Each individual tier-3 trade loses only a little — a few dollars of spread, a small adverse move — but the *frequency* compounds. A trader who takes thirty tier-3 trades a month, each with a small negative expectancy after costs, has manufactured a steady drain that no amount of skill on the real events can offset.

![A tier-1 hot CPI reaction of over four percent dwarfs a typical tier-3 release day of about a fifth of a percent](/imgs/blogs/common-mistakes-trading-the-news-6.png)

The figure puts the scale in perspective. A genuine tier-1 reaction — the hot August 2022 CPI, an S&P move of about 4.3% — towers over a typical tier-3 release day, where the index barely twitches by perhaps a fifth of a percent. The edge available on the left bar is real and worth preparing for; the edge on the right bar is smaller than your transaction costs. Trading the right bar is paying to play.

**The real episode.** This one is less a single date than a pattern. Across any quarter, the calendar carries dozens of minor releases — construction spending, the trade balance, durable-goods orders excluding transport — each of which barely registers in index volatility. Compare that to the genuine event days: the −4.32% S&P day on the hot CPI, the +5.54% day on the cool one, the −1.84% day on the weak August 2024 jobs report. The market reserves its big moves for tier-1 events; the rest is chop. A trader's P&L over a year is dominated by a tiny number of days, and those days are tier-1.

#### Worked example: the tier-3 drain

Say you take twenty tier-3 trades in a month on a \$10,000 account, each round-trip costing roughly 0.10% in spread and commission with essentially zero directional edge. That's 20 × 0.10% × \$10,000 = −\$200 in pure friction, before any adverse moves. Over a year that's −\$2,400 — a 24% drag on a \$10,000 account, manufactured entirely from trading noise. Skip the tier-3 prints and that \$2,400 stays in your account. The intuition: every trade has a cost, so a trade with no edge is a guaranteed slow loss — silence is a position.

**The fix.** Build a tier-1 calendar and trade *only* it. The [macro calendar deep-dive](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi) and this series' [global calendar post](/blog/trading/event-trading/the-global-economic-calendar-what-actually-moves-markets) rank what actually matters. For everything below tier-1, the correct action is to watch, not trade. Most prints aren't worth a position, and treating "no trade" as the default decision is the single biggest source of saved capital for the over-trader.

### Mistake 7: wrong size for the event

Event days are not normal days. The expected move is larger, the spreads are wider, the liquidity is thinner around the release, and the gaps can jump straight past your stop. A position size that's perfectly reasonable on a quiet Tuesday can be reckless going into an FOMC decision, because the same dollar position is exposed to a move several times larger. The trader who uses their everyday size into a high-volatility event has, without realizing it, multiplied their risk.

The mirror error is also common: sizing *too small* into a genuine, well-read surprise because event volatility scared the trader into timidity, leaving most of a hard-won edge on the table. But the dangerous direction is oversizing, because that's the one that ends accounts. The fix for both is the same — size to the *event's* expected move, not to your habitual size or your conviction.

**The real episode.** The August 2024 cascade again illustrates the cost of size. On 5 August, an unlevered position in the broad market lost −3.0% on the day; a 3× levered position lost −9%; and traders carrying large leveraged carry positions saw far worse as the unwind compounded. The intraday VIX hitting 65.73 meant the *expected* daily move had quintupled from its calm-market baseline near 13. A position sized for a 1% day was suddenly exposed to a 3%+ day. The traders who survived had sized for the volatility regime they were actually in; the ones who didn't were liquidated.

#### Worked example: sizing to the expected move

Suppose your risk budget is to lose no more than \$500 on any single event. On a normal day the index moves about ±1%, so you can hold \$50,000 (1% of \$50,000 = \$500). But going into an FOMC decision the expected move is ±2.5%, so to keep the same \$500 risk you must cut the position to \$20,000 (2.5% of \$20,000 = \$500). A trader who kept the full \$50,000 into the event risked 2.5% × \$50,000 = \$1,250 — two and a half times their budget — on a single decision. The intuition: hold your *risk* constant by shrinking your *size* as the expected move grows; the event decides the size, not your habit.

**The fix.** Read the expected move before the event — from the options straddle for a data release, from the historical event-day range for a known event — and size so that a full expected-move adverse outcome stays within your risk budget. The [expected-move pricing deep-dive](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options) shows how to extract the number from option prices. Then account for the *gap* risk: stops don't work in a vacuum when price can jump, so the position must be small enough to survive a gap past your stop. The [liquidity and gaps around news post](/blog/trading/event-trading/liquidity-and-gaps-around-news) covers why event-day stops are unreliable.

### Mistake 8: buying options into the vol crush

The final mistake is the most technical and, for that reason, the most surprising to its victims. A trader expects a big move on an event — say a CPI release — and decides to express it by buying an option (or a straddle, to be direction-agnostic). The logic feels airtight: if the move is big, the option pays. But options are priced with an *implied volatility* premium that inflates before a scheduled event, precisely because everyone knows a move is coming. You are not buying the move cheaply; you are buying it at peak price, with the uncertainty premium baked in.

Then the event passes. The uncertainty is resolved — the number is out, there's nothing left to be uncertain about — and implied volatility *collapses*. This is the **vol crush**. Even if the underlying moves exactly the way you predicted, the option can *lose* value, because the premium you paid for the uncertainty deflates faster than the move adds intrinsic value. You were right on direction and still lost money — the most demoralizing outcome in trading.

**The real episode.** This is a structural feature of every scheduled event, but the mechanics are clearest with a concrete setup. Going into a CPI print, an at-the-money straddle on the S&P at the 5000 level might cost around \$60 — implying an expected move of about ±1.2%. That \$60 is mostly implied-vol premium. If the index then moves only +0.8% — a real move, in the right direction for a call buyer — but the post-event implied vol collapses, the straddle can be worth less than \$60 afterward, because the vol crush outweighs the modest realized move. The straddle buyer needed the move to *exceed* the priced ±1.2% just to break even, and a merely-correct call wasn't enough.

#### Worked example: right direction, killed by the vol crush

You buy a \$1,000 straddle (a call plus a put) on a stock going into earnings, expecting a big move. The stock does move in a direction you'd have profited from — but it moves *less* than the implied volatility priced in, and after the report implied vol crushes. Your straddle is now worth \$700. Result: \$700 − \$1,000 = −\$300, a 30% loss, despite being broadly right that the stock would move. The intuition: when you buy an option into an event you're betting the move will *beat* the priced expected move; merely being right on direction loses to the vol crush.

The deeper framing is the gap between *implied* and *realized* volatility. The price you pay for an option encodes the market's *implied* move — the volatility it expects. Your P&L on a long straddle held through the event depends on whether the *realized* move (what actually happened) exceeds that implied move. If realized < implied, the option buyer loses even on a correct direction; if realized > implied, they win. Buying options into an event is therefore a bet that the market has *under-priced* the move — and since the whole market is staring at the same calendar, that's a hard bet to win consistently. The [implied-vs-realized and vol-crush deep-dive](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush) and the [volatility-surface post](/blog/trading/quantitative-finance/volatility-surface) develop this in full.

#### Worked example: the breakeven the option buyer forgets

You buy that \$60 at-the-money S&P straddle at the 5000 level, which prices an expected move of ±1.2% (60 ÷ 5000). For the straddle to break even at expiry, the index must move *at least* ±1.2% — 60 points. If the index moves +0.8% (+40 points), a correct directional call, the call leg is worth only about \$40 of intrinsic value while the put expires worthless: \$40 − \$60 = −\$20 per straddle, a loss on a right call. You needed +1.2% just to break even and *more* than that to profit. The intuition: the priced expected move is the bar you must clear, not the direction — being right on direction is necessary but not sufficient.

**The fix.** Respect the vol crush. If you want event exposure through options, recognize you need the realized move to *exceed* the implied (priced) move, not just to go your way — so only buy premium when you have a strong view that the move will be *bigger* than consensus expects. Otherwise, prefer being a *seller* of the inflated premium (with defined risk), or express the view in the underlying instead of in options. The [event-volatility and vol-crush deep-dive](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush) and the [options-theory primer](/blog/trading/quantitative-finance/options-theory) lay out the math of implied versus realized volatility.

## Common misconceptions that feed the mistakes

The eight mistakes are sustained by a handful of beliefs that *feel* like trading wisdom but are quietly wrong. Naming them helps, because each misconception is the seed of one of the errors.

**"A good number is bullish."** This is the seed of Mistake 1, and it's wrong because the sign of a number's effect on price depends on the regime, not on the number. In 2022, a *strong* economy was *bearish* for stocks, because it meant more Fed hikes — strong jobs of +517k produced a −1.04% S&P day. The correction: there is no fixed mapping from number to direction; the reaction function, which can flip across regimes, decides the sign.

**"If I'm right about the economy, I'll make money."** This is the seed of Mistakes 2 and 8 — being right is necessary but not sufficient. You can be right about a cool print and lose because it was priced (Mistake 2), or right about a big move and lose because you bought the vol crush (Mistake 8). The correction: P&L comes from the *gap between your view and what's priced*, not from your view being correct in absolute terms.

**"Spikes always fade, so fade them."** This is the seed of Mistake 3, and it's wrong because some spikes are the front of a trend. On the hot September 2022 CPI, the knee-jerk down-move *was* the trend, and there was no bounce to fade — the S&P bled −4.32% in a near-straight line. The correction: require the market to prove a fade before betting on one; never fade the first leg of a clear surprise.

**"More trades, more chances to win."** This is the seed of Mistake 6 — every trade carries a cost, so more trades on no edge means a faster, more certain loss. The correction: your annual P&L is dominated by a tiny number of tier-1 days; the rest is friction. Trading less is often the highest-expectancy decision available.

**"A stop-loss caps my risk on event day."** This is the seed of the size error in Mistake 7, and it's dangerously wrong because event-day liquidity is thin and prices *gap* — they can jump straight past your stop, filling you far worse than your stop level. The correction: stops are unreliable around news; the only real risk control is a small enough *size* to survive a gap, which is exactly what the [liquidity and gaps post](/blog/trading/event-trading/liquidity-and-gaps-around-news) shows.

Spot the misconception behind a trade you're about to put on, and you've usually caught the mistake before it costs you anything.

## How they show up: real episodes

The eight mistakes are easiest to see when laid against real reaction data, because the same numbers that traders got right on the macro level still produced losing trades. This section uses the curated CPI, NFP, and FOMC reactions to show the recurring pattern: *right call, wrong trade*.

![Three CPI days showing a correct cool-print call can be a small move if it was already priced](/imgs/blogs/common-mistakes-trading-the-news-3.png)

The bar chart is the clearest single picture of Mistake 1 and Mistake 2 together. Three CPI days: the hot August 2022 print (S&P −4.32%), the cool October 2022 print (+5.54%), and the cool October 2023 print (+1.91%). Notice that the two cool prints — both "correct" bullish reads — produced *wildly* different P&L. The 2022 cool print was a regime-shifting surprise and paid 2.9× what the 2023 cool print paid, because by 2023 the cooling was priced. A trader who treated "cool CPI" as a uniform buy signal sized identically into both and was badly mismatched to the actual move on the second.

**The CPI episodes.** The cross-asset reaction to these prints rewards the *reaction-reader* and punishes the *number-reader*. On the hot 2022 print, not only did the S&P fall −4.32%, the Nasdaq fell −5.16%, Bitcoin dropped roughly −9.4%, and the dollar rose +1.4% — a clean risk-off cascade. A trader who simply saw "high inflation number" and didn't connect it to "more Fed hikes = sell everything risky" missed the *direction* (Mistake 1). On the cool prints, the trader who didn't check what was priced (Mistake 2) overpaid for the smaller 2023 move. The full cross-asset breakdown lives in the [CPI case-studies post](/blog/trading/event-trading/cpi-case-studies-the-prints-that-broke-the-tape).

**The NFP episodes.** The jobs report shows Mistakes 3, 4, and 5 vividly. On 3 February 2023, payrolls printed a blowout +517k versus ~187k expected — and the S&P *fell* −1.04% while the 2-year yield jumped +18 basis points. A number-reader saw "strong jobs, good economy, buy" and was wrong on direction, because in that regime strong data meant more hikes (Mistake 1). On 2 August 2024, payrolls came in weak at +114k with unemployment rising to 4.3%, the S&P fell −1.84%, and the 2-year yield dropped −28bp — the start of the recession scare that fed the carry unwind three days later (Mistake 4, positioning). And the 2024 payroll figures, repeatedly revised down, are the canonical revisions trap (Mistake 5). The [NFP cross-asset playbook](/blog/trading/event-trading/trading-the-nfp-release-cross-asset-playbook) walks the full reaction.

**The FOMC episodes.** The Fed meetings show Mistakes 2 and 7. On 16 March 2022, the first hike of the cycle, the S&P *rallied* +2.24% — the hike was fully priced, and the relief of clarity (plus the removal of uncertainty) sparked a rally that confused anyone expecting "hike = sell." On 18 September 2024, the first cut, the S&P *fell* −0.29% — the cut was priced, and the move had already happened. And the December 2018 hawkish hike triggered a −1.54% day that extended into a −19.8% quarterly drawdown, a reminder that event-day size matters because the move can keep going (Mistake 7). The [FOMC case-studies post](/blog/trading/event-trading/fomc-case-studies-taper-tantrum-2018-pivot-2022-hikes) has the dated detail.

There's a Vietnam-flavored version of the same pattern worth noting, because the series scope spans VN markets. When the State Bank of Vietnam raised its refinancing rate from 4.0% to 6.0% across autumn 2022 to defend the dong, the VN-Index was already in a brutal slide — it fell from a January 2022 peak near 1,528 to a trough of 911 on 15 November 2022, roughly −40%. A trader who reacted to each individual SBV announcement as "rate hike = sell" was trading the number, but the deeper driver was the cross-asset and positioning story: foreign outflows, a margin-debt unwind, and a global risk-off backdrop. The SBV rate path was a symptom of the same global tightening that crushed the S&P, not an independent VN signal — and reading it in isolation, divorced from the global reaction function, was its own version of Mistake 1. The [VN events post](/blog/trading/event-trading/vietnam-events-sbv-vn-cpi-and-how-vn-index-reacts) traces how VN-Index actually reacts to policy.

The thread through all of these is the same: the traders who lost weren't wrong about the economy. They traded the number instead of the reaction, ignored what was priced, faded too early, got caught by positioning, chased revised-away headlines, or sized wrong. The macro call was the easy part. The hard part — the part that separates the trader who keeps their account from the one who doesn't — is the eight-item execution discipline that turns a correct read into a profitable trade instead of a vindicated loss.

## The meta-mistake: no plan, no review

Underneath the eight specific mistakes sits a ninth, and it's the one that makes the other eight chronic rather than occasional: **trading without a written plan, and never reviewing what happened.** Every one of the eight errors is something a disciplined process would catch. Trading the number not the reaction is caught by a plan that forces you to write down the *reaction* you expect and why. Ignoring what's priced is caught by a plan that requires you to record the market-implied probability before you trade. Wrong size is caught by a plan that fixes your risk budget per event in advance.

Without a plan, you make every decision in the heat of the moment, under the adrenaline of a fast-moving tape, which is exactly when the brain reaches for the lazy heuristic — "good number, buy" — that the eight mistakes are made of. The plan is what moves the decision from the panicked present to the calm past, when you could think clearly.

And without a *review*, you never learn which mistakes are *yours*. Every trader has a signature error — some chronically fade too early, some chronically over-size, some can't resist tier-3 prints. You cannot fix a pattern you can't see, and you can't see it without a trade journal that records, for each event trade: the consensus, the surprise you expected, the actual surprise, the position, the size, the reaction you predicted, the reaction that happened, and the result. After thirty entries, your signature mistake stares back at you from the page.

Here is what a usable event-trade journal entry looks like, filled out for a hypothetical CPI trade. Before the event: *consensus = 3.3% headline; market-implied = mostly priced for a cool print; expected move = ±1.0% on the S&P; my view = in-line, no edge; planned trade = none, or a small fade of any spike; risk budget = \$500.* After the event: *actual = 3.2%, a −0.1pp cool surprise; reaction = +0.4% knee-jerk that faded to flat by the close; my trade = I chased the spike anyway; result = −\$120.* The journal entry doesn't lie: it records that the *plan* said "no trade or fade," and the *trader* chased — that's a discipline failure, not an analysis failure, and it's invisible without the written comparison.

After a few dozen of these, the pattern becomes undeniable. Maybe your "planned trade" and "actual trade" disagree every time there's a fast spike — you're a chronic chaser (Mistake 1 or 3). Maybe your losses cluster on tier-3 days you said you'd skip (Mistake 6). Maybe your size column creeps up whenever you're confident, right before the biggest losses (Mistake 7). The journal converts a vague sense of "I keep losing on news" into a specific, fixable diagnosis: *here is the one column where my plan and my behavior diverge.* That is worth more than any new indicator.

There's a second, subtler benefit to the written plan: it pre-commits you to *inaction*. Most of the eight mistakes are sins of *commission* — chasing, fading, over-trading, oversizing — and the antidote to a sin of commission is a default of *no action* that you have to actively override. When your plan says "no trade unless the surprise exceeds X," and the number comes in below X, the plan has already made the decision for you, and you don't have to summon discipline in the heat of the moment — you just follow the page. Discipline-in-advance beats willpower-in-the-moment, because willpower is exactly what the adrenaline of a fast tape destroys.

The meta-fix is therefore process, not insight: a pre-event checklist that forces the eight questions, and a post-event journal that catches your recurring error. The insight in this whole post is worthless if it lives only in your head on a calm afternoon and evaporates the moment the number hits the tape. A modest trader with an iron checklist will, over a year, beat a brilliant one who trades from the gut — because the checklist trader makes the eight mistakes a handful of times and the gut trader makes them daily.

## The playbook: the checklist of fixes

Here is the entire post compressed into a checklist you can run before every event trade. It is deliberately ordered: reading first (is there even a trade?), then timing (when?), then sizing and instrument (how much, with what?).

![The fixes checklist as a tree from one rule into reading timing and sizing habits](/imgs/blogs/common-mistakes-trading-the-news-7.png)

The figure shows the structure: one rule at the root — *trade the reaction, not the number* — branching into the three families, each with its concrete habits. Run it top to bottom:

**Reading (is there a trade?).**
1. *Trade the reaction, not the number.* Write down the *reaction* you expect (direction and size of the price move), not your classification of the number as good or bad. Confirm the regime's reaction function — does good news help or hurt risk in *this* regime?
2. *Check what's already priced.* Read the market-implied probability or the options-implied expected move. If the outcome is mostly priced, the headline is a non-event for direction; your edge, if any, is in the second-order detail.
3. *Distrust revisions-prone first prints.* For payrolls, GDP, and retail sales, treat the first headline as low-confidence. Weight the steadier components and the broader mosaic; size down if you must trade the first print.

**Timing (when?).**
4. *Don't fade the first move.* Let the knee-jerk settle. Require the market to prove a fade — a stall, a reversal of flow — before you bet on one. Fade the *second* leg, never the first; with a real surprise, the first move is the trend.
5. *Respect positioning.* Ask who is offside before the event. In a crowded trade, the pain trade — the move that forces the crowd out — matters more than the fundamentals at the margin.

**Sizing and instrument (how much, with what?).**
6. *Skip tier-3 data.* Trade only tier-1 events. For everything below, the default decision is *no trade*. Silence is a position, and it's usually the right one.
7. *Size to the expected move, not your conviction.* Fix your risk budget per event, read the expected move, and shrink the position so a full adverse expected-move outcome stays within budget. Account for gap risk — event stops are unreliable.
8. *Don't fight the vol crush.* If you express the view in options, you need the realized move to *beat* the priced move, not just to go your way. Otherwise sell the inflated premium with defined risk, or trade the underlying.

And wrapping all eight: **have a written plan and review every trade.** The plan forces the eight questions; the journal reveals your signature mistake. Process beats insight, because process survives the adrenaline of the tape and insight doesn't.

If you internalize one sentence from this entire post, make it the root of that tree: *you are not trading the number, you are trading the reaction — and the reaction depends on the surprise, what's priced, the positioning, and the timing, not on whether the number was "good."* Every one of the eight mistakes is a way of forgetting that sentence, and every fix is a way of remembering it.

## Further reading & cross-links

Within this series — the foundations each mistake leans on:

- [Consensus, expectations, and 'priced in'](/blog/trading/event-trading/consensus-expectations-and-priced-in) — the engine behind Mistakes 1 and 2: how the market builds a consensus and why only the surprise moves price.
- [Anatomy of a news reaction: spike, fade, trend](/blog/trading/event-trading/anatomy-of-a-news-reaction-spike-fade-trend) — the timing model behind Mistake 3: how to tell a fade from a trend.
- [Positioning and the pain trade](/blog/trading/event-trading/positioning-and-the-pain-trade) — the crowd dynamics behind Mistake 4: why markets move to where it hurts most.
- [Event volatility: implied vs realized and the vol crush](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush) — the options mechanics behind Mistake 8.
- [The reaction function: why the same number moves differently](/blog/trading/event-trading/the-reaction-function-why-the-same-number-moves-differently) — why "good news" flips sign across regimes, the deeper layer under Mistake 1.

For the macro mechanism beneath the reactions, see the macro-trading series: the [macro calendar](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi) for what's tier-1, the [flows and positioning post](/blog/trading/macro-trading/following-the-flows-positioning-cot-dealer-hedging) for reading the crowd, and the [carry-unwind history](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) for how positioning turns a small catalyst into a cascade.

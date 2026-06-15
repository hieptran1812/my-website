---
title: "Trading the FOMC: The Statement, the Dot Plot, and the Press Conference"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "A beginner-friendly deep dive into how the FOMC meeting actually moves markets — why the rate decision is usually a non-event, and how the statement wording, the dot plot, and Powell's press conference create the surprise you can trade."
tags: ["macro", "monetary-policy", "fomc", "federal-reserve", "dot-plot", "interest-rates", "fed-funds-futures", "event-trading", "volatility", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Eight times a year the FOMC meeting is the single biggest scheduled event on the calendar, but the move almost never comes from the rate decision itself — that is usually priced in beforehand. The move comes from the **surprise versus what was priced**: the statement wording change, the dot plot, and Powell's press conference.
>
> - The market pre-prices the decision through **fed funds futures**. If a hike is 96% priced, the hike landing changes nothing; only a deviation from the priced path moves price. Learn to read "what's priced" first, or you will trade the wrong thing.
> - On FOMC day the action clusters in three places: the **2:00pm statement** (diff the wording against last time), the **dot plot / SEP** (only four of the eight meetings, the quarterly projections), and the **2:30pm press conference**, where Powell can completely undo what the statement said.
> - The classic pattern is **knee-jerk then reversal**: the first move on the 2:00 statement is often faded during the 2:30 presser. The first move is frequently the *wrong* move. An FOMC day's range can be two to three times a normal day.
> - The one place surprises land hardest: the **2-year Treasury yield**. It is the market's read on the average policy rate over two years, so a hawkish dot shift or a hawkish presser repriced it instantly — in 2022-23 the 2Y ran from ~0.7% to a 5.05% peak as the Fed hiked, including four straight 75bp meetings.

On the afternoon of an FOMC decision, the Fed does exactly what everyone expected. At 2:00pm Eastern the statement crosses the wires and the central bank raises its policy rate by a quarter point — the move every economist had forecast, the move that fed funds futures had been pricing at better than ninety-percent odds for weeks. By the textbook, nothing should happen. The decision was known. Stocks twitch a little, the dollar ticks, and then the screens settle. For thirty minutes the market looks bored.

Then at 2:30pm the chair walks to the podium for the press conference, and within ten minutes the entire afternoon inverts. A single phrase — that the committee is "not even thinking about thinking about" something, or that it is "a long way" from a particular outcome, or that it would "need to see more progress" before changing course — reframes the whole decision. The first move, the calm one, reverses violently. The stock index that drifted up on the statement now sells off two percent. The two-year yield that barely moved at 2:00 jumps fifteen basis points. Traders who bought the initial pop are stopped out into the reversal. The rate decision was a non-event. The *words around it* were the event.

This is the thing almost every beginner gets backwards. They watch FOMC day for the rate decision, as if the number is the news. But the number is usually the most predictable thing on the entire calendar — the Fed telegraphs it for weeks precisely so it does not shock anyone. What is *not* fully knowable in advance, and therefore what actually moves price, is the surprise: the wording the committee chose, the path the dot plot drew, and the tone the chair struck at the podium. Learn the anatomy of FOMC day — what is priced, what can surprise, and where the surprise lands — and you stop guessing the event and start trading it. We build every piece from zero.

![FOMC day timeline showing the 2pm statement, the dot plot, the 230pm press conference, and where volatility clusters](/imgs/blogs/trading-the-fomc-statement-presser-dot-plot-1.png)

## Foundations: the meeting, the statement, the dots, and the presser

Before any trade, you need a clear picture of what the FOMC meeting *is* and what it produces. Most of the confusion around Fed days comes from people who have never separated the four distinct things that come out of one. So we start there, slowly, with no jargon left undefined.

### What the FOMC is and when it meets

The **FOMC** is the Federal Open Market Committee — the part of the US Federal Reserve that sets the country's main interest rate. It has twelve voting members: the seven Fed governors in Washington, the president of the New York Fed (always votes), and four of the other eleven regional Fed-bank presidents on a rotation. The chair (Jerome Powell, through this writing) runs the meeting and is the public face of the decision.

The single most important scheduling fact for a trader: **the FOMC meets on a fixed calendar, eight times a year**, roughly every six weeks. The dates are published a year in advance. This is why FOMC day is a *scheduled* event — unlike an inflation surprise or a geopolitical shock, you know exactly when it is coming, down to the minute. That predictability is the whole reason it is tradeable as an event: the market gets weeks to price the likely outcome, which is what creates the "priced-in versus surprise" dynamic this entire post is about. The mechanics of how the Fed actually pushes the rate to its chosen target are covered in [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates); here we focus on the *meeting day* and how it moves markets.

Each meeting lasts two days. The decision is announced on the afternoon of the second day. The output of that afternoon is not one thing — it is up to four distinct artifacts, released in a precise sequence, and a trader has to read each one differently.

### The four things an FOMC meeting produces

1. **The policy decision** — the new target range for the federal funds rate (for example, "raise the target range to 4.25%-4.50%"). Released at exactly 2:00pm Eastern. This is the headline number, and it is usually the *least* surprising thing in the whole package.

2. **The statement** — a short, formulaic document (a few paragraphs) released at the same 2:00pm. It describes the committee's view of the economy and, crucially, contains the **forward guidance**: hints about what the Fed plans to do next. The statement is built from a fixed template that changes only a few words each meeting, which is exactly what makes it readable — you diff it against last time.

3. **The Summary of Economic Projections (SEP), including the "dot plot"** — released *only four times a year* (the March, June, September, and December meetings), also at 2:00pm. The SEP is a set of forecasts from each committee member for growth, unemployment, inflation, and the policy rate. The **dot plot** is the rate part: a chart where each member places a dot for where they think the rate should be at the end of this year, next year, the year after, and in the long run. It is the committee's own picture of the *path* of rates.

4. **The press conference** — the chair takes the podium at **2:30pm**, reads a brief opening statement, and then takes reporters' questions for roughly forty-five minutes. This is live, unscripted Q&A, and it is where the chair can clarify, soften, or harden everything the 2:00 statement said. It is, empirically, the single most volatile window of the entire day.

That sequence — 2:00 decision and statement and (sometimes) dots, then 2:30 presser — is the skeleton of FOMC day. The cover figure lays it out on a timeline and marks where the volatility actually clusters: not on the decision, but on the statement diff, the dot surprise, and above all the press conference. Keep that picture in mind; everything below hangs on it.

### What "priced in" means

Here is the concept a beginner has to internalize before anything else makes sense. When traders say a rate decision is **"priced in,"** they mean the market has *already moved* in anticipation of it, so when it happens, there is nothing left to react to.

Think of it like a known birthday. If everyone in the office knows your birthday is Friday and has already chipped in for the cake, then Friday arriving is not a surprise — the cake was bought Tuesday. The "event" already happened in advance, in expectation. Markets work exactly this way. If a quarter-point hike is virtually certain, then bond yields, the dollar, and stock prices have *already* adjusted to a world with that higher rate days or weeks before the meeting. When the hike is announced, the world is unchanged from what was expected, so prices barely move. The hike was "bought Tuesday."

This is the deepest idea in event trading and it flips a beginner's intuition completely. A rate *hike* — which sounds bad for stocks — can cause stocks to *rally*, if the hike was smaller or less aggressive-sounding than what the market had already priced. And a rate *cut* — which sounds good — can cause stocks to *fall*, if the accompanying message was more cautious than priced. The direction of the price move is set by the gap between **outcome and expectation**, not by the outcome alone. We will hammer this with worked examples, because it is the single most common way people lose money around the Fed.

### Why the surprise — not the level — is what moves

It is worth being precise about *why* only the surprise moves price, because the mechanism is the whole logic of event trading and once you see it clearly you stop making the rookie error for good. The price of any asset already embeds the market's best forecast of the future. A stock's price reflects expected earnings and the rates used to discount them; a bond's yield reflects the expected path of short-term rates over its life. When a forecastable event arrives *exactly as forecast*, no new information has entered the world — the forecast was already in the price — so there is nothing for price to do. Price only moves when the world turns out *different* from what was expected, because only then does the embedded forecast need correcting.

Apply that to the Fed. Weeks before a meeting, every participant — pension funds, hedge funds, dealers, money-market funds — has formed a view on what the FOMC will do, and they have positioned accordingly. Those positions *are* the priced path: bond yields, the dollar, and equity valuations have already adjusted to the expected decision. The aggregate of all those private forecasts is what fed funds futures print. So when the decision lands as priced, every position was already set for it; the marginal buyer and seller are in balance; price is still. The "energy" was spent in the weeks of repricing *before* the meeting. What is left to release on the day is only the part that *nobody had correctly forecast* — the surprise.

This is why a single changed word in the statement, or a single dot moving 25 basis points, can move billions in a heartbeat while a fully-expected half-point hike does nothing. The word and the dot are *new information*; the hike is *old news*. A useful way to hold it: **the meeting does not inject the decision into the market — the market injected the decision into prices weeks ago. The meeting injects only the error in the market's forecast.** Your job as a trader is to estimate that error — the surprise — before and as it lands, because the surprise is the only thing that pays.

A second consequence follows, and it explains the violence of FOMC reactions. Because everyone has crowded into the same priced view, the positioning going into the meeting is often *lopsided* — the whole market leaning one way. When the surprise comes the other way, all those positions have to be unwound at once, and a forced unwind moves price far more than the information alone would justify. A modest dovish surprise into a market positioned for hawkishness does not produce a modest rally; it produces a stampede of short-covering. This is why FOMC moves overshoot, why the range is multiples of a normal day, and why the *positioning* into the meeting matters as much as the surprise itself. The bigger the consensus, the bigger the unwind when it breaks.

### Fed funds futures: the market's printed expectation

So how do you *know* what is priced in? You do not have to guess — there is a market that prints it for you. **Fed funds futures** are exchange-traded contracts whose price reflects the market's expectation of the average fed funds rate over a given month. Because they trade continuously, they give you a live, numerical readout of what the market thinks the Fed will do at upcoming meetings, expressed as a probability.

The convention is simple once you see it: a fed funds futures contract is quoted as **100 minus the expected average rate**. A contract trading at 96.50 implies an expected rate of `100 − 96.50 = 3.50%`. If the same contract trades at 96.25, the implied expected rate is 3.75%. As the market's expectation of the Fed's next move shifts, the futures price shifts, and you can back out exactly what rate — and therefore what probability of a hike or cut — is baked into the price. (Data vendors and the CME's "FedWatch" tool do this arithmetic for you and show it as "the market prices a 92% chance of +25bp," but you should understand the mechanism underneath.)

This is your single most important pre-meeting tool. **Before any FOMC meeting, the first thing a trader checks is fed funds futures: what is priced?** Everything else — the statement, the dots, the presser — is then read as *surprise relative to this number*. If you skip this step, you are trading the rate decision as if it were news, which is the rookie error this whole post is built to prevent.

### The blackout period and how expectations get set

One more piece of the anatomy shapes how the priced path forms in the days before the meeting: the **blackout period.** Starting the second Saturday before a meeting and running through the Thursday after it, Fed officials are barred from speaking publicly about monetary policy. This matters for trading because it means the *last* communication the market gets before the decision is the speeches and interviews in the window *just before* blackout begins. Officials know this, so they often use those final pre-blackout appearances to steer expectations toward where they want the market positioned — nudging fed funds futures toward the outcome they intend to deliver, precisely so the decision lands as a non-surprise.

This is the mechanism by which the decision becomes "priced in." The Fed does not want to shock markets, so it telegraphs. A well-placed speech or a *Wall Street Journal* article from a known Fed-watcher reporter (sometimes called the "Fed whisperer") in the week before blackout can move fed funds futures sharply, locking in the priced path before anyone is silenced. By the time the meeting arrives, the market has usually been guided to the right answer on the *decision* — which is exactly why the surprise has to come from the harder-to-telegraph parts: the precise wording, the dot dispersion, and Powell's live answers. Understanding the blackout rhythm tells you *when* the priced path gets set (the pre-blackout window) and *why* the decision is rarely the surprise (the Fed spent that window removing the surprise). When you build your pre-meeting view, the pre-blackout commentary is where you calibrate what is priced.

## The realized path versus the priced path

There are two "paths" you have to keep straight, and confusing them is a classic mistake. There is the **realized** policy path — the actual rate the Fed has set, which only changes on FOMC days — and the **priced** path — the market's forecast of where the rate is going, which moves every second in fed funds futures and in the 2-year yield.

The realized path is a staircase. The rate sits flat between meetings and steps up or down only when the FOMC acts. Figure 2 draws that staircase for the last cycle: the emergency cut to near-zero in March 2020, the long flat bottom, then the violent climb of 2022 — including **four consecutive 75-basis-point hikes** from June to November 2022, the steepest part of the staircase and the part that taught a generation of traders to respect the meeting. The rate then plateaus at a 5.50% upper bound through 2023 before the easing begins.

![Fed funds target upper bound step chart 2019 to 2024 with the 2022 75bp hikes marked](/imgs/blogs/trading-the-fomc-statement-presser-dot-plot-2.png)

Notice the shape: long flat stretches punctuated by sudden steps, and every step lands on a scheduled FOMC date. That is the realized path. It is *backward-looking* — it tells you what the Fed has already done. The realized path is not what you trade; by the time a step prints, the futures had already priced it.

What you trade is the *priced* path, and how it shifts when the meeting reveals something the priced path did not expect. The priced path lives in fed funds futures and, most readably, in the 2-year Treasury yield (we will get to why the 2Y is the key instrument). The whole game of FOMC trading is: the priced path is some curve; the meeting either confirms that curve (no trade) or bends it (the trade). Figure 3 makes this explicit.

![Priced path versus surprise diagram showing only the deviation from expectations moves price](/imgs/blogs/trading-the-fomc-statement-presser-dot-plot-3.png)

The left column is what the market walked in already pricing: the hike, the projected path of future moves, and the expected tone of the chair. The right column is what actually happens to price — and the lesson is that the expected pieces produce *near-zero* reaction (top row), while only the *deviations* move markets: a dot plot that shifts up versus what was priced sends the 2-year up and stocks down (a hawkish surprise), and a chair who sounds more dovish than feared produces a relief rally even on a hike (a dovish surprise). Same decision; opposite outcomes; the difference is the gap from pricing.

A quick vocabulary note, because these two words appear constantly. **Hawkish** means leaning toward tighter policy — higher rates, more worried about inflation. **Dovish** means leaning toward looser policy — lower rates, more worried about growth and jobs. A "hawkish surprise" is an outcome more aggressive on rates than priced; a "dovish surprise" is the opposite. Markets care about the *surprise direction*, not the absolute stance.

#### Worked example: reading fed funds futures into an implied rate

Suppose the fed funds futures contract for next month is trading at **96.50**. By the quoting convention, the implied expected average rate is `100 − 96.50 = 3.50%`. The current target range is 3.25%-3.50% (a 3.50% upper bound), so the futures are saying: the market expects the rate to *stay put* — no hike priced for the meeting in that month.

Now news comes out — a hot inflation print — and the contract falls to **96.25**. The new implied rate is `100 − 96.25 = 3.75%`. The market has shifted from pricing "no change" to pricing a full quarter-point hike to a 3.75% upper bound. Nothing has happened at the Fed yet; the *expectation* moved, and the futures repriced 25bp of tightening into existence.

When the meeting then arrives, what matters is whether the Fed delivers relative to that 96.25 / 3.75% expectation. If the Fed hikes to 3.75% exactly as priced, the futures barely move on the day — it was already in the price. If the Fed *holds* at 3.50% (a dovish surprise versus the priced hike), the futures snap back up toward 96.50 and risk assets typically rally. **The takeaway: the futures price is the expectation you measure every surprise against, and a 0.25 move in the contract is a full quarter-point of policy repriced.**

## Reading the statement: diff the wording

The 2:00pm statement is the first artifact to cross, and the skill of reading it is almost mechanical: you **diff it against the previous statement.** The statement is built from a stable template, so most of it is identical meeting to meeting. The handful of words that *changed* are the entire signal. Wire services and Fed-watchers publish a redlined version within seconds — additions in one color, deletions in another — and the market reacts to the changes, not the boilerplate.

Two sentences in the statement carry almost all the weight:

- **The forward-guidance sentence** — the line about what the committee anticipates doing next. The classic tell is whether a phrase like "the Committee anticipates that some additional policy firming may be appropriate" is *present*, *softened*, or *removed*. When the Fed dropped its explicit tightening-bias language and shifted to data-dependent wording, that was a dovish signal even with no change to the rate. The presence, softening, or removal of the firming-bias phrase is the most-watched single edit in the document.

- **The risk-assessment sentence** — how the committee characterizes the balance of risks. A shift from "inflation remains elevated" to "inflation has eased" is dovish; a shift toward language stressing that risks "remain tilted to the upside" is hawkish. The committee also signals through which mandate it emphasizes — inflation or employment — which connects directly to its [reaction function and the dot plot](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot).

Figure 5 lays out the statement-diff method as a repeatable procedure.

![Statement diff method pipeline showing how the changed wording becomes the trading signal](/imgs/blogs/trading-the-fomc-statement-presser-dot-plot-5.png)

The procedure: pull last meeting's statement and this one side by side; diff the guidance line; diff the risk line; score the net change as more hawkish or more dovish than the prior statement; then — and this is the step beginners skip — compare that net change against what was *priced*. A statement that drops the firming bias is only a dovish *surprise* if the market had not already expected the Fed to drop it. If everyone knew the firming bias was going away, removing it confirms the price and moves nothing. The statement is read for its delta versus last time *and* its delta versus expectations. Both diffs matter.

One subtlety worth flagging: the *dissents*. The statement lists which members voted against the decision and which way (a member preferring a larger hike is a hawkish dissent; one preferring no hike is dovish). A surprise dissent — especially the first in a long while, or a governor rather than a regional president — is a signal the consensus is fraying, and markets read it as a hint about the *next* meeting. The vote tally is part of the statement diff.

## The dot plot: the committee's picture of the path

When the meeting is one of the four with a Summary of Economic Projections, the dot plot is often the biggest single mover, because it is the Fed telling you, in numbers, where it thinks the *path* of rates is going — not just today's decision but the trajectory.

Here is exactly what the dot plot is. Each FOMC participant — all nineteen of them, voters and non-voters alike — anonymously marks a dot for the level they think the federal funds rate *should* be at the end of the current year, the end of each of the next two years, and "in the longer run" (their estimate of the neutral rate the economy gravitates to). Stack all the dots for each year into a column and you get a scatter. The **median dot** — the middle value — is taken as the committee's central projection for that year. The *spread* of the dots is how much they disagree.

So the dot plot encodes two completely different signals, and reading them as one thing is a common error. Figure 6 separates them.

![Dot plot reading matrix median dot is the path the dispersion is the uncertainty](/imgs/blogs/trading-the-fomc-statement-presser-dot-plot-6.png)

- **The median dot for each year is the projected path.** It is the committee's best guess of where rates will be. When the median dot for next year moves up versus the prior SEP, that is the committee signaling a higher-for-longer path — a hawkish surprise if the market had been pricing cuts. The shift in the median dot from one quarterly SEP to the next is one of the cleanest hawkish/dovish signals the Fed produces.

- **The dispersion of the dots is the committee's uncertainty.** Tightly clustered dots mean high agreement — the projected path is firm. Widely scattered dots mean the committee itself does not know — the path is a guess, and you should weight it less. The longer-run dot, in particular, is usually scattered, because nobody agrees on where neutral is. The dispersion tells you how much *conviction* sits behind the median, which is exactly how much weight to put on it as a trade.

The crucial discipline — and this is a "common misconception" we will return to — is that **the dots are projections, not promises.** They are each member's view *as of that meeting*, conditional on their economic forecast. When the data changes, the dots change. The Fed has repeatedly drawn a dot plot showing, say, three hikes for the coming year and then delivered something completely different as conditions shifted. The dot plot is a snapshot of the committee's *current* thinking about the path, not a commitment. You trade the *change* in the dots versus expectations on the day; you do not bank on the dots being delivered months later. The path the dots imply, and how the market prices the terminal rate and the cutting cycle, is the subject of [the terminal rate and rate-cut cycles](/blog/trading/macro-trading/terminal-rate-and-rate-cut-cycles-pricing-the-path).

### The dots are downstream of the rest of the SEP

The dot plot gets all the attention, but a sharper reader checks the *rest* of the Summary of Economic Projections too, because the dots are downstream of it. Alongside the rate dots, every participant submits projections for **real GDP growth**, the **unemployment rate**, and **inflation** (both headline and core PCE), for this year, the next two years, and the longer run. The rate dots are not arbitrary — each member draws the rate path that, given their own growth-inflation-unemployment forecast, would deliver the dual mandate. So the SEP is internally linked: a member who marks higher inflation and lower unemployment will, to be consistent, mark a higher rate dot.

This linkage gives you two extra signals. First, you can sanity-check the dots against the macro forecast: if the median dot rose but the inflation and growth projections barely changed, the committee has turned more hawkish *for the same economy* — a pure stance shift, which is a stronger signal than dots that merely tracked a higher inflation forecast. Second, the *direction of the forecast revisions* foreshadows future dot moves. If the committee marks up its inflation projection while holding the dots, the market infers that the dots are likely to follow upward at the next meeting — a delayed hawkish tell. The SEP is a connected web; the dots are the visible thread, but the inflation and unemployment numbers are where the committee's *view of the economy* — its reaction function in action — actually lives. Pairing the dot shift with the forecast shift tells you whether the path changed because the economy changed or because the committee's tolerance changed, and those are different trades.

There is one more SEP subtlety worth internalizing: the **longer-run rate dot** is the committee's estimate of the *neutral* rate — the rate that neither stimulates nor restrains the economy over time. It moves slowly and the dots for it are usually widely scattered (nobody agrees on neutral), but when the *median* longer-run dot drifts — say from 2.5% toward 3.0% — it tells you the committee now believes the economy can sustain higher rates indefinitely, which is a structurally hawkish signal that reprices the entire long end of expectations, not just the next year. A drifting longer-run dot is a slow, powerful signal that beginners miss because they only look at the near-year column.

#### Worked example: a hawkish dot-plot surprise

Go into a September SEP meeting. Fed funds futures and the 2-year yield imply the market expects the median dot for *next year* to show the rate ending around **3.50%** — that is, a couple of cuts from the current 4.25% upper bound, then a pause. The dot plot is released at 2:00pm and the median dot for next year prints at **4.00%** instead. The committee, in aggregate, has shifted its projected path **up by 50 basis points** versus what was priced. It is signaling fewer cuts — higher for longer.

This is a hawkish surprise, and it lands hardest on the front end. The 2-year Treasury yield, which reflects the expected average policy rate over two years, jumps as the market reprices a higher path: say it moves from 4.30% to 4.50%, a 20bp leap in minutes. Higher rates for longer are a discount-rate headwind for equities, so stock indices fall — an S&P move of −1.5% in the half hour after the dots is entirely ordinary on a surprise of this size. The dollar firms (higher US rates attract capital). In dollar terms the front-end move is brutal: on a \$10,000,000 position in 2-year notes (a DV01 near \$1,900 per basis point), that 20bp jump is a \$38,000 loss booked in thirty minutes, and a \$50,000,000 book bleeds \$95,000 before the press conference even begins. Notice the rate *decision* that same day might have been a perfectly expected 25bp cut; the cut was priced, the *dots* were not, and the dots drove the day. **The takeaway: a 50bp upward shift in the median dot versus the priced path is a hawkish surprise that the 2-year yield prices in minutes, dragging stocks down even if the rate move itself was as expected.**

#### Worked example: the priced-versus-actual gap — a hike that rallies risk

Now the counterintuitive case that proves the whole thesis. The Fed is in a hiking cycle and the market is *terrified* of how hawkish it might be. Fed funds futures price a 25bp hike as certain, but the *bigger* fear — visible in the 2-year yield and in options pricing — is that the dot plot will signal three or four more hikes ahead and that Powell will sound aggressive. So the market walks in positioned for pain: stocks have sold off into the meeting, the 2-year is elevated, hedges are on.

The Fed hikes 25bp exactly as priced. But the dot plot shows only *one* more hike penciled in, not three, and Powell's tone at the presser is measured — he notes progress on inflation and says the committee is "getting closer" to the end. Relative to a market braced for something much more hawkish, this is a **dovish surprise even though the Fed hiked.** The 2-year yield *falls* as the market unwinds the extra hikes it had priced. Stocks *rally* — a +2% relief move — because the realized outcome was less hawkish than the priced outcome. Shorts cover, hedges come off, and the index that "should" have fallen on a rate hike closes sharply higher.

This is the single most important pattern to internalize: **the sign of the rate move tells you almost nothing; the sign of the surprise tells you everything.** A hike can be bullish (less hawkish than priced) and a cut can be bearish (more cautious than priced). The takeaway: when the market is positioned for maximum hawkishness, a merely-hawkish outcome is a dovish surprise, and risk assets rally on the relief.

## The press conference: where Powell can undo the statement

If the statement and dots are the scripted part of FOMC day, the press conference is the live wire. At 2:30pm the chair takes questions, unscripted, for roughly forty-five minutes, and this window is — by a wide margin — the most volatile of the day. The reason is structural: the statement is a few carefully negotiated paragraphs that cannot capture nuance, while the presser is the chair explaining, qualifying, and emphasizing in real time. The chair can make a hawkish-looking statement sound dovish, or a dovish-looking statement sound hawkish, simply by what he stresses and how he answers a pointed question.

This is why the **knee-jerk move on the 2:00 statement is so often reversed.** The first reaction is the algorithms and fast traders pricing the statement diff and the dots in milliseconds. But the *full* information set is not complete until Powell speaks. A statement that reads hawkish can be softened the moment Powell says the committee will be "patient" or "data-dependent" or notes that policy is "well into restrictive territory." A statement that reads dovish can be hardened when Powell refuses to declare victory on inflation or pushes back on market expectations of cuts. The 2:30-to-3:00 window routinely erases or doubles the 2:00 move.

A few presser dynamics worth knowing:

- **The chair pushes back on market pricing.** A recurring move is the chair using the presser to *fight* what the market has priced. If futures price aggressive cuts and the Fed does not want financial conditions to ease prematurely, the chair will pour cold water on cut expectations — a hawkish presser that reverses a dovish statement reaction. Conversely, if the market is pricing a hard, prolonged tightening that the Fed thinks is overdone, the chair can lean dovish to talk it down.

- **The "is this the last hike / first cut?" question.** Reporters always probe for the turning point. How the chair handles whether the cycle is ending — whether he leaves the door open or shuts it — is usually the biggest single mover of the presser, because it reprices the *path*, not just one meeting.

- **The opening statement versus the Q&A.** The chair reads a prepared opening (often a continuation of the written statement's tone), then takes questions. The Q&A is where the unscripted reversals happen — an off-the-cuff answer to a sharp question can move markets more than anything prepared. The first move within the presser is often itself reversed by a later answer.

The practical consequence for a trader is enormous: **you do not have your full information until the press conference is well underway.** Acting on the 2:00 move alone is trading on half the data. We will turn that into a concrete rule in the playbook.

## The front end is where surprises land

We have said repeatedly that the surprise "lands on the 2-year." It is worth making that precise, because it tells you *which instrument* to watch and trade.

A bond's yield reflects the average short-term interest rate expected over its life, plus a small term premium. The **2-year Treasury yield** therefore reflects the market's expectation of the average fed funds rate over the next two years. That is *exactly* the horizon the FOMC's decisions and dots speak to. So when the Fed surprises on the path — hawkish dots, a hawkish presser — the 2-year reprices almost instantly, because the thing it represents (the expected two-year average policy rate) just changed. The 10-year moves too, but less and more slowly, because it is dominated by longer-run growth and inflation expectations the FOMC affects only indirectly.

Figure 4 shows the 2-year through the 2022-23 cycle, overlaid on the realized policy staircase.

![Two year Treasury yield versus fed funds 2020 to 2026 showing the front end reacts to FOMC](/imgs/blogs/trading-the-fomc-statement-presser-dot-plot-4.png)

Two features are the whole point. First, the 2-year *leads* the policy rate: it climbs ahead of the hikes as the market prices them in, because it is the expectation and the funds rate is the realization. Second, the 2-year *peaked before the last hike* — it topped near 5.05% in October 2023 while the Fed was still at its plateau, because the market had begun pricing the *end* of the cycle and the cuts beyond it. The gap between the dashed realized line and the solid 2-year line is, quite literally, the expected future path of policy. When an FOMC surprise bends that path, the 2-year is the instrument that moves first and most. If you want to know whether a meeting was hawkish or dovish in one number, look at how the 2-year closed versus where it opened.

This is also why the 2-year is where event traders express an FOMC view directly — through the 2-year note, fed funds futures, SOFR futures, or short-dated rate options — rather than through the 10-year or through stocks, which carry a lot of non-Fed noise. The front end is the cleanest read on, and the cleanest expression of, a policy surprise. How the front end then relates to the rest of the curve — and what an inverted curve says about the cycle — is the subject of [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession).

#### Worked example: decomposing a 2-year move into "path" versus "level"

It helps to split a 2-year move into its two components, because it tells you whether a meeting changed *today's* rate or the *future path* — and the path is what FOMC surprises move. The 2-year yield is approximately the average expected overnight rate over the next two years (ignore the small term premium for this example). Suppose the current overnight rate is 4.50% and, before a meeting, the market expects it to average 4.80% over the next two years — so the 2-year yields about 4.80%. The 30bp gap above today's 4.50% is the *priced path*: the market expects rates to be net higher over the horizon.

Now the meeting delivers a hawkish dot surprise — the projected path lifts. The market revises its expected two-year average rate from 4.80% to 5.00%. The 2-year yield therefore jumps about 20bp, from 4.80% to 5.00% — and crucially, *the overnight rate has not moved at all today*. The entire 2-year move came from repricing the *future path*, not from any change in the current rate. That is the signature of an FOMC path surprise: the front end moves while today's policy rate is unchanged (or even moves the *opposite* way, as in the hawkish-cut case, where the overnight rate drops 25bp but the 2-year rises because the path was lifted). **The takeaway: a 2-year move on FOMC day is almost entirely about the repriced *path* of rates, which is exactly why the dots and the presser — both path signals — move it far more than the single decision does.**

#### Worked example: the 2022 sequence — four 75bp meetings and the 2-year

Walk the 2022 tightening through the front end, using the realized staircase from Figure 2 and the 2-year from Figure 4. Entering 2022 the 2-year yielded about **0.73%** (end-2021) — the market priced an easy Fed for years. Then inflation forced a violent repricing. The Fed went 25bp in March, 50bp in May, then **four straight 75bp hikes** at the June, July, September, and November meetings — the most aggressive burst in forty years.

Watch the 2-year track the *expected* path, not just the realized one. By March 2022 it was already at **2.28%**, far above the 0.50% upper bound the Fed had actually reached — the front end had priced most of the coming hikes in advance. By September 2022, with the funds upper bound at 3.25%, the 2-year was at **4.22%** — again pricing well beyond the current rate, anticipating the hikes still to come. The 2-year was not reacting to each hike as news; it was running ahead of the staircase because each hawkish meeting and each hawkish dot revision *raised the priced path*.

The trade implication: a trader short the 2-year note (positioned for higher front-end yields) through this sequence made money not on the hikes themselves but on the *repeated upside surprises* — each meeting where the dots ratcheted higher and Powell refused to blink pushed the priced path up another notch, lifting the 2-year. The 2-year finally peaked near **5.05% in October 2023**, *ahead* of the last hike, when the market judged the surprises were finally exhausted. **The takeaway: through 2022-23 the 2-year priced the path ahead of the Fed, and each hawkish FOMC surprise — not each hike — was what pushed it higher, until the surprises ran out and it peaked before the final hike.**

## Common misconceptions

Five beliefs cost beginners real money on FOMC day. Each is corrected with a number or a mechanism.

**"The rate decision is the event."** It is the *least* surprising part. The Fed telegraphs the decision for weeks precisely so it does not shock markets; by meeting day a hike or hold is typically 90%+ priced in fed funds futures. The event is the *surprise* — the statement wording, the dots, the presser — relative to that pricing. In the dovish-hike example above, the rate decision (a hike) was bullish for stocks (+2%) because the surprise was dovish. If you trade the decision, you are trading the one thing that already happened in advance.

**"The first move is the right move."** The 2:00 knee-jerk on the statement is frequently reversed during the 2:30 press conference. The first reaction is fast algorithms pricing an *incomplete* information set — the full picture is not in until Powell speaks. A measurable fraction of FOMC days see the post-statement move reverse sign during the presser. Chasing the first tick is how you get stopped out into the reversal. The first move is a hypothesis, not a conclusion.

**"The dots are promises."** They are conditional projections, anonymous, as of that meeting, and they change when the data changes. The Fed has drawn dot plots implying several hikes and then done the opposite as conditions shifted. The *change* in the median dot versus expectations is a tradeable signal *on the day*; the dots as a *forecast you can bank on months out* are unreliable. Trade the surprise in the dots, not the literal dot.

**"A hike is bearish and a cut is bullish."** Only relative to expectations. A hike that is less hawkish than priced rallies risk; a cut delivered with cautious "we are not committing to more" language can sell off. The sign of the policy move tells you little; the sign of the *surprise versus pricing* tells you almost everything. This is the inversion that catches every beginner.

**"FOMC day is a normal trading day."** The realized range on an FOMC afternoon is routinely two to three times a normal day, and almost all of it is compressed into the 2:00-to-3:30 window. Volatility spikes, spreads widen, and stops get run by the whipsaw. Sizing a position the way you would on a quiet Tuesday is how a correct *view* turns into a losing *trade* — you get stopped out by noise before your thesis plays out. Size for the vol, not for the conviction.

## How it shows up in real markets

These patterns become concrete in a few recurring shapes. None of these requires you to *predict* the Fed; they require you to read the surprise and the sequence correctly.

### The 2:00-to-2:30 reversal pattern

The textbook FOMC-day shape is: a knee-jerk move on the 2:00 statement and dots, then a reversal during the 2:30 press conference. It happens because the 2:00 reaction prices the *written* word — fast and mechanical — while the presser delivers the *interpreted* word, which can land differently. A statement that reads hawkish (firming bias retained, inflation-risk language stressed) gets an initial hawkish reaction; then Powell, in Q&A, emphasizes the progress already made and the lags in policy, and the move fades. The reverse happens too: a dovish-looking statement followed by a Powell who refuses to validate market expectations of cuts, hardening the tone and reversing the dovish pop.

The practical read: treat the 2:00 move as *provisional*. The market is voting on incomplete information. The 2:30-to-3:00 window — when the chair has spoken enough to reframe the decision — is where the day's *real* close-to-close move usually sets up. Several of the largest single-day equity-index reversals in recent years happened precisely in this window, with the index swinging from up to down (or down to up) by two to three percent between the statement and the late presser.

### The hawkish-cut surprise

A particularly tradeable configuration: the Fed *cuts* (which sounds dovish) but pairs it with a hawkish message — a dot plot showing few further cuts, or a chair who frames the cut as a "recalibration" rather than the start of an easing cycle, or who stresses that inflation is not yet beaten. The market, which may have been pricing an aggressive cutting cycle, has to unwind those expectations. The result is the strange-looking outcome of a *rate cut* on which the 2-year yield *rises* and stocks *fall* or fade — because the path was repriced higher even though today's move was lower. This is the "hawkish cut," and it is a clean illustration that the path (the dots and the presser) dominates the single decision.

### The dovish-hold surprise

The mirror image: the Fed *holds* (no change, which sounds neutral-to-hawkish if a cut was hoped for) but pairs it with a dovish message — a statement that drops the firming bias, dots that pencil in cuts ahead, a chair who signals the next move is likely down. The market reads the *path* as turning even though today nothing changed. The 2-year *falls*, stocks *rally*, the dollar softens. A "dovish hold" can be one of the biggest risk-on days of a cycle precisely because it signals the inflection — the moment the market becomes confident the tightening is over and easing is coming.

Both patterns make the same point in opposite directions: **the decision is one data point; the path (dots plus presser) is the trade.** A cut can be hawkish and a hold can be dovish, depending entirely on the message wrapped around it.

### A walk through one real meeting structure

To make the sequence concrete, walk a stylized but realistic SEP meeting from the 2022-23 hiking cycle, minute by minute, using the front-end levels from Figure 4. Going in, the funds upper bound is at 4.50% and fed funds futures price a 25bp hike to 4.75% at roughly 95% odds — so the hike is *priced*. The 2-year yield sits near 4.40%. The market's bigger question is the path: futures price the cycle topping around 5.0% with cuts beginning later in the year. That is the priced path you measure everything against.

At 2:00 the statement crosses: a 25bp hike, exactly as priced, and the firming-bias sentence is *retained* ("ongoing increases will be appropriate"). The dot plot lands simultaneously and the median dot for the year-end rises to 5.1% from 4.6% at the prior SEP — a **half-point upward shift in the projected path** versus three months earlier, and above the ~5.0% the market had priced. This is a hawkish surprise on the path. The knee-jerk is immediate: the 2-year jumps from 4.40% toward 4.55% in minutes, the dollar firms, equity futures drop roughly 1%. So far, the day is textbook hawkish.

Then at 2:30 Powell takes the podium, and the reversal risk is live. Suppose, in Q&A, he repeatedly stresses that "the disinflationary process has begun" and that the committee can "afford to be patient" — language softer than the hawkish statement. The 2:00 move now fades: the 2-year gives back part of its spike, equities claw back from −1% toward flat or higher. The *durable* close-to-close move is much smaller than the 2:00 knee-jerk implied, because the presser partially undid the statement. A trader who shorted the 2-year aggressively on the dot surprise at 2:05 — chasing the knee-jerk — gets squeezed as Powell reframes; a trader who waited for the presser to read the *net* of statement-plus-Powell sized correctly and on the right side. This is the entire playbook in one afternoon: the dots were hawkish, the presser softened them, and the surprise that actually paid was the *combination*, knowable only after 2:30.

### When the decision itself is the surprise

The one case where the rate decision *is* the event is when the Fed does something fed funds futures did *not* fully price — an unexpected 50bp when 25bp was priced, a surprise hold when a hike was priced, or an inter-meeting emergency move. These are rare precisely because the Fed hates to shock markets, but when they happen the reaction is large and immediate, because the priced path was wrong about the *decision*, not just the message. The March 2020 emergency cut to near-zero (Figure 2) is the canonical example — an unscheduled, fully surprising move that repriced everything at once. When the decision deviates from pricing, the decision *is* the surprise; the rest of the time, it is not.

## How to trade it: the playbook

Everything above converges on a small set of rules. The goal is not to *predict* the Fed — that is a fool's errand against a market that prices it for weeks — but to read the surprise correctly and to survive the volatility. Figure 7 contrasts the trap (chasing the knee-jerk) with the disciplined trade.

![FOMC day playbook before-after the knee-jerk trap versus the disciplined trade](/imgs/blogs/trading-the-fomc-statement-presser-dot-plot-7.png)

**1. Establish what is priced before the meeting.** Pull fed funds futures (or the FedWatch implied probabilities) and note the priced decision and the priced *path*. Note where the 2-year is trading. Write down, explicitly, what outcome would be a hawkish surprise and what would be a dovish surprise relative to this pricing. If you cannot state the surprise thresholds in advance, you are not ready to trade the meeting. **The signal you are measuring is always outcome-minus-pricing, and you must know the pricing first.**

**2. Diff the statement, separate the dots.** At 2:00, read the redlined statement: did the guidance/firming line change? Did the risk line change? Were there surprise dissents? If it is an SEP meeting, read the median dot for next year *versus the prior SEP and versus what was priced* (the surprise), and note the dispersion (how much to trust it). Score the net as hawkish or dovish *relative to pricing*, not relative to nothing.

**3. Respect the presser reversal — do not chase the 2:00 tick.** The single most important execution rule. The 2:00 move is provisional; the press conference at 2:30 can reverse it. Unless the statement/dots delivered an unambiguous, large surprise versus pricing, **wait for Powell.** Let the chair confirm or undo the initial read before you commit size. Most blown FOMC trades are entries into the 2:00 spike that the presser then reverses. Patience through the first thirty minutes is an edge.

**4. Trade the surprise, in the front end.** Where the outcome clearly diverged from pricing, the cleanest expression is the front end — the 2-year, fed funds/SOFR futures, or short-dated options — because that is where the path-surprise lands first and with least noise. A hawkish path surprise: position for higher front-end yields and for risk-off (the 2-year up, the dollar up, equities down). A dovish path surprise: the mirror. Stocks and the long end react too, but with more cross-currents; the front end is the purest read.

**5. Size for the volatility, set wide stops, respect the 3:30 settle.** An FOMC range is 2-3x a normal day, so a position sized for a quiet day will be stopped out by noise even when the view is right. Cut size (half or less of normal), give stops room to clear the whipsaw, and recognize that the *durable* close-to-close move usually does not settle until after the presser (around 3:00-3:30). The repricing into the new path can then run for *days* — the meeting's information bleeds into the front end well after the close. Trade the path repricing, not the first-hour chaos.

**The invalidation.** Your read is wrong, and you should be out, when the 2-year moves *against* your surprise thesis after the presser has fully digested — for example, you judged the meeting hawkish and positioned for higher front-end yields, but by the 3:30 settle the 2-year is *lower*. The front end is the scoreboard: if it disagrees with your hawkish/dovish call once the presser is done, the market has read the surprise the other way, and you defer to the scoreboard rather than your interpretation. Likewise, if pre-meeting you cannot identify a *gap* between pricing and a plausible outcome — if the meeting looks fully priced with no realistic surprise — the correct trade is *no trade*: there is no edge in betting on an outcome the market has already nailed.

Put together, the discipline is small and repeatable: know what's priced, read the surprise across statement-dots-presser, wait for the press conference, express it in the front end, and size for the vol. You will not catch every move, and you will sometimes be on the wrong side of a reversal. But you will have stopped guessing the Fed and started trading the one thing that is actually tradeable on FOMC day — the gap between what the market expected and what the committee delivered.

## Further reading & cross-links

- [Inflation and the Fed reaction function: reading the dot plot](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot) — how the committee decides what to do, and why the dots reflect a reaction function rather than a forecast.
- [The terminal rate and rate-cut cycles: pricing the path](/blog/trading/macro-trading/terminal-rate-and-rate-cut-cycles-pricing-the-path) — how the market prices the *whole* future path the dots and presser describe.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the mechanics of how the FOMC's chosen target actually becomes the market's overnight rate.
- [Reading the yield curve: slope, inversion, recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) — how the front-end repricing from FOMC surprises propagates into the shape of the whole curve.

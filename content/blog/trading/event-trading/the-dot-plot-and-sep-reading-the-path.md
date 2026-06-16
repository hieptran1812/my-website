---
title: "The Dot Plot and the SEP: Reading the Fed's Path"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "What the FOMC's Summary of Economic Projections and its famous dot plot actually are, how to read the median dot, the dispersion, the terminal rate and the longer-run anchor, and why markets trade the gap between the dots and what futures already price."
tags: ["event-trading", "macro", "dot-plot", "sep", "fomc", "fed-funds", "terminal-rate", "neutral-rate", "treasury-yields", "rates-trading"]
category: "trading"
subcategory: "Event Trading"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — Four times a year the Fed publishes a set of projections, and inside them sits the "dot plot": every official's guess at where the policy rate will be at year-end. Markets don't trade the dots in isolation — they trade the **gap** between the median dot path and the path the market itself has already priced through interest-rate futures. Close that gap and the tape moves.
>
> - The **SEP** (Summary of Economic Projections) is the Fed's quarterly forecast for growth, unemployment, inflation and the policy rate. The **dot plot** is the rate part: 19 dots, one per official, for each of the next few year-ends plus a "longer-run" value.
> - The number that moves markets is the **median dot** for each horizon, and how it changed versus the *last* SEP. A median that rises a notch is hawkish; a notch lower is dovish — relative to what was already in the price.
> - The trade is the **gap**: the dots say one path, fed funds futures price another. When data forces the market toward the dots, the front end of the curve reprices fast — and 2-year yields, rate-sensitive stocks, gold and the dollar all move with it.
> - The one thing to remember: **the dots are projections, not promises.** They have been wrong by hundreds of basis points within a single year. Trade the *repricing*, not the forecast.

## A meeting where the dots moved two notches

It is 2:00 p.m. on a Wednesday in the spring of 2022. The Federal Reserve has just released its statement and, alongside it, a fresh Summary of Economic Projections. For the previous few minutes the market has been calm — the rate decision itself was fully expected, priced to the basis point. Then the dot plot hits the wire, and within ninety seconds the entire short end of the Treasury curve lurches.

What happened was not the rate decision. It was the *forecast*. The median official's projection for where the policy rate would sit at the end of the year had jumped sharply higher than the previous quarter's projection — the committee was now telling the world it expected to raise rates much faster and much further than it had said just three months earlier. Traders who had been pricing a gentle path of hikes suddenly had to reprice for a steep one. Two-year Treasury yields, the part of the curve most sensitive to the expected path of the Fed, ripped higher in minutes. Rate-sensitive growth stocks sagged. The dollar firmed. The dots had moved up a couple of notches, and the whole curve repriced toward them before the chair had even sat down for the press conference.

This is the strange power of the dot plot. The Fed insists, every single time, that the dots are *not a plan* and *not a commitment* — they are each official's best guess, made on a particular day, and they will change as the data changes. And yet a shift in the median dot can move trillions of dollars of bonds in the span of a coffee break. The reason is the same idea that runs through this whole series: markets trade *expectations*, and the dots are the cleanest published snapshot of what the people who set the rate expect to do. When that snapshot moves relative to what the market had already priced, somebody has to be wrong — and the repricing toward the new information is the trade. This post is about how to read that snapshot, why it matters, and how to position around it. For the mechanism that produces the dots — the Fed's reaction function — lean on the companion piece on [inflation and the Fed's reaction function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot).

![Stylized dot-plot grid with the median, dispersion, terminal rate and longer-run dot labeled](/imgs/blogs/the-dot-plot-and-sep-reading-the-path-1.png)

## Foundations: what the SEP and the dot plot actually are

Let us build this from absolute zero, because the dot plot is one of those market objects that sounds intimidating and is, underneath, a very simple grid of guesses.

### The FOMC and the policy rate

The **Federal Open Market Committee (FOMC)** is the group inside the U.S. Federal Reserve that sets the country's main interest rate. That rate is the **federal funds rate** — the rate at which banks lend reserves to each other overnight. The Fed doesn't dictate a single number; it sets a **target range**, usually 25 basis points wide (a basis point, "bp", is one hundredth of a percentage point — so 0.25% is 25bp). When people say "the Fed funds rate is 5.25%–5.50%", the 5.50% is the upper bound of that range. Almost every other interest rate in the economy — mortgages, business loans, the yield on short-term Treasury bills — keys off this rate, which is why the world watches it so closely. If you want the full machinery of how the Fed moves this rate, the companion explainer on [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) covers it.

The FOMC meets eight times a year. At every meeting it decides whether to raise, cut, or hold the target range. But at **four** of those eight meetings — roughly March, June, September and December — it does something extra: it publishes the **Summary of Economic Projections**.

### Why the dot plot exists at all

The dot plot is younger than most people assume. The Fed only began publishing it in **January 2012**, as part of a broader push toward transparency under then-chair Ben Bernanke. The idea was to give the public a clearer sense of where the committee thought rates were heading — to make policy more predictable and to anchor expectations. Before 2012, the market had to infer the Fed's intentions from speeches and statement language alone; after 2012, there was a literal chart of where each official saw the rate going.

That transparency had an unintended consequence: the dots became a *tradeable object* in their own right. The Fed never meant for the median dot to be read as a promise — it explicitly designed the dots to be anonymous and individual, precisely so they couldn't be mistaken for a committee decision. But markets, being markets, immediately started trading the median and its revisions as if they were guidance. The Fed has spent the decade since gently reminding everyone that the dots are forecasts, not commitments — a reminder that, as we'll see, the historical record makes necessary. The tension between "we publish a path" and "the path is not a plan" is built into the instrument, and it's exactly what creates the trade.

### The SEP: a committee forecast you can read

The **Summary of Economic Projections (SEP)** is a short document in which each of the Fed's policymakers writes down their individual forecasts for four things, for each of the next few calendar year-ends and for the "longer run":

- **Real GDP growth** — how fast the economy expands.
- **The unemployment rate** — how many people are out of work.
- **Inflation** — measured by the personal consumption expenditures (PCE) price index, the Fed's preferred gauge.
- **The appropriate federal funds rate** — where *they* think the policy rate should be at each year-end, given their own forecasts for the three variables above.

The first three are economic forecasts. The fourth — the appropriate policy rate — is the one that produces the dot plot, and it is the one markets trade most directly. Two of the variables, inflation and unemployment, are the Fed's legal mandate (the "dual mandate"): roughly 2% inflation and maximum sustainable employment. The dots are each official's answer to the question, "given where I see inflation and jobs going, where should the rate be?"

Crucially, the SEP also reports a **range** and a **central tendency** for each variable, not just a median. The full range is every projection from lowest to highest. The central tendency strips out the three highest and three lowest projections and reports the band the middle of the committee falls into. A trader uses these to gauge agreement: when the central tendency is narrow, the committee is aligned; when it's wide, there's a genuine fight inside the room about the right policy. The dots are the *visual* version of this — you can see the range and the clustering at a glance instead of reading it off a table. The four economic projections and the rate dots are internally consistent for each official: a policymaker who forecasts hotter inflation will, almost by definition, place their rate dot higher, because their reaction function says hotter inflation calls for tighter policy. That's why a hawkish dot revision and an upward inflation-forecast revision usually travel together.

### The dot plot: 19 dots, one per official

The **dot plot** takes the policy-rate projections and plots them. The horizontal axis is the time horizon — the current year-end, next year-end, the year after, and a final column labeled "longer run". The vertical axis is the interest rate. Each official's projection for a given year is a single **dot**. There are 19 seats on the committee (7 Board governors plus 12 Reserve Bank presidents, though not all vote at any one meeting — all of them submit dots). So each column holds up to 19 dots stacked at the rate levels people chose.

The figure above shows the anatomy. Four things matter, and a trader reads all four:

1. **The median dot.** With 19 dots, the median is the 10th from the bottom (or top). This is *the* number the headlines report and the market trades — "the median dot for next year is 3.4%." It is a robust summary because outliers at the top and bottom can't drag it around.
2. **The dispersion.** How spread out the dots are. A tight cluster means the committee broadly agrees; a wide spread means deep disagreement about the right path.
3. **The terminal rate.** The peak of the projected path — the highest the rate is expected to reach in the cycle before the Fed starts cutting. It's not a labeled value on the chart; you read it off as the high point of the median path across the years.
4. **The longer-run dot.** The final column. This is the committee's estimate of the **neutral rate** — the rate that neither stimulates nor restrains the economy once inflation is back at target. It's the anchor the whole path eventually returns to.

### The quarterly cadence — and "projection, not promise"

Two features of the SEP shape how it's traded. First, the **cadence**: it comes out only four times a year, at the March, June, September and December meetings. That means a dot-plot meeting carries more event risk than a "plain" meeting, because there's a fresh forecast to react to. The four "off" meetings have only the statement and the press conference.

Second, and most important: **the dots are projections, not promises.** Each official writes down what they think is *appropriate* given their forecast on that day. If the data comes in differently — inflation cools faster, the labor market cracks — the appropriate rate changes, and the next SEP's dots will move. The Fed says this explicitly, every time. The dots are conditional forecasts, not a pre-committed schedule. We'll see later just how far reality has drifted from the dots — by hundreds of basis points inside a single year. Internalize this now: **you trade the change in the dots and the repricing it triggers, not the dots as a plan.**

## How to read the dots: median, range, and the path across years

Reading a dot plot well is a skill, and most beginners read it wrong by fixating on the level when they should be reading the *change* and the *shape*.

### The median is the signal; the level is the context

When the new SEP lands, the first thing every desk does is compare the **new median** for each year to the **old median** from the previous SEP. The level (say, 3.4% for next year) tells you where the committee thinks the rate is going. But the level was already roughly in the price — the market had three months to digest the last SEP and the speeches since. What's *new* is the **revision**: did the median for a given year move up, down, or hold?

A median that rises is **hawkish** — the committee now sees a higher rate as appropriate, which usually means they're more worried about inflation. A median that falls is **dovish** — they see a lower rate as appropriate, usually because they're more worried about growth or jobs, or because inflation is cooling. The size of the move matters: medians move in 25bp "notches" (because the rate moves in 25bp steps), so a median jumping from 3.4% to 3.9% is a two-notch hawkish revision, a meaningful surprise.

There's a subtlety worth internalizing about how the median moves. With 19 dots, the median is a single point in the middle of the stack. It can lurch by a full 25bp notch if just *one or two* officials shift their dots across the midpoint — even if the average projection barely changed. This makes the median both powerful and treacherous: a headline-grabbing "the Fed raised its 2025 dot" can be the work of one regional Fed president nudging their forecast, not a wholesale change in committee thinking. That's why experienced rates traders never read the median in isolation — they look at *how many dots* sit just above and just below the new median. A median that moved because the whole cluster shifted up is a strong signal; a median that moved because it was perched on a knife-edge between two officials is a weak one, easily reversed at the next meeting. The dispersion picture (figure 5, below) is exactly what tells you which kind of move you're looking at.

### The path, not the point

The dots aren't a single number; they're a **path** across years. The market reads the whole shape: how high does the path peak (the terminal rate), how fast does it come down (the pace of cuts), and where does it end (the longer-run dot)? A SEP can leave the current year's median unchanged but raise the *out-year* medians — telling you the Fed expects to stay higher for longer. That's a hawkish message even though the near-term dot didn't budge.

The figure below shows exactly this — two real SEP median paths side by side, and how the shape shifted between them.

![Two SEP median dot paths plotted as lines across 2024, 2025, 2026 and longer run](/imgs/blogs/the-dot-plot-and-sep-reading-the-path-2.png)

Look at the two paths. The December 2023 SEP (dashed) and the September 2024 SEP (solid) projected almost the same rate for 2025 and 2026 — both saw the rate working down toward roughly 3.4% and 2.9%. The near-term 2024 dot actually came *down* a touch (4.6% to 4.4%) as cuts got closer. But the eye-catching change is at the far right: the **longer-run dot rose from 2.5% to 2.9%**. The committee re-estimated where "neutral" sits — higher than before. That single revision tells rates traders something profound: if neutral is higher, the whole path is anchored higher, and the market should price fewer cuts at the back end. A four-tenths move in a forecast you can't even date precisely still moves the long end of the curve, because it changes the destination.

Here is the disciplined reading sequence to run on any new dot plot, using these numbers as the worked case:

1. **Read the near-term median and its revision.** The 2024 year-end median moved from 4.6% (Dec 2023) to 4.4% (Sep 2024) — a 20bp downward drift, i.e. the Fed got slightly more dovish on the immediate horizon as the first cut approached. Modest, and largely expected, so little surprise content.
2. **Read the out-year medians.** 2025 went 3.6% → 3.4%, 2026 held at 2.9%. So the projected pace of cuts is broadly intact: the Fed still expects to work the rate down toward neutral over the next two years.
3. **Read the longer-run dot.** 2.5% → 2.9%. *This* is the surprise — the destination moved up 40bp. The committee is saying the floor under rates is structurally higher than it thought nine months earlier.
4. **Translate to a curve view.** A near-term that's flat-to-dovish but a *higher destination* is a flattening-then-steepening story: the front end can still rally on near-term cuts, but the long end should sell off (yields up) because the average expected rate over ten years just rose. A long-end trader reads this SEP as *bearish duration at the back end* even though the near-term dots barely moved.

That four-step read is the whole skill: separate the near-term path (what trades the 2-year) from the destination (what trades the 30-year), and weigh each revision against what the futures curve already held.

#### Worked example: pricing a one-notch hawkish median surprise

Say the new SEP raises the median dot for next year by 50bp versus the last SEP — from 3.4% to 3.9% — while the market had only priced the old 3.4% path through futures. The front end of the curve has to reprice toward the dots. Suppose this drags the 2-year Treasury yield up by 20bp on the day.

A trader is long \$1,000,000 face of the 2-year note. The **DV01** — the dollar change in the position's value for a 1bp move in yield — is about \$190 for a \$1,000,000 2-year. Bond prices move *opposite* to yields, so a rise in yield is a loss for a long.

- Yield move: +20bp.
- Loss per bp: \$190.
- Total: 20 × (−\$190) = **−\$3,800** on the \$1,000,000 position.

The intuition: a hawkish dot revision that the market hadn't priced makes existing bonds worth less, and the 2-year — the maturity that tracks the expected Fed path most closely — takes the hit fastest.

### Dispersion: conviction versus disagreement

Two dot plots can share the exact same median and send opposite signals, depending on how the dots are spread. This is the single most under-read part of the chart.

![Tight dot cluster versus wide dot spread with the same median highlighted](/imgs/blogs/the-dot-plot-and-sep-reading-the-path-5.png)

On the left, the dots cluster tightly around the median — every official is within a notch or two. That's **high conviction**: the committee agrees, the guidance is firm, and a single data point is unlikely to swing the median. On the right, the dots span a wide range with the same median in the middle. That's **disagreement**: the committee is split, and the median is fragile — one or two officials changing their dots at the next meeting could move it. A tight cluster says "trust this path"; a wide spread says "this path is one surprise away from changing." When you see a wide dispersion, you weight the *incoming data* more and the median less, because the median is more likely to move next time.

## The terminal rate and the longer-run dot

Two points on the dot-plot path matter more than any other: the peak and the end. Get these two anchors right and you understand the whole cycle.

![Projected path rising to a terminal-rate peak then descending to a longer-run neutral dot](/imgs/blogs/the-dot-plot-and-sep-reading-the-path-6.png)

### The terminal rate: where the cycle peaks

The **terminal rate** is the highest point the policy rate is expected to reach in the current cycle — the top of the median path before cuts begin. In a hiking cycle, this is the number the whole market obsesses over, because it defines how restrictive policy will get. During the 2022–2023 hiking cycle, the terminal-rate question dominated every meeting: was the peak going to be 5%, 5.25%, 5.5%, or higher? Each upward revision to the terminal rate repriced the entire front end of the curve. The deep-dive on [the terminal rate and rate-cut cycles](/blog/trading/macro-trading/terminal-rate-and-rate-cut-cycles-pricing-the-path) walks through how the market builds and trades that number across a full cycle.

The terminal rate isn't printed on the dot plot as a label — you read it as the peak of the median path. If the median is 5.4% this year, 4.4% next year, and 3.4% the year after, the cycle has already peaked (this year is the high) and the path is descending. If instead the median rises year over year, the terminal rate is still ahead and the market prices more hikes.

The *shape* of the descent matters as much as the peak. A path that falls steeply from the peak (say 5.4% to 3.4% over two years) tells the market the Fed expects to cut aggressively once it's done hiking — typically because it expects a sharp slowdown. A path that descends gently (5.4% to 4.9% to 4.4%) says the Fed expects to hold restrictive policy for a long time — the "higher for longer" posture. Two SEPs can show the same terminal rate yet very different descents, and the descent is a genuine signal about how worried the committee is. A steep projected descent that the market *hadn't* priced is a dovish surprise even if the peak is unchanged, because it pulls forward the expected cuts that the front end keys off.

#### Worked example: a terminal-rate shift on a 10-year position

Suppose a hawkish SEP pushes the projected terminal rate up, and the move bleeds into longer maturities — the 10-year Treasury yield rises 15bp because a higher peak means rates stay restrictive longer.

A trader holds \$500,000 face of the 10-year note. The DV01 of a \$500,000 10-year is roughly \$430 per basis point (longer maturities have bigger DV01s because their cash flows are discounted over more years).

- Yield move: +15bp.
- Loss per bp: \$430.
- Total: 15 × (−\$430) = **−\$6,450**.

The intuition: even though the dot plot is about the *short-term* policy rate, a higher projected peak lifts the entire curve, and the longer the bond, the more a yield move costs you — the 10-year loses far more dollars than the 2-year on the same 15bp.

### The longer-run dot: where neutral sits

The **longer-run dot** is the committee's estimate of the **neutral rate** of interest — often written r* ("r-star") — the rate that, in the long run, neither speeds the economy up nor slows it down, once inflation is back at the 2% target. It's the gravitational center of the whole path: the rate the Fed expects to settle at after the cycle plays out. For years this dot sat around 2.5%. The drift up to 2.9% in the September 2024 SEP (shown in figure 2) was a quiet but important signal — the committee was telling markets that the new normal for rates might be structurally higher than the post-2008 era.

Why does a vague, undated "longer-run" number move markets? Because the entire bond market is, at its core, a bet on the average expected short-term rate over the life of the bond. If the destination — neutral — moves up, the average expected rate over the next ten years moves up, and 10-year yields rise to match.

#### Worked example: a longer-run revision repricing a 30-year position

The longer-run dot is revised from 2.5% to 2.9% — a 40bp upward shift in where neutral sits. The 30-year Treasury, whose value depends on expected rates far into the future, is the most exposed maturity. Suppose the long-run repricing lifts the 30-year yield by 12bp.

A trader holds \$100,000 face of the 30-year bond. The DV01 of a \$100,000 30-year is roughly \$200 per basis point (a small face but a very long duration).

- Yield move: +12bp.
- Loss per bp: \$200.
- Total: 12 × (−\$200) = **−\$2,400** on the \$100,000 position.

The intuition: the longer-run dot is the one part of the plot that speaks directly to the long end of the curve. A higher neutral rate is bad for the longest bonds because their value rests almost entirely on the far-future path of rates — exactly the part the longer-run dot re-anchors.

## Dots versus the market-implied path: the gap is the trade

Here is the crux of the entire post. The Fed publishes its dots. But the market has its *own* path for the policy rate, priced continuously through **interest-rate futures** — contracts that let traders bet on where the Fed funds rate will be at future dates. (Read off as probabilities, these are what tools like CME FedWatch report; the mechanics live in the companion piece on [consensus, expectations, and "priced in"](/blog/trading/event-trading/consensus-expectations-and-priced-in).)

These two paths — the dots and the futures curve — **almost never agree**. The dots are the median policymaker's *forecast*; the futures curve is the market's *bet*, weighted by real money and often more dovish (markets tend to price more cuts than the Fed projects, betting the economy will weaken). The difference between them is the gap, and the gap is the trade.

### How the market builds its own path

It helps to know what you're comparing the dots *to*. The market-implied path comes from two main families of contracts. **Fed funds futures** settle on the average daily fed funds rate over a given month, so their price directly implies the expected rate for that month. **SOFR futures** (and the related options) reference the Secured Overnight Financing Rate, a market rate that tracks the Fed's target closely; they're the deeper, more liquid market for trading the longer path. By stringing together the prices of contracts for each future month, traders construct a continuous **expected-rate curve** that says, in effect, "the market thinks the Fed funds rate will average X% in March, Y% in June, Z% in December."

The popular way to read this is as **probabilities of a move at each meeting** — the format tools like CME FedWatch publish. If the futures imply an expected rate 18bp below the current rate at the next meeting, that's read as roughly a 72% chance of a 25bp cut (18 ÷ 25 ≈ 0.72). The same arithmetic, run across every meeting on the calendar, gives the market's full projected path — its answer to the same question the dots answer. The mechanics of turning futures prices into these odds are laid out in [consensus, expectations, and "priced in"](/blog/trading/event-trading/consensus-expectations-and-priced-in); here the point is just that the market has a continuous, money-weighted path, and the dots are a quarterly, forecast-weighted path, and the two are rarely the same.

#### Worked example: turning the gap into a cut count

Suppose at the end of next year the median dot sits at 3.9% while the futures curve implies 3.4% — a 50bp gap, with the market pricing two more 25bp cuts than the Fed is projecting. A macro fund holds \$1,000,000 of 2-year notes positioned for the *market's* dovish path. A hot inflation print then forces the market to abandon one of those two extra cuts, lifting the 2-year yield 25bp toward the dots.

- Gap closed: 25bp (one cut priced out).
- 2-year DV01: about \$190 per bp.
- Loss on the long: 25 × (−\$190) = **−\$4,750**.

The intuition: the dovish bet was a wager that the gap would close *the Fed's way* (dots falling to the market). When the data instead closed it *the market's way* (the market rising to the dots), every basis point of repricing was a loss on a position built for the opposite outcome.

![Median dot path above the futures-implied path with the gap between them shaded](/imgs/blogs/the-dot-plot-and-sep-reading-the-path-3.png)

The figure makes it concrete. The blue median-dot path sits above the green futures-implied path — the Fed is projecting a higher rate than the market is pricing. The amber gap between them is the disputed territory. Only one side can be right. As data arrives, one of two things happens:

- **The market converges to the dots.** If inflation runs hot and growth holds, the data validates the Fed's higher path. The market is forced to price out the cuts it had penciled in. Front-end yields rise, the gap closes from the bottom up — and anyone positioned for the dovish futures path loses.
- **The dots converge to the market.** If the economy weakens, the Fed's next SEP revises the dots *down* toward where the market already was. The dots fall, the gap closes from the top down — and the market's dovish bet pays off.

The event-trade is to ask, on every dot-plot day: which way is the gap going to close, and is the new SEP a step in that direction? A SEP that *widens* the gap (dots move further from the market) is a bigger surprise than one that narrows it.

#### Worked example: the gap closing against an equity book

A trader runs a \$25,000 book of rate-sensitive equities — high-growth tech names whose valuations depend heavily on low discount rates. A hawkish dot-plot surprise pushes the market's path up toward the dots, the front end sells off, and the rate-sensitive book falls 1.5% on the day as higher rates compress valuations.

- Book value: \$25,000.
- Move: −1.5%.
- Dollar impact: \$25,000 × (−0.015) = **−\$375**.

The intuition: the dots don't only move bonds. When the market is forced to reprice the rate path higher, the assets most sensitive to discount rates — long-duration growth stocks, gold (which pays no yield and suffers when real rates rise), and rate-sensitive small caps — all feel it. The dot plot is a cross-asset event, not just a bond event. For the full transmission map, see [the cross-asset rotation](/blog/trading/cross-asset/risk-on-risk-off-the-cross-asset-rotation).

## Why the dots shift, and how markets reprice

The dots move because the *data* moves. Each official's dot is the answer to "given my forecast for inflation and jobs, what rate is appropriate?" Change the inflation forecast and the appropriate-rate answer changes. So between two SEPs, the things that move the dots are the things that move the Fed's forecast: inflation prints (CPI and PCE), the jobs report, growth data, financial conditions, and any shock to the outlook.

When the dots shift at a meeting, the repricing happens in a specific order and a specific shape:

1. **The dot plot hits the wire** at 2:00 p.m. ET (statement and SEP are released together).
2. **The front end moves first.** Two-year Treasury yields — the maturity that tracks the expected Fed path — reprice within seconds. Fed funds futures and SOFR futures adjust to the new path.
3. **The curve reshapes.** A hawkish dot revision tends to *flatten* the curve (short rates rise more than long rates, because the market expects the Fed to eventually tame inflation and cut later). A dovish revision tends to *steepen* it.
4. **Cross-asset transmission.** The dollar firms on a hawkish surprise (higher U.S. rates attract capital), gold dips (higher real yields hurt the zero-yield metal), rate-sensitive equities sell off, and risk assets broadly wobble. The reverse on a dovish surprise. The chain runs from rates outward: a hawkish dot revision raises the expected path, which lifts front-end yields and real yields, which raises the discount rate on every future cash flow — so long-duration growth stocks (whose value sits far in the future) fall hardest, value and short-duration sectors hold up better, and assets that pay no yield (gold, and to a degree crypto) lose their relative appeal. Emerging-market assets and high-yield credit, which depend on cheap dollar funding, wobble too. None of this is on the dot plot itself; it's the second-order consequence of the market repricing the path the dots imply.
5. **The press conference can override everything.** Thirty minutes after the SEP, the chair speaks. If the chair "walks back" the hawkish dots — emphasizing data-dependence, downplaying the median — the initial move can fade or fully reverse. Trading [the FOMC statement and press conference](/blog/trading/macro-trading/trading-the-fomc-statement-presser-dot-plot) is its own skill, because the presser and the dots can point in opposite directions.

This is why the dot plot is treated as a knee-jerk-then-reassess event. The first move on the dots is mechanical; the durable move depends on whether the chair confirms the dots' message in the press conference.

### Why the front end moves and the long end lags

It's worth being precise about *why* the 2-year reacts more than the 10-year to a dot shift. A Treasury yield is, broadly, the average expected short-term rate over the bond's life, plus a term premium. The 2-year's life is almost entirely inside the window the dots cover — the next two years of Fed policy — so a change in the projected path over those two years moves the 2-year almost one-for-one. The 10-year, by contrast, averages expected rates over a *decade*; the next two years are only a fifth of that window, so a near-term dot shift moves it proportionally less. This is the mechanical reason a hawkish dot surprise flattens the curve: the short end, which is mostly "Fed path," jumps; the long end, which is mostly "long-run neutral plus term premium," barely twitches unless it's the *longer-run* dot that moved. Understanding this lets you pick the right instrument — trade the 2-year (or a 2s10s flattener) on a near-term dot surprise, and the long end on a longer-run-dot surprise. The full anatomy of how a curve moves around these events lives in [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession).

### The dots and the chair can disagree on purpose

One nuance traders learn the hard way: the dots and the press conference are *separate signals*, and the chair sometimes sets them against each other deliberately. A chair who wants to keep optionality might let the dots print hawkish (to anchor inflation expectations) while sounding dovish at the podium (to avoid spooking markets), or vice versa. The 2:00 p.m. dot reaction and the 2:30 presser reaction can therefore point in opposite directions, and the closing print is a blend of the two. A disciplined event trader treats the dots as the *committee's median view* and the presser as the *chair's spin on it*, and waits to see whether the chair confirms or contradicts before committing real size. This is why so many "dot plot trades" are actually two trades: the initial dot reaction, then the presser fade-or-confirm.

## The dot plot's limits: it is not a commitment

Now the most important section, and the one that separates traders who use the dots well from those who get burned. **The dots are projections, not promises** — and the historical record is brutal on this point.

![Actual fed funds rate path as a step chart with the Sep-2024 SEP median dots overlaid as points](/imgs/blogs/the-dot-plot-and-sep-reading-the-path-4.png)

The figure overlays the *delivered* policy rate — the actual fed funds upper bound, drawn as a step function — against the projected dots from one SEP. The blue steps are what the Fed actually did; the amber dots are what one SEP projected it would do. They don't line up. Reality drifts from the dots, sometimes by a lot, because the future arrives differently than any forecast.

The canonical cautionary tale is **2021**. Through most of that year, the dots projected essentially *no rate hikes* until 2023 — the committee believed inflation was "transitory." Then inflation exploded, and in 2022 the Fed delivered the fastest hiking cycle in four decades, lifting the rate from near zero to over 4% in a single year. The dots from late 2021 were wrong by *hundreds of basis points* within twelve months. Anyone who treated those dots as a commitment — who positioned for a near-zero rate through 2022 — was destroyed.

It's worth sitting with how large that miss was. In mid-2021, the median dot saw the policy rate at roughly **0.1% through the end of 2022** — i.e. no meaningful hikes. The rate actually finished 2022 at **4.25%–4.50%**. The forecast error was on the order of **400+ basis points in eighteen months** — not a rounding error, but a complete regime miss. The dots weren't lying; the committee genuinely believed the inflation surge was transitory, and when that belief collapsed, so did the projected path. For a trader, the takeaway is not "the Fed is incompetent" but "even the people who set the rate cannot forecast their own path through a regime change." If the median dot-setters can be off by 400bp, a trader treating the dots as a fixed schedule is building on sand. The dots are most reliable in calm regimes and least reliable exactly when the outlook is shifting — which is exactly when the biggest moves happen.

The lesson is not that the dots are useless. They're the cleanest read on the committee's current thinking, and the *change* in the median is a genuine signal. The lesson is about how to *use* them:

- The dots tell you what the Fed expects **today**, given today's data. They are a conditional forecast, not a schedule.
- The market knows this, which is why the futures curve so often disagrees with the dots — traders are pricing the *distribution* of outcomes, not the committee's point estimate.
- The tradeable signal is the **revision** (how the median changed) and the **gap** (dots versus market), not the absolute level treated as gospel.
- When a regime is uncertain — inflation could go either way — the dispersion widens and the dots become *less* reliable, exactly when you most want certainty.

Trade the repricing the dots trigger. Never trade the dots as a promise the Fed has to keep.

## How it reacted: real episodes

Theory is cheap. Here are dated episodes where the dots — or the absence of a dot-plot surprise — drove the tape.

### December 2018: the dots that didn't bend

On 19 December 2018, the Fed hiked 25bp to a 2.25%–2.50% range. The hike itself was expected. The damage came from the *dots*: the SEP still projected **two more hikes for 2019**, when a wobbling stock market and slowing global growth had the market begging for a pause. The dots were more hawkish than the market wanted, and the chair's press conference reinforced it. The S&P 500 reversed to close **−1.54%** on the day, and the index was already mid-collapse — the fourth quarter of 2018 saw a **−19.8%** drawdown from the September high, including a brutal **−2.71%** on Christmas Eve. The market was screaming that the Fed's projected path was too high; within weeks the Fed pivoted to "patience," and by mid-2019 it was *cutting*. The dots had been wrong, and the market had front-run their correction violently.

#### Worked example: the December 2018 dot-driven equity hit

A trader holds a \$25,000 S&P 500 index position into that December 2018 meeting. The hawkish dots and presser drive the index to close −1.54%.

- Position: \$25,000.
- Move: −1.54%.
- Dollar impact: \$25,000 × (−0.0154) = **−\$385**.

And that was just one session. Over the full Q4 2018 drawdown of −19.8%, the same \$25,000 buy-and-hold would have been marked down by \$25,000 × (−0.198) = **−\$4,950** at the trough — a vivid reminder that misjudging the Fed's path is expensive across the whole quarter, not just on the day.

### 2013: the taper tantrum, a path-repricing in miniature

Before the dot plot was even three years old, the market had already learned how violently it reprices to a shift in the *expected path* — even without a single rate change. In May 2013, chair Bernanke merely *suggested* the Fed might begin slowing ("tapering") its bond purchases later that year. No rate hike, no change to the target range — just a hint that the easy-money path might shorten. The market repriced the entire expected path of policy in weeks. The 10-year Treasury yield rose from **1.66% in early May to 2.31% by the June FOMC, and on to 3.04% by year-end** — nearly a doubling of the 10-year yield off a change in the *expected path*, not the current rate. The "taper tantrum" is the clearest proof that markets trade the path, not the level: the dots, when they shift, can trigger exactly this kind of repricing because they *are* the path. A hawkish dot revision is the modern, formalized version of a Bernanke taper hint.

### 2023 to 2024: the SEP revisions and the higher-for-longer story

Across 2023 and into 2024, the SEP told a slow-motion story that the market traded the whole way. As inflation proved sticky, successive SEPs nudged the out-year dots up and trimmed the number of cuts projected — the "higher for longer" theme. The December 2023 SEP (figure 2) projected the rate working down toward 3.6% in 2025; by the September 2024 SEP the near-term path had shifted as cuts finally arrived, but the **longer-run dot had been quietly raised from 2.5% to 2.9%**. That back-end revision was the market's signal that neutral had moved up structurally — the bond market priced a higher floor for long-term yields, and the 10-year traded accordingly.

### September 2024: the first cut, and a muted-then-reversed reaction

On 18 September 2024 the Fed delivered its first cut of the cycle — a larger-than-usual **50bp**, to a 4.75%–5.00% range. You might expect a big rally. Instead the S&P 500 closed **−0.29%** on the day, essentially flat, before rallying the next session. Why so muted? Because the cut was largely priced, and the *dots* alongside it mattered as much as the cut: the new SEP's path showed a measured pace of further easing, not an emergency-cut trajectory, and the longer-run dot at 2.9% kept the back end anchored. The market had to weigh a dovish action (the 50bp cut) against a not-especially-dovish set of dots. The 10-year yield actually rose about 7bp on the day — the long end said "this is a normalization, not a panic." The dots tempered the action, exactly as the gap framework predicts.

The September 2024 meeting is a perfect teaching case for the gap framework because the action and the dots pointed *opposite* ways. The headline — a jumbo 50bp cut — was unambiguously dovish, the kind of move a panicking central bank makes. But the dots said the opposite: a measured glide path of roughly 25bp cuts thereafter and a *higher* neutral anchor. A trader who only read the headline ("50bp cut! risk-on!") and bought duration got the front end roughly right but the long end exactly wrong — the 10-year *rose*. The lesson: when the action and the dots disagree, the dots usually win the longer-dated part of the curve, because they speak to the destination, while the action moves the very front. The 50bp cut steepened the curve (front down, long up) rather than rallying it across the board, and the disciplined read was a steepener, not a duration long.

These episodes share a pattern: the *decision* is usually priced, but the *dots* and the press conference carry the surprise. The market trades the revision and the gap, not the headline action. Notice too how each episode keys off the *expected path*, not the current rate: December 2018 was about dots projecting hikes the market didn't want; the taper tantrum was about a path shortening with no rate change at all; September 2024 was about dots refusing to confirm a panic the 50bp cut might have implied. In every case the tradeable surprise lived in the projected path versus the priced path — the gap.

## Common misconceptions

A handful of beliefs about the dot plot are both widespread and wrong. Each one costs money.

### "The dots are a promise the Fed will keep"

This is the big one. The dots are **projections that move**, and the 2021 episode proves it — the late-2021 dots projected near-zero rates through 2022, and the Fed instead hiked over 400bp that year. Treating the dots as a binding schedule is the fastest way to get run over. The dots are conditional on the data; when the data changes, the dots change, and so does the rate. Trade the revision, not the forecast.

### "The median dot is the only number that matters"

The median is the headline, but the **dispersion** (figure 5) and the **path shape** carry independent information. A wide spread of dots means the median is fragile and the next print could swing it — you should weight incoming data more. A SEP that holds the near-term median but raises the out-years is hawkish even though the headline dot didn't move. Reading only the median misses half the signal.

### "The dots and the market agree, so there's no trade"

They almost never agree, and even when they're close the *direction the gap is closing* is the trade. A 25bp gap that's about to widen on a hot CPI print is more actionable than a 100bp gap that's stable. The trade lives in the *change* in the gap, not its level.

### "A dot-plot surprise always sticks"

It often doesn't. The press conference can fully reverse the initial dot-driven move — the chair can "walk back" hawkish dots by stressing data-dependence, or amplify them by sounding alarmed about inflation. December 2018 is the cautionary case where the dots *and* the presser were hawkish and the move stuck (and then some); but plenty of meetings see the dots spike yields at 2:00 p.m. and the presser fade them by 2:45. Never assume the 2:00 move is the close.

### "The longer-run dot is academic and doesn't trade"

The opposite. Because the long end of the curve is a bet on the average future short rate, a revision to the longer-run dot — the destination — feeds directly into 10- and 30-year yields. The 2.5%-to-2.9% revision in 2024 was one of the more consequential quiet moves of the year for long-end traders.

## The playbook: how to trade a dot-plot meeting

Pull it together into an if-then map you can run on the day. Dot-plot meetings (March, June, September, December) carry more event risk than plain meetings, so size accordingly.

![Decision tree comparing the new median to the last SEP and to the futures path with trade outcomes](/imgs/blogs/the-dot-plot-and-sep-reading-the-path-7.png)

The figure is the decision flow. Two questions, four outcomes.

**Before the meeting — build your baseline.**

- Pull the **last SEP's median path** for each year and the longer-run dot. This is your "old" reference.
- Pull the **market-implied path** from Fed funds / SOFR futures (FedWatch probabilities are the readable form). This is what's priced.
- Note the **gap** between them and which way the recent data has been pushing it. Hot inflation/jobs → gap likely to close toward the dots (hawkish risk); weak data → dots likely to fall toward the market (dovish risk).
- Check the **expected move** the options market is pricing for the session, so you size risk to the event. (See [the expected move](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options) for the method.)

**At 2:00 p.m. — read the SEP.**

- **Q1: Did the median move versus the last SEP?** Up a notch or two = hawkish surprise; down = dovish surprise; unchanged = trade the path shape and dispersion instead.
- **Q2: Where does the new median sit versus the futures path?** Above the curve and the data supports it → hawkish repricing: front-end yields rise, curve flattens, dollar firms, gold and rate-sensitive equities dip. Below the curve → dovish repricing: front-end yields fall, curve steepens, dollar softens, rate-sensitive assets pop. Already matching the curve → likely a non-event on the median; trade the dispersion and the presser.

**The positions per asset (hawkish-surprise case).**

- **Rates:** the cleanest expression. A hawkish dot surprise sells the front end — short the 2-year, or position for a flatter curve (2s10s). The 2-year has the tightest link to the Fed path.
- **FX:** long the dollar (higher U.S. rates attract capital). See [trading the dollar](/blog/trading/macro-trading/trading-the-dollar-dxy-carry-dollar-smile).
- **Gold:** lower (higher real yields hurt the zero-yield metal).
- **Equities:** rate-sensitive growth and small caps underperform; defensives hold up better.
- Reverse all of these for a dovish surprise.

**The invalidation — when to cut.**

- **The press conference contradicts the dots.** If the chair walks back a hawkish median, the 2:00 move can fully reverse by 2:45. If your trade was on the dots and the presser kills it, you're wrong — cut.
- **The gap was already priced.** If the median matched the futures curve, the median is a non-event and any knee-jerk will fade. Don't chase it.
- **The dispersion is wide.** A fragile median is one print away from changing; don't oversize a trade on a split committee.

**Sizing and risk.** Treat the dot plot as a two-stage event: the 2:00 SEP release and the 2:30 press conference. Volatility is elevated across both. Size to the options-implied expected move, keep dry powder for the presser, and never assume the first 2:00 print is the settle. The durable trade is the one the chair *confirms*.

#### Worked example: sizing a rates trade to the dot-plot risk

A trader wants to express a hawkish-dot view by shorting \$1,000,000 of the 2-year note ahead of the meeting. The 2-year DV01 is about \$190/bp, and the trader's pre-meeting risk budget for the event is \$5,000 of loss tolerance.

- The options market and recent dot-plot sessions suggest the 2-year could move about 15bp against the position in an adverse (dovish) surprise.
- Worst-case loss at full size: 15 × \$190 = \$2,850 — comfortably inside the \$5,000 budget, so full size is acceptable.
- But the trader also reserves half the budget for the presser, where a dovish walk-back could add another ~10bp against the position: 10 × \$190 = \$1,900. Combined potential adverse move: \$2,850 + \$1,900 = **\$4,750**, just under the \$5,000 limit.

So the trader either holds full size and accepts the two-stage risk, or halves the position to \$500,000 (DV01 ~\$95/bp) to keep room for adding after the presser confirms. The intuition: a dot-plot trade is two events stacked in 30 minutes, so you size for *both* the SEP and the presser, not just the headline median.

**A worked dovish case, for symmetry.** Suppose instead the median is revised *down* a notch and sits below the futures curve — a dovish surprise. A trader long \$1,000,000 of the 2-year (positioned for falling rates) benefits as the yield drops 12bp: 12 × \$190 = **+\$2,280**. The same logic, opposite sign — the long 2-year gains because the dovish dots pull the expected path down and lift bond prices.

The core discipline never changes: the dots are a forecast, the futures curve is the bet, and you trade the gap between them and the way it closes — not the dots as a promise.

## Further reading and cross-links

- [Inflation and the Fed's reaction function](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot) — the mechanism that *produces* the dots: how the Fed maps inflation and jobs to an appropriate rate.
- [The terminal rate and rate-cut cycles](/blog/trading/macro-trading/terminal-rate-and-rate-cut-cycles-pricing-the-path) — how the market builds and trades the peak of the path across a full cycle.
- [Trading the FOMC statement and press conference](/blog/trading/macro-trading/trading-the-fomc-statement-presser-dot-plot) — the presser that can confirm or reverse the dot-driven move.
- [Consensus, expectations, and "priced in"](/blog/trading/event-trading/consensus-expectations-and-priced-in) — how the market builds the implied path the dots are measured against.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the plumbing behind the policy rate the dots project.

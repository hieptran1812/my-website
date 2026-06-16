---
title: "Relative Strength: Reading Which Sector Is Actually Leading VN-Index"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Up is not the same as leading. A sector can rise and still lag the index. This is how to use relative strength -- a sector's price divided by VN-Index -- to see which group is actually winning the flow, how to compute and read it, and the honest limits of the tool."
tags: ["vietnam-stocks", "sector-rotation", "relative-strength", "vn-index", "technical-analysis", "sector-ranking", "rrg", "leadership", "momentum", "trading"]
category: "trading"
subcategory: "Vietnam Stocks"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** -- "Up" is not "leading." A sector can close green and still be losing the race against the index. Relative strength (RS) -- a sector index divided by VN-Index -- strips out the whole-market move and shows you who is actually winning the flow.
>
> - **RS is a ratio, not a feeling.** RS = sector index / VN-Index. When the ratio rises, the sector is beating the index; when it falls, the sector is lagging -- *even if its price is up*. Normalize the ratio to 100 (or 1.00) at a start date so every sector shares one scale.
> - **Rising RS in an index uptrend is the cleanest leadership signal you can read** with a single line. You rank sectors by their 13-week relative performance, watch the top and bottom movers, and rotate toward improving RS and away from deteriorating RS.
> - **RS divergence is an early handoff tell.** When a sector's price makes a new high but its RS line does *not*, the flow is quietly leaving -- leadership is being handed to the next group before price admits it.
> - **RS is not RSI.** RS = relative strength versus the index (an unbounded ratio line). RSI = the Relative Strength *Index*, a bounded 0--100 momentum oscillator of a single price against *itself*. Confusing the two is the single most common mistake here.
> - The one number to remember: a sector up **4%** while VN-Index is up **7%** has *falling* RS. Green on the screen, losing the race. This is educational, not financial advice.

On a single trading session in March 2024, two sectors on the Ho Chi Minh exchange (HOSE) closed green. Banks finished up about 1.2% on the day; a mid-cap real estate basket finished up about 1.1%. VN-Index itself -- the headline benchmark, a market-cap-weighted index of the stocks listed on HOSE -- closed up roughly 0.4%. A retail investor scanning the board would see two green sectors, both beating a green market, and conclude that both groups were "leading."

One of them was. The other was a passenger.

Here is the thing the daily color cannot tell you: over the trailing three months, banks had been quietly winning a *larger and larger share* of the index's move, while that real estate basket had been winning a *smaller and smaller* share. Both were rising in absolute price. But only one was rising *relative to the index*. If you had been ranking sectors by relative strength rather than by daily color, you would have known -- weeks earlier -- which group the smart money was actually rotating into, and which one was just floating up on the tide.

This post is about that single lens. Relative strength is not a magic indicator and it does not predict the future. What it does is answer one question cleanly that absolute price cannot: *of all the sectors that are up, which ones are actually leading?* That question sits at the center of sector rotation -- the whole discipline of moving capital from the group that is finishing its run into the group that is starting one. Let us build the tool from zero, learn to read it honestly, build a weekly VN-Index dashboard around it, and then be very clear about everything it cannot do.

![Two panel diagram: top shows two sector prices both rising, bottom shows the same two as RS lines vs the index, one above 1.0 and one below](/imgs/blogs/relative-strength-reading-which-sector-leads-vn-index-1.png)

Look at the figure. The top panel is what your eyes see on the price board: two sectors, both climbing, both green on the year. Sector A is up 25%, Sector B is up 18% -- both *winners*, surely. The bottom panel divides each one by VN-Index. Now the story flips. Sector A's RS line *rises* -- it is taking share of the index's move, it is genuinely leading. Sector B's RS line *falls* -- it is up in price, but it is losing the race, lagging the very benchmark it is part of. Same two green sectors, opposite verdicts. The entire post lives in the gap between those two panels.

## Foundations: what relative strength actually is

Start with the most basic distinction in all of performance measurement: **absolute** versus **relative**.

*Absolute performance* is what you make in money. A sector index goes from 850 to 900; that is +5.9% in absolute terms. It is the number that hits your account. It is real and it matters -- you cannot eat relative performance.

*Relative performance* is what you make **compared to a benchmark**. If your sector rose 5.9% but VN-Index rose 9% over the same window, then in relative terms you *underperformed* by about three percentage points. You made money and you still lost the race. If instead VN-Index rose only 2%, your same 5.9% means you *outperformed* by nearly four points -- you led.

Here is the everyday analogy before the formula. Imagine two runners in a marathon, both moving forward, both making progress down the course. From a helicopter, both look like they are "winning" -- they are both going toward the finish line. But the race is not against the course; it is against each other. To know who is *actually winning*, you stop measuring distance-from-start and start measuring the *gap between them*. Are they pulling apart, and in whose favor? A rising market is the course; it carries every sector forward. Relative strength is the gap measurement -- it ignores how far the pack has run and asks only: of these two, who is pulling ahead?

Relative strength is simply the disciplined, continuous version of that comparison. Instead of eyeballing two percentage numbers, you compute a single **ratio** at every point in time:

```
RS ratio = sector index / VN-Index
```

That is the whole core idea. One number, one line. The sector index is the numerator (the thing you care about), VN-Index is the denominator (the benchmark you measure against). The genius of the ratio is that it *divides out the market move*. When the whole market rallies, both the numerator and the denominator rise -- so the ratio barely moves unless the sector is rallying *more* (ratio up) or *less* (ratio down) than the market. RS isolates exactly the thing daily color hides: the sector's performance *net of the index*.

It helps to see *why* the division works, because once you feel it you will never again confuse "up" with "leading." Write the sector index at time *t* as its starting level times a growth factor, and the same for VN-Index. The ratio is then the sector's growth factor divided by the index's growth factor (the starting levels fold into the normalization). So the RS line is literally *the sector's cumulative growth divided by the market's cumulative growth*. If the sector compounds at the same rate as the index, the ratio is flat -- a horizontal RS line means "this sector is exactly the market." If the sector compounds faster, the ratio rises no matter what the absolute direction is; if it compounds slower, the ratio falls no matter what. This is why a sector can be up 30% in a bull market and still have a *falling* RS line: 30% is real money, but if the index made 40%, the sector's growth factor (1.30) divided by the market's (1.40) is 0.93 -- below where it started. Growth is absolute; the ratio is relative; the RS line plots only the second.

A subtle but important consequence: the RS line is *symmetric to the market's direction*. In a bear market where everything falls, a sector that drops 10% while the index drops 20% has a *rising* RS line (0.90 / 0.80 = 1.125) -- it is leading, even though it lost you money. RS does not know or care whether the market is going up or down; it only measures who is winning the relative race. That symmetry is exactly why RS must always be paired with a separate read of the *index trend itself* -- a point we will hammer in the playbook, because leadership of a falling market is still a losing position.

A quick word on what these "sector indices" are, because a beginner deserves it. On HOSE, the headline benchmark is VN-Index. Beneath it sit narrower baskets: VN30 (the 30 largest, most liquid names), and a family of sector or industry sub-indices (VNFIN for financials, VNREAL for real estate, VNIT for technology, and so on), plus various third-party and broker-built sector baskets. Any of these can be the numerator. When this post says "the banks index" or "the securities index," it means a basket of bank stocks (VCB, BID, CTG, TCB, MBB, ACB, VPB and the rest) or broker stocks (SSI, VND, VCI, HCM and friends), tracked as a single number. Divide that basket by VN-Index and you have the sector's RS line.

### The RS line and why we normalize to 100

The raw ratio is awkward to read because it depends on the absolute *levels* of the two indices. If the banks index is around 1,800 and VN-Index is around 1,280, the raw ratio is about 1.41 -- a number that tells you nothing on its own. What you care about is *how the ratio changes over time*, not its level on any one day.

So you **normalize**: pick a start date, divide every RS value by the RS value on that start date, and multiply by 100 (or just leave it as a multiple of 1.00). Now every sector's RS line starts at exactly 100 on day one. A line that climbs to 112 means the sector has beaten VN-Index by 12% since the start date. A line that falls to 94 means it has lagged the index by 6%. Every sector is on one comparable scale, anchored at a common origin. This normalized ratio over time is what people draw and call **the RS line**.

#### Worked example: the RS ratio from scratch

Take a single sector and watch its RS evolve. Suppose on day one the sector index is at **850** and VN-Index is at **1,250**.

```
day 1:  RS = 850 / 1,250 = 0.680
```

A month later, the market has rallied. The sector index is now **900** and VN-Index is **1,280**.

```
day 30: RS = 900 / 1,280 = 0.703
```

The RS ratio rose from 0.680 to 0.703, a gain of about 3.4%. Notice what happened: VN-Index itself rose (1,250 to 1,280, +2.4%), so a casual observer would say "the market went up, this sector went up, fine." But the *sector outpaced the index* -- it rose 5.9% while the index rose 2.4% -- and the RS line captured exactly that 3.4% of relative outperformance. **A rising RS ratio means leadership even when the index is also rising; the ratio cares only about who is winning the gap.**

To put a familiar anchor on the levels: VN-Index trading near 1,280 with the dong around **\$1 = 25,400 VND** corresponds to a market whose total capitalization runs into the hundreds of **\$bn**; a single large bank like Vietcombank alone is a **\$15bn**-plus company. We will keep dropping these **\$** anchors so non-Vietnam readers have a sense of scale, but the RS math is unit-free -- it would give the same 3.4% whether the indices were quoted in points, dong, or dollars.

### What actually drives a sector's relative strength

The RS line is a measurement, but it is measuring something real underneath. A sector's relative strength rises because *capital is flowing into that sector faster than into the broad market* -- and capital flows for reasons. Understanding the reasons keeps you from treating RS as a mysterious squiggle.

A sector index is just the aggregate price of a basket of companies that share a common economic driver. Banks all earn on the spread between loan and deposit rates (NIM) and on credit growth; brokers all earn on trading turnover and margin lending; steel and construction names all live or die on real estate and infrastructure demand; consumer staples all sell things people buy regardless of the cycle. Because the companies in a sector share a driver, their *earnings move together* -- and because earnings move together, their *prices move together*. That common movement is what makes a sector index (and therefore an RS line) a meaningful object rather than noise. When the driver improves for one sector faster than for the average company in the market, that sector's earnings expectations rise faster, money rotates in, and its RS line climbs. The RS line is the *visible shadow* of a shift in the relative outlook for one group's earnings.

This is why RS leadership tends to *track the business cycle*. Early in a recovery, as the State Bank of Vietnam (SBV) cuts rates and credit re-accelerates, the earnings outlook for banks, brokers, and cyclical industrials improves fastest -- and their RS lines lead. Late in the cycle, as inflation and rates rise, the outlook tilts toward real-asset and commodity sectors. In a downturn, defensives (utilities, staples, healthcare) hold their earnings best, so their RS lines turn up even as the market falls. RS does not *cause* rotation; it *reflects* the rotation that the changing economic outlook drives. That is also why an RS dashboard works as a leadership tool: it surfaces, in price, the same rotation the macro cycle is producing in earnings -- often before the earnings numbers are even reported. (The cross-asset post on rotation through the cycle, linked at the end, maps each cycle phase to the sectors whose RS *should* be improving in it.)

#### Worked example: why two sectors with identical price gains can have opposite RS

Make the driver idea concrete. Over one quarter, VN-Index returns **+6.0%**. Two sectors both happen to return exactly **+6.0%** in price -- identical absolute gains. But trace the flow:

- **Banks** rose 6.0% on *broadening* participation -- ten of the twelve names in the basket rose, on rising credit-growth expectations. Its RS held flat at 6.0% - 6.0% = **0.0%**, exactly matching the index, and its RS *momentum* (the slope of the RS line over the quarter) was turning *up* from below.
- **Real estate** rose 6.0% on *narrowing* participation -- two large names carried the basket while the rest stagnated on lingering bond-market fears. Its RS also reads 0.0% on the quarter, but its RS momentum was rolling *over* from a higher level -- the rise was the tail end of an old move, not the start of a new one.

Same price gain, same point-in-time RS, but opposite *trajectories*: banks improving from below, real estate weakening from above. **The RS *level* tells you where a sector stands; the RS *momentum* -- the slope -- tells you which way it is heading, and the two together are what the relative-rotation graph plots.** This is why a single RS number is never enough; you read the line, not the dot.

## Computing relative strength three ways

There is not one "RS number." There are three closely related views, and a serious sector analyst uses all three. They answer slightly different questions.

### 1. The RS ratio line -- the trend view

This is what we just built: ratio = sector / index, normalized to 100, plotted over time. Its job is to show you the **trend** of leadership. Is this sector winning more share of the index over weeks and months (RS line sloping up), or bleeding share (sloping down), or treading water (flat)? The slope is the signal. The RS line is the single most information-dense view because a human eye reads a trend in a line instantly.

![Line chart of three stylized RS ratios over 18 months, one leading above 1.0 and rising, one neutral near 1.0, one lagging below 1.0 and falling](/imgs/blogs/relative-strength-reading-which-sector-leads-vn-index-2.png)

The figure shows three stylized RS lines over 18 months, each starting at 1.00 (each sector exactly matching the index on day one). The green line is a genuine leader: it grinds steadily above 1.00, ending around 1.22 -- it has beaten VN-Index by roughly 22% over the window. The slate line is neutral: it wobbles around 1.00 with no net direction -- this sector is simply *being the market*, neither leading nor lagging. The red line is a laggard: it slides to about 0.84, having underperformed the index by roughly 16%. Crucially, *all three sectors could be up in absolute price* over these 18 months -- in a bull market, the laggard's underlying price might still be 30% higher. RS does not tell you whether a sector made money; it tells you whether it beat the benchmark. That is a different, and for rotation a more useful, question.

### 2. The rate-of-change view -- the momentum snapshot

The RS line shows the trend, but to *rank* sectors against each other you want a single number per sector that summarizes recent relative performance. The standard tool is **rate of change (ROC)** of relative performance over a fixed lookback -- commonly **13 weeks** (one quarter) for sector work, sometimes 4 weeks for faster signals or 26/52 weeks for slower ones.

The 13-week relative ROC for a sector is just:

```
rel_ROC = (sector return over 13 weeks) - (VN-Index return over 13 weeks)
```

A positive number means the sector beat the index over the last quarter; a negative number means it lagged. This is the number you sort on to build a ranking. (More formally you can take the percentage change of the *RS line itself* over 13 weeks, which is mathematically almost the same thing for modest moves; the subtraction form above is the intuitive version and is what most desks quote.)

Why 13 weeks specifically? It is a deliberate compromise between two failure modes. Too short a lookback (say 2 weeks) and the relative ROC swings wildly on noise -- in a retail-driven market like Vietnam, a single week's margin-call cascade or foreign-flow headline can flip a 2-week relative number from +5% to -5% and back, generating a "leadership change" that is pure randomness. Too long a lookback (say 52 weeks) and the number is so smoothed that by the time it confirms a rotation, the rotation is mostly over and you have missed the move. Thirteen weeks -- one quarter -- is long enough to average out weekly noise and align with the earnings-reporting rhythm that actually drives sector leadership, but short enough to catch a rotation while it still has room to run. You can and should look at multiple lookbacks: a 4-week relative ROC for the fast, early tell, the 13-week for the core signal, and a 26-week for the slow, structural trend. When all three agree -- 4-week, 13-week, and 26-week relative ROC all positive and rising -- you have a high-conviction leader. When they disagree (4-week negative, 26-week positive), the sector is a long-term leader having a short-term pullback, which is often exactly when you want to add.

There is one practical wrinkle worth flagging: the relative ROC is path-independent over the window, so it can mask the *shape* of how the outperformance happened. A sector that beat the index by 6% in a smooth grind reads the same +6% as a sector that beat it by 20% then gave back 14% in the final weeks -- but the first is a healthy leader and the second is a sector rolling over. This is exactly why you never rely on the ROC number alone; you look at the RS *line* to see the shape, and you watch the *change* in the ROC week to week (is +6% this week up from +3% last week, or down from +12%?). The number ranks; the line and its slope diagnose.

#### Worked example: 13-week rate-of-change ranking

Suppose over the trailing 13 weeks VN-Index returned **+5.0%**. Three sectors did this:

- **Securities (brokers):** +11.5% over the 13 weeks.
- **Banks:** +8.0% over the 13 weeks.
- **Utilities:** +1.5% over the 13 weeks.

Compute each sector's relative ROC by subtracting the index's +5.0%:

```
Securities: +11.5% - 5.0% = +6.5%  (leading by 6.5 points)
Banks:      + 8.0% - 5.0% = +3.0%  (leading by 3.0 points)
Utilities:  + 1.5% - 5.0% = -3.5%  (lagging by 3.5 points)
```

Rank them by relative ROC: **Securities (+6.5) > Banks (+3.0) > Utilities (-3.5)**. Note that *all three sectors made money* -- utilities was up 1.5% in absolute terms -- yet utilities is the clear laggard because it captured far less of the move than the index did. **The ranking is by who beat the benchmark, not by who was green; a sector can be up 1.5% and still sit at the bottom of the RS table.** A 13-week window long enough to filter daily noise but short enough to catch a real rotation is the workhorse lookback for VN sector analysis.

### 3. The RS ranking across sectors -- the dashboard view

Do the rate-of-change calculation for *every* sector, then sort. The result is a leaderboard: the sectors currently winning the most share of VN-Index at the top, the ones bleeding the most share at the bottom. This is the single most actionable artifact in the whole discipline, and we will build it in detail later.

![Horizontal bar chart ranking nine VN sectors by 13-week relative performance versus VN-Index, positive bars green and negative bars red](/imgs/blogs/relative-strength-reading-which-sector-leads-vn-index-4.png)

The figure is an illustrative snapshot of such a ranking. Nine sectors, sorted by their 13-week relative performance versus VN-Index. IT leads at +9.0%, banks at +6.0%, securities at +4.5%, retail at +3.0%, steel barely positive at +1.0%. Below the zero line, oil & gas at -1.0%, real estate at -2.5%, utilities at -3.5%, food & beverage at -4.0%. Everything above zero is leading the index; everything below is lagging it. A rotation-minded investor reads this top-down: the leadership is concentrated in financials and tech; the laggards are the defensives (utilities, staples) and the rate-sensitive (real estate) -- a textbook risk-on, growth-leaning regime. We will return to *why* that pattern means what it means.

## Reading an RS chart

Computing RS is arithmetic. Reading it is where the skill lives. Here is the honest hierarchy of what an RS line tells you, from most reliable to least.

**An RS uptrend equals leadership.** The single most reliable read: a rising RS line, sustained over weeks, means the sector is consistently beating the index. That is leadership, full stop. It is descriptive (it tells you what *has* happened), but a trend that has persisted for a quarter has a decent chance of persisting a while longer -- momentum in relative performance is one of the more robust empirical regularities in markets, including emerging markets like Vietnam. When VN-Index is itself in an uptrend, a sector with a rising RS line is the highest-conviction long you can identify with a single line: the market is going up *and* this sector is going up faster.

**RS new highs versus price new highs -- the confirmation test.** Here is the subtle, powerful read. Watch what the RS line does when the sector's *price* makes a new high. If the RS line *also* makes a new high, the leadership is real and confirmed -- the sector is making new highs *and* winning more share of the index. But if the price makes a new high while the RS line does *not* -- if RS has rolled over and is making lower highs while price grinds up -- you have a **divergence**. The sector is still rising in absolute terms, but it is rising *slower than the index*; it is losing share even as it makes new price highs. That is the fingerprint of distribution: the flow is leaving the sector, rotating elsewhere, before the price admits it.

![Before and after diagram contrasting RS rolling over while price makes a new high against RS confirming a new high with price](/imgs/blogs/relative-strength-reading-which-sector-leads-vn-index-3.png)

The figure lays the two cases side by side. On the right (the confirming case), price prints a new high and the RS line makes a new high too -- the ratio versus VN-Index is rising, the move is confirmed, flow is arriving, the sector is genuinely winning share. On the left (the diverging case), price prints a new high but the RS line has already rolled over and peaked weeks earlier -- no new RS high -- which means the flow is leaving early and leadership is being handed off to the next sector. **The same green price candle means opposite things depending on whether RS confirms it; the RS line is the lie detector for a price high.** This is the early-handoff tell, and it is the reason serious rotation traders watch RS lines and not just price.

#### Worked example: beating the index while "up less"

This is the trap from the opening, made concrete. Two sectors over the same quarter, with VN-Index up **+7.0%**:

- **Sector X:** up **+10.0%** in price. RS rising: +10.0% - 7.0% = **+3.0%** relative. Leading.
- **Sector Y:** up **+4.0%** in price. RS *falling*: +4.0% - 7.0% = **-3.0%** relative. Lagging.

Sector Y is **green**. Its price is up 4%. An investor looking only at absolute returns is happy with Sector Y -- it made money. But its RS line is *falling*, because it captured only 4 of the 7 points the index delivered. In a portfolio sense, holding Sector Y instead of just buying the index cost you 3 points of relative performance over the quarter. To make the dollars vivid: on a **500 million VND** position (about **\$19,700** at \$1 = 25,400 VND), being up 4% earns you 20 million VND (~**\$790**) -- real money -- while the index would have earned you 35 million VND (~**\$1,380**) on the same capital. You were green and you still left **\$590** of relative performance on the table. **"Up" felt like winning; relative strength shows it was losing the race, and the RS line would have flagged it in real time.**

**RS divergence as an early handoff tell.** Combine the two reads above and you get the most valuable signal RS provides: a leader whose price is still rising but whose RS has clearly topped and turned down is *handing off leadership*. The flow that was lifting it is rotating into the next group. This is your cue to start trimming the old leader and hunting for the new one -- the sector whose RS line is just beginning to turn *up* from below. Rotation is, at its heart, the transfer of relative strength from one sector to the next, and the RS line is the only place you can watch that transfer happen in real time.

**One layer deeper: RS and breadth.** A sector's RS line can be lifted by the whole basket rising together, or by one or two mega-caps dragging the index-weight up while the rest of the sector stagnates. These two situations look identical on the RS line but mean very different things. Broad-based RS strength -- most of the names in the sector outperforming -- is durable; it reflects a genuine improvement in the sector's shared driver. Narrow RS strength -- one giant carrying the basket -- is fragile, because the moment that one name stumbles, the RS line collapses. In Vietnam this matters acutely: a banking RS line can be almost entirely Vietcombank (a **\$15bn**-plus weight), or a property RS line can be almost entirely Vinhomes. When you see strong RS, glance at the breadth underneath: how many names in the basket are above their own rising trend? If RS is strong but breadth is thin, treat the leadership as borrowed, not owned -- it can reverse fast. RS tells you the basket is winning; breadth tells you whether the win is real or a one-stock illusion.

**RS and foreign flows.** A final read specific to emerging markets. In Vietnam, foreign investors (the *room ngoại*, or foreign ownership room, which caps how much of certain stocks foreigners can hold) can be a meaningful marginal buyer or seller in large-cap sectors. When a sector's RS line turns up *at the same time* as sustained foreign net buying (measured in **\$mn** of daily net inflow), the leadership has an extra leg of support -- foreign money tends to be stickier and more cycle-driven than retail. When RS strength comes *despite* foreign net selling, it is being driven by domestic retail margin, which is faster and more fickle. The RS line does not show you who is buying; pairing it with the foreign-flow print tells you *how durable* the leadership is likely to be.

## Building a VN sector-RS dashboard

Reading one RS line is useful. Reading *all* the sectors at once, ranked, refreshed on a schedule, is where RS becomes a genuine rotation system. Here is how to build a simple one for VN-Index.

**The ingredients.** You need, for each sector you track, a sector index or a representative basket: financials (VNFIN or a bank basket), securities/brokers, real estate (VNREAL), technology (VNIT or FPT-led basket), industrials and steel (HPG-led basket), retail and consumer (MWG, PNJ, MSN-led), oil & gas (GAS, PLX, BSR), utilities (POW, NT2, GEG), food & beverage (VNM, SAB, MSN-staples). For VN-Index, use the headline index. Eight to twelve sectors is plenty -- enough to see rotation, not so many that the dashboard becomes noise.

**The recipe.** For each sector, every week:

1. Pull the sector index level and VN-Index level at the weekly close.
2. Compute the RS ratio (sector / index) and the 13-week relative ROC.
3. Rank all sectors by 13-week relative ROC, strongest at the top.
4. Flag the *movers*: which sectors climbed in the ranking this week (improving RS) and which fell (deteriorating RS). The change in rank often matters more than the rank itself.
5. Act inside the index trend -- which we will cover in the playbook.

**Why weekly, not daily.** Daily RS is mostly noise, especially in Vietnam's retail-dominated market where a single session can swing on margin-call cascades or a foreign-flow headline. The 13-week ROC on weekly closes filters that. You are looking for *persistent* shifts in leadership, not one-day wiggles. Refresh weekly; act on changes that survive two or three weeks.

**Read the *change* in rank, not just the rank.** The most useful column on the dashboard is not this week's rank -- it is the *change* from last week. A sector sitting at rank 3 tells you it is a current leader. A sector that *climbed from rank 8 to rank 3 in three weeks* tells you something far more actionable: leadership is *arriving* there, the flow is rotating in, the RS momentum is strongly positive. Conversely a sector that *fell from rank 1 to rank 5* is shedding leadership even if it is still technically in the top half -- the flow is leaving. The biggest climbers and the biggest fallers are where rotation is happening; the stable middle is mostly inertia. Keep a small "rank delta" column and let your eye go straight to it.

**Watch the extremes, not the middle.** The top two or three sectors and the bottom two or three carry almost all the information. The top is where leadership lives -- your long candidates. The bottom is where the rotation *out* is happening -- both a list to avoid and, paradoxically, the pool from which the *next* leaders will emerge (a sector cannot move into the Improving quadrant without first having been in Lagging). The middle of the ranking is sectors performing roughly in line with the index -- they are "being the market," and there is little edge in them either way. Spend your attention on the head and tail of the list.

#### Worked example: reading a one-week shift in the leaderboard

Suppose your dashboard last week and this week looks like this for four sectors, ranked by 13-week relative ROC (the index returned about +5% over the quarter in both readings):

```
sector        last week   this week   rank change
IT            +9.0%  (#1)  +7.5%  (#2)   down 1
Banks         +3.0%  (#4)  +6.0%  (#1)   up 3
Securities    +6.0%  (#2)  +4.5%  (#3)   down 1
Real Estate   +4.0%  (#3)  -2.5%  (#4)   down 1
```

The naive read is "IT is still the leader, hold it." The rotation read is sharper. Banks *climbed three ranks* to the top on improving RS -- leadership is arriving there; that is your add. Real estate *collapsed* from +4.0% to -2.5%, the single biggest deterioration on the board -- the flow is fleeing; that is your trim or avoid, regardless of whether real estate is still green in absolute terms. IT and securities are gently fading from the top -- still strong on level, but losing momentum, the Weakening quadrant. **The leaderboard's value is in its *deltas*: banks improving, real estate deteriorating -- those two moves, not the static ranking, are the actionable signal this week.**

![Pipeline diagram of the weekly relative strength routine: pull sector indices, divide by VN-Index, rank, flag movers, then act inside the trend](/imgs/blogs/relative-strength-reading-which-sector-leads-vn-index-7.png)

The figure walks the loop end to end. Pull the sector indices at the weekly close. Divide each by VN-Index and normalize to a common scale. Rank by 13-week relative performance, strongest to weakest. Flag the sectors whose RS is improving (turning up the table) and those deteriorating (falling down it), watching the top and bottom of the list hardest. Then act -- but only *inside* the index trend, adding to improving RS in an uptrend and cutting deteriorating RS. **Reading sector leadership is not a flash of insight; it is a five-step weekly routine you can run in twenty minutes.**

### The relative-rotation graph (RRG)

There is one more way to look at the whole sector universe at once, and it is worth knowing because it makes rotation *visible* as motion. It is called the **relative-rotation graph**, or RRG. Do not be intimidated -- it is just the RS dashboard plotted on two axes instead of in a list.

The horizontal axis is **RS level**: how strong the sector's relative strength is right now (left = weak/lagging, right = strong/leading). The vertical axis is **RS momentum**: whether that relative strength is getting better or worse (down = deteriorating, up = improving). Two axes, four quadrants, and every sector is a dot somewhere in the plane.

![Quadrant diagram of a relative rotation graph with improving leading weakening and lagging quadrants and clockwise arrows showing the rotation path](/imgs/blogs/relative-strength-reading-which-sector-leads-vn-index-5.png)

The figure names the four quadrants. **Improving** (top-left): weak RS level but rising momentum -- the next leaders forming, like banks turning up through 2024. **Leading** (top-right): strong RS and still rising -- the current winners, like securities in a 2021 melt-up. **Weakening** (bottom-right): still strong on level but momentum rolling over -- time to take profits, a late-stage broker run. **Lagging** (bottom-left): weak and still deteriorating -- the avoid pile, like utilities in a risk-on tape. The arrows show the path: sectors tend to rotate **clockwise** -- from Improving, up into Leading, over into Weakening, down into Lagging, and eventually back around to Improving. **An RRG turns the abstract idea of rotation into a literal clockwise journey you can point at; a healthy sector you want to own is one moving from Improving into Leading, and a sector to lighten is one sliding from Leading into Weakening.** You do not need fancy software for the intuition: the RS ratio line tells you the horizontal position (level), and the RS line's recent slope tells you the vertical position (momentum). The RRG is just those two facts plotted together.

## The honest limits of relative strength

If this post left you thinking RS is a crystal ball, it has failed you. RS is genuinely useful and also genuinely limited, and a good analyst holds both ideas at once. Here are the limits, stated plainly.

**RS lags.** It is built from past prices -- a ratio of where the sector and the index *have been*. By construction it can only tell you that a rotation *has been happening*, never that one is *about* to happen. The RS line turns up after the flow has already started arriving, and turns down after it has already started leaving. You are reading a wake, not a forecast. This is fine -- riding an established trend is a perfectly good strategy -- but never mistake a descriptive tool for a predictive one.

**RS whipsaws in a narrow market.** Vietnam's market is dominated by retail investors trading on margin (ký quỹ -- borrowed money from brokers), and breadth is often thin -- a handful of large-cap names can swing a whole sector index. In a narrow, choppy tape, RS lines cross back and forth across their trends, generating false leadership signals that reverse within weeks. The 13-week lookback helps, but in a sideways, low-conviction market even a quarter of relative performance can be mostly noise. RS works best when there is a real trend for it to measure; in a directionless market it mostly measures randomness.

**RS says nothing about value or risk.** A sector with the strongest RS in the market might be the most expensive, most crowded, most vulnerable group in the market -- RS is silent on valuation, on balance-sheet quality, on how far price has run ahead of earnings. The strongest RS line in 2021 belonged to securities brokers trading at nosebleed multiples right before they fell 60%. RS told you they were leading; it could not tell you they were a bubble. RS is one input -- the *who is winning the flow* input -- and it must be combined with valuation, the macro cycle, and risk management. On its own it is a momentum signal, and momentum signals are exactly the ones that hurt most when they break.

**RS is relative, so everything can be falling.** A sector with the best RS in a bear market is still losing you money -- it is just losing *less* than the index. RS ranks sectors against each other; it does not tell you whether you should be in the market at all. That decision -- the index trend itself -- is a separate, and more important, judgment. Leadership inside a downtrend is leadership of a sinking fleet.

## Common misconceptions

**Myth 1: "RS is the same as RSI."** This is the big one, and the names are cruelly similar. **RS** -- relative strength -- is the *ratio of two prices*: a sector divided by the index, an unbounded line that can rise or fall without limit, measuring one thing against *another* thing. **RSI** -- the Relative Strength *Index* -- is a bounded **0--100 momentum oscillator** computed from a *single* price's own up-moves versus down-moves over a lookback (classically 14 periods); it measures a price against *itself*, and it is "overbought" near 70 and "oversold" near 30. They share a word and nothing else. RS answers "is this sector beating the market?" RSI answers "has this one price moved up too far, too fast, relative to its own recent range?" If you take one thing from this post: **RS is a ratio versus the index; RSI is an oscillator of a price against itself.** (For the RSI side of that distinction in depth, see the technical-analysis post on momentum oscillators linked below.)

**Myth 2: "Strong RS means buy at any price."** No. RS tells you a sector is leading; it does not tell you that *this moment* is a good entry. The strongest-RS sector can be wildly overbought on its own price, extended far above its moving averages, ripe for a sharp pullback. The number proves it: a broker basket can have a 13-week relative ROC of **+20%** -- screaming leadership -- and still drop 15% in price over the next three weeks as the overextension unwinds, even while its RS stays strong. Use RS to pick *which* sector; use trend, support, and risk management to pick *when and at what price*. RS chooses the horse; it does not choose the entry gate.

**Myth 3: "RS predicts the future."** RS is descriptive and lagging, as we said. It tells you who *has been* winning, with the reasonable but not guaranteed inference that leadership tends to persist. The moment you treat an RS uptrend as a *promise* rather than a *description with momentum*, you will be holding the old leader when its RS rolls over -- which is exactly when the divergence tell is screaming at you to leave. RS is a rear-view mirror with a slight forward tilt, not a windshield.

**Myth 4: "A flat RS line is useless."** Beginners chase the steep, exciting RS lines and ignore the flat ones. But a flat RS line carries real information: it says "this sector is *being the market* -- neither leading nor lagging." That is exactly what you should expect from a sector that is roughly index-weighted and index-driven, and it is the correct null hypothesis. A flat RS line that suddenly *turns* -- breaks up out of a long sideways stretch -- is one of the cleaner early-leadership signals you will get, because the break is unambiguous against the prior flat baseline. The number makes it concrete: a sector whose 13-week relative ROC sat in a tight **-1% to +1%** band for two quarters and then prints **+5%** has done something genuinely new; the same +5% from a sector that has been swinging **±8%** is just more noise. Flat is not nothing; flat is the baseline against which a real move is measured.

**Myth 5: "The sector with the highest RS is always the best buy."** The top of the RS table is the *most extended*, not necessarily the *best risk-reward*. A sector at the very top has, by definition, already done most of its outperforming -- the flow has largely arrived. The better risk-reward is often one or two ranks below the top, in a sector whose RS is *improving fast from a lower base* (the Improving quadrant moving toward Leading), because that sector still has the bulk of its relative move ahead of it. Buying the #1-ranked sector at the top of its RS run is how investors end up holding the late stage of a move; buying the sector climbing the ranking is how they catch the middle of one. Rank tells you who is winning; rank *change* tells you who is *starting* to win, and the second is usually the better entry.

## How it shows up on VN-Index

Abstractions are cheap. Here is RS doing real work on real Vietnamese market history.

**The 2021 broker melt-up, and the RS top before the price top.** Through 2021, Vietnam's securities sector (the brokers -- SSI, VND, VCI, HCM) was the loudest leader on the market. Trading volumes exploded as a wave of new retail accounts (so-called F0 investors) opened during the pandemic, and brokers, whose earnings scale directly with trading turnover and margin lending, saw profits surge. Their RS line versus VN-Index climbed relentlessly through 2021 -- a textbook leader, sitting deep in the Leading quadrant of an RRG. But here is the lesson: the brokers' *RS line peaked and began rolling over in late 2021, weeks before the price peaked in early 2022*. While VND and SSI were still grinding to marginal new price highs, their relative strength had already topped -- they were rising slower than the index, losing share, distributing. The RS divergence flagged the handoff before the brutal 2022 decline, in which the sector fell more than 60% from its highs. An RS-aware investor was trimming brokers while a price-only investor was still buying the "new highs."

#### Worked example: a position rotated from deteriorating to improving RS

Make the 2021-into-2022 lesson concrete with a real-sized position. Suppose in late 2021 you held a **500 million VND** sleeve (about **\$19,700** at \$1 = 25,400 VND) in the securities sector, which had been your best performer all year. Your weekly RS dashboard now shows securities' 13-week relative ROC rolling over from +18% to +6% to -2% over three weeks -- deteriorating RS, a clear handoff signal -- while banks' relative ROC is turning up from -4% to +1% to +5% -- improving RS, a fresh leader forming.

You rotate the full sleeve from securities into banks. Over the following quarter, securities falls another 25% while banks rises 8%. Had you stayed:

```
stay in securities: 500m VND x (1 - 0.25) = 375m VND  (down 125m, ~ -$4,900)
rotate to banks:    500m VND x (1 + 0.08) = 540m VND  (up  40m, ~ +$1,575)
```

The rotation is worth the difference -- **165 million VND**, roughly **\$6,500** -- on a single sleeve, purely from acting on the RS handoff rather than holding the old leader. **The signal was not "sell because price fell"; it was "rotate because relative strength left securities and arrived at banks," and RS let you read that transfer weeks before price confirmed it.** (This is illustrative arithmetic, not a record of specific trades.)

**Banks' RS turning up into 2024.** Run the mirror image. Through much of 2022 and 2023, Vietnamese banks were a market laggard -- weighed down by the corporate-bond crisis (the trái phiếu doanh nghiệp blowup that hit property developers and the banks exposed to them), by net-interest-margin (NIM -- the spread between what a bank earns on loans and pays on deposits) compression, and by asset-quality fears. Their RS line versus VN-Index drifted lower; banks sat in the Lagging quadrant. But through late 2023 and into 2024, as the State Bank of Vietnam (SBV) cut policy rates, credit growth resumed, and bond-market fears faded, the banks' RS line *turned up from below* -- the Improving quadrant. Months before banks became an obvious consensus leader, their relative strength was already inflecting. An RS dashboard caught the turn early; a headline-reader caught it after banks had already run.

**The defensive RS tell at a market top.** A subtler one. When defensives -- utilities (POW, NT2), water and power, consumer staples (VNM) -- quietly start *outperforming* in a market that is still making new highs, their RS lines turning up while cyclicals' RS rolls over, it is often a late-cycle warning. Money is rotating toward safety even as the index grinds higher. The RS dashboard surfaces this rotation-into-defensives before the index itself turns, because the index can keep rising on a few mega-caps while *under the surface* leadership has already shifted to the boring, defensive groups. RS reads the surface; the index reads only the average.

![Dual axis chart of a sector price versus its RS line showing RS peaking around month twelve while price grinds to a higher high around month sixteen](/imgs/blogs/relative-strength-reading-which-sector-leads-vn-index-6.png)

The figure is the broker-style divergence in stylized form. The blue line is the sector's absolute price; the amber line is its RS versus VN-Index. The RS line peaks around month 12 and rolls over -- leadership fading -- while the *price* grinds on to a marginally higher high around month 16 before finally turning. The four-month gap between the RS top and the price top is the early warning: by the time price admits the trend is over, RS has been telling you for a quarter. **RS rolls over before price because flow leaves a sector before the last buyers do; that lead time is the entire practical value of watching relative strength.**

## The playbook: using RS for rotation

Everything above converges here. This is how you actually use relative strength to position in VN-Index sectors. It is deliberately simple, because complexity in this domain mostly adds noise.

**Signal 1 -- the regime filter (do this first).** RS is *relative*; it ranks sectors against each other but says nothing about whether you should be invested at all. So before anything, check the index trend. Is VN-Index above its rising 30- or 40-week moving average (an uptrend), or below a falling one (a downtrend)? RS-based rotation is a strategy for *uptrends and neutral markets*. In a confirmed downtrend, the right move is usually less exposure overall, not picking the prettiest RS line on a sinking ship. Leadership of a falling market is still a falling position.

**Signal 2 -- buy improving RS in an uptrend.** When VN-Index is in an uptrend, your longs come from the top of the RS ranking *and*, even better, from sectors moving *up* the ranking -- the Improving and Leading quadrants. A sector whose 13-week relative ROC has turned positive and is climbing, while the index trends up, is the highest-conviction long the dashboard produces. You are stacking two edges: the market is rising, and this sector is rising faster.

**Signal 3 -- cut deteriorating RS.** The exit discipline. When a sector you hold sees its RS line roll over and start making lower highs -- especially the price-high-without-RS-high divergence -- you trim or exit, regardless of whether its absolute price is still green. This is the hardest rule emotionally, because you are selling something that is *still going up*. But that is precisely the point: you are selling the old leader into strength, before the price catches down to what RS has already told you.

**Sizing and the invalidation.** RS picks the sector; it does not size the position or set the stop -- you still need those. Size to a fixed fraction of the sleeve and set a hard invalidation: the trade thesis is *this sector is leading*, so the thesis is **invalidated when its RS line breaks its uptrend** -- when the 13-week relative ROC turns clearly negative and the RS line makes a decisive lower low. That is your line in the sand. Not a price stop (though you should have one of those too for risk), but a *relative-strength stop*: the moment the sector stops leading, the reason you owned it is gone, and you rotate to wherever the RS has gone. Combine RS leadership with the liquidity backdrop -- in Vietnam, leadership only persists while margin balances and turnover are expanding; when the margin cycle rolls over, even the strongest RS lines tend to whipsaw, so size down when liquidity tightens.

#### Worked example: sizing a rotation around the RS invalidation

Put the invalidation into position-sizing terms. You have a **1 billion VND** sector sleeve (about **\$39,400** at \$1 = 25,400 VND) and your dashboard puts banks at the top of the RS table with rising momentum, in a confirmed VN-Index uptrend. You commit 40% of the sleeve -- **400 million VND** (~**\$15,700**) -- to a bank basket. Your relative-strength invalidation is the level where the banks' RS line breaks its rising trend; in price terms that sits about 8% below your entry. So your risk on the position, if the relative-strength stop is hit, is roughly 8% of 400 million VND = **32 million VND** (about **\$1,260**), which is 3.2% of the total sleeve -- a sane single-sector risk. If instead the RS strength is narrow (one mega-cap carrying it) or liquidity is tightening, you halve the commitment to 20% of the sleeve, cutting the same-percentage risk to about **16 million VND** (~**\$630**). **The RS signal chooses the sector and defines the invalidation; position sizing then translates that invalidation into a fixed, survivable dong amount -- RS without sizing is a tip, RS with sizing is a system.**

**Where RS sits in the toolkit.** Be clear about what RS is and is not in your process. It is the *which sector* filter -- the single best tool for ranking groups by who is winning the flow. It is *not* the *whether to be invested* tool (that is the index-trend regime filter), *not* the *what price to enter* tool (that is trend, support, and your trigger), *not* the *how much to risk* tool (that is position sizing), and *not* the *is it cheap* tool (that is valuation). A complete process layers all five: regime filter says yes, RS ranking says which sector, price structure says where to enter, sizing says how much, valuation says whether the leader is a reasonable price or a bubble. RS is one clean, honest input among several -- the input that answers the one question the daily color cannot.

**Put it together.** Each week: confirm the index trend (regime filter). Rank sectors by 13-week relative RS. Buy or add to the improving leaders inside an uptrend. Trim the sectors whose RS is deteriorating or diverging from price. Hold your invalidation -- exit when RS breaks down, not when your feelings change. That is the entire system, and its honesty is its strength: it makes no prediction, it just keeps you aligned with wherever the flow is *currently* winning, and rotates you when the flow moves. RS will not catch the exact top or bottom of any rotation -- it lags, it whipsaws, it ignores value. But used as a *who-is-leading* filter inside a sound regime and risk framework, it is the single most useful lens for sector rotation on VN-Index, and the cleanest answer to the question the daily color cannot answer: of all the sectors that are up, which one is actually leading?

## Further reading & cross-links

- [Sector rotation explained: leaders and laggards](/blog/trading/vietnam-stocks/sector-rotation-explained-leaders-and-laggards) -- the framework RS feeds into: how leadership passes from one group to the next across the cycle.
- [Anatomy of a stock sector: why industries move together](/blog/trading/vietnam-stocks/anatomy-of-a-stock-sector-why-industries-move-together) -- why a whole sector shares a common driver, which is what makes a sector index (and thus an RS line) meaningful in the first place.
- [Liquidity and the margin cycle in Vietnam](/blog/trading/vietnam-stocks/liquidity-and-the-margin-cycle-vietnam) -- the backdrop that decides whether RS leadership persists or whipsaws: when margin balances expand, leaders run; when the cycle rolls over, RS signals get noisy.
- [Sector rotation through the cycle](/blog/trading/cross-asset/sector-rotation-through-the-cycle) -- the cross-asset version of the same idea, with the cycle-phase playbook (early/mid/late/recession) that tells you *which* sectors' RS should be improving in each regime.
- [RSI and momentum oscillators](/blog/trading/technical-analysis/rsi-and-momentum-oscillators) -- the other "relative strength," the bounded oscillator RS is so often confused with; read this to keep RS and RSI permanently separate in your head.

*This article is educational and not financial advice. Relative strength is a descriptive, lagging tool; it does not predict returns, and past leadership does not guarantee future leadership. Position sizes and dollar figures are illustrative, computed at roughly \$1 = 25,400 VND. Do your own research and manage risk.*

---
title: "Building a Macro-Correlation Monitor: The Weekly Routine"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A repeatable weekly process to keep a live read on which macro correlations are active and which are about to flip: the regime read, the watchlist, the flip check, and the exposure audit that turns three bets into one trade."
tags: ["macro", "correlation", "monitoring", "weekly-routine", "regime", "watchlist", "risk-management", "stock-bond-correlation", "exposure-audit", "positioning"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A correlation is not a fact you look up once; it is a live reading that drifts and flips, so you need a *routine* to keep it current. The monitor is a five-step weekly loop — read the regime, scan a fixed watchlist, check for flips, audit your concentration, write one sentence — and its whole job is to make sure you trust only the correlations your current regime actually turns on.
>
> - The single most important number to monitor is the **stock-bond correlation**. It sat near **−0.40** for two disinflation decades, flipped to **+0.60** in 2022, and that one sign change broke the 60/40 portfolio. If you watch only one line, watch this one.
> - **The regime selects the correlations.** In a deflation scare bonds hedge stocks at about −0.55; in a stagflation scare the same pair runs +0.55 and falls together. The quadrant you are in tells you which relationships to trust this week.
> - **A flip is worth more than a level.** Catching the stock-bond correlation crossing zero a quarter early on a \$100,000 60/40 book is the difference between a −8% hedged drawdown and a −18% naked one — roughly **\$10,000** saved.
> - **The exposure audit is the part everyone skips.** Long gold, long EM, short dollar feel like three independent bets; because each correlates about −0.55 with the dollar, they are one dollar trade wearing three hats. The monitor's last job is to catch that before the market does.

In late September 2022 a friend who runs a modest discretionary book — call it \$100,000, the size of a serious individual investor's portfolio — sent me a screenshot of his risk report. It was green across the board. His model said his positions were "well diversified": long gold as an inflation hedge, long emerging-market equities for growth, and short the US dollar as a macro view. Three different asset classes, three different theses, three different lines on the screen. The risk model netted them out to a comfortable, moderate number.

Six weeks later the dollar index (DXY) had pushed to a multi-decade high near 114, and all three positions were down together. Gold had fallen. EM had fallen harder. The short-dollar bet was the only thing that should have worked, and it had — but it was a third the size of the losses on the other two. His "diversified" book had behaved like one concentrated bet on a *weaker* dollar, placed at exactly the moment the dollar was at its strongest in twenty years. The risk model never flagged it because nobody had asked the one question that mattered: *are these three bets actually the same trade?*

That question, asked every week, in a fixed order, with a fixed watchlist, is what this post is about. We are not building code here — [the Python dashboard that pulls the data and draws the heatmap](/blog/trading/macro-correlations/building-a-macro-asset-correlation-dashboard-in-python) is a separate, complete post, and you should read it for the *machinery*. This post is the **operating manual**: what to look at, in what order, and what each reading means for what you do on Monday morning. The dashboard is the instrument; the monitor is the routine you run on it. A telescope is useless without a habit of looking up.

![Weekly macro correlation monitor loop refresh regime watchlist flip audit thesis](/imgs/blogs/building-a-macro-correlation-monitor-the-weekly-routine-1.png)

## Foundations: why a correlation needs monitoring at all

Before the routine, we need to be honest about *why* a routine is necessary — because if correlations were stable, you could look one up in a textbook and never think about it again. The entire reason this monitor exists is that **a correlation is a regime, not a constant**, and that single idea, developed in depth in [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant), is the foundation everything else stands on.

Let us define the term carefully, because the whole series rests on it. A **correlation** is a single number between −1 and +1 that summarizes how two things move *together*. The specific number we use is the **Pearson correlation coefficient**, written *r*. If *r* = +1, two series move in perfect lockstep — when one is above its average, so is the other, by a proportional amount. If *r* = −1, they move in perfect opposition: one up means the other down. If *r* = 0, knowing one tells you nothing about the other. The clean way to picture it: line up the week-by-week *moves* of two assets, and ask how reliably they point the same direction. Gold and the dollar usually point opposite ways (a negative correlation); stocks and bonds *used* to (negative), then started pointing the same way (positive). The full taxonomy — Pearson versus Spearman versus beta — lives in [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta); here we only need the intuition: a correlation is a number that says "how together do these two move."

Now the load-bearing fact: **that number is not fixed.** It is computed over some window of time, and as the world changes regime, the number changes too — sometimes it merely weakens, sometimes it flips sign entirely. The classic example, which we will return to again and again because it is the most important correlation in all of macro investing, is the relationship between stocks and bonds. For two decades after the late 1990s, US stocks and Treasury bonds were *negatively* correlated. When stocks fell, bonds rose, and the rally in bonds cushioned the loss in stocks. That negative correlation is the entire engine of the "balanced" 60/40 portfolio — it is *why* holding both is supposed to be safer than holding either alone. Then in 2022 inflation became the dominant risk, the Federal Reserve hiked aggressively, and stocks and bonds fell *together*. The correlation flipped from about −0.40 to about +0.60. The hedge stopped being a hedge.

Here is the deep point. If you had looked up "the stock-bond correlation" in 2011 and written −0.40 into a spreadsheet, you would have been *right* — for 2011. And you would have carried that number, untouched, into 2022, where it was catastrophically wrong. The number did not betray you; *your failure to keep watching it* did. A correlation is like a tide table that is only valid for one coastline in one season. Use last season's table on this season's beach and you will get wet. The monitor is the discipline of re-checking the tide table every week.

There is a second foundational concept: the **window**. A correlation is computed over a span of observations, and the choice of span is a choice, not a default. A short window — say the last 13 weeks (one quarter) — is twitchy: it reacts fast to a regime change, but each reading carries a wide cloud of uncertainty because it is built from few data points. A long window — say 104 weeks (two years) — is smooth and statistically reliable, but it lags: by the time a two-year window confirms a flip, the flip happened a year ago. The art of monitoring, developed in [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters), is to watch *both*: the short window to catch flips early as a warning, the long window to confirm them before you bet real money. We will use that two-window logic in the flip check.

There is one more reason a routine beats a one-time look, and it is psychological rather than statistical. Markets do not announce regime changes; they drift through them, and a drift is invisible from inside it. If you check a correlation once a year, each check is a fresh shock — you see −0.40 in 2021 and +0.60 in 2022 and it feels like a discontinuous jump, a thing that came out of nowhere. If you check it every week, you watch the line *crawl* from −0.20 to −0.05 to +0.10 to +0.30, and the move is no longer a shock; it is a trend you saw building for a quarter. The weekly cadence does not make you a better forecaster of the endpoint; it makes you an earlier *noticer* of the journey. And in markets, noticing early is most of the edge — the investor who saw the stock-bond correlation creeping positive in spring 2022 did not need to predict the year; she only needed to act on what the line was already telling her.

So the monitor exists because correlations drift and flip, because no single look at a single window is trustworthy, and because a weekly habit turns invisible drifts into visible trends. The job of the routine is to convert "correlation is a moving regime" from a slogan into a Monday-morning habit.

### The monitor versus the dashboard: routine versus code

It is worth drawing a sharp line between this post and its sibling, because they are easy to confuse. The [correlation dashboard](/blog/trading/macro-correlations/building-a-macro-asset-correlation-dashboard-in-python) is *code*: roughly 60 lines of pandas that pull macro series and asset prices, align them onto one calendar, convert them to changes, and compute a rolling correlation matrix and a few rolling-correlation lines. Its output is a heatmap and some line charts. That is the **instrument**.

The monitor is the **routine** you run on the instrument: which numbers on the heatmap to read first, what each reading means for your positioning, how to spot a flip before it is obvious, and the one question — the exposure audit — that no dashboard answers automatically because it depends on *your* positions, not on the market's. You can run the monitor on the dashboard, or on a paid data terminal, or — for an individual investor — on a handful of charts you eyeball every Monday. The instrument is interchangeable; the routine is the value. This post is the routine.

The governing principle of the routine, the one sentence to tattoo on the inside of your eyelids: **identify the regime first, then trust only the correlations that regime activates.** Most people do the opposite. They memorize a correlation ("gold goes up when inflation goes up" — which, as we will see, is wrong), apply it in every weather, and are blindsided when the regime that made it true ends. The monitor inverts the order: regime first, correlations second. Let us build it.

## The five components of the monitor

The monitor has four things you *look at* and one thing you *write down*. In order:

1. **The regime read.** Where are we on the growth-by-inflation map, what is the yield curve saying, what are ISM and the CPI trend doing? This is step one because the regime selects which correlations are live.
2. **The watchlist scan.** A fixed, short list of the correlations that matter most — stock-bond above all, then gold-real yield, BTC-Nasdaq, credit-equity, dollar-EM. You check the same list every week so you notice movement.
3. **The flip check.** Which correlation on the watchlist is near a sign change, and does the short window disagree with the long window? A flip in the making is the highest-value thing the monitor finds.
4. **The exposure audit.** Given the live correlations, are your supposedly-independent positions actually one concentrated trade? This is the step that saved nobody in 2022 because nobody did it.
5. **The one-line thesis.** You compress everything into a single sentence — the regime, the live risk, the one trade to size or cut. If you cannot write the sentence, you do not understand your own book.

The figure at the top of this post is that loop, with the Monday data refresh prepended as the cadence's first step — so the picture shows six boxes (refresh plus the five analytical components). We will take each component in turn, and each one comes with a worked example on a \$100,000 book so the routine never floats free of dollars and decisions.

## Component one: the regime read

You cannot trust a correlation until you know what regime you are in, because the regime *is* what makes the correlation true or false. So the monitor always starts here.

The simplest, most durable regime map is the **growth-by-inflation quadrant**: a 2×2 grid of whether growth is accelerating or slowing, crossed with whether inflation is rising or falling. The four boxes have names you will see everywhere in macro:

- **Goldilocks** (growth up, inflation down): the dream regime. Stocks lead, bonds are fine, gold is quiet.
- **Reflation** (growth up, inflation up): the early-cycle boom. Commodities and cyclicals lead; bonds start to struggle.
- **Stagflation** (growth down, inflation up): the nightmare. Cash and gold are the only refuges; stocks *and* bonds fall together.
- **Deflation** (growth down, inflation down): the bust. Bonds are king; long Treasuries are the trade.

The reason this map sits at the front of the monitor is that **the quadrant selects the stock-bond correlation sign**, and therefore decides whether your portfolio's core diversifier is working. The figure below shows the expected stock-bond correlation per quadrant — and notice how it ranges from a comfortable −0.55 in deflation to a portfolio-breaking +0.55 in stagflation.

![Stock bond correlation by macro regime quadrant Goldilocks reflation stagflation deflation](/imgs/blogs/building-a-macro-correlation-monitor-the-weekly-routine-4.png)

That single chart is the spine of the regime read. In deflation and Goldilocks the correlation is comfortably negative — bonds hedge stocks, the 60/40 works, and you can run a normal balanced book. In stagflation it is strongly positive — bonds and stocks fall together, the hedge is gone, and you must cut size or find a different diversifier (cash, gold). In reflation it is a weak −0.10, fading toward zero — the hedge is unreliable, a yellow light. The full development of this is in [correlation by regime: the four macro quadrants](/blog/trading/macro-correlations/correlation-by-regime-the-four-macro-quadrants); the monitor's job is to *locate you on this map every week*.

How do you locate yourself? Four readings, in order:

1. **The growth axis: ISM/PMI.** The ISM manufacturing index is a survey of purchasing managers; above 50 means expansion, below 50 contraction. It leads the hard data by months, which is exactly what you want from a regime gauge — [ISM/PMI: the leading correlation with cyclicals](/blog/trading/macro-correlations/ism-pmi-the-leading-correlation-with-cyclicals) shows it leads S&P earnings growth by roughly six months. Is ISM rising or falling, above or below 50? That sets your growth axis.
2. **The inflation axis: the core CPI trend.** Not the level, the *direction* — is core CPI year-over-year accelerating or decelerating over the last three prints? That sets your inflation axis. The mechanics of which CPI components actually correlate with assets are in [core CPI, shelter, and supercore](/blog/trading/macro-correlations/core-cpi-shelter-and-supercore-what-actually-correlates).
3. **The yield curve: 2s10s.** The slope between the 2-year and 10-year Treasury yields. An inverted curve (2-year above 10-year) has led every modern US recession by 12–18 months — it is the market's own vote on whether growth is about to slow. It is a cross-check on your growth axis. [The yield curve as a growth signal](/blog/trading/macro-correlations/the-yield-curve-as-a-growth-signal-and-its-asset-correlation) covers the lead times.
4. **The dollar: DXY direction.** Not a quadrant axis, but a master cross-asset variable — a rising dollar pressures everything outside the US. You note its direction here because it will dominate the exposure audit later.

Put those four together and you have a regime label: "slowing growth, sticky inflation, inverted curve, strong dollar → late-cycle drifting toward stagflation." That label is the output of component one, and it is what you carry into the watchlist.

But the regime read does more than label a box — it hands you a *whole correlation profile*. Each quadrant does not merely set the stock-bond sign; it nominates one specific correlation as the dominant one for the week and tells you what to do with the book. The matrix below is the regime read in its most useful form: read down to your quadrant, and the row tells you the stock-bond sign, the single correlation that will drive positioning, and the implication for your portfolio.

![Regime read matrix which correlation each macro quadrant turns on stock bond sign book implication](/imgs/blogs/building-a-macro-correlation-monitor-the-weekly-routine-5.png)

Walk the four rows, because each is a different operating manual:

- **In Goldilocks**, the stock-bond correlation is −0.40 (bonds hedge), and the dominant correlation is the one between ISM/PMI and cyclical stocks — the *growth read* is what drives risk, so you watch the leading indicators and run your equity overweight with confidence that bonds will cushion any wobble. This is the regime where the textbook 60/40 actually behaves like the textbook says.
- **In Reflation**, the stock-bond correlation has faded to −0.10 — the hedge is weakening, a yellow light — and the dominant correlation shifts to oil and breakevens versus real assets. Commodities lead. You tilt toward real assets and stop relying on bonds to diversify, because the relationship that protected you in Goldilocks is decaying toward zero.
- **In Stagflation**, the stock-bond correlation is +0.55 (the red row), and the dominant correlation is real yields versus gold — *rate fear* drives everything, so the same shock hits stocks and bonds together. The 60/40 breaks. The book's job is to cut size, raise cash, and hold gold; there is no equity-bond combination that diversifies in this corner.
- **In Deflation**, the stock-bond correlation is −0.55 — the *strongest* bond hedge — and the dominant correlation is credit spreads versus equities, the canary for a debt-deflation spiral. Long Treasuries are the trade; you de-risk equity and let the bond rally do the work.

This is the deep reason the regime read sits first: it does not just tell you the weather, it hands you the *right instrument panel for that weather*. The mistake is to carry one panel — say, the Goldilocks "growth-read drives risk" panel — into a stagflation regime where the rate-fear panel is the one that matters. The monitor swaps panels with the regime.

#### Worked example: reading the regime into a stock-bond decision

You run a \$100,000 portfolio in the textbook 60/40 split: \$60,000 in an S&P 500 index fund, \$40,000 in long-term Treasuries. Your whole premise is that the \$40,000 in bonds will cushion a stock drawdown. Monday's regime read shows: ISM at 48 and falling (growth slowing), core CPI ticking *up* for the third straight month (inflation rising), curve inverted. That is the stagflation corner of the map. The figure tells you the expected stock-bond correlation in stagflation is about **+0.55** — positive. Your \$40,000 "hedge" is no longer a hedge; it is a second \$40,000 bet on the same risk that is hurting your stocks. Concretely: if a rate shock takes the S&P down 10% (a \$6,000 loss on the equity sleeve) in this regime, your bonds do not rally to offset it — at +0.55 they fall too, perhaps 6%, another \$2,400 loss, for a total drawdown near **\$8,400** instead of the \$3,000-ish you would have suffered if bonds had hedged at −0.40. The regime read just told you, *before* the shock, that your portfolio is twice as fragile as your mental model assumes. The intuition: the regime is not background color — it sets the sign of your single most important diversifier, so you read it first or you fly blind.

## Component two: the watchlist scan

Now that you know the regime, you scan a *fixed* list of correlations. The discipline of "the same list, every week, in the same order" is what makes the scan work — you are not hunting for whatever looks interesting (that way lies finding patterns in noise), you are checking known, important relationships for movement. A weather forecaster does not invent new instruments each morning; she reads the same barometer, thermometer, and wind gauge and notices when one of them has changed.

Here is the master watchlist, ranked by how much it should occupy your attention:

1. **Stock-bond correlation** — the number one line, because it decides whether your core diversifier works. Developed below and in [the stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime).
2. **Gold versus real yields** — the cleanest macro correlation when it works (about −0.8), and a famous teaching case in how a strong correlation can *break*. [Real yields and the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation).
3. **Bitcoin versus the Nasdaq** — tells you whether crypto is trading as a macro-liquidity / high-beta-risk asset or off on its own. [Crypto as a macro asset](/blog/trading/macro-correlations/crypto-as-a-macro-asset-the-liquidity-correlation).
4. **Credit spreads versus equities** — the risk-on/risk-off canary; tightens toward −1 in stress. [Credit spreads: the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary).
5. **Dollar (DXY) versus everything outside the US** — gold, EM, oil, copper, crypto. The cross-asset gravity that drives the exposure audit. [The dollar (DXY) cross-asset correlation](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation).

You do not need to memorize a hundred relationships. You need to *scan five* every week. To make the scan a single glance rather than five separate lookups, build a **scorecard**: a grid of which macro driver pushes which asset which way and how hard. The figure below is that scorecard, built from the master correlation map in [the macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix).

![Macro driver versus asset correlation scorecard heatmap rates real yield CPI dollar oil ISM credit](/imgs/blogs/building-a-macro-correlation-monitor-the-weekly-routine-2.png)

Read the scorecard like a dashboard idiot-light panel. Each row is a macro driver (a rate rise, a hot CPI print, a stronger dollar, wider credit spreads); each column is an asset. Green means the driver pushes the asset *up*, red means *down*, and the deeper the color, the stronger the relationship. Three things to notice on every weekly scan:

- **The reddest column tells you what is most exposed.** When rates and real yields are rising, the "US 10Y bond" column is deep red (−0.95, −0.70) — bonds are getting hammered by their most direct driver. When the dollar is strengthening, the "Gold" and "EM equity" cells go red. The reddest column under the *active* driver is where your portfolio is most at risk this week.
- **The +1.00 cell — the dollar against itself — is the anchor.** It is a tautology (a stronger dollar is perfectly correlated with itself), but it is on the scorecard deliberately, as a visual reminder that the dollar is the one driver that is *also* an asset, which is exactly why it dominates the exposure audit.
- **The sign of a cell is the regime's signature.** In the inflation regime, "hot CPI surprise" runs red across stocks, bonds, *and* EM simultaneously — that single red row *is* the stock-bond correlation flipping positive, because the same driver (rate fear) is hitting everything at once.

The scorecard is the snapshot. But a snapshot cannot show you a flip — for that you need the movie, which is component three. The most important single line from the scorecard, the stock-bond relationship, deserves its own time-series view because it is the one you watch most closely.

![Stock bond rolling correlation 1990 to 2025 the number one line to monitor](/imgs/blogs/building-a-macro-correlation-monitor-the-weekly-routine-3.png)

This is the line. For most of 1998–2021 it sat below zero — bonds hedged stocks, the green band, the 60/40 era. In 2022 it shot up to +0.60 — the red band, where both fall together. The whole point of watching this as a *movie* rather than a *number* is visible in the chart: the full-sample average of this line is roughly −0.1, a number that describes *no actual year*. Average a −0.40 era with a +0.60 era and you get a meaningless midpoint. The monitor never reads the average; it reads *where the line is now* and *which way it is moving*.

#### Worked example: the watchlist names your biggest live risk

Your \$100,000 book is the same 60/40: \$60,000 stocks, \$40,000 bonds. You do the watchlist scan. The stock-bond line is sitting at +0.30 and the scorecard's "real yield rise" row is deep red across both stocks and bonds. The watchlist is screaming that your two sleeves are now positively correlated — they will move together, not offset. Quantify the damage to the diversification you *thought* you had. A 60/40 portfolio's risk is not the weighted average of the two sleeves' risks; the cross term depends on the correlation. With stocks at ~16% annual volatility and bonds at ~10%, a −0.40 correlation gives a portfolio volatility near **9.5%**. Flip the correlation to +0.30 and the same two sleeves now carry a portfolio volatility near **12.0%** — your "balanced" book just got about 25% riskier without you trading a single share, purely because the correlation moved. The watchlist scan caught it. The intuition: diversification is not a property of *holding two things*; it is a property of the *correlation between them*, and the watchlist is how you check that the diversification you are paying for still exists.

### Scanning in lead-lag order: which correlation warns you first

There is a subtlety in *how* you read the watchlist that separates a competent scan from an expert one. The five correlations do not all move at the same time — some lead, some lag — so the order you read them in is not arbitrary. You want to scan the *leading* relationships first, because they warn you before the lagging ones confirm.

The empirical lead times, drawn from decades of business-cycle research, are striking. The yield curve (2s10s) leads recessions by roughly 14 months. ISM new orders lead S&P earnings growth by about 6 months. Building permits lead GDP by about 9 months. Credit spreads lead equity drawdowns by about 3 months. Initial jobless claims lead the unemployment rate by about 2 months. CPI and PCE are essentially coincident, with CPI a touch earlier; PPI leads core goods CPI by about a month. The full development of this is in [lead-lag: leading, coincident, and lagging indicators](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators).

What does that mean for the scan? Read the *leading* gauges first — the curve and ISM in the regime read, then credit spreads on the watchlist — because they are the smoke before the fire. By the time a *coincident* correlation (stocks-and-the-economy moving together right now) confirms the regime, you have lost the lead time the leading gauges already gave you. A practical example: in late 2007, credit spreads were widening — the leading canary — months before equity prices rolled over. A monitor that scanned credit spreads first (the 3-month leader) flagged risk a quarter before a monitor that waited for the coincident equity-economy correlation to turn. The discipline is to weight the *leaders* in your scan and treat the laggards as confirmation, not as the trigger.

This also reframes the flip check. A flip in a *leading* correlation (credit spreads versus equities widening, the curve inverting) is a forecast; a flip in a *coincident* correlation (the stock-bond relationship itself) is happening *now*. Both matter, but you act on the leaders with more lead time and the coincident flips with more urgency. The next component formalizes how to catch either one early.

## Component three: the flip check

A level tells you where you are. A *flip* tells you the world is changing — and a correlation that is about to change sign is the single highest-value thing the monitor can find, because it means a relationship you have been relying on is about to reverse. The flip check is where the monitor earns its keep.

A flip has two warning signs, and you check both:

1. **The level is near zero.** A correlation at +0.50 or −0.50 is in no danger of flipping next week; one hovering at +0.05 or −0.10 is on the knife's edge. Scan the watchlist for any correlation sitting close to zero — those are the candidates.
2. **The short window disagrees with the long window.** This is the early-warning system. Recall from the foundations that a short window (13 weeks) reacts fast and a long window (104 weeks) lags. When the short window has already crossed a threshold the long window has not, the short window is *forecasting* a flip the long window will confirm later. That disagreement is your lead time. The formal machinery for detecting this — including statistical change-point tests — is in [rolling-correlation regimes and change-point detection](/blog/trading/macro-correlations/rolling-correlation-regimes-and-change-point-detection).

Why does a flip matter so much more than a level? Because your positioning is built on the *sign*. A diversifier with correlation −0.40 and one with −0.20 are both diversifiers — the difference is one of degree. But −0.10 versus +0.10 is a difference of *kind*: one hedges you, the other doubles you up. Catching the moment a key correlation crosses zero is worth more than any refinement of a correlation that is comfortably signed.

The 2022 stock-bond flip, dissected in [when correlations break: the 2022 stock-bond flip](/blog/trading/macro-correlations/when-correlations-break-the-2022-stock-bond-flip), is the canonical case. The 13-week window crossed into positive territory in the first quarter of 2022; the 104-week window did not confirm until late in the year. An investor watching only the smooth long window saw nothing wrong until the damage was done. An investor running the flip check — short window crossed, long window has not, the level is near zero, the regime read says inflation is now the dominant risk — had a quarter of warning to cut bond duration or add an explicit hedge.

#### Worked example: spotting a flip early and the dollars it saves

It is early 2022. Your \$100,000 is in the 60/40 split. The flip check fires: the 13-week stock-bond correlation has crossed from −0.20 to +0.15, the 104-week is still −0.35, the regime read shows core CPI accelerating past 6% and the Fed signaling hikes. Three of three flip conditions are lit. You act: you cut your long-Treasury sleeve from \$40,000 to \$20,000 and park the \$20,000 in T-bills (cash), which actually diversify in an inflation shock because their price barely moves. Now run the year forward. In 2022 long Treasuries fell about 25%. Had you held the full \$40,000, that sleeve loses about **\$10,000**. By cutting to \$20,000 you lose about \$5,000 there, and the \$20,000 in bills is roughly flat — total bond-side loss about **\$5,000**, versus \$10,000. The flip check, caught a quarter early, saved you roughly **\$5,000** on a \$100,000 book — about 5% of the entire portfolio — and that is *before* counting that the cash gave you dry powder to buy stocks lower. The intuition: you do not get paid for knowing a correlation is −0.40; you get paid for being the one who notices, early, that it is no longer −0.40.

## Component four: the exposure audit

This is the component nobody runs, the one that cost my friend his September, and — done weekly — the one with the highest payoff-to-effort ratio in the entire routine. The first three components are about the *market's* correlations. The exposure audit is about *yours*: it takes your actual positions and asks whether they are as independent as you think.

The setup is always the same. You have several positions. Each has its own thesis. On the surface they look diversified — different asset classes, different stories, different lines on the screen. But if they all load on the *same underlying macro driver*, they are not several bets; they are one bet, sized up. And the most common shared driver, by a wide margin, is the dollar.

Look back at the watchlist scorecard, then at this concentration view, which lays out how a stronger dollar correlates with each asset:

![Asset correlation with a stronger US dollar gold EM oil copper bitcoin concentration audit](/imgs/blogs/building-a-macro-correlation-monitor-the-weekly-routine-6.png)

Gold, EM equities, oil, and copper all sit at −0.45 to −0.55 against the dollar. They are not four independent things; they are four expressions of one view — that the dollar will weaken. Buy all four and you have not diversified, you have *concentrated*, you have put on a single short-dollar trade four times and told yourself it was four ideas. [The dollar: cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity) develops why the dollar pulls on everything; the audit's job is to translate that gravity into a hard look at your own book.

The audit is a three-step calculation you can do on the back of an envelope:

1. **List your positions and their dominant driver.** For each, ask: what is the one macro variable this position is really a bet on? Long gold → short dollar / falling real yields. Long EM → short dollar / risk-on. Short DXY → short dollar. Long Nasdaq → falling real yields / liquidity.
2. **Group by shared driver.** How much of your book, in dollars, loads the same way on the same variable? If \$30,000 of gold, \$30,000 of EM, and \$30,000 short-dollar all load short-the-dollar, that is \$90,000 of a \$100,000 book on one variable.
3. **Net it and resize.** That \$90,000 is your *real* position size on the dollar. If you would never knowingly put 90% of your book on a single macro call, you must cut legs or hedge the driver directly — even though your risk report, which sees three different tickers, says you are fine.

The before-and-after of that audit is the figure below: three positions that *look* independent, revealed as one concentrated dollar trade.

![Exposure audit three independent bets revealed as one concentrated dollar trade](/imgs/blogs/building-a-macro-correlation-monitor-the-weekly-routine-7.png)

The deep lesson here connects to the most uncomfortable fact in all of portfolio construction: **correlations rise toward one exactly when you need diversification most.** In a calm market the average pairwise correlation across risk assets is about 0.25 — genuine diversification. In a normal selloff it climbs to about 0.45. In a full crisis, with forced deleveraging, it spikes to about 0.80 — everything sells together. [Correlation during crises: when diversification fails](/blog/trading/macro-correlations/correlation-during-crises-when-diversification-fails) and [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) develop this. The exposure audit is your defense: if you have *already* netted your book down to its true number of independent bets in the calm, you will not be ambushed when the crisis collapses the rest of your "diversification" into a single trade.

#### Worked example: three bets that are really one trade

Your \$100,000 book holds: \$30,000 long gold ("inflation hedge"), \$30,000 long EM equities ("growth bet"), \$30,000 short the dollar ("macro view"), and \$10,000 in cash. Your risk report shows three positions in three asset classes and reports a comfortable, diversified risk number. Run the audit. From the concentration chart, gold correlates −0.55 with the dollar and EM −0.55 with the dollar; the short-dollar position is, definitionally, −1.0 with a stronger dollar. Convert each to its dollar-exposure equivalent: the \$30,000 gold position behaves like roughly \$16,500 of short-dollar exposure (0.55 × \$30,000), the \$30,000 EM like another \$16,500, and the \$30,000 short-dollar is \$30,000 of it outright. Sum: about **\$63,000** of net short-dollar exposure — nearly two-thirds of your book riding on one macro variable. If the dollar instead *rises* 5% (as it did in 2022), a rough −0.55 beta on the two correlated legs plus the direct short means a combined loss in the ballpark of **\$3,500** across the three positions, all at once, with nothing offsetting because they are the *same trade*. The audit's fix: keep the highest-conviction expression (say, short-dollar directly), cut one of the correlated legs, and redeploy into something with a *different* driver — for instance long-duration Treasuries, which load on falling growth, not on the dollar. The intuition: the number of lines on your screen is not the number of bets in your book; the audit is how you count the bets that actually matter.

## Component five: the one-line thesis

The final step is the cheapest and the most clarifying. You compress the entire week's monitor into a single sentence. The format is fixed:

> *"We are in [regime]; the live risk is [the correlation that just moved or is near flipping]; the one trade is [size up / cut / hedge X]."*

For example: *"We are late-cycle drifting toward stagflation; the live risk is the stock-bond correlation crossing positive, so bonds no longer hedge; the one trade is to cut Treasury duration and add cash as the real diversifier."*

The discipline of the sentence is the point. If you cannot write it, one of three things is true and all of them are useful to discover: you have not actually identified the regime; you have not found the live correlation risk; or your book is so tangled that there is no single trade to name. Each of those is a signal to go back and finish the monitor. The one-liner is the forcing function that turns four screens of numbers into a decision. [From correlation to signal: building a macro overlay](/blog/trading/macro-correlations/from-correlation-to-signal-building-a-macro-overlay) shows how this one-liner becomes a sized, rules-based overlay; and [building a macro thesis: from data to a trade](/blog/trading/macro-trading/building-a-macro-thesis-from-data-to-a-trade) is the full mechanism behind turning a macro read into a position.

There is a second, quieter benefit to writing the sentence: it is a *commitment device*. A vague mental impression that "things feel risky" cannot be checked against reality next week, but a written claim — "stagflation, stock-bond flipping positive, cut Treasury duration" — can. When you re-read it the following Monday, you find out whether your read was right, and over a few months your log of one-liners becomes the only honest scorecard of your own macro judgment. Most investors never keep one, which is precisely why most investors never improve: they cannot remember what they believed before the outcome was known, so they cannot learn from being wrong. The one-line thesis, logged weekly, is the cheapest self-improvement tool in the discipline. It costs two minutes and it is the difference between ten years of experience and one year of experience repeated ten times.

#### Worked example: sizing to the regime from the one-liner

Your monitor produces: *"Goldilocks — growth firming (ISM 54 and rising), inflation cooling (core CPI decelerating three prints), curve steepening; the stock-bond correlation is comfortably negative at −0.35, so bonds hedge again; the one trade is to add equity risk."* The regime read has *earned* you the right to lean into stocks, because in Goldilocks the stock-bond correlation is −0.40 and your bonds will cushion any wobble. So on your \$100,000 book you shift from a defensive 50/50 (\$50,000 stocks, \$50,000 bonds) to a 70/30 (\$70,000 stocks, \$30,000 bonds), adding \$20,000 of equity exposure. The correlation math justifies the risk: with the stock-bond correlation at −0.35, the extra \$20,000 in stocks raises portfolio volatility only modestly (from roughly 8% to roughly 10%) because the bond sleeve still offsets equity wobbles — the negative correlation is *buying down* the volatility cost of the equity add. Had the regime been stagflation (correlation +0.55), that same \$20,000 equity add would have pushed volatility past 13% with no offset, and the one-liner would have read "cut, don't add." The intuition: the regime does not just tell you *which way* to trade — it tells you *how much you can afford to*, because the correlation sets the price of taking the risk.

## The weekly cadence: a checklist you can copy

The monitor is only valuable as a habit. Here is the cadence, in the exact order to run it, designed to take about twenty minutes on a Monday morning. Copy it.

1. **Monday data refresh.** Pull (or re-pull) your series and prices through Friday's close. If you run the [Python dashboard](/blog/trading/macro-correlations/building-a-macro-asset-correlation-dashboard-in-python), this is one `run`; if you eyeball charts, open the same set you opened last week. Sanity-check that nothing is stale or silently missing — a dropped series quietly poisons every correlation downstream.

2. **Regime read (5 minutes).** Note ISM (above/below 50, rising/falling), the core CPI trend direction, the 2s10s slope (positive/inverted), and the dollar's direction. Write the quadrant label. Mark which of the four quadrants you are in or drifting toward.

3. **Watchlist scan (5 minutes).** Read the scorecard and the five key rolling lines, top to bottom: stock-bond, gold-real yield, BTC-Nasdaq, credit-equity, dollar-EM. For each, note the current level and whether it moved meaningfully versus last week. The "versus last week" comparison is why you keep a log.

4. **Flip check (3 minutes).** Flag any watchlist correlation that is (a) near zero in level and (b) showing the short window crossing a threshold the long window has not. That is your flip candidate and your highest-priority finding.

5. **Exposure audit (5 minutes).** List your positions, tag each with its dominant macro driver, group by shared driver, and net out your true exposure to each. Flag any single driver carrying more than your risk tolerance allows (a common personal rule: no single macro variable above ~40% of the book).

6. **One-line thesis (2 minutes).** Write the sentence: regime, live risk, one trade. If you cannot, you are not done — go back to whichever step is unclear.

A few cadence rules learned the hard way:

- **Keep a log.** A one-paragraph note each week — the regime label, the five levels, the audit's net exposures, the one-liner. The value of the monitor is in the *deltas*, and you cannot see a delta without last week's number written down. The log is also your honest backtest: did the flips you flagged actually happen? Calibrate yourself against your own record. The discipline of not fooling yourself with these numbers is the whole subject of [backtesting a correlation without fooling yourself](/blog/trading/macro-correlations/backtesting-a-correlation-without-fooling-yourself).
- **Same order, every week.** The fixed order is not bureaucracy; it is what stops you from cherry-picking the correlation that confirms the trade you already wanted to make. Regime first, *then* watchlist — never the reverse.
- **Tie it to the calendar.** The releases that move these correlations cluster around CPI, NFP, FOMC, and the PMIs. Run the monitor *after* the week's key prints, not before — a regime read taken the day before CPI is a guess. [The macro calendar: CPI, NFP, FOMC, PMI](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi) lays out the schedule to anchor your weekly timing to.
- **Resist the urge to add instruments.** The temptation, after a few months, is to expand the watchlist from five lines to fifteen — every interesting correlation you stumble on wants a seat. Don't. A monitor you actually run with five lines beats a comprehensive one you skip because it takes an hour. The five core relationships — stock-bond, gold-real yield, BTC-Nasdaq, credit-equity, dollar-EM — cover the vast majority of the macro risk in a typical book. Add a sixth line only when you hold an asset whose dominant driver is not already on the list, and retire any line you have not learned something from in two months. The whole value of the routine is that it is short enough to be a habit; a watchlist that grows without bound quietly destroys the very discipline it was meant to enforce.

## Common misconceptions

The monitor exists partly to inoculate you against a handful of seductive, expensive errors. Each one is a place where a static view of correlation, or a memorized rule of thumb, leads you wrong.

**Misconception 1: "Gold hedges inflation, so I'll hold it when CPI is high."** No — gold tracks *real yields*, not inflation. Over 2007–2021 the correlation between the 10-year real yield and gold was about −0.82: gold rises when real yields fall, regardless of what headline inflation is doing. High inflation with *rising* real yields (because the Fed hikes faster than inflation) is actually bad for gold, which is exactly what happened in 2022's early innings. And then the relationship *broke*: over 2022–2025 the correlation flipped to roughly +0.6 as central-bank buying drove gold higher even with high real yields. So the rule "gold hedges inflation" is wrong twice over — wrong about the driver (it's real yields, not CPI) and wrong to assume any correlation is permanent. The monitor catches both because it watches the gold-real-yield line directly. See [inflation and gold: the real-yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story).

**Misconception 2: "Bonds always diversify stocks."** Only in some regimes. As the quadrant chart showed, bonds hedge stocks (correlation −0.40 to −0.55) in Goldilocks and deflation, but they *amplify* stock losses (correlation +0.55) in stagflation. The 2022 lesson cost the average 60/40 investor double-digit drawdowns precisely because they treated "bonds diversify" as a law of nature rather than a regime-dependent fact. The monitor's number-one watchlist line exists to keep this from happening to you.

**Misconception 3: "More positions means more diversification."** Not if they share a driver. As the exposure audit showed, four positions all loading on the dollar are one bet, not four. Diversification is about the *number of independent drivers*, not the number of tickers. You can hold twenty positions and have, in truth, two bets. The audit is the only step that catches this, which is why it is the step everyone skips and the step that does the most damage when skipped.

**Misconception 4: "The full-sample correlation is the real one."** The full-sample number is an *average across regimes*, and an average across a −0.40 era and a +0.60 era is roughly −0.1 — a number that describes no actual period and would mislead you in every one. The monitor never reads the average; it reads the current rolling level and its trend. [Rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters) develops why the full-sample number is the most dangerous number on the page.

**Misconception 5: "A correlation I can see is a correlation I can trade."** Two warnings. First, a correlation off a short window is statistically noisy — a 26-week estimate has a standard error near 0.20, so a "0.3 correlation" is barely distinguishable from zero. Second, many striking correlations are *spurious* — two trending series will correlate near +0.9 even when their week-to-week moves are unrelated, which is why the dashboard correlates *changes*, never levels. [Spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data) and [common correlation mistakes and how to avoid them](/blog/trading/macro-correlations/common-correlation-mistakes-and-how-to-avoid-them) catalog the full set of traps the monitor is designed to route around.

## How it shows up in real markets

The monitor is not a theoretical exercise. Here are three dated episodes where running it — or failing to — was the difference between a controlled outcome and a blindside.

**2022: the stock-bond flip.** The defining macro event of the decade for portfolio construction. An investor running the flip check saw the 13-week stock-bond correlation cross positive in Q1 2022 while the 104-week stayed negative, with a regime read screaming inflation (core PCE peaked at 5.6% in February 2022). The signal: cut bond duration, the hedge is breaking. An investor reading only the smooth long-window number, or the full-sample average, saw nothing until the S&P had fallen ~18% and long Treasuries ~25% *in the same year* — the worst year for the balanced portfolio in living memory. The monitor's value was entirely in the *order*: regime first told you to distrust the comforting long-window correlation.

**2020: COVID and the everything-correlation.** In March 2020 the average pairwise correlation across risk assets spiked toward 0.85 — gold, stocks, even Treasuries sold off together in the dash for cash before the Fed intervened. An investor who had run the exposure audit in the calm of January knew that their "diversified" book would collapse to a single risk-asset bet in a panic, and had either held genuine dry powder (cash, which actually held up) or trimmed the correlated legs in advance. The monitor does not predict the crash; it makes sure you are not surprised by the *collapse of diversification* that always accompanies one.

**2023–2024: the gold decoupling.** For fifteen years the gold-real-yield correlation was a reliable −0.8. Then in 2023–2024 it broke: real yields rose to nearly 2% (which historically would crush gold) yet gold rallied from ~\$1,940 to ~\$2,390, driven by central-bank buying. An investor watching the gold-real-yield line on the watchlist saw the relationship weakening in real time — the short window decoupling from the long — and learned the regime-shift lesson without paying for it: a correlation that has held for a decade is still not a law, and the monitor is how you notice the day it stops being true. [Structural shifts: why today's correlations aren't yesterday's](/blog/trading/macro-correlations/structural-shifts-why-todays-correlations-arent-yesterdays) develops this case at length.

**The monitor outside the US: a Vietnam example.** The same routine works on any market once you swap in the right watchlist. For an investor holding Vietnamese equities, the regime read still leads — but the dominant correlations are different. The VN-Index correlates about +0.45 with the S&P 500 and +0.55 with broad emerging-market equities, so a US risk-off event imports directly. It correlates about −0.40 with the dollar and about −0.25 with the US 10-year yield — a stronger dollar and higher US yields pull foreign money out of Vietnam, and the foreign-net-buy flow on the Ho Chi Minh exchange correlates about +0.50 with the index, amplifying both directions. On the domestic side, the VN-Index runs about −0.45 against the State Bank of Vietnam's policy-rate changes and about −0.30 against the local CPI trend. The watchlist for a VN book is therefore: US risk (S&P/EM beta), the dollar, US yields, foreign flows, and the SBV rate path. Crucially, the *exposure audit still applies*: a VN equity book plus a long-EM ETF plus a short-dollar position is, once again, three expressions of one short-dollar / risk-on bet, because each correlates the same way with the dollar. The monitor is portable; only the names on the watchlist change.

The common thread across all four: the people who came out fine were not smarter forecasters. They simply *kept looking*, in a fixed order, every week, and noticed the change a beat before the people who had written a correlation into a spreadsheet and stopped checking.

## How to read it and use it

Here is the whole monitor compressed into a usable mental flowchart, the way you should carry it in your head.

**The signal.** Every week, in order: name the regime (growth × inflation, curve, dollar). Scan the five watchlist correlations off the scorecard. Flag any near-zero correlation where the short window is crossing ahead of the long window — that is your flip warning. Audit your own book for hidden concentration on a single driver. Write the one-line thesis.

**The regime check.** Before you act on any correlation, confirm the regime supports it. A negative stock-bond correlation is a *Goldilocks/deflation* fact, not a universal one — do not lean on bonds as a hedge in a stagflation read. A gold-real-yield correlation is a *normal-regime* fact — do not assume it survives a central-bank-buying episode. The regime is the validity condition stamped on every correlation; check it first.

**What invalidates the signal.** Three things should make you distrust a reading and go back. (1) The correlation is computed off a short window with few observations — treat any short-window correlation as a rumor until the long window confirms it. (2) The two series are correlated in *levels* rather than *changes* — that is a trend artifact, not a relationship. (3) The regime read and the correlation disagree — if the regime says stagflation but a correlation says bonds are still hedging, the correlation is probably a lagging long-window number that has not caught up, and you trust the regime. When in doubt, the order of authority is: regime first, then the long-window correlation, then the short window as a forward warning, never the full-sample average.

**The one rule above all.** Identify the regime, then trust only the correlations that regime activates. Every component of the monitor — the regime read at the front, the scorecard's regime-dependent signs, the flip check's regime context, the exposure audit's shared-driver logic — is in service of that one rule. Run it every Monday, keep the log, and you will be the investor who saw the flip coming, not the one explaining the drawdown afterward.

The full series ties together in [the macro-correlation playbook](/blog/trading/macro-correlations/the-macro-correlation-playbook-capstone), which assembles every relationship in this monitor into one reference. This post is the *habit* that keeps that playbook alive — because a playbook you consult once is a book on a shelf, and a playbook you run every week is an edge.

## Further reading and cross-links

**The instrument and the playbook (this series):**
- [Building a macro-asset correlation dashboard in Python](/blog/trading/macro-correlations/building-a-macro-asset-correlation-dashboard-in-python) — the code that produces the heatmap and rolling lines this monitor reads.
- [The macro-correlation playbook](/blog/trading/macro-correlations/the-macro-correlation-playbook-capstone) — the full reference this weekly routine keeps alive.
- [The macro-asset correlation matrix](/blog/trading/macro-correlations/the-macro-asset-correlation-matrix) — the master driver × asset map behind the scorecard.
- [Correlation by regime: the four macro quadrants](/blog/trading/macro-correlations/correlation-by-regime-the-four-macro-quadrants) — the growth × inflation map at the front of the monitor.
- [Rolling-correlation regimes and change-point detection](/blog/trading/macro-correlations/rolling-correlation-regimes-and-change-point-detection) — the statistics behind the flip check.
- [Common correlation mistakes and how to avoid them](/blog/trading/macro-correlations/common-correlation-mistakes-and-how-to-avoid-them) — the traps the monitor routes around.

**The lines on the watchlist (this series):**
- [The stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) and [when correlations break: the 2022 stock-bond flip](/blog/trading/macro-correlations/when-correlations-break-the-2022-stock-bond-flip) — the number-one line.
- [Real yields and the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation) and [inflation and gold: the real-yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story).
- [The dollar (DXY) cross-asset correlation](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation) — the driver behind the exposure audit.
- [Credit spreads: the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary) and [crypto as a macro asset](/blog/trading/macro-correlations/crypto-as-a-macro-asset-the-liquidity-correlation).
- [Correlation during crises: when diversification fails](/blog/trading/macro-correlations/correlation-during-crises-when-diversification-fails).

**The mechanism and the calendar (other series):**
- [Building a macro thesis: from data to a trade](/blog/trading/macro-trading/building-a-macro-thesis-from-data-to-a-trade) — turning the one-line thesis into a position.
- [The macro calendar: CPI, NFP, FOMC, PMI](/blog/trading/macro-trading/the-macro-calendar-cpi-nfp-fomc-pmi) — anchoring the weekly timing to the releases that move these correlations.
- [How policy moves every asset: the cross-asset transmission map](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map) — why the correlations exist in the first place.

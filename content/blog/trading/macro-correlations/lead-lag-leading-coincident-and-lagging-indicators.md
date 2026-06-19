---
title: "Lead, Lag, or Coincident: The Time Axis of Every Correlation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Every correlation has a clock. This post builds the cross-correlation function from zero, sorts indicators into leading, coincident, and lagging, and shows why you trade the leader, not the lagger."
tags: ["macro", "correlation", "lead-lag", "cross-correlation", "leading-indicators", "yield-curve", "ism", "business-cycle", "granger-causality", "indicators"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — A correlation is incomplete without its lead/lag: some indicators turn *before* the market (yield curve, ISM new orders, building permits, credit spreads, claims), some turn *with* it (payrolls, CPI, industrial production), and some only turn *after* (unemployment rate, GDP, earnings). The cross-correlation function finds the lag at which two series line up best, and that lag is where the money is.
>
> - **The number to remember:** the 2s10s yield curve has historically inverted roughly **12–18 months** (peak cross-correlation near **14 months**) before the recession it warns about. By the time the lagging confirmation (rising unemployment) arrives, the leading signal has already paid you.
> - **You trade the leader, not the lagger.** ISM new orders lead reported S&P earnings by about 6 months; you position in cyclicals on the survey, not on the earnings beat.
> - **The cross-correlation function** is just "slide one series past the other and re-measure correlation at each shift." The shift with the highest correlation is the lead.
> - **Be honest:** the lead time is itself regime-dependent and noisy, and a high cross-correlation at some lag is *not* proof of causation. Granger-causality only says "X helps predict Y," which is a much weaker claim than "X causes Y."

In late 2022 a strange thing happened to anyone watching the economic data. The headline numbers looked *fine*. Nonfarm payrolls were printing 250,000-plus jobs a month. The unemployment rate sat at 3.5%, a half-century low. Consumer spending was solid. By every coincident measure, the US economy was humming. And yet the bond market was screaming that a recession was on the way: the 2s10s yield curve — the gap between the 10-year and 2-year Treasury yield — had inverted in July 2022 and kept getting *more* inverted, eventually reaching −1.08 percentage points in July 2023, the deepest since 1981.

Two groups of people were looking at the same economy and seeing opposite things. The first group read the coincident data — jobs, spending, production — and concluded all was well. The second group read a *leading* indicator — the shape of the yield curve — and concluded a slowdown was coming. The disagreement was not about the facts. It was about *which clock* each group was reading. Coincident data tells you where the economy is right now. Leading data tells you where it is going. They almost never agree at a turning point, because the turning point is exactly when the future stops resembling the present.

This is the single most under-appreciated property of a correlation. We spend enormous energy arguing about whether two things are correlated, and how strongly, and we almost never ask the question that actually decides whether the correlation is *tradeable*: **when?** Does the indicator move before the asset, with it, or after it? A correlation with no lead is a coincidence you can admire but not profit from. A correlation with a *positive* lead — where the indicator turns first — is a clock you can set your portfolio by. This post is about that time axis, and about the tool macroeconomists use to measure it: the cross-correlation function.

![Leading coincident and lagging indicators arranged in time around the market](/imgs/blogs/lead-lag-leading-coincident-and-lagging-indicators-1.png)

If you have not yet read [what a correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta), start there: it covers Pearson's r, Spearman's rank correlation, beta, and what a correlation of 0.6 really means. This post assumes you know that a correlation has a *sign* and a *strength*, and adds the third coordinate — the lead/lag — that turns a static number into a moving signal.

## Foundations: what "X leads Y" actually means

Let us build the idea from absolutely nothing, because the everyday intuition is so strong that it usually gets in the way.

Suppose you measure two things every month: the number of building permits issued (a permit is the legal go-ahead to start construction), and the level of GDP (the total output of the economy). You plot them on top of each other over twenty years. The two wiggly lines clearly move together — both rise in good times, both fall in bad times. A naive correlation would say "permits and GDP are positively correlated, r ≈ 0.7" and stop there.

But look more carefully at the *timing*. Every time permits turn up, GDP turns up too — about nine months later. Every time permits roll over, GDP rolls over too — again, about nine months later. The two series have the same *shape*, but one of them is shifted in time relative to the other. Permits are GDP's shape, slid nine months into the future. That shift is the lead.

Here is the everyday version. Think of a freight train and the sound of its whistle. If you stand near the tracks, you see the train and hear the whistle at almost the same instant — they are *coincident*. But if you stand a mile down the line, you hear the whistle first and the train arrives later. The whistle *leads* the train. Nothing about the relationship between whistle and train changed; what changed is where you are standing, which sets the lag. In markets, "where you stand" is which variable you choose to watch. Building permits are the whistle; GDP is the train; the lead is how far down the track you are.

### Why anything leads anything

Leads are not magic. They exist for a concrete, mechanical reason: **economic activity is a chain of steps, and each step happens before the next.** A company decides to build a factory (that is a permit). Months later it pours concrete (that is construction spending). Months after that the factory opens and starts producing (that is GDP). The decision *causes* the output, and the decision necessarily comes first. So if you watch the decision, you are watching the future of the output.

The same logic explains every leading indicator:

- **Yield curve → recession.** The bond market is a giant prediction machine. When investors expect the central bank to cut rates in the future (which it does in recessions), they buy long-term bonds now, pushing long yields *below* short yields — the curve inverts. The inversion is the market's *forecast* of a slowdown, made well before the slowdown is visible in the hard data. [Reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) covers the mechanism in depth.
- **ISM new orders → earnings.** A factory's new orders today become its production next quarter and its booked revenue the quarter after. Earnings sit at the *end* of that chain, so the order survey leads the earnings print.
- **Building permits → GDP.** The permit is the first observable step in a construction project that will add to output much later.
- **Initial jobless claims → unemployment rate.** A layoff shows up as a new claim for unemployment benefits within a week. It only shows up in the *unemployment rate* after enough layoffs accumulate to move a stock variable. The flow (claims) leads the stock (the rate).

In every case the indicator captures an *early* link in a causal chain, and the asset or the headline number captures a *late* link. The lead is just the length of the chain.

This reframing — *a lead is the length of a causal chain* — is the single most useful idea in this post, because it tells you how to *find* leads you have not been told about. Whenever you want to know whether X might lead Y, ask: is there a chain of cause and effect that runs from X through several steps to Y? If yes, X probably leads Y by roughly the time it takes the chain to play out, and that chain also tells you *why the lead might change*. Whenever you cannot draw such a chain — when X and Y just happened to move in sequence over your sample — you should distrust the lead no matter how clean the cross-correlation function looks, because there is nothing to make it persist. A lead with a mechanism is a tool; a lead without one is a trap. We will lean on this distinction repeatedly: it is what separates a robust leading indicator from a statistical coincidence dressed up as a forecast.

### Leading, coincident, lagging — the three classes

Economists formalize this into three buckets, made famous by the Conference Board's Leading Economic Index (LEI), a composite of ten indicators chosen specifically because they turn *before* the cycle does.

- **Leading indicators** turn before the cycle turns. They are forecasts. Examples: the yield-curve slope, ISM new orders, building permits, the average workweek, stock prices themselves, credit spreads, and initial jobless claims. You read these to *position*.
- **Coincident indicators** turn at roughly the same time as the cycle. They define "now." Examples: nonfarm payrolls, industrial production, real personal income, real manufacturing-and-trade sales. The National Bureau of Economic Research (NBER), which officially dates US recessions, leans heavily on coincident data. You read these to *mark where you are*.
- **Lagging indicators** turn after the cycle turns. They are confirmations. Examples: the unemployment rate, the average duration of unemployment, the prime rate, unit labor costs, corporate earnings, and the final GDP print. You read these to *confirm* a move that has already happened.

The figure above arranges all three around the market in the middle. Notice the asymmetry of *use*: leaders are for action, laggards are for confirmation, and confirmation is worth far less than action because by the time you have it, the price has already moved. A trader who waits for the unemployment rate to rise before selling stocks is selling into a market that bottomed months ago.

> [!note]
> A subtle but important point: **stock prices are themselves a leading indicator.** The S&P 500 is in the Conference Board's LEI precisely because the market prices the future, not the present. This is why "the market" sits in the *middle* of our figure rather than on the lagging side — relative to the real economy, the market leads; relative to the bond market's curve signal, the stock market can lag. Lead and lag are always *relative to a chosen reference*, never absolute.

## The cross-correlation function: how you actually measure a lead

So far "X leads Y by k months" has been a story. Now let us make it a number, because a number is what you trade on. The tool is the **cross-correlation function (CCF)**, and despite the intimidating name it is one of the most intuitive ideas in all of time-series analysis.

Recall ordinary correlation. You have two series, X and Y, both measured at the same times. You line up X(t) with Y(t), point for point, and compute Pearson's r. That gives you the correlation *at lag zero* — assuming the two move together at the same instant.

The cross-correlation function asks: what if they *don't* move at the same instant? What if X moves first? To check, you **slide** X relative to Y. Take X and shift it forward in time by one month, so that X from last month lines up with Y from this month. Compute the correlation of that shifted alignment. Then shift by two months and recompute. Then three, then four, and also shift the other way (X *after* Y) by one, two, three months. You end up with a correlation value for *every* possible lag, k = …, −3, −2, −1, 0, +1, +2, +3, …. Plotting correlation against k gives you the cross-correlation function.

![Cross correlation function shift one series past the other to find the peak](/imgs/blogs/lead-lag-leading-coincident-and-lagging-indicators-3.png)

The convention (which trips up everyone at first) is this: if the **peak** of the cross-correlation function sits at a **positive lag k**, then **X leads Y by k periods.** The intuition: you had to shift X *forward* by k to make it line up with Y, which means X's pattern naturally arrives k periods *earlier* than Y's. Positive lag = X is early = X leads.

Look at the figure. When we shift X to the wrong place (k = −6, X six months too early), the curves barely overlap and the correlation is a weak 0.15. When we shift X to k = +6, the curves snap into alignment and the correlation jumps to 0.72 — that is the peak, and it tells us X leads Y by six months. Over-shift to k = +12 and the correlation falls off again to 0.30. The whole point of the exercise is to find that single lag where the alignment is best.

#### Worked example: finding the lead between two series

Suppose you have monthly data on ISM new orders (X) and S&P 500 trailing EPS growth (Y), and you compute the cross-correlation at several lags:

```
lag k (months)   corr(X shifted by k, Y)
   0                 0.31
  +2                 0.48
  +4                 0.64
  +6                 0.71   <- peak
  +8                 0.66
 +10                 0.52
```

The correlation is *highest* at k = +6, with r = 0.71. The interpretation: ISM new orders lead S&P EPS growth by about six months, and at that optimal lead the relationship is strong (r ≈ 0.71, so R² ≈ 0.50, meaning roughly half the variance in earnings growth is statistically associated with new orders six months earlier). The contemporaneous correlation (k = 0) is only 0.31 — if you had naively measured the relationship at lag zero you would have *understated* it by more than half. **The lead is not a footnote to the correlation; it is the difference between a weak relationship and a strong one.**

That last sentence is the whole reason the cross-correlation function matters. The same two series can look weakly correlated, moderately correlated, or strongly correlated depending entirely on whether you bother to align them in time. Most people who say "X and Y aren't really correlated" simply measured at the wrong lag.

### Computing it, carefully, without the heavy math

You do not need the formula to use the cross-correlation function, but you do need to respect a handful of practical rules, because the CCF is easy to fool. Here, in plain language, is what a careful practitioner does — and what a careless one gets wrong.

First, **work with changes, not levels.** Two series that both *trend* upward over decades (say, nominal GDP and the price level) will show a sky-high correlation at almost every lag, simply because both are going up. That correlation is spurious — it is the shared trend talking, not a genuine lead/lag relationship. The fix is to compute the CCF on *growth rates* or *differences* (month-over-month or year-over-year changes), which strips out the common trend and leaves the cyclical co-movement you actually care about. A cross-correlation function computed on raw, trending levels is the single most common rookie error in this whole area.

Second, **mind the sample size at each lag.** When you shift X forward by k months, you lose k months of overlap at the edges. Shift by twelve months on a twenty-year monthly sample and you are still fine, but on a short sample the long-lag correlations are computed on very few points and become unreliable. A "peak" at lag +18 estimated from thirty data points is noise wearing a costume.

Third, **the peak you find is an estimate, with error bars.** If you computed the CCF on a different decade, the peak might land a few months away. Honest practitioners do not report "the lead is 6 months"; they report "the peak is around 4–8 months, centered near 6." This is not hedging — it is an accurate representation of how much the data actually tells you.

Fourth, **a peak is not a guarantee of stability.** The CCF is computed over a *historical sample*, and it describes the *average* lead over that sample. If the relationship's lead drifts over time — which macro leads routinely do — the single peak smears out the variation into one number that may not describe any single episode well. We will see exactly this when we get to regime-dependence.

#### Worked example: changes versus levels

Imagine you compute corr(building permits, GDP) on twenty years of *raw levels* and get a stunning r = 0.95 at lag zero, with barely any peak when you slide the series. You conclude "permits and GDP are coincident and almost perfectly correlated." Both conclusions are wrong. The 0.95 is mostly the shared upward trend (both grew with the economy and population over twenty years), and it swamps the cyclical signal at every lag, which is why you saw no clear peak. Recompute on *year-over-year growth rates* and the level-correlation collapses to something modest at lag zero but a clear peak emerges at about +9 months — *now* you can see that permit growth leads GDP growth by three quarters. The lead was always there; the trend was hiding it. **Always difference before you cross-correlate.**

### The shape of the cross-correlation function tells you something too

A clean leading relationship produces a cross-correlation function with a single, sharp peak at a positive lag and lower values on both sides. But the shape carries extra information:

- **A broad, flat peak** means the lead is *imprecise* — the relationship is roughly as strong whether X leads by 4, 6, or 8 months. This is the common case in macro, and it is why honest practitioners quote a *window* ("a couple of quarters") rather than a single number.
- **Two peaks** suggest two different transmission channels operating at two different speeds, or a seasonal artifact you forgot to remove.
- **A peak at lag zero** is a coincident relationship — the two move together with no usable lead. CPI and PCE inflation are like this: they measure overlapping baskets, so they are nearly coincident (CPI a touch earlier because it is released first, but the *economic* lead is essentially zero).
- **A peak at a negative lag** means you had the leader and the lagger backwards. If you thought X led Y but the peak is at k = −3, then in fact Y leads X by three months.

## The lead-time map: who leads whom, by how much

Now we can put real numbers on the chain. The chart below is the centerpiece of this post: the approximate lead time at which each indicator's cross-correlation with its target peaks. These are researched approximations from business-cycle literature (the Conference Board's LEI methodology, NBER dating studies, and decades of practitioner research), rounded for teaching — they are the documented *regime*, not a tick-exact reproduction, and you should treat every one as "give or take a quarter."

![Horizontal bar chart of lead times in months for each indicator and its target](/imgs/blogs/lead-lag-leading-coincident-and-lagging-indicators-2.png)

Read it from the top:

- **Yield curve (2s10s) → recession: ~14 months.** The longest and most famous lead. The curve inverts well over a year before the recession it warns about. This is both a blessing and a curse: a long lead gives you time to position, but it also means the signal can fire and then *nothing happens for a year*, which is agonizing for anyone holding a position against the consensus.
- **Building permits → GDP: ~9 months.** Construction is slow; the lead is the length of a building project from approval to output.
- **ISM new orders → S&P EPS growth: ~6 months.** Roughly two quarters from the order book to the earnings line.
- **Credit spreads → equity drawdown: ~3 months.** Credit markets sniff out trouble before equity markets fully price it. Widening high-yield spreads have repeatedly preceded equity drawdowns by a quarter or so.
- **Initial claims → unemployment rate: ~2 months.** The flow (new layoffs) leads the stock (the accumulated jobless rate) by a couple of months.
- **PPI → core goods CPI: ~1 month.** Producer prices feed into consumer goods prices with a short lag as cost increases pass through the supply chain.
- **CPI → PCE: ~0 months (coincident).** Two measures of the same thing; no usable lead.

Notice how the leads *stack*: the yield curve (a financial-market forecast) leads everything, then come the survey-based and order-based indicators, then the price pass-throughs that are nearly coincident. The further "upstream" in the causal chain an indicator sits, the longer its lead — and, usually, the noisier it is. There is a fundamental trade-off here, which we will return to: **long leads are early but unreliable; short leads are reliable but late.**

#### Worked example: turning a lead into a position decision

Suppose it is month 0 and the 2s10s curve has just inverted to −0.20 pp. The historical lead to recession is about 14 months, with a wide band (say 6 to 24 months in past episodes). You run a portfolio and you want to know what to do.

A naive reader says "recession in 14 months, sell stocks." That is wrong, and dangerously so. The lead is a *window*, not a date. If you sell everything the moment the curve inverts, you are likely to sit in cash for a year or more while equities keep rising — the curve inverted in mid-2022 and the S&P went on to rally hard through 2023 and 2024. The correct use of a long-lead signal is **gradual and conditional:** you note that the *base rate* of a slowdown over the next 1–2 years has risen, you begin tilting *at the margin* toward defensives, and you arm a faster, shorter-lead trigger (credit spreads widening past, say, 5%, which leads equity drawdowns by only ~3 months) to tell you when the slow-burning risk is finally igniting. The long-lead indicator changes your *prior*; the short-lead indicator changes your *position*. Confusing the two is how traders go broke being right too early.

## Real markets: three leads you can watch

Concepts are cheap. Let us ground each part of the lead/lag spectrum in something you can actually pull up on a screen.

### The yield curve: a ~14-month lead with a brutal patience cost

The 2s10s spread is the textbook leading indicator, and the 2021–2026 cycle is a perfect case study in both its power and its pain.

![Two year ten year Treasury spread over time with the inverted region shaded](/imgs/blogs/lead-lag-leading-coincident-and-lagging-indicators-4.png)

The spread started 2021 at a healthy +1.20 pp. Through 2022 it collapsed as the Fed hiked aggressively, dragging the 2-year yield up faster than the 10-year. It crossed zero in mid-2022 and kept falling, reaching its deepest point of −1.08 pp in July 2023 — the most inverted the curve had been since 1981. It then slowly normalized, un-inverting back above zero in late 2024 and reaching +0.55 pp by 2026.

The leading-indicator story is in the *timing*. The inversion began in mid-2022, when coincident data (jobs, spending) was still strong. If the historical ~14-month lead held, that pointed to a slowdown sometime in 2023–2024. The signal was *early* — and being early is indistinguishable from being wrong, right up until it isn't. Anyone who shorted equities on the 2022 inversion got run over for more than a year.

This is the central honesty of leading indicators: **a long lead means a long period of looking foolish.** The shaded red region in the chart is the entire window during which a yield-curve bear was "right" but losing money. The lead is real and the signal has genuine predictive content, but its *length* makes it useless as a market-timing trigger on its own. You use it to set the prior, not to pull the trigger.

#### Worked example: the cost of acting on a long-lead signal too literally

Imagine two traders, both watching the July 2022 inversion. Trader A sells the S&P at 3,900 the day the curve inverts, convinced a recession is ~14 months out. Trader B notes the inversion, raises her *base-rate* estimate of a recession over the next two years, but keeps most of her equity exposure and waits for a *shorter-lead* confirmation.

Over the next 18 months the S&P does not crash; it rallies to roughly 4,800 by the end of 2023 and beyond. Trader A, sitting in cash, misses a gain of about (4,800 − 3,900) / 3,900 ≈ **+23%**, plus dividends. On a \$1,000,000 equity book, that is a forgone gain of roughly \$230,000 — real money, given up because a 14-month forecast was traded as if it were a tomorrow forecast. Trader B, who stayed mostly invested, captures most of that \$230,000. The yield curve's signal was *informationally* correct — recession risk genuinely rose — but Trader A converted a long-lead *forecast* into a short-lead *trade*, and the mismatch between the signal's horizon and his action's horizon cost him \$230,000 on a million-dollar book. The lesson: **match the action's horizon to the signal's horizon.** A 14-month signal is a 14-month signal, not a tomorrow signal.

### ISM new orders: a ~6-month lead into earnings

The Institute for Supply Management's manufacturing survey asks purchasing managers a simple question every month: are new orders rising or falling? The "new orders" sub-index is one of the cleanest leading indicators of the industrial cycle, and it flows downstream into corporate earnings with a lead of about six months. (For the survey's mechanics and its release-day market reaction, see [the ISM/PMI business surveys that lead](/blog/trading/event-trading/ism-pmi-the-business-surveys-that-lead).)

![Chain showing ISM new orders leading production revenue and reported earnings](/imgs/blogs/lead-lag-leading-coincident-and-lagging-indicators-5.png)

The lead is mechanical, not mystical. An order booked today becomes production in a month or two, shipments and recognized revenue a couple of months after that, and finally a line in the quarterly earnings report around six months out. Earnings are the *last* link in the chain, which is exactly why earnings are a lagging indicator while the order survey is a leading one. The same economic event — a wave of new demand — shows up in the survey first and in the headline EPS number last.

The trading implication follows directly: **you position in cyclicals on the survey, not on the earnings beat.** By the time a company reports a blowout quarter, the demand surge that produced it was visible in the ISM new-orders index two quarters earlier, and the stock has usually already moved. The reader who waits for the confirmation (earnings) is trading on a lagging indicator; the reader who acts on the survey (new orders crossing back above 50) is trading on the leading one.

#### Worked example: ISM new orders as an earnings-turn signal

Say ISM new orders has been below 50 (contraction) for several months and then prints 53.0, crossing decisively back above the expansion line. History suggests S&P EPS growth tends to inflect upward roughly two quarters later. With a contemporaneous-to-peak correlation profile that rises from r ≈ 0.31 at lag 0 to r ≈ 0.71 at lag +6, the survey is telling you something the current earnings data cannot: the *next* earnings season is more likely than not to surprise to the upside.

If you believe the lead, the trade is to overweight economically sensitive sectors (industrials, materials, semiconductors) *now*, accepting that the earnings confirmation is two quarters away and that the position will look premature until then. Concretely, on a \$500,000 portfolio you might shift, say, \$50,000 (10%) from defensives into cyclicals on the survey signal, two quarters ahead of the earnings beats that the survey is forecasting. If new orders had instead rolled from 53 to 47, the same logic in reverse would warn of an earnings *slowdown* coming in ~6 months, and you would move that \$50,000 the other way — trimming cyclicals before the bad quarters print. The single number — the lead — converts a survey into a portfolio tilt.

### Initial claims: a ~2-month lead into the unemployment rate

The shortest, most reliable lead in our map is the one from initial jobless claims to the unemployment rate. When companies lay people off, those workers file for unemployment benefits within a week, so claims are a near-real-time *flow* of job losses. The unemployment rate is a *stock* — the share of the labor force that is jobless — and a stock only moves once enough flow accumulates. So claims lead the rate by about two months.

This pairing is the cleanest illustration of why a flow leads a stock. The flow is the *change*; the stock is the *level*; and a change always shows up before its effect on the level. It is the same reason your bathtub's water level (stock) lags the faucet (flow): turn the tap and the level only starts visibly rising a moment later. Claims are the faucet; the unemployment rate is the bathtub.

The practical payoff: a sustained rise in initial claims is one of the earliest hard-data confirmations that the labor market — a key input to Fed policy and to consumer spending — is genuinely deteriorating. It leads the unemployment rate (a lagging indicator) and complements the much-longer-lead yield-curve signal. When the long-lead financial signal (inverted curve) is *finally* corroborated by a short-lead hard-data signal (rising claims), the slow-burning risk is igniting, and the two horizons line up.

There is one more reason claims are the most *trustworthy* of the leads even though they have the shortest horizon, and it is worth stating explicitly because it generalizes. Claims sit very *close* to the thing they predict — the labor market — with a short, well-understood mechanism (layoff → claim → eventually the rate). Short, mechanical leads are reliable. The yield curve, by contrast, sits *far* upstream (it is a market's forecast of a forecast of policy), so its lead is long but its reliability is lower; it can fire and fizzle. This is the general law of the lead/lag spectrum: **the further upstream you read, the longer your warning and the lower your hit rate.** Claims give you a short, dependable warning; the curve gives you a long, unreliable one. A complete process uses both, weighting the long-lead signal for the prior and the short-lead signal for the trigger, and never confusing the dependable-but-late with the early-but-flaky.

#### Worked example: layering claims under the curve

Suppose the curve inverted 14 months ago (your long-lead prior: recession risk is elevated) and initial claims have just risen for six straight weeks from a four-week average of 215,000 to 275,000 — a clear, sustained uptrend rather than a one-week blip. Individually, neither is a precise market-timing trigger; together they are powerful. The curve told you *months ago* that the structural risk was building; the claims uptrend now tells you the labor market is *actually cracking*, with a short ~2-month lead into the unemployment rate. The two horizons have finally aligned: the long-lead forecast is being corroborated by short-lead hard data. *This* is the moment to convert the elevated prior into an actual defensive tilt — not when the curve first inverted (too early, ~14 months of patience cost), and not when the unemployment rate finally rises (too late, the market has bottomed). The edge is in waiting for the short-lead confirmation of the long-lead warning.

### Credit spreads: a ~3-month lead into equity drawdowns

Between the long-lead yield curve and the short-lead claims sits credit. The corporate-bond market — especially the high-yield ("junk") segment — is famous for sniffing out trouble before the equity market fully prices it. The mechanism is straightforward: bondholders care above all about *getting paid back*, so they are exquisitely sensitive to any deterioration in a company's ability to service its debt. When that ability looks shakier, the extra yield investors demand to hold risky corporate debt — the *credit spread* over Treasuries, measured as the option-adjusted spread or OAS — widens. Widening spreads are an early warning that the credit cycle is turning, and the equity drawdown tends to follow about a quarter later.

There is a second, subtler relationship that makes credit a useful *leading* signal for *forward* equity returns, and it runs in the opposite direction from what most people expect. The chart below plots the high-yield OAS *today* against the S&P 500's return over the *next twelve months*.

![Scatter of high yield spread today versus next twelve month S and P return with fit line](/imgs/blogs/lead-lag-leading-coincident-and-lagging-indicators-6.png)

The fitted line slopes *upward*: r ≈ +0.77, with a slope of roughly +3.4% of forward return per additional percentage point of spread. In plain terms, the *wider* spreads are today — the more fear is already priced into credit — the *higher* the S&P's return has tended to be over the following year. That sounds backward until you see the logic: very wide spreads mark moments of maximum fear, when risk assets are cheap and the bad news is already in the price. Spreads of 8% or 10% (the deleveraging panics) have historically been followed by strong equity returns; tight spreads of 3% (complacency) have been followed by weak ones.

This is a *contrarian* leading relationship, and it illustrates a deep point about lead/lag. The *change* in spreads (widening) leads equity drawdowns by a few months — that is the warning signal. But the *level* of spreads, once wide, leads the *recovery* — that is the opportunity signal. Same variable, two leads, opposite directions, depending on whether you are watching the change or the level. The cross-correlation function would find a *negative*-correlation peak at a short lead for "spread change → equity return" and a *positive*-correlation relationship for "spread level today → forward return." A reader who knows only "spreads up, stocks down" is missing half the signal.

#### Worked example: reading the spread level for forward returns

Suppose high-yield OAS has blown out to 8.0% in a panic. Plugging into the fitted line from the chart, the model's central estimate for the S&P's next-12-month return is roughly intercept + 3.4 × 8.0. With the fit shown (which passes near +18% at an 8% spread), the historical base rate points to a strongly positive forward year — even though the *current* mood is terror and the *recent* return has been awful. Contrast that with a 3.0% spread (calm markets), where the same line points to only single-digit forward returns.

The trade is contrarian and uncomfortable: wide spreads are a *buy* signal for forward equity returns precisely because they coincide with maximum fear. To put a dollar on it, an investor who deployed \$100,000 into the S&P at an 8% spread, with the model's ~+18% central estimate, would expect roughly \$18,000 of gain over the next year; the same \$100,000 deployed at a complacent 3% spread, with a single-digit central estimate, might expect closer to \$8,000. The spread level is, in effect, telling you the *expected payoff* on your next \$100,000 of risk. But note the honesty caveat — r ≈ 0.77 leaves real scatter (one of the data points at a 5.7% spread had a *negative* forward year), so the spread level shifts the *odds*, it does not guarantee the \$18,000. You size the position to the edge, not to the hope.

## How the three classes fit together: the indicator clock

Step back and look at all three classes as a system. The figure below lays out the taxonomy: each class, its examples, its timing relative to the cycle, its lead time, and — most importantly — the *job* you give it in your process.

![Matrix of leading coincident and lagging indicator classes with examples and uses](/imgs/blogs/lead-lag-leading-coincident-and-lagging-indicators-7.png)

The three jobs are distinct and non-substitutable:

- **Leading indicators → position.** Act on the warning, not the confirmation. These set your direction and your prior.
- **Coincident indicators → time.** Mark where the cycle is *right now*. These tell you which phase you are in. The [business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock) is built on exactly this idea — that asset-class correlations rotate as you move around the cycle.
- **Lagging indicators → confirm.** Verify a move that already happened. These are for after-the-fact validation and for slow-moving structural reads (unit labor costs, debt-to-income), not for timing.

A complete process uses all three, but it *weights* them by the decision at hand. To decide whether to add risk, you weight the leaders. To decide what regime you are in, you weight the coincident data. To check whether your earlier call was right, you read the laggards. The classic mistake is to use a lagging indicator (most often the unemployment rate, or the most recent GDP print) as if it were a *timing* tool. By the time unemployment is visibly rising, the recession is usually already underway and the equity market has often already bottomed — selling on that signal is selling at the lows.

#### Worked example: reading the clock at a turning point

It is the depth of a recession scare. Here is what each class is telling you:

- **Leading:** the yield curve, having inverted 16 months ago, has now *re-steepened* sharply (often a sign the cut cycle is starting); ISM new orders has ticked up from 44 to 48; credit spreads have stopped widening. The leaders are tentatively turning *up*.
- **Coincident:** payrolls are still negative, industrial production is flat. The coincident data says "we are still in the contraction."
- **Lagging:** the unemployment rate is *still rising* (5.2% and climbing) and earnings estimates are *still being cut*. The laggards say "things are bad and getting worse."

A trader who reads only the lagging data sells the bottom. A trader who reads the *leading* data buys it — the leaders are the first to turn, and a re-steepening curve plus improving new orders plus stabilizing spreads is the classic early-recovery signature, even while the headline (unemployment) is still deteriorating. The whole edge is in knowing that the leaders turn first and the laggards turn last, so a turning-point disagreement among the three classes is *expected*, not a contradiction. **You believe the leaders at the turn and the laggards in the middle of a trend.**

## Granger causality: "X helps predict Y" (and what it is not)

If X leads Y, a natural next question is: does X *cause* Y? The cross-correlation function cannot answer that — it only tells you that X's pattern arrives before Y's. The economist's slightly more rigorous tool for "does X help predict Y?" is **Granger causality**, named after the econometrician Clive Granger, and it is worth understanding at the level of intuition (we will keep the math out of it, as promised).

Before the test itself, fix the intuition with a wager. If I let you bet on next month's value of Y, you would want every scrap of useful information you could get. Granger's question is simply: *is the past of X one of those useful scraps, over and above what Y's own past already tells you?* If knowing X's history lets you place better bets on Y's future, then X "Granger-causes" Y in the only sense the test means — it carries incremental predictive information. If knowing X's history does not improve your bets at all, X does not Granger-cause Y, even if the two are highly correlated at lag zero.

The idea is a betting test. Suppose you are trying to predict Y next month. You build two forecasters:

- **Forecaster 1** uses only the past values of Y itself (Y's own history).
- **Forecaster 2** uses the past values of Y *plus* the past values of X.

If Forecaster 2 is *reliably better* at predicting Y than Forecaster 1 — if knowing X's history genuinely reduces your prediction error for Y — then we say "**X Granger-causes Y**." It is a statement about *predictive content*: X carries information about Y's future that Y's own past does not already contain.

That is a useful test, and it formalizes the intuition behind leading indicators: a good leading indicator is one that Granger-causes the thing it leads. The yield curve Granger-causes recessions in the technical sense — its history improves recession forecasts beyond what GDP's own history provides.

But notice what Granger causality is *not*:

- **It is not true causation.** The name is misleading. Granger causality says X *helps predict* Y, full stop. A rooster's crow Granger-causes the sunrise — the crow reliably precedes and "predicts" the dawn — but the rooster obviously does not *cause* the sun to rise. Both are driven by a hidden third variable (the Earth's rotation). In markets, the hidden third variable is everywhere: two indicators can both be driven by an unseen common factor (the credit cycle, central-bank liquidity, the business cycle itself), so that one "Granger-causes" the other without any direct mechanism between them.
- **It is regime-dependent.** A variable that Granger-caused another in one decade may stop doing so in the next, when the structure of the economy or the policy regime changes.
- **It can be fooled by a third leading variable.** If a common driver Z hits X first and Y second, then X will appear to Granger-cause Y even though Z is doing all the work.

#### Worked example: Granger causality versus real causation

Consider ISM new orders (X) and S&P earnings (Y). Forecaster 2 (Y's history + X) beats Forecaster 1 (Y's history alone) — new orders genuinely improve earnings forecasts — so X Granger-causes Y. And in this case there *is* a plausible direct mechanism (orders → production → revenue → earnings), so the Granger result and real causation point the same way.

Now consider gold (X) and the stock market (Y) during a particular crisis where both happened to fall together with gold a few days earlier. A naive Granger test on that window might say gold "Granger-causes" stocks. But there is no mechanism by which gold *causes* equities to fall — instead, a common driver (a liquidity squeeze forcing the sale of everything) hit both. The Granger test detected a *predictive* relationship that is not *causal*. The practical rule: **let Granger causality and the cross-correlation function tell you the timing, but only a mechanism tells you whether it will hold.** If you cannot name the chain of cause and effect, treat the lead as a fragile statistical artifact, size the position smaller, and watch for it to break.

## The lead itself is a regime, not a constant

This series has one repeated, non-negotiable theme: a correlation is a *regime*, not a constant. It has a sign, a strength, a lead/lag — and all of them *flip and drift across regimes*. The lead/lag is no exception. In fact, the lead is arguably the *most* unstable of the three properties, and treating it as a fixed number is the deepest trap in this whole subject.

Consider how the credit-spread-to-equity lead behaves in different regimes. In a slow, grinding deterioration — a classic late-cycle slide — spreads widen gently over months, and the equity drawdown follows with a lead of a full quarter or more. There is plenty of time to react. But in a fast, violent crash — a 2008 or a March 2020 — spreads and equities collapse almost *together*; the lead compresses toward zero because everyone is selling everything at once and there is no time for the signal to "lead" anything. The same correlation, the same two variables, but a lead that ranges from three months to three days depending on the *speed* of the regime. A strategy that hard-codes "spreads lead stocks by three months" will be perfectly calibrated in a slow grind and catastrophically too slow in a crash.

The yield-curve lead shows the *opposite* drift. Across recent cycles, the lead from inversion to recession has, if anything, *lengthened* — partly because central-bank intervention has stretched out cycles, partly because the structure of the economy has shifted toward services (which are less interest-rate-sensitive than the manufacturing and housing that dominated earlier cycles). A 12-month lead in one era can be an 18- or 24-month lead in another. This is exactly why the 2022 inversion was so painful for the bears: the lead had *lengthened*, so the recession that "should" have come in 2023 simply did not arrive on the old schedule.

Why do leads drift? Three structural reasons, all of which trace back to the mechanism:

- **The length of the causal chain changes.** If supply chains shorten or lengthen, the lead from orders to earnings shifts with them. If construction gets faster (prefabrication) or slower (permitting bottlenecks), the permits-to-GDP lead moves.
- **The policy reaction function changes.** Leads that run *through* the central bank — like the yield-curve-to-recession lead — depend on how the central bank responds. Change the reaction function and you change the lead.
- **The composition of the economy changes.** An economy dominated by interest-rate-sensitive sectors (housing, autos, capital goods) transmits monetary shocks faster than a service-heavy one. As the mix shifts, every monetary lead shifts with it.

The practical consequence is that you must **re-estimate leads as you go**, and you must read the *current* regime to know which version of the lead you are in. Are spreads grinding wider slowly, or gapping wider violently? That tells you whether the credit lead is three months or three days. Is the economy goods-heavy or services-heavy this cycle? That tells you whether the monetary leads are short or long. The number from the textbook is a *starting prior*, not a *setting* you dial in once and forget. The [business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock) develops this regime-rotation idea across whole asset classes; here the point is narrower and sharper — **even the timing of a relationship is conditional on the regime you are in.**

#### Worked example: the lead that compressed in a crash

In a slow late-cycle deterioration, suppose high-yield OAS drifts from 3.5% to 5.5% over four months, and the S&P begins its drawdown about three months after the spread first started widening — a clean ~3-month lead, plenty of warning. Now replay a crash: spreads gap from 3.5% to 8.0% in *two weeks* as a liquidity panic hits, and the S&P is already down 20% by the time the spread move is complete. The "lead" has collapsed from three months to essentially zero — the signal and the damage are now simultaneous. A trader who sized a hedge expecting three months of warning gets only a few days, and the position that was supposed to be a calm rotation becomes a frantic scramble. **The lead is part of the regime; when the regime speeds up, the lead shrinks, and a strategy calibrated to the slow regime fails in the fast one.**

## Common misconceptions

Lead/lag is intuitive enough to be dangerous: people get the gist and then over-apply it. Here are the five most expensive mistakes, each corrected with a number or a fact.

**Myth 1: "A leading indicator tells you *when* something will happen."** No — it tells you *that the odds have risen*, over a *window*. The 2s10s curve leads recessions by ~14 months *on average*, but past leads have ranged from roughly 6 to 24 months. Treating a 14-month average as a 14-month deadline is how traders end up short for a year while the market rallies. A leading indicator changes your prior, not your calendar.

**Myth 2: "If two things are correlated, you can trade one off the other."** Only if the correlation has a *lead*. A purely coincident correlation (CPI and PCE, r high at lag 0) is informative but not *actionable for timing* — by the time you see X move, Y has already moved too. The whole value of the cross-correlation function is to find which correlations have a usable lead and which are coincident dead ends.

**Myth 3: "The unemployment rate is a key recession signal."** It is, but it is a *lagging* one — it usually rises *after* the recession has begun and *after* the equity market has bottomed. Using it to time the market means selling near the lows and buying near the highs. The unemployment rate is for confirming a recession in the rear-view mirror, not for positioning. Watch *initial claims* (a ~2-month-lead flow) if you want an early labor-market read.

**Myth 4: "Granger causality proves X causes Y."** No. Granger causality means "X's history improves the forecast of Y," which is consistent with a hidden common driver, reverse causation in a different time frame, or pure coincidence over the sample. The rooster does not cause the sunrise. Always demand a mechanism before you believe a lead will persist.

**Myth 5: "The lead time is a stable constant."** It is the *most* regime-dependent of all the correlation's properties. The lead from credit spreads to equity drawdowns can be a month in a fast crash and a quarter in a slow grind. The lead from the yield curve to recession has lengthened over recent cycles. Quote leads as *windows* ("a couple of quarters"), re-estimate them as you go, and never hard-code a single number into a strategy.

## How to read it and use it: the lead/lag playbook

Here is the practical distillation — how a thoughtful macro reader actually uses the time axis of a correlation.

**1. For every correlation you care about, ask "what's the lead?"** Before you trade a relationship, run the cross-correlation function (or look up the documented lead). A relationship with a positive lead is a clock; one with a zero lead is a coincidence. Spend your attention on the clocks.

**2. Match your action's horizon to the signal's horizon.** A 14-month signal (yield curve) is for slowly shifting your prior and your strategic tilt. A 3-month signal (credit spreads) is for tactical positioning. A 2-month signal (claims) is for confirming a labor-market turn. Never convert a long-lead forecast into a short-lead trade — that mismatch is the single most common way to be "right but broke."

**3. Layer the leads.** Use the long-lead indicator to set the base rate ("recession risk is elevated") and a shorter-lead indicator to time the entry ("spreads are now widening past 5%"). The long lead changes the prior; the short lead pulls the trigger. The signal you act on should be the one whose horizon matches your holding period.

**4. Believe the leaders at the turn, the laggards in the trend.** At a turning point the three classes disagree by design — leaders turn first, laggards turn last. Weight the *leading* data near suspected turns. In the middle of an established trend, the slower coincident and lagging data is more reliable and less prone to false signals.

**5. Demand a mechanism.** A lead with a clear causal chain (orders → revenue → earnings) is robust; a lead that is purely statistical (this index happened to precede that one over the sample) is fragile. Granger causality and the cross-correlation function find candidates; only a mechanism tells you which ones will survive a regime change.

**6. Re-estimate the lead, and watch for it to break.** The lead is regime-dependent. Recompute it periodically. If the cross-correlation peak is drifting toward lag zero, the lead is collapsing (the relationship is becoming coincident — less useful). If the peak is vanishing, the relationship is breaking. A correlation that breaks at the wrong moment — see [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) on full-sample numbers lying — will hurt you most exactly when you are leaning on it.

**What invalidates the whole framework:** a structural break in the economy or the policy regime. If the central bank changes its reaction function, or the economy's composition shifts (more services, fewer factories), the old leads can lengthen, shorten, or disappear. The yield-curve lead, the ISM-earnings lead, the claims-unemployment lead — none are laws of nature. They are empirical regularities with mechanisms, and a mechanism can change. When the lead you have relied on stops working, do not force it; re-estimate, find the new lead, or stand aside.

The deepest lesson of the time axis is also the simplest: **the future shows up first in the indicators that sit earliest in the causal chain.** The bond market's curve sees the slowdown before the factory does; the factory's order book sees it before the earnings report does; the layoff sees it before the unemployment rate does. Your job is to read as far upstream as you can without drowning in the noise — to find the longest lead you can actually trust, match your action to its horizon, and let the laggards merely confirm what the leaders already told you.

And keep the humility that runs through this whole series front of mind: every lead you rely on is an *empirical* regularity, measured over a finite history, conditional on a regime that can change without asking your permission. The curve's lead can lengthen, the credit lead can compress, the ISM-earnings lead can scramble in a supply shock, and a relationship that Granger-caused another for a decade can quietly stop. So treat the time axis the way you treat the sign and the strength of any correlation — as a property to monitor, re-estimate, and stress-test, never as a constant to set and forget. The traders who survive are not the ones who memorized the lead times; they are the ones who understood the *mechanisms* well enough to notice, early, when a lead is about to break.

## Further reading & cross-links

Within this series:
- [What a correlation actually measures: Pearson, Spearman, beta](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) — the sign and strength of a correlation, before you add the time axis covered here.
- [The yield curve as a growth signal and its asset correlation](/blog/trading/macro-correlations/the-yield-curve-as-a-growth-signal-and-its-asset-correlation) — the deep dive on the longest-lead indicator in this post, and its correlation with banks, cyclicals, and the dollar.
- [The business-cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock) — how asset-class correlations rotate by cycle phase, built on the coincident-indicator idea of "where are we now."

The mechanism (how policy actually moves an asset):
- [Reading the yield curve: slope, inversion, recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) — why the curve inverts and what the inversion means.
- [The business cycle: four phases for traders](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) — the phase framework that the leading/coincident/lagging classes map onto.

The release-day reaction (how a print hits markets intraday):
- [ISM/PMI: the business surveys that lead](/blog/trading/event-trading/ism-pmi-the-business-surveys-that-lead) — the survey mechanics and the same-session market reaction to the ISM new-orders print discussed here.

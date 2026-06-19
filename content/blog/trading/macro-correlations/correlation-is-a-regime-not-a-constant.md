---
title: "Correlation Is a Regime, Not a Constant: How Macro Data Moves Asset Prices"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The thesis for a 45-post series: every link between an economic indicator and an asset price has a sign, a strength, a lead-lag, and it flips across regimes. Here is how to measure it and read it honestly."
tags: ["macro", "correlation", "stock-bond-correlation", "real-yields", "gold", "regime", "diversification", "asset-allocation", "inflation", "macro-trading", "60-40-portfolio"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — The relationship between any macro indicator and any asset price is not a fixed number you can memorize; it is a *regime*. Each link has a **sign** (do they move the same way or opposite?), a **strength** (how tight is the link?), a **lead-lag** (who moves first?), and it **flips** when the dominant macro risk changes.
>
> - The stock-bond correlation was about **−0.5 for two decades** (bonds rose when stocks fell — the engine of the 60/40 portfolio), then flipped to **+0.60 in 2022**, and the 60/40 had its worst year since the 1930s.
> - The cleanest macro link there is — gold versus the real yield — was **−0.96 in 2007–2021**, **+0.8 in 2022–2025**, and **−0.01 over the full sample**. The full-sample number is a lie that averages two opposite regimes into nothing.
> - A correlation rises with the inflation regime: stock-bond corr is **−0.45 below 2% inflation** but **+0.50 above 4%**. The regime is the variable.
> - **The one fact to remember:** a single correlation number with no regime attached is noise. Always ask "in which regime, measured over which window, leading or lagging?" — that is the whole series.

In October 2022, a man who had done everything right opened his retirement statement and felt the floor drop. He held the most boring, most recommended portfolio in finance: 60% stocks, 40% bonds. For forty years that mix had been sold as the closest thing to a free lunch in investing — when stocks crashed, bonds rallied, and the two losses never landed in the same year with full force. The bonds were the airbag. That was the whole point of owning them.

In 2022 the airbag did not deploy. US stocks fell about 18%. Long-term Treasury bonds — the safe part, the boring part, the part that was supposed to *go up* when stocks went down — fell roughly 30%, their worst year in modern history. A 60/40 portfolio lost around 17%, the worst calendar year for that mix since 1937. Nothing offset anything. The diversification that "always works" did not work, and it failed in the exact year people needed it most.

Here is the thing nobody tells beginners: the people who got crushed were not wrong about the *correlation between stocks and bonds*. They were wrong to think that correlation was a **constant**. For roughly twenty years the correlation had been negative — stocks and bonds genuinely did move opposite each other. Then, in 2022, it flipped positive. The relationship did not weaken or get noisy; it *changed sign*. And once you understand why it flipped, you can see it coming next time. That insight — that correlation is a regime, not a constant — is the entire thesis of this series.

![Four properties of a macro correlation: sign, strength, lead-lag, and flip](/imgs/blogs/correlation-is-a-regime-not-a-constant-1.png)

This is the first of 45 posts about a single, deceptively simple question: **how does macro data move asset prices?** Not "why does the Fed raise rates" — another series already covers the *mechanism*. Not "what happens in the ten minutes after the CPI report drops" — a third series covers the *release-day reaction*. This series is about the thing in between and underneath both: the **measurable statistical relationship** between an economic number and a price, how to measure it without fooling yourself, and how to read which regime is live right now. We will build it from absolute zero, so if you have never heard the word "correlation" used in anger, you are exactly who this is written for.

## Foundations: what a correlation actually is

Let me define the one word the whole series rests on, because most people who use it do not actually know what it means.

A **correlation** measures whether two things tend to move *together*, and how reliably. That is it. Forget the formula for a moment and start with your own life.

You have noticed that on days you sleep badly, you drink more coffee. Not every single time — some bad nights you skip coffee, some great nights you have a double shot out of habit — but *on average*, less sleep goes with more coffee. The two move in opposite directions: sleep down, coffee up. If you could put a single number on how reliable that opposite-movement is, you would have a correlation. A strong, reliable opposite-movement gets a number near **−1**. A strong, reliable same-direction movement (say, hours studied and exam score) gets a number near **+1**. Two things with no relationship at all (your shoe size and the weather) get a number near **0**.

That number is called the **correlation coefficient**, written as the letter *r*, and it always lives between −1 and +1:

- **r = +1** — perfect lockstep. When one goes up, the other always goes up, proportionally. (Almost never happens in markets.)
- **r = 0** — no linear relationship. Knowing one tells you nothing about the other.
- **r = −1** — perfect mirror. When one goes up, the other always goes down, proportionally.

In markets, the two "things" are usually a macro indicator (inflation, the unemployment rate, the 10-year Treasury yield, the price of oil) and an asset price (the S&P 500, gold, the dollar, Bitcoin). When we say "stocks and bonds have a correlation of −0.5," we mean: over some window of time, when stocks had an unusually good month, bonds tended to have a slightly-worse-than-average month, and vice versa, reliably enough to be worth about −0.5 on the −1-to-+1 scale.

### The everyday version of the four properties

The figure above introduced the spine of this whole series: a macro correlation is fully described by four properties. Let me ground each one in something you already understand before we go anywhere near a chart.

**1. The sign — same way or opposite way?** Sleep and coffee move opposite (negative). Studying and grades move together (positive). In markets: a rising real interest rate and the gold price move opposite (negative); a rising business survey and cyclical stocks move together (positive). The sign tells you whether an asset *hedges* your other holdings (moves opposite — good for diversification) or *amplifies* them (moves the same way — secretly doubling your bet).

**2. The strength — how tight is the link?** You drink more coffee on bad-sleep days *reliably*; you eat more ice cream on hot days *sometimes*. The first is a strong correlation, the second weak. In markets, r = −0.9 is a relationship you can lean on; r = −0.2 is mostly noise that will betray you. A weak correlation is not a small signal — it is mostly *luck*, and we will spend a whole later post (`spurious-correlation-and-the-traps-of-macro-data`) on why traders fool themselves with weak ones.

**3. The lead-lag — who moves first?** Dark clouds lead rain; rain does not lead clouds. The clouds are the *leading* indicator. In markets, the shape of the yield curve leads recessions by more than a year; the unemployment rate lags the economy. A leading link is *tradeable* — it gives you warning. A lagging one just confirms what already happened. Knowing which is which is the difference between a forecast and a post-mortem.

**4. The flip — when does the sign or strength change?** This is the property nobody respects, and it is the one that destroys portfolios. Ice cream sales and drowning deaths are correlated — but only because both rise in summer; the link "flips" to nothing in a heated indoor pool. In markets, the correlation between stocks and bonds was reliably negative for twenty years and then turned positive in a single year. **The correlation itself changes with the environment.** That environment is what we call the *regime*, and reading the regime is the master skill this series teaches.

A correlation with three of these four properties is still a trap. You can know the sign, the strength, and the lead-lag perfectly, and still get destroyed because you assumed the regime you measured in would last forever. The honest practitioner never quotes a correlation without attaching a regime to it.

#### Worked example: turning a co-movement into a number

Let me show you, in plain arithmetic, where the −0.5 comes from — no statistics background needed.

Suppose over five months, stocks returned **+3%, −2%, +1%, −4%, +2%**, and bonds over the same months returned **−1%, +2%, 0%, +3%, −1%**. Eyeball it: the big stock-down month (−4%) was the big bond-up month (+3%); the best stock month (+3%) was the worst bond month (−1%). They lean opposite. To put a number on it, you ask: when stocks were *above their own average*, were bonds *below their own average*? Stocks averaged 0%; bonds averaged +0.6%. In the −4% stock month, stocks were 4 points below average and bonds (+3%) were 2.4 points above average — opposite, as expected. You multiply each month's "stock deviation × bond deviation," add them up (a negative sum means they lean opposite), and divide by a scaling factor that forces the answer between −1 and +1. Do the arithmetic on these numbers and you get **r ≈ −0.86** — a strong negative link. **The intuition:** a correlation is just the averaged answer to "when one was unusually high, was the other unusually low?" — nothing more mystical than that.

We will derive the full formula carefully in the next post (`what-correlation-actually-measures-pearson-spearman-beta`), including the cousin you actually trade — *beta*, which measures not just the direction but the *magnitude* of the move. For now, the mental model is enough: positive means together, negative means opposite, near zero means unrelated, and the size tells you how much to trust it.

### Correlation versus beta: direction versus dosage

There is a distinction here that trips up almost everyone, and getting it straight early will save you a lot of confusion later. **Correlation (r) measures how *reliably* two things move together; beta measures *how much* one moves per unit of the other.** They are different questions, and you need both.

Take two thermometers. One is a high-quality thermometer that always reads exactly two degrees too high — it is *perfectly* correlated with the real temperature (r = 1.0, it never disagrees about the direction), and its *beta* to the real temperature is exactly 1.0 (one degree of real change is one degree on the dial, just shifted). A cheap second thermometer swings wildly — sometimes it over-reacts, sometimes under-reacts, but on average it still tracks the truth. It might have a *beta* of 2.0 (it amplifies real changes) but a *correlation* of only 0.6 (it is unreliable about it). Same direction, very different usefulness.

In markets you care about both. When the macro-trading series says "a hot CPI print sends the S&P down," the correlation tells you *whether you can count on the down-move*, and the beta tells you *how big it will be*. A relationship with high correlation but tiny beta (reliable but small) is a weak trade; one with high beta but low correlation (big but unreliable) is a gamble. The sweet spot — high correlation *and* meaningful beta — is what every later post is hunting for. We will state both wherever the data supports it, and Track G shows you how to estimate the beta with a real regression in Python (`measuring-beta-to-data-surprises-an-event-study-in-python`).

#### Worked example: same direction, different beta

The S&P 500 and the Nasdaq 100 both fall when a CPI surprise comes in hot — same sign. But the Nasdaq is full of long-duration tech, whose value sits far in the future and is therefore more sensitive to the discount rate. So a +0.1 percentage-point upside core-CPI surprise might send the S&P down about 0.7% and the Nasdaq down about 1.0%. Both have a similarly *strong* correlation to the surprise (both reliably fall), but the Nasdaq's *beta* is roughly 1.4 times the S&P's (1.0 ÷ 0.7). If you knew only the correlation, you would treat them as interchangeable hedges; knowing the beta tells you the Nasdaq is the higher-octane expression of the same trade. **The intuition:** correlation tells you they will move the same way; beta tells you the tech-heavy index will move *harder*, which is exactly the information that sizes a position.

### What a scatter plot is actually telling you

Almost every correlation in this series can be read off a *scatter plot* — one dot per period, the indicator on the horizontal axis, the asset on the vertical. You already saw one for gold versus real yields. Learning to read the *shape* of the cloud is most of the skill:

- **A tight downward-sloping cloud** (like the green dots in the gold chart) is a strong negative correlation — knowing the x value tells you the y value with confidence.
- **A tight upward-sloping cloud** is a strong positive correlation.
- **A round, shapeless blob** is a correlation near zero — the x value tells you nothing about y.
- **Two separate clouds with different slopes** (like the gold chart's green cloud versus red diamonds) is the visual signature of a *regime flip* — and the single most important pattern to learn to spot, because it is invisible in a single correlation number.

That last point is worth dwelling on. When you compute one correlation over a data set that secretly contains two regimes, you get a number — but the *scatter plot shows you the two clouds*. This is why every honest correlation analysis starts with the chart, not the number. The number can hide a regime flip; the eye cannot. We make this a discipline in `spurious-correlation-and-the-traps-of-macro-data`: always plot before you trust.

### Levels versus changes: the trap that manufactures fake correlations

There is one more foundational point that quietly ruins more macro analysis than any other, and it is so subtle that even professionals fall into it: **do you correlate the *levels* of two series, or their *changes*?** The answer is almost always *changes*, and getting this wrong manufactures correlations out of thin air.

Here is why. Take two completely unrelated things that both happen to *trend upward over time* — say, the total number of streaming subscriptions worldwide and the nominal price level (which rises with inflation every year). Plot the *levels* of those two series against each other and you will get a beautiful upward-sloping line with a correlation near +0.95 — because both are simply growing over time, and *anything* that grows over time correlates with *anything else* that grows over time. The correlation is real arithmetically and completely meaningless economically. It is an artifact of the shared trend, not a relationship between the two things.

The fix is to correlate the *changes* — month-over-month or year-over-year moves — which strips out the shared trend and asks the real question: "when one *accelerated*, did the other *accelerate* too?" This is why, throughout this series, we correlate *returns* (the change in an asset's price), *surprises* (the change in a data release versus what was expected), and *yield moves* (the change in a yield) — not raw price levels or raw index levels. A correlation of price *levels* is one of the classic spurious-correlation traps we dissect in `spurious-correlation-and-the-traps-of-macro-data`, and it is part of why the careful version of every claim here is stated in terms of changes, not levels.

This connects to a deeper statistical idea called **stationarity** — roughly, whether a series' statistical properties stay constant over time. A trending series (like a price level) is *non-stationary*; its changes (returns) are usually much closer to stationary. Correlating non-stationary series is the express lane to fooling yourself, which is why Track G's backtesting post (`backtesting-a-correlation-without-fooling-yourself`) opens with a stationarity test before anything else. For now, just internalize the rule: **correlate changes, not levels** — and be deeply suspicious of any chart that shows two upward-trending lines and calls their co-movement a relationship.

### A note on R-squared — the "how much is explained" number

You will also meet **R-squared** (R²), which is simply the correlation squared, expressed as a percentage. If gold and real yields have r = −0.9, then R² = 0.81, which reads as "about 81% of gold's variation is explained by real yields in this regime." It is a useful sanity number because it punishes weak correlations harshly: an r of 0.3 sounds like *something*, but R² = 0.09 means it explains only 9% of the variation — barely better than nothing. Whenever someone waves an r = 0.3 at you as a "signal," square it, and watch the signal shrink. We will use R² throughout as the honesty check on whether a correlation is worth acting on.

### Lead-lag: the property that turns a correlation into a forecast

Of the four properties, lead-lag is the one that separates a *forecast* from a *coincidence*. Two things can be perfectly correlated and still be useless if they move at the *same instant* — by the time you see one, the other has already happened. The value is in the *gap*: an indicator that reliably leads an asset by months is a warning bell; one that lags is a post-mortem.

Macroeconomists sort all indicators into three bins on exactly this basis. **Leading** indicators turn before the economy does — the yield curve, building permits, new orders in the manufacturing survey, the stock market itself. **Coincident** indicators move with the economy in real time — employment, industrial production, real income. **Lagging** indicators turn after the fact — the unemployment rate (which keeps rising for months after a recession has technically ended), inflation, the average duration of unemployment. The same logic applies to any indicator-asset pair: you want to find the time shift at which the correlation is *strongest*, and check whether the indicator sits before or after the asset.

Here is roughly how long some of the most important macro links lead, drawn from decades of business-cycle research:

- The **yield curve** (specifically the 2-year-versus-10-year spread) inverts about **14 months** before a recession begins — the longest and most famous lead in macro.
- **Building permits** lead GDP by about **9 months** — people pull permits before they pour concrete before the spending shows up in output.
- **ISM new orders** lead S&P earnings growth by about **6 months** — orders come in, then get built, then get booked as revenue.
- **Credit spreads** lead equity drawdowns by about **3 months** — the bond market smells trouble before stocks price it.
- **Initial jobless claims** lead the unemployment rate by about **2 months** — firings show up in weekly claims before they show up in the monthly rate.
- **CPI** and **PCE** (the two inflation gauges) are roughly *coincident* — CPI a touch earlier, but they essentially move together.

The practical payoff is enormous. If credit spreads lead equity drawdowns by three months, then a sudden widening in spreads is a three-month early-warning system for stocks — and that is a tradeable edge in a way that a coincident correlation never is. We devote a full post to the cross-correlation function and how to find the lead-lag at which a correlation peaks (`lead-lag-leading-coincident-and-lagging-indicators`), and the macro-trading series covers the recession-prediction mechanism of the curve in `reading-the-yield-curve-slope-inversion-recession`.

#### Worked example: trading the lead, not the coincidence

You are watching two signals in early 2007. The unemployment rate is still near a multi-year low — a *lagging* indicator, telling you the economy *was* fine. But the yield curve inverted in mid-2006 — a *leading* indicator with a ~14-month lead to recession. The naive reader trusts the reassuring unemployment rate; the lead-lag-aware reader trusts the curve and counts forward: mid-2006 plus 14 months points at roughly late 2007, which is almost exactly when the recession that became the financial crisis began (December 2007, per the NBER). The unemployment rate did not start climbing meaningfully until 2008 — by which point the warning was a year late. **The intuition:** a correlation is only a forecast if the indicator *leads*; the same data, read as a coincident or lagging signal, would have told you "all clear" right up to the cliff edge.

## Why a single full-sample number lies

Here is the most important idea in this entire series, and it is the reason the 60/40 crowd got blindsided.

When you compute a correlation, you have to choose a *window* — over how long a stretch of history do you measure it? The instinct of a careful person is "use as much data as possible — more data is more reliable." For a relationship that never changes, that instinct is correct. For a relationship that *flips between regimes*, that instinct is **exactly backwards**: averaging over a long window blends two opposite regimes into a mush that describes neither.

Let me prove it with the cleanest example in all of macro: **gold versus the real interest rate.**

First, two definitions. A **nominal** interest rate is the rate printed on the bond — say 4%. The **real** interest rate is what is left after subtracting expected inflation; it is the *true* reward for lending money once you account for the loss of purchasing power. We measure it directly from a special government bond called a TIPS (Treasury Inflation-Protected Security); the 10-year TIPS yield *is* the 10-year real yield. The macro-trading series treats the real yield as the master signal that prices everything — see [real versus nominal rates and real yields](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) for the full mechanism. Here we care only about its statistical link to gold.

Why should gold care about the real yield at all? Because gold pays you nothing — no interest, no dividend. Its only competition is a safe bond. When the real yield is high, a safe bond pays you a fat real return and gold (which pays zero) looks expensive to hold; when the real yield is negative — when safe bonds are guaranteed to lose you purchasing power — gold's zero yield suddenly looks great by comparison. So the textbook says: **real yield up → gold down.** A clean, strong, negative correlation. And for fifteen years, it was *exactly* that.

![Gold price versus the 10-year real yield, with the 2022 to 2025 break highlighted](/imgs/blogs/correlation-is-a-regime-not-a-constant-3.png)

Look at the green cloud — every year from 2007 to 2021. The dots fall almost perfectly on a downward line: as the real yield rose along the x-axis, gold fell along the y-axis. The correlation over that window is **−0.96**, which is about as tight a relationship as you will *ever* find in markets. If you had measured it in 2021 and written "gold is negatively correlated with real yields, r = −0.96" on a slide, you would have been completely, defensibly correct.

Now look at the red diamonds — 2022, 2023, 2024, 2025. The real yield rose sharply (from about 0.4% to over 2%), which by the −0.96 rule should have *crushed* gold. Instead gold went *up*, from about \$1,800 to \$2,650. The dots fly up and to the right — the opposite of the green line. Measured over 2022–2025 alone, the correlation is **+0.8**. The sign flipped.

Here is the punchline. If you take all the years together — green and red, the whole sample from 2007 to 2025 — the correlation is **−0.01**. Essentially zero. A naive analyst running the full sample would conclude "gold and real yields are unrelated; ignore the real yield when you trade gold." That conclusion is catastrophically wrong. Gold and real yields are *intensely* related — they were −0.96 in one regime and +0.8 in another. The full-sample −0.01 is not a measurement of a weak relationship; it is **two strong, opposite relationships averaging each other into nothing.** The number is technically true and completely useless.

#### Worked example: how two strong correlations average to zero

Make it concrete. Take a relationship that is +0.9 in regime A and −0.9 in regime B, and your data set is half regime A, half regime B. The full-sample correlation is *not* somewhere around ±0.9 — it is approximately **zero**, because the up-moves in regime A and the down-moves in regime B systematically cancel when you pool them. With gold-versus-real-yields the cancellation is not perfectly symmetric (−0.96 over 15 years versus +0.8 over 4 years), and the units differ, so the pooled number lands at −0.01 rather than dead zero — but the mechanism is identical. **The intuition:** pooling data across a regime flip does not give you "the average relationship"; it gives you a meaningless number that hides the fact that the relationship has two faces. The fix is never "get more data" — it is "split the data by regime."

So why *did* gold and real yields decouple after 2022? Because a new, dominant buyer arrived: central banks. After Western governments froze Russia's dollar reserves in 2022, central banks across the developing world — China, Turkey, India, Poland — concluded that dollar reserves could be weaponized, and rushed to buy gold instead, regardless of its yield disadvantage. Their buying was *price-insensitive to the real yield*; they were not running the gold-versus-bonds trade-off at all. A new force overwhelmed the old one, and the regime changed. We dedicate an entire post to this (`inflation-and-gold-the-real-yield-story`), but the lesson here is structural: **a correlation lives only as long as the force that creates it stays dominant.** When a bigger force arrives, the correlation breaks.

## The signature flip: stocks and bonds

Now back to the portfolio that lost 17%. The stock-bond correlation is the most consequential macro correlation on Earth, because trillions of dollars in pension funds, endowments, and retirement accounts are allocated on the assumption that it is negative. Let me show you the full history.

![Rolling stock-bond correlation from 1990 to 2025 with regime bands](/imgs/blogs/correlation-is-a-regime-not-a-constant-2.png)

This is a **rolling correlation** — at each point in time, we compute the correlation over the trailing two years or so, then slide the window forward. (Why a rolling window and not the full sample? Because we just learned the full sample lies; we will dig into the bias-variance trade-off of window choice in `rolling-correlation-and-why-the-window-matters`.) The dashed line at zero is the dividing line between "diversifying" (below zero) and "amplifying" (above zero).

Read the three bands:

- **1990–1997 (amber):** correlation positive, around +0.3 to +0.4. Stocks and bonds moved *together*. This was the tail end of the high-inflation era that began in the 1970s.
- **1998–2021 (green):** correlation negative, often −0.4 to −0.5. This is the "diversifier" era — the two decades that taught everyone that bonds protect you when stocks fall. Every textbook, every robo-advisor, every "set it and forget it" 60/40 default was built on this band.
- **2022–2025 (red):** correlation snaps positive to **+0.60**, the highest in the chart, then eases but stays positive. The diversifier became an amplifier.

Notice this is not a gentle drift. The correlation was around −0.10 in 2021 and +0.60 in 2022 — a 0.70 swing in a single year. That is what a regime flip looks like: not a slow slide, but a sudden change in *which macro force is in charge*.

### What actually drives the flip

The flip is not random and it is not unknowable. It is governed by **what the market is most afraid of**, and that fear has two flavors:

**When the dominant fear is a growth scare** (recession, deflation, a financial accident), money behaves like this: bad growth news hits → stocks fall because earnings will shrink → investors flee to the safety of government bonds → bond prices rise (yields fall). Stocks down, bonds up: **negative correlation, the diversifier.** This was the world of 1998–2021, an era of low and stable inflation where the central bank could always ride to the rescue by cutting rates, and cutting rates pushed bond prices up exactly when stocks needed help.

**When the dominant fear is an inflation scare,** the logic inverts: hot inflation data hits → the central bank must hike interest rates to fight it → higher rates mean a higher discount rate, which *lowers the present value of both future earnings (stocks) and future coupons (bonds)* → both fall together. Stocks down, bonds down: **positive correlation, the amplifier.** This was 2022. Inflation hit 9%, the Fed hiked at the fastest pace in forty years, and the single rate-shock variable drove both asset classes down in lockstep.

![Two regimes for the same stock and bond pair, with the correlation flipping from negative to positive](/imgs/blogs/correlation-is-a-regime-not-a-constant-5.png)

The figure lays the two regimes side by side. Same two assets — stocks and bonds. The only thing that changed is which shock is in the driver's seat. In the low-inflation regime (left), a growth scare sends money *from* stocks *to* bonds, and they offset. In the high-inflation regime (right), an inflation scare sends a rate shock through *both*, and they sink together. The portfolio did not change. The world changed.

The deepest version of this is `the-stock-bond-correlation-regime` in this series and the full 2022 case study in `when-correlations-break-the-2022-stock-bond-flip`; the *mechanism* of how rates move every asset is in the macro-trading series at [how policy moves every asset](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map). But you can already see the master principle: **the sign of a correlation is set by which macro variable is the dominant source of risk.** Change the dominant risk, and you change the sign.

### The single number that predicts the flip

The remarkable, practical thing is that this flip is *conditionable*. You do not have to guess which regime you are in — you can read it off one variable: the level of inflation.

![Stock-bond correlation grouped by the inflation regime](/imgs/blogs/correlation-is-a-regime-not-a-constant-4.png)

Group every historical period by its average core inflation rate and compute the stock-bond correlation within each group:

- **Inflation below 2%:** correlation **−0.45** (strong diversifier)
- **2–3%:** correlation **−0.30** (still diversifying)
- **3–4%:** correlation **+0.05** (the link is dead — bonds no longer hedge)
- **Above 4%:** correlation **+0.50** (full amplifier — bonds make it worse)

There is a *threshold*, somewhere around 3–4% inflation, where the stock-bond correlation crosses from negative to positive. Below it, the 60/40 free lunch is real. Above it, it is a trap. This is the single most useful conditioning variable in cross-asset investing, and it tells you that the question is never "are stocks and bonds correlated?" — it is "what is the inflation regime, and therefore what is the *sign* of the correlation right now?" The sibling post `inflation-and-stocks-the-correlation-that-flips` walks the full U-shape; for the diversification logic, the cross-asset series covers it in [correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch).

#### Worked example: sizing the 60/40 airbag in each regime

Put dollars on it. You hold a \$1,000,000 portfolio, 60% stocks (\$600,000) and 40% long bonds (\$400,000). A growth scare hits and stocks fall 20%, a \$120,000 loss on the equity sleeve.

In the **diversifier regime** (inflation under 2%, correlation about −0.45), bonds typically rally in a growth scare; say long bonds gain 8%, a \$32,000 gain on the bond sleeve. Net portfolio loss: \$120,000 − \$32,000 = **\$88,000, or −8.8%.** The bonds absorbed more than a quarter of the equity hit. The airbag worked.

In the **inflation regime** (inflation over 4%, correlation about +0.50), the same 20% equity fall is driven by a rate shock that *also* hits bonds; say long bonds fall 15%, a \$60,000 *loss* on the bond sleeve. Net portfolio loss: \$120,000 + \$60,000 = **\$180,000, or −18.0%.** The "safe" sleeve doubled your pain. **The intuition:** the *exact same allocation* lost 8.8% in one regime and 18.0% in the other, purely because the correlation flipped sign — the portfolio's risk was never a property of the portfolio, it was a property of the regime.

## How this series is organized

You now have the entire thesis: every macro-to-asset link has a sign, strength, lead-lag, and a regime in which it flips, and the full-sample number lies. The remaining 44 posts apply that lens systematically — first to teach you to *measure* a correlation properly, then to walk every family of indicators against every asset, then to teach you to *read the live regime* and act on it.

![The 45 post series map across measuring, indicators, regimes, and the playbook](/imgs/blogs/correlation-is-a-regime-not-a-constant-7.png)

The series runs in eight tracks:

- **Track A — Measuring it.** The toolkit: Pearson versus Spearman versus beta, the rolling window and why it matters, lead-lag and the cross-correlation function, correlating the *surprise* not the level, and the traps (spurious correlation, non-stationarity, p-hacking). It ends with `the-macro-asset-correlation-matrix`, the master heatmap that maps every driver to every asset.
- **Track B — Inflation indicators × assets.** CPI, core CPI, PPI, PCE, breakevens, and the gold/real-yield story you just met.
- **Track C — Growth and labor indicators × assets.** NFP (the jobs report), jobless claims, ISM/PMI surveys, GDP, the yield curve as a recession signal, and the business-cycle "correlation clock."
- **Track D — Rates and the bond complex × assets.** The master transmission: the 10-year yield, real yields, the Fed-funds path, the stock-bond regime, credit spreads, and the dollar.
- **Track E — Cross-asset and commodities.** Oil, the copper/gold ratio, crypto as a macro asset, the VIX and risk-on/risk-off, and global liquidity.
- **Track F — Regimes, breakdowns, and time-variation.** The four macro quadrants, the 2022 flip in depth, correlations during crises, and structural shifts.
- **Track G — The quant toolkit, in Python.** Runnable pandas/numpy/statsmodels code: build a correlation dashboard, estimate betas to data surprises, detect regime change-points, and backtest a correlation without fooling yourself. This is the spine no other series has.
- **Tracks H and I — Vietnam and the playbook.** The imported correlation (VN-Index versus US macro), USD/VND and the rate differential, the domestic links, then the practitioner routine and the capstone.

The thread through all of it is the four properties and the regime question. Whether we are looking at CPI and the dollar or the unemployment rate and bank stocks, the same four questions apply: what is the sign, how strong, who leads, and when does it flip?

### The map of the whole series in one chart

The master deliverable of Track A is a single heatmap that captures the *sign and strength* of every important macro driver against every major asset. Here is a preview — the kind of map we will spend the series building, validating, and (crucially) annotating with the regimes in which each cell holds.

![Heatmap of macro drivers versus asset returns showing correlation sign and strength](/imgs/blogs/correlation-is-a-regime-not-a-constant-6.png)

Read it like a weather map. Each row is a macro driver (a rate rising, a hot CPI surprise, a stronger dollar). Each column is an asset. Green means "moves the same way as the driver"; red means "moves opposite." The deeper the color, the stronger the link.

A few patterns you can already see, each of which becomes its own deep-dive later:

- **The top three rows are a wall of red across risk assets.** A rising 10-year yield, a rising real yield, and a hot CPI surprise all push stocks, bonds, gold, crypto, and emerging markets *down*. This is the discount-rate channel — higher rates lower the present value of everything — and it is the single most important transmission in macro. Track D is built around it.
- **The "stronger USD" row is mostly red, with a +1.00 against USD itself** (trivially, the dollar moves with itself). A strong dollar pressures commodities, gold, crypto, and emerging markets. The dollar is *cross-asset gravity*; we cover it in `the-dollar-dxy-cross-asset-correlation` and the cross-asset series covers it in `the-dollar-cross-asset-gravity`.
- **The "ISM/PMI rise" row is green for risk assets and red for bonds** — good growth news lifts stocks and cyclicals but hurts safe bonds (it pushes yields up). This is the growth channel, the mirror image of the rate channel.
- **The "credit spread wider" row is deep red for equities** — when the price of corporate borrowing jumps, stocks fall hard, because widening spreads are the bond market's early warning of trouble. Credit leads equity; we cover that, and how it goes to −1 in a crisis, in `credit-spreads-the-risk-correlation-and-the-canary`.

But — and this is the entire point of the series — **every single cell in that map has a regime in which it flips.** The −0.10 for "hot CPI surprise versus gold" is an average across regimes where gold sometimes loves inflation (1970s) and sometimes ignores it (2010s) and sometimes rallies *despite* high real yields (2022–25). A static heatmap is a starting point, not an answer. The honest version of that map carries a regime label on every cell, and learning to attach those labels is the skill.

## How this series differs from its two siblings

If you have read around the blog, you may have hit two adjacent series and wondered how this one is different. The distinction is worth nailing down, because confusing the three is the most common way people misuse macro data.

**The macro-trading series is about the *mechanism* — the "why."** It answers "*how* does the Fed raising rates move stock prices?" by tracing the causal chain: higher policy rate → higher discount rate → lower present value of earnings → lower stock prices. It is about cause and effect, the plumbing of the economy. When you want to understand *why* a relationship exists, you read macro-trading — for example `interest-rates-the-price-of-money-master-variable` or `the-business-cycle-four-phases-for-traders`. This series *cites* those mechanisms; it does not re-derive them.

**The event-trading series is about the *release-day reaction* — the "what happens in the next ten minutes."** It answers "the CPI report just printed 0.2 above expectations — what does the S&P do in the next hour?" It is about the intraday move, the surprise versus consensus, the whipsaw around a data release. When you want to know how to position around a specific scheduled report, you read event-trading — for example `cpi-the-report-that-moves-the-world` or `why-news-moves-markets-the-surprise-framework`. This series builds on the *surprise* concept from event-trading (you correlate the surprise, not the level — see `the-surprise-not-the-level-betas-to-data-surprises`) but it is not about the intraday reaction.

**This series is about the *measurable correlation structure itself* — the "how related, how reliably, leading or lagging, and in which regime."** It sits underneath both. Before you can trade the mechanism or the reaction, you have to know: *is* there a reliable relationship, how strong, which way does it lead, and — the whole point — does it hold in the regime you are actually in? The mechanism tells you a link *should* exist; this series tells you whether it *does*, how much to trust it, and when it stops being true.

A quick way to remember it: macro-trading is *physics* (the laws of cause and effect), event-trading is *the weather report for tomorrow* (the immediate forecast), and this series is *the climate study* (the long-run statistical structure and how it shifts between regimes). You need all three, and they do not substitute for one another.

## Common misconceptions

Most of what people "know" about macro correlations is a half-truth that holds in one regime and fails in another. Here are the five that cost the most money, each corrected with a number.

**Myth 1: "Gold is an inflation hedge."** This is the most quoted and most wrong belief in all of macro. The correlation between gold and *CPI itself* is weak and unstable — over 2010–2020, inflation was low and gold went sideways then ripped; over the 1980s–90s, inflation was real and gold *fell* for two decades. Gold's actual master variable is the **real yield**, where the correlation was −0.96 over 2007–2021 — and that link itself broke in 2022. The honest statement is "gold is mostly a play on real yields, except in regimes where central-bank buying dominates." `inflation-and-gold-the-real-yield-story` is the full autopsy.

**Myth 2: "Bonds always protect a stock portfolio."** False in any high-inflation regime. The stock-bond correlation is −0.45 when inflation is under 2% but **+0.50 when inflation is over 4%.** In 2022, the protective sleeve fell 30%. Bonds protect against *growth* scares, not *inflation* scares — and which kind of scare is live is exactly the regime question.

**Myth 3: "A correlation near zero means no relationship."** This is the gold-versus-real-yields trap. The full-sample correlation was **−0.01** — apparently nothing — while the two underlying regimes were −0.96 and +0.8. A near-zero pooled correlation can hide two *intense* opposite relationships. Always split by regime before you conclude "unrelated"; we formalize this in `spurious-correlation-and-the-traps-of-macro-data`.

**Myth 4: "Correlation is causation."** A correlation tells you two things move together; it tells you *nothing* about whether one *causes* the other, or whether a hidden third variable drives both. Ice cream sales and drownings correlate (both driven by summer). In markets, a hidden third variable — usually the level of liquidity or the dominant regime — drives many apparent correlations, and when that hidden driver changes, the surface correlation evaporates. We never trade a correlation without a plausible mechanism behind it, which is exactly why this series leans on the macro-trading mechanism posts.

**Myth 5: "More data gives a more reliable correlation."** Only if the relationship is stationary — that is, if it does not change over time. Macro correlations are *not* stationary; they flip with the regime. For a non-stationary relationship, a longer window is *worse*, because it pools more regimes together. The right answer is a *rolling* window short enough to capture the current regime but long enough to be statistically meaningful — the bias-variance trade-off we tackle in `rolling-correlation-and-why-the-window-matters`.

## How it shows up in real markets

Theory is cheap. Here are dated episodes where the regime nature of correlation was the whole story — and money was made or lost on whether people understood it.

**2008 — the diversifier at its best.** The global financial crisis was the purest *growth scare* in modern history. Stocks fell more than 50% peak to trough; long Treasury bonds *rallied* about 20% as terrified money fled to safety and the Fed slashed rates to zero. The stock-bond correlation in this window was deeply negative, around −0.45 to −0.5. A 60/40 portfolio still lost money, but the bonds cushioned it enormously. Anyone who held the textbook negative-correlation belief was vindicated — *because the regime was a growth scare, where that belief holds.*

**2013 — the "taper tantrum" preview.** When the Fed merely *hinted* it would slow its bond-buying, real yields jumped sharply. Gold, sitting on its tight −0.96 link to real yields, fell about 28% on the year — its worst since 1981. This was the negative gold/real-yield correlation working *exactly as advertised*, and it caught a lot of "gold is an inflation hedge" believers off guard, because inflation that year was *low* and gold fell anyway. The lesson: gold was never tracking inflation; it was tracking real yields.

**2022 — the flip that broke 60/40.** The dominant fear switched from growth to inflation. The stock-bond correlation went from about −0.1 to +0.6; gold *defied* its real-yield link (+0.8 instead of the historical −0.96) as central banks piled in; the dollar (DXY) ripped to a 20-year high, dragging everything priced against it lower. Three "reliable" correlations either flipped or broke in the same twelve months. The investors who survived best were not the ones with the best forecasts of inflation — they were the ones who *recognized the regime had changed* and stopped assuming bonds would hedge.

**2020–2024 — Bitcoin's correlation that grew, then faded.** Not every regime story is a *sign* flip; some are *strength* changes, and Bitcoin is the cleanest example. Before 2020, Bitcoin's correlation with the Nasdaq was essentially zero (around +0.05) — it traded as its own strange thing, deaf to macro. Then, as institutional money and macro funds piled into crypto and the world flooded with central-bank liquidity, Bitcoin started trading as a high-beta macro-liquidity asset: its 90-day correlation with the Nasdaq spiked to about **+0.65 in 2022**, right as the liquidity tide went out and *both* sold off hard together. By 2024–2025, as crypto-specific drivers (spot ETFs, halving cycles, idiosyncratic flows) reasserted themselves, the correlation *faded* back toward +0.2 to +0.3. The sign never flipped — it stayed positive throughout — but the *strength* swung from "ignore it" to "this is a macro asset" and back to "mostly its own thing." **The same kill-switch logic applies:** Bitcoin behaves like a macro asset when macro liquidity is the dominant driver, and like an idiosyncratic asset when it is not. The full treatment is in `crypto-as-a-macro-asset-the-liquidity-correlation`. The lesson: a correlation can betray you by *weakening* just as much as by flipping sign — if you sized a "crypto hedges nothing, it's uncorrelated" book in 2021 and carried it into 2022, the +0.65 correlation meant your "diversifier" sank with everything else.

**2022 — the dollar as cross-asset gravity.** The third correlation that mattered in 2022 was the US dollar. As the Fed out-hiked the rest of the world, the dollar index (DXY) surged to a 20-year high, and a strong dollar is *gravity* for everything priced against it. The dollar's longer-run correlations are deeply negative across the risk complex: roughly **−0.55 with gold, −0.45 with oil, −0.50 with copper, −0.55 with emerging-market equities, and −0.35 with Bitcoin**, while it runs about **+0.40 with US 10-year yields** (higher US yields pull capital toward the dollar). When the dollar rips, commodities priced in dollars get mechanically more expensive for the rest of the world and demand softens; emerging markets that borrowed in dollars face a tightening vise; gold, which competes with dollar-denominated safe assets, loses its shine. In 2022, this single variable dragged half the macro map lower at once, and it is why no cross-asset analysis is complete without the dollar. We cover it in `the-dollar-dxy-cross-asset-correlation`, and the cross-asset series frames it as `the-dollar-cross-asset-gravity`.

**2024–2025 — the new normal, watching for the next flip.** Inflation cooled toward 3% — right at the threshold band where the stock-bond correlation hovers near zero. Bonds neither reliably hedged nor reliably amplified; the correlation drifted between +0.2 and +0.3. Gold kept rising on continued central-bank buying even as real yields stayed high near 2%, the decoupling persisting. The honest read of this regime is not "bonds are back as a hedge" or "they are dead" — it is "we are near the threshold, the correlation is unstable, and a move in inflation in *either* direction will decide the sign." That is what reading a live regime actually feels like: not certainty, but knowing which variable to watch.

#### Worked example: reading the live regime to set your hedge

You manage a \$10,000,000 fund in mid-2025. Core inflation is running about 3.2% — squarely in the 3–4% band where the stock-bond correlation is roughly +0.05, essentially zero. You want to hedge a \$6,000,000 equity book against a sell-off. The naive move is to buy \$4,000,000 of long bonds and call it diversified — but at a correlation near zero, those bonds will neither help nor hurt in an equity drawdown; you have spent capital on a hedge that does not hedge. The regime-aware move is to ask "which way will inflation break?" If inflation re-accelerates above 4%, the correlation goes to +0.5 and bonds become an *anti-hedge* — you would want *less* duration, not more, and a different hedge entirely (cash, or puts). If inflation falls below 2%, bonds become a real hedge again and the duration is worth holding. **The intuition:** in a near-threshold regime, the *level* of your hedge matters far less than correctly forecasting which side of the threshold inflation lands on — the hedge decision is really a regime forecast in disguise.

## How to read and use a macro correlation

Here is the playbook that distills the whole series into a procedure you can run on any indicator-asset pair. Every later post is an application of these five steps.

**Step 1 — Establish the mechanism first.** Before you measure anything, ask: *why should these two be related?* If you cannot name a plausible causal chain (real yields raise the opportunity cost of holding gold; rate hikes lower the discount rate on stocks), do not trade the correlation no matter how strong it looks — it is probably spurious. This is why we lean on the macro-trading mechanism posts. A correlation without a mechanism is a coincidence waiting to betray you.

**Step 2 — Measure the four properties, on a rolling window.** Get the *sign* (same way or opposite), the *strength* (r, and ideally the beta — how big a move per unit of driver), and the *lead-lag* (does the indicator lead the asset, and by how long?). Crucially, measure it on a **rolling window**, not the full sample, so you can see whether it is stable or drifting. A correlation that is steady at −0.5 for ten years is very different from one oscillating between −0.5 and +0.5 even if both average −0.5.

**Step 3 — Identify the current regime.** This is the master step. Ask: what is the dominant macro fear right now — growth or inflation? What is the inflation level (above or below the 3–4% threshold)? Is liquidity expanding or contracting? Is the dollar strong or weak? The regime determines which sign of each correlation is live. Track F and the business-cycle clock (`the-business-cycle-correlation-clock`) are entirely about this step.

**Step 4 — Check what would invalidate the correlation.** Every correlation has a "kill switch" — the condition under which it breaks. For gold/real-yields, it is *price-insensitive central-bank buying*. For stock-bond, it is *inflation crossing the threshold*. For credit-spreads/equity, it is *forced deleveraging that takes the correlation to −1*. Know the kill switch before you put on the trade, and watch for it. A correlation you cannot break in your head is one you do not understand.

**Step 5 — Size to the regime, and re-check when the regime shifts.** Use the correlation to size positions and hedges *for the current regime*, and set a trigger — usually a regime variable like the inflation level or a liquidity gauge — that tells you to re-measure. The single biggest mistake is "set and forget": measuring a correlation once, in one regime, and assuming it holds forever. The whole 60/40 disaster of 2022 was a set-and-forget failure.

A concrete way to operationalize the trigger: pick the *one variable* that defines the regime for your trade and watch its level against the threshold where the correlation flips. For the stock-bond pair, that variable is core inflation and the threshold is roughly 3–4%; as long as inflation sits comfortably below it, you can lean on bonds as a hedge, and the moment it climbs through it you re-measure everything. For gold, the trigger is the *pace of central-bank buying* relative to the real-yield move — when official-sector demand overwhelms the real-yield signal, the −0.96 link is suspended. For credit-versus-equity, the trigger is *forced deleveraging*, which pulls the correlation toward −1 and means the early-warning value of spreads collapses into a simultaneous crash. The practitioner's weekly routine — what to watch and when to re-measure — is the subject of `building-a-macro-correlation-monitor-the-weekly-routine`, and the full catalog of mistakes (set-and-forget, full-sample bias, the diversification illusion, p-hacking) is `common-correlation-mistakes-and-how-to-avoid-them`. The discipline is simple to state and hard to keep: a correlation is a *living* number, and the regime variable is its pulse.

If you run those five steps honestly, you will never again say "stocks and bonds are negatively correlated" as if it were a law of nature. You will say "stocks and bonds are negatively correlated *in a low-inflation, growth-scare regime, where they have been for two decades* — and that correlation will flip positive if inflation crosses about 3–4%, which is the variable I am watching." That sentence — sign, strength, regime, and kill switch all attached — is the difference between someone who quotes correlations and someone who uses them.

### The one-paragraph version to keep

If you remember nothing else from this entire series, remember this. A macro correlation is not a number; it is a *number plus a regime*. The number tells you the sign, the strength, and the lead-lag *within* a regime; the regime tells you when that number stops being true. The single most expensive mistake in macro investing is treating a correlation measured in one regime as a constant that holds across all of them — that mistake is exactly what made 2022 the worst year for the 60/40 portfolio since 1937. Every post that follows is a worked application of one principle: **measure the correlation, name the regime, and watch the kill switch.**

## Further reading and cross-links

This is the intro; the rest of the series builds out every cell of the map.

**Next in this series (Track A — measuring it):**
- `what-correlation-actually-measures-pearson-spearman-beta` — covariance, Pearson versus rank correlation, R², and the beta you actually trade.
- `rolling-correlation-and-why-the-window-matters` — why the full sample lies, and how to choose a window.
- `the-macro-asset-correlation-matrix` — the full indicator-by-asset heatmap, with regime labels.

**Key later posts referenced above:**
- `the-stock-bond-correlation-regime` and `when-correlations-break-the-2022-stock-bond-flip` — the flip in full.
- `inflation-and-gold-the-real-yield-story` — why gold tracks real yields, and when it does not.
- `inflation-and-stocks-the-correlation-that-flips` — the inflation-threshold U-shape.
- `credit-spreads-the-risk-correlation-and-the-canary` — the canary that goes to −1 in a crisis.
- `the-dollar-dxy-cross-asset-correlation` — cross-asset gravity.

**The mechanism (macro-trading series — *why* the links exist):**
- [Interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — the master variable behind the discount-rate channel.
- [Real versus nominal rates and real yields](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal) — why the real yield prices gold and long-duration assets.
- [How policy moves every asset](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map) — the full cross-asset transmission map.
- [The business cycle in four phases](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) — the cycle that rotates the regime.
- [Reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) — the leading recession signal.

**The release-day reaction (event-trading series — *what happens in the next ten minutes*):**
- [Why news moves markets: the surprise framework](/blog/trading/event-trading/why-news-moves-markets-the-surprise-framework) — correlate the surprise, not the level.
- [CPI: the report that moves the world](/blog/trading/event-trading/cpi-the-report-that-moves-the-world) — the intraday reaction to the master inflation print.

**The allocator's lens (cross-asset series):**
- [Correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch) — why correlation is the engine of diversification.
- [The stock-bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) — the allocator's view of the flip.
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis) — why diversification fails when you need it.
- [The dollar as cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity) — the dollar's pull on every risk asset.

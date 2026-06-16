---
title: "Reading the Regime in Real Time: The Cross-Asset Dashboard"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "The growth-and-inflation regime is only obvious in hindsight, so you have to nowcast it. This is the handful of public dials — PMIs, the yield curve, breakevens, real yields, credit spreads, the dollar — that tell you which quadrant you are in now, in time to act."
tags: ["asset-allocation", "cross-asset", "macro-dashboard", "nowcasting", "yield-curve", "credit-spreads", "pmi", "real-yields", "regime-detection", "copper-gold-ratio", "vix", "tactical-allocation"]
category: "trading"
subcategory: "Cross-Asset"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — The growth-and-inflation regime that decides what you should own is only obvious *after* it has happened, so the whole game is to **nowcast** it from a handful of public dials and act on confirmed turns before the move is over.
>
> - Pin down **three dials**: the direction of **growth** (PMIs, payrolls and claims, the yield-curve slope, the copper/gold ratio), the direction of **inflation** (CPI/PCE trend, breakevens, oil, wages), and the **policy/liquidity** stance (the Fed, real yields, financial conditions, the dollar, credit spreads).
> - **Markets vote faster than the data prints.** The yield curve, high-yield credit spreads, the dollar, the copper/gold ratio, the VIX and equity-sector leadership re-price the regime in real time; CPI and GDP arrive weeks to a quarter late. When the two disagree, **markets usually move first** — weight them for speed, weight the data for confirmation.
> - Combine the dials into a **scorecard** that maps to one of four quadrants — **Goldilocks** (growth up, inflation down), **Overheat** (growth up, inflation up), **Stagflation** (growth down, inflation up), **Recession** (growth down, inflation down) — and each quadrant has a mapped tilt.
> - The one thing to remember: in **June 2022** every public dial pointed the same way — CPI **9.06%** and rising, ISM rolling over, the 10y real yield surging from **−1.04%** to **+1.74%**, the dollar near **114.8** — a stagflation read that, acted on, turned a **−16%** 60/40 year into roughly flat. No forecast was needed; just a checklist, read honestly.

In hindsight, every market regime looks obvious. Of course you should have been in commodities and cash in 2022, when inflation hit a 40-year high and the Federal Reserve — America's central bank, the institution that sets the country's base interest rate — was hiking faster than at any time since the 1980s. Of course you should have bought stocks in the spring of 2009, when the economy was still bleeding jobs but the recovery had quietly begun. Of course Treasuries were the place to be in October 2008. The chart, read backward, tells a clean story.

But you do not get to trade backward. You trade *forward*, into a fog, where the regime is never labeled. Nobody rings a bell that says "Overheat begins today." The official data that would confirm the regime — the inflation print, the jobs report, the GDP number — arrives weeks or even a quarter after the fact, and the body that officially dates US recessions, the **National Bureau of Economic Research** (a private group of economists), often does not call a recession until it is more than a year underway. By the time the regime is *certain*, the move you wanted to catch is mostly over.

This is the problem a dashboard solves. A **dashboard** is a fixed, repeatable set of indicators you check on a schedule to **nowcast** the regime — to estimate, from signals available *right now*, which growth-and-inflation quadrant the economy currently sits in, rather than which one it sat in last quarter. The diagram below is the mental model for the whole post: three dials — growth, inflation, and a third dial for policy and liquidity — each of which you can read from public data, combining into a single scorecard that points at one of four quadrants, and therefore at a tilt. The rest of this post is the user's manual for that dashboard.

![Three dials of growth inflation and policy feeding a scorecard that points to one of four regime quadrants](/imgs/blogs/reading-the-regime-in-real-time-the-dashboard-1.png)

This is the capstone of the timing track of the Cross-Asset Playbook. Earlier posts built the theory: [the business cycle and the Investment Clock](/blog/trading/cross-asset/the-business-cycle-and-the-investment-clock) taught you *that* a different asset leads in each growth-and-inflation phase, and [correlation by regime](/blog/trading/cross-asset/correlation-by-regime-growth-and-inflation) taught you *how* the relationships between assets re-wire when the regime flips. Those posts answered *what to own when*. This one answers the harder, more practical question: **how do you know which "when" you are in, in real time, before the data confirms it?** By the end you will have a concrete checklist — a list of dials, the levels that matter on each, and a procedure for turning their combined read into a portfolio tilt — that you can run in twenty minutes on the first weekend of every month.

## Foundations: why you need a dashboard, and the three dials it reads

Before we wire up the dials, let's define the job from absolute zero. No economics background is assumed. The whole framework rests on three ideas: that the regime is what matters, that it is invisible in real time, and that it can nonetheless be triangulated from a small set of readable signals.

### What "the regime" is, and why it decides everything

A **regime** here means the prevailing combination of two forces: the *direction of economic growth* (is the economy speeding up or slowing down?) and the *direction of inflation* (are prices accelerating or decelerating?). That is it — two axes, each with a direction, giving four combinations. The reason this two-by-two grid is so powerful is that **each asset class has a favorite regime**. Stocks love growth and falling inflation. Government bonds love falling inflation and falling interest rates. Commodities love rising inflation and strong demand. Cash and gold love chaos and rising real yields — the inflation-adjusted return on safe assets, which we will define properly below.

Because each asset has a favorite quadrant, knowing which quadrant you are in is worth more than knowing almost anything else about an individual asset. In 2022, an investor who simply identified the quadrant — growth slowing, inflation surging — would have owned commodities (**+16.1%**) and cash (**+1.5%**) instead of the classic 60/40 mix of 60% stocks and 40% bonds, which fell roughly **−16%**, its worst year since 1937. The regime call dominated the stock-picking.

A subtle but load-bearing point, carried over from the Investment Clock post: **direction beats level.** An economy with 4% inflation that is *falling* toward 2% behaves like a recovery; an economy with 2% inflation that is *rising* toward 4% behaves like an overheat. The dashboard is a map of momentum, not altitude. Every dial we read, we read for its *direction* and its *rate of change*, not just its current value.

### Why the regime is invisible in real time

Here is the uncomfortable truth that makes a dashboard necessary rather than optional. The cleanest measures of the regime are the slowest to arrive:

- **Inflation (CPI)** is published once a month, roughly two weeks after the month it describes. So in mid-June you are reading May's number, and you are reading a single noisy print that may be revised.
- **Payrolls** — the monthly count of jobs added — comes out about a week into the following month and is then revised *twice* over the next two months, sometimes by hundreds of thousands of jobs.
- **GDP** — the broadest measure of growth — arrives roughly a month after the *quarter* ends, and the first estimate is revised twice more.
- **Recession dating** by the NBER often lags the actual onset by **6 to 18 months**.

So if you wait for the data to *confirm* the regime, you are acting on information that is weeks to quarters stale — and the market, which prices in expectations continuously, has usually already moved. This is why the dashboard splits into two families of signals, a distinction that runs through the entire post: **macro data** (the official growth and inflation numbers — accurate but slow) and **market-based signals** (prices that re-rate the regime in real time — fast but noisy). The art is using the fast signals to *see* the turn and the slow signals to *confirm* it.

It helps to understand *why* the two families have opposite speed-and-accuracy profiles, because that is what tells you how to weight them. A data release like CPI is a careful *measurement* of something that already happened: the statisticians collect prices over a month, average them, seasonally adjust them, and publish weeks later — accurate, but backward-looking by construction. A market price is the opposite: it is a live *wager* on what will happen, set by millions of participants staking real money on their best guess, updated continuously. The price is noisier because it includes everyone's mistakes and emotions, but it is faster because it does not wait for the event to finish before reacting. So the two families are not redundant — they are complementary. The market gives you a fast, noisy estimate; the data gives you a slow, clean confirmation; and the dashboard's discipline is to act tentatively on the first and scale up on the second, never demanding the certainty of the data before you do anything at all.

### The three dials

We pin the regime down with three dials, shown in the figure above:

1. **The GROWTH dial** — is the economy speeding up or slowing down? Read from PMIs, payrolls and jobless claims, the yield-curve slope, and the copper/gold ratio.
2. **The INFLATION dial** — are prices accelerating or decelerating? Read from the CPI/PCE trend, breakeven inflation, oil and commodity prices, and wage growth.
3. **The POLICY/LIQUIDITY dial** — is money getting easier or tighter? Read from what the Fed is doing, real yields, financial conditions indices, the dollar, and credit spreads.

The first two dials place you on the growth-and-inflation grid. The third dial — policy and liquidity — is the throttle that tells you *how forcefully* the regime is being pushed, and it is often the thing that turns the other two. When the Fed tightens hard, it bends growth down and inflation down with a lag; when it floods the system with liquidity, it bends them up. A reader who tracks only growth and inflation but ignores the policy dial will be repeatedly blindsided by the Fed.

#### Worked example: how much being one quadrant early is worth

Let's make the cost of lag concrete. Suppose you run a simple \$100,000 portfolio and you are deciding between the classic 60/40 mix and a regime-aware tilt, in the year 2022. The 60/40 returned about **−16%**, so by year-end it was worth \$100,000 × (1 − 0.16) = **\$84,000** — a loss of \$16,000.

Now suppose your dashboard flagged the stagflation regime by mid-year (we will see exactly how, later) and you tilted to a defensive mix: cut equities, add commodities and cash. A simple such mix — say one-third commodities (**+16.1%**), one-third cash (**+1.5%**), and one-third short-duration bonds (roughly flat, call it **−2%**) — returned about (0.161 + 0.015 − 0.02) / 3 = **+0.052**, or **+5.2%**. That portfolio ended the year at \$100,000 × 1.052 = **\$105,200**.

The gap between the two outcomes is \$105,200 − \$84,000 = **\$21,200** on a \$100,000 base — more than 20% of the portfolio — and it came not from picking better stocks but from *reading the regime one or two months before the crowd capitulated*. The intuition: in a regime shift, the dashboard's payoff is not a few basis points of edge; it is the difference between owning the assets that win and the assets that lose, which is enormous.

## The GROWTH gauges: is the economy speeding up or slowing down?

The first dial answers a single question: which way is growth moving? Four gauges read it, ranging from official-but-slow to market-based-and-fast. No single one is reliable alone; together they form a robust read. The figure below lays out the four.

![Four growth gauges PMIs jobless claims yield curve slope and copper gold ratio with their read levels](/imgs/blogs/reading-the-regime-in-real-time-the-dashboard-2.png)

### PMIs: the 50 line

A **PMI** — Purchasing Managers' Index — is a monthly survey of the executives who buy materials for companies, asking whether activity (new orders, production, employment, deliveries) is rising or falling versus last month. The most-watched in the US is the **ISM Manufacturing PMI**, published by the Institute for Supply Management. Its genius is a single threshold: **50 is the line.** Above 50, a majority of firms report expansion; below 50, contraction. The further from 50, the stronger the signal.

PMIs are the workhorse growth gauge because they are timely (out on the first business day of the next month, ahead of most hard data), forward-looking (they ask about orders, which precede production), and they have that clean 50 line you can read at a glance. Watch two things: the *level* relative to 50, and the *direction*. A PMI of 52 and falling is a different message than 48 and rising.

The 2021–2024 sequence in the ISM tells the recent regime story cleanly. It ran hot in the post-COVID boom — **63.7** in March 2021 — then rolled over through **58.8** (December 2021) to **53.0** (mid-2022), crossed below the line to **48.4** by December 2022, and bottomed around **46.0** in mid-2023 before grinding back toward the line at **49.2** by late 2024. A dashboard reader watching that descent saw growth weakening through 2022 in real time, months before the GDP statistics would have made it official.

### Payrolls and jobless claims

**Payrolls** is the monthly net change in the number of US jobs. **Jobless claims** is the *weekly* count of people newly filing for unemployment benefits. These read the labor market, which is the heart of the growth picture: when firms are hiring, the economy is expanding; when they start cutting, it is rolling over.

The tradeoff is timeliness versus reliability. Claims are nearly real-time (weekly, two-week lag) but noisy week to week. Payrolls are monthly, heavily revised, and the unemployment rate they feed tends to turn *late* — joblessness often does not rise until a slowdown is well advanced. In the 2022–2024 cycle, unemployment crept from **3.5%** (early 2022) up to **4.1%** (late 2024) and **4.3%** by 2026 — a slow, clean uptrend that confirmed weakening growth but lagged the faster gauges. Use the labor data as *confirmation*, not as your early-warning system.

### The yield-curve slope

This is the single most famous growth gauge, and it is market-based, which makes it fast. The **yield curve** plots the interest rate (yield) the US government pays to borrow across maturities — 2 years, 10 years, and so on. The **slope** we watch is the 10-year yield minus the 2-year yield (written **2s10s**). Normally it is positive: long-term loans pay more than short-term ones, to compensate for the longer wait.

When the slope goes **negative — inverts** — short rates exceed long rates, which is the bond market's way of saying it expects the Fed to be *cutting* rates in the future, which it only does when it expects the economy to weaken. An inverted curve has preceded every US recession of the last half-century: it inverted before 2001, before 2008, before 2020, and then *deeply* — to about **−1.0%** (specifically −1.08% in July 2023, the most inverted since 1981) — in 2022–2023, before re-steepening in 2024. We will give this its own chart shortly, because the curve is also a market-internal, and the way it un-inverts is itself a signal.

### The copper/gold ratio: the market's growth vote

The last growth gauge is purely market-based, and it is a favorite of macro traders precisely because it sidesteps the data lag entirely. **Copper** is the most economically sensitive metal — it goes into wiring, plumbing, motors, construction — so its price rises when the world expects more building and manufacturing. It is nicknamed "Dr. Copper" for its supposed PhD in economics. **Gold**, by contrast, is a fear-and-money asset: it rises when people want safety and when real yields fall. So the **copper/gold ratio** — the price of copper divided by the price of gold — rises when growth optimism beats fear (reflation) and falls when fear beats optimism (slowdown). It is, in effect, a continuous referendum on growth that updates every second the metals markets are open. When the data and the copper/gold ratio disagree, the ratio has usually moved first.

Why divide the two rather than just watch copper? Because the ratio cancels out the forces common to *both* metals — a general commodity boom, dollar weakness, a liquidity flood — and isolates the one thing that distinguishes them: growth optimism versus fear. Copper alone can rise simply because all commodities are rising on a weak dollar; the *ratio* rises only when copper is outpacing gold, which happens specifically when the market is voting for growth over safety. That is what makes it a cleaner growth read than copper's raw price. As reference points, copper traded around **\$2.80/lb** in the slow 2020 spring and climbed to **\$4.23/lb** in the 2021 reflation, while gold ran from about **\$1,770** to **\$1,799/oz** over the same span — so the ratio jumped, correctly flagging the growth surge before the hard data caught up.

#### Worked example: reading the PMI cross with a simple rule

Let's build a tiny, mechanical rule from the PMI to show how a gauge becomes a decision. Suppose your rule is: *"Growth dial = UP while ISM is above 50 or rising for two straight months; DOWN while ISM is below 50 or falling for two straight months."* Apply it to the real 2021–2023 path.

In March 2021 the ISM is **63.7** — far above 50 — so the dial reads UP. By December 2021 it is **58.8**: still above 50, but it has fallen from the peak. Two more falling months in early 2022 flip your rule's second clause, and by mid-2022, with the ISM at **53.0** and still sliding, the dial flips to DOWN. It crosses below 50 to **48.4** in December 2022, confirming the DOWN read, and bottoms at **46.0** in mid-2023.

So the rule had you reading "growth slowing" from roughly mid-2022 — months before the official GDP and unemployment statistics caught up. A reader who had instead waited for two consecutive quarters of falling GDP, or for unemployment to rise meaningfully (which did not happen until 2024), would have been a year late. The intuition: a simple, written-down rule on a fast gauge beats a sophisticated judgment on a slow one, because the rule fires while the move is still happening.

## The INFLATION gauges: are prices accelerating or decelerating?

The second dial answers: which way is inflation moving? Again, four gauges, again ranging from slow-official to fast-market. The figure below lays them out.

![Four inflation gauges CPI trend breakevens oil prices and wage growth with their read speeds](/imgs/blogs/reading-the-regime-in-real-time-the-dashboard-3.png)

### The CPI/PCE trend

**CPI** — the Consumer Price Index — measures the average price of a basket of goods and services a household buys; its year-over-year change is the headline inflation number. **PCE** — Personal Consumption Expenditures — is a related measure that the Fed actually prefers, because it adjusts for how people substitute between goods. The two move together; CPI is the one in the headlines, PCE the one the Fed targets (at 2%).

For the dashboard, what matters is the **trend** — the direction and slope — not the single print. The 2021–2024 CPI path is the textbook example of an inflation dial swinging across the grid: it ran from **4.99%** (May 2021) up to a 40-year high of **9.06%** (June 2022), then back down to **2.97%** (June 2023) and held near **2.89%** (December 2024). A dashboard reader watching that *rise* through 2021 and the first half of 2022 saw the inflation dial pointing firmly UP — the half of the regime that, combined with the growth dial turning DOWN, defines stagflation.

A practical refinement: watch **core** inflation (which strips out volatile food and energy) for the underlying trend, and watch the *3-month annualized* rate, not just the year-over-year, because the year-over-year number is slow to reflect a turn (it averages in old months). Core PCE peaked at **5.6%** in February 2022 and fell to **2.8%** by end-2024 — a cleaner read of the disinflation than the noisier headline.

### Breakeven inflation: the market's forecast

Here is the market-based, real-time inflation gauge. The US Treasury issues two kinds of bonds: ordinary ones (nominal) and **TIPS** — Treasury Inflation-Protected Securities — whose payments rise with the CPI. The gap between the yield on a 10-year nominal bond and a 10-year TIPS is the **breakeven inflation rate**: the average inflation rate over the next 10 years at which you would be indifferent between the two. It is, literally, the bond market's inflation forecast, and it updates every second markets are open. When breakevens are rising, the market is pricing higher inflation ahead — often *before* the CPI prints confirm it. We will see the cousin of this — the *real* yield — in the policy section.

### Oil and commodity prices

Energy is the inflation engine. **Oil** (WTI crude is the US benchmark) feeds directly into gasoline and heating costs, which are a big chunk of the CPI, and indirectly into the cost of making and shipping almost everything. So a spike in oil flows into headline inflation within weeks. The recent path is dramatic: WTI ran from about **\$26** (early 2016) and **\$45** (late 2018) to a negative print in April 2020, then surged to **\$124** in March 2022 as Russia invaded Ukraine — directly feeding the 2022 inflation spike — before easing back to about **\$72** by late 2024. When oil and broad commodity indices are climbing, the inflation dial is getting a fast, real-time push upward.

### Wage growth: the stickiest signal

Wages are the *slowest* inflation gauge, and that is exactly why they matter. When prices rise, workers eventually demand higher pay; once wages rise, businesses raise prices to cover them; and round it goes — a **wage-price spiral**. Because wages are sticky (employers are reluctant to cut them), wage growth is the last thing to rise in an inflation and the last thing to fall in a disinflation. So wage growth *confirms* the inflation trend but never leads it. A dashboard reader uses wages to answer "is this inflation entrenched or transitory?" — entrenched inflation shows up in wages; a one-off supply shock does not.

#### Worked example: computing a breakeven and reading it

Let's compute a breakeven so the gauge stops being abstract. Suppose the 10-year nominal Treasury yields **4.0%** and the 10-year TIPS yields **1.7%** (a level the real yield actually reached in late 2022). The breakeven inflation rate is simply the difference:

$$\text{breakeven} = y_{\text{nominal}} - y_{\text{TIPS}} = 4.0\% - 1.7\% = 2.3\%$$

So the market is pricing average inflation of **2.3%** per year over the next decade. Now here is how you read it as a dial. If next week the nominal yield rises to **4.3%** while TIPS hold at **1.7%**, the breakeven jumps to **2.6%** — the market just raised its inflation forecast by 0.3 percentage points, *in real time*, with no CPI print involved. Conversely, if the nominal yield falls to 3.7% with TIPS unchanged, the breakeven drops to **2.0%** — the market is pricing disinflation.

Put a \$1,000 number on why this matters to a bondholder. If you own a \$1,000 nominal 10-year bond and breakevens (and thus nominal yields) rise from 2.3% to 3.3% inflation expectations, the bond's price falls roughly by its duration times the yield change — for a 10-year bond, duration is about 8, so a 1-percentage-point yield rise costs about 8% of price, or **−\$80** on your \$1,000. The intuition: the breakeven is not just a forecast; it is the live wire that moves bond prices the instant inflation expectations shift, which is why it leads the CPI.

## The POLICY/LIQUIDITY dial: is money getting easier or tighter?

The third dial does not place you on the growth-and-inflation grid directly — it tells you how hard the regime is being *pushed*, and it is frequently the cause of the next turn. Money getting tighter slows growth and inflation with a lag; money getting easier accelerates them. Five gauges read this dial.

### The Fed: hiking or cutting

The most direct read is simply what the **Fed** is doing to its policy rate (the federal funds rate). Hiking = tightening = leaning against growth and inflation. Cutting = easing = supporting them. The 2022–2024 cycle was the most violent in 40 years: the Fed took the upper bound of its target range from **0.25%** (early 2022) to **5.50%** (mid-2023) in about 16 months, then began cutting to **4.50%** by December 2024. The *direction* and *pace* of the Fed are the master setting of the policy dial. For the mechanics of how the Fed actually moves rates, see [how the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates).

### Real yields: the cleanest policy read

The single best gauge of the policy *stance* is the **real yield** — the inflation-adjusted return on a safe bond, read straight off the 10-year TIPS. Roughly, real yield ≈ nominal yield − expected inflation. When the real yield is *negative*, holding safe assets loses you purchasing power, which pushes investors out into risk: that is *easy* policy. When the real yield is *positive and high*, safe assets pay you a real return, which pulls money out of risk and into bonds and cash: that is *tight* policy.

The 2021–2024 path of the 10-year real yield is the clearest possible picture of the policy dial swinging from easy to tight: it ran from **−1.04%** (December 2021 — deeply negative, deeply easy) to **+1.74%** (October 2022 — sharply positive, sharply tight) and on to **+2.20%** (December 2024). That is a roughly 3-percentage-point tightening in the real cost of money — an enormous move that, by itself, re-priced every asset on Earth. Because this variable is so central, it has its own post: [real yields, the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything).

### Financial conditions, the dollar, and credit spreads

Three more gauges round out the policy/liquidity dial, and all three are market-based and fast:

- **Financial conditions indices** bundle rates, credit spreads, equity levels and the dollar into a single number measuring how easy or tight overall financial conditions are. Tightening conditions = a headwind for growth.
- **The dollar (DXY)** — the US Dollar Index, the dollar's value against a basket of major currencies — is a global tightening gauge. A *rising* dollar drains liquidity from the rest of the world (dollars get scarcer and more expensive), so it tightens conditions; a *falling* dollar loosens them. The dollar peaked near **114.8** in September 2022 — the same month real yields and the Fed were tightening hardest — and that surge was a major reason 2022 was so brutal for risk assets worldwide. See [the dollar: cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity) for the full mechanism.
- **Credit spreads** — the extra yield risky corporate bonds pay over safe Treasuries — widen when lenders get nervous and narrow when they get comfortable. They are a real-time read on financial stress, and we treat them in detail in the next section as a market-internal.

A useful habit is to read the policy dial as a single composite: when *most* of these gauges point the same way — Fed hiking, real yields rising, dollar climbing, conditions tightening, spreads widening — the policy push is unambiguous and powerful, and it will bend growth and inflation with a lag of roughly 6 to 18 months. When they conflict — say the Fed is on hold but the dollar is falling and spreads are tightening — the net push is mild, and the growth-and-inflation dials are more likely to drift than to swing. The policy dial is less about any one gauge's exact level and more about whether the gauges are *aligned*.

#### Worked example: how the real yield re-prices a stock

The policy dial sounds abstract until you see it move a price. A stock is, at bottom, a claim on future cash flows, and those future dollars are worth less today the higher the rate you discount them at — and the discount rate moves with the real yield. Take a simple growth stock that you expect to pay \$10 of cash flow per year, growing modestly, far into the future. A rough valuation is the cash flow divided by the discount rate minus the growth rate. Suppose the growth rate is **2%**.

In December 2021, the 10-year real yield was **−1.04%**, so the discount rate was deeply low — say a real discount rate of about **3%** once you add a risk premium. The stock is worth roughly \$10 / (0.03 − 0.02) = \$10 / 0.01 = **\$1,000**. Now run the policy dial forward to October 2022: the real yield has surged to **+1.74%**, lifting the discount rate to about **5.8%**. The same \$10 cash flow is now worth \$10 / (0.058 − 0.02) = \$10 / 0.038 = **\$263**.

The stock's *fundamentals* did not change — same \$10, same 2% growth — yet its fair value fell from \$1,000 to \$263, a **−74%** repricing, driven entirely by the policy dial swinging from easy to tight. That arithmetic is why the highest-growth, longest-duration stocks were hit hardest in 2022: their value lives furthest in the future, so a rising real yield discounts it most. Reading the real-yield dial told you, in real time, that long-duration equities were about to be repriced — no earnings forecast required.

## The market-internals that vote fastest

Now we get to the heart of the dashboard's edge over the official data. The macro gauges above are accurate but slow; the **market-internals** are prices, so they re-rate the regime *continuously*. When data and markets disagree, markets usually move first — not because markets are clairvoyant, but because a price is a real-time aggregation of everyone's bets, while a data release is a backward-looking measurement. The figure below makes the speed difference explicit.

![Two columns contrasting slow lagging official data against fast leading market based signals](/imgs/blogs/reading-the-regime-in-real-time-the-dashboard-6.png)

Six market-internals do most of the work. Several already appeared as gauges above — that is the point: the best gauges are the ones that are *also* tradeable prices, because those are the ones that move first.

### The yield curve (again, but now as a market signal)

We met the 2s10s slope as a growth gauge. As a market-internal it has an extra wrinkle worth its own chart: *how* it un-inverts matters. The curve usually inverts well before a recession, then **re-steepens** as the recession approaches — because the front end (the 2-year) falls fast when the market starts pricing Fed cuts. So an *inverted* curve is a slow-burning warning, but a *rapidly re-steepening* curve (the 2-year dropping toward the 10-year) is often the more urgent signal that the slowdown is arriving. The chart below tracks the real 2s10s through its 2022–2024 cycle.

![Yield curve 10y minus 2y spread over time with the inversion zone shaded red and the deepest point marked](/imgs/blogs/reading-the-regime-in-real-time-the-dashboard-4.png)

Read the shape: a clean march from **+1.20** (mid-2021) down through zero in July 2022 into a deepening inversion that bottomed at **−1.08** in July 2023 — the deepest since 1981 — followed by a re-steepening back above zero to **+0.33** by December 2024. The red band is the inversion zone, the bond market's recession warning, lit for roughly two and a half years.

### Credit spreads: the stress gauge

The **high-yield credit spread** — specifically the ICE BofA US High Yield Option-Adjusted Spread, or **HY OAS** — is the extra yield that bonds issued by riskier ("junk") companies pay over safe Treasuries, expressed in **basis points** (a basis point is one hundredth of a percent, so 100 bps = 1%). It is arguably the single best real-time gauge of financial stress, because it directly measures how much lenders demand to take corporate default risk. The calibration:

- **Calm: ~300–400 bps.** Lenders are comfortable; credit is flowing; risk-on.
- **Stress: >800 bps.** Lenders are nervous; defaults are being priced in; risk-off.
- **Crisis: >1100–2000 bps.** Panic. The HY OAS hit about **1100 bps** in March 2020 (COVID) and about **2000 bps** in 2008 (the global financial crisis).

When the HY OAS is widening, the policy/liquidity dial is tightening *de facto* regardless of what the Fed is doing, because risky borrowers are being cut off. Widening credit spreads led almost every major risk-off episode of the past two decades. The mechanics and history of credit spreads get the full treatment in the corporate-credit post of this series.

### The dollar (DXY)

Covered above as a policy gauge; as a market-internal, the dollar is a fast vote on global risk and liquidity. A surging dollar (toward that **114.8** peak in 2022) is risk-off and tightening; a falling dollar is risk-on and loosening. The dollar tends to *lead*, because it re-prices the instant global capital shifts toward or away from safety.

### The copper/gold ratio

Covered above as a growth gauge; as a market-internal, it is a real-time growth vote that often turns before the PMIs do, because the metals markets price expectations continuously while the PMI is a once-a-month survey.

### The VIX: the fear dial

The **VIX** is the market's expected volatility of the S&P 500 over the next 30 days, derived from options prices — informally, the "fear gauge." Its calibration is well-worn:

- **Long-run average: ~19.5.**
- **Calm: <15.** Complacency, risk-on.
- **Stressed: >25.** Anxiety rising.
- **Panic: >40.** It spiked to about **82.7** in March 2020 (COVID), **37.3** in February 2018 (the "volmageddon" spike), and **65.7** in August 2024 (the yen-carry-unwind scare).

The VIX is the fastest of all — it moves in minutes — but it is also the noisiest and the most prone to false alarms, so it confirms stress rather than predicting regime *direction*. A high VIX tells you risk is being repriced *right now*; it does not by itself tell you whether you are entering stagflation or recession.

### Equity-sector leadership and breadth

The subtlest market-internal is *which stocks are leading*. In a healthy expansion, **cyclical** sectors — those whose fortunes swing with the economy, like industrials, materials, consumer discretionary, banks — lead. When the market starts to fear a slowdown, leadership rotates to **defensives** — utilities, consumer staples, healthcare — whose earnings hold up in a downturn. This **defensives-over-cyclicals** rotation is a real-time regime vote *inside* the stock market: it flips toward defense as the growth dial turns down, often before the PMIs confirm it. Closely related is **breadth** — the share of stocks participating in a rally. Narrowing breadth (a few giant stocks holding the index up while most stocks fall) is a classic late-cycle warning that the rally is running out of fuel. For the deeper mechanics of how money rotates risk-on and risk-off, see [risk-on, risk-off: how money rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates).

#### Worked example: the HY spread as a position-sizing dial

Let's turn the HY OAS into a concrete risk decision. Suppose you hold \$100,000 of high-yield bonds, and you size your credit exposure by a simple rule tied to the spread: full size when the HY OAS is calm (<400 bps), half size when it crosses into stress (>600 bps), and out when it goes >1000 bps.

In a calm January the HY OAS sits at **350 bps**, so you hold the full \$100,000. Over the spring it widens to **650 bps** as the growth dial turns down and the VIX rises — your rule cuts you to **\$50,000**, moving \$50,000 into cash. Then a stress event hits and the spread blows out to **1100 bps** (the 2020 level) — your rule takes you to **\$0** in high yield.

Now price the avoided pain. High-yield bonds fell about **−11.2%** in 2022 and about **−26.2%** in 2008. Had you stayed full size into a −26.2% move, your \$100,000 would have become \$73,800 — a **−\$26,200** loss. By cutting as the spread widened, you sidestepped most of it: you were already half-sized at \$50,000 when the worst hit, and out before the −26%. The intuition: the HY spread is not just a thermometer — read mechanically, it is a position-sizing dial that tells you to take risk off *while* stress is building, not after it has peaked.

## Putting it together: the scorecard and the four quadrants

Three dials, a dozen gauges. The dashboard's job is to collapse all of that into a single read: **which quadrant am I in?** The procedure is deliberately mechanical — you are not forecasting, you are scoring what the dials currently say.

Score the two axes:

- **Growth axis (UP or DOWN):** Take the four growth gauges — ISM relative to 50 and its direction, jobless claims direction, the curve slope, the copper/gold ratio direction. If a majority point to acceleration, growth = UP; if a majority point to deceleration, growth = DOWN.
- **Inflation axis (UP or DOWN):** Take the four inflation gauges — CPI/PCE trend (3-month annualized), breakevens direction, oil/commodity direction, wage growth. If a majority point up, inflation = UP; if down, inflation = DOWN.

The two axes give four quadrants, and the policy dial tells you how hard the regime is being pushed. Here is the mapping, which is the spine of the entire framework:

| Quadrant | Growth | Inflation | Lead asset | Tilt |
|---|---|---|---|---|
| **Goldilocks** (Recovery) | UP | DOWN | Stocks, credit | Overweight equities, add high yield, trim cash |
| **Overheat** | UP | UP | Commodities, energy | Add commodities, energy, TIPS, value; cut long duration |
| **Stagflation** | DOWN | UP | Cash, gold, energy | Raise cash, add gold and energy; cut growth stocks and duration |
| **Recession** | DOWN | DOWN | Bonds, duration | Buy duration and quality bonds, defensives; trim cyclicals and high yield |

The figure below is the same mapping in visual form — quadrant, signature dashboard read, lead asset, and tilt — so that once the dials place you, the action is a lookup rather than a fresh decision.

![Matrix mapping four regime quadrants to dashboard reads lead assets and portfolio tilts](/imgs/blogs/reading-the-regime-in-real-time-the-dashboard-7.png)

A few rules for scoring honestly:

- **Weight the fast signals for *seeing* the turn, the slow ones for *confirming* it.** The copper/gold ratio and breakevens flip first; the CPI and unemployment confirm last. If the fast gauges have turned but the slow ones have not, you are in the *transition* — the highest-value moment to act, and the moment you most want a written rule so you do not freeze.
- **Direction beats level, everywhere.** A 6% CPI that is *falling* fast is a disinflation read; a 3% CPI that is *rising* is an inflation read.
- **The policy dial breaks ties and sets urgency.** Two economies with the same growth-and-inflation read but opposite policy stances (one with the Fed slashing, one with the Fed hiking into the slowdown) call for very different durations of the tilt.

What do you do when the gauges within an axis disagree — say the ISM is below 50 but the copper/gold ratio is rising, so the growth dial is split two-to-two? Three rules resolve it. First, **go with the majority** when there is one; a single dissenting gauge is usually noise. Second, when there is no majority, **default to the faster signals** for the early read and wait for the slower ones to break the tie — the market-internals are wrong less often at turning points than a single survey. Third, **treat a split as information in itself**: a divided dial often means you are *in the transition* between quadrants, which is precisely the moment to size a tilt tentatively rather than fully, and to watch the board more closely than your monthly cadence would normally require. A split is not a failure of the dashboard; it is the dashboard telling you the regime is mid-flip.

#### Worked example: scoring the dashboard in mid-2022

Let's run the full scorecard for **June 2022**, using only data that was public *at the time*, and watch the quadrant fall out. The figure below is this exact read.

![Four dials in mid-2022 all pointing to the stagflation verdict and the mapped defensive tilt](/imgs/blogs/reading-the-regime-in-real-time-the-dashboard-5.png)

Score the **inflation axis** first. CPI had just printed **9.06%** — a 40-year high — and it was *rising* (up from 4.99% a year earlier). Breakevens were elevated. Oil had spiked to **\$124** in March. Wages were climbing. All four inflation gauges point UP. **Inflation = UP.**

Now the **growth axis**. The ISM had fallen from 63.7 (early 2021) to **53.0** and was sliding toward the 50 line. The yield curve was about to invert (it crossed zero in July 2022). The copper/gold ratio was falling as growth fears mounted. A majority point to deceleration. **Growth = DOWN.**

Growth DOWN, inflation UP places you squarely in the **Stagflation** quadrant. The **policy dial** confirms the urgency and the duration of the tilt: the 10-year real yield was surging from **−1.04%** toward **+1.74%**, the Fed was hiking from 0.25% toward 5.50%, the dollar was charging toward **114.8**, and the HY OAS was widening — every policy gauge screaming *tightening*. Defensives were already outperforming cyclicals inside the equity market, the market-internal vote confirming the macro read.

So the scorecard, read mechanically, said: stagflation, with a hard tightening push. The mapped tilt — cut duration and growth stocks, add commodities, energy, gold and cash — would have turned the 60/40's **−16%** year into roughly flat-to-positive. Put the dollar number on it once more: a \$100,000 60/40 became \$84,000; the stagflation-tilted mix from our earlier example became about \$105,200. The intuition that closes the whole post: *every single one of those dials was public and readable in June 2022.* No forecast was required — just a checklist, scored honestly, in time to act.

## Common misconceptions

A dashboard fails most often not because the gauges are wrong but because the reader misuses them. Here are the five mistakes that do the most damage, each corrected with a number.

**"The regime is whatever the latest headline says."** No — the regime is the *direction* of growth and inflation, read from a basket of gauges, and it changes far more slowly than the news cycle. In 2022 the headlines oscillated daily between "inflation peaking" and "recession imminent," but the dashboard read — growth down, inflation up — held steady for most of the year. A reader who chased headlines was whipsawed; a reader who scored the dials monthly was not. Check on a fixed cadence; ignore the noise in between.

**"Wait for the data to confirm before you act."** This sounds prudent and is often ruinous. The official data lags by weeks (CPI), months (revised payrolls), or a quarter (GDP), and the NBER dates recessions 6–18 months late. By the time the data confirms, the move is largely over — the S&P bottomed in March 2009 while the economy was still shedding jobs. Use the fast market-internals to *see* the turn and the data to *confirm* it; do not wait for confirmation to act at all.

**"An inverted yield curve means sell stocks now."** The curve inverts *early* — often a year or more before the recession — and stocks frequently rise during the inversion. The 2s10s inverted in July 2022 and stayed inverted through 2023, a period in which stocks fell in 2022 but rose sharply in 2023. The inversion is a *warning to start watching the other dials*, not a same-day sell signal. The more urgent signal is the *re-steepening*, when the front end collapses as cuts get priced.

**"More indicators are always better."** A dashboard with 50 gauges is unreadable and contradicts itself constantly; the noise drowns the signal. The dozen gauges here are chosen because each reads a distinct facet (growth vs. inflation vs. policy; fast vs. slow). Adding a 20th growth gauge that correlates 0.95 with the PMI adds nothing but the illusion of rigor. Fewer, well-chosen, diverse gauges beat a wall of redundant ones.

**"Market signals are noise; only the hard data is real."** The opposite error. Yes, the VIX throws false alarms and the copper/gold ratio is volatile — but in aggregate, prices are a real-time aggregation of every participant's information, and they have led the official data at every major turn. The dollar surged and credit spreads widened *before* the 2022 damage showed up in the GDP statistics. Markets are noisy *and* fast; the discipline is to weight them for speed while waiting for the slow data to confirm, not to dismiss them.

## How it shows up in real markets

Five episodes, each showing the dashboard's dials lighting up in real time — and what a reader who scored them honestly would have seen.

**2022 — the stagflation year (the worked case, in the wild).** Every dial we have discussed fired in unison. CPI ran to **9.06%** and rising (inflation UP); the ISM fell from 53.0 toward 48.4 (growth DOWN); the 10-year real yield surged **−1.04% → +1.74%** and the Fed hiked **0.25% → 4.50%** by year-end (policy tightening hard); the dollar peaked at **114.8**; HY spreads widened; defensives beat cyclicals. The scorecard read stagflation by mid-year, and the assets it pointed to — commodities (**+16.1%**), cash (**+1.5%**), energy — were the only winners while the 60/40 fell **−16%**. This is the dashboard's cleanest validation: a slow-motion regime shift, fully readable from public dials, that crushed the assets the textbook 60/40 was overweight.

**October 2008 — the recession/flight-to-quality.** The HY OAS blew out toward **2000 bps**, the VIX spiked toward the 80s, the dollar surged (global dash for dollars), and the curve had inverted in 2006–2007 and was re-steepening violently. Every stress dial was maxed. The dashboard read recession with acute liquidity stress — growth DOWN, inflation about to roll DOWN — and the one asset it pointed to, long Treasuries, returned **+25.9%** while the S&P fell **−37%**. A reader who weighted the market-internals saw the crisis in the spreads and the VIX *as it built*, not after the GDP confirmed it.

**March 2020 — COVID, the fastest regime shift ever.** In a matter of weeks the VIX hit **82.7**, the HY OAS spiked to **~1100 bps**, oil collapsed (eventually to a *negative* print in April), and the S&P fell **−33.9%** peak to trough. The dashboard's fast market-internals registered the recession instantly — far ahead of any data, since the official numbers for March did not exist yet. Then the policy dial inverted just as fast: the Fed slashed to 0.25% and flooded the system with liquidity, real yields went deeply negative, and the regime flipped to recovery within months — which a reader watching the policy dial and credit spreads (rapidly *tightening* back in) could see in the prices long before GDP rebounded.

**August 2024 — the false alarm.** A spike in the VIX to **65.7** on a yen-carry-trade unwind looked, for a day, like the start of something. But the *other* dials did not confirm: the HY OAS barely moved, the curve was re-steepening benignly, growth gauges were stable, inflation was still falling. The dashboard's discipline — *one dial spiking is not a regime change; you need a majority* — kept a reader from panic-selling into what proved to be a brief, technical air-pocket. This is the value of scoring the whole board: it filters the VIX's false alarms.

**2024 — the re-steepening and the soft landing.** Through 2024 the curve un-inverted from **−0.50** back to **+0.33**, the ISM ground back toward 50 (**49.2**), CPI held near **2.89%** (inflation DOWN and contained), and the Fed began cutting (**5.50% → 4.50%**). The dials read a turn from the late-cycle/recession-watch quadrant toward recovery — growth stabilizing, inflation contained, policy easing — which favored a rotation back toward equities and credit. The episode shows the dashboard working in the *less* dramatic direction: confirming that the stress had passed and it was time to add risk back.

**2021 — the overheat that the dashboard caught and the Fed missed.** This is the cautionary tale, because the famous voice in the room got it wrong while the dials got it right. Through 2021 the Fed insisted inflation was "transitory" and kept policy at emergency settings. But the dashboard's inflation gauges were already lighting up: CPI ran from **4.99%** (May) to **6.81%** (November); oil climbed; commodities (BCOM) returned **+27.1%** for the year, the best of any major asset; breakevens were elevated. Meanwhile the growth dial was still UP — the ISM was at **63.7** early in the year. Growth UP and inflation UP is the **Overheat** quadrant, whose lead asset is commodities — exactly what won. A reader scoring the dials, rather than listening to the "transitory" narrative, was overweight the right asset a full year before the Fed capitulated and began the 2022 hiking cycle. The lesson: the dashboard is a check *against* the official narrative, not a follower of it.

## When to own it: the dashboard playbook

This is the payoff — the concrete, repeatable procedure. The dashboard is worthless if you check it erratically or override it on a hunch; its value comes entirely from running it the same way, on a schedule, and acting on confirmed turns.

**Check on a fixed cadence — monthly.** The regime changes on a scale of months, not days, so monthly is the right rhythm: it is frequent enough to catch turns and infrequent enough to ignore noise. Pick a date — say the first weekend after the monthly CPI and ISM releases — and score all three dials. Twenty minutes. Daily checking invites you to react to the VIX's hourly mood; monthly checking forces you to act on the regime, not the headline.

**Act on confirmed turns, not single-gauge noise.** A turn is confirmed when a *majority* of an axis's gauges agree, not when one spikes. The August 2024 VIX spike was one dial; the rest of the board said "stable," so there was no turn. Conversely, mid-2022 had all four inflation gauges up and three of four growth gauges down — an unambiguous, majority-confirmed stagflation read. Require the majority.

**Weight market signals for speed, data for confirmation.** Use the curve, credit spreads, breakevens, the dollar, copper/gold, and sector leadership to *see* the turn early; use the CPI, PMI level, payrolls, and (eventually) GDP to *confirm* it. The trade is sized up as confirmation accumulates: a fast-signal-only turn gets a *tentative* tilt; once the slow data agrees, you scale the tilt to full size.

**Map the quadrant to the tilt mechanically.** Once the dials place you, the action is the lookup in the table and figure above. Goldilocks → overweight stocks and credit. Overheat → add commodities, energy, TIPS; cut long duration. Stagflation → raise cash, add gold and energy; cut growth stocks and duration. Recession → buy duration and quality bonds and defensives; trim cyclicals and high yield. You are not re-deriving the allocation each month; you are reading which row of the table is live. The full machinery of turning these tilts into an actual portfolio — sizing, rebalancing bands, risk budgeting — is the next step, and it builds directly on this dashboard.

**Write down what would change your mind.** This is the discipline that separates a dashboard from a vibe. For every tilt, record the specific gauge readings that would invalidate it. "I am tilted to stagflation; I will reverse if the CPI 3-month annualized falls below 3% *and* the real yield stops rising *and* the ISM turns back above 50." Writing the exit condition in advance does two things: it forces honesty (you cannot move the goalposts after the fact), and it lets you act decisively when the regime *does* turn, because you decided the trigger while you were calm. The reader who keeps this log is running a dashboard; the one who does not is just watching charts.

A closing honesty note, because finance rewards humility: this is an educational framework, not individualized advice, and no dashboard is a crystal ball. The dials lag, they occasionally disagree, false signals happen (August 2024), and the mapping from quadrant to tilt is a starting point, not a guarantee — assets sometimes ignore the regime for quarters at a time. What the dashboard buys you is not certainty but *speed and discipline*: a repeatable way to read the regime from public data before the official numbers confirm it, and a written rule that keeps you from fighting the tape. In a business where the regime call dominates the security selection, that is most of the edge there is.

## Further reading and cross-links

This post is the practical capstone of the timing track — the dials that operationalize the whole framework. To go deeper on the pieces it rests on:

- [The business cycle and the Investment Clock](/blog/trading/cross-asset/the-business-cycle-and-the-investment-clock) — the four-quadrant model the dashboard reads, and why a different asset leads in each phase.
- [Correlation by regime: how the whole map re-wires](/blog/trading/cross-asset/correlation-by-regime-growth-and-inflation) — why bonds stop hedging stocks when the inflation dial points up, and how every correlation changes with the regime.
- [Real yields: the variable that prices everything](/blog/trading/cross-asset/real-yields-the-variable-that-prices-everything) — the deep version of the single most important gauge on the policy dial.
- [The dollar: cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity) — why a rising DXY tightens conditions worldwide and shows up as a market-internal that leads.
- [Risk-on, risk-off: how money rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates) — the mechanics behind the sector-leadership and credit-spread internals that vote fastest.

The natural next step is turning these regime reads into an actual portfolio — position sizes, rebalancing rules, and a risk budget — which is the subject of the allocation-and-rebalancing post that closes the series.

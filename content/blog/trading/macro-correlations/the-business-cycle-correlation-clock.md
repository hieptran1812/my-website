---
title: "The Business-Cycle Correlation Clock: Why the Map Rotates"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The business cycle is a clock, and which correlations are live rotates with it: stocks lead the recovery, commodities lead the overheat, bonds win the recession, and the stock-bond correlation flips along the way."
tags: ["macro", "correlation", "business-cycle", "investment-clock", "regime", "stock-bond-correlation", "asset-rotation", "sector-rotation", "recession", "inflation"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Asset-class leadership and the cross-asset correlation structure rotate predictably with the business cycle, so the phase you are in is the regime selector that tells you which correlations are currently live.
>
> - The cycle has four phases — **early/recovery, mid/expansion, late/overheat, recession** — and each one has a different leader: stocks lead the recovery (about +20% real), commodities lead the overheat (about +14%), and bonds plus cash win the recession (bonds about +10% while stocks fall about −10%).
> - The **stock-bond correlation is not a constant**: it is negative (bonds diversify stocks) in disinflationary expansions and positive (both fall together) in the late/overheat inflation phase. That single flip is why the 60/40 portfolio worked for two decades and then broke in 2022.
> - You date the phase with four indicators read **together** — ISM above or below 50 and rising or falling, the unemployment trend, the yield-curve slope, and the inflation regime — and the quadrant they point to tells you which correlations to trust.
> - The one fact to remember: **a correlation is a property of a regime, not of an asset pair.** Find the phase first, and the correlation map falls out of it.

In October 2022, a portfolio that had quietly delivered for forty years stopped working all at once. The classic 60/40 — 60% in stocks, 40% in bonds — is built on one assumption: when stocks fall, bonds rise, so the bond sleeve cushions the equity sleeve. For most of the 2000s and 2010s that assumption held beautifully. The correlation between the two was *negative*, often around −0.5, which is exactly what a diversifier is supposed to do. Then 2022 arrived. The S&P 500 fell 18%. And long Treasuries, the supposed cushion, fell **31%** — worse than the stocks they were meant to protect. The 60/40 lost about 16% in a single year, its worst since 1937.

Nothing was broken with bonds, and nothing was broken with the idea of diversification. What had changed was the *regime*. For two decades the dominant macro risk was weak growth, and in a growth scare stocks fall while bonds rally — a negative correlation. In 2022 the dominant risk became **inflation**, and inflation is the one shock that hits stocks and bonds *at the same time*: it forces the central bank to raise rates, which discounts both future earnings and future coupons harder. The correlation didn't drift. It flipped sign, from about −0.3 to about **+0.6**, because the economy had moved to a different point on the business cycle.

This is the synthesis post of the whole series. Everything else we have studied — the way CPI moves stocks, the way real yields move gold, the way the yield curve leads recessions, the way growth surprises move cyclicals — turns out to be *conditional on where we are in the cycle*. The relationships are real, but each one is strongest in some phases and absent or reversed in others. The business cycle is a clock, and as it turns, the correlation map rotates with it. Learn to read the clock and you have the regime selector for every other correlation in the series.

![The business cycle correlation clock with four phases and their leaders](/imgs/blogs/the-business-cycle-correlation-clock-1.png)

## Foundations: the business cycle, the investment clock, and what "correlation rotates" means

Let us build this from absolute zero, because the payoff at the end depends on the foundations being solid.

**What is the business cycle?** An economy does not grow at a steady, smooth pace. It expands for a while, runs hot, then contracts, then recovers — over and over. One full loop is the *business cycle*. The official US arbiter of when a recession starts and ends is the National Bureau of Economic Research (NBER), and post-war US expansions have lasted anywhere from about one to eleven years, while recessions are usually short — six to eighteen months. There is no fixed length. The cycle is a *sequence of phases*, not a calendar.

**The four phases.** We will split each loop into four pieces. The exact names vary across textbooks, but the substance is standard:

- **Early / Recovery** — the economy is just climbing out of a recession. Growth is turning up from a low base; unemployment is high but starting to fall; inflation is low because there is still slack (idle factories, unemployed workers); the central bank is still easy. This is the most explosive phase for stocks because everything is improving from a depressed level.
- **Mid / Expansion** — growth is solid and self-sustaining; unemployment is low and stable; inflation is tame, near the central bank's target; policy is roughly neutral. The longest and calmest phase. Stocks grind higher; the "normal" correlations hold.
- **Late / Overheat** — the economy is running above its sustainable speed. Unemployment is very low, wages and prices are rising, inflation is high and climbing, and the central bank is hiking to cool things down. Commodities and real assets shine; bonds suffer; this is where the stock-bond correlation flips positive.
- **Recession** — growth turns negative; unemployment rises fast; inflation cools (and the central bank starts cutting). Stocks and commodities fall; government bonds and cash are the only winners.

**What is the "investment clock"?** It is an old idea, popularized by Merrill Lynch in a widely-cited 2004 note, that draws these phases on a circle and places the asset class that tends to lead at each position — like the twelve hours on a clock face. The original framing was deliberately simple: as the economy moves clockwise from recovery through overheat to recession, leadership passes from stocks to commodities to bonds to cash and back. It caught on because it gave a *single picture* to a relationship that practitioners had long traded by feel — that different assets have their day at different points in the cycle. The deeper version, which we use here, organizes the phases by two axes: **growth** (rising or falling) and **inflation** (rising or falling). Those two axes make four quadrants, and each quadrant has a natural leader:

| Growth × Inflation | Phase | Natural leader |
|---|---|---|
| Growth up, inflation down | Early / Recovery | **Stocks** |
| Growth up, inflation up | Mid → Late | Stocks → **Commodities** |
| Growth down, inflation up | Late / Overheat | **Commodities, real assets** |
| Growth down, inflation down | Recession | **Bonds, cash** |

If that two-axis picture feels familiar, it is the same machinery as our companion post [correlation by regime: the four macro quadrants](/blog/trading/macro-correlations/correlation-by-regime-the-four-macro-quadrants) — the clock is what you get when you let the quadrants *rotate over time* instead of treating them as a static grid.

**What does "correlation" mean here, precisely?** Correlation, written *r*, is a number between −1 and +1 that measures how tightly two things move together. A correlation of **+1** means they move in lockstep the same way; **−1** means perfect opposite movement; **0** means no linear relationship. For two asset returns, +0.6 means "when one is up, the other is usually up too"; −0.5 means "when one is up, the other is usually down" — a *diversifying* relationship. If correlation is new to you, [what correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) builds it from the ground up; here we take the definition as given and focus on why it *moves*.

**The central claim of this post:** correlation is a property of a *regime*, not of an asset pair. "Stocks and bonds are negatively correlated" is not a law of nature like "copper conducts electricity." It is a statement that *happens to be true in disinflationary expansions* and *false in inflationary overheats*. The same is true of nearly every correlation in this series. So if you want to know which correlations to trust today, you do not look up a long-run average — you first locate the phase, and the phase tells you the map.

**Why two axes and not one?** It is tempting to collapse the cycle to a single dial — "good times" versus "bad times" — but that throws away the information that actually drives correlations. The reason the clock needs *two* axes (growth and inflation) is that the two shocks pull assets in *different directions*. A growth shock and an inflation shock are not the same thing wearing different hats; they reorganize the entire correlation map differently.

Think about what each axis does to the discount rate, which is the master variable that prices every asset (the deep version of this lives in [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable)). A **growth** surprise raises the *expected cash flows* of risky assets (earnings, demand for commodities) and, separately, nudges rates up because more growth eventually means more borrowing and inflation. The cash-flow effect dominates for stocks, so stocks rise; the rate effect dominates for bonds, so bonds fall — hence the *negative* stock-bond correlation in growth-driven regimes. An **inflation** surprise barely touches real cash flows in the short run but forces the central bank to hike, raising the discount rate on *everything at once* — so stocks and bonds both fall, a *positive* correlation. Two axes, because there are two fundamentally different ways the economy can surprise you, and each one redraws the map.

This is also why a "soft landing" and a "stagflation" are both possible late-cycle outcomes that look superficially similar (growth slowing) but have *opposite* correlation maps. In a soft landing inflation falls as growth slows, so the stock-bond correlation eases back toward negative and bonds start hedging again. In stagflation inflation stays high *while* growth slows, so the stock-bond correlation stays positive and bonds keep failing as hedges. The single-axis "things are slowing" framing cannot tell these apart; the two-axis clock can.

**A note on what "the leader" means.** Throughout this post, "the leader" of a phase is the asset class with the highest *expected* return in that phase, on average, across many historical cycles. It does not mean the leader wins *every* time — these are averages with wide dispersion, and any single cycle can defy them. The investment clock is a base-rate tool: it tells you which way to lean, not what will certainly happen. Treating it as a guarantee is the most common way to misuse it, and we will return to that in the misconceptions.

#### Worked example: why inflation flips the stock-bond sign

Here is the mechanism in numbers, because the whole post hangs on it. The price of a long bond and the price of a stock are both *present values of future cash flows*, discounted by an interest rate. A stock paying \$3 of earnings growing at 5%, discounted at 7%, is worth roughly \$3 / (0.07 − 0.05) = \$150. A 10-year bond paying \$5 coupons is worth the present value of those coupons plus principal.

Now ask: what happens to each when the discount rate rises by 1 percentage point? For the stock: \$3 / (0.08 − 0.05) = \$100 — a fall of about **33%** from \$150. For the bond, a 1-point rate rise on a 10-year instrument cuts the price by roughly its duration, about **8–9%**. The key point is the *sign*: **both fall**. When the shock driving markets is "rates are going up because inflation is hot," the discount rate rises for *everything*, so stocks and bonds drop together — a positive correlation. When instead the shock is "growth is weakening," rates *fall* (the central bank eases), which *lifts* bond prices while *hurting* stock earnings — they move opposite, a negative correlation. The intuition: **what is driving rates decides the sign.** Inflation-driven rates pull stocks and bonds the same way; growth-driven rates pull them opposite ways.

## The clock face: leadership rotates phase by phase

Now we can put real numbers on the rotation. The single most useful object in this post is the **investment-clock matrix**: the four phases as rows, the four major asset classes as columns, and in each cell the average annual *real* (inflation-adjusted) return that asset class has tended to deliver in that phase. Read it as a map of who leads when.

![Heatmap of average real return by cycle phase and asset class](/imgs/blogs/the-business-cycle-correlation-clock-2.png)

Walk the rows top to bottom and you can feel the clock turn:

- **Early / Recovery:** Stocks **+20%**, commodities +8%, government bonds +6%, cash −1%. Stocks dominate. Everything is recovering from a depressed level, earnings rebound fastest, and equities have the most upside. Cash *loses* in real terms because the central bank is keeping rates pinned below inflation to nurse the recovery. The recovery is also the phase where being *wrong about the timing is least costly* — because nearly everything except cash goes up, even a clumsy allocation does fine, which is why the early phase forgives mistakes that the late phase punishes.
- **Mid / Expansion:** Stocks **+12%**, commodities +6%, bonds +2%, cash 0%. Stocks still lead but the easy money is made; this is the long grind. The famous "growth correlations" (cyclicals, copper, yields all rising with the economy) are at their strongest and most reliable here.
- **Late / Overheat:** **Commodities +14%**, stocks +6%, cash +1%, bonds **−2%**. Leadership has rotated. The economy is running hot, demand for raw materials is high, inflation is eating into bond returns, and equities are wobbling as the central bank hikes. Real assets are the place to be.
- **Recession:** Bonds **+10%**, cash +1%, commodities **−8%**, stocks **−10%**. The defensive assets win. Growth collapses, the central bank cuts rates, bond prices soar, and the cyclical assets — stocks and commodities — get hit hardest.

Notice the *anti-correlation across the cycle*: the asset that leads in one phase is rarely the leader in the next. Stocks lead recovery; commodities lead overheat; bonds lead recession. This is exactly *why* a single static correlation number is misleading — the relationships are rotating underneath it.

**Follow one asset class down its column.** Reading the matrix the other way — tracing a single asset class through all four phases — is just as instructive, and it is how you learn each asset's *personality* across the cycle:

- **Stocks** go +20% → +12% → +6% → −10%. A steadily declining return as the cycle ages, turning sharply negative in recession. Stocks are the ultimate pro-cyclical asset: they love the early and mid phases and hate the recession. The decay from +20% to +6% across recovery, mid, and late is the market "running out of cheap upside" as valuations rise and growth matures.
- **Government bonds** go +6% → +2% → −2% → +10%. They are *counter-cyclical*: their best phase (recession, +10%) is precisely the stock's worst, and their worst phase (overheat, −2%) is when stocks are still positive. This counter-cyclicality is *why* bonds diversify — except in the overheat, where the −2% comes *alongside* a stock wobble rather than against it.
- **Commodities** go +8% → +6% → +14% → −8%. They peak late (the overheat, +14%), because that is when demand is hottest and inflation is highest, and they crater in the recession (−8%) when demand collapses. Commodities are the late-cycle specialist.
- **Cash** goes −1% → 0% → +1% → +1%. It is the flat, boring line — it loses a little in the recovery (rates held below inflation), breaks even mid, and earns a small positive real return only late and in recession. Cash never *leads*, but it never *crashes*; its value is optionality, not return.

This column view is the source of the diversification *itself*: because the four assets peak in four different phases, a portfolio holding all of them is never fully exposed to any single phase's loser. That is the intuition behind risk parity and all-weather investing, which try to size each asset so the portfolio is balanced across the clock rather than betting on one quadrant ([all-weather and risk parity](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime)).

#### Worked example: a phase-appropriate vs. phase-blind \$100,000 portfolio

Suppose you have **\$100,000** and you must choose between two strategies across one full cycle made of four equal-length phases. Strategy A is "phase-appropriate": each phase you hold the leader from the matrix above. Strategy B is a fixed 60/40 (60% stocks, 40% bonds) held through all four.

Using the matrix's real returns, one period each:

- **Strategy A (hold the leader):** Early → stocks +20%; Mid → stocks +12%; Late → commodities +14%; Recession → bonds +10%. Compounding: \$100,000 × 1.20 × 1.12 × 1.14 × 1.10 = **\$168,500**, a +68.5% real gain.
- **Strategy B (static 60/40):** blended returns per phase = 0.6×stocks + 0.4×bonds. Early: 0.6×20 + 0.4×6 = 14.4%. Mid: 0.6×12 + 0.4×2 = 8.0%. Late: 0.6×6 + 0.4×(−2) = 2.8%. Recession: 0.6×(−10) + 0.4×10 = −2.0%. Compounding: \$100,000 × 1.144 × 1.08 × 1.028 × 0.98 = **\$124,400**, a +24.4% real gain.

The phase-appropriate path ends with roughly **\$44,000 more** on the same \$100,000 over one idealized cycle. The numbers are stylized — you will never time the phases perfectly and transaction costs and taxes are real — but the gap is the *value of the clock*: knowing which leadership and which correlations are live is worth real money. The intuition: most of the long-run advantage of "rotating with the cycle" comes not from being clever in any one phase, but from *not owning the loser* — especially not holding cyclical assets into the recession.

## The correlation that flips: stock-bond by inflation regime

The clock's most important consequence for a normal investor is what it does to the **stock-bond correlation**, the engine of every balanced portfolio. We will now look at it directly, because it is the single clearest example of "the correlation is a regime."

The cleanest way to see it is to condition the correlation on the *inflation regime*, since inflation is the variable that drives the late/overheat phase.

![Bar chart of stock-bond correlation by inflation regime](/imgs/blogs/the-business-cycle-correlation-clock-3.png)

The pattern is monotonic and stark:

- **Inflation below 2%:** r = **−0.45**. Strongly diversifying. This is the disinflationary expansion — bonds are a genuine hedge for stocks.
- **2–3% inflation:** r = **−0.30**. Still diversifying. The comfortable middle, where 60/40 does its job.
- **3–4% inflation:** r = **+0.05**. The diversification has basically *evaporated*. Bonds no longer cushion stocks.
- **Above 4% inflation:** r = **+0.50**. Strongly positive. Stocks and bonds now fall *together*. The 60/40 cushion turns into a second source of loss.

The driver is exactly the mechanism from the foundations worked example. At low inflation, the market's dominant fear is weak growth, so rates move with growth and stocks and bonds move opposite. Once inflation climbs above roughly 3–4%, the dominant fear becomes inflation itself, rates move with inflation, and stocks and bonds move together. The threshold is not magic — it is where inflation becomes the *primary* thing the central bank and the market are reacting to.

The same flip is visible if you stop bucketing by inflation and instead watch the correlation roll forward through history. The chart below traces the rolling stock-bond correlation from 1990 to 2025 — and it reads like a tour of the regimes.

![Rolling stock-bond correlation from 1990 to 2025 with regime shading](/imgs/blogs/the-business-cycle-correlation-clock-4.png)

The line tells a three-act story. In the early **1990s** the correlation was *positive* (around +0.35 to +0.45) — a hangover from the high-inflation 1970s and 80s, when inflation was still the market's reflex fear. Around **1998–2000** it crossed zero and went durably *negative*, ushering in the great two-decade "diversification" era: from 2002 through 2021 the correlation sat between about −0.2 and **−0.55**, the deepest readings in the post-2008 disinflation. This is the window in which the 60/40 earned its reputation; bonds reliably rallied in every stock scare (2008, 2011, 2018, the March 2020 crash). Then in **2022** the line snapped *up* to **+0.60** — the sharpest, fastest sign change in the whole series — as inflation reclaimed the driver's seat. By 2023–24 it eased back toward +0.30 as inflation cooled, but it has not yet returned to the deep-negative diversifying regime. The single most important visual takeaway: this is not a noisy wiggle around a constant. It is a slow-moving *regime variable* that spends years in one sign and then flips — and the flips line up with the cycle's inflation phase.

This is the heart of the matter for [the stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime), which dives into this single relationship in full; and it is the macro reason behind [the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) breaking. Here the point is narrower and bigger at once: *the inflation regime is just a coordinate on the clock.* "Above 4% inflation" almost always means "late/overheat phase." So the same flip can be described two ways — by inflation regime or by cycle phase — and they are the same fact.

#### Worked example: the stock-bond correlation as a hedge ratio in dollars

Correlation is abstract; let us turn it into a dollar cushion. Take a **\$100,000** 60/40 portfolio: \$60,000 in stocks, \$40,000 in bonds. Suppose stocks fall **10%**, a \$6,000 loss on the equity sleeve. The question is what the bond sleeve does, and that depends entirely on the correlation regime.

In a **low-inflation regime** (r ≈ −0.45), the negative correlation means bonds tend to *rise* when stocks fall. Say bonds rise about 4% — a +\$1,600 gain on \$40,000. Net portfolio loss: −\$6,000 + \$1,600 = **−\$4,400**, or −4.4%. The bond sleeve absorbed about a quarter of the equity loss. The diversification worked.

In a **high-inflation regime** (r ≈ +0.50), the positive correlation means bonds tend to *fall with* stocks. Say bonds fall 6% — a −\$2,400 loss on \$40,000. Net portfolio loss: −\$6,000 − \$2,400 = **−\$8,400**, or −8.4%. The bond sleeve nearly *doubled* the damage instead of cushioning it. That −8.4% versus −4.4% — almost twice the loss on the same equity move — is precisely what happened to balanced portfolios in 2022. The intuition: the bond sleeve's job is to be negatively correlated, and in the overheat phase it stops doing that job exactly when you need it most.

## The correlation that peaks mid-cycle: growth assets

If the late phase is defined by the stock-bond flip, the mid phase is defined by *growth correlations* being at their cleanest. When the economy is expanding steadily, a positive growth surprise — say the ISM manufacturing survey ticking up — pulls a very recognizable set of assets in a very recognizable way.

![Bar chart of asset correlations with a positive growth surprise](/imgs/blogs/the-business-cycle-correlation-clock-5.png)

Read the chart as "what is correlated with the economy speeding up":

- **Cyclical sectors (industrials): +0.65** — the most growth-sensitive part of the stock market; they make the machines and materials that a growing economy buys more of.
- **Copper: +0.60** — "Dr. Copper," the metal with a PhD in economics; it is in everything that gets built, so its price reads the growth pulse.
- **S&P 500: +0.55** and **small caps: +0.55** — broad equity and especially smaller, more domestic, more cyclical companies rise with growth.
- **US 10-year yield: +0.45** — yields *rise* with growth, because faster growth implies more demand for capital and, eventually, more inflation. This is the *growth-driven* rate move, the one that makes stocks and bonds move opposite.
- **Defensives (staples): −0.10** and **gold: −0.05** — roughly flat; people do not buy more toothpaste because the economy is booming, and gold is not a growth play.
- **Long Treasuries: −0.45** — they *fall* when growth accelerates, the mirror image of the +0.45 on yields.

The crucial detail is that this clean fan-out of correlations is *strongest in mid-cycle*. In the recovery the relationships are still forming (everything bounces together off the bottom); in the overheat they get distorted by the inflation overlay; in the recession they invert (defensives win, cyclicals lose). The growth correlation is a mid-cycle phenomenon, which is why our companion posts on [ISM as the leading correlation with cyclicals](/blog/trading/macro-correlations/ism-pmi-the-leading-correlation-with-cyclicals) and [the yield curve as a growth signal](/blog/trading/macro-correlations/the-yield-curve-as-a-growth-signal-and-its-asset-correlation) both insist you check the phase before you trust the signal.

**Rotation happens inside the stock market too.** The clock does not only rotate *between* asset classes; it rotates *within* equities, across sectors. This is "sector rotation," and it is the same idea one level down. Early in the cycle, the most beaten-down, most cyclical, most rate-sensitive sectors lead — financials, consumer discretionary, technology, real estate — because they have the most upside from a recovering economy and falling rates. Mid-cycle, broad cyclicals like industrials and materials carry the baton as capital spending ramps. Late-cycle, the sectors tied to inflation and real assets lead — energy and materials — while rate-sensitive sectors fade. In the recession, the *defensive* sectors win on a relative basis: consumer staples, utilities, and health care, because people keep buying food, electricity, and medicine no matter what the economy does.

The 2022 sector returns make this vivid: energy returned **+65.7%** (the inflation/overheat winner) while consumer discretionary fell **−37.0%** and technology fell **−28.2%** (the rate-sensitive losers). Staples (−0.6%), health care (−2.0%), and utilities (+1.6%) held up far better than the cyclicals — a textbook late-cycle/early-recession defensive rotation. The point for correlations: the *correlation of a sector with the broad market* is itself phase-dependent. Defensives have a low or even negative correlation with the market's cyclical swings; cyclicals have a high one. Knowing the phase tells you not just which asset class to own, but which corner of the equity market — and that is the same clock turning at a finer resolution.

#### Worked example: a \$50,000 growth-surprise tilt and its beta

Suppose you read the ISM print on the first business day of the month and it comes in well above expectations — a clear positive growth surprise — and you judge you are in mid-cycle, where the growth correlations are reliable. You put **\$50,000** to work tilting toward the growth winners and away from the losers.

Translate the correlations into a rough one-month tilt. If a strong-growth month historically moves cyclicals about +3% and long Treasuries about −2% (consistent with their +0.65 and −0.45 correlations to the growth factor), then:

- **\$30,000 into cyclicals** at +3% → +\$900.
- **Short \$20,000 of long Treasuries** at −2% (you profit when they fall) → +\$400.
- Combined: **+\$1,300** on \$50,000 deployed, about **+2.6%** in a month.

Now run the same trade in the *wrong* phase. If you are actually late-cycle and the strong ISM is the *last* good print before the rollover, the inflation overlay can mean cyclicals stall (+0.5%) while long Treasuries *also* fall on inflation fear (−2%) — but now both legs are noise, and the trade earns roughly \$150 + \$400 = **+\$550**, less than half, with far more risk of a sharp reversal. The intuition: the growth correlation is real, but its *strength* is a mid-cycle property; the identical trade is worth less than half as much, with worse odds, if you misread the phase.

## Why the clock turns: the engine behind the rotation

It is worth pausing on *why* the cycle moves at all, because understanding the engine is what lets you anticipate the next turn instead of just reading the current one. The clock is not driven by a calendar or by mysticism; it is driven by a **feedback loop between the economy and the central bank**, and that loop is what makes the correlations rotate.

Here is the loop in plain terms. In the recovery, the central bank holds rates low to stimulate. Cheap money pulls forward demand, the economy speeds up, and we move into expansion. As the expansion matures, slack disappears — factories run near capacity, the labor market tightens — and prices start to rise. That rising inflation is the late/overheat phase, and it forces the central bank to *hike* rates to cool things down. Higher rates eventually bite: borrowing slows, demand falls, and the economy tips into recession. The recession kills inflation, which lets the central bank *cut* rates again, which seeds the next recovery. Around and around. The clock turns because the policy response to one phase *creates* the conditions for the next.

This is why the correlations rotate in the order they do. The stock-bond correlation flips positive in the late phase precisely because that is when the central bank is hiking *into* inflation, pulling stocks and bonds down together. It flips back negative in the recession because the central bank pivots to cutting, which rallies bonds while growth fear hurts stocks. The correlation map is downstream of the policy loop. If you understand the loop, you understand why the map looks the way it does in each phase — and why a central bank that *breaks* the normal loop (cutting into inflation, or hiking into a slowdown) will scramble the correlations in ways the clock can't predict.

**Lead and lag: not everything turns at once.** The phases do not switch cleanly like a light; different indicators turn at different times, and the *order* in which they turn is itself a signal. This is the lead/lag structure that [leading, coincident, and lagging indicators](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators) studies in full. A few of the most useful lead times:

- **The yield curve leads the recession by about 14 months** on average. The 2s10s inverts well before the downturn arrives, because the bond market prices future rate cuts before the economy actually rolls over.
- **ISM new orders lead earnings (S&P EPS growth) by about 6 months.** The survey of what businesses are *ordering* turns before the profits show up — which is why ISM sits first on the dating checklist.
- **Building permits lead GDP by about 9 months,** because a permit today is construction (and spending) several quarters out.
- **Credit spreads lead equity drawdowns by about 3 months.** The bond market often sniffs out trouble before the stock market sells off.
- **Initial jobless claims lead the unemployment rate by about 2 months.** Claims are the high-frequency tremor before the slower unemployment quake.

The practical upshot: because leading indicators turn *first*, you can often see the next phase coming before the coincident data confirms it. When the yield curve inverts, ISM rolls over, and credit spreads start widening — all of which lead — you should be *preparing* for the recession quadrant's correlation map even while unemployment (a lagging indicator) is still low and the stock market is still near highs. The leaders are the early-warning system; the laggards are the confirmation. Trading the laggards alone means you are always one phase late.

**The mid-cycle correlation peak, explained.** Earlier we saw that growth correlations are strongest mid-cycle. The engine explains why. In the recovery, *every* asset is bouncing off a depressed bottom, so they all rise together regardless of their growth sensitivity — the growth signal is drowned out by the broad rebound. In the overheat, the inflation overlay distorts everything — yields rise on inflation rather than growth, so the normal "growth up, yields up, cyclicals up, bonds down" pattern gets scrambled. Only in mid-cycle, when growth is the *clean* driver with no recovery-bounce noise and no inflation distortion, do the growth correlations show their true strength. The correlation is most reliable exactly when the macro story is simplest.

## Dating the phase: the four-indicator checklist

Everything above is useless if you cannot tell which phase you are in. The good news: you do not need to forecast the cycle — you only need to *locate* it, which is far easier. Four indicators, read **together**, pin you to a quadrant.

![Matrix of four indicators and their reading in each cycle phase](/imgs/blogs/the-business-cycle-correlation-clock-6.png)

Here is the checklist. No single indicator is decisive; the *combination* is.

1. **ISM / PMI (the manufacturing survey).** The level relative to 50 (the expansion/contraction line) *and* its direction matter. Below 50 but **rising** = early/recovery. Above 50 and **rising** = mid. Above 50 but **rolling over** = late. Below 50 and **falling** = recession. The ISM leads the hard data by months, which is why it is the first thing on the list. See [ISM PMI: the business surveys that lead](/blog/trading/event-trading/ism-pmi-the-business-surveys-that-lead) for the release mechanics.
2. **Unemployment (the trend, not the level).** High but **falling** = recovery. Low and **stable** = mid. Very low, bottoming out = late (the labor market cannot get much tighter). **Rising** = recession — and rising fast is the classic recession tell (the Sahm rule fires when the 3-month average unemployment rate rises half a point above its 12-month low). [Unemployment claims and the recession correlation](/blog/trading/macro-correlations/unemployment-claims-and-the-recession-correlation) goes deep on the labor leg.
3. **The yield curve (the 2s10s slope).** Steep and positive = early. Flattening = mid. Flat or **inverted** = late (the bond market is pricing future cuts). Re-steepening *from* an inversion = the recession is arriving — historically the curve un-inverts right *before* the downturn, not at the bottom. See [the yield curve as a growth signal](/blog/trading/macro-correlations/the-yield-curve-as-a-growth-signal-and-its-asset-correlation) and the mechanism post [reading the yield curve](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession).
4. **Inflation (level and direction).** Low and cooling = recovery. Tame near 2% = mid. High and **rising** = late/overheat (this is the one that flips the stock-bond correlation). Falling fast = recession.

The skill is *triangulation*. If ISM is above 50 and rising, unemployment is low and stable, the curve is gently flattening, and inflation is near 2% — you are squarely mid-cycle, and the growth correlations are your best friend. If ISM is rolling over, unemployment is bottoming, the curve is inverted, and inflation is high and rising — you are late/overheat, the stock-bond correlation has flipped positive, and you should not be relying on bonds to hedge stocks.

#### Worked example: dating the phase from a live indicator snapshot

Let us date a real-ish snapshot using the curated series. Take the readings around late 2022:

- **ISM** had fallen from 60+ in 2021 to **48.4** by December 2022 — below 50 and falling.
- **Unemployment** was **3.5%** — historically very low, near the cycle bottom.
- **The 2s10s curve** was **−0.55 percentage points** — deeply inverted.
- **Core PCE inflation** was **4.9%** — high, though just starting to ease from a 5.6% peak.

Score it: ISM says we have rolled over from late toward recession; unemployment at a multi-decade low says we are still late-cycle; a −0.55 inversion is the textbook *late-cycle* warning; 4.9% inflation, well above 4%, is firmly in the overheat zone. The triangulated read is **late/overheat tipping toward recession** — which is exactly the phase where the stock-bond correlation is at its most positive.

What did that imply for a portfolio? It said *do not trust bonds to hedge stocks*, and indeed in 2022 they did not — the 60/40 fell about 16% precisely because the phase had moved both stocks and bonds into the same losing quadrant. A trader who simply read the four indicators and concluded "late/overheat, stock-bond correlation positive" would have known to cut duration and lean on commodities and cash rather than long bonds — and would have sidestepped the cushion that wasn't. The intuition: you did not have to *forecast* anything; you only had to *read four numbers* and look up the quadrant.

## Which correlations to trust in each phase

We can now assemble the payoff: a single map of which correlations are reliable in each phase, and which hedges to lean on versus avoid. This is the clock's verdict on the rest of the series.

![Matrix of which correlation and hedge to trust in each cycle phase](/imgs/blogs/the-business-cycle-correlation-clock-7.png)

Read each column as a regime briefing:

- **Early / Recovery:** stock-bond correlation negative (bonds hedge), growth correlations turning positive. What hedges: long bonds and cyclicals both work as the economy lifts. What to avoid: sitting in cash, which loses about 1% real per year while everything else rallies.
- **Mid / Expansion:** stock-bond correlation negative (60/40 works), growth correlations *strongest*. What hedges: a normal balanced book — equities plus credit, with bonds genuinely diversifying. What to avoid: over-hedging growth, i.e., being too defensive in the calmest, most rewarding phase.
- **Late / Overheat:** stock-bond correlation **positive (60/40 breaks)**, growth correlations fading as inflation takes over. What hedges: commodities and real assets, which rise with the inflation that is hurting everything else. What to avoid: **long bonds** — the very thing you reach for as a hedge is now positively correlated with stocks and will compound your loss.
- **Recession:** stock-bond correlation negative again (bonds rally hard as the central bank cuts), growth correlations *inverted* (defensives win, cyclicals lose). What hedges: bonds, cash, and long duration — the recession is the one phase where Treasuries truly pay. What to avoid: commodities and high-beta assets, which fall hardest as demand collapses.

The single most expensive mistake in the table is in the *Late* column: reaching for long bonds as a hedge when the stock-bond correlation has flipped positive. That is the 2022 trap in one cell. The single most common mistake is in the *Mid* column: being permanently defensive — holding too much cash and too many hedges through the long expansion, and giving up the bulk of the cycle's returns out of caution.

This map is the bridge to [the macro correlation playbook capstone](/blog/trading/macro-correlations/the-macro-correlation-playbook-capstone), which strings the whole series into a single decision process: read the phase, look up the live correlations, size the trade.

**A caution on transitions.** The clean column-by-column map is most reliable in the *middle* of a phase, when the four indicators all agree. The dangerous moments are the *transitions* — the handful of months when the economy is moving from one quadrant to the next. There, correlations are unstable: the stock-bond correlation might be crossing zero, the growth correlations might be weakening but not yet inverted, and the indicators disagree (one leads, one lags). Two errors are common at transitions. The first is *anchoring* — continuing to trust the previous phase's map after it has rotated, which is exactly how investors got hurt holding long bonds into 2022. The second is *jumping the gun* — assuming the next phase has arrived the moment a single leading indicator turns, when the confirming data has not followed (selling all your stocks the day the curve inverts). The discipline is to *fade your conviction at transitions*: when the four indicators split, cut position sizes, widen your stops, and wait for two or three of the four to align before committing to the new phase's correlation map. The clock is most tradeable when it is firmly *in* an hour, least tradeable when it is striking the next one.

**The clock is not only domestic.** Everything above is framed on the US cycle because the US sets the global financial weather, but the same machinery applies to any economy with its own cycle — and the *interaction* of cycles is itself a correlation story. When the US dollar strengthens (typically late-cycle, as US rates rise), it pressures commodities, emerging-market equities, and gold (see [the dollar: cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity)). So a US late/overheat phase does not only flip the domestic stock-bond correlation — it exports tighter conditions to the rest of the world through the dollar, which is why emerging markets often have their worst phase when the US is hiking into its overheat. The clock you read at home has spillover hands that move other markets' clocks. For a single-market application of the same phase logic, the Vietnam and cross-asset tracks show how a domestic cycle and the imported US cycle can be in *different* hours at once — and how the correlation between them is itself a function of where each one is on its own dial.

## Common misconceptions

**Myth 1: "Stocks and bonds are negatively correlated."** No — they are negatively correlated *in some regimes and positively correlated in others*. Over 1990–2025 the rolling stock-bond correlation ranged from about −0.55 (the 2010s disinflation) to **+0.60** (the 2022 inflation shock). A "long-run average" near zero hides the whole story. The sign depends on whether growth or inflation is the dominant risk — i.e., on the phase. Anyone quoting a single stock-bond correlation number is quoting the average of two opposite regimes.

**Myth 2: "The 60/40 is dead / the 60/40 is fine."** Both overstate. The 60/40 is *regime-dependent*: it is an excellent portfolio in disinflationary expansions (the phase it was designed in and tested through) and a poor one in inflationary overheats. It is not dead; it just stops diversifying in exactly one phase. The fix is not to abandon it but to recognize that in the late/overheat quadrant you need a *third* leg — commodities or real assets — because bonds cannot do the hedging job there.

**Myth 3: "Cash is always safe."** In real (inflation-adjusted) terms, cash *loses* in the early/recovery phase, where the matrix shows about **−1%** — the central bank deliberately holds rates below inflation to stimulate. Cash only earns its keep in the late and recession phases, when rates are high or falling from a high level. "Safe" is also a regime statement.

**Myth 4: "Gold is the inflation hedge, so buy it in the overheat."** Gold is not primarily an inflation hedge; it tracks *real yields* (see [inflation and gold: the real yield story](/blog/trading/macro-correlations/inflation-and-gold-the-real-yield-story)), and its growth-surprise correlation is roughly **−0.05** — essentially zero. In the late phase, the real asset that reliably leads is *commodities* (about +14% real), not gold specifically. Gold can do well in the overheat, but for the real-yield reason, not the inflation reason — and that distinction will save you in the phase where real yields rise even as inflation stays high.

**Myth 5: "A 35-year average correlation is the best estimate of today's correlation."** It is the *worst* of both worlds. If you average the stock-bond correlation from 1990 to 2025 you get a number near zero — and zero is true in *no single regime*. The relationship was +0.4 in the early 90s, −0.5 through the 2010s, and +0.6 in 2022; the long-run average is a blend of opposite truths that describes no actual market you will ever trade in. A short window is noisy but at least tells you about the *current* regime; a very long window is precise but tells you about an *average of regimes that have ended*. The fix is to estimate the correlation *conditional on the phase* (the bucketed bars in this post) rather than unconditionally — which is the whole lesson of [rolling correlation and why the window matters](/blog/trading/macro-correlations/rolling-correlation-and-why-the-window-matters).

**Myth 6: "The clock always ticks in order."** It does not. The cycle can skip, stall, or sprint. A policy mistake or external shock can collapse mid-cycle straight into recession (2020's COVID crash skipped the late phase entirely); a "soft landing" can stall late-cycle indefinitely without a recession (the 2023–24 episode, where the curve inverted in 2022 but no recession had arrived). The clock is a *map of the typical sequence*, not a guarantee of timing. You re-date the phase every month from the indicators; you never assume the next phase just because time has passed.

## How it shows up in real markets

**2020 — the recovery, where stocks led (and skipped late).** The COVID crash was a recession compressed into weeks. By April 2020 the ISM bottomed at **41.5**; the central bank slashed rates to zero and flooded the system with liquidity. What followed was a textbook *early/recovery* phase: stocks led explosively (the S&P bottomed in March and was at new highs within months), the stock-bond correlation stayed negative, and bonds still cushioned. The clock had reset to the recovery position, and the recovery playbook — own stocks, especially the beaten-down cyclicals — paid off enormously.

**2021 — mid-cycle, the calm grind.** Through 2021 the ISM ran above 50 (peaking around **63.7** in March), unemployment fell steadily toward 3.9%, and inflation, though rising, was still being dismissed as "transitory." This was the mid/expansion phase: stocks ground higher, growth correlations were strong, cyclicals and copper did well (copper rose from \$2.80 to \$4.23 a pound). A balanced book worked. The stock-bond correlation, still around −0.10, was negative enough that bonds remained a (weakening) hedge. The clock was at the top, in its calmest hour.

**2022 — late/overheat, where commodities won and the 60/40 broke.** This is the phase that defines the post. Inflation hit a 40-year high of **9.06%** in June 2022; core PCE peaked at 5.6%; the central bank hiked from 0.25% to 4.50% in a single year; the 2s10s curve inverted to −0.55. Every box on the late/overheat checklist was ticked. And the asset returns matched the matrix exactly: commodities (BCOM) **+16.1%**, the US dollar +8.2%, while the S&P fell 18.1%, long Treasuries fell **31.2%**, and the 60/40 lost 16.1%. The stock-bond correlation flipped to **+0.60**. The lesson printed in real money: in the overheat, the hedge that works is *commodities*, and the hedge that fails is *long bonds*.

#### Worked example: the recession flight-to-quality in dollars

The recession quadrant is the one phase where the textbook "flight to safety" actually pays, and it is worth pricing out because it is the mirror image of the 2022 trap. Suppose you correctly date a recession arriving — ISM below 50 and falling, unemployment turning up, the curve re-steepening from inversion — and you move a **\$200,000** portfolio from cyclical to defensive: out of stocks and commodities, into long Treasuries and cash.

Using the recession-row returns: stocks −10%, commodities −8%, bonds +10%, cash +1%. Compare two books:

- **Stayed cyclical (60% stocks, 40% commodities):** 0.6×(−10%) + 0.4×(−8%) = −6.0% + −3.2% = **−9.2%** → on \$200,000, a loss of **\$18,400**.
- **Rotated defensive (70% long bonds, 30% cash):** 0.7×(+10%) + 0.3×(+1%) = +7.0% + +0.3% = **+7.3%** → on \$200,000, a gain of **\$14,600**.

The swing between the two books is **\$33,000** on the same \$200,000 — a 16.5-point gap — and it comes entirely from being on the right side of the rotation. Note the symmetry with the 2022 worked example: in the overheat, long bonds were the *trap*; in the recession, long bonds are the *trade*. Same instrument, opposite role, one phase apart. The intuition: the recession is the only phase where reaching for Treasuries is reaching for the leader rather than the loser — which is exactly why dating the phase, not the instrument, is what matters.

**2023–24 — the disinflationary recovery (the clock stalls).** Here the clock did something instructive: it *stalled*. The curve had inverted in 2022, which historically leads a recession by 12–18 months — yet no recession arrived. Instead, inflation cooled (core PCE fell from ~5% toward 3%) while growth held up — a *disinflationary expansion*, the regime where the stock-bond correlation eases back toward negative (it fell from +0.60 in 2022 to +0.45 in 2023 to +0.30 by 2024). Stocks recovered strongly. This is the "soft landing" case where the recession the curve predicted simply did not come on schedule. The disciplined response was *not* "the curve is broken, ignore it," but "the phase has shifted to disinflationary recovery, so the correlations are rotating back toward their diversifying state — re-trust bonds gradually." The clock did not break; it paused, and the indicators told you so.

#### Worked example: why the clock can skip or stall, in numbers

The clock's typical lead times are real but variable. Historically, a 2s10s inversion has led the recession by an average around 14 months — but the spread is enormous: about **18 months** in 1989, **13** in 2000, **22** in 2006, only **6** in 2019 (COVID shortened it), and in 2022 the inversion led to **no recession at all** within the following two-plus years. If you had mechanically assumed "inversion in July 2022 → recession by roughly September 2023" and sold all your stocks, you would have missed one of the strongest equity rallies in a decade: the S&P rose roughly **24%** in 2023 and another **20%+** in 2024. On a **\$100,000** equity position, sitting in cash from mid-2023 to end-2024 instead of staying invested cost roughly **\$49,000** in forgone gains (\$100,000 × 1.24 × 1.20 − \$100,000 ≈ \$48,800). The intuition: the clock tells you the *typical order*, not the *timing*; you trade the phase you can *measure today*, never the phase you *assume* is coming next.

## How to read it and use it

Here is the whole post compressed into a repeatable process. Run it monthly, when the key data lands.

**Step 1 — date the phase.** Pull the four indicators and read them together: ISM level and direction, the unemployment trend, the 2s10s slope, and the inflation level and direction. Triangulate to one quadrant — early, mid, late, or recession. Do not over-engineer it; the point is to be *roughly right about the quadrant*, not precisely right about the week.

**Step 2 — look up the live correlations.** Once you know the quadrant, the correlation map is fixed. Mid-cycle: trust the growth correlations (cyclicals, copper, yields up with the economy) and trust bonds to diversify. Late/overheat: distrust the stock-bond hedge (it has flipped positive), and lean on commodities and real assets. Recession: re-trust bonds and cash; avoid cyclicals and high-beta. Recovery: own stocks, especially beaten-down cyclicals; don't sit in cash.

**Step 3 — size the trade to the phase, and the conviction to the clarity.** The four indicators rarely line up perfectly. When all four agree — ISM, unemployment, curve, and inflation all pointing to the same quadrant — your conviction (and position size) should be high. When they conflict (e.g., ISM says late but inflation is already cooling), you are at a *transition*, where correlations are unstable and you should size down and wait for the picture to resolve.

**What invalidates the read.** The clock framework is wrong, or at least suspended, when an *exogenous shock* overrides the cycle: a pandemic, a war, a banking panic, a policy error. In those moments correlations do not rotate gently — they all rush toward +1 as everything sells off together (see [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis)). The tell is a volatility spike with *no* phase-consistent story; when that happens, the clock is off and you fall back to crisis rules (raise cash, cut leverage) until the phase reasserts itself. The other invalidation is the *stall*: when the curve or another lead indicator fires but the confirming indicators (unemployment, ISM) refuse to follow, do not force the next phase — re-date from what you can measure, as 2023–24 taught.

**The one-sentence version.** Find the phase first; the correlations follow from it. A correlation you look up without knowing the phase is the *average of two opposite regimes*, which is to say, a number that is true on average and wrong almost always.

Two habits make the clock pay off in practice. First, *date the phase before you form any view on a correlation* — not after. It is tempting to decide "stocks and bonds diversify" and then go looking for the phase that confirms it; reverse the order, let the indicators speak first, and accept whatever map they hand you. Second, *write down your phase call each month with the four readings that justify it*. When the call later turns out wrong, the written record shows you which indicator misled you — and over a few cycles you build the pattern-recognition that turns the checklist from a chore into instinct. The clock rewards discipline far more than cleverness: the investors who do best with it are not the ones who predict the next phase, but the ones who never fight the phase they are demonstrably in.

The clock is the regime selector for the entire series. The mechanism behind each phase — *why* policy and growth move each asset — lives in the macro-trading posts ([the business cycle: four phases for traders](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) and [asset rotation across the business-cycle quadrants](/blog/trading/macro-trading/asset-rotation-across-the-business-cycle-quadrants)). The *measurement* of each correlation — how strong, which lead, when it flips — lives in the rest of this series. And the *strategy* of owning every regime so you never have to time the clock perfectly lives in [all-weather and risk parity](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime). Put them together and you have a complete loop: measure the phase, read the live correlations, choose the assets, and re-check next month.

## Further reading and cross-links

Within this series:

- [Correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) — the founding idea this post operationalizes.
- [Correlation by regime: the four macro quadrants](/blog/trading/macro-correlations/correlation-by-regime-the-four-macro-quadrants) — the static-grid sibling of the rotating clock.
- [The stock-bond correlation regime](/blog/trading/macro-correlations/the-stock-bond-correlation-regime) — the single most important flip, in full.
- [ISM PMI: the leading correlation with cyclicals](/blog/trading/macro-correlations/ism-pmi-the-leading-correlation-with-cyclicals) — the indicator that dates the growth axis.
- [The yield curve as a growth signal and its asset correlation](/blog/trading/macro-correlations/the-yield-curve-as-a-growth-signal-and-its-asset-correlation) — the indicator that dates the late phase.
- [Unemployment claims and the recession correlation](/blog/trading/macro-correlations/unemployment-claims-and-the-recession-correlation) — the labor leg of the checklist.
- [What correlation actually measures](/blog/trading/macro-correlations/what-correlation-actually-measures-pearson-spearman-beta) — the statistics, from zero.
- [The macro correlation playbook capstone](/blog/trading/macro-correlations/the-macro-correlation-playbook-capstone) — the whole series as one process.

The mechanism (why policy moves each asset):

- [The business cycle: four phases for traders](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders)
- [Asset rotation across the business-cycle quadrants](/blog/trading/macro-trading/asset-rotation-across-the-business-cycle-quadrants)
- [Reading the yield curve: slope, inversion, recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession)

Portfolio construction across regimes:

- [Stock-bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine)
- [All-weather and risk parity: owning every regime](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime)
- [Correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch)
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis)

---
title: "The Yield Curve as a Growth Signal and Its Asset Correlation"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Why the slope of the Treasury curve is the market's best-known recession forecast, how its sign correlates with banks, cyclicals, and the dollar, and why the deepest inversion in forty years still has not produced a recession."
tags: ["macro", "correlation", "yield-curve", "recession", "2s10s", "inversion", "interest-rates", "business-cycle", "banks", "fixed-income"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The slope of the yield curve (10-year yield minus 2-year yield, the "2s10s") is the single best-known leading recession indicator: an inversion has preceded every modern US recession with roughly 12 to 18 months of lead, and the slope itself is *correlated* with bank profitability, cyclical-versus-defensive leadership, and the dollar.
>
> - The correlation has a **sign** (a flatter or inverted curve is risk-off for cyclicals and banks), a **strength** (the inversion → recession hit rate is near-perfect over the modern era), a **lead** (~14 months at the cross-correlation peak), and it can **break** (the 2022-24 inversion was the deepest since 1981 yet produced no recession at the time of writing).
> - Inversion is the *warning*; the **bull re-steepening** from the front end — when the Fed actually starts cutting — is the historical *trigger*. Recessions tend to arrive as the curve un-inverts, not while it is most inverted.
> - The slope correlates with bank net interest margins (banks borrow short and lend long, so a flat curve squeezes their spread), with the rotation from cyclicals to defensives, and with a firmer dollar when front-end yields are high.
> - The one number to remember: the 2s10s reached **−1.08 percentage points in July 2023**, the deepest inversion in four decades, and the recession that the rule promised had still not arrived two years later — a live lesson that a strong correlation is a regime, not a law.

In the summer of 2022, a chart that almost nobody outside of bond desks had ever looked at became front-page news. The yield on the 2-year US Treasury note climbed *above* the yield on the 10-year note. On paper this is a small thing — two numbers, a few basis points apart, crossing over. In practice it set off every recession alarm in the financial world, because that crossover, the "inverted yield curve," had preceded every US recession since the 1960s. Economists who had spent careers studying it appeared on television to explain that the clock had started: history said a recession would arrive in roughly a year to a year and a half.

Then something strange happened. The curve did not just invert — it inverted *more* than it had at any point since 1981, reaching −1.08 percentage points in July 2023. And the recession did not come. Unemployment stayed near 4%. The S&P 500 made new all-time highs. Corporate earnings grew. By late 2024 the curve had quietly un-inverted, the 10-year back above the 2-year, and the most reliable recession indicator in macroeconomics had, for the first time in living memory, apparently cried wolf.

This post is about that signal — what the yield curve is, why its slope encodes the market's forecast for growth, why an inversion is supposed to mean trouble, how the slope *correlates* with the assets you actually trade (banks, cyclicals, the dollar, bonds), and why 2022-24 was either a once-in-a-generation false alarm or a sign that the rules have shifted. We build it from zero. By the end you will be able to look at a single number — the 2s10s spread — and read off what the market thinks about the next eighteen months, while staying honest about how often that reading has been wrong.

![Normal versus inverted yield curve and what each implies for growth and assets](/imgs/blogs/the-yield-curve-as-a-growth-signal-and-its-asset-correlation-1.png)

## Foundations: what the yield curve is and why its slope is a forecast

Start with the rawest possible definition. The US Treasury borrows money by selling bonds — IOUs — of many different lengths. You can lend the government money for 3 months, 2 years, 5 years, 10 years, or 30 years. Each of those loans pays you an interest rate, called the *yield*. The **yield curve** is just a picture: maturity on the horizontal axis (how long you lend), yield on the vertical axis (what you earn). Connect the dots and you get a curve.

Most of the time the curve slopes *upward*: lending for 10 years pays more than lending for 2 years, which pays more than lending for 3 months. That feels intuitive — if you tie your money up for longer you should be paid more for the inconvenience and the risk. The amount of extra yield you earn for going longer is the **slope** of the curve. When people say the curve is "steep," they mean long yields are well above short yields. When they say it is "flat," long and short yields are close together. And when they say it is "inverted," they mean the curve slopes *downward* — long yields are *below* short yields, which is the world turned upside down.

The headline measure of slope is the **2s10s spread**: the 10-year yield minus the 2-year yield, quoted in percentage points (or, equivalently, in *basis points*, where one basis point is one-hundredth of a percentage point, so 0.50 percentage points is "50 basis points," written 50bp). When the 2s10s is +0.60, the curve is normal and moderately steep. When it is 0.00, the curve is flat. When it is −1.08, the curve is deeply inverted. That single number is the protagonist of this entire post.

### Why the curve is normally upward-sloping

Two forces push long yields above short yields in normal times.

The first is the **term premium** — extra compensation you demand for the risk of lending long. If you lend the government money for 10 years, a lot can go wrong over that decade: inflation could erode the real value of your repayments, interest rates could rise so that newer bonds pay more than yours, you might need your cash back and have to sell at a loss. To bear those risks you want to be paid, and that payment is a positive term premium baked into long yields.

The second is the **expectations component**. A 10-year yield is, very roughly, the market's average guess of where short-term interest rates will be over the next 10 years. If the economy is growing and the central bank is expected to keep rates around a normal level (or raise them as growth and inflation firm up), then the *average* expected short rate over a decade sits above today's short rate, and the long yield is higher. Put the two together — a positive term premium plus an expectation that short rates stay around normal or rise — and you get the textbook upward slope.

This is why slope is a *forecast*. The 2-year yield is anchored to where the market thinks the central bank's policy rate will be over the next two years. The 10-year reflects the longer arc. When the spread between them is wide and positive, the market is implicitly saying "growth and inflation are fine; the central bank will keep rates around here or higher." The shape of the curve is the bond market's collective economic forecast, priced in real money.

### What an inversion actually means

So what is the market saying when the curve inverts — when the 2-year yield rises *above* the 10-year? It is saying something specific and a little counterintuitive: *short rates are high now, but they will be lower later.* The central bank has hiked rates to fight inflation or cool an overheating economy, which pins up the front end (the 2-year). But investors expect that this tight policy will slow the economy enough that the central bank will have to *cut* rates in the future — and those expected future cuts pull the 10-year down below the 2-year.

An inversion, in other words, is the market pricing a coming downturn. It is not magic and it is not a prophecy; it is the aggregated bet of millions of investors that today's high rates will choke off growth and force the central bank into reverse. That is why it correlates with recessions: the same conditions that invert the curve — restrictive policy meeting a fragile economy — are the conditions that historically produce recessions. The curve does not *cause* the recession; it *prices* the conditions that tend to lead to one. (For the deeper mechanism of how the central bank's policy rate ripples out to every maturity, see [interest rates: the price of money](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) and the fixed-income primer on [what moves the yield curve](/blog/trading/fixed-income/what-moves-the-yield-curve-the-fed-growth-inflation-and-supply).)

Figure 1 above lays the two worlds side by side. A normal curve prices growth, lets banks earn on the spread, and supports cyclicals; an inverted curve prices cuts, squeezes bank margins, and starts the risk-off rotation. Hold that contrast in mind — everything else in this post is an elaboration of it.

### Which slope: 2s10s, 3m10y, and why the 2-year

You will see the curve's slope measured in more than one way, and the differences are not pedantic — they change *when* the signal fires and *what* it is telling you.

The **2s10s** (10-year minus 2-year) is the headline. Its appeal is that the 2-year is a clean expression of the market's *expected policy path over the next couple of years* — it already bakes in anticipated hikes or cuts. That makes the 2s10s a forward-looking measure: it can invert *early*, as soon as the market starts pricing future cuts, even before the central bank has finished hiking.

The **3m10y** (10-year minus 3-month) uses the very front of the curve, the 3-month bill, which is pinned almost exactly to the *current* policy rate. The Federal Reserve's own research has often favored this spread, arguing it has the cleanest statistical link to recession because the 3-month is the purest read on *current* monetary tightness. Because the 3-month tracks today's policy rather than expected future policy, the 3m10y tends to invert *later* than the 2s10s — it needs the central bank to have actually hiked, not merely to be expected to.

The practical upshot: in 2022 the 2s10s inverted in July (the market pricing future cuts) while the 3m10y did not invert until October (the Fed needing to hike enough to drag the 3-month above the 10-year). When the two disagree, that gap is itself a signal — the 2-year is telling you about *expected* policy and the 3-month about *current* policy. Most traders watch both: the 2s10s for the early warning, the 3m10y for the confirmation. For this post we use the 2s10s as the protagonist because it is the most widely quoted and the one in our data, but everything generalizes.

### A quick word on "correlation" for this series

Throughout this series we treat every macro relationship as a *statistical correlation* with four properties: a **sign** (does the asset go up or down when the indicator moves?), a **strength** (how reliable is the relationship, often measured by the correlation coefficient r, which runs from −1 to +1?), a **lead/lag** (does the indicator move *before* or *after* the asset, and by how long?), and a **regime-dependence** (when does the relationship flip?). The yield curve is a beautiful teaching case because it scores high on all four — strong sign, near-perfect historical strength, a long and useful lead — and yet 2022-24 shows even the best correlation is a regime, not a constant. If the term "correlation" is new to you, the series opener [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) builds the idea from scratch.

#### Worked example: reading the slope off two yields

Suppose you pull up two numbers on a screen: the 2-year Treasury yields 5.00% and the 10-year yields 3.90%. The 2s10s spread is 3.90 − 5.00 = **−1.10 percentage points**, i.e. −110 basis points. The curve is deeply inverted. Now translate that into English. The market is paying you *more* to lend for 2 years than for 10 — which only makes sense if it expects short rates to fall sharply over the coming decade. To make the 10-year (the average of expected future short rates plus a term premium) sit 110bp *below* the current 2-year, investors must be pricing a meaningful run of rate cuts ahead. If you held \$100,000 and naively chose the higher yield, you would lock in the 5.00% 2-year and collect \$5,000 a year — but you would be doing exactly what the inversion warns against: reaching for front-end yield right before the cuts that the curve is forecasting arrive and strand you reinvesting at far lower rates. The slope is not just a number; it is a sentence about the future, and here the sentence reads "cuts are coming, and growth is the reason."

### Decomposing the slope: the arithmetic of expectations

It is worth doing the decomposition once explicitly, because it makes the inversion intuitive rather than mysterious. The 10-year yield can be approximated as the average expected short rate over the next ten years, plus the term premium:

```
y10  ≈  average(expected short rates, years 1 to 10)  +  term_premium_10y
y2   ≈  average(expected short rates, years 1 to 2)   +  term_premium_2y
```

In a normal expansion, the central bank's policy rate is around, say, 2.5%, and the market expects it to stay near there. The average expected short rate over 2 years and over 10 years is roughly the same (~2.5%), but the 10-year carries a larger term premium (say +0.8%) than the 2-year (say +0.2%) because ten years of risk demands more compensation. So the 10-year sits around 3.3% and the 2-year around 2.7% — an upward slope of about +0.6%, driven entirely by the term premium.

Now run a tightening cycle. The central bank hikes the policy rate to 5.0% to fight inflation. The 2-year, dominated by the next two years of policy, jumps to roughly 5.0% plus its small term premium, call it 5.1%. But the 10-year is the *average* over ten years, and the market expects those 5.0% rates to be cut back toward 2.5% within a couple of years — so the average expected short rate over the full decade might only be ~3.5%, and even with a 0.8% term premium the 10-year sits around 4.3%. The slope is now 4.3% − 5.1% = **−0.8%, inverted**. Nothing exotic happened. The inversion is just arithmetic: when the front end is dominated by *current* high policy and the long end is dominated by *expected lower future* policy, the curve flips. The depth of the inversion is, roughly, a measure of *how many cuts and how soon* the market is pricing. A −1.08 reading like July 2023 says the market was pricing an aggressive cutting cycle within the decade — i.e. a forecast that the high rates would break something.

This decomposition also explains *why the term-premium argument matters so much for 2022-24*. If years of quantitative easing artificially suppressed `term_premium_10y` — pushing it toward zero or even negative — then the 10-year is held down *not* because the market is forecasting deep cuts, but because a non-economic buyer is sitting on the long end. In that world the curve can invert without carrying its usual "recession is coming" message, because part of the inversion is a term-premium distortion rather than a pure growth forecast. We return to this in the honesty section, but the arithmetic is the foundation: an inversion *can* mean "cuts coming because growth is dying," or it *can* mean "the long end is artificially depressed" — and the two look identical on the screen.

## The slope as a forecast: the empirical correlation with recessions

We have the intuition. Now the evidence. The reason the yield curve became famous is not theory — it is an empirical track record that is almost embarrassingly good. In the modern US era (roughly the late 1960s onward), an inversion of the 2s10s curve has preceded every single recession, and there have been very few "false positives" where the curve inverted and no recession followed. Few macro indicators come close to that hit rate.

Figure 2 charts the recent episode in detail: the 2s10s from 2021 through 2026, the zero line that separates "normal" from "inverted," the shaded inverted region, the record −1.08 trough in July 2023, and the un-inversion in late 2024.

![The 2s10s spread from 2021 to 2026 with the inversion shaded and the deepest point marked](/imgs/blogs/the-yield-curve-as-a-growth-signal-and-its-asset-correlation-2.png)

Read the path left to right. In mid-2021 the curve was steep, +1.20 percentage points — a normal, growth-pricing shape. As the Federal Reserve began the fastest hiking cycle in four decades to fight 2022's inflation, the front end (the 2-year) shot up, and the spread collapsed: +0.79 at the end of 2021, +0.19 by April 2022, and then *below zero* in July 2022. It kept sinking through 2022 and into 2023, bottoming at −1.08 in July 2023 — the deepest inversion since 1981. It then clawed back toward zero and finally crossed above it in late 2024, un-inverting after roughly two and a half years below the line.

### How much lead time, exactly?

The correlation between an inversion and a recession is not contemporaneous — it *leads*. That is what makes it valuable: it warns you before the downturn arrives, giving you time to position. But how much warning?

Figure 3 shows the historical lead from the first 2s10s inversion to the official start of the recession, for each modern episode.

![Months from yield curve inversion to recession start by historical episode](/imgs/blogs/the-yield-curve-as-a-growth-signal-and-its-asset-correlation-3.png)

The 1989 inversion led the recession by about 18 months. The 2000 inversion led by 13 months. The 2006 inversion led by 22 months — nearly two years. The 2019 inversion led by only 6 months, but that was COVID, an exogenous shock that pulled the recession forward; absent the pandemic the lead would likely have been longer. Across those four episodes the *average* lead is roughly 14-16 months. That figure matches the cross-correlation work in the [lead-lag taxonomy](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators) post, where the yield curve's cross-correlation with recessions peaks at about a 14-month lead — the longest, earliest warning of any single indicator we track.

And then there is the fifth bar: the 2022 inversion, with a lead of **zero** — because as of this writing no recession had arrived at all. That is the false-signal case, and we devote a whole section to it below. But note what it does to the "average": include it and the mean lead collapses, which is itself a lesson in how a single regime-breaking observation can wreck a clean historical relationship.

### What "strength" means for this correlation

A signal can have the right sign and the right lead and still be useless if it fires randomly. So how *strong* is the curve's recession correlation, in the statistical sense?

The honest answer is that it is hard to put a clean correlation coefficient on, precisely because recessions are rare. There have only been a handful of modern US recessions, so any "the curve predicts recessions with r = 0.x" claim rests on a sample of roughly seven or eight events. That is a tiny dataset, and it is exactly why you should be suspicious of confident precision. What we *can* say is two things. First, the *hit rate* — the fraction of inversions followed by a recession — was essentially perfect from the 1960s through 2019: every inversion was followed by a recession, and there were very few inversions without one. Second, the *cross-correlation* of the curve slope with the future output gap peaks at a lead of roughly 12-18 months, with a meaningfully negative coefficient (an inverted curve associates with a future contraction). Those two facts are what earned the curve its reputation.

But here is the statistical trap, and it is the spine of this whole series. A near-perfect hit rate over seven events is *not* the same as a near-perfect hit rate over a thousand events. With a small sample, even a genuinely informative signal will show an inflated, too-clean track record, because you have not yet seen the cases where it fails. The 2022-24 episode may simply be the first observation in the part of the distribution we had never sampled — the inversion that does *not* lead to recession. One failure out of eight events drops the apparent hit rate from "100%" to "88%," a huge revision, which is precisely what fragile small-sample statistics do. The deeper treatment of why small macro samples mislead is in [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data); the takeaway here is that the curve's "strength" is real but *overstated* by a small sample, and you should treat it as a strong prior rather than a law of nature.

#### Worked example: turning lead time into a position calendar

Say the 2s10s inverts for the first time this month and you take the signal at face value. The historical lead is roughly 14 months (call the useful range 12 to 18). That does *not* mean "sell everything today." It means you have a window. A practitioner reading this would not flip to cash on day one — they would build a calendar. Roughly the first 6-9 months after inversion have historically been *fine* for equities (the late-cycle melt-up often happens *with* the curve inverted), so you stay invested but start trimming the most cyclical, most leveraged exposure. As you move into months 10-18, you rotate progressively toward defensives, lengthen bond duration, and raise cash. Concretely: if you manage a \$1,000,000 portfolio and your plan is to cut equity beta from 1.0 to 0.6 over the back half of the window, you might sell \$400,000 of high-beta cyclicals between months 9 and 15 and park it in long Treasuries and cash. The point is that the inversion's *lead* is the whole value of the signal — it converts a vague worry into a dated plan — but the lead is also a *range*, and acting on the first day is as much an error as ignoring it entirely.

## The asset correlation: how the slope moves what you trade

The yield curve is not just a recession oracle; it is a *correlation tool*. The slope is mechanically linked to several asset and sector outcomes, which means you can use it to position long before the recession verdict is in. Three linkages matter most: banks, the cyclical-versus-defensive rotation, and the dollar.

### Banks and the net interest margin

The cleanest mechanical link is to banks. A bank's core business is *maturity transformation*: it borrows short (deposits, which it can repay on demand and which pay a low rate tied to the front end) and lends long (mortgages, business loans, which pay a rate tied to longer yields). Its profit on that activity is the **net interest margin** (NIM) — roughly the long lending rate minus the short funding rate. Notice that NIM is, almost literally, *the slope of the yield curve*.

So when the curve is steep, banks earn a fat spread on every dollar they intermediate, and bank stocks tend to do well. When the curve flattens, the spread compresses and NIM falls. When the curve *inverts*, a bank that borrows at the high front-end rate and lends at the lower long rate is being squeezed from both sides — its new lending is barely profitable, and the market marks bank stocks down accordingly. The correlation between curve slope and bank-sector relative performance is one of the most dependable in macro.

There is an important subtlety here that the 2023 regional-bank crisis made painfully concrete. The NIM squeeze is the *flow* problem — new lending earns a thin spread. But an inverted curve also creates a *stock* problem: a bank that already holds a portfolio of long-dated, low-yielding bonds and loans (bought when long rates were low) sees the market value of that portfolio *fall* as rates rise, even as its funding cost (deposits, which can flee to higher-yielding alternatives) *rises*. That is a duration mismatch, and it is what felled Silicon Valley Bank in March 2023 — a deeply inverted curve is the macro environment in which the duration mismatch on a bank's balance sheet turns from a footnote into a solvency question. So the curve's correlation with bank stress runs through two channels at once: the margin on new business (flow) and the mark-to-market on the existing book (stock). Both worsen as the curve inverts, which is why bank equity and the curve slope move together so reliably. The duration mechanics behind that mark-to-market hit are worked through in the fixed-income post on [duration, the most important number in fixed income](/blog/trading/fixed-income/duration-the-most-important-number-in-fixed-income).

#### Worked example: a dollar bank-versus-defensive trade on a curve move

Suppose the 2s10s steepens by 30 basis points over a quarter — say from −0.50 to −0.20 — because the market starts pricing rate cuts that will drop the front end. You believe this re-steepening helps bank margins relative to rate-insensitive defensives. You put on a \$100,000 long position in a bank-sector ETF and a matching \$100,000 short in a consumer-staples (defensive) ETF — a market-neutral pair, so you are betting purely on the *spread* between them, not on the market direction.

Historically a re-steepening of that size has been worth a few percent of relative outperformance for banks over a quarter. Say banks rise 6% and staples rise 2% over the period. Your long earns \$100,000 × 6% = **\$6,000**; your short *loses* you the 2% the staples rose, i.e. −\$100,000 × 2% = **−\$2,000**. Net profit on the pair: \$6,000 − \$2,000 = **\$4,000**, on roughly \$100,000 of capital at risk (the margin for the short), or about a 4% return — and crucially, you made it whether the overall market went up or down, because the curve move drove the *relative* performance. The intuition: the curve slope *is* the bank margin, so trading banks-versus-defensives is a clean way to express a view on the curve without taking outright market risk.

There is, however, a sting in the tail. The *kind* of steepening matters enormously, and that is where the four-quadrant map comes in.

### The four curve moves: bull versus bear, steepen versus flatten

"The curve steepened" is ambiguous. It could mean the long end rose (good for growth, good for banks) or the short end fell (the central bank is cutting because the economy is sick). Practitioners disambiguate with two words. **Bull** versus **bear** says which direction *yields* moved (bull = yields fell, prices rose; bear = yields rose, prices fell). **Steepen** versus **flatten** says what happened to the *slope*. Combine them and you get four moves, each with a different cycle meaning and a different trade.

![The four yield curve moves: bull and bear, steepen and flatten, and what each means](/imgs/blogs/the-yield-curve-as-a-growth-signal-and-its-asset-correlation-4.png)

Figure 4 lays out the 2×2. Read it carefully — this is the grammar of the curve:

- **Bear steepener** (long end rises fastest, yields up): reflation. Growth and inflation expectations are rising, often with a dose of fiscal-supply worry pushing long yields up. This is early-cycle or reflationary, and it is *good* for banks and cyclicals. Trade: long cyclicals and banks.
- **Bear flattener** (front end rises fastest, yields up): the central bank is hiking into the economy. This is the mid-cycle tightening phase — the front end gets dragged up by policy faster than the long end. Trade: fade long duration, expect the curve to keep flattening toward inversion.
- **Bull flattener** (long end falls fastest, yields down): growth and inflation fears are *fading*; investors pile into long bonds for safety and the long end rallies. Late-cycle, duration outperforms. Trade: long the 10-year, fade cyclicals.
- **Bull steepener** (front end falls fastest, yields down): the central bank is *cutting* into a slowdown. The front end collapses as cuts are priced and delivered. **This is the classic recession trigger.** Trade: long bonds, long defensives, reduce risk.

The progression around the clock is itself the business cycle: bear steepener (early) → bear flattener (mid, hiking) → inversion (late) → bull flattener (fears building) → bull steepener (the cut, the recession). For the cross-asset view of those phases, see [the business cycle's four phases](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) and the sibling [business cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock).

Figure 5 draws the four moves as actual curves, so you can *see* which end did the moving in each case.

![Four panels showing the curve shape before and after a bull steepener, bull flattener, bear steepener, and bear flattener](/imgs/blogs/the-yield-curve-as-a-growth-signal-and-its-asset-correlation-5.png)

#### Worked example: bull-steepening versus bear-steepening P&L on the same slope move

Here is the trap that catches beginners. Two different scenarios both "steepen the curve by 40bp," and a naive trader who is just "long steepeners" treats them identically. They are opposites.

*Scenario A — bear steepener.* The economy is reflating. The 10-year rises from 4.0% to 4.6% (+60bp) while the 2-year rises from 4.2% to 4.4% (+20bp). The slope went from −0.20 to +0.20, a 40bp steepening. Bank stocks love this — wider NIM, growth intact. If you are long \$50,000 of banks, a typical reflationary quarter might give them +8%, or **+\$4,000**. Long bonds, meanwhile, *lost* money because yields rose; a \$50,000 position in a long-Treasury fund with ~17-year duration falls roughly 17 × 0.60% ≈ 10%, or **−\$5,000**.

*Scenario B — bull steepener.* The economy is rolling over. The Fed is cutting, so the 2-year *collapses* from 4.4% to 3.6% (−80bp) while the 10-year only eases from 4.0% to 3.6% (−40bp). The slope went from −0.40 to 0.00, a 40bp steepening — *the same 40bp as Scenario A*. But now banks are pricing recession and underperform; your \$50,000 of banks might fall 6%, or **−\$3,000**. Long bonds, by contrast, *rallied* as yields fell; that same \$50,000 long-Treasury position rises about 17 × 0.40% ≈ 7%, or **+\$3,500**.

Same 40bp steepening, mirror-image P&L. The lesson: never trade "the slope" in isolation — trade *which end is moving and why*. A steepening driven by the long end rising (bear) is risk-on; a steepening driven by the front end falling (bull) is risk-off and is the recession signal. Confusing the two is the single most expensive mistake in curve trading.

### The dollar

The third linkage is the US dollar. When the front end of the curve is pinned high — the central bank holding policy rates up to fight inflation — US short-term assets pay a lot relative to the rest of the world, which attracts capital and tends to *firm the dollar*. The mechanism is the interest-rate differential: a global investor choosing between parking cash in US 2-year bills at 5% or in a foreign equivalent at 2% will, all else equal, prefer the dollar asset and bid the dollar up. So a deeply inverted curve (high front end) often coexists with a strong dollar, and the relationship between the dollar and risk assets is itself a correlation we cover elsewhere: a stronger dollar is cross-asset gravity, pressuring commodities, emerging markets, gold, and crypto. The DXY's broad correlation footprint is mapped in [the dollar as cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity); here the point is just that the curve's *level* (how high the front end sits) feeds the dollar, and the dollar feeds everything else.

This is why the inversion's asset footprint is broader than just banks and cyclicals. An inverted curve typically means: a high front end → a firm dollar → pressure on dollar-priced commodities (oil, copper), on emerging-market assets that have dollar-denominated debt, and on gold and crypto, which compete with the high yields now available on cash. So when you read an inversion, you are not just reading "recession risk in ~14 months" — you are reading a whole regime: defensives over cyclicals, quality over junk, a firm dollar, and a headwind for anything that lives on the other side of a strong dollar. The curve slope is one of the most information-dense numbers in all of macro precisely because it sits at the junction of growth expectations, policy, and the dollar.

#### Worked example: the inversion's cross-asset checklist in dollars

Suppose you run a \$1,000,000 multi-asset book and the 2s10s has just inverted to −0.50 while the front end sits at 5%. You decide to lean the portfolio toward the regime the inversion describes, shifting \$200,000 of exposure. The inversion checklist argues for: trimming \$80,000 of high-beta cyclicals and adding it to defensives and quality; trimming \$60,000 of emerging-market equity (vulnerable to the firm dollar) into US large-cap quality or cash; and trimming \$60,000 of gold and high-beta commodity exposure given the firm-dollar, high-real-yield headwind, parking it in T-bills earning that 5% front-end yield (worth \$60,000 × 5% = **\$3,000** a year in carry while you wait). None of these are all-or-nothing bets; each is a *tilt* in the direction the curve is pointing. The intuition: an inversion is not one trade, it is a coordinated regime shift across banks, sectors, the dollar, and everything the dollar touches — and a disciplined book leans into all of those tilts at once, sized modestly, rather than making a single dramatic recession bet.

## The un-inversion is the real trigger

Here is the most important — and most misunderstood — fact about the yield-curve signal. The recession does not usually arrive while the curve is *most* inverted. It arrives as the curve *un-inverts* — specifically, as the front end falls in a bull steepener because the central bank has started cutting.

Why? Walk the logic. The curve inverts because policy is tight and the market expects cuts. For months, the inversion persists while the economy, running on momentum and accumulated savings, holds up. Then the data finally cracks — unemployment ticks up, the central bank pivots to cutting — and the front end *plunges*. That is the bull steepener. And the cuts that cause it are happening *because* the economy is now visibly weakening. So the curve un-inverts and the recession lands at roughly the same time. The signal that should scare you is not the deepest inversion; it is the *re-steepening from the front end* after a long inversion. By the time the curve looks "normal" again, the damage is often already underway.

This is the single most counterintuitive thing about the indicator, and it trips up almost everyone the first time. The natural instinct is "the more inverted the curve, the closer the recession" — as if depth of inversion were a countdown timer. It is not. Depth of inversion measures *how aggressively the market expects the central bank to cut*, which is a measure of how tight policy is *right now* relative to where it is expected to go — not a measure of how imminent the downturn is. The downturn becomes imminent when the cuts the curve has been forecasting actually start, because cuts only start when the economy is visibly breaking. So the un-inversion is not the "all clear"; it is the alarm. A trader who buys risk because "the curve is normalizing, the recession scare is over" is making exactly the wrong read at exactly the wrong moment. The normalization *is* the recession arriving. Watch the 2-year: a sharp, sustained fall in the 2-year yield after a long inversion — that is the front end pricing real, delivered cuts — is the practical trigger to shift toward the recession playbook.

Figure 6 shows what that recession regime does to asset classes — the rotation the un-inversion delivers.

![Average real returns by asset class across the four business cycle phases with the recession phase highlighted](/imgs/blogs/the-yield-curve-as-a-growth-signal-and-its-asset-correlation-6.png)

In the recession column, the pattern is stark and consistent with everything above: stocks fall hard (roughly −10% real), commodities fall (−8%), while government bonds rally strongly (+10%) and cash holds its value (+1%). This is the "flight to quality" — capital flees risk and crowds into the safest, most liquid assets. It is also exactly why the bull steepener (front-end-led rally in bonds) coincides with it: the same cuts that re-steepen the curve are the cuts that send bonds higher and stocks lower. Note the mirror-image in the *Late / Overheat* column (commodities lead, bonds suffer) — that is the bear-flattener/inversion phase that precedes the recession. The full rotation by phase, with the underlying numbers, is in the [business cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock).

Figure 7 ties the whole transmission together as a chain: from inversion, through the three asset linkages, to the re-steepening that triggers the recession.

![Transmission chain from yield curve inversion through bank margins, sector rotation, and the dollar to the re-steepening recession trigger](/imgs/blogs/the-yield-curve-as-a-growth-signal-and-its-asset-correlation-7.png)

#### Worked example: timing the un-inversion, not the inversion

Imagine you went to cash the day the curve inverted in July 2022, with the S&P 500 around 3,900. "The recession is coming," you told yourself. Over the next two years the market climbed past 5,000 — a gain of roughly 28% you missed by acting on the *inversion* rather than the *trigger*. On a \$200,000 portfolio, sitting in cash through that move cost you about \$200,000 × 28% = **\$56,000** of forgone gains (before counting the few percent of interest you earned on the cash).

Now walk through the disciplined version. You note the inversion, you mark the calendar, but you stay invested with a *plan*: trim risk only as the front end starts to fall in a genuine bull steepener. You watch the 2-year. The day it drops 80-100bp over a quarter because the central bank has pivoted to cutting — *that* is your signal to rotate into long bonds and defensives, because the un-inversion is the trigger. By waiting for the re-steepening you keep the late-cycle gains and still position before the recession rotation. The intuition: the inversion tells you a storm is *possible*; the front-end re-steepening tells you it has *started*. Trade the second, not the first.

## The honesty caveat: the 2022-24 false signal

We have spent this whole post praising the yield curve. Now the discipline: it just gave its most famous false signal in decades, and an honest practitioner has to sit with that rather than explain it away.

The facts are not in dispute. The 2s10s inverted in July 2022. It reached −1.08 in July 2023, the deepest reading since 1981. It stayed inverted for about two and a half years — longer than any inversion in the modern record. By every historical rule, a recession should have arrived somewhere in the 2023-24 window. None did. Growth stayed positive, unemployment stayed near full employment, corporate earnings grew, and equities made new highs. That is the "0" bar in Figure 3, and it is the elephant in the room for every yield-curve enthusiast.

So what happened? There are several non-exclusive explanations, and the honest answer is that we do not yet know which dominates:

1. **Structural shift in the term premium.** A decade-plus of central-bank bond-buying (quantitative easing) and strong global demand for safe US assets may have *artificially depressed* long yields. If the long end is held down by forces unrelated to the growth forecast, then the curve can invert without carrying its usual recessionary message — the inversion is partly a term-premium artifact, not a pure growth signal. This is the leading "the rules changed" argument, and it connects directly to the broader question of [why today's correlations are not yesterday's](/blog/trading/macro-correlations/structural-shifts-why-todays-correlations-arent-yesterdays).

2. **A genuinely unusual cycle.** The 2020-21 fiscal stimulus left households and firms with unusual cash buffers and locked-in low-rate debt, blunting the usual transmission from high policy rates to economic pain. The economy may simply have been more rate-insensitive this cycle, delaying (or canceling) the recession the curve forecast.

3. **It is not a false signal yet — just a long lead.** The 2006 inversion led by 22 months; leads of two years are within the historical range. The recession may simply be later than usual, not absent. This is the "be patient" camp, and it cannot be ruled out.

The deeper statistical lesson is the spine of this whole series: **a correlation is a regime, not a law.** The yield curve's recession-forecasting power was estimated over a particular monetary regime (positive term premia, conventional policy). Change the regime — via QE, via fiscal dominance, via a rate-insensitive economy — and the correlation can weaken or break exactly when you are most relying on it. This is also a textbook case of why you must be suspicious of any single indicator: small samples (there have only been a handful of modern recessions) make the "always works" claim statistically fragile, a trap explored in [spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data).

#### Worked example: sizing a position when the signal might be broken

You believe the inversion still carries *some* recession signal, but you also take seriously that the regime may have shifted, so the probability is lower than history implies. How do you size?

Suppose history says an inversion implies a 90% chance of recession within 18 months, but you haircut that to a 50% chance given the structural-shift arguments and the fact that the signal has already been "wrong" for a year. A full recession-hedge position — long bonds, short cyclicals, raised cash — might cost you, in expectation, 3% of carry and forgone upside per year if you are *wrong* (the market keeps rising and you are under-invested), while it saves you perhaps 15% if you are *right* (you dodge the drawdown). On a \$500,000 portfolio, being wrong costs about \$500,000 × 3% = **\$15,000**; being right saves about \$500,000 × 15% = **\$75,000**. Naive expected value at a 50% probability still favors *some* hedge: 0.50 × \$75,000 − 0.50 × \$15,000 = **+\$30,000** expected. But the right move is not "all in" — it is to scale the hedge to your *haircut* probability: hedge maybe a third to a half of the portfolio rather than the whole thing, so a broken signal does not wreck you and a working one still protects you. The intuition: when a famous correlation might be in a regime break, you do not abandon it *or* bet the farm on it — you size it down and stay humble.

## Common misconceptions

Several beliefs about the yield curve are widespread, intuitive, and wrong. Each is worth correcting with a number.

**Myth 1: "The recession hits when the curve is most inverted."** No — it hits as the curve *un-inverts* from the front end. In the 2007-08 episode, the curve had already re-steepened well off its lows by the time the recession bit; the deepest inversion came months *before* the worst of the downturn. The signal to act on is the bull steepener (front-end-led re-steepening after a long inversion), not the trough. Trading the trough means going risk-off during the late-cycle melt-up, which is often the most expensive time to be in cash — see the \$56,000 missed-gain example above.

**Myth 2: "An inversion means sell stocks now."** No — the lead is roughly 14 months on average, and equities have historically performed *fine* for the first 6-9 months after an inversion. The 2022 inversion was followed by the S&P rising roughly 28% over the next two years. Inversion is a warning to *plan*, not a trigger to *act* on day one.

**Myth 3: "The 3m10y and the 2s10s say the same thing."** Mostly, but not always, and the difference matters. The 3-month/10-year spread uses the very front of the curve, which is pinned almost exactly to the current policy rate, so it inverts a bit later and is favored by some researchers (the Fed's own preferred model uses it). The 2s10s incorporates a couple of years of rate *expectations* in the 2-year, so it can invert *earlier*, as the market begins pricing cuts. In 2022 the 2s10s inverted in July while the 3m10y did not invert until October. When the two disagree, the 2-year is telling you about *expected* policy while the 3-month is telling you about *current* policy — and the gap between them is itself information.

**Myth 4: "A steeper curve is always bullish for risk."** No — it depends on *which end moved*. A bear steepener (long end rising on reflation) is risk-on; a bull steepener (front end falling on rate cuts) is the recession trigger and is risk-off. As the worked example showed, the same 40bp steepening can produce mirror-image P&L depending on whether bonds rallied or sold off into it. "Steepening" without specifying bull or bear is a half-sentence.

**Myth 5: "The yield curve is broken now, so ignore it."** Too strong. The 2022-24 episode is a genuine puzzle and a real warning that the correlation may be in a regime shift. But "weaken" is not "reverse," and the recession may yet have been merely delayed. The disciplined response is to *down-weight* the signal and combine it with confirming indicators (claims, credit spreads, ISM), not to throw it out. A signal that has been right for fifty years and wrong once deserves a haircut, not a funeral.

## How it shows up in real markets

Abstract rules become believable when you watch them play out on real dates. Three episodes show the curve doing its job — and one shows it apparently failing.

**2000-2001: the dot-com inversion.** The 2s10s inverted in early 2000 as the Fed hiked into a frothy, late-cycle economy with the Nasdaq in full mania. The curve was telling you the same thing it always does — high front-end rates would eventually break something — and for a while the bulls dismissed it as the bond market being too gloomy about a "new economy." The recession arrived in March 2001, about 13 months after the inversion, almost exactly on the historical schedule. The tech-heavy equity market had already begun its collapse in March 2000, and the recession formalized what the curve had forecast. This episode is a clean reminder that the curve does not care about the narrative of the day ("this time the internet changes everything"); it prices the arithmetic of policy and growth, and the arithmetic won.

**2006-2008: the textbook run.** The 2s10s inverted in early-to-mid 2006 with the housing boom still roaring and the Fed holding rates high. For more than a year the inversion was the punchline of "the bond market is wrong" commentary while equities ground higher into late 2007. Then the cracks appeared — subprime, Bear Stearns — the Fed began cutting hard in late 2007, the front end collapsed, and the curve bull-steepened violently. The recession that the 2006 inversion had forecast (with a ~22-month lead, the longest in our sample) arrived in December 2007, and the equity drawdown that followed was the worst since the Depression. The curve did exactly what the theory says: inverted as the warning, re-steepened from the front end as the trigger, recession on the un-inversion.

**2019: the short fuse.** The 2s10s briefly inverted in mid-2019 amid a trade-war growth scare and a Fed that had over-tightened in 2018. The historical clock said a recession was likely within a year or so. It came in early 2020 — but as COVID, an exogenous shock, not the organic slowdown the curve had been pricing. The lead was only 6 months, the shortest in the modern record, because the pandemic pulled everything forward. This episode is a reminder that the curve forecasts the *conditions* for recession; it cannot forecast a meteor.

**2022-2024: the great false alarm (so far).** As covered above, the deepest inversion since 1981 produced no recession at the time of writing. Banks did suffer — the March 2023 regional-bank crisis (Silicon Valley Bank and others) was in part a story of institutions caught with long-duration assets funded by suddenly expensive short deposits, exactly the NIM squeeze the inverted curve implies. So the *asset correlation* (inversion → bank stress) fired even though the *recession correlation* did not. That is an important nuance: the curve's linkage to bank profitability worked precisely as advertised; it was the economy-wide recession call that misfired. The credit-market angle on that stress — spreads as the risk canary — is covered in [credit spreads: the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary).

**The labor-market cross-check.** A practitioner never trades the curve alone. The yield curve is a *leading* indicator with a long, noisy lead; the labor market is a *coincident-to-lagging* confirmation. When the curve has been inverted for a year and *then* initial jobless claims start trending up and the unemployment rate ticks above its 12-month average (the so-called Sahm rule), the leading signal and the real-time signal agree, and the recession case firms up dramatically. Pairing the curve with claims is one of the most robust setups in macro — the curve gives you the early warning, the claims give you the confirmation. That pairing is the subject of [unemployment claims and the recession correlation](/blog/trading/macro-correlations/unemployment-claims-and-the-recession-correlation).

## How to read it and use it: the playbook

Pulling everything together, here is how a disciplined trader or allocator actually uses the yield curve as a growth signal and a correlation tool.

**The signal.** Watch the 2s10s spread (10-year minus 2-year, in basis points). Above roughly +50bp, the curve is normal and risk-on for cyclicals and banks. Between 0 and +50, it is flattening — late-cycle, start trimming. Below 0, it is inverted — the recession clock is running, but you have a 12-18 month *window*, not a same-day trigger. The trigger to act on is not the trough; it is the **bull steepener**, the front-end-led re-steepening after a long inversion, which signals the cuts that historically coincide with the recession.

**The regime check.** Before you trust the magnitude of the signal, ask whether the regime supports it. Is the term premium being artificially depressed by central-bank bond-buying or strong safe-asset demand? Is the economy unusually rate-insensitive (locked-in low-rate debt, large cash buffers)? If yes, *haircut* the recession probability the inversion implies — the 2022-24 episode is the cautionary tale. The curve is a strong prior, not a certainty.

**The cross-asset map.** Use the slope to position relatives even before the recession verdict is in. A flattening or inverting curve argues for: underweight banks (NIM squeeze), underweight cyclicals versus defensives, and respect for a firmer dollar (front-end yields high). A bull steepener (the trigger) argues for: long government bonds (they rally as the front end falls and as recession hits), long defensives and quality, raised cash, and reduced equity beta. A bear steepener (reflation) argues for the opposite: long banks and cyclicals, fade duration.

**What invalidates it.** The signal is weakened or invalidated if (a) the inversion is plausibly a term-premium artifact rather than a growth forecast (regime shift); (b) the confirming indicators — claims, credit spreads, ISM new orders — *fail* to deteriorate even many months into the inversion (as in 2023-24); or (c) the un-inversion happens as a *bear* steepener (long end rising on reflation) rather than a *bull* steepener (front end falling on cuts), which means the economy is reaccelerating, not rolling over. Always specify *which end is moving and why* before you trade the slope. And never trade the curve in isolation: pair the leading curve signal with a coincident labor-market confirmation before sizing up the recession bet.

The yield curve earned its reputation as the premier growth signal honestly — over half a century it has been the earliest, most reliable warning of recession we have. But 2022-24 is the reminder that even the best correlation is a regime, not a constant: the slope is a sentence about the future written in the bond market's collective hand, and like any forecast it can be wrong, especially when the rules change underneath it. Read it, respect it, confirm it, and size it humbly. The curve is the best single growth signal we have ever found, which is exactly why it deserves to be handled with care rather than worship — a tool this powerful is most dangerous in the hands of someone who has forgotten it can be wrong.

## Further reading and cross-links

Within this series:

- [Correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) — the foundational idea behind every post here.
- [Unemployment claims and the recession correlation](/blog/trading/macro-correlations/unemployment-claims-and-the-recession-correlation) — the coincident confirmation that pairs with the leading curve.
- [The business cycle correlation clock](/blog/trading/macro-correlations/the-business-cycle-correlation-clock) — the full asset rotation by cycle phase.
- [Lead, lag, leading, coincident, and lagging indicators](/blog/trading/macro-correlations/lead-lag-leading-coincident-and-lagging-indicators) — where the curve sits in the indicator taxonomy and its ~14-month lead.
- [Credit spreads: the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary) — the other great recession canary, and how it confirms the curve.
- [Structural shifts: why today's correlations aren't yesterday's](/blog/trading/macro-correlations/structural-shifts-why-todays-correlations-arent-yesterdays) — the term-premium and regime-break arguments for why 2022-24 misfired.
- [Spurious correlation and the traps of macro data](/blog/trading/macro-correlations/spurious-correlation-and-the-traps-of-macro-data) — why small recession samples make "always works" fragile.

The mechanism and fixed-income depth:

- [Reading the yield curve: slope, inversion, recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession) — the macro-mechanism companion to this correlation post.
- [Interest rates: the price of money, the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — how the policy rate ripples to every maturity.
- [The business cycle: four phases for traders](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) — the cycle framework behind the curve moves.
- [The yield curve explained: the most important chart in finance](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance) — the fixed-income primer on curve construction.
- [Why the yield curve usually slopes up: term premium and expectations](/blog/trading/fixed-income/why-the-yield-curve-usually-slopes-up-term-premium-and-expectations) — the two forces behind the normal slope.
- [Yield curve inversion: the recession signal that mostly works](/blog/trading/fixed-income/yield-curve-inversion-the-recession-signal-that-mostly-works) — the fixed-income deep dive on the inversion signal.

---
title: "The Macro-Asset Correlation Matrix: One Map for Every Indicator and Asset"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "A single grid that lays out the sign and strength of how every macro driver moves every asset price, how to read a row versus a column, and why each cell is a regime-average that can flip."
tags: ["macro", "correlation", "cross-asset", "interest-rates", "real-yields", "dollar", "credit-spreads", "regime", "asset-allocation", "matrix"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Almost every macro driver (yields, real yields, hot inflation, a stronger dollar, oil, growth surveys, credit spreads) moves risk assets in a predictable direction, and you can lay the whole field out on one grid where each cell is the sign and strength of a driver-asset correlation — but every cell is a regime-average, and the famous ones flip.
>
> - **Read a row** to see one driver's footprint across all assets; **read a column** to see everything that moves one asset.
> - **Rising rates and real yields are the dominant negative row** — they pull down stocks, bonds, gold, crypto and emerging markets all at once. A stronger dollar is the cross-asset headwind; wider credit spreads are the risk-off row; rising growth surveys are the one cleanly risk-on row.
> - The matrix is a **starting point, not the truth**: the stock-bond correlation flipped from roughly −0.45 to +0.6 in the 2022 inflation shock, and gold's clean −0.8 link to real yields broke as central banks bought.
> - **The one number to remember:** a rise in the 10-year real yield carries about a **−0.8** correlation with gold and roughly **−0.5 to −0.65** with every risk asset — real yields are the closest thing the matrix has to a master switch.

In late 2022, a fund manager who had spent a career believing bonds protect you when stocks fall watched both halves of a classic 60/40 portfolio fall together, month after month. The S&P 500 dropped about 18% on the year. Long Treasuries — the supposed safe haven — dropped *more*, around 30% for the longest-duration funds. The "diversification" that had worked for two decades simply stopped working. Nothing was broken in the machinery of markets. What had changed was the *correlation*: the statistical relationship between stocks and bonds had flipped sign, from negative (they zig when stocks zag) to positive (they fall together), because the thing the whole market was suddenly afraid of had changed from recession to inflation.

That episode is the entire thesis of this series in one story. Markets are not a list of separate assets each marching to its own drummer. They are a web of relationships — stocks to bonds, gold to real yields, emerging markets to the dollar, crypto to the Nasdaq — and those relationships have a *sign*, a *strength*, a *lead-lag*, and, crucially, they *change with the regime*. If you can hold the whole web in your head as a single map, you stop being surprised. You can look at any macro release — a hot inflation print, a strong jobs number, a wider credit spread — and immediately read off which way every asset you hold is likely to move, and roughly how hard.

This post is that map. It is the hub of the whole "Macro Correlations" series: a single matrix with macro drivers down the rows and assets across the columns, where each cell is the sign-and-strength of one correlation. Every later post in the series zooms into one row or one cell and goes deep. Here we build the map, learn to read it, walk the four most important rows, and — most importantly — learn why no cell on it is a fixed constant.

![Macro drivers reaching every asset price through a few channels](/imgs/blogs/the-macro-asset-correlation-matrix-1.png)

The figure above is the mental model. Notice that four very different-looking drivers on the left — rates, growth surveys, the dollar, credit spreads — funnel into just a handful of *channels* in the middle, and the most important channel by far is the discount rate. That is the secret that makes a single map possible: most macro news reaches most assets through the *same* mechanism, so the correlations are not a random scatter of 49 independent numbers. They are structured, and the structure is learnable.

## Foundations: what a correlation actually is, and what the matrix measures

Before we read the map, we need to define the one word the whole series rests on, because most people use it loosely. A **correlation** is a single number, written **r**, that measures how two things move *together*. It runs from −1 to +1.

- **r = +1** means perfect positive co-movement: every time one goes up by some amount, the other goes up in lockstep.
- **r = −1** means perfect negative (inverse) co-movement: one up, the other down, every time, in proportion.
- **r = 0** means no linear relationship at all: knowing one tells you nothing about the other.

Everyday analogy first, before any math. Think of two friends walking. If they are holding hands, when one steps forward the other steps forward too — that is r near +1. If they are on opposite ends of a seesaw, one rises exactly as the other falls — that is r near −1. If they are strangers in a crowd wandering independently, their steps have no relation — that is r near 0. A correlation is just a number that says *how tightly the two are linked, and in which direction*.

The formula, for completeness, is the covariance of the two series divided by the product of their standard deviations:

```
r = Cov(X, Y) / (sigma_X * sigma_Y)
   = sum((x_i - x_bar)(y_i - y_bar)) / sqrt(sum((x_i - x_bar)^2) * sum((y_i - y_bar)^2))
```

You almost never compute this by hand — a spreadsheet or one line of Python does it (`np.corrcoef`, or `df.corr()`). What matters is the *interpretation*: the sign tells you direction, and the magnitude tells you reliability. An r of −0.9 is a near-iron law; an r of −0.2 is a faint tendency you will see broken constantly.

A close cousin you will meet constantly in this series is **beta**. Where correlation is dimensionless (just a number from −1 to +1), beta is a *slope with units*: it says how much asset Y moves *per unit* of move in driver X. If gold has a beta of −0.8% per +0.1 percentage point of real yield, that is a concrete, tradeable magnitude. Correlation tells you whether a relationship is reliable; beta tells you how big the move is. We will use both, and most cells in our matrix encode the *sign and relative strength* — closer to a normalized correlation — while the worked examples translate them into betas with real dollar magnitudes.

Now, what exactly does our matrix measure? Each cell is the representative correlation between a *change or surprise* in a macro driver (the row) and the *return* of an asset (the column). "Change or surprise" is doing important work there. Markets are forward-looking: they price in what everyone already expects. So it is not the *level* of inflation that moves the S&P on CPI day — it is the *surprise*, the gap between the print and what was expected. (We build that idea fully in the event-trading series; here we just take it as given.) A cell that reads −0.55 for "hot CPI surprise" versus "S&P 500" means: when inflation comes in hotter than expected, the S&P tends to fall, and the relationship is moderately strong and reliable.

There is a fourth property of every relationship that the static matrix cannot show, and it is the reason this series exists rather than just a single grid: **lead and lag.** Two variables can be correlated *contemporaneously* (they move on the same day), or one can *lead* the other (it moves first, and the second follows weeks or months later). The yield curve, for instance, tends to lead recessions by over a year; credit spreads lead equity drawdowns by a few months; ISM new orders lead earnings by about half a year. Lead-lag is what turns a correlation from a description into a *signal*: a contemporaneous correlation just tells you two things moved together, but a *leading* correlation tells you what is coming. The matrix cell records the sign and strength; the lead-lag — which we return to throughout the series — tells you the timing. Keep all four properties in mind for every cell: sign, strength, lead-lag, and regime-dependence.

One more foundation, because it is the load-bearing caveat for the entire series: **correlation is not causation, and a full-sample correlation can be an outright lie.** Two assets can be correlated because one drives the other, because a third thing drives both, or by pure coincidence over a short window. The classic absurd example is that the divorce rate in one US state correlates almost perfectly with margarine consumption over a decade — a coincidence with no mechanism, the kind of spurious correlation you find by data-mining enough series. In markets the danger is subtler: real mechanisms produce real correlations, but those mechanisms switch on and off with the regime. And — the subtlest trap — a correlation measured over a long sample can hide two opposite regimes that cancel out. We will see exactly this with gold and real yields: strongly negative for fifteen years, then positive for three, so the full-sample number is near zero and tells you nothing. Hold that thought; it is why the matrix is "the starting point, not the truth." The discipline this series drills is: never accept a correlation without a *mechanism* you can name and a *regime* you can check.

For the deeper statistics of correlation — why it only captures *linear* relationships, how it differs from regression, and how rolling windows reveal regimes — lean on the [math-for-quants treatment of correlation and covariance](/blog/trading/math-for-quants/covariance-matrix-linear-algebra-math-for-quants) and the cross-asset framing in [correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch). This series uses those tools; it does not re-derive them.

## The map: the macro-asset correlation matrix

Here is the whole field on one grid. Rows are the macro drivers; columns are the assets. Each cell is the sign-and-strength of the correlation between a *rise* (or a hot surprise) in that driver and the asset's return. Green is positive, red is negative, and the deeper the color, the stronger the relationship.

![The macro-asset correlation matrix heatmap with values in each cell](/imgs/blogs/the-macro-asset-correlation-matrix-2.png)

Spend a minute with it before reading on. Squint and look at the colors as blocks. The top three rows — 10-year yield rise, real-yield rise, hot CPI surprise — are a wall of red across the risk-asset columns and a wall of red down the bond column, with green only in the last column (the dollar). That is the single most important pattern in all of macro: **rising rates and the inflation that drives them are bad for almost everything, and good for the dollar.** The bottom rows are more mixed — oil is a near-wash, growth surveys are the one mostly-green row, and credit spreads are red across risk assets again.

These are *researched approximations* drawn from public cross-asset studies (FRED data, published bank and asset-manager notes, exchange data), rounded for teaching and representing the documented sign and relative strength rather than a single-sample point estimate. Treat the exact decimals as illustrative; treat the *signs* and the *relative magnitudes* as the durable lesson.

Before we walk the rows one at a time, step back and notice the *shape* of the whole grid, because the shape is itself a lesson. There is a clear block structure. The top three rows (yields, real yields, hot inflation) are essentially the *same* row repeated with slightly different emphasis — they are all expressions of "the cost of money is rising" — and they form a red block across every risk column. That block is the single dominant feature of the matrix, and it is why so many traders compress all of macro down to "what are rates doing?" The bottom rows are more idiosyncratic: oil has its own supply-and-demand story, growth surveys are the lone risk-on row, and credit spreads are a fear gauge that overlaps with but is not identical to the rates block. And one column stands apart from all the others: the dollar column is mostly green (a rise in almost any driver that signals tighter US conditions lifts the dollar), and the dollar's own row is mostly red (a stronger dollar pressures almost everything). The dollar is the matrix's pivot — both an output of the other drivers and an input to every asset.

### Why one map can cover seven assets

It is worth pausing on *why* a single grid is even possible. With seven drivers and seven assets there are forty-nine cells, and you might expect forty-nine independent stories. There are not, for the reason the opening figure showed: the drivers do not reach the assets through forty-nine separate channels — they funnel through a *handful* of them, and the discount-rate channel dominates. Rising yields, rising real yields, and a hot inflation print are three different ways of saying "the discount rate is going up," and the discount rate hits every future cash flow in the economy at once. That shared channel is what *couples* the assets together and what makes their correlations structured rather than random. When you internalize that, the matrix stops being a lookup table you have to memorize and becomes something you can *derive*: ask "does this driver raise or lower the discount rate, tighten or loosen the dollar, raise or lower the fear of default?" and the signs of the row fall out almost automatically.

### The oil row: the one that is genuinely two-sided

The oil row is worth a special note because it is the most ambiguous in the matrix, and the ambiguity is instructive. Higher oil shows roughly +0.10 with the S&P, −0.05 with the Nasdaq, +0.20 with gold, −0.40 with the dollar, and small positives with EM and Bitcoin. Why so muddled? Because oil is both a *growth signal* and a *cost shock*, and which one dominates depends on why oil is rising. If oil is climbing because global demand is strong (a healthy expansion), that is mildly good for risk assets and for the energy-heavy parts of the equity market — hence the small positive with the S&P. But if oil is spiking because of a supply shock (war, an OPEC cut), it acts as a tax on consumers and a push on inflation, which is bad for stocks and good for nothing except energy producers and, indirectly, gold. The cell records the *average* of these two stories, which is why it is close to zero and unreliable. The lesson generalizes: a correlation cell near zero does not always mean "no relationship" — sometimes it means "two strong opposite relationships averaging out," and you have to ask which one is operative right now.

### How these numbers are actually measured

A fair question at this point: where do these cells come from, and how would you compute your own? The honest answer is that there is no single canonical number for any cell, because the value depends on choices you make: the *sample window* (last year? last decade? a specific regime?), the *frequency* (daily, weekly, monthly returns), whether you use *levels or changes* (you almost always want changes/returns, because two trending series will show a spurious correlation just from sharing a trend), and whether you measure *contemporaneous* or *lagged* relationships. Change any of those and the number moves. The cells in this matrix are a synthesis: they take the documented sign and rough magnitude from many published studies and from the underlying return data, and round them to a representative value for a *typical* recent regime. The next post in the series shows the machinery — computing a rolling correlation in a few lines of pandas, choosing a window, and watching the number breathe over time. For now, the takeaway is that a correlation is an *estimate*, not a fact, and a responsible reader always asks "estimated over what window?"

#### Worked example: computing a correlation cell by hand

To demystify the number, compute one cell from scratch. Take a tiny five-point sample of a driver X (the change in the 10-year yield, in basis points) and an asset Y (the S&P's daily return, in percent) on five hypothetical CPI days: X = [+10, −5, +8, −12, +4], Y = [−1.0, +0.6, −0.8, +1.4, −0.5].

- Means: X-bar = (10 − 5 + 8 − 12 + 4) / 5 = +1.0 bp; Y-bar = (−1.0 + 0.6 − 0.8 + 1.4 − 0.5) / 5 = −0.06%.
- Deviations from the mean, multiplied pairwise, and summed gives the covariance numerator: (10−1)(−1.0+0.06) + (−5−1)(0.6+0.06) + (8−1)(−0.8+0.06) + (−12−1)(1.4+0.06) + (4−1)(−0.5+0.06) = (9)(−0.94) + (−6)(0.66) + (7)(−0.74) + (−13)(1.46) + (3)(−0.44) = −8.46 − 3.96 − 5.18 − 18.98 − 1.32 = **−37.9**.
- The denominator is the product of the two spread terms: sqrt(sum of squared X-deviations) x sqrt(sum of squared Y-deviations) = sqrt(81+36+49+169+9) x sqrt(0.88+0.44+0.55+2.13+0.19) = sqrt(344) x sqrt(4.19) = 18.5 x 2.05 = **37.9**.
- r = −37.9 / 37.9 = about **−1.0**.

The intuition: in this hand-built sample, every time yields jumped the S&P fell and vice versa, so the correlation comes out near −1 — and you can see directly that the *sign* comes from the cross-products (negative when the two move oppositely) while the *magnitude* gets normalized by each series' own spread, which is what keeps r between −1 and +1 regardless of the units (basis points versus percent).

### How to read a row versus a column

The matrix has two reading directions, and they answer two different questions. This is the single most useful skill the series teaches, so let us make it concrete.

![Reading a row gives a driver footprint, reading a column gives what moves an asset](/imgs/blogs/the-macro-asset-correlation-matrix-3.png)

**Reading across a row** gives you one driver's *footprint* — everything that driver touches, in one glance. Read the top row of the figure above (10-year yield rises) left to right: S&P −0.45, Bonds −0.95, Gold −0.55. The story the row tells is "when long yields rise, almost everything falls, and bonds fall hardest." That is the question you ask on a day when one variable is moving: *the 10-year just jumped 15 basis points — what does that do to my whole book?* You read the row.

**Reading down a column** gives you everything that *moves one asset*, ranked by strength. Read the Bonds column of the figure top to bottom: 10Y yield −0.95, stronger USD +0.10, ISM/PMI −0.25. The story is "bonds are dominated by yields (almost mechanically), barely care about the dollar, and fall modestly when growth surveys rise." That is the question you ask when you own something and want to know its risk drivers: *I hold long Treasuries — what should I watch?* You read the column.

#### Worked example: reading a hot CPI print off the row

Suppose core CPI comes in at +0.2 percentage points above the consensus expectation — a meaningful upside surprise. To see what happens to your portfolio, you read the **"hot CPI surprise" row** of the matrix and translate the correlations into betas using the event-study magnitudes in our data. The data file's `CPI_SURPRISE_BETA` (calibrated to the 2022-23 inflation-fear regime) gives the move per +0.1pp of upside surprise, so a +0.2pp surprise is roughly double:

- **S&P 500:** −0.70% per 0.1pp x 2 = about **−1.4%**.
- **Nasdaq 100:** −1.00% x 2 = about **−2.0%** (longer-duration, more rate-sensitive, so a bigger hit).
- **US 10-year yield:** +7.0bp x 2 = about **+14bp** (yields rise as the market prices more Fed hikes).
- **US 2-year yield:** +9.0bp x 2 = about **+18bp** (the front end reprices most).
- **Gold:** −0.80% x 2 = about **−1.6%** (real yields rose, and gold hates real yields).
- **Bitcoin:** −1.60% x 2 = about **−3.2%** (the highest-beta risk asset).

The intuition: one number — the inflation surprise — pushed every asset in the direction its matrix cell predicted, and the *size* of each move scaled with how rate-sensitive the asset is. That is the whole game: the row tells you the signs, the betas tell you the magnitudes, and the most rate-sensitive assets (Nasdaq, gold, Bitcoin) move most.

## The dominant row: rates and real yields

If you only learn one row of this matrix, learn this one. Interest rates — and especially their inflation-adjusted cousin, real yields — are the closest thing markets have to a master variable. The reason is the discount rate, the same channel that sat in the middle of our very first figure.

Here is the mechanism in plain terms. The value of any asset is the present value of the cash it will throw off in the future — dividends from a stock, coupons from a bond, rents from a building. To turn future cash into a price today, you *discount* it by an interest rate: a dollar next year is worth less than a dollar today, and the higher the rate, the less that future dollar is worth now. So when the discount rate rises, the present value of *every* future cash flow falls — and that is why rising rates pull down stocks, bonds, gold, and crypto all at once. They are not falling for separate reasons. They are falling for the *same* reason, which is exactly why this row is a wall of red.

The macro-trading series builds this mechanism in full — see [interest rates, the price of money and the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable). What this series adds is the *measured correlation*: not just "rates matter" but "a rise in the 10-year yield carries roughly −0.45 with the S&P, −0.95 with bonds, −0.55 with gold."

Why is the bond cell almost −1? Because for a bond, the relationship is nearly mechanical, not statistical. A bond's price *is* the present value of fixed coupons, so when its yield rises the price falls by definition — that is what duration measures. The correlation between yields and bond prices is the strongest, most reliable cell in the entire matrix, which is why we round it to −0.95 rather than leaving room for doubt. Everything else in markets is fuzzier; bonds versus yields is close to a law of physics.

Now the subtlety that earns real yields their own row. The **nominal** yield (the headline 10-year rate) bundles two things together: a real return and compensation for expected inflation. The relationship is captured by a simple identity — nominal yield is approximately the real yield plus expected inflation (the "breakeven" inflation rate the market is pricing). The **real** yield strips out the inflation piece and shows the true cost of money. For risk assets, the real yield is the cleaner signal, because it isolates how much the discount rate genuinely tightened versus how much it just kept pace with inflation. A nominal yield can rise for a *good* reason (real growth picking up) or a *bad* one (inflation fears spiking); the real yield tells you which, and risk assets care most about the bad kind. The macro-trading series develops this in [real vs nominal: inflation and the real-yield master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal); we will see its cleanest correlation — with gold — in a moment.

This is also why a hot inflation *surprise* gets its own row even though it works through the same rates channel. When CPI comes in hotter than expected, the market does not just mark up inflation — it marks up the *path of the Fed*, pricing in more or faster rate hikes. That pushes both nominal and real yields up, which is why the "hot CPI surprise" row looks so much like the rates rows. The one cell where it differs sharply is gold: the rates rows hit gold hard (−0.55 to −0.80), but the *direct* CPI-surprise cell for gold is only about −0.10, because a hot inflation print is genuinely two-sided for gold — it raises real yields (bad) but also raises the long-run inflation gold supposedly hedges (good), and the two roughly cancel in the moment. That near-zero gold cell in the inflation row, sitting right next to a strongly negative gold cell in the real-yield row, is the single most important clue to debunking the "gold is an inflation hedge" myth — and we will use it.

There is also a lead-lag dimension to the rates block that the static matrix hides but that matters enormously for the cycle. The *shape* of the yield curve — the gap between long and short yields — is one of the most powerful leading indicators in all of macro. When short yields rise above long yields (an "inverted" curve), it has historically led recessions by over a year, with the cross-correlation peaking around fourteen months out. So the rates block is not only the dominant *contemporaneous* row of the matrix; embedded in it is the single best *leading* signal for the whole business cycle. The macro-trading series unpacks this in [reading the yield curve: slope, inversion, recession](/blog/trading/macro-trading/reading-the-yield-curve-slope-inversion-recession).

#### Worked example: a 50bp rise in real yields and a long-duration stock

Imagine a high-growth tech stock whose entire value is in far-future earnings — a "long-duration" equity, the stock-market equivalent of a 30-year bond. Suppose its effective duration is about 25 years (its price is that sensitive to the discount rate). Real yields rise by 0.50 percentage points, from 1.5% to 2.0%.

A rough first-order estimate of the price hit is duration times the yield change: 25 x 0.50% = about **−12.5%**. Compare that to a stable, dividend-heavy utility with an effective duration of, say, 8 years: 8 x 0.50% = about **−4%**. Same macro move, the same real-yield row of the matrix, but the *magnitude* depends entirely on how far in the future the asset's cash flows sit.

The intuition: the real-yield row's sign (negative) is universal across risk assets, but the size of each move is set by duration — which is why a 50bp real-yield jump barely scratches a utility and guts a money-losing growth name.

### Gold and real yields: the cleanest correlation, and the one that broke

Of all the cells in the matrix, gold versus real yields is the one textbooks love, because the logic is airtight and the historical fit was beautiful. Gold pays no interest. So its great competitor is a real, inflation-protected government bond, which *does* pay interest. When the real yield on that bond rises, holding gold gets more expensive in opportunity-cost terms — you are giving up more real return to hold the shiny metal — so money rotates out of gold and the price falls. When real yields fall (or go negative, as in 2020-21), gold's zero yield suddenly looks great, and the price soars.

For roughly 2007 to 2021, this relationship was almost a straight line.

![Gold versus the ten-year real yield scatter with the two regimes split](/imgs/blogs/the-macro-asset-correlation-matrix-6.png)

Look at the blue dots and the down-sloping line: as real yields rose, gold fell, year after year, with a correlation around −0.8 to −0.9 over that window. That is one of the cleanest macro correlations you will ever find, which is why this series gives it a dedicated post: [real yields and the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation).

Now look at the orange points — 2022 through 2025. They sit *above* the line, completely off the old relationship. Real yields rose sharply (from negative to around +2%), and by the textbook gold should have collapsed. Instead it climbed to record highs above \$2,600/oz. The correlation in that window flipped to roughly +0.6. What broke it? A new buyer that does not care about opportunity cost: central banks, especially in emerging markets, bought gold aggressively to diversify away from the dollar after the 2022 reserve freezes. A clean fifteen-year correlation can break in a single regime, and this is the canonical case.

#### Worked example: full-sample correlation versus the regime split

Here is the lie of the full sample, made arithmetic. Using the gold-versus-real-yield data:

- Over **2007-2021**, the correlation between the real yield and the gold price is about **−0.9** (a strong, clean inverse link).
- Over **2022-2025**, the correlation is about **+0.8** (the decoupling — gold and real yields rose together).
- Over the **full 2007-2025 sample**, the two regimes partly cancel and the correlation collapses toward roughly **0** — "no relationship at all."

If you had naively computed one correlation over the whole period, you would have concluded gold and real yields are unrelated — exactly backwards from the truth, which is that they are *very* strongly related, just with the sign depending on the regime. The intuition: never trust a single full-sample correlation; always ask whether it is averaging across two opposite worlds.

## The cross-asset headwind: the dollar

The next row to internalize is the US dollar. The dollar is best understood not as one more asset but as *cross-asset gravity* — a force that pulls on almost everything else, because so much of the world's commodities, debt, and trade is priced and funded in dollars.

The mechanism has two legs. First, the **pricing leg**: oil, copper, gold, and most commodities are quoted in dollars worldwide. When the dollar strengthens, a barrel of oil that costs \$80 becomes more expensive for a buyer holding euros or yen, so demand softens and the dollar price tends to fall — a stronger dollar mechanically pushes dollar-priced commodities down. Second, the **funding leg**: emerging-market governments and companies borrow heavily in dollars. When the dollar rises, their debt gets more expensive to service in local-currency terms, financial conditions tighten, and EM assets sell off. Put both legs together and you get a row that is negative across commodities, EM, and crypto.

![The dollar footprint bar chart showing correlations with each asset](/imgs/blogs/the-macro-asset-correlation-matrix-4.png)

The bar chart makes the footprint vivid. A stronger dollar carries roughly −0.55 with gold and EM equities, −0.45 to −0.5 with oil and copper, −0.35 with Bitcoin, and a milder −0.20 with the S&P 500 (US large-caps earn a lot abroad, so a strong dollar dents their foreign earnings, but they are less exposed than EM). The single green bar is US yields: higher US yields *attract* capital into dollars, so the dollar and US yields move together — that is the one positive cell in the dollar's footprint, and it is the engine behind the whole row.

There is a third, more reflexive leg worth knowing: the dollar is itself the world's premier safe haven, so in a risk-off panic everyone rushes *into* dollars at the same moment they are dumping risk assets. That makes the dollar's row partly a *consequence* of risk-off as much as a *cause* of it — when fear spikes, risk assets fall and the dollar rises together, reinforcing the negative cells. This is why the dollar is the matrix's pivot: it is simultaneously an output of tighter US conditions (higher US yields pull capital in) and an input to every other asset (a stronger dollar pressures them), and in a crisis it is the flight-to-safety magnet that ties the whole risk-off move together. The mechanism is developed in [the dollar system: why the USD rules markets](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy) and [the dollar as cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity); this series measures it in [the dollar (DXY) cross-asset correlation](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation).

#### Worked example: a 5% dollar rally and an EM portfolio

Suppose the dollar index (DXY) rallies 5% over a quarter — a meaningful but not extreme move, like the run into the 2022 high near 114.8. You hold an emerging-market equity basket. EM equities carry roughly a −0.55 correlation with the dollar, but to size the move you need a beta. Empirically, EM equity returns run something like −1.5 to −2x the dollar move during dollar-up regimes. Taking the middle, −1.75x:

- Estimated EM equity move: −1.75 x 5% = about **−8.75%** from the dollar channel alone.
- Gold, with a similar correlation but a smaller beta of roughly −1x: about **−5%**.
- Oil, beta around −1.2x: about **−6%**.

The intuition: the dollar's row is one of the few macro variables that hits commodities, EM, *and* crypto simultaneously, which is why a dollar rally is the classic "everything-risk-off-at-once" trigger and why allocators watch DXY as a single dial for global risk appetite.

## The risk-on row and the risk-off row

Two more rows complete the core of the map: growth surveys (risk-on) and credit spreads (risk-off). They are mirror images.

**Growth surveys — the ISM and PMI business indices — are the one cleanly risk-on row.** These are monthly surveys asking purchasing managers whether business is getting better or worse; a reading above 50 means expansion, below 50 contraction. When they rise, the market reads stronger demand and better earnings ahead, so risk assets rise. The matrix shows ISM/PMI at about +0.55 with the S&P, +0.50 with the Nasdaq, +0.55 with EM equities, +0.40 with Bitcoin — a row of green across risk assets. The two non-green cells are instructive: bonds are slightly negative (strong growth lifts yields, which hurts bonds) and gold is near zero (it has no growth story; it is a rates-and-fear asset). These surveys also *lead* — ISM new orders tend to lead S&P earnings growth by about six months — which makes them prized as an early read on the cycle. See [the business cycle: four phases for traders](/blog/trading/macro-trading/the-business-cycle-four-phases-for-traders) for the regime framing, and the event-trading take in [ISM/PMI: the business surveys that lead](/blog/trading/event-trading/ism-pmi-the-business-surveys-that-lead).

**Credit spreads are the risk-off row, and the market's most honest canary.** A credit spread is the extra yield a risky corporate bond must pay over a safe Treasury — the market's price for the risk that the borrower defaults. When spreads *widen*, the bond market is getting scared, and it usually gets scared before the stock market does. The matrix shows "credit spread wider" at about −0.70 with the S&P (the strongest equity cell of any driver), −0.65 with the Nasdaq and EM, −0.60 with Bitcoin. The reason spreads correlate so tightly with equity drawdowns is that both are pricing the same thing — the probability and severity of a recession — but credit, being a bond market dominated by sober institutions, often moves first. This series gives it a dedicated post: [credit spreads, the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary).

#### Worked example: high-yield spreads as a forward-return signal

There is a beautiful, counterintuitive twist in the credit row. *Wider* spreads correlate negatively with equity prices *right now* (stocks are falling as spreads blow out), but they correlate *positively* with future returns. Using the high-yield-spread-versus-forward-return pairs in our data:

- When the high-yield spread is tight at about **3.0%**, the S&P's subsequent 12-month return averages around **+10%** — decent, but you are buying when everyone is calm and valuations are full.
- When the spread blows out to about **8.0%**, the subsequent 12-month return averages around **+18%**.
- At a crisis-level **10.8%** spread (think the depths of 2008), the forward return averages around **+35%**.

The intuition: wide spreads mean fear is already in the price, and fear that is already priced is the friend of future returns — the credit canary that warns you of the drawdown *as it happens* is the same signal that, at its extreme, marks the best entry points. (Contemporaneous correlation: strongly negative. Forward-looking correlation: strongly positive. Same data, opposite sign, depending on whether you mean "now" or "next year" — a perfect illustration of why lead-lag matters.)

### The most-exposed column: emerging markets and crypto

There is one more reading worth doing before we leave the rows, and it is a *column* reading that ties the whole map together. Look down the EM-equity and Bitcoin columns and notice that they are red for almost *every* driver: rising yields hurt them, rising real yields hurt them, a hot CPI hurts them, a stronger dollar hurts them, wider credit spreads hurt them, and only rising growth surveys help. These two columns are the matrix's "everything hits them" assets — the highest-beta expressions of global risk appetite. The reason is that both are at the *end* of the risk chain: EM borrows in dollars and depends on global growth and capital inflows, so it amplifies every tightening; crypto is the purest high-beta risk asset, the last thing bought in a boom and the first thing sold in a squeeze. The practical consequence is that EM and crypto are where the matrix's correlations show up *largest* in magnitude, which makes them both the most rewarding in a risk-on regime and the most punishing when the rates-and-dollar block turns red. If you want to feel the matrix's full force, watch these two columns — they are the amplifier.

## The cell that famously flips: stocks and bonds

We have walked the rows. Now we return to the cell from the opening story, because it is the matrix's most important lesson about *change*. The relationship between stocks and bonds is the foundation of the 60/40 portfolio and of most institutional asset allocation, and it is not stable.

![Rolling stock-bond correlation flipping from negative to positive in 2022](/imgs/blogs/the-macro-asset-correlation-matrix-5.png)

Trace the line. In the 1990s the stock-bond correlation was *positive* — both were driven by inflation, so they tended to fall together when inflation rose. Then around 1998-2000 it turned firmly *negative* and stayed there for two decades: bonds became the great diversifier, rallying whenever stocks fell because the shared fear was recession (and recessions are good for bonds, which benefit from rate cuts). This negative correlation is the entire engine of the 60/40 portfolio — see [the stock-bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine). Then 2022 happened: inflation became the dominant fear again, the correlation snapped to roughly **+0.60**, and the diversifier stopped diversifying exactly when investors needed it most.

What governs the flip? The dominant risk. The data tells a clean story when you condition on the inflation regime:

- When inflation is **below 2%**, the stock-bond correlation averages about **−0.45** (bonds diversify well).
- At **2-3%** inflation, about **−0.30**.
- At **3-4%**, it crosses zero to about **+0.05**.
- **Above 4%** inflation, it flips to about **+0.50** (bonds and stocks fall together).

The mechanism: when recession is the dominant fear, bad news for stocks is *good* news for bonds (rate cuts coming), so they move oppositely. When inflation is the dominant fear, bad news is the *same* news for both — higher rates ahead — so they move together. The sign of the most important diversification relationship in finance is a function of which fear is in charge. This is the heart of the series' angle, developed in [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant).

## Common misconceptions

The matrix is powerful precisely because it compresses a lot of truth into one grid — but that compression breeds myths. Here are the ones that cost people money.

**Myth 1: "Gold is an inflation hedge."** Half-true at best. Over the long run gold roughly keeps pace with the price level, but on the horizons that matter to a trader, gold does *not* track inflation — it tracks **real yields**. In 2022, inflation hit a 40-year high of about 9% and gold went essentially nowhere, because real yields were rising hard at the same time. The matrix shows it plainly: the "hot CPI surprise" cell for gold is only about −0.10 (near zero), while the "real-yield rise" cell is −0.80. Gold cares about the real cost of money, not the headline inflation number.

**Myth 2: "Bonds always protect you when stocks fall."** Only in the right regime. As the flip chart shows, the stock-bond correlation was around +0.6 in 2022 — bonds fell *harder* than stocks. The protection bonds offer is conditional on recession being the dominant fear, not inflation. A diversifier you cannot count on in the one scenario you fear most is not a reliable hedge; it is a fair-weather friend.

**Myth 3: "Bitcoin is digital gold / an uncorrelated diversifier."** Time-varying, and at the wrong times it correlates most. Bitcoin's correlation with the Nasdaq was near zero before 2020, spiked to roughly **+0.65** in the 2022 liquidity squeeze (it traded as the highest-beta risk asset, not as a safe haven), then faded back toward +0.2-0.3 by 2024-25. The diversification was real in calm times and evaporated in the stress of 2022 — the textbook failure mode of "when correlations go to one in a crisis." See [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).

**Myth 4: "A strong economy is good for stocks, full stop."** Only when growth is the market's focus. In the 2022-23 "good news is bad news" regime, a *strong* jobs report sent stocks *down*, because strong jobs meant more Fed hikes. The same +100k upside surprise in non-farm payrolls carried a beta of about −0.50% for the S&P in 2022-23, versus about +0.35% in a normal expansion. The sign of the growth-to-stocks relationship itself flips with the regime — which is why the matrix is a regime-average, not a law.

**Myth 5: "A full-sample correlation is the truth."** The most dangerous myth of all, and the one we already disproved with gold: a correlation measured over a long window can average two opposite regimes into a meaningless near-zero, hiding a relationship that is actually very strong in each regime separately. Always check whether the number is stable over time before you trust it. This is why the next post in the series is about rolling correlations, not static ones.

## How it shows up in real markets

The matrix is not an abstraction. Here are three dated episodes where reading it (or failing to) was the whole ballgame.

**2022 — the year the matrix flipped.** This is the master case. Inflation ran to a 40-year high (CPI peaked at 9.06% in June 2022), the Fed hiked the funds rate from near zero to 4.5% in nine months, and the 10-year real yield rocketed from about −0.95% in late 2021 to roughly +1.7% by late 2022. Read the rates row of the matrix and you can predict the whole year: stocks down (S&P −18%), bonds down *harder* (long Treasuries −30%), gold flat despite record inflation (real yields up), the dollar up (DXY to a peak of 114.8 in September), EM hammered, crypto crushed (Bitcoin from \$46k to \$16.5k). Every one of those moves is the rates-and-real-yields row playing out, plus the dollar row reinforcing it. The matrix did not fail in 2022 — it *worked*, for anyone who knew which row was driving.

**2023-2024 — gold breaks its leash.** Real yields stayed high (around +1.7% to +2.2%), so by the textbook gold should have languished. Instead it rallied to records above \$2,600/oz by 2025. The "clean" gold-versus-real-yield correlation broke because a price-insensitive buyer — central banks diversifying reserves — entered the market. Anyone reading the matrix as gospel would have shorted gold into a 40%+ rally. The lesson is not that the matrix is wrong; it is that every cell can flip when a *new mechanism* shows up, and you have to keep asking "is the usual driver still in charge?"

**2024-2025 — Bitcoin's correlation fades.** As the acute liquidity stress of 2022 receded, Bitcoin's correlation with the Nasdaq drifted from +0.6 back toward +0.2-0.3. An asset that traded as pure high-beta risk in the squeeze started to trade somewhat more on its own (ETF flows, halving cycle, idiosyncratic crypto news). The cell did not flip sign, but its *strength* changed a lot — a reminder that even when the direction holds, the magnitude is regime-dependent, and a hedge sized for a +0.6 correlation is mis-sized when it drops to +0.2.

What unites all three episodes is a single discipline. In each case, the matrix's *signs* were broadly right for the dominant driver, but a naive reader who treated the cells as fixed constants got hurt in the details — shorting gold into a record rally because "real yields are up," over-hedging crypto because "it correlates +0.6 with stocks," or assuming bonds would cushion an equity drawdown. The traders who navigated 2022-2025 well were not the ones with the most precise correlation estimates; they were the ones who kept asking *which regime is in charge* and *is the usual driver still the operative one*. The matrix gave them the map; the regime question kept them oriented as the terrain shifted. That is the entire skill, compressed: hold the map in your head, but never stop checking the ground.

## The crucial caveat: the matrix is a starting point, not the truth

Everything above builds to this. The matrix you have learned is a **regime-average map**. Every cell is the historical *tendency*, averaged across many market environments — and several of the most important cells flip sign depending on which regime you are in.

![The cells that flip between the textbook regime and an inflation-shock regime](/imgs/blogs/the-macro-asset-correlation-matrix-7.png)

The before-after figure collects the famous flips. In the textbook low-inflation regime: stocks and bonds are negatively correlated (bonds hedge), good jobs prints lift stocks, gold tracks real yields with a clean −0.8, and Bitcoin is roughly its own asset. Flip into an inflation-shock regime and *every one of those reverses*: stocks-bonds goes to +0.6, good jobs sends stocks down, gold's real-yield link breaks, and Bitcoin's correlation with stocks jumps to +0.6. Same assets, same matrix structure, opposite signs — because the market's dominant fear changed from recession to inflation.

There is an even harsher version of this caveat, and it is the most important sentence in risk management: **diversification fails exactly when you need it most.** In a genuine crisis — 2008, March 2020, the worst weeks of 2022 — almost every risk correlation rushes toward +1. The careful diversification you built in calm times, spreading across stocks, EM, credit, and crypto, collapses into a single bet, because in a panic everyone sells everything liquid at once to raise cash and meet margin calls. The matrix's pleasant patchwork of greens and reds does not hold in the tail; it homogenizes into "all risk assets down together." This is not a flaw in the matrix so much as a regime the matrix has to be read *for*: the "crisis regime" is its own column of behavior, where the only correlations that hold are the flight-to-safety ones (cash up, the dollar up, sometimes Treasuries up — though even that failed in 2022). The cross-asset series treats this directly in [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis), and it is the reason a serious allocator never relies on a single diversifying correlation to save them: the one you are counting on is the one most likely to vanish in the storm.

The deeper point is that a correlation is a *behavior of a regime*, not a property of two assets. Gold and real yields are not "negatively correlated" the way water is wet; they were negatively correlated *during a regime* in which the marginal buyer of gold was a yield-sensitive Western investor, and that correlation changed when the marginal buyer became a price-insensitive central bank. Stocks and bonds are not "diversifiers" by nature; they diversify *during a regime* in which recession is the shared fear. Read every cell of the matrix with that grammar — "during the current regime, this driver tends to push this asset this way, with this strength" — and you will never again be blindsided by a flip, because you will have been watching for the regime change all along.

So how should you actually use the matrix, given that it moves? Three rules.

1. **Use it for signs and relative strength, not precise numbers.** The durable lesson is "rising real yields hurt risk assets, and gold most of all" — not "the correlation is exactly −0.80." Trust the direction and the ranking; treat the decimals as approximate.

2. **Always ask which regime you are in first.** Before you apply any row, identify the dominant fear: recession or inflation? Growth or liquidity? The regime decides whether several cells are positive or negative. The single fastest regime check is the inflation rate — above ~3-4% core, assume the inflation-shock signs; below ~2%, assume the textbook signs.

3. **Measure the current correlation, do not assume the historical one.** This is what the rest of the series teaches: compute *rolling* correlations on a window that matches your horizon, watch for the sign changing, and re-size your hedges when it does. A static matrix tells you what *usually* happens; a rolling correlation tells you what is happening *now*.

## How to read it and use it: the playbook

Put it all together into a repeatable process. When any macro variable moves — a data release, a yield spike, a dollar rally — run this loop:

- **Identify the driver (the row).** Which variable just moved? Rates, real yields, inflation surprise, dollar, growth survey, credit spread? Pick the row.
- **Read the row for signs.** Across that row, which assets get the green and which get the red? That is your first-pass map of which way your whole book moves.
- **Check the regime.** What is the dominant fear right now — recession or inflation? If inflation is in charge (core CPI above roughly 3-4%), apply the flipped signs: stock-bond positive, good-news-is-bad for stocks, gold decoupled from inflation.
- **Size with betas, not just signs.** Translate the correlation into a magnitude using the asset's rate-sensitivity (duration for stocks and bonds, historical beta for EM and crypto). The most rate-sensitive assets — long-duration tech, gold, Bitcoin — move most.
- **Verify against the live correlation.** Before you trust a hedge, confirm the rolling correlation still has the sign and strength you are counting on. If the stock-bond correlation has flipped positive, your "diversified" 60/40 is actually concentrated in one bet on rates.
- **Know what invalidates the read.** A cell can break when a new mechanism appears (central-bank gold buying broke the gold cell; a liquidity crisis pushes every correlation toward +1). If a relationship you are relying on stops behaving, stop relying on it and find out what changed.

#### Worked example: re-sizing a 60/40 portfolio when the correlation flips

Concretely, watch what the stock-bond flip does to the risk of a "balanced" portfolio. Take a classic 60% stocks / 40% bonds mix. Suppose stocks have an annual volatility of 16% and bonds 7%. Portfolio variance is the weighted sum of each variance plus a cross term that depends on the correlation:

- **Diversifying regime, correlation −0.4:** variance = (0.6 x 16)^2 + (0.4 x 7)^2 + 2 x (0.6 x 16)(0.4 x 7)(−0.4) = 92.2 + 7.8 + 2(9.6)(2.8)(−0.4) = 92.2 + 7.8 − 21.5 = 78.5, so portfolio volatility = sqrt(78.5) = about **8.9%**. The bonds are actively *cancelling* some of the stock risk.
- **Inflation-shock regime, correlation +0.6:** the cross term flips: 92.2 + 7.8 + 2(9.6)(2.8)(+0.6) = 92.2 + 7.8 + 32.3 = 132.3, so portfolio volatility = sqrt(132.3) = about **11.5%**.

The same portfolio weights, the same asset volatilities, but the realized risk jumped from roughly 8.9% to 11.5% — a 29% increase — purely because one correlation cell flipped sign. The intuition: when the stock-bond correlation turns positive, your "diversified" 60/40 quietly becomes a far riskier, more concentrated bet on rates, and if you have not re-checked the live correlation you will not even know it happened until the drawdown teaches you.

The single most valuable habit this whole series can give you: **never assume a correlation; always check whether the regime that produced it still holds.** The matrix is the map. The regime is the terrain. A map is priceless right up until the terrain changes underneath it — and in markets, the terrain always changes eventually.

This post is the hub. From here, each row and each famous cell gets its own deep dive: rolling correlations and why the window matters in [correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant); the master rates row in [bond yields, the master correlation with every asset](/blog/trading/macro-correlations/bond-yields-the-master-correlation-with-every-asset); the cleanest cell in [real yields and the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation); the cross-asset headwind in [the dollar (DXY) cross-asset correlation](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation); the canary in [credit spreads, the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary); and it all comes together in [the macro correlation playbook](/blog/trading/macro-correlations/the-macro-correlation-playbook-capstone).

## Further reading and cross-links

Within this series (each fills in a row or a cell of the map):

- [Correlation is a regime, not a constant](/blog/trading/macro-correlations/correlation-is-a-regime-not-a-constant) — rolling correlations, regime detection, why the window matters.
- [Bond yields: the master correlation with every asset](/blog/trading/macro-correlations/bond-yields-the-master-correlation-with-every-asset) — the dominant row, in full.
- [Real yields and the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation) — gold, the −0.8, and the break.
- [The dollar (DXY) cross-asset correlation](/blog/trading/macro-correlations/the-dollar-dxy-cross-asset-correlation) — cross-asset gravity, measured.
- [Credit spreads: the risk correlation and the canary](/blog/trading/macro-correlations/credit-spreads-the-risk-correlation-and-the-canary) — the risk-off row and the forward-return twist.
- [The macro correlation playbook (capstone)](/blog/trading/macro-correlations/the-macro-correlation-playbook-capstone) — the whole process, end to end.

The mechanism behind the correlations (why, not just how much):

- [How policy moves every asset: the cross-asset transmission map](/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map) — the causal chain this matrix measures.
- [Interest rates: the price of money, the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable).
- [Real vs nominal: inflation and the real-yield master signal](/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal).
- [The dollar system: why the USD rules markets](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy).

The cross-asset and statistical foundations:

- [Correlation and the diversification free lunch](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch).
- [When correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis).
- [The stock-bond correlation: the 60/40 engine](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine).
- [The dollar as cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity).

The release-day reaction (the intraday version of these correlations):

- [Cross-asset transmission: how one print hits every market](/blog/trading/event-trading/cross-asset-transmission-how-one-print-hits-every-market).
- [CPI: the report that moves the world](/blog/trading/event-trading/cpi-the-report-that-moves-the-world).

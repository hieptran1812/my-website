---
title: "Inflation and Stocks: The Correlation That Flips"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The link between inflation and equities is not a single number but a U-shape: stocks do best with moderate inflation and lose real value in both deflation and high inflation. Here is why the sign flips around 3-4% and how to read the regime you are in."
tags: ["macro", "correlation", "inflation", "cpi", "equities", "stocks", "real-returns", "pe-compression", "regime", "stock-bond-correlation", "discount-rate", "macro-trading"]
category: "trading"
subcategory: "Macro Correlations"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The relationship between inflation and stocks is not a fixed number; it is a **U-shape**. Equities deliver their best real returns in *moderate* inflation (~1-3%) and turn negative in real terms in BOTH deflation and high inflation (> ~4-5%). The correlation between inflation surprises and stock returns *flips sign* around a 3-4% threshold.
>
> - Average real S&P returns by inflation bucket trace a U: about **+10% in the 2-3% sweet spot**, but roughly **−2% in deflation** and **−5% above 5%** inflation. A single straight-line correlation averages two opposite stories into mush.
> - The correlation of equity returns with inflation surprises is **mildly positive (~+0.10) when inflation is low and stable**, but turns **strongly negative (~−0.45) when inflation is rising or high** — the sign literally flips.
> - High inflation is where the stock-bond hedge breaks: the same rising discount rate that compresses equity multiples also crushes bond prices, so the stock-bond correlation went to **+0.6 in 2022** and both fell together.
> - **The one fact to remember:** there is no such thing as "the" inflation-stock correlation. There is a moderate-inflation regime where it is mildly positive and a high-inflation regime where it is sharply negative — and the whole game is knowing which one you are standing in.

In January 2022, an investor with a perfectly sensible playbook would have told you that stocks are an inflation hedge. And he would have had decades of textbooks on his side. Stocks are claims on real businesses — companies that sell real goods, raise prices when costs rise, and own factories and brands whose value floats up with the general price level. Bonds pay back fixed dollars that inflation erodes; stocks own *things*. So when inflation comes, the logic goes, you want to own stocks, not bonds. For most of the post-war era that logic held up well enough to become conventional wisdom.

Then 2022 happened. US headline CPI tore from under 2% to a 40-year high of **9.06% in June 2022**. If stocks were really an inflation hedge, this was their moment to shine. Instead the S&P 500 fell about **18%** on the year, the tech-heavy Nasdaq 100 fell **32.5%**, and — here is the part that broke people's mental models — long-term Treasury bonds, the supposed *opposite* of stocks, fell about **31%** at the same time. The "inflation hedge" and the "safe asset" went down together, hard, in the exact year inflation arrived. The conventional wisdom did not just underperform; it inverted.

Here is what almost nobody tells beginners: both the "stocks hedge inflation" camp and the "stocks got crushed by inflation" camp are right — they are just describing different *regimes*. The relationship between inflation and stock prices is genuinely **non-linear**. It is shaped like a U. A little inflation is good for stocks; a lot of inflation is poison. Somewhere around 3-4% inflation, the sign of the correlation flips from positive to negative. The investor in January 2022 was not wrong about the textbook — he was wrong to assume the textbook's regime would hold as inflation blew through the flip point. Once you can see the U, you can stop being surprised by it.

![The inflation sweet spot for stocks showing deflation and high inflation as two loss tails](/imgs/blogs/inflation-and-stocks-the-correlation-that-flips-1.png)

This is the inflation chapter of a series about a single, deceptively simple question: how does macro data move asset prices? We are not re-deriving *why* the Fed hikes when inflation runs hot — the macro-trading series covers that mechanism, and we will cite it. We are not replaying the ten-minute reaction to a CPI print — the event-trading series covers that, and we will cite it too. This post is about the thing in between: the **measurable statistical relationship** between the inflation rate and the level of stock prices, why that relationship is a curve rather than a line, where it flips, and how to read which side of the flip you are on right now. We build it from zero, so if "real return" and "P/E compression" are unfamiliar phrases, you are exactly the reader this is for.

## Foundations: inflation, real returns, and what a correlation is

Before we can talk about how inflation moves stocks, we need three plain-English ideas locked down: what inflation actually is, the difference between a nominal and a real return, and what a correlation measures. If you have read earlier posts in this series, treat this as a refresher; if this is your first one, this section gives you everything you need.

### Inflation: the price of everything, rising

**Inflation** is the rate at which the general level of prices is rising. If a basket of groceries that cost \$100 last year costs \$104 this year, the inflation rate is 4%. The most-watched measure in the US is the **Consumer Price Index (CPI)**, reported monthly by the Bureau of Labor Statistics; when people say "inflation is 4%," they almost always mean CPI rose 4% over the prior twelve months ("CPI year-over-year"). There are cousins — core CPI strips out volatile food and energy, and the Fed's preferred gauge is core PCE — but the intuition is identical: a single number for how fast your money is losing purchasing power.

A crucial nuance for this whole post: inflation is not one thing with one effect. **Mild, stable inflation** (say 2%) is what a healthy growing economy runs; central banks literally target about 2% on purpose. **Deflation** (negative inflation, falling prices) sounds nice — cheaper stuff! — but it is dangerous, because falling prices make people delay purchases, crush business revenue, and increase the real burden of debt. **High inflation** (5%, 9%, or worse) is corrosive: it scrambles planning, erodes savings, and forces central banks to slam the brakes. The same word, "inflation," describes a friendly companion at 2% and a wrecking ball at 9%. That is the seed of the U-shape.

### Nominal versus real: the only return that matters

Here is the single most important distinction in this entire post. Your **nominal return** is the headline number — the dollars-and-cents change in your investment. Your **real return** is the nominal return *minus inflation* — what your money actually bought you after prices moved.

Suppose your stock portfolio rose 6% in a year. Sounds good. But if inflation was 9% that year, your money buys *less* than it did before: your real return was roughly 6% − 9% = **−3%**. You have more dollars and less stuff. That is the trap of high inflation for stocks: a positive nominal return can hide a negative real return. Throughout this post, when we ask "is inflation good or bad for stocks," we mean **real returns** — because that is the only return that pays your rent. The U-shape figure above is a U-shape *in real returns*; in nominal terms, stocks often look fine even in high inflation, which is precisely how the "stocks hedge inflation" myth survives.

#### Worked example: the real return in a high-inflation year

Take 2022 in round numbers. The S&P 500's *price* fell about 18%; with dividends, the total nominal return was roughly **−18%**. Headline CPI for 2022 averaged about 8% and peaked at 9.06% in June. So the real total return was approximately −18% − 8% = **−26%** — a brutal year that the nominal figure already showed, made worse once you account for the purchasing power lost. Now flip it: in a calm 2% year where stocks return a nominal 10%, the real return is 10% − 2% = **+8%** — almost the entire gain survives. **The intuition:** inflation is a tax that gets subtracted from every nominal return, so the higher inflation runs, the more of your stock gains it quietly confiscates — and at high enough inflation, it turns a positive nominal year into a real loss.

![Nominal versus real stock returns in a high inflation year and a calm year](/imgs/blogs/inflation-and-stocks-the-correlation-that-flips-8.png)

The chart makes the tax visible: in a calm 2% year almost the whole nominal gain survives as real return, while in a high-inflation year the same subtraction can flip a modest nominal gain into a real loss.

### Correlation: do they move together, and how reliably?

A **correlation** measures whether two things tend to move together, and how dependably. It is summarized by the **correlation coefficient**, written *r*, which always sits between −1 and +1. An *r* near **+1** means the two move in lockstep the same direction; near **−1** means they move in mirror-image opposite directions; near **0** means knowing one tells you nothing about the other. (We build this from absolute scratch — including the cousin you actually trade, *beta* — in `what-correlation-actually-measures-pearson-spearman-beta`; here we only need the gist.)

When we say "the inflation-stock correlation," we mean: over some window of time, when inflation surprised higher than expected, did stocks tend to rise (positive *r*), fall (negative *r*), or do nothing (*r* near 0)? The entire thesis of this post is that **this number is not stable**. It is mildly positive in one regime and sharply negative in another, and a full-sample average over both regimes lands near zero — which is worse than useless, because it tells you the relationship is weak when in fact it is *strong in two opposite directions*.

### Why a single linear correlation is a lie here

Most of the relationships in this series are roughly *monotone*: more of the indicator means reliably more (or less) of the asset, at least within a regime. The inflation-stock link is different — it is **non-monotone**. More inflation is good for stocks up to a point, then bad beyond it. The relationship literally changes direction as you move along the inflation axis.

This matters because the standard Pearson correlation coefficient only measures **linear** association — straight-line, one-direction co-movement. Run a single linear regression of stock returns on the inflation rate across all of history and you get a coefficient near zero, because the positive slope on the left half of the U cancels the negative slope on the right half. You would conclude "inflation doesn't matter for stocks." That conclusion is dead wrong. Inflation matters enormously — it just matters in opposite directions on the two sides of the sweet spot. **The U-shape is the single most important picture in this post**, and the linear correlation is precisely the wrong tool to see it with. We cover this class of trap in detail in `spurious-correlation-and-the-traps-of-macro-data`, but the inflation-stock link is its cleanest real-world example.

### Expected inflation versus the surprise

One more distinction will pay off repeatedly, and it is the one beginners skip. Markets do not respond to inflation; they respond to *unexpected* inflation. By the time a CPI number is released, the consensus forecast for it is already baked into prices. If inflation comes in exactly as expected, prices barely move — the news was already known. What moves markets is the **surprise**: the gap between the actual print and what the market had priced. We devote a whole post to this (`the-surprise-not-the-level-betas-to-data-surprises`), but you need the idea here because it explains a subtlety in the U-shape.

The *level* of inflation determines which regime you are in (which side of the U). But within that regime, it is the *surprise* that drives the day-to-day correlation. In the high-inflation regime, an upside inflation surprise is read as "even more hikes than feared," and stocks fall — the correlation of returns with surprises is sharply negative. In the low-inflation regime, a modest upside surprise reads as "the economy is a bit stronger than thought," and stocks shrug or rise — mildly positive. So when we measure "the inflation-stock correlation by regime," we are usually measuring the correlation of stock returns with inflation *surprises*, conditioned on the inflation *level*. Two different roles for inflation: the level sets the regime, the surprise drives the move. Keeping them straight is what separates a careful read from a confused one.

There is a second, market-based way to see expected inflation: the **breakeven inflation rate**, the gap between a nominal Treasury yield and the matching inflation-protected (TIPS) yield. It is the inflation rate at which holding the two bonds breaks even, so it reveals what the bond market *expects* inflation to average. In the 2022 scare, the 10-year breakeven spiked to about **3.02% in April 2022** — the market briefly pricing persistent above-target inflation — before settling back near 2.3% as the Fed's resolve became clear. When breakevens are anchored near the central bank's 2% target, the market trusts the sweet spot will hold; when they unanchor upward, the market is pricing a slide toward the right tail of the U. Watching breakevens is watching the market vote on which regime is coming.

## The mechanism: why moderate inflation helps and high inflation hurts

To trust the U-shape, you need to understand the *machinery* underneath it. A stock's price is, at bottom, the present value of the cash a company will hand its owners in the future. Two things move that price: how much cash the company will generate (its **earnings**), and what multiple the market pays for each dollar of those earnings (its **valuation**, usually quoted as the **P/E ratio**, price divided by earnings). Inflation pushes on both — but in *opposite directions* as it rises, and that tug-of-war is the U-shape.

### The good side: nominal revenue, pricing power, and the low-rate backdrop

In the left and middle of the U — deflation up through moderate inflation — rising prices are mostly *good* for a company's earnings, for three reasons.

**Nominal revenue grows.** A company's sales are measured in dollars. If the general price level rises 3%, a company that simply holds its market share and volumes sees its dollar revenue rise about 3% too — for free, just from prices floating up. Earnings, being a slice of revenue, tend to grow with the nominal economy. Stocks are claims on this nominal cash flow, so a little inflation lifts the numerator (earnings) of every valuation.

**Pricing power lets margins hold or expand.** A business with a strong brand or a tight market can raise its prices *as fast as or faster than* its costs rise. In mild inflation, demand is usually healthy, so firms can pass cost increases through to customers without losing volume. Margins hold. Some firms even widen margins because their costs (say, long-term debt fixed at a low rate, or wages that adjust with a lag) rise slower than their selling prices. This is why investors prize "pricing power" in inflationary periods — it is the difference between inflation lifting your earnings and inflation eating them.

**The discount rate stays low.** Crucially, *moderate* inflation does not force the central bank to tighten aggressively. Rates stay low, which keeps the *denominator* of the valuation — the discount rate applied to future earnings — gentle. Cheap money supports high P/E multiples. So in the sweet spot you get the best of both: earnings rising with nominal growth, and a low discount rate keeping multiples generous. That is why the U bottoms out (its best point) around 2-3% inflation.

### The bad side: the discount-rate channel and margin squeeze

Now push inflation past roughly 4-5% and the same forces invert. This is the part the "stocks hedge inflation" crowd misses, and it operates through three channels.

**P/E compression via the discount rate.** This is the dominant channel, and it deserves its own careful explanation, which is coming in the next section. The short version: high inflation forces the central bank to raise interest rates hard to bring inflation back down. Higher rates mean a higher **discount rate** — the rate at which the market converts future earnings into today's price. A higher discount rate makes future earnings worth less today, which pushes the P/E multiple *down*. Even if earnings hold up, the multiple the market pays for them collapses, and the stock falls. This is **multiple compression**, and it is the engine of equity losses in high inflation.

**Margin squeeze.** In high inflation, costs — wages, materials, energy, shipping — often rise *faster* than a company can lift its selling prices without losing customers. Demand may be softening at the same time (because the central bank is deliberately cooling the economy). So margins get squeezed from both ends: costs up, pricing power down. Earnings, which looked like they would float up with nominal revenue, instead stall or fall. The numerator of the valuation stops helping.

**Real yields rise and Fed tightening drains liquidity.** The cleanest macro variable in markets is the **real yield** — the interest rate after subtracting expected inflation (we devote a whole post to it: `real-yields-and-the-cleanest-macro-correlation`). When the Fed hikes to fight high inflation, it pushes real yields up sharply. A positive, rising real yield is the single biggest headwind for risky assets: it raises the bar every investment must clear, draws money out of stocks and into newly-attractive bonds, and tightens financial conditions across the board. In 2022 the 10-year real yield swung from about **−0.95% at end-2021 to +1.57% by end-2022** — a roughly 250 basis-point swing that, by itself, justified a large chunk of the equity multiple compression that year.

![Why high inflation breaks the stock-bond link through the rising discount rate](/imgs/blogs/inflation-and-stocks-the-correlation-that-flips-4.png)

Put the two sides together and you have the U. On the left, inflation lifts earnings and the discount rate stays tame, so stocks do well. On the right, the discount rate explodes and margins squeeze, so stocks do badly — *despite* nominal revenue still rising. The crossover, where the two forces roughly balance, sits around 3-4% inflation. That is the flip point.

## The P/E compression channel, in detail

Because multiple compression is the heart of why high inflation hurts stocks, let us slow down and make it concrete with arithmetic. This is the most important mechanism in the post.

### A stock is the present value of future cash

Imagine a company that will pay you \$5 per share in dividends next year, and that dividend grows a bit each year forever. What is the share worth today? You cannot just add up all the future dividends — a dollar ten years from now is worth less to you than a dollar today, because you could have invested today's dollar in the meantime. So you **discount** each future dollar back to its present value using a discount rate *r*. A simple, classic formula (the Gordon dividend-discount model) says:

```
Price = Dividend_next_year / (r - g)
```

where *r* is the discount rate (roughly: the risk-free interest rate plus an equity risk premium) and *g* is the growth rate of the dividend. The number that matters for us is *r − g*, the gap between the discount rate and the growth rate. The smaller that gap, the higher the price. The bigger that gap, the lower the price.

Now watch what high inflation does. It forces the central bank to raise rates, which raises *r*. The growth rate *g* of nominal dividends might rise a little with inflation too, but the discount rate rises *more*, because the central bank is deliberately tightening to slow the economy. So *r − g* widens, and the price falls. The valuation multiple — the P/E — is just another way of expressing how generous *r − g* is. When *r − g* widens, the P/E compresses.

#### Worked example: P/E compression from a real-yield rise

Let us make this painfully concrete. Suppose a stock earns \$5 per share. At the start of 2022, with the 10-year real yield around −0.95% and rates pinned near zero, the market was happy to apply an effective discount rate of, say, 6% against a 4% growth assumption: *r − g* = 6% − 4% = 2%. With next-year cash of \$5, the implied value is \$5 ÷ 0.02 = **\$250**, an eye-watering P/E of 50 (\$250 price ÷ \$5 earnings).

Now inflation forces the Fed's hand. The real yield rises about 2.5 percentage points over the year, dragging the equity discount rate up with it to roughly 8.5%, while growth expectations actually fade to 3.5% as the economy cools: *r − g* = 8.5% − 3.5% = 5%. Same \$5 of earnings, but now valued at \$5 ÷ 0.05 = **\$100**. The P/E has compressed from 50 to 20. The stock fell from \$250 to \$100 — a **60% loss** — and earnings never even dropped. Every dollar of the loss came from the *multiple*, not the *earnings*. **The intuition:** when rates jump, the market pays far less for each dollar of future profit, and the most expensive, longest-duration stocks fall the hardest because their value sits furthest out in the discounted future.

That last sentence is the key to the *cross-sectional* flip we will see in 2022: the highest-multiple, longest-duration stocks (think unprofitable tech, the "growth" basket) have the most P/E to lose, so they fall the most. Cheap, cash-flowing, short-duration stocks (energy, value, staples) have less multiple to give back. We will see this dispersion in living color shortly.

### Equity duration: why some stocks are more rate-sensitive than others

The worked example above hinted at the most useful idea in this whole post for picking *which* stocks to own in each regime: **equity duration**. Borrowed from the bond world, duration measures how sensitive an asset's price is to a change in the discount rate. A bond that pays you back soon (short duration) barely moves when rates change; a 30-year bond (long duration) swings violently. Stocks have the same property. A company whose profits arrive *soon* — a mature, cash-gushing energy major or consumer-staples brand — is short-duration: most of its value is near-term cash that a higher discount rate barely touches. A company whose profits arrive *far in the future* — a fast-growing, barely-profitable tech firm whose justification is enormous earnings a decade out — is long-duration: nearly all its value sits in distant cash flows that a higher discount rate slashes.

This is why high inflation does not hit all stocks equally. When the discount rate jumps, the present value of cash arriving in year 10 falls far more than the present value of cash arriving next year. So long-duration growth stocks get hammered while short-duration value and cash-flow stocks hold up. The same logic explains why the *flip point* on the U is not identical for every stock: a cheap, short-duration value stock can keep delivering positive real returns even at 4-5% inflation, while an expensive long-duration growth stock may already be in negative-real territory at 3%. "The inflation-stock correlation" is really a *family* of correlations, ordered by duration, and the high-inflation regime spreads them apart violently.

#### Worked example: the duration spread in present-value terms

Take two companies, each worth \$100 today, but with different cash-flow timing. Company A (short duration) returns most of its value as cash within 3 years; Company B (long duration) returns most of its value as cash around year 10. Now the discount rate rises 3 percentage points. The present value of a dollar 3 years out, discounted at the higher rate, falls by roughly 3% × 3 = **~9%**. The present value of a dollar 10 years out falls by roughly 3% × 10 = **~30%** (duration ≈ years to the cash). So Company A drops to about \$91 and Company B to about \$70 — same rate shock, more than three times the damage to the long-duration name. **The intuition:** a rise in the discount rate is a tax on the future, and the further out a company's profits sit, the bigger the tax — which is the entire reason energy (+65.7%) and tech (−28.2%) ended up 94 points apart in 2022.

### Why bonds fall at the same time — the stock-bond link

Notice something in the P/E worked example: the thing that crushed the stock was the *rising discount rate*, which is the same thing as a rising bond yield. And when bond yields rise, **bond prices fall** (bond prices and yields move inversely — see `bond-yields-the-master-correlation-with-every-asset`). So the identical force — the rate rising to fight inflation — pushes *both* stocks and bonds down at once.

This is the deep reason the inflation-stock story is inseparable from the stock-bond story. In a low-inflation world, stocks and bonds are driven by *different* fears: stocks fear weak growth, bonds fear strong growth and inflation, so they tend to move *opposite* — bonds rally when stocks fall, the engine of the 60/40 portfolio. But in a high-inflation world, *one fear dominates both*: the fear that the central bank will hike rates. That single fear hits stocks (via P/E compression) and bonds (via higher yields) simultaneously, so they fall **together**. The stock-bond correlation flips from negative to positive. We devote whole posts to this — `the-stock-bond-correlation-regime` and `when-correlations-break-the-2022-stock-bond-flip` — but you cannot understand the inflation-stock U-shape without it, because the right tail of the U is *exactly* the regime where the stock-bond hedge fails.

![Stock-bond correlation rising with the inflation regime from negative to positive](/imgs/blogs/inflation-and-stocks-the-correlation-that-flips-6.png)

The chart makes the point with conditional correlations: when average core inflation runs **below 2%**, the stock-bond correlation is about **−0.45** (bonds diversify beautifully); when it runs **above 4%**, the correlation flips to roughly **+0.50** (both fall together). The inflation regime is the variable that controls whether your "diversified" portfolio is actually diversified. In the high-inflation regime, your stocks and bonds are secretly the same bet on rates.

## The empirical U-shape: the centerpiece

Enough mechanism — let us look at what stocks actually delivered. The single most important empirical fact in this post is that long-run real equity returns, sorted by the inflation regime they occurred in, trace out a U.

![Real S&P 500 returns by inflation bucket forming a U-shape](/imgs/blogs/inflation-and-stocks-the-correlation-that-flips-2.png)

Read the bars left to right. In **deflation (below 0%)**, average real S&P returns are negative, about **−2%** — falling prices crush nominal revenue and signal a sick economy. In the **0-2%** bucket, real returns jump to about **+8%**; in the **2-3%** sweet spot they peak around **+10%** — this is the regime stocks love, where nominal revenue grows and rates stay friendly. Then the descent begins: **3-4%** delivers about **+5%**, the **4-5%** bucket is roughly **flat (0%)**, and **above 5%** real returns turn negative again, around **−5%**. The relationship is unmistakably non-monotone. Inflation is good for stocks until it is not, and the turn happens around the 3-4% mark.

This is the empirical fingerprint of everything the mechanism predicted. Moderate inflation = the earnings tailwind dominates = best real returns. High inflation = the discount-rate headwind dominates = negative real returns. Deflation = a different poison (collapsing nominal revenue and debt-deflation) = also negative. Two bad tails, one good middle.

#### Worked example: reading the U-shape correctly

A naive analyst takes these six buckets and fits a single straight line of real return against the inflation rate. The left half slopes up (−2% to +10% as inflation rises from deflation to 3%); the right half slopes down (+10% to −5% as inflation rises from 3% to above 5%). Averaged together, the best-fit line is nearly **flat** — its slope rounds to roughly zero, and the linear correlation coefficient comes out small, maybe +0.1. The analyst concludes: "inflation has essentially no relationship with stock returns." But look at the buckets: the *spread* from the best bucket (+10%) to the worst (−5%) is **15 percentage points** of annual real return. That is one of the largest, most decision-relevant effects in all of macro. **The intuition:** when a relationship is U-shaped, the linear correlation hides a huge effect by cancelling its two opposite halves — you must condition on the regime to see the truth, which is why this series never quotes a correlation without a regime attached.

### The sign flip in the correlation itself

The U-shape in *levels* implies a flip in the *correlation*. If you measure the correlation between monthly equity returns and inflation surprises separately in each regime, you get two opposite signs.

![The inflation-stock correlation flips sign across regimes from positive to negative](/imgs/blogs/inflation-and-stocks-the-correlation-that-flips-3.png)

When inflation is **low and stable (below ~3%)**, the correlation of stock returns with inflation surprises is **mildly positive, about +0.10**. A small upside inflation surprise in a calm world is read as a healthy-growth signal — demand is firm, the economy is humming — and stocks shrug or even tick up. But when inflation is **rising or high (above ~4%)**, the correlation flips to **strongly negative, about −0.45**. Now an upside inflation surprise is terrifying: it means more rate hikes, more discount-rate pressure, more multiple compression, so stocks fall. **Same data release, opposite market reaction, depending on the regime.** This is the statistical signature of the U: the slope (and therefore the correlation) is positive on the left and negative on the right.

This regime-dependence of the *reaction to a surprise* is exactly what the event-trading series calls "good news is bad news" — in 2022, strong data of any kind sent stocks *down* because it meant more Fed tightening. We cover the surprise-reaction mechanism in `the-surprise-not-the-level-betas-to-data-surprises` and the intraday version in the event-trading post `cpi-the-report-that-moves-the-world`. The point here is that the *sign* of the beta to a CPI surprise is itself a regime variable.

### The U is asymmetric, and volatility matters too

Two refinements keep the U-shape honest. First, the U is **asymmetric**: the right tail (high inflation) is deeper and steeper than the left (deflation) in the data — roughly −5% real versus −2%. There is a structural reason. High inflation triggers an *active* policy response: the central bank deliberately hikes rates to break it, and that hike is what crushes equity multiples. Deflation triggers easing (rate cuts, which help stocks), so the equity damage in deflation comes mainly from the collapsing economy rather than from a hostile discount rate. The right tail has *two* things hurting stocks (discount rate up *and* margins squeezed); the left tail has the economy hurting them but the discount rate trying to help. So the right side of the U falls faster. For an equity investor, **high inflation is the more dangerous tail to be wrong about.**

Second, it is not just the *level* of inflation that matters but its **volatility and predictability**. Stocks tolerate steady, predictable 3% inflation far better than they tolerate inflation lurching unpredictably between 2% and 6%, even if the *average* is the same. Predictable inflation can be planned around: companies set prices, sign contracts, and the central bank stays calm. Volatile inflation scrambles planning, widens the equity risk premium (investors demand a bigger discount for the uncertainty), and keeps the central bank twitchy. This is why the *anchoring* of inflation expectations matters so much — a market that believes inflation will return to 2% prices a calm regime even if the current print is high, while a market that has lost faith prices the chaotic right tail. The breakeven inflation rate we discussed earlier is precisely the market's read on whether expectations are still anchored. A 3% print with anchored expectations is a different regime from a 3% print with unanchored ones.

#### Worked example: same average inflation, different equity multiple

Imagine two economies, both averaging 3% inflation over a decade. Economy A runs a steady 3% every year; Economy B alternates 1% and 5%. Same mean. But Economy B forces the central bank to react to each swing, generates a higher *expected volatility* of rates, and so commands a higher equity risk premium — say the market demands an extra 1 percentage point of discount rate for the uncertainty. Using the Gordon model with \$5 of earnings and a baseline *r − g* of 4% (value \$125), adding 1 point of risk premium makes *r − g* = 5%, dropping the value to \$100 — a **20% lower multiple** for the volatile economy despite identical average inflation. **The intuition:** stocks do not just dislike high inflation, they dislike *uncertain* inflation, because uncertainty inflates the risk premium and compresses the multiple — which is why central banks fight to keep expectations anchored, not just to keep the level low.

## Common misconceptions

The inflation-stock relationship is a graveyard of confident, wrong one-liners. Here are the five that cost people the most, each corrected with a number.

**Myth 1: "Stocks are an inflation hedge."** Half-true, in a way that is dangerous. Stocks hedge *moderate* inflation well — their nominal returns float up with nominal revenue, and real returns stay healthy in the 1-3% zone. But they are an *anti*-hedge in high inflation: in the 2022 inflation shock, real total returns were roughly **−26%**. Stocks hedge the inflation you do not need hedging from and fail you in the inflation you do. The honest statement is "stocks hedge mild inflation, not high inflation."

**Myth 2: "There is a stable inflation-stock correlation you can plug into a model."** No. The correlation is **+0.10 in the low-inflation regime and −0.45 in the high-inflation regime**. The full-sample number that mixes them lands near zero and is meaningless. Any model that uses a single static inflation-stock correlation will be confidently wrong in exactly the regime where it matters most.

**Myth 3: "Deflation is good for stocks because cheaper inputs lift margins."** Backwards. Mild disinflation can help, but actual *deflation* (falling prices) is associated with **negative real equity returns of about −2%**, because falling prices crush nominal revenue, increase the real burden of corporate debt, and signal a contracting economy. Japan's lost decades and the 2008-2009 deflation scare are the cautionary tales. Stocks fear deflation almost as much as they fear high inflation — that is why the U has *two* bad tails.

**Myth 4: "Real assets like gold are the inflation hedge, so stocks do not need to be."** Gold's relationship to inflation is itself misunderstood — it tracks **real yields**, not the inflation rate (see `inflation-and-gold-the-real-yield-story`). In 2022, with CPI at 9%, gold returned roughly **−0.3%** — flat — because rising real yields offset the inflation tailwind. Neither stocks nor gold reliably hedge high inflation; the asset that did in 2022 was *energy and commodities* (BCOM up 16.1%) and *cash* (which at least did not fall). Do not assume any single asset is your inflation airbag.

**Myth 5: "If inflation comes down, stocks automatically recover."** It depends on *how* it comes down. If inflation falls because the economy is cooling toward a recession, earnings fall and stocks can keep dropping even as inflation declines. If inflation falls while growth holds up (a "soft landing"), the discount-rate headwind eases and multiples re-expand — that is the bullish path, roughly what 2023-2024 delivered as CPI fell from 6.45% (end-2022) toward 3% and the S&P rallied hard. The level of inflation matters, but so does the *direction of travel and the growth backdrop*. There is no automatic recovery; there is a regime to read.

## How it shows up in real markets

Theory is cheap. Let us walk the two regimes through real, dated episodes with real numbers.

### 2022: the right tail of the U, in full

2022 is the cleanest high-inflation case study in modern markets. Headline CPI ran from 7.48% in January to its **9.06% peak in June**, the highest in four decades. The Fed responded with the fastest hiking cycle in a generation, lifting the funds-rate upper bound from 0.25% in March to 4.50% by December. The 10-year real yield swung from about **−0.95% to +1.57%**. Everything the mechanism predicted then happened.

The headline indices fell, but the *dispersion* tells the real story. Look at the 2022 returns by S&P sector.

![S&P 500 sector returns in 2022 showing energy winning and growth losing](/imgs/blogs/inflation-and-stocks-the-correlation-that-flips-5.png)

**Energy returned +65.7%** — it is short-duration (you get the cash now), it benefits directly from the high commodity prices *driving* the inflation, and it has tons of current cash flow that does not get discounted away. Defensive, cash-rich sectors held up: Utilities +1.6%, Consumer Staples roughly flat at −0.6%, Health Care −2.0%. At the other end, the **long-duration growth** sectors got destroyed: Communication Services **−39.9%**, Consumer Discretionary **−37.0%**, Technology **−28.2%**, Real Estate **−26.2%**. The Nasdaq 100, stuffed with exactly these long-duration names, fell **32.5%**. This is the P/E-compression worked example playing out across the whole market: the stocks with the most multiple to lose lost the most.

#### Worked example: the cross-sectional flip in one number

Compare two slices of the market in 2022. Energy (+65.7%) and Technology (−28.2%) sat in the *same* index, in the *same* economy, in the *same* year. The spread between them was **about 94 percentage points**. What separated them was duration: energy's cash flows are immediate and rise with inflation, so a higher discount rate barely dents them; technology's cash flows sit far in the future, so a higher discount rate slashes their present value. The inflation-stock correlation was not just negative for "the market" in 2022 — it was *violently* negative for long-duration growth and even *positive* for energy. **The intuition:** in a high-inflation, rising-rate regime, "stocks versus inflation" is really "duration versus the discount rate," and the correlation's sign within the market depends on how far out each company's cash flows sit.

And critically: bonds did not save anyone. Long Treasuries fell **31.2%**, IG corporate bonds **−15.4%**, the classic 60/40 portfolio **−16.1%**, its worst year since 1937. The stock-bond correlation that year was about **+0.6**. The right tail of the inflation U is *also* the regime where diversification fails — the same rate-fear hit everything at once. (For the full anatomy of that flip, see `when-correlations-break-the-2022-stock-bond-flip`.)

### The 1970s: the original right tail, in slow motion

2022 was sharp and fast; the 1970s were the long, grinding version of the same lesson, and they are why "stocks hedge inflation" got such a brutal stress test. Through that decade, US inflation repeatedly spiked into double digits — peaking near 13-14% by 1979-1980 — driven by oil shocks, loose policy, and unanchored expectations. The conventional wisdom going in was the same one investors carried into 2022: stocks own real assets, so they will keep up with inflation. They did not.

In nominal terms, the S&P 500 *looked* roughly flat-to-up across the decade, which is exactly how the "hedge" myth survived — the index ended the 1970s not far from where it started in price. But in **real** terms, after subtracting that relentless inflation, the decade was a disaster: real equity returns were deeply negative, and an investor who bought in 1968 did not recover real purchasing power for well over a decade. The P/E multiple on the S&P compressed from the low-to-mid twenties at the start of the era to single digits by the early 1980s — the market's willingness to pay for a dollar of earnings collapsed as the discount rate climbed with inflation and rates. This is the U-shape's right tail stretched over ten years: nominal revenue and earnings rose with inflation, but the multiple compressed faster, and high inflation quietly confiscated the real return.

The 1970s also gave us the word **stagflation** — high inflation *plus* stagnant growth — which is the worst possible spot on the U for stocks: the discount-rate headwind of high inflation without the earnings tailwind of a strong economy. Both ends of the valuation equation work against you. It is no accident that the great bull market only began in 1982, the moment the Fed (under Volcker) had crushed inflation back toward the sweet spot, real yields could fall, and multiples could finally re-expand. The 1970s are the long-form proof that the right tail of the U is real, persistent, and capable of stealing a decade of real returns while the nominal chart looks deceptively calm.

### 2008-2009 and Japan: the left tail of the U

The U has *two* bad tails, and the left one — deflation — gets far less attention because it is rarer in the modern US. But it is just as real. In late 2008, as the financial system seized, the US briefly tipped into outright deflation: headline CPI went *negative* year-over-year in 2009. Falling prices were not a relief; they were a symptom of collapsing demand, and the S&P fell about 57% peak-to-trough in the GFC. Deflation is dangerous for stocks because it crushes nominal revenue (the numerator of every valuation goes *down*, not up), it raises the real burden of corporate debt (a fixed debt is harder to repay when revenue is shrinking), and it signals a contracting economy where earnings fall outright.

Japan is the long-form deflation case study, the mirror image of the 1970s US. After its asset bubble burst around 1990, Japan spent much of the next two decades in mild deflation or near-zero inflation, and the Nikkei spent those decades far below its 1989 peak — it did not reclaim that high until 2024, *35 years later*. Persistent deflation kept nominal revenue stagnant, kept animal spirits suppressed, and kept equity multiples low despite rock-bottom interest rates. The lesson: rates near zero do *not* guarantee high multiples if the deflation regime kills the growth side of the equation. The left tail of the U is shallower than the right in the data (about −2% real versus −5%), but it is more *persistent* — deflationary regimes, once entrenched, are hard to escape, which is exactly why central banks fear deflation enough to target 2% inflation rather than 0%.

### 2013-2019: the sweet spot, quietly working

Now the boring, profitable middle of the U. From 2013 through 2019, US CPI mostly ran between about 1% and 2.5% — squarely in the sweet spot, occasionally a touch below. The Fed kept rates near zero for most of it and hiked only gently and gradually from 2015. Real yields stayed low. And stocks? The S&P 500 delivered one of the great bull runs in history, compounding double-digit nominal returns for most of the decade with healthy real returns on top, because inflation was low enough to barely tax them. This is the +8% to +10% bucket of the U-shape in action: nominal revenue grew with the nominal economy, margins were fine, and the discount rate stayed gentle because inflation never forced the Fed's hand. Nobody talked about the inflation-stock correlation because, in the sweet spot, it is mildly positive and undramatic — exactly as the left side of the U predicts.

There is a telling detail in this era that proves the regime point. In those years, when an inflation print or a jobs report came in *hot*, stocks frequently *rose* — the very opposite of 2022. A slightly-stronger-than-expected economy was unambiguously good news, because inflation was so far below target that no one feared the Fed would tighten in response. The correlation of equity returns with inflation surprises was mildly *positive*, just as the data file's low-regime figure (+0.10) says. The identical CPI surprise that would terrify the market in 2022 was *welcomed* in 2017, because the regime was different. This is the cleanest possible demonstration that the sign of the inflation-stock correlation is not a property of stocks or of inflation — it is a property of the *regime*, and specifically of the central bank's likely reaction. In the sweet spot, more inflation means "healthy growth, no policy threat," so stocks like it. In the right tail, more inflation means "more hikes coming," so stocks hate it. Same variable, opposite sign, set entirely by where you sit on the U.

### 2023-2024: walking back down the right side of the U

The most instructive recent episode is the *descent* from the right tail. After peaking at 9.06% in mid-2022, CPI fell steadily: 6.45% at end-2022, ~3% by mid-2023, hovering near 3% through 2024. As inflation walked back toward the sweet spot — and crucially, as it did so *without* a recession (the "soft landing") — the discount-rate headwind eased, real yields stabilized, and the P/E multiple re-expanded. The S&P 500 rallied roughly 24% in 2023 and again in 2024. This is Myth 5 corrected in real time: inflation coming down was bullish *because growth held up*, letting multiples recover. The same variable (inflation) that crushed stocks on the way up to 9% helped them on the way back down to 3% — the U-shape works in both directions, and the regime you are walking *toward* matters as much as the one you are in.

#### Worked example: estimating the multiple-recovery tailwind

A back-of-envelope on 2023. Suppose, as inflation fell from ~6.5% toward 3%, the market's implied equity discount rate eased by about 1 percentage point (real yields stopped rising and growth fears faded), narrowing *r − g* from, say, 5% back toward 4%. Using the Gordon model with \$5 of earnings: at *r − g* = 5% the value is \$5 ÷ 0.05 = \$100; at *r − g* = 4% it is \$5 ÷ 0.04 = \$125 — a **+25% gain purely from the multiple re-expanding**, with earnings flat. That is roughly the magnitude of the 2023 S&P rally, and most of it was multiple expansion, not earnings growth. **The intuition:** just as the discount rate did all the damage on the way up the inflation tail, it did most of the healing on the way back down — which is why watching the *direction* of inflation and rates often matters more for equity returns than the level of earnings.

## How to read it and use it

You now understand the U. Here is how to turn it into a decision process rather than a surprise. This is the practitioner payoff.

### Step 1: Identify which regime you are in

The whole game is locating yourself on the U. Ask three questions:

- **What is the inflation *level*?** Below 2%, in the 1-3% sweet spot, or above 4%? Use core inflation (core CPI or, better, the Fed's core PCE) to strip out noisy food and energy, and check the *trend*, not just the latest print. Above roughly 4% and rising, assume the right-tail regime: the inflation-stock correlation is negative, and rate-fear is the dominant driver.
- **What is the *direction*?** Inflation rising toward the flip point is dangerous for stocks even before it crosses it, because the market front-runs the Fed. Inflation falling toward the sweet spot (with growth intact) is the bullish multiple-recovery setup.
- **What are *real yields* doing?** The real yield is the cleanest single read on the discount-rate headwind (`real-yields-and-the-cleanest-macro-correlation`). Rising real yields = the right-tail regime is live, regardless of what the inflation print says this month. In 2022, the real-yield surge *was* the equity bear market.

### Step 2: Set the position to the regime

The inflation regime tells you not just whether to own stocks but *which* stocks, and whether your bonds are still a hedge.

![The inflation regime playbook for equities showing winners and the correlation sign by regime](/imgs/blogs/inflation-and-stocks-the-correlation-that-flips-7.png)

In the **moderate regime (1-3%)**: own broad equities, lean into growth and quality, and trust the 60/40 — bonds still diversify because the stock-bond correlation is negative here. This is the sweet spot; do not overthink it.

In the **high regime (> 4-5%)**: the market rotates *inside* equities toward short-duration, cash-rich, inflation-beneficiary sectors — energy, value, commodities, materials — and away from long-duration growth and rate-sensitive sectors (tech, discretionary, real estate). Expect the stock-bond correlation to be positive, so stop relying on bonds as your equity hedge; the diversifiers in this regime are commodities, energy equities, cash, and (sometimes) the dollar. Treasury Inflation-Protected Securities (TIPS) and floating-rate instruments also earn their keep here.

In the **deflation regime (< 0%)**: the playbook flips again — long-duration, high-quality assets and government bonds tend to win, because the discount rate is falling and the fear is contraction, not overheating. This is the classic flight-to-quality regime where the stock-bond correlation is reliably negative.

### Step 3: Know what invalidates the read

Every signal needs a kill-switch. The inflation-regime read is wrong if:

- **The Fed's reaction function changes.** The right tail of the U is really about the *central-bank response* to inflation. If the Fed signals it will tolerate above-target inflation rather than crush it (a credible "we are done hiking" pivot), the discount-rate headwind can fade even with inflation still elevated — and stocks can rally before inflation normalizes. The 2023 rally began partly on the *expectation* the Fed was near done, not on inflation hitting target.
- **Inflation falls for the wrong reason.** If inflation drops because the economy is collapsing into recession, earnings fall and stocks can keep declining even as you walk back down the right side of the U. The U-shape is about real returns *conditional on a functioning economy*; a hard landing is a different chart (an earnings-driven bear market, covered in `the-business-cycle-correlation-clock`).
- **You measured the correlation over the wrong window.** A full-sample correlation, or one estimated over a window that straddles the flip point, will be near zero and useless. Always condition on the regime and use a sensible rolling window (`rolling-correlation-and-why-the-window-matters`).

### A real-time monitoring checklist

You do not need a model to run this; you need a short dashboard you check whenever inflation is in play. Here is the practitioner's version, in order of importance:

- **Core inflation level and trend.** Core CPI and core PCE, three-month and twelve-month. Is the level below 3% (sweet-spot side) or above 4% (right-tail side)? Is it rising or falling? This places you on the U.
- **The 10-year real yield.** The single cleanest read on the discount-rate headwind. Rising real yields = the right-tail regime is live and equity multiples are under pressure, *regardless* of this month's CPI print. This is the variable that actually moved stocks in 2022.
- **Breakeven inflation (expectations anchoring).** Is the 10-year breakeven near the 2% target (anchored, calm regime) or drifting up toward 3% (unanchoring, right-tail risk)? Unanchored expectations are the early warning that the regime is about to shift.
- **The Fed's reaction function.** Is the central bank hiking, on hold, or signaling cuts? The right tail of the U is fundamentally about the policy response; a credible pivot can ease the discount-rate headwind before inflation normalizes.
- **Sector dispersion inside equities.** When energy and value are crushing growth and tech, the market is *already* pricing the right-tail regime — the dispersion is a confirming signal that rate-fear is the dominant driver.
- **The stock-bond correlation itself.** A rolling stock-bond correlation turning positive is the market telling you the inflation regime has flipped and your bonds have stopped hedging your stocks.

Run down that list and you can answer the only question that matters: which side of the U am I on, and is the regime stable or about to flip? Everything else — which sectors, whether to trust the 60/40, whether to hold TIPS — falls out of the answer.

### Sizing to conviction about the regime

One last practical point: the U-shape should change not just *what* you own but *how much*. In the moderate regime, the inflation-stock correlation is mildly positive and stable, so inflation is a minor input and you can run a normal equity weight. As inflation pushes toward and through the 3-4% flip point, the correlation goes strongly negative and the *dispersion* of outcomes widens — this is when inflation becomes the dominant risk to your portfolio and deserves the most attention and the most hedging. The mistake is to treat inflation as equally important in every regime. It is nearly irrelevant in the sweet spot and the single most important variable in the right tail. Size your attention, your hedges, and your sector tilts to the regime, not to a constant.

### The one-paragraph version

If you remember nothing else: the inflation-stock relationship is a U, not a line. Stocks earn their best real returns in moderate inflation (~1-3%) and lose real value in both deflation and high inflation. The correlation between inflation surprises and stock returns is mildly positive in the low regime (~+0.10) and sharply negative in the high regime (~−0.45) — it *flips* around 3-4%. The right tail is also where bonds stop hedging and the stock-bond correlation goes positive, because one fear — the rising discount rate — hits everything at once. Locate yourself on the U, set the position to the regime, and never quote a single inflation-stock correlation without saying which regime it came from.

## Further reading and cross-links

Within this series:
- `cpi-and-asset-prices-the-master-inflation-correlation` — the master post on how CPI moves every asset class, the parent of this deep dive into the equity slice.
- `the-stock-bond-correlation-regime` — the full treatment of why stocks and bonds flip between diversifying and co-moving.
- `when-correlations-break-the-2022-stock-bond-flip` — the anatomy of the 2022 regime, the right tail of the inflation U in action.
- `inflation-and-gold-the-real-yield-story` — why gold tracks real yields, not the inflation rate, and what actually hedges inflation.
- `real-yields-and-the-cleanest-macro-correlation` — the single variable that prices the discount-rate headwind behind the U.
- `the-surprise-not-the-level-betas-to-data-surprises` — why the *surprise* (and its regime-dependent sign) moves markets, not the level.
- `spurious-correlation-and-the-traps-of-macro-data` — why a single linear correlation lies about a U-shaped relationship.

Mechanism and reaction (other series):
- `/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine` — the allocator's view of the stock-bond relationship that the inflation regime controls.
- `/blog/trading/macro-trading/how-policy-moves-every-asset-cross-asset-transmission-map` — how Fed policy reaches stocks through discount rates and sectors.
- `/blog/trading/macro-trading/real-vs-nominal-inflation-real-yields-master-signal` — the nominal-versus-real distinction and the real-yield master signal, built from the ground up.
- `/blog/trading/event-trading/cpi-the-report-that-moves-the-world` — the intraday reaction to a CPI print, and how "good news is bad news" emerges in the high-inflation regime.

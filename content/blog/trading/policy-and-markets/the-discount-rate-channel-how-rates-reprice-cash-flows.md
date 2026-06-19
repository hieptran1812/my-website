---
title: "The Discount-Rate Channel: How Rates Reprice Cash Flows"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How the policy rate flows into the discount rate and reprices every future cash flow — the present-value math, equity duration, and why higher rates hammer growth stocks hardest."
tags: ["monetary-policy", "discount-rate", "present-value", "equity-duration", "valuation", "interest-rates", "growth-vs-value", "central-banks", "asset-valuation", "fed-model", "dcf"]
category: "trading"
subcategory: "Policy & Markets"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A central bank sets one number, the policy rate, and that number flows into the *discount rate* every investor uses to convert future cash flows into a price today. Raise the discount rate and every future dollar is worth less now — so when the Fed went from 0.25% to 5.50% in 2022, the assets whose cash flows sit farthest in the future fell the hardest.
>
> - The value of any asset is the present value of its future cash flows, and the discount rate is the dial that converts "later" into "now". Higher rate, lower present value — mechanically, with no change in the cash flows themselves.
> - **Equity duration** explains the cross-section: a long-duration growth stock, whose profits arrive 10-30 years out, gets crushed by a rate rise; a short-duration value stock, paid mostly in the next few years, barely flinches.
> - In 2022 the Nasdaq fell about 33% and the most speculative growth fell 60%+, while value held up — one channel, a brutal demonstration. The number to remember: a single cash flow due in 25 years loses about **21%** of its value when the discount rate rises just one percentage point.

In the first eight months of 2022, the Federal Reserve did something it had not done in forty years: it raised its policy rate from a floor of 0.25% toward a peak of 5.50%, the fastest tightening cycle since Paul Volcker. Nothing about the companies in the stock market changed on those announcement days. Apple still sold iPhones; a money-losing software startup still had the same product roadmap; a pipeline still pumped the same oil. And yet the prices investors were willing to pay for those future profits moved violently — and they did not move uniformly.

The Nasdaq Composite, stuffed with long-dated growth companies, fell roughly 33% on the year. The most speculative corner — pre-profit tech, the kind of company whose payoff is supposed to arrive a decade from now — fell 60%, 70%, in some cases more. Meanwhile the cheap, cash-generating "value" stocks (banks, energy, consumer staples) held up far better; some sectors were nearly flat. Same economy, same earnings season, wildly different fates. The thing that sorted the winners from the losers was not the business. It was *when* each business pays you — its sensitivity to the one number the Fed was moving.

That sorting mechanism is the **discount-rate channel**, and it is the single most important way monetary policy reaches asset prices. This post builds it from the ground up: what a discount rate is, why a dollar later is worth less than a dollar now, how that one rate reprices an entire stream of cash flows, and why "equity duration" makes growth stocks the most rate-sensitive instruments in the market. By the end you will be able to look at a rate move and estimate, with arithmetic you can do on a napkin, roughly how much a given asset should reprice — and why.

![Pipeline showing the policy rate flowing through the risk-free rate, the discount rate, present value, the equity multiple, and the asset price](/imgs/blogs/the-discount-rate-channel-how-rates-reprice-cash-flows-1.png)

The chain above is the whole argument in one line. A central bank pulls a lever (the policy rate). That lever sets the *risk-free rate* — the yield on a government bond. The risk-free rate is the base of the *discount rate* every investor applies to risky cash flows. The discount rate determines the *present value* of those cash flows. And present value, divided by earnings, is the *equity multiple* (the P/E) the market is willing to pay. Move the first box and the last box moves — that is the entire transmission mechanism. Let us walk it backwards, from what an asset is worth to the policy rate that sets it.

## Foundations: present value and the discount rate

Start with the most basic question in finance: what is anything worth? A stock, a bond, a rental apartment, a toll bridge — strip away the story and each is just a **claim on a stream of future cash flows**. A bond pays coupons and then your principal back. A stock pays dividends, or buys back shares, or reinvests in growth that becomes dividends later. An apartment pays rent. The "fair value" of any of these is the answer to one question: *how much would I pay today, in a lump sum, to receive that future stream?*

The reason you would not pay the full face amount is captured in an idea everyone understands from everyday money: **a dollar today is worth more than a dollar a year from now.** Three reasons. First, you could take the dollar today, put it in a safe government bond, and have more than a dollar next year — so a future dollar must be discounted to compete with that. Second, inflation erodes a future dollar's buying power. Third, a promise of a future dollar carries risk: the company might not pay, the tenant might not renew. To convert a future dollar into its worth today, you shrink it by a **discount rate** — the annual percentage by which you mark down each year of waiting.

Think of the discount rate as the mirror image of an interest rate. If a bank pays you 5% a year, then \$100 today grows to \$105 next year and \$110.25 the year after, because the second year earns interest on the first year's interest — that is *compounding*, and it makes money grow faster the longer you wait. Discounting runs that machine in reverse. If money grows at 5% a year, then \$105 next year is "only" worth \$100 today, and \$110.25 two years out is also "only" worth \$100 today. The question "what is a future cash flow worth now?" is exactly "how much would I have to put in the bank today, at the going rate, to end up with that amount on that date?" The going rate *is* the discount rate. A higher rate means your money would have grown faster, so you need less of it today to reach the same future sum — which is precisely why a higher discount rate makes every future cash flow worth *less* right now.

The compounding works against you fastest at long horizons, and that asymmetry is the heart of everything that follows. At a 5% rate, one year of waiting costs you about 5% of the value; but twenty-five years of waiting compounds that 5% haircut twenty-five times over, leaving you with only about thirty cents on the dollar. Distance in time is not penalized linearly — it is penalized *exponentially*, because the discount factor `1/(1+r)^t` has the waiting-time `t` in the exponent. This single fact — that the haircut compounds — is why far-future cash flows are so much more rate-sensitive than near ones, and therefore why long-duration assets get repriced so violently when the rate moves.

The mechanics are one formula, the **present value (PV)** of a single future cash flow:

```
PV = CashFlow / (1 + r)^t
```

where `r` is the discount rate (per year) and `t` is how many years out the cash flow arrives. The term `1 / (1 + r)^t` is called the **discount factor** — the price today of one dollar delivered in year `t`. At a 5% discount rate, a dollar arriving in one year is worth `1 / 1.05 = 0.952` today; in five years, `1 / 1.05^5 = 0.784`; in twenty-five years, `1 / 1.05^25 = 0.295` — under thirty cents. The farther out the dollar, the smaller its discount factor, because you compound the haircut every year.

The value of a whole asset is just the sum of the present values of every cash flow it will ever pay — a process called **discounted cash flow (DCF)** analysis:

```
Value = CF1/(1+r)^1 + CF2/(1+r)^2 + CF3/(1+r)^3 + ...
```

That is the entire engine. Everything else in this post is what happens when you change one input — `r`, the discount rate — and watch `Value` move.

One more piece of vocabulary makes the rest effortless. A stock that pays cash flows *forever* — a perpetual stream — has a famous closed-form value: if it pays a constant `C` every year, discounted at `r`, it is worth `C / r`. A toll bridge that nets \$1 million a year forever, discounted at 5%, is worth `$1,000,000 / 0.05 = $20 million`. Raise the discount rate to 6% and it is worth `$1,000,000 / 0.06 = $16.7 million` — a 17% drop from a one-point rate rise, with the bridge unchanged. Allow the cash flow to *grow* at rate `g`, and the value becomes `C / (r − g)`, the Gordon growth model we will lean on throughout. These two formulas — the perpetuity and the growing perpetuity — are the discount-rate channel in its most compact form, and they show the key fact immediately: value is *divided* by the discount rate (or by `r − g`), so a rising rate shrinks value, fast.

![Bar chart of a five-year stream of one hundred dollar cash flows with their shrinking present values discounted at five percent](/imgs/blogs/the-discount-rate-channel-how-rates-reprice-cash-flows-2.png)

The figure makes the discounting concrete. Imagine an asset that pays you \$100 at the end of each of the next five years (green bars, all equal). Discounted at 5%, the present value of each \$100 shrinks the farther out it sits (blue bars): \$95 in year 1, \$91 in year 2, \$86 in year 3, \$82 in year 4, and \$78 in year 5. The \$100 promised five years out is worth only 78 cents on the dollar today. Add up the five blue bars and you get the asset's fair value: about \$433. Notice the shape — the near cash flows keep most of their value; the far ones lose the most. Hold that picture; it is the key to equity duration.

#### Worked example: the present value of \$100, ten years out, at 4% versus 5%

Take a single \$100 cash flow due in ten years and price it at two discount rates one percentage point apart.

- At `r = 4%`: `PV = 100 / 1.04^10 = 100 / 1.4802 = `**`$67.56`**.
- At `r = 5%`: `PV = 100 / 1.05^10 = 100 / 1.6289 = `**`$61.39`**.

The cash flow did not change — it is still exactly \$100 in ten years. But raising the discount rate by one point cut its value today from \$67.56 to \$61.39, a drop of about **9%**. *One point of discount rate erased nine percent of a ten-year cash flow's value, with the business untouched — that is the discount-rate channel in a single number.*

### Where the discount rate comes from: risk-free plus a risk premium

The discount rate is not a number an analyst invents. It is built from two pieces:

```
discount rate (r) = risk-free rate + risk premium
```

The **risk-free rate** is what you earn for taking essentially no default risk — the yield on a government bond, usually the 10-year Treasury for long-lived assets like stocks. It is the "you could just buy a bond instead" baseline. The **risk premium** is the extra return you demand for bearing the uncertainty of a risky asset: the chance the company stumbles, the volatility you have to stomach, the illiquidity. For the stock market as a whole this is the **equity risk premium (ERP)**, historically around 4-5%.

This decomposition is why monetary policy has such direct leverage. The central bank sets the policy rate, which anchors the front end of the bond market; expectations of the future policy path set the 10-year Treasury yield; and that Treasury yield is the *risk-free rate* sitting inside every investor's discount rate. The Fed does not need to touch a single stock. It moves the risk-free base, and the discount rate on every risky cash flow in the economy moves with it. (For how the policy rate propagates out the curve into that 10-year yield, see the companion post [how policy sets the bond market](/blog/trading/policy-and-markets/how-policy-sets-the-bond-market-the-yield-curve); for the statistical strength of the link, [the Fed funds path and front-end correlation](/blog/trading/macro-correlations/the-fed-funds-path-and-front-end-correlation).)

It helps to understand *why* the risk premium exists, because it tells you which piece the Fed controls and which it does not. The equity risk premium is the market's collective answer to a simple bargain: "stocks are riskier than Treasuries, so how much extra annual return do I demand to hold them instead?" Historically that answer has averaged around 4-5%, but it is not constant — it widens when investors are frightened (they demand more compensation, so they pay less for the same cash flows) and narrows when they are complacent (they accept thinner compensation, so they pay more). The Fed controls the *risk-free* leg directly and the *risk-premium* leg only indirectly, through confidence and liquidity. Most of the time the channel works through the risk-free leg — the Fed hikes, the Treasury yield rises, the discount rate rises, multiples compress. But in the dangerous episodes (a credibility crisis, a liquidity freeze) the risk premium blows out at the same time, and the two legs reinforce each other into a brutal repricing. We will see both kinds in the case studies.

There is a second reason the 10-year Treasury, specifically, is the right risk-free rate for stocks: **stocks are perpetual.** A bond matures and returns your principal; a healthy company never "matures" — it keeps generating cash flow indefinitely. To value an infinitely-long stream you need a long-term discount rate, and the 10-year (sometimes blended toward the 30-year) is the market's best single proxy for "the cost of money over the long run." This is why a move at the *long end* of the yield curve matters far more for equity valuations than a move at the very front end — a point we return to when we discuss the term structure of the discount rate.

![Step chart of the Fed funds upper-bound target rate from 2015 to 2025 rising from 0.25 percent to 5.50 percent and falling to 3.75 percent](/imgs/blogs/the-discount-rate-channel-how-rates-reprice-cash-flows-4.png)

Here is the lever itself. The Fed funds target (upper bound) sat at a 0.25% floor through the pandemic, then stepped up eleven times in sixteen months to a 5.25-5.50% peak by July 2023 — 525 basis points, the steepest climb since Volcker. It held there for over a year before three cuts in late 2025 brought it to 3.50-3.75%. Each step on this staircase pushed up the risk-free base of every discount rate in dollar markets. The next chart shows where that base actually lives for stock valuations.

![Line chart of the 10-year Treasury nominal yield and the 10-year real yield from 2020 to 2026, showing real yields moving from negative to positive](/imgs/blogs/the-discount-rate-channel-how-rates-reprice-cash-flows-5.png)

The 10-year Treasury yield (blue) is the risk-free rate most long-duration valuations key off. But the cleaner driver is the **real yield** (amber) — the yield after subtracting expected inflation, which the market reads off inflation-protected Treasuries (TIPS). The real yield is what actually matters for valuing real future profits, and look at the swing: it was about **−1%** in 2021 (you were paying the government to hold your money in real terms) and rose to about **+2%** by late 2023. A three-point swing in the real discount rate is enormous. When the real yield is deeply negative, almost nothing competes with a growth stock's distant payoff; when it turns sharply positive, those distant payoffs suddenly have a much sterner hurdle to clear. This is why [real yields and the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation) is the single most reliable cross-asset relationship a macro trader watches.

## How one rate reprices an entire stream: equity duration

We have the engine (present value) and the input the Fed moves (the discount rate). Now the crucial step: *why does the same rate move hit different stocks so differently?* The answer is a concept borrowed from the bond market — **duration**.

In bonds, duration measures how much a bond's price moves when its yield changes. A long-maturity bond (a 30-year Treasury) is far more sensitive to a rate move than a short-maturity bond (a 2-year note), because its cash flows are pushed further into the future, where the discount factor does the most work. The rule of thumb: a bond's percentage price change is approximately `−(duration) × (change in yield)`. A bond with a duration of 25 years loses about 25% × 1% = 25% of its value when its yield rises one percentage point. That is the whole reason long bonds got annihilated in 2022 — the Bloomberg Aggregate bond index fell about 13%, its worst year in modern history, purely because its duration ran into a rate spike.

**Equity duration** applies the exact same idea to stocks. A stock's "duration" is the weighted-average number of years until its cash flows arrive, where each year is weighted by how much of the present value it contributes. The longer that horizon, the more its value sits in distant, heavily-discounted cash flows — and the more violently a discount-rate change reprices it. It is the same physics as a seesaw: the farther a weight sits from the pivot, the more leverage it has. A stock's cash flows are weights along the time axis; the ones sitting far out give the price the most leverage to a change in the discount rate. The trick is that stocks differ enormously in *when* they pay you:

- A **value stock** — a bank, a utility, a tobacco company — pays most of its cash flow soon: a fat dividend now, steady earnings, modest growth. Its cash flows are front-loaded, so its effective duration is short, maybe 8-12 years. It behaves like a short bond.
- A **growth stock** — a young software or biotech company — pays you almost nothing now and promises explosive profits a decade or two out. Its cash flows are back-loaded, so its effective duration is long, often 20-30+ years. It behaves like a very long bond.

This is the deepest point in the post: *a growth stock is a long-duration instrument, and the discount rate is the lever that reprices long-duration instruments the most.* When the Fed raises rates, it is not "bad for tech" in some vague sentiment sense — it is mechanically, mathematically, most damaging to the assets whose value lives farthest in the future.

![Before and after comparison of a long-duration growth stock and a short-duration value stock repriced by a one point rise in the discount rate](/imgs/blogs/the-discount-rate-channel-how-rates-reprice-cash-flows-3.png)

The figure lays the two side by side. The growth stock (left) carries its cash flows 10-30 years out, an effective duration around 25 years; bump its discount rate from 4% to 5% and its fair value falls about 20%. The value stock (right) is paid mostly in the next few years, duration around 8; the same one-point rate rise costs it only about 7%. Same policy move, same percentage change in the discount rate — a 3x difference in the damage, entirely because of *when* each one pays. Let us put real numbers on that claim.

#### Worked example: a 25-year cash flow repriced by a +1pp discount rate (~20% hit)

The cleanest possible "long-duration asset" is a single cash flow far in the future — say \$100 due in 25 years, the purest 25-year-duration object. Price it at 4% and at 5%:

- At `r = 4%`: `PV = 100 / 1.04^25 = 100 / 2.666 = `**`$37.51`**.
- At `r = 5%`: `PV = 100 / 1.05^25 = 100 / 3.386 = `**`$29.53`**.

That is a drop from \$37.51 to \$29.53 — a **−21.3%** repricing from a single percentage point on the discount rate. The cash flow is identical; only the rate moved. *A one-point rate rise takes more than a fifth off the value of a 25-year cash flow — this is the arithmetic that turned the 2022 rate shock into a 30%+ haircut for long-duration tech.*

Compare that to the same +1pp move on a cash flow due in just 2 years: `100/1.04^2 = $92.46` versus `100/1.05^2 = $90.70`, a drop of only **−1.9%**. The near cash flow barely notices the rate; the far one is devastated. That gap — 1.9% versus 21.3% — *is* equity duration.

#### Worked example: growth versus value, side by side (the cross-section)

Model both stocks with the **Gordon growth model**, the cleanest one-line DCF: a stock that pays a dividend `D` growing forever at rate `g`, discounted at `r`, is worth `P = D / (r − g)`. The `(r − g)` denominator is what makes growth stocks long-duration: a high `g` makes the denominator tiny, which both inflates the price *and* makes it hypersensitive to changes in `r`.

Take a cost of equity (discount rate) `r = 8%`, and a next-year dividend `D = $2` for both. The growth stock grows at `g = 4%`; the value stock at `g = 0%`.

- **Growth stock**, before: `P = 2 / (0.08 − 0.04) = 2 / 0.04 = $50.00`. Now raise `r` by one point to 9%: `P = 2 / (0.09 − 0.04) = 2 / 0.05 = $40.00`. That is a **−20.0%** repricing.
- **Value stock**, before: `P = 2 / (0.08 − 0.00) = 2 / 0.08 = $25.00`. Raise `r` to 9%: `P = 2 / (0.09 − 0.00) = 2 / 0.09 = $22.22`. That is only a **−11.1%** repricing.

The same +1pp discount-rate move costs the growth stock 20% and the value stock 11% — nearly double the damage, purely because the growth stock's cash flows are pushed further out by its higher growth rate. *The higher a company's growth, the longer its duration, and the more a rate rise reprices it — high growth is a double-edged sword the moment rates move against you.*

![Line chart of fair P/E multiple versus discount rate for a growth stock and a value stock, showing the growth multiple falling faster](/imgs/blogs/the-discount-rate-channel-how-rates-reprice-cash-flows-8.png)

The relationship is a curve, and the curve is the whole story of multiple compression. Each line is the fair P/E multiple (`payout / (r − g)`) as a function of the discount rate. The growth stock (red, `g = 6%`) starts at a high multiple but its curve is steep — as the discount rate rises, its fair P/E collapses fast, because its small `(r − g)` denominator is exquisitely sensitive. The value stock (green, `g = 2%`) sits at a lower multiple but its curve is gentle — its fair P/E barely moves. The dashed lines mark a +1pp discount-rate move: trace it up to each curve and you see the growth multiple fall far more than the value multiple. *This single picture is why "rates up, growth down" is not a slogan but a law of arithmetic.*

### The math of multiple compression

We keep using the word **multiple compression** — let us define it precisely, because it is the form the discount-rate channel takes in everyday market commentary. A stock's price is often written as `Price = Earnings × Multiple`, where the multiple is the price-to-earnings (P/E) ratio. The multiple is not arbitrary; it *is* the discounted-cash-flow value expressed per dollar of earnings. The Gordon model says the justified multiple is `Multiple = payout / (r − g)`. Rearrange the present-value engine any way you like and the same fact falls out: **the fair multiple is inversely related to the discount rate.** Rate up, multiple down.

So a stock's price can fall for two completely different reasons. Either earnings disappointed (the `E` fell) — a business problem — or the discount rate rose and the *multiple* compressed (the `M` fell) — a policy problem, with the business unchanged. The discount-rate channel is the second kind, and in 2022 it dominated: S&P 500 earnings actually *grew* modestly that year, yet the index fell about 19%, because the forward P/E compressed from roughly 21x to 17x as the 10-year yield climbed from 1.5% to 4.3%. Almost the entire decline was multiple compression — a discount-rate event, not an earnings event.

There is a reason the multiple is *so* sensitive for high-growth firms, and it lives in that `(r − g)` denominator. For a value stock with `g = 0%` and `r = 8%`, the denominator is 0.08; a one-point rate rise to 9% changes the denominator to 0.09, a 12.5% increase, so the price falls about 11%. For a growth stock with `g = 6%` and `r = 8%`, the denominator is only 0.02; the same one-point rate rise changes it to 0.03 — a *50%* increase in the denominator, so the price halves toward a 33% fall. The closer the growth rate creeps to the discount rate, the tinier the denominator, and the more explosively a small change in `r` swings the price. This is why the market's very longest-duration darlings can fall 50-80% on a rate cycle that only moved the discount rate two or three points: their `(r − g)` was so small to begin with that even a modest rise in `r` was a huge proportional change in the thing dividing their cash flows. High growth buys you a high multiple and a hair-trigger sensitivity to rates in the same breath.

#### Worked example: multiple compression from the rate move alone

Suppose a stock earns \$5 per share and trades at a 20x multiple, so its price is `$5 × 20 = $100`. Earnings hold flat all year. But the discount rate rises enough that the market's justified multiple falls from 20x to 16x. New price: `$5 × 16 = $80`.

The stock fell 20% — from \$100 to \$80 — and **not one cent of it came from the business.** Earnings were identical at \$5. The entire move was the multiple compressing because the discount rate rose. *When you read that a stock "fell on no news," this is usually the news: the discount rate moved, and the multiple did the rest.*

### The duration spectrum: how the channel sorts the whole market

The single most useful mental model the discount-rate channel gives you is a *spectrum* of the entire market, ranked by equity duration. At one end sit the longest-duration assets, whose value lives almost entirely in distant cash flows; at the other end sit short-duration assets, paid mostly now. A rate move runs along this spectrum like a wave, hitting the long end hardest and the short end least. Roughly, from longest to shortest:

- **Pre-profit growth (longest, duration 30+ years).** A money-losing software, biotech, or clean-energy company whose payoff is supposed to arrive in the 2030s or 2040s. Almost 100% of its value is in distant cash flows. These are the most rate-sensitive instruments in the equity market — more rate-sensitive than a 30-year Treasury bond, because at least the bond's coupons start arriving now.
- **High-multiple profitable growth (long, duration ~20-25 years).** A profitable, fast-compounding business trading at 30-50x earnings. It earns money now, but the market's price reflects expectations of much bigger profits a decade out, so its effective duration is long.
- **The market average (medium, duration ~15 years).** The S&P 500 as a whole, a blend of growth and value.
- **Quality value and dividend payers (short, duration ~8-12 years).** Banks, industrials, consumer staples — steady earnings, fat dividends, modest growth. Most of the value arrives soon.
- **Cash and short bonds (shortest, duration near 0).** A Treasury bill is paid back in months; its value barely moves when rates change. This is why "cash is king" in a rising-rate world — it is the only zero-duration asset.

The practical payoff: when you see the Fed about to move, you do not ask "is this good or bad for stocks?" You ask "where on the duration spectrum am I, and which way is the discount rate going?" A rate rise is a transfer of relative performance from the long end to the short end of this spectrum, every single time.

#### Worked example: the same rate shock across three durations

Take a single \$100 cash flow and place it at three different horizons — 2 years (a value-like near payoff), 10 years (a market-average payoff), and 30 years (a pre-profit growth payoff). Reprice each from a 4% to a 5% discount rate:

- **2-year (short duration):** `100/1.04^2 = $92.46` falls to `100/1.05^2 = $90.70`, a **−1.9%** hit.
- **10-year (medium duration):** `100/1.04^10 = $67.56` falls to `100/1.05^10 = $61.39`, a **−9.1%** hit.
- **30-year (long duration):** `100/1.04^30 = $30.83` falls to `100/1.05^30 = $23.14`, a **−24.9%** hit.

One identical policy move — a single point on the discount rate — costs the near cash flow 2%, the medium one 9%, and the far one 25%. *The discount-rate channel is not one number applied evenly; it is a gradient that punishes patience, and the longer your cash flows wait, the more a rate rise costs you.*

### The term structure: it is the long rate that reprices stocks

One subtlety trips up even seasoned observers: *which* rate is the discount rate? The Fed moves the overnight policy rate, but the discount rate that reprices a 25-year cash flow is a long-term rate. These can move very differently. In a tightening cycle the front end (driven directly by the Fed) can rise far while the long end rises less — or even falls, if the market believes the hikes will slow the economy and force cuts later. The yield curve can *invert*, with short rates above long rates.

For the discount-rate channel, what matters is the long end, because that is the rate appropriate to a perpetual cash-flow stream. This is why the channel sometimes appears "muted" even as the Fed hikes aggressively: if the hikes are pushing up the front of the curve but the 10-year is anchored (because the market expects a slowdown), the discount rate on long-duration equities may not rise much, and growth stocks may hold up better than the headline rate move suggests. Conversely, when the *long* end breaks higher — as it did in late 2022 and again in the 2025 tariff scare, when the 10-year spiked — that is when long-duration equities take the real damage. Watch the 10-year and the 10-year *real* yield, not the Fed funds rate, when you are reading the channel's pressure on stocks. The policy rate is the lever; the long real yield is the actual force on the valuation.

## Common misconceptions

**"Rates only matter for bonds; stocks are about earnings."** Stocks are about earnings *and* the rate at which those earnings are discounted. In a year like 2022, the discount rate did almost all the work: the S&P fell about 19% while earnings rose, because the multiple compressed. Ignoring the discount rate means missing the dominant driver of equity returns in any year the Fed is moving.

**"Higher rates are bad for all stocks equally."** No — the cross-section is the whole point. The hit scales with *equity duration*. In 2022 the long-duration Nasdaq fell about 33% and unprofitable tech fell 60%+, while short-duration value sectors (energy was actually up sharply on the separate oil shock) held far better. A rate move is a sorting machine, not a blanket.

**"A growth stock falls because investors lost faith in the story."** Sometimes. But more often in a rate shock the story is intact and the *discount rate on the story* changed. The 2021-22 de-rating of profitable, fast-growing software companies happened with their revenue still compounding 30-40% — the businesses delivered; the discount rate repriced them anyway.

**"If the Fed cuts, stocks must go up."** Only if the cut lowers the discount rate *more than* it signals falling earnings. Cuts that arrive because the economy is collapsing (2008, early 2020) can coincide with falling stocks, because the earnings cut (`E` down) outruns the multiple expansion (rate down). The channel is real, but it competes with the earnings channel — never read one in isolation. (For how traders actually position around the cut decision itself, see [how monetary policy moves stocks](/blog/trading/macro-trading/how-monetary-policy-moves-stocks-discount-rates-sectors).)

**"The discount rate is the nominal rate."** The cleaner driver of long-duration valuations is the *real* rate. Inflation lifts both the discount rate and (eventually) nominal cash flows, so they partly cancel. It is the real yield — the rate above expected inflation — that does the heavy lifting on growth-stock multiples, which is why the 2021-23 swing in real yields from −1% to +2% was so destructive even though some of the nominal move was just inflation.

**"Duration is something only bond traders need to think about."** Equity duration is invisible on a balance sheet — no company reports it — which is exactly why it is so often missed. But it is the property that decides how your portfolio behaves when the Fed moves, and it cuts across the usual sector and style labels. Two stocks in the same sector can have very different durations (a mature, dividend-paying chipmaker versus a pre-revenue chip startup); two stocks in different sectors can have similar durations. The label that predicts rate sensitivity is not "tech" or "value" — it is *when the cash flows arrive*. Train yourself to look through the sector to the cash-flow timing, and the channel stops surprising you.

## The Fed model: when does the discount rate make stocks expensive?

There is a simple, powerful way to see the discount-rate channel as a competition between two assets: the stock market versus the risk-free bond. Flip the P/E upside down and you get the **earnings yield** — earnings divided by price, the return the stock "yields" you per dollar invested. A stock at a 20x P/E has an earnings yield of `1/20 = 5%`. The **Fed model** (a piece of market folklore, not gospel, but a useful lens) compares that earnings yield to the 10-year Treasury yield. When the earnings yield comfortably exceeds the bond yield, stocks pay you more than the risk-free alternative, so investors accept a high multiple. When the bond yield rises to meet — or exceed — the earnings yield, the safe bond starts to out-compete stocks, and the multiple has to fall (the earnings yield has to rise) to stay attractive.

![Matrix comparing the S&P earnings yield and the 10-year bond yield across 2021, 2022, and 2025, showing the gap closing as rates rose](/imgs/blogs/the-discount-rate-channel-how-rates-reprice-cash-flows-7.png)

The matrix tracks the contest across three regimes. In **2021**, the S&P earnings yield was about 4.5% (a ~22x multiple) while the 10-year sat at 1.5% — a 3.0-point cushion in stocks' favor, so a rich multiple was easy to justify. Through **2022**, the bond yield rose to 4.3% while the earnings yield climbed only because the multiple compressed to ~17x — the cushion vanished, and the multiple *had* to fall. By **2025**, with the 10-year near 4.1% and the multiple back up near 22x (a 4.5% earnings yield), the cushion was a thin 0.4 points — stocks were priced for strong growth with very little margin against the bond. The Fed model does not time markets, but it makes the channel visible: as the risk-free yield rises into the earnings yield, the equity multiple is under pressure.

#### Worked example: the earnings-yield-versus-bond-yield comparison

Suppose the S&P 500 trades at a 20x forward P/E. Its earnings yield is `1 / 20 = 5.0%`. The 10-year Treasury yields 1.5%. The "yield gap" in stocks' favor is `5.0% − 1.5% = 3.5 points` — stocks pay 3.5 points more than the safe bond, a comfortable cushion.

Now the Fed hikes and the 10-year rises to 4.5%, with earnings unchanged. If the market wants to keep that same 3.5-point cushion, the earnings yield must rise to `4.5% + 3.5% = 8.0%` — which means the multiple must compress from 20x to `1 / 0.08 = 12.5x`. At a flat \$5 of earnings, the "fair" price falls from `$5 × 20 = $100` to `$5 × 12.5 = $62.50`, a **−37.5%** repricing.

In practice the cushion itself shrinks (investors accept a thinner premium when they are optimistic), so the real move is gentler — but the direction is iron-clad. *When the risk-free yield rises and the equity cushion holds, the multiple must compress; the only question is how much of the move the market absorbs by accepting a thinner premium.*

## Case studies: the channel in the real world

### 2020-21: zero rates and the everything-multiple-expansion

When the pandemic hit in March 2020, the Fed cut to zero and launched unlimited quantitative easing. The 10-year Treasury yield fell to 0.5%, and the *real* 10-year yield went to about −1%. Plug a near-zero, even negative, real discount rate into the present-value engine and the result is mechanical: the present value of distant cash flows *explodes*, because you are barely discounting them at all. Recall the discount factor for a 25-year cash flow: at a 5% rate it is 0.30; at a 1% rate it is 0.78. Cut the rate toward zero and a dollar in 2045 is suddenly worth almost a dollar today.

This is exactly what happened. The S&P 500's forward P/E expanded from the mid-teens to about 22x. The longest-duration assets ran the most: unprofitable growth stocks, "story" stocks promising profits in the 2030s, and the most speculative corners of the market (meme stocks, SPACs, crypto) all soared, because a near-zero discount rate is the kindest possible environment for assets that pay you far in the future. The companion post [the liquidity channel](/blog/trading/policy-and-markets/the-liquidity-channel-qe-qt-and-the-everything-bid) covers the *flow* side of that boom (QE pushing money into risk); the discount-rate channel is the *valuation* side — the same policy that flooded markets with liquidity also drove the discount rate to its lowest level in history, and the two together produced the everything-rally. Nothing in this period was magic; it was a negative real discount rate doing precisely what the arithmetic says it must.

It is worth dwelling on *why* the most speculative assets ran the most, because it is the duration spectrum at work. A profitable mega-cap that pays a dividend is a medium-duration asset; a near-zero discount rate helps it, but only so much, because a good chunk of its value arrives soon and was never discounted heavily. A pre-revenue startup promising profits in 2035 is almost pure duration — *all* of its value is in distant cash flows, every one of which gets multiplied by a discount factor that just leapt from 0.30 toward 0.80. When the rate falls toward zero, the assets whose value lives entirely in the far future get the largest percentage uplift, mechanically. The 2021 mania in unprofitable tech, story stocks, and crypto was not (only) crowd psychology — it was the rational consequence of pricing the longest-duration assets in existence at the lowest discount rate in history. The psychology amplified it; the arithmetic permitted it. And the same arithmetic guaranteed that whenever the rate reversed, those exact assets would fall the hardest — which is precisely what 2022 delivered.

### 2022: the great growth de-rating

Then it reversed, hard. As inflation hit a 40-year high of 9.1%, the Fed hiked 525 basis points in sixteen months — the staircase in the rate chart above. The 10-year yield tripled from about 1.5% to 4.3%; the real 10-year yield swung from −1% to about +1.7%, a move of nearly three points in the real discount rate. The present-value engine ran in reverse, and equity duration sorted the carnage with brutal precision:

- The long-duration **Nasdaq fell about 33%**, while the broad S&P 500 fell about 19% and the short-duration Dow fell only about 9%.
- The most speculative, longest-duration corners fell far more: unprofitable tech baskets dropped 60-70%; the most speculative names lost 80-90%.
- Critically, this was a **multiple-compression** event, not an earnings event. S&P 500 earnings actually grew modestly in 2022. The index fell almost entirely because the forward multiple compressed from ~21x to ~17x as the discount rate rose. The businesses were fine; the discount rate repriced them.
- And the cruelest detail: **stocks and bonds fell together.** The classic 60/40 portfolio fell about 16% in 2022, its worst year in a century, because a single shared input — the discount rate — drove both. When the rate is the thing moving, diversification across stocks and bonds fails, because both are present-value machines keyed off the same rate. (See [how policy prices equities](/blog/trading/policy-and-markets/how-policy-prices-equities-the-multiple-and-the-earnings) for the multiple-versus-earnings split in detail.)

2022 is the cleanest natural experiment the channel has ever produced. Same companies, same year, and the only variable that explained the cross-section of returns was *duration* — how far in the future each company's cash flows sat.

There is a deeper lesson hiding in the 60/40 wreck that is worth making explicit, because it overturns a piece of conventional wisdom. For forty years, the standard portfolio held 60% stocks and 40% bonds on the theory that bonds zig when stocks zag — when a recession hits earnings, the Fed cuts rates, bonds rally, and the bond gains cushion the equity loss. That relationship held because for forty years the *dominant* shock was a growth shock. But the discount rate is a shared input to both assets: a stock is a present-value machine and a long bond is a present-value machine, and both are discounted by the same long rate. When the dominant shock is a *rate* shock rather than a growth shock — when the thing moving is the discount rate itself — stocks and bonds fall *together*, because the one input they share is the one that moved. 2022 was a discount-rate shock, so the diversification evaporated exactly when it was needed. The discount-rate channel does not just explain the cross-section of stocks; it explains when stock-bond diversification works and when it fails.

### 2025: tariffs, "higher for longer," and the discount-rate link

The most recent demonstration is subtler, and it shows the channel operating through a *fiscal* lever. On April 2, 2025 — "Liberation Day" — the administration announced a 10% universal tariff plus steep "reciprocal" rates on dozens of partners (46% on Vietnam, 34% on China, 20% on the EU). The average effective US tariff rate jumped from about 2.4% to roughly 12-15%, the highest since the 1930s. Tariffs are a tax on imports, and the market's immediate worry was a clean macro chain: **tariffs raise the price of imported goods → inflation reaccelerates → the Fed has to stay "higher for longer" → the discount rate stays elevated → long-duration assets stay pressured.**

![Line chart of the S&P 500 in 2025 falling from a February high of 6,144 to an April trough of 4,983 then recovering](/imgs/blogs/the-discount-rate-channel-how-rates-reprice-cash-flows-6.png)

The repricing was fast. The S&P 500 fell from an all-time high of 6,144 on February 19 to a trough of 4,983 by April 8 — a **−18.9%** drawdown, with a brutal four-day stretch around the announcement. Two things in this episode are worth pinning down because they show the channel's machinery under stress.

First, the Fed's reaction function was the transmission. Through the first half of 2025 the Fed *held* rates — it explicitly waited to see whether tariffs would show up as higher inflation (which argues for higher-for-longer) or weaker jobs (which argues for cuts). That "higher for longer" stance kept the discount rate elevated precisely when growth assets were most vulnerable. Only once the inflation pass-through looked contained did the Fed cut three times late in the year, easing the discount rate back down — and stocks recovered to new highs, closing the year at 6,680.

Second, the bond market behaved *strangely*, in a way that reveals the discount rate is not only set by the Fed. Normally a stock-market crash sends investors fleeing into Treasuries, pushing yields *down*. Instead, during the worst of the April selloff, the **10-year yield rose about 50 basis points in a week** (from 3.99% to 4.49%), and the dollar fell more than 10% over the half-year — its worst first-half since 1973. That is a *policy-credibility* tell: foreign holders questioning US assets pushed the risk-free rate and risk premium *up* at the same time, the opposite of the usual safe-haven reflex. When the discount rate rises for credibility reasons rather than growth reasons, it is the most dangerous version of the channel — the lever moves against you with no offsetting earnings benefit. (For the policy-mechanics of that episode see [tariffs and trade policy as a market force](/blog/trading/policy-and-markets/tariffs-and-trade-policy-as-a-market-force) and the macro positioning view in [how monetary policy moves stocks](/blog/trading/macro-trading/how-monetary-policy-moves-stocks-discount-rates-sectors).)

The episode also shows the channel reaching beyond US borders. Vietnam faced one of the steepest "reciprocal" rates announced on April 2 — 46%, the fourth-highest of any partner — before negotiating it down to 20% in a July 2 deal (with a 40% rate on goods deemed to be transshipped through the country). Because roughly 29% of Vietnam's exports go to the US, that headline rate was an enormous shock to the cash flows of every export-exposed Vietnamese company, and the discount rate the market applied to them jumped with the uncertainty. The mechanism is identical to the US story, just with a different lever and a different country: a government tax (a tariff) raised the perceived risk on a stream of future cash flows, the discount rate on those cash flows rose, and the prices fell — until the deal cut the rate and dialed the discount rate back down. The same arithmetic governs a Hanoi-listed exporter and a Silicon Valley software company; only the cash flows and the country differ.

The 2025 episode closes the loop the whole series is built on: a *government* lever (a tariff) reached the stock market not by changing any company's products but by changing the *discount rate path* the market expected — through inflation, through the Fed's reaction function, and through the credibility of US assets. Different lever, same channel.

### The long view: 1981 and the discount rate that ran a generation

It is easy to treat the discount-rate channel as a thing that happens over months. Step back, and it has set the tide for entire generations of asset returns. When Paul Volcker took the Fed funds rate to a peak of about 19% in 1981 to break double-digit inflation, the 10-year Treasury yield hit nearly 16% — the highest long-term discount rate in modern US history. A 16% discount rate makes distant cash flows nearly worthless; it is the most hostile possible environment for long-duration assets, and stock multiples spent the early 1980s deeply compressed (the S&P traded near 8x earnings, a third of today's multiple).

Then the discount rate began a forty-year decline, from that ~16% peak in 1981 to about 0.5% in 2020. That single, multi-decade fall in the discount rate was the great tailwind behind the long bull markets in stocks, bonds, real estate, and eventually the longest-duration assets of all — the pre-profit growth and tech that defined the 2010s. The same arithmetic that crushed valuations at a 16% discount rate inflated them at a 1% discount rate. Most investors who built careers between 1981 and 2021 were, whether they knew it or not, riding a forty-year decline in the discount rate. The 2022 reversal was so shocking partly because it was the first time in a generation the channel ran the *other* way for a sustained stretch. *The discount rate is not just a quarterly market driver; over decades it is the single most powerful force shaping what assets are worth — and a multi-year change in its direction is the most important thing a long-term investor can recognize.*

## What it means for asset values: the playbook

Pull the threads together into something you can use.

**The direction is iron-clad; the magnitude scales with duration.** When the discount rate rises, every present-value asset falls, and it falls in proportion to how far its cash flows sit in the future. Rank your holdings by duration — money-losing growth and 30-year bonds at the long end, cash-generating value and short bonds at the short end — and you have ranked them by rate sensitivity. A rough field estimate: for a +1pp move in the long-term real discount rate, expect a long-duration growth stock (duration ~25) to reprice on the order of −20%, a market-average stock (duration ~15) on the order of −12%, and a short-duration value stock (duration ~8) on the order of −7%. You can do this estimate in your head with the bond rule of thumb — `percentage move ≈ −(duration) × (rate change)` — and it will put you within a few points of what actually happens, which is far more than most market commentary manages.

**The channel reaches everything, not just stocks.** Every present-value asset answers to the same dial. A rental property is valued at a *cap rate* (its yield), which is a discount rate by another name — higher rates push cap rates up and property values down. Gold pays no cash flow at all, so it is valued against the *opportunity cost* of holding it, which is the real yield — when real rates fall, gold's lack of yield stops being a disadvantage and it rallies (a big part of gold's record run as real-rate expectations shifted). A startup's valuation, a bond's price, a toll road, a farmland lease — all are present-value machines, all keyed to the discount rate. Once you see the channel, you see it everywhere.

**Watch the real yield, not the headline.** The 10-year *real* yield (the TIPS yield) is the single cleanest signal for the discount-rate channel. When it is falling — especially when it is negative — the channel is a tailwind for long-duration assets; when it is rising, it is a headwind, strongest at the long-duration end. A trader watching one number for the channel should watch this one.

**Separate the multiple from the earnings.** When a stock or index moves, decompose it: did earnings change (a business event) or did the multiple compress/expand (a discount-rate event)? In a year the Fed is moving, the multiple usually dominates — and a multiple-compression selloff in a business that is still executing is the channel handing you a repricing, not a deterioration.

**The Fed model gives you a cushion gauge.** Compare the market's earnings yield to the 10-year yield. A wide gap (earnings yield well above the bond yield) means stocks have a fat cushion and can absorb higher rates; a thin gap means the multiple is stretched against the risk-free alternative and is vulnerable to even a small further rate rise.

**What would invalidate the read.** The channel competes with the earnings channel. If rates rise *because* the economy is booming and earnings are accelerating faster than the discount rate, stocks can rise even as rates rise (the discount-rate headwind is overpowered by the earnings tailwind). And if a rate cut arrives *because* a recession is hitting earnings, stocks can fall even as the discount rate eases. Never read the discount-rate channel in isolation — read it against what is happening to the cash flows in the numerator.

The discount rate is the most powerful single dial in finance because it sits inside the value of *everything*: every stock, every bond, every building, every business. A central bank that moves the policy rate is reaching through the risk-free rate, into the discount rate, and out to the present value of every future dollar the economy will ever produce. That is why "rates up, stocks down" is not sentiment, and not a correlation that might break — it is the present-value engine running in reverse, and the assets that pay you farthest in the future are always the ones that feel it most.

## Further reading and cross-links

- [The liquidity channel: QE, QT, and the everything bid](/blog/trading/policy-and-markets/the-liquidity-channel-qe-qt-and-the-everything-bid) — the flow side of the same 2020-21 boom; how balance-sheet policy pushes money into risk assets.
- [How policy prices equities: the multiple and the earnings](/blog/trading/policy-and-markets/how-policy-prices-equities-the-multiple-and-the-earnings) — decomposing an equity move into its multiple and earnings components.
- [How policy sets the bond market: the yield curve](/blog/trading/policy-and-markets/how-policy-sets-the-bond-market-the-yield-curve) — how the policy rate propagates out the curve into the 10-year risk-free rate.
- [Tariffs and trade policy as a market force](/blog/trading/policy-and-markets/tariffs-and-trade-policy-as-a-market-force) — the 2025 tariff lever and its inflation pass-through.
- [How monetary policy moves stocks: discount rates and sectors](/blog/trading/macro-trading/how-monetary-policy-moves-stocks-discount-rates-sectors) — the trader's positioning playbook around the rate decision.
- [Real yields and the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation) — the statistical strength of the real-yield-to-growth-stock link.
- [The Fed funds path and front-end correlation](/blog/trading/macro-correlations/the-fed-funds-path-and-front-end-correlation) — how the policy path maps to the front end of the curve.

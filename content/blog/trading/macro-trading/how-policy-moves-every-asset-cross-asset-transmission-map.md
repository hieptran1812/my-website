---
title: "How Policy Moves Every Asset: The Cross-Asset Transmission Map"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "The capstone of the asset-class cluster: a single policy force — the Fed's fastest tightening in 40 years — drove the 2022 return of nearly every asset on earth. This post builds the master map of how a hike or a cut transmits to stocks, bonds, credit, real estate, FX, gold, and crypto, ranks each asset by its rate-beta, and shows how to read one policy move into a whole portfolio."
tags: ["macro", "macro-trading", "monetary-policy", "cross-asset", "rate-beta", "transmission", "discount-rate", "2022-selloff", "correlation", "portfolio-construction", "regime-analysis", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Under the price of every asset on earth sits one number: the risk-free rate. Because the Fed sets that number, a single policy move — a hike, a cut, a round of QE or QT — reprices stocks, bonds, credit, real estate, FX, gold, and crypto all at once. 2022 was the cleanest proof in 40 years: the Fed went from near-zero to 5.5% and almost every asset fell together.
>
> - **One variable runs underneath everything.** Every asset is priced as the present value of its future cash flows, discounted at a rate built on the risk-free rate. Raise the risk-free rate and you lower the present value of *all* future cash flows simultaneously — that is why policy moves everything at once.
> - **Each asset has a rate-beta** — how hard its price moves per unit of policy move. Crypto and long-duration growth stocks have the highest positive rate-beta (they fall hardest when rates rise); the dollar and commodities carry *negative* rate-beta (they rise when the Fed hikes); gold sits near zero.
> - **In a rate shock, correlations go to one.** In normal times each asset answers to its own driver and they diversify. When the discount rate becomes the dominant force, every asset answers to the *same* driver, so a "diversified" book falls as one — diversification fails exactly when you need it.
> - **The number to remember:** in 2022 a classic 60/40 stock-bond portfolio lost **−16.1%**, because for the first time in a generation stocks (−18.1%) and bonds (−31.2% at the long end) fell *together* — both repriced off the same rising rate.

In calendar year 2022, almost nothing worked. The S&P 500 lost **18.1%**. The Nasdaq 100 lost **32.5%**. Long-dated Treasury bonds — the asset that is supposed to *protect* you when stocks fall — lost **31.2%**, one of the worst years for bonds in US history. Investment-grade corporate credit lost **15.4%**, high-yield credit **11.2%**, real-estate stocks **26.2%**, and Bitcoin a staggering **64.3%**. The classic 60/40 portfolio, the default "balanced" allocation that millions of people own, lost **16.1%** — its worst year since 2008. A trader who had spread money across a dozen different asset classes, exactly as every textbook tells you to, still watched the whole book go down together.

This was not a coincidence of a dozen separate problems. It was *one* problem, hitting a dozen places at once. In March 2022 the Federal Reserve's policy rate was pinned near zero — the target range topped out at **0.25%** — while inflation ran at four times the Fed's target. Over the next eighteen months the Fed hiked from 0.25% to **5.50%**, the fastest tightening cycle in four decades. That single force — the price of money repricing higher — is what drove the return of nearly every asset on earth that year. Stocks, bonds, credit, real estate, currencies, gold, and crypto did not each have their own bad year for their own reasons. They all repriced off the same variable.

This is the capstone of the asset-class cluster of the *Macro for Traders* series. Across the cluster we walked each channel one at a time: [how monetary policy moves stocks](/blog/trading/macro-trading/how-monetary-policy-moves-stocks-discount-rates-sectors), [how it moves bonds](/blog/trading/macro-trading/how-monetary-policy-moves-bonds-duration-convexity), and the underlying [transmission mechanism](/blog/trading/macro-trading/monetary-policy-transmission-how-rate-changes-reach-markets) that carries a rate change out to markets. Now we pull every channel together into a single master map. The payoff is concrete: by the end you will be able to look at one policy move — a hike, a cut, a QE, a QT — and read it into a *whole* portfolio at once, because you will know each asset's rate-beta and the direction it moves. We build the entire idea from zero.

![Risk-free rate at the center with policy feeding it and stocks, bonds, credit, real estate, dollar, gold, and crypto fanning out](/imgs/blogs/how-policy-moves-every-asset-cross-asset-transmission-map-1.png)

## Foundations: the risk-free rate under everything

Before any map, we need the one idea the whole map is built on. It is the single most important concept in cross-asset macro, and once it clicks, the rest of this post is just working out its consequences. The idea is this: **every asset is priced as a stream of future cash flows, and every stream of future cash flows is discounted at a rate that sits on top of the risk-free rate.** Move the risk-free rate, and you move the price of everything.

### Every asset is a stream of future money

Start with what an asset actually is. Forget the ticker and the chart. An asset is a *claim on future money*. A bond is a claim on a stream of fixed coupon payments plus your principal back at the end. A stock is a claim on a company's future profits — dividends now and a much larger stream of earnings far into the future. A rental property is a claim on a stream of future rents. Even a thing that pays no cash, like gold or a barrel of oil, is a claim on a future *sale price* — money you expect to receive when you sell it later.

So the question "what is this asset worth today?" is always the same question: *what is a stream of future money worth right now?* And the answer to that question depends on one thing above all else — how much you discount the future.

### The discount rate, and why a dollar later is worth less than a dollar now

A dollar you will receive in ten years is worth less than a dollar in your hand today, for a simple reason: if you had the dollar today, you could put it in the safest possible investment and earn interest on it. The "safest possible investment" is lending to the US government by buying a Treasury — an asset so close to riskless that finance treats its yield as the **risk-free rate**, the return you can earn with essentially zero chance of loss.

To convert a future dollar into today's value, you *discount* it by that rate. If the risk-free rate is `r`, then a dollar received `t` years from now is worth `1 / (1 + r)^t` today. That formula is the engine under every asset price in the world. The present value of an asset is the sum of all its future cash flows, each one discounted by `1 / (1 + r)^t`. The bigger `r` is, the more you shrink every future dollar, and the lower the asset's price today.

Here is the part that matters for the whole post. The discount rate applied to any asset is built in two layers:

- **The risk-free rate** — the Treasury yield, set in large part by Fed policy. This is the *floor* under every discount rate. It is the same floor for every asset on earth.
- **A risk premium** — extra return demanded for the chance that *this particular* asset's cash flows don't show up (a company goes bankrupt, a tenant stops paying, a coin goes to zero). Riskier assets carry bigger premiums.

The total discount rate for any asset is `risk-free rate + risk premium`. The risk premium is specific to the asset. But the risk-free rate is *shared by everything*. And that shared component is exactly what the Fed moves when it changes policy.

### The unifying idea: one rate sits under every price

Now the punchline. When the Fed raises the policy rate, it pushes the risk-free rate up. That higher risk-free rate flows into the discount rate of *every* asset — the stock, the bond, the building, the coin — all at the same time. A higher discount rate shrinks the present value of every future cash flow, everywhere, simultaneously. That is the mechanical reason a single policy move can reprice the entire investable universe in the same direction at the same moment.

This is what the cover figure shows: policy moves the risk-free rate, and the risk-free rate is the discount rate that sits inside the price of stocks, bonds, credit, real estate, FX, gold, and crypto. There is no asset that escapes it, because there is no asset whose price isn't a discounted stream of future money. We explored the discount-rate channel for one asset class in detail in [how monetary policy moves stocks](/blog/trading/macro-trading/how-monetary-policy-moves-stocks-discount-rates-sectors); here we generalize it to the whole board.

### Rate-beta: how sensitive each asset is to a policy move

If every asset reprices off the same risk-free rate, why didn't every asset fall by the *same amount* in 2022? Because they don't all have the same sensitivity. The Nasdaq fell 32.5% and gold fell 0.3%, yet both faced the identical move in the risk-free rate. The difference is what we will call **rate-beta**: how much an asset's price moves per unit of move in the policy rate.

The term "beta" comes from how traders describe sensitivity. A stock with a market-beta of 2 moves twice as much as the market. By analogy, an asset's *rate-beta* describes how much it moves for a given change in rates. We will use it loosely — not as a precise regression coefficient but as a ranking of sensitivity — and that loose version is enough to be powerful. The sign and the size of rate-beta are what the whole transmission map is about:

- **High positive rate-beta**: the asset falls a lot when rates rise. These are the long-duration assets — ones whose cash flows are pushed far into the future, so a change in the discount rate compounds over many years. Long bonds, growth stocks, real estate, and crypto live here.
- **Low rate-beta**: the asset barely reacts. Cash and very short-dated instruments sit here, because their cash flows arrive almost immediately and there is little future to discount.
- **Negative rate-beta**: the asset *rises* when rates rise. The dollar is the cleanest example — higher US rates pull global capital into dollars, so a hike strengthens the currency. Some commodities behave this way through the growth-and-inflation channel rather than the discount-rate channel.

Why does *duration* — how far out the cash flows sit — drive rate-beta so strongly? Because of how the discount formula compounds. Discounting a cash flow one year out by a 1% higher rate barely changes it; discounting a cash flow thirty years out by a 1% higher rate shrinks it dramatically, because the `(1 + r)^t` term has `t = 30` in the exponent. An asset whose value is mostly far-future cash — a profitless growth stock priced on earnings a decade away, a 30-year bond, a building valued on rents stretching out forever — is enormously sensitive to the discount rate. An asset whose value is near-term cash is not. Long-duration is high rate-beta; short-duration is low rate-beta. That single relationship explains most of the ranking we are about to build, and it is the same logic we developed for fixed income in [how monetary policy moves bonds](/blog/trading/macro-trading/how-monetary-policy-moves-bonds-duration-convexity).

### The two channels: rate versus growth

There is a second wrinkle we have to name, because it explains the assets that *don't* fit the simple discount-rate story. A policy move reaches assets through **two channels**, not one:

- **The rate channel (the discount-rate effect).** This is everything above: a higher risk-free rate lowers the present value of future cash flows. It hits *every* asset and it hits long-duration assets hardest. This channel works almost instantly, the moment the market reprices rate expectations.
- **The growth channel (the cash-flow effect).** Higher rates also slow the real economy — they make borrowing expensive, cool demand, and eventually shrink corporate earnings, rents, and commodity consumption. This channel changes the *numerator* of the valuation (the cash flows themselves), not just the discount rate in the denominator. It works with a long lag, often a year or more, because it takes time for higher rates to bite into the economy.

Most assets feel both channels, and sometimes they pull in opposite directions. Stocks get hit by the rate channel immediately (the discount rate rises) and by the growth channel later (earnings slow). Commodities are the interesting case: the rate channel says they should fall (higher discount rate), but the growth-and-inflation channel can push them *up* when the policy tightening is itself a response to a hot, inflationary economy — which is exactly why commodities *rose* 16% in 2022 even as nearly everything else fell. Keeping the two channels separate is what lets you predict the exceptions instead of being surprised by them.

### How to estimate an asset's rate-beta in practice

We have been ranking rate-betas by eye, using the 2022 returns. It is worth knowing where the number actually comes from, because once you can estimate it you can apply the map to *any* asset, not just the ten on our chart.

There are three ways to pin a rate-beta down, in rising order of rigor:

- **Duration as a proxy.** For any asset that is fundamentally a discounted stream of cash flows — bonds, real estate, dividend stocks — the duration (the weighted average time to its cash flows, in years) *is* its rate-beta in disguise. A bond with a duration of 17 years loses about 17% of its price for a 1-percentage-point rise in yield. A growth stock priced on earnings centered a decade out behaves like a 20-to-30-year-duration instrument. The longer the duration, the bigger the rate-beta. This is why the single question "how far out are this asset's cash flows?" predicts most of the ranking.
- **Historical regression.** The cleaner empirical method is to regress the asset's returns on changes in the risk-free rate (say, the 2-year or 10-year Treasury yield) over a stretch of history. The slope coefficient is the rate-beta: how many percent the asset moves per percentage-point move in the rate. This is the formal version of what "beta" means, and it is how a quant desk would actually measure it. The catch is that the coefficient is *unstable* — it changes with the regime, which is the whole reason correlations go to one. The rate-beta you measure in a calm year understates the rate-beta you'll feel in a shock.
- **The dominant-shock readout.** The most honest gauge is exactly what we used: look at how the asset behaved during the *last big rate shock*, when the rate channel overwhelmed everything else, and treat that as the asset's rate-beta when it matters. The whole reason 2022 is so useful is that it is a high-signal readout of every asset's rate-beta under stress — which is the only time rate-beta actually decides your P&L.

In practice you blend all three: duration tells you the structural sensitivity, the regression gives you a number in calm times, and the last shock tells you how that number explodes when policy takes over. The crucial humility is that **rate-beta is bigger in a shock than in calm** — assets that look only mildly rate-sensitive in a quiet year become violently rate-sensitive when the discount rate becomes the dominant force.

### Why 2022 was the cleanest natural experiment in decades

Scientists love a clean experiment: one variable changes, everything else holds still, and you watch the result. Markets almost never give you that, because dozens of forces move at once. 2022 came as close as macro ever does. One enormous force — the Fed's fastest tightening in 40 years, from 0.25% to 5.50% — dominated everything else. There was no competing positive story big enough to matter; growth was decent, earnings were fine, there was no banking crisis (that came in early 2023). The risk-free rate simply went vertical, and you could watch its shadow fall across every asset class in real time, each one moving in proportion to its rate-beta. That is why this post uses 2022 as its anchor: it is the closest thing to a controlled experiment that proves the entire thesis. The whole point of a [macro thesis built from data](/blog/trading/macro-trading/building-a-macro-thesis-from-data-to-a-trade) is to find exactly this kind of dominant variable.

## The cross-asset map: ranking the rate-betas

With the foundations in place, we can build the actual map: every major asset class, ranked from highest positive rate-beta (falls hardest when policy tightens) to negative rate-beta (rises when policy tightens). The cleanest available proxy for rate-beta is each asset's 2022 return, because that year the rate shock was the dominant force, so the loss each asset took is a real-world measurement of its policy sensitivity. Here is the master chart — every asset's 2022 return, ranked.

![Horizontal bar chart of 2022 total returns by asset class, ranked from Bitcoin worst to commodities best](/imgs/blogs/how-policy-moves-every-asset-cross-asset-transmission-map-2.png)

Read this chart slowly, because it is the entire thesis in one picture. Eight of the ten assets are red. The two green ones — commodities and the dollar — are precisely the two that carry *negative* rate-beta, the ones that benefit from the same force that crushed everything else. Gold sits at the dividing line, essentially flat. The order from worst to best is not random; it is a ranking by policy sensitivity, and it goes almost exactly the way the duration logic predicts.

Let me walk down the map, asset by asset, naming the channel that drives each one.

### Crypto — the highest rate-beta of all (−64.3%)

Bitcoin and the broader crypto complex sit at the extreme. Crypto has *no cash flow at all* — no coupon, no dividend, no earnings. Its entire value is a claim on a future sale price, which makes it the longest-duration asset there is: pure future, no near-term cash to anchor it. With nothing to discount but a distant, speculative payoff, it is maximally sensitive to the cost of money and to risk appetite. When liquidity is abundant and rates are low, crypto inflates spectacularly; when the Fed drains liquidity and raises the discount rate, it deflates just as spectacularly. In 2022 it fell 64.3%, more than three times the S&P. We built this case fully in [crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset): crypto is the purest high-beta expression of the global cost of money.

### Long-duration growth stocks — Nasdaq (−32.5%)

The Nasdaq 100 is concentrated in high-growth technology — companies whose value is overwhelmingly in *future* earnings, not current ones. A profitless software company valued on cash flows a decade away is, in discount-rate terms, a very long-duration asset. When the discount rate jumps, the present value of those far-future earnings collapses. That is why growth stocks fell roughly twice as much as the broad market in 2022. The S&P 500 (−18.1%) is a blend of long-duration growth and shorter-duration value, so it sits in the middle.

### Long-dated bonds — the supposed safe haven (−31.2%)

Here is the result that shocked a generation of investors. Long Treasuries lost 31.2% — nearly as much as the Nasdaq. Bonds are supposed to be safe. But a 20-or-30-year bond is, by construction, an extremely long-duration asset: a fixed stream of coupons stretching out decades, whose present value moves inversely and powerfully with the discount rate. When yields go from under 2% to over 4%, the price of a long bond falls violently. The "safety" of a bond is about *credit* risk (will you get paid?), not *rate* risk (what is the price along the way?). On the rate dimension, a long bond is one of the highest rate-beta assets in existence — a point we developed in detail in [how monetary policy moves bonds](/blog/trading/macro-trading/how-monetary-policy-moves-bonds-duration-convexity).

### Credit — the double hit (IG −15.4%, HY −11.2%)

Corporate bonds get hit *twice*. First by the rate channel, like all bonds (their yields rose with Treasuries). Second by the spread channel: in a tightening, the *risk premium* over Treasuries usually widens too, because tighter policy raises default fears. So credit faces both a rising risk-free rate and a rising risk premium. Investment-grade credit, with longer duration and tighter starting spreads, fell more than high-yield in 2022 — an unusual ordering caused by the fact that 2022's damage was mostly the rate channel, not a default scare, so the longer-duration IG bonds took the bigger rate hit.

### Real estate — rates with leverage (−26.2% for REITs)

Property is valued almost exactly like a long bond: a stream of future rents discounted at a rate built on the risk-free rate. The number that does this in real estate is the **cap rate** — the rent yield on a property — and cap rates track the risk-free rate. When rates rise, cap rates rise, and property values fall. Worse, real estate is highly *leveraged* — bought with borrowed money — so when the cost of that borrowing jumps, the equity slice gets squeezed hard. Real-estate stocks fell 26.2% in 2022, and mortgage rates told the same story: the 30-year fixed went from a record-low **2.65%** in January 2021 to **7.79%** by October 2023, the highest since 2000. That move alone froze the housing market.

### The dividing line — gold (−0.3%)

Gold is the hinge of the whole map, and it teaches the deepest lesson. Gold pays no cash flow, so you'd expect it to behave like crypto — crushed by a rising discount rate. But gold is priced primarily off the **real** rate (the rate *after* inflation), not the nominal rate. In 2022 nominal rates soared, but so did inflation, so the real rate's rise was more muted, and gold's two opposing forces — a higher real rate (bad for gold) and high inflation plus geopolitical fear (good for gold) — roughly cancelled. The result: essentially flat, −0.3%. Gold's near-zero rate-beta is *exactly why it sits in the middle of the ranking* and why it can be a genuine diversifier when the rate-beta of everything else goes to one.

### Negative rate-beta — the dollar (+8.2%) and commodities (+16.1%)

At the green end are the two assets that *rose*. The **dollar** has negative rate-beta through the capital-flows channel: when the Fed raises US rates faster than other central banks, global capital chases the higher yield into dollars, and the dollar strengthens. The DXY dollar index spiked to a two-decade high of nearly **115** in September 2022. **Commodities** rose through the growth-and-inflation channel: the very inflation that *forced* the Fed to hike also lifted the price of oil, food, and metals. The Bloomberg Commodity Index gained 16.1%. These two are not exceptions to the rule — they are the rule's other half. The same force (tightening policy) that crushes long-duration risk assets *lifts* the currency and, when the tightening is fighting inflation, the inflation hedges. We trace the dollar mechanism in [trading the dollar](/blog/trading/macro-trading/trading-the-dollar-dxy-carry-dollar-smile).

#### Worked example: the 2022 cross-asset tally on a \$1,000,000 book

Let's make the master chart concrete with money. Suppose at the start of 2022 you held a genuinely "diversified" \$1,000,000 portfolio — \$100,000 spread across each of ten asset classes — exactly the kind of spread a textbook recommends. Apply each asset's actual 2022 return and watch what happens.

- **S&P 500**, \$100,000 × (−18.1%) = **−\$18,100**
- **Nasdaq 100**, \$100,000 × (−32.5%) = **−\$32,500**
- **Long Treasuries (20y+)**, \$100,000 × (−31.2%) = **−\$31,200**
- **IG corporate bonds**, \$100,000 × (−15.4%) = **−\$15,400**
- **High-yield bonds**, \$100,000 × (−11.2%) = **−\$11,200**
- **Real estate (REITs)**, \$100,000 × (−26.2%) = **−\$26,200**
- **Gold**, \$100,000 × (−0.3%) = **−\$300**
- **Commodities**, \$100,000 × (+16.1%) = **+\$16,100**
- **US dollar**, \$100,000 × (+8.2%) = **+\$8,200**
- **Bitcoin**, \$100,000 × (−64.3%) = **−\$64,300**

Sum it up: the total change is **−\$174,900**, so the \$1,000,000 book ends the year at **\$825,100**, a loss of **17.5%**. Eight of your ten sleeves lost money; only commodities and the dollar gained, and together they added back just \$24,300 against \$199,200 of losses elsewhere. The takeaway is stark: spreading across ten "different" asset classes barely helped, because eight of them were really the same trade — a bet on the cost of money — wearing different costumes.

## Why the correlations all went to one in 2022

We have to dwell on the single most counterintuitive fact in that worked example: the diversification didn't work. You owned ten different things and they fell as one. To understand why, we need to talk about what *correlation* actually is and why it changes with the regime.

### Correlation is not a constant — it depends on what's driving

**Correlation** measures whether two assets move together. A correlation of +1 means they move in perfect lockstep; 0 means no relationship; −1 means they move in exact opposition. The entire promise of diversification rests on the idea that different assets have *low* correlation — when one zigs, another zags, and the bumps cancel out so the portfolio is smoother than any single holding.

But correlation is not a fixed property of two assets. It depends on *what is driving them at the moment*. Two assets are uncorrelated when they answer to *different* dominant forces. Stocks normally answer to earnings and growth; bonds normally answer to the rate; gold answers to real rates and fear. In normal times these drivers are independent, so the assets are independent, so the portfolio diversifies. That is the world on the left of the next figure.

### When one force dominates, everything answers to it

Now suppose one force becomes so large it overwhelms every asset's individual driver. When the Fed hikes from 0.25% to 5.50% in eighteen months, the discount rate stops being a quiet background variable and becomes the *loudest* thing in the room. Suddenly stocks are not trading on earnings — they are trading on the discount rate. Bonds are trading on the discount rate. Real estate, credit, crypto — all of them are trading on the discount rate, because the move in that one variable is bigger than anything happening in their individual fundamentals. When every asset answers to the *same* driver, every asset moves *together*. Correlations collapse toward +1. That is the world on the right.

![Before and after panels showing many independent asset drivers collapsing into one shared driver in a rate shock](/imgs/blogs/how-policy-moves-every-asset-cross-asset-transmission-map-6.png)

This is the cruel mechanics of a rate shock: **diversification fails exactly when you need it most.** In calm markets, when you don't really need protection, your assets are nicely uncorrelated. In a policy shock, when you desperately need something to hold up, the correlations all rush to one and the whole book goes down together. The diversification was never a property of the assets — it was a property of the *regime*, and the regime can flip in an instant when one force takes over. We explore this regime-dependence of correlation, and the few assets that genuinely hold up, in [when correlations go to one in a crisis](/blog/trading/macro-trading/when-correlations-go-to-one-in-a-crisis).

### The death of the stock-bond hedge

The most painful version of this in 2022 was the breakdown of the classic stock-bond relationship. For most of the previous two decades, stocks and bonds were *negatively* correlated: when stocks fell (usually on growth fears), the Fed would cut rates, bonds would rally, and the bond gains would cushion the stock losses. This is the engine of the 60/40 portfolio — 60% stocks, 40% bonds — and it is why "balanced" funds felt safe.

That hedge depends on the *cause* of the stock decline. If stocks fall because of slowing growth, bonds rally and the hedge works. But if stocks fall because the *discount rate is rising*, bonds fall too — because the very same rising rate that crushes stocks also crushes bonds. In 2022, stocks didn't fall on a growth scare; they fell on a rate shock, and so bonds fell *with* them instead of hedging them. Both legs of the 60/40 went down together for the first time in a generation. We unpack this engine in detail in [the stock-bond correlation and the 60/40 engine](/blog/trading/macro-trading/stock-bond-correlation-the-60-40-engine).

#### Worked example: the 60/40 book when stocks and bonds fall together

Take a \$1,000,000 portfolio built the classic way: **\$600,000 in stocks** (the S&P 500) and **\$400,000 in bonds**. In a normal bad year for stocks, bonds rally and soften the blow. Let's see what 2022 — a rate shock — did instead.

- The stock sleeve: \$600,000 × (−18.1%) = **−\$108,600**.
- The bond sleeve: a typical "aggregate" bond holding (a blend of maturities) lost about **13%** in 2022 as yields surged; \$400,000 × (−13%) = **−\$52,000**.
- Total loss: −\$108,600 + (−\$52,000) = **−\$160,600**, leaving the book at **\$839,400**, a loss of about **16.1%** — matching the published 60/40 return for the year.

Now compare to a *normal* bad year, where the same stock loss is cushioned by a bond rally. If bonds had instead gained +5% (as they often did when stocks fell pre-2022), the bond sleeve would have *added* \$400,000 × 5% = **+\$20,000**, and the book would have lost only **−\$88,600**, or 8.9% — barely half the damage. The difference between a −16.1% year and an −8.9% year is entirely the sign of the stock-bond correlation, and that sign is set by *why* stocks are falling. The takeaway: the 40% in bonds only protects you when stocks fall on growth; in a rate shock the bonds become a *second* losing bet on the very same variable.

## The policy-to-asset matrix: hike vs cut, QE vs QT

So far we have measured rate-beta with one real episode (the 2022 tightening). To make the map a true tool, we need to generalize: not just "tightening crushes risk assets," but the *direction every asset moves under every kind of policy move*. The Fed has four basic levers, and each one transmits to each asset class with a predictable sign. Here is the full matrix.

![Heatmap matrix of asset classes by policy action showing price up in green and price down in red](/imgs/blogs/how-policy-moves-every-asset-cross-asset-transmission-map-4.png)

The columns are the four policy actions: a **rate hike**, a **rate cut**, **QE** (quantitative easing — the Fed buying bonds, which lowers long rates and floods the system with liquidity), and **QT** (quantitative tightening — the Fed shrinking its balance sheet, which drains liquidity). The rows are the asset classes. Each cell is green if the asset's price typically *rises* under that policy action and red if it typically *falls*. (The signs are a synthesis of the transmission channels we have built throughout this post and the cluster; we explored the balance-sheet tools themselves in [QE vs QT](/blog/trading/macro-trading/qe-vs-qt-how-balance-sheet-policy-moves-markets).)

Three patterns jump out of the matrix, and they are the whole reason to build it:

**First, the columns come in mirror-image pairs.** A hike and a cut are opposites: almost every cell flips sign between them. The same is true of QE and QT — QE is "easing through the balance sheet," QT is "tightening through the balance sheet," so their columns are near-mirrors. This means you don't have to memorize four columns; you memorize *two directions* (easing and tightening) and the four levers collapse into them. A hike and QT both *tighten*; a cut and QE both *ease*.

**Second, most assets share the same sign.** Look down the "rate hike" column: long bonds, credit, stocks, growth, real estate, and crypto are *all red*. That vertical wall of red *is* the correlation-to-one phenomenon, drawn as a matrix. When the dominant policy move is a hike, the high rate-beta assets all fall together — there is your 2022.

**Third, two rows break ranks.** The **dollar** row is the inverse of everything above it — green where the others are red. A hike *lifts* the dollar (negative rate-beta). **Commodities** are mixed and often move with the growth-and-inflation channel rather than the discount-rate channel, so they don't cleanly follow the others. These two rows are your potential hedges and your "other side" of the trade: in a tightening, you don't just sell the red rows — you can *buy* the dollar.

### Why QE and QT add a second channel: liquidity, not just the rate

The rate levers (hike and cut) work through the discount-rate channel we have built. The balance-sheet levers (QE and QT) work through that channel *and* a second one — the **liquidity channel** — which is worth separating because it explains why QE and QT can move risk assets even when the policy rate isn't changing.

QE is the Fed buying bonds with newly created money. It does two things at once. First, by buying long-dated bonds it pushes their yields *down*, which lowers the long end of the risk-free curve — that's the rate channel. Second, the money it creates lands in the banking system as reserves and cash that has to go *somewhere*, and a lot of it flows into risk assets, bidding them up — that's the liquidity channel. QT is the reverse: the Fed lets bonds roll off its balance sheet, which drains cash out of the system and pushes long yields up. The Fed's balance sheet went from about \$4.2 trillion before COVID to nearly \$9 trillion at the 2022 peak under QE, then began draining under QT toward \$6.6 trillion by 2025.

The liquidity channel is why the highest rate-beta assets — crypto especially — are even more sensitive to QE and QT than to the policy rate itself. Crypto has no cash flow for the discount rate to act on, so its price is almost pure demand, and demand rides the tide of liquidity. When QE floods the system, the marginal dollar chases the highest-octane risk asset; when QT drains it, that same dollar leaves first. This is why the crypto and growth rows in the matrix are deep red under QT and deep green under QE: they feel both the rate channel and the liquidity channel at full force. We built the liquidity-tide idea in full in [the liquidity cycle and asset prices](/blog/trading/macro-trading/the-liquidity-cycle-and-asset-prices); the point for the map is that the four policy levers are really *two channels* — rate and liquidity — and the balance-sheet levers hit both.

### The same dispersion shows up inside the stock market

A beautiful confirmation of the whole rate-beta idea is that it doesn't just rank *asset classes* — it ranks the *sectors inside the stock market* by the same logic. A stock market is itself a portfolio of businesses with very different durations. Some sectors are short-duration cash cows (energy, utilities) and some are long-duration growth bets (technology, communications). If rate-beta is real, the rate shock should disperse the sectors in exactly the same order it disperses the asset classes. It did.

![Horizontal bar chart of 2022 S&P 500 sector total returns ranked from energy best to communication services worst](/imgs/blogs/how-policy-moves-every-asset-cross-asset-transmission-map-5.png)

Energy *gained* 65.7% in 2022 — the same growth-and-inflation channel that lifted commodities lifted the energy companies that produce them. Utilities and consumer staples, the "bond-proxy" defensives with stable near-term cash flows, were roughly flat. At the bottom, the long-duration growth sectors got crushed: technology −28.2%, consumer discretionary −37.0%, communication services −39.9%. Real estate, the most rate-leveraged sector, fell 26.2%. This is the *same chart* as the asset-class map, one level down. The rate-beta ranking is fractal: it orders asset classes, and inside the stock asset class it orders sectors, by the identical duration logic. We mapped this sector dispersion in [how monetary policy moves stocks](/blog/trading/macro-trading/how-monetary-policy-moves-stocks-discount-rates-sectors).

#### Worked example: ranking assets by rate-beta with a classifier

Here is the rate-beta map as a tiny piece of code. It takes the dictionary of asset returns, ranks the assets from most to least policy-sensitive, classifies each one's beta sign, and prints a regime read. The point is to show that "the cross-asset map" is mechanical enough to be a function — give it a policy direction and it tells you which assets lead.

```
def classify_rate_beta(returns_2022):
    ranked = sorted(returns_2022.items(), key=lambda kv: kv[1])
    rows = []
    for name, ret in ranked:
        if ret <= -25:
            beta = "very high positive rate-beta"      # crushed by a hike
        elif ret < -5:
            beta = "high positive rate-beta"
        elif ret <= 5:
            beta = "near-zero rate-beta (diversifier)"  # gold-like
        else:
            beta = "negative rate-beta (rises on a hike)"
        rows.append((name, ret, beta))
    return rows

def read_regime(rows, policy):
    lead = "SELL high-beta, BUY dollar" if policy == "tighten" else "BUY high-beta, SELL dollar"
    print(f"Policy = {policy}  ->  {lead}")
    for name, ret, beta in rows:
        print(f"  {name:<26} {ret:+6.1f}%   {beta}")

rows = classify_rate_beta(ASSET_RETURNS_2022)
read_regime(rows, "tighten")
```

Run it on the 2022 returns and the ordering falls out exactly as the chart shows: Bitcoin and the Nasdaq and long bonds flagged as the very-high-beta assets to sell in a tightening, gold flagged as the near-zero diversifier, the dollar and commodities flagged as the negative-beta assets to buy. To turn the classifier into dollars, weight your \$1,000,000 book by the flags: put the largest *short* risk against the very-high-beta names — say \$50,000 of risk short the Nasdaq and crypto sleeve — and your *long* risk into the negative-beta names — \$50,000 long the dollar — while parking gold and cash as the near-zero ballast. Flip the `policy` argument to `"ease"` and every recommendation inverts, so the same \$1,000,000 book swings from short the high-beta names to long them. The takeaway: the cross-asset map is not a vibe — it is a deterministic function from one policy direction to a sized, dollar position in every asset at once.

![Horizontal bar chart ranking assets by rate-beta using 2022 returns, with high-beta fallers and negative-beta gainers labeled](/imgs/blogs/how-policy-moves-every-asset-cross-asset-transmission-map-3.png)

That ranking — drawn here as a chart — is the heart of the whole post. The dark-red bars at the bottom (Bitcoin, Nasdaq, long bonds) are the high positive rate-beta assets that fall hardest when policy tightens; the green bars at the top (commodities, the dollar) are the negative rate-beta assets that *rise* when the Fed hikes. Gold sits on the line. Memorize this order, because it is the same order in every tightening and the *reverse* order in every easing.

#### Worked example: reading a hypothetical rate cut into the book

We have done tightening to death. Let's run the map the *other* way, because the real test of a model is whether it predicts the symmetric case. Suppose it is some future month and the Fed surprises the market with the start of a genuine easing cycle — a cut, plus a signal that more cuts and eventually QE are coming. The discount rate is about to *fall*. Run it down the same \$1,000,000 book, but now invert every sign.

- **The highest positive rate-beta assets lead the rally.** The same crypto, growth stocks, and long bonds that fell hardest in 2022 now *rise* hardest, because a falling discount rate lifts long-duration cash flows the most. If you wanted to express maximum conviction in the easing, you'd put risk into Bitcoin and the Nasdaq — a \$100,000 sleeve in each is your highest-octane bet on the cut.
- **Long bonds lead among the "safe" assets.** This is the key asymmetry: in an easing, bonds become a hedge *and* a winner again. Buying duration early — long Treasuries — captures the falling-rate move directly. A \$200,000 allocation to long bonds at the start of an easing cycle is a classic high-conviction easing trade.
- **The negative rate-beta assets become headwinds.** The dollar, which rose 8.2% in the 2022 tightening, tends to *fall* in an easing as capital leaves for higher-yielding currencies elsewhere; commodity-beta that helped in the tightening now hurts. So you'd *cut* the dollar sleeve and trim commodity exposure.
- **The order of leadership is the reverse of 2022.** Where the tightening ranking ran crypto-worst to commodities-best, the easing ranking runs crypto-*best* to dollar-worst. Same assets, same ranking, sign flipped.

So a single read — "the Fed is easing more than priced" — positions the entire book: overweight the high-beta long-duration assets (crypto, growth, long bonds), own duration as both an engine and a hedge, and underweight the dollar. The takeaway: one policy read, run through each asset's rate-beta, gives you a *complete* portfolio tilt — and the easing playbook is just the tightening playbook with every sign reversed.

## Common misconceptions

The cross-asset map overturns three beliefs that feel like common sense and cost people real money. Each is corrected with a number.

### Misconception 1: "Diversification protects you in a rate shock"

The whole point of diversification is supposed to be that owning many different things protects you. But we just watched a ten-asset portfolio lose 17.5% and a 60/40 lose 16.1% in 2022, because the assets weren't really *different* — they were all bets on the cost of money. Diversification protects you against *asset-specific* risk (one company goes bankrupt, one sector falls out of favor). It does **nothing** against a force that hits the discount rate under *all* of them at once. The deeper lesson: count your *bets*, not your *positions*. Ten positions that all reprice off the risk-free rate are one bet held ten ways. True diversification means owning assets with genuinely different *drivers* — and in a rate shock, almost the only ones left are cash, the dollar, and (sometimes) commodities and gold.

### Misconception 2: "Some assets are policy-immune"

People reach for an asset they believe is outside the system — gold, Bitcoin, real estate, "real" assets — on the theory that it can't be touched by what the Fed does. The map says otherwise. Bitcoin, far from being immune, had the *highest* rate-beta of all and fell 64.3%. Real estate, the asset people trust most as a store of value, fell 26.2% and froze as mortgage rates went from 2.65% to 7.79%. Even gold, the closest thing to a genuine diversifier, is priced off the real rate and is only "immune" in the narrow sense that its two opposing forces happened to cancel. There is no asset whose price isn't a discounted stream of future money, and there is therefore no asset that is truly policy-immune. The honest version of "policy-immune" is "low or negative rate-beta" — and the only assets that fit are cash, short-duration debt, the dollar, and the inflation hedges, not the speculative ones people usually reach for.

### Misconception 3: "The effects on each asset are independent"

It's tempting to analyze each asset on its own: stocks for their earnings, bonds for their yields, crypto for its adoption, gold for its fear bid. But that misses the structure entirely. The effects are *not* independent — they are different projections of the *same* underlying variable. The risk-free rate doesn't hit stocks and then separately hit bonds and then separately hit crypto; it hits the discount rate that is *shared* by all of them, simultaneously. That is why they correlate to one. Treating the assets as independent is what leads people to think they're diversified when they're not. The correct mental model is one root cause (the policy rate) and many branches (the asset classes), as the cover figure draws it — not many separate trees.

### Misconception 4: "If I get the policy call right, every asset trade works"

The other side of the error. Knowing the Fed will hike is necessary but not sufficient, because the *same* correct call lands with completely different force on different assets. In 2022, the correct call ("the Fed hikes more than priced") was enormously profitable short the Nasdaq or long the dollar, roughly neutral in gold, and *positive* if you were long commodities. The policy direction tells you the *sign* of each asset's move via the matrix; the rate-beta tells you the *size*. Getting paid requires choosing the asset whose rate-beta is large and clean — which is exactly why a macro view should be expressed through its highest-beta, cleanest instrument, the lesson at the heart of [building a macro thesis from data to a trade](/blog/trading/macro-trading/building-a-macro-thesis-from-data-to-a-trade).

## How it shows up in real markets

The map is not just a 2022 story. The same mechanics run in both directions, and the two cleanest episodes of the last decade are mirror images of each other.

### 2022: the synchronized selloff (tightening)

We have anatomized this throughout, so here is the compressed version as a case study in transmission. The Fed went from 0.25% to 5.50% in eighteen months. The risk-free rate went vertical: the 2-year Treasury yield rose from under 1% to over 5%, the 10-year from under 2% to over 4%. Every long-duration asset repriced down in proportion to its rate-beta — crypto worst (−64%), then growth stocks and long bonds (~−32%), then the broad market and credit and real estate, with gold flat and the dollar and commodities up. Correlations went to one; the 60/40 lost 16.1%; the stock-bond hedge died. It was the rate channel in its purest form: the growth channel (slowing earnings) barely had time to bite, so 2022 was almost *entirely* a discount-rate event. The full case is in [the 2022 case study: stocks and bonds both fell](/blog/trading/macro-trading/case-study-2022-stocks-and-bonds-both-fell).

### 2020: the synchronized rally (easing)

Now run the film backward. In March 2020, as COVID hit, the Fed slammed the policy rate to 0.25% and launched the largest QE in history — its balance sheet went from about \$4.2 trillion to nearly \$9 trillion. The discount rate *collapsed*, and liquidity *flooded* in. The result was the mirror image of 2022: nearly every asset rallied together, hardest at the high-beta end. Bitcoin went from about \$5,000 in March 2020 to nearly \$69,000 by late 2021. The Nasdaq doubled off its lows. Long bonds rallied as yields fell to record lows. Gold hit an all-time high. Even the assets that "shouldn't" move together all rose, because the same force — a collapsing discount rate plus a tide of liquidity — lifted everything at once. This is the "everything rally," and it is the correlation-to-one phenomenon with the sign flipped: when one force dominates, assets correlate to one *whether that force is pushing up or down*.

### When the map breaks: growth shocks and credit events

The map is built for the *rate channel* dominating. It is worth being precise about when it doesn't, because trading the map in the wrong regime loses money. There are two cases where the cross-asset signs differ from the tightening matrix:

- **A pure growth shock (a recession scare with no rate move).** When stocks fall because the *economy* is cracking rather than because rates are rising, the discount rate often falls (the market expects the Fed to cut), so bonds *rally* and the old stock-bond hedge works again. In a growth shock, the rate-beta map is the wrong tool — you want the *growth-beta* map, where defensive sectors and long bonds protect you. The 2020 COVID crash had both: a violent growth shock *and* an enormous easing, which is why long bonds rallied (rate channel) even as stocks briefly collapsed (growth channel).
- **A credit event (a banking or default scare).** When the fear is that someone won't get paid, risk premiums blow out independently of the risk-free rate. Credit spreads gap wider, low-quality assets fall hardest, and "safe" assets (Treasuries, the dollar, gold) catch a flight-to-quality bid even if rates aren't moving. In March 2023, when several regional banks failed, the high rate-beta map briefly inverted: long bonds *rallied* hard as the market priced rate cuts, even though the prior eighteen months had been pure tightening.

The discipline is to identify which channel is in charge *before* applying the map. If the dominant variable is the discount rate, the rate-beta map rules and the signs are as drawn. If it's growth or credit, a different map rules. Most of the time in the modern era the rate channel dominates — but the cost of applying it during a growth or credit shock is getting the sign of the bond hedge exactly backwards.

### The general lesson: read the dominant force first

The two episodes together teach the master skill: before you analyze any single asset, ask *what force is dominating right now?* When one force — usually policy and liquidity — is large enough to overwhelm individual fundamentals, the cross-asset map takes over and you trade the *regime*, not the asset. When no single force dominates (the calm, range-bound years), correlations fall, diversification works again, and you can trade assets on their own merits. The art of cross-asset macro is recognizing which world you are in — and the dashboard for that is the next section.

## How to trade it / the playbook

Here is the payoff: how to use the map to read one policy move into a whole book. The discipline is a single dashboard you run on every policy surprise.

![Matrix playbook mapping tightening versus easing surprises to positions across rate-beta, the dollar, bonds, and what to size up](/imgs/blogs/how-policy-moves-every-asset-cross-asset-transmission-map-7.png)

### Step 1 — read the policy direction relative to what's priced

The map keys off *surprises*, not levels. Markets have already priced in the expected policy path, so the asset moves come from the *gap* between what the Fed does and what was priced. The question is never "is the Fed hiking?" but "is the Fed hiking *more or less than the market expects*?" You read this from the rates futures curve, the [dot plot](/blog/trading/macro-trading/inflation-and-the-fed-reaction-function-dot-plot), and the inflation and jobs data that drive the Fed's reaction. A hawkish surprise (more tightening than priced) sends you down the left column of the playbook; a dovish surprise (more easing than priced) sends you down the right.

### Step 2 — run the move down every asset by rate-beta

Once you have the direction, the matrix does the rest mechanically. On a **tightening surprise**: sell the highest rate-beta assets (crypto, growth stocks, long bonds), buy the dollar, lean toward commodities and cash, and *do not* count on bonds to hedge — they'll fall with stocks. On an **easing surprise**: buy the highest rate-beta assets (the same crypto, growth, and long bonds now lead the rally), own duration as both an engine and a hedge, and short or underweight the dollar. The same dashboard reads a hike and a cut in opposite directions — that's the elegance of it: one read positions the whole book.

### Step 3 — size by rate-beta, not by dollars

The map doesn't just tell you direction — it tells you *how much* of each asset to hold, because the rate-beta is also a risk number. A \$100,000 position in Bitcoin carries far more rate-risk than \$100,000 in gold, because Bitcoin's rate-beta is enormous and gold's is near zero. So you size *down* the high-beta sleeves to keep each position's *risk* contribution comparable, or you concentrate your conviction in the highest-beta clean instrument when you want maximum exposure to the policy view. Either way, size flows from rate-beta. This is the cross-asset version of the sizing discipline we built for single trades: the position is the opinion, and rate-beta is how the opinion translates into risk.

### Step 4 — watch the correlation regime, not just the assets

The dashboard's most important signal is *correlation itself*. When cross-asset correlations are rising toward one — when stocks, bonds, credit, and crypto all start moving together — that is the tell that a single force (policy/liquidity) has taken over and the map is in control. In that regime, diversification is fake, hedges fail, and you must reduce *gross* risk because you have fewer real bets than positions. When correlations fall back apart, the regime has passed, individual fundamentals matter again, and diversification is real. The correlation regime *is* the on/off switch for the whole map — track it on the [regime dashboard](/blog/trading/macro-trading/reading-the-regime-in-real-time-the-dashboard).

### Step 5 — set the invalidation on the policy variable

Because the whole book is keyed to one variable, the invalidation is keyed to it too. If your read is "the Fed eases more than priced, so own duration and high-beta," the trade is wrong the moment the data forces the Fed *hawkish* — a hot CPI or jobs print that pushes the priced rate path back up. Pre-commit to that: the print that flips the policy direction is the print that kills the whole positioning, all at once. The beauty and the danger of cross-asset macro are the same thing — *one* variable moves the entire book, so *one* data surprise can validate or destroy it. Know which print that is before you put the book on.

### The one-line version

Find the dominant force. In the modern era, more often than not, it is policy and the liquidity it controls. Read whether that force is tightening or easing *relative to what's priced*. Then run it down every asset by rate-beta: high-beta long-duration assets move most and in the direction of the easing/tightening (down on tightening, up on easing), the dollar and commodities move the *other* way, gold and cash sit still, and bonds hedge only when the move is about growth, not rates. One policy read, the whole book — that is the cross-asset transmission map.

## Further reading & cross-links

This post is the capstone of the asset-class cluster. To go deeper on any single channel:

- [How monetary policy moves stocks: discount rates and sectors](/blog/trading/macro-trading/how-monetary-policy-moves-stocks-discount-rates-sectors) — the discount-rate channel and sector rate-beta in full.
- [How monetary policy moves bonds: duration and convexity](/blog/trading/macro-trading/how-monetary-policy-moves-bonds-duration-convexity) — why long bonds are such a high rate-beta asset.
- [Monetary policy transmission: how rate changes reach markets](/blog/trading/macro-trading/monetary-policy-transmission-how-rate-changes-reach-markets) — the plumbing that carries a rate change out to prices.
- [Building a macro thesis: from data to a trade](/blog/trading/macro-trading/building-a-macro-thesis-from-data-to-a-trade) — how to turn a policy read into a single, sized, falsifiable position.

And the supporting pieces referenced along the way: [crypto as a macro liquidity asset](/blog/trading/macro-trading/crypto-as-a-macro-liquidity-asset), [the stock-bond correlation and the 60/40 engine](/blog/trading/macro-trading/stock-bond-correlation-the-60-40-engine), [when correlations go to one in a crisis](/blog/trading/macro-trading/when-correlations-go-to-one-in-a-crisis), [QE vs QT](/blog/trading/macro-trading/qe-vs-qt-how-balance-sheet-policy-moves-markets), and [reading the regime in real time](/blog/trading/macro-trading/reading-the-regime-in-real-time-the-dashboard).

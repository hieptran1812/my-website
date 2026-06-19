---
title: "2021-23: The Fastest Hiking Cycle and the 60/40 Wreck"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "The case study of the modern discount-rate shock: how the Fed's 525bp of hikes sank stocks and bonds together, wrecked the 60/40 portfolio, and buried Silicon Valley Bank's bond book."
tags: ["monetary-policy", "interest-rates", "discount-rate", "inflation", "sixty-forty-portfolio", "silicon-valley-bank", "duration", "asset-valuation", "central-banks", "bond-market", "case-study"]
category: "trading"
subcategory: "Policy & Markets"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Between March 2022 and July 2023 the Federal Reserve raised its policy rate from 0.25% to 5.50% — 525 basis points in sixteen months, the fastest hiking cycle since Volcker — to fight inflation that had peaked at 9.1%. That one lever moved the discount rate behind *every* asset at once, so in 2022 stocks **and** bonds fell together: the S&P 500 lost 19.4%, the Aggregate bond index lost 13%, and a classic 60/40 portfolio lost about 16% — the diversification that had worked for forty years failed in a single year.
>
> - A 60/40 portfolio is built on the assumption that bonds rise when stocks fall. That works when stocks fall because of a *recession* (the Fed cuts, bonds rally). It fails when stocks fall because of *rising rates*, because the same rising rate that compresses stock multiples also crushes bond prices. In 2022 the cushion *was* the wrecking ball.
> - The mechanism is pure discount-rate arithmetic: the 10-year Treasury yield went from about 1.5% to a 2022 high of 4.34%, and because that yield sits in the denominator of every valuation, it repriced bonds down (duration loss), equities down (multiple compression), and anything rate-sensitive down — all in the same direction at the same time.
> - The tail risk arrived in March 2023: Silicon Valley Bank had parked a flood of pandemic-era deposits in long bonds at ~1.5%. The rate move buried that book in unrealized losses; a deposit run forced the bank to sell and crystallize the loss, which wiped out its equity in days. The one sentence to remember: **when policy moves the discount rate hard enough, correlations go to one.**

In October 2021, the Federal Reserve's official position on the fastest-rising inflation in a generation was a single word: *transitory*. The price spikes, the argument went, were the temporary indigestion of an economy reopening from a pandemic — snarled supply chains, a one-time surge in demand for goods, a base effect from the deflationary spring of 2020. Give it a few months and it would pass on its own. The Fed's policy rate sat at essentially zero, where it had been since March 2020, and the central bank was still *buying* bonds — adding stimulus — even as consumer prices climbed past 6%.

Eight months later, that word had been retired, inflation had reached **9.1%** — a four-decade high — and the Fed was in the middle of the most violent tightening campaign since Paul Volcker broke the back of 1970s inflation. It raised rates at four consecutive meetings by an enormous 75 basis points each, a pace it had not used since 1994, and kept going until the policy rate hit 5.50% in July 2023. The journey from 0.25% to 5.50% — 525 basis points in sixteen months — was the fastest hiking cycle in forty years.

This post is the case study of what that single lever did to asset prices. It is the cleanest modern demonstration of the idea this whole series turns on: that a central bank pulling *one* dial — the discount rate — can reprice *every* asset at once, in the same direction, at the same time. We will watch it wreck the most trusted portfolio construction in finance (the 60/40), bury a long-bond book deep enough to take down a bank (Silicon Valley Bank), and teach a generation of investors a lesson their parents learned under Volcker and then forgot: diversification is a regime, not a law of nature, and when the discount rate moves hard enough, the correlations you were counting on go to one.

![Graph showing the Fed hike feeding the discount rate which splits to the stock leg and bond leg and sinks the 60/40 portfolio](/imgs/blogs/2021-23-the-fastest-hiking-cycle-and-the-60-40-wreck-1.png)

The figure above is the whole argument in one picture. The Fed pulled a single lever — 525 basis points of hikes. That lever pushed up the discount rate, proxied by the 10-year Treasury yield going from about 1.5% to 4.3%. And that one discount rate fanned out to hit *both legs* of a 60/40 portfolio: it compressed the stock leg's valuation multiple and it crushed the bond leg's price through duration. The S&P fell 19.4%, the bond index fell 13%, and because both legs went down together, the blended portfolio fell about 16%. One box at the front; two very different boxes at the back, both moved by the same hand. Everything below is that diagram, expanded with the dated numbers and the napkin arithmetic.

## Foundations: the discount rate, the 60/40, and why diversification usually works

Before we can understand why 2022 was a catastrophe, we have to understand why the years *before* it were so calm — and that requires three building blocks: what the discount rate is, what a 60/40 portfolio is, and why the two of them normally cooperate to give investors a smooth ride.

### The discount rate: the master valuation dial

Every asset is, at bottom, a claim on future money. A stock is a claim on a company's future profits. A bond is a claim on future coupon payments plus the return of your principal. An apartment building is a claim on future rent. To turn any of those *future* dollars into a *price today*, you have to discount them, because a dollar in the future is worth less than a dollar in your hand right now — the dollar in your hand could be put in a safe bond and grow.

The rate you discount by is called the **discount rate**, and the single most important fact in this post is that it lives in the *denominator* of every valuation. The present value (PV) of a cash flow `CF` arriving `t` years from now, discounted at rate `r`, is:

```
PV = CF / (1 + r)^t
```

and the value of any whole asset is the sum of the present values of all the cash flows it will ever pay:

```
Value = CF1/(1+r)^1 + CF2/(1+r)^2 + CF3/(1+r)^3 + ...
```

Because `r` sits in the denominator, raised to the power of time, raising `r` shrinks every single term — and the far-future terms shrink fastest, because the haircut compounds every year. A cash flow 25 years out, discounted at 2%, is worth about 61 cents on the dollar; bump the rate to 5% and it is worth about 30 cents. *Half its value gone* from a three-percentage-point move in a number set by a committee in Washington. This series' companion post [the discount-rate channel](/blog/trading/policy-and-markets/the-discount-rate-channel-how-rates-reprice-cash-flows) derives this mechanism in full; what matters here is that the Fed's policy rate anchors the discount rate, so when the Fed moves, the denominator under *everything* moves.

One refinement is worth making now because it explains a puzzle later in the post. The discount rate has two parts: a **real rate** (the return after stripping out inflation) and an **inflation expectation** added on top. The *nominal* rate you actually see — the 4.3% on a 10-year Treasury — is roughly the real rate plus expected inflation. In 2022 *both* parts rose at once: real rates climbed from deeply *negative* (about −1% in 2021, because the Fed had pinned nominal rates below inflation) toward strongly positive, *and* inflation expectations spiked before the hikes brought them back down. The jump in the *real* rate is the part that did the most damage to asset values, because real rates are what compete with the real returns on stocks, gold, and property. When the Fed dragged the real 10-year yield from −1% to +2% — a three-point swing in the *real* discount rate — it reset the bar that every risky asset had to clear, and most of them failed to clear it. The macro-correlations post [real yields and the cleanest macro correlation](/blog/trading/macro-correlations/real-yields-and-the-cleanest-macro-correlation) shows just how tightly asset prices tracked that real-rate move.

### The 60/40 portfolio: the cushion that built the industry

The "60/40" is the most famous recipe in personal finance: put 60% of your money in stocks (for growth) and 40% in bonds (for safety), rebalance once a year, and ride. For roughly four decades it was close to magic. From the early 1980s — when Volcker's rate peak marked the all-time high in yields — through 2021, the 60/40 returned something like 9% a year with drawdowns far shallower than stocks alone. It became the default setting of pension funds, target-date retirement funds, and robo-advisors. Trillions of dollars sit in some version of it.

The magic was never the bonds' return on its own; it was the *cushion*. The whole point of the 40% in bonds was that when stocks crashed, bonds would rise and soften the blow. The two legs were *negatively correlated* — they moved in opposite directions — so blending them gave you a smoother total than either alone.

#### Worked example: the cushion in a normal recession

Take a \$100,000 60/40 portfolio: \$60,000 in stocks, \$40,000 in bonds. Now imagine a textbook recession — say 2008 or the early-2020 COVID crash. Stocks fall hard, but the Fed responds by *cutting* rates toward zero to support the economy, which pushes bond prices *up*.

- Stock leg: \$60,000 falls 30% → loses \$18,000 → worth \$42,000.
- Bond leg: \$40,000 rises 8% (rates fell, bond prices rose) → gains \$3,200 → worth \$43,200.
- Total: \$42,000 + \$43,200 = \$85,200, a loss of **\$14,800, or about −14.8%**.

The stock leg alone lost 30%; the blended portfolio lost under 15%. The bond leg's \$3,200 gain offset part of the stock leg's \$18,000 loss. *That* is the cushion — and notice it only works because the Fed *cut* rates, lifting bonds while stocks fell. The intuition to hold: in a recession-driven sell-off, the bond cushion inflates exactly when you need it, because falling rates are good for bonds.

### Why the cushion normally works — and the hidden assumption

Walk through *why* bonds rose in that example and you find the load-bearing assumption underneath the entire 60/40 edifice. Stocks fell because of a recession. A recession means the Fed *cuts* rates to stimulate the economy. Cutting rates means a *falling* discount rate. A falling discount rate is *good* for bond prices (smaller denominator, bigger present value). So the bond leg rose because the same event that hurt stocks — economic weakness — triggered the policy response (rate cuts) that helps bonds.

The hidden assumption, then, is this: **stocks fall because of recessions, and recessions bring rate cuts.** As long as that holds, stocks and bonds are negatively correlated and the cushion works. But it contains a trapdoor. What if stocks fall not because of a recession, but because of *rising rates themselves*? Then the policy response is the *opposite* — the Fed is *hiking*, not cutting — and a rising discount rate is *bad* for bonds too. Both legs fall together. The cushion becomes a second source of loss. That trapdoor is exactly what opened in 2022.

## The lever: from "transitory" to 525 basis points

To understand the violence of the 2022 repricing, you have to understand how *late* the Fed was, because the lateness is what forced the speed.

When the economy reopened in 2021, prices began to climb for reasons that genuinely *looked* temporary: semiconductor shortages, snarled ports, a spike in used-car prices, and the arithmetic "base effect" of comparing 2021 prices against the deflationary lows of spring 2020. The Fed, scarred by a decade of *under*-shooting its 2% inflation target in the 2010s, chose to look through it. Through most of 2021 the policy rate stayed at the zero lower bound and the Fed kept buying \$120 billion of bonds a month — active stimulus — even as headline inflation pushed past 5%, then 6%.

![Line chart of CPI inflation rising from near zero to a 9.1 percent peak in June 2022](/imgs/blogs/2021-23-the-fastest-hiking-cycle-and-the-60-40-wreck-3.png)

The chart above is the disease the Fed was diagnosing. Consumer-price inflation, year-over-year, ran near the 2% target for most of 2020. Then it took off: 5% by mid-2021, where the "transitory" framing held; 7% by January 2022; and a peak of **9.1% in June 2022**, the highest in four decades. By the time the Fed acted, inflation was not a one-time price-level adjustment — it had broadened into services, rents, and wages, the stickier categories that signal it has become *embedded*. The companion macro-trading post [2021-2023: inflation and the fastest hiking cycle](/blog/trading/macro-trading/2021-2023-inflation-and-the-fastest-hiking-cycle) walks the inflation data in detail; here the point is simpler: the Fed fell behind, and falling behind is what dictated the brutal catch-up.

The catch-up began in March 2022 with a cautious 25 basis points. Then, as inflation kept rising, the Fed abandoned caution. It hiked 50bp in May, then *four consecutive 75bp hikes* (June, July, September, November) — a pace not seen since 1994 — then 50bp and a string of 25bp moves, until the upper bound of the target range reached **5.50% in July 2023**. Eleven hikes, 525 basis points, sixteen months.

![Step chart of the Fed funds rate rising from 0.25 percent to 5.50 percent between 2022 and 2023](/imgs/blogs/2021-23-the-fastest-hiking-cycle-and-the-60-40-wreck-2.png)

The step chart shows the path. Each riser is a hike; the shaded band is the hiking window. The defining feature is the *slope*: from the floor to 5.50% in sixteen months. For comparison, the previous tightening cycle (2015-2018) took *three years* to climb from 0.25% to 2.50% — a tenth of the speed. The 2022 cycle earned its nickname, "the fastest hiking cycle since Volcker," not because the *level* was extreme (Volcker's 19% dwarfs it; see this series' [Volcker case study](/blog/trading/policy-and-markets/volcker-1979-82-how-20-percent-rates-repriced-everything)) but because the *speed* gave markets and balance sheets no time to adjust. A slow hike lets institutions roll into higher rates gradually; a fast one repriced everything before anyone could reposition.

### The second lever: quantitative tightening

The rate hikes were the headline, but they did not act alone. Running alongside them was a second, quieter lever: **quantitative tightening (QT)**, the reversal of the bond-buying the Fed had done during the pandemic. To understand QT you first have to understand its mirror image, **quantitative easing (QE)**. In 2020-2021 the Fed created new money and used it to *buy* trillions of dollars of Treasuries and mortgage bonds, swelling its balance sheet from about \$4.2 trillion to a peak of roughly \$9.0 trillion. That buying did two things: it pushed bond prices up (yields down), and it pumped cash — liquidity — into the financial system, where it found its way into every risk asset and inflated their prices. This series' [liquidity channel](/blog/trading/policy-and-markets/the-liquidity-channel-qe-qt-and-the-everything-bid) post builds the mechanism in full; the macro-trading companion [QE vs QT](/blog/trading/macro-trading/qe-vs-qt-how-balance-sheet-policy-moves-markets) walks the trading side.

Starting in mid-2022, the Fed put that engine in reverse. Instead of buying bonds, it let up to \$95 billion of them *roll off* its balance sheet each month — as bonds matured, it simply did not replace them, draining cash out of the system rather than adding it. Over the cycle the balance sheet shrank from \$9.0 trillion toward \$6.5 trillion. QT is to liquidity what rate hikes are to the discount rate: a second tightening force, working on the *quantity* of money rather than its *price*. For our story, QT matters because it reinforced the rate move — it pushed bond yields *higher* (the Fed was no longer a buyer, so the market had to absorb more supply at lower prices) and pulled liquidity *out* of risk assets, deepening the equity de-rating. The 2022 selloff was the discount-rate channel and the liquidity channel pulling in the *same direction* at once, which is part of why it was so unrelenting.

There is a subtle but important asymmetry here. QE in 2020-2021 lifted *all* assets together — the "everything rally," in which stocks, bonds, real estate, gold, and crypto all rose on the tide of liquidity and low rates. QT plus hikes in 2022 sank *all* assets together — the "everything selloff." The symmetry is the whole lesson of this series in miniature: when policy moves the common factors (the discount rate and the liquidity tide) hard enough, the *correlation structure* of markets changes. Assets that normally diversify each other start moving as one, in whichever direction policy is pushing. The 60/40 wreck is just the most famous casualty of that correlation flip.

## The transmission: how the rate move hit both legs

Now we follow the lever into the two legs of the 60/40. The unifying claim, stated before the details bury it: **because the same discount rate sits in the denominator of both stock and bond valuations, driving it from 1.5% to 4.3% repriced both *down*, at the same time.** Take them in order of how directly the channel hits: bonds first (they *are* the discount rate), then equities.

### The bond leg: duration is just discounting in reverse

A bond is the purest expression of the discount-rate channel, because a bond *is* a stream of fixed cash flows — set coupons plus the return of principal — and its price is nothing but the present value of that stream. Crucially, the coupons are *fixed in dollars* the day the bond is issued. So when the market's discount rate rises, the bond's cash flows do not change at all; only the rate you divide them by changes. Higher rate → bigger denominator → lower price. The relationship is mechanical and inverse: **bond prices move opposite to yields, always.**

The longer the bond, the more violent the move, because a long bond's value is concentrated in cash flows that sit far in the future — exactly the cash flows whose discount-factor haircut compounds the most. This sensitivity has a name: **duration**, roughly the percentage a bond's price falls for each one-percentage-point rise in its yield. A 2-year note has a duration near 2; a 10-year bond, a duration around 8 or 9; a 30-year bond, well over 15. When yields jump, the long end is devastated and the short end merely bruised.

![Line chart of the 10-year Treasury yield rising from 0.6 percent in 2020 to a 4.34 percent high in 2022](/imgs/blogs/2021-23-the-fastest-hiking-cycle-and-the-60-40-wreck-8.png)

The discount rate behind every asset is plotted above: the 10-year Treasury yield. It bottomed at **0.62% in July 2020** — the lowest in the history of the United States — as the COVID Fed flooded the system. By the end of 2021 it had crept back to about 1.5%. Then the hiking cycle took it to a 2022 intraday high of **4.34%**. That is a near-tripling of the discount rate in under two years. Hold that move — 1.5% to 4.3% — because it is the input to everything that follows.

#### Worked example: the duration loss on a 10-year bond

Suppose at the end of 2021 you bought a brand-new 10-year Treasury note with a 1.5% coupon at par — \$100 of face value for \$100, the going rate at the time. Each year it pays you \$1.50, and at year 10 it returns your \$100. Now run the discount rate from 1.5% to 4.3% and re-price that *same* bond — the cash flows are unchanged, only the discount rate moves.

Discounting ten annual \$1.50 coupons plus the \$100 principal at 4.3% instead of 1.5%:

- At a 1.5% yield: price = \$100.00 (it was issued at par).
- At a 4.3% yield: price = **\$77.62**.

You have lost **\$22.38 per \$100, about −22.4%** — on a *U.S. Treasury*, the asset everyone calls "risk-free." Your annual income did not change; the rate the market discounts it at did, and that alone vaporized nearly a quarter of your principal. The intuition: a bond's coupons are frozen, so when the discount rate jumps, the *only* thing that can give is the price — and on a 10-year bond, a 2.8-point rate move costs you over a fifth of your money.

![Curve of bond price falling as yield rises, with the issue point at 1.5 percent and the 2022 point at 4.3 percent](/imgs/blogs/2021-23-the-fastest-hiking-cycle-and-the-60-40-wreck-6.png)

The price-yield curve above plots that worked example. The green dot is where you bought (yield 1.5%, price \$100); the red dot is where the market repriced you in 2022 (yield 4.3%, price \$78). The dotted red line *is* the loss. Note the curve's downward bow — that is **convexity**, the second-order effect that makes the price fall a little *less* than a straight-line duration estimate would predict, and which becomes a *gift* on the way back down when yields fall. But in 2022, on the way up, all the investor felt was the fall. And remember: a 10-year was the *gentle* case. The Bloomberg US Aggregate bond index — the "Agg," the benchmark the bond leg of a 60/40 tracks — has an average duration around 6, and it lost **13% in 2022**, its worst year in the index's history, which began in 1976. A 30-year Treasury bought at the 2020 low lost closer to *half* its value.

### The equity leg: the multiple is one over the discount rate

Now the stock leg. Stocks feel less obviously rate-driven than bonds — a company's profits are not fixed coupons; they grow, and a good year can offset a rate move. But the discount-rate channel reaches equities just as surely, through the **valuation multiple** — the price-to-earnings (P/E) ratio, how many dollars investors will pay for one dollar of annual earnings.

Here is the link, stripped to its core. A simple model values a stock as next year's earnings divided by (the discount rate minus the growth rate): `Value = E / (r − g)`. The P/E multiple is therefore `1 / (r − g)`. When the discount rate `r` rises, the denominator `(r − g)` gets bigger, so the multiple — what investors will pay per dollar of earnings — shrinks. *Even if earnings never change*, the price falls, because each dollar of future profit is now discounted harder. And the effect is most violent for "long-duration" stocks: high-growth companies whose profits sit mostly far in the future (think a young tech firm earning little today but promising fortunes in a decade). Their value is concentrated in distant cash flows, exactly the ones the rising discount rate haircuts the most. This series' companion post [how policy prices equities](/blog/trading/policy-and-markets/how-policy-prices-equities-the-multiple-and-the-earnings) builds this from scratch.

#### Worked example: the multiple compression on the S&P 500

The S&P 500 entered 2022 trading around 21 times forward earnings — a rich multiple, inflated by years of near-zero rates that made even modest future profits worth a lot in present value. Then the discount rate roughly tripled. Watch what happens to a stock's price when *only* the multiple moves, holding earnings flat:

- Start: earnings of \$220 per index "share," a 21× multiple → price = 220 × 21 = **\$4,620**.
- After the rate shock, the multiple compresses to about 17× (the de-rating of 2022) → price = 220 × 17 = **\$3,740**.
- That is a fall of **\$880, about −19%**, with earnings *completely unchanged*.

The entire decline came from investors paying less per dollar of the same earnings, because the discount rate behind those earnings rose. In reality the S&P 500 fell **19.4% in 2022** — and almost all of that was multiple compression, not falling profits (S&P earnings were roughly flat to slightly up that year). The intuition: a stock's price is earnings times a multiple, and the multiple is just one-over-the-discount-rate — so when the Fed triples the discount rate, the multiple shrinks and the price falls even if the business is fine.

There is a second, reinforcing reason stocks fell, and it sharpens the point. Through the 2010s and into 2021, investors had a slogan: **TINA** — "there is no alternative." With cash and bonds yielding almost nothing, the only place to earn a return was stocks, so money piled into equities almost regardless of price, inflating multiples. The hiking cycle killed TINA. By 2023 you could earn 5% on a Treasury bill or a money-market fund — risk-free, in cash — for the first time in fifteen years. Suddenly there *was* an alternative. The "equity risk premium" — the extra return investors demand for holding risky stocks over safe bonds — got squeezed, because the safe leg now paid a real return. Money that had been *forced* into stocks by zero rates was *pulled back* toward cash and bonds by 5% rates. That flow compounded the mechanical multiple compression: stocks fell both because the discount rate in the denominator rose *and* because the rising risk-free rate gave capital somewhere safer to go.

#### Worked example: the de-rating of a long-duration growth stock

Make the duration point concrete with two stocks that earn the *same* total profit but on different timelines. Stock V (value) earns \$10 a year, steadily, starting now. Stock G (growth) earns almost nothing today but is expected to earn \$30 a year a decade from now. At a 1.5% discount rate, the far-future profits of Stock G are barely haircut, so it commands a sky-high multiple. Now triple the discount rate to 4.3% and watch the gap:

- Stock V's value rests on *near-term* cash flows. Discounting \$10/yr harder costs it something, but its cash flows arrive soon, so the haircut is modest — say its fair value falls about 12%.
- Stock G's value rests on cash flows a *decade out*. A dollar ten years away, discounted at 4.3% instead of 1.5%, is worth `(1.015/1.043)^10 ≈ 0.76` of its former value — a **24% haircut on each distant dollar**, before any change in the business.

That is exactly the pattern 2022 produced: the value-heavy Dow fell about 9%, the broad S&P fell 19.4%, and the growth-heavy Nasdaq fell about 33% — *the same rate move, scaled by duration*. The intuition: a growth stock is a long-dated bond in disguise, and a rising discount rate punishes long duration whether it wears a coupon or a P/E.

Notice the symmetry now. The bond leg fell because a rising discount rate shrank the present value of *fixed coupons*. The stock leg fell because the *same* rising discount rate shrank the *multiple* on earnings. **Same variable, same direction, both legs.** The cushion was gone — not because anything went wrong with the bond market specifically, but because the one input both legs share moved against both of them simultaneously.

## The 60/40 wreck, in dollars

We now have everything we need to watch the most trusted portfolio in finance come apart. The key statistic is the *correlation* between stocks and bonds. For most of the four decades from 1982 to 2021, that correlation was *negative* — bonds zigged when stocks zagged, which is the entire source of the cushion. In 2022 it flipped *positive*: stocks and bonds fell *together*, month after month, because the shared driver (the rising discount rate) overwhelmed everything else.

![Grouped bar chart showing the S&P 500 down 19.4 percent, the Agg bond index down 13 percent, and the 60/40 blend down 16 percent in 2022](/imgs/blogs/2021-23-the-fastest-hiking-cycle-and-the-60-40-wreck-4.png)

The three bars above are the wreck. The stock leg fell 19.4%. The bond leg — the part that was *supposed to be safe*, the cushion — fell 13%. And the 60/40 blend, getting hit on both legs at once, fell about 16%. There was nowhere to hide inside the portfolio: the very asset you held *for protection* was a second source of loss. This was, by most measures, the worst year for a balanced 60/40 portfolio since the 1930s.

To see why this was such a shock to the people who run trillions of dollars, you have to understand that the negative stock-bond correlation they relied on was itself a *historical accident* of the era they grew up in. Across the four decades from roughly 1982 to 2021, the dominant macro risk was *deflation and recession* — the worry was always that growth would slow, so bad news for stocks (weak growth) was good news for bonds (rate cuts coming), and the correlation sat reliably negative. But that was not always true. In the 1970s — the *last* time inflation was the dominant risk — stocks and bonds were *positively* correlated, exactly as they were in 2022, because the shared driver back then was also inflation and the policy response to it. The negative correlation that made the 60/40 look like free diversification was a feature of the *disinflationary regime*, not a permanent law. When the regime flipped back to inflation in 2022, the correlation flipped with it, and a generation of investors discovered that the cushion they had treated as a constant was actually a *regime-dependent variable*. The companion macro-correlations work makes this its central thesis: correlation is a regime, not a constant.

#### Worked example: the 60/40 wreck on a \$100k portfolio

Run the same \$100,000 portfolio from the Foundations section — \$60,000 stocks, \$40,000 bonds — but through 2022 instead of a normal recession:

- Stock leg: \$60,000 falls 19.4% → loses \$11,640 → worth \$48,360.
- Bond leg: \$40,000 falls 13.0% → loses \$5,200 → worth \$34,800.
- Total: \$48,360 + \$34,800 = \$83,160, a loss of **\$16,840, or about −16.8%**.

Now compare this to the "normal recession" worked example earlier, where the same portfolio lost just 14.8% *despite* stocks falling a much larger 30%. In that case the bond leg *gained* \$3,200 and cushioned the blow. Here the bond leg *lost* \$5,200 and made the blow *worse*. The difference between the two scenarios is entirely about *why* stocks fell — recession (cushion works) versus rising rates (cushion fails). The intuition that should now feel inevitable: the 60/40 isn't diversified against a discount-rate shock at all, because the discount rate is the one variable both of its legs depend on.

That is the lesson in one line. Diversification protects you against *idiosyncratic* risk — the risk specific to one asset, one company, one sector. It does *not* protect you against a *common factor* that prices every asset, and the discount rate is the most common factor there is. When the Fed moves it hard, the correlation between assets that normally diversify each other snaps toward +1, and a portfolio that looked diversified turns out to have been a single concentrated bet on low rates all along.

## The tail risk: how the rate move buried a bank

A 16% loss on a balanced portfolio is painful but survivable. The same rate move did something far more dangerous on the balance sheet of a bank that had made the identical mistake — holding long bonds bought at low yields — but with *borrowed* money and *no* loss-absorbing cushion. That bank was Silicon Valley Bank, and in March 2023 it became the second-largest bank failure in U.S. history, undone in 48 hours by the exact discount-rate arithmetic we just walked through.

### The setup: a deposit flood meets a bond binge

During the 2020-2021 tech boom, Silicon Valley Bank — the banker to a huge share of venture-backed startups — was buried in cash. Its clients, flush with venture funding raised in the cheap-money era, parked enormous deposits at the bank. SVB's deposits exploded from around \$60 billion at the end of 2019 to about \$189 billion by early 2022 — more than tripling in two years.

A bank has to *do something* with deposits — it earns money by investing them at a higher rate than it pays depositors. With loan demand from its startup clients limited, SVB poured roughly \$120 billion of that deposit flood into bonds: long-dated Treasuries and mortgage-backed securities, bought in 2020-2021 at yields around 1.5%. Critically, it classified a large chunk of these as **"held-to-maturity" (HTM)** — an accounting category that lets a bank carry the bonds at their original purchase price on its books and *not* mark them down as market yields rise, on the promise that it intends to hold them to maturity and collect par. On paper, the losses were invisible.

![Graph showing the SVB doom loop from deposit flood to long bonds to the rate shock to the deposit run to the equity wipeout](/imgs/blogs/2021-23-the-fastest-hiking-cycle-and-the-60-40-wreck-7.png)

The figure above traces the doom loop. A deposit flood (\$60bn → \$189bn) went into long bonds (~\$120bn at ~1.5%) classified as held-to-maturity, so the losses never showed on the books. Then the Fed hiked 525bp and yields tripled, which — by exactly the price-yield arithmetic of the worked example above — buried that bond book in unrealized losses. When depositors fled, SVB was forced to *sell* those bonds, which turned the invisible paper loss into a real, realized loss large enough to wipe out the bank's equity. Read the diagram left to right and you are reading the same discount-rate channel that wrecked the 60/40 — only here it killed an institution.

### The mark-to-market loss the accounting hid

The held-to-maturity label hid the loss; it did not make it disappear. The bonds were genuinely worth far less than the price SVB carried them at, because — as we proved with the price-yield curve — a bond bought at 1.5% is worth dramatically less once the market yield is 4-5%.

#### Worked example: a \$10bn HTM book's hidden mark-to-market loss

Take a \$10 billion slice of SVB's held-to-maturity book: 10-year bonds bought at a 1.5% yield. SVB carries them on its books at \$10 billion (the HTM accounting trick). But the *market* now yields about 4.3%, so what are they actually worth? Use the exact same per-\$100 numbers from the duration worked example, where a 10-year 1.5% bond reprices from \$100 to \$77.62:

- Book value (carried at purchase): \$10.0 billion.
- True market value: \$10.0 billion × (77.62 / 100) = **\$7.76 billion**.
- Hidden unrealized loss: \$10.0 billion − \$7.76 billion = **\$2.24 billion** — on this slice alone.

Scale that across SVB's roughly \$120 billion securities book (much of it longer-dated mortgage bonds, which behave even worse), and the firm was sitting on something like **\$15 billion of unrealized losses** by late 2022. That number matters enormously because of what sat next to it: SVB's total equity — the capital that absorbs losses before depositors are at risk — was only about **\$16 billion**. The hidden bond loss was nearly the *entire* equity cushion of the bank. The intuition: an HTM book is a time bomb when rates rise, because the accounting lets you ignore a loss that is economically real, right up until the moment you are forced to sell.

### The run: forcing the loss into the open

A held-to-maturity loss is harmless *if* you can actually hold to maturity. The fatal flaw is that you can only hold the bonds if your depositors let you — and a bank's depositors can leave at any time, while its bonds are locked up for years. This is the **maturity mismatch** at the heart of all banking, and the rate shock made SVB's version lethal.

By early 2023, SVB's depositors — the same venture-backed startups — were *burning* cash (the funding boom had ended, partly *because* of the same rate hikes) and pulling deposits to fund operations. To meet those withdrawals, SVB had to sell assets. On March 8, 2023, it announced it had sold \$21 billion of bonds at a *realized* loss of \$1.8 billion and needed to raise \$2.25 billion of fresh capital to plug the hole. That announcement did exactly the wrong thing: it told the world the losses were real and the bank was short of capital. SVB's depositors — sophisticated, concentrated, mostly *uninsured* (above the \$250,000 FDIC limit), and all connected through the same venture-capital networks — did the rational thing and ran. On March 9, customers tried to withdraw **\$42 billion in a single day** — about a quarter of all deposits. The bank could not raise the capital, could not sell bonds fast enough without crystallizing catastrophic losses, and on March 10, 2023, regulators seized it.

#### Worked example: how the forced sale wiped the equity

Put the pieces together to see why the run was fatal. SVB had roughly \$16 billion of equity and roughly \$15 billion of *unrealized* losses sitting silently in its HTM bond book. As long as nobody forced a sale, the \$16 billion of equity looked intact and the \$15 billion loss stayed invisible.

- Stated equity (book): ~\$16 billion.
- Hidden bond loss if marked to market: ~\$15 billion.
- *Economic* equity (book equity minus the hidden loss): \$16bn − \$15bn ≈ **\$1 billion**.

The bank was, on a true mark-to-market basis, already a hair from insolvent — its real equity cushion was almost entirely consumed by the bond losses the rate move had created. The deposit run did not *cause* the insolvency; it *revealed* it, by forcing the sales that turned the hidden \$15 billion loss into a realized one that wiped out the \$16 billion of equity. The intuition: the rate shock had already destroyed SVB's capital months earlier — the accounting just hadn't admitted it yet, and the run was the moment the truth got marked to market.

### The policy response: the BTFP backstop

The story has a coda that is itself a policy lever. To stop the panic from spreading to every other bank holding underwater bonds — and that was *most* banks, because the rate shock hit all of them — the Fed created an emergency facility two days later: the **Bank Term Funding Program (BTFP)**. Its trick was elegant. It let banks borrow against their underwater bonds *at the bonds' par value* — pretending, for collateral purposes, that the bonds were worth what the bank paid, not their crushed market price. That gave any bank facing a run a way to raise cash without being forced to *sell* its bonds at a loss, breaking the doom loop at the "forced sale" step of the diagram. It worked: the panic stopped. The deeper lesson — that the same rate move which reprices a portfolio can threaten the *plumbing* of the financial system, and that regulators then reach for new tools — is the subject of this series' [macroprudential and regulatory policy](/blog/trading/policy-and-markets/macroprudential-and-regulatory-policy) post.

### Why it was not just SVB

It is tempting to file SVB under "one badly-run bank," but the rate shock had loaded the same gun across the whole regional-banking sector, and SVB was simply the first to fire. The same weekend SVB failed, **Signature Bank** was seized — a New York bank with a similar profile of concentrated, uninsured, flighty deposits (in its case, crypto-linked) and its own duration mismatch. Seven weeks later, in May 2023, **First Republic Bank** failed and was sold to JPMorgan in the largest U.S. bank failure since 2008 — again the same story: a deposit base of wealthy, uninsured clients who could move billions with a few taps, and a balance sheet of low-yielding long assets (in First Republic's case, jumbo mortgages written at 2-3%) that the rate move had rendered worth far less than book. Three banks, one disease.

What links all three is that the rate shock created two simultaneous wounds that fed each other. The *asset* side was bleeding (long bonds and mortgages worth less as yields rose), and the *liability* side was running (depositors leaving for the 5% they could now earn in a money-market fund instead of the ~0% the bank paid). Rising rates *caused* both: they crushed the assets *and* they gave depositors a reason to leave. A bank can survive one wound; the doom loop is when the two combine, because the deposit flight forces asset sales that realize the losses that scare more depositors. The thing to internalize is that this was not a *credit* crisis like 2008 — no one defaulted, the bonds were money-good Treasuries and agency mortgages. It was an *interest-rate* crisis, the banking-system version of the exact 60/40 wreck, with leverage and a maturity mismatch turning a price loss into a solvency event.

There is a final irony worth naming. The deposit flight that killed these banks was, in part, *caused by the same hikes that crushed their assets*. As money-market funds began paying 4-5%, retail and corporate depositors woke up to the fact that leaving cash in a checking account paying nothing was leaving real money on the table, and they swept hundreds of billions out of bank deposits into money funds. So the rate move was a pincer: it hit the banks' assets and their funding from both sides at once, with the *same* lever. That is the discount-rate channel operating not on a portfolio but on the architecture of the banking system itself.

## Common misconceptions

**"Bonds are safe, so the bond leg can't lose much."** Bonds are safe from *default* risk (a Treasury will pay you back) but not from *price* risk. A 10-year Treasury lost 22% in our worked example, and a 30-year lost about half its value off the 2020 low, purely from the rate move — no default required. "Safe" means the cash flows are certain; it does not mean the price is stable when the discount rate triples. Duration risk is real risk.

**"The 60/40 is diversified, so it's protected against any crash."** Diversification protects against *idiosyncratic* risk — one company, one sector failing. It does *not* protect against a *common factor* that prices every asset at once, and the discount rate is the ultimate common factor. In 2022 the stock-bond correlation flipped from its usual negative to strongly positive, and the "diversified" portfolio behaved like one big concentrated bet on rates staying low. Diversification is a regime, not a guarantee.

**"SVB failed because of crypto / a niche tech problem."** SVB failed because of plain-vanilla *interest-rate risk* — it bought long bonds with short-term deposit money and the discount rate moved against it. The very same arithmetic that cost a 60/40 investor 16% cost SVB its solvency, only with leverage and a maturity mismatch. There was nothing exotic about it; it was a textbook bank run triggered by a textbook duration loss.

**"The Fed could have avoided all this by hiking sooner."** Hiking sooner would have meant a *slower, smaller* cycle and less violence — but the inflation that justified the hikes had a real cause (a supply-and-demand shock from the pandemic and the war in Ukraine), and waiting for "transitory" to prove itself is what forced the catch-up to be so fast. The lesson isn't "the hikes were a mistake"; it's that *falling behind* converts a gentle tightening into a fast one, and fast moves break things that slow ones don't.

**"A high held-to-maturity bond loss is just paper — it doesn't matter unless you sell."** This is precisely the belief that killed SVB. An unrealized loss is harmless *only if* you can hold to maturity, which requires that your funding (deposits) stays put. The moment depositors run, "held-to-maturity" becomes "sold-at-a-loss," and the paper loss becomes a real one. Paper losses on a leveraged, deposit-funded balance sheet are solvency risk in disguise. The accounting label says nothing about the economics; it only controls *when* the loss is forced into the open, and a run controls the timing far more ruthlessly than any treasurer's intention to hold.

## Case studies: the same shock across every asset

We have walked the two legs of the 60/40 and the bank failure in the tail. Step back and look across *every* asset class in 2022-23 and the unity of the episode becomes clear: one variable, the discount rate, moved everything in the same direction.

**Stocks: multiple compression, led by the longest-duration names.** The S&P 500 fell 19.4% in 2022, almost entirely from the P/E de-rating rather than falling earnings. Within the index, the pattern was textbook discount-rate behavior: the longest-duration stocks fell hardest. The Nasdaq, dominated by high-growth tech whose value sits in distant cash flows, fell about 33% — far worse than the broad market — because rising rates haircut far-future profits the most. Unprofitable, story-driven growth stocks (the most extreme "long-duration" equities of all) fell 60-80%. The *same* mechanism, scaled by duration: the further out an asset's cash flows, the harder a rising discount rate hits it.

**Bonds: the worst year on record.** The Bloomberg US Aggregate fell 13% in 2022, its worst calendar year since the index began in 1976. Long Treasuries (20+ year) fell roughly 30%. This was not a credit event — issuers paid every coupon — it was a pure discount-rate repricing, the bond leg of the wreck.

**Real estate: cap rates re-rated and the lock-in effect.** Property is valued by a "cap rate" — net rental income divided by price — which behaves like a bond yield: when the risk-free rate rises, cap rates rise and property values fall. Commercial real estate, especially offices, repriced down 25-40%. On the residential side, the 30-year mortgage rate rocketed from sub-3% in 2021 to about 7.8% by late 2023, freezing the housing market into the "lock-in effect" (nobody with a 3% mortgage will sell to take a 7% one). This series' [how policy prices real estate](/blog/trading/policy-and-markets/how-policy-prices-real-estate-cap-rates-and-mortgages) post details the cap-rate channel.

**Gold and crypto: the liquidity-sensitive assets.** Gold, which pays no coupon, competes with bonds; when real yields rose sharply in 2022, gold's appeal dimmed and it ended the year roughly flat-to-down despite the inflation scare — a reminder that for gold, the *real rate* matters more than the inflation rate. Bitcoin, the purest liquidity-regime asset, collapsed from about \$69,000 (November 2021) to about \$16,000 by late 2022 — a 77% drawdown — as the liquidity tide that had lifted it went out. Both moved the same direction as everything else: down, with the rising discount rate.

**The cross-asset verdict.** This is the deepest point of the case study. In a *normal* year, these assets diversify each other — stocks, bonds, gold, real estate, and crypto respond to different things, so a basket of them is smoother than any one. In 2022 they all fell together, because they were all being priced off the *same* tripling discount rate. When the policy lever moves the common factor hard enough, the diversification *across* asset classes fails for exactly the same reason it failed *within* the 60/40: correlations go to one. The companion macro-correlations post [the Fed funds path and front-end correlation](/blog/trading/macro-correlations/the-fed-funds-path-and-front-end-correlation) quantifies how tightly the front end of the curve tracked the policy path that year.

## What it means for asset values: the playbook

So what is the actual, durable takeaway — the thing to *do* with this case study? Boil it to a set of rules about how policy prices assets, what signal to watch, and what would invalidate the read.

**The repricing rule.** When the central bank moves the discount rate fast and far, *every* long-duration asset reprices down together, in rough proportion to its duration. Long bonds and high-growth stocks (both "long-duration") fall hardest; short bonds and value stocks (both "short-duration") fall least; cash, whose duration is zero, *wins* — for the first time in fifteen years, holding cash at 5% was a real, positive-real-yield choice in 2023. The direction and the *ranking* are predictable; the magnitude scales with how far the discount rate moves.

**The 60/40 caveat.** The stock-bond cushion works against *recession* shocks (Fed cuts, bonds rally) and fails against *inflation/rate* shocks (Fed hikes, bonds fall with stocks). The diagnostic question is always: *why* are stocks falling? If the answer is "the economy is weakening," your bonds will protect you. If the answer is "rates are rising," they won't — and you want shorter duration, more cash, and real assets instead. After 2022, the textbook response has been to add diversifiers that *don't* depend on the discount rate (trend-following, commodities, cash) rather than to abandon bonds entirely.

**The bank-balance-sheet tell.** The SVB episode generalizes to a signal you can watch: when rates rise fast, look for institutions holding *long* assets funded by *short, flighty* liabilities and carrying large *unrealized* (held-to-maturity) bond losses relative to their equity. That combination — long assets, short funding, hidden losses near the size of the capital cushion — is the doom-loop setup, and it is visible in bank filings *before* the run. The run is the trigger; the rate shock is the loaded gun.

**The signal to watch.** The single number that drives all of this is the 10-year Treasury yield (the discount rate). Watch its *level* and especially its *speed*: a fast 150bp+ move in a few months is the regime where correlations go to one and the 60/40 cushion fails. The Fed funds path sets the floor under it; the companion post [how policy sets the bond market](/blog/trading/policy-and-markets/how-policy-sets-the-bond-market-the-yield-curve) maps the whole curve.

**What would invalidate the read.** The "stocks and bonds fall together" regime is specific to *inflation-driven* tightening. The moment the dominant risk flips from inflation to recession — the moment the market starts pricing *cuts* instead of *hikes* — the stock-bond correlation should flip back negative and the bond cushion should return. That is roughly what began in late 2023 and 2024, as inflation cooled toward target and the Fed pivoted to cuts (three cuts in 2025, taking the policy rate from 4.50% to 3.75%): bonds rallied, the cushion partially reappeared, and the worst of the wreck was behind. The regime, in other words, is not permanent — it is a property of *why* policy is moving, which is the whole thesis of this series.

### The aftermath: what the cycle proved

It is worth closing the case study by looking at what happened *after* the wreck, because the recovery is as instructive as the crash. By the end of 2023, inflation had fallen from 9.1% to about 3.4%, and the market began to believe the next move was a *cut*, not a hike. The moment that belief took hold, the machinery ran in reverse: the 10-year yield, which had peaked at 4.34%, fell back; bond prices rose (the duration that hurt on the way up *helped* on the way down, with the convexity bonus from the price-yield curve); and stock multiples re-expanded as the discount rate eased. A 60/40 investor who had simply *held* through 2022 recovered much of the loss over the following two years — the wreck was a repricing, not a permanent impairment, because the underlying cash flows (coupons, earnings, rents) never actually stopped.

That recovery teaches the deepest lesson of the episode. The 2022 losses were not caused by anything *breaking* in the real economy — companies kept earning, bonds kept paying, rents kept coming in. The losses were caused entirely by the *discount rate* changing, and a discount-rate loss is *reversible* when the rate comes back down, in a way that a credit loss (a default, a bankruptcy) is not. This is why the cycle was so different from 2008: 2008 was a *solvency* crisis (assets were genuinely impaired by bad loans), while 2022 was a *valuation* crisis (good assets repriced by a higher discount rate). The investor's takeaway is precise and worth holding: a discount-rate drawdown is a price you pay *up front* in exchange for a higher *future* return — every asset you hold now yields more than it did in 2021 — whereas a credit drawdown is money simply gone. The 60/40 wreck of 2022 was the first kind, which is exactly why patience, not panic, was the correct response.

The 2021-23 cycle is the modern textbook for one idea: a central bank holds a single dial that prices everything, and when it turns that dial hard, the diversification you thought you had evaporates because it was all, secretly, the same bet. The 60/40 investor, the long-bond holder, and the bank treasurer all made the *identical* wager — that rates would stay low — and the discount rate settled all three accounts at once. The next time the Fed has to move fast, the assets will be different, but the arithmetic will be exactly this.

## Further reading & cross-links

- [COVID 2020: infinite QE, the fiscal bazooka, and the everything rally](/blog/trading/policy-and-markets/covid-2020-infinite-qe-fiscal-bazooka-and-the-everything-rally) — the easy-money boom that set up the 2022 bust; the low yields SVB and the 60/40 both bought into.
- [The discount-rate channel: how rates reprice cash flows](/blog/trading/policy-and-markets/the-discount-rate-channel-how-rates-reprice-cash-flows) — the present-value arithmetic that drives every repricing in this post, derived from scratch.
- [How policy sets the bond market: the yield curve](/blog/trading/policy-and-markets/how-policy-sets-the-bond-market-the-yield-curve) — how the Fed funds path translates into the whole curve and the discount rate behind every asset.
- [Macroprudential and regulatory policy](/blog/trading/policy-and-markets/macroprudential-and-regulatory-policy) — the SVB failure, the BTFP backstop, and how regulators respond when a rate shock threatens the financial plumbing.
- [The 2025 tariff shock: policy uncertainty reprices markets](/blog/trading/policy-and-markets/the-2025-tariff-shock-policy-uncertainty-reprices-markets) — the next chapter, when a *different* policy lever (trade) repriced markets again.
- [Volcker 1979-82: how 20% rates repriced everything](/blog/trading/policy-and-markets/volcker-1979-82-how-20-percent-rates-repriced-everything) — the original, far larger version of this exact shock, and why 2022 was called "the fastest since Volcker."
- [2021-2023: inflation and the fastest hiking cycle](/blog/trading/macro-trading/2021-2023-inflation-and-the-fastest-hiking-cycle) — the trader's-positioning companion to this case study.

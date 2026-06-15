---
title: "Correlation by Regime: How the Whole Map Re-Wires with Growth and Inflation"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "Which assets diversify which is not a fixed fact — it depends on the macro regime. Two axes, growth and inflation, carve the world into four quadrants, each with a different leader and a different correlation structure."
tags: ["asset-allocation", "cross-asset", "correlation", "macro-regimes", "investment-clock", "stagflation", "diversification", "stocks-bonds", "commodities", "gold", "all-weather"]
category: "trading"
subcategory: "Cross-Asset"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Correlation is not a constant; it is a function of the macro regime. Two axes — is growth rising or falling, is inflation rising or falling — carve the world into four quadrants, and each quadrant has a different leading asset and a different correlation structure.
>
> - In a **growth shock** (recession fear), bonds rally as stocks fall, so they hedge each other — correlation is negative and a 60/40 portfolio works.
> - In an **inflation shock** (price fear), bonds and stocks fall *together* — correlation flips positive, the 60/40 hedge vanishes, and commodities and gold decouple to the upside.
> - The four-quadrant map: **Goldilocks** (growth up, inflation down) → stocks lead; **Overheat** (growth up, inflation up) → commodities lead; **Stagflation** (growth down, inflation up) → cash and gold lead; **Recession** (growth down, inflation down) → long bonds lead.
> - The one number to remember: the rolling stock-bond correlation ran near **−0.40 in 2008** (bonds hedged) and **+0.55 in 2022** (bonds did not). Same two assets, opposite relationship — because a different shock was in charge.

In 2008, the worst year for stocks in three generations, the boring part of your portfolio saved you. The S&P 500 fell 37% with dividends, but long-dated US Treasury bonds *gained* 25.9%. The bonds did exactly what the textbook promised: when stocks crashed, they zigged while stocks zagged. A 60/40 portfolio — 60% stocks, 40% bonds — lost far less than stocks alone, because the bond sleeve was a shock absorber. Diversification worked.

In 2022, the same textbook portfolio had its worst year since 1937. Stocks fell 18.1%. And the bonds, the supposed shock absorber, fell *13.0%* right alongside them. The 60/40 lost about 16% — there was no hedge, no zig against the zag, nothing to cushion the fall. The only major asset that made money was commodities, up 16.1%. The exact same pair of assets that protected each other in 2008 sank together in 2022. Nothing about the *assets* changed. What changed was the **regime** — the kind of shock the economy was absorbing.

The diagram above is the mental model for this whole post. Most people treat correlation — the number that says whether two assets move together or apart — as a fixed property, like a chemical bond. It is not. It is a regime-dependent question. Whether bonds hedge stocks, whether gold protects you, whether commodities help or hurt, all flip depending on where the economy sits on two axes: is growth rising or falling, and is inflation rising or falling. Those two axes carve the world into four quadrants, and the entire cross-asset map re-wires as you move between them.

![Four-quadrant map of macro regimes by growth and inflation, each with its leading asset](/imgs/blogs/correlation-by-regime-growth-and-inflation-1.png)

This is the conceptual bridge of the series. We have spent the correlation track learning *that* assets move together or apart and *why* the [stock-bond correlation is the engine of the 60/40 portfolio](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine). Now we generalize that single insight to the whole matrix and hand it to the timing track: **the correlations between every pair of assets re-wire with the macro regime, so "which assets diversify which" is a question you have to answer fresh for each quadrant.** By the end, you will be able to look at the world, place it in one of four boxes, and know what should lead, what should hedge, and what to avoid.

## Foundations: two axes, four quadrants, and what "correlation" actually means

Before we can talk about how the map re-wires, we need three things defined from zero: what correlation is, what the two macro axes are, and what a "regime" means. None of these require any finance background — just a willingness to think about money as a system.

### What correlation means, in plain English

**Correlation** is a single number, between −1 and +1, that summarizes whether two things tend to move in the same direction or opposite directions.

- **+1** means they move in perfect lockstep: when one goes up, the other always goes up by a proportional amount.
- **0** means they are unrelated: knowing one moved tells you nothing about the other.
- **−1** means they move in perfect opposition: when one goes up, the other always goes down.

Real-world correlations live in between. Two large US tech stocks might have a correlation of +0.8 — they mostly move together, but not identically. Stocks and gold over the last decade had a correlation of about +0.05 — basically unrelated. Stocks and the US dollar ran around −0.30 — they lean in opposite directions, but loosely.

Here is why this number is the whole game for a portfolio. The benefit of diversification — owning many things so no single bad event sinks you — comes almost entirely from owning assets with *low or negative* correlation to each other. We covered the mechanism in [correlation and the diversification "free lunch"](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch): two risky assets that don't move together combine into a portfolio that is *less* risky than either one alone. That is the only free lunch in finance. And it is paid for entirely by correlation staying low. The instant two assets you thought were diversifiers start moving together, the free lunch is gone — and you find out at the worst possible time, in a crash, when everything you own falls at once.

So the question "is this asset a good diversifier?" reduces to "what is its correlation to the rest of my portfolio?" And the thesis of this post is that *that correlation is not a fixed number*. It changes — sometimes it changes sign — depending on the macro regime. A diversifier in one regime is dead weight, or worse, a second source of loss, in another.

#### Worked example: how much a correlation flip costs you

Let's make this concrete with the simplest possible portfolio. You hold \$50,000 in stocks and \$50,000 in bonds, \$100,000 total. The standard deviation — the typical year-to-year swing — of each sleeve is 15%. (Standard deviation just measures how much an asset bounces around; a 15% standard deviation means a "normal" year is plus or minus 15%.)

How risky is the *combined* \$100,000 depends entirely on the correlation between the two sleeves. The formula for the portfolio's standard deviation with two equal-weighted, equal-risk sleeves is:

$$\sigma_p = \sigma \sqrt{\tfrac{1 + \rho}{2}}$$

where $\sigma$ is each sleeve's standard deviation (15%) and $\rho$ (rho) is the correlation between them.

- **If correlation is −0.40** (the 2008 regime, bonds hedging): $\sigma_p = 15\% \times \sqrt{(1 - 0.40)/2} = 15\% \times \sqrt{0.30} = 15\% \times 0.548 = 8.2\%$. Your \$100,000 has a typical swing of about \$8,200.
- **If correlation is +0.55** (the 2022 regime, bonds and stocks falling together): $\sigma_p = 15\% \times \sqrt{(1 + 0.55)/2} = 15\% \times \sqrt{0.775} = 15\% \times 0.880 = 13.2\%$. The same \$100,000 now has a typical swing of about \$13,200.

The portfolio is **61% more volatile** just because the correlation flipped from −0.40 to +0.55 — and you did not change a single holding. The intuition: a hedge that disappears does not just remove protection, it converts your second asset from a shock absorber into a second engine of loss.

### The two macro axes: growth and inflation

Now the two axes. Almost everything that drives asset prices across the whole economy can be compressed into two questions:

1. **Growth** — is the real economy speeding up or slowing down? Are companies selling more, hiring more, earning more (growth rising)? Or are sales softening, layoffs starting, earnings shrinking (growth falling)? The cleanest single proxy is whether GDP and corporate earnings are accelerating or decelerating.

2. **Inflation** — is the general price level rising faster or slower? When inflation is *rising*, each dollar buys less next year than this year, and the central bank tends to raise interest rates to cool things down. When inflation is *falling* (disinflation, or outright deflation), prices are cooling and the central bank can cut rates. The real interest rate — the rate after subtracting expected inflation — is the master signal underneath this axis: it is what gold and bonds ultimately respond to.

Why these two and not, say, ten variables? Because growth and inflation are the two forces that pull asset prices in *different and identifiable* directions. Stocks love growth and tolerate low inflation. Bonds love low-and-falling inflation and falling rates. Commodities love rising inflation and rising demand. Gold loves falling real yields. Cash loves rising rates and chaos. Each asset has a different favorite quadrant — and that is exactly why the correlations between them re-wire as you cross from one quadrant to the next.

### What a "regime" is

A **regime** is simply a stretch of time in which the same macro force is in charge of asset prices. Markets are not always driven by the same thing. Sometimes a *growth shock* — a recession scare, a credit crisis, a pandemic — dominates everything, and the question on every trader's screen is "how bad will the slowdown be?" Other times an *inflation shock* — an oil embargo, a supply-chain breakdown, a wage-price spiral — dominates, and the question is "how high will rates have to go to kill inflation?"

The regime is the answer to "what shock is in charge right now?" And the key fact, the one this entire post is built on, is that **the dominant shock determines the sign of the correlations.** When a growth shock is in charge, falling growth drags stocks down *and* drags interest rates down (the central bank cuts to fight the slowdown), and falling rates push bond *prices up* — so bonds rise while stocks fall. Negative correlation. When an inflation shock is in charge, rising inflation hurts stocks (higher discount rates compress valuations) *and* pushes interest rates up (the central bank hikes to fight inflation), and rising rates push bond *prices down* — so bonds and stocks fall together. Positive correlation.

Same two assets. Opposite relationship. The difference is entirely which axis is doing the shocking.

### The one mechanism underneath everything: cash flows over a discount rate

There is a single mechanism that explains *why* growth and inflation are the two axes that re-wire the whole map, and it is worth installing now because it makes every quadrant fall out almost automatically. Almost every financial asset is, at bottom, a claim on a stream of future cash flows. A stock is a claim on a company's future profits. A bond is a claim on future coupon payments and the return of principal. A piece of real estate is a claim on future rents. Even gold and commodities can be thought of as claims on future purchasing power. And the value of any such claim is its future cash flows *divided by* a discount rate:

$$\text{Value} = \frac{\text{future cash flows}}{(1 + r)^t}$$

where the numerator is what you expect to receive and $r$ is the discount rate — roughly, the interest rate plus a risk premium — that converts future money into today's money. The further out the cash flow ($t$) and the higher the discount rate ($r$), the less a future dollar is worth today.

Now look at what the two macro axes do to this fraction:

- **Growth moves the numerator.** When the economy is growing, expected future cash flows rise — companies earn more, rents rise, defaults fall. Stronger growth lifts the top of the fraction, which is good for risk assets (stocks, credit, real estate). Weaker growth shrinks the numerator and hurts them.
- **Inflation moves the denominator.** When inflation rises, central banks raise interest rates, which raises the discount rate $r$ at the bottom of the fraction. A higher denominator shrinks the value of *every* asset whose worth comes from far-off cash flows — long-dated bonds and expensive "growth" stocks get hit hardest, because their cash flows are furthest out and so most sensitive to the discount rate.

This is the engine of the entire re-wiring. A *growth shock* moves the numerator down and, via rate cuts, moves the denominator down too — and the denominator effect is what lifts bond prices, so bonds rise as stocks fall. An *inflation shock* moves the denominator *up* (rate hikes) while also hurting the numerator for stocks — so bonds and stocks both fall, their prices crushed by the same rising $r$. The reason commodities and gold decouple in an inflation shock is that they sit *outside* this fraction: they are not claims on nominal cash flows discounted at $r$, they are physical stores of value that rise *with* the inflation that is wrecking everything paper. Once you see that growth works on the top of the fraction and inflation works on the bottom, you can derive every quadrant's leaders and laggards yourself — the interest rate is the denominator's master switch, and the central bank moves it in response to the inflation axis.

## The regime-dependent matrix: the same pair, opposite sign

The heart of the idea is that one correlation — say, between stocks and bonds — does not have one value. It has at least two, depending on the regime, and they have opposite signs. Let's trace exactly why, then generalize from the stock-bond pair to the whole map.

![Same assets opposite relationship across a growth shock and an inflation shock](/imgs/blogs/correlation-by-regime-growth-and-inflation-2.png)

The figure above lays out the two regimes side by side, the same chain of cause and effect running through each. Read the left column top to bottom — the **growth shock**, a recession fear. Read the right column — the **inflation shock**, a price fear. Watch what happens to the same four assets.

**In a growth shock**, the dominant fear is that the economy is slowing. Stocks fall because weaker growth means weaker future earnings, and a stock is a claim on future earnings. But here is the mechanism that makes bonds a hedge: a slowing economy makes the central bank *cut* interest rates to stimulate. When market interest rates fall, the price of existing bonds *rises* (a bond paying a fixed 4% coupon is worth more when new bonds only pay 2%). So bond prices rise *exactly when* stock prices fall. The two are negatively correlated, and the 60/40 portfolio gets its shock absorber. Commodities, meanwhile, fall or go flat — a slowing economy needs less oil, less copper, less of everything — so they offer no help but do no harm.

**In an inflation shock**, the dominant fear is that prices are rising out of control. Stocks fall — partly because higher inflation forces higher interest rates, and higher rates make a dollar of future earnings worth less today (you discount it more harshly). But now the bond mechanism *reverses*: to fight inflation, the central bank *raises* rates. When market rates rise, existing bond prices *fall*. So bonds fall *at the same time* as stocks. The hedge has not weakened — it has inverted. The two are now positively correlated, and the 60/40 has lost its second leg. Commodities, though, are a different story entirely: in an inflation shock, commodities *are* the inflation. Rising oil and food and metal prices are often the very thing driving the inflation print higher. So commodities and gold decouple to the upside — they rise while everything paper falls. They become the only thing that hedges an inflation shock.

This is the entire re-wiring in one picture. The relationships you depend on — bonds hedge stocks, commodities are a sideshow — are true in a growth-shock regime and false in an inflation-shock regime. Owning the right diversifier means knowing which regime you are in.

### From one pair to the whole map: the calm baseline

Before we watch the whole matrix re-wire, we need a baseline — what the cross-asset map looks like in calm, "normal" times, the regime markets spend most of their years in. Here is the cross-asset correlation matrix computed from monthly total returns over 2015 to 2024, a decade dominated by the benign growth-up, inflation-down regime.

![Cross-asset correlation heatmap for eight asset classes 2015 to 2024](/imgs/blogs/correlation-by-regime-growth-and-inflation-3.png)

Read the heatmap as a map of who diversifies whom in calm times. Red cells are positive correlations (assets that move together — bad for diversification); green cells are negative (assets that move apart — good for diversification). A few clusters jump out:

- **Stocks, high-yield bonds, and REITs are all near +0.75 with each other.** They are all "risk assets" — they all rise when the economy is healthy and fall when it is scared. Owning all three feels like diversification but is mostly the same bet three times. High-yield bonds (corporate bonds from lower-rated companies) correlate +0.75 with stocks because their main risk — that the company defaults — is exactly the risk that spikes in a recession.
- **Bonds (high-quality government and aggregate bonds) sit near +0.10 with stocks** in this calm-decade average — close to zero, which is *why* the 60/40 worked so well across these years. That near-zero is the period average; underneath it, the relationship was sharply negative during the growth scares of 2018 and 2020 and sharply positive in 2022.
- **The US dollar is the great diversifier on this map**, negative against almost everything: −0.30 to stocks, −0.35 to commodities, −0.40 to gold. When fear hits, money flees to the dollar — the world's reserve currency — and the dollar rises while risk assets fall. A rising dollar is itself one of the cleanest signs that a growth-shock, risk-off regime has taken over, because frightened capital runs to the safest currency first.
- **Gold is loosely positive with bonds (+0.30) and commodities (+0.35), near zero with stocks (+0.05).** It marches to its own drummer, which is exactly what makes it interesting in a portfolio. Whether gold is [money, insurance, or just a rock](/blog/trading/cross-asset/gold-money-insurance-or-just-a-rock) is its own deep question.

Now hold this map in your head, because every number on it is a *calm-regime* number. The moment a real shock arrives, the cells re-color. The stocks-bonds +0.10 swings to −0.40 in a growth shock and +0.55 in an inflation shock. The commodities row, near-zero-to-slightly-positive in calm times, swings sharply negative against stocks in an inflation shock as commodities rip while stocks fall. The map you diversify against in good times is not the map you live with in a crisis.

#### Worked example: a diversifier that quietly stops diversifying

Suppose in 2021 you built a "diversified" \$200,000 portfolio: \$100,000 in stocks and \$100,000 in high-yield bonds. You reasoned that you had spread across two different asset classes — stocks and bonds — so you were diversified.

Look at the calm matrix: stocks and high-yield bonds correlate **+0.75**. They are not two different bets; they are nearly the same bet. So when 2022's inflation shock hit, stocks fell 18.1% (−\$18,100 on your stock sleeve) and high-yield bonds fell 11.2% (−\$11,200 on your HY sleeve). Total loss: −\$29,300, or −14.7%. There was no offset, because the two sleeves were never diversifiers of each other — they were both risk-on bets wearing different labels.

Had you instead split \$100,000 stocks and \$100,000 *government* bonds, you would still have been hurt in 2022 (the inflation-shock regime breaks even that hedge), but the lesson stands: the label "bonds" hid a +0.75 correlation that made the diversification an illusion. The intuition: diversification lives in correlation, not in the number of tickers you own.

## Quadrant by quadrant: what leads, what hedges, what to avoid

Now we build the full four-quadrant map. The framing comes from the **Merrill Lynch Investment Clock**, a classic way of mapping the business cycle onto asset leadership. The clock says: as the economy rotates through its cycle, growth and inflation trace a loop, and a different asset class leads in each quadrant. We do not need the full cyclical-loop machinery — we just need the four boxes and their leaders.

![Asset leadership grid showing what leads hedges and suffers in each of four regimes](/imgs/blogs/correlation-by-regime-growth-and-inflation-6.png)

The grid above is the reference table for the rest of the post. Each row is a regime; reading across gives you the asset that leads, the asset that still protects you, the asset that suffers most, and what the stock-bond correlation does. Let's walk each quadrant.

### Goldilocks (growth up, inflation down): stocks and credit lead

This is the dream quadrant — "not too hot, not too cold." The economy is growing, so earnings rise and stocks climb. Inflation is low and falling, so the central bank is relaxed and interest rates are low, which is also good for stocks (cheap money, low discount rates) and fine for bonds. Risk assets compound, volatility is low, and almost everything works.

The leader is **stocks**, with **credit** (corporate bonds, which earn extra yield over government bonds) right behind. Credit shines in Goldilocks for a specific reason: the extra yield a corporate bond pays over a government bond — the *credit spread* — compensates you for the risk the company defaults, and in a growing economy defaults are rare, so that spread is almost pure bonus return. You collect the premium without the risk showing up. The years 2017 (+21.8%), 2019 (+31.5%), and 2023 (+26.3%) were classic Goldilocks years for the S&P 500 — strong gains, low drama. The stock-bond correlation is **negative** here, so the 60/40 portfolio thrives: when the occasional growth wobble appears, bonds rally and cushion it. The only thing that "suffers" in Goldilocks is anything defensive — cash earns little, and gold and commodities lag because there is no inflation or fear to bid them up. Holding too much cash in Goldilocks costs you the rally.

### Overheat / Reflation (growth up, inflation up): commodities and energy lead

The economy is still growing, but now it is running hot enough that inflation is rising. This is late-cycle. Demand is strong, capacity is tight, and prices of physical things — oil, metals, food — start to climb. The leader rotates to **commodities and energy**, and within the stock market, **value** stocks (banks, energy, industrials) beat **growth** stocks (expensive tech), because rising rates punish the long-dated cash flows that growth stocks are valued on.

Here the cracks in the 60/40 begin. Bonds suffer — rising inflation means rising rates, and rising rates mean falling bond prices. The stock-bond correlation starts turning positive. 2021 was a textbook Overheat year: the Bloomberg Commodity Index returned +27.1% while the aggregate bond index *lost* 1.5%. Growth was strong, inflation was building, and the asset that led was the one that *is* inflation. If your portfolio held no commodities and lots of long bonds, Overheat is where you first feel the regime turning against you.

The second-order effect inside the stock market is the value-over-growth rotation, and it falls straight out of the cash-flow-over-discount-rate mechanism. A "growth" stock — an expensive technology company whose profits are expected to arrive years in the future — has most of its value in far-off cash flows, deep in the denominator's reach. A "value" stock — a bank, an energy producer, an industrial — earns cash *now* and is less sensitive to a higher discount rate. When inflation pushes rates up in Overheat, the discount rate rises and the far-off cash flows of growth stocks get marked down hardest, while value stocks, earning today, hold up better. So within stocks, the leadership rotates from growth to value at exactly the moment commodities take over the cross-asset lead.

#### Worked example: why a higher discount rate hurts growth stocks more than value stocks

Let's price the value-over-growth rotation. Two companies each promise \$100 of cash flow, but at different horizons. The "value" company pays its \$100 next year ($t = 1$); the "growth" company pays its \$100 in ten years ($t = 10$). Start with a discount rate of 4%, then let an inflation shock push it to 8% and see which stock falls more.

At a 4% discount rate, using $\text{Value} = 100 / (1 + r)^t$:

- Value stock: \$100 / (1.04)^1 = **\$96.15**
- Growth stock: \$100 / (1.04)^10 = **\$67.56**

Now the inflation shock lifts the discount rate to 8%:

- Value stock: \$100 / (1.08)^1 = **\$92.59** — a fall of about **3.7%**
- Growth stock: \$100 / (1.08)^10 = **\$46.32** — a fall of about **31.4%**

The same 4-percentage-point rise in the discount rate barely scratched the value stock but wiped nearly a third off the growth stock — because the growth stock's cash flow sits ten years out, where the compounding of a higher $r$ does its real damage. The intuition: in an inflation shock, the longer your cash flows are dated, the harder the rising discount rate hits you, which is why long bonds and expensive growth stocks suffer together while near-term-cash value stocks and commodities lead.

### Stagflation (growth down, inflation up): cash and gold lead — the 60/40 nightmare

This is the dangerous quadrant. Growth is *falling* — recession is a real risk — but inflation is *still high*. The economy is weak and prices are high at the same time. This is the worst-case for a conventional stock-bond portfolio, because *both* its sleeves get hurt: stocks fall on weak growth, and bonds fall because high inflation forces the central bank to keep rates high even as the economy weakens. There is no hedge inside the 60/40. The stock-bond correlation is firmly **positive** — they fall together.

The only things that lead are **cash** (which now earns a real return because rates are high, and loses nothing) and **gold** and **real assets / commodities** (which hold value when paper money is being eroded by inflation). The proof is 1973–74, the canonical stagflation: the S&P 500 fell about 37% across the two years, but **gold rose roughly 210%**, oil quadrupled, and commodities soared. A portfolio of stocks and bonds had nothing to hold onto. A portfolio with gold and commodities had its only winners. Stagflation is the quadrant that breaks the most portfolios, because most portfolios are built — implicitly — for the other three quadrants.

#### Worked example: the 60/40 in a stagflation quarter vs a quadrant-aware mix

Let's price the stagflation nightmare directly. You have \$100,000. Compare two ways to hold it through a stagflation quarter in which stocks fall 10% and bonds fall 6%.

**Portfolio A — the classic 60/40**, built for Goldilocks, where it shines:

- Stocks: \$60,000 × (−10%) = **−\$6,000**
- Bonds: \$40,000 × (−6%) = **−\$2,400**
- Total: **−\$8,400**, a loss of **−8.4%**, with no offset anywhere. Both sleeves fell because both are hurt by a stagflation shock.

**Portfolio B — a four-quadrant-aware mix**: \$40,000 stocks, \$25,000 bonds, \$20,000 commodities, \$15,000 gold. In the same quarter, commodities rise 12% and gold rises 8% (the stagflation winners), while stocks and bonds fall the same as before:

- Stocks: \$40,000 × (−10%) = **−\$4,000**
- Bonds: \$25,000 × (−6%) = **−\$1,500**
- Commodities: \$20,000 × (+12%) = **+\$2,400**
- Gold: \$15,000 × (+8%) = **+\$1,200**
- Total: **−\$1,900**, a loss of just **−1.9%**.

Portfolio B lost roughly a quarter as much — \$1,900 versus \$8,400 — because it *owned the assets that lead when growth falls and inflation rises*. The intuition: in stagflation, diversification does not come from owning more stocks and bonds; it comes from owning the things that go up when stocks and bonds go down together.

### Recession / Deflation (growth down, inflation down): long bonds and quality lead

Growth is falling *and* inflation is falling — a classic recession or, in the extreme, deflation. Demand collapses, prices soften, and the central bank slashes rates to the floor. Now **long-dated government bonds** are the kings: falling rates push their prices up hard, and they are the safest thing to own when the world is scared. **Cash** and **high-quality** assets hold up too. This is the one downturn where the 60/40 works beautifully again, because the stock-bond correlation goes sharply **negative** — bonds rally exactly as stocks crash.

2008 is the template. The S&P fell 37%, but long Treasuries gained **25.9%**, the broad bond index gained 5.2%, and gold added 5.5%. The asset that suffered most was **commodities**, down 35.6% — in a deflationary collapse, there is no demand for physical things and no inflation to bid them up, so the commodity that led in Overheat becomes the worst performer in Recession. The leader and the loser have completely swapped places from one quadrant to the diagonally opposite one.

There is a subtle but important split *within* the bond sleeve here, and it is the difference between high-quality and low-quality credit. In a recession, **government bonds and investment-grade corporate bonds** rally — they are the safe-haven duration play, and falling rates lift their prices. But **high-yield bonds** — debt of the riskiest companies — fall hard, because the recession is exactly when those companies default. High-yield bonds lost 26.2% in 2008, behaving far more like stocks than like bonds, which is precisely the +0.75 correlation we saw on the calm-regime heatmap turning lethal in a real downturn. The word "bonds" hides a fork: in a recession, the safe end of the bond market is your best hedge and the risky end is just stocks in disguise. Knowing which "bonds" you own is the difference between a hedge and a second helping of the crash.

### Notice the diagonal symmetry of the map

Step back and look at the four quadrants as a whole and a pattern appears: the leaders sit on diagonals. Goldilocks (growth up, inflation down) and Recession (growth down, inflation down) share the *inflation-down* axis, and in both the stock-bond correlation is negative — bonds work as a hedge in both, which is why the calm decade and the 2008 crash both flattered the 60/40. Overheat (growth up, inflation up) and Stagflation (growth down, inflation up) share the *inflation-up* axis, and in both the stock-bond correlation is positive — bonds fail as a hedge in both, which is why 2021 and 2022 both punished the conventional portfolio. The inflation axis, not the growth axis, is what controls the sign of the stock-bond hedge. Growth decides whether you want to be in risk assets at all; inflation decides whether your bonds will save you when you are not. That single observation is most of what you need to position a portfolio: watch the inflation axis to know if your hedge works, watch the growth axis to know how much risk to carry.

## The master flip: stock-bond correlation as a function of regime

If you remember only one relationship from this post, make it the stock-bond correlation, because it is the cleanest, most-studied case of a correlation that re-wires — and it is the load-bearing assumption of the most popular portfolio in the world.

![Stock-bond correlation by era, positive in inflation regimes and negative in growth regimes](/imgs/blogs/correlation-by-regime-growth-and-inflation-5.png)

The chart shows the rolling 24-month correlation between US stock and Treasury returns at era midpoints, going back to the 1970s. Read the green bars as "bonds hedged stocks" (negative correlation) and the red bars as "bonds did not hedge" (positive correlation). Notice the pattern is not random — it tracks the inflation regime:

- **The 1970s–1990s: positive correlation** (red), around +0.20 to +0.35. This was a high-and-volatile-inflation era. Inflation shocks were the dominant force, so bonds and stocks moved together and the 60/40 was a much weaker hedge than people remember.
- **2000–2021: negative correlation** (green), reaching −0.40 in 2008 and staying negative through 2020. This was the era of low, stable inflation — the "Great Moderation" — when growth shocks dominated and bonds reliably hedged stocks. This is the era in which the 60/40's reputation as a magic shock absorber was forged. A generation of investors learned a rule that was only true *for that regime*.
- **2022 onward: positive again** (red), +0.55 in 2022, +0.50 in 2023. Inflation came roaring back as the dominant driver, and the correlation flipped sign for the first time in two decades. The 60/40 had its worst year since the 1930s precisely because the regime had changed and the hedge had inverted.

The single most important takeaway: the famous negative stock-bond correlation that makes the 60/40 work is **not a law of nature — it is a feature of the low-inflation regime.** When inflation is the dominant shock, the correlation goes positive and the hedge disappears. The number that decides whether your "diversified" portfolio is actually diversified is the answer to "is growth or inflation the shock in charge right now?"

![Total returns of stocks bonds commodities and gold across four macro regimes](/imgs/blogs/correlation-by-regime-growth-and-inflation-4.png)

The bar chart drives the point home with real returns. The same four assets — stocks, bonds, commodities, gold — are plotted across four real regimes. In **stagflation 1973–74**, gold and commodities soar while stocks crash. In the **2008 recession**, bonds (long Treasuries +25.9%) and gold protect while commodities (−35.6%) are crushed. In the **2022 Overheat/inflation shock**, commodities (+16.1%) are the only winner while both stocks and bonds fall. In **Goldilocks 2019**, stocks lead (+31.5%) with everything calm. The leader rotates every single time. There is no asset that is "the diversifier" in all regimes — only the right diversifier for each regime.

#### Worked example: the all-weather idea falls out of the math

Here is where the four-quadrant map points to a portfolio. If no single asset leads in every regime, then the only way to have *something* working in every quadrant is to own a slice of each quadrant's leader. This is the seed of the "all-weather" idea, and we can size it with a simple example.

Take \$100,000 and split it into four equal \$25,000 buckets, one tuned to each quadrant: \$25,000 stocks (Goldilocks), \$25,000 commodities (Overheat), \$25,000 gold (Stagflation), \$25,000 long bonds (Recession). Now run it through each regime's leader-vs-loser spread (using the real returns above, roughly):

- **Goldilocks year** (stocks +25%, others muted): stocks +\$6,250, the rest roughly flat → portfolio up a few thousand. You participate in the boom through the stock bucket.
- **Stagflation** (gold +30%, stocks −20%): gold +\$7,500, stocks −\$5,000, commodities up, bonds down → the gold and commodity buckets offset the stock loss. You survive the quadrant that kills the 60/40.
- **Recession** (long bonds +25%, commodities −35%): bonds +\$6,250, commodities −\$8,750, stocks down → the bond bucket cushions the crash that the commodity bucket suffers.

No quarter is a disaster, because in every regime *at least one bucket is the leader*. You give up the explosive upside of being all-in on the right quadrant, in exchange for never being all-in on the wrong one. The intuition: you cannot predict the regime reliably, so you pay a small "insurance premium" — the drag of always holding three buckets that are not leading — to guarantee you always hold the one that is. We develop this fully in the timing track; for now, note that it falls directly out of the four-quadrant map.

## The dangerous illusion of the average correlation

There is a trap hiding in every correlation table, including the calm-regime heatmap we built earlier, and it is worth dismantling because it is how careful investors still get blindsided. A correlation computed over a long sample is an *average across regimes*, and an average can be a number that the relationship almost never actually equals.

Recall that the period-average stock-bond correlation over 2015 to 2024 was about +0.10 — close to zero, which sounds like "stocks and bonds are basically unrelated, so my 60/40 is well diversified." But +0.10 is the blend of two very different regimes that happened to occur in that window. The relationship was sharply *negative* during the growth scares (2015–16, 2018, the 2020 COVID crash) and sharply *positive* during the inflation surge of 2022. The +0.10 average is a number the correlation spent almost no time at — it is the midpoint of a relationship that was swinging between −0.40 and +0.55, not a stable resting value.

#### Worked example: how a benign average hides a lethal regime

Let's decompose the average. Suppose a ten-year sample is split into two regimes: eight years of a growth-shock world where the stock-bond correlation runs −0.30, and two years of an inflation-shock world where it runs +0.55. The time-weighted average is:

$$\bar{\rho} = \frac{8 \times (-0.30) + 2 \times (+0.55)}{10} = \frac{-2.40 + 1.10}{10} = \frac{-1.30}{10} = -0.13$$

So the table reports a comforting −0.13 — "bonds hedge stocks on average." But put real money on it. Build a \$100,000 60/40 portfolio and assume each sleeve has a 15% standard deviation. In the eight good years, the realized correlation of −0.30 gives a portfolio swing of $15\% \times \sqrt{(1 - 0.30)/2} = 15\% \times 0.592 = 8.9\%$ — a calm \$8,900 typical move. But in the two inflation-shock years, the realized correlation of +0.55 gives $15\% \times \sqrt{(1 + 0.55)/2} = 15\% \times 0.880 = 13.2\%$ — a \$13,200 swing, and crucially with *both* sleeves pointing down at once, so the bad outcomes cluster instead of offsetting. The −0.13 you budgeted for never shows up in the year that matters. The intuition: an average correlation tells you what *usually* happens, but a portfolio is killed by what happens in the *worst* regime, and that regime's correlation can be the opposite sign from the average you planned around.

This is why the regime view beats the table view. A correlation matrix gives you one number per pair; the regime map gives you the *conditional* correlation — the value the relationship takes given the current quadrant — which is the only number that protects you in a crisis.

## Common misconceptions

**"Bonds always hedge stocks."** No — bonds hedge stocks *in a growth-shock regime* and fail to hedge them in an inflation-shock regime. The reliable negative correlation that defines the 60/40 was a feature of the 2000–2021 low-inflation era. In 2022 it went to +0.55 and the hedge vanished. "Bonds hedge stocks" is a regime-specific truth dressed up as a universal law.

**"Diversification means owning many different assets."** No — diversification means owning assets with *low correlation to each other*. You can own eight tickers that all correlate +0.75 (stocks, high-yield bonds, REITs, emerging-market equities, and so on) and be barely diversified at all, because they are all the same risk-on bet. Real diversification requires owning things that lead in *different quadrants* — and that means deliberately holding assets that will lag most of the time.

**"Gold is a permanent hedge against everything."** No — gold's correlation to the rest of the map is itself regime-dependent. Gold shines in stagflation (gold +210% in 1973–74) and in growth scares with falling real yields (gold +25.1% in 2020). But it can do nothing for years in a Goldilocks regime, and it fell during parts of the rate-hiking 2022 even as inflation raged, because rising *real* yields are gold's kryptonite. Gold is a regime hedge, not an all-regime hedge.

**"Commodities are too volatile to own."** They are volatile in isolation, but a volatile asset that is *negatively* correlated to your core holdings in the regime where your core holdings get hurt is exactly what a portfolio wants. Commodities returned +27.1% in 2021 and +16.1% in 2022 — the years the 60/40 was breaking — precisely because they are the asset that leads when inflation is the shock. Their standalone volatility is the price of admission for their regime-hedging power.

**"I can just hold the asset that's working now."** This confuses the *level* with the *turn*. By the time an asset has obviously been leading for a year, the regime may already be rotating. The whole point of the four-quadrant map is to recognize the regime *and* the regime *change*, and to hold a hedge for the quadrant you are not in — because the costliest mistake is being maximally positioned for the regime that is about to end. [Risk-on / risk-off rotations](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates) are how the market repositions when the regime turns.

## How it shows up in real markets

**1973–74, stagflation, the textbook case.** The OPEC oil embargo quadrupled oil prices, inflation hit double digits, and growth stalled — growth down, inflation up. The S&P 500 fell about 37% across the two years and bonds gave almost no protection (long Treasuries up only ~4% nominal, deeply negative after inflation). The winners were exactly the stagflation leaders: gold rose roughly 210%, oil went from about \$3 to about \$12 a barrel, and commodities broadly soared. Any portfolio of stocks and bonds was devastated; any portfolio with real assets had a lifeline. This is the historical anchor for why "stagflation breaks the 60/40."

**2008, the deflationary collapse.** The global financial crisis was a growth-down, inflation-down shock — a credit crisis that threatened a deflationary spiral. The Fed slashed rates to zero. Long Treasuries gained 25.9% as rates collapsed, the broad bond index gained 5.2%, and gold added 5.5% — the recession-quadrant leaders all worked. Meanwhile commodities cratered 35.6% and high-yield bonds fell 26.2%, because in a deflationary bust there is no demand for physical things and the weakest borrowers default. The 60/40 worked beautifully here — the *opposite* outcome from 2022 — because the regime was the opposite kind of shock.

**2020, the COVID growth shock.** A sudden growth collapse with policy flooding the system. Stocks crashed 33.9% from the February peak to the March trough, then the regime stayed a growth shock with falling rates: bonds rose 7.5% for the year, gold rose 25.1% (falling real yields), and stocks roared back to +18.4% as stimulus hit. Commodities lagged (−3.1%) and oil briefly printed *negative* in April. A growth-shock regime, and the growth-shock hedges (bonds, gold) did their job.

**2021, the Overheat.** Growth rebounded hard from COVID while inflation began building — growth up, inflation up. Commodities led with +27.1% while bonds *lost* 1.5%, the first crack in the bond hedge. Value stocks beat growth. This was the warning shot: the regime had rotated from the 2020 growth shock into Overheat, and the assets that led rotated with it. Investors who were still positioned for the falling-rate 2020 regime — long-duration bonds, expensive growth stocks — started to bleed.

**2022, the inflation shock and the 60/40's worst year since 1937.** Inflation hit 9%, the Fed hiked rates at the fastest pace in 40 years, and the regime was unambiguously an inflation shock. Stocks fell 18.1%, bonds fell 13.0%, and the stock-bond correlation went to +0.55 — both sleeves of the 60/40 fell together, and the portfolio lost about 16%. The only major winner was commodities, +16.1%, the inflation-shock leader. This is the single cleanest real-world proof that the cross-asset map re-wires by regime: an event that would have been cushioned by bonds in 2008 was un-cushioned in 2022 because a different axis was doing the shocking.

**The early 1980s, the regime *transition* that minted a generation's intuition.** This is the most instructive episode of all, because it shows what happens when an axis flips. Coming out of the 1970s, the world was stuck in stagflation — high inflation, weak growth, gold and commodities leading. Then the Federal Reserve under Paul Volcker pushed interest rates to nearly 20% to break inflation, and it worked: inflation collapsed from double digits toward the low single digits over the following years. The inflation axis turned hard from "up" to "down." And as it did, the entire map re-wired in slow motion. Bonds, which had been dead money for a decade, began a thirty-year bull market as rates fell from 20% toward zero. Stocks, freed from the inflation drag, began the great bull market of the 1980s and 1990s. Gold, the stagflation king, peaked around \$850 an ounce in 1980 and then went nowhere for two decades. The negative stock-bond correlation that the 60/40 generation took as gospel was *born* in this transition — it was a property of the low-inflation regime that Volcker created, not a permanent feature of markets. An investor who learned "bonds always hedge stocks" in 1995 was learning a rule that had been false in 1975 and would be false again in 2022.

**2024–2025, the disinflation-with-growth tilt.** As inflation cooled while growth held up, the world leaned back toward Goldilocks — and stocks (+25.0% in 2024) and gold (+27.2% in 2024, extending above \$3,000 an ounce into 2025) both did well, an unusual pairing that reflected easing real yields plus resilient growth. The lesson: regimes are not always clean, and the map is a guide to the dominant force, not a guarantee of every cross-correlation.

## The allocation playbook: trading the four-quadrant map

This is the payoff. The four-quadrant map is not just a way to understand history — it is a repeatable loop for positioning a portfolio. Here is the playbook.

![Four-step regime-aware allocation playbook from identify to rebalance](/imgs/blogs/correlation-by-regime-growth-and-inflation-7.png)

**Step 1 — Identify the quadrant.** Answer two questions from the data: is growth accelerating or decelerating, and is inflation rising or falling? You do not need to forecast precisely; you need the *direction* of each axis. Growth direction shows up in purchasing-manager surveys, employment trends, and corporate earnings revisions; inflation direction shows up in the trend of the consumer price index and in the real-yield signal — the interest rate after subtracting expected inflation. Two axes, four boxes — place the world in one box.

**Step 2 — Tilt toward that quadrant's leaders.** Once you know the box, you know what should lead: Goldilocks → overweight stocks and credit; Overheat → add commodities and energy, tilt to value, shorten bond duration; Stagflation → raise cash, add gold and real assets, cut long bonds and growth stocks; Recession → extend bond duration, hold quality and cash, cut commodities. The tilt is a lean, not an all-in bet — you are overweighting the leader, not betting the whole portfolio on it.

**Step 3 — Hold a hedge for the opposite quadrant.** This is the discipline most people skip. Always own a slice of the regime you are *not* in, sized small. If you are positioned for Goldilocks, keep some gold and commodities for the stagflation tail. If you are positioned for an inflation shock, keep some long bonds for the deflationary-recession tail. The reason is humility: regime calls are uncertain, regimes turn faster than you expect, and the hedge is what keeps a wrong call from being a catastrophe. A portfolio that owns something for each quadrant — the all-weather idea — is just this principle taken to its logical end.

**Step 4 — Rebalance on the turn, not the noise.** Act when growth or inflation changes *direction* — when the regime itself rotates — not on every wiggle in the monthly data. The signal is the turn (growth rolling over, inflation re-accelerating), not the level. Over-trading on noise will cost you more in fees and whipsaws than the regime tilt earns. The regime turn is rare and slow; let it come to you.

### Reading the two axes in real time

The playbook is only as good as your read of the two axes, so it is worth being concrete about the everyday signals that tell you which way growth and inflation are pointing. None of these require a forecast — they are coincident or slightly leading indicators of *direction*, which is all the four-quadrant map needs.

For the **growth axis**, the most-watched single number is the purchasing-managers' index (PMI) — a monthly survey of company managers asking whether activity is expanding or contracting, where a reading above 50 means expansion and below 50 means contraction. A PMI rolling from 55 down toward 48 is the growth axis turning down, often before it shows up in the official GDP data. Alongside it, watch the direction of jobless claims (rising claims means weakening growth) and the trend in corporate earnings revisions (analysts cutting estimates means the growth numerator is shrinking). The shape of the yield curve is a slower-moving growth tell: when short-term rates rise above long-term rates — an "inverted" curve — the bond market is pricing a coming slowdown.

For the **inflation axis**, the headline is the consumer price index (CPI), but the *direction* matters more than the level. A CPI that printed 6% but is decelerating month over month is a falling-inflation signal, even though 6% is high. Watch the trend of commodity prices — especially oil and food, which feed straight into headline inflation — and the breakeven inflation rate, the gap between the yield on a normal Treasury and an inflation-protected one, which is the market's own forecast of future inflation. When breakevens are rising, the market is telling you the inflation axis is pointing up, and your bond hedge is in danger.

#### Worked example: placing the world in a quadrant from two signals

Suppose you read two numbers this month. The PMI fell from 52 to 49 (growth turning down, now in contraction), and the breakeven inflation rate rose from 2.2% to 2.8% (inflation turning up). Two signals, two directions: growth down, inflation up. That is the **stagflation** quadrant.

What does the map tell you to do with a \$100,000 portfolio sitting in a standard 60/40? Stagflation is the quadrant where both your sleeves are exposed — \$60,000 of stocks vulnerable to falling growth and \$40,000 of bonds vulnerable to rising inflation. The playbook says: trim the most exposed pieces and tilt toward the stagflation leaders. A modest move — shift, say, \$15,000 from the stock-and-long-bond sleeves into \$8,000 of gold and \$7,000 of broad commodities — converts a portfolio with *no* stagflation hedge into one with two. You have not predicted the future; you have read two directional signals, placed the world in a box, and tilted toward the assets that lead in that box. The intuition: the regime call is a reading exercise, not a forecasting one — you respond to the direction the axes are already moving, not to a guess about where they will be.

A few sizing notes, framed as ranges rather than prescriptions, because the right numbers depend entirely on your situation and this is education, not advice:

- **The tilt is modest.** A quadrant-aware investor might shift 10–20% of a portfolio toward the leader and away from the loser as a regime call firms up — not flip the whole book. The base portfolio still does most of the work; the regime tilt is the seasoning.
- **The hedge is small but non-zero.** A 5–15% allocation to the opposite quadrant's leader is enough to soften a wrong call without dragging the portfolio badly in the regime you are actually in.
- **What invalidates the case.** If the two axes disagree with your positioning — you are tilted for Goldilocks but inflation is clearly re-accelerating and growth is rolling over — that is the signal that the regime has turned and the tilt is now backwards. The map tells you not just what to own, but when you are wrong.

The deepest lesson of this post is one sentence: **a portfolio diversified for one regime can be completely undiversified for another.** The 60/40 that was bulletproof in 2008 was naked in 2022, not because anyone changed it, but because the regime changed underneath it. Correlation is not a number you can look up once and trust forever. It is a regime-dependent variable, and the whole cross-asset map — who leads, who hedges, who to avoid — re-wires every time the world crosses from one growth-by-inflation quadrant to the next. The investor who internalizes that does not ask "what's a good diversifier?" — they ask "what's a good diversifier *for the regime we're heading into?*"

## Further reading & cross-links

- [Stock-bond correlation: the engine of the 60/40](/blog/trading/cross-asset/stock-bond-correlation-the-60-40-engine) — the single-pair version of this post's whole-map argument, where the sign flip is unpacked in full.
- [Correlation and the diversification "free lunch"](/blog/trading/cross-asset/correlation-and-the-diversification-free-lunch) — why low correlation is the only free lunch in finance, and the math behind it.
- [The map of asset classes: what you can own](/blog/trading/cross-asset/the-map-of-asset-classes-what-you-can-own) — the eight asset classes on the heatmap, defined from zero.
- [Gold: money, insurance, or just a rock?](/blog/trading/cross-asset/gold-money-insurance-or-just-a-rock) — why gold is a regime hedge, not an all-regime hedge.
- [Risk-on / risk-off: how money rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates) — the market mechanism that repositions capital when the regime turns.

*This article is educational, not individualized financial advice. Historical returns are not predictions; regimes are uncertain and turn without warning.*

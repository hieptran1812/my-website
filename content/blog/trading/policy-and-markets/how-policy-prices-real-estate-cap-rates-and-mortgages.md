---
title: "How Policy Prices Real Estate: Cap Rates and Mortgages"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Real estate is the most rate-sensitive asset class on earth — this is how the policy rate reaches it through mortgages and cap rates, why one rate variable repriced offices down 30% and froze the housing market, and how to do the math yourself."
tags: ["monetary-policy", "real-estate", "cap-rates", "mortgage-rates", "interest-rates", "asset-valuation", "lock-in-effect", "commercial-real-estate", "central-banks", "macroprudential", "housing", "vietnam"]
category: "trading"
subcategory: "Policy & Markets"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — Real estate is a leveraged bond with a roof, and a bond is the most rate-sensitive thing there is. So when a central bank moves the policy rate, it reaches property through two channels at once — the *mortgage rate* that decides what a buyer can afford, and the *cap rate* (the yield a building's income must offer) that decides what an investor will pay. Move one rate and you reprice the largest asset class on the planet.
>
> - The mortgage channel: policy rate → mortgage rate → monthly payment → what a buyer can borrow → the price they can bid. When the 30-year mortgage went from sub-3% in 2021 to ~7.8% in 2023, the payment on the same loan jumped ~58%, and affordability collapsed.
> - The cap-rate channel: a building's value is its net operating income divided by the cap rate, and the cap rate is roughly the risk-free rate plus a risk premium minus growth. Raise the risk-free rate and the cap rate re-rates up, so the *same income* is worth less. US office values fell 25-40% on exactly this.
> - The lock-in effect: a homeowner sitting on a 3% mortgage will not sell into a 7% market, because moving means swapping a cheap loan for an expensive one on the same house. That single fact froze existing-home supply — policy reshaping a market through *behavior*, not just price.
> - The number to remember: a building throwing off **\$1 million** of income a year is worth **\$20 million** at a 5% cap rate and only **\$14.3 million** at 7% — a **28% loss** with the building completely unchanged.

In the autumn of 2020, you could borrow money to buy a house in America at 2.68% — the lowest 30-year mortgage rate Freddie Mac had ever recorded. Three years later, in October 2023, the same loan cost 7.79%, the highest since the year 2000. Nothing about the houses changed. The same three-bedroom in the same suburb with the same school district was, by every physical measure, identical. But the *price a buyer could afford to pay for it* had collapsed, because almost nobody buys a house with cash — they buy a monthly payment, and that payment had jumped by more than half.

Across town, in the office towers of every major US city, something even more dramatic was happening. The buildings were still standing, still leased (mostly), still collecting rent. And yet their *values* fell 25%, 30%, in the worst cases 40% or more. Some trophy towers in San Francisco and New York sold for less than half what they had fetched a few years earlier. The rents had barely moved. What had moved was the *cap rate* — the yield an investor demanded to own the income — and that cap rate is built directly on top of the interest rate the Federal Reserve had just spent eighteen months pushing from a 0.25% floor to a 5.50% ceiling.

This is the central fact of real estate as an asset class: **property is the most rate-sensitive thing most people will ever own.** A house is not really a physical object you trade; it is a *stream of housing value financed with an enormous, leveraged loan*. A commercial building is not really concrete and glass; it is a *bond-like stream of rent* that investors price off the risk-free rate. Both descriptions make property a creature of interest rates — and interest rates are set, at the base, by policy. This post builds the whole machine from the ground up: how the policy rate becomes a mortgage rate, how a mortgage rate becomes an affordable price, what a cap rate actually is and why a rising risk-free rate re-rates buildings down, why low-rate mortgages froze the housing market, how regulators use loan-to-value caps as a direct price lever, and why offices got hit so much harder than apartments. By the end you will be able to look at a rate move and estimate, on a napkin, roughly how much a given property should reprice — and why.

![Branching diagram showing the policy rate reaching property through a mortgage-affordability channel and a cap-rate discount channel that both converge on the property value](/imgs/blogs/how-policy-prices-real-estate-cap-rates-and-mortgages-1.png)

The figure above is the entire argument in one picture. A central bank pulls one lever — the policy rate. That lever splits into two channels. On the left, it sets the *mortgage rate*, which sets the monthly payment, which decides what a buyer can borrow and therefore what they can bid: the **affordability channel**. On the right, it sets the *risk-free rate*, which is the base of the *cap rate*, which is the yield an investor's income must clear: the **discount channel**. Both channels converge on the same place — the price of the property — and they both push it the same way. Raise the policy rate and property gets squeezed from both sides at once. Let us walk each channel, starting from the most basic question of all: what is a building actually worth?

## Foundations: a building is a leveraged bond with a roof

Before the mechanics, hold one number in mind: real estate is, by a wide margin, the **largest asset class on earth**. Global real estate is worth somewhere around \$380 trillion — more than all the world's stocks and bonds combined, and several times global GDP. For the typical household, the home is by far the largest asset they will ever own, and the mortgage by far the largest liability. So when a central bank's rate decision reprices property, it is not repricing a niche — it is repricing the single biggest store of wealth in the economy, the collateral behind most bank lending, and the balance sheet of nearly every household. That is why the rate-sensitivity of property is not an academic curiosity. It is the channel through which monetary policy reaches ordinary people most directly, and the reason a housing downturn so reliably drags the whole economy with it.

Strip away the curb appeal and the granite countertops, and any income-producing property is three things stacked on top of each other. Understanding the stack is the whole game, because each layer is rate-sensitive in its own way.

The first layer is **the rent stream**. A building you rent out throws off cash every month: tenants pay rent. Subtract the costs of running it — property taxes, insurance, maintenance, management, but *not* the mortgage — and what is left is the **net operating income**, almost always abbreviated **NOI**. NOI is the building's "earnings", the cash the asset itself produces before any financing. And here is the key insight: a stream of fairly predictable cash payments is *exactly what a bond is*. A bond pays you a fixed coupon every period; a building pays you NOI every period. The resemblance is not loose — it is the foundation of how the entire industry values property. A building is, financially, a bond whose coupon is its NOI.

The second layer is **the cap rate**, the yield that converts that bond-like income into a price. We will spend a whole section on it below, but the one-line version is: the cap rate is NOI divided by price, the rental yield of the building, and it behaves just like a bond's yield — when the cap rate goes up, the price goes down, even though the income (the "coupon") has not changed at all. The cap rate is where the risk-free interest rate enters the picture, because investors will not accept a lower yield on a risky building than they can get on a safe government bond, so the cap rate is anchored to the risk-free rate plus a premium.

The third layer is **leverage** — the mortgage. Almost nobody buys property with cash. A homeowner puts down 10-20% and borrows the rest; a commercial investor might put down 30-40% and borrow the rest. That borrowed money is a fixed-rate (or sometimes floating-rate) loan, and it sits *between* the investor and the asset. Leverage is a magnifier: it amplifies both the gains and the losses on the slice of equity the buyer actually put in. A 28% fall in a building's value, on a property bought with 65% debt, can wipe out *80%* of the owner's equity — the same arithmetic that makes a margined stock position so dangerous. So property is rate-sensitive *twice*: once because its value is a bond priced off rates, and again because it is bought with a leveraged loan whose cost is a rate.

On top of all three sits the one layer rates do *not* touch: the physical asset itself — the roof, the walls, the land, the shelter and use-value a property provides. That is the part that makes real estate more than a paper bond. But financially, the roof rides on top of a structure that is rate-sensitive top to bottom.

![Layered stack showing a property decomposed into a rent stream, a cap rate, a leveraged mortgage, and the physical roof on top](/imgs/blogs/how-policy-prices-real-estate-cap-rates-and-mortgages-3.png)

The stack above is why "real estate is the most rate-sensitive asset class" is not a slogan — it is structural. The income is bond-like (rate-sensitive), the price is a yield calculation off the risk-free rate (rate-sensitive), and the whole thing is bought with a rate-priced loan (rate-sensitive). Three of the four layers move with interest rates. For the foundational mechanics of *why* a higher discount rate makes any future cash flow worth less today, see the companion post on [the discount-rate channel](/blog/trading/policy-and-markets/the-discount-rate-channel-how-rates-reprice-cash-flows); this post is the property-specific application of that same engine.

### Where the policy rate actually enters

A central bank does not set mortgage rates or cap rates directly. It sets one thing: the very-short-term policy rate (in the US, the federal funds rate). Everything else is transmission. The policy rate anchors the front end of the bond market; the market's expectation of the *future* policy path sets longer-term yields like the 10-year Treasury; and those longer-term yields are the base of both the mortgage rate *and* the cap rate. (For how a single front-end rate propagates out the curve into the 10-year, see the companion post on [how policy sets the bond market](/blog/trading/policy-and-markets/how-policy-sets-the-bond-market-the-yield-curve).)

So the chain is: **policy rate → Treasury yields → (mortgage rate, cap rate) → property prices.** The Fed never has to touch a single house or office tower. It moves the base of money, and the cost of borrowing to buy property — and the yield investors demand from property — moves with it. That is leverage in the policy sense: one rate, two channels, the entire built environment repriced.

### Why property has such long "duration"

There is one more piece of foundational vocabulary that explains *why* real estate, specifically, is so violently rate-sensitive: **duration**. In the bond world, duration measures how much a security's price moves for a given change in yield — the longer the cash flows stretch into the future, the higher the duration, and the more the price whipsaws when rates move. A 2-year Treasury barely flinches when yields rise a point; a 30-year Treasury can lose 15% or more on the same move, because so much of its value sits in distant payments that get discounted harder.

Real estate has *enormous* duration, and that is the heart of its rate sensitivity. A building is valued as a near-perpetual stream of NOI — cash flows that, in principle, go on forever. A perpetuity has the longest duration of any cash-flow structure there is, because the cash flows never stop. That is why the cap-rate formula `value = NOI / cap rate` is so explosive: dividing by a small number (the cap rate) means a small *change* in that small number swings the value wildly. Going from a 5% cap to a 7% cap is only a 2-percentage-point move in absolute terms, but in *relative* terms it is a 40% increase in the rate you divide by — and that is why it lops 28% off the value. The same 2-point move from a 9% cap to an 11% cap (a smaller relative change) does far less damage. The lower the cap rate to begin with, the more a given rate rise hurts — which is exactly why the low-cap-rate, "priced-for-perfection" prime apartments and trophy assets of 2021 had the furthest to fall.

This duration intuition is also why the *long* end of the yield curve matters far more for property than the very front end. The Fed's overnight rate matters because it anchors expectations, but what actually sets mortgage rates and cap rates is the 10-year Treasury — the market's price of money over a long horizon, which is the right discount rate for an asset whose cash flows stretch out indefinitely. A Fed that hikes the overnight rate but convinces the market the hikes are temporary may leave the 10-year — and therefore mortgages and cap rates — relatively unmoved. It is the *persistent* level of long rates, not the overnight rate by itself, that reprices the built environment.

## The mortgage channel: you do not buy a house, you buy a payment

Start with residential, because it is where almost everyone meets this machine personally. The crucial fact is that a house buyer is not price-sensitive in the way a stock buyer is — they are *payment-sensitive*. A typical buyer figures out the largest monthly payment they can stomach (lenders cap it at roughly 28-36% of gross income), and then the mortgage rate decides how big a loan that payment will support. A higher rate means the same payment supports a smaller loan, which means the buyer can bid less, which means — across millions of buyers — house prices have to give.

The mortgage rate itself is not set by the Fed. It is set by the bond market: a 30-year fixed mortgage is funded by selling mortgage-backed securities to investors, who demand a yield a bit above the 10-year Treasury (because mortgages can be prepaid and carry some risk). Historically that spread runs around 1.7-3.0 percentage points. So the mortgage rate is, to a very good approximation, **the 10-year Treasury yield plus a spread** — which means it tracks the policy rate's influence on the long end of the curve.

![Scatter plot of the 30-year mortgage rate against the 10-year Treasury yield with a fitted line showing the spread](/imgs/blogs/how-policy-prices-real-estate-cap-rates-and-mortgages-4.png)

The scatter makes the relationship concrete. Each dot is a period, plotting the 10-year Treasury yield on the horizontal axis against the 30-year mortgage rate on the vertical. They line up on a clean upward line: when the 10-year was 0.93% at the end of 2020, the mortgage was 2.68%; when the 10-year hit 4.88% in October 2023, the mortgage hit 7.79%. The vertical gap between the dot and the 10-year — averaging around 2.6 points here, wider than usual in 2022-23 because mortgage-market stress fattened the spread — is the financing premium a homeowner pays over the government. The point is the *slope*: the mortgage rate rides the Treasury yield, and the Treasury yield rides the policy rate. Move the Fed, move the mortgage.

Now watch what the mortgage rate does to a payment.

#### Worked example: a \$500,000 mortgage at 3% versus 7%

Take a \$500,000 loan on a 30-year fixed mortgage and price it at the 2021 rate and the 2023 rate. The monthly payment formula is the standard amortizing-loan formula, but the numbers are what matter:

- At **3%**: the monthly principal-and-interest payment is about **\$2,108**.
- At **7%**: the monthly payment is about **\$3,327**.

That is a jump of **\$1,219 a month**, or about **\$14,600 a year**, on the *exact same loan amount* — same house, same borrower, same everything except the rate. Over the full 30-year life of the loan, the higher rate adds roughly \$439,000 in total interest. *The rate did not change the house by a single brick, but it raised the lifetime cost of owning it by nearly as much as the house's original price — which is why buyers simply vanished when rates spiked.*

The flip side of the same arithmetic is even more revealing, because it tells you what happens to *prices*. Buyers do not hold the loan amount fixed; they hold the *payment* fixed and let the loan amount float.

#### Worked example: how much house a \$2,400 payment buys at 3% versus 7%

Suppose a buyer can comfortably afford \$2,400 a month for principal and interest. Run the amortization backwards to find the largest 30-year loan that payment supports:

- At **3%**: \$2,400 a month supports a loan of about **\$569,000**.
- At **7%**: the same \$2,400 a month supports only about **\$361,000**.

The buyer's budget did not change — they can still pay \$2,400 a month. But the rate move cut their *purchasing power by about 37%*, from \$569k of house to \$361k of house. *If every buyer in the market loses 37% of their borrowing power, prices cannot stay where they were unless something else gives — which is exactly why the 2022-23 rate spike was such a violent affordability shock.* In practice, US home prices did *not* fall 37%, for a reason we are about to meet: the lock-in effect choked off supply at the same moment, so the market cleared on frozen *volume* instead of crashing *prices*.

![Step-and-line chart of the 30-year fixed mortgage rate from 2012 to 2025 falling to a 2.68 percent trough and spiking to a 7.79 percent peak](/imgs/blogs/how-policy-prices-real-estate-cap-rates-and-mortgages-2.png)

The chart shows the master variable itself — the 30-year fixed mortgage rate. It drifted down for a decade, bottomed at 2.68% in December 2020 (the green dot) as the Fed cut to zero and bought mortgage bonds during COVID, and then rocketed to 7.79% by October 2023 (the red dot) as the Fed reversed course. That climb — nearly tripling the cost of a mortgage in under two years — is the single most consequential move in the housing market in forty years, and every consequence in this post flows from it. (For how the Fed's balance-sheet operations directly bought down mortgage rates and then let them rise, see [the liquidity channel](/blog/trading/policy-and-markets/the-liquidity-channel-qe-qt-and-the-everything-bid).)

### Why the mortgage *structure* changes everything

A crucial subtlety hides inside the word "mortgage": the *structure* of the loan determines how the rate shock hits households, and it differs enormously across countries. This is not a detail — it is the difference between a frozen market and a foreclosure crisis.

In the United States, the dominant product is the **30-year fixed-rate, freely-prepayable, non-portable** mortgage. Each word matters. *Fixed-rate* means an existing borrower's payment never changes when rates rise — the homeowner who locked 3% in 2021 still pays 3% in 2024, fully insulated. *Prepayable* means they can refinance for free when rates fall, capturing the upside. *Non-portable* means the loan is tied to the house, not the person — sell the house and the loan is gone, so moving forces a new loan at today's rate. This combination is why the US gets the lock-in effect: existing owners are protected *and* handcuffed, so a rate spike freezes supply rather than crushing payments.

Contrast this with much of the rest of the world. In the UK, mortgages are typically fixed for only 2-5 years and then reset to the prevailing rate; in Canada, 5-year fixes that must be renewed; in Australia and much of Europe, outright *floating-rate* loans that reprice with the policy rate within months. In these markets there is no lock-in cushion. When the central bank hikes, *existing borrowers' payments rise* — sometimes brutally — as their fixed period expires or their floating rate resets. That forces distressed selling and genuine price declines, which is exactly why the UK, Canada, Australia, New Zealand, and Sweden all saw sharper *price* corrections in 2022-23 than the US did, despite similar rate moves. *The very same policy lever produces a frozen market in one country and a price crash in another, purely because the plumbing of the mortgage differs — a reminder that transmission depends on institutions, not just on the rate.*

The US 30-year fixed is itself a *policy artifact*: it exists at scale only because the government, through Fannie Mae and Freddie Mac, guarantees and securitizes these loans, making investors willing to fund a 30-year fixed-rate, prepayable instrument that private lenders would otherwise never offer. So even the *structure* that creates the lock-in effect is a product of housing policy. (For the legal and regulatory architecture of the GSEs and housing finance, see [real estate and housing law](/blog/trading/law-and-geopolitics/real-estate-and-housing-law-zoning-rent-control-and-the-gses).)

### The lock-in effect: when low rates freeze the market

Here is the second-order effect that surprised even seasoned economists. When mortgage rates spiked, the obvious prediction was that house prices would fall hard. They did not — at least not nearly as much as the affordability math implied. The reason is one of the most elegant and underappreciated consequences of monetary policy on real estate: the **lock-in effect** (sometimes called the "golden handcuffs").

In the US, the standard mortgage is a *30-year fixed* loan, and crucially it is *not portable* — when you sell your house, you pay off the old mortgage and take out a brand-new one on the next house at today's rate. So consider a homeowner who refinanced into a 3% loan in 2021. To move — even across the street, even to a cheaper house — they would have to give up that 3% loan and borrow at 7% on the next one. For most people, that is financially irrational: the monthly payment on the *same loan balance* would jump enormously, so they stay put. Multiply that decision across tens of millions of households and the result is a *supply freeze*: existing homes simply stop coming onto the market, because the people who own them are handcuffed to their cheap loans.

![Comparison matrix contrasting staying with a 3 percent loan versus selling and rebuying at 7 percent across rate, payment, annual cost, and the rational choice](/imgs/blogs/how-policy-prices-real-estate-cap-rates-and-mortgages-8.png)

The matrix above lays out the homeowner's decision. Stay, and you keep a 3% loan with its low payment. Sell and rebuy the same-priced house, and you face a 7% loan — about \$975 more a month on a \$400k balance, roughly \$11,700 a year, for the privilege of living in a house no nicer than the one you have. The rational choice in the bottom row is the same for almost everyone: *do not move.* And when nobody moves, the supply of existing homes for sale collapses.

#### Worked example: the lock-in math for a \$400,000 loan

Take a homeowner with a \$400,000 mortgage balance at 3.0% who is thinking about selling and buying a similar house, which would require a new \$400,000 mortgage at the 2023 market rate of 7.0%.

- Current payment at **3%**: about **\$1,686 a month**.
- New payment at **7%**: about **\$2,661 a month**.
- The difference: **\$975 a month**, or about **\$11,700 a year**, *for an equivalent house with no upgrade*.

To justify moving, the homeowner would need a reason worth \$11,700 a year after tax — a job relocation, a growing family, a divorce. Ordinary "we'd like a slightly bigger yard" moves simply stop happening. *The low-rate mortgage the Fed encouraged in 2021 became, in 2023, a cage that locked owners in place and starved the market of supply — a policy effect that operates through human behavior, not through the valuation formula.* This is why, paradoxically, a brutal affordability shock produced *frozen sales volume and resilient prices* rather than the price crash the payment math alone would predict. Existing-home sales fell to their lowest level in nearly thirty years, even as prices barely dipped.

## The cap-rate channel: how a rising risk-free rate re-rates buildings down

The mortgage channel governs residential, where buyers are payment-driven. Commercial real estate — offices, warehouses, apartment complexes, shopping centers bought by investors — works through a different but parallel mechanism: the **cap rate**. This is where property most resembles a bond, and where a rising risk-free rate does its most visible damage.

The capitalization rate, or cap rate, is defined with disarming simplicity:

```
cap rate = NOI / price
```

It is the building's income yield — what percent of the purchase price you earn in net rent each year before financing. A building generating \$1 million of NOI that sells for \$20 million has a 5% cap rate (\$1M / \$20M). Rearrange the same equation and you get the valuation formula, the single most important line in commercial real estate:

```
value = NOI / cap rate
```

This is the perpetuity formula in disguise — the same `C / r` that values any infinite stream of cash flows. The building is treated as a stream of NOI that goes on essentially forever, and the cap rate is the discount rate you divide by. And just like a bond, *value moves inversely to the rate*: a higher cap rate means a lower price for the same income. Now the crucial question — what *sets* the cap rate? The answer is what links it to policy:

```
cap rate ≈ risk-free rate + risk premium − NOI growth
```

Read that carefully, because it is the whole channel. The cap rate is the *risk-free rate* (the 10-year Treasury) — the yield you could earn with no risk — *plus* a **risk premium** (extra yield demanded for the building's vacancy risk, illiquidity, and uncertainty) *minus* the **growth** you expect in NOI (because a building whose rents will grow can justify a lower current yield, the way a growth stock justifies a low dividend yield). When the Fed pushes up the risk-free rate, the first term rises, so the cap rate rises, so — holding NOI fixed — *value falls*. Mechanically, with not a single tenant lost.

![Before-and-after panels showing a building with one million dollars of NOI valued at twenty million at a five percent cap rate and fourteen point three million at a seven percent cap rate](/imgs/blogs/how-policy-prices-real-estate-cap-rates-and-mortgages-5.png)

The before-and-after panels show the re-rating in its purest form. On the left, the "before": \$1 million of NOI, a 5% cap rate, a value of \$20 million. On the right, the "after": the *same* \$1 million of NOI, but now investors demand a 7% cap rate because the risk-free rate rose two points — and the value falls to \$14.3 million. Nothing about the building changed. The same tenants pay the same rent. But the market's required yield re-rated, and 28% of the value evaporated. This is the discount-rate channel in its most brutal form, because property carries no earnings-growth surprise to cushion the blow the way a stock sometimes does.

#### Worked example: a \$1 million NOI building re-rated from a 5% to a 7% cap rate

Take an office building producing \$1 million of net operating income a year.

- At a **5%** cap rate: `value = $1,000,000 / 0.05 = `**`$20,000,000`**.
- At a **7%** cap rate: `value = $1,000,000 / 0.07 = `**`$14,290,000`**.

The value fell from \$20.0M to \$14.3M — a loss of **\$5.7 million, or about 28%** — driven entirely by a 2-point rise in the cap rate, with the income unchanged. And note the leverage angle: if that building had been bought with 65% debt (\$13M loan, \$7M equity), the \$5.7M value loss does not fall on the \$20M — it falls on the \$7M of equity, which would be cut by more than 80%. *A 28% drop in the asset became an ~80% drop in the owner's equity — leverage turning a re-rating into a wipeout, which is exactly how so many commercial-property owners ended up handing buildings back to lenders in 2023-24.*

![Horizontal bar chart of representative US cap rates by property sector showing apartments and industrial lowest and office highest](/imgs/blogs/how-policy-prices-real-estate-cap-rates-and-mortgages-6.png)

Not every property type re-rated equally, and the bar chart shows why. Cap rates vary by sector because the *risk premium* and the *growth* terms differ. Apartments and industrial/warehouse (the logistics boom, e-commerce demand) carry the lowest cap rates — around 5.4-5.6% — because investors see steady demand and rent growth, so they accept a thinner yield. Retail sits higher at ~6.8%. And office sits highest of all, around 8.5%, because the work-from-home shift after 2020 did something no rate move alone could: it *cut the growth term and raised the risk premium at the same time*. Demand for office space structurally fell, vacancy rose, and investors demanded a fat yield to touch it. So office got hit by a triple whammy — a higher risk-free rate, a higher risk premium, *and* a negative growth outlook — which is why office values fell 25-40% while apartments held up far better. The cap-rate formula explains the entire cross-section.

### The refinancing cliff: where commercial leverage detonates

Residential's release valve is the lock-in freeze. Commercial real estate has the opposite problem — a *forced* repricing event built into the structure of the debt — and it is worth understanding because it is where rate policy turns a paper loss into a real default.

Unlike a 30-year fixed home loan, commercial mortgages are typically short — 5, 7, or 10 years — and **interest-only or partially-amortizing**, with a large **balloon payment** of principal due at maturity. The owner does not pay the loan down; they refinance it with a new loan when it comes due. That works beautifully when rates are stable or falling. It becomes a catastrophe when a loan taken out at 3.5% in 2019 or 2021 comes due in 2024 and must be refinanced at 7-8% — on a building that, thanks to the cap-rate re-rating, is now worth less than the loan balance. This bunching of maturities into a high-rate window is the **maturity wall** or **refinancing cliff**, and trillions of dollars of US commercial mortgages hit it in 2024-26.

The mechanism that bites is the **debt-service coverage ratio (DSCR)** — the building's NOI divided by its annual loan payment. Lenders require NOI to cover the payment with a cushion (a DSCR of, say, 1.25×). When the refinance rate doubles, the required payment doubles, and the same NOI may no longer cover it — so the lender will not refinance the full balance, forcing the owner to inject fresh cash (a "cash-in refinance") or hand the building back.

#### Worked example: a refinancing cliff at maturity

Take an office building with **\$1 million** of NOI and a **\$15 million** interest-only loan taken out at **3.5%**, coming due for refinancing.

- The old annual interest payment at **3.5%** was `$15,000,000 × 0.035 = `**`$525,000`**, comfortably covered by the \$1M NOI (a DSCR of 1.9×).
- At the new market rate of **7.5%**, the interest on the same \$15M would be `$15,000,000 × 0.075 = `**`$1,125,000`** — *more than the entire \$1M of NOI* (a DSCR of 0.89×, below 1.0).

The building no longer earns enough to cover the interest on its existing debt at current rates. The lender will refinance only a smaller balance that the NOI can support — say, an amount whose payment fits a 1.25× DSCR, which at 7.5% is a loan of about `$1,000,000 / 1.25 / 0.075 ≈ $10.7 million`. To refinance, the owner must come up with the **\$4.3 million** gap in cash — or default. *Rising rates did not merely lower the building's value on paper; they made the existing debt mathematically un-refinanceable, converting a valuation problem into a default — which is precisely the dynamic that put office loans at the center of the 2023-24 bank stress.*

### Where property is *not* a pure bond: the growth term and inflation

If property were nothing but a bond, the story would end with the cap-rate formula and a uniformly grim verdict on rising rates. But the "leveraged bond with a roof" has one feature a real bond lacks, and it lives in the *growth* term of the cap-rate equation: **a bond's coupon is fixed forever, but a building's rent resets.** Leases expire, and when they do, the landlord re-prices to the current market — often upward, especially during inflation. So NOI is not the frozen coupon of a Treasury; it is a stream that can *grow*, and that growth is what makes real estate a partial **inflation hedge**.

This is the saving grace that separated the winners from the losers in 2022-24. Consider why apartments and industrial held up while office cracked. Apartment leases turn over every year, so when inflation pushed up rents, landlords captured it almost immediately — NOI grew, partly offsetting the cap-rate re-rating. Industrial leases, riding the e-commerce and reshoring boom, saw double-digit rent growth that more than offset the higher discount rate for many owners. Office, by contrast, had long leases (5-10 years) locked at pre-2020 rents *and* collapsing demand, so its NOI was falling exactly when its cap rate was rising — both terms working against it. The same rate shock that devastated office barely dented industrial, because one had growing rent and the other had shrinking rent.

The lesson sharpens the whole framework: the rate move sets the *headwind* (a higher cap rate, lower value for fixed income), but the *rent-growth* of the specific property determines whether that headwind is survivable. A property whose rents reprice quickly and upward — short leases, scarce supply, inflationary demand — can grow its way through a rate shock. A property with long fixed leases and weak demand has no defense; it is, for the duration of those leases, an actual fixed-coupon bond, and it takes the full force of the discount-rate channel. This is why "real estate is an inflation hedge" is true *on average and over time*, but false in the short run for any property whose rents are locked. The hedge lives entirely in the growth term, and the growth term is specific to the building.

## Macroprudential policy: when regulators set property prices directly

The mortgage and cap-rate channels are how the *price* of money reaches property. But governments have a second, blunter lever that bypasses the interest rate entirely and acts *directly* on how much can be borrowed against a property. These are **macroprudential** tools — rules a central bank or regulator imposes on lending itself, designed to cool a property market or protect the banking system. The two most common are the **loan-to-value (LTV) cap** and the **debt-service-to-income (DSR/DSTI) cap**.

A loan-to-value cap limits how much you can borrow as a fraction of the property's price. If the LTV cap is 80%, you must put down at least 20%; if regulators tighten it to 60%, you must put down 40%. This is a direct lever on the price a buyer can pay, because most buyers are constrained by the *cash they have for a down payment*, not by their income. Lower the maximum LTV and you instantly shrink the pool of money chasing properties. Countries from South Korea to Singapore to New Zealand have used LTV caps to deflate housing bubbles without touching the policy rate — a surgical tool aimed at property alone. (For the broader toolkit of capital, liquidity, and lending rules regulators use to steer markets, see the companion post on [macroprudential and regulatory policy](/blog/trading/policy-and-markets/macroprudential-and-regulatory-policy).)

#### Worked example: an 80% to 60% LTV cap on a buyer's maximum price

Suppose a buyer has **\$200,000** in cash for a down payment and wants to buy the most expensive property they can.

- Under an **80% LTV** cap, the \$200,000 is the required 20% down, so it supports a purchase price of `$200,000 / 0.20 = `**`$1,000,000`** (a \$800,000 loan).
- Tighten the cap to **60% LTV**, and the \$200,000 must now be the required 40% down, so it supports only `$200,000 / 0.40 = `**`$500,000`** (a \$300,000 loan).

The buyer's wealth did not change — they still have exactly \$200,000. But the regulator, by tightening one ratio, cut the maximum price that cash can buy *in half*, from \$1,000,000 to \$500,000. *A macroprudential rule is the most direct policy lever on property prices there is: it does not work through expectations or yields, it simply caps the number — which is why regulators reach for it when they want to cool housing without raising rates on the whole economy.* A DSR cap works the same way on the income side, limiting the loan to a multiple of income; both compress the price a constrained buyer can pay, immediately.

## REITs: where property gets marked to market every second

There is one corner of real estate that does *not* hide its volatility behind slow appraisals: **real estate investment trusts (REITs)** — companies that own portfolios of property and trade on the stock exchange like any equity. A REIT is just a basket of buildings wrapped in a stock, so its price is the market's *real-time* opinion of what that property is worth, updated every second the exchange is open. And because REITs are leveraged property exposure trading on a screen, they are one of the most rate-sensitive corners of the entire stock market — they are the place the cap-rate channel shows up *instantly* instead of with an 18-month appraisal lag.

This makes REITs a useful leading indicator. When the Fed began hiking in 2022, listed REITs fell hard and fast — the office REITs worst of all, some down 50-60% — long *before* the private appraisals of the same buildings caught up. The public market had already done the cap-rate math; the private market just had not marked it yet. The gap between where REITs trade and where private appraisals sit is itself a signal: a wide discount tells you the public market expects further cap-rate re-rating that the appraisals have not yet acknowledged. REITs also pay out most of their income as dividends (by law, to keep their tax status), which makes them behave even more like long-duration bonds — their dividend yield competes directly with the Treasury yield, so when the risk-free rate rises, REIT prices fall until their yield is competitive again. It is the cap-rate channel, transposed into a stock price, visible in real time.

#### Worked example: a REIT's yield competing with the Treasury

Suppose a REIT pays a **\$4** annual dividend and trades at **\$100**, a **4% dividend yield**, back when the 10-year Treasury yielded 1.5% — a comfortable 2.5-point spread for taking property risk.

- Now the 10-year Treasury rises to **4.5%**. For the REIT to keep offering the same 2.5-point risk premium, it would need to yield **7%**.
- Holding the \$4 dividend fixed, a 7% yield means a price of `$4 / 0.07 = `**`$57`** — a **43% fall** in the share price.

The buildings the REIT owns did not change, and the dividend did not change. But the risk-free rate rose three points, so the price the market would pay for that \$4 income stream collapsed to keep the yield competitive. *A REIT is the cap-rate re-rating made visible on a stock ticker — the same NOI-divided-by-a-rising-yield arithmetic, just marked to market in real time instead of every eighteen months by an appraiser.* (For the general mechanics of how rising rates compress equity valuations, see [how policy prices equities](/blog/trading/policy-and-markets/how-policy-prices-equities-the-multiple-and-the-earnings).)

## Common misconceptions

**"Real estate is a safe, low-volatility asset."** The *prices* look smooth because property does not trade on a screen every second — appraisals are infrequent and lag the market, so the volatility is hidden, not absent. Once you mark a building to what it would actually sell for, real estate is *more* rate-sensitive than most stocks. US office values fell 25-40% in 2022-24, and commercial real estate as a whole drew down ~20% — comparable to a serious equity bear market, just reported with a delay. The smoothness is an accounting artifact.

**"If rents don't fall, my building's value is safe."** This is the cap-rate trap. Value is NOI *divided by* the cap rate, and the cap rate is driven by the risk-free rate, which the building's landlord does not control. The \$1M-NOI building above lost 28% of its value with rents perfectly flat, purely because the Fed raised rates and the cap rate re-rated up. Stable income is no protection against a rising discount rate.

**"Higher mortgage rates always crash home prices."** Not necessarily — and 2022-24 is the proof. Affordability collapsed (a 37% loss of purchasing power in the worked example), which *should* have crashed prices. But the lock-in effect froze *supply* at the same moment, so the market cleared on collapsed sales volume rather than collapsed prices. Whether a rate spike crashes prices or freezes volume depends on the mortgage market's structure — countries with portable or floating-rate mortgages (the UK, Australia, Canada) saw far more price pain, because there was no lock-in to choke off supply.

**"Cap rates and interest rates move one-for-one."** They are linked but not glued. The cap rate is risk-free *plus a premium minus growth*, and both the premium and the growth term move independently. In 2021, cap rates *fell* even as the economy recovered, because the growth outlook and risk appetite improved faster than rates rose. The risk-free rate sets the floor and the trend; the spread above it does its own thing.

**"Lower rates are always good for property."** Lower rates lift values, yes — but they also encourage the leverage and the lock-in that make the *next* rate rise more dangerous. The 2.68% mortgages of 2020-21 were a gift that became a trap: they pumped up prices, then handcuffed owners and set up the office leverage that blew up when rates normalized. Policy that is too easy for too long builds the fragility that policy tightening then exposes.

**"Residential and commercial real estate move together."** They share the rate channel but diverge sharply in how it transmits, and 2022-24 proved it. Residential, on long fixed-rate non-portable mortgages, *froze* — volume collapsed but prices held. Commercial, on short balloon-payment loans that must be refinanced, *broke* — forced repricing and defaults. And within commercial, office cratered while industrial and apartments held. "Real estate" is not one asset; it is a family of assets that share a discount rate but differ enormously in lease structure, leverage structure, and demand outlook — which is exactly why the cap-rate formula, with its separate risk-premium and growth terms, is the only honest way to value any single one of them.

**"A higher cap rate means a property is cheaper, so it's a better buy."** Not by itself. A high cap rate can mean a genuine bargain — or it can mean the market is pricing in falling rents and high risk (office in 2024 had high cap rates *because* it was distressed, not because it was cheap). The cap rate is risk-free plus a risk premium minus growth; a fat cap rate driven by a fat risk premium and negative growth is a warning, not a discount. You have to decompose *why* the cap rate is high before you can judge whether the price is attractive.

## Case studies: three real repricings

### The 2021→2023 mortgage spike and the great freeze

The cleanest case study is the one that just happened. Through 2020-21, the Fed cut its policy rate to a 0.25% floor and bought hundreds of billions of dollars of mortgage bonds, driving the 30-year mortgage to a record-low **2.68%** in December 2020. Cheap money inflated home prices roughly 40% nationally over 2020-22. Then inflation arrived, and the Fed reversed hard: **525 basis points** of hikes in sixteen months, the fastest cycle since Volcker, dragging the 10-year Treasury — and the mortgage rate riding on top of it — up to **7.79%** by October 2023, the highest since 2000.

The affordability shock was historic. The monthly payment on a median-priced home roughly doubled from its 2021 low. By the purchasing-power math, buyers had lost more than a third of what their income could borrow. And yet national home *prices* barely fell — they dipped a few percent and then resumed rising in many markets. The release valve was the **lock-in effect**: with the typical existing homeowner sitting on a sub-4% mortgage, almost nobody listed their home, and existing-home sales collapsed to their lowest level in nearly three decades. The market did not crash; it *froze*. This is policy reshaping the largest asset class on earth not through a clean price signal but through a behavioral handcuff — a consequence the affordability models entirely missed. (For how the same rate cycle hit stocks through the discount rate, see [how policy prices equities](/blog/trading/policy-and-markets/how-policy-prices-equities-the-multiple-and-the-earnings); for the trader's playbook on the rate-and-credit cycle, see [how monetary policy moves real estate and credit](/blog/trading/macro-trading/how-monetary-policy-moves-real-estate-and-credit).)

### The office cap-rate re-rating, 2022-2024

If residential froze, commercial *broke* — and office most of all. Through 2022-24, the same rate cycle pushed commercial cap rates up across the board, but office got hit by three forces at once. First, the **risk-free rate** rose ~5 points, lifting the base of every cap rate. Second, the **work-from-home** shift after 2020 structurally cut demand for office space, raising vacancy and gutting the *growth* term in the cap-rate formula — landlords could no longer assume rents would climb; many faced falling rents and tenants not renewing. Third, the elevated vacancy and uncertainty fattened the **risk premium** investors demanded. The result, visible in the sector bar chart above, is that office cap rates blew out to ~8.5% while apartments and industrial held near 5.4-5.6%.

Run the formula and the carnage is no surprise: a building whose cap rate goes from ~6% to ~8.5% loses roughly 30% of its value from the re-rating alone, *before* counting any drop in NOI from rising vacancy. Count the NOI hit and the worst towers fell 40-50%+. Because these buildings were bought with heavy leverage, the equity was frequently wiped out entirely, and a wave of owners — including some of the largest institutional landlords — simply stopped paying and handed keys back to lenders. The regional banks that had financed them took the losses; office-heavy lending became the single biggest stress point in the 2023 banking scare. One rate variable, working through the cap-rate channel, repriced an entire asset class and rippled into the banking system. (For the credit-spread side of that stress, see [the discount-rate channel](/blog/trading/policy-and-markets/the-discount-rate-channel-how-rates-reprice-cash-flows).)

### Vietnam 2022: a property crackdown meets a rate-defense

The third case shows the same machine running in an emerging market, with an extra lever the US does not use. In 2022, Vietnam's property sector — long financed by a booming corporate-bond market — hit a wall when the government launched an **anti-fraud crackdown** on bond issuance, arresting executives at major developers (the Tan Hoang Minh and Van Thinh Phat cases). That froze the developers' funding overnight. At the same time, the global tightening cycle forced the State Bank of Vietnam (SBV) to **defend the dong**: it hiked its policy rates by **+200 basis points** in September and October 2022 and let the currency slide past 24,000 per dollar before defending it. Vietnamese banks also operate under an SBV-set **credit growth quota** — a hard ceiling on how much each bank may lend each year — which, when binding, chokes off property credit directly, the way an LTV cap does in other countries.

The collision of a credit squeeze, a rate-defense, and a funding crackdown hit the most rate-sensitive asset class hardest. The VN-Index fell from a **1,528** high in April to an intra-year low of **874** in mid-November — a **-32.8%** year, one of the worst in the world that year — with property and bank stocks at the epicenter, because property *is* the collateral and the credit in Vietnam's financial system.

The Vietnam case illustrates a lever the US simply does not have: the **credit growth quota**. Each year the SBV assigns every commercial bank a ceiling on how much its loan book may grow — 14%, say, for the system as a whole, parceled out bank by bank. When that quota is loose, credit floods into property and prices rise; when the SBV tightens it, or when banks hit their ceiling mid-year, property lending stops *regardless of the interest rate*. It is a quantity control rather than a price control — a direct rationing of credit that acts on property even more bluntly than an LTV cap, because it can shut off the flow of new loans entirely. In 2022, the combination of a binding quota, the bond-market crackdown, and the rate hikes removed three separate sources of property funding at once. There was simply no money left to bid for property, and the most leveraged developers — those who had funded themselves with short-term bonds rolled over indefinitely — could not refinance and collapsed.

There is a second lesson in the Vietnam case that generalizes to every emerging market: when a country must **defend its currency**, it loses control of its domestic property cycle. The SBV did not hike in 2022 because Vietnamese inflation demanded it; it hiked because the dong was sliding and capital was leaving, and a higher domestic rate is the classic defense — it makes holding the local currency more attractive. But that defensive hike landed on a property sector that was already starved of credit, deepening the squeeze. This is the emerging-market bind in miniature: external pressure (a strong dollar, a hawkish Fed) forces a rate-defense, and the rate-defense reprices the most rate-sensitive domestic asset — property — even when the domestic economy would have preferred easier policy. (For how rate differentials and currency defense interact, see [the currency channel](/blog/trading/policy-and-markets/the-currency-channel-rate-differentials-and-carry).)

![Dual-axis chart of the VN-Index falling through 2022 while the dong weakens past 24,000 per dollar amid SBV rate hikes](/imgs/blogs/how-policy-prices-real-estate-cap-rates-and-mortgages-7.png)

The chart traces the year: the VN-Index (red, left axis) sliding from its 1,528 peak as the dong (blue, right axis) weakened toward 24,800 per dollar at the October stress point before the SBV defended it back. It is the same policy-to-property machine — a credit-and-rate squeeze repricing the most leveraged asset class — running through Vietnam's particular toolkit of bond enforcement, rate hikes, FX defense, and credit quotas. (For the legal and institutional machinery behind that toolkit, see [SBV monetary and banking law](/blog/trading/law-and-geopolitics/sbv-monetary-and-banking-law-credit-quotas-and-the-dong).)

## What it means for asset values: the playbook

Put the whole machine together and a few concrete rules fall out for anyone trying to read how policy will reprice property.

**When the policy rate is rising**, expect both channels to push property down. Mortgage rates climb, affordability erodes, and buyer purchasing power shrinks — but watch the *mortgage structure* to know whether you get a price crash or a volume freeze: fixed-rate, non-portable markets (the US) freeze; floating-rate or portable markets (UK, Canada, Australia) crash faster. On the commercial side, cap rates re-rate up roughly with the risk-free rate, so values fall about `−(Δcap rate / cap rate)` for the same income — a 2-point cap-rate rise from a 6% base is roughly a 25% value hit before any NOI change. The most leveraged owners get wiped out first; the banks that financed them are the second-order casualty to watch.

**When the policy rate is falling**, the machine runs in reverse: mortgage rates drop, the lock-in thaws (owners can finally move), supply unfreezes, and cap rates compress so values rise for the same income. This is generally bullish for property, but the most rate-sensitive sectors (the ones with the longest income duration and highest leverage) move most in *both* directions.

**The signals to watch** are the 10-year Treasury yield (the base of both the mortgage and the cap rate), the mortgage-to-Treasury spread (a fattening spread signals mortgage-market stress, as in 2022-23), the level of the cap rate relative to the risk-free rate (a thin spread means property is priced for perfection and vulnerable), and — uniquely for real estate — *sales volume*, because a frozen market with stable prices is telling you the lock-in effect is absorbing a shock that has not yet shown up in price.

**The feedback loop back into the economy** is why central banks watch property so closely in the first place. Real estate is not just an asset that policy reprices — it is one of the main channels through which policy *works*. When higher rates lower home values, households feel poorer and spend less (the "wealth effect"), cooling demand exactly as the central bank intends. When property values fall, the collateral behind bank loans shrinks, banks tighten lending, and credit contracts across the whole economy. And construction — a large, rate-sensitive, employment-heavy industry — slows when financing gets expensive, directly cutting GDP and jobs. So the property channel is *part of the transmission mechanism itself*: the central bank reprices real estate in order to slow the economy. The 2022-24 cycle is the textbook case — the Fed hiked to cool inflation, property repriced, construction stalled, bank lending tightened, and the slowdown the Fed wanted arrived partly *through* the housing market. This is also why a property crash is so dangerous: the same feedback loop that helps policy work in moderation becomes a doom loop in excess, as falling values, shrinking collateral, and tightening credit reinforce each other downward — the mechanism behind 2008.

**What would invalidate the read**: a sharp move in the *risk premium* or *growth* term that swamps the rate move. Cap rates fell in 2021 even though the recovery was underway, because risk appetite and the growth outlook improved faster than rates. And a structural demand shock — like work-from-home for offices — can re-rate a sector independent of the policy rate entirely. The rate sets the floor and the trend; the spread above it has a mind of its own. Real estate is a leveraged bond with a roof, and most of the time the bond math wins — but never forget the roof, the leverage, and the human being deciding whether to move.

## Further reading and cross-links

- [The discount-rate channel: how rates reprice cash flows](/blog/trading/policy-and-markets/the-discount-rate-channel-how-rates-reprice-cash-flows) — the foundational present-value engine that the cap-rate formula is a special case of.
- [Macroprudential and regulatory policy](/blog/trading/policy-and-markets/macroprudential-and-regulatory-policy) — LTV/DSR caps, capital rules, and the direct levers on credit and property.
- [How policy sets the bond market: the yield curve](/blog/trading/policy-and-markets/how-policy-sets-the-bond-market-the-yield-curve) — how the policy rate becomes the 10-year Treasury that anchors mortgages and cap rates.
- [The liquidity channel: QE, QT, and the everything bid](/blog/trading/policy-and-markets/the-liquidity-channel-qe-qt-and-the-everything-bid) — how the Fed's mortgage-bond purchases drove rates to record lows and back.
- [How monetary policy moves real estate and credit](/blog/trading/macro-trading/how-monetary-policy-moves-real-estate-and-credit) — the trader's positioning playbook for the rate-and-credit cycle.
- [SBV monetary and banking law: credit quotas and the dong](/blog/trading/law-and-geopolitics/sbv-monetary-and-banking-law-credit-quotas-and-the-dong) — the legal machinery behind Vietnam's credit-quota and FX toolkit.

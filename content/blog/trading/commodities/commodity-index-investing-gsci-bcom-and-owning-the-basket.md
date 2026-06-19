---
title: "Commodity Index Investing: GSCI, BCOM, and Owning the Basket"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "What an institution actually buys when it wants commodities exposure: a fully-collateralized basket of rolled futures, where the GSCI is mostly an oil bet and BCOM is the truer diversifier."
tags: ["commodities", "index-investing", "gsci", "bcom", "roll-yield", "asset-allocation", "futures", "diversification", "inflation-hedge", "collateral-yield", "financialization"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — When an institution says it owns "commodities," it almost never owns a barrel or a bushel; it owns a fully-collateralized basket of near-month futures that get mechanically rolled forward each month, and the index's return is spot move plus roll yield plus collateral yield.
>
> - The two big benchmarks weight the basket in opposite ways. The **S&P GSCI** is production-weighted, so it is roughly **62% energy** and behaves like a leveraged long-oil position. The **Bloomberg Commodity Index (BCOM)** is capped and diversified, with energy near **30%** and meaningful weights in ags, metals, and precious — a far truer cross-commodity diversifier.
> - "Owning the index" is not owning the spot price. You own **futures that must be rolled**, so in a contango market (back months dearer) the long-only roll quietly **bleeds**, and in backwardation it **earns**. The roll is the single most important thing the brochure underplays.
> - Because the cash that backs the futures sits in T-bills, **collateral yield** is a real and currently large part of the return — at a 5% short rate, that is roughly 5 points a year before the commodities even move.
> - The one number to remember: the GSCI is about **62% energy**. If you "own commodities" through it, you mostly own oil, plus a roll bet, plus a cash yield.

In the spring of 2004, a quiet thing started happening on the world's commodity futures exchanges. Pension funds, endowments, and insurance companies — institutions that had never traded a barrel of oil or a bushel of corn in their lives — began wiring billions of dollars into a new kind of product: the commodity index swap. A landmark academic paper, Gary Gorton and Geert Rouwenhorst's "Facts and Fantasies about Commodity Futures," had just argued that a basket of commodity futures earned equity-like returns with bond-like volatility and, crucially, rose when stocks and bonds fell. Allocators read it, nodded, and went shopping. By 2008, an estimated \$200 billion or more had poured into long-only commodity index products. Then oil ran from roughly \$60 to \$147 a barrel, food riots broke out from Haiti to Bangladesh, and Congress hauled the "index speculators" in front of a Senate subcommittee to ask whether all that passive money had broken the world's commodity markets.

That story is the subject of a sibling post on [the financialization of commodities](/blog/trading/commodities/the-financialization-of-commodities-when-wall-street-arrived). What this post is about is the *vehicle* — the thing all that money was buying. When a \$50 billion pension fund decides it wants "5% in commodities," it does not call a grain elevator. It buys an index. And an index, it turns out, is a surprisingly mechanical, surprisingly leaky machine: a basket of futures, picked by a weighting rule, rolled on a published schedule, backed by a pile of Treasury bills. Understanding that machine — what it owns, how it rolls, why it lags the spot price, and why two of these indices can disagree wildly about what "commodities" even means — is the difference between an allocator who knows what they hold and one who is surprised every quarter.

![A commodity index is a collateralized basket of rolled futures from weights to total return](/imgs/blogs/commodity-index-investing-gsci-bcom-and-owning-the-basket-1.png)

The diagram above is the whole machine in one line. An index picks weights, buys front-ish futures, rolls them every month, parks the unspent cash in T-bills, and reports the sum as a single total-return number. Three of those gears — the spot move, the roll yield, and the collateral yield — each contribute to what you actually earn. Miss any of them and you will misread your own portfolio.

## Foundations: what "owning the commodity index" actually means

Let us build this from absolute zero, because almost every confusion about commodity investing comes from skipping these definitions.

**A commodity** is a physical, fungible good — crude oil, copper, corn, gold, live cattle — that trades on standardized terms. One barrel of West Texas Intermediate crude is interchangeable with any other barrel that meets the spec. (For the full definition and why fungibility matters, see [what is a commodity](/blog/trading/commodities/what-is-a-commodity-the-physical-asset-that-trades-on-paper).) The defining feature for an investor is awkward: commodities are *consumption and industrial* assets, not monetary ones. They cost money to store, they can rot or evaporate, and they pay no dividend or coupon. A share of stock represents a claim on a stream of future profits; a bond pays interest; a barrel of oil just sits there, costing you tank rent. That single fact drives everything that follows.

**A futures contract** is a standardized agreement to buy or sell a fixed quantity of a commodity at a fixed price on a fixed future date. If you buy the December WTI future at \$72, you have locked in a price of \$72 per barrel for delivery in December. You post a small good-faith deposit (margin), not the full \$72, and the contract is marked to market daily. The vast majority of futures are *never* delivered — traders close them out before expiry. This is the instrument the index actually holds. The index does not own oil; it owns the *promise* of oil, written on paper. (The deep mechanics of why the futures price and the spot price differ are covered in [spot vs futures](/blog/trading/commodities/spot-vs-futures-the-two-prices-of-the-same-barrel).)

**A commodity index** is a rules-based recipe for combining many such futures into one number. The recipe specifies three things:

1. **Which commodities and how much of each** — the *weights*. The S&P GSCI and Bloomberg BCOM differ here more than anywhere else, and it is the most consequential choice the index makes.
2. **Which contract month to hold** — usually the *front* (nearest-to-expiry) or near-front contract, because that is the most liquid.
3. **When and how to roll** — the schedule for selling the expiring contract and buying the next one before delivery is forced.

**Fully collateralized** is the phrase that turns a leveraged futures bet into an investable asset. A futures position is inherently leveraged: you control \$72 of oil for maybe \$5 of margin. A *fully collateralized* index does not use that leverage. For every \$72 of oil exposure, it holds \$72 of actual cash, almost always in Treasury bills. So the index investor takes the full price exposure of the commodity, with no embedded borrowing — and the cash earns the T-bill rate while it waits. This is why a commodity index has *three* return components, not one.

**Total return** of a fully-collateralized long-only commodity index decomposes cleanly:

```
total return = spot return        (the change in the front contract's price)
             + roll yield         (gain or loss from rolling front -> next month)
             + collateral yield   (T-bill interest earned on the cash backing it)
```

Almost everyone who is new to this thinks "I bought commodities, so my return is the change in the oil price." That is the *spot return* — only one of the three pieces. The roll yield can be large and is usually negative for long-only investors. The collateral yield is currently several percent a year and is pure tailwind. We will dollar-out every piece below.

### The roll, in one paragraph

Because a futures contract expires, a long-only index that wants to stay invested cannot just buy and hold. As the front contract nears its delivery date, the index must **sell it and buy the next contract out** — this is "rolling." If the next contract costs *more* than the one you are selling (an upward-sloping forward curve, called **contango**), you are perpetually selling low and buying high: the roll *bleeds*. If the next contract costs *less* (a downward-sloping curve, **backwardation**), you sell high and buy low: the roll *earns*. The shape of the forward curve therefore decides whether holding the index is a tailwind or a headwind, entirely separate from whether the price of oil goes up or down. This is the single most important — and most under-explained — feature of commodity index investing, and it gets its own dedicated treatment in [roll yield and why long-only commodity ETFs bleed](/blog/trading/commodities/roll-yield-and-why-long-only-commodity-etfs-bleed). The curve's *shape* and its causes (storage cost, the convenience yield) are explored in [contango vs backwardation](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) and [the forward curve](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities).

Hold those three components in mind. Everything from here is just filling in the numbers.

### The annual rebalance, and why energy dominates "production value"

One more piece of plumbing before we compare the two benchmarks. The weights are not frozen forever — both indices **rebalance once a year** (typically in January), recalculating each commodity's weight from updated production and liquidity data and resetting the basket to those targets. Through the year, weights drift as prices move (a sector that rallies hard becomes a larger share until the next rebalance), and then January snaps them back to the rule. This annual reset is itself a mechanical buy-low-sell-high: the rebalance trims whatever rallied and adds to whatever lagged, a small "rebalance bonus" that diversified indices like BCOM capture more than concentrated ones.

Now, *why* does production-weighting hand energy such a crushing share? Because the GSCI weights by the **dollar value of world production**, and energy's production value swamps everything. Run the rough numbers: the world consumes on the order of 100 million barrels of oil a day, and at, say, \$75 a barrel that is about \$7.5 billion *per day*, or roughly \$2.7 trillion a year — for crude alone, before refined products and natural gas. Compare that to copper: world mine output is around 22 million tonnes a year at roughly \$9,000 a tonne, or about \$0.2 trillion — an order of magnitude smaller. Gold's *annual mine production* is a few thousand tonnes worth well under \$0.3 trillion. So when you weight by what the world actually produces and burns in dollar terms, oil is simply in a different league, and any production-weighted index inherits that lopsidedness. That arithmetic — energy's production value dwarfing the rest — is the deep reason the GSCI is an oil bet, and it is not an accident of one year's prices; it is structural.

## The two benchmarks: GSCI versus BCOM

There are dozens of commodity indices, but two dominate the institutional world, and they could not be more different in philosophy. The disagreement is entirely about *weights*, and the weights are entirely about one design choice: should the biggest market get the biggest slice, or should every sector be kept on a leash?

### The S&P GSCI: production-weighting, so energy eats the index

The Standard & Poor's Goldman Sachs Commodity Index (GSCI), launched by Goldman Sachs in 1991 and now owned by S&P, weights each commodity by **world production**. The logic sounds neutral and even principled: weight the basket by the economic importance of each commodity, measured by how much of it the world actually produces and consumes. The biggest, most-produced markets get the biggest weights.

The problem is that, measured in dollars of annual production, **energy dwarfs everything else**. The world produces and burns a colossal value of crude oil, refined products, and natural gas every year — far more than all the copper, corn, gold, and coffee combined. So production-weighting hands energy a crushing majority of the index.

![S&P GSCI sector weights production weighting makes it an energy bet](/imgs/blogs/commodity-index-investing-gsci-bcom-and-owning-the-basket-2.png)

The bar chart above shows the result: the GSCI is roughly **62% energy**, with agriculture around 15%, industrial metals near 11%, livestock 7%, and precious metals a mere 5%. Within that energy sleeve, crude oil (WTI and Brent) plus gasoline, heating oil, and gas make up the bulk. The practical consequence is blunt: **the S&P GSCI is, for all intents and purposes, a leveraged long-oil position with some agricultural and metals garnish.** When you "buy the GSCI," you are mostly buying oil. Its correlation to crude is very high; its volatility is dominated by energy's swings. An allocator who buys the GSCI thinking they have diversified across "all commodities" has, in reality, made a concentrated energy bet.

### Bloomberg BCOM: capped and diversified

The Bloomberg Commodity Index (BCOM, formerly the Dow Jones-UBS Commodity Index) was designed in direct reaction to that concentration. It also starts from production and liquidity data, but then it applies **caps**: no single sector may exceed roughly one third of the index, and no single commodity may exceed about 15%. Related commodities (all the energy products, say) are also grouped and capped together so the index cannot back-door its way into an energy overweight.

![Same asset class two very different baskets GSCI vs BCOM weights](/imgs/blogs/commodity-index-investing-gsci-bcom-and-owning-the-basket-3.png)

The grouped bars above place the two indices side by side. Where the GSCI is 62% energy, BCOM holds energy near **30%**. Agriculture rises to roughly 24%, industrial metals to about 16%, and precious metals — barely a footnote in the GSCI at 5% — jump to around 19% in BCOM (gold and silver carry real weight). The result is a basket that genuinely spreads risk across the commodity complex rather than concentrating it in the oil patch. BCOM is the index most allocators reach for when they want a *diversified* commodity sleeve; the GSCI is the index you choose if you specifically want energy beta.

![Two weighting rules two different bets you are actually making](/imgs/blogs/commodity-index-investing-gsci-bcom-and-owning-the-basket-4.png)

The contrast figure above frames the underlying choice. Production-weighting (left) makes the biggest market — energy — the biggest slice, so the GSCI moves like crude and concentrates its risk. Capping (right) deliberately limits every sector and every commodity, so BCOM stays balanced and behaves like a truer diversifier. Neither is "wrong." They answer different questions. The mistake is owning one while believing you own the other.

#### Worked example: a blended index move from sector weights

Suppose over a month energy falls **−10%** and industrial metals rise **+5%**, with all other sectors flat. What does each index do, just from these two moves? We will use the published sector weights and assume each sector moves as one block.

For the **GSCI** (energy 62%, metals 11%):

```
energy contribution = 0.62 x (-10%) = -6.20%
metals contribution = 0.11 x (+5%)  = +0.55%
GSCI move           = -6.20% + 0.55% = -5.65%
```

For **BCOM** (energy 30%, metals 16%):

```
energy contribution = 0.30 x (-10%) = -3.00%
metals contribution = 0.16 x (+5%)  = +0.48%
BCOM move           = -3.00% + 0.48% = -2.52%
```

The exact same world — oil down 10%, copper up 5% — costs a GSCI holder **−5.65%** but a BCOM holder only **−2.52%**, a gap of more than three percentage points in a single month. The intuition: when you own the GSCI you are mostly betting on one sector, so that sector's bad month is your bad month, while BCOM's caps spread the pain (and the gain).

## What "owning the index" delivers: the three-piece return

Now we open up the actual return. A fully-collateralized long-only commodity index is, in plain arithmetic, the sum of three things, and a serious investor tracks all three separately because they have completely different drivers and completely different long-run signs.

### Piece one: spot return

This is the part everyone expects — the change in the price of the front futures contract, which roughly tracks the change in the underlying commodity's price. If WTI's front contract goes from \$72 to \$79 over a year, the spot return on the oil sleeve is about +9.7%. Spot return is the headline, the thing on the news, the number people mean when they say "oil went up." Over very long horizons, spot returns on commodities have been roughly flat in *real* (inflation-adjusted) terms — a commodity is not a productive asset that compounds; it is a thing that costs about as much, in real terms, decade after decade, with enormous swings around that flat trend.

![Why the GSCI tracks oil WTI crude annual average 2000 to 2025](/imgs/blogs/commodity-index-investing-gsci-bcom-and-owning-the-basket-7.png)

Because the GSCI is roughly 62% energy, its spot return is dominated by exactly the line in the chart above: WTI crude's annual average from 2000 to 2025. You can see the index's whole life in that line — the run to nearly \$100 average in 2008, the 2014–2016 shale crash, the 2020 COVID collapse (the front contract even printed *negative* on one extraordinary day), the 2022 war spike back toward \$95. An investor who held a GSCI product over any of those windows did not really "own commodities"; they owned that oil rollercoaster, scaled by the index's energy weight. The cross-asset post on [energy, oil, gas, and the inflation engine](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine) digs into why oil sits at the center of the whole commodity complex's behavior.

### Piece two: roll yield

This is the piece that the brochures whisper and the statements scream. As covered in the Foundations, a long-only index must keep rolling its expiring front contract into the next month. The sign and size of that roll depend entirely on the forward curve:

- In **contango** (back months dearer), the index sells the cheap expiring contract and buys a dearer one. It pays up every single month. Over a year of steady contango, that drag can run anywhere from a couple of percent (mild) to *double digits* (steep contango, as natural gas routinely shows).
- In **backwardation** (back months cheaper), the index sells dear and buys cheap. The roll *adds* return — historically, crude oil spent long stretches in backwardation, which is a big reason commodity indices earned positive long-run returns despite flat real spot prices.

![Why a commodity fund lags spot the long-only roll drag in contango](/imgs/blogs/commodity-index-investing-gsci-bcom-and-owning-the-basket-5.png)

The chart above is the picture every disappointed commodity-fund holder eventually draws. The grey line is the spot price index — it wanders up and down but ends roughly where it started, even a touch higher. The red line is the long-only *total return*, dragged steadily lower by the monthly roll cost in a contango market. Over the illustrative five-year window, the spot is essentially flat-to-up while the total-return index has bled to 70 — a gap of about 31 points purely from the roll. This is not a fee; it is not a tracking error; it is the structural cost of staying long futures when the curve slopes up. The infamous United States Oil Fund (USO) lived this in 2009 and again in 2020: oil's spot price recovered sharply off its lows, yet USO holders lost money for years because the curve was in deep contango and the fund bled on every roll. The full anatomy is in [roll yield and why long-only commodity ETFs bleed](/blog/trading/commodities/roll-yield-and-why-long-only-commodity-etfs-bleed).

#### Worked example: the roll-plus-collateral total-return decomposition

Let us build a full year of returns for a single-commodity collateralized position so the three pieces are concrete. Suppose:

- The front WTI contract starts the year at \$72 and ends at \$79 — a spot move of +9.7%.
- The curve is in mild contango all year, and rolling the front each month costs about **−4%** annualized.
- The T-bills backing the position earn a **5%** collateral yield over the year.

The total return is the sum:

```
spot return       = (79 - 72) / 72        = +9.72%
roll yield         (mild contango)         = -4.00%
collateral yield   (T-bills at 5%)         = +5.00%
total return                               = +10.72%
```

Now flip the curve. Same +9.72% spot move and same 5% collateral, but the curve is in **backwardation** worth +6% on the roll:

```
spot return       = +9.72%
roll yield         (backwardation)         = +6.00%
collateral yield                           = +5.00%
total return                               = +20.72%
```

The *price of oil did exactly the same thing in both cases* — up from \$72 to \$79 — yet your total return was 10.7% in one world and 20.7% in the other, a ten-point swing driven entirely by the shape of the curve and the cash yield. The intuition: when you own a commodity index, the oil price is only one of three engines, and in many years it is not even the biggest.

### Piece three: collateral yield

The unsung hero, and the piece that has roared back to relevance since 2022. Because a fully-collateralized index holds real cash (in T-bills) behind every dollar of futures exposure, that cash earns the short-term interest rate. In the zero-rate decade from 2009 to 2021, this was a rounding error — T-bills yielded near zero, so collateral added almost nothing. But when the Fed hiked rates to roughly 5%+ in 2022–2023, collateral yield jumped to **about 5% a year of pure, riskless tailwind** on top of whatever the commodities did. The macro-trading post on [how monetary policy moves commodities](/blog/trading/macro-trading/how-monetary-policy-moves-commodities-real-rates-gold) explains why the level of real rates matters so much across the complex. For an index investor specifically, the lesson is mechanical: in a high-rate world, a chunk of your "commodity return" is really just the T-bill rate wearing a commodity costume. That is a tailwind to celebrate, but also one to be honest about — it is not a commodities edge.

There is a subtlety in the collateral leg worth naming, because it trips up careful people. The collateral yield is the *cash* return, but it is not free risk-free money you could have had anyway — the question is whether you are being *compensated for the commodity risk on top of it*. A T-bill alone gives you 5% with no commodity volatility. A collateralized commodity index gives you 5% *plus* the spot-and-roll bundle, but that bundle carries 15–20% annualized volatility. So the real test for an allocator is not "did I make money" (in a 5%-rate year you probably did, from collateral alone) but "did the spot-and-roll part of the return justify the volatility it added." If spot is flat and the roll is bleeding, the answer is no — you are taking commodity risk to capture a yield you could have had risklessly. That is the most common way a commodity sleeve quietly fails to earn its keep, and it hides precisely because the collateral leg keeps the headline number positive.

## How you actually buy an index: swaps, ETFs, and direct futures

So far we have talked about "the index" as if you can simply own it. You cannot — an index is a number, a calculation. To get the exposure, you need a *vehicle*, and the vehicle you choose changes your costs, your tax treatment, and even your counterparty risk. There are three main routes, and it is worth knowing how each one works because the differences are not academic.

**The total-return swap.** This is how the large institutions did it in the financialization wave and still do it today. The pension fund enters a swap with a bank: the fund pays the bank a small spread (and, in older structures, a fixed financing leg), and the bank pays the fund the total return of the GSCI or BCOM. The bank, in turn, hedges its obligation by actually buying and rolling the futures. The advantage is operational simplicity — the pension never touches an exchange, never rolls a contract, never posts margin to a clearinghouse; it just gets the index return delivered as a single cash flow. The cost is the swap spread the bank charges and the *counterparty risk*: if the bank fails, the fund is an unsecured creditor for the swap's value. This is the structure that put the "index speculator" debate on the map, because billions flowed through bank swap desks that then mechanically bought futures.

**The exchange-traded fund or note.** For everyone smaller than a pension fund, the ETF (or ETN) is the retail and advisor route. A fund like a broad commodity-index ETF holds the collateralized futures basket directly and issues shares you can buy in a brokerage account. The convenience is enormous — one ticker, daily liquidity, no derivatives paperwork — but it comes with an expense ratio (often 0.25% to 1.0%+ a year) and, critically, the same roll drag baked into the underlying. The ETF does not protect you from contango; it *delivers* contango to you, net of fees. (Single-commodity ETFs like USO are the extreme case, since they cannot diversify away a bad curve.) A note (ETN) adds the issuer's credit risk on top, since an ETN is an unsecured debt obligation of the issuing bank, not a fund holding real assets.

**Direct futures.** The largest and most sophisticated allocators sometimes replicate the index themselves — buying and rolling the actual futures contracts in-house. This cuts out the swap spread and the ETF fee, and it gives total control over the roll timing (letting them dodge the Goldman-roll front-running crowd). The price is operational complexity: you need a futures desk, margin management, and the discipline to roll thousands of contracts on schedule. Only the biggest funds bother, but when they do, they capture the cost savings the smaller players pay away.

The takeaway across all three: the index return you read about in the methodology document is a *gross, frictionless* number. What you actually pocket is that number minus a swap spread, an ETF fee, or your own trading costs — and minus the roll drag, which no vehicle can wish away because it lives in the futures themselves. When you compare two commodity products, you are really comparing two answers to the question: *how cheaply and cleanly does this vehicle deliver the same underlying roll-and-collateral bundle?*

## The roll mechanics: the Goldman roll and why it is front-run

The roll is not just a concept; it happens on a *published schedule*, and that publicity creates a cost that has become folklore in the futures pits.

The GSCI rolls its expiring front contracts into the next month over a fixed window: the **fifth through ninth business days** of each roll month, moving 20% of the position each day. This window is so well known it has a name — **the Goldman roll**. Because the schedule is public, anyone can calculate, weeks in advance, roughly how many contracts the index will sell and buy on which days. And when a large, price-insensitive, *forced* flow is fully predictable, faster traders position ahead of it: they sell the expiring contract before the index does (pushing its price down) and buy the next contract before the index does (pushing its price up). When the index finally rolls, it sells into a depressed front and buys into an elevated next month — getting a slightly worse price on both legs.

![The Goldman roll window a published schedule everyone can front-run](/imgs/blogs/commodity-index-investing-gsci-bcom-and-owning-the-basket-6.png)

The timeline above lays out the window. As the front contract nears expiry, the index must move on; on the fifth business day the roll begins, 20% per day for five days; faster traders front-run the predictable flow, pushing the front down and the next month up before the index trades; the index pays slightly worse prices — a small but recurring drag — and then holds the new front until next month, when the whole cycle repeats. Academic studies have estimated this front-running cost at somewhere between a few tenths of a percent and over one percent per year, depending on the period and how crowded the trade was. It is exactly the kind of predictable, mechanical edge that microstructure folks love; the strategic logic of trading against a forced, telegraphed flow is the bread and butter of the [game-theory-of-markets](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold) way of thinking about who is on the other side of your trade.

This is also the seed of the **second-generation index**. If a published, predictable roll costs the investor money, the obvious fix is to roll *less predictably* and *more cleverly* — which is exactly what enhanced indices do.

## First-generation versus second-generation indices

The original GSCI and BCOM are "first-generation": they always hold the front (or near-front) contract and roll on a fixed schedule. That design is simple and transparent, but it maximizes exposure to two costs — contango bleed (because the front-to-second roll is where contango bites hardest) and front-running (because the schedule is public).

**Second-generation** or "enhanced" indices try to keep the commodity exposure while reducing the roll cost. The common tricks:

- **Optimized-roll / curve-picking.** Instead of always holding the front contract, the index picks the point on the forward curve where the roll cost is *least bad* — often a contract several months out, where the curve is flatter and the monthly roll loss is smaller. Deutsche Bank's "Optimum Yield" indices popularized this.
- **Roll-timing flexibility.** Rather than rolling on the same five days as everyone else, enhanced indices roll on a wider or variable window to dodge the front-running crowd.
- **Active sector tilts.** Some "third-generation" products actively over- and under-weight sectors based on the curve shape (favoring backwardated commodities, avoiding deeply contangoed ones).

These reduce drag, but they introduce two new things an allocator must judge: **higher fees** (active cleverness is not free) and **tracking risk** (you no longer simply own "the front of the curve," so you can underperform the naive index in years when the curve does something the optimizer did not expect). The honest framing: first-generation indices are cheap, transparent, and structurally bleed in contango; second-generation indices cost more and trade some transparency for a smaller, smarter roll. Whether the trade is worth it depends on the curve environment, which is precisely why an allocator must understand the roll before choosing a vehicle.

#### Worked example: why my commodity fund lagged spot

A reader holds a popular long-only oil ETF. Over a year, the *spot* price of WTI rose from \$50 to \$60 — a gain of 20%. The reader checks their statement and finds the fund returned only **+2%**. Where did 18 points go? Let us decompose, assuming the curve was in steep contango that year (common after an oil crash, when near-term storage is glutted) costing about **−20%** on the roll, and assume a near-zero collateral yield (a low-rate year):

```
spot return       = (60 - 50) / 50         = +20.0%
roll yield         (steep contango)         = -20.0%
collateral yield   (low-rate year)          = + 0.5%
expense ratio                               = - 0.7%
fund total return                           = - 0.2%   (~flat, vs the +2% quoted)
```

Even with the spot price up a fat 20%, the steep-contango roll erased almost the entire gain; the fund was roughly flat while the headline oil price soared. (The exact arithmetic varies with the curve; the point is the *direction* and *magnitude* of the surprise.) The intuition: a long-only commodity ETF tracks the *futures roll*, not the spot price, and in steep contango those two can diverge by 20 points or more in a single year — so "oil doubled and my oil fund went nowhere" is not a glitch, it is the design.

## The financialization wave: 2004–2008

We started with it, so let us close the loop on what all that index money did. Between roughly 2004 and 2008, long-only commodity index investment exploded from tens of billions to a couple hundred billion dollars, as pension funds and endowments — chasing the Gorton–Rouwenhorst diversification thesis — poured into GSCI- and BCOM-tracking swaps and funds. Because these were *long-only* and *passive*, they did three structural things to the market:

1. **They added persistent buying pressure to the front of the curve**, especially during the monthly roll, and were price-insensitive (an index buys regardless of value).
2. **They changed the shape of forward curves.** A wall of long-only money concentrated at the front arguably pushed several markets toward contango more often, which — circularly — increased the roll cost those same investors paid.
3. **They became the scapegoat for the 2008 spike.** When oil hit \$147 and food prices soared, "index speculators" were blamed in Senate hearings and in the press. The academic verdict is genuinely mixed: some studies found index flows had measurable price impact; others (and the bulk of the careful work) concluded that fundamentals — Chinese demand, a weak dollar, tight supply — explained most of the move, with index money amplifying rather than causing it.

The regulatory response (position limits, the Dodd-Frank swap rules) and the full debate are the subject of [the financialization of commodities](/blog/trading/commodities/the-financialization-of-commodities-when-wall-street-arrived). For the index investor, the lasting lesson is uncomfortable: *you were the predictable flow.* The roll front-running, the contango that deepened as everyone crowded the front — those costs were, in part, the market extracting a toll from the very passive money that thought it had bought a clean diversifier.

## The diversification thesis, revisited

The whole reason institutions bought commodity indices in the first place was a *correlation* argument, so it deserves a hard look. The Gorton–Rouwenhorst thesis rested on three claims: commodities earned equity-like long-run returns, with bond-like volatility, and — the crown jewel — *negative correlation with stocks and bonds*, especially during inflation. A genuinely uncorrelated asset that rises when everything else falls is the holy grail of portfolio construction, because it lets you raise expected return without raising portfolio risk. That is why the pension money came.

Two decades later, the verdict is more nuanced, and an honest allocator has to sit with the nuance. The diversification *does* show up — but it is episodic and regime-dependent, not constant. Commodities are a superb diversifier in **supply-shock inflation** (an oil embargo, a war, a drought): prices spike, real bond yields fall, equities derate, and the commodity sleeve is the one green box on the page. That is exactly what happened in 1973–74, in the 2007–08 spike, and again in 2021–22. But in a **demand-driven slowdown** (a recession that crushes both demand and prices), commodities fall *with* equities — they are pro-cyclical industrial assets, after all — so the diversification evaporates precisely when a falling stock market makes you crave it most. The 2008 second half is the cautionary tale: commodities had been the great diversifier through the first-half spike, then collapsed alongside equities in the autumn crash as demand cratered. Correlation, here, is not a constant you can plug into an optimizer; it is a *regime*, and the sign flips with the kind of shock.

There is also the awkward question of the long-run *return*. The Gorton–Rouwenhorst study covered a period (1959–2004) that was unusually kind to commodities — heavy on backwardation, which made the roll a tailwind. The decade that followed (2012–2020) was the opposite: persistent contango, a strong dollar, a shale supply glut, and a commodity index that lost money for years on end. An allocator who bought in 2011 on the back of the diversification paper sat through a brutal lost decade before the 2021–22 inflation finally vindicated the sleeve. So the second revision is sobering: the long-run return is *not* reliably equity-like; it is closer to flat-real-spot plus collateral plus a roll that has been negative more often than positive in the modern era. The diversification benefit is real but conditional, and the return is best thought of as the price of insurance rather than a growth engine — which is exactly how the playbook below treats it. This conditional, regime-dependent character is the heart of the companion post on [commodities as an inflation hedge and when they are not](/blog/trading/commodities/commodities-as-an-inflation-hedge-and-when-they-are-not), and it is why a commodity sleeve belongs in a regime-balanced framework like the one in [all-weather and risk parity](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime) rather than being judged on its standalone Sharpe ratio.

## Common misconceptions

**"Owning a commodity index means owning the spot price of commodities."** No. You own collateralized futures that are rolled monthly. Your return is spot *plus* roll yield *plus* collateral yield. In a contango market the roll can cost you 10–20 points a year, so you can easily lose money while the spot price rises. The 2009–2011 USO experience — oil recovering off its lows while the fund bled — is the canonical proof. If you remember one thing, remember that the index tracks the *roll*, not the barrel.

**"Commodities are a great diversifier, so I should buy the biggest, most liquid index."** The biggest, most liquid index is the GSCI — and it is roughly 62% energy. Buying it is not diversifying across commodities; it is concentrating in oil. If diversification is the actual goal, BCOM (or an equal-weight or capped index) is far closer to the spirit of the thesis. The mistake is conflating "liquid and famous" with "diversified."

**"The collateral yield is a minor detail."** It was, from 2009 to 2021, when rates were near zero. Since 2022 it has been worth roughly 5% a year — a meaningful chunk of total return, and sometimes the largest of the three pieces in a flat-commodity year. Ignoring it makes you misattribute return: you might credit "commodities" for a gain that was really just the T-bill rate.

**"Enhanced (second-generation) indices are strictly better because they reduce the roll cost."** They reduce the *expected* roll cost, but at the price of higher fees and tracking risk. In a year of unexpected backwardation, an optimizer that moved out the curve to dodge contango can *underperform* the naive front-month index. There is no free lunch — only a different bet on the curve's behavior, with a higher management fee.

**"Index speculators caused the 2008 commodity spike."** This is the popular story and the political one, but the careful academic evidence is mixed and leans toward fundamentals (Chinese demand, a weak dollar, supply tightness) doing most of the work, with index flows amplifying at the margin. Treating "passive money broke the market" as settled fact is overconfident; treating it as having *zero* effect is also wrong. The honest answer is "some, but probably not most."

**"GSCI and BCOM are basically the same thing — they are both commodity indices."** They share a category and almost nothing else. In a year when energy rallies and grains fall, the GSCI (62% energy) soars while BCOM (30% energy, 24% ags) is roughly flat — and vice versa when softs spike and oil sags. Their annual returns can differ by 15–20 percentage points. Treating them as interchangeable is like treating an oil ETF and an agriculture ETF as the same trade because both say "commodity" on the label. The weighting rule is not a footnote; it is the entire identity of the index.

**"If I buy a commodity index, I get the rebalance bonus for free."** Only partly. The annual rebalance does mechanically trim winners and add to laggards, which can add a small return for a *diversified* index over time. But a concentrated index like the GSCI has little to rebalance *between* — when 62% of the basket is one sector, there is no meaningful diversification to harvest. The rebalance bonus is a benefit of *diversification*, so it accrues to BCOM-style baskets far more than to the energy-dominated GSCI. You do not get it just by buying "an index"; you get it by buying a *spread-out* one.

**"A commodity index pays me to wait, like a bond or a dividend stock."** The opposite, in contango. A bond pays a coupon for holding it; a contangoed commodity index *charges* you a roll cost for holding it. The only "yield" you reliably collect is the collateral yield on the cash — which is the T-bill rate, available without touching commodities at all. The commodity-specific carry (the roll) is just as often negative as positive.

## How it shows up in real markets

### The USO contango bleed, 2009 and 2020

The United States Oil Fund is the textbook case of an investor learning the roll the hard way. In early 2009, with oil having crashed from \$147 to the \$30s, the front-month curve went into deep contango — near-term storage was glutted, so prompt barrels traded far below later-dated ones. USO, a first-generation front-month tracker, had to roll every month at a steep loss. Crude's spot price roughly doubled off its lows through 2009, yet USO investors saw the fund badly lag, losing money in some stretches while the headline oil price rose. The fund eventually restructured to hold contracts further out the curve to soften the bleed. Then came April 2020: the May WTI contract settled at **−\$37.63** as storage at Cushing, Oklahoma filled to the brim, and the super-contango was so violent that USO was forced to abandon front-month exposure entirely and spread across multiple contracts to survive. The lesson, twice taught: a long-only, front-month commodity vehicle is a roll machine first and a price tracker second.

### The pension allocation that "diversified" into oil

A recurring real-world pattern: an institutional board, told that "commodities diversify the portfolio," approves a 5% allocation and — because it is the most famous, most liquid benchmark — implements it via a GSCI-linked product. They believe they have spread risk across energy, metals, and agriculture. In reality, with the GSCI at ~62% energy, they have placed a concentrated bet on oil. When energy has a bad multi-year stretch (2014–2016, when WTI fell from over \$100 to under \$30), the "diversifier" delivers a deep drawdown perfectly correlated with the energy shock they thought they had hedged. Boards that did the homework — and chose BCOM, an equal-weight index, or a multi-index blend — got a meaningfully different (and more diversified) experience. The vehicle choice *was* the bet.

### The collateral-yield comeback, 2023–2024

For most of the 2010s, the collateral leg of commodity index returns was dead money — T-bills paid near zero, so the entire return came from spot and roll. Then the Fed's 2022–2023 hiking cycle (see [the 2021–2023 inflation and the fastest hiking cycle](/blog/trading/macro-trading/2021-2023-inflation-and-the-fastest-hiking-cycle)) lifted the short rate above 5%. Suddenly a fully-collateralized commodity index earned roughly 5% a year *before the commodities did anything*. In a year when commodities were flat, that collateral yield could be the difference between a positive and a negative total return — and it was earned with no commodity risk at all. Sophisticated allocators noted the irony: a large slice of their "commodity return" was now just the risk-free rate, and they could have captured most of it by holding T-bills directly. The collateral leg is a reminder that an index return is a *bundle*, and you should know which part of the bundle you are actually being paid for.

### Robusta and softs: when a diversified index earns its keep

In 2024–2025, while oil drifted sideways, a clutch of agricultural and soft commodities went vertical: robusta coffee roughly doubled on Vietnamese supply scarcity, cocoa printed a record above \$12,000 a tonne on West African crop failure, and arabica coffee surged. A GSCI investor — 62% energy, with softs a tiny sliver — barely felt these moves. A BCOM or diversified-index investor, with a real ~24% agriculture weight, captured a chunk of them. This is exactly the scenario the diversification thesis is built for: a basket spread across uncorrelated supply shocks (drought in Brazil, disease in Ivory Coast, a tight Vietnamese robusta crop) smooths the ride, while a concentrated energy index sits out the action entirely. (The underlying markets are profiled in [softs](/blog/trading/commodities/softs-coffee-sugar-cocoa-and-cotton-the-tropical-markets).)

### The 1970s: when commodities won and everything else lost

The deepest case for a commodity sleeve comes from the 1970s. Stagflation — high inflation, weak growth — crushed both stocks and bonds in real terms, but commodities (and the indices that would later be built on them) soared as oil embargoes and food shocks drove prices up. That decade is the historical anchor for "commodities hedge inflation," and it is why allocators keep a sleeve even through long barren stretches. The full episode is covered in the cross-asset [1970s stagflation case study](/blog/trading/cross-asset/case-study-1970s-stagflation-commodities-win). The catch — and there is always a catch — is that the inflation-hedge property is *episodic*: commodities are a powerful hedge in supply-shock inflations like the 1970s and 2021–2022, and a poor one in demand-driven or disinflationary periods, where they can lag inflation for a decade. That conditional nature is the whole subject of [commodities as an inflation hedge and when they are not](/blog/trading/commodities/commodities-as-an-inflation-hedge-and-when-they-are-not).

### The China supercycle and the great backwardation, 2003–2008

The golden age of commodity index investing — and the reason the diversification papers looked so good — was the China supercycle of roughly 2003 to 2008. China's industrialization sucked in copper, iron ore, oil, and coal at a pace the world's mines and wells could not match, so prices ran for years and, crucially, the demand was so insatiable that prompt barrels and prompt tonnes traded at a *premium* — the forward curves were in **backwardation**. For a commodity index that was a double gift: spot prices rose *and* the roll *earned* money instead of bleeding. A GSCI investor in that window enjoyed all three return engines firing at once — rising oil, positive roll yield, and (until 2008) a decent collateral rate. It was the best of times, and it convinced a generation of allocators that commodities were a permanent free lunch. The hangover came after 2011, when the supercycle faded, the shale revolution flooded the oil market, curves flipped to persistent contango, and the same indices bled for the better part of a decade. The lesson for an index investor: the spectacular 2003–2008 returns were not the *normal* case — they were a backwardation-plus-supercycle alignment that has not repeated since. The full arc is in the cross-asset [2000s China commodity supercycle case study](/blog/trading/cross-asset/case-study-2000s-china-commodity-supercycle).

### The 2014–2016 oil crash inside the index

When WTI fell from over \$100 in mid-2014 to under \$30 in early 2016, the GSCI did not fall by the average of all commodities — it fell roughly *with oil*, because oil is roughly the index. An investor who had been told their GSCI sleeve was a "diversified commodity allocation" watched it draw down 60%+, almost perfectly tracking the energy collapse, while a BCOM holder — with energy capped near 30% and real weights in gold (which actually *rose* as rates stayed low) and grains — took a far shallower hit. This episode is the cleanest natural experiment in the whole post: the same macro event (an oil crash), two indices, two wildly different outcomes, driven entirely by the weighting rule. It is the strongest single argument for reading the energy weight before you buy.

## The playbook: how an allocator should think about a commodity sleeve

So you sit on an investment committee, and someone proposes "let's add commodities for diversification and inflation protection." Here is how to think it through, given everything above.

**Start by naming what you actually want.** "Commodities" is not one thing. Do you want (a) an inflation hedge, (b) a diversifier uncorrelated with stocks and bonds, or (c) a tactical bet on a specific commodity or supercycle? The answer changes the vehicle. If you want inflation protection, you want broad real-asset exposure and you must accept the episodic nature of the hedge. If you want diversification, you want the *capped* index (BCOM), not the energy-concentrated one. If you want an oil bet, then — and only then — the GSCI is the honest choice, and you should call it an oil bet out loud.

**Know that you are mostly buying energy and a roll bet, not "all commodities."** Internalize the 62% energy number for the GSCI. Any committee memo that says "diversified commodity exposure" while implementing the GSCI is misdescribing the position. The single most important due-diligence question is: *what is the energy weight of the index we are buying?* If it is north of 50%, you are buying oil.

**Decompose the expected return into its three pieces before you buy.** Ask explicitly: what do we expect from spot (probably flat in real terms over the long run), from roll (depends on the curve — currently contango or backwardation?), and from collateral (the T-bill rate)? If the answer is "flat spot, contango drag, plus the T-bill rate," you should ask the uncomfortable follow-up: are we really being paid to take commodity risk, or are we paying a roll cost to earn the risk-free rate with extra volatility? Sometimes the honest answer kills the trade.

**Size it small and accept the flat stretches.** Most institutions that hold commodities cap the sleeve at a few percent — commonly 3% to 5% — precisely because the asset class delivers long, demoralizing flat-to-down stretches (the entire 2012–2020 period was brutal) punctuated by violent inflationary spikes when you are glad you held it. A commodity sleeve is portfolio insurance for supply-shock inflation, not a compounding growth engine. Size it as insurance: enough to matter in a 1970s or 2022, small enough that a lost decade does not sink the fund.

**Pick the vehicle deliberately.** First-generation index (cheap, transparent, bleeds in contango) versus second-generation enhanced (pricier, smarter roll, tracking risk) versus a BCOM-style diversified versus a GSCI energy bet versus individual-commodity ETFs. Each is a different combination of cost, transparency, and exposure. The vehicle is not an implementation detail bolted on after the asset-allocation decision — *the vehicle is a large part of the bet.*

**Mind the collateral leg in a high-rate world.** When the T-bill rate is 5%, a meaningful slice of your commodity return is just the risk-free rate. Be honest in your attribution: do not credit "commodities" for the collateral yield. And recognize the corollary — when rates fall back toward zero, that tailwind disappears and the roll drag is exposed in full.

**Watch the curve, not just the price.** Because the roll is so large a part of the return, the single most useful ongoing signal is the *shape* of the forward curve for your index's biggest sectors. A broad, persistent contango (as in 2015–2016 or 2020) is a warning that the sleeve will bleed even if spot prices recover; a shift into backwardation (as in 2003–2008 or 2022) is a green light that the roll is now a tailwind. You do not need to time it precisely — but an allocator who ignores the curve and watches only the spot price is reading the wrong gauge entirely. The curve is the dashboard; the spot price is just the speedometer.

To compress all of that into a due-diligence checklist a committee can actually use before approving a commodity allocation: (1) What is the **energy weight** of this index — above 50% means you are buying oil, not commodities. (2) What is the **current curve shape** for the dominant sectors — contango is a structural headwind, backwardation a tailwind. (3) What is the **collateral yield** right now, and how much of the expected return is really just the T-bill rate. (4) What **vehicle** delivers it — swap, ETF, or direct futures — and what does that vehicle cost in spread, fee, or operational drag. (5) What **size** turns a −15% commodity year into a tolerable fund-level drag while still mattering in an inflation spike. Answer those five and you have done more analysis than most of the money that flooded in during 2004–2008.

#### Worked example: sizing a 5% commodity sleeve

A \$2 billion endowment decides to add a 5% commodity sleeve via a diversified (BCOM-style) collateralized index, and wants to understand its contribution. The allocation is:

```
sleeve size = 5% x $2,000,000,000 = $100,000,000
```

Now suppose, over a year, the sleeve's three return pieces come in at: spot +3%, roll −2% (mild contango), collateral +5% (high-rate environment). The sleeve total return and dollar contribution:

```
sleeve return  = +3% (spot) - 2% (roll) + 5% (collateral) = +6.0%
dollar gain    = 6.0% x $100,000,000                      = +$6,000,000
contribution to total fund = $6,000,000 / $2,000,000,000  = +0.30%
```

Now stress it: in a bad commodity year, spot −15%, roll −5%, collateral +5%:

```
sleeve return  = -15% - 5% + 5% = -15.0%
dollar loss    = -15.0% x $100,000,000 = -$15,000,000
drag on total fund = -$15,000,000 / $2,000,000,000 = -0.75%
```

A 5% sleeve turns a brutal −15% commodity year into a −0.75% drag on the whole fund — survivable — while still being large enough that a +40% inflation-spike year (as 2021–2022 delivered) would add a full +2 points to the fund. The intuition: size the sleeve as insurance, where a disaster costs less than a percent of the fund but a payoff in a supply-shock inflation is worth multiples of that — and where, in any year, you can name exactly how much came from spot, roll, and collateral.

That is the whole discipline. A commodity index is not a magic exposure to "real assets"; it is a transparent, mechanical machine — a basket of futures, weighted by a rule, rolled on a schedule, backed by Treasury bills. Its return is spot plus roll plus collateral, and for the most famous version of it, the "spot" part is mostly oil. Own it deliberately: know that the GSCI is an energy-and-roll bet wearing a diversification label, know that BCOM is the truer basket, know that the curve decides whether holding it pays or bleeds, and size it as the episodic inflation insurance it actually is. The allocator who can say all of that out loud is the one who will not be surprised by their own statement — and being un-surprised, in this asset class, is most of the battle.

## Further reading & cross-links

- [Roll yield and why long-only commodity ETFs bleed](/blog/trading/commodities/roll-yield-and-why-long-only-commodity-etfs-bleed) — the full anatomy of the roll cost that an index investor wears, with the USO case in depth.
- [The financialization of commodities: when Wall Street arrived](/blog/trading/commodities/the-financialization-of-commodities-when-wall-street-arrived) — what the 2004–2008 index-money wave did to price formation, and the "did speculators cause the spike" debate.
- [Commodities as an inflation hedge — and when they are not](/blog/trading/commodities/commodities-as-an-inflation-hedge-and-when-they-are-not) — why the inflation-hedge property is episodic, and the spot-versus-roll distinction behind it.
- [Contango vs backwardation](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) and [the forward curve](/blog/trading/commodities/the-forward-curve-the-most-important-chart-in-commodities) — the curve shapes that decide whether the roll pays or bleeds.
- [Spot vs futures](/blog/trading/commodities/spot-vs-futures-the-two-prices-of-the-same-barrel) and [what is a commodity](/blog/trading/commodities/what-is-a-commodity-the-physical-asset-that-trades-on-paper) — the building blocks the index is made of.
- [Energy: oil and gas, the inflation engine](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine) — why energy dominates the commodity complex's behavior (and the GSCI).
- [All-weather and risk parity: owning every regime](/blog/trading/cross-asset/all-weather-and-risk-parity-owning-every-regime) — where a commodity sleeve fits in a regime-balanced portfolio.
- [Metals: copper and silver, the economy's pulse](/blog/trading/cross-asset/metals-copper-silver-the-economys-pulse) — the non-energy half of a diversified basket.
- [How monetary policy moves commodities](/blog/trading/macro-trading/how-monetary-policy-moves-commodities-real-rates-gold) — why real rates (and the collateral yield) matter so much.
- [The 1970s stagflation case study](/blog/trading/cross-asset/case-study-1970s-stagflation-commodities-win) — the historical anchor for the commodity inflation hedge.

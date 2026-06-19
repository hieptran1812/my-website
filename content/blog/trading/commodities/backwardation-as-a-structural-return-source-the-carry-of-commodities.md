---
title: "Backwardation as a Structural Return Source: The Carry of Commodities"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why a backwardated forward curve hands a long holder a positive roll, why that roll is a genuine harvestable risk premium and not a free lunch, and how a systematic investor harvests commodity carry by tilting toward backwardation instead of buying the basket blind."
tags: ["commodities", "backwardation", "carry", "roll-yield", "risk-premium", "normal-backwardation", "futures", "forward-curve", "carry-factor", "crude-oil"]
category: "trading"
subcategory: "Commodities"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — When a commodity's forward curve is **backwardated** (prompt dearest, deferred cheaper), the act of rolling a long position from the expiring front contract to the next one is "sell high, buy low" — a positive **roll yield** that pays the long holder even if the spot price never moves. That positive carry is not an accident; it is the harvestable form of the commodity **risk premium**.
>
> - The same rolling mechanism that *bleeds* a long-only fund in contango *pays* it in backwardation — only the sign of the curve's slope changes. The carry is the slope.
> - John Maynard Keynes called the structural version of this **normal backwardation**: producers are net short and pay a premium to speculators to carry their price risk, so the futures price sits a touch below the expected spot and drifts up toward delivery on average.
> - The empirical commodity premium is **concentrated in backwardated markets**. This is the basis of the carry factor: go long the steeply backwardated commodities, short the steeply contangoed ones, and harvest the spread between the two rolls.
> - The premium is **not free**: backwardation is the signature of a *tight* physical market, and a tight market is one shock away from a violent break. You are being paid to be short a fat left tail. The one number to remember: a steeply backwardated curve can hand a long holder on the order of **10-15% a year** in roll yield — for as long as the tightness lasts.

In the autumn of 2021, the oil market did something it had not done at scale in years: it went deep into backwardation. The prompt barrel of crude traded several dollars above the barrel for delivery a year out, and the curve sloped steadily downhill. To most headline-watchers this was invisible — they saw only that oil was "around \$80." But to anyone who held crude through futures, the downhill slope was quietly doing something remarkable. Every month, when they rolled their position from the expiring contract to the next one, they were selling a dear barrel and buying a cheaper one. They were being *paid* to hold oil, on top of any move in the price itself. Over the year that followed, that positive roll added up to a double-digit annual tailwind — a return that came not from oil going up, but from the *shape* of the curve.

This is the exact mirror image of the story most commodity investors know. The famous warning — the one in every brochure footnote — is that long-only commodity funds *bleed* in contango, losing money to the roll month after month even when the spot price is flat. We told that story in [Roll yield and why long-only commodity ETFs bleed](/blog/trading/commodities/roll-yield-and-why-long-only-commodity-etfs-bleed). But the roll is not a curse. It is a *sign-flipping* mechanism. Flip the curve from contango to backwardation and the very same act of rolling flips from a steady loss into a steady gain. The bleed becomes a *bonus*. And because backwardation is the normal resting state of a healthy, demand-led commodity market, that bonus is not a fluke — it is a genuine, repeatable risk premium that a disciplined investor can deliberately harvest.

This post is about that premium: where it comes from, why it lives in backwardation, the theory (Keynes) that explains why producers hand it to you, the empirical evidence that it is real, the strategy (the commodity carry factor) that systematizes it, and the crash risk that is the reason it pays at all. By the end you will understand why "long the backwardated, short the contangoed" is one of the oldest and most durable trades in the commodity complex — and why it occasionally runs people over.

![Backwardation pays the roll: the long earns the slope while contango bleeds](/imgs/blogs/backwardation-as-a-structural-return-source-the-carry-of-commodities-1.png)

The figure above is the whole post in one frame. On the left is the bleed you already know: a contango curve where the front is cheap and the back months are dear, so each roll sells the cheap front and buys the dearer next month — buy high, sell low, a quiet negative carry that loses money even when spot is flat. On the right is the carry: a backwardated curve where the front is dear and the back months are cheap, so each roll sells the dear front and buys the cheaper next month — sell high, buy low, a quiet positive carry that *makes* money even when spot is flat. Same mechanism, opposite sign. The entire art of commodity carry is making sure you stand on the right side of that picture.

## Foundations: how the roll turns the curve's slope into a return

Before we can call backwardation a "return source," we have to be precise about the machine that converts a curve shape into dollars. That machine is the **roll**, and it is worth rebuilding from zero because everything downstream depends on getting it exactly right.

Start with the basic fact about commodity futures: every contract has an **expiry**. A January crude contract stops trading in December and either settles to physical delivery or to a cash price. A long-term holder of commodity exposure — an index fund, a pension's commodity sleeve, a macro trader expressing a multi-year view — does not want to take delivery of a thousand barrels of oil in a tank in Cushing, Oklahoma. So before the front contract expires, they **roll**: they sell the expiring contract and buy a later-dated one to keep the exposure alive. This is not optional. A long-run commodity position is a *chain* of rolls, one after another, forever. The return you actually earn is the sum of two things: the change in the price level (spot going up or down) **plus** the cumulative effect of every roll.

That second piece is the **roll yield**, and it is determined entirely by the slope of the curve at the point where you roll. One way to feel the inevitability of it: a long-run commodity holder *never* actually captures the spot price, because they never hold spot — they hold a *succession of futures contracts*, each bought somewhere on the curve and sold somewhere else on the curve as it rolls forward. The spot price is a reference they orbit but never touch. What they actually bank is the price at which they buy each contract versus the price at which they sell it, and the gap between those — the roll — is set by the curve's shape. This is why the roll yield is not a fee bolted onto the side of a commodity return; it *is* a structural component of the return, as fundamental as the spot move itself. Here is the mechanic, slowed down. Suppose the front contract is trading at \$85 and the next contract is trading at \$83.20 — a downward, backwardated slope. To roll, you *sell* your front at \$85 and *buy* the next at \$83.20. You have just sold high and bought low: you pocket the \$1.80 difference per barrel as a roll gain, and you now hold a contract that — if spot stays put — will itself drift *up* toward \$85 as it approaches delivery and becomes the new prompt. Either way you read it, the downhill curve pays you. Now flip it: in contango the front is at \$72 and the next is at \$73.40. To roll you sell low (\$72) and buy high (\$73.40), losing \$1.40 per barrel. The uphill curve charges you.

The crucial and counterintuitive point — the one that trips up almost everyone — is that **this happens whether or not the spot price moves at all**. The roll yield is a separate, additive source of return that comes purely from the curve's geometry. A flat spot price with a backwardated curve still makes you money; a flat spot price with a contango curve still loses you money. The level is one story; the slope is a completely separate story, and over long horizons the slope often dominates. We hammered this distinction in [Contango vs backwardation: what the shape of the curve means](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means); here we take the next step and treat the *positive* version of the slope as something you can deliberately go and collect.

#### Worked example: a single positive roll, annualized

Take the backwardated front of our illustrative crude curve: the prompt contract at \$85.00, the next at \$83.20. You hold one contract (1,000 barrels). When you roll, you sell the prompt at \$85.00 and buy the next at \$83.20.

```
Sell prompt   85.00 per bbl
Buy next      83.20 per bbl
Roll gain      1.80 per bbl  =  1.80 / 85.00  =  +2.12% on the position
```

That \+2.12% is earned on *one* roll. If the curve held its shape and you rolled roughly once a month, the carry would compound through the year. A crude approximation: 12 rolls of \+2.12% compounds to about \((1.0212)^{12}-1 \approx +28.6\%\) before any change in spot — and even a gentler, more realistic schedule that decays as you roll into the flatter part of the curve still lands in the **\+10% to \+15% a year** range that steep oil backwardation has historically delivered. The intuition: in backwardation, doing nothing but maintaining your position pays you, and over a year that "nothing" can be worth more than most people make from being right about the price.

### The total-return decomposition: three pieces, and the one that compounds

It helps to write down, plainly, what a long futures position actually earns, because the carry premium is one named term in a short equation. A fully-collateralized long commodity futures position — one where you post cash equal to the notional and hold futures on top — earns three things added together:

```
Total return  =  Spot return       (the change in the price level)
              +  Roll return       (the carry from rolling the curve)
              +  Collateral return  (the interest on the cash you posted)
```

The **spot return** is what everyone watches: oil went from \$80 to \$88, that is \+10%. The **collateral return** is the boring, reliable piece — because a futures position requires only margin, the cash backing it sits in Treasury bills earning the risk-free rate, so a fully-collateralized commodity index quietly adds the T-bill yield on top. The **roll return** is the carry — the term this whole post is about — and it is the only one of the three whose *sign* is set by the shape of the curve rather than the direction of the world. In backwardation the roll return is positive and adds to the other two; in contango it is negative and eats into them.

The reason the roll return matters so disproportionately over long horizons is that it is *persistent and compounding*. The spot return is a random walk that ends roughly where it started in real terms over decades — commodities are consumption goods, not growth assets, so the price level mean-reverts to the cost of production and goes nowhere on average. The collateral return is a modest, steady positive. The roll return, by contrast, is a *structural drift* with a consistent sign for as long as the curve keeps its shape — and a consistent-sign drift, compounded over years, swamps a mean-reverting random walk. This is the deep reason that, in the historical record, the *carry* and not the *spot move* explains most of the difference between the commodities that rewarded long holders and the ones that punished them.

#### Worked example: decomposing a year of total return

Suppose over one year crude spot rises 8%, the curve is backwardated and pays a \+11% roll, and the cash you posted earns the 5% T-bill rate. Your fully-collateralized long position earns:

```
Spot return        + 8%
Roll return        +11%
Collateral return  + 5%
Total return       +24%  (approx, before compounding interactions)
```

Now flip the curve: same 8% spot rise, but the commodity is in contango paying a \-11% roll. Total return is \(8 - 11 + 5 = +2\%\) — the *same* correct call on the price level, a 22-percentage-point worse outcome, entirely from the sign of one term. The intuition: you can be right about the price and still earn almost nothing if you are on the wrong side of the roll, and right about the price and earn a fortune if you are on the right side — the carry is the term that decides which.

That last point deserves its own picture, because the magnitude is the whole reason anyone bothers.

![Roll yield by curve regime showing the carry lives in backwardation](/imgs/blogs/backwardation-as-a-structural-return-source-the-carry-of-commodities-2.png)

This is the roll yield broken out by curve regime, with the positive backwardation bars in green — the carry. A steep contango (think natural gas through the winter, or oil in a storage glut) can cost a long holder on the order of \-12% a year; a steep backwardation in a tight oil market can hand the same holder \+12%. The flat case sits at zero. The asymmetry the picture makes visceral is this: the curve regime is worth roughly *plus or minus a dozen percent a year* before you have made a single correct call about the direction of the price. For a long-only investor that is a tax; for a carry investor it is the entire edge. The job, then, is to systematically position on the green side of this chart and avoid the red side — and to understand *why* the green side exists in the first place, which is where the theory comes in.

## Why the curve rolls you up into money: the mechanics of backwardation

Let us look directly at the curve you want to be long, and trace exactly where the money comes from as time passes.

![The backwardated curve you roll up into money with per-roll gain annotations](/imgs/blogs/backwardation-as-a-structural-return-source-the-carry-of-commodities-3.png)

The green line is a backwardated crude curve: the prompt sits at \$85 and the curve slides down to about \$79 a year out. Now hold the deferred contract — say the one priced at \$83.20 — and let time pass with spot unchanged at \$85. As that contract marches toward the front of the curve, it has to *converge* to the prevailing spot price at delivery, because at expiry a futures contract and the physical commodity are the same thing — a point we built carefully in [Spot vs futures: the two prices of the same barrel](/blog/trading/commodities/spot-vs-futures-the-two-prices-of-the-same-barrel). So a contract bought at \$83.20 drifts *up* toward \$85 as it ages. That upward drift, repeated on every contract you hold, *is* the roll yield. The downhill curve is not a forecast that prices will fall; it is a structure that pays you for the passage of time, as long as the shape persists.

This is the single most misunderstood point in the whole topic, so it bears stating bluntly: **a backwardated curve does not predict that the spot price will go down.** If the market actually expected spot to fall by the amount of the slope, the long holder would earn nothing — the gain from rolling up the curve would be exactly offset by the loss from the falling spot. The backwardation premium exists precisely *because* the curve is steeper than the market's honest expectation of where spot will go. The deferred contracts are priced *below* the expected future spot, and the long holder collects the gap between that depressed forward price and the spot price that actually materializes. That gap has a name and a theory behind it, and they belong to Keynes.

## Keynes's normal backwardation: who pays you, and why

In the 1920s and 30s, John Maynard Keynes and later John Hicks asked a deceptively simple question: in a futures market, is the futures price an *unbiased* forecast of the future spot price, or is it systematically biased one way? Their answer launched the entire theory of the commodity risk premium, and it is called **normal backwardation**.

The argument starts with who actually *needs* to use the futures market, and it is the producers. A farmer who has planted a corn crop, an oil company that has drilled a well, a copper miner who will produce metal for years — these are people who are *naturally long* the physical commodity and terrified of the price falling before they can sell it. Their dominant motive is to lock in a price *now* and remove that risk. To do so, they **sell futures** — they go short the paper to hedge their long physical. Across the whole market, hedging producers are typically **net short** the futures.

But every short needs a long on the other side. Who buys the futures that producers want to sell? Speculators — macro funds, CTAs, index investors, and yes, you. And here is Keynes's insight: speculators do not provide that service for free. They are being asked to take on the price risk that producers are desperate to shed, and bearing risk demands compensation. The only way to compensate them is to let them buy the futures *cheap* — to set the futures price a touch **below** the expected future spot price. That price concession is the producers' insurance premium, and it is the speculators' fee. Because the futures price is set below the expected spot, it tends to *rise* toward spot as delivery approaches, and the long speculator collects that rise. A market structurally priced this way — futures below expected spot — is in **normal backwardation**.

![Keynes normal backwardation pipeline showing why the long speculator gets paid](/imgs/blogs/backwardation-as-a-structural-return-source-the-carry-of-commodities-4.png)

The pipeline traces the chain: producers are net short and want out of price risk; to find a buyer they accept a futures price below expected spot; that depressed forward is what the long speculator buys; as delivery nears the futures converges *up* toward the realized spot; and the long collects the gap as compensation for carrying the risk the producer dumped. Read it as an insurance market and it clicks immediately — the producer is buying a policy against a price crash, the speculator is the underwriter, and the premium flows from the hedger to the speculator. This is exactly the "who is on the other side" framing we built in [The four players: producers, consumers, hedgers, and speculators](/blog/trading/commodities/the-four-players-producers-consumers-hedgers-and-speculators): the speculator is not a parasite on the market but a paid risk-absorber, and normal backwardation is the wage.

#### Worked example: the premium the long collects on average

Suppose corn for delivery in six months has an *expected* spot price — the market's honest best guess — of \$5.00 a bushel. Producers, net short and eager to hedge, are willing to sell the six-month future at \$4.88. A speculator buys that future at \$4.88.

```
Expected future spot     5.00 per bushel
Futures price (today)    4.88 per bushel
Embedded premium         0.12 per bushel  =  +2.46% over the six months
```

If the expected spot is what materializes, the future the speculator bought at \$4.88 converges to \$5.00, and the speculator earns \$0.12 a bushel — the premium the producer paid to offload the risk. The speculator did not need corn to *rise* above \$5.00; they only needed the futures to be priced below the expected spot, which normal backwardation guarantees. The intuition: the long earns a structural premium not by forecasting prices but by underwriting the producers' desire for certainty.

There is an important honesty here. Normal backwardation is a statement about the *expected* relationship between futures and spot, and it holds *on average and over time*, not on any single contract. Some months the producer's fear is overwhelming and the premium is fat; some months consumers (airlines hedging fuel, food companies hedging grain) are the more anxious side and tip the market the other way; sometimes a supply glut pushes the whole curve into contango and the sign flips entirely. The premium is a *tendency*, harvested across many contracts and many commodities — which is exactly why it is captured systematically, not bet on one trade at a time.

### Two theories of the curve: hedging pressure vs the theory of storage

Keynes's normal backwardation is one of *two* great theories of why the commodity curve has the shape it does, and a serious carry investor needs both, because they explain different things and they interact.

The first is the one we just built: the **hedging-pressure** theory (Keynes, Hicks, and later refined into the "hedging pressure hypothesis"). It is a theory about *risk preferences and who is desperate*. Whichever side of the market is the more anxious hedger — usually producers, who are net long the physical and net short the futures — pays a premium to the speculators on the other side, and that premium is the risk-bearing wage. When producers dominate the hedging, the futures sit below expected spot (normal backwardation, the long is paid); when consumers dominate (a war-scared airline desperate to lock fuel, say), the futures can sit *above* expected spot (a "contango of fear," where the *short* speculator is paid). The hedging-pressure theory says the premium's sign and size track the *net hedging imbalance*, which is exactly what the Commitments of Traders positioning data lets you observe.

The second is the **theory of storage** (Kaldor, Working, Brennan). It is a theory not about risk but about *physical inventories and arbitrage*. It says the relationship between the futures price and the spot price is pinned by the cost of carrying the physical: futures should equal spot plus storage plus financing *minus* the convenience yield — the benefit of having the physical on hand. When inventories are *high*, the convenience yield is near zero, storage dominates, and the curve is in contango (futures above spot). When inventories are *low and the market is tight*, the convenience yield is *large* — having the barrel today is worth a lot — and it overwhelms storage, dragging futures *below* spot into backwardation. We built this full cost-of-carry identity in [Convenience yield and the cost of carry](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape).

The two theories are not rivals; they are two lenses on the same curve, and together they explain why the carry premium is *concentrated in backwardation*. The theory of storage tells you *when* a market backwardates — when inventories are low and tight. The hedging-pressure theory tells you *why the long gets paid* when it does — because a tight, stressed market is also one where producers are most anxious to hedge and most willing to pay for the privilege. Low inventory is the physical cause; producer hedging pressure is the premium it generates. Backwardation is where both forces point the same way, which is precisely why the long carry is reliable there and absent in slack, well-supplied, contangoed markets.

## Why the premium is concentrated in backwardated markets

If normal backwardation were a constant, every commodity would always pay the long and the whole business would be trivial. It is not constant — and the reason it isn't is the reason the premium is *concentrated* in backwardated markets specifically. The empirical record, built up over decades of academic work on commodity returns, is strikingly clear on one point: the long-run return to holding a commodity is far more about its *curve shape* than its identity. The commodities that have rewarded long holders are the ones that spent most of their time in backwardation; the ones that punished long holders are the ones that spent most of their time in contango.

The mechanism is the one we have already built. In backwardation the roll pays, so a long holder earns the spot return *plus* a positive carry. In contango the roll bleeds, so a long holder earns the spot return *minus* a negative carry. Over a multi-decade sample where spot prices roughly track inflation and go nowhere in real terms, the *carry* is what is left — and it is overwhelmingly positive for the historically backwardated commodities (often energy, sometimes industrial metals in tight cycles) and overwhelmingly negative for the historically contangoed ones (often natural gas, frequently the precious-metals and grains complexes in glut years). The premium is not spread evenly; it pools where the curve slopes down.

This is exactly the failure mode of the naive long-only index. A broad commodity index buys *everything* — the backwardated names that pay and the contangoed names that bleed — and lets the bleed of the contangoed majority drag down the whole basket. We see this in the divergence between spot and total return.

![Same roll opposite sign comparing the contango drag against harvested carry](/imgs/blogs/backwardation-as-a-structural-return-source-the-carry-of-commodities-5.png)

The slate line is the spot index — the underlying barrel — which over this window wobbles and ends roughly flat. The red line is the naive long-only total return: the contango drag compounds underneath the flat spot, and the holder ends up far below where the barrel itself sits. The green line is the same investor with one change — they tilt toward backwardated markets, so the very same roll mechanism *adds* instead of subtracts, and they end up far *above* the spot. The picture's whole message is that the roll is not destiny. The long-only holder eats the drag because they buy the curve shape blind; the carry-aware investor flips the sign by being deliberate about *which* curves they hold. The red and green areas are the same mechanism with the slope reversed.

#### Worked example: the carry-factor spread (the spread harvested)

A carry strategy does not just go long backwardation; it goes long the backwardated *and short the contangoed*, harvesting the spread between the two rolls. Suppose you find crude in steep backwardation paying a \+10% annual roll, and natural gas in steep contango costing its longs \-12% a year. You put \$1,000,000 long crude futures and \$1,000,000 short gas futures.

```
Long crude (backwardated)   +10% roll on 1,000,000  =  +100,000
Short gas (contangoed)      you collect the bleed the
                            long side pays:  +12% on 1,000,000  =  +120,000
Combined carry                                          =  +220,000
                                                        =  +11% on the 2,000,000 gross
```

Because you are long one and short the other, much of the *flat-price* market risk cancels — if all commodities sell off together, your long loses while your short gains. What is left is the **carry spread**: the difference in roll yields between the leg that pays and the leg that bleeds. The intuition: a carry trade is not a bet on commodities going up; it is a bet on the curve-shape spread, harvested whether the complex rallies or falls.

There is more than one way to put that spread on, and the difference matters in practice.

### Two ways to express carry: cross-sectional and time-series

There are two distinct disciplines for harvesting commodity carry, and they behave differently enough that a serious investor often runs both.

**Cross-sectional carry** is the long-short factor we just described: at each rebalance, rank the *whole universe* of commodities by curve slope, go long the most backwardated quintile and short the most contangoed quintile, and hold the spread. It is *relative* — it does not care whether commodities as a class are going up or down, only that the backwardated names out-carry the contangoed ones. Because the long and short legs largely cancel the common market move, cross-sectional carry is close to market-neutral and its return is almost purely the carry spread. This is the cleanest expression of the premium and the one most academic studies measure.

**Time-series carry** (also called the curve-tilt or "smart roll") is different: it stays *long-only* in spirit but uses the slope to decide *how much* and *which contract* to hold. When a commodity is backwardated, hold it long and roll into the cheap deferred; when it tips into contango, cut the position or move to a part of the curve where the bleed is smallest. Time-series carry does not short the contangoed names — it simply *avoids* them, or rolls around the worst of the contango. It keeps some exposure to the upside of the commodity complex (useful if you want commodities as an inflation hedge) while dodging the structural roll tax that sinks the naive index. The optimized-roll commodity indices that emerged after the 2000s — the ones that pick the contract with the best carry rather than mechanically buying the front month — are time-series carry in product form.

The trade-off is the usual one. Cross-sectional carry is purer and more market-neutral but gives up the commodity beta entirely; time-series carry keeps the beta but is more exposed to the whole complex selling off. A diversified harvester typically runs cross-sectional carry as a standalone factor *and* uses time-series carry logic to optimize the roll on any long-only commodity holding — squeezing the premium from both the relative and the absolute side.

That structure — rank by slope, buy the payers, sell the bleeders — is the carry factor, and it deserves to be seen as a recipe.

![The carry factor go long the backwardated short the contangoed](/imgs/blogs/backwardation-as-a-structural-return-source-the-carry-of-commodities-6.png)

The grid lays the factor out as a procedure. The long leg buys steep backwardation, where the roll pays you to sell dear and buy cheap. The short leg sells steep contango, where the roll costs the *other* holder, and you collect that cost from the short side. The net result is the carry spread — the slope difference between the two legs — harvested largely independent of the price level, roughly market-neutral. This is the systematic, curve-aware way to own commodity carry, and it is the direct opposite of the long-only basket that buys both legs and eats the bleed on one of them. The same family of carry ranking shows up across asset classes; the commodity version is just unusually clean because the curve slope is a directly observable, physically grounded signal rather than an estimated one.

## The premium is not free: backwardation is a tight market, and tight markets break

Everything above might read like a free lunch: backwardation pays, so tilt toward it and collect. If it were free, it would have been arbitraged away decades ago. It has not been, which tells you it is *compensation for a risk* — and the risk is real, specific, and occasionally devastating. Understanding it is what separates a carry investor from a carry casualty.

Recall what backwardation *means* physically. The curve slopes down — prompt dear, deferred cheap — because the commodity is **scarce right now**. Buyers are paying a premium to have the physical thing today rather than later; inventories are low; there is little spare cushion in the system. This is the **convenience yield** at work, the hidden dividend of holding the physical when it is tight, which we built in [Convenience yield and the cost of carry: why the curve has a shape](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape). Backwardation is, by construction, the curve shape of a *tight, stressed, low-inventory market*.

And a tight market with no inventory cushion is exactly the market that can break violently. When there is plenty of inventory (contango), a demand shock is absorbed by tanks and warehouses and the price barely flinches. When there is *no* cushion (backwardation), a shift in supply or demand has nowhere to go but straight into the price — and because the market was tight on the *upside*, the dangerous direction is a sudden *collapse*: a demand crack (a recession, a sudden efficiency gain, a Chinese slowdown) or a supply flood (OPEC opens the taps, a war ends, a mine restarts) can flip a steeply backwardated curve toward contango almost overnight. The prompt, which had been bid to a premium by desperate buyers, collapses fastest of all. The long carry holder, who was quietly collecting \+10% a year, gets run over.

![Why the carry premium is not free short the tail](/imgs/blogs/backwardation-as-a-structural-return-source-the-carry-of-commodities-7.png)

The decision graph shows the shape of the bet. Backwardation says the market is tight. *Most* of the time, the tightness persists and the roll keeps paying the steady carry — small positive returns, month after month, that look like free money. But *rarely*, a shock hits a market that had no cushion, the curve breaks, the prompt collapses, and the long is run over. The premium you were collecting is the *payment for bearing that crash risk*: you are structurally **short the fat left tail**. This is the defining signature of carry trades everywhere — a long stretch of small gains punctuated by occasional large losses, often described as picking up pennies in front of a steamroller. The carry is the pennies; the tight-market break is the steamroller. The premium is real, but it is *earned*, and the way it is earned is by standing in front of a tail that occasionally arrives.

### Measuring the slope: how you actually read the carry signal

To harvest carry you need to *measure* it, and there is a standard, model-free way to turn a curve into a single carry number. Take the front contract price \(F_1\) and a deferred contract price \(F_2\) that is \(n\) months further out, and compute the annualized slope:

```
Annualized roll signal  =  (F1 - F2) / F2  x  (12 / n)
```

For our backwardated crude, with \(F_1 = 85.00\) (prompt) and \(F_2 = 83.20\) (two months out, so \(n = 2\)):

```
(85.00 - 83.20) / 83.20  =  0.0216  per 2 months
0.0216  x  (12 / 2)       =  +12.98%  annualized
```

A positive number means backwardation (the long is paid); a negative number means contango (the long bleeds). This single figure is the carry signal you rank commodities on — it is observable today, requires no forecast, and is grounded in real, physical prices rather than an estimated expected return. That last property is what makes commodity carry unusually clean compared with carry in other asset classes: the slope of a forward curve is a *price you can see*, not a parameter you have to guess. A practical refinement is to measure the slope over a *consistent* part of the curve across commodities (say, the first versus the twelfth month) so you are comparing like with like, and to be aware that the front of the curve is the noisiest, most squeeze-prone, most crowded part — which is why optimized strategies often read the carry signal from a slightly deferred section of the strip rather than the volatile prompt.

#### Worked example: a carry-crash loss

Suppose you are long a steeply backwardated crude position collecting \+1% a month in carry. You hold \$1,000,000 of exposure. For eight months you collect the carry, and then a demand shock hits a market with no inventory cushion: the prompt drops 20% in three weeks as the curve flips toward contango.

```
8 months of carry   +1% / month on 1,000,000  ≈  +83,000  (compounded)
The break           prompt falls 20% on 1,000,000  =  -200,000
Net                                                =  -117,000
```

Eight months of patiently harvested pennies are wiped out — and then some — by a single three-week break. The intuition: the carry premium and the crash risk are *the same coin*; you cannot collect the one without being exposed to the other, and any honest carry strategy budgets for the steamroller rather than pretending the pennies are free.

This is also why measured volatility is such a treacherous guide to carry risk. For the eight calm months in the example, the strategy's return stream looks like a smooth, low-volatility line — a Sharpe ratio that would make any allocator's mouth water. Then the ninth month delivers a loss many times larger than anything in the trailing window. A risk model fed only the calm months would have sized the position *up*, precisely because it looked safe, and walked straight into the steamroller with extra leverage. The pathology is general to every short-tail strategy: the periods of apparent safety are not evidence of low risk; they are the *accumulation phase* of a risk that is being stored up and paid out in lumps. The carry harvester who survives is the one who sizes for the lump, not for the smooth line — who treats a long quiet stretch as a *warning that the bill is overdue*, not as proof the bill will never come.

## When the premium disappears

The carry premium is not a law of nature; it is a *behavioral and structural* phenomenon, and there are identifiable conditions under which it shrinks, vanishes, or reverses. A serious harvester watches for all of them.

**The market goes structurally contango.** The most basic way the premium disappears is that the curve flips. If a commodity's normal state becomes contango — because new supply is abundant, storage is plentiful, or the commodity is cheap to hold — then the roll bleeds rather than pays, and the long carry simply is not there. Natural gas, with its huge storage and brutal seasonality, spends long stretches in contango; gold, a near-monetary metal with negligible storage cost and no convenience yield, is almost always in mild contango (the full cost-of-carry curve), which is one reason it lives in [the gold series](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) and is treated as a monetary asset rather than a carry source. When the structural shape is contango, there is no premium to harvest on the long side — the carry is on the *short*.

**Everyone crowds the trade.** The carry premium is, in part, a payment for providing a scarce service — bearing the producers' risk. When *too much* speculative capital piles into the long-backwardation trade, two things happen. First, the extra demand for the backwardated contracts bids up the deferred prices, *flattening* the curve and shrinking the very slope that was the source of return. Second, a crowded carry trade is a fragile carry trade: when the inevitable shock comes, everyone tries to exit the same tight position at once, and the crash is deeper and faster than it would otherwise be. The financialization of commodities after the mid-2000s — when index funds and ETFs poured into the front of the curve — is widely argued to have *compressed* the historical roll premium precisely by crowding it, a story we pick up in the broader complex's [allocation lens](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine). A premium that everyone harvests is a premium that gets arbitraged thin.

**The hedging imbalance reverses.** Because the premium is, at root, the wage for absorbing the *net* hedging pressure, it shrinks or flips whenever that pressure reverses. If consumers become the more desperate hedgers — an airline in a geopolitical scare locking fuel at any price, a food company panicking about a harvest failure — they bid the futures *up* relative to expected spot, and the long carry shrinks or turns negative even in a tight market. This is why the Commitments of Traders positioning data is part of the carry toolkit: it lets you see *which* side is the anxious hedger and therefore which way the premium is flowing. A carry signal read purely off the curve slope, without a check on who is doing the hedging, will occasionally be long a "backwardation" that is really a consumer panic about to unwind.

**The risk simply shows up.** The premium does not "disappear" so much as get *paid back* when the tail arrives. A carry strategy can look brilliant for years and then surrender a large fraction of its gains in a single regime break. This is not a failure of the strategy; it is the strategy *doing what it is paid to do* — absorbing a risk that occasionally materializes. The premium is the long-run average *after* the crashes, not the smooth line between them. A useful discipline is to judge a carry strategy only over a window long enough to *contain* at least one regime break — a track record that has never seen the steamroller has not yet been tested, and its smooth line is a measure of luck, not of edge.

## Common misconceptions

**"Backwardation means the price is going to fall."** No. This is the single most common error, and it is the opposite of how the premium works. A backwardated curve that simply forecast a falling spot would pay the long holder *nothing* — the roll-up gain would be exactly cancelled by the spot decline. The premium exists *because* the deferred contracts are priced below the *expected* spot, not because spot is expected to drop. Backwardation is a statement about scarcity *now*, plus a risk premium — not a directional forecast. We dismantled the level-versus-slope confusion in detail in [Contango vs backwardation](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means); the carry premium is the proof that they are independent.

**"The roll yield is just an accounting artifact, not a real return."** Wrong, and dangerously so. The roll yield is as real as any return you will ever earn: it is the actual change in the dollar value of your position from selling a dear contract and buying a cheaper one. A long-only fund that loses \-12% a year to contango has *really lost* that money — its NAV is genuinely lower. A carry fund that earns \+12% a year from backwardation has *really earned* it. The roll yield is not a bookkeeping curiosity; it is, for a long-horizon holder, often the *dominant* component of total return.

**"Carry is a low-risk, market-neutral free lunch."** No. The long-short structure cancels much of the *flat-price* market risk, which makes carry look low-volatility most of the time — and that is exactly the trap. The risk is not in the day-to-day wobble; it is in the *tail*: the tight-market break that flips the curve and collapses the prompt. Carry strategies have low *measured* volatility right up until they don't, which is why naive risk models that look only at recent volatility systematically *under*estimate carry risk. You are short a tail, and tails do not show up in trailing volatility.

**"If carry works, just buy and hold the most backwardated commodity forever."** No — concentration is how you get steamrolled. The premium is a *cross-sectional, diversified* phenomenon: it is reliable across a *basket* of backwardated commodities harvested over many rolls, not in any single name held forever. Any one backwardated commodity is one supply shock from a violent reversal. The carry factor works because it spreads the bet across many curves and rebalances as the slopes change — not because any single curve is a perpetual money machine.

**"Backwardation and a high convenience yield are speculator constructs."** No. They are *physical* facts about a tight market, grounded in real inventories, storage costs, and the value of having the commodity on hand right now. The convenience yield is the dividend the physical pays its holder in a shortage; backwardation is that dividend showing up in the curve. The speculator's premium rides *on top* of this physical structure — it does not invent it.

**"Gold and precious metals are a great carry source."** No — the near-monetary metals are almost the opposite. Gold has negligible storage cost, no industrial convenience yield to speak of, and is held as a store of value rather than consumed, so its curve sits in steady *contango* (spot plus the full cost of carry) essentially all the time. A long gold-futures position therefore *bleeds* a small roll cost rather than earning carry — which is one more reason gold is treated as a monetary asset in [its own series](/blog/trading/gold/is-gold-money-a-commodity-or-a-currency-the-framing-that-decides-everything) rather than as a consumption commodity. The carry premium lives in the commodities that are *used up* — burned, eaten, smelted — because only a consumable can be genuinely scarce *right now* in a way that backwardates the curve. A thing you merely store is never scarce in that sense.

## How it shows up in real markets

**Crude oil, 2021-2022: the carry tailwind.** As the world reopened after the pandemic and OPEC+ kept supply tight, crude moved into steep, persistent backwardation through late 2021 and most of 2022. Spot crude rose, yes — but a meaningful chunk of the *total* return to a long crude position over that stretch came from the roll, not the level. Holders rolling each month sold the dear front and bought the cheaper next, harvesting a double-digit annualized carry on top of the price move. This is normal backwardation in its natural habitat: a tight, demand-led oil market where producers were hedging into strength and longs were paid handsomely to carry the risk.

**Natural gas: the perpetual carry tax.** US natural gas is the textbook *anti-carry* commodity. Because gas is expensive and seasonal to store, the curve is frequently in steep contango — you buy cheap summer gas and pay up for dear winter gas, and the roll bleeds. A long-only gas holder, or a broad index overweight gas, eats a structural roll tax that can run double digits a year. This is why a carry strategy is usually *short* the gas leg, collecting the bleed that the long-only holder suffers — and why "just buy commodities for the long run" has been a poor idea for anyone who bought the gas-heavy part of the basket.

**The nickel break of March 2022: backwardation meets the steamroller.** Nickel had been tight and the LME curve stressed when a massive short squeeze sent the price from around \$25,000 to over \$100,000 a tonne in two days, and the exchange cancelled trades. The detail is in [Aluminum, nickel, and the 2022 nickel squeeze](/blog/trading/commodities/aluminum-nickel-and-the-2022-nickel-squeeze-when-the-market-broke), but the lesson for carry is sharp: a tight, low-cushion metals market is precisely the kind of structure where a shock produces a violent, discontinuous move. Carry traders positioned in stressed metal curves are short exactly this tail. The premium they collect in the calm months is the price of standing in front of a break that, when it comes, comes all at once.

**The financialization compression.** Academic and practitioner work on the post-2004 inflows into commodity index products documents a measurable *flattening* of the historical roll premium as front-month contracts were crowded by passive longs. The premium did not vanish, but it thinned — a real-world example of the "everyone crowds the trade" mechanism. A carry harvester adapted by rolling *away* from the crowded front (using deferred or optimized roll schedules) rather than buying the same congested contract as everyone else, capturing slope where the crowd was not. The premium survives, but it migrates away from where the money is most concentrated.

**The 2014-16 oil crash: backwardation flips to super-contango.** Through the first half of the 2010s, oil sat in steady backwardation — a tight, demand-led market where long holders were paid to roll. Then US shale flooded the world with crude, OPEC declined to cut, and over 2014-15 the price collapsed from over \$100 to under \$30. The relevant point for carry is what happened to the *curve*: the glut pushed oil into deep contango, because suddenly there was a surplus that had to be stored, and storage demanded a premium. Carry investors who had been long the backwardated curve were hit twice — once by the spot crash and once by the curve flipping from a tailwind to a headwind. The episode is the textbook illustration of the steamroller: years of pleasant backwardated carry surrendered as a tight market gorged on new supply and broke. The shale story behind it lives in [The shale revolution](/blog/trading/commodities/the-shale-revolution-how-the-us-became-the-swing-producer).

**The 2000s supercycle: backwardation as a tell.** Through the China-led commodity supercycle of the 2000s, broad swaths of the metals and energy complex spent long stretches in backwardation — the curve was telling you, in real time, that the physical world was *tight*, that demand was outrunning supply, and that inventories were being drawn down. A carry harvester who simply followed the slope was structurally long the right commodities during the boom, not because they forecast Chinese demand but because the backwardation *was* the signal. The cross-asset allocation lens on that cycle is in [the 2000s China commodity supercycle case study](/blog/trading/cross-asset/case-study-2000s-china-commodity-supercycle); the carry lesson is that the curve shape carried the macro information for free.

## The playbook: how a systematic investor harvests commodity carry

Pull it together into how a curious investor should actually think about commodity carry — not as a long-only "own the basket" bet, but as a curve-aware, deliberate harvest of a real premium with a known crash risk.

**Tilt toward the slope, not the level.** The first and most important shift is to stop thinking "is oil cheap?" and start thinking "what is the *shape* of oil's curve?" The carry investor's primary signal is the slope: backwardation is a tailwind, contango is a headwind, and the slope is worth roughly plus-or-minus a dozen percent a year before any price move. Before going long *anything* in commodities, check the shape — a long-only position in a contangoed market is fighting a structural drag, no matter how right you are about the price.

**Rank cross-sectionally; go long the payers, short the bleeders.** The premium is most cleanly harvested as a long-short carry factor: rank the commodity universe by curve slope, go long the most backwardated, short the most contangoed, and collect the spread between the two rolls. This cancels much of the flat-price market risk and isolates the carry. It is the opposite of the long-only index, which buys both legs and eats the bleed on one.

**Budget for the tail.** A carry strategy that does not respect the crash risk is a strategy that will eventually be destroyed by it. Backwardation is a tight market; tight markets break; you are short the tail. Size the position so that a sudden curve flip — a 20%+ collapse in a prompt you are long — is survivable, not fatal. The strategies that have harvested carry for decades are the ones that treated the occasional steamroller as a *budgeted cost*, not an unforeseeable disaster. Many overlay an explicit tail hedge (cheap out-of-the-money puts on the long leg) to cap the worst outcome, accepting a slightly lower average carry in exchange for surviving the break — the protective-put logic from [hedging a portfolio with options](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk).

**Watch for crowding and structural shifts.** The premium thins when everyone harvests it and disappears when a commodity goes structurally contango. Watch the front-month positioning (the financialized crowd), roll *away* from the congested contract when the crowd is heavy, and accept that some commodities (gas, often gold) are carry *costs* on the long side, not sources. The premium migrates; the harvester follows the slope, not the crowd.

**Mind breadth, turnover, and costs.** The carry premium is harvested *across a diversified set* of curves, not concentrated in one or two. Breadth matters because any single commodity's carry can reverse violently; spreading the long and short legs across energy, metals, grains, and softs smooths the harvest and dilutes the steamroller. But breadth costs money: commodity futures have real bid-ask spreads and roll-execution costs, and a strategy that rebalances too often or trades the illiquid contracts can give back much of the premium in frictions. The practical discipline is to rebalance on a sensible cadence (monthly is common), to trade liquid contracts, to roll *gradually* rather than all at once on the same congested day everyone else rolls, and to weight the legs by liquidity and risk rather than equally. A carry strategy is only as good as its net-of-cost implementation; the gross premium on a chart is not the premium you keep.

**Read it as a macro signal too.** Beyond the return, the carry's existence *tells you something*. Widespread backwardation across the energy and metals complex says the physical economy is *tight* — demand is outrunning supply, inventories are low, and inflationary pressure is building from the real side. Persistent contango says the opposite: glut, slack, and disinflation. The carry is not only a return source; it is a read on the state of the physical world, which is why commodity curve shape sits in the macro toolkit alongside the level — see [commodities as macro signals](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold).

Here is the spine of this series, made explicit one last time. A commodity price is a physical thing forced through a financial contract. The forward curve, the cost of storage, and the convenience yield are the gears — and they decide who profits from the roll. The long-only investor who buys the basket blind lets those gears grind them down in contango. The carry investor reads the gears, stands on the backwardated side where the roll pays, shorts the contangoed side where it bleeds, and collects the producers' insurance premium that Keynes identified a century ago — knowing full well that the premium is the wage for absorbing a tail risk that, every so often, comes due. Backwardation is not a forecast. It is a *structure* — and a structure, unlike a forecast, can be harvested.

## Further reading & cross-links

- [Roll yield and why long-only commodity ETFs bleed](/blog/trading/commodities/roll-yield-and-why-long-only-commodity-etfs-bleed) — the mirror image of this post: the same roll mechanism as a *cost* in contango.
- [Contango vs backwardation: what the shape of the curve means](/blog/trading/commodities/contango-vs-backwardation-what-the-shape-of-the-curve-means) — the two curve shapes and why slope, not level, drives the long-only return.
- [The four players: producers, consumers, hedgers, and speculators](/blog/trading/commodities/the-four-players-producers-consumers-hedgers-and-speculators) — who is on the other side, and why the speculator is paid.
- [Convenience yield and the cost of carry](/blog/trading/commodities/convenience-yield-and-the-cost-of-carry-why-the-curve-has-a-shape) — the physical reason a tight market backwardates.
- [Spot vs futures: the two prices of the same barrel](/blog/trading/commodities/spot-vs-futures-the-two-prices-of-the-same-barrel) — why a futures contract converges to spot at delivery.
- [Aluminum, nickel, and the 2022 nickel squeeze](/blog/trading/commodities/aluminum-nickel-and-the-2022-nickel-squeeze-when-the-market-broke) — what a tight-market break looks like in practice.
- [Energy, oil and gas: the inflation engine](/blog/trading/cross-asset/energy-oil-gas-the-inflation-engine) — the allocation lens on owning the energy complex.
- [Commodities as macro signals: oil, copper, gold](/blog/trading/macro-trading/commodities-as-macro-signals-oil-copper-gold) — reading the curve as a tell on the state of the physical economy.
- [Hedging a portfolio with options: protective puts, collars, and tail risk](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk) — how a carry harvester caps the steamroller.
- [Gold futures: COMEX, contango, backwardation, and paper vs physical](/blog/trading/gold/gold-futures-comex-contango-backwardation-and-paper-vs-physical) — why the near-monetary metal is a carry *cost*, not a source.

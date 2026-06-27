---
title: "Commodity Valuation: How Gold, Oil, and Raw Materials Are Priced"
description: "Commodities have no earnings. Learn how spot/futures pricing works via cost-of-carry theory, why gold is a monetary metal, how oil is priced, and what contango vs backwardation signals to traders."
date: 2026-06-27
publishDate: 2026-06-27
tags: ["commodity valuation", "futures pricing", "cost of carry", "gold valuation", "oil pricing", "contango", "backwardation", "hedging", "commodity markets", "SJC premium", "OPEC"]
category: "trading"
subcategory: "Finance"
author: "Hiep Tran"
featured: false
readTime: 45
---

> [!important]
> **TL;DR** — Commodities have no earnings or dividends, so price is determined by physical supply/demand equilibrium plus the mathematics of holding the commodity through time.
>
> - The core formula is **F = S × e^(r+u-y)T** — futures price equals spot price grown by financing and storage costs, shrunk by convenience yield.
> - **Gold is a monetary metal**, not just an industrial input — it trades primarily against real interest rates and currency debasement fears, with near-zero convenience yield.
> - **Oil is priced at the intersection of OPEC quotas, shale breakeven costs, and demand cycles** — WTI and Brent differ by geography and grade.
> - **Contango** (futures > spot) is normal; **backwardation** (futures < spot) is a signal of immediate scarcity and pays a roll yield to long holders.
> - Valuing a **commodity company** (miner, oil producer) is not the same as holding the commodity — the stock adds operating leverage, costs, and management risk on top of the raw commodity exposure.

---

On April 20, 2020, the price of West Texas Intermediate crude oil fell to negative \$37.63 per barrel. Not low. Not cheap. **Negative.** Sellers were paying buyers to take oil off their hands.

To anyone who had only ever seen oil trade in two-digit or three-digit positive numbers, this was incomprehensible. Oil is real. It exists. You can touch it. How can something physical, something that powers the global economy, have a *negative* price?

The answer lies entirely in the economics of commodity valuation — specifically in storage costs and what happens when storage runs out. In April 2020, COVID-19 had cratered demand overnight while production continued. The Cushing, Oklahoma storage hub — the physical delivery point for WTI contracts — was nearly full. Traders holding futures contracts that expired the next day faced a stark choice: take physical delivery of oil they had nowhere to put, or pay someone else to take the contract. Paying \$37.63/barrel to offload the obligation was cheaper than renting emergency storage or defaulting on delivery commitments.

This episode crystallizes the core truth about commodity markets: **commodities are physical things, not paper claims on future earnings.** They take up space. They can spoil. They can overflow storage. Their price at any given moment is not a discounted stream of future cash flows — it is the clearing price between people who have the physical stuff and people who need it, adjusted for what it costs to bridge the gap between now and the future. 

Understanding that adjustment — the mathematics of cost of carry, convenience yield, and the futures curve — is the entire subject of commodity valuation. It applies whether we are talking about crude oil, gold, natural gas, copper, wheat, or coffee. The formulas are the same. What varies is which inputs dominate, and why.

This post builds the full picture from first principles: what determines the spot price, how the futures curve is constructed mathematically, and then the specific mechanics of gold and oil — two commodities so economically important and behaviorally distinct that they deserve dedicated treatment. Along the way we will cover contango and backwardation, how producers and consumers hedge using futures, and how valuing a commodity company differs fundamentally from holding the commodity itself.

![Commodity pricing layers — stack diagram showing spot, carry costs, and futures price](/imgs/blogs/commodity-valuation-gold-oil-futures-pricing-1.png)

---

## Foundations: Why Commodities Are Priced Differently

Every valuation method we use for financial assets starts with the same basic question: *what future cash flows does this asset generate, and how much are they worth in today's money?* For a stock, we project earnings, dividends, or free cash flow and discount them back. For a bond, we discount the coupon stream plus par value. For real estate, we capitalize rental income. The [time value of money](/blog/trading/asset-valuation/time-value-of-money-engine-every-valuation-model) is always the engine running underneath.

Commodities break this framework completely. A barrel of WTI crude oil does not pay dividends. An ounce of gold does not generate earnings. A metric ton of copper does not have a cash flow statement. If you hold physical gold in a vault for five years, you still have an ounce of gold — not an ounce of gold plus interest. In fact you have *less* than an ounce of gold's worth in real terms, because you paid insurance premiums and storage fees along the way.

This absence of income means the standard DCF toolkit is useless for pricing physical commodities. Instead, commodity prices are governed by three distinct forces:

**1. Supply and demand equilibrium.** At any given moment, the spot price is simply the price at which the amount of the commodity being produced and sold matches the amount being consumed and bought. This sounds trivial, but the complexity is immense: production is slow to adjust (you cannot open a copper mine overnight), demand can shift abruptly (a global recession slashes industrial metal demand in weeks), and inventories act as a buffer that smooths but cannot eliminate price swings. The spot price is the market's real-time answer to the question: *at what price do the marginal supplier and marginal buyer agree to trade right now?*

**2. Cost of carry.** Once we know the spot price, we can compute the *fair value* of any futures contract — an agreement to buy or sell the commodity at a fixed price on a future date. The futures price is not a separate forecast; it is derived mathematically from the spot price via the cost-of-carry formula. To see why, consider a simple arbitrage argument: if I can buy oil today at \$80, store it for three months at \$2.40, finance the purchase at \$1.00, and sell it three months from now at a futures price of \$83.40 — I have a riskless profit. Markets compete away these opportunities, which forces the futures price to embed exactly these costs. We will derive the formula precisely in the next section.

**3. Convenience yield.** This is the subtlest concept, and the one most often ignored. Physical possession of a commodity has value beyond just the financial return on a futures contract. A refinery that holds physical crude oil can run its operations today without waiting for futures delivery. An airline that holds jet fuel inventory is protected if the spot price spikes next month. A trader who holds physical copper can satisfy a customer order immediately. This *operational optionality* of having the physical thing in hand is called convenience yield — and it acts as a negative cost that reduces the fair-value futures price. When convenience yield is high, futures trade *below* spot (backwardation). When it is low or zero, futures trade *above* spot (contango).

The contrast with equities is instructive. When you run a DCF on Apple, you forecast revenues, margins, and reinvestment for five to ten years, then estimate a terminal value. Every number in that model reflects beliefs about a business: how competitive Apple's ecosystem will be, whether iPhone demand grows, whether margins expand or compress. For a barrel of oil, there is no business to model. The fair futures price for oil in three months is almost entirely mechanical: spot price plus financing plus storage minus convenience yield. The *level* of the spot price itself comes from macro supply/demand — OPEC decisions, shale production rates, Chinese industrial demand — but given that level, the futures price follows arithmetically.

**Physical vs financial commodities.** A further distinction matters for understanding valuation. Some commodities are purely physical: crude oil, wheat, natural gas. To take delivery, you need actual storage infrastructure — a tank farm, a grain elevator, a liquefied natural gas terminal. This physical constraint creates real limits on arbitrage and can cause dramatic price dislocations when storage fills up (as in April 2020) or runs short. Other commodities, particularly gold, have very large above-ground stocks relative to annual production flows, and gold is easily stored in small volumes relative to its value. This makes gold's market structure quite different: its convenience yield is low, its carry costs are modest, and its futures market is highly efficient.

**What cannot be valued with a commodity framework.** Importantly, while you can derive fair-value futures prices from the spot price using cost-of-carry math, this does not tell you what the *spot price should be*. The spot price of oil is not derivable from first principles — it emerges from the interaction of millions of producers, consumers, traders, and policy actors. What valuation theory gives us is the *relative pricing* of futures versus spot, and the internal consistency of the futures curve. For absolute commodity price forecasting, you need to model supply and demand fundamentals — a different and far messier discipline.

**Inventory cycles and the commodity supercycle.** One of the most important features of commodity markets that differentiates them from financial markets is the role of inventory cycles. When commodity prices are high, producers invest in new capacity — new mines, new wells, new refineries. This investment takes years to complete. Meanwhile, demand might soften. The result is a commodity price cycle that can span a decade or more: rising prices stimulate investment, investment eventually creates oversupply, oversupply pushes prices down, low prices kill investment, supply eventually tightens, and prices rise again. The commodity supercycle of 2002–2012, driven by Chinese industrialization, followed exactly this pattern. Understanding that commodity spot prices are not random walks but are anchored by these fundamental supply cycles is essential context for any valuation exercise.

**Seasonality and basis.** Many agricultural and energy commodities exhibit strong seasonal patterns in both spot prices and the futures curve. Natural gas in the US is typically cheapest in spring (low heating demand, high storage levels) and most expensive in winter (high heating demand, low storage). Corn prices typically peak in late spring before the US harvest and trough in autumn when harvest abundance hits markets. These seasonal patterns create predictable but not risk-free trading opportunities, and they explain a significant portion of the variation in the futures term structure across calendar months. For energy commodities, the front-month contract often diverges significantly from seasonal norms when extreme weather events disrupt supply/demand balances — as in the February 2021 Texas deep freeze, which temporarily pushed natural gas at the Waha hub in West Texas to -\$150/MMBtu as pipeline capacity froze and production collapsed.

**The role of commodity indices.** A large amount of commodity exposure in financial markets is held not through direct futures positions but through commodity indices — instruments like the Bloomberg Commodity Index or the S&P GSCI. These indices hold baskets of commodity futures with rules-based allocation and periodic rebalancing. When investors buy commodity index ETFs, the flow of money into futures positions can affect futures prices independently of underlying physical supply and demand — a phenomenon sometimes called "financialization" of commodity markets. Research has shown that since the early 2000s, commodity prices have become more correlated with financial markets, partly because the same investor base (institutional asset managers) holds both equities and commodity index products. During a market panic (such as March 2020), everything was sold — including commodity futures — driving prices down even when physical supply/demand fundamentals had not yet shifted.

---

## The Cost-of-Carry Framework

The cost-of-carry framework is the mathematical backbone of all commodity futures pricing. Its logic is simple: if physical storage and forward delivery are both possible, then the futures price must equal the cost of owning the commodity today and delivering it at the contract's expiry. Any deviation creates an arbitrage opportunity that sophisticated traders will immediately close.

**Derivation from first principles.** Suppose the spot price of a commodity today is S. You want to agree today on a price to buy or sell the commodity at time T (measured in years from now). What is the fair forward price F?

To replicate a long futures position (the obligation to buy the commodity at time T at price F), you can:
1. Borrow \$S at the risk-free rate r.
2. Buy one unit of the commodity today at the spot price S.
3. Store the commodity from now until time T, paying storage costs at continuous rate u per year.
4. Receive the convenience yield at continuous rate y per year (the operational benefit of holding physical).
5. Deliver the commodity at time T.

The total cost of this strategy, compounded continuously, is:

**F = S × e^(r + u − y) × T**

If the actual futures price is *above* this level, a trader can buy physical, store it, and sell futures — locking in a riskless profit. If the actual futures price is *below* this level, a trader can sell physical short, lend out the proceeds, and buy futures — again locking in a riskless profit. Competition pushes the futures price to exactly F = S × e^(r+u-y)T.

**Defining each term:**

- **S** (spot price): the current cash price for immediate delivery of the commodity. The spot market is where physical buyers and sellers meet.
- **r** (risk-free rate): the cost of financing the purchase. If you borrow money at 5% per year to buy the commodity, that cost must be reflected in the futures price.
- **u** (storage cost): warehousing, transportation, insurance, and any quality-loss costs expressed as a fraction of the commodity's value per year. For oil, this runs roughly 3–5% of value per year. For gold, which is dense, non-corrosive, and easily vaulted, storage is closer to 0.15–0.30% per year.
- **y** (convenience yield): the non-monetary benefit of holding physical inventory. This is hard to observe directly but can be inferred from the spread between futures and spot. When supply is tight, y is large, pulling futures below spot. When supply is ample, y approaches zero and futures trade above spot.
- **T** (time to expiry): measured in years as a fraction. A 6-month futures contract has T = 0.5.

The formula uses continuous compounding (the mathematical notation e^x) for precision, but you can approximate it for short time horizons with simple arithmetic: F ≈ S × (1 + r + u − y) × T for small T. The difference between continuous and simple compounding is negligible for contracts under a year.

![Cost of carry pipeline — spot price flows through financing, storage, and convenience yield to produce futures price](/imgs/blogs/commodity-valuation-gold-oil-futures-pricing-2.png)

#### Worked example:

**Gold 6-month futures pricing.** Suppose gold spot is trading at \$1,900 per troy ounce. The risk-free rate is 5% per year. Gold storage costs 0.30% per year (insurance, vault fees at a reputable custodian). Gold has essentially zero convenience yield — there is no operational advantage to holding physical gold versus a futures contract for industrial purposes, since gold's industrial consumption is small relative to its stock.

Plugging into F = S × e^(r+u-y)T:

- r = 0.050 (5% per year)
- u = 0.003 (0.30% per year)
- y = 0.000 (near zero for gold)
- T = 0.5 (six months)
- Combined exponent = (0.050 + 0.003 − 0.000) × 0.5 = 0.053 × 0.5 = 0.0265
- F = \$1,900 × e^0.0265 = \$1,900 × 1.0268 = **\$1,950.90**

The six-month gold futures contract should trade at approximately \$1,951, about \$51 above spot. This premium is purely a carry cost, not a forecast that gold will be worth more in six months. If actual futures are at \$1,960, there is \$9/oz of riskless profit for a trader who can buy physical, store it, and sell the futures. In practice these opportunities close within seconds in liquid markets.

**Intuition check:** the cost-of-carry premium is small for gold because storage is cheap and there is no financing urgency. For oil, where storage is much more expensive and convenience yield is volatile, the spread between spot and futures can move dramatically — as the April 2020 episode demonstrated.

---

## Gold: The Monetary Metal

Gold occupies a unique position in the commodity universe: it is simultaneously a physical commodity with genuine industrial uses (electronics, dentistry, aerospace) and a *monetary metal* — a store of value with a 5,000-year track record and a balance-sheet role in central bank reserves. This dual identity creates a pricing behavior quite unlike that of oil, copper, or agricultural goods.

**Why gold is different from other commodities.** The world mines approximately 3,300 metric tons of gold per year. But the above-ground stock of gold — all the gold ever mined — exceeds 200,000 metric tons. The ratio of stock to flow is roughly 60 years, meaning the global gold inventory is equivalent to 60 years of current mine production. No other commodity has this characteristic. Oil's stock-to-flow ratio is measured in months. Even silver's is under two years. This enormous stock-to-flow ratio means that annual mine production barely affects the global gold supply. The relevant supply is the stock of existing gold, most of which sits in vaults, jewelry boxes, and central bank reserves around the world, potentially available at the right price.

This structure creates near-zero convenience yield for gold. Unlike oil, which refineries need to keep running, or copper, which manufacturers need for production, most gold holders are not using their gold operationally. They are holding it as a store of value. The operational urgency that drives convenience yield in industrial commodities is largely absent. A gold futures contract and a physical gold bar are near-perfect substitutes for most holders.

**The real interest rate relationship.** If gold earns nothing but competes with government bonds that pay a real return, then gold's opportunity cost is the real yield on safe assets — most commonly approximated by the yield on US Treasury Inflation-Protected Securities (TIPS). When real yields are high, the opportunity cost of holding non-yielding gold is high, and gold tends to be cheap. When real yields are negative — as they were in 2020–2021 when the Federal Reserve held nominal rates near zero while inflation ran above 5% — the opportunity cost of holding gold is actually *negative*. Investors are rewarded, in real terms, for owning gold rather than Treasury bonds.

This is why gold rallied to \$2,075/oz in August 2020 as the Fed cut rates to zero and the 10-year TIPS yield fell to -1.0%: holding gold was beating the real return on government bonds. And it partially explains why gold sold off in 2013 (TIPS yields rose as the market anticipated Fed tapering) and in 2022 (aggressive rate hikes sent real yields sharply positive even as inflation rose).

**Central bank demand.** Since the global financial crisis, and accelerating after the West froze Russia's foreign reserves following the 2022 Ukraine invasion, central banks in the Global South have been buying gold aggressively. Countries including China, India, Poland, Turkey, and many others have added hundreds of tons to their reserves, explicitly reducing dollar exposure. Central bank demand has averaged over 1,000 metric tons per year since 2022 — nearly a third of annual mine production. This structural demand has supported gold prices even in periods when Western investors were net sellers.

**Gold's price drivers in summary:** The gold price reflects (1) the real yield environment, (2) currency debasement fears (demand for an alternative to fiat), (3) central bank reserve diversification, and (4) physical demand from jewelry (especially India and China) as a distant fourth. Most professional gold investors focus almost entirely on the first two.

![Gold price history 2010 to 2024 with key events annotated](/imgs/blogs/commodity-valuation-gold-oil-futures-pricing-5.png)

![WTI oil futures term structure — contango vs backwardation curves by month to expiry](/imgs/blogs/commodity-valuation-gold-oil-futures-pricing-6.png)

**The Vietnam SJC context.** Vietnam is one of the world's largest per-capita gold-consuming countries, with a deeply embedded cultural tradition of gold savings. The State Bank of Vietnam (SBV) holds a near-monopoly on legitimate gold trading through Saigon Jewelry Company (SJC) branded bars, which are the only form legally used in domestic transactions. Decree 24 (2012) effectively banned other gold bar brands from trading, creating a captive domestic market.

The result is a persistent and often extreme premium of SJC gold over international prices. When global gold trades at \$2,300/oz (approximately 71 million VND per tael — a Vietnamese unit equal to roughly 1.2 troy ounces), SJC bars often trade at 80 million VND or above. This premium is not explained by the cost-of-carry formula. The formula assumes free arbitrage: if domestic gold is expensive, import more gold. But Vietnam severely restricts gold imports, controlling who can bring gold into the country and in what quantities. With no import safety valve, domestic demand shocks (such as the 2023–2024 period when Vietnamese savers rushed into gold amid currency concerns) cannot be arbitraged away. The premium simply reflects captive demand paying whatever the local market bears.

#### Worked example:

**SJC premium calculation.** In mid-2024, the global gold spot price reached approximately \$2,300 per troy ounce.

- 1 troy ounce = 31.1 grams
- 1 Vietnamese *tael* (or *lượng*) = 37.5 grams
- So 1 troy oz ≈ 0.829 tael, or 1 tael ≈ 1.206 troy oz

At \$2,300/oz and an exchange rate of approximately 25,000 VND/USD:
- 1 troy oz in VND = \$2,300 × 25,000 = 57,500,000 VND
- 1 tael at international price = 57,500,000 × 1.206 = **69.35 million VND/tael**

At the same time, SJC gold was trading at approximately **81–82 million VND/tael** in domestic shops, a premium of about **17–18%** over international parity.

This premium persisted because:
1. The SBV controlled import licenses and only released gold in small batches
2. No private entities could import gold bars legally
3. Local demand from Vietnamese savers was surging due to VND depreciation concerns
4. The domestic market was effectively isolated from global arbitrage

For a Vietnamese saver, paying an 18% premium for domestic gold versus international gold still made sense if they feared further VND devaluation exceeding 18% — effectively a forced bet on the domestic gold market structure, not global commodity pricing.

---

## Oil Valuation: Where Supply and Demand Meet Geopolitics

If gold is the monetary metal, oil is the *geopolitical metal*. The price of crude oil is simultaneously a commodity price, a strategic weapon, a measure of industrial health, and a function of decisions made in Riyadh, Moscow, Washington, and Beijing. No commodity has more moving parts in its valuation.

**WTI vs Brent.** Two benchmarks dominate global oil pricing. **West Texas Intermediate (WTI)** is a light, sweet crude produced in the US and delivered physically at Cushing, Oklahoma — the benchmark for US oil futures on the CME (NYMEX). **Brent crude** is produced in the North Sea and is the benchmark for European and globally traded oil (ICE). Historically Brent traded at a slight premium to WTI due to logistical factors; the US shale boom reversed this for a time as Cushing filled up and WTI discounted heavily. Today WTI and Brent typically trade within \$2–5 of each other, with the spread varying based on US export capacity, pipeline bottlenecks, and quality differences.

**The production cost floor.** One of the most important concepts in oil valuation is the production cost floor — the minimum price at which producers can profitably drill new wells. This matters because at prices below the breakeven, new investment stops, supply eventually falls, and prices recover. The floor is not a single number; it is a supply curve. The lowest-cost oil comes from Middle Eastern supergiant fields: Saudi Aramco can produce oil for \$3–8 per barrel all-in. The marginal barrel is much more expensive.

US shale (tight oil) has become the swing producer in the global market. Shale wells have high upfront drilling costs but relatively low ongoing operating costs, and new wells can be permitted, drilled, and brought online in 60–90 days. The breakeven for typical Permian Basin shale ranges from roughly \$40 to \$55 per barrel (full-cycle, including capital costs). This effectively creates a floor: when oil drops below \$45, US shale drilling rigs are idled, supply growth stalls, and the market typically finds support. When oil is above \$70, shale producers drill aggressively, adding supply.

**OPEC as price manager.** The Organization of Petroleum Exporting Countries (OPEC) and its partners (Russia, UAE, and others in "OPEC+") collectively control roughly 40% of world oil production and hold the majority of proven low-cost reserves. OPEC functions as a production cartel: it sets production quotas among members to manage supply and, by extension, price. When the cartel is disciplined and adheres to quotas, it can maintain prices above \$70. When discipline breaks down (2014–2016 Saudi-Russia price war, 2020 COVID collapse), prices can plunge to \$30 or lower before eventually recovering.

**Demand cycles.** Oil demand is relatively inelastic in the short run — businesses and consumers cannot immediately change their oil consumption when prices move. Over the medium term, demand responds to prices (vehicle efficiency standards, mode substitution, industrial process changes) and to economic growth. China's industrialization from 2000–2015 was the single largest demand driver of the commodity supercycle that took oil from \$20 to \$147 in 2008. The energy transition — electric vehicles, renewable power — is now reducing the long-term demand growth outlook, though current global oil demand continues to set records.

#### Worked example:

**Oil futures arbitrage and implied convenience yield.** WTI spot is trading at \$80 per barrel. The 3-month futures contract is trading at \$82. Storage at Cushing costs \$0.80 per barrel per month (including insurance and logistics). The risk-free rate is 5% per year (1.25% for 3 months).

Under the cost-of-carry formula, the fair-value futures price with zero convenience yield would be:
- Storage for 3 months: \$0.80 × 3 = \$2.40/barrel
- Financing: \$80 × 0.05 × 0.25 = **\$1.00**/barrel
- Fair F (zero convenience yield) = \$80 + \$2.40 + \$1.00 = **\$83.40**

The actual futures price is \$82, which is **\$1.40 below** the zero-convenience-yield fair value. This gap implies a positive convenience yield. To solve for y:

Using the approximate form: \$82 = \$80 × (1 + (0.05 + u − y) × 0.25)
Implied: (0.05 + 0.03 − y) × 0.25 = 0.025 → y ≈ 0.03 − 0.025/0.25 × ... 

More directly: implied y = (Fair F − Actual F) / (S × T) = (\$83.40 − \$82.00) / (\$80 × 0.25) = \$1.40 / \$20 = **7% per year (annualized)**

This 7% convenience yield tells you that the market is pricing in meaningful scarcity value of physical oil at Cushing right now — refineries and traders are willing to accept a lower futures price (i.e., pay a premium for physical delivery) because they value having the oil in hand immediately. When this convenience yield disappears (storage fills up, no urgency), the futures will move back above spot, and you will see a contango structure.

---

## Contango and Backwardation

The terms **contango** and **backwardation** describe the shape of the futures curve — the relationship between the current futures price and contract expiry dates.

**Contango** exists when futures prices are *higher* than the current spot price, and further-dated contracts are priced above near-dated ones. This is the "normal" condition for most storable commodities when supply is abundant. The futures curve slopes upward because it costs money to store and finance the commodity, and the market needs to compensate holders for those costs. If you look at the WTI futures curve in a normal market, you might see: spot at \$80, 1-month at \$80.80, 3-month at \$82.50, 6-month at \$85.00. Each step up reflects accumulated carry costs.

**Backwardation** exists when futures prices are *below* spot, and longer-dated contracts are cheaper than near-dated ones. The curve slopes downward. This happens when the market desperately needs the commodity *right now* — tight current supply, high immediate industrial demand, or fear of disruption. The spot price trades at a premium because physical holders are scarce and buyers are competing for immediate delivery. Examples: oil during supply disruptions, natural gas during cold snaps, agricultural commodities before a harvest when inventories are low.

**Why does this matter for investors?** The term structure of futures has enormous implications for investors who hold commodity exposure via futures (as most ETF and index investors do). When you hold a long futures position, you must "roll" expiring contracts into later-dated ones periodically. In contango, you sell the cheaper near contract and buy the more expensive far contract — this rolling process destroys value. In backwardation, you sell the more expensive near contract and buy the cheaper far contract — this rolling process *creates* value.

This is called **roll yield**, and it can dwarf the impact of the actual commodity price movement. During 2009–2011, oil prices were relatively stable, but investors in WTI futures lost money consistently due to heavy contango (sometimes 5–10% per month roll cost). Conversely, during periods of heavy backwardation (2021–2022 energy crisis), investors in oil futures earned significant roll yield on top of price appreciation.

![Contango vs backwardation comparison — before-after showing different market structures](/imgs/blogs/commodity-valuation-gold-oil-futures-pricing-3.png)

#### Worked example:

**Roll yield in backwardation.** The oil market is in backwardation. The 1-month futures contract is trading at \$85 per barrel. The 2-month contract is at \$83. You are long the 1-month contract and it is approaching expiry.

**Rolling the position:**
- You sell your 1-month contract (now nearly expiry) at approximately \$85 (the spot price at expiry, since near-dated futures converge to spot).
- You buy the 2-month contract at \$83.
- Net gain from the roll: \$85 − \$83 = **\$2 per barrel**.

If you roll every month and the curve maintains this shape (not guaranteed), the annualized roll yield is approximately:
- Roll yield per month ≈ \$2 / \$83 ≈ 2.4% per month
- Annualized ≈ 2.4% × 12 ≈ **28.9% per year**

This is a hypothetical example with an extreme backwardation structure, but similar dynamics played out in 2021–2022. During the energy crisis, oil futures were in steep backwardation and long commodity investors earned substantial roll yield simply from the curve structure, independent of whether oil prices went up or down. The intuition is simple: in backwardation, the market is *paying you* to hold the commodity forward in time, because the spot market urgently needs physical oil right now more than the future market does.

**When should you expect contango vs backwardation?** As a rule of thumb: contango dominates when inventories are comfortable and supply is adequate; backwardation dominates when inventories are low and supply is tight. Watching the shape of the futures curve gives you a real-time read on market tightness that is often more reliable than any supply/demand forecast.

---

## How Producers and Consumers Hedge

Commodity markets exist primarily to serve the hedging needs of real-economy participants — the oil companies, mining firms, airlines, food manufacturers, and utilities that produce or consume commodities in their core business. Speculation by financial investors provides liquidity, but the fundamental purpose of futures markets is risk transfer between producers and consumers.

**Why producers hedge.** An oil company that extracts crude at \$35 per barrel faces devastating uncertainty if oil prices can swing from \$40 to \$120. At \$40, the company barely covers costs. At \$80, it earns enormous profits. At \$40 again, it cannot service its debt and may face bankruptcy. For an oil company that needs to plan capital budgets, hire workers, invest in new wells, and service bond obligations, this volatility is existential — not because of any single bad year, but because volatile cash flows prevent rational long-term planning and scare away lenders.

The solution is to sell futures contracts corresponding to expected production. If a company plans to produce one million barrels over the next six months and the 6-month futures price is \$80, it can sell \$80M of futures today. If the oil price falls to \$55 at delivery time, the company sells its physical oil for \$55M in the spot market but gains \$25M on its short futures position (it agreed to sell at \$80, buys back at \$55). Total revenue: \$80M — exactly as planned. If prices rise to \$100, the company gets \$100M from spot sales but loses \$20M on the futures. Total revenue: still \$80M.

The hedge transforms unpredictable commodity revenue into a fixed, predictable cash flow stream. The producer gives up upside in exchange for eliminating downside. This is rational for a company with fixed obligations and a long-term plan — the stability is worth more than the option on commodity price upside.

**Consumer hedging.** The other side of every commodity hedge is a consumer. Airlines are the canonical example: jet fuel typically represents 20–30% of operating costs, and jet fuel prices swing with crude oil. An airline cannot easily pass fuel cost spikes to customers (fares are set months in advance, competition is fierce). So airlines hedge aggressively, buying futures or call options on jet fuel (or crude oil, which correlates closely). Southwest Airlines famously ran one of the most aggressive and successful fuel hedging programs in aviation history, locking in low prices ahead of the 2000s oil price surge and maintaining profitability when competitors were hemorrhaging money.

**The mechanics: futures vs options hedges.** Producers typically use **short futures** — they agree today to sell a fixed quantity at a fixed price. This fully neutralizes price risk in both directions. A variation is to buy **put options** — the right to sell at a guaranteed floor price — which limits downside but preserves upside. Consumers typically use **long futures** (buy at a fixed price) or **call options** (right to buy at a ceiling price). In practice, large commodity users layer multiple instruments: long futures for core certain demand, call options for uncertain upside demand, and basis swaps for location or grade differences.

**Hedge ratios and basis risk.** One of the practical challenges of commodity hedging is basis risk — the risk that the price of the hedging instrument (e.g., WTI crude futures) does not move in perfect lockstep with the actual commodity the producer sells or the consumer buys. A mid-continent refinery might buy crude at a local price that differs from WTI by a regional basis — typically a function of local pipeline capacity, transportation costs, and regional supply/demand. If the refinery hedges its crude input costs using WTI futures but the local basis widens unexpectedly, the hedge is imperfect. Basis risk cannot be eliminated entirely; it is the residual risk after hedging that traders must manage with careful hedge ratio calculations and ongoing monitoring.

The **optimal hedge ratio** is not always 1:1 (matching quantity hedged to futures contracts exactly). It is adjusted for the correlation between the hedging instrument and the exposure:

h* = ρ × (σ_S / σ_F)

Where h* is the optimal fraction of the exposure to hedge, ρ is the correlation between spot and futures price changes, σ_S is the standard deviation of spot price changes, and σ_F is the standard deviation of futures price changes. For most major commodity benchmarks, correlation is very high (0.95+), and σ_S ≈ σ_F, so the optimal hedge ratio is close to 1.0. For cross-commodity hedges (e.g., hedging jet fuel exposure with crude oil futures), the correlation is lower and the hedge ratio must be computed carefully from historical data.

**Mark-to-market and margin calls.** Futures contracts are mark-to-market daily — the gain or loss on a futures position is settled in cash every day through the exchange's clearing house. This creates an important asymmetry for hedgers: if an oil producer has sold futures at \$80 and oil rises to \$90, the producer must post \$10/barrel in additional margin on the short futures position, even though the producer's physical production is worth more. The cash margin call is real and immediate; the offsetting gain on future production is unrealized and spread over months. This cash-flow mismatch has caused serious problems for hedgers in rapidly rising markets — most famously, Metallgesellschaft (MG) in 1993, which had sold long-dated oil supply contracts and hedged with short-dated futures. When oil prices rose, MG faced massive margin calls on its futures, causing a near-bankruptcy even though the total economic position was hedged.

**Costless collars: a practical hedging structure.** A common hedging structure for oil producers that want both a price floor and some upside participation is the **costless collar**. The producer simultaneously buys a put option at \$70/barrel (protecting against price below \$70) and sells a call option at \$90/barrel (capping upside above \$90). The premium received from selling the call funds the cost of buying the put — hence "costless." The result: the producer is guaranteed to receive between \$70 and \$90 per barrel regardless of where spot oil trades. If oil falls to \$50, the put pays the difference to bring total revenue to \$70. If oil rises to \$110, the short call loses money, capping total revenue at \$90. Many US shale producers use costless collars extensively, especially when entering a period of heavy capital investment where stable cash flows are critical to funding drilling programs.

![Producer hedging workflow — pipeline from extraction to stable cash flow](/imgs/blogs/commodity-valuation-gold-oil-futures-pricing-4.png)

![Asset class risk-return scatter — gold highlighted against other asset classes](/imgs/blogs/commodity-valuation-gold-oil-futures-pricing-7.png)

The risk-return chart above shows gold (highlighted in amber) plotted against other major asset classes from 2000 to 2024. Gold's annualized return of 8.9% with 15.8% standard deviation puts it in the mid-tier of the risk-return spectrum — comparable returns to US stocks with slightly lower volatility over this particular period. This period includes two gold bull markets (2001–2011 and 2018–2024), which flatters gold's average return. The key takeaway is not that gold is definitively better or worse than equities, but that it provides a genuinely different return profile, often uncorrelated with stocks — which is exactly what hedgers and portfolio diversifiers need.

---

## Valuing Commodity Companies vs the Commodity Itself

Owning a gold mining stock and owning gold are not the same investment. This seems obvious when stated directly, but many investors conflate them — buying Barrick Gold shares when they want gold exposure, or buying crude oil futures when they want to invest in oil company earnings.

**Operating leverage: the miner's hammer.** A gold miner has fixed costs: drilling equipment, labor, processing facilities, environmental compliance, administration. Suppose a miner's all-in sustaining cost (AISC — the industry standard measure of total production cost) is \$1,200 per ounce. At a gold price of \$1,300, profit is \$100/oz. At \$1,600, profit is \$400/oz — a 4× increase from a 23% price rise. This is **operating leverage**: a given percentage move in the commodity price produces a larger percentage move in profit.

This leverage is a double-edged sword. It amplifies gains but also amplifies losses. If gold falls to \$1,100 (below AISC), the miner loses money on every ounce it produces and may be forced to curtail operations, impair reserves, or raise expensive equity. The miner cannot simply wait on the sidelines the way a physical gold holder can — it has ongoing fixed costs and obligations. This is why gold mining stocks often trade at significant discounts to their estimated gold reserves value; the market applies a large discount for operational risk and management quality.

**NAV approach for miners.** The primary valuation method for mining companies is net asset value (NAV), which estimates the present value of all the gold (or copper, or whatever) in the ground, minus the all-in cost to extract it, discounted at an appropriate rate. The key inputs are:

1. **Proven and probable reserves**: the quantity of metal confirmed to be economically recoverable at current prices and extraction methods (a regulatory classification)
2. **Mining cost profile**: how much it costs per ounce over the mine life, broken down by year
3. **Metal price assumptions**: the price deck used to calculate future revenues — the most contentious input
4. **Discount rate**: the cost of capital for mining projects ([covered here](/blog/trading/asset-valuation/risk-required-return-capm-beta-cost-capital)), which is high due to geopolitical, operational, and commodity price risks, typically 8–12%

A simplified NAV calculation: if a miner has 10 million ounces of proven gold reserves and can extract them at an average all-in cost of \$1,200/oz, and the gold price is \$1,900/oz, the per-ounce profit is \$700. Discounting 10 years of \$700/oz production at a 10% discount rate: NPV/oz ≈ \$700 × 6.14 (present value factor, 10-yr annuity at 10%) / 10 years of reserves ≈ ... in practice you build the full cash flow model year by year, but the core point is that the miner is not worth 10M × \$1,900 = \$19B; it is worth the present value of the spread between revenues and costs, discounted heavily for risk. In this case, perhaps \$700M–\$1.5B depending on cost timing and reserve quality.

**DCF with a commodity price deck.** For oil companies, integrated majors, and diversified miners, analysts use a modified DCF. The key adaptation is the commodity price deck: instead of forecasting a single constant price forever, the analyst builds scenarios (bull, base, bear) for the long-run commodity price and runs separate DCFs. The base case might assume \$75/barrel Brent long-run; the bear case \$55/barrel; the bull case \$95/barrel. Each produces a different value, and the analyst weights them by subjective probability. This scenario-weighted DCF acknowledges the fundamental uncertainty in commodity price forecasting in a way that a single-price DCF cannot.

**Optionality value.** This is where [real options valuation](/blog/trading/asset-valuation/real-options-valuation-flexibility-strategic-investments) becomes critical for commodity companies. A miner sitting on a deposit that is uneconomic at today's gold price of \$1,600 but would be profitable at \$2,000 has a real option — the right (but not obligation) to mine that deposit if prices rise. This option has positive value even though the deposit generates nothing today. Standard DCF fails to capture this because it assumes the company will produce continuously; real options models recognize that management can pause, expand, contract, or abandon operations in response to price signals. Junior miners and exploration companies in particular derive most of their equity value from this option value, not from current production cash flows.

**Commodity company valuation multiples.** Beyond NAV and DCF, commodity companies are also valued using sector-specific multiples that attempt to normalize for commodity price cycles:

- **EV/EBITDA** is commonly used for oil majors and diversified miners. The challenge is that EBITDA is highly sensitive to commodity prices, so analysts typically use "through-the-cycle" price assumptions rather than spot prices. A major oil company might be valued at 5–7× normalized EBITDA using a long-run oil price assumption of \$65–70/barrel, regardless of whether spot is at \$50 or \$100.
- **Price-to-NAV (P/NAV)** is the gold mining–specific equivalent of price-to-book. It compares the current market cap to the estimated net asset value of the company's reserves. Senior producers like Barrick and Newmont typically trade at 0.8–1.3× NAV; junior explorers with higher-risk, unproven deposits trade at 0.3–0.6× prospective NAV. When the gold price is rising and sentiment is positive, producers can trade above 1.0× NAV as investors price in future price appreciation; when gold is falling or costs are rising, P/NAV compresses.
- **EV/oz of reserves or resources** is used to quickly compare acquisition targets. If the market is paying an average of \$150/oz in the ground for gold reserve acquisitions, a company sitting on 5 million proven ounces might be valued at \$750M on this basis — a quick sanity check against any DCF-derived value. The per-ounce metrics vary enormously by reserve quality, jurisdiction risk, and stage of development.

**The dividend question.** For commodity companies at full production, an important valuation driver is dividend and capital returns policy. Oil supermajors — ExxonMobil, Chevron, Shell, BP — have historically been valued partly as yield stocks, with dividend yields of 3–6%. When oil prices collapse, these companies face hard choices: cut dividends (destroying their yield investment thesis) or maintain dividends by taking on debt (destroying balance sheet quality). Many majors committed during the 2014–2016 oil crash to "progressive" or "resilient" dividend policies, maintaining payouts even as oil fell to \$26/barrel. This commitment had value — it preserved the investor base — but it also meant borrowing money to pay dividends when cash flows were negative. The commodity cycle creates structural tension in capital returns that does not exist for stable, earnings-predictable businesses.

**The AISC floor and mine economics.** Understanding all-in sustaining cost (AISC) is essential for gold company valuation. AISC is a standardized metric that includes: cash operating costs (mining, processing, site overhead), royalties and production taxes, sustaining capital expenditures (equipment replacement, mine development to maintain current production), and corporate costs allocated to mining operations. It does *not* include expansion capital (building new mines), acquisition costs, or exploration spending — those are discretionary and cannot be meaningfully allocated per ounce.

For context on the cost curve in 2024: the first quartile of global gold producers had AISC below \$900/oz; the median (50th percentile) was around \$1,300/oz; the 90th percentile (highest-cost marginal producers) was near \$1,800/oz. At a gold price of \$2,300 in 2024, essentially the entire industry was profitable — which is why capital flows into gold equities are typically strong at these gold price levels, and why the industry tends to discipline investment (avoiding overinvestment that would kill the cycle) when prices are well above costs.

---

## Common Misconceptions

**Myth 1: "Gold is a safe investment."** Gold is a low-correlating investment, not a low-volatility one. Its annual standard deviation from 2000–2024 is 15.8% — essentially identical to US large-cap stocks (15.2%). In 2013 alone, gold fell approximately 28%, its worst year in three decades. Anyone who held gold expecting safety was unpleasantly surprised. The correct statement is that gold tends to *not fall at the same time* as equities — its correlations with the S&P 500 are typically near zero or slightly negative — so it provides diversification benefit in a portfolio context. Safety? No. Diversification? Yes.

**Myth 2: "Rising oil demand is bullish for oil stocks."** The relationship between commodity demand and the equity price of producers is mediated by many intervening variables: cost inflation, capital allocation decisions, hedging programs, balance sheet leverage, and supply response. During 2021–2022, oil demand recovered strongly from the COVID crash, and oil prices rose above \$100. But many oil companies underperformed broad market indices because investors worried about ESG mandates, stranded asset risk, and the energy transition. Conversely, during periods of high prices, oil companies that hedged aggressively at lower levels (locking in \$50/barrel when spot was \$90) badly underperformed the commodity itself.

**Myth 3: "Contango means the market expects prices to fall."** Contango simply means the futures price is above spot — which is usually just the result of positive carry costs (financing and storage). It says nothing about where the spot price will be at expiry. If storage costs are 3% per year and financing is 5%, futures should trade 8% above spot for a one-year contract in normal conditions. The "market expectation" embedded in futures prices is not the futures price itself, but the futures price *minus* the expected carry premium. In efficient commodity markets, futures are poor predictors of future spot prices beyond the carry component; the spot price at expiry is driven by supply and demand conditions at that future time, which are not known today.

**Myth 4: "Futures prices are forecasts of future spot prices."** Related to the above: the economics of cost-of-carry shows that futures prices are anchored to *today's* spot price, not a forecast of the future spot. The futures price reflects what it costs to own the commodity today and deliver it in the future, not a market consensus on what the spot price will be. Academic research consistently shows that commodity futures are poor forecasters of future spot prices — in many cases, today's spot price is a better forecast of the future spot price than the futures curve. Futures are pricing mechanisms for carry, not forecasting mechanisms for price level.

**Myth 5: "Commodities always hedge inflation."** Gold does have a long-run correlation with inflation and has served as a reasonable store of value over multi-decade periods. But in the short run (5-year windows), the relationship is unreliable. Gold lost 27% in real terms from 2011 to 2015 despite positive inflation throughout. Oil *caused* some of the 2022 inflation surge but then crashed 40% in 2023. Agricultural commodities hedge food inflation but may be highly correlated with other risks (droughts, war) that you may not want to bet on. The empirical evidence on commodity-inflation hedging is much weaker than the intuition suggests — especially for short holding periods.

---

## How It Shows Up in Real Markets

**April 2020: WTI oil goes to -\$37/barrel.** As described in the introduction, April 20, 2020 was the most dramatic commodity pricing event in modern history. The May WTI futures contract expired the next day. Traders holding long positions faced physical delivery at Cushing, Oklahoma — but Cushing's storage was nearly full. The cost of emergency storage was prohibitive. Unable to take delivery and unable to find buyers, holders sold at any price, driving the settlement to **-\$37.63/barrel**. The June contract, which did not face immediate delivery, traded at +\$20 — a \$57 spread between adjacent contracts. This was the cost of carry inverted: instead of futures being above spot (normal carry), the spot was catastrophically below futures because the *immediate physical reality* of no storage was driving the expiring contract to absurd extremes. The event illustrated that commodity price theory is backed by physical reality, and when physical reality becomes extreme, so does price.

**2022: European natural gas surges 10× due to Russia-Ukraine war.** In 2021, European natural gas at the TTF hub (Netherlands) averaged around €25 per MWh. Russia had historically supplied roughly 40% of Europe's gas via pipelines, and European storage was optimistically lean. When Russia invaded Ukraine in February 2022 and gas flows were weaponized, TTF gas prices spiked to over **€340/MWh** by August 2022 — a 13× increase. The futures curve went into extreme backwardation: spot winter contracts were priced for catastrophe, while summer 2023 contracts reflected a (correct, as it turned out) assumption that the emergency would pass as alternative supplies were secured. Long futures holders during the backwardation period earned enormous roll yields. The entire episode was a stark illustration of what extreme supply disruption does to cost-of-carry dynamics: when you cannot get physical supply today, the convenience yield of having it in hand becomes enormous.

**August 2020: Gold reaches \$2,075/oz.** In the summer of 2020, the Federal Reserve had cut rates to essentially zero and was running QE at unprecedented scale. 10-year TIPS yields fell to approximately **-1.08%** — the lowest level ever recorded. With government bonds paying a guaranteed negative real return, the opportunity cost of holding gold (which earns 0% nominal and is a real asset) was *negative*. Every day you held a Treasury bond, you were guaranteed to lose purchasing power. Gold, which at least maintains purchasing power over time, became the dominant alternative for institutional money. The August 7, 2020 all-time high of \$2,075.47/oz corresponded almost exactly to the trough in real yields. When the Fed began telegraphing tapering and eventual rate hikes, TIPS yields rose, the opportunity cost of gold rose, and gold fell from its peak — eventually hitting \$1,618 in September 2022 as the Fed aggressively hiked. This is the real-rate relationship in action: gold is not a safe haven from all risks, but it is a near-perfect inverse of the real yield environment.

**Vietnam SJC premium 2023–2024.** In 2023, the State Bank of Vietnam's tight control of gold imports combined with surging domestic demand pushed SJC prices to extraordinary premiums. By early 2024, SJC gold bars were trading at 82–83 million VND/tael against an international parity of approximately 65–68 million VND/tael — a premium exceeding **20%**. This premium made Vietnamese gold one of the most expensive in the world on a parity-adjusted basis. The premium reflected:

1. **Import restrictions**: the SBV had issued no meaningful gold import licenses since 2012, and the domestic supply of SJC bars was nearly fixed.
2. **Domestic demand shock**: Vietnamese households rushed to gold as a safe haven amid currency volatility, property market stress, and bank sector scandals.
3. **No arbitrage mechanism**: even though international gold was 20% cheaper, private citizens and companies had no legal way to import the cheaper international gold and sell it domestically at SJC prices.

The SBV eventually intervened in mid-2024, conducting SJC gold auctions directly to increase domestic supply and narrow the premium. This intervention required using the SBV's own foreign reserves to buy international gold for domestic release — a reminder that maintaining an artificial market structure has real costs. The episode demonstrates that commodity pricing in Vietnam is not purely driven by global cost-of-carry theory, but by institutional and regulatory constraints that isolate the domestic market from international arbitrage.

---

## Further Reading & Cross-Links

The commodity valuation framework sits at the intersection of several deeper topics covered elsewhere in this series and across the site:

**Within this series:**
- [Time Value of Money: The Engine Behind Every Valuation Model](/blog/trading/asset-valuation/time-value-of-money-engine-every-valuation-model) — The continuous compounding mathematics in F = S × e^(r+u-y)T is the same compounding framework used across all valuation. Review this post to build intuition for why e^rT appears in the cost-of-carry formula.
- [Risk and Required Return: CAPM, Beta, and the Cost of Capital](/blog/trading/asset-valuation/risk-required-return-capm-beta-cost-capital) — When valuing commodity companies (miners, oil producers) rather than the commodity itself, you need an appropriate discount rate. CAPM-based approaches apply, with beta estimated from equity returns.
- [Real Options Valuation: How to Price Flexibility and Strategic Investments](/blog/trading/asset-valuation/real-options-valuation-flexibility-strategic-investments) — Junior miners and exploration companies derive most of their value from options on undeveloped deposits — the right to mine if prices rise. Real options valuation captures this in ways that standard DCF cannot.

**Cross-series:**
- [Gold: Price Drivers and Analysis](/blog/trading/gold/gold-price-drivers-analysis) — For a deeper dive into the macroeconomic, geopolitical, and structural drivers of the gold price itself (beyond the valuation mechanics covered here), this post in the gold series provides comprehensive empirical analysis.

**For further study:**
- John Hull's *Options, Futures, and Other Derivatives* (Chapters 3 and 5) — the canonical textbook derivation of cost-of-carry, futures pricing, and hedging mechanics. Hull's treatment is rigorous and complete; the framework here is faithful to his approach.
- Geman (2005), *Commodities and Commodity Derivatives* — the most comprehensive academic treatment of commodity-specific pricing, including seasonality, mean-reversion in spot prices, and the behavior of the convenience yield over time.
- CFTC Commitments of Traders reports (free, weekly) — show the positioning of commercial hedgers (producers/consumers) vs speculative traders in every major commodity futures market. The composition of long vs short commercial positions is a real-time read on hedging demand and market structure.

---

*Commodity valuation is grounding: no matter how sophisticated the model, the fundamental anchor is the physical reality of supply and demand, the actual cost of renting a storage tank, and the genuine operational value of having the physical commodity in hand. The formulas are simple — F = S × e^(r+u-y)T fits on a business card — but the judgment required to estimate S, u, and y correctly in any given market environment is the entire craft.*

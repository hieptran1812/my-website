---
title: "FX and Currencies: The Relative-Value Layer"
date: "2026-06-16"
publishDate: "2026-06-16"
description: "A currency is never worth something on its own — it is only ever worth another currency. This is how exchange rates work, what really moves them, why the dollar sits at the center of everything, and how currency quietly rides inside every foreign asset you own."
tags: ["asset-allocation", "cross-asset", "fx", "currencies", "carry-trade", "us-dollar", "dxy", "interest-rate-differentials", "hedging", "portfolio-construction"]
category: "trading"
subcategory: "Cross-Asset"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A currency has no intrinsic value and no long-run "up." An exchange rate is just the *relative price of two monies*, so FX is always a pair and always zero-sum: for every currency that rises, another falls by exactly as much. Your return from holding a currency is the interest-rate gap you earn, plus or minus how the spot price moves — nothing else.
>
> - **There is no "buy and hold" in FX.** Stocks claim profits, bonds pay coupons, but a euro just sits there earning the euro interest rate. The whole game is the *differential* between two rates and the move in the price between them.
> - **The dollar is the hinge of the entire cross-asset system.** It is the world's reserve and funding currency, and it is *negatively* correlated with almost everything risky — about **−0.30 with stocks, −0.35 with commodities, −0.40 with gold**. A strong dollar tightens financial conditions everywhere.
> - **Carry — borrow a low-yield currency, hold a high-yield one — goes up the stairs and down the elevator.** It earned a ~4.5% rate gap holding dollars against yen in 2022, then those trades can unwind in days when fear spikes.
> - **The one number to remember:** in a foreign stock that returns +10% locally, a 10% move in the currency can wipe the entire gain to **0%**. Currency is not a side issue in international assets — it can be the whole return.

In 2022, the most important price in global finance was not a stock index or an oil quote. It was the dollar. Over that single year the US Dollar Index — a measure of the dollar against a basket of major currencies — climbed from about 96 to an intraday peak near 114.8 in late September, its strongest level in two decades. Nothing about America's economy justified that on its own. What happened was simpler and more violent: the Federal Reserve raised interest rates faster than it had in 40 years, while Japan's central bank kept its rate pinned near zero and Europe lagged behind. Money does what money always does — it chased the higher rate. The dollar rose because *holding* dollars suddenly paid you, and holding yen did not.

That one move rippled through everything. A strong dollar made oil, copper, and wheat — all priced in dollars — more expensive for the rest of the world, so commodities wobbled. It crushed the currencies of emerging-market countries that had borrowed in dollars, because their debts got heavier in their own money. It pulled the yen down from 115 to 131 per dollar, and then well beyond. The Japanese government spent tens of billions trying to slow the slide. A single rate gap, expressed as a currency, reorganized the entire risk landscape.

This is the deep-dive on currencies in the *Cross-Asset Playbook* series — the strangest asset class on [the map of what you can own](/blog/trading/cross-asset/the-map-of-asset-classes-what-you-can-own). Stocks, bonds, gold, and commodities all have *something* you can point to: a profit stream, a coupon, a lump of metal, a barrel of oil. A currency has none of that. It is the one asset that is purely *relative* — only ever worth another currency, never worth anything by itself. The diagram above is the mental model we will build the whole post around: the dollar, uniquely, tends to strengthen at *both* ends of the global cycle — in a booming America that pulls in capital, and in a terrified world that runs for safety — and weakens only in the calm middle. That shape is called the dollar smile, and once you see it, the dollar stops being mysterious and becomes the single most useful dial in cross-asset investing.

![The dollar smile: the dollar rises in a global boom and again in a global panic and falls in the calm middle](/imgs/blogs/fx-currencies-the-relative-value-layer-1.png)

## Foundations: what an exchange rate actually is

Let us start from absolute zero, because currencies break the usual way of thinking about value, and almost every mistake people make about FX comes from skipping this step.

### A currency is only ever worth another currency

Pick up a stock and you can ask "what is it worth?" and get a real answer: the present value of the profits the company will earn. Pick up a bond and you can ask the same thing: the value of the interest and principal it will pay. Pick up a \$100 bill and ask "what is it worth?" and the question has no answer in isolation. Worth *what*? A hundred dollars is worth a hundred dollars — that is a tautology, not a price. The only way to give the dollar a price is to quote it *against something else*: \$100 is worth about €92, or about ¥15,700, or about 0.04 ounces of gold.

This is the first and most important fact about foreign exchange, usually shortened to *FX*: **a currency has no price of its own — it only has a price relative to another currency.** That is why every FX quote is a *pair*. You never trade "the euro." You trade EUR/USD — euros against dollars. You never trade "the yen." You trade USD/JPY — dollars against yen. The thing being priced is the *exchange rate*: how many units of one currency it takes to buy one unit of another.

An *exchange rate*, then, is the relative price of two monies. EUR/USD at 1.10 means one euro costs 1.10 US dollars. USD/JPY at 150 means one US dollar costs 150 Japanese yen. Read the pair left to right: the *first* currency is the one you are pricing (the *base*), and the *second* is the one you are pricing it in (the *quote*). When EUR/USD goes from 1.10 to 1.20, the euro got *more* expensive in dollars — the euro strengthened and the dollar weakened. When USD/JPY goes from 130 to 150, the dollar got *more* expensive in yen — the dollar strengthened and the yen weakened. (That last one trips up beginners constantly: a *rising* USD/JPY means a *falling*, weaker yen, because you need more and more yen to buy the same dollar.)

### FX is zero-sum, and that changes everything

Here is the consequence that makes currencies unlike any other asset class. Because an exchange rate is one currency *divided by* another, **every currency that rises does so at the exact expense of another currency that falls.** There is no such thing as "all currencies going up." If the dollar strengthens against the euro, the euro by definition weakens against the dollar by the same amount. FX is *zero-sum*: across the system, the gains and losses cancel to exactly zero, before you even count trading costs.

Compare that to stocks. The world's companies, taken together, genuinely create wealth over time — profits grow, the pie expands, and a diversified stock investor can earn a real long-run return of roughly 5–6% a year for a century without anyone else having to lose. The stock market has a *long-run up*. The currency market does not. There is no "world currency index" that rises over decades the way the world stock market does, because a currency is a ratio, and ratios do not trend to infinity. The dollar in 2025 closed the year near 97 on the index — almost exactly where it sat in 2004. Twenty years, round trip, net nothing. This is structural, not bad luck.

So if currencies don't go up over time, why does anyone hold them as an investment at all? Because of the *other* half of the return.

### Your FX return: the rate gap, plus or minus the spot move

When you hold a foreign currency, two things happen to your money.

First, the currency earns *its* interest rate. Dollars parked in a money-market account earn the US short-term rate; euros earn the euro rate; yen earn the (near-zero) yen rate. This is the *carry* — the yield you collect just for holding the currency, set by that country's central bank. (We have a whole post on why [interest rates are the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) of all asset prices; here we just need the rate itself.)

Second, the exchange rate *moves*. The currency you hold can get more or less valuable against your home currency — the *spot move*, where "spot" just means the price for immediate delivery, today's exchange rate.

Put them together and you get the one equation that defines an FX return:

$$\text{FX return} \approx \underbrace{(r_{\text{foreign}} - r_{\text{home}})}_{\text{the carry / rate gap}} \; + \; \underbrace{\%\Delta \text{ spot}}_{\text{the price move}}$$

where $r_{\text{foreign}}$ is the interest rate you earn on the currency you bought, $r_{\text{home}}$ is the rate you gave up at home, and $\%\Delta \text{ spot}$ is the percentage change in the exchange rate. That is the *whole* return. There is no dividend, no earnings growth, no coupon beyond the rate itself. A currency position is a bet on the rate gap *and* the spot move, nothing more.

#### Worked example: what you actually earn holding a foreign currency

You are a US investor with \$100,000. The US short rate is 1% and the New Zealand short rate is 5% — New Zealand has long been a classic high-rate currency. You convert your dollars to New Zealand dollars and park them for a year earning 5%.

- **The carry.** You earn 5% in New Zealand dollars instead of 1% at home, a rate gap of \$5,000 − \$1,000 = a **+4% advantage**, worth about **\$4,000** on your \$100,000.
- **Case A — the spot doesn't move.** You convert back at the same exchange rate. You keep the full +4% carry: about **\$4,000** profit. Lovely.
- **Case B — the New Zealand dollar falls 3%.** Your currency lost 3% against the US dollar over the year. Net: +4% carry − 3% spot = **+1%**, about **\$1,000**. The spot move ate most of your carry.
- **Case C — the New Zealand dollar falls 8%.** Net: +4% − 8% = **−4%**, a **\$4,000 loss**. You earned the highest interest rate in the developed world and *still* lost money, because the currency fell more than the rate gap you collected.

The one-sentence intuition: in FX, the interest you earn is the easy, visible part — the spot move is the part that actually decides whether you made money, and it is the part nobody can reliably predict.

### The plumbing: the biggest market on earth, and how it trades

A few facts to ground the scale. The foreign-exchange market is, by a wide margin, the largest financial market in the world: it turns over roughly **\$7.5 trillion every single day** (Bank for International Settlements, 2022 triennial survey). For comparison, the entire US stock market trades a few hundred billion dollars a day. FX is an order of magnitude bigger, because every cross-border trade — every import, every foreign bond purchase, every tourist exchanging cash — passes through it.

It trades in three main forms, and the names matter because they recur throughout the post:

| Instrument | What it is | What it's for |
|---|---|---|
| **Spot** | Exchange at today's rate, settled in ~2 days | Pay for goods, convert a portfolio now |
| **Forward** | Lock in an exchange rate *today* for a swap on a *future* date | Hedge a known future foreign payment or asset value |
| **FX swap** | Spot leg now + a forward leg to reverse it later | Borrow one currency against another short-term |

The *forward* is the workhorse of currency hedging, and its price is not a forecast — it is pinned by the interest-rate gap. If euros earn 1% less than dollars, the one-year EUR/USD forward sits about 1% *higher* than spot, because otherwise you could earn the rate gap risk-free. (This iron link is called *covered interest parity*, and it is one of the few things in FX that is close to a law.) The practical point: when you hedge a foreign currency, the cost or benefit of the hedge *is* the rate differential. Hedging a high-rate currency back to dollars costs you the gap; hedging a low-rate currency *pays* you the gap.

### Reading a quote, and the cost of trading one

Two practical wrinkles round out the foundations. First, FX prices are quoted to absurd precision because the numbers are huge. A quote like EUR/USD 1.0875 is read to the fourth decimal place, and that last digit has a name: a *pip* (one ten-thousandth of the rate, 0.0001). When a trader says "the euro moved 50 pips," they mean it went from, say, 1.0875 to 1.0925. On a large position those tiny digits add up — \$1 million of EUR/USD moving 50 pips is a \$5,000 swing.

Second, like any market, FX has a *bid-ask spread* — the gap between the price you can sell at (the *bid*) and the price you can buy at (the *ask*). In the most-traded pairs (EUR/USD, USD/JPY) that spread is razor-thin, often under a pip, because the market is so deep and liquid; trading them is nearly frictionless. In thinly traded *emerging-market* currencies the spread can be ten or a hundred times wider, and that cost is part of why carry trades in exotic currencies are riskier than the headline rate gap suggests — some of your yield goes to the dealer every time you get in or out. The depth of a currency's market is itself a feature: it is exactly *because* the dollar's market is the deepest on earth that it serves as the world's refuge in a crisis. You can always sell dollars; you cannot always sell the currency of a small economy when everyone wants out at once.

#### Worked example: the bid-ask cost of a round trip

You convert \$100,000 into euros and back again in a day, doing nothing in between. The dealer quotes EUR/USD at 1.0874 bid / 1.0876 ask — a 2-pip spread.

- **Buying euros.** You pay the *ask*, 1.0876 dollars per euro, so \$100,000 buys €91,946.
- **Selling them back.** You sell at the *bid*, 1.0874, so €91,946 fetches back **\$99,982**.
- **The cost.** You lost about **\$18** to the spread on a \$100,000 round trip — roughly 0.018%. Trivial in EUR/USD. Now imagine the same trade in a currency quoting a 50-pip spread: the same round trip costs about **\$460**, twenty-five times more.

The intuition: in the major pairs FX is one of the cheapest markets to trade in the world, but in thin currencies the spread quietly taxes every move — and that hidden tax is part of carry's true risk.

## What actually drives an exchange rate

If a currency has no cash flow to anchor it, what sets the price? Four forces, working on different timescales. The diagram below is the hub we will unpack — the rate gap, growth and trade, capital flows, and the slow long-run anchor all pushing on one number.

![What drives an exchange rate: rate differential growth and trade capital flows and the slow PPP anchor](/imgs/blogs/fx-currencies-the-relative-value-layer-3.png)

### Driver 1: interest-rate differentials (the dominant short-run force)

In the short run — months to a couple of years — nothing moves currencies more reliably than the *gap* between two countries' interest rates. Money is mobile and lazy: it flows toward wherever it earns the most for the least risk. When a central bank raises rates, its currency typically strengthens, because global capital rotates *into* that currency to capture the higher yield, and you have to buy the currency to hold it.

This is the engine behind the carry trade, which gets its own section below. For now, hold the headline: **a widening rate gap pulls capital toward the higher-yielding currency and tends to strengthen it.** The dollar's 2022 surge was, at its core, the Fed opening a rate gap against everyone else. The yen's long collapse was the *opposite* — the Bank of Japan refusing to raise rates while everyone else did.

There is an important subtlety here, and getting it right separates people who understand FX from people who guess. What matters is not the *level* of rates but *expected changes* and the *gap*. A currency whose central bank is widely expected to hike *faster than the market already assumes* tends to strengthen; a currency whose hikes are already "priced in" may not move at all when those hikes arrive. Markets trade the *surprise*, not the news everyone saw coming. This is why a currency can fall on a rate *hike* (if the hike was smaller than expected) and rise on a *hold* (if a cut was feared). The rate differential is the gravity; the *change in expectations* about that differential is what produces the day-to-day moves. When you read that "the dollar rallied on hawkish Fed comments," the mechanism is precisely this: the market revised *upward* its expectation of the future US rate gap, and bought dollars to get ahead of it.

### Driver 2: growth and the terms of trade

Currencies also track the relative health of economies. A country growing faster, with rising productivity and an attractive place to invest, draws in foreign capital — for factories, for stocks, for bonds — and that demand for its assets means demand for its currency. Faster relative growth tends to support a currency, slower relative growth tends to drag on it.

A close cousin is the *terms of trade* — the price of what a country *sells* versus what it *buys*. A nation that exports commodities (Australia's iron ore, Canada's oil, Brazil's soy) sees its currency rise when those commodity prices rise, because the world has to buy more of that currency to pay for the exports. The Australian dollar, the Canadian dollar, and the Norwegian krone are "commodity currencies" for exactly this reason — they go up when their key export goes up. This is one of the threads that ties FX to the rest of the cross-asset world: a commodity boom is, partly, a commodity-currency boom.

### Driver 3: capital flows and risk sentiment

The third force is the one that produces the violent, short-term moves: where global capital wants to *be* depends on how scared it is. In calm, optimistic markets ("risk-on"), money fans out across the world hunting for return — into emerging markets, into high-yield currencies, into anything that pays more. In frightened markets ("risk-off"), that money sprints home to safety, and "safety" overwhelmingly means the US dollar, the Swiss franc, and the Japanese yen — the classic *safe-haven* currencies.

This is why the dollar can rise in a global *panic* even when nothing good is happening in America. When fear spikes, investors everywhere sell risky assets and buy US Treasuries — the deepest, safest market on earth — and to buy Treasuries you must first buy dollars. The demand for dollars in a crisis has almost nothing to do with US fundamentals and everything to do with the dollar being the world's refuge. We cover this rotation in depth in the macro post on [risk-on / risk-off and how money rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates); the FX takeaway is that the dollar is the asset the whole world flees *into*.

### Driver 4: PPP — the slow long-run anchor

The fourth force barely matters month to month but dominates over a decade: *purchasing power parity*, or PPP. The idea is intuitive. If a basket of goods costs \$100 in the US and €100 in Europe, then "fair value" for EUR/USD is about 1.00 — one euro should buy roughly what one dollar buys. If a country runs persistently higher inflation, its currency should slowly weaken to keep purchasing power in line, because its money buys less and less at home.

The honest caveat: PPP is almost useless for predicting next year, and currencies can stay 20–30% away from PPP "fair value" for years on end. The famous shorthand is *The Economist*'s "Big Mac Index" — comparing the price of a Big Mac across countries to spot rich and cheap currencies. It is a real signal, but a slow one. Think of PPP as gravity: it doesn't tell you where the ball is right now, but it tells you which way it is eventually being pulled. Over five-plus years, currencies that drifted far from PPP tend to drift back.

### The dollar smile, restated

Now we can name the cover figure precisely. Stack these drivers together and the dollar — uniquely among currencies — gets pushed *up* at both ends of the global cycle:

- **Left side of the smile (global panic):** fear spikes, the world flees into Treasuries, and that flight *requires buying dollars*. The dollar rises in the storm.
- **Bottom of the smile (calm growth):** in a steady, risk-on world, money leaves the safe dollar for higher returns abroad — emerging markets, carry currencies. The dollar drifts *down* in the calm.
- **Right side of the smile (US boom):** when America grows faster than the rest and the Fed hikes, high US yields pull capital *in*. The dollar rises again, this time on strength rather than fear.

Only in the soft middle — a placid, broadly risk-on world where the US is not notably out-growing everyone — does the dollar weaken. That is the smile, and it is the single most useful regime map in cross-asset investing, because it tells you that the dollar is strong in *exactly* the two regimes (boom and bust) where you most need to know what it's doing.

## The dollar's special role: the hinge of the whole system

Every currency obeys the four drivers above. The dollar gets its own section because it is not just *a* currency — it is the operating system the others run on.

### Reserve currency, funding currency, invoice currency

The US dollar is the world's *reserve currency*: the money central banks hold as their rainy-day savings. As of 2024, the dollar was still about **57.8%** of all disclosed global FX reserves (IMF COFER) — down from **71.5%** in 2001 and **63.8%** in 2008, a slow erosion, but still more than every other currency *combined*. The euro is around 20%, the yen and pound a few percent each, the Chinese renminbi barely 2%. The dollar's dominance has faded at the pace of a glacier, not a landslide.

It is also the world's *funding currency* and *invoice currency*. A huge share of global trade is priced in dollars even when neither party is American — oil, most commodities, much of Asia's exports. A huge share of the world's debt is *borrowed* in dollars: companies and governments outside the US owe trillions in dollar-denominated bonds and loans. That second fact is the load-bearing one for cross-asset investors, because it means the dollar's price sets the cost of the world's debts.

Being the reserve currency confers what a French finance minister once called an *exorbitant privilege*: because the world *needs* dollars to trade and to hold as reserves, the US can borrow more cheaply and run larger deficits than any other country could get away with. The world is effectively obliged to lend to America by buying its Treasuries. The flip side — known as the *Triffin dilemma* — is that to supply the world with the dollars it needs, the US must run persistent trade deficits, which over the very long run is a slow weakening pressure on the currency. These two forces, privilege and dilemma, are why the dollar can dominate global finance *and* drift sideways for decades at once: it is structurally in demand as the world's money, yet structurally supplied in ever-greater quantity. Neither force trends the exchange rate; together they keep it ranging.

### A strong dollar tightens financial conditions everywhere

Here is the mechanism that makes the dollar the cross-asset hinge. Suppose a company in Brazil or Turkey or Indonesia borrowed \$1 billion in dollars (because dollar borrowing is cheap and deep). Its revenues are in its local currency. When the dollar strengthens 20%, that \$1 billion debt just got 20% *heavier* in local-currency terms — the company has to find 20% more of its own money to make the same payment. Multiply that across thousands of borrowers and you get a global *tightening*: a rising dollar squeezes everyone who owes dollars, drains liquidity, and forces selling. A strong dollar is a headwind for the entire risky world.

This is why traders watch the dollar as a *financial-conditions* gauge, not just an exchange rate. When the dollar rips higher, global financial conditions tighten whether or not any central bank intended it. The full machinery — the eurodollar system, why the dollar rules markets, and how the DXY works — is laid out in the macro post on [the dollar system](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy); here the point is the *consequence*: the dollar is the master valve on global risk appetite.

### The DXY, and the dollar's missing trend

The **DXY** — the US Dollar Index — measures the dollar against a basket of six developed-market currencies (heavily weighted to the euro). It is the single most-watched number in FX. The chart below is the DXY's path over a decade, and it makes the "no long-run up" point visceral.

![US Dollar Index yearly close 2014 to 2025 ranging between 90 and 109 with the 2022 peak marked](/imgs/blogs/fx-currencies-the-relative-value-layer-2.png)

Look at what it does *not* do: it does not trend. It ranges. It bottomed near **89.9** at the end of 2020 — zero US rates, a recovering risk-on world, the bottom of the smile. It surged to an intraday **~114.8** in September 2022 — the Fed hiking into a frightened world, the right side *and* left side of the smile firing at once. By the end of 2025 it had fallen back to **97.0**, roughly where it started the decade. Across eleven years the dollar went up, down, up, and back, netting close to nothing. A stock index over the same period roughly tripled. The dollar is a *range*, not a *trend* — which is exactly what you'd expect from a zero-sum relative price.

#### Worked example: reading the dollar as a financial-conditions dial

You manage a portfolio with emerging-market stocks, gold, and commodities, and you want one number that tells you whether the wind is at your back. Watch the DXY.

- **The DXY falls from 105 to 95** (about −9.5%). History says a weakening dollar loosens global conditions: dollar debts get lighter, capital flows out to higher-return markets. With USD correlated about **−0.45** with EM equities, a 9.5% dollar drop has tended to coincide with EM equities up *strongly* over the same window. The wind is at your back.
- **The DXY rises from 95 to 105** (about +10.5%). Now the same correlation works against you: dollar debts get heavier, money runs home, and your EM stocks, gold, and commodities all face a headwind at once. With gold at about **−0.40** to the dollar, a +10% dollar move has often coincided with gold down several percent.

The intuition: you don't need to forecast every asset separately — the dollar's direction is a single dial that turns most of your risk book the same way at once.

## The carry trade: up the stairs, down the elevator

The most important *strategy* in currencies — and the one that produces FX's signature blow-ups — is the carry trade. It deserves its own deep section because it is how most currency *return* is actually earned, and how it is occasionally, spectacularly, lost.

### The mechanics

The carry trade is simple to state: **borrow a currency with a low interest rate, and use the proceeds to hold a currency with a high interest rate. Pocket the difference.** The textbook version for the last decade has been borrowing Japanese yen (interest rate pinned near 0%) to hold US dollars (rate around 4–5% in 2023–24), or to hold even-higher-yielding currencies like the Mexican peso or Brazilian real.

The figure below is the shape of the trade — and the trap inside it.

![The carry trade borrow low yield hold high yield earn the gap in calm then unwind hard in a shock](/imgs/blogs/fx-currencies-the-relative-value-layer-4.png)

In calm markets, the carry trade is beautiful. You earn the rate gap every single day, like rent. And often the spot rate *drifts in your favor* too — when lots of investors are doing the same trade, their collective buying of the high-yield currency pushes it up, so you make the carry *and* a capital gain. Returns are steady, positive, and boring. This is the "up the stairs" part: slow, reliable, one step at a time.

#### Worked example: the 2022 yen carry, in dollars

You borrow ¥15,000,000 of Japanese yen at roughly **0%** (Japan's rate). At USD/JPY of 130, that converts to about **\$115,000**. You park it in US money markets earning roughly **4.5%** (the US rate in late 2022).

- **The carry.** Over one year you earn 4.5% on \$115,000 = about **\$5,175** in interest, and you owe ~0% on the yen loan. Pure rate gap: **+\$5,175**.
- **The spot bonus.** Over 2022 the dollar *rose* against the yen — USD/JPY went from about 115 toward 131, roughly **+20%** in the dollar's favor over the move's full extent. Your dollars are now worth ~20% more yen, so when you eventually buy back the yen to repay the loan, you need far fewer dollars. On the \$115,000 that's another ~**\$23,000** of currency gain.
- **The total.** Carry (+\$5,175) plus spot (~+\$23,000) ≈ **+\$28,000** on \$115,000 of borrowed money you put almost none of your own cash into. That is the dream version, and it is why the yen carry trade ballooned to enormous size.

The intuition: in a trending, calm regime the carry trade pays you twice — the rate gap *and* a spot tailwind — which is exactly what lures so much money in that the eventual exit becomes dangerous.

### Why it goes "down the elevator"

Now the other side. The carry trade's deadly feature is that its *risks are the mirror image of its rewards*. Everyone is in the same trade (short the low-yield currency, long the high-yield one), it is built on borrowed money (*leverage*), and it works only while markets stay calm. When a shock hits — a recession scare, a banking wobble, a sudden central-bank surprise — volatility spikes, and every carry trader tries to get out *at the same moment*.

Getting out means *buying back* the low-yield currency you borrowed (the yen) and *selling* the high-yield one. When thousands of leveraged traders do that simultaneously, the low-yield currency rockets up and the high-yield one collapses — in *days*, sometimes hours. The spot move that took two years to build in your favor can reverse in a week, wiping out months or years of patiently collected carry. The phrase traders use is exact: **the carry trade goes up the stairs and down the elevator.** Slow, steady gains; sudden, vertical losses.

The clearest recent example came in early August 2024. A surprise Bank of Japan rate hike plus a weak US jobs report sent the yen sharply stronger in days, USD/JPY tumbled from around 161 toward 142, and the global yen-carry unwind cratered stocks worldwide in a single session — Japan's market had its worst day since 1987. Nothing fundamental about the world economy changed that week. A crowded, leveraged FX trade simply ran for the exit at once. That is the elevator.

This fat-tailed shape — small positive returns most of the time, rare enormous losses — is the defining behavior of currencies as an investment, and it is why carry must be *sized* and *hedged* like the risk it is, never treated as free yield.

There is a deep reason the rate gap doesn't simply vanish through everyone arbitraging it away. Economists call it the *forward premium puzzle*: in theory, a high-yield currency should *weaken* over time by exactly enough to cancel its rate advantage (so that no free lunch exists). In practice it often *doesn't* — high-yield currencies have historically tended to *hold* or even *gain*, which is why carry has earned a real long-run return. But that return is not free; it is the *premium you collect for bearing the crash risk*. You are, in effect, being paid to stand in front of a rare but devastating move. The carry trade's average return is positive precisely *because* its worst-case is so bad that most investors won't hold it without compensation. Understand it that way and you'll never mistake the steady drip of carry for safety — you'll recognize it as an insurance premium you are quietly collecting, with the claim event still out there somewhere.

## How currencies behave: range-bound, regime-driven, fat-tailed

Pull the threads together and you get a clear behavioral profile, very different from stocks or bonds.

**Currencies range; they don't trend.** Because FX is a relative price with no long-run up, major-currency pairs spend most of their lives oscillating inside a band. The dollar index has lived between roughly 90 and 115 for over a decade. That makes FX feel mean-reverting — until a regime breaks.

**They are regime-driven.** A currency can sit quietly for years and then move 30% in eighteen months when the regime shifts — a central bank that diverges (the BoJ vs the Fed), a terms-of-trade shock, a sovereign crisis. The yen weakened from 108 to 157 per dollar between 2019 and 2024 — about a 45% depreciation — almost entirely because Japan held rates near zero while the rest of the world hiked. That is not noise; it is one regime (a giant rate divergence) playing out.

![USD JPY yearly close 2019 to 2025 weakening from 108 to 157 with intervention peaks marked](/imgs/blogs/fx-currencies-the-relative-value-layer-5.png)

**Their returns are fat-tailed.** The carry section already showed why: long calm stretches of small gains, punctuated by violent unwinds. Statistically, currency carry returns look like picking up small coins steadily and occasionally getting hit by a truck. The "average" return badly understates the risk, because the losses cluster in exactly the worst moments — when everything else you own is also falling.

**Volatility itself is regime-dependent.** In quiet years a major pair like EUR/USD might move 5–7% top to bottom. In a crisis it can move that in a week. And FX volatility tends to *spike when risk assets fall*, which means currency risk shows up precisely when you can least afford it.

**And they are famously hard to forecast.** A landmark finding in economics — going back to a 1983 study by Richard Meese and Kenneth Rogoff — is that over short horizons, no model of exchange rates reliably beats a simple "random walk," the assumption that tomorrow's rate is just today's rate plus noise. All the fundamentals we listed — rates, growth, trade, flows — *explain* moves after the fact and dominate over *years*, but they predict next month's rate barely better than a coin flip. For the cross-asset investor this is liberating rather than discouraging: it means the goal is not to *forecast* the dollar but to *respond* to it — to read which regime you're in (calm, panic, US-boom, US-lag) and take the matching stance, rather than to predict the next 3% wiggle that no one can. The dollar is a dial to read, not an oracle to interrogate.

#### Worked example: a "stable" currency that quietly halved your yield

You are a US investor who bought a Japanese government bond in 2019 yielding almost nothing, mostly to hold yen, when USD/JPY was about **108**. You held it through 2024, when USD/JPY hit about **157**.

- **The local return.** Japanese government bonds paid you close to **0%**. Call it flat in yen.
- **The currency.** The yen fell from 108 to 157 per dollar. To find your loss in dollars: your yen are now worth 108/157 ≈ **0.688** of what they were — a **−31% currency loss**.
- **The total.** 0% local return × (1 − 0.31) ≈ **−31%** in US dollar terms. You held one of the world's "safest" assets in one of the world's "safest" currencies and lost nearly a third of your money — purely on the exchange rate.

The intuition: in a foreign asset, the currency is not a footnote to the return — over years it can dwarf the return, for better or worse, which is the whole reason hedging exists.

## How the dollar correlates with everything else — the cross-asset hinge

This is the section that earns the currency its place in the *Cross-Asset Playbook*, because the dollar's value to an allocator is almost entirely about how it moves *against* the rest of the book.

![The dollar is negatively correlated with EM equities gold commodities US stocks REITs and high yield](/imgs/blogs/fx-currencies-the-relative-value-layer-6.png)

Over monthly returns from 2015 to 2024, the dollar (DXY) was *negatively* correlated with essentially every major risk asset:

| Asset | Correlation with USD | Why |
|---|---|---|
| **EM equities** | ≈ −0.45 | Dollar debt + capital flight when USD rises |
| **Gold** | ≈ −0.40 | Gold is priced in dollars and competes with cash |
| **Commodities** | ≈ −0.35 | Priced in dollars; strong USD makes them costlier abroad |
| **US stocks** | ≈ −0.30 | Strong USD tightens conditions, hurts exporters |
| **High yield** | ≈ −0.30 | Risk asset; sells off as conditions tighten |
| **REITs** | ≈ −0.25 | Rate-sensitive risk asset |

Every number is negative. That is the dollar's defining cross-asset property: **it goes up when almost everything else goes down.** A rising dollar is, mechanically, a tightening of global conditions, and a falling dollar is a loosening. The dollar is therefore the closest thing markets have to a single *risk-on / risk-off dial*. When it rallies hard, it is usually because the world is either booming (right side of the smile) or scared (left side) — and in the scared case, it rallies *while* your stocks, gold, EM, and commodities all fall together.

This makes long-dollar exposure a genuine *hedge*. Most diversifiers fail you in a crisis because everything risky falls together (correlations "go to one"). The dollar is one of the few things that reliably does the opposite: it tends to *rise* in the exact panic where your risk assets are cratering. That is why "long dollar" and "long Treasuries" are the two classic crisis hedges — and why a strong-dollar tailwind quietly flatters a US investor's foreign losses, while a weak-dollar tailwind quietly boosts foreign gains.

#### Worked example: how currency flips a foreign stock's return

A US investor holds **€100,000** of European stocks. Over the year, the stocks rise **+10%** (to €110,000) — a genuinely good year for European equities. But over the same year the euro falls from **1.10** to **1.00** dollars.

- **Start value in dollars.** €100,000 × 1.10 = **\$110,000**.
- **End value, unhedged.** The stock is now worth €110,000, but each euro is worth only \$1.00. €110,000 × 1.00 = **\$110,000**.
- **The result.** You started at \$110,000 and ended at \$110,000. A **+10% local gain became a flat 0%** in dollars. The 10% currency drop *exactly* cancelled the 10% stock gain.
- **The hedged version.** Had you *hedged* the euro — locked in 1.10 via a forward at the start — you would have kept the full **+10%**, about **\$11,000** of profit, because the currency move would have been neutralized.

The intuition: in a foreign asset, the currency is a second, invisible position riding alongside the one you chose — and over a year it can be the entire difference between a great return and nothing.

## Common misconceptions

**"A strong currency is good for a country / a strong dollar is good for my US stocks."** Mostly backwards for investors. A *strong* dollar makes US exporters less competitive, tightens global financial conditions, and is *negatively* correlated (≈ −0.30) with US stocks. Patriotism aside, for a multi-asset investor a *falling* dollar has usually been the tailwind, not a rising one.

**"FX has no return, so it doesn't matter for my portfolio."** It has no *systematic long-run* return (it's zero-sum), but it absolutely has a return *you experience* — the carry you earn and the spot moves you ride. And inside foreign assets, currency can be the *entire* return: the worked examples above showed a +10% European stock turning into 0%, and a 0% Japanese bond turning into −31%, purely on FX. "No long-run trend" is not the same as "doesn't matter."

**"Carry is free money — you just earn the higher rate."** This is how people blow up. Carry is *insurance you've sold*: small steady premiums in calm, catastrophic payouts in a crisis. The yen carry earned its rate gap for years and then unwound violently in August 2024. The rate gap is real, but it is *compensation for tail risk*, not a free lunch.

**"The dollar will collapse because of US debt / de-dollarization."** People have predicted the dollar's demise for fifty years. The reality is a glacial decline in reserve share — from 71.5% (2001) to 57.8% (2024) — over two *decades*, and it is still larger than every rival combined. The dollar's role rests on the depth and safety of US Treasuries, and no rival market is remotely as deep. Slow erosion, not collapse.

**"A rising USD/JPY means the yen is getting stronger."** No — it's the opposite, and it catches everyone. USD/JPY is *yen per dollar*, so a higher number means each dollar buys *more* yen, i.e. the yen is *weaker*. When USD/JPY went from 108 to 157, the yen *lost* about a third of its value. Always check which currency is the base.

**"I can forecast the dollar from the news, so I'll trade it."** The Meese-Rogoff result above is the polite version of why this fails: over short horizons, fundamentals predict exchange rates barely better than a random walk, and the moves that matter come from *surprises* relative to what was already priced in — which by definition you can't see coming. Even professional macro funds, with armies of economists, find FX one of the hardest things to time. The useful posture is humility: use the dollar as a *regime read* (is it broadly rising or falling, loosening or tightening conditions?), not as a precise forecast you bet the book on.

## How it shows up in real markets

**2022 — the dollar wrecking ball.** The Fed hiked from ~0% to ~4.5% in a single year while the BoJ stayed at zero. The dollar index surged to ~114.8 in September, the right *and* left sides of the smile firing together (US strength plus global fear). The fallout was global: the yen fell to 131, the pound briefly cratered toward parity, EM currencies buckled, and a strong dollar tightened conditions everywhere at once. The single rate gap, expressed as a currency, was the year's dominant cross-asset force.

**The yen, 2019–2024 — a regime, not noise.** The yen weakened from 108 to 157 per dollar — roughly 45% — almost entirely because Japan held rates near zero while everyone else hiked. Japan's Ministry of Finance intervened, spending tens of billions of dollars, as USD/JPY hit ~151.9 in October 2022 and ~161.9 in July 2024. Intervention can slow a slide, but it cannot beat the rate gap: as long as the differential persisted, the yen stayed weak. The lesson — fundamentals (the rate gap) beat intervention.

**August 2024 — the carry unwind.** A surprise BoJ hike plus a weak US jobs report sent the yen sharply stronger, USD/JPY fell from ~161 toward ~142 in days, and the unwinding of the crowded yen-carry trade cratered global equities — Japan's Nikkei had its worst day since 1987. No fundamental shift; just a leveraged FX trade hitting the exit at once. The cleanest illustration of "down the elevator" in a generation.

**2008 — the dollar in the storm.** In the depths of the global financial crisis, with the *US itself* at the epicenter, the dollar *rose*. Why? A global scramble for dollars to repay dollar debts and to hold Treasuries — the left side of the smile. The currency of the country *causing* the crisis strengthened because it was the world's refuge. This is the single best proof that the dollar's safe-haven role is about plumbing, not patriotism.

**Switzerland, 2015 — the floor that snapped.** The Swiss National Bank had capped the franc against the euro for years. In January 2015 it abruptly abandoned the cap; the franc instantly jumped ~30% against the euro, and several FX brokers went bankrupt as leveraged clients were wiped out in minutes. A reminder that currencies can gap *enormously* when a regime breaks, and that leverage in FX is uniquely unforgiving.

**The British pound, September 2022 — when fiscal policy fights the bond market.** Britain announced a package of large unfunded tax cuts, and the market's verdict was instant and brutal: the pound plunged toward parity with the dollar — its weakest ever — in a matter of days, and UK government bond yields spiked so violently that the Bank of England had to intervene to stop pension funds from collapsing. The mechanism was the FX one we built: investors lost confidence in the credibility of UK policy, capital fled (Driver 3), and the currency took the hit. Within weeks the policy was reversed and the pound recovered. The lesson — a currency is, among other things, a real-time confidence vote on a country's policy, and that vote can be merciless. It also shows the dollar's smile from the *other* side: when a rival currency cracks, the dollar is where that fleeing capital lands.

## When to own it: the FX and dollar playbook

This is the payoff. Currency exposure is not optional — you already hold it the moment you own any foreign asset — so the real questions are: when is the dollar a *return source*, when is it a *risk to hedge*, and how do you read it as a *regime signal*? The matrix below maps the four regimes to a stance.

![The FX and dollar playbook mapping each regime to a dollar stance carry call and hedging decision](/imgs/blogs/fx-currencies-the-relative-value-layer-7.png)

### FX as a return source — carry, sized for the tail

Carry is the only durable *return* in currencies, and it works best in **calm, high-differential regimes**: low volatility, a wide and stable rate gap, no looming central-bank surprise. In those windows, borrowing a 0% currency to hold a 4–5% one has paid handsomely. But size it as the tail-risk trade it is: keep positions small relative to the book, diversify across several currency pairs rather than one crowded trade, and *cut fast* when volatility spikes — the whole point of "down the elevator" is that waiting to exit is how you get hurt. Carry is a position you rent in calm and abandon at the first siren, not something you marry.

### FX as a risk to hedge — the foreign-asset decision

If you own international stocks or bonds, you hold currency exposure whether you wanted it or not. The decision is whether to *hedge* it (lock in today's exchange rate via forwards) or leave it open:

- **Hedge when the currency risk is large relative to the asset's return** — most obviously *foreign bonds*. A bond yields a few percent; a currency can move 10–30%. Unhedged foreign bonds are mostly a currency bet wearing a bond costume. Most professional global-bond funds hedge the currency for exactly this reason.
- **Often leave equities unhedged** — for *foreign stocks*, the currency exposure can be a *useful diversifier*, because foreign currencies often *fall* when the dollar rises in a US-led boom, partly offsetting. But know the cost: hedging a high-rate currency back to dollars *costs* you the rate gap (recall: the forward is pinned by the differential), while hedging a low-rate currency *pays* you.

The honest framing: there is no free hedge. Hedging removes the currency's risk *and* its potential return, and the carry of the hedge can be a meaningful drag or boost. The worked examples showed both faces — a hedge that would have saved a European-stock investor's +10%, and the −31% a yen-bond holder ate for *not* hedging.

A middle path many allocators take is a *partial* or *strategic* hedge — covering, say, half the currency exposure rather than all or none. This caps the worst-case currency loss without giving up all of the diversification benefit, and it spares you the painful regret of being fully hedged in a year the currency moved hard in your favor (or fully unhedged in a year it moved against you). Some go further and *vary* the hedge ratio with the regime: hedge more when the dollar looks set to strengthen (you don't want foreign currencies dragging on you), hedge less when the dollar looks set to weaken (you want the foreign-currency tailwind). That is a more active stance, and it depends on reading a regime the Meese-Rogoff result warns is hard to time — so most long-term investors keep it simple: hedge the bonds, partially hedge or leave the equities, and revisit only when the dollar's *trend* (not its daily wiggle) clearly turns.

### The dollar as a regime dial — the risk-on / risk-off signal

The dollar's highest-value use to a cross-asset investor is not as a trade at all — it is as a *dial*. Because it is negatively correlated with nearly everything risky, its direction is a single read on whether global conditions are loosening or tightening:

- **A falling dollar** loosens global conditions — a tailwind for EM, commodities, gold, and risk broadly. This is the regime to *lean into* risk and let foreign exposure run unhedged.
- **A rising dollar** tightens conditions — a headwind for the same assets, and a signal to trim risk, raise hedges, or hold dollars as ballast. In a genuine panic (left side of the smile), *long dollars and long Treasuries* are among the few hedges that pay off while everything else falls.

#### Worked example: long dollars as crisis insurance

You hold a \$1,000,000 portfolio that is 70% risk assets (stocks, EM, commodities) and you fear a risk-off shock. You add a **\$100,000** long-dollar position (say, via cash dollars against a basket) as a hedge.

- **The shock hits.** Risk assets fall **−20%**: your \$700,000 of risk loses about **−\$140,000**.
- **The dollar rises.** In the panic the DXY jumps **+8%** (left side of the smile). Your \$100,000 long-dollar hedge gains about **+\$8,000**, and because the dollar is ≈ −0.30 to −0.45 correlated with your risk book, that gain reliably shows up *in the same moment* the rest is falling.
- **The trade.** The hedge offsets only a slice of the loss (≈ \$8,000 of a \$140,000 hit), but it does so *exactly* when correlations everywhere else have gone to one — that timing, not the size, is the value. In calm times that same long-dollar position quietly costs you the rate gap; you are paying a small premium for crisis insurance.

The intuition: the dollar's job in a portfolio is to pay off in precisely the regime where your diversifiers fail — you hold it not to make money on average, but to cushion the moment everything else falls together.

### What invalidates the case

Every stance has a kill-switch. **For carry:** a volatility spike or a hawkish surprise from the funding-currency's central bank (the August 2024 BoJ hike) — exit immediately, don't average down. **For a long-dollar hedge:** a clear pivot to a global-recovery, weak-dollar regime (the bottom-to-left-of-smile transition), where holding dollars just bleeds carry. **For leaving foreign assets unhedged:** evidence that the currency risk now dwarfs the asset's expected return (low-yielding foreign bonds, or an unstable EM currency) — then hedge. The discipline is the same one that runs through this whole series: name the regime, take the matching stance, and write down in advance what would tell you the regime has changed.

## Further reading & cross-links

Currencies are the relative-value layer that sits *underneath* every other asset class — the exchange rate is the lens through which a US investor sees the entire foreign half of the world. To go deeper on the pieces this post leaned on:

- [The map of asset classes: what you can own](/blog/trading/cross-asset/the-map-of-asset-classes-what-you-can-own) — where currencies fit among stocks, bonds, gold, and commodities, and why FX is the odd one out.
- [Gold: money, insurance, or just a rock?](/blog/trading/cross-asset/gold-money-insurance-or-just-a-rock) — the other "no-cash-flow" asset, and its own ≈ −0.40 relationship with the dollar.
- [The dollar system: why USD rules markets and how the DXY works](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy) — the full machinery of the reserve and funding currency.
- [Interest rates: the price of money and the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — why the rate differential drives FX in the first place.
- [Risk-on / risk-off: how money rotates](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates) — the sentiment cycle that turns the dollar into a single risk dial.

This is educational, not investment advice. The mechanisms and history here are tools for understanding currency risk and the dollar's role — not a recommendation to put on any specific trade. The currency you can't see — the one riding inside your foreign assets — is usually the one that matters most, so the first job is simply to *know it's there*.

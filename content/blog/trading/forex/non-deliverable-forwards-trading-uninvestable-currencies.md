---
title: "Non-Deliverable Forwards: Trading Currencies You Cannot Deliver"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How the offshore market prices currencies you cannot freely move — the renminbi, won, rupee and dong — using cash-settled forwards in dollars, and what the onshore/offshore gap tells you about devaluation pressure."
tags: ["forex", "currencies", "ndf", "non-deliverable-forward", "capital-controls", "vietnam", "dong", "renminbi", "emerging-markets", "offshore-fx", "devaluation"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A non-deliverable forward (NDF) is a bet on a currency you are not allowed to deliver. Two parties agree a rate today; on the fixing date they compare it to an official reference rate and settle only the dollar difference. No restricted currency ever changes hands.
>
> - NDFs exist because **capital controls** block free delivery of currencies like the dong (VND), renminbi (CNY), won (KRW) and rupee (INR) — so the world built a parallel offshore market that settles in dollars.
> - The offshore NDF price can **diverge from the onshore rate**. When the offshore forward prices the currency cheaper than the onshore band allows, that gap is the market pricing in a **devaluation** the central bank is trying to prevent.
> - The whole trade is a **dollar trade**: you post dollars, you settle dollars, your profit and loss are in dollars — the restricted currency is just a number you read off a screen on one day.
> - The one number to remember: a USD/VND NDF spread can be **~40 pips** versus ~0.2 for EUR/USD — restriction is not free, and the offshore premium spiked to **3.5%** in the 2022 dollar squeeze.

In late October 2022 the dong was quietly having its worst year in a generation. Onshore, you would not have known it from the screen: the State Bank of Vietnam (SBV) held USD/VND inside a band, the official rate crawled up by a fraction of a percent a week, and to a tourist changing money in Hanoi nothing looked broken. But a few thousand kilometres away, on dealer screens in Singapore and London, a different price was flashing. The one-year offshore forward on the dong — a contract that can never actually deliver a single dong — was pricing the currency several percent weaker than the onshore market was allowed to go. The offshore market was, in effect, screaming a devaluation that the onshore market was legally forbidden from showing.

That gap is the subject of this post. It is one of the strangest and most useful instruments in all of currency trading: a forward contract on a currency you cannot buy, cannot sell, cannot wire across a border, and will never receive. You settle the entire thing in US dollars against an official "fixing." And precisely because the contract is offshore and unrestricted while the real currency is onshore and caged, the price of the contract becomes a pressure gauge — a way to read what global capital really thinks a controlled currency is worth, versus what its government insists it is worth.

This is the spine of the whole series in its purest form. An exchange rate is the relative price of two monies, and you never own a currency in isolation — you own a *bet on the gap*. With an NDF, the series' principle becomes almost literal: you genuinely never own the currency, not even for a moment. You own nothing but the difference between two numbers, paid in dollars. Let us build it from the ground up.

![NDF settlement mechanic pipeline agree rate fixing date cash settle the difference in USD](/imgs/blogs/non-deliverable-forwards-trading-uninvestable-currencies-1.png)

## Foundations: What a non-deliverable forward is

Start with the word "deliverable," because the entire instrument is defined by its negation. In an ordinary FX forward — the workhorse covered in [FX forwards and swaps](/blog/trading/forex/fx-forwards-and-swaps-the-real-workhorses-of-the-market) — two parties agree today to exchange two currencies at a set rate on a future date. A US importer who owes 100 million yen in three months can lock the rate now: on the delivery date, real dollars go one way and real yen come back the other. The contract *delivers*. Both currencies physically move between accounts.

A non-deliverable forward removes the delivery. Nothing physically moves except dollars. The two parties still agree a rate today, they still pick a future date, but on that date they do not exchange the two currencies. Instead they look up an agreed official "fixing" rate, compare it to the rate they locked in, and the loser wires the winner the difference — in US dollars. The restricted currency appears nowhere in the cash flows. It is a *reference number*, not a thing you hold.

So an NDF has four moving parts, and you need all four to read a quote:

- **The currency pair**, always written as the restricted currency against the dollar — USD/VND, USD/KRW, USD/INR, USD/CNY. The dollar is the settlement currency.
- **The notional**, denominated in dollars — say a one-million-dollar contract. This is the size of the bet, not an amount of dong.
- **The NDF rate** — the forward rate the two parties agree today. This is what you are locking in.
- **The fixing** — the specific, named, official reference rate that both sides will read on the settlement date to decide who pays whom.

#### Worked example: the anatomy of a \$1,000,000 dong NDF

Suppose a hedge fund and a dealer agree a one-month USD/VND NDF on a notional of \$1,000,000 at an NDF rate of **25,450 dong per dollar**. That single line of a term sheet says: "In one month, we will look up the official USD/VND fixing. If the dong is weaker than 25,450 — more dong per dollar — the party that is long dollars (short the dong) wins, and the other side pays the difference, in dollars, on this \$1,000,000 notional." No dong is bought today. No dong is owed in a month. The \$1,000,000 is never actually exchanged for dong either — it is purely the scaling factor that turns a percentage move into a dollar amount. The takeaway: an NDF is a side bet on where an official number lands, sized in dollars, and the restricted currency never touches your account.

### Why "non-deliverable": capital controls

The reason delivery is off the table is **capital controls** — legal restrictions a government places on moving its currency in and out of the country, and on who may convert it. A freely convertible currency like the dollar, euro, yen or pound can be bought, sold, wired and held by anyone, anywhere, for any reason. A restricted currency cannot. To get dong, you generally need to be inside Vietnam's financial system, with a licensed reason — an import to pay, a salary to receive, a verified investment. You cannot, as a London hedge fund, simply call a bank and ask for a billion dong to be delivered to your account in Singapore. There is no legal pipe for it.

That creates a problem for anyone with a real economic exposure to the dong but no access to it. A foreign manufacturer with a factory in Vietnam earns dong revenue and wants to hedge it. A bond investor holding dong-denominated debt wants to protect against depreciation. A multinational consolidating its accounts wants to lock a rate. All of them have a genuine need to trade the dong's value — and none of them can legally deliver it across a border. The NDF is the financial system's answer: keep the *exposure*, drop the *delivery*. You get the economic profit-and-loss of a dong forward without ever needing a dong.

### The fixing: the most important word in the contract

If an NDF settles against "the fixing," then the fixing is the single most consequential detail in the whole contract — get it wrong and you are settling against the wrong number. So it deserves its own treatment, because in a deliverable forward there is no fixing at all (you just exchange the currencies), and beginners coming from deliverable FX often skip right over it.

A fixing is a published, official, auditable reference rate, sampled at a stated time on a stated day from a stated source. The phrase "USD/VND fixing on the second business day before maturity, as published by the State Bank reference, at 11:00 local time" is not boilerplate — every clause is load-bearing. The *source* matters because different sources can print different numbers (an interbank average versus a central-bank reference versus a calculated benchmark). The *time* matters because a currency can move within a day, and you want a rate that is hard to manipulate. The *date* matters because settlement happens a couple of days after the fixing, to allow the dollar wire to clear.

Each NDF currency has its own conventional fixing, and a trader has to know them:

- **The dong (VND)** settles against a published interbank or central-bank reference rate on a named day.
- **The won (KRW)** settles against the Seoul market average rate (the MAR), a volume-weighted interbank average.
- **The rupee (INR)** settles against the Reserve Bank of India reference rate.
- **The renminbi (CNY)** NDF settles against the daily central-parity or a calculated benchmark.

Why so much care about a single number? Because the fixing is where a thin market is most vulnerable. If a fixing is calculated from a small window of illiquid trades, a large player can push the print and skew everyone's settlement. This is not hypothetical — regulators have fined banks for attempting to influence FX fixings in deliverable currencies, and the same incentive exists, magnified, in a thin NDF where the fixing decides millions in settlement. A good fixing is robust, transparent, and hard to move; a bad one is a single point of failure for the entire market. When you trade an NDF, you are trusting that one published number to be honest.

#### Worked example: how the fixing day decides everything

Two desks hold opposite sides of the same \$1,000,000 USD/VND NDF struck at 25,450. The fixing is scheduled for a Thursday. On Wednesday the dong is trading around 25,500 offshore — the long-dollar side is winning by ~50 points. But the fixing has not happened yet, so nobody has settled anything. On Thursday morning, before the fixing window, a large dollar-buying flow hits and the official print comes in at 25,750. Now settle:

```
Settlement = 1,000,000 × (25,750 − 25,450) / 25,750 = +$11,650
```

The long-dollar desk collects \$11,650 — but notice that the *entire* P&L was decided by where the fixing printed on that one morning, not by where the dong traded for the prior 29 days. Had the fixing instead caught a moment of dong strength at 25,400, the same desk would have *paid* `1,000,000 × (25,400 − 25,450) / 25,400 = −$1,969`. The takeaway: in an NDF the fixing is not a formality — it is the single observation that converts a month of price action into one dollar number, which is exactly why its source and timing are negotiated so carefully.

### Where NDFs came from

NDFs are not an academic invention; they grew out of necessity. As global capital pushed into emerging markets in the 1990s, investors and corporates accumulated real exposure to currencies — the Brazilian real, the Russian ruble, the Korean won, eventually the renminbi and the dong — that they could not freely deliver across borders. The deliverable-forward market simply could not serve them, because delivery was blocked. So dealers in the big offshore hubs improvised a cash-settled forward: keep the price exposure, settle the difference in dollars, never touch the restricted currency. The market grew up where the liquidity and the legal freedom were — Singapore and Hong Kong for the Asian currencies, New York and London for the Latin American ones.

The Asian financial crisis of 1997 was a brutal proving ground. As pegs broke across the region — the baht, the rupiah, the won all collapsed, a story told in the [Asian financial crisis](/blog/trading/finance/asian-financial-crisis-1997) post — the NDF markets for those currencies became the place where the offshore world repriced them, fast, while onshore markets froze or were defended. The crisis taught a generation of traders that the offshore NDF is where stress shows up first and most violently, precisely because it is unrestricted. Today the NDF market is mature, electronically traded, and partly cleared, but its DNA is unchanged: it exists to let the world trade currencies that their own governments will not let move freely.

### Onshore versus offshore: two markets for one currency

This is the conceptual fork that makes everything else click, so go slowly here. A restricted currency lives in two worlds at once.

**Onshore** is the regulated home market — inside Vietnam, inside China, inside India. Here the currency is real and deliverable, but access is gated. Banks trade it under the central bank's rules, within whatever band or reference-rate regime the authorities run. For the dong, the SBV sets a daily central reference rate and allows the market to trade within a band around it — a managed crawl, covered in detail in [USD/VND and the managed float](/blog/trading/forex/usd-vnd-and-the-managed-float-how-the-sbv-runs-the-dong). The onshore rate is what a Vietnamese importer actually pays. It is **deliverable but caged**.

**Offshore** is everywhere else — Singapore, Hong Kong, London, New York. Here the currency is not deliverable, because no real dong can leave Vietnam freely, but it is *unrestricted as a price*. Anyone with a dollar account can write an NDF. There is no central bank setting a band on the offshore forward. The offshore price floats to wherever global supply and demand for dong exposure puts it. It is **uncaged but undeliverable**.

![Onshore deliverable forward versus offshore cash settled NDF before and after comparison](/imgs/blogs/non-deliverable-forwards-trading-uninvestable-currencies-2.png)

The figure above lays the two side by side. On the left, the onshore deliverable world: you need a license, real dong moves, real dollars move, and you must have a permitted reason. On the right, the offshore NDF world: you sit outside the controls, no dong ever moves, only the net dollar difference settles, and any global account can play. Same currency, two completely different pipes — and, crucially, two prices that do not have to agree, because there is no free flow of capital welding them together. Hold that thought; the gap between those two prices is the entire payoff of this post.

## The NDF mechanic: how a contract settles step by step

Let us trace one NDF from agreement to settlement, because the mechanics are where people get confused, and the confusion is always about *which currency moves*. (The answer is always: only dollars.)

**Day zero — agreement.** Two parties agree the four parts above: pair (USD/VND), notional (\$1,000,000), NDF rate (25,450), and the fixing source and date. Crucially they name the fixing precisely. For the dong, a common reference is a published interbank or central-bank rate on a stated day; for the won it is the Seoul market average rate; for the rupee it is the Reserve Bank of India reference rate. The fixing is not "whatever the rate is" — it is a specific, auditable, published number, so neither side can argue about it later.

**Between day zero and the fixing date — nothing.** No payments. No margin of currency. (There may be collateral or variation margin between dealers, posted in dollars, but no dong-related cash flow.) The contract just sits there as an exposure on both books.

**The fixing date — read the number.** On the agreed day, both sides look up the official fixing. Say it prints at **25,800 dong per dollar** — the dong is weaker than the 25,450 they locked. The contract is now in the money for whichever side was betting on a weaker dong (long USD/VND), and out of the money for the side betting on a stronger dong.

**Settlement — pay the dollar difference.** Now comes the only cash flow that involves the currencies, and it is entirely in dollars. The settlement formula converts the rate difference into a dollar P&L. Because the pair is quoted as dong-per-dollar (the dollar is the base of the quote in this convention), the dollar amount is:

```
Settlement (USD) = Notional_USD × (Fixing − NDFrate) / Fixing
```

We divide by the fixing, not the NDF rate, because the gain or loss is realized at the rate prevailing on settlement day — that is the rate at which the dong difference would notionally be converted back to dollars. This is a standard and subtle point that trips up first-timers; we will do the arithmetic in the next worked example so it is concrete.

The figure that opens this post is exactly this chain: agree the rate, carry the notional in dollars, reach the fixing date, compare the fixing to the contract rate, compute the net dollar P&L with no dong changing hands, and cash-settle in dollars in an offshore account. Six steps, one cash flow, zero deliveries of the restricted currency.

#### Worked example: the dollar P&L when the fixing differs from the contract rate

Take the contract from before: \$1,000,000 notional, NDF rate 25,450 VND/USD, and the dong weakens so the fixing prints at **25,800**. The party that was long USD/VND (betting the dong would weaken) wins. Plugging into the formula:

```
Settlement = 1,000,000 × (25,800 − 25,450) / 25,800
           = 1,000,000 × 350 / 25,800
           = 1,000,000 × 0.013566
           = $13,566
```

So the long-dollar side receives **\$13,566**, wired from the loser, and the contract is over. No dong was ever bought, held, or delivered. The dong moved about 1.37%, and on a million-dollar notional that 1.37% is worth roughly \$13.6k. Note the asymmetry baked into the division-by-fixing convention: if instead the dong had *strengthened* to 25,100, the same 350-point move would settle to `1,000,000 × (25,100 − 25,450) / 25,100 = −$13,944` — the long-dollar side pays \$13,944. The numbers are close but not identical, because each is divided by a different fixing. The takeaway: an NDF turns a percentage move in a currency you cannot touch into a clean dollar wire, and the exact convention (divide by fixing) matters down to the last hundred dollars.

### Why settle in dollars and not the local currency

There is a second, subtler reason delivery is dropped rather than merely restricted: even where a foreign party *could* obtain a small amount of the currency, doing so for a hedge would be wildly impractical. To hedge a billion dong of revenue with a deliverable forward, you would need a billion dong to deliver — sitting in a Vietnamese bank account, subject to Vietnamese rules, repatriable only with paperwork. The NDF erases all of that operational friction. You hold dollars, you post dollars, you settle dollars, and the dong exists only as a number you read on one day. The instrument does not just route around the legal block — it routes around the entire operational nightmare of holding a caged currency.

The settlement-in-dollars design is not an accounting nicety; it is the whole point. If the contract settled in dong, you would need dong to settle it — and the entire reason the NDF exists is that you *cannot get dong*. Settling in dollars sidesteps the capital control completely. Neither party needs access to the restricted currency's banking system. A fund in the Cayman Islands and a dealer in Singapore can run dong risk against each other forever and never once interact with a Vietnamese bank. The dollar is the universal solvent of FX, on ~88% of all currency trades (per the BIS), so it is the natural settlement leg for a market whose defining feature is that the *other* currency is unreachable. This connects directly to the broader [dollar system](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy): the reason an offshore market for the dong can even exist is that the dollar provides a neutral, liquid, deliverable counter-currency that everyone already holds.

### Who trades NDFs, and how they clear

It helps to know who is on each side of an NDF, because the players explain the price. The natural *sellers of the local currency* (long-dollar) are foreign investors and corporates hedging downside risk to a restricted currency they are exposed to — a fund holding local bonds, a manufacturer with local revenue. The natural *buyers of the local currency* (short-dollar) include exporters and inbound investors who will receive the currency and want to lock a stronger rate, plus speculators taking the other side of the fear premium when they think it has overshot. And in the middle sit the dealers — the global banks that make two-way prices, warehouse the risk, and charge the wide spread we saw earlier for the privilege.

Mechanically, NDFs trade over the counter between these parties, increasingly on electronic platforms, and a growing share is now novated to a central counterparty for clearing — which reduces the risk that one side fails to pay the dollar settlement. That settlement risk is real but small relative to a deliverable trade: because only a net dollar difference changes hands, the amount at risk is the *P&L*, not the full notional, which is one quiet advantage of the cash-settled design. The instrument that grew out of a restriction turns out to carry less settlement exposure than the deliverable forward it replaces.

## The onshore/offshore gap: where the two prices come apart

Now the part traders actually care about. We have two prices for the same currency — onshore (caged, deliverable) and offshore (free, non-deliverable) — and nothing forces them to be equal.

In a freely convertible currency, arbitrage welds the forward to the spot via **covered interest parity**: if the forward drifted away from the rate implied by the two countries' interest rates, you could borrow in one currency, lend in the other, lock the forward, and pocket a riskless profit until the gap closed. That mechanism is the subject of the [FX forwards](/blog/trading/forex/fx-forwards-and-swaps-the-real-workhorses-of-the-market) post and it works beautifully for EUR/USD. But it relies on being able to *move money freely between the two currencies*. Capital controls break exactly that. You cannot borrow dong onshore and lend the proceeds offshore, because you cannot move the dong offshore. The arbitrage pipe is cut. And when the pipe is cut, the two prices are free to diverge.

So what determines the offshore NDF price if not clean arbitrage? Supply and demand for dong *exposure* among people who only have dollars. If global investors are nervous about the dong — capital outflows, a widening trade gap, fear the SBV will let the currency slide — they pile into the long-USD/VND side of the NDF (betting the dong weakens). That demand pushes the offshore forward to price the dong cheaper than the onshore band allows. The onshore rate, meanwhile, is pinned by the central bank, which is actively defending the band by selling reserves and managing the reference rate. The result is a wedge: **the offshore NDF sits weaker than the onshore rate, and the size of that wedge is the offshore premium.**

![What the offshore NDF gap signals graph offshore premium leading to devaluation pressure](/imgs/blogs/non-deliverable-forwards-trading-uninvestable-currencies-5.png)

Read the figure as a flow of cause and effect. Two forces push from the left: reserves draining (the central bank is spending ammunition to hold the line) and the rate gap versus the dollar (when US rates tower over local rates, holding the local currency is expensive, so the offshore forward must price it cheaper to compensate — a carry cost). Both forces feed two outcomes: the NDF prices the currency cheaper, while the onshore rate stays pinned by the band. Those two diverging prices open the offshore premium, and that premium *is* devaluation pressure, priced. The figure's claim is the thesis of the section: an offshore NDF premium above the onshore rate is the market pricing in a devaluation the authorities are resisting.

#### Worked example: an offshore short pricing devaluation

Suppose the onshore spot is 25,450 dong per dollar and the SBV's band caps it near 25,557 — the onshore market simply is not *allowed* to trade weaker than that ceiling. A global macro fund believes the dong is genuinely worth more like 26,200 — it sees reserves falling, a wide US–Vietnam rate gap, and outflow pressure. Onshore it can do nothing: the band forbids the price it believes in. Offshore, it sells the dong via a one-year NDF, agreeing an NDF rate of **26,200**. It is now short the dong at a level the onshore market is forbidden from reaching. If a year later the SBV capitulates and the fixing prints at 26,500 on a \$1,000,000 notional:

```
Settlement = 1,000,000 × (26,500 − 26,200) / 26,500 = +$11,321
```

The fund collects **\$11,321** for being right that the band would eventually give. But notice the deeper signal: the *moment it put the trade on* at 26,200 — roughly 3% beyond the onshore ceiling of 25,557 — its NDF quote itself became a public statement that smart money expected a ~3% devaluation. Multiply that across every fund doing the same, and the offshore curve as a whole becomes a forecast of the devaluation the onshore market cannot legally display. The takeaway: the offshore NDF is where the devaluation gets *priced* before the central bank lets it *happen*.

### Reading the gap as a pressure gauge

The dong NDF gap over time is the headline chart of this entire post, because it turns an abstract idea — "devaluation pressure" — into a number you can watch.

![The dong NDF gap offshore depreciation premium bar chart 2018 to 2025](/imgs/blogs/non-deliverable-forwards-trading-uninvestable-currencies-3.png)

The bars show the one-year NDF-implied depreciation premium — how much cheaper the offshore forward priced the dong than the onshore spot — for each year. In calm years like 2020, when the dong was stable and reserves were ample, the premium sat below 1%. Then look at **2022**: the premium spiked to **3.5%**, the highest in the window. That was the year the Federal Reserve hiked aggressively, the dollar surged (the dollar index hit a two-decade high), and capital fled emerging markets globally — the classic "dollar wrecking ball." Vietnam's reserves were drawn down hard defending the dong, and the offshore market priced in the strain. The premium then eased toward 2% in 2023 as the dollar peaked and rolled over, before ticking back up to 2.8% in 2024. The chart is, quite literally, a continuous readout of how much pressure the offshore world is putting on a currency its government is trying to hold steady.

This is why NDF gaps are watched as an early-warning system. The onshore rate, by design, moves smoothly — the central bank smooths it. The offshore NDF is free to lurch, and it lurches *first*, when stress is building. A widening offshore premium is one of the cleanest market-based signals that a managed currency is under devaluation pressure, weeks or months before the band actually gives.

But read it with discipline, because the signal has two well-known failure modes. The first is **liquidity distortion**: a thin NDF can gap on a single large flow that has nothing to do with fundamentals — a corporate hedging a one-off deal, a fund forced to close a position — so a one-day spike in a market as shallow as the dong NDF can be noise, not signal. You want a *sustained* widening across the curve, confirmed by reserves draining and the rate gap, before you treat it as real pressure. The second is **the central bank's reaction function**: a premium prices the *market's expectation* of devaluation, and central banks routinely defeat that expectation. The SBV has, repeatedly, leaned against a wide offshore premium with reserves and a measured crawl, and the devaluation the offshore market priced simply never arrived in full. So the gap is a forecast with a known forecaster on the other side actively trying to falsify it. The skill is not reading the number — it is judging whether the central bank has the reserves, the credibility, and the will to hold the line the offshore market is betting against. That judgment is where the reserve chart, the policy framework, and the NDF premium come together.

## What sets the NDF rate: carry plus fear

We said the offshore NDF price is set by supply and demand for dong exposure rather than clean arbitrage. That is true, but it is too vague to trade on. Let us be precise about the two forces that actually move an NDF rate, because they are the same two forces that move every forward in this series — interest-rate carry and risk premium — just with one of the usual brakes (arbitrage) removed.

**Force one: carry, the interest-rate gap.** Even without free arbitrage, the NDF curve is anchored loosely to the interest-rate differential between the two currencies, because that is the cost of holding the position. The dollar pays a high interest rate; the dong pays a different one. A forward rate has to compensate for that gap, or one side of the trade is getting paid to wait. The mechanics of how a rate gap builds a forward curve are covered in [FX forwards and swaps](/blog/trading/forex/fx-forwards-and-swaps-the-real-workhorses-of-the-market) and rest on the rate-differential logic from [how monetary policy moves currencies](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials) — and we link out rather than re-derive it. The short version: if local rates are high relative to the dollar, the currency trades at a forward *discount* (the forward prices it weaker), because you are being paid extra interest to hold the high-rate currency and the forward has to take that back. The dong's high domestic rates are one reason its NDF curve slopes toward depreciation even in calm times — part of the offshore premium is just carry, not panic.

**Force two: the risk premium, the fear leg.** On top of carry sits a premium for the *risk* that the currency gaps — that the band breaks, the central bank devalues, or capital flight accelerates. In a freely convertible currency, arbitrage would compete this premium down. In a restricted currency, with the arbitrage pipe cut, the premium can balloon, because there is no riskless trade to fade it. The 2022 spike to 3.5% in the dong NDF was mostly this second force: not a 3.5% interest-rate gap, but a fear premium that the dollar squeeze would force the dong's hand. Disentangling the two — how much of an offshore premium is boring carry versus genuine devaluation fear — is one of the real skills of trading these markets.

#### Worked example: splitting an NDF premium into carry and fear

Suppose the one-year USD/VND NDF prices a 2.8% offshore premium over onshore spot (the 2024 reading from the headline chart). You estimate that Vietnamese one-year rates run about 1.5 percentage points above comparable dollar rates — wait, the dollar's rate is the *higher* one in this period, so the carry actually pushes the dong to a forward discount of roughly 1.8% on the rate gap alone. That leaves:

```
Total offshore premium      = 2.8%
Carry (rate-gap) component  ≈ 1.8%
Implied fear premium        ≈ 1.0%
```

So of the 2.8% the offshore forward is pricing, roughly 1.8 points is mechanical carry that any forward would show, and only about 1.0 point is a genuine *extra* premium for devaluation risk. A naive reader sees "2.8% — big devaluation coming." A sharper reader strips out the carry and sees a more modest ~1% fear premium — meaningful, but not a crisis. The takeaway: the headline NDF gap mixes boring carry with real fear, and separating them is what turns the number from a scary headline into a usable signal.

## The onshore crawl versus the offshore lurch

To feel why the gap matters, you have to see how differently the two prices behave. The onshore rate is a managed crawl — deliberately gradual, smoothed by intervention.

![USD VND year-end managed crawl line chart 2014 to 2025](/imgs/blogs/non-deliverable-forwards-trading-uninvestable-currencies-4.png)

The onshore path climbs in a remarkably orderly staircase: 21,340 in 2014, 23,250 in 2018, 24,270 in 2023, 25,450 in 2024, 26,300 in 2025. The dong depreciates — but in a managed, one-directional crawl, never a free fall. Even in the stressed year of 2022, the onshore move was contained to a few percent. This smoothness is a *policy choice*, the heart of Vietnam's [managed float](/blog/trading/forex/usd-vnd-and-the-managed-float-how-the-sbv-runs-the-dong): the SBV would rather grind the rate weaker slowly than let it gap. The offshore NDF, by contrast, is allowed to price the *whole expected future move at once*. When the offshore premium hit 3.5% in 2022, it was effectively pricing a chunk of the crawl that the onshore market would only deliver over the following year or two.

That difference — gradual onshore reality versus front-loaded offshore expectation — is the structural reason the two prices diverge under stress. The onshore market shows you where the rate *is* under the band's smoothing. The offshore NDF shows you where global capital thinks it is *going*, with no smoothing at all.

#### Worked example: hedging a dong receivable when you cannot deliver

A US electronics firm earns **2.5 billion dong** a month from its Vietnam sales — roughly \$98,000 at 25,450. It worries the dong will weaken before it repatriates, shrinking the dollar value. It cannot buy a deliverable dong forward offshore (no delivery pipe), so it uses an NDF. It sells the dong forward — long USD/VND — on a notional sized to its exposure, roughly \$98,000, at an NDF rate of, say, 25,700 (the offshore forward, already pricing some depreciation). Three months later the fixing prints at 26,100. On the NDF:

```
Settlement = 98,000 × (26,100 − 25,700) / 26,100 = +$1,502
```

The firm collects **\$1,502** on the hedge. Meanwhile its actual 2.5 billion dong of revenue, converted at the weaker 26,100, is now worth only ~\$95,785 instead of the ~\$98,000 it expected — a shortfall of about \$2,200. The NDF gain of \$1,502 offsets most of that loss. The hedge is imperfect because the firm hedged at 25,700 (already pricing some weakness) rather than today's spot, and the move overshot — but it converted an uncontrollable currency risk into a small, known residual. The takeaway: an NDF lets a real-economy business hedge a currency it can never physically deliver, paying for the protection in the one currency everyone can settle — dollars.

## What restriction costs: the spread you pay

Nothing about an NDF is free, and the price of restriction shows up most visibly in the bid-offer spread — the gap between the price at which a dealer will buy and sell. In a deep, free market like EUR/USD, that spread is razor thin. In a restricted, non-deliverable currency it is wide, because the market is thinner, harder to hedge, and riskier for the dealer to warehouse.

![Dealer spread by currency tier horizontal bar chart majors crosses EM and NDF](/imgs/blogs/non-deliverable-forwards-trading-uninvestable-currencies-6.png)

On a log scale, the hierarchy is stark. EUR/USD trades at about **0.2 pips**, the most liquid price on earth. The other majors — USD/JPY, GBP/USD, AUD/USD — sit between 0.3 and 0.6. A cross like EUR/GBP costs 0.8. Then you step into emerging markets and the spread jumps an order of magnitude: USD/MXN around 8 pips, USD/TRY around 25. And the USD/VND NDF? About **40 pips** — two hundred times the EUR/USD spread. That spread is the toll for trading a currency the market cannot freely deliver: thinner liquidity, fewer dealers, harder hedging, and the constant risk that the band or the regime shifts under you. When you read an NDF level, you are reading a price with a wide margin baked in, and that margin is itself information — it tells you how restricted, and how risky, the market considers the currency.

### Why the spread is the restriction, quantified

It is worth dwelling on *why* the NDF spread is so wide, because the reasons map onto everything restricted about the currency. A dealer quoting EUR/USD can lay off risk instantly in a multi-trillion-dollar market and hedge perfectly. A dealer quoting a USD/VND NDF cannot: there is no deep deliverable market to hedge into, the offshore NDF pool is shallow, and the dealer's only true hedge — an offsetting onshore position — is behind the same capital controls that created the NDF in the first place. The dealer also bears *regime risk*: the central bank could change the band, devalue overnight, or change the fixing methodology, and the dealer's book would gap. All of that risk is priced into the spread. The 40-pip spread is not greed; it is the honest cost of warehousing exposure to a currency the dealer cannot freely trade out of.

And the spread is not constant — it *breathes* with stress, which is itself a signal. In calm conditions a USD/VND NDF might trade near that 40-pip benchmark. In a dollar squeeze, when everyone wants to be long dollars at once and dealers cannot find anyone to take the other side, the spread can widen several-fold and the quote sizes shrink: you can no longer move size without paying up. A widening *spread*, on top of a widening *premium*, is a second-order stress signal — it tells you not just that the market expects depreciation, but that liquidity itself is drying up, which is exactly when a managed currency is most vulnerable to a disorderly break. Watching the spread alongside the gap is how experienced traders distinguish a market that is merely repricing from one that is genuinely seizing up. The cost of trading the currency, in other words, is information about the health of the currency.

#### Worked example: what the spread costs you on a round trip

You want to express a view on the dong with a \$1,000,000 NDF, and the dealer quotes you a spread of 40 pips around a mid of 25,450 — so you buy (long dollars) at roughly 25,470 and could sell back at 25,430. If you opened and closed immediately, you would cross the full 40-pip spread. In dollar terms, a 40-pip move on this notional is about `1,000,000 × 40 / 25,450 ≈ $1,572` — that is your round-trip transaction cost before the currency has moved at all. On EUR/USD, the same \$1,000,000 round trip at a 0.2-pip spread costs cents. The takeaway: restriction is expensive to trade, and that ~\$1,572 toll is the market charging you for access to a currency it cannot freely deliver — a cost you must clear before any view pays off.

## The dong, the renminbi, and the won: the big NDF markets

NDFs exist for any currency that is economically important enough to need hedging but restricted enough to block delivery. The deepest markets are the big restricted Asian currencies.

![The big four NDF currencies grid renminbi won rupee dong](/imgs/blogs/non-deliverable-forwards-trading-uninvestable-currencies-7.png)

The grid maps the four anchors of the NDF world. **CNY/CNH (China)** is the giant — the renminbi is restricted onshore (CNY) but China deliberately built a deliverable *offshore* version (CNH, the Hong Kong-traded renminbi), so the renminbi has an onshore rate, an offshore deliverable rate, *and* an NDF market, with the gaps between them watched obsessively as a barometer of Chinese capital-flow stress. **KRW (South Korea)** has the most liquid pure NDF market — the won is non-deliverable offshore, and the Korea NDF is the textbook deep, active NDF. **INR (India)** has a huge onshore market but historically thin offshore NDF liquidity, with regulators periodically trying to pull NDF trading back onshore. **VND (Vietnam)** is the thin end: a small, wide-spread NDF on a tightly managed currency, exactly the market we have been dissecting. The pattern is clear — the NDF currencies are the restricted Asian currencies, each with its own flavor of the onshore/offshore split.

### The CNH twist: when the offshore version is deliverable

China is the instructive special case, so it earns its own paragraph. Beijing did something most restricted-currency countries did not: it created CNH, a deliverable offshore renminbi that trades freely in Hong Kong, alongside the restricted onshore CNY and the older CNY NDF market. This gives three prices for one currency — onshore CNY, offshore deliverable CNH, and the NDF-implied rate — and the *spreads between them* are a real-time stress gauge. When capital wants out of China, CNH trades weaker than CNY (offshore demand for dollars outstrips onshore), and the gap blows out. The People's Bank of China watches this CNH–CNY spread the way the SBV watches the dong NDF premium. The lesson generalizes: whenever a currency is caged onshore but priced freely offshore, the *difference* between the two prices is where the truth leaks out. China just built a more liquid leak.

## Common misconceptions

**"An NDF means you own the foreign currency."** No — you never own one unit of it. The restricted currency is a reference number you read off a fixing on one day. The only currency that ever touches your account is the dollar. If you buy a USD/VND NDF you do not have dong; you have a dollar-settled bet whose payoff is keyed to the dong's official rate. This is the single most common confusion, and it is why people misread their own P&L.

**"The offshore NDF rate is the real exchange rate."** Neither rate is uniquely "real." The onshore rate is real for someone inside the country with a license to deliver; the offshore NDF rate is real for global capital that can only settle in dollars. They are two genuine prices for the same currency under two different sets of constraints. Under stress they diverge by design — and the *gap between them*, not either level, is the most informative number.

**"A wide offshore premium means a devaluation is certain."** It means devaluation is being *priced*, not that it is guaranteed. The offshore premium hit 3.5% on the dong in 2022, yet the onshore rate only ground a few percent weaker over the following years — the SBV absorbed much of the pressure with reserves and a managed crawl rather than a one-off devaluation. The premium is a probability-weighted expectation, and central banks frequently lean against it successfully. Treat it as pressure, not prophecy.

**"NDFs are exotic instruments only hedge funds use."** They are mainstream plumbing. Multinationals hedging Asian revenues, bond investors hedging local-currency debt, and corporate treasurers all use NDFs as routine risk management. Per the BIS, forwards (including NDFs) are a large slice of the ~\$7.5 trillion-a-day FX market. The dong NDF is niche, but the NDF *structure* — cash-settled, dollar-denominated, fixing-based — is everyday machinery for the restricted Asian and Latin American currencies.

**"Because no currency is delivered, there's no real risk."** The market risk is fully real — your dollar P&L swings with the fixing, and a managed currency can gap if the band breaks. What you avoid is *delivery and settlement* risk in the restricted currency, not market risk. You can lose your entire notional's worth of move; you just lose it in dollars.

**"The onshore and offshore prices must converge by expiry."** They converge only in the sense that an NDF settles against the *onshore-derived fixing*, so on the fixing date the contract is marked to the official onshore number. But that is convergence at a single point in time against a managed rate — not the continuous arbitrage-driven convergence that welds a deliverable forward to spot. Between now and the fixing, the offshore curve can sit persistently away from the onshore band for the entire life of the trade, because no riskless trade forces them together. A currency can carry a wide offshore premium for years without the onshore rate ever "catching up" — the central bank can simply hold the band and absorb the pressure indefinitely, as long as the reserves last.

**"An NDF is the same as buying the offshore deliverable currency, like CNH."** They are cousins, not twins. CNH is a *deliverable* offshore renminbi — you actually receive and hold real offshore yuan. An NDF delivers nothing; you only ever settle a dollar difference. The distinction matters for funding, for collateral, and for what happens at expiry: holding CNH leaves you with a currency balance to manage, while an NDF simply pays out and vanishes. China is unusual in having both; most restricted currencies, the dong included, have only the non-deliverable version.

## How it shows up in real markets

**The 2022 dollar squeeze.** This is the cleanest modern case, and the headline chart already showed it: as the Fed hiked and the dollar surged to a two-decade high, the dong's offshore NDF premium spiked to 3.5% — the offshore market pricing the strain that the onshore band was smoothing out. The same dynamic hit every managed Asian currency at once; the NDF curves across the region all steepened together, a synchronized warning that the [dollar's cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity) was pulling restricted currencies down. The onshore crawl, meanwhile, kept its orderly staircase — the divergence between the two was the whole story of the year.

**The reserve angle.** Whether a central bank can hold the line against an offshore NDF premium comes down to ammunition — foreign reserves. A central bank defending a restricted currency sells dollars to buy its own currency, supporting the onshore rate. The more reserves, the longer it can lean against the offshore pressure.

![Reserves in months of imports bar chart emerging markets 2024 with three month adequacy line](/imgs/blogs/non-deliverable-forwards-trading-uninvestable-currencies-8.png)

The bars rank emerging markets by reserves measured in months of imports — the classic adequacy yardstick, with the rule-of-thumb three-month line drawn in. China sits at 14 months and India at 11 — deep war chests, which is part of why their NDF gaps, while watched, rarely run away. At the other end, **Vietnam sits at about 3 months** — right at the adequacy line, not below it but with little cushion. That thin reserve buffer is precisely why the dong's offshore NDF premium can spike: the market knows the SBV's ammunition is limited, so when dollar stress hits, the offshore forward prices in the real chance that the band will have to give. Reserves and the NDF gap are two sides of one coin — one is the ammunition, the other is the pressure on the trigger.

#### Worked example: sizing a one-month dong NDF against the fixing one more time

Let us close the loop with the canonical trade stated cleanly, because repetition cements the mechanic. You put on a **\$1,000,000 one-month USD/VND NDF at 25,450** — you are long the dollar, short the dong, expecting weakness. Three outcomes at the fixing:

```
Fixing 25,450 (unchanged): 1,000,000 × (25,450 − 25,450) / 25,450 = $0
Fixing 25,800 (dong −1.4%): 1,000,000 × (25,800 − 25,450) / 25,800 = +$13,566
Fixing 25,200 (dong +1.0%): 1,000,000 × (25,200 − 25,450) / 25,200 = −$9,921
```

If the dong is unchanged at the fixing, you settle flat — the band held and the offshore expectation came to nothing. If it weakens to 25,800 you collect \$13,566. If it strengthens to 25,200 you pay \$9,921. Every outcome is a dollar wire; the dong is never bought, sold, or moved. The takeaway: an NDF compresses an entire view on a restricted currency into a single dollar number paid on one day — the purest possible expression of this series' principle that you never own a currency, only a bet on the gap.

**The renminbi's three-way spread.** The most-watched NDF-adjacent signal in the world is not the dong — it is the renminbi, where China's deliberate creation of an offshore deliverable version (CNH) gives a live, liquid readout of capital-flow stress. During the 2015–2016 capital-outflow scare, when investors feared Beijing would let the yuan slide, CNH traded persistently weaker than onshore CNY, and the gap blew out to multiple big figures at the worst moments. The People's Bank of China intervened in the offshore market directly — buying CNH, squeezing offshore yuan funding rates to punish short-sellers — to close the gap and break the speculative attack. It is the same contest we keep returning to: the offshore market prices the devaluation the authorities are resisting, and the authorities fight back through reserves and funding pressure. The renminbi just has the deepest, most liquid version of the fight, which is why every macro desk on earth watches the CNH–CNY spread as a China stress gauge. The lesson transfers directly to the dong's thinner NDF: a widening offshore premium is the first place the strain becomes visible.

**The won as the liquid benchmark.** If the dong is the thin end of the NDF spectrum, the Korean won is the deep, liquid benchmark — the most actively traded pure NDF in the world. Korea runs a far more open capital account than Vietnam, so the won NDF tracks the deliverable market closely in calm times, with a tight gap. But in risk-off episodes — the won is a high-beta, export-heavy, semiconductor-cycle currency — the NDF can still lurch ahead of the onshore rate as global funds hedge Korean equity and bond exposure fast. The contrast is instructive: the more open the capital account, the tighter the onshore/offshore gap stays, because more arbitrage capital can lean against it. The dong's wide, spiky premium and the won's tight, well-behaved one are two points on the same spectrum, and the width of the gap is a direct readout of how restricted — and how stressed — the currency is.

**Vietnam's policy ties it together.** The reason the dong even has a well-behaved managed crawl, rather than a baht-1997-style collapse, is the policy framework run by the SBV — the credit ceiling, the reference-rate band, the reserve management — covered in [Vietnam's monetary policy](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling). The offshore NDF is the market's continuous referendum on whether that framework is holding. When the framework is credible and reserves are ample, the offshore premium stays small. When stress builds, the premium widens, and the SBV must choose: spend reserves, let the crawl steepen, or accept a wider gap. The NDF gap is the scoreboard for that ongoing contest between policy and capital.

## The takeaway: read the gap, not the level

The single most useful thing an NDF gives you is not a way to trade an obscure currency — it is a *window*. A managed currency's onshore rate is, by construction, a smoothed and politically managed number. It will not tell you when pressure is building, because its whole job is to hide that pressure behind a gentle crawl. The offshore NDF has no such job. It floats to wherever global dollars want it, and so it shows you the pressure raw and front-loaded.

So when you look at a restricted currency, train your eye on the *gap*, not the level. The onshore rate tells you where the central bank is holding the line today. The offshore NDF tells you where the market thinks the line will move. The premium between them — 0.8% in a calm year, 3.5% in the 2022 squeeze — is a direct, market-priced measure of devaluation pressure, available weeks before the onshore band actually gives. It is one of the few times in markets where you can read a government's currency stress straight off a screen, in a contract that, fittingly, can never deliver a single unit of the currency it is about.

A practical checklist falls out of everything above. When an offshore NDF premium widens, run four questions before you act on it. First, **strip the carry**: how much of the premium is just the rate gap that any forward would show, and how much is genuine fear? Second, **check the reserves**: does the central bank have the ammunition — measured in months of imports — to defend the line, or is it near the three-month edge like Vietnam? Third, **read the flows**: is the premium confirmed by capital outflows, a widening trade gap, a draining reserve balance — or is it a one-off liquidity gap in a thin market? Fourth, **judge the will**: does the regime have the credibility and the appetite to spend reserves rather than devalue? Only when carry-stripped fear, thin reserves, confirming flows, and a wavering central bank line up does the NDF premium graduate from a curiosity to a tradeable signal. Each of those four checks is itself a thread of this series — carry, reserves, flows, and policy credibility — and the NDF gap is where they all converge into a single number.

That is the deep lesson of the non-deliverable forward, and it is this series' spine in its sharpest form. You never own a currency in isolation; you own a bet on the gap between two monies. With an NDF the gap is doubled — there is the gap between two countries' interest rates that drives the forward, and the gap between the caged onshore price and the free offshore price that signals the stress. Learn to read both, and a currency you cannot even legally touch becomes one of the most honest gauges on your screen.

## Further reading & cross-links

- [FX forwards and swaps: the real workhorses of the market](/blog/trading/forex/fx-forwards-and-swaps-the-real-workhorses-of-the-market) — the deliverable forward an NDF is built from, and the covered-interest-parity arbitrage that capital controls break.
- [The offshore NDF market for the dong and Asian currencies](/blog/trading/forex/the-offshore-ndf-market-for-the-dong-and-asian-currencies) — a deeper tour of the offshore plumbing and the regional NDF landscape.
- [USD/VND and the managed float: how the SBV runs the dong](/blog/trading/forex/usd-vnd-and-the-managed-float-how-the-sbv-runs-the-dong) — the onshore band-and-crawl regime whose ceiling the offshore NDF prices against.
- [The impossible trinity: pick two of three](/blog/trading/forex/the-impossible-trinity-pick-two-of-three) — why a country that wants a managed rate and monetary independence must keep the capital controls that make NDFs necessary.
- [Vietnam monetary policy: the State Bank, the dong, and the credit ceiling](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling) — the policy framework that keeps the dong's crawl orderly and sets the backdrop the NDF gap measures.
- [The dollar system: why USD rules markets](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy) — why every NDF settles in dollars, and why a rising dollar pressures restricted currencies everywhere at once.

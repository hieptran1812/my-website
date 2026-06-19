---
title: "Spot, Forward and Swap: The Three Ways to Trade a Currency"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Spot, outright forwards and FX swaps are three different instruments for three different jobs — and the forward price is just today's spot rate adjusted by the gap between two interest rates."
tags: ["forex", "currencies", "fx-swap", "forward-points", "covered-interest-parity", "spot-fx", "interest-rate-differential", "fx-plumbing"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — There are three ways to trade a currency, and a forward price is not a forecast: it is today's spot rate mechanically adjusted by the gap between two countries' interest rates.
>
> - **Spot** sets a rate today and delivers the money two business days later (T+2). It is the price everyone quotes, but it is *not* the biggest part of the market.
> - **An outright forward** locks a single exchange rate for a single future date. Its price equals spot plus or minus the **forward points**, and those points come entirely from the interest-rate differential — no view on the future is baked in.
> - **An FX swap** bolts a spot leg to an offsetting forward leg. It is a collateralised loan dressed as two currency trades, and it carries no net currency risk — just a funding cost.
> - **The one number to remember:** with spot USD/JPY at 150.00, a US rate of 5.0% and a Japanese rate of 0.5%, the fair 1-year forward is **≈ 143.6 yen per dollar** — about 6.4 yen *below* spot, purely because of the 4.5-point rate gap.

On the afternoon of 5 August 2024, traders watching their screens saw something that, on paper, should have been impossible. The Japanese yen — a currency famous for moving in slow, grinding trends — strengthened against the US dollar by several percent in a matter of hours. USD/JPY, which had touched almost 162 yen per dollar in early July, was changing hands near 142. Hedge funds that had borrowed yen cheaply to buy higher-yielding assets were being forced to buy that yen back all at once. And underneath the headline price, a quieter market was doing the real work: the market for **forwards** and **swaps**, where the price of borrowing one currency against another is set, was repricing violently.

Most people who follow markets know the word "exchange rate." Far fewer know that the rate flashing on a TV ticker — the *spot* rate — is only one of three distinct ways to trade a currency, and that it accounts for barely a quarter of all the currency trading on Earth. The other two — outright forwards and FX swaps — are where banks, exporters, fund managers and central banks actually do most of their business. If you want to understand how currencies are priced, hedged, funded and occasionally blown up, you have to understand these three instruments and how they fit together.

This post is the foundation for everything that follows in this series. We will build the three instruments from zero, then prove the single most important relationship in all of currency trading: **the forward price is just the spot price adjusted by the interest-rate gap between the two currencies.** No crystal ball, no forecast — just arithmetic. Get that one idea, and a huge amount of the FX market suddenly makes sense.

![Forward price is spot adjusted by the interest-rate gap, shown as a five-step pipeline](/imgs/blogs/spot-forward-and-swap-the-three-ways-to-trade-a-currency-1.png)

## Foundations: The three ways to trade a currency

Before anything else, one rule that runs through this entire series: **you never own "a currency" in isolation.** Every currency trade is a swap of one money for another, so every position is a *pair* — a relative bet. When you "buy dollars," you are always buying them *with* something: yen, euros, pounds. The price of that pair is the exchange rate.

Let me fix the vocabulary, because the rest of the post leans on it.

- **The pair.** USD/JPY means "US dollars priced in Japanese yen." The first currency (USD) is the **base** — the thing you are buying or selling one unit of. The second (JPY) is the **quote** — the currency the price is measured in. USD/JPY = 150.00 means one dollar costs 150 yen. EUR/USD = 1.0800 means one euro costs 1.08 dollars.
- **A pip.** The smallest standard increment of a quote. For most pairs it is the fourth decimal place (EUR/USD moving from 1.0800 to 1.0801 is one pip); for yen pairs it is the second decimal place (USD/JPY 150.00 to 150.01).
- **The rate gap (interest-rate differential).** The difference between the interest rate you can earn on the two currencies. If US one-year money pays 5.0% and Japanese one-year money pays 0.5%, the gap is 4.5 percentage points. This single number is the engine of everything below.

Here is an everyday-money version of the rate gap before any formulas. Suppose a friend in the US will pay you 5% a year to hold their dollars, and a friend in Japan will pay you only 0.5% a year to hold their yen. If someone offers to swap your dollars for yen today and swap them back in a year at an agreed rate, what is a fair "swap-back" rate? It cannot leave you better off for having parked in the higher-paying dollars — if it did, everyone would do the same trade until the advantage vanished. So the swap-back rate has to give back exactly the 4.5% head-start the dollars earned. The dollar must come back "cheaper" in yen by about that 4.5%. That give-back is the forward points, and the 4.5% is the rate gap. You have just derived the entire pricing logic of forwards with no maths at all — the rest of the post is only making that precise.

With that in hand, here are the three instruments. Each does a genuinely different job.

**1. Spot — exchange now, settle in two days.** A spot trade fixes the exchange rate *today* and delivers the two currencies two business days later. That two-day lag is the famous **T+2** convention. You agree the rate on the trade date (T); the actual cash moves on T+2. (A few pairs settle faster — USD/CAD is T+1 — but T+2 is the global default.) Spot is the price you see quoted everywhere, the one a tourist gets at the airport bureau de change (badly) and the one a bank's screen shows (tightly). It is the *cash* market for currency.

**2. Outright forward — lock a future rate today.** A forward fixes an exchange rate now for delivery on some agreed date *further* out than spot — one month, three months, a year, sometimes years. An exporter who knows they will receive a foreign payment in six months can sell that currency forward today, removing all uncertainty about what the rate will be. The forward rate is almost never equal to spot; it is spot plus or minus the **forward points**, and we will spend much of this post deriving exactly where those points come from.

**3. FX swap — borrow one currency against another, then unwind.** An FX swap is two trades stapled together: a **near leg** (usually a spot trade) and a **far leg** (an offsetting forward in the opposite direction). You exchange currencies now and agree to reverse the exchange at a future date. The net effect is not a currency bet at all — it is a *collateralised loan*. You hand over dollars, get yen for a while, then swap back. The "interest" on that loan is the difference between the two legs, which is — once again — the forward points.

Three instruments, three jobs. Spot moves money now. A forward locks a rate for a real future payment. A swap funds or rolls a position without taking a fresh currency view. The rest of this post takes each apart, then shows how the forward and the swap are really the same arithmetic seen from two angles.

## Spot: the price everyone quotes, settled two days later

Start with the simplest instrument, because even it has a subtlety most beginners miss: spot is not instant.

When two banks agree a spot EUR/USD trade at 1.0800 for \$1,000,000 worth of euros, nothing physically happens at that moment. The trade is *binding* — the rate is locked — but the actual euros and dollars do not change hands until two business days later. On trade date (T), the deal is struck. On T+1, the back offices match and confirm the details. On T+2 — **settlement date** — the two legs settle simultaneously: the euro-seller delivers euros, the dollar-seller delivers dollars, and both land in the right accounts at once.

![Spot settlement timeline showing trade date T, confirmation on T+1, and delivery on T+2](/imgs/blogs/spot-forward-and-swap-the-three-ways-to-trade-a-currency-2.png)

Why two days? History and plumbing. The convention dates from an era when settlement instructions crossed time zones by telex and humans reconciled them by hand; two business days gave a London desk and a New York desk time to confirm, fund their accounts, and instruct their correspondent banks. The standard survived even as the technology sped up, because the whole market is wired around it. That two-day window also means **time-zone risk** is real: if you trade USD/JPY, the yen leg settles during the Tokyo day and the dollar leg during the New York day, and for a few hours one party may have paid out before receiving. That gap is **settlement risk** — and it has a famous cautionary tale, Bankhaus Herstatt, which failed mid-settlement-day in June 1974 after counterparties had already paid it Deutsche marks but before it paid out dollars. The plumbing that fixes this (continuous linked settlement, where both legs move or neither does) is a story for a later post; for now, just hold the idea that even "instant" spot carries a two-day tail.

#### Worked example: a spot EUR/USD trade, start to finish

A US importer needs €1,000,000 to pay a German supplier and buys euros spot at EUR/USD = 1.0800.

- **Cost in dollars:** €1,000,000 × \$1.0800 per euro = **\$1,080,000**.
- **Trade date (Monday):** the rate 1.0800 is locked. No cash moves.
- **T+1 (Tuesday):** back offices confirm; still no cash.
- **T+2 (Wednesday):** the importer's bank pays \$1,080,000 and receives €1,000,000; both legs settle together.

Even though the importer "bought euros on Monday," the dollars only leave the account on Wednesday — a two-day float that matters for cash management even when it is invisible on the rate.

There is more hiding inside "T+2" than first appears, and it is worth a beat because the same logic governs every forward and swap date too. The settlement date — the day the cash actually moves — is called the **value date**, and the whole forward and swap market is really a market in value dates. T+2 is not "two calendar days"; it is two *business days* in *both* currencies' financial centres. A USD/JPY trade has to skip a day that is a holiday in either New York or Tokyo, because one side cannot pay if its banks are shut. This is why a spot trade struck on a Thursday before a Monday holiday might not settle until the following Wednesday. Dealers track a holiday calendar for every currency precisely so they agree on the right value date — a mismatch there is one of the most common sources of a failed settlement.

The convention also explains a quirk that confuses newcomers: the **value-tomorrow** and **value-today** trades. If you genuinely need the money sooner than T+2 — say a corporate that must fund an account today — you can deal "tom-next" or "value today," but the rate is adjusted by a tiny amount of forward points to account for the day or two of interest you are pulling forward. Even shortening the settlement by one day is, mechanically, a one-day FX swap. The rate gap reaches all the way down to the overnight tenor. Hold that thought — it is the first hint that *every* deviation from plain spot, in either direction, is priced off the same interest-rate differential we are about to derive.

Now the headline fact that reorders how you should think about this market. **Spot is not the biggest piece of FX.** When the Bank for International Settlements ran its 2022 triennial survey of the global foreign-exchange market, it measured average daily turnover of about **\$7.5 trillion per day** — and broke it down by instrument. Spot was \$2.10 trillion a day. FX swaps were \$3.81 trillion — bigger than spot and outright forwards *combined*.

![FX turnover by instrument in 2022, with FX swaps the largest single slice](/imgs/blogs/spot-forward-and-swap-the-three-ways-to-trade-a-currency-3.png)

Read that chart again, because it is the single most clarifying fact about how currencies actually trade. The thing everyone watches — spot — is roughly 28% of the market. The thing almost nobody outside the industry has heard of — the FX swap — is more than half. The market is not mostly people taking directional bets on where the euro is going. It is mostly the world *funding and rolling* its currency positions: a fund hedging its foreign bonds, a bank managing its dollar funding, a corporate treasurer rolling a hedge forward. Spot is the tip; swaps are the iceberg. We will come back to *why* that is, but the takeaway is already clear: if you only understand spot, you understand a quarter of FX.

One more thing about spot before we leave it: the *price* you get depends entirely on who you are. A spot quote is always two numbers — a **bid** (the price at which the dealer buys the base currency from you) and an **offer** or ask (the price at which it sells to you). The difference is the **spread**, and it is the dealer's compensation for warehousing the risk and providing liquidity. For the deepest, most-traded pair, EUR/USD, the interbank spread in normal conditions is about 0.2 of a pip — almost nothing. For USD/JPY it is around 0.3 pips; for a major like AUD/USD, around 0.6. Step out to an emerging-market pair such as USD/MXN and the spread widens to several pips; to an exotic like USD/TRY and it can be 25 pips or more. The spread is a tax on every round trip, and it scales with how risky and illiquid the currency is for the dealer to hold. The retail customer at an airport kiosk pays a spread *hundreds* of times wider than the interbank rate — which is why "the exchange rate" you read about and "the exchange rate you actually transact at" are never quite the same number. Forwards and swaps inherit this same bid-offer structure, layered on top of the points, as we will see.

## The outright forward: a future rate, locked today

A spot rate answers "what does one currency cost *now*?" An outright forward answers "what rate can I lock in *today* for an exchange that happens *later*?"

Concretely: a forward is a contract to exchange two currencies at an agreed rate on an agreed future date beyond spot. If today's spot USD/JPY is 150.00, the one-year outright forward might be 143.60. That is not a prediction that the yen will strengthen to 143.60 in a year — the market has no idea where spot will actually be. It is the rate at which a buyer and seller will trade *regardless* of where spot ends up. The forward rate is a contractual price, agreed now, for settlement in a year.

Who needs this? Anyone with a *known future currency exposure*. The textbook case is an exporter. Suppose a Japanese electronics maker has sold goods to a US buyer and will be paid \$10,000,000 in six months. The company's costs are in yen, so it cares about how many yen those dollars will buy. If it does nothing and the dollar weakens over six months, its revenue in yen shrinks — a pure currency loss on top of a perfectly good sale. By selling the \$10,000,000 forward today at a locked rate, the exporter converts an uncertain future yen amount into a certain one. The currency risk is gone. It has *hedged*.

The price the exporter locks is the spot rate plus the **forward points** — and the entire rest of this post is about where those points come from. The crucial promise, which I will now prove, is this: the forward points are not a guess about the future. They fall out of arithmetic, from the gap between the two currencies' interest rates.

Two practical features round out the instrument. First, forwards trade at *any* date, not just round ones. Dealers quote standard tenors — one week, one month, three months, six months, a year — but the customer can ask for delivery on an exact day to match a real cash flow (the precise date the supplier invoice is due). A non-standard date is a **broken date**, and the dealer prices it by interpolating the points between the two surrounding standard tenors. Second, a forward is a *firm obligation*, not an option: both sides must exchange the currencies on the value date, whatever spot has done. That is what distinguishes a forward from an FX option, where the buyer pays a premium for the *right* but not the obligation to trade — a separate instrument this series covers elsewhere, and one whose pricing leans on the same forward as its anchor. For the option layer, see [the volatility smile and skew](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more); for now, keep the forward as the plain, binding, no-optionality lock.

### Where the forward points come from: no-arbitrage

Here is the argument, built from money you can actually earn. Suppose you have a million dollars and you want to end up with yen in one year. You have two routes, and the market must price them so neither beats the other — otherwise free money exists, and traders pile in until it doesn't. (This pinning-together of two routes is called **no-arbitrage**, and it is the backbone of all derivatives pricing.)

**Route A — convert now, earn yen.** Take your \$1,000,000, convert to yen at spot 150.00 → ¥150,000,000. Deposit that yen for a year at the Japanese rate of 0.5%. After a year you have ¥150,000,000 × (1 + 0.005) = **¥150,750,000**.

**Route B — earn dollars first, convert later via a forward.** Keep your \$1,000,000 in dollars and deposit it for a year at the US rate of 5.0%. After a year you have \$1,000,000 × (1 + 0.05) = **\$1,050,000**. But you wanted yen, and you don't want to gamble on where spot will be — so you lock a forward rate *today* to convert those dollars in a year.

Both routes start with \$1,000,000 and end with yen in a year. If they delivered different amounts of yen, you could borrow to do the better one, sell the worse one, and pocket the difference risk-free. So the market forces them equal. Route A gives ¥150,750,000. For Route B to give the same ¥150,750,000 from \$1,050,000, the forward rate must be:

**Forward = ¥150,750,000 ÷ \$1,050,000 = 143.57 yen per dollar.**

That is it. The forward rate of ≈ 143.6 is not a forecast — it is the only rate that makes the two routes pay the same. Notice what drives it: the dollar leg grew faster (5.0% vs 0.5%), so to keep the two routes equal, each dollar must buy *fewer* yen in the forward than it does at spot. The high-interest currency (the dollar) trades at a forward **discount**; the low-interest currency (the yen) trades at a forward **premium**. That is a law, not a tendency: **the currency with the higher interest rate always trades at a forward discount, by exactly enough to cancel the interest advantage.**

Written as a formula, for a pair quoted as *quote currency per base currency* (USD/JPY = yen per dollar, so base = USD, quote = JPY):

```
Forward = Spot * (1 + r_quote * T) / (1 + r_base * T)
```

where `r_quote` and `r_base` are the two interest rates and `T` is the time in years. Plugging in: 150.00 × (1 + 0.005) / (1 + 0.05) = 143.57. This relationship has a name — **covered interest parity (CIP)** — and it is the closest thing FX has to a law of physics. "Covered" because the forward leg covers (removes) the currency risk; "parity" because the two routes end up equal.

It helps to see *who enforces it*. Suppose a dealer quoted a 1-year forward of 145.00 instead of the fair 143.57 — too high, meaning the dollar is too "cheap" forward. An arbitrageur would pounce: borrow yen at 0.5%, convert to dollars at spot 150.00, deposit the dollars at 5.0%, and sell the dollar proceeds forward at the rich 145.00. Every leg is locked, no currency risk is taken, and the trade nets a riskless profit because the forward over-pays. Traders would do this in size until the forward fell back to 143.57 and the profit vanished. The mirror trade — borrow dollars, hold yen, sell forward — disciplines a forward that is quoted too low. This two-sided arbitrage is what *welds* the forward to spot and the rate gap; the forward is not "usually" near the parity level, it is *pinned* there by the threat of free money, which is why CIP held to within a basis point for decades before 2008. When we later find it drifting, that drift will be telling us something specific and important about dollar scarcity.

#### Worked example: building the 1-year USD/JPY forward from the rate gap

Spot USD/JPY = 150.00, US 1-year rate = 5.0%, Japan 1-year rate = 0.5%, tenor = 1 year.

- **Quote leg (yen) growth factor:** 1 + 0.005 × 1 = 1.0050.
- **Base leg (dollar) growth factor:** 1 + 0.050 × 1 = 1.0500.
- **Forward = 150.00 × (1.0050 ÷ 1.0500) = 150.00 × 0.95714 = 143.57 yen per dollar.**
- **Forward points = forward − spot = 143.57 − 150.00 = −6.43 yen** (≈ −643 pips).

The dollar is "expensive" today in interest terms (it pays 4.5% more), so the forward makes the dollar *cheaper* by 6.43 yen — exactly enough that lending dollars and converting forward earns you no more than just holding yen.

### Forward points across tenors: the gap compounds with time

The −6.43 yen we just found was for one year. Shorter forwards have smaller points; longer forwards have bigger ones, because the interest gap has more time to accumulate. Build the whole curve from the same three inputs (spot 150.00, US 5.0%, Japan 0.5%) and you get a smooth, monotonic spread of forward points that widens with the tenor.

![Forward points for USD/JPY across tenors, growing from one month to three years](/imgs/blogs/spot-forward-and-swap-the-three-ways-to-trade-a-currency-4.png)

A one-month forward sits only about 0.56 yen below spot; three months, about 1.67 yen; six months, about 3.29 yen; one year, the 6.43 we derived; three years, roughly 17.6 yen below spot. The shape is the rate gap, integrated over time. This is why dealers quote forwards not as full rates but as *points* — they quote "minus 6.43 for the one-year" and you add that to whatever spot is doing. It separates the part that moves second-by-second (spot) from the part that moves with interest rates (the points), and lets a forward desk price a thousand forwards off one spot feed plus a points grid.

#### Worked example: a Japanese exporter hedges a \$10,000,000 receivable

The exporter from earlier will receive \$10,000,000 in one year and sells it forward at the 1-year rate of 143.57.

- **Locked yen proceeds:** \$10,000,000 × 143.57 = **¥1,435,700,000**.
- If instead it waited and spot in a year turned out to be 150.00 (unchanged), it would get \$10,000,000 × 150.00 = ¥1,500,000,000 — more yen.
- If spot fell to 135.00 (dollar weakened), the unhedged result would be ¥1,350,000,000 — far less.

The forward "costs" the exporter the 6.43-yen discount versus today's spot, but that is the *price of certainty*, not a loss: it is just the rate gap, and it removes all guessing about where the dollar lands. The exporter swapped an unknown yen amount for a known one — which, for a business with yen costs, is the whole point.

### How a forward is actually quoted: spot plus a points grid

On a real trading desk nobody types a full forward rate. They quote *spot* and *forward points* separately, and the customer adds them. There is a practical reason: spot ticks several times a second, but the forward points move only when interest rates move, which is far slower. Splitting them lets a dealer price hundreds of forwards off one live spot feed and a points grid that is refreshed when rates change, rather than re-quoting every full rate on every spot tick. It also makes the *risk* legible: a forward desk runs an interest-rate book (the points) separately from the spot book (the level), because those are two different exposures hedged by two different teams.

The points are quoted as a bid and an offer, just like spot. A one-year USD/JPY might be quoted "−6.45 / −6.40," meaning the desk will sell you the forward at 6.40 yen below spot and buy it from you at 6.45 below — a half-pip-of-yen spread on the *points*, layered on top of the spread on spot itself. The customer's all-in forward is "spot ± the relevant side of the points." Crucially, the bid-offer on the points reflects the *interest-rate* risk and the funding cost of carrying the position to the far date, not a view on the currency.

#### Worked example: reading a dealer's USD/JPY forward quote

A corporate asks its bank for a 1-year USD/JPY forward to buy dollars (sell yen) forward. The bank shows spot 150.00 and 1-year points of −6.45 / −6.40.

- The corporate is *buying dollars forward*, so it pays the offer side of the points: −6.40.
- **All-in forward = 150.00 + (−6.40) = 143.60 yen per dollar.**
- On \$10,000,000, it commits to pay ¥1,436,000,000 in a year — locked today.
- The dealer's profit is the spread: it can cover its own forward at roughly the mid (−6.43) and earns the difference, a fraction of a yen on each dollar.

The corporate never had to know covered interest parity to use the quote — but the −6.40 it paid is nothing other than the rate gap, marked up by a sliver for the dealer's spread.

This is also where the difference between a **par forward** and an **off-market forward** shows up. The forward we have been pricing — spot adjusted by the rate gap — is the *par* or *fair* forward: the rate at which the contract has zero value at inception, because both routes pay the same. A customer can instead ask for an *off-market* forward at a chosen rate (say, a round 145.00), but then the contract has value on day one — better than fair for one side — and that value is settled with an up-front cash payment. Treasurers sometimes do this to match an internal budget rate, but it is just the par forward plus a loan; there is no free lunch. The fair forward, the one the rate gap dictates, is the anchor everything else is measured against.

## Reading the same law on EUR/USD: which way do the points go?

The USD/JPY example had the high-rate currency as the base. Flip the convention and the same law gives the *opposite-looking* answer — which trips up a lot of beginners, so let's do it deliberately.

Take EUR/USD, quoted as *US dollars per euro*. Here the base is the euro and the quote is the dollar. Suppose spot EUR/USD = 1.0800, the US one-year rate is 5.0% and the euro-area one-year rate is 3.0%. Now the *quote* currency (the dollar) has the higher rate. Apply the same formula:

```
Forward = Spot * (1 + r_quote * T) / (1 + r_base * T)
        = 1.0800 * (1 + 0.05) / (1 + 0.03)
        = 1.0800 * 1.01942
        = 1.1010 US dollars per euro
```

The forward is *above* spot — the euro buys *more* dollars forward than it does today. At first glance that looks backwards: didn't we just say the high-rate currency trades at a discount? It does — and it still does here. The high-rate currency is the **dollar**, and the dollar trades at a forward discount, which means it takes *more* dollars to buy a euro forward. Same law, read through the EUR/USD quote convention: the euro (low rate) is at a forward **premium** of about 210 pips, the dollar (high rate) is at a forward **discount**. The arithmetic never lies; you just have to keep track of which currency is the base.

#### Worked example: the EUR/USD covered-interest-parity forward

Spot EUR/USD = 1.0800, US rate 5.0%, euro rate 3.0%, tenor 1 year.

- **Quote leg (USD) factor:** 1 + 0.05 = 1.0500.
- **Base leg (EUR) factor:** 1 + 0.03 = 1.0300.
- **Forward = 1.0800 × (1.0500 ÷ 1.0300) = 1.0800 × 1.01942 = 1.1010.**
- **Forward points = 1.1010 − 1.0800 = +0.0210 = +210 pips.**

A US investor who buys German bonds and hedges the euro back to dollars locks this forward — and the +210-pip forward premium on the euro almost exactly eats the extra yield the dollar's higher rate seemed to offer. That cancellation is covered interest parity doing its job, and it is why "just earn the higher rate and hedge" is not free money. For the deeper mechanics of how policy rates set this gap in the first place, see [How monetary policy moves currencies](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials).

The same rate gap that prices the forward also pushes the *spot* pair around over time. When the gap between US and Japanese rates blew out to roughly 4.4 points in 2022, the yen weakened from about 108 to about 157 over the following two years. The forward points and the spot trend are two readings of the same underlying force.

![USD/JPY spot level against the US-minus-Japan two-year rate gap from 2019 to 2025](/imgs/blogs/spot-forward-and-swap-the-three-ways-to-trade-a-currency-6.png)

I want to be careful here, because the series spine matters: the rate gap *prices the forward exactly* (that is arithmetic, covered interest parity), but it only *influences* spot (that is economics, and many other flows push back). The chart shows them moving together because the same interest-rate story drove both — but do not confuse "the forward equals spot plus the gap" (a law) with "spot must follow the gap" (a tendency). For the rates themselves, this series defers to the fixed-income world: see [forward rates: what the market expects rates to be](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be) for how a *rate* forward is built — the FX forward is the same no-arbitrage idea applied to two currencies instead of one yield curve.

## The FX swap: a collateralised loan wearing a currency costume

Now the biggest, least-understood instrument — the one that was more than half of all FX turnover. Once you have the forward, the swap is easy, because **an FX swap is just a spot trade and an offsetting forward, done as a single package.**

Here is the structure. In an FX swap you agree two exchanges at once:

- **Near leg (today):** exchange currency A for currency B at the spot rate.
- **Far leg (later):** exchange currency B back for currency A at the forward rate, on an agreed future date.

You give dollars and get yen now; you give the yen back and get dollars later. The currency you "bought" in the near leg you "sell" in the far leg, in the same size. So at the end you are back where you started in currency terms — you took on *no net currency position*. What you actually did was *borrow yen and lend dollars for a year*, fully secured. The yen you received was collateralised by the dollars you handed over, and vice versa. An FX swap is a **collateralised loan**: a way to swap funding from one currency into another for a set period, with zero credit risk on the principal because each side is holding the other's money the whole time.

![An FX swap shown as a near leg at spot and a far leg at the forward, forming a collateralised loan](/imgs/blogs/spot-forward-and-swap-the-three-ways-to-trade-a-currency-5.png)

And the interest on that loan? It is the difference between the two legs — the spot rate on the way out, the forward rate on the way back. Which is to say: **the forward points are the interest rate on the swap.** This is the deep unity of the whole post. The forward points we derived from covered interest parity *are* the cost of funding one currency with another. The forward and the swap are not two different prices; they are the same number seen from two angles. The forward is the points expressed as a future exchange rate; the swap is the points expressed as the cost of a loan.

#### Worked example: a one-year USD/JPY FX swap as a dollar loan against yen

A US bank has surplus dollars and wants to hold yen for a year without taking a yen view. It does a 1-year USD/JPY swap on \$1,000,000.

- **Near leg (today, spot 150.00):** the bank gives \$1,000,000, receives ¥150,000,000.
- **Far leg (in 1 year, forward 143.57):** the bank gives back ¥150,000,000, receives ¥150,000,000 ÷ 143.57 = **\$1,044,580**.
- **Net dollar result:** it put in \$1,000,000 and got back \$1,044,580 — a gain of **\$44,580**, about **4.46%**.

That 4.46% is not a currency profit — the bank had no net yen position at any point. It is the *interest* on lending dollars for a year, which is what you would expect from roughly the 5% US rate minus a little. The whole trade was a way to park dollars and hold yen for a year at a known funding cost, the rate gap, with the principal fully collateralised throughout. That is why FX swaps dominate turnover: this is how the world's banks and funds manage which currency their cash sits in, day in and day out.

This is also why "FX swap" should not be confused with "currency swap" (also called a cross-currency swap). They sound identical and are different animals. An **FX swap** is short-dated, exchanges only principal (near leg out, far leg back), and embeds the interest in the forward points. A **currency swap** is long-dated (years), exchanges *periodic interest payments* in two currencies along the way, and is really a tool for swapping a stream of foreign-currency debt service into your home currency. In the BIS turnover table, FX swaps were \$3.81 trillion a day and currency swaps just \$0.12 trillion — two orders of magnitude apart. When this series says "swap" without qualification, it means the FX swap.

### Why the swap is the busiest instrument: the rolling hedge

To see *why* swaps dwarf everything, follow one ordinary investor: a US pension fund that buys €100,000,000 of German government bonds. It wants the German yield, not a bet on the euro, so it hedges the currency. It cannot use a single forward to the bond's maturity — the fund may hold the bonds for years and trades in and out — so instead it sells the euro forward for, say, three months at a time, and *rolls* the hedge every quarter. Each roll is an FX swap: close the expiring forward and open a new one, in one package, for a fresh three-month value date.

That single decision generates a stream of swap trades stretching out for as long as the fund holds the bonds. Multiply it by every insurer, pension fund, sovereign wealth fund and bank on Earth that owns foreign assets and hedges them, and you have the \$3.8-trillion-a-day swap market. It is not speculation; it is the *maintenance* of the world's hedged cross-border portfolios. Each roll also re-prices the funding cost at the current rate gap, so a fund's hedging cost rises and falls with interest-rate differentials even though its underlying bond never moves.

#### Worked example: the quarterly cost of a euro hedge

The pension fund hedges €100,000,000 by rolling a 3-month EUR/USD swap. With the US rate at 5.0% and the euro rate at 3.0%, the dollar (the currency it is effectively lending while it holds euro assets and hedges them) earns the higher rate, so the hedge has a cost roughly equal to the 2-point gap over three months.

- **Gap per quarter:** about (5.0% − 3.0%) ÷ 4 = **0.5%** of the hedged notional.
- On €100,000,000 ≈ \$108,000,000 hedged, that is roughly **\$540,000 per quarter**, or about **\$2,160,000 a year**, in hedging cost.
- That cost is the forward points on the roll — and it is exactly the extra yield the fund's dollar-side advantage gave up by holding euro assets.

The fund's "free" foreign yield is not free: the rolling swap quietly charges it the rate gap every quarter, which is covered interest parity asserting itself one roll at a time.

## Who uses which, and why

We now have three instruments and a clear sense that they share one engine — the rate gap. The last piece is *matching the instrument to the job*, because choosing the wrong one is a common and expensive mistake.

![A matrix mapping spot, outright forward, and FX swap to what each does, who uses it, and its currency risk](/imgs/blogs/spot-forward-and-swap-the-three-ways-to-trade-a-currency-7.png)

- **Spot** is for someone who wants the money *now* and is willing to take whatever the rate is today. A tourist, an importer settling an invoice this week, a trader putting on a fresh directional view. Spot *creates* a currency position: after a spot trade you hold the new currency and you are exposed to its moves.
- **An outright forward** is for someone with a *single, known, future* exposure who wants to remove the uncertainty: the exporter with a receivable, the corporate with a foreign bill due in March, the fund that knows it must convert a dividend next quarter. A forward locks one rate for one date. It *is* a one-sided currency bet on the future rate — you have committed to a level — but a *deliberate* one taken to neutralise a real underlying exposure.
- **An FX swap** is for someone who needs to *fund or roll* a position without changing their currency view at all. A fund that owns foreign bonds and hedges them rolls its hedge with swaps every few months. A bank short of dollars borrows them against yen with a swap. A central bank lends dollars to its banking system through swap lines. The defining feature is **no net currency risk** — the two legs cancel, leaving only the funding cost.

Notice the throughline. Spot and outright forwards both *take* currency risk (now, or for a future date). The FX swap is the only one of the three that is *risk-neutral* on the currency and purely about funding. That is exactly why it is the biggest: the world has far more need to *fund and roll* currency positions than to take fresh bets on them. Every hedged foreign-bond portfolio on the planet is rolling FX swaps in the background, and that hedging demand — not speculation — is what makes the swap market the deep, central plumbing of FX. The same hedging-and-funding lens shows up across asset classes; for how the dollar specifically sits at the centre of it, see [the dollar: cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity).

A speculator, interestingly, uses all three at once. To bet that the dollar will rise against the yen, a trader buys USD/JPY — but a leveraged fund rarely wants to fund the full dollar amount in cash. Instead it takes the directional position and *finances* it with FX swaps, rolling the funding every few days or weeks. The directional view lives in the spot-or-forward leg; the financing lives in the swap. This is precisely why the carry trade — borrow a low-rate currency, hold a high-rate one — is a swap-funded position at its core, and why its unwinds tear through the swap market. The instruments are not rival choices for the speculator; they are layers of the same trade. And the bank on the other side of every one of these — the **market maker** — is using the swaps to manage its own funding and the forwards to lay off the rate-gap risk it warehouses. Understanding which instrument does which job for which player is most of understanding who is on the other side of your currency trade.

## Common misconceptions

**"The forward rate is the market's forecast of the future spot rate."** No. This is the single most common and most costly misunderstanding. The forward is set by *today's* interest rates through covered interest parity, full stop. With US rates 4.5 points above Japan's, the 1-year forward sits about 6.4 yen below spot — but that is not the market predicting the yen will strengthen to 143.6. It is the no-arbitrage price. Empirically the forward is a *terrible* forecast of future spot; the well-known "forward premium puzzle" is precisely the finding that high-interest currencies do *not*, on average, depreciate by the amount the forward implies — which is what makes the carry trade profitable on average. The forward tells you the rate gap, not the future.

**"Spot is the whole FX market."** Spot is about 28% of it. FX swaps alone (\$3.81 trillion a day in 2022) are larger than spot and outright forwards combined. If your model of FX is "people betting on where the euro goes," you are looking at the smallest of the three instruments.

**"A forward or a swap means you've paid or borrowed money up front."** A vanilla forward has *no* up-front cash exchange — you agree a rate now and exchange currencies only on the settlement date. An FX swap exchanges principal on the near leg, but it is collateralised: you receive the other currency in return, so neither side is out any money. These are not loans of cash you must fund today; they are agreements about future exchanges (forward) or fully-secured currency funding (swap).

**"FX swap and currency swap are the same thing."** They are not, and conflating them will get you the wrong instrument. The FX swap is short-dated, principal-only, with the interest baked into the forward points (\$3.81 trillion/day). The cross-currency swap is long-dated and exchanges streams of interest payments in two currencies (\$0.12 trillion/day). Off by a factor of about thirty in size and entirely different in use.

**"Hedging with a forward 'costs' you when the points go against you."** A treasurer sometimes refuses to hedge because "the forward is worse than spot" — the dollar buyer who sees a forward of 143.60 versus spot 150.00 feels they are paying up. But the points are not a fee; they are the rate gap. The yen you would have held instead earns 0.5% while the dollars earn 5.0%, so the 6.4-yen "cost" of the forward is exactly offset by the interest you forgo by not holding the higher-yielding currency. Hedged or unhedged, the rate gap is yours to pay either way. Refusing to hedge to "save the points" is just choosing to take currency risk for free — which is rarely what the treasurer means to do.

**"Covered interest parity always holds exactly, so the forward is mechanical."** Almost — but not quite, and the gap is informative. Since 2008, the real forward has drifted slightly away from the textbook rate-gap forward. That residual is the **cross-currency basis**, and it is a live signal, not noise — which is where we go next.

## How it shows up in real markets

The clean covered-interest-parity story — forward equals spot times the rate-gap factor — describes calm markets almost perfectly. But the most interesting moments in FX are when it *bends*. The amount by which the actual forward deviates from the textbook rate-gap forward is the **cross-currency basis**, and it is one of the best stress gauges in all of finance.

![The three-month USD/JPY cross-currency basis at stress and calm dates, deeply negative in 2008 and 2020](/imgs/blogs/spot-forward-and-swap-the-three-ways-to-trade-a-currency-8.png)

Read the basis as "how much *extra* you have to pay, on top of the interest-rate gap, to get dollars through the swap market." When dollars are abundant, the basis sits near zero and covered interest parity holds tightly. When the world scrambles for dollars — a crisis — non-US banks and funds that *must* roll their dollar funding will pay over the odds to get it, and the swap-implied cost of dollars rises above what the rate gap alone says it should be. The basis goes deeply negative.

Why doesn't the arbitrage we just described close the gap instantly? Because after 2008 the arbitrage stopped being free. Doing it requires a bank to expand its balance sheet — to borrow one currency and lend another in size — and post-crisis regulation (leverage ratios, capital charges) made balance sheet a scarce, costly resource. So a non-zero basis can persist: the riskless profit is real, but the *balance-sheet cost* of capturing it eats the profit. The basis is therefore a clean read on how expensive bank balance sheet is at that moment, which spikes precisely when funding is tight. It is the residual that the textbook formula leaves out — and the fact that it exists at all is one of the more important things that changed in markets after the global financial crisis. In the 2008 crisis the 3-month USD/JPY basis blew out to roughly −220 basis points; in the March 2020 COVID dash-for-cash it hit about −145; even in calm 2024 it sat around −35, never fully closing. A persistently negative basis is the market telling you that dollar funding is structurally scarce outside the US — the deep reason the Federal Reserve runs swap lines to other central banks in a crisis, lending them dollars to relieve exactly this pressure.

Now connect this back to the hook. On 5 August 2024, the yen-carry unwind was, mechanically, a stampede out of FX swaps. For years, funds had used swaps to *fund in yen* — borrow cheap yen, hold higher-yielding dollars and other assets, rolling the swap every few months and paying only the small yen funding cost. When the Bank of Japan hiked and the yen suddenly strengthened, every one of those positions had to be unwound: the funds had to buy yen back to close their swaps, all at once. USD/JPY collapsed from about 162 in early July toward 142 by 5 August, and the VIX — Wall Street's fear gauge — spiked to an intraday 65.7. The "currency move" everyone saw on the ticker was the surface; underneath, it was the swap-funding market reversing. The carry trade *is* a leveraged FX-swap position, and its unwinds are why the swap market, not the spot market, is where currency crises actually detonate. For the anatomy of these unwinds across 1998, 2008 and 2024, see [carry-trade unwinds: when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks).

This is not new behaviour. In October 1998, when Russia defaulted and the hedge fund Long-Term Capital Management was failing, the very same dynamic ran in reverse on the same pair: USD/JPY collapsed from about 136 to 112 in roughly three trading days as another generation of yen-funded carry trades was force-unwound. Different decade, identical mechanism — a leveraged FX-swap-funded position, a shock, a stampede to buy back the funding currency. The instrument and the rate gap that made the trade attractive were the constants; only the trigger changed. When you read that "the yen surged" in a crisis, the sentence underneath is almost always "the swap-funded carry trade unwound," and the pair moves most violently precisely because the funding leg, not a fresh directional view, is doing the buying.

It is worth closing the loop on the settlement-risk thread we opened with spot, because it is the reason the swap and forward markets can be as large as they are without seizing up on counterparty fear. The fix to the Herstatt problem is **payment-versus-payment** settlement: a system where both legs of an FX trade move simultaneously or neither moves at all, so no party can pay out and then fail to receive. Since 2002 the bulk of interbank FX settles this way through a dedicated utility, and roughly 40% of global FX turnover now settles payment-versus-payment. That plumbing is invisible when it works and catastrophic when it is absent — and it is what lets a \$7.5-trillion-a-day market of two-legged trades run on trust between counterparties who will never meet. The instruments in this post are only usable at scale because the settlement risk underneath them was engineered down.

One more real-market wrinkle worth knowing: not every currency settles cleanly through forwards at all. Where capital controls or thin offshore markets prevent free delivery — the Vietnamese dong, the Indian rupee, the Brazilian real onshore — the market uses **non-deliverable forwards (NDFs)**, which fix a rate and settle only the *difference* in dollars rather than exchanging the restricted currency. The dong's onshore-versus-offshore NDF gap is a standing gauge of devaluation pressure on the currency. The instrument is the same idea — lock a future rate from the rate gap — adapted to a currency you cannot freely move across the border. We will give NDFs their own post; for the broader picture of how managed currencies trade, the Vietnam thread runs through [Vietnam monetary policy and the dong](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling).

## The takeaway: read the rate gap, and you read the forward

Strip away the jargon and the whole of this post reduces to one sentence you can carry into every currency you ever look at: **a forward price is today's spot rate adjusted by the gap between two interest rates, and an FX swap is that same adjustment expressed as the cost of a loan.**

That single idea reorganises the market for you. When you see a forward quoted below spot, you do not wonder whether the market is "bearish" on the currency — you immediately know the base currency has the higher interest rate, and by how much. When you see a swap point, you read it as a funding cost. When you hear that "the yen is at a forward premium," you translate it instantly: Japanese rates are below US rates, so the yen must trade richer forward to keep the no-arbitrage books balanced. You stop seeing three mysterious instruments and start seeing one engine — the rate gap — driving all three.

It also tells you *where to look* when currencies get violent. The headlines will show you spot, because spot is the price on the ticker. But spot is a quarter of the market. The other three-quarters — the forwards and especially the swaps — are where the world funds, hedges and leverages its currency positions, and that is where the breaks happen. The cross-currency basis blowing out in 2008 and 2020, the carry unwind detonating through the swap market in August 2024: these were not spot stories that spilled into funding. They were *funding* stories that surfaced as spot moves. If you want an early read on currency stress, watch the points and the basis, not just the rate.

And it sets up everything to come. The rate gap that prices every forward is the same gap that, when it is wide enough and the currency stays put, *pays you to take the risk* — the carry trade. The same swap structure that funds a hedge funds a leveraged bet. The same covered-interest-parity logic that holds in calm breaks in crisis, and the size of the break is a number you can read. Three instruments, one engine, and a rate gap that runs through all of it. That is the foundation. From here, the series builds up: what moves the rate gap, who is on the other side of your trade, and what happens when the whole leveraged structure has to unwind at once.

Concretely, here is what to do with this the next time you look at a currency. Find the two interest rates — the policy or short-term rate in each country of the pair. Their difference is the rate gap, and that gap, scaled by the time horizon, *is* the forward points; you can sanity-check a quoted forward against it in your head. If the forward sits below spot, the base currency has the higher rate; if above, the lower. When you hear a currency is "at a forward premium," read it as "its interest rate is below the other side's." And when markets get violent, do not stop at the spot ticker — ask what the swap market and the cross-currency basis are doing, because that is where the funding stress lives and where the next unwind will start. Three instruments, one engine, and a habit of always finding the rate gap first.

You never own a currency in isolation. Every position is a pair, every pair is a spread of two interest rates, and the forward is where that spread is priced down to the last pip.

## Further reading & cross-links

- **The market these instruments trade in:** [The biggest market on Earth: inside the interbank FX market](/blog/trading/forex/the-biggest-market-on-earth-inside-the-interbank-fx-market) — who quotes spot, forwards and swaps, and how the \$7.5-trillion-a-day plumbing fits together.
- **The variable that prices every forward:** [Interest-rate differentials: the master variable of FX](/blog/trading/forex/interest-rate-differentials-the-master-variable-of-fx) — the rate gap is the engine of forward points; this post is where it gets the full treatment.
- **The workhorses themselves, in depth:** [FX forwards and swaps: the real workhorses of the market](/blog/trading/forex/fx-forwards-and-swaps-the-real-workhorses-of-the-market) — NDFs, the cross-currency basis, and how the swap market actually clears.
- **The rates side, deferred to fixed income:** [Forward rates: what the market expects rates to be](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be) — the same no-arbitrage idea applied to a single yield curve instead of two currencies.
- **How policy sets the gap:** [How monetary policy moves currencies: rate differentials](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials) — where the interest rates that drive the forward points come from.
- **When the leveraged version breaks:** [Carry-trade unwinds, 1998–2008–2024: when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) — why the swap-funding market, not spot, is where currency crises detonate.

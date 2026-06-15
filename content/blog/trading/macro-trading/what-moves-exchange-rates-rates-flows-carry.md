---
title: "What Actually Moves a Currency: Rates, Flows, and the Carry Trade"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Over the horizons traders care about, currencies are driven by interest-rate differentials and capital flows, not trade balances — and the carry trade explains the biggest FX moves until it violently unwinds."
tags: ["macro", "monetary-policy", "foreign-exchange", "carry-trade", "interest-rate-parity", "yen", "currencies", "capital-flows", "volatility", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Over the horizons traders actually care about, a currency is priced by interest-rate differentials and capital flows, not by trade balances; the carry trade — borrow a low-yield currency, hold a high-yield one — explains most of the big FX moves, and when it unwinds, volatility explodes.
>
> - **Rates lead, trade lags.** Day to day, currencies move on where rate expectations go and where capital is flowing. The textbook "trade surplus equals strong currency" link is real over decades and nearly invisible over months.
> - **The carry trade is the engine.** Borrow yen at 0.1%, hold dollars at 5.3%, and you collect ~5.2% a year just for holding the position — until a shock makes everyone exit the same door at once.
> - **Parity is the rail, not the prediction.** Covered interest parity pins the forward rate exactly; uncovered interest parity "should" predict the spot move but reliably fails, and that failure *is* the carry premium you earn.
> - **The one number to remember:** USD/JPY went from 103 in 2020 to a 161.9 intraday peak in July 2024 — a ~57% move — driven almost entirely by the US-Japan rate gap blowing out from near zero to ~5 points, then unwound ~12% in days when the gap started to close in August 2024.

On the morning of Monday, 5 August 2024, the Japanese stock market fell 12.4% in a single session — its worst day since 1987. Across the world, traders woke up to red screens, a yen that had ripped higher, and a volatility index (the VIX) that briefly spiked above 65. Nothing had blown up. No bank had failed. No war had started. The trigger was almost comically mundane: the Bank of Japan had nudged its policy rate up by a quarter of a point a few days earlier, and the US jobs report had come in soft. Two small pieces of news about *interest rates*.

So why did markets convulse? Because for three years, one of the most crowded trades on the planet had been to borrow cheap Japanese yen and use it to buy higher-yielding assets everywhere else — US dollars, Mexican pesos, US tech stocks, you name it. That trade is called the **carry trade**, and it had quietly printed money for years. When the rate gap that powered it started to close, everyone tried to unwind the same position at the same moment. The exit was too small for the crowd. The yen spiked, the assets bought with borrowed yen got dumped, and the whole thing fed on itself.

This post is about the machinery underneath that day — and underneath every big currency move. We are going to build, from absolute zero, an understanding of what actually moves an exchange rate. The headline you will walk away with is blunt and a little heretical against the textbooks: **over the horizons a trader cares about, currencies are driven by interest-rate differentials and capital flows, not by trade balances.** Purchasing power parity, the trade deficit, the "fundamentals" you read about in the news — those are slow, decade-scale anchors. The fast money is rates and flows. And the single most important expression of that is the carry trade.

![Carry trade earn the spread and the unwind reversal flow](/imgs/blogs/what-moves-exchange-rates-rates-flows-carry-1.png)

The figure above is the entire mental model on one page, and we will spend the rest of the post earning the right to read it. The top row is the carry: borrow a low-yield currency, convert it, hold a high-yield one, and collect the spread every single day. The bottom row is the unwind: a shock arrives, everyone reverses every leg at once, and the FX loss erases years of accumulated carry in days. Hold these two rows in your head. Everything else is detail.

## Foundations: how to read a currency, from zero

Before we can talk about what *moves* a currency, we have to be ruthlessly clear about what a currency price even *is*. This trips up beginners more than any other concept in macro, because exchange rates are quoted in a way that feels backwards. Let us fix that permanently.

### An exchange rate is a price, and prices have a numerator and a denominator

An exchange rate is just the price of one currency expressed in another. Like any price, it is a ratio: how many units of *this* do you pay for one unit of *that*. The confusion is entirely about which currency is on top.

The market convention is to write a pair as `BASE/QUOTE`, and the number tells you **how many units of the quote currency you need to buy one unit of the base currency.** For `USD/JPY`, the dollar is the base and the yen is the quote, so `USD/JPY = 150` means **one US dollar costs 150 yen.** The dollar is the thing being priced; the yen is the money you pay with.

That phrasing — "one dollar costs 150 yen" — is the single most useful sentence in this whole article. Whenever you see a pair, translate the number into "one [base] costs [number] [quote]." Do that and you will never get the direction wrong.

Here is the everyday-money analogy. Think of a dollar as an apple and the yen as pennies. `USD/JPY = 150` means one apple costs 150 pennies. If tomorrow the same apple costs 160 pennies, the apple got *more expensive* — apples (dollars) went **up**, pennies (yen) went **down**. The number going up means the base currency strengthened.

### Appreciation versus depreciation: which way is "up"?

Two words you must internalize:

- **Appreciation** = a currency becomes *more valuable* (it buys more of the other currency).
- **Depreciation** = a currency becomes *less valuable* (it buys less of the other currency).

For `USD/JPY`, when the number rises from 150 to 160, the **dollar appreciates** (one dollar now buys more yen) and the **yen depreciates** (you need more yen to buy one dollar). The rising number = stronger base, weaker quote. This is exactly backwards from how a beginner's intuition wants it to work, because a "rising USD/JPY" *sounds* like it should be good for the yen. It is not. A rising USD/JPY is a **weakening yen**.

The opposite convention shows up in `EUR/USD`, where the euro is the base. `EUR/USD = 1.08` means one euro costs 1.08 dollars. If it rises to 1.12, the euro appreciated and the dollar depreciated. Same rule: rising number = stronger base.

#### Worked example: read USD/JPY when the rate moves from 150 to 160

Suppose `USD/JPY` starts the year at 150 and ends at 160. Did the dollar rise or fall, and by how much?

First, the direction. The rate is yen-per-dollar. It went *up*, from 150 to 160. A higher yen-per-dollar means each dollar now commands more yen, so **the dollar appreciated and the yen depreciated.**

Now the magnitude — and here is a subtlety that catches people. The dollar's appreciation and the yen's depreciation are *not the same percentage*, because they are measured against different bases.

From the dollar's point of view, it went from buying 150 yen to buying 160 yen:

```
dollar move = (160 - 150) / 150 = 10 / 150 = +6.67%
```

So the dollar appreciated **6.67%**.

From the yen's point of view, we have to flip the quote. One yen was worth `1/150 = 0.006667` dollars and is now worth `1/160 = 0.006250` dollars:

```
yen move = (0.006250 - 0.006667) / 0.006667 = -6.25%
```

So the yen depreciated **6.25%**, not 6.67%. The two numbers differ because percentage changes are not symmetric — a thing that doubles rises 100% but the reverse is a 50% fall. **The dollar rising 6.67% against the yen and the yen falling 6.25% against the dollar are the same event seen from two sides.** Always state which currency you are measuring.

### Spot versus forward: today's price and a locked-in future price

A **spot** exchange rate is the price for settling a currency trade essentially now (in practice, two business days out — "T+2"). When someone quotes "USD/JPY at 150," they mean spot.

A **forward** exchange rate is a price agreed *today* for exchanging the two currencies at a *fixed future date* — one month, three months, a year out. You lock the rate now; you exchange the money later. Forwards are not a forecast. They are a contract. The forward rate is not the market's *prediction* of where spot will be; it is a number pinned by arbitrage to the two countries' interest rates, as we will see. The gap between the forward rate and the current spot rate is called the **forward points**, and those points are pure interest-rate math.

This distinction — spot now, forward locked for later — is the foundation of the entire parity machinery and the carry trade. Hold onto it.

### The real exchange rate and purchasing power parity

Everything above is the **nominal** exchange rate — the actual market number. There is a second concept, the **real exchange rate**, which adjusts for the fact that prices (inflation) differ between countries.

Here is the intuition with no math. Suppose one US dollar buys 150 yen, and a basket of goods costs \$100 in America and ¥15,000 in Japan. Convert the Japanese basket to dollars: ¥15,000 ÷ 150 = \$100. The basket costs the same in both places. The real exchange rate is "fair" — your dollar buys the same amount of stuff whether you spend it at home or convert it and spend it abroad.

Now suppose the yen weakens to 160 per dollar but Japanese prices have not changed. The Japanese basket still costs ¥15,000, which is now only ¥15,000 ÷ 160 = \$93.75 in dollars. Suddenly Japan is *cheap* for an American. The nominal rate moved, prices did not keep up, so the **real** exchange rate moved too — Japan became cheaper in real terms.

**Purchasing power parity (PPP)** is the idea that, in the long run, exchange rates should drift toward the level that makes a basket of goods cost the same everywhere — that the real exchange rate should return to "fair." The famous toy version is *The Economist's* Big Mac Index: if a Big Mac costs \$5 in the US and the equivalent local price implies a different exchange rate, the currency is "overvalued" or "undervalued" relative to burgers.

Here is the load-bearing point for a trader: **PPP is a long-run anchor and nothing more.** Over five to ten years, currencies do tend to drift back toward PPP-fair levels. Over the weeks, months, and even years a trader positions over, PPP is almost useless as a predictor. The yen spent years getting *cheaper and cheaper* against the dollar on a PPP basis from 2021 to 2024, blowing right through "undervalued" and staying there. PPP did not stop it. Rates and flows drove it. We will return to this asymmetry — it is the heart of the post.

### Cross-rates and why the dollar sits in the middle of everything

One more piece of plumbing before we leave the foundations, because it explains why almost every FX move you read about runs *through the dollar*. Most currency pairs are quoted against the US dollar — `EUR/USD`, `USD/JPY`, `GBP/USD`, `USD/CAD`, `AUD/USD`. The dollar is on one side of roughly 88% of all FX trades, even when neither party is American. It is the world's *vehicle currency*: if a Korean firm wants to buy Brazilian reais, the trade very often goes won → dollars → reais, because the dollar leg is the deepest, cheapest, most liquid market on each side.

A **cross-rate** is the price of two currencies *neither of which is the dollar*, computed by chaining their dollar rates. If `EUR/USD = 1.08` (one euro costs 1.08 dollars) and `USD/JPY = 150` (one dollar costs 150 yen), then the `EUR/JPY` cross is simply the product:

```
EUR/JPY = (EUR/USD) × (USD/JPY) = 1.08 × 150 = 162.0
```

One euro costs 162 yen. You did not need a separate euro-yen market to find that price — you chained two dollar pairs. This matters for a trader for a concrete reason: **a move in the dollar shows up in almost every pair at once.** When the dollar broadly strengthens — say on a hawkish Fed surprise — `EUR/USD` falls, `GBP/USD` falls, `AUD/USD` falls, and `USD/JPY` rises, all in the same breath, because they all have a dollar leg. The dollar index (DXY) is exactly a weighted basket of those dollar pairs, which is why a single "the dollar is up" headline moves your whole FX screen.

The practical upshot: before you form a view on, say, the euro, separate "is this a euro story or a dollar story?" If `EUR/USD` is falling but `EUR/JPY` and `EUR/GBP` are flat, the euro is fine — the dollar is just strong, and the move is a *dollar* event you would also see in yen and sterling. If `EUR/USD`, `EUR/JPY`, and `EUR/GBP` are *all* falling, *now* you have a genuine euro-weakness story. Decomposing a pair into its dollar leg and its idiosyncratic leg is the first thing a macro trader does, and it is only possible because the dollar sits in the middle of the whole system.

## What moves FX short-run versus long-run

Now we can state the central claim precisely and defend it.

Over **short horizons** (intraday to a couple of years — the trader's world), a currency is moved by two forces:

1. **Interest-rate differentials and rate expectations.** Money flows toward where it earns more. If US rates are 5.3% and Japanese rates are 0.1%, capital wants to be in dollars, all else equal — and crucially, it is the *change in expectations* about that gap that moves the price, often violently and instantly when a central bank surprises.
2. **Capital flows.** The actual buying and selling: foreigners buying US Treasuries, funds rotating into emerging-market bonds, pension funds hedging, hedge funds putting on or taking off carry. FX is the largest market in the world — roughly \$7.5 trillion a day changes hands — and the vast majority of that is *financial* flow, not payment for imported cars and soybeans.

Over **long horizons** (five to ten years), the slow anchor of **PPP** and the cumulative effect of trade and competitiveness reassert themselves. A country that runs persistent inflation will see its currency depreciate over the long haul. A country whose goods become structurally cheaper attracts trade and, eventually, currency strength. But "eventually" is doing enormous work in that sentence.

![Drivers of FX with PPP as a long run anchor](/imgs/blogs/what-moves-exchange-rates-rates-flows-carry-5.png)

The figure separates the fast lane from the slow lane. Rate expectations, capital flows, and central-bank intervention set the exchange rate *minute by minute*. PPP defines a long-run fair value that *tugs* the rate back — but the tug operates over five to ten years, so on any given Tuesday it is essentially silent. A trader who positions on PPP fights the flows and loses for years before being "right." A trader who positions on the rate gap and the flows captures the move that actually happens.

The crucial practical consequence: **when you are forecasting a currency over a trading horizon, ask about rates and flows first, and treat the trade balance as background.** The next sections build out exactly how the rate machinery works, because that is where the tradable signal lives.

### The four kinds of flow, ranked by speed

"Capital flows" is a bucket word. It pays to break it into its components, because they move at different speeds and respond to different triggers — and a trader who knows which flow is driving a move knows how durable it is.

- **Speculative / leveraged flow (fastest).** Hedge funds, prop desks, and systematic carry strategies. This is the money that moves *first* on a rate surprise, because it is the most leveraged and the most rate-sensitive. It is also the money that *unwinds* first and hardest. When USD/JPY moves 2% in an hour on a CPI print, this is the flow doing it. It is fast, fickle, and the source of both trends and crashes.
- **Portfolio flow (fast).** Real-money investors — pension funds, insurers, mutual funds — buying foreign bonds and equities. When US 10-year yields are far above Japanese yields, Japanese life insurers face a choice: buy US Treasuries for the yield (which requires buying dollars) or stay home. These flows are huge and persistent and respond to *rate differentials* and *relative asset performance*. They are slower than leveraged flow but far larger.
- **Hedging flow (variable).** The same real-money investors deciding whether to *hedge* the currency risk on their foreign holdings. A Japanese investor holding US Treasuries can sell dollars forward to remove FX risk — and the *cost* of that hedge is exactly the forward points we computed, driven by the rate gap. When the rate gap is huge, hedging is expensive, so investors hedge *less*, which leaves more unhedged dollar demand. This second-order effect amplified the yen's slide: the wide rate gap made yen-hedging US assets prohibitively expensive, so Japanese capital flowed into dollars *unhedged*, adding to dollar demand.
- **Trade / FDI flow (slowest).** Payment for imports and exports, foreign direct investment, dividend repatriation. This is the "fundamentals" flow that the textbooks lead with — and it is the *slowest and smallest* relative to the financial flows above on any given day. It matters over years. It does not explain Tuesday.

Rank them and the central claim of this post becomes mechanical rather than rhetorical: **the fastest, most rate-sensitive flows dominate the price on a trading horizon, and they are precisely the ones the trade balance has nothing to do with.**

## Covered interest parity and forward points

Let us start with the one piece of FX theory that *actually holds* almost perfectly: **covered interest parity (CIP)**.

The setup is a no-arbitrage argument. You have one dollar. You have two ways to turn it into dollars one year from now, both completely risk-free:

- **Stay home.** Put the dollar in a US deposit earning, say, 5.3%. After a year you have \$1.053.
- **Go abroad, fully hedged.** Convert your dollar to yen at the spot rate (150 yen). Put the 150 yen in a Japanese deposit earning 0.1%. *At the same time*, lock in a forward contract today to convert your yen back to dollars in a year at a guaranteed forward rate. After a year you have your yen plus 0.1% interest, converted back at the pre-agreed rate.

Both paths are riskless — in the second, the forward contract removed all the exchange-rate uncertainty up front ("covered"). If the two paths gave different dollar amounts, you could borrow in the cheaper one and lend in the richer one and pocket free money forever. Arbitrageurs do exactly that, and their trading forces the two paths to give **the same answer.** That equality *defines* the forward rate.

![Interest rate parity no arbitrage box linking spot forward and rates](/imgs/blogs/what-moves-exchange-rates-rates-flows-carry-3.png)

The figure is the no-arbitrage box. The left node is your dollar. The top branch goes abroad — convert at spot, earn the foreign rate, lock a forward, sell back. The bottom branch stays home and earns the domestic rate. The two branches *must* end at the same dollar amount, and that requirement is what pins the forward rate. Read it once and the formula below will feel obvious rather than memorized.

Algebraically, covered interest parity says:

```
Forward = Spot × (1 + r_quote) / (1 + r_base)
```

where `r_base` is the interest rate on the base currency (USD here) and `r_quote` is the rate on the quote currency (JPY). The currency with the **higher** interest rate trades at a **forward discount** — its forward price is weaker than spot — and the currency with the lower rate trades at a forward premium. This is not a market opinion. It is arithmetic enforced by arbitrage.

#### Worked example: compute the forward points from a rate differential

Spot `USD/JPY` is 150.00. The one-year US interest rate is 5.3%; the one-year Japanese rate is 0.1%. What is the one-year forward rate, and what are the forward points?

Apply covered interest parity. The base is USD (5.3%), the quote is JPY (0.1%):

```
Forward = 150.00 × (1 + 0.001) / (1 + 0.053)
        = 150.00 × 1.001 / 1.053
        = 150.00 × 0.95061
        = 142.59
```

The one-year forward is **142.59 yen per dollar**, *below* the spot of 150. The forward points are:

```
forward points = 142.59 - 150.00 = -7.41 yen
```

That negative number is the whole story. Because the dollar earns 5.3% and the yen earns 0.1%, the dollar is *forward-discounted* by about 7.4 yen. In plain English: **the market will pay you, in the forward, to give up the high US yield — the forward rate already bakes in a weaker dollar by exactly the interest advantage.** The 5.2-point rate gap and the ~5% forward discount are the *same number wearing two hats.* This is mechanical, it always holds, and it is the launchpad for understanding why the carry trade is so seductive and so dangerous.

The practical takeaway: when you see "the yen is cheap to hold in the forward" or "dollar forward points are deeply negative," that is not a forecast of yen strength — it is just the rate gap expressed as a forward. The forward is the *fair* price given rates, not a prediction of spot.

## Uncovered interest parity and why it "fails"

Now we get to the interesting part — the place where theory and reality split, and where money is made.

**Uncovered interest parity (UIP)** takes the same idea but removes the forward hedge. It claims that the *expected* future spot rate should equal today's forward rate. Put differently: UIP says the high-yield currency should *depreciate* by exactly its interest advantage, so that — on average — you earn the same return holding either currency. The extra interest you collect on the high-yielder should be precisely cancelled by the currency falling.

If UIP held, the carry trade would be pointless. You would borrow yen at 0.1%, hold dollars at 5.3%, collect your 5.2% — and the dollar would, on average, fall 5.2% against the yen, leaving you with nothing. The free lunch would be eaten by the exchange rate.

Here is the empirical fact that built a thousand hedge funds: **uncovered interest parity reliably fails.** High-yield currencies, on average, do *not* fall by their interest advantage. They often hold roughly flat, or even appreciate, for long stretches. This is called the **forward premium puzzle** or the **UIP anomaly**, and it is one of the most robust findings in international finance. The high-yielder earns its rate advantage *and* fails to depreciate as much as theory says — so the carry trade earns a positive average return.

Why does UIP fail? There are competing explanations, and you do not need to pick a winner, but the trader's-eye version is this: **the extra return is compensation for a risk that is real but rare.** Carry pays you a steady premium most of the time *because* every so often it blows up spectacularly. You are being paid to hold a position that occasionally suffers a sudden, violent loss — a loss that is correlated across everyone holding it, arriving exactly when markets are most stressed. That is not a free lunch; it is an insurance premium you collect for selling crash protection, whether you realize that is what you are doing or not.

This reframing is the most important idea in the post: **the carry premium is the UIP failure, and the UIP failure is a crash risk you are short.** Keep it in mind as we build the trade.

### Why does UIP fail? Three lenses, none of them comforting

You do not have to pick one explanation, but it helps to hold all three, because each implies a different early-warning signal.

The **risk-premium** lens, which we just met, says the carry return is compensation for crash risk. The implication: carry is a *risk-on* trade dressed as a yield trade. It will pay you in calm times and take it all back in the stress event, and there is no escaping that bargain — only sizing for it. The signal this lens gives you is *volatility*: when the price of crash insurance (FX implied vol, the VIX) is cheap, carry is being underpriced for risk and will keep paying; when it ticks up, the premium is about to be claimed.

The **slow-moving-capital** lens says UIP fails because real-money capital does not reallocate instantly. When a rate gap opens, the portfolio and hedging flows that *should* arbitrage it away (selling the high-yielder forward until UIP holds) take *months to years* to fully respond — pension mandates are sticky, hedging policies are reviewed quarterly, and there is only so much balance sheet to put on the trade. So the gap persists, and the carry trader is paid for being the fast capital that shows up before the slow capital. The signal here is *positioning*: the trade keeps paying while capital is still flowing in, and turns when the flow is exhausted and one-sided.

The **peso-problem** lens says the historical record *understates* carry's true risk, because the worst events are rare enough that any finite sample under-counts them. A carry strategy can run for years looking like a Sharpe-2 machine simply because the steamroller has not arrived *yet* in your data window. This is the most dangerous lens for a backtester: the strategy that looks best is often the one whose crash simply has not been sampled. The signal it gives is humility — assume your backtest has not seen the real tail.

All three point the same way for a trader. **The carry premium is real, it is earned for bearing a crash risk, and the crash is rarer and larger than your intuition (or your backtest) wants to believe.** That is the entire risk-management problem of carry, and the rest of the post is about respecting it.

## The carry trade: mechanics, and why it works until it doesn't

A **carry trade** is, at its core, embarrassingly simple:

1. **Borrow** money in a currency with a low interest rate (the *funding currency* — historically the Japanese yen and the Swiss franc).
2. **Convert** it to a currency with a high interest rate (the *target* or *asset currency* — the US dollar, the Australian dollar, the Mexican peso, the Brazilian real).
3. **Hold** the high-yielding asset and collect the interest-rate spread — the *carry* — every day.
4. **Eventually** convert back and repay the loan.

Your profit, before any exchange-rate movement, is the spread between what you earn and what you pay. If you borrow yen at 0.1% and hold dollars at 5.3%, you pocket roughly 5.2% per year just for keeping the position open. The position throws off positive cash flow continuously, like rent.

The catch is the exchange rate. You borrowed in yen, so to repay the loan you eventually have to buy yen back. If the yen *weakened* while you held the trade (USD/JPY rose), you get a bonus — your dollars buy back the cheaper yen with room to spare, *on top of* the carry. If the yen *strengthened* (USD/JPY fell), you take an FX loss when you repay, and that loss can dwarf the carry.

#### Worked example: the annual carry on a \$1,000,000 yen-funded dollar position

You put on a classic carry trade. You borrow ¥150,000,000 in yen, which at a spot of 150 yen per dollar is exactly \$1,000,000. You convert it to dollars and hold a \$1,000,000 dollar position.

Use the real US funding cost. From our data, the Fed funds upper bound peaked at 5.50% (set 26 July 2023), so a dollar deposit earned roughly 5.3% after the small spread to the effective rate. The yen borrowing cost is ~0.1%.

Annual interest *earned* on the dollar leg:

```
$1,000,000 × 5.3% = $53,000
```

Annual interest *paid* on the yen loan (¥150,000,000 at 0.1%):

```
¥150,000,000 × 0.1% = ¥150,000
at 150 yen per dollar  =  $1,000
```

Net carry, before any FX move:

```
carry = $53,000 - $1,000 = $52,000 per year
```

That is **5.2% of the position, paid to you for doing nothing but holding it.** On a \$1,000,000 notional, you collect \$52,000 a year, dripping in daily, as long as the rate gap and the exchange rate cooperate. This is why carry is so seductive: it *looks* like a high-yield bond with no credit risk. The risk is hiding entirely in that last clause — "as long as the exchange rate cooperates."

Now scale your intuition. A real carry book is *leveraged*. If a fund holds that \$1,000,000 position on \$100,000 of capital (10× leverage), the \$52,000 carry is a **52% annual return on capital.** That is the dream the carry trade sells. It is also why the unwind is so dangerous: leverage cuts both ways, and a modest FX move against a 10× position is a catastrophe.

![USD JPY year end close 2019 to 2025 with peaks marked](/imgs/blogs/what-moves-exchange-rates-rates-flows-carry-2.png)

The chart shows why this trade printed money for years. USD/JPY started 2020 near 103 — a strong-yen world — and ground relentlessly higher to 131 (2022), 144 (2023), and 157 (2024), with intraday spikes to 151.9 in October 2022 and an astonishing 161.9 in July 2024. Every tick higher on this chart is the yen *weakening*, which means the yen-funded dollar carry trader was earning the 5.2% carry *and* a tailwind from the falling yen. For three years it was the best risk-adjusted trade on the desk. The problem with trades that work for three years is that everyone notices.

### Why it works until it doesn't

The carry trade has a payoff profile that is the opposite of what most investors want. It earns a small, steady, positive return the overwhelming majority of the time, punctuated by rare, sudden, large losses. Practitioners call it "picking up pennies in front of a steamroller" or "up the stairs, down the elevator."

![Carry pays slowly then crashes fast illustrative payoff shape](/imgs/blogs/what-moves-exchange-rates-rates-flows-carry-6.png)

This chart is **illustrative** — a stylized shape, not a market quote — but the shape is the truth. The green line is the carry income accruing slowly, month after month, for years. The red line is the unwind: a drawdown that arrives in days, not months, and erases years of accumulated carry. The asymmetry is the entire risk profile of the strategy. A backtest that only spans the green part of the curve looks like a Sharpe-ratio miracle. A backtest that includes one red event looks like exactly what it is: selling disaster insurance.

The unwind mechanism is a crowding problem. When carry works for years, more and more capital piles into the same trade — short yen, long high-yielders. The positioning becomes enormous and one-sided. Then a shock arrives that threatens the rate gap or simply spikes risk aversion: the funding central bank hikes, the target central bank cuts, or a risk-off event makes leveraged players cut exposure. The first movers buy back yen to close their positions. That buying *strengthens* the yen (USD/JPY falls), which inflicts losses on everyone *still* in the trade, which triggers their stops and margin calls, which forces *them* to buy back yen, which strengthens the yen further. The position is so crowded that the exit is too small for the crowd.

![Carry unwind cascade from crowded positioning to volatility spike](/imgs/blogs/what-moves-exchange-rates-rates-flows-carry-7.png)

The cascade figure is the playbook warning in one picture. A crowded carry trade sits calm until a shock triggers the first losses; those losses force deleveraging; the forced buying of the funding currency *is itself* the price move that triggers the next round of losses; and the whole thing ends in a volatility spike that overshoots fair value. Notice the self-reinforcing middle: in a carry unwind, the selling causes the price move that causes more selling. That feedback loop is why carry unwinds are so fast and so violent — and why "it's only a 25-basis-point hike" can produce a 12% currency move in a week.

#### Worked example: the carry-unwind hit when USD/JPY drops from 160 to 145

You are holding that \$1,000,000 yen-funded dollar carry trade. You put it on when USD/JPY was near its July 2024 peak of around 160 (call it 160 for round numbers; the intraday high was 161.9). Then the unwind hits and USD/JPY falls to 145 — a move squarely inside the August 2024 range, on the way back down from the 157.2 year-close.

You borrowed ¥150,000,000... wait — you actually borrowed at the rate you entered, 160. So your yen loan is:

```
$1,000,000 × 160 = ¥160,000,000 borrowed
```

When USD/JPY falls to 145, repaying that ¥160,000,000 loan now costs:

```
¥160,000,000 / 145 = $1,103,448 to buy back the yen
```

Your FX loss on repaying the loan:

```
FX loss = $1,103,448 - $1,000,000 = $103,448
```

You lost **\$103,448** on a \$1,000,000 position from a single ~9.4% drop in USD/JPY (160 → 145 is a 9.375% fall). Now compare that to the carry you were earning — \$52,000 a year:

```
years of carry wiped out = $103,448 / $52,000 = ~2.0 years
```

**A ~9% currency move erased almost two full years of carry in a matter of days.** And remember the leverage: if you held this on \$100,000 of capital at 10×, that \$103,448 loss is *more than your entire capital* — you are wiped out and then some, facing a margin call you cannot meet. This single example is the whole reason carry unwinds are market-moving events rather than line items. The slow drip up and the cliff down are not symmetric, and leverage turns the cliff into a crater.

## Common misconceptions

### Myth 1: "A trade surplus means a strong currency."

This is the most stubborn intuition in macro, and it is wrong on the horizons that matter. The reasoning *sounds* airtight: a country that exports more than it imports has foreigners buying its goods, who must buy its currency to pay, so demand for the currency rises. True in a vacuum. But trade flows are a rounding error next to *financial* flows. Total goods trade is measured in tens of trillions per *year*; the FX market turns over ~\$7.5 trillion per *day*. Japan ran a structural current-account position for decades while the yen swung wildly on rate differentials. The US runs a massive, persistent trade deficit and the dollar has spent long stretches *strengthening* — the DXY dollar index closed 2024 at 108.5, near multi-year highs, in the teeth of a record trade deficit. **Capital flows, driven by rate differentials and risk appetite, swamp trade flows over any trading horizon.** The number to remember: ~\$7.5 trillion a day in FX, against which annual trade is a trickle.

### Myth 2: "Purchasing power parity tells you where the currency is going."

PPP is a real force — over five to ten years. On a trading horizon it is close to useless as a directional signal. From 2021 to 2024 the yen got *more and more* undervalued on every PPP metric, blowing past "cheap" into "absurdly cheap," and it kept falling because the rate gap kept widening. A trader who shorted USD/JPY in 2022 because "the yen is undervalued on PPP" was run over for two more years. PPP is a long-run gravity, not a short-run map. Use it to know which way the *very* long-run wind blows, never to time a trade. The number: the yen fell roughly 35% against the dollar from 2020 to its 2024 peak *while* getting cheaper on PPP the entire way.

### Myth 3: "A weak currency is always bad."

A weakening currency is routinely reported as a national humiliation, but it is a two-sided event. A weaker currency makes a country's exports cheaper and more competitive abroad, supports domestic producers against imports, and inflates the home-currency value of foreign earnings — which is why exporter-heavy stock indices often *rise* when their currency falls. The flip side is imported inflation: a weaker currency makes imported energy, food, and goods more expensive, which is precisely why the Bank of Japan eventually had to care about a yen past 160. "Weak currency = bad" and "weak currency = good" are both lazy. The correct frame is: **a currency move redistributes pain and gain between importers, exporters, savers, and the central bank's inflation mandate.** Whether it is "good" depends entirely on which of those you are.

### Myth 4: "The forward rate predicts the future spot rate."

The forward rate is *not* a forecast. It is the spot rate adjusted by the interest-rate differential, pinned by covered interest parity arbitrage. When dollar forwards point to a weaker dollar (negative forward points against a high US rate), that is not the market predicting dollar weakness — it is the rate gap expressed in forward terms. In fact, the *failure* of the forward to predict spot — uncovered interest parity breaking down — is exactly the carry premium. If forwards predicted spot, carry would not pay. They don't, so it does.

### Myth 5: "Central banks can defend any level if they want to."

Intervention can change the *timing and speed* of a move, and a credible threat can squeeze short-term speculators, but a central bank fighting the rate differential is fighting the tide. Japan's Ministry of Finance intervened heavily in 2022 and again in 2024, spending tens of billions of dollars to buy yen at the 152–162 zone. The yen bounced each time — and then resumed weakening, because the *rate gap was still ~5 points*. Intervention without a change in the underlying rate path is a speed bump, not a wall. The yen only truly turned when the Bank of Japan actually started hiking and the Fed started cutting — when the *gap itself* began to close.

## How it shows up in real markets

### The yen carry trade, 2021–2024: the rate gap as engine

The cleanest case study in modern macro is the US-Japan rate divergence. Through 2020 and 2021, both the Fed and the Bank of Japan sat near zero, and USD/JPY hovered around 103–115. Then, in 2022, US inflation hit a 40-year high of 9.1%, and the Fed hiked at the most aggressive pace in modern history — from 0.25% to 4.50% in nine months, eventually peaking at 5.50% in mid-2023. The Bank of Japan, fighting its own decades-long battle against deflation, kept its policy rate pinned near zero.

![Fed funds versus BOJ rate gap driving USD JPY dual axis](/imgs/blogs/what-moves-exchange-rates-rates-flows-carry-4.png)

This is the single most important chart in the post. The blue step line is the Fed funds upper bound climbing from 0.25% to 5.50%; the amber dashed line is the Bank of Japan glued to ~0.1%; the red line (right axis) is USD/JPY. Watch how the yen tracks the *gap*. As the blue line steps up through 2022, the rate differential blows out from essentially zero to roughly five percentage points — and USD/JPY climbs in near lockstep, from ~115 to ~131 to ~144 to ~157. The trade balance did nothing to explain this. The rate gap explained almost all of it. **This is what "currencies are driven by rate differentials" looks like in the actual data.**

For three years, the carry trade on this divergence was the most reliable money on Wall Street: borrow yen near zero, hold dollars near 5%, and watch the yen weaken on top of it. The position grew enormous — yen-funded longs spread far beyond the dollar, into Mexican pesos (yielding 11%), Brazilian reais, Indian rupees, and into US equities themselves. By 2024, estimates of the total yen-funded carry book ran into the trillions of dollars.

### The August 2024 unwind: when the gap started to close

Then the gap began to narrow from both ends. The Bank of Japan, with the yen at a 38-year low past 160 and imported inflation biting, finally moved — it had exited negative rates in March 2024 and, on 31 July 2024, hiked again to 0.25%. Days later, on 2 August, a soft US jobs report ignited fears the Fed would have to cut aggressively. The rate gap that had powered the carry for three years was now, for the first time, *visibly closing*.

The crowded trade reversed exactly as the cascade figure predicts. The first funds to buy back yen pushed USD/JPY down; the falling pair triggered losses across the enormous, leveraged, one-sided carry book; margin calls and stops forced more yen buying; the buying drove the pair lower still. Over a few sessions in early August, USD/JPY fell from around 160 toward 145 — a ~9–10% move in days — Japan's Nikkei fell 12.4% in one session, and the VIX spiked above 65 intraday before collapsing back almost as fast. Nothing fundamental had broken. A crowded carry trade had simply tried to exit through a door too small for it. By year-end, with the dust settled and the gap still wide-ish, USD/JPY had recovered to 157.2 — but the August spike is the textbook illustration of *carry unwinds*: violent, self-reinforcing, and triggered by a rate move that, in isolation, looked trivial.

Notice what the unwind did *not* respect: the trade balance, PPP, or any "fundamental." Japan's external position in August 2024 was essentially what it had been in July. The yen did not strengthen 10% because Japan suddenly exported more cars. It strengthened because a giant, leveraged, one-sided *financial* position had to be unwound through a market too thin to absorb it gracefully — and then, with the position cleared and the rate gap still wide, it weakened straight back toward 157 by December. If you needed one episode to prove that flows and rate expectations drive FX while the trade balance watches from the sidelines, this is it: the entire round trip happened with the "fundamentals" frozen.

### The intervention that wasn't a wall

It is worth lingering on Japan's interventions, because they are a perfect lesson in what a central bank can and cannot do to a currency. In September and October 2022, with USD/JPY pushing toward 152, Japan's Ministry of Finance spent roughly ¥9 trillion (about \$60 billion) buying yen. In 2024, defending against the slide past 160, it spent on the order of ¥10 trillion more. Each time, the yen jolted stronger by several yen within minutes — intervention *does* move the price on impact, because it is real, large, market-order flow.

And each time, the yen resumed weakening within weeks. Why? Because the interventions did nothing to the *rate gap* — the Fed was still at 5.50%, the Bank of Japan still near zero, and the ~5-point differential that powered the carry was untouched. Selling \$60 billion of reserves to buy yen is a one-time flow; the rate gap is a *continuous* flow that pulls capital toward dollars every single day. A one-time flow cannot beat a continuous one. The intervention bought time and punished the most leveraged short-term speculators, but it could not reverse the trend, because the trend's engine — the rate differential — was still running. The yen only truly turned in 2024 when the engine itself changed: the Bank of Japan started hiking and the Fed started cutting. **A central bank can fight the *speed* of an FX move with its reserves; it can only fight the *direction* with its policy rate.**

### EM carry: the same trade, more yield, more fragility

The carry trade is not just a yen story. Emerging-market currencies — the Mexican peso, Brazilian real, Turkish lira, South African rand — often carry double-digit interest rates, and for years they were funded by cheap dollars and yen. The peso in particular was a star carry currency through 2022–2024, with Mexican rates above 11% offering a juicy spread. EM carry pays even more than developed-market carry, which is exactly the point: the higher yield is compensation for higher crash risk. EM currencies are *more* prone to sudden, brutal devaluations — a political shock, a commodity crash, a sudden-stop in capital inflows — and when global risk appetite turns, EM carry unwinds alongside the yen, often harder. The same "steady drip, sudden cliff" shape, with a steeper cliff.

The peso's own 2024 story is instructive. It had been one of the best-performing currencies in the world, riding an 11% carry and a "nearshoring" narrative, when Mexico's June 2024 election delivered a surprise super-majority for the ruling party. The peso fell roughly 10% in days on the fear of unchecked constitutional changes — and then the August yen unwind hit it again, because the same leveraged funds long the peso were funded in the same yen they now had to buy back. Two unrelated shocks, one positioning problem: that is the EM carry trap. The yield looks like free money right up until a local political surprise and a global deleveraging arrive in the same quarter, and the currency that paid you 11% a year gives back two years of it in a fortnight. The lesson generalizes: **the higher the carry, the fatter and closer the tail — EM carry is the developed-market carry trade with the volume turned up on both the income and the crash.**

## How to trade it / The playbook

Everything above earns its keep here. If you remember one thing, remember that **a currency over your trading horizon is a bet on the rate differential and the flows, with the carry trade as the dominant expression and its unwind as the dominant risk.** Here is how to act on that.

**1. Trade the rate-differential *divergence*, not the level.** The big, durable FX trends come from two central banks moving in *opposite* directions — one hiking while the other holds or cuts. That is what drove USD/JPY from 103 to 161. So your primary signal is not "where is the rate gap now" but "is the gap *widening or narrowing*, and is the market correctly pricing the path?" Watch the policy-rate trajectories, the inflation prints that move them (a hot CPI pulls the hiking central bank's path higher), and the forward-rate markets (OIS, fed funds futures) that price expectations. The currency moves on the *change in expected* rates, which is why a single surprising CPI or jobs report can move a pair 2% in an hour. Position *with* a widening divergence; be very careful holding a carry trade into a *narrowing* one.

**2. Use covered interest parity to read forwards correctly.** When you see deeply negative dollar forward points or "the yen is cheap to hold forward," do not read it as a forecast — read it as the rate gap. The forward is fair given rates. The *opportunity* is in the gap between the forward-implied move and the move that actually happens (the UIP failure), which is the carry premium you are harvesting. Knowing this keeps you from the rookie error of "the forward says the dollar falls, so I'll short it" — the forward is not predicting anything.

**3. Size carry for the crash, not the carry.** The cardinal sin of carry is sizing the position off the steady income — "it pays 5%, and it's been so calm" — and ignoring the tail. Size it off the *unwind*. Ask: if this pair moves 10% against me in a week (it can, and it does), what is my loss, and can I survive it? Our worked example showed a 9% move erasing two years of carry; at 10× leverage it erases your capital. The correct carry size is one where a sudden 10–15% adverse move is a survivable drawdown, not a margin call. Most blown-up carry traders sized off the green part of the curve and never priced the red cliff.

**4. Watch carry *crowding* and volatility as the unwind tell.** The unwind does not come from nowhere — it comes from a crowded trade meeting a shock. So your early-warning dashboard is: (a) **positioning** — CFTC Commitment of Traders data showing record one-sided short-yen positioning, sell-side estimates of the carry book size, and "everyone is in this trade" sentiment; (b) **the funding central bank turning** — any sign the Bank of Japan (or whoever funds the trade) is about to hike, which closes the gap from the cheap end; (c) **the target central bank turning dovish** — soft data pulling the high-yielder's rate path down; and (d) **volatility itself** — FX implied vol (and broad risk vol like the VIX) ticking up off the floor is the canary. Carry thrives in low-vol, calm regimes and dies in vol spikes; **a rising vol regime is the single best signal to cut carry**, because the same shock that lifts vol forces the deleveraging that drives the unwind.

**5. The invalidation: when the gap starts to close, the trade is over.** The clean exit signal for a carry position is the *rate differential beginning to narrow* — the funding bank hiking, the target bank cutting, or the market repricing either path. That is the structural turn, and it front-runs the cascade. In the yen case, the tell was the Bank of Japan exiting negative rates in March 2024 and hiking in July, combined with a softening US data path. If you were still maximally long the yen carry into August 2024 after both of those, the cascade found you. **Do not wait for the move to confirm the unwind — the move *is* the unwind, and it is too fast to exit into.** The discipline is to cut as the *gap* turns, before the price does.

**6. Trade the unwind itself, carefully.** For the more aggressive: a carry *unwind* is a tradable event in its own right. When a crowded carry meets a clear catalyst (a funding-bank hike into record positioning), the highest-conviction trade can be to *fade the carry* — go long the funding currency (long yen), expecting the cascade. But this is timing a crowded reversal, which is treacherous: the carry can stay calm and bleed you on negative carry for months before the unwind arrives (you are now paying the spread, not collecting it). Size small, define a clear catalyst window, and respect that being early on a carry unwind is indistinguishable from being wrong until it suddenly isn't.

**7. The concrete prints to put on your calendar.** A currency view is only as good as the events that confirm or break it, so here is the watchlist that actually moves FX, roughly in order of punch. *Central-bank meetings* (FOMC, Bank of Japan, ECB) and especially the *dot plots, statements, and press conferences* — these reprice the entire rate path in seconds, and the move is in the *surprise* relative to what was priced, not the decision itself. *Inflation prints* (US CPI above all) — a hot CPI pulls the hiking bank's path higher and is the single most reliable intraday FX mover; the August 2024 unwind was lit by the jobs report's implication for the *Fed's* path, not by anything in Japan. *Labor data* (US non-farm payrolls) — same logic, it moves the rate path. *Forward-rate markets* — fed funds futures and OIS are where the rate path is priced; read them to know what is already in the price, so you can position for the *re-pricing* rather than the level. And the *positioning prints* — CFTC Commitment of Traders (weekly), which shows you when speculative short-yen positioning has gone to a record one-sided extreme, the precondition for an unwind. Put those on your calendar and you are watching the actual drivers, not the trade-balance release that the news leads with and the market ignores.

The deepest lesson is the one the August 2024 morning taught with a 12% one-day equity crash off a 0.25% rate hike: **in FX, the rate differential is the engine, the carry trade is the gearbox, and crowding plus a vol spike is the thing that strips the gears.** Read the rate paths, respect the crowding, watch the vol, and size for the cliff — and you will be on the right side of the moves that actually happen, instead of the right side of a PPP model that takes a decade to be proven correct. The trade balance will still be in the headlines. Let it be. Your edge is knowing that the money is in the rate gap and the flows, that the carry trade is how that edge is expressed, and that the day it stops working is the day the gap starts to close and the crowd starts to run — so you watch the gap and the crowd, not the burgers and the trade deficit.

## Further reading & cross-links

- [The dollar system: why the USD rules markets and what the DXY tells you](/blog/trading/macro-trading/dollar-system-why-usd-rules-markets-dxy) — the dollar is the other side of most FX pairs and the funding currency of the global financial system; this post is the macro backdrop to everything here.
- [How the Fed sets interest rates](/blog/trading/finance/how-the-fed-sets-interest-rates) — the rate differential that drives FX starts with the Fed; understand the mechanics of how the policy rate is actually set.
- [Interest rates: the price of money and the master variable](/blog/trading/macro-trading/interest-rates-the-price-of-money-master-variable) — the deeper treatment of why interest rates sit upstream of nearly every asset price, currencies included.
- [Vietnam's monetary policy: the State Bank, the dong, and the credit ceiling](/blog/trading/finance/vietnam-monetary-policy-state-bank-dong-credit-ceiling) — a concrete look at how an emerging-market central bank manages its currency and rates, the EM-carry side of this story.

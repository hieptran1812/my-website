---
title: "Uncovered Interest Parity and Why It Fails: The Forward Puzzle"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A from-scratch tour of uncovered interest parity, the forward as a forecast, the forward-premium puzzle, and why UIP's failure is the entire reason the carry trade exists."
tags: ["forex", "currencies", "uncovered-interest-parity", "carry-trade", "forward-premium-puzzle", "interest-rate-differentials", "risk-premium", "fx-forecasting", "macro"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Uncovered interest parity (UIP) says a currency that pays a higher interest rate should weaken by exactly that much, so two countries' bonds give you the same return once the currency move is counted; in the real world, high-yield currencies tend to *hold or rise* instead of falling, and that broken prediction is the entire reason the carry trade makes money.
>
> - UIP is built by taking covered interest parity (the airtight, arbitrage-locked relationship between spot, forward, and the rate gap) and **dropping the hedge** — you replace the known forward price with a *guess* about the future spot rate.
> - The "guess" UIP plugs in is the forward rate itself. So UIP is really two claims welded together: the forward is set by the rate gap (true, by arbitrage), and the forward is the market's unbiased forecast of spot (false, in the data).
> - The **forward-premium puzzle**: across decades and dozens of currencies, the high-yield currency on average appreciates a bit instead of depreciating. The regression slope that should be \+1 under UIP comes out negative.
> - That gap between "what UIP predicts" and "what happens" is a harvestable return — the **carry trade** — and it pays a positive average return precisely *because* it is compensation for a risk that occasionally bites hard (the crash).
> - The one number to remember: UIP predicts a regression coefficient of **\+1**; the empirical estimate is roughly **\−0.5 to \−1**. The sign is wrong, not just the size.

## When the textbook and the tape disagree

On a trading desk in early 2024, two screens told two stories. One screen ran the textbook. It said: the United States pays you about 5% to hold a one-year dollar deposit; Japan pays you about 0.1% to hold a one-year yen deposit. That is a 4.9-percentage-point gap, the widest in a generation. The textbook screen — running the logic of uncovered interest parity — drew the obvious conclusion: the yen must be *expected* to strengthen by roughly 4.9% over the year, because otherwise lending in dollars instead of yen would be free money, and free money cannot survive in a market this large. So the textbook quietly forecast the yen up.

The other screen ran the tape. And the tape said the opposite was happening in slow motion: the yen had been *falling* against the dollar for two straight years while that rate gap widened, from about 109 yen per dollar at the end of 2019 to 157 by the end of 2024. The currency that paid almost nothing was getting weaker, not stronger. The currency that paid 5% was the one going up. Every month the gap stayed open, a trader who had *ignored* the textbook and simply borrowed cheap yen to hold dollars had banked the 4.9% and watched the exchange rate move in their favour on top of it.

This is not a quirk of one pair in one year. It is the single most studied anomaly in international finance, and it has a name with two halves. The economists call it the **failure of uncovered interest parity** or, in its sharpest empirical form, the **forward-premium puzzle**. Traders call it, simply, **carry** — and they have built a multi-trillion-dollar industry on the fact that the textbook is wrong. This post is about why the textbook says what it says, exactly how and why reality disagrees, and why that disagreement is not a bug in the market but the very thing that pays the people willing to bear its risk.

![UIP prediction versus what currencies actually do, a before and after comparison](/imgs/blogs/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle-1.png)

Remember the spine of this whole series: an exchange rate is the *relative* price of two monies, and you never hold a currency alone — every position is a spread, a bet on one country's money against another's. UIP is the cleanest possible statement of what that spread *should* pay. The puzzle is what happens when the market refuses to obey.

## Foundations: From covered to uncovered parity

Before we can say why UIP fails, we have to build it carefully, because most of the confusion about the puzzle comes from people skipping the step where UIP is born. UIP is not a fundamental law of nature. It is a *hypothesis* that you reach by taking a law that genuinely is airtight and then deliberately removing the thing that made it airtight. Let us do that slowly.

### A 30-second vocabulary so the equations read cleanly

Two pieces of FX grammar make every formula below readable. First, an exchange rate is always quoted as a **pair** with a **base** currency and a **quote** currency: "EUR/USD at \$1.0800" means one unit of the *base* (the euro) costs 1.0800 units of the *quote* (the dollar). The number is "quote per one base." When the number goes up, the base got stronger and the quote got weaker — EUR/USD rising from \$1.0800 to \$1.1000 means the euro strengthened and the dollar weakened. Keep that straight and the parity equations stop being confusing: the left side is always "future or expected pair level" over "today's pair level."

Second, a **pip** is the smallest standard increment a pair is quoted in — for most pairs the fourth decimal place (0.0001), and for yen pairs the second decimal (0.01). When USD/JPY moves from 150.00 to 150.50 it has moved 50 pips. Pips are just the unit traders count moves in; nothing in the parity logic depends on them, but you will see them whenever we talk about how far a currency actually travelled. With base, quote, and pip in hand, every number below is unambiguous.

### Two ways to earn interest on cash

Suppose you have one dollar and a one-year horizon, and you want to earn interest. There are two roads.

**Road one — stay home.** Put the dollar in a one-year US deposit paying an interest rate we will call `r_us`. A year later you have `1 + r_us` dollars. Done. No currency was involved.

**Road two — go abroad.** Convert the dollar into a foreign currency today at the **spot rate** (the price for immediate exchange), put the foreign cash into a foreign one-year deposit paying `r_foreign`, and a year later convert the proceeds back into dollars. You earn the foreign interest rate, but your final dollar amount depends on what the exchange rate does in between. If the foreign currency strengthens, you gain twice; if it weakens, the currency loss eats into the interest you earned.

The entire subject of interest parity is a comparison of these two roads. The only question is: *what makes them give the same answer?* And the answer depends entirely on one choice — whether you lock in the exchange rate for the return trip in advance, or leave it to fate.

### Covered interest parity: the airtight version

If on day one you also sign a **forward contract** — an agreement today to exchange currencies back at a fixed price one year from now, a price called the **forward rate** `F` — then road two has *no* uncertainty left. You know your foreign interest, and you know the rate you will convert back at. Both roads are now risk-free, so by the iron logic of arbitrage they must pay exactly the same. If they did not, you could borrow on the cheap road, lend on the rich road, and pocket the difference with zero risk and zero capital. Markets erase such opportunities in milliseconds.

That equality is **covered interest parity** (CIP) — "covered" because the forward contract *covers* (hedges) the currency risk. Written in the convention where the exchange rate is "quote currency per one unit of base currency," it says:

```
F / S = (1 + r_quote) / (1 + r_base)
```

In words: the forward rate `F` differs from the spot rate `S` by exactly the ratio of the two interest rates. The currency with the *higher* interest rate trades at a forward **discount** (it is expected to buy less in the future), and the currency with the *lower* rate trades at a forward **premium**. This is not a forecast — it is mechanical, enforced by arbitrage, and it holds to within a few basis points in normal markets. (When it *does* break, as in a dollar-funding crunch, that gap — the cross-currency basis — is itself a tradeable stress signal, but that is a different post.) CIP is the bedrock; everything else in this article is built by chipping away at it.

#### Worked example: the covered, risk-free round trip

Take real, internally consistent numbers. Spot EUR/USD is \$1.0800 (one euro costs \$1.0800). The one-year dollar rate is 5.0%; the one-year euro rate is 3.0%. You have \$1,080,000 and you want a year of euro interest without currency risk.

Convert to euros at spot: \$1,080,000 ÷ \$1.0800 = €1,000,000. Earn 3.0% euro interest: €1,000,000 × 1.03 = €1,030,000 a year out. To remove all currency risk, you locked a forward today. CIP says that forward must be `F = 1.0800 × (1.05 / 1.03) = \$1.1010` per euro — the euro trades at a forward *premium* because the dollar carries the higher rate. Convert back at the locked forward: €1,030,000 × \$1.1010 = \$1,134,030.

Now check road one: \$1,080,000 × 1.05 = \$1,134,000. The two roads give \$1,134,030 versus \$1,134,000 — identical to rounding. **That equality is not luck; it is what CIP guarantees, and the tiny residual is just rounding in the forward.** The forward premium on the euro exactly cancels its lower interest rate.

### Uncovered interest parity: drop the hedge, keep the hope

Now perform the surgery that creates UIP. Take CIP and remove the forward contract. You no longer lock the return rate; you let the future spot rate be whatever it turns out to be. Road two is now risky again — your final dollars depend on an exchange rate you cannot know in advance.

For the two roads to *still* offer the same return, something has to replace the locked-in forward. UIP supplies it with a single bold assumption: that the exchange rate you *expect* a year from now, call it `E[S₁]`, behaves just like the forward did. Substitute the *expected* future spot for the *contracted* forward, and CIP becomes:

```
E[S_1] / S = (1 + r_quote) / (1 + r_base)
```

Read that aloud and you have the whole of UIP: **the expected change in the exchange rate just offsets the interest-rate gap.** The high-yield currency is *expected* to depreciate by exactly enough to wipe out its yield advantage. So an investor is indifferent between the two roads — not because risk is hedged, but because the expected currency move is assumed to compensate for the rate gap precisely. UIP is CIP with the certainty ripped out and replaced by a forecast.

![From covered to uncovered parity, the forward as the market's forecast](/imgs/blogs/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle-3.png)

### The unbiasedness hypothesis: the forward as a forecast

There is a slicker way to state UIP that exposes its weak point. Stack UIP next to CIP. CIP says `F / S = (1 + r_quote)/(1 + r_base)`. UIP says `E[S₁] / S = (1 + r_quote)/(1 + r_base)`. The right-hand sides are identical. Therefore:

```
F = E[S_1]
```

The forward rate equals the expected future spot rate. This is the **unbiasedness hypothesis** (sometimes the "forward rate unbiasedness" condition): the forward price the market quotes today is supposed to be the market's best, unbiased guess of where spot will actually be. If that were true, the forward would be a *forecast* — and a free one, printed on every dealer's screen.

This is the seam where everything tears. CIP is arbitrage; it is true. The unbiasedness step is a *behavioural* claim about how the market forms expectations and prices risk, and it is the part the data demolishes. The forward is built by arbitrage from the rate gap — that much is solid — but the leap from "the forward is arbitrage-free" to "the forward is the right forecast" is exactly the leap that fails. Hold onto that distinction; it is the spine of the rest of this post, and it is why we never re-derive the rate mechanics here — the curve and the forward-rate machinery live in the [forward rates explainer](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be); we are only asking whether that arbitrage-fair forward is a *good forecast*.

#### Worked example: what UIP would forecast for USD/JPY

Plug the 2024 numbers into UIP and see the forecast it spits out. Spot USD/JPY is 150.00 (150 yen per dollar). The one-year dollar rate is 5.0%; the one-year yen rate is 0.5%. Here the dollar is the base, the yen the quote, so UIP says:

`E[S₁] = 150.00 × (1 + 0.005) / (1 + 0.05) = 150.00 × 1.005 / 1.05 = 143.57` yen per dollar.

UIP forecasts that a year out, the dollar buys only 143.6 yen instead of 150 — the dollar *weakens* about 4.3%, the yen *strengthens* about 4.3%, almost exactly cancelling the 4.5-point interest gap you earned by holding dollars. On \$1,000,000 held in dollar deposits, you would earn \$50,000 of interest and, if UIP held, lose about \$43,000 to a stronger yen on the conversion, netting roughly the same as if you had held yen all along. **UIP says the rate gap is an illusion: the FX move is supposed to give back almost everything the interest paid you.** The next sections are about how badly that prediction missed.

## UIP versus CIP: same equation, opposite trustworthiness

It is worth dwelling on why two equations that look like algebraic twins behave so differently in practice, because conflating them is the single most common mistake beginners make. They share a right-hand side, the rate ratio. They differ only in the left-hand side: CIP uses the *contracted* forward `F`, a price you can lock today; UIP uses the *expected* future spot `E[S₁]`, a number nobody can lock and nobody can even observe until the year is over.

That single substitution changes the *kind* of statement you are making. CIP is a no-arbitrage condition: it must hold or money is left on the table, and the enforcement is automatic and instantaneous. UIP is an equilibrium condition that depends on two fragile assumptions — that investors are risk-neutral (they do not demand extra compensation for bearing currency risk) and that they form expectations rationally (their average guess is correct). Strip either assumption and UIP can fail while CIP stays perfectly intact. That is exactly what the data shows: CIP holds to a few basis points, UIP fails by enough to build a trade on.

Here is the cleanest way to feel the difference. The forward is *knowable and tradeable* — you can sell the euro one year forward at \$1.1010 right now and that price is real. The expected future spot is *unknowable and untradeable* — it is a belief about 2025 that lives only in aggregate, and you cannot buy or sell it. CIP relates two real, contemporaneous prices. UIP relates a real price to a hidden expectation. Markets are extraordinarily good at policing relationships between real prices and quite bad at making the unobservable expectation come true. So:

- **CIP is a fact you can verify on a screen.** It is a constraint on prices.
- **UIP is a hypothesis about the future you can only test after the fact** — by collecting decades of realized exchange-rate moves and asking whether they, on average, matched what the rate gap predicted. They did not.

#### Worked example: the two roads when you do NOT hedge

Return to the EUR/USD trade, but this time refuse to lock the forward. You convert \$1,080,000 to €1,000,000 at spot \$1.0800, earn 3% euro interest to reach €1,030,000, and a year later convert back at *whatever spot turns out to be*. UIP forecast that spot would slide to about \$1.0590 per euro (the euro weakening to give back its rate disadvantage). If that forecast came true, you would get €1,030,000 × \$1.0590 = \$1,090,770 — *less* than the \$1,134,000 you would have made just staying in dollars, exactly as UIP claims, because the euro's depreciation more than ate its interest.

But suppose the euro instead *held* at \$1.0800 (UIP's prediction simply did not happen). Then you collect €1,030,000 × \$1.0800 = \$1,112,400 — and your euro deposit, plus a currency that refused to fall, beat the textbook by over \$21,000 on a million-dollar position. **The whole carry edge lives in that "refused to fall": UIP says the currency must drop, and historically it usually does not drop by nearly enough.** Across hundreds of such bets, that shortfall compounds into a real, repeatable return.

## The forward-premium puzzle: when the sign comes out wrong

Now we make the failure precise, because vague hand-waving about "UIP not holding" misses how *strange* the failure is. It is not that UIP is a little off, predicting the right direction but the wrong magnitude. It is that, on average, UIP gets the **direction backwards**.

### The regression that should slope up at 45 degrees

Economists test UIP with a simple regression. Line up many periods. On the horizontal axis put the interest-rate gap (equivalently, the forward premium — they are the same thing by CIP). On the vertical axis put the *realized* change in the exchange rate over the following period. UIP makes a razor-sharp prediction about the line through that cloud of points: its slope should be exactly **\+1**. A one-point-bigger rate gap should produce, on average, a one-point-bigger depreciation of the high-yield currency. The intercept should be zero, the slope should be one — a clean 45-degree line.

When researchers actually run this regression — Eugene Fama did the canonical version in 1984, and the result has been replicated across dozens of currencies and four decades since — the slope does not come out at \+1. It comes out *negative*, typically somewhere around **\−0.5 to \−1**. A negative slope means that when a currency pays a higher rate, it tends to *appreciate* afterward, not depreciate. The high-yield currency does the opposite of what UIP commands. The textbook does not merely overstate the depreciation; it points the arrow the wrong way.

This is **the forward-premium puzzle**: the forward rate, far from being an unbiased forecast of the future spot, is a *biased* one, and it is biased in the most profitable possible direction for anyone willing to take the other side. The forward over-predicts depreciation of the high-yield currency so systematically that betting *against* the forward has paid a positive average return for as long as we have data.

### Reading the regression slope like a trader

It is worth slowing down on what the slope coefficient actually *means*, because it is the single most quoted number in this literature and most people parrot it without feeling it. Write the test as: the realized depreciation of the high-yield currency equals some intercept, plus a slope times the rate gap, plus noise. UIP nails three things at once: the intercept is zero (no free average return), the slope is exactly one (a bigger gap produces a proportionally bigger depreciation), and the noise has no exploitable structure.

A slope of **\+1** would mean the forward is a perfect, unbiased forecast — the rate gap predicts the FX move one-for-one, and carry earns nothing on average. A slope of **0** would already be a scandal: it would mean the rate gap tells you *nothing* about the future FX move, so the high-yielder neither tends to fall nor rise, and you simply pocket the entire rate gap as carry with no offset at all. The actual estimate of roughly **\−0.5 to \−1** is stranger still: a *negative* slope says the high-yield currency tends to *appreciate* afterward. You collect the rate gap *and* a currency tailwind, on average. The forward did not just fail to forecast the move — it forecast the move with the wrong sign.

For a trader, the slope is a recipe. A slope of \+1 says "trade the forward, it is right." A slope near 0 says "ignore the forward's FX forecast and just harvest the gap." A negative slope says "fade the forward — bet the high-yielder does the opposite of what the forward implies." Forty years of data say we live in the third world, which is why systematic carry is long the high-yielder and short the funder, full stop.

#### Worked example: turning the slope into a position size

Suppose the estimated slope on a developed-market pair is \−0.5 and the current one-year rate gap is 4.0 percentage points (the high-yielder pays 4 points more). UIP's forecast for the high-yielder's depreciation is \−4.0% (it should fall 4%). The regression's forecast is the slope times the gap: \−0.5 × 4.0% = \−2.0% — but with the sign convention of the test, a negative slope flips UIP's predicted depreciation into a predicted *appreciation* of roughly \+2.0%.

So on a \$1,000,000 carry position into the high-yielder, the *naive UIP trader* expects to net about zero: \+\$40,000 of carry minus ~\$40,000 of expected currency loss. The *regression-aware carry trader* expects \+\$40,000 of carry *plus* roughly \+\$20,000 of expected currency gain, for about \+\$60,000 — and sizes the position to that expected edge against the pair's volatility. **The slope is not academic trivia; it is the difference between expecting zero and expecting a six-figure edge on a million-dollar bet, and the data sides with the carry trader.**

### The picture that started a trillion-dollar trade

The 2019-to-2025 yen story is the puzzle in one chart, so let us read it carefully. The US-minus-Japan two-year rate gap blew out from roughly 1.5 points in 2019 to around 4.3 points in 2022 and stayed wide. UIP's instruction was unambiguous: with the dollar paying so much more, the dollar should be *expected to fall* against the yen — the yen should strengthen to offset. Instead the yen collapsed, from 109 per dollar to 157. The currency UIP said should rise fell by nearly a third while the rate gap was at its widest.

![USD JPY versus the US Japan two year rate gap from 2019 to 2025](/imgs/blogs/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle-6.png)

A trader who, in 2021, did the textbook thing — sold dollars and bought yen because UIP said the yen was cheap relative to its rate disadvantage — got run over for three years. A trader who did the *opposite* — borrowed yen at almost nothing, held dollars at 5%, and ignored the textbook entirely — earned the rate gap *and* a 40-plus-percent currency gain. That is the forward-premium puzzle paying out in real time. The rate gap that UIP says is "free money that cannot exist" turns out to be money the market hands you for bearing a particular risk, which we will get to.

#### Worked example: the yen carry that ignored UIP

Take \$1,000,000 at the end of 2021, when USD/JPY was about 115. You borrow the equivalent in yen — ¥115,000,000 — at roughly 0.1%, and you hold it in dollars earning, by 2023, about 5%. Ignore compounding niceties and run two years.

Interest leg: holding \$1,000,000 at ~5% for roughly two years earns on the order of \$100,000, while the yen funding cost is a few thousand dollars — call the net carry about \$95,000. UIP said this \$95,000 would be erased by the yen strengthening. The exchange leg: USD/JPY went from 115 to about 157, a roughly 36% move *in the dollar's favour* for someone long dollars against yen. On the ¥115,000,000 you borrowed, you now need far fewer dollars to repay it — that currency move alone is worth several hundred thousand dollars of gain on a million-dollar notional.

So instead of UIP's promised wash, the trade earned the carry *and* a large currency gain — easily a 30-to-40% total return over two years on the notional, before leverage. **UIP predicted zero excess return; the trade delivered a fortune, because the yen did the precise opposite of what parity demanded.** This is not hindsight cherry-picking; it is the median experience of high-yield-versus-low-yield FX over decades, which is why carry is a recognized, persistent return factor.

## Why UIP fails: the risk premium and the peso problem

A puzzle this large and this durable demands an explanation that is not "markets are dumb." Smart money has been trying to arbitrage this for forty years and it is *still here*. The leading explanations all say the same structural thing: the rate gap is not free money, so UIP's premise — that the two roads must pay the same — was wrong from the start. The gap is *compensation*. The two roads do not have to pay the same because one of them carries a risk the other does not. The question is: what risk?

### Decomposition: the rate gap is three things, not one

The clean way to see it is to decompose the interest-rate gap into its parts. UIP assumes the gap is entirely "expected depreciation" — that the whole rate advantage is given back by an FX move. But the realized data forces us to split the gap into three components:

![Decomposing the rate gap into expected move, risk premium, and crash term](/imgs/blogs/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle-5.png)

1. **Expected depreciation** — the piece UIP says is everything. There *is* some genuine expected currency move; it just is not the whole gap, and historically it is far too small to eat the gap.
2. **A risk premium** — extra return that high-yield currencies pay because holding them is *risky*, and risk-averse investors must be bribed to do it. This is the piece UIP's risk-neutral assumption wrongly sets to zero. If you demand to be paid for bearing currency risk, the high-yield currency must offer a return *above* what its expected depreciation costs you — and that surplus is the carry profit.
3. **A peso / crash term** — a rare-disaster premium. The sample we measure over may simply not contain enough of the catastrophic events that would, eventually, justify the textbook. More on this below.

UIP fails because it pretends only component one exists. The data says components two and three are large — large enough that, on average, the high-yield currency does not fall at all and the holder pockets the rate gap as a risk premium.

### The risk-premium story: you are an insurer

The cleanest economic explanation is that the carry trader is selling insurance. High-yield currencies — the Mexican peso, the Brazilian real, the Australian dollar, the South African rand — are *pro-cyclical*. They are strong when the global economy is booming and risk appetite is high, and they crash hardest exactly when the world panics and everyone wants safe assets. Low-yield funding currencies — the yen, the Swiss franc — do the reverse: they *strengthen* in a panic, because they are where frightened money flees.

So when you put on a carry trade — long the high-yielder, short the low-yielder — you are taking a position that pays a steady drip in calm times and blows up in a crisis. You are, functionally, *selling crash insurance* on the global economy. And selling insurance pays a premium precisely because the buyer wants protection. The rate gap is your premium. UIP fails because it ignored that you are being paid to bear a real, correlated, occasionally devastating risk. The currency does not need to depreciate to make UIP hold; the *risk* is what closes the books.

This reframes the puzzle entirely. There is no free lunch and never was. The carry return is a fair price for a nasty risk — it just happens to look like free money for years at a time between the disasters. For the mechanics of how that risk appetite drives the whole pair — rates, flows, and carry together — see [what moves exchange rates](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry).

There is a second, deeper consequence of the risk-premium view: the premium is *not constant*. It swells when the world is anxious and shrinks when it is calm, which is exactly backwards from what naive carry traders assume. In quiet, low-volatility regimes the premium is thin — everyone has piled in, the rate gap is fully arbitraged down to the bare risk compensation, and the trade feels safe right up until it is not. In stressed regimes, after a crash has flushed the leverage out, the premium is fat — few are willing to hold the high-yielder, so those who do are paid handsomely. The cruel irony is that the premium is largest precisely when it is hardest to collect, because the crash that made it large has just destroyed the capital that would have collected it. A time-varying risk premium of this shape is the single most consistent academic explanation for why UIP's slope is not just wrong but *unstable* across sub-periods — the relationship between rates and FX moves is a regime, not a constant.

The funding currencies deserve their own line, because their behaviour is the mirror image of the high-yielders. The yen and the Swiss franc do not pay you to hold them — they pay you to *borrow* them, in effect, because they appreciate in panics. They are the safe-haven leg, the place money runs to when the carry trade implodes. That safe-haven property is *why* they can fund carry at almost no rate: investors accept a near-zero yield in exchange for a currency that rallies when everything else is falling. So the funder's low rate and the high-yielder's high rate are two ends of the same risk axis — one is insurance you buy, the other is insurance you sell — and UIP, by assuming risk-neutrality, erased the entire axis. The dollar sits awkwardly in the middle as both a high-ish yielder in the 2022–2024 cycle *and* the ultimate safe haven, which is why its behaviour follows the "dollar smile" rather than a clean carry logic; that cross-asset role is the subject of [the dollar as cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity).

### The peso problem: the disaster that has not happened yet

The "peso problem" is the subtlest explanation and it is named after a real episode. For years in the 1970s the Mexican peso was pegged to the dollar but paid a much higher interest rate than the dollar. UIP said: if the peso pays more, it must be expected to devalue. For years it did not — anyone collecting the peso's high rate looked like they were earning free money, and the regression over that window showed UIP failing. Then in 1976 the peg broke and the peso devalued massively in one shot, wiping out years of accumulated carry gains in days. The high rate *had* been compensation for a devaluation risk all along; the sample just had not contained the devaluation yet.

The general lesson: when a rare, large, one-directional event lurks in the distribution but has not shown up in your sample, *every* statistic you compute is biased. The carry trade can look like it earns a steady premium for decades and still be, in truth, fairly priced — you are just being paid in advance for a crash that is coming but has not arrived in your data window. This is why the carry index does not look like a smooth uptrend; it grinds up and then gives a chunk back in violent, clustered crashes. The smoothness in between is the illusion; the crashes are the bill.

![The carry trade total return index from 2007 to 2025 with crash years marked](/imgs/blogs/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle-2.png)

Look at the shape: a long up-grind from 2007, punctured by the 2008 global financial crisis (the index fell from 100 to 72), the 2011 eurozone scare, the 2015 franc shock, and the August-2024 yen unwind. The average return is positive. The *path* is brutal. That signature — steady gains, rare cliffs — is exactly what a risk premium for crash insurance looks like, and it is the fingerprint of UIP's failure. The deeper anatomy of those cliffs lives in [carry-trade unwinds](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks).

#### Worked example: the steamroller, in dollars

Run the August-2024 unwind on a leveraged carry book. Say you ran \$1,000,000 of equity at five-times leverage, so \$5,000,000 of notional long high-yielders funded in yen, earning a blended carry of, say, 4% — about \$200,000 a year of income on your \$1,000,000 of equity, a tidy 20% before any FX move. Looks magnificent. Then in early August 2024 the BoJ hiked, the yen rocketed from about 162 to 142 per dollar in days — a roughly 12% move against the trade.

A 12% adverse FX move on \$5,000,000 of notional is a \$600,000 loss. Against \$1,000,000 of equity, that is a 60% drawdown in under a week — and it wiped out three years of that 20%-a-year carry in five trading days. **The carry "income" of \$200,000 a year is real, but it is the premium on an insurance policy whose claim, when it comes, dwarfs years of premiums.** That asymmetry is not a flaw in the strategy; it *is* the strategy, and it is the precise mechanism by which UIP's "missing" depreciation eventually shows up — all at once, in a crash.

### Does UIP fail at every horizon? The timescale caveat

An honest treatment has to add a caveat that mature FX traders know well: UIP fails most spectacularly at the *short* horizons most people test it over — months to a couple of years — and it appears to partially *re-assert itself* over very long horizons of five to ten years and beyond. Over a decade, currencies do tend to drift toward their purchasing-power-parity (PPP) fair values, and a chronically high-yielding, high-inflation currency does eventually depreciate roughly in line with its inflation gap. The yen, which sat far below its PPP fair value for years while carry traders shorted it, is the textbook example: the *level* was unsustainable on a fundamental basis even as the *carry* kept paying.

So the precise statement is: at the one-month-to-two-year horizon where carry trades actually live, UIP's predicted depreciation does not show up and the rate gap accrues as profit. At the multi-year horizon, valuation gravity (PPP) slowly pulls currencies back, and the high-yielder does eventually give some of it back. The two are not contradictory — they are different forces operating on different clocks. Carry is a short-to-medium-horizon harvest of a risk premium; PPP is a decade-scale anchor. A trader who confuses the two — who shorts a high-yielder today because it is "overvalued on PPP" — typically gets ground down by carry for years before the valuation correction arrives, if it ever does on their timeframe. The valuation factor and the carry factor are *both* in the FX factor zoo precisely because they pay off on different horizons and even hedge each other somewhat. The point for UIP: its failure is a statement about the tradeable horizon, not about eternity, and that is exactly the horizon where money is made and lost.

## Carry: harvesting the failure on purpose

If UIP failed at random — sometimes the high-yielder rose, sometimes it fell, no pattern — there would be no trade, just noise. The reason carry is an *industry* is that UIP fails in a *consistent direction*: high-yield currencies under-depreciate on average, across currencies and across decades. A consistent, repeatable bias is, by definition, a harvestable return. The carry trade is simply the machine built to harvest it.

![How the carry trade harvests the failure of uncovered interest parity](/imgs/blogs/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle-7.png)

The recipe is mechanical: **borrow the low-yield currency, hold the high-yield currency, collect the gap.** Because UIP fails, the gap is not given back by the exchange rate on average — so it accrues as profit. You are not forecasting anything. You are not picking tops or bottoms. You are systematically betting that the forward is wrong in the way it is *always* wrong, and collecting the spread between the forward's prediction and reality. The full anatomy of the trade — sizing, funding, which currencies, the carry-to-vol ratio — is its own deep dive in [the carry trade](/blog/trading/forex/the-carry-trade-getting-paid-to-hold-a-currency); here we only need the link to UIP: carry is *long the UIP failure*.

### Why arbitrage has not killed it

The natural objection: if this is so reliable, why has the army of hedge funds and bank desks not arbitraged it to zero? Three reasons, all of which trace back to the risk-premium explanation.

First, it is *not* arbitrage — it is a risk premium, and risk premia do not get arbitraged away because bearing the risk is the whole job. You cannot make the crash risk disappear by trading harder; you can only choose whether to be paid for bearing it. Second, the crashes deter capital. A strategy that loses 60% in a week is one most investors cannot stomach and most leveraged players are forced out of at the worst moment — which is exactly why the premium survives. Third, the crowding *is* the risk: when too much capital piles into carry, the unwind is more violent, which raises the very crash risk the premium compensates. The trade self-regulates through its own blow-ups. The companion post on [carry crashes](/blog/trading/forex/carry-crashes-picking-up-pennies-in-front-of-a-steamroller) is entirely about this dynamic.

There is also a "limits to arbitrage" wrinkle worth naming, because it explains why even the players who *know* UIP fails cannot fully correct it. To lean against the puzzle in the size required, you need cheap leverage and a long, patient horizon — and the players with the deepest pockets (banks, pensions) are also the most regulated and the most allergic to the kind of fat-tailed drawdown carry delivers. Capital that *could* arbitrage the premium down is structurally reluctant to do so, because a 60% drawdown ends careers and breaches risk limits, regardless of the strategy's long-run Sharpe. The mispricing persists not because the market is stupid but because the people who could correct it are, sensibly, unwilling to bear the precise risk that would require them to. That is the general shape of every durable risk premium: it is the return to doing the uncomfortable thing that most capital is constrained from doing.

A fourth, subtler point: even the academics who have spent careers on this disagree about *how much* of the carry return is risk premium versus peso problem versus slow-moving expectational errors, and that very disagreement is healthy evidence that the premium is real and not a data artifact. If carry were a measurement error, it would have been found and erased; instead, decades of attack have only refined our understanding of *which* risk it pays for. The puzzle has survived every attempt to explain it away precisely because there is a genuine, bearable, occasionally catastrophic risk at its core.

### The rate menu UIP says should not matter

Here is the menu of policy rates that carry traders actually shop from. Under UIP, this menu is *irrelevant* — every currency should offer the same risk-adjusted return because the FX move offsets the rate. The fact that this menu is the *most-watched table on an FX desk* is itself the loudest evidence that UIP fails: practitioners trade off exactly the variable the theory says carries no information.

![Funding versus high yield policy rates across currencies in 2024](/imgs/blogs/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle-4.png)

The funders sit at the bottom — yen at 0.25%, franc at 0.50%. The high-yielders tower above — Mexico at 10.25%, Brazil at 12.25%, Turkey at a startling 50%. UIP insists the Turkish lira *must* be expected to depreciate about 50% a year to offset that rate, and indeed high-inflation currencies like the lira *do* often depreciate a lot, which is the honest caveat: UIP works far better for extreme cases than for moderate G10 gaps. But for the developed-market carry universe — Australia, Norway, the dollar, funded in yen or francs — the depreciation that UIP demands simply does not show up in anything like the required size, and the gap accrues to the carry holder. The master variable behind this whole table is the subject of [interest-rate differentials](/blog/trading/forex/interest-rate-differentials-the-master-variable-of-fx).

### Carry as a factor: the failure has a Sharpe ratio

The final piece of evidence that UIP's failure is real and harvestable is that "carry" stands shoulder to shoulder with the other recognized return *factors* in finance — value, momentum, the equity premium. Strip the carry trade down to a diversified, systematic strategy and it has earned a respectable risk-adjusted return over decades, with a Sharpe ratio in the same neighbourhood as a trend-following or value strategy.

![FX style factor Sharpe ratios with carry highlighted](/imgs/blogs/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle-8.png)

Carry stands out as the headline FX factor — and it *only exists if UIP fails*. If the forward were an unbiased forecast, the carry factor would have a zero expected return and a Sharpe ratio of zero. That it has a positive, persistent Sharpe is the market's verdict on the unbiasedness hypothesis: rejected. The factor is the puzzle, monetized. The whole zoo of FX factors and how they relate sits in the [cross-asset relative-value layer](/blog/trading/cross-asset/fx-currencies-the-relative-value-layer).

It is worth being precise about what that Sharpe ratio does and does not promise. A Sharpe near 0.5 is respectable but unspectacular — it is roughly in the range of a diversified equity-market exposure, not a money machine. And it is computed *including* the crashes, which means the smooth-looking premium between disasters overstates the true reward; the honest, crash-inclusive number is the one above. The Sharpe also understates the pain, because Sharpe measures volatility symmetrically while carry's risk is violently one-sided — the returns are negatively skewed, with many small gains and rare enormous losses. A skew-aware investor demands a *higher* premium for that shape than the Sharpe alone implies, which is yet another reason the premium survives. The single most important thing the factor evidence proves, though, is binary and decisive: the expected excess return is *not zero*, and a not-zero excess return is a flat, empirical rejection of uncovered interest parity. The textbook said the number should be zero. The market has paid it for forty years.

## Common misconceptions

Because UIP is taught as a "law" and then quietly contradicted by every trading desk, the topic accumulates myths. Here are the ones that cause real money to be lost.

**"The forward rate is a forecast, so I can read tomorrow's price off it."** No — and this is the central confusion the whole post exists to kill. The forward is set by *arbitrage from the rate gap*, not by anyone's view of the future. It equals the expected spot only if the unbiasedness hypothesis holds, and it does not. Empirically the forward is a *biased* predictor: it systematically over-predicts depreciation of the high-yield currency. If you must use the forward as a forecast, the historical correction is to *fade* it — the realized move is usually smaller than, and often opposite to, what the forward implies. The forward is a tradeable price, not a crystal ball.

**"UIP fails, so high-yield currencies always go up."** Also wrong, and dangerously so. UIP fails *on average*, over diversified baskets and long horizons. On any single pair over any single period, the high-yielder can and does crater — that is the crash risk you are being paid for. The 2018 Argentine peso lost about half its value; the lira has lost the overwhelming majority of its dollar value over a decade despite paying enormous rates. The correct statement is "high-yield currencies depreciate *less than UIP predicts, on average*," which leaves enormous room for individual disasters. Confusing the average with the guarantee is how leveraged carry books die.

**"Carry is free money — the rate gap with no downside."** The whole risk-premium section is the refutation. The 4% you earn each year is the premium on a policy that pays out a 60% loss in a bad week. Over a long enough horizon, including the crashes, the *risk-adjusted* return is positive but modest, not a free lunch. The pennies are real; so is the steamroller.

**"If UIP fails, CIP must fail too — they are the same equation."** They share a right-hand side but they are *categorically* different. CIP is enforced by riskless arbitrage between observable prices and holds to a few basis points in normal times. UIP is a behavioural hypothesis about expectations and risk preferences. CIP can hold perfectly while UIP fails wildly — and that is precisely the observed state of the world. Never let the algebraic resemblance fool you into thinking they have the same reliability.

**"UIP is useless, so ignore interest rates when trading FX."** The opposite lesson. UIP fails in a *direction*, and that direction is tradeable — the rate gap is the single most important predictor of carry returns. The rate differential is not noise to be ignored; it is the master signal. UIP's failure does not demote interest rates; it *promotes* them from "the thing the FX move will cancel" to "the thing that pays you." The mechanism by which policy rates move currencies is covered in [how monetary policy moves currencies](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials).

## How it shows up in real markets

The theory is clean; the tape is where it bites. Four episodes show UIP's failure — and its occasional, violent vindication — in the wild.

**The yen, 2021 to 2024: the textbook humiliated.** Already our running example, and the cleanest modern case. The Fed hiked to 5.5% while the BoJ held near zero, opening the widest G10 rate gap in decades. UIP screamed that the yen would strengthen to offset. It fell from 115 to 157 instead — a textbook-perfect failure of UIP, and a textbook-perfect carry payout, for three straight years. Anyone trading on the forward as a forecast was short the dollar against the yen and bled the entire time.

**August 2024: the steamroller arrives.** Then, in a few days at the start of August 2024, the BoJ hiked, the carry trade everyone loved unwound, and USD/JPY collapsed from about 162 to 142 — a roughly 12% move that spiked the VIX to an intraday 65.7. This is the *other* half of the truth: UIP's "missing" depreciation does not vanish, it *clusters*. Three years of carry gains were partly handed back in a week. The peso problem made flesh: the premium was real, and so was the claim when it finally came due. The blow-by-blow of leverage breaking in unwinds is in [carry-trade unwinds](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks).

**The 1992 sterling crisis: when high rates could not hold the line.** On Black Wednesday, 16 September 1992, the Bank of England jacked its base rate from 10% to 12% to 15% in a single day, trying to make sterling so high-yielding that nobody would dare sell it — a brute-force appeal to UIP logic: *we will pay you so much to hold pounds that the rate gap must defend the currency*. It failed within hours. The pound was forced out of the European Exchange Rate Mechanism and fell about 15% by year-end. The lesson: when the market believes a currency is going to fall, no interest rate is high enough to satisfy UIP, because the *expected* depreciation has gone to infinity. (The speculative-attack game behind this is modelled in [the central-bank game](/blog/trading/game-theory/the-central-bank-game-credibility-commitment-and-dont-fight-the-fed); we keep only the narrative here.)

**The 1998 Russia/LTCM unwind: correlation goes to one.** When Russia defaulted in August 1998, the yen carry trade — short yen, long higher-yielders — unwound so fast that USD/JPY fell from about 147 to 112 in three days that October. Long Term Capital Management, levered into exactly these convergence and carry bets, was destroyed. This is the canonical demonstration that the carry risk premium is *correlated across everything*: in the crash, every high-yielder falls together and every funder rallies together, which is why the premium has to be so large. The diversification you thought you had evaporates exactly when you need it.

**The emerging-market carry-and-sudden-stop cycle.** The puzzle is most violent in emerging markets, where rate gaps are largest and crashes are deepest. Hot money chases the high yields of Mexico, Brazil, Indonesia, or Turkey for years — portfolio inflows ran near a quarter-trillion dollars in 2012 — then reverses in a single panicked quarter when the dollar strengthens or US rates rise. The 2013 "taper tantrum" and the 2022 outflow wave (roughly \−\$80 billion of EM portfolio flows) are the textbook sudden stops: the carry that looked so reliable in the inflow years was, again, an insurance premium, and the stop is the claim. For an EM currency, UIP's failure and its violent vindication can both happen inside a single business cycle, which is why EM carry is sized smaller and watched harder than G10 carry. The country-risk layer on top of the rate gap is covered in [emerging-market and sovereign debt](/blog/trading/fixed-income/emerging-market-and-sovereign-debt-yield-with-country-risk).

Across all four, the same structure repeats: UIP fails quietly for years (high-yielders hold or rise, carry pays), then is violently vindicated for days (the crash hands back the depreciation all at once). The forward was a biased forecast the whole time; the bias was the premium; the premium was insurance; the insurance paid out in the crisis. That is the complete life cycle of the forward-premium puzzle.

## The takeaway: read the rate gap as a price, not a prophecy

Here is the shift in how you read a currency once you internalize UIP and its failure. The interest-rate gap between two countries is the most important single number for the pair — but it does *not* tell you, as the textbook claims, how much the currency will move to cancel it. It tells you something better and more useful: how much the market is *paying you to bear the risk* of holding the high-yielder against the low-yielder.

When you see a 5-point rate gap, do not think "the high-yielder will fall 5% to make the trade a wash" — that is the UIP reflex, and it is wrong on average. Think instead: "the market is offering me 5% a year to sell crash insurance on this pair; am I being paid enough for how bad the crash could be, and is the crowd so heavy into this trade that the unwind will be vicious?" That question — *carry versus crash risk* — is the entire decision. The rate gap is the premium; the volatility, the positioning, and the funding-currency safe-haven status tell you the size of the potential claim.

This reframes the forward, too. A forward price is not the market's forecast — it is a tradeable line in the sand set by arbitrage from the rate gap. When the forward implies the high-yielder will weaken sharply, the historical record says: *fade that*. The realized move is usually milder, and frequently the other way. The forward is the thing carry traders sell *into*, not the thing they believe.

Practically, this converts into a discipline. Rank currencies by their rate gap, go long the top, fund in the bottom, and *size by the ratio of carry to volatility* — because the premium is real but the crash is real too, and the only protection is to take less risk when the implied volatility and the crowding say the unwind would be vicious. Cut the position when funding-currency safe-haven demand spikes (the franc and yen bid in a risk-off lurch is the early warning), and never let leverage turn a 4%-a-year premium into a 60%-a-week obituary. The rate gap tells you the *reward*; the volatility surface and positioning tell you the *risk*; UIP's failure is the reason there is a reward to weigh against that risk at all. Read the gap as a price for bearing risk, and the whole apparatus of carry — the funding leg, the high-yield leg, the crash hedge — falls into place as the rational response to a forecast the market refuses to make come true.

And it reconnects to the spine of this series. An exchange rate is the relative price of two monies, and the rate gap is the price of *time-and-risk* in that relationship. UIP tried to say that price is always fair — that the gap is exactly offset by the expected move. The data says it is not fair; it is a risk premium, and a harvestable one. Every carry trade ever placed is a bet that the textbook is wrong in the way it is reliably wrong. The forward-premium puzzle is not an embarrassment for finance to explain away. It is the load-bearing fact that makes one of the largest trades in the world pay — and the steamroller that, every few years, makes everyone remember why.

## Further reading & cross-links

- [Interest-rate differentials: the master variable of FX](/blog/trading/forex/interest-rate-differentials-the-master-variable-of-fx) — the rate gap that UIP says is offset and carry says is paid; the input to everything here.
- [The carry trade: getting paid to hold a currency](/blog/trading/forex/the-carry-trade-getting-paid-to-hold-a-currency) — the full mechanics of harvesting the UIP failure: funding, sizing, the carry-to-vol ratio.
- [Carry crashes: picking up pennies in front of a steamroller](/blog/trading/forex/carry-crashes-picking-up-pennies-in-front-of-a-steamroller) — the other half: why the premium exists because the unwind is catastrophic.
- [What moves exchange rates: rates, flows, carry](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry) — how rate differentials, capital flows, and risk appetite combine to drive a pair.
- [Carry-trade unwinds: 1998, 2008, 2024 — when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) — the anatomy of the crashes that vindicate UIP all at once.
- [How monetary policy moves currencies: rate differentials](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials) — where the rate gaps come from in the first place.
- [Forward rates: what the market expects rates to be](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be) — the forward-rate machinery we cite rather than re-derive.

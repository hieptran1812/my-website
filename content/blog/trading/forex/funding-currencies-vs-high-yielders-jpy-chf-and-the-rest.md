---
title: "Funding Currencies vs High-Yielders: JPY, CHF, and the Rest"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The two sides of carry — why JPY and CHF make ideal funding legs, what makes an emerging-market currency a high-yielder, and how a diversified, risk-weighted carry basket is actually built."
tags: ["forex", "currencies", "carry-trade", "funding-currency", "high-yield", "jpy", "chf", "emerging-markets", "risk-weighting", "volatility"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A carry trade has two legs that do opposite jobs: you *borrow* a cheap, deep, crisis-proof **funding currency** (yen, franc) and you *hold* an expensive, fragile **high-yielder** (peso, real, rand), and the gap between their interest rates is your income.
>
> - A good **funder** is cheap (0.25–0.50% rate), deeply liquid, and *rallies* in a panic because money runs home to it — the yen and franc fit perfectly.
> - A good **high-yielder** pays a fat 8–12%+ local rate, but it is thinner, more volatile, and tends to *crash* in exactly the same risk-off moment your funder is rallying.
> - You never hold one pair — you build a **basket**: short three funders, long three yielders, so the blended rate gap is earned across many independent legs.
> - The number to remember: in a clean 2024 snapshot you could borrow yen at **0.25%** and hold Brazilian real at **12.25%** — an 12-point gap — but the August 2024 unwind shows that gap can vanish in a week.

On the morning of 5 August 2024, a trade that had paid out, quietly and reliably, for the better part of three years detonated. The yen — which had spent those years sliding from 115 to nearly 162 to the dollar as the Bank of Japan held rates near zero while everyone else hiked — suddenly screamed higher. USD/JPY collapsed from 161.9 in early July to 141.7 by 5 August, a roughly 12% move in the world's third-most-traded currency in barely a month. Global equity indices cratered. Japan's Nikkei had its worst single day since 1987. The VIX, Wall Street's fear gauge, spiked to an intraday 65.7. And the proximate cause was not a war, a default, or a bank failure. It was the unwinding of millions of positions that all shared the same skeleton: *short the yen, long something that pays more.*

That is the carry trade, and the August 2024 episode is the cleanest modern illustration of why its two legs matter so much. One leg — the yen — is a **funding currency**: cheap to borrow, infinitely deep, and the thing money flees *into* when the world gets scared. The other leg — whatever you bought with the borrowed yen, from Mexican pesos to Australian dollars to US tech stocks — is the **high-yield** side, the thing that pays you to hold it but that falls apart precisely when the funder rallies. Understanding what makes each side of that trade tick, and how professionals stitch many such pairs into a single risk-controlled **basket**, is the whole game. This post is about that anatomy.

![The rate menu showing funding currencies at the bottom and high-yielders at the top](/imgs/blogs/funding-currencies-vs-high-yielders-jpy-chf-and-the-rest-1.png)

The chart above is the entire trade on one axis. At the bottom sit the funders — the yen near 0.25%, the franc near 0.50% — currencies where borrowing costs almost nothing. At the top sit the high-yielders — the Mexican peso above 10%, the Brazilian real above 12%, the Turkish lira at an eye-watering 50%. The carry trade is, mechanically, nothing more than a bet that you can sit in the gap between the bottom of this menu and the top, collect the difference, and get out before the gap snaps shut. Everything else in this post is detail on *which* currencies belong where, *why*, and *how much of each* you should hold.

## Foundations: The two sides of every carry trade

Before anything else, let us nail down the vocabulary, because the carry trade is one of those topics where the jargon does most of the obscuring.

A **carry trade** is borrowing money in a place where interest rates are low and parking it where rates are high, pocketing the difference. The everyday-money version: suppose your bank lets you borrow at 1% and a foreign bank pays 11% on deposits. You borrow \$100,000 at 1% (you owe \$1,000 a year), convert it, deposit it abroad at 11% (you earn \$11,000 a year), and — if the exchange rate doesn't move — you keep \$10,000. That \$10,000 is the **carry**: the income that drops out of holding the position, before any change in the exchange rate. In FX, the "where you borrow" and "where you deposit" are different *currencies*, so the borrowed leg is called the **funding currency** and the held leg is the **high-yielder** (or **investment currency**, or **target currency** — all the same thing).

An **exchange rate** is the relative price of two monies, and that is the spine of this entire series: you never own "a currency" in isolation. Every carry position is a *pair* — short one money, long another — and what determines whether it makes or loses money is two things stacked on top of each other: the **interest-rate gap** (your income, which is positive and steady) and the **change in the exchange rate** (your capital gain or loss, which can be either sign and is the thing that bites). The carry trade is a wager that the income outweighs the currency move. Often it does. Occasionally — August 2024, 2008, 1998 — it spectacularly does not.

> **Why the rate gap exists at all.** This series leans on the fixed-income and macro posts for the *why* of interest rates, so we won't re-derive them here. The short version: a central bank sets a country's short-term rate to manage inflation and growth, and a country fighting high inflation (Brazil, Turkey, Mexico) sets a high rate, while a country fighting deflation or low growth (Japan, Switzerland) sets a near-zero rate. For the mechanism, see [how monetary policy moves currencies](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials). What matters for *this* post is that the rate gap is a fact you can observe and trade, and that the currencies at the two ends behave very differently.

![A matrix comparing funding leg and high-yield leg across rate, depth, crisis behaviour, use, and the catch](/imgs/blogs/funding-currencies-vs-high-yielders-jpy-chf-and-the-rest-2.png)

The matrix above lays out the two sides trait by trait, and it is worth reading slowly because almost everything in this post is an elaboration of one of these rows. The funder is cheap, deep, and rallies in a crisis — so you short it. The high-yielder pays well but is thin, volatile, and crashes in a crisis — so you go long it and collect the rate. The two "catch" cells at the bottom are the same catch said two ways: in a risk-off unwind your short funder rallies *against* you while your long yielder crashes *against* you, and both losses land at once. The carry trade is short two kinds of insurance simultaneously, and the August 2024 yen spike is what it looks like when both insurance claims pay out on the same day.

### The funding leg: cheap, deep, and safe

A funding currency is the one you borrow. To be *good* at the job, it needs three properties, and the magic of the yen and the franc is that they have all three at once.

**First, a low interest rate.** This is the obvious one. Your cost of carry is the funder's interest rate — every basis point you pay to borrow the funder is a basis point off your income. The yen at 0.25% and the franc at 0.50% are the cheapest funding in the developed world. If you funded in dollars at 4.50% instead, your gap to a 10%-yielder shrinks from roughly 9.7 points to 5.5 — you'd give up nearly half your income just by choosing the wrong funder.

**Second, deep liquidity.** You are going to borrow this currency in size, short it, and — critically — you need to be able to *cover* that short (buy it back) in any market condition, including a panic. The yen is on roughly 16.7% of all FX trades and the franc on 5.2%; both trade in essentially unlimited size around the clock. A currency you can't reliably buy back is a trap: the moment you most need to close your short is the moment everyone else is trying to close theirs, and a thin market gaps against you. Depth is what lets you exit at a price close to where you see the screen, rather than 5% worse.

**Third — and this is the subtle, defining trait — it must *rally* in a risk-off panic.** This is the safe-haven property, and it is what separates a merely cheap currency from a great funder. When global markets seize up, capital does not stay put; it runs *home* to the safest, most liquid harbors. The yen and franc are exactly those harbors, for a structural reason we'll unpack: both countries are large net creditors to the rest of the world, so in a panic their investors repatriate foreign assets and their currencies rally.

Notice the cruel irony embedded in trait three. The very property that makes a currency a good funder — that it rallies in a crisis — is *also* the thing that hurts you most, because *you are short it.* When the yen rallies 12% in a panic, your short-yen funding leg loses 12%. So the funder is doing two opposite things for you: it gives you cheap, deep, reliable financing in calm times, and it stabs you in the back in the exact moment the rest of your basket is also falling apart. That tension is the heart of why carry is a "picking up pennies in front of a steamroller" trade.

There's a fourth property worth naming because it's easy to miss: a great funder needs **a credible commitment to stay cheap.** It's not enough for a currency to have a low rate *today* — you're going to hold the short for months or years, and if the funding currency's central bank suddenly hikes, your cost of carry jumps and the currency rallies on you at the same time. This is exactly what made the yen the funder of the century: for decades the Bank of Japan was *structurally* stuck near zero, fighting deflation, with no credible path to meaningful hikes. The market could be confident the funding would stay cheap. When that confidence finally cracked in 2024 — when the BoJ started hiking and signaled more to come — the funder stopped being reliably cheap, and the whole edifice of short-yen carry wobbled. A funder is only as good as the market's belief that it will *stay* a funder. The franc has a different version of this risk: the Swiss National Bank can and does intervene, so its commitment to cheapness is more discretionary than Japan's was structural.

These four traits — low rate, deep liquidity, safe-haven rally, credible commitment to stay cheap — are why the funding universe is so small. In practice there are only two pure funders (yen, franc), one quasi-funder (the euro, low-rate and a creditor bloc but with weaker safe-haven flows), and everyone else is either too expensive or too unreliable. That scarcity matters: because the funders are so few and so similar, *everyone* funds in the same two currencies, which is precisely why the unwind is so violent. When the trade turns, the whole world is short the same two funders and they all rush to cover at once.

#### Worked example: the cost of choosing the wrong funder

Suppose you have \$1,000,000 to put to work and you want to be long the Mexican peso at a 10.25% local rate. The only question is what to fund it with. Compare three funders:

- **Fund in yen at 0.25%.** Your gross income is the gap: 10.25% − 0.25% = 10.00%. On \$1,000,000 that is \$100,000 a year.
- **Fund in francs at 0.50%.** Gap = 10.25% − 0.50% = 9.75% → \$97,500 a year.
- **Fund in dollars at 4.50%.** Gap = 10.25% − 4.50% = 5.75% → \$57,500 a year.

The difference between the best funder (yen) and the dollar is \$100,000 − \$57,500 = **\$42,500 a year**, on the exact same long-peso position. The funder choice is worth 42 basis points of *gross* income per point of rate difference, and on a million dollars that is real money. The takeaway: half the carry trade is choosing the *cheapest* deep, safe place to borrow — the leg most beginners ignore because it looks passive.

### The high-yield leg: paid well to take real risk

The other leg is the currency you *hold* — the one that pays you a high local interest rate. To be a high-yielder, a currency essentially needs one thing: a high policy rate. But high rates do not exist in a vacuum. A central bank sets a 10% or 12% rate because it has to — because inflation is high, or the currency is under pressure, or foreign investors demand a premium to hold the country's debt. So the high yield is a *symptom*, and the disease it's a symptom of is exactly what makes the high-yielder dangerous.

The classic high-yielders fall into two buckets:

- **Emerging-market currencies** like the Mexican peso (MXN), Brazilian real (BRL), South African rand (ZAR), Indonesian rupiah, Indian rupee, and — at the extreme — the Turkish lira (TRY). These pay high rates because their central banks fight high domestic inflation and their bonds carry a country-risk premium. See [emerging-market sovereign debt](/blog/trading/fixed-income/emerging-market-and-sovereign-debt-yield-with-country-risk) for where that premium comes from.
- **Commodity currencies** like the Australian dollar (AUD), New Zealand dollar, Canadian dollar (CAD), and Norwegian krone. These are developed-market currencies that historically paid a bit more than the G3 (dollar, euro, yen) and rise and fall with their export commodities — iron ore for the Aussie, oil for the loonie and the krone. They are *milder* high-yielders: less explosive than EM, but still pro-cyclical, meaning they fall in a global downturn.

The properties of a high-yielder are the mirror image of a funder's. Where the funder is deep, the yielder is **thinner** — wider bid-ask spreads, and a market that can gap when flows reverse. Where the funder rallies in a panic, the yielder **crashes**, because the same hot money that chased its high yield runs for the exit all at once. And where the funder's catch is that it rallies against your short, the yielder's catch is more brutal: *the yield is never large enough to cover the size of the crash.* A currency might pay you 10% a year and then fall 30% in a week. You collected pennies for years and gave them all back, plus your principal, in days.

It's worth being precise about *why* a high-yielder is structurally fragile, because it's not a coincidence — it's the same fact viewed from three angles. A country that pays a high interest rate is, almost always, a country that *depends on foreign capital inflows* to fund itself: it runs a current-account deficit, it has to attract money from abroad, and the high rate is the bribe it offers to pull that money in. But money that comes for yield leaves when the yield no longer compensates for the risk — it is the opposite of the funder's sticky creditor flows. So the high-yielder is built on a foundation of *fickle* capital. In calm times, that capital floods in and the currency is stable or even strengthens (which makes the carry trade pay double — yield *plus* appreciation). In stressed times, the capital floods out, and because the currency was never deeply liquid to begin with, the outflow becomes a stampede with no buyers on the other side. The high rate that drew the money in is the same high rate that signals how fragile the underlying balance is.

There is also a **spread** cost the funder mostly escapes. Look at the dealer spreads across the tiers: a major like EUR/USD trades on a fraction of a pip, USD/MXN on roughly 8 pips, and an exotic like USD/TRY on 25 pips. Every time you enter or exit a high-yield leg you pay that spread, and on a thin EM pair traded with any frequency the spread quietly erodes a real slice of your carry. The funder's depth means you keep almost all of your gross gap on the funding side; the yielder's thinness means you leak some of it on every transaction. The high yield is partly an illusion of the headline rate — net of spread and the occasional gap, you keep less than the menu suggests.

#### Worked example: gross carry versus net carry after the spread

Take a \$1,000,000 long-peso position funded in yen, gross carry 10.0% = \$100,000 a year. Now add the trading frictions a real desk pays. USD/MXN trades on roughly 8 pips of spread; on a \$1,000,000 notional at a USD/MXN level around 20, 8 pips is about \$400 to enter and \$400 to exit. If you also roll or adjust the position monthly, those costs recur.

- **Round-trip entry/exit spread:** ~\$800.
- **If you rebalance monthly** (12 round trips a year): ~12 × \$800 = **\$9,600** in spread alone.
- **Net carry after spread:** \$100,000 − \$9,600 ≈ **\$90,400**, or 9.0% rather than the headline 10.0%.

A whole percentage point of your yield vanished into the spread on a moderately liquid EM pair — and on a thinner exotic like the lira (25 pips), the same active management could cost several points. The intuition: the headline rate menu is *gross*; what you actually keep depends on how deep the pair is and how often you touch it, which is one more reason the deepest, cleanest legs beat the highest-yielding ones.

![Developed-market policy rates with JPY and CHF as the cheap funding legs](/imgs/blogs/funding-currencies-vs-high-yielders-jpy-chf-and-the-rest-3.png)

This chart shows just the developed-market policy rates, to make a narrower point: even *within* the rich-world G10, there is a clear funding hierarchy. The Bank of England, the Fed, the ECB, and the RBA all sat between 3% and 4.75% at end-2024, while the Bank of Japan held at 0.25% and the Swiss National Bank at 0.50%. The gap from the BoJ floor up to the rest of the pack — the dotted green line — is the *developed-market* carry you can earn without ever touching an emerging market. Many of the most durable carry trades are entirely G10: long Aussie or long dollar, funded in yen. You give up the fat EM yield, but you get a far deeper, less crash-prone book.

#### Worked example: the gross gap on a single EM pair

Let's build the simplest possible carry position and trace the cash. You borrow \$1,000,000-equivalent of Japanese yen at 0.25% and use it to buy Brazilian real, which earns the Brazilian policy rate of 12.25%.

- **Annual cost of the funding leg:** 0.25% of \$1,000,000 = \$2,500.
- **Annual income from the high-yield leg:** 12.25% of \$1,000,000 = \$122,500.
- **Net carry (income minus cost):** \$122,500 − \$2,500 = **\$120,000 a year**, or 12.0% of the notional.

That is a stunning 12% annual yield *if the exchange rate doesn't move*. But now stress it: the Brazilian real only has to fall about 12% against the yen over the year to wipe out the entire \$120,000 of carry, and EM currencies routinely move that much. A 25% real depreciation — well within an EM crisis range — turns your +\$120,000 income into roughly −\$130,000 net. The intuition: the carry is the *width of the cushion*, and the cushion on a single EM pair is thinner than the moves that pair makes in a bad month.

## The rate menu: where every currency sits

Let's return to the full menu, because the spread of rates across currencies is the raw material of every carry decision. At the bottom of the menu — the funding zone — you have the developed safe-havens: yen (0.25%), franc (0.50%). In the middle you have the developed core: dollar (4.50%), Aussie (4.35%). And at the top — the high-yield zone — you have the emerging markets: rand (7.75%), peso (10.25%), real (12.25%), and the lira sitting alone at 50%.

The single most important thing this menu teaches is that **the spread, not the level, is what you trade.** It does not matter that the peso pays 10.25% in absolute terms; what matters is that 10.25% is 10 full points above the yen's 0.25%. If every currency in the world paid 10%, there would be no carry trade at all — borrowing and lending would cost the same everywhere, and the gap you live on would be zero. Carry exists *because* central banks set different rates for different domestic reasons, and FX is the market where those differences get arbitraged (or not — see the [uncovered interest parity puzzle](/blog/trading/forex/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle), which is precisely the observation that the carry gap is *not* fully arbitraged away by currency moves, which is why the trade has historically paid).

The Turkish lira deserves a special mention as the cautionary tail of the menu. A 50% policy rate looks like the greatest carry trade on earth — borrow yen at 0.25%, earn lira at 50%, pocket 49.75 points a year! In practice the lira has depreciated so violently and so persistently (losing the majority of its value against the dollar over recent years) that the enormous yield has been a trap: the currency falls faster than the rate pays. The lira is the living proof that *yield is a warning label, not a free lunch*. A 50% rate is the market screaming that this currency is expected to collapse; the rate is the compensation demanded for standing in front of it. The higher the yield, the louder the warning.

#### Worked example: when the high yield is a trap

Take the lira at face value. You borrow \$1,000,000 of yen at 0.25% and hold lira at 50%.

- **Gross carry if nothing moves:** 50.00% − 0.25% = 49.75% → \$497,500 a year. Spectacular.
- **But the breakeven depreciation** — how far the lira can fall before you lose money — is about 33% over the year. (Earning 49.75% on the notional means the currency could fall to roughly two-thirds of its value and you'd still be flat, because the carry cushions the drop.)
- **The reality:** the lira has frequently depreciated *more* than 33% in a year. Lose 40% on the currency and your net is roughly +49.75% − 40% ≈ +10%; lose 50% and you are deep in the red despite the gargantuan yield.

The lesson: a yield that looks too good to be true is the market pricing in a depreciation you should expect to happen. The carry on the lira is not income — it is a fee the market pays you for absorbing a near-certain currency loss, and historically the fee has not covered the loss.

## What makes a great funder: the safe-haven property

Of all the traits in this post, the one that separates a *cheap* currency from a *great funder* is the safe-haven property: it must rally when the world is scared. Let's understand precisely why the yen and franc have it and most low-rate currencies don't.

![A graph showing how a low rate, current-account surplus, and deep liquidity combine into a safe-haven funder](/imgs/blogs/funding-currencies-vs-high-yielders-jpy-chf-and-the-rest-4.png)

The diagram traces three structural traits converging into the funder's dual nature. Start at the top: a **low policy rate** makes the currency cheap to borrow and short. A **current-account surplus** — running for decades in both Japan and Switzerland — means the country, over time, has *sold more to the world than it bought*, and the proceeds have been recycled into foreign assets. Japanese pension funds, insurers, and households own trillions of dollars of foreign stocks and bonds; Switzerland's investors and its central bank likewise hold vast foreign portfolios. The country is a giant **net creditor**: the world owes it money, not the other way around.

Now watch what happens in a panic. When markets crash, those Japanese and Swiss investors — and the global funds that borrowed yen and francs to buy risky assets abroad — all want to come home. The Japanese pension fund sells its US stocks and buys yen to repatriate. The hedge fund that funded its peso trade in yen rushes to buy back the yen to close the short. Every one of those flows is a *purchase of the funder*, so the funder rallies. **A current-account surplus today is the seed of a safe-haven rally tomorrow:** because the country owns foreign assets, a crisis triggers a wave of repatriation that lifts its currency exactly when everything else falls.

This is why a low rate alone is not enough. There are plenty of low-rate currencies that are *not* safe havens — typically because the country runs a deficit and depends on foreign capital *inflows*. When a panic hits, those inflows stop, and the currency falls rather than rises. The euro is a borderline case: low-ish rate, but a creditor bloc, so it has partial safe-haven character. The dollar is the strangest case of all — it pays a high rate yet is the ultimate safe haven, because the entire world's debts are denominated in dollars, so a panic creates a global *scramble for dollars* regardless of the rate. The dollar's behavior gets its own treatment in [trading the dollar and the dollar smile](/blog/trading/macro-trading/trading-the-dollar-dxy-carry-dollar-smile); for the funder's purposes, just note that the yen and franc are the *purest* safe-haven funders because they combine the low rate *and* the creditor-nation repatriation flow.

The right-hand side of the diagram closes the loop with the funder's curse. The very repatriation that makes the funder rally in a panic is what *squeezes the shorts* — and in a carry trade, you are the short. So the funder's safe-haven rally is simultaneously its greatest virtue (it draws the funding flows that make it deep and cheap) and your single biggest risk (it rips higher against your short in the unwind).

#### Worked example: the safe-haven rally costing you on the funding leg

Return to the August 2024 unwind with numbers. You're funded in yen, short \$1,000,000 of it, holding a high-yield basket. Over the year you've collected, say, 9% of carry = \$90,000. Then the unwind hits and USD/JPY falls from 161.9 to 141.7 — the yen *rallies* about 12.5% against the dollar.

- Your short-yen leg loses roughly **12.5% of \$1,000,000 = \$125,000** as you have to buy back yen that has become more expensive.
- Your full-year carry of \$90,000 does not cover it. Net on the funding leg alone you're down \$125,000 − \$90,000 = **−\$35,000**, and that's *before* counting losses on the high-yield leg, which was also falling.

The intuition: the funder is the best thing you can borrow precisely because it's the thing that rallies in a crisis — and that same rally is the loss you take on your short when the crisis comes. Carry pays you to be short crisis insurance.

## What makes a great high-yielder: paid for fragility

Flip to the other leg. We've established that a high-yielder pays a high rate because its central bank is forced to, and that the high yield is a symptom of underlying fragility. Let's quantify that fragility, because the single cleanest measure of it is **volatility** — how much the currency's price jumps around.

![Implied volatility by currency pair showing high-yielders carry higher volatility](/imgs/blogs/funding-currencies-vs-high-yielders-jpy-chf-and-the-rest-5.png)

This chart plots the one-month implied volatility — the market's priced-in expectation of how much each pair will move — across the spectrum. The developed majors sit low: EUR/USD around 7%, USD/JPY around 9.5%, AUD/USD around 10.5%. The emerging-market pairs sit higher: USD/MXN around 13%, and USD/TRY at a towering 22%. **Reaching for yield means reaching for volatility** — the two rise together, almost mechanically, because the same fragility that forces a high rate also makes the currency prone to violent moves. (Implied volatility is the options market's forecast of movement; for how it's read and priced, see [reading the vol surface](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear). We cite the Greeks rather than re-derive them.)

This is the deep relationship that governs the whole trade, and it has a name in the literature: carry is **short volatility**. The income you earn (the rate gap) is roughly compensation for bearing the risk that the currency lurches. In calm times, vol is low, the lurches don't happen, and you keep the carry. In a crisis, vol explodes, the high-yielders lurch *down*, the funders lurch *up*, and you give back years of carry at once. The relationship between carry and vol is so central to the trade that it gets its own post: [carry and volatility, the relationship that runs the trade](/blog/trading/forex/carry-and-volatility-the-relationship-that-runs-the-trade). For now, hold onto the mental model: **the yield is the price the market pays you for standing under the volatility.**

There's a second-order point hiding in this chart. Volatility doesn't just measure how bad the crash *can* be — it tells you how much risk each leg contributes to your *basket*. A 22%-vol lira position and a 9.5%-vol yen position are not equal risks even at equal dollar size; the lira leg will dominate your daily profit-and-loss swings. That observation is the seed of risk-weighting, which we'll get to. First, let's assemble the basket itself.

#### Worked example: the yield-per-unit-of-volatility test

A useful screen for a high-yielder is its *carry-to-vol ratio* — roughly how much income you collect per unit of risk you take. Compare two yielders, both funded in yen at 0.25%:

- **Mexican peso:** carry = 10.25% − 0.25% = 10.0%; implied vol ≈ 13%. Carry-to-vol = 10.0 / 13 ≈ **0.77**.
- **Turkish lira:** carry = 50.0% − 0.25% = 49.75%; implied vol ≈ 22%. Carry-to-vol = 49.75 / 22 ≈ **2.26**.

At first glance the lira looks *better* — more carry per unit of vol! But this is where the naive ratio lies: implied vol captures normal-times wobble, not the regime-shift risk of a one-way collapse. The peso's 0.77 is earned in a currency that mean-reverts; the lira's 2.26 is earned in a currency that has trended relentlessly weaker. The takeaway: carry-to-vol is a starting filter, not a verdict — you still have to ask *what kind* of risk the volatility is hiding, and a structurally one-way currency is more dangerous than its vol number suggests.

## Building the basket: don't bet the trade on one pair

Now the construction. No serious carry investor holds a single pair. The reason is diversification: any one high-yielder can blow up for an idiosyncratic, country-specific reason — a coup, a default, a surprise rate cut, a commodity crash — that has nothing to do with the others. If your whole position is long-peso/short-yen and Mexico has a bad election week, you lose everything at once. But if you spread the long side across the peso, the real, and the rand, and the short side across the yen and the franc, a single country's blowup costs you only one slice. You build a **carry basket**.

![A stack diagram showing a carry basket built from short funders and long high-yielders](/imgs/blogs/funding-currencies-vs-high-yielders-jpy-chf-and-the-rest-6.png)

The stack above shows the canonical structure: a short funding base (yen and franc, borrowed cheap) at the top, three long high-yield legs (peso, real, rand) in the middle, the **blended carry** computed below, and the shared risk at the bottom. The blended carry is the average yield of the longs minus the average cost of the shorts. With longs averaging roughly 10.1% (10.25 + 12.25 + 7.75, divided by three) and shorts averaging roughly 0.375% (0.25 + 0.50, divided by two), the gross blended gap is about 9.7%. That single number — the basket's blended carry — is what the whole construction earns per year if exchange rates stand still.

The diversification benefit is real but limited, and the limit is the most important thing to understand about carry baskets. The legs diversify each other's *idiosyncratic* risk (Mexico's election, Brazil's fiscal scare, South Africa's power crisis) — those are independent, so spreading across them smooths your returns in normal times. But the legs do *not* diversify the *systematic* risk, the risk that *all* high-yielders crash together in a global risk-off event. In a panic, the correlations between every high-yielder snap to nearly 1: they all fall, simultaneously, because they're all the same trade ("long global risk, short safe-haven") wearing different country costumes. This is the carry trade's fatal flaw, and it has a name across markets — [when correlations go to one in a crisis](/blog/trading/cross-asset/when-correlations-go-to-one-in-a-crisis). Your basket looks beautifully diversified on a calm Tuesday and behaves like a single concentrated bet on the one day it matters.

How do you actually *choose* the legs? The professional basket isn't just "the three highest yields" — that would load you into the lira and the most fragile currencies, exactly the ones whose yield is a warning label. A better construction screens on several axes at once: the **carry** (the rate gap, your income), the **valuation** (is the currency cheap or expensive on purchasing-power-parity grounds — a cheap high-yielder has further to *fall back to* and is safer than an expensive one), the **liquidity** (can you trade and exit the leg in size), and the **stability of the regime** (is the central bank credible, is the political situation calm). The peso, real, and rand are perennial basket members not because they pay the *most* but because they balance a fat yield against tradable liquidity and reasonably credible institutions. The lira pays far more and almost never makes a disciplined basket, because its fragility swamps its yield. Choosing legs is the act of asking, for each candidate, "is the yield *compensation* for risk I can survive, or a *warning* about risk I can't?"

You also want the legs to be as *independent* of one another as you can manage, because independence is what makes diversification work in the calm periods. A basket of the peso, the real, and the rand spreads you across North America, South America, and Africa, across an oil-and-manufacturing economy, a soft-commodity economy, and a metals-and-mining economy. That's far better than a basket of three Latin American currencies that all move on the same commodity cycle and the same Fed decisions. The more genuinely different the drivers behind each leg, the more your basket's normal-times path smooths out — even though, again, *none* of that independence survives the systematic crash.

#### Worked example: building a \$1,000,000 carry basket

Let's build the basket concretely. You have \$1,000,000 to deploy. The simplest construction is **equal dollar weights**: short two funders and long three yielders, sizing each long leg at \$333,333 (one-third of the capital) funded by an equal split of yen and franc shorts.

Long legs (one-third each of \$1,000,000):

- **Long MXN at 10.25%:** income = 10.25% × \$333,333 = \$34,167.
- **Long BRL at 12.25%:** income = 12.25% × \$333,333 = \$40,833.
- **Long ZAR at 7.75%:** income = 7.75% × \$333,333 = \$25,833.
- **Total long income:** \$34,167 + \$40,833 + \$25,833 = **\$100,833.**

Funding legs (the \$1,000,000 is borrowed, split half yen, half franc):

- **Short JPY at 0.25%:** cost = 0.25% × \$500,000 = \$1,250.
- **Short CHF at 0.50%:** cost = 0.50% × \$500,000 = \$2,500.
- **Total funding cost:** \$1,250 + \$2,500 = **\$3,750.**

**Net annual carry of the basket:** \$100,833 − \$3,750 = **\$97,083**, or about **9.7%** of the \$1,000,000 notional. The intuition: the basket earns nearly 10% a year in calm conditions, spread across three independent country bets — but every one of those bets is the same systematic wager, so the diversification protects you against a bad week in Brazil, not against a bad week for the world.

## Risk-weighting: stop the loudest leg from running the book

The equal-dollar-weight basket above has a hidden flaw, and fixing it is what separates a professional carry book from a retail one. Recall the volatility chart: the lira moves at 22% vol, the peso at 13%, the yen at 9.5%. If you give every leg an equal *dollar* weight, the high-vol legs will dominate your *risk* — your daily profit-and-loss will swing mostly with the loudest, most volatile currency, regardless of how much income it contributes. A \$333,333 lira leg and a \$333,333 peso leg are equal in dollars but the lira leg generates nearly twice the daily risk. Your "diversified" basket is secretly a concentrated bet on the wildest currency in it.

The fix is **risk-weighting**: instead of equal dollars, you size each leg *inversely to its volatility*, so each leg contributes roughly the same *risk* to the basket. A 22%-vol leg gets a smaller dollar weight than an 11%-vol leg, scaled so that 22% × weight ≈ 11% × weight for every leg.

![A pipeline showing how a carry basket is risk-weighted by scaling legs inversely to volatility](/imgs/blogs/funding-currencies-vs-high-yielders-jpy-chf-and-the-rest-8.png)

The pipeline above walks the full risk-weighting workflow. You start with equal dollar weights, measure each leg's volatility, scale each weight inversely to its vol (the formula is just `weight ∝ 1/vol`, then normalized so the weights sum to one), watch the loud legs shrink, cap any single name so no one currency can sink the book, and finally stress-test the whole thing against a correlated unwind where *every* leg falls together. That last step is the one beginners skip and professionals obsess over, because it's the only step that accounts for the correlations-go-to-one problem. Risk-weighting smooths your normal-times P&L beautifully; it does *nothing* to protect you from the systematic crash, which is why the final step is to size the *whole basket* down to a total loss you can actually survive.

#### Worked example: risk-weighting the three high-yield legs

Take three high-yield legs with these implied vols: MXN at 13%, ZAR at 14%, and a TRY leg at 22%. (We'll risk-weight just the long side; the funding split follows the same logic.) The recipe:

- **Step 1 — inverse vols:** 1/13 = 0.0769, 1/14 = 0.0714, 1/22 = 0.0455. Sum = 0.1938.
- **Step 2 — normalize to weights:**
  - MXN: 0.0769 / 0.1938 = **39.7%**
  - ZAR: 0.0714 / 0.1938 = **36.8%**
  - TRY: 0.0455 / 0.1938 = **23.5%**
- **Step 3 — apply to \$1,000,000:** MXN \$396,700, ZAR \$368,000, TRY \$235,300.

Compare to equal weights (\$333,333 each): the loud 22%-vol TRY leg shrinks from 33.3% to 23.5%, and the quieter MXN leg grows from 33.3% to 39.7%. Now check the risk contributions: MXN 0.397 × 13% = 5.16%, ZAR 0.368 × 14% = 5.15%, TRY 0.235 × 22% = 5.17% — all three legs now contribute essentially *equal risk* (~5.2% each), which is exactly the goal. The intuition: risk-weighting doesn't ask which currency pays the most; it asks which currency *moves* the most, and shrinks it so no single loud leg secretly runs your whole book.

#### Worked example: the correlated-unwind stress test

Risk-weighting balances normal-times risk, but the killer is the day all legs fall together. So stress-test it. Take the risk-weighted \$1,000,000 basket above and assume the historical carry-crash scenario: in the August 2024 unwind the carry trade fell about 12%, so model *every* high-yield leg dropping 12% against the funders at once.

- **MXN leg:** −12% × \$396,700 = −\$47,604.
- **ZAR leg:** −12% × \$368,000 = −\$44,160.
- **TRY leg:** −12% × \$235,300 = −\$28,236.
- **Total basket loss in the unwind:** −\$47,604 − \$44,160 − \$28,236 = **−\$119,000**, or about −11.9% of the notional.

Against that, your annual carry was around 9.7%, or \$97,000. So a single August-2024-style unwind erases more than a full year of carry. The intuition: risk-weighting made the legs contribute equal risk *and* it told you the brutal truth — that the diversified-looking basket loses ~12% in a correlated crash, which is why the last step is always to size the *whole book* down to a drawdown you can survive, not up to the carry you'd like to earn.

## The funder weakens — until it doesn't

There's one more dynamic that makes carry seductive and dangerous in equal measure: the trend. In the years *between* crashes, the funder doesn't just sit still — it tends to *weaken* against the high-yielders, which adds a capital gain on top of your carry income. This is the part that makes carry feel like free money for years at a time.

![USD/JPY level versus the US-Japan 2Y rate gap from 2019 to 2025](/imgs/blogs/funding-currencies-vs-high-yielders-jpy-chf-and-the-rest-7.png)

The chart traces the cleanest example: USD/JPY against the US-Japan two-year rate gap from 2019 to 2025. As the Fed hiked and the BoJ held, the rate gap blew out from roughly zero in 2020 to over 4 points by 2022 — and the yen weakened in lockstep, sliding from 103 to 157. A short-yen carry trader over those years earned the rate gap *and* a capital gain as the yen fell, a double payout that pulled in more and more capital. This is the funder's seductive trend: a widening rate gap both *increases* your carry and *weakens the funder against you in the right direction*, so the trade pays twice and looks unstoppable.

And then the line snaps. The flat patch at the right end of the chart — 2024 into 2025 — is where the rate gap started to narrow (the BoJ finally hiked, the Fed finally cut), and the yen's multi-year weakening abruptly reversed. The August 2024 unwind sits right at that inflection. The lesson, woven back into this series' spine: an exchange rate is the relative price of two monies driven by the *gap* between their rates, and when that gap *stops widening*, the trend that powered the carry trade for years can reverse in days. The funder's slow weakening is borrowed time, and the bill comes due all at once.

## Common misconceptions

**"High yield means high return."** No — high yield means high *income*, which is a different thing from return. Return is income *plus* the change in the exchange rate, and for the highest-yielders the exchange rate change is reliably negative. The Turkish lira's 50% yield has been one of the *worst* trades of recent years because the currency fell faster than the rate paid. The rule: treat a sky-high yield as a *warning label* the market has stapled to a currency, not a free coupon. The higher the yield, the more depreciation the market is pricing in.

**"A diversified basket is safe."** It's safe against *idiosyncratic* risk — a single country's bad week — but not against *systematic* risk. In a global risk-off event, every high-yielder falls together and every funder rallies together, so a five-leg basket behaves like one concentrated trade. The 2008, 1998, and 2024 unwinds all hit *every* carry pair at once. Diversification across high-yielders buys you smoothness in calm times and almost nothing in a crisis, when correlations snap to one.

**"The funder is the boring, safe leg."** Exactly backwards. The funder is the leg that *rallies against you* in a crisis, and that rally is frequently your single biggest loss. In August 2024 the move that broke the trade was the *yen* — the funder — spiking 12% in weeks, not any one high-yielder collapsing. The funder is cheap and reliable in calm markets and lethal in a panic, precisely because its safe-haven property and your short position are on a collision course.

**"You earn the full interest-rate gap."** Not quite. The headline gap (say 10 points) is your *gross* carry; your *net* carry is reduced by the funding cost, the bid-ask spread on entering and exiting each leg (much wider on EM pairs — tens of pips on USD/MXN versus a fraction of a pip on EUR/USD), and any rollover or financing costs. On a thin EM pair the spread alone can eat a meaningful slice of a year's carry if you trade it actively. The deeper the pair, the more of the gross gap you keep — another reason the cleanest funders are the deepest currencies.

**"Risk-weighting protects you in a crash."** It protects you from one leg *dominating* in normal times, but it does nothing about the correlated crash. Risk-weighting equalizes how much each leg contributes to your daily wobble; it cannot change the fact that all the legs fall together when the world panics. The only real protection against the crash is to *size the whole basket smaller* — to hold less carry than the seductive yield invites you to hold.

## How it shows up in real markets

**The 2024 yen unwind (August).** The textbook case, and the one this post opens with. For three years, short-yen carry was the consensus trade: borrow at 0.25%, buy anything that paid more, ride the weakening yen. By mid-2024, USD/JPY had reached 161.9 and the positioning was enormous and crowded. Then two things converged: the Bank of Japan hiked rates (narrowing the gap that powered the trade), and US data softened (raising the odds of Fed cuts). The rate gap started closing, the yen's trend reversed, and the rush for the exit fed on itself — everyone short the yen had to buy it back at once. USD/JPY fell to 141.7 in weeks, the VIX hit 65.7 intraday, and the carry index dropped about 12%. The funder rallied, the high-yielders fell, and a year of carry evaporated. For the full anatomy, see [carry trade unwinds: 1998, 2008, 2024](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks).

**The 1998 LTCM/Russia unwind.** A generation earlier, the same skeleton. Russia defaulted in August 1998, triggering a global flight to safety. The yen — the funder of choice then as now — rallied violently: USD/JPY collapsed from 136 to 112 in a *three-day* window in October 1998, a roughly 18% move, as carry positions funded in yen were unwound en masse. Long-Term Capital Management, which had carry-like relative-value positions across the globe, was a casualty. The mechanism was identical to 2024: a shock made the high-yielders fall, which made traders cover their funder shorts, which made the funder spike, which forced more covering — a self-reinforcing squeeze on the cheapest, deepest, safest currency in the world.

**The Swiss franc as the other funder.** The franc earns its place beside the yen because Switzerland is the other great creditor-nation safe-haven. But the franc carries a specific extra risk: the Swiss National Bank actively manages it, and in January 2015 it abandoned its EUR/CHF 1.20 floor without warning, sending the franc up roughly 30% in *minutes*. Anyone short the franc as a funding leg was annihilated. The episode is a reminder that a funder's safe-haven rally can be turbocharged by a central-bank policy shift, and that the "boring" funding leg can deliver the single most violent move in the entire FX market. The franc remains a premier funder; it just comes with intervention risk the yen mostly lacks.

**The commodity currencies as mild high-yielders.** Not every high-yield leg is an emerging market. The Australian dollar spent decades as the developed world's favorite high-yielder — a G10 currency that paid more than the yen and franc and rose with iron-ore demand. A "long AUD, short JPY" trade was the canonical G10 carry: deep on both legs, far less crash-prone than an EM basket, but still pro-cyclical — the Aussie falls in a global slowdown just as iron ore demand and risk appetite fade together. The commodity currencies show that the funder/high-yielder split is a *spectrum*, not a binary: the Aussie is a high-yielder relative to the yen and a funder relative to the rand.

**The EM carry-and-sudden-stop cycle.** Emerging-market high-yielders attract waves of "hot money" — foreign capital chasing the yield — during calm, risk-on periods, and then suffer violent **sudden stops** when sentiment turns and that money reverses all at once. The 2013 taper tantrum and the 2022 outflows are the modern examples; portfolio flows into EM can swing from +\$290 billion in a good year to −\$80 billion in a bad one. The high yield is the *bait* that draws the flows, and the sudden stop is the trap that springs when the flows reverse — covered in depth in [the carry into emerging markets and the sudden stop](/blog/trading/forex/the-carry-into-emerging-markets-and-the-sudden-stop).

**Why the funder always rallies the most.** Across all of these episodes — 1998, 2008, 2015, 2024 — notice the consistent signature: the most violent single move is almost always in the *funder*, not in any one high-yielder. The reason is positioning. The high-yield side of the world's carry trades is spread across dozens of currencies — pesos, reals, rands, rupiahs, liras — so when they all crash, the selling is diffused across many markets. But the *funding* side is concentrated in just two currencies, the yen and the franc. Every carry trade on earth is short one of those two. So when the unwind comes and everyone scrambles to buy back their funder at once, that buying pressure lands on an unusually narrow target, and the funder spikes harder than any single high-yielder falls. This is the structural reason the funder is the dangerous leg: it's not just that it rallies in a crisis, it's that the *whole world's* short positions are crammed into it, so the cover is a stampede through a single door. The narrowness of the funding universe — which we noted makes for great, deep, reliable funders in calm times — is exactly what makes the unwind so concentrated and so brutal.

## The takeaway: read a currency by which leg it is

Once you internalize the funder/high-yielder split, you read every currency differently. You stop asking "is this currency going up or down?" and start asking "which leg of the carry trade is this, and what regime are we in?"

A **funder** — yen, franc — is a currency you expect to *grind weaker in calm times and spike violently stronger in a panic*. Its low rate is its job description; its safe-haven rally is both why it's the funder of choice and why being short it is the trade's hidden time bomb. When you see the yen falling month after month, you're watching the carry trade work — and accumulating the fuel for the unwind. The flatter and more crowded the trend, the closer the snap.

A **high-yielder** — peso, real, rand, lira — is a currency that *pays you to hold it and crashes when you most need it not to*. Its high rate is a symptom of fragility, and the size of the yield is the size of the warning. You hold it for the income, you spread it across many legs to dampen the idiosyncratic blowups, and you size the whole thing small enough to survive the day all your legs fall together. The yield is never the question; the question is always whether the rate gap is wide enough to pay you for the volatility you're standing under — and in a crisis the answer is always no.

And the **basket** ties it together: short the deepest, cheapest, most crisis-proof funders; long a diversified set of high-yielders; risk-weight so no loud leg dominates; cap any single name; and stress-test the whole book against the correlated unwind that *will* eventually come. The carry trade is not a free lunch and it is not a scam — it is a real, persistent risk premium for being short global volatility, paid out in steady pennies and clawed back in occasional avalanches. Read a currency by which leg it is, size for the avalanche, and the carry is yours to keep. Forget which leg it is — or believe the high yield is free — and you become the liquidity for the next unwind.

The deepest version of the spine, then, is this: an exchange rate is the relative price of two monies, and a carry trade is a bet that the *gap between their interest rates* will outpay the *change in their relative price*. The funder and the high-yielder are just the two ends of that bet. Everything — the safe-haven rally, the volatility, the diversification, the risk-weighting, the unwind — falls out of that single relationship.

## Further reading & cross-links

- [The carry trade: getting paid to hold a currency](/blog/trading/forex/the-carry-trade-getting-paid-to-hold-a-currency) — the foundational mechanics of carry, which this post builds the two legs of.
- [Carry and volatility: the relationship that runs the trade](/blog/trading/forex/carry-and-volatility-the-relationship-that-runs-the-trade) — why carry is structurally short volatility, and how vol governs the unwind.
- [FX as a factor zoo: carry, value, momentum, and dollar](/blog/trading/forex/fx-as-a-factor-zoo-carry-value-momentum-and-dollar) — where carry sits among the systematic FX style factors.
- [The carry into emerging markets and the sudden stop](/blog/trading/forex/the-carry-into-emerging-markets-and-the-sudden-stop) — the hot-money-then-reversal cycle of the EM high-yield leg.
- [How monetary policy moves currencies: rate differentials](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials) — the central-bank mechanism that sets the rate gap you trade.
- [Carry trade unwinds: 1998, 2008, 2024 — when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks) — the recurring anatomy of the crash that ends every carry cycle.

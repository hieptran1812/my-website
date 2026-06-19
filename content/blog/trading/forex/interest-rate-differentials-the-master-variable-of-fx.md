---
title: "Interest-Rate Differentials: The Master Variable of FX"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The gap between two countries' interest rates is the gravitational center of an exchange rate — and covered interest parity proves the forward price is nothing but spot adjusted by that gap, from a pure no-arbitrage argument."
tags: ["forex", "currencies", "interest-rate-differential", "covered-interest-parity", "forward-points", "carry", "rate-gap", "no-arbitrage"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 44
---

> [!important]
> **TL;DR** — The single most important number in currency trading is the gap between two countries' interest rates. It is the master variable that sets the forward price exactly and pulls hard on the spot price.
>
> - **An exchange rate is the relative price of two monies, and the relative reward for holding them is the rate gap.** When US money pays 5.0% and euro money pays 3.0%, that 2-point gap is the gravitational center the pair orbits.
> - **Covered interest parity (CIP) is not a theory — it is arithmetic enforced by arbitrage.** A fully hedged round trip (borrow dollars, convert, deposit euros, convert back at a locked forward) must net to zero, and that one constraint pins the forward rate.
> - **The currency with the *higher* interest rate trades at a forward *discount*.** This trips up almost everyone: the dollar earning 5.0% buys *fewer* euros forward, not more.
> - **The one number to remember:** spot EUR/USD 1.0800, USD 5.0%, EUR 3.0% gives a fair 1-year forward of **≈ 1.1010** — the euro at a forward premium, the higher-yielding dollar at a forward discount, purely because of the 2-point gap.

On 5 August 2024, the Japanese yen did something that looked, on a chart, like a small financial earthquake. In a single session the yen strengthened against the dollar by several percent, with USD/JPY falling from the high-150s toward 142, after the Bank of Japan nudged its policy rate up by a sliver and signalled more to come. Trillions of dollars of positions that had been built on one assumption — that Japanese money would stay cheap and American money expensive — were unwound in a panic. The trade everyone loved broke in an afternoon.

What every one of those positions was really a bet on was a single number: the **gap between US and Japanese interest rates**. For two years the Federal Reserve had hiked toward 5.5% while the Bank of Japan held near zero. That widening gap had dragged the yen from about 103 per dollar in 2020 to nearly 162 by mid-2024. When the gap looked like it might start to close — when the BoJ finally moved — the whole edifice tilted. The pair did not move because of "sentiment" or a news headline in isolation. It moved because the master variable underneath it shifted.

This post is the spine of everything that follows in this series on what moves a currency. The claim is simple and almost arrogant in its reach: **the interest-rate differential between two countries is the gravitational center of their exchange rate.** It sets the forward price *exactly*, through an iron law called covered interest parity, and it pulls hard on the spot price through the flow of money chasing yield. Get this one idea — really get it, down to the arithmetic — and a huge fraction of the currency market stops being mysterious.

![USD JPY versus the US Japan rate gap 2019 to 2025](/imgs/blogs/interest-rate-differentials-the-master-variable-of-fx-1.png)

The chart above is the whole post in one picture. The blue line is USD/JPY — how many yen one dollar buys. The amber dashed line is the gap between US and Japanese two-year interest rates. They move together because one *causes* the other. Our job is to understand exactly why, and to make it precise enough that you could price a forward yourself on the back of a napkin.

## Foundations: The rate gap, from first principles

Before any formula, fix the vocabulary, because the rest of the post leans on three terms.

- **The pair, base and quote.** A currency is never priced alone; it is always priced *against* another. EUR/USD means "euros priced in dollars." The first currency (EUR) is the **base** — the thing you buy or sell one unit of. The second (USD) is the **quote** — the currency the price is measured in. EUR/USD = 1.0800 means one euro costs 1.08 dollars. USD/JPY = 150.00 means one dollar costs 150 yen.
- **Spot and forward.** The **spot** rate is the price for delivery now (technically two business days out). A **forward** rate locks an exchange rate *today* for delivery on some agreed future date. The two are almost never equal, and the gap between them is the subject of this whole post.
- **The rate gap (the interest-rate differential).** The difference between the interest you can earn holding the two currencies for the same period. If one-year US money pays 5.0% and one-year euro money pays 3.0%, the gap is 2.0 percentage points in the dollar's favor. This single number is the engine of everything below.
- **Interest rate.** Just to be fully concrete: an interest rate is the price of money over time — the rent you collect for lending it, or pay for borrowing it. Every currency has its own term structure of rates, anchored at the short end by what its central bank pays, and rising or falling out to longer maturities. When this post says "the US rate is 5.0%," it means a safe, one-year US deposit pays 5.0% — and "the euro rate is 3.0%" means the matching safe euro deposit pays 3.0%. We compare *like with like*: same maturity, same credit quality, two different currencies. That apples-to-apples comparison is what makes the rate gap a clean number.

One more framing before we build. The whole series rests on a single sentence: **an exchange rate is the relative price of two monies.** You never own "a currency" the way you own a stock; you always own one money *against* another. So when you ask "what is the dollar worth?" the honest answer is "against what?" — against the euro it might be strengthening while against the yen it is flat. Every currency position is therefore a *spread*: long one money, short another, by construction. And the cleanest, most measurable reason to prefer holding one money over another for a stretch of time is the interest it pays. That is why the rate gap, not any single country's rate, is the master variable. A bet on a currency is always a bet on a *difference*.

Here is the rate gap in everyday money, with no maths. Suppose a friend in America will pay you 5% a year to hold their dollars, and a friend in Europe will pay you 3% a year to hold their euros. Both are completely safe. Now a third person offers a deal: swap your dollars for euros today, and swap them back in exactly one year at a rate agreed *right now*. What is a fair swap-back rate?

It cannot leave you richer for having parked in the higher-paying dollars. If it did — if you could earn the extra 2% *and* get your euros back at today's rate — you would have found free money, and so would everyone else, and they would all pile into the same trade until the free lunch vanished. So the fair swap-back rate has to claw back exactly the 2-point head start the dollars earned. The dollar has to come back "cheaper" in euros by about that 2%. That claw-back is the **forward points**, and the 2% is the rate gap. You have just derived the entire pricing logic of forwards with no algebra. The rest of this section only makes that precise and gives it its real name: **covered interest parity**.

Notice what this everyday story already rules out. It rules out the forward being a *forecast*. Nobody in the story made a prediction about where the exchange rate would actually go; the fair swap-back rate fell out purely from "don't leave free money on the table." That is the single most important thing to internalize before the algebra: the forward rate is not a guess about the future, it is a no-free-lunch *price*. It would be exactly the same number even if every trader in the world were certain the dollar would soar — because if it weren't, someone would run the loop and collect a riskless profit. Forecasts live in the *uncovered* world we'll reach at the end; the forward lives in the covered world, and the covered world is pure arithmetic.

### Two deposits, two rates, one comparison

Strip the currency market down to its atom. You have \$1,000,000 and a one-year horizon. There are exactly two safe ways to be holding dollars again in a year.

**Path A — stay home.** Deposit the \$1,000,000 in a US bank at 5.0%. In a year you have \$1,050,000. Done. No currency ever changed hands.

**Path B — take a detour through Europe.** Convert the \$1,000,000 into euros at today's spot rate, deposit those euros at 3.0% for a year, and convert the euro proceeds back into dollars. But here is the catch that makes this *safe*: you do not wait a year and gamble on what the exchange rate will be. You lock the conversion-back rate *today*, with a one-year forward contract. Now Path B has no currency risk at all — every rate is known the moment you start.

Two paths, same start (\$1,000,000), same finish (dollars, one year out), both completely safe. A market with no free lunches *must* make them pay the same. If Path B paid more, everyone would borrow to do Path B until it stopped; if it paid less, they would do the reverse. The only thing in Path B that is free to adjust is the forward rate. So the forward rate must be exactly the value that makes Path B equal Path A. That is covered interest parity, and it is the most reliable relationship in all of currency trading.

The word **covered** is doing important work in that name. It means the future conversion is *covered* — locked in advance with a forward contract — so there is no currency risk left over. The whole logic only holds because Path B has been stripped of any gamble: every rate is known at the start. Drop the forward, leave the future conversion to chance, and you have an *uncovered* position — a completely different animal that we will meet at the end. For now, hold the distinction: covered means hedged, and a hedged position is the one CIP nails down to the cent.

It is worth pausing on *why* the arbitrage is so much more forceful than the soft "supply and demand" stories that move other prices. A stock can stay overvalued for years because betting against it is risky and costly. But a mispriced forward offers a *riskless* profit — you take no view, you bear no risk, you simply harvest the discrepancy and walk away. Riskless profit attracts effectively unlimited capital instantly. That is why CIP is not a tendency or an average that holds "in the long run." It is a tight, fast constraint that holds to within a basis point or two, second by second, in any liquid pair — enforced not by belief but by the certainty that someone, somewhere, will pick up a free dollar lying on the floor before you can blink.

#### Worked example: the two paths must tie

Let me put numbers on both paths so the equality is concrete. Spot EUR/USD is 1.0800 (one euro costs \$1.08), the US rate is 5.0%, the euro rate is 3.0%.

- **Path A.** \$1,000,000 at 5.0% grows to \$1,000,000 × 1.05 = **\$1,050,000**.
- **Path B, step 1.** Convert \$1,000,000 to euros at 1.0800: \$1,000,000 ÷ 1.0800 = **€925,926**.
- **Path B, step 2.** Deposit €925,926 at 3.0% for a year: €925,926 × 1.03 = **€953,704**.
- **Path B, step 3.** Convert €953,704 back to dollars at the forward rate F. For Path B to equal Path A, we need €953,704 × F = \$1,050,000, so F = 1,050,000 ÷ 953,704 = **1.1010**.

The forward rate that makes the two paths tie is EUR/USD **1.1010** — higher than the spot 1.0800. The euro buys *more* dollars forward than spot; the dollar buys *fewer* euros forward than spot. That is the 2-point rate gap showing up as a price. The lesson in one line: the forward is not a forecast — it is the only number that stops Path B from being free money.

![Two paths to hold dollars for a year, stay in dollars versus detour through euros, that must pay equally](/imgs/blogs/interest-rate-differentials-the-master-variable-of-fx-3.png)

### The no-arbitrage loop, drawn as a circle

The cleanest way to *see* covered interest parity is to walk the loop as a closed circle and insist that you end exactly where you started. Borrow dollars, convert to euros at spot, deposit the euros, and convert back at the forward rate you locked at the outset. If the loop returns even one cent more than you owe, you have a riskless money pump; if it returns less, you run the loop backwards. Either way, the forward rate is forced to the one level where the circle closes with zero profit.

![The covered interest parity loop, borrow dollars convert to euros deposit and convert back](/imgs/blogs/interest-rate-differentials-the-master-variable-of-fx-2.png)

The figure traces it. Borrow \$1,000,000 at 5.0%, so you owe \$1,050,000 in a year. Convert to €925,926 at spot 1.0800. Deposit at 3.0% to reach €953,704. Sell those euros forward today at rate F. In a year, convert back: €953,704 × F dollars. The whole loop is hedged — you took no view on where the rate would go — so the only way the market permits it is if the dollars you get back exactly equal the \$1,050,000 you owe. Set €953,704 × F = \$1,050,000 and out pops F = 1.1010, the same number as before. The arbitrage isn't a strategy you'd run; it is the *enforcer* that keeps the forward honest.

This is the same equality as the two-paths story, just dressed as a single circular trade rather than two parallel investments. The two-paths version asks "would you be indifferent between holding dollars and detouring through euros?" The loop version asks "could you borrow, detour, and come back with a profit?" They are logically identical — both pin the forward to the level where there is neither indifference-breaking gain nor a free profit — but the loop is the one a real arbitrage desk actually executes, which is why it is the more honest picture of what *enforces* the price. No human believes in covered interest parity the way one believes a forecast; it holds because a desk in London, Tokyo and New York stands ready to run this exact loop the instant the forward strays a basis point from fair.

#### Worked example: what happens if the forward is mispriced

Suppose a sloppy dealer quotes the one-year EUR/USD forward at 1.0900 instead of the fair 1.1010. Run the loop and harvest the error.

- Borrow \$1,000,000 at 5.0%; you will owe **\$1,050,000**.
- Convert to euros at spot 1.0800: **€925,926**.
- Deposit at 3.0%: **€953,704** in a year.
- Sell those euros forward *now* at the mispriced 1.0900: €953,704 × 1.0900 = **\$1,039,537**.

That returns \$1,039,537 against a \$1,050,000 debt — you'd *lose* \$10,463, so you run the loop the other way instead: borrow euros, convert to dollars, deposit at the high US rate, and buy euros back forward at the cheap 1.0900. That direction nets roughly \$10,400 of riskless profit per million. Arbitrage desks would slam that quote shut in seconds, dragging the forward back to 1.1010. The takeaway: any forward away from the CIP level is a standing invitation to a money pump, which is why, in normal markets, the forward sits glued to the no-arbitrage value.

### The CIP formula, finally

Now name the pattern. For a pair quoted as BASE/QUOTE (like EUR/USD, where EUR is base and USD is quote), covered interest parity says:

```
F = S * (1 + r_quote) / (1 + r_base)
```

Here S is the spot rate, r_quote is the interest rate on the *quote* currency (USD), and r_base is the rate on the *base* currency (EUR). Plug in: F = 1.0800 × (1.05 ÷ 1.03) = 1.1010. The currency in the numerator's denominator — the base, the euro, the lower-rate leg — comes out *dearer* forward. The mechanical rule that falls out is worth carving in stone: **the higher-rate currency trades at a forward discount; the lower-rate currency trades at a forward premium.** The dollar pays more, so the dollar is worth fewer euros in the future. The reward for holding the high-rate currency over the period is exactly given back in the forward price. Nothing is free.

For short tenors people often use the simpler linear approximation, which makes the gap explicit:

```
F is approximately S * (1 + (r_quote - r_base) * T)
```

with T the fraction of a year. The forward moves off spot by (roughly) the spot times the **rate gap** times the time. The rate gap is right there in the formula, doing all the work. Everything else is spot and a clock.

A note on which rate goes where, because the BASE/QUOTE bookkeeping is where careful people still slip. The rule is mechanical: the *quote* currency's rate sits in the numerator, the *base* currency's rate in the denominator. For EUR/USD the quote is the dollar, so the dollar's 5.0% goes on top and the euro's 3.0% on the bottom, pushing F *above* spot. For USD/JPY the quote is the yen, so the yen's rate goes on top and the dollar's on the bottom, pushing F *below* spot. The two examples look like they disagree about whether the high-rate currency's forward rises or falls — but they don't; they only differ in which currency is being *quoted*. Strip away the quoting convention and the economics is invariant: **the currency that pays more interest is worth less in the future, by exactly the interest it paid.** Always sanity-check the *direction* against that sentence, never against the formula's surface.

### Reading the gap across many countries at once

The rate gap is a comparison between two specific countries, but it helps to see the whole field laid out, because every currency pair is really a pick of two bars off the same chart. The wider the distance between the two you pick, the harder the rate channel pulls on that pair; pick two countries sitting at nearly the same level and the rate gap barely registers, leaving the pair to be driven by other forces.

![Ten year government bond yields by country end 2024 sorted as horizontal bars](/imgs/blogs/interest-rate-differentials-the-master-variable-of-fx-8.png)

At end-2024 the ten-year benchmark yields ranged from Switzerland at just 0.30% and Japan at 1.07% up through Germany at 2.36%, Canada at 3.23%, Australia at 4.36%, the UK at 4.57% and the US at 4.58%. Read any pair as the *distance* between two of these bars. A US-versus-Japan trade rides a gap of about 3.5 points (4.58% − 1.07%) — wide, and so the yen carries a fat forward premium and the dollar a fat discount. A US-versus-Switzerland trade rides an even wider ~4.3 points. But a US-versus-UK trade rides almost nothing (4.58% versus 4.57%): the forward points are near zero, and you would not expect the rate channel to dominate GBP/USD at all — that pair will answer to growth, flows and risk sentiment instead. The chart is a map of where the rate pull is strong and where it is weak. (These are *bond* yields, which fixed-income owns in depth; for how the whole curve is read, see [The yield curve explained: the most important chart in finance](/blog/trading/fixed-income/the-yield-curve-explained-the-most-important-chart-in-finance).)

## The rate gap sets the forward exactly — and the forward points are its fingerprint

The difference between the forward and the spot rate has a name traders use a hundred times a day: **forward points**. Forward points = forward − spot. They are not a separate quantity with a life of their own; they are the rate gap, translated into the units of the pair. A pair with a big rate gap has big forward points. A pair with two countries at identical rates has forward points of essentially zero — the forward sits right on top of spot.

Why call them "points"? Because dealers quote them in the pair's smallest increment — pips — rather than as a full price. A trader won't say "the one-year EUR/USD forward is 1.1010"; they'll say "the one-year is *plus 210 points*," meaning add 0.0210 to whatever spot is right now. Quoting the *adjustment* instead of the *level* is not a quirk — it is a tell about what the forward actually is. Spot wanders around all day; the forward points barely move, because they are pinned to two slow-moving interest rates. By quoting the points, dealers separate the fast part of the price (spot) from the slow part (the rate gap). When you hear a forward quoted as "+210 points" or "−640 points," you are hearing the rate gap read aloud.

Let me switch to USD/JPY, because it makes the sign vivid and ties back to the cover chart. Here the dollar is the *base* and the yen is the *quote*, so the CIP formula flips which rate sits where.

![Forward points curve for USD JPY computed from covered interest parity by tenor](/imgs/blogs/interest-rate-differentials-the-master-variable-of-fx-6.png)

#### Worked example: USD/JPY forward points

Spot USD/JPY is 150.00 (one dollar buys 150 yen). The US one-year rate is 5.0%; the Japanese one-year rate is 0.5%. For a BASE/QUOTE pair, F = S × (1 + r_quote) ÷ (1 + r_base), and here quote = JPY (0.5%), base = USD (5.0%):

- F = 150.00 × (1.005 ÷ 1.05) = 150.00 × 0.95714 = **143.57 yen per dollar**.
- Forward points = 143.57 − 150.00 = **−6.43 yen**.

The dollar buys *fewer* yen forward (143.6) than spot (150.0). The dollar earns the high rate (5.0%), so it trades at a forward *discount* — exactly the same rule as the euro example, just with the dollar now playing the high-rate role. The yen, earning almost nothing, trades at a forward *premium*. If you sold dollars forward and bought yen forward at 143.6, and a year later spot was unchanged at 150.0, you'd profit — but you'd have given up the 4.5-point interest carry to get there. The forward already paid you the gap up front. The intuition: forward points are just the rate gap with a price tag.

The chart above plots those forward points across tenors, all computed from the same CIP arithmetic. They scale with time: a one-month forward gives back about a twelfth of the gap, a one-year forward the whole gap, a three-year forward roughly three times over. The fingerprint of a wide rate gap is a steep forward-points curve sloping away from spot. When a trader glances at the forward points, they are reading the rate gap directly off the screen — the two are the same fact wearing different clothes.

Notice what the *shape* of that curve tells you. A forward-points curve that fans out sharply with tenor is the signature of a wide, persistent rate gap — the longer you go out, the more interest the high-rate currency has earned, and the bigger the discount it must give back. A flat forward-points curve sitting near zero means the two countries' rates are roughly equal at every maturity. And a curve that is steep at the front but flattens further out is telling you the *market expects the gap to narrow* — because the forward at each tenor compounds the rate gap *for that tenor*, and if longer rates already price in convergence, the far points stop growing. The forward-points curve is, in this sense, a picture of the market's expected *path* for the rate gap, not just its level today. Reading it is reading two yield curves differenced against each other.

This is also why the FX swap market — by far the biggest slice of all currency trading — is really a money market in disguise. An FX swap is a near leg plus an offsetting far leg, and the price difference between the two legs *is* the forward points, which *is* the rate gap. Banks fund themselves across currencies through this market, which is why the cross-currency basis (a topic for a later post) is watched as a dollar-funding stress gauge. For the full plumbing of how spot, forwards and swaps fit together, see [Spot, forward and swap: the three ways to trade a currency](/blog/trading/forex/spot-forward-and-swap-the-three-ways-to-trade-a-currency); for the bond-market view of what those forward rates encode, see [Forward rates: what the market expects rates to be](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be).

It is worth sitting with how big this is, because it changes how you read the FX market. Most people assume currency trading is tourists and exporters swapping spot. But the largest single chunk of the roughly \$7.5 trillion *per day* that flows through global FX is not spot at all — it is FX swaps, which are pure rate-gap instruments. A bank with surplus dollars and a need for euros for a week does not "buy euros"; it does an FX swap, lending dollars and borrowing euros, and the price it pays or receives is the forward points, i.e. the rate gap for that week. In other words, the single biggest activity in the biggest market on Earth is the trading of interest-rate differentials wearing a currency costume. The rate gap isn't a side-show in FX; it is most of the volume.

#### Worked example: the FX swap is a loan priced by the gap

A US bank has \$1,000,000 idle and needs euros for one year. Instead of buying euros outright (taking currency risk), it does a one-year EUR/USD swap: convert at spot now, reverse at the forward later.

- **Near leg.** Hand over \$1,000,000, receive €925,926 at spot 1.0800.
- **Far leg.** In a year, hand back the euros and receive dollars at the forward 1.1010.
- **What it cost.** The bank effectively lent dollars and borrowed euros. The forward (1.1010 vs spot 1.0800) means it gets back more dollars per euro than it paid — compensation for having lent the 5.0% dollars while only owing on 3.0% euros. The 2-point gap is the *interest* on this disguised loan.

No net currency position was ever taken — the near and far legs cancel. The only economic content is the funding cost, and that cost *is* the rate differential. The intuition: an FX swap is a collateralized cross-currency loan, and its price is the rate gap, full stop.

## Why the high-rate currency trades cheaper forward

This is the single most counterintuitive thing in FX, and it is worth a section to itself, because almost everyone gets it backwards on the first pass. Surely the *strong*, high-yielding currency should be worth *more* in the future, not less?

No — and the reason is that the interest you earn and the forward price are two sides of the same coin. They cannot both reward you. If holding dollars pays you 2 points more than holding euros over the year, then a market with no free lunches must take those 2 points back somewhere. It takes them back in the forward: the dollar buys fewer euros forward than spot. The total return from a fully hedged position is identical whichever currency you hold — that is the *entire content* of covered interest parity. The extra yield on the high-rate currency is exactly offset by its forward discount. There is no edge in a covered position; the gap is a wash.

![The higher-rate currency trades at a forward discount, dollar versus euro before and after](/imgs/blogs/interest-rate-differentials-the-master-variable-of-fx-7.png)

The figure lays the two currencies side by side. The dollar (5.0%) earns the high rate and pays it back as a forward discount: it buys fewer euros forward, EUR/USD rising from 1.0800 to 1.1010. The euro (3.0%) earns the low rate and is rewarded with a forward premium: it buys more dollars forward. The 2-point gap shows up as a 2.1-cent move from spot to forward, and it nets to zero economic advantage for anyone who hedges. The strong-currency-must-rise instinct is wrong because it double-counts: you cannot collect both the interest *and* a favorable forward.

#### Worked example: the hedged return is identical either way

Prove the "no edge" claim directly. You have \$1,000,000 to park for a year and you fully hedge any currency exposure. Compare holding it in dollars versus in euros.

- **Hold dollars.** \$1,000,000 × 1.05 = **\$1,050,000**. Return: 5.0%.
- **Hold euros, hedged.** Convert at spot 1.0800 → €925,926. Earn 3.0% → €953,704. Convert back at the forward 1.1010 → €953,704 × 1.1010 = **\$1,050,000**. Return: 5.0%.

Identical to the dollar. The euro paid 2 points *less* in interest, but its 2-point forward *premium* made up the difference precisely. The hedged investor is indifferent between the two currencies — which is the whole point of covered interest parity. The intuition: hedging converts every currency's return into the same number, so a covered position can never harvest the rate gap.

The reason this matters so much is that it tells you exactly where any *edge* must come from. If the hedged bet is a guaranteed wash, then every dollar of profit anyone has ever made trading currencies on the rate gap came from *not hedging* — from keeping the interest carry and accepting the spot risk. The gap is not a free lunch; it is a paid wager. CIP doesn't say "currencies don't move with rates"; it says the *risk-free* slice of that move is already priced into the forward, and what is left over — the part that can actually pay you — is risk you chose to keep.

Here is the crucial pivot, and it is what makes the rest of the series interesting. CIP says the *covered* (hedged) position has no edge. But what if you *don't* hedge? What if you borrow cheap yen, buy high-yielding dollars, and simply leave the position exposed to wherever spot drifts? Now you keep the full interest carry, and your only risk is that spot moves against you. That uncovered bet is the **carry trade**, and whether it pays is the question of whether the high-rate currency *actually* falls by as much as the forward predicted. History says: usually less, sometimes catastrophically more. The systematic failure of the forward to predict the future spot is the **forward premium puzzle**, and it is the engine of carry. We build CIP here so that those two posts have a foundation — see [Uncovered interest parity and why it fails: the forward puzzle](/blog/trading/forex/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle) and [The carry trade: getting paid to hold a currency](/blog/trading/forex/the-carry-trade-getting-paid-to-hold-a-currency).

To see why the distinction is the whole ballgame, lay the two bets side by side. The covered investor sells the high-rate currency forward at 1.1010, locking in a 5.0% dollar return no matter what spot does — a guaranteed wash against simply holding dollars. The uncovered investor *doesn't* sell forward; they hold the high-rate currency naked and collect the 2-point yield, betting that spot does *not* drift down to 1.1010 over the year. If spot ends the year unchanged at 1.0800, the uncovered investor pockets the full 2 points the covered investor gave away. If spot falls all the way to 1.1010 (the high-rate currency depreciating exactly as the forward "predicted"), the two bets tie. And if spot falls *past* 1.1010 — the high-rate currency weakening more than the forward said — the uncovered investor loses, potentially badly. The forward rate is, in effect, the *breakeven* line for the carry trade. Everything in FX speculation comes down to whether spot will end up above or below the forward — which is to say, whether the high-rate currency will fall by less or more than the rate gap.

Decades of data say it usually falls by *less* — high-rate currencies tend to *not* depreciate as much as CIP forwards imply, so carry pays on average. But "on average" hides a vicious skew: long stretches of small, steady gains punctuated by sudden, violent losses when everyone unwinds at once, exactly as happened to the yen carry trade in August 2024. That asymmetry — pick up pennies for years, lose dollars in an afternoon — is *the* defining feature of currency speculation, and it falls straight out of the gap between the covered world (where the forward is law) and the uncovered world (where it is merely a bet). This post builds the covered floor; the carry and uncovered-parity posts build the dangerous structure on top of it.

## Why the gap moves: the rate-gap is the head of the causal chain

So far the rate gap has been a fixed input — 5.0% versus 3.0%. In the real world it is the most fast-moving, news-sensitive number in macro, because it is set by two central banks making decisions on their own schedules. When one bank hikes and the other holds, the gap blows out; when they converge, it collapses. And because the gap sits at the *head* of the causal chain, every move in it ripples all the way to spot.

![How a central bank rate decision widens the rate gap and reaches the spot rate](/imgs/blogs/interest-rate-differentials-the-master-variable-of-fx-5.png)

The chain runs left to right. A central bank moves its **policy rate** — the rate it pays banks to park money overnight, which anchors all the short rates in that currency. That shifts the **rate gap** against the other country. Capital, always hunting for yield, **chases** the higher rate: money flows into the high-rate currency and out of the low-rate one. That flow is real buying and selling of the currency, so it pushes the **spot rate**. And separately, CIP instantly **re-prices the forward** off the new gap. The gap is not one factor among many — it is the source that the others flow from. (For the full transmission mechanism, the macro-trading desk owns it: [How monetary policy moves currencies: rate differentials](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials).)

There is a subtle but crucial split inside that chain. The forward re-prices *mechanically and instantly* — it has no choice, because CIP is arithmetic and the new rates are known the moment they're set. The *spot* move is different: it is the result of *flows*, of real money deciding to chase the higher yield, and flows are a matter of behavior, not arithmetic. This is why the forward is rock-solid (CIP holds to a basis point) while the spot is noisy and only *tends* to follow the gap. Two channels, two speeds. The forward channel is a law; the spot channel is a strong, reliable, but breakable tendency. Keeping them separate in your head is the difference between understanding why a forward is what it is (always, exactly) and why a spot rate is roughly where the gap says it should be (usually, approximately).

And the spot channel works through more than one mechanism, which is worth spelling out:

- **The carry incentive.** A wider gap makes it more attractive to borrow the low-rate currency and hold the high-rate one. As long as the trade is on, it is a steady buy-pressure on the high-rate currency. This is the flow that built the yen's multi-year slide.
- **The expectations channel.** Markets are forward-looking, so spot moves on the *expected future path* of the gap, not just today's level. A central bank that merely *signals* future hikes can move its currency before it touches rates at all. This is why forward guidance is a currency tool.
- **The valuation re-rating channel.** Higher rates change the present value of every future cash flow in that economy — bonds, equities, real estate — and global investors rebalancing into or out of those assets buy or sell the currency to do it. The rate decision ripples through the whole asset complex, and the currency is the entry and exit door.

All three push the same way for a widening gap, which is why the rate channel is usually the dominant force on spot over a horizon of months — even though, day to day, it gets drowned out by noise.

![Policy rates by central bank, Fed BoE ECB RBA SNB BoJ, 2023 peak versus end 2024](/imgs/blogs/interest-rate-differentials-the-master-variable-of-fx-4.png)

The grouped bars show where the gap *comes from*: the policy rate each major central bank parked at. At end-2024 the Fed sat at 4.50% while the Bank of Japan had crept up to just 0.25% — a gap of more than four points that is the entire reason the dollar carries a large forward discount against the yen. Notice the Reserve Bank of Australia at 4.35% and the Bank of England at 4.75%, both close to the Fed; a US-versus-UK or US-versus-Australia trade rides a far thinner gap than a US-versus-Japan trade. The width of the gap is set by the *distance between two bars*, and that distance is what a currency pair orbits.

#### Worked example: pricing the forward off a live gap

Take the end-2024 policy rates and price a one-year USD/JPY forward off them. Fed at 4.50%, BoJ at 0.25%, and say spot is at 157.00 (roughly where USD/JPY closed 2024). Using the linear approximation, forward ≈ spot × (1 + (r_jpy − r_usd) × T):

- gap = 0.25% − 4.50% = **−4.25 points** (the quote, yen, earns far less than the base, dollar).
- forward ≈ 157.00 × (1 + (−0.0425) × 1) = 157.00 × 0.9575 = **150.3 yen per dollar**.

So the one-year forward sits about 6.7 yen *below* spot. The dollar's 4.25-point yield advantage is handed straight back as a forward discount of roughly 4.3%. If the gap later narrowed — say the BoJ hiked toward 1.0% — that forward discount would shrink, and the forward would creep back toward spot. The takeaway: you can read tomorrow's forward off today's two policy rates, because the gap between them *is* the forward, scaled by time.

A few things move the gap, and they are worth naming because they are what a currency trader actually watches:

- **Relative inflation.** A central bank facing higher inflation hikes more, widening its rate gap and (over the medium term) lifting its currency on the carry side — even as inflation erodes the currency's purchasing power on the [purchasing-power-parity](/blog/trading/forex/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle) side. These two forces fight; the rate channel usually wins over months, PPP over decades.
- **Relative growth.** Strong growth lets a central bank keep rates high; a recession forces cuts. The gap tracks the *relative* growth cycle of the two economies.
- **Policy surprises.** The gap reprices not on the rate decision itself but on the *surprise* versus what was already priced. The August 2024 yen move was a BoJ hike that markets had under-priced; the gap's *expected path* shifted, and the pair convulsed.

For the deeper machinery of how central banks set and signal these rates, the macro-trading and fixed-income desks carry it: [Central bank toolkit: rates, QE, QT, forward guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance).

There is also a real/nominal distinction lurking here that separates beginners from professionals. The *nominal* rate gap (5.0% versus 3.0%) is what sets the forward through CIP — that part is pure arithmetic and uses the quoted rates as they are. But the flows that move *spot* often respond to the *real* (inflation-adjusted) rate gap, because what a global investor actually earns is the interest minus the inflation eating into the currency's purchasing power. A country with a high nominal rate but even higher inflation has a *negative* real rate, and money may flee it despite the headline yield. This is why a currency can have the highest interest rate in the world and still collapse: Turkey through much of the 2018–2023 period ran enormous nominal rates while the lira fell relentlessly, because real rates were deeply negative and the carry never compensated for the depreciation. The nominal gap prices the forward; the real gap shapes the flows. Most of the time they point the same way, but when they diverge, the real gap usually wins the spot fight.

## Common misconceptions

**"The strong, high-yielding currency rises in the forward market."** It falls. The dollar earning 5.0% against a euro earning 3.0% trades at a forward *discount* — EUR/USD goes from 1.0800 spot to 1.1010 forward, meaning the dollar buys fewer euros forward. The forward gives back exactly the interest advantage. If you remember one thing from this post, make it this: high rate → forward discount.

**"The forward rate is the market's forecast of where spot will be."** No. The forward is a *no-arbitrage price* derived mechanically from two interest rates; it contains no view about the future. The market's forecast may be anything. The systematic gap between the forward and the actual future spot is precisely the forward premium puzzle that powers the carry trade — if the forward *were* an unbiased forecast, carry would not pay. The forward is arithmetic; the forecast is opinion; they are different objects.

**"Covered interest parity is a textbook idealization that doesn't hold in real markets."** It holds astonishingly tightly — within a basis point or two in normal times — precisely because violations are arbitraged away in seconds. It *does* break in dollar-funding crises (2008, March 2020), when balance-sheet constraints stop banks from running the arbitrage; the residual is the **cross-currency basis**, which spiked to −150 basis points and beyond in those episodes. But the basis being watched as a stress gauge is the exception that proves the rule: the whole reason a −50bp basis is alarming is that CIP normally holds to within a couple of bp. And note *why* it broke: not because the economics changed, but because post-2008 bank regulation made it costly to hold the balance sheet the arbitrage requires. The free lunch was still on the floor; the rules just made it expensive to bend down and pick it up. That is a profound lesson — a "law" of finance is only as strong as the capital willing to enforce it.

**"Forward points are a fee the bank charges."** They are not a fee — they are the rate gap, and they can be in your favor. If you sell a low-yielding currency forward (buy the high-yielder forward), the points work against you; if you do the reverse, they pay you. The points are symmetric and fair; the dealer's actual fee is the much smaller bid-ask spread layered on top.

**"A pair with two countries at similar rates won't move."** The rate gap sets the *forward* and pulls on flows, but it is not the only force on spot. Two countries at identical rates (US and UK at end-2024, both near 4.6% on the 10-year) have near-zero forward points, yet GBP/USD still moves on growth surprises, the current account, risk sentiment and the dollar's global role. The gap is the master variable, not the *only* variable — it dominates the forward exactly and dominates spot *often*, but spot also answers to flows the gap doesn't capture.

## How it shows up in real markets

**USD/JPY, 2021–2024: the cleanest textbook case in modern markets.** Walk the cover chart again with the mechanism in hand. In 2020–21 the Fed and the BoJ were both near zero, the US-Japan two-year gap was essentially nil, and USD/JPY sat around 103–115. Through 2022 the Fed hiked at the fastest pace in forty years while the BoJ refused to move; the two-year gap exploded from near zero to about 4.35 points. The yen collapsed to 131, then 144, then 157 by end-2024. The pair tracked the gap almost line for line, because the gap was both re-pricing the forwards and pulling capital toward dollars. This is the rate differential as master variable, in its purest form.

Put rough numbers on the carry side of that move to feel why it was so powerful. A trader who borrowed yen at roughly 0.1% and held dollars at roughly 5% earned about 4.9 percentage points a year in pure interest carry — *and* watched the dollar appreciate from 103 to 157 yen, a further ~52% capital gain over the period. Carry plus a trending spot is the dream combination, and it is exactly what a widening rate gap produces: the gap pays you to hold the position *and* pushes the spot your way. That is why the trade became so crowded — and why, when the gap's expected path turned in 2024, the exit was so violent. Crowded carry plus a turning gap is the recipe for an unwind.

**August 2024: the gap's *expected path* matters as much as its level.** The BoJ's tiny hike on 31 July 2024 barely moved the *level* of the gap. What it moved was the market's belief about the *future* gap — the sudden sense that Japan was finally starting a hiking cycle while the Fed was about to cut. The expected convergence of the gap was enough to trigger a violent unwind of years of carry positioning, dragging USD/JPY from about 162 to 142 in days. The episode is a reminder that markets price the *path* of the gap, not just today's snapshot. The cross-asset and macro desks dissect the unwind itself: [Carry-trade unwinds 1998, 2008, 2024: when leverage breaks](/blog/trading/macro-trading/carry-trade-unwinds-1998-2008-2024-when-leverage-breaks).

**EUR/USD and the 2022 Fed-ECB divergence.** When the Fed hiked aggressively in 2022 and the ECB lagged, the US-euro rate gap widened and EUR/USD fell below parity (under 1.00) for the first time in two decades — the euro buying less than a dollar. As the ECB caught up through 2023 and the gap narrowed, EUR/USD recovered toward 1.10. The same gap, the same mechanism, a different pair. The lesson generalizes: you do not need a different theory for each currency. EUR/USD, USD/JPY, GBP/USD, AUD/USD — all four obey the same CIP arithmetic in the forward and the same rate-pull on spot. What differs is only *which* two central banks are diverging, and by how much. Learn to read one pair as a bet on a gap and you have learned to read them all.

**The mirror image: when the gap closes.** The mechanism runs in reverse just as cleanly. Whenever a high-rate central bank starts cutting toward a low-rate peer, the gap narrows, the carry incentive fades, the forward discount shrinks, and the high-rate currency tends to give back its gains. Traders watch the *expected* convergence of policy paths as closely as the current gap, because spot front-runs the change. A pair that has trended for two years on a widening gap can reverse hard the moment the market becomes convinced the gap has peaked — which is exactly the dynamic that detonated the yen carry trade in 2024.

**When CIP cracked: September 2008 and March 2020.** In the 2008 crisis and the March 2020 COVID panic, banks outside the US scrambled for dollars and could not get them at the CIP-implied rate. The cross-currency basis blew out — the effective cost of borrowing dollars via the FX swap market ran 100–200 basis points *above* what CIP said it should — because balance-sheet limits stopped arbitrageurs from closing the gap. The Fed had to open dollar swap lines with other central banks to relieve the squeeze. CIP didn't *fail* as economics; it failed as *arbitrage* because the arbitrage capital wasn't there. That distinction — the relationship is real, but enforcing it needs willing balance sheets — is one of the deepest lessons in the plumbing of the dollar system: [The dollar: cross-asset gravity](/blog/trading/cross-asset/the-dollar-cross-asset-gravity).

**Reading a screen like a pro.** Pull up any forward on a dealer terminal and you can reverse-engineer the rate gap it implies. Spot USD/JPY 150, one-year forward 143.6 → the dollar is at a 4.3% forward discount → the implied gap is about 4.3 points → that matches a Fed near 5% and a BoJ near 0.5%. The forward *is* the gap. Once you see that the forward points and the rate differential are the same number, the screen stops being a wall of quotes and becomes a map of where every central bank is sitting relative to every other.

You can run the trick the other way, too, and it is how professionals sanity-check markets in real time. If you know the two interest rates, you know what the forward *should* be — so if a forward is quoted away from that level, either you have the rates wrong, or there is a funding stress (a cross-currency basis) pricing in. During the March 2020 dash for dollars, traders watched forwards that implied dollar borrowing costs *far* above the Fed's policy rate and read it correctly as a balance-sheet emergency, not a mispricing to arbitrage. The forward, read against the rate gap, is one of the cleanest real-time stress gauges in all of finance — precisely because it is normally pinned so tightly to the rates.

## The takeaway: read every pair as a bet on a gap

Step back to the spine of this whole series: an exchange rate is the relative price of two monies, and you never own a currency in isolation — every position is a *pair*, a relative bet. This post sharpens that into a single operational claim. The thing you are *really* betting on, in every currency position, is the **gap between two interest rates** — its level today, and, more than that, where it is heading.

That reframes how you read the market. A currency does not have a value; a *pair* has a price, and that price orbits the rate gap. When you see a pair move, the first question is not "what's the news?" but "what just happened to the gap, or to the market's view of the gap's path?" When you want to know if a forward is cheap, you don't guess — you compute the CIP value from two interest rates and compare. When you want to know why the high-yielder is the one trading at a forward discount, you remember that the interest and the forward are the same reward seen twice, and a market with no free lunch can only pay it once.

Here is a concrete checklist for reading any pair through the rate-gap lens, so the idea becomes a habit rather than a theory:

- **Identify the two policy rates.** Which two central banks set this pair? Where is each parked, and which way is each leaning? That tells you the gap's level and its likely direction.
- **Read the forward points as the gap.** Pull the forward, subtract spot, and you have the rate differential priced in. A big number means a wide gap; a near-zero number means two countries at similar rates. The sign tells you which currency is the high-yielder.
- **Separate the law from the tendency.** The forward *is* the gap, exactly, always — that part you can compute and trust. Spot only *leans* on the gap, through flows that can be overwhelmed by risk sentiment, the dollar's global role, or a current-account story. Don't confuse the two.
- **Watch the expected path, not just the level.** The pair moves on changes in the *anticipated* gap. A central bank that signals a turn can move its currency before it touches rates. The August 2024 yen move was a shift in the expected path, not the level.
- **Check nominal versus real.** The nominal gap prices the forward; the real (inflation-adjusted) gap shapes the flows. A high nominal yield with negative real rates is a trap, not a carry.

Run that checklist and a currency screen stops being a fog of numbers and becomes a readout of where every central bank sits relative to every other — which is, in the end, what an exchange rate *is*.

And the deepest payoff is the door this opens. Covered interest parity says the *hedged* bet has no edge — the gap is a wash. So every interesting question in FX is really the question of what happens when you *stop hedging*: when you keep the carry and take the spot risk. Does the high-rate currency fall as much as the forward said it would? Usually not — until, one August afternoon, it falls far more. That asymmetry is where the money and the danger both live. But you cannot understand the uncovered bet, the carry trade, or the crashes that punctuate them without first nailing the covered one. That is what we did here: we made the rate gap the master variable explicit, and we proved — from nothing but a refusal to leave free money on the table — that the forward price is just spot, adjusted by the gap. Master that, and you are no longer guessing at why a currency moved. You are reading the gap.

## Further reading & cross-links

- [Uncovered interest parity and why it fails: the forward puzzle](/blog/trading/forex/uncovered-interest-parity-and-why-it-fails-the-forward-puzzle) — what happens when you drop the hedge: the forward is *not* an unbiased forecast, and that bias is the puzzle that powers carry.
- [The carry trade: getting paid to hold a currency](/blog/trading/forex/the-carry-trade-getting-paid-to-hold-a-currency) — the uncovered bet on the rate gap, its steady grind and its violent crashes.
- [Spot, forward and swap: the three ways to trade a currency](/blog/trading/forex/spot-forward-and-swap-the-three-ways-to-trade-a-currency) — the instruments that turn the rate gap into a tradeable price.
- [How monetary policy moves currencies: rate differentials](/blog/trading/macro-trading/how-monetary-policy-moves-currencies-rate-differentials) — the full transmission mechanism from a central-bank decision to the spot rate.
- [Forward rates: what the market expects rates to be](/blog/trading/fixed-income/forward-rates-what-the-market-expects-rates-to-be) — the bond-market view of the forward curve that FX forwards ride on.
- [Central bank toolkit: rates, QE, QT, forward guidance](/blog/trading/macro-trading/central-bank-toolkit-rates-qe-qt-forward-guidance) — how the policy rate that anchors the gap actually gets set and signalled.

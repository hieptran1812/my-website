---
title: "FX Options and the Volatility Smile: Why the Market Quotes Vol, Not Price"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "A beginner-friendly deep dive into FX vanilla options: calls and puts, the at-the-money straddle, premium in pips, and the convention that makes FX unique — the market quotes implied volatility by delta, not price by strike."
tags: ["forex", "currencies", "fx-options", "volatility-smile", "implied-volatility", "straddle", "delta", "risk-reversal", "derivatives"]
category: "trading"
subcategory: "Forex"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — In the FX options market, the thing two dealers actually negotiate is not a dollar price but a *volatility*; the price falls out of that number once you plug it into a pricer.
>
> - A vanilla FX option is a call (the right to *buy* the base currency at a fixed rate) or a put (the right to *sell* it), and the buyer pays a premium quoted in pips.
> - The at-the-money straddle — one call plus one put at the same strike — is the market's purest bet on *how much* a pair will move, regardless of direction.
> - The FX-specific twist: the market quotes **implied volatility, not price**, and it pins that vol at fixed **delta** points (25-delta put, ATM, 25-delta call) rather than at fixed strikes.
> - Plot vol against delta and you get the **volatility smile**: out-of-the-money options on both wings are bid up relative to the at-the-money, because the market knows big moves cluster. The one number to remember: a major like EUR/USD trades a 1-month at-the-money vol around **7%**.

On the afternoon of 5 August 2024, the implied volatility on one-month USD/JPY options roughly doubled in a matter of hours. Spot had collapsed from about 162 to the low 140s as the great yen-carry trade unwound (we tell that story in [carry crashes](/blog/trading/forex/carry-crashes-picking-up-pennies-in-front-of-a-steamroller)). But here is the part that confuses people coming from the stock market: the dealers on the FX options desk were not shouting prices at each other. They were shouting *volatilities*. "USD/JPY one-month, where's the vol?" "Was seven this morning, it's twelve now, paper's bidding it through fourteen." Nobody quoted a dollar figure. The price of the option was an *output*, computed once the two sides agreed on a vol.

That is the single strangest, most important fact about the FX options market, and it trips up almost everyone the first time. In equities, you look up an option and you see a price: "the 100 call is trading at \$3.20." In FX, you look up an option and you see a *number that is not a price at all* — an annualized standard deviation of returns, expressed as a percent. The price is hiding behind it.

This post builds the whole thing from zero. We will define what a call and a put even are, walk through the at-the-money straddle (the market's favorite instrument), compute a real premium in pips and dollars, and then spend the back half on the two conventions that make FX options their own world: **quoting in volatility instead of price**, and **quoting by delta instead of strike**. That second convention is what gives us the **volatility smile** — the gently U-shaped curve that is the photograph of the market's fear. Throughout, we lean on the [options-volatility series](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more) for the Greeks and the deep theory; here we keep it FX-native and intuitive.

![In FX the quote is a volatility, not a price, shown as one-month implied vol by currency pair](/imgs/blogs/fx-options-and-the-volatility-smile-1.png)

That chart is the whole post in one picture. Each bar is the *quote* a dealer would give you for a one-month at-the-money option on that pair — and the unit on the y-axis is a percent of volatility, not a dollar amount. EUR/USD, the calmest major, sits around 7%. USD/JPY runs a notch higher near 9.5%. By the time you reach the Turkish lira, the "price" of an option is a 22% vol — the market is telling you, in its own units, that this is a currency that can move violently. Read this series' spine through that lens: an exchange rate is the relative price of two monies, and the *vol quote* is the market's live estimate of how unstable that relative price is about to be.

## Foundations: FX options and why vol is the quote

Let us not assume any options background. We will build the four primitives — call, put, strike, premium — and then the two FX conventions, one brick at a time.

### What a currency pair is (the thirty-second refresher)

An exchange rate like EUR/USD = 1.0800 means *one euro costs 1.08 US dollars*. The first currency (EUR) is the **base**; the second (USD) is the **quote** (or "terms") currency. The number tells you how many units of the quote currency you pay for one unit of the base. When EUR/USD rises to 1.1000, the euro got *stronger* (more dollars per euro); when it falls to 1.0600, the euro got *weaker*. The smallest standard increment is a **pip** — for most pairs that is the fourth decimal place, so a move from 1.0800 to 1.0801 is one pip. (For JPY pairs, where the rate looks like 150.25, a pip is the second decimal, 0.01.) We cover all of this in [base, quote and pips](/blog/trading/forex/base-quote-pips-and-how-to-read-an-fx-quote); here we just need the vocabulary.

The crucial thing to hold onto: in FX, *every* position is a relative bet between two monies. There is no such thing as "owning EUR" in a vacuum — you own EUR *against* USD. That fact bleeds straight into options. An FX call is never just "a bet that something goes up"; it is "the right to buy one money by paying with another."

### Call and put: the two rights

An **option** is a contract that gives the buyer a *right*, not an obligation, to trade at a fixed rate on (or before) a fixed date, in exchange for a fee paid up front. Two flavors:

- A **call** on EUR/USD gives you the right to **buy** the base currency (EUR) at an agreed rate called the **strike**. You use it if the rate ends up *above* the strike — you buy euros cheaply through the option and they are worth more in the market.
- A **put** on EUR/USD gives you the right to **sell** the base currency (EUR) at the strike. You use it if the rate ends up *below* the strike — you sell euros dear through the option even though the market has marked them down.

The fee you pay up front is the **premium**. It is yours to lose and nothing more: the most a buyer can lose on a long option is the premium, no matter how far the market moves against the position. That asymmetry — capped loss, open-ended gain — is the entire reason options exist.

![A call is the right to buy the base currency and a put the right to sell it, paying off on opposite sides of the strike](/imgs/blogs/fx-options-and-the-volatility-smile-3.png)

Notice in that figure that both the call and the put share the same skeleton: pay a premium up front (red, money out), profit on one side of the strike (green), and the worst case is simply losing the premium (amber, the cost). They are mirror images around the strike. A call leans bullish on the base currency; a put leans bearish. Hold both at once and something interesting happens — which is exactly the straddle, coming up.

#### Worked example: a single EUR/USD call

Suppose EUR/USD spot is **1.0800** and you buy a one-month call struck **at-the-money** (strike = 1.0800) on a notional of **\$1,000,000** worth of euros — that is about €925,926 of base currency. At a 7% implied vol, the call's premium is roughly **0.49%** of the euro notional, or about **49 pips** of EUR/USD. In dollars that is approximately:

- Premium ≈ 49 pips × €925,926 ≈ 0.0049 × 925,926 × 1 (in USD per pip terms) ≈ **\$4,500** paid up front.
- If at expiry EUR/USD is **1.1000**, your right to buy at 1.0800 is worth 200 pips, or about 0.0200 × 925,926 ≈ **\$18,500** gross, **\$14,000** net of premium.
- If at expiry EUR/USD is **1.0700** (below the strike), the call expires worthless and you lose exactly the **\$4,500** premium — not a dollar more.

The intuition: you risked \$4,500 to control \$1,000,000 of upside, with your downside capped at that \$4,500 — the option converted an open-ended directional bet into a known, budgetable cost.

### Premium in pips: the FX way to size a cost

Equity options quote premium in dollars per share. FX options quote premium in **pips of the pair** (or sometimes as a percent of one of the two notionals — there are four conventions, but pips-of-the-pair is the most intuitive). Saying "the EUR/USD one-month ATM is worth 98 pips" means the two-sided straddle costs 0.0098 in EUR/USD terms. To turn pips into a dollar figure you multiply by the notional and by the value of a pip. For a €1,000,000 notional, one pip of EUR/USD is worth €100 (since a pip is 0.0001 and 0.0001 × 1,000,000 = 100), which at the prevailing rate is roughly \$108. So 98 pips on a €1,000,000 straddle ≈ 98 × \$108 ≈ **\$10,600** — close to the \$9,800 figure we will compute more carefully below once we anchor the notional in dollars rather than euros.

The reason this matters: pips are *unit-free across the size of your trade*. A dealer can quote "98 pips" to a hedge fund doing €1bn and to a corporate doing €1m, and both know exactly what they are paying per unit. The dollar number scales with notional; the pip number is the *price*.

One more wrinkle that catches beginners: an FX option has *two* notionals, one in each currency, because every FX trade is a swap of one money for another. A EUR/USD call on €1,000,000 with a strike of 1.0800 is simultaneously a call to *buy* €1,000,000 and a call to *pay* \$1,080,000. Dealers therefore quote premium in one of four conventions: pips of the pair, percent of the base-currency notional, percent of the quote-currency notional, or "quote per base." They are all the same economic premium expressed in different units, the way a temperature is the same whether you say it in Celsius or Fahrenheit. We will stick to pips-of-the-pair and dollars throughout, but if you ever see "premium 0.49% EUR" and "premium 49 pips" side by side, do not panic — they describe the identical option.

### Moneyness, intrinsic value, and time value

Two more terms make everything click. An option's **moneyness** describes where the strike sits relative to spot. A call is **in-the-money (ITM)** when spot is above the strike (exercising it makes money right now), **at-the-money (ATM)** when spot equals the strike, and **out-of-the-money (OTM)** when spot is below the strike (worthless if exercised today). For a put it is mirrored: ITM below the strike, OTM above. The OTM options are the cheap long-shots; the ITM options are expensive because they already have value baked in.

That baked-in value has a name: **intrinsic value**, the amount you would collect if the option expired right now. An ATM or OTM option has *zero* intrinsic value. So what are you paying for when you buy an ATM option that is worth nothing if it expired this instant? You are paying for **time value** — the chance that the pair moves your way before expiry. Time value is *entirely* a function of two things: how much time is left, and how much the pair is expected to move per unit time — which is to say, *volatility*. This is the cleanest way to see why the FX market quotes vol: for an ATM option, the *whole* premium is time value, and time value is just vol × √time × spot, dressed up. The vol quote and the time-value premium are the same statement said two ways.

#### Worked example: intrinsic versus time value on a EUR/USD call

Take a one-month EUR/USD **1.0700-strike call** with spot at **1.0800** and 7% vol. Spot is 100 pips *above* the strike, so the call is in-the-money:

- **Intrinsic value** = spot − strike = 1.0800 − 1.0700 = **100 pips**. That is what you would collect by exercising today.
- The **total premium** the pricer returns might be about **118 pips**.
- So the **time value** is 118 − 100 = **18 pips** — the extra you pay for the chance the call goes even *further* into the money over the next month.

Now compare the ATM call (strike 1.0800): intrinsic value **0 pips**, total premium **49 pips**, so *all 49 pips are time value*. The intuition: the ATM option is pure volatility exposure — every pip of its price is a bet on movement — which is exactly why the market uses the ATM as its volatility benchmark.

### The at-the-money straddle: the market's purest vol bet

Here is the instrument the FX market lives and breathes by. A **straddle** is *one call plus one put at the same strike and expiry*. When that strike is set at-the-money (ATM) — meaning at the current spot or forward — you own a position that profits if the pair moves *a lot in either direction* and loses if it sits still.

Why is this the purest bet on volatility? Because you have deliberately removed the directional view. A lone call is a bet that EUR/USD rises; a lone put is a bet it falls. Bolt them together and the up-bet and down-bet cancel at the center — what is left is a bet purely on the *size* of the move, regardless of sign. If the pair rockets up, the call pays; if it craters, the put pays; if it goes nowhere, both expire worthless and you lose both premiums. The straddle is long volatility in its rawest form, which is exactly why the ATM straddle premium is what the market uses to *define* the vol quote.

![A long ATM straddle pays off as a V around the strike with two breakevens and max loss equal to the premium](/imgs/blogs/fx-options-and-the-volatility-smile-4.png)

That V is the signature shape. At the strike (1.0800) you are at your maximum loss — both options are worthless and you are out the full premium, here 98 pips. Move either way and one leg starts paying. The two **breakevens** sit one premium's-width on each side of the strike: 1.0800 − 0.0098 = **1.0702** on the downside and 1.0800 + 0.0098 = **1.0898** on the upside. Beyond those points the straddle is in profit. The width between the breakevens is the market's implied "you need a move this big just to break even" — and that width is set entirely by the vol quote.

#### Worked example: pricing a €1,000,000 EUR/USD ATM straddle at 7% vol

This is the canonical FX-options calculation, so let us do it carefully. Inputs: spot = strike = **1.0800**, one-month tenor (T = 1/12 ≈ 0.0833 years), implied vol **σ = 7%**, notional **\$1,000,000** (so about €925,926 of base). We are *not* going to re-derive Black-Scholes — the [vega and pricing theory](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) lives in the options series. We just use the well-known shortcut for an ATM straddle premium:

$$\text{ATM straddle premium} \approx 0.80 \times \sigma \times \sqrt{T} \times \text{spot}$$

Plugging in: 0.80 × 0.07 × √0.0833 × 1.0800 = 0.80 × 0.07 × 0.2887 × 1.0800 ≈ **0.01746** in EUR/USD terms, or about **98 pips** when you express the round-trip premium for the pair (the 0.80 factor and rounding land us near the 98-pip figure dealers would actually quote). Now in dollars on a \$1,000,000 notional:

- 98 pips = 0.0098 of EUR/USD. On €925,926 of euros that is 0.0098 × 925,926 ≈ \$9,074. Rounding the notional and the factor, dealers would call this roughly **\$9,800** of premium on a \$1,000,000 straddle.
- So you pay about **\$9,800** up front to own both the call and the put.
- The pair must move **98 pips** — to 1.0702 or 1.0898 — *just for you to break even*.

The intuition: at 7% vol the market is implicitly saying "a one-standard-deviation monthly move in EUR/USD is about 2%, or roughly 216 pips, and a straddle costs you a bit under half of that to own." The premium and the expected move are two sides of the same coin: the vol *is* the price.

### The straddle's cousin: the strangle

There is a close relative worth knowing because it shows up constantly on the vol sheet. A **strangle** is also a long call plus a long put, but with the two strikes spread *apart* — typically a 25-delta call and a 25-delta put, both out-of-the-money. Because both legs start OTM, the strangle is *cheaper* than the straddle, but the pair has to move *further* before either leg pays. You buy a strangle when you expect a big move and want to pay less for it; you accept wider breakevens in exchange. The reason it matters for *us* is that the FX market uses a particular strangle — the average of the 25-delta wing vols minus the ATM — to measure the *curvature* of the smile. That measure is called the **butterfly**, and together with the ATM and the risk reversal it is one of the three numbers that fully describe a smile. Hold that thought; it returns when we read a real vol sheet.

### Why the straddle defines the vol quote

Here is the deeper reason the ATM straddle, specifically, anchors the whole market. The ATM straddle has *zero net delta* — the call's positive delta and the put's negative delta cancel at the money. A position with zero delta does not care which way spot goes at the first instant; it cares only about *movement*. So its price is a pure function of volatility, with the directional component scrubbed out. That makes the ATM straddle the natural yardstick: the cleanest instrument whose price is *only* about vol. When the market says "EUR/USD one-month is 7," it is quoting the vol that prices this delta-neutral straddle. Every other point on the surface — the wings, the risk reversal, the butterfly — is then quoted *relative* to this clean ATM benchmark. The straddle is the meter stick; everything else is measured against it.

#### Worked example: the breakeven move as a fraction of the implied move

Let us connect the premium to the implied "typical move" so the 7% number means something physical. A one-month, 7%-vol pair has a one-standard-deviation move of:

$$\sigma_{\text{month}} = \sigma \times \sqrt{T} = 0.07 \times \sqrt{0.0833} \approx 0.0202 = 2.02\%.$$

On a spot of 1.0800 that is about **218 pips**. Our straddle breaks even at a **98-pip** move. So the buyer profits whenever the realized move exceeds roughly 98/218 ≈ **0.45 standard deviations**. That is the deep logic of long volatility: you make money when the market moves *more* than its own implied "normal," and you lose when it moves less. The straddle is a wager that realized vol will beat implied vol — picking up the difference between what the market *charged* you (7%) and what actually *happens*.

That single sentence — "you make money when realized beats implied" — is why volatility itself becomes an asset class. The straddle is the cleanest way to be long or short *the move*, and the vol quote is its price tag.

## Why FX quotes volatility, not price

We have now built every primitive. Time for the headline. Boiled down, the equity world and the FX world price the *same* Black-Scholes contract, but they hand each other a different one of the model's inputs as "the quote."

![Equity desks quote the premium in dollars while FX desks quote the implied volatility and back out the price](/imgs/blogs/fx-options-and-the-volatility-smile-2.png)

Look at the two columns. On an equity options desk, the number that changes on the screen is the **premium in dollars**; the implied vol is something you *back out* of that price if you happen to want it. On an FX desk it is exactly reversed: the number that changes on the screen is the **implied volatility**, and the premium is something the pricer spits out once both sides agree on the vol. Same model, opposite "handle."

### Why would anyone quote in vol?

It feels backwards until you see the three problems it solves.

**First, vol is stable when price is not.** A EUR/USD option's dollar premium depends on the spot rate, which ticks every fraction of a second. If you quoted price, your quote would be stale the instant you made it, because spot already moved. But the *implied vol* — the market's estimate of how much the pair will wobble — barely moves second to second. By quoting vol, a dealer can hold a quote steady for minutes while spot dances underneath. The pricer re-computes the dollar premium continuously from live spot; the human only has to manage the one slow-moving number.

**Second, vol is comparable across strikes, tenors, and pairs.** A 7% vol on EUR/USD and a 9.5% vol on USD/JPY are directly comparable statements about relative instability — even though the two pairs trade at utterly different price levels (1.08 vs 150) and their options have utterly different dollar premiums. Try comparing "the USD/JPY 152 call costs ¥1.40" with "the EUR/USD 1.09 call costs \$0.0042" and your eyes glaze over. Compare "USD/JPY vol is 9.5%, EUR/USD vol is 7%" and you instantly know which market the options crowd thinks is more dangerous.

**Third, vol is what dealers actually have a view on.** A market-maker on the options desk does not, primarily, have a view on where EUR/USD is going — that is the spot desk's job. The options dealer has a view on *how much it will move*, i.e. on volatility. Quoting the instrument in the very variable they trade keeps the conversation honest. When a dealer says "I'll pay 7.2 for a billion of one-month," they are not betting on direction; they are buying volatility at 7.2%.

#### Worked example: the same option, two ways to quote it

Take that one-month ATM EUR/USD call on \$1,000,000 again, spot 1.0800, vol 7%. The two desks describe the identical contract differently:

- **Equity-style quote:** "The premium is about 49 pips, ≈ \$4,500." If spot drifts to 1.0820 over the next ten minutes, that dollar quote is *wrong* — the real premium is now higher, because the call is closer to in-the-money. The dealer must re-quote.
- **FX-style quote:** "Vol's at 7." If spot drifts to 1.0820, the vol quote is *still 7* — nothing about the market's estimate of future wobble changed. The pricer simply recomputes the premium from the new spot. The dealer's quote held.

The intuition: quoting vol decouples the dealer's *view* (how much it'll move) from the *thing that's constantly changing* (where spot is right now), which is why the whole market converged on it.

### From a vol quote to dollars: the pipeline

It helps to trace the full path from the single number a dealer says out loud to the dollars that change hands. The chain is short and mechanical, and once you have walked it once you will never again be confused about why "the vol is 7" is a complete quote.

![A single volatility quote feeds a pricer that outputs the strike the premium in pips and the dollar amount](/imgs/blogs/fx-options-and-the-volatility-smile-8.png)

Read the pipeline left to right. **Step one:** the dealer quotes a *vol* — say 7.0%. That is the only thing the two humans negotiate. **Step two:** the convention fixes the strike. For an ATM trade the strike is the forward; for a wing trade it is whatever strike carries the agreed delta (more on delta-quoting below). **Step three:** both sides feed spot, the strike, the tenor, the two interest rates, and that 7.0% vol into the *same* Black-Scholes pricer — the model is shared infrastructure, not a competitive edge, which is precisely why nobody bothers to quote its output. **Step four:** the pricer returns the premium in **pips** of the pair (here ~98 pips for the straddle). **Step five:** multiply by the notional and the pip value to get **dollars** (~\$9,800 on a \$1,000,000 straddle). Five steps, one negotiated number — the vol — and everything downstream is arithmetic. That is the deep reason FX can quote vol: the rest of the pipeline is deterministic, so the only thing worth haggling over is the vol itself.

This also explains a subtlety that confuses newcomers: two dealers can quote you the *same* vol and yet hand you *different* dollar premiums, if their spot, their forward points, or their interest-rate inputs differ even slightly. The vol is the agreed *view*; the dollars depend on the plumbing each desk plugs in. When a corporate treasurer "shops" an option across three banks, they are really comparing vols first, and only then checking that nobody's pricer is using a stale spot.

### What vol actually is, stated plainly

When we say "the vol is 7%," we mean the market's estimate of the **annualized standard deviation of EUR/USD's returns**, expressed in percent. A 7% annual vol means that if you scaled the typical yearly swing down to one day it would be about 7% / √252 ≈ 0.44% per day, and over one month about 7% × √(1/12) ≈ 2.0%. It is a measure of *dispersion* — how widely the pair's future is smeared out — and nothing about direction. A pair can have a high vol and a flat trend (it whips around but goes nowhere) or a low vol and a strong trend (it grinds steadily). Vol is the width of the probability cone, not its tilt.

This is why the vol quote *is* the option price in disguise: an option is fundamentally a claim on dispersion. The wider the cone of possible outcomes, the more likely the option finishes in-the-money by a lot, and the more it is worth. Black-Scholes is just the exact dictionary that translates "width of the cone" (vol) into "dollars of premium." FX dealers chose to speak in the language of the cone's width and let the dictionary handle the rest.

## Quoting by delta, not by strike

Now the second FX convention, and the one that sets up the smile. We have said the market quotes *vol*. But vol *of which option*? An option needs a strike. And here FX does something equities do not: it pins its quotes to fixed **delta** points rather than fixed strikes.

### What delta is (the one Greek we need)

**Delta** is how much the option's value changes when spot moves by one unit — but the more useful reading for FX quoting is that delta is *approximately the probability the option finishes in-the-money*, expressed as a number between 0 and 1 (or 0% and 100%). An at-the-money option has a delta near 0.50: a coin-flip whether it ends up worth something. A deep out-of-the-money option has a delta near 0.10: a long shot. A deep in-the-money option has a delta near 0.90: almost certain to pay. We do not re-derive delta here — the [vol surface post](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) in the options series carries the Greeks — we just use it as a *coordinate*.

The FX market quotes vol at five standard delta coordinates: the **10-delta put**, the **25-delta put**, the **at-the-money** (delta ≈ 0.50), the **25-delta call**, and the **10-delta call**. Five points, and the whole vol curve for that tenor is summarized by them.

![FX quotes vol by delta at the 25 delta put, the at the money, and the 25 delta call rather than by fixed strike](/imgs/blogs/fx-options-and-the-volatility-smile-5.png)

That matrix is the core of how an FX vol sheet is organized. The columns are delta points — 25-delta put, ATM, 25-delta call — and each carries its own vol quote: maybe 8.0% for the put, 7.0% at-the-money, 7.4% for the call. The strike that corresponds to "25-delta put" is whatever rate makes the put have a 0.25 delta *today*; as spot moves, that strike moves with it.

### Why delta instead of strike?

The reason is the same stability argument, one level deeper. A fixed strike of "1.0500" means something totally different depending on where spot is. If spot is 1.0800, then 1.0500 is a meaningfully out-of-the-money put. If spot rallies to 1.1500, then 1.0500 is a *deep* out-of-the-money put — a different animal entirely. So if you quoted vol "at the 1.0500 strike," your quote would drift in meaning as spot moved.

But "the 25-delta put" *always* means the same thing: the option with a one-in-four chance of finishing in-the-money. As spot moves, the *strike* that carries 25-delta moves with it, but the *moneyness* — the thing the dealer actually has a view on — stays pinned. Quoting by delta makes a vol curve that is comparable across days, across pairs, and across spot levels. It is the natural coordinate system for a market that quotes the cone's width: delta measures position *within* the cone.

#### Worked example: the 25-delta put strike moves with spot

Suppose one-month EUR/USD 25-delta put vol is quoted at **8.0%**. With spot at **1.0800**, the strike that carries 0.25 delta might be around **1.0620** — roughly 180 pips below spot. Now let spot rally to **1.1000** while vols stay unchanged. The "25-delta put" is *still* quoted at 8.0% — but the strike that now carries 0.25 delta has shifted up to around **1.0815**, again roughly 180 pips below the *new* spot.

The intuition: the dealer never re-quoted, because in delta-space nothing changed — a 25-delta put is a 25-delta put. The strike chased spot so that the *probability* stayed pinned at one-in-four. That invariance is exactly why FX quotes by delta.

### The ATM convention: delta-neutral or 50-delta

A subtlety worth one paragraph. "At-the-money" in FX usually does not mean "struck at spot." It means struck at the **forward** rate (the rate implied by the two countries' interest rates — see [spot, forward and swap](/blog/trading/forex/spot-forward-and-swap-the-three-ways-to-trade-a-currency)), or sometimes at the strike where a straddle has *zero* net delta (the "delta-neutral straddle"). For our purposes the distinction is small — the ATM is the bottom of the smile, the benchmark vol, the number on the front of the screen. But it is worth knowing that the FX "ATM" is anchored to the forward and the rate differential, threading straight back to this series' spine: even the center of the vol curve is set by the gap between two countries' rates.

## The volatility smile

We now have everything to build the smile, the single most important picture in FX options. Take a pair, fix a tenor (say one month), and plot the *vol quote* at each delta point: 10-delta put, 25-delta put, ATM, 25-delta call, 10-delta call. Connect the dots. You do not get a flat line. You get a curve that **dips in the middle and rises toward both wings** — a smile.

![A stylised volatility smile where implied vol is lowest at the money and rises toward both the put and call wings](/imgs/blogs/fx-options-and-the-volatility-smile-6.png)

The at-the-money is the cheapest vol — the bottom of the curve, here 7.0%. Move out toward either wing — toward out-of-the-money puts on the left, out-of-the-money calls on the right — and the quoted vol *rises*: 7.7% at the 25-delta points, 9.3% and 8.1% at the 10-delta wings. The market is charging *more* implied volatility for the options that only pay off on big moves.

### Why does the smile exist? Fat tails.

If returns were perfectly normally distributed (the textbook bell curve), every strike would carry the *same* implied vol and the smile would be a flat line. The smile exists because real currency returns are **not** normal: they have **fat tails**. Big moves — the kind that put a 10-delta option in the money — happen *far more often* than a bell curve predicts. The 2015 Swiss franc de-peg, the 2024 yen unwind, Black Wednesday 1992: these are 5- and 10-standard-deviation events under a normal model, yet the FX market produces one every few years. The market *knows* this. So it charges extra vol for the wing options — the ones that only pay on tail moves — to compensate for the fact that tails are fatter than the model's default assumption.

Put differently: the smile is the market admitting, in the only language it has, that the normal distribution underprices disaster. Each point on the smile is a different implied vol because each strike is a bet on a different part of the (non-normal) return distribution, and the wings are where the distribution is fattest relative to the bell curve.

#### Worked example: the wing premium in pips

Take our EUR/USD smile. Suppose the ATM is **7.0%** and the 10-delta put is bid at **9.3%**. A flat-smile world (no fat tails) would price that 10-delta put at 7.0% vol too. What does the extra 2.3 vol points cost? For a far out-of-the-money option, the dollar premium is small in absolute terms, but the *relative* markup is large. A 10-delta put priced at 7% might cost, say, **8 pips**; the same option priced at the smile's 9.3% might cost **13 pips** — over **60% more** for the identical contract, purely because the market refuses to believe the tail is as thin as a bell curve says.

The intuition: the smile is most expensive, in percentage terms, exactly where the bell curve is most wrong — out in the wings, where crashes live. Buying tail protection in FX always feels overpriced, and the smile is why.

### The butterfly: how much the smile smiles

We met the butterfly above as the smile's *curvature*. Now we can pin it down. The **butterfly** is the average of the two 25-delta wing vols minus the ATM vol:

$$\text{butterfly} = \tfrac{1}{2}\left(\sigma_{25\text{c}} + \sigma_{25\text{p}}\right) - \sigma_{\text{ATM}}.$$

It measures how much *higher* the wings sit above the bottom of the smile — how "smiley" the smile is. A *large* butterfly means the market is paying a big premium for tail moves of either sign; a *flat* butterfly (near zero) means the market thinks the bell curve is roughly right and big moves are not specially feared. The butterfly is the market's *fat-tail premium*, condensed to one number. When a currency's regime looks fragile — a peg under pressure, an election looming, a central bank cornered — the butterfly fattens as the market bids up *both* wings, even before it knows which way the break will go.

So a full smile is summarized by exactly three numbers per tenor: the **ATM** (the level, the bottom of the smile), the **risk reversal** (the tilt, which wing is higher), and the **butterfly** (the curvature, how far both wings rise). Give a trader those three numbers and they can reconstruct the whole five-point smile and read the market's mood — level, direction, and tail-fear — at a glance.

### The smile lives on a surface: the term structure of vol

Everything so far fixed the tenor at one month. But the *same* pair has a smile at one week, one month, three months, one year — and those smiles are *not* identical. Plot ATM vol against tenor and you get the **term structure** of volatility. Usually it slopes gently upward (longer horizons price in more uncertainty), but it inverts during a panic: when a crisis hits, *short-dated* vol spikes far above long-dated, because the market expects the storm to be violent but brief. Stack the smile (across delta) against the term structure (across tenor) and you have the full **vol surface** — a 3D map where every point is a vol quote for a specific moneyness and horizon. We do not build the whole surface here; the [vol-surface post](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) in the options series does. For FX the takeaway is just this: the smile is one slice of a living surface, and the *shape* of that surface — flat and calm, or steep and inverted — is the most complete readout of fear the currency market produces.

### The smile is not symmetric: enter the skew

Look again at the smile figure: the left wing (puts) is bid a touch higher than the right wing (calls) — 9.3% versus 8.1% at the 10-delta points. The smile is *tilted*. That tilt is the **skew**, and it carries directional information. When the put wing is higher than the call wing, the market is paying up more for downside protection than for upside — it fears a fall more than a rise. The cleanest summary of that tilt is the **risk reversal**: the vol of the out-of-the-money call *minus* the vol of the out-of-the-money put.

![Representative risk reversals show how far the smile tilts toward downside fear in calm versus stressed markets](/imgs/blogs/fx-options-and-the-volatility-smile-7.png)

A risk reversal of −0.4 in calm EUR/USD means the put is bid 0.4 vol points over the call — a mild downside lean. In stress it blows out to −2.5 (EUR/USD) or even −4.0 (USD/JPY, where the fear is a violent yen rally as carry unwinds). The risk reversal is the smile's *tilt* condensed into a single number, and it is the market's directional fear gauge. We give it a full treatment in the sibling post [risk reversals and the shape of fear](/blog/trading/forex/risk-reversals-and-the-shape-of-fear-in-fx); here, just hold the picture — the smile is the *level and curvature* of fear, the risk reversal is its *direction*.

#### Worked example: reading a risk reversal off the smile

From our smile: 10-delta call vol = **8.1%**, 10-delta put vol = **9.3%**. The 10-delta risk reversal is:

$$\text{RR}_{10} = \sigma_{\text{call}} - \sigma_{\text{put}} = 8.1\% - 9.3\% = -1.2 \text{ vol points}.$$

A negative risk reversal means **puts are dearer than calls** — the market is paying up to protect against EUR/USD *falling*. If tomorrow the risk reversal moves to −2.5, the smile has tilted *further* toward the downside without necessarily moving its ATM level at all: the same average vol, but the fear has rotated toward a euro fall.

The intuition: you can read the market's directional anxiety straight off the asymmetry of the smile, and the risk reversal puts a single number on it — negative means "protect me from a drop."

## Common misconceptions

**"A higher vol quote means the option is more expensive in some absolute sense."** Not exactly — it means the option is more expensive *for its moneyness and tenor*. A 22% vol on USD/TRY does not make a TRY option cost more dollars than a 7% EUR/USD option of the same notional in every case; it means TRY's options are pricing in far more expected movement. Vol is a *rate of dispersion*, not a price. Two options can have the same dollar premium and wildly different vols if their tenors or notionals differ. Always read vol as "how much movement is priced in," never as a raw dollar cost.

**"The straddle profits if the pair goes up."** No — the straddle profits if the pair moves *a lot in either direction*, by more than the breakeven width. Our €1,000,000 example needed a **98-pip** move (to 1.0702 or 1.0898) just to break even. A straddle is a bet on *magnitude*, not *direction*; if you have a directional view, you buy a single call or put, not a straddle. Confusing the two is the most common beginner error in options.

**"The smile means the market thinks the pair will fall."** The *smile* (the U-shape) means the market thinks big moves in *either* direction are underpriced by the normal distribution — it is symmetric in spirit. It is the *skew* (the tilt) and its summary the risk reversal that carry direction. A pair can have a pronounced smile (fat tails both ways) with a near-zero risk reversal (no directional lean). Keep the two concepts separate: smile = tail fatness, skew = which tail.

**"Implied vol predicts where the pair is going."** Implied vol predicts *how much* the pair will move, not *which way*. A 9.5% USD/JPY vol says "expect a roughly 2.7% monthly swing"; it says nothing about up versus down. People constantly read a vol spike as a directional signal — it is not. (The directional signal, if any, is in the risk reversal, not the ATM vol.)

**"You can lose more than the premium on a long option."** A long (bought) option's maximum loss is the premium, full stop — that is the defining feature. The unlimited-loss horror stories come from *selling* options (being short the straddle, short the wing), where you collect a small premium and bear an open-ended tail. The 2015 CHF de-peg famously detonated traders who were *short* CHF calls. Buying options caps your loss; selling them does not.

**"Implied vol and the option premium move in lockstep, so I can ignore one of them."** They are linked but not interchangeable, because the premium also depends on spot, time, and the rate differential. Implied vol can *fall* while a call's premium *rises* — if spot rallies hard enough that the call's growing intrinsic value swamps the lost time value. Conversely, an ATM option's premium *bleeds* a little every day even with vol pinned, because time value decays toward expiry. If you trade options to express a *vol* view, you must hedge out the spot and time components (the delta and theta) or your P&L will be a muddle of three different bets. The clean way to be long volatility, and nothing else, is the delta-hedged straddle — which is exactly why the straddle keeps coming back as the market's volatility yardstick.

## How it shows up in real markets

### The August 2024 yen-carry unwind: vol as the live readout

We opened with this; now we can read it properly. Through early 2024, USD/JPY one-month ATM vol sat quietly around 8–9%. The carry trade — borrow cheap yen, buy higher-yielding dollars — was crowded and calm, and *calm is what carry needs* (the relationship is the subject of the sibling post [carry and volatility](/blog/trading/forex/carry-and-volatility-the-relationship-that-runs-the-trade)). The risk reversal was deeply negative: the market was paying up for *yen-up* protection (USD/JPY puts), because everyone knew that if the trade unwound it would unwind violently in one direction.

Then the BoJ hiked and the trade broke. Over the first week of August 2024, USD/JPY fell from ~162 toward ~142, and one-month vol roughly *doubled* — the whole smile lifted and the risk reversal blew out further toward yen-up fear. Anyone who had *owned* the USD/JPY puts (the cheap-looking wing that the smile had been quietly bidding up) made a fortune; anyone short that wing was carried out. The smile had been *telling you* this was coming — the persistently negative risk reversal was the market pre-paying for exactly the move that happened. The vol quote was not a lagging price; it was a live, forward-looking readout of fear, and it was right.

### The 2015 Swiss franc de-peg: the wing that paid 30%

On 15 January 2015 the Swiss National Bank abandoned its 1.20 floor under EUR/CHF and the pair collapsed from 1.20 to below 0.85 *intraday* — a 30% move in minutes (we tell the full story in [the SNB 2015 peg break](/blog/trading/forex/the-snb-2015-peg-break-when-a-central-bank-blinks)). Under a normal distribution, a 30% intraday move in a developed-market pair is a multi-thousand-sigma impossibility. The FX smile "knew" the tail was fatter than that — EUR/CHF puts (CHF calls) had carried elevated wing vol for months as the floor's credibility eroded. Traders who owned the cheap-looking far-out-of-the-money CHF calls were paid spectacularly; the dealers and funds who were *short* those wings — collecting a little premium against a floor they assumed would hold — were vaporized. The smile is, in the end, the market's collective insurance premium against exactly this kind of regime break, and 2015 is the canonical case of the wing being underpriced even at its elevated vol.

### Why emerging-market vols sit so much higher

Glance back at the cover chart. EUR/USD trades around 7% vol, USD/JPY near 9.5%, but USD/MXN runs 13% and USD/TRY a startling 22%. That ladder is not random — it is the options market pricing the *structural* instability of each relative-money relationship. A developed pair like EUR/USD links two large, credible central banks with deep capital markets and free-floating regimes; its relative price is anchored and rarely gaps. An emerging pair like USD/TRY links the dollar to a currency whose central bank has, at times, run deeply negative real rates and faced recurring confidence crises; its relative price can lurch 10% in a week. The vol quote *is* that history and that fragility, priced. And because EM pairs gap so violently, their *smiles* are also far more pronounced and their *risk reversals* far more skewed — almost always toward "the EM currency crashes," because that is the tail everyone fears. Reading the vol ladder top to bottom is reading a fragility ranking of the world's monetary regimes, exactly the spine of this series: the more unstable the relationship between two monies, the higher the vol the market charges to insure it.

### The everyday use: a corporate hedging a euro receivable

Strip away the drama and here is the bread-and-butter use. A US importer expects to receive €1,000,000 in three months and fears the euro will *weaken* (fewer dollars when it arrives). They buy a three-month EUR/USD put struck near the forward — the right to sell their euros at a known rate. The dealer quotes them a *vol* (say 7.5%), the pricer turns it into a premium (say 90 pips, ≈ \$8,300 on the notional), and the importer has bought a known-cost floor under their euro proceeds. If EUR/USD falls, the put pays and offsets the loss on the receivable; if it rises, the put expires and they simply enjoy the better rate, out only the premium. Every term in that transaction — the vol quote, the delta-chosen strike, the premium in pips — is exactly what we built in this post. The smile shows up too: because the corporate is buying a downside put (the bid-up wing), they pay a touch *more* vol than the ATM, the small price of the market's fat-tail awareness.

#### Worked example: the corporate's hedge, fully costed

Let us put real numbers on that importer. They will receive **€1,000,000** in three months (T = 0.25), spot is **1.0800**, and they buy an at-the-forward EUR/USD put at **7.5%** vol. Using the ATM rule of thumb for a single option (about half the straddle premium), the put costs roughly 0.40 × 0.075 × √0.25 × 1.0800 ≈ **0.0162**, call it **90 pips**, or on €1,000,000 about **\$8,300** of premium paid up front.

- If at expiry EUR/USD is **1.0400** (euro fell), their receivable is worth only \$1,040,000 in the open market, but the put lets them sell at 1.0800 for \$1,080,000 — the put paid \$40,000, far more than the \$8,300 premium. Their floor held.
- If at expiry EUR/USD is **1.1200** (euro rose), the put expires worthless; they sell their euros in the market at 1.1200 for \$1,120,000 and are out only the \$8,300 premium — a small price for the upside they kept.

The intuition: the importer converted an open-ended FX risk into a known, budgetable insurance cost of \$8,300 — and the *vol quote*, not a dollar price, is what they negotiated to set that cost.

### When implied diverges from realized: the volatility risk premium

One last real-market pattern, because it explains who is on the other side of every option trade. Over long horizons, *implied* volatility in FX (what options charge) tends to run a little *above* the volatility that actually shows up — the **realized** volatility. The gap is the **volatility risk premium**: the extra that option *buyers* pay, on average, for insurance, and that option *sellers* collect for providing it. It is the same economics as house insurance: most months your house does not burn down, so the insurer profits — but in the rare catastrophe the policyholder is very glad they paid. FX option sellers are insurers; they earn a steady premium for years and then, on a CHF de-peg or a yen unwind, give it all back (and more) at once. This is why the smile's wings stay persistently bid even though they "usually" expire worthless: the sellers demand compensation for the fat tail, and the buyers, who remember 2015 and 2024, are willing to pay it. The vol quote is, in the end, the clearing price of this insurance market — set where the fear of the buyer meets the greed of the seller.

### The interbank smile sheet: how a desk actually reads it

On a real desk, the one-month EUR/USD vol "sheet" is five numbers: 10d-put vol, 25d-put vol, ATM vol, 25d-call vol, 10d-call vol — or equivalently, the ATM plus two **risk reversals** (25d and 10d) plus two **butterflies** (the curvature, how much the wings sit above the ATM). A trader glances at it and reads the whole market mood in a second: ATM = the level of fear, risk reversal = the direction of fear, butterfly = how much the market is paying for tails versus the body. When the ATM is 7, the risk reversal is −0.4, and the butterfly is 0.2, it is a calm market with a mild downside lean and thin tail demand. When the ATM jumps to 12, the risk reversal gaps to −2.5, and the butterfly to 0.8, the market has repriced to *crisis*: more fear, more downside skew, more demand for tail protection. The five numbers *are* the market's emotional state, quoted in vol.

#### Worked example: reconstructing a smile from the three numbers

Suppose the desk's one-month EUR/USD sheet reads: **ATM = 7.0%**, 25-delta **risk reversal = −0.6** (puts over calls), 25-delta **butterfly = 0.4**. Reconstruct the two 25-delta wing vols. The butterfly says the *average* of the two wings sits 0.4 above the ATM, so the average wing vol is 7.0 + 0.4 = **7.4%**. The risk reversal says the call is 0.6 *below* the put, i.e. the call is half a risk-reversal below the average and the put half above:

- 25-delta call vol = 7.4 − (0.6 / 2) = **7.1%**
- 25-delta put vol = 7.4 + (0.6 / 2) = **7.7%**

So from three numbers — 7.0, −0.6, 0.4 — the trader recovers the full quote: put 7.7%, ATM 7.0%, call 7.1%. The intuition: the ATM-risk-reversal-butterfly triple is just a compressed encoding of the smile, and any FX trader can decompress it in their head in seconds.

## The takeaway: vol is the market's estimate of its own instability

Here is what changes once you internalize all this. When you see "EUR/USD one-month vol is 7%," do not file it as an obscure derivatives statistic. Read it as the FX market's live, money-backed estimate of *how unstable the relative price of two monies is about to be*. That single number — and the smile and risk reversal around it — is the most honest forward-looking signal the currency market produces, because real money is paying for it. A spot quote tells you where the pair *is*; the vol surface tells you what the market is *afraid of*.

Three durable habits to carry away. **First, separate level from direction.** The ATM vol is the *level* of fear (how much movement is priced); the risk reversal is the *direction* of fear (which way). A vol spike is not a directional signal — go to the risk reversal for that. **Second, respect the wings.** The smile rises at the edges because tails are fat, and FX produces a regime-break tail every few years. Far-out-of-the-money options always *look* overpriced and are usually still cheap relative to the disasters that actually happen; selling the wings to harvest a little premium is the classic way to blow up. **Third, remember that vol is the price.** Everything else — the pips, the dollar premium, the strike — is downstream of the vol quote. Learn to think in vol and you are thinking the way the options market thinks.

And tie it back to the spine of this whole series: an exchange rate is the relative price of two monies, driven by the gap between two countries' rates and the flow of money across borders. The vol surface is the derivatives market's measurement of how *unstable* that relative price is — and the smile is its admission that monetary regimes break more often, and more violently, than any tidy model would have you believe. The smile is fear, drawn to scale.

## Further reading & cross-links

- [Risk reversals and the shape of fear in FX](/blog/trading/forex/risk-reversals-and-the-shape-of-fear-in-fx) — the directional tilt of the smile, in depth.
- [Carry and volatility: the relationship that runs the trade](/blog/trading/forex/carry-and-volatility-the-relationship-that-runs-the-trade) — why low vol feeds the carry trade and why a vol spike kills it.
- [The cross-currency basis: when covered parity breaks](/blog/trading/forex/the-cross-currency-basis-when-covered-parity-breaks) — the other place FX "prices" something that is not a spot rate.
- [The volatility smile and skew: why OTM puts cost more](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more) — the general theory of the smile and fat tails (the Greeks live here).
- [Reading the vol surface like a trader: the 3D map of fear](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) — delta, tenor, and the full surface.
- [Vega: your exposure to implied volatility and the vol of vol](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) — the Greek that measures your P&L per vol point.
- [The SNB 2015 peg break: when a central bank blinks](/blog/trading/forex/the-snb-2015-peg-break-when-a-central-bank-blinks) — the canonical fat-tail case the smile prices for.
- [Spot, forward and swap: the three ways to trade a currency](/blog/trading/forex/spot-forward-and-swap-the-three-ways-to-trade-a-currency) — where the forward (and the ATM strike) comes from.

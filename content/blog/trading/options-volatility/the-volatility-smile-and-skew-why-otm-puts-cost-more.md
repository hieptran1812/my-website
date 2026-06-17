---
title: "The Volatility Smile and Skew: Why OTM Puts Cost More"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Why each strike trades at its own implied volatility, why downside puts are structurally rich, and how to read the skew so you stop pricing options off a single number."
tags: ["options", "volatility", "skew", "smile", "implied-volatility", "risk-reversal", "vol-surface", "tail-risk", "put-skew"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Textbook Black-Scholes assumes one volatility for every strike; the real market quotes a *different* implied vol at every strike, and on equity indices the downside is systematically more expensive. That tilt is the skew, and it is a price you pay or collect on every options trade whether you notice it or not.
>
> - **Each strike trades its own implied vol.** Plot implied vol against strike and you get a curve — a smile in FX, a downward "smirk" in equity indices — not the flat line the textbook model assumes.
> - **OTM puts are structurally rich.** In the representative SPX 30-day skew, the 90% put prints 22% implied vol while the 110% call prints 14.8%. Persistent demand for crash protection plus a supply of overwriting calls bids the left wing up and offers the right wing down.
> - **Choosing a strike is choosing a point on the vol curve.** The same 10%-out-of-the-money strike is a far more expensive bet on the put side than the call side, which reprices every spread, collar, and risk reversal you build.
> - **The number to remember:** at a flat 17% vol a 90% put is worth about \$0.021; at its true skew vol of 22% it is worth about \$0.103 — roughly five times as much. Price a put off the at-the-money number and you will be shocked at the fill.

A trader wants to hedge a \$100 stock. The plan is simple and sensible: buy a 30-day put 10% out of the money — the 90 strike — as crash insurance. The trader knows the option-pricing model, knows the stock's at-the-money implied volatility is running around 17%, and does the arithmetic in advance. Plug 17% into Black-Scholes for a 90-strike, 30-day put on a \$100 name and the model spits out about \$0.021 per share — call it two cents, \$2.10 per contract. Cheap insurance. The trader sizes up to buy a thousand contracts, expecting to pay around \$2,100.

The actual quote comes back at roughly **\$0.10 per share** — \$10 a contract, about \$10,000 for the thousand. Five times the estimate. The trader did nothing wrong with the model. The stock hadn't moved. Nothing about the company changed between the estimate and the quote. The model was simply fed the wrong number, because the market does not price that 90-strike put off the at-the-money 17%. It prices it off **22%**. The downside is dearer than the middle, and it is dearer for reasons that have held for almost forty years.

That gap — the fact that the 90 put trades at 22% while the at-the-money option trades at 17% and the 110 call trades at 14.8% — is the **volatility skew**. It is the single most important departure of real options markets from the clean theory, and it is the subject of this post. We will build it from zero: why a flat-vol world is a fiction, what shape the curve actually takes on different assets, *why* equity-index puts are structurally bid, how traders read and quote the skew, and — because this is a practitioner series — exactly how the skew reprices the trades you will actually put on.

![The equity-index volatility skew: implied volatility falling from 26% at the 85% strike to 14.5% at the 115% strike, with the at-the-money point marked at 17%](/imgs/blogs/the-volatility-smile-and-skew-why-otm-puts-cost-more-1.png)

We met implied volatility as one of the five inputs to an option's price in [what sets an option's price](/blog/trading/options-volatility/what-sets-an-options-price-the-five-inputs-and-the-intuition), and we treated it there, for simplicity, as a single number for the whole name. This post takes that simplification away. The honest statement is that there is no single implied volatility for a stock or an index; there is a *curve* of implied vols, one for every strike, and the shape of that curve is itself tradable information. Learning to read it is what separates someone who can price an at-the-money straddle from someone who can actually trade options across the strike ladder.

## Foundations: one model, one volatility — and why that's a lie

Start with what the textbook says, because the skew is best understood as a deviation from it. The Black-Scholes model — whose full derivation lives in [the Black-Scholes post](/blog/trading/quantitative-finance/black-scholes), and which this series does not re-derive — prices an option from five inputs: the stock price, the strike, the time to expiry, the risk-free rate, and **one** volatility number, the famous sigma. That sigma is assumed to be a constant: a single annualized standard deviation that describes how the stock will diffuse between now and expiry, identical for every strike and every expiry on the name.

Under that assumption, the math is clean. The stock's future price is **lognormally distributed** — its logarithm is a normal bell curve — and from that single bell curve you can compute the fair price of *every* option on the stock at once. Feed in the strike you care about, out comes the price. Crucially, if you then ran that machinery in reverse — took the model's own prices and backed out the implied volatility at each strike — you would get the same sigma back at every strike. A flat line. That is the world the textbook describes: one volatility, one bell curve, a flat implied-vol curve across strikes.

Now go look at a real options screen. Take the market prices of options across the strike ladder — the actual bids and offers traders are transacting at — and run Black-Scholes in reverse at each one: *what volatility would I have to plug in to reproduce this market price?* That number is the **implied volatility** of that strike. Do it for every strike and plot the answers against strike. You do not get a flat line. You get the curve in the figure above: high on the left (low strikes, the puts), sloping down through the at-the-money point, and lower still on the right (high strikes, the calls). The market is telling you, through its prices, that it does *not* believe in a single lognormal bell curve. Every strike is quoting its own opinion about volatility.

This is worth sitting with, because it sounds like a contradiction. Implied volatility is supposed to be a property of the *stock* — how much it will move. How can the stock have a different volatility "for the 90 strike" than "for the 110 strike"? It is the same stock with the same future. The resolution is that implied vol is not really a forecast of the stock's volatility at all. **It is a translation device** — a single number that converts an option's dollar price into a unit traders can compare across strikes and expiries. The market sets the *prices* of the options directly, by supply and demand, and implied vol is just what those prices look like after you pass them through the Black-Scholes lens. When the market pays up for downside puts, the implied vol of those puts goes up, not because the stock is "more volatile below 90" but because the *price* of that insurance is bid. The skew is the fingerprint that the real distribution of returns is not the tidy lognormal the model assumes.

It helps to be precise about the mechanics of "backing out" the vol, because it demystifies the whole exercise. Black-Scholes is a function that maps a volatility to a price: feed in a sigma and it returns one dollar value for a given strike and expiry. That function is *monotonic* in sigma — a higher sigma always produces a higher option price, because more volatility means a wider range of outcomes and an option only captures the favorable side. Monotonic functions are invertible, so for any market price there is exactly one sigma that reproduces it. Finding that sigma is a one-dimensional root-find: guess a vol, price the option, compare to the market, adjust the guess, repeat until they match. The answer is the implied vol. Nothing about that procedure assumes the answer will be the same across strikes; it simply reports, strike by strike, the vol consistent with each observed price. The market hands you seven prices on seven strikes, and the inversion hands you back seven different vols. The curve is what those seven numbers trace out.

So the deepest way to state the situation is this: Black-Scholes is not *wrong* — it is being used as a quoting convention rather than as a belief about the world. Traders quote and risk-manage in vol units because vol is comparable across strikes, expiries, and even underlyings in a way that raw dollar premiums are not. A \$2 option and a \$0.10 option are not obviously comparable; "17% vol" and "22% vol" instantly are. The model survives precisely because everyone agreed to *disagree with it* in a structured way — by feeding it a different sigma at every strike. The skew is the shape of that structured disagreement, and it is stable enough to be a market in its own right.

### The smile, the smirk, and the vocabulary

A few terms, defined once so the rest of the post is unambiguous.

- **Implied volatility (IV):** the volatility number that makes Black-Scholes reproduce an option's market price. Quoted in annualized percentage points (e.g. 17%).
- **Moneyness:** where a strike sits relative to the spot price. A strike below spot is a low strike; above spot, a high strike. We will quote moneyness as strike ÷ spot, so the 90 strike on a \$100 stock is 90% moneyness, the 110 strike is 110%.
- **The volatility smile:** the curve of implied vol plotted against strike (or moneyness). When it is roughly symmetric — both wings higher than the middle, like a valley — it literally looks like a smile. This is the classic shape in foreign-exchange options.
- **The skew (or smirk):** when the curve is *tilted* rather than symmetric — one wing much higher than the other. Equity indices show a pronounced **downward** skew: the low-strike (put) wing is far higher than the high-strike (call) wing. The curve looks less like a smile and more like a smirk, hence the nickname.
- **At-the-money (ATM):** the strike nearest the current spot (here, the 100 strike). Its implied vol is the reference point everyone quotes the skew relative to. (For the difference between at-the-money, in-the-money, and out-of-the-money, see [moneyness and the strike](/blog/trading/options-volatility/moneyness-and-the-strike-itm-atm-otm-and-what-you-are-really-buying).)

The numbers we will use throughout come from the series' curated **representative SPX 30-day skew** — an illustrative but realistic post-1987 equity-index shape, labelled as representative rather than a single day's quote:

```
    moneyness:  85%   90%   95%   100%  105%  110%  115%
    implied vol: 26.0  22.0  19.0  17.0  15.6  14.8  14.5  (%)
```

That table *is* the skew. Notice it falls monotonically from left to right. The 85% strike (deep OTM put) implies 26% vol; the 115% strike (deep OTM call) implies 14.5%. The at-the-money 100 strike sits at 17%. Every dollar figure in this post is computed from the Black-Scholes pricer using *the implied vol for that strike*, not a single flat number — that is the whole point.

#### Worked example: the put that cost five times the estimate

Take the hook trader's hedge and price it both ways. Stock at \$100, 30 days to expiry (T = 30/365 ≈ 0.082 years), risk-free rate 4%, and a 90-strike put.

**Priced off the at-the-money 17%** (the mistake): Black-Scholes gives a 90-strike put value of about **\$0.021** per share — \$2.10 per 100-share contract. That is the trader's estimate.

**Priced off the skew** — the 90 strike trades at **22%** vol, not 17% — Black-Scholes gives about **\$0.103** per share — \$10.30 per contract. Almost exactly five times the estimate.

The difference is entirely the vol input: 22% versus 17%. Same model, same spot, same strike, same expiry. The extra eight cents per share is the price of the skew, and on a thousand contracts it is the difference between paying \$2,100 and paying \$10,300. The intuition: the model is fine, but it must be fed the strike's *own* implied vol — the 90 put lives at 22%, and pricing it off the middle of the curve understates its cost by a factor of five.

## How the skew reprices the two wings

The cleanest way to feel the skew is to hold everything fixed except the vol input and watch the two wings move in opposite directions. The figure below does exactly that: it prices the out-of-the-money options on each side of the curve twice — once at each strike's true skew vol, once at a flat 17% at-the-money vol — and shows the gap.

![Grouped bars comparing OTM put and OTM call prices at the skew implied vol versus a flat 17% volatility, with puts dearer and calls cheaper under the skew](/imgs/blogs/the-volatility-smile-and-skew-why-otm-puts-cost-more-2.png)

Read the left panel first. Every OTM put is *more* expensive under the skew than under a flat vol, because every put strike's skew vol (19%, 22%, 26%) is above the flat 17%. The 95 put goes from \$0.31 (flat) to \$0.44 (skew). The 90 put goes from \$0.02 to \$0.10 — the five-fold jump from the worked example. The 85 put, almost worthless at flat vol, is \$0.03 at skew vol. The further out you go, the *larger* the proportional markup, because the deep wing is where the skew vol diverges most from the middle.

The right panel is the mirror image. Every OTM call is *cheaper* under the skew, because every call strike's skew vol (15.6%, 14.8%, 14.5%) is below the flat 17%. The 105 call falls from \$0.47 (flat) to \$0.37 (skew). The 110 call falls from \$0.058 to \$0.024 — less than half. The skew does not just raise the puts; it simultaneously lowers the calls. The curve pivots around the at-the-money point.

This is the practical heart of the matter. **The same distance out of the money is a completely different trade on the two sides.** A 10%-OTM put (the 90 strike) costs about \$0.103; a 10%-OTM call (the 110 strike) costs about \$0.024. The put is more than four times the price of the call, for the same 10% move in the *opposite* direction over the same 30 days. If you naively think "I'll buy a 10%-OTM option for cheap protection / cheap lottery," the answer is wildly different depending on which side you are on, and the skew is the reason.

#### Worked example: the same 10%-OTM strike, both sides

Stock \$100, 30 days, r = 4%.

- **10%-OTM put — the 90 strike — at its skew vol of 22%:** Black-Scholes value ≈ **\$0.103** per share.
- **10%-OTM call — the 110 strike — at its skew vol of 14.8%:** Black-Scholes value ≈ **\$0.024** per share.

The put costs **4.4×** the call. If the skew didn't exist and both traded at the flat 17%, the put would be worth \$0.021 and the call \$0.058 — and the *call* would be the more expensive of the two (calls carry a touch more value than equidistant puts at the same vol because of the positive interest-rate drift). The skew completely reverses that ranking: it makes the downside the expensive side. The intuition: in equity-index land, betting on a 10% drop is structurally pricier than betting on a 10% rise, because everyone wants the former hedge and the market charges them for it.

## Why the equity-index skew exists

The skew is not an accident or a mispricing waiting to be arbitraged away. It has persisted on equity indices for decades because it is fed by *structural*, repeating order flow — the same forces, year after year. There are three of them, and the figure below traces how they combine.

![Cause and effect diagram showing demand for downside puts, supply of overwriting calls, and crash memory feeding dealer inventory and producing rich OTM puts and cheap OTM calls](/imgs/blogs/the-volatility-smile-and-skew-why-otm-puts-cost-more-3.png)

**1. Structural demand for downside protection.** The biggest holders of equities — pension funds, insurance companies, asset managers, and increasingly retail investors — are *long the market* and want to be hedged against a crash. The natural hedge is to buy index puts, often 5–15% out of the money. This is portfolio insurance, and the demand for it is relentless: there is always someone with a large long book who needs to cap their downside. That demand is one-directional — these players are persistent net *buyers* of OTM puts, almost never sellers — and persistent buying pressure on a contract bids up its price, which shows up as elevated implied vol on the left wing. Insurance you cannot easily live without is never cheap.

**2. A supply of overwriting calls.** On the upside, the flow runs the other way. A large class of investors runs **covered-call** or "overwriting" strategies: they own the stock or index and sell OTM calls against it to harvest premium income. Pension funds, structured-product desks, and yield-focused retail funds all do this. The result is a persistent supply of OTM calls hitting the market — sellers outnumber buyers on the right wing — and persistent selling pressure offers a contract's price *down*, which shows up as depressed implied vol on the call side. The same imbalance that makes puts rich makes calls cheap.

**3. The 1987 crash memory and the asymmetry of crashes.** On October 19, 1987, the S&P 500 fell 20.5% in a single day. Before that day, equity options traded with very little skew — close to the flat-vol textbook world. After it, the skew appeared and never left. The reason is partly behavioral (the market learned that the left tail is fatter than the lognormal admits) and partly mechanical: **markets fall faster than they rise.** Crashes are gap-down, correlated, panic events; rallies tend to grind. A dealer who is short downside puts is short exactly the risk that blows up violently and all at once, so the dealer demands a higher price — higher implied vol — for carrying it. We will tell the full 1987 story and trace how it *created* the modern skew in [the 1987 crash case study](/blog/trading/options-volatility/case-study-the-1987-crash-and-the-birth-of-the-skew).

All three forces push the same way: they bid the puts and offer the calls. The market-maker community sits in the middle, ending up net short OTM puts and net long OTM calls, and prices its inventory accordingly — charging more (higher IV) for the downside it is short and the public keeps demanding, and accepting less (lower IV) for the upside it is long and the public keeps supplying. The skew is the equilibrium price of that flow. Because the flow is structural and repeating, the skew is structural and repeating — it does not arbitrage away.

#### Worked example: the dealer's-eye view of the markup

Suppose a dealer is asked to make a market in the 90-strike put. The "fair" value off the at-the-money 17% vol is \$0.021. But the dealer knows that if they sell this put, they join the crowd of everyone short the left tail, and they will struggle to buy it back in a sell-off when everyone wants it at once. So the dealer marks the put up — quotes it at a vol of 22% — which prices it at **\$0.103**. The \$0.082 of extra premium per share is the dealer's compensation for warehousing a risk that (a) is in structural excess demand and (b) blows up in a correlated, gap-down fashion exactly when it is hardest to hedge. The intuition: the skew is not the market predicting a crash on a particular day; it is the *standing price* of a risk that is both in short supply and unusually nasty to be short.

The natural question is why the skew is not competed away. If OTM puts are "expensive," why doesn't a wave of sellers show up, sell the rich vol, and flatten the curve? Two answers. First, sellers *do* show up — that is the variance-risk-premium trade — but selling far-OTM put vol is a strategy that earns small, steady premiums and then occasionally loses a fortune in a single gap. The supply of capital willing to be short the left tail is *limited*, because the people with the balance sheet to warehouse crash risk demand a high price for the privilege, and the people who don't have that balance sheet get carried out the first time it goes wrong. The premium persists because it is paying for a real and limited risk-bearing service, not because no one has noticed it. Second, the skew is not a free lunch you can lock in: there is no riskless arbitrage that turns "puts look expensive in vol terms" into guaranteed profit, because the only way to harvest it is to take on the tail risk the skew is compensating. A genuine arbitrage — buy low, sell high, no risk — would be competed away in minutes. A *risk premium* — get paid to hold something nasty — can persist for decades, and the skew is a risk premium. That distinction is the difference between a mispricing and a market.

## Reading the skew: slope, risk reversals, and what traders actually quote

Practitioners do not stare at the whole curve all day. They compress it into a couple of numbers that capture its *steepness* — how much more expensive the downside is than the upside — and quote those. The most important is the **risk reversal**.

A **25-delta risk reversal** is the implied vol of the 25-delta put minus the implied vol of the 25-delta call. ("25-delta" means the strike whose option has a delta of about 0.25 in absolute value — roughly a quarter of the way out of the money in probability terms; see [delta](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio) for what delta measures. Quoting by delta rather than by fixed strike is the market convention, because it keeps the comparison consistent as spot moves.) On equity indices the risk reversal is strongly *negative* in the call-minus-put convention, or equivalently *positive* in the put-minus-call convention we will use: the put vol is meaningfully above the call vol. It is the single number that says "the downside is X vol points more expensive than the upside."

![Implied volatility plotted against equivalent call-delta, with the 25-delta put at 18.2% and the 25-delta call at 16.0% and the 2.2 vol-point risk reversal gap annotated](/imgs/blogs/the-volatility-smile-and-skew-why-otm-puts-cost-more-5.png)

The figure plots the same skew, but now in **delta space** rather than strike space — implied vol against the option's delta. This is how a vol trader actually sees it. The 25-delta put (a strike around 97 on our curve) sits at about 18.2% implied vol. The 25-delta call (a strike around 103.5) sits at about 16.0%. The gap — the 25-delta risk reversal — is about **2.2 vol points**, put-over-call. That single number is what a trader means when they say "the SPX 25-delta risk reversal is bid 2.2 vols": the market is paying 2.2 extra vol points to be long the downside relative to the upside.

Two related measures round out the vocabulary:

- **The skew slope** is simply how fast IV changes per unit of moneyness or per delta. A steep skew (a big risk reversal) means a sharply tilted curve and a market very afraid of the downside; a flat skew means the two wings are priced similarly. The slope itself moves around — it steepens in fear, flattens in complacency — and trading *changes in the slope* is its own discipline, covered in [trading skew](/blog/trading/options-volatility/trading-skew-risk-reversals-collars-and-the-shape-of-fear).
- **The butterfly** (or "smile") measures how much the *wings together* sit above the at-the-money — the curvature, not the tilt. A big butterfly means both tails are bid relative to the middle (a true smile); a small butterfly with a big risk reversal is a pure one-sided smirk. Risk reversal captures tilt; butterfly captures curve.

#### Worked example: pricing a 25-delta risk reversal

A risk reversal as a *trade* is: buy the 25-delta put, sell the 25-delta call (the bearish/protective version). Price both legs off their own skew vols. Stock \$100, 30 days, r = 4%.

- **Buy the 25-delta put** — strike ≈ 97, vol ≈ 18.2% — costs about **\$0.80** per share.
- **Sell the 25-delta call** — strike ≈ 103.5, vol ≈ 16.0% — collects about **\$0.69** per share.
- **Net cost ≈ \$0.11** per share (a small debit).

Now imagine the skew didn't exist and both traded at the flat 17%. The put (strike 97) would be worth about \$0.69 and the call (strike 103.5) about \$0.78 — so selling the call to buy the put would actually be a **\$0.09 credit**; you'd get *paid* to put on downside protection. The skew flips that: the protection now costs you \$0.11 instead of paying you \$0.09, a swing of about \$0.20 per share. The intuition: the risk-reversal price *is* the skew, distilled to one trade — the steeper the skew, the more it costs to be long the downside relative to the upside, and that cost is exactly what the 25-delta risk reversal quotes.

## The implied distribution: skew is a fat left tail

There is a deeper way to read the skew, and it pays off in intuition: the skew is the market telling you the shape of its *probability distribution* for the stock's future price. This is the Breeden-Litzenberger idea, and we will use it lightly here — the full no-arbitrage treatment of the surface lives in [the volatility surface post](/blog/trading/quantitative-finance/volatility-surface).

The key fact, stated without the calculus: **the prices of options across strikes encode the market's risk-neutral probability distribution of the terminal stock price.** A whole strip of options is, in effect, a strip of bets on different outcomes, and from their prices you can read off how much probability the market is assigning to each region. Breeden and Litzenberger showed precisely how — the second derivative of the call price with respect to strike *is* the risk-neutral density — but the qualitative consequence is all we need: a flat implied-vol curve corresponds to the lognormal bell curve, and a *tilted* curve corresponds to a *tilted* distribution.

![Two risk-neutral density curves: a symmetric lognormal and a left-skewed skew-implied density with a fat left tail and a thinner right tail](/imgs/blogs/the-volatility-smile-and-skew-why-otm-puts-cost-more-4.png)

The figure shows both. The dashed curve is the lognormal density that flat-vol Black-Scholes assumes — symmetric in log terms, with thin, well-behaved tails. The solid curve is the density implied by the put skew: its peak is nudged slightly to the left, its **left tail is fat** (a meaningful chunk of probability mass sits down in the crash region around 80–90), and its **right tail is thinner** (less probability of a big melt-up). The shaded region is the *extra* crash probability the skew prices in over and above the lognormal.

This makes the skew click. OTM puts are expensive precisely because the market assigns *more* probability to a big down-move than the lognormal would — the left tail is fat, so options that pay off in that tail are worth more, so their implied vols are bid. OTM calls are cheap because the right tail is thin — less probability of a big up-move, so upside lottery tickets are worth less, so their implied vols are offered. The skew and the fat left tail are two views of the same object: the market's belief that crashes are more likely (and more violent) than a clean bell curve admits. When you buy a cheap OTM call and an expensive OTM put on the same name, you are not being treated inconsistently — you are being quoted off a distribution that genuinely is lopsided.

#### Worked example: the skew as a probability statement

On our curve, the 85 put (15% OTM) trades at 26% vol and is worth about \$0.031; under a flat 17% vol it would be worth about \$0.0004 — essentially nothing. The skew makes that deep crash put worth roughly **75 times** its flat-vol value. Translate that into probabilities: the market is pricing the 85-strike put as if a 15%+ drop in 30 days is *far* more likely than a 17% lognormal would say. A 17% annual vol implies a 30-day standard deviation of about 17% × √(30/365) ≈ 4.9%, so a 15% drop is roughly a 3-standard-deviation move — under a normal distribution about a 1-in-740 event, but the skew is pricing it as something much more common. The intuition: the steep left wing is the market quoting a left tail far fatter than the bell curve — paying real money today against a crash the lognormal calls almost impossible.

## What it means for traders: choosing a strike is choosing a vol

Everything above converges on one practitioner takeaway: **when you pick a strike, you are picking a point on the vol curve, and you are paying or collecting that strike's implied vol — not the at-the-money number.** This reframes a lot of common trades.

The cleanest mental reframe is to stop thinking in dollars of premium and start thinking in *vol you are buying or selling*. A dollar of premium spent on a 90 put at 22% vol is buying you the most expensive volatility on the board; the same dollar spent on a 110 call at 14.8% is buying you some of the cheapest. Two traders who both "spent \$0.10 on an OTM option" did not do the same thing at all — one bought rich vol, the other bought cheap vol, and over many repetitions that difference dominates their results. The skew is precisely the exchange rate between dollars and vol across the strike ladder, and it is never one-to-one. When you internalize that every strike has its own price of vol, the question stops being "is this option cheap in dollars?" and becomes "is this *vol* cheap relative to what I think will be realized, *given where it sits on the skew?*" That is the question a vol trader actually asks, and the skew is half the answer.

- **Buying downside protection is expensive, and gets more expensive the further out you go.** The deep OTM puts that look "cheap" in dollar terms are the *richest* in vol terms. You are buying the most overpriced volatility on the board. That does not mean don't hedge — it means know that the skew is a structural tax on insurance, and consider structures (spreads, collars) that sell some of that expensive vol back.
- **Buying upside is relatively cheap.** OTM calls are offered. A trader with a genuinely bullish view can express it with OTM calls that are inexpensive in vol terms — though "cheap" upside is cheap for a reason (the market thinks big rallies are less likely than big crashes).
- **Selling a put is selling rich vol; selling a call is selling cheap vol.** If your edge is collecting the variance risk premium by selling options, the put side pays you more per unit of risk — but it also loads you onto the fat, gap-prone left tail. The premium is compensation for a genuinely worse risk, not free money.
- **Every multi-leg structure inherits the skew.** A vertical spread, a collar, a risk reversal, a butterfly — each combines strikes at different points on the curve, so the skew is baked into the net price. Sometimes it helps you, sometimes it hurts you, and a trader who ignores it will systematically misprice their own structures.

Let's make that last point concrete with the two structures the skew touches most.

### How skew prices into a vertical spread

A **put debit spread** — buy a higher-strike put, sell a lower-strike put — is a defined-risk way to bet on (or hedge) a moderate down-move. (The full mechanics of debit and credit verticals are in [vertical spreads](/blog/trading/options-volatility/vertical-spreads-debit-and-credit-defining-your-risk).) Both legs are puts, so both sit on the steep part of the skew — but the *lower* strike sits on the *steeper* part, so it carries a higher vol. That changes the net cost.

#### Worked example: a put spread, skew vs flat

Buy the 95 put, sell the 90 put, 30 days, stock \$100, r = 4%. Max payoff is the 5-point width (\$5.00 per share); your cost is the net debit.

- **At skew vols** (95 put at 19%, 90 put at 22%): long 95 put ≈ \$0.44, short 90 put ≈ \$0.10 → **net debit ≈ \$0.34** per share.
- **At a flat 17%** (both legs): long 95 put ≈ \$0.31, short 90 put ≈ \$0.02 → net debit ≈ \$0.29.
- **At a flat 19%** (both at the long leg's vol): long ≈ \$0.44, short ≈ \$0.04 → net debit ≈ \$0.40.

The skew actually makes this spread slightly *cheaper* than pricing both legs at the long leg's 19% would suggest, because the short 90 leg you sell carries an even higher vol (22%) — you collect more on the leg you are short. Pricing the whole spread off the at-the-money 17% understates the long leg and overstates how little you collect on the short, getting the net debit (\$0.29 vs the true \$0.34) wrong by about 15%. The intuition: in a spread you are long one point on the skew and short another, so what matters is the *difference* in vols between the two strikes — and ignoring the skew misprices that difference every time.

### How skew finances a collar

A **collar** on a long stock position — own the stock, buy a protective OTM put, sell an OTM call to pay for it — is the textbook use of the skew, and it is where the skew is your *friend*. You are buying the expensive (put) wing and selling the cheap (call) wing... which sounds bad. But you are a *natural* holder of the position: you want the put for protection and you are happy to cap your upside by selling the call. The skew means the put you buy is dear and the call you sell is cheap, so a symmetric collar costs you money — but you can pick strikes so the cheap call nearly fully finances the expensive put.

#### Worked example: a skew-financed collar

Own 100 shares at \$100. Buy the 95 put (protection), sell a call to finance it. 30 days, r = 4%.

- **Symmetric 95/105 collar at skew vols:** buy 95 put (19%) ≈ \$0.44, sell 105 call (15.6%) ≈ \$0.37 → **net cost ≈ \$0.07** per share. The collar caps your stock between \$95 and \$105 for about 7 cents.
- **Same 95/105 collar at a flat 17%:** buy 95 put ≈ \$0.31, sell 105 call ≈ \$0.47 → net **credit ≈ \$0.16**. Under flat vol you'd get *paid* to collar.
- **Zero-cost collar under the skew:** to fully finance the \$0.44 put, you must sell a slightly *closer* call — around the 104.5 strike (vol ≈ 15.7%, value ≈ \$0.44) — giving up a touch more upside than the flat-vol world would require.

The skew costs the collar about \$0.23 per share (the swing from a \$0.16 credit to a \$0.07 debit) and forces you to sell a nearer call to get to zero cost. The intuition: the skew makes downside insurance dearer and upside cheaper, so financing protection by selling calls *always* means giving up more upside on an equity index than a flat-vol intuition suggests — the structure is feasible, but the skew sets the price of admission.

## Other smile shapes: the market's fear, by asset

The downward equity-index smirk is the most famous shape, but it is not the only one. The *shape* of the smile is a direct read on which tail the market fears, and different assets fear different tails.

![Three implied volatility curves across moneyness: an equity-index put smirk, a symmetric FX smile, and an upward-sloping commodity forward skew](/imgs/blogs/the-volatility-smile-and-skew-why-otm-puts-cost-more-7.png)

- **Equity indices — the downward smirk** (blue curve). Puts bid, calls offered, for all the structural reasons above. The market fears the *downside*: stocks crash, they rarely melt up. Fat left tail.
- **FX major pairs — the symmetric smile** (green curve). In a major exchange rate like EUR/USD, a big move in *either* direction is a shock, and there is two-sided demand for protection — both sides of the pair have someone who wants to hedge a sharp move against them. The result is a roughly symmetric smile: both wings bid relative to the at-the-money, with little tilt. The market fears *both* tails. (When a currency has a clear asymmetry — say a peg that can only break one way, or an emerging-market currency that can devalue but not surge — the smile tilts toward the feared direction.)
- **Energy and some commodities — the forward (reverse) skew** (amber curve). In crude oil, natural gas, and electricity, the violent, gap-prone move is often to the *upside* — a supply shock, a cold snap, a refinery outage spikes the price up, not down. So the OTM *calls* are bid and the curve slopes *up* to the right: a forward skew, the mirror image of equities. The market fears the *upside* in the thing it must buy. (Single stocks sometimes show a forward skew too, around an anticipated takeover or a binary upside catalyst.)

The lesson generalizes: **the shape of the smile is a map of the market's fear distribution.** A fat tail on one side bids the options that pay off in that tail, which lifts their implied vols, which tilts the curve toward that side. Read the tilt and you read what the market is afraid of. We will assemble these curves across strikes *and* expiries into the full three-dimensional object — the vol surface — in [reading the vol surface like a trader](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear).

## Sticky-strike vs sticky-delta: how the curve moves with spot

So far we have looked at the skew as a static snapshot. But spot moves all day, and a crucial — and often overlooked — question is: **when spot moves, what happens to the curve?** Does the implied vol at each fixed strike stay put, or does the whole curve slide along with spot? The answer determines what happens to the vol of the option you are *already holding*, and traders name two idealized regimes.

![Two-column comparison of sticky-strike and sticky-delta regimes when spot falls 5%, with at-the-money vol rising in one and unchanged in the other](/imgs/blogs/the-volatility-smile-and-skew-why-otm-puts-cost-more-6.png)

- **Sticky-strike.** The implied vol attached to each *fixed strike* does not move when spot moves. The curve stays pinned to the strike axis. Consequence: when spot falls, your at-the-money option is no longer at-the-money — a strike that *was* at-the-money is now an OTM-put strike on the curve, so it reads a higher vol. Equivalently, the new at-the-money strike (lower down) reads a higher vol than the old one did. **At-the-money vol rises as spot falls.** This is the typical regime in calm, range-bound markets.
- **Sticky-delta (or sticky-moneyness).** The curve is pinned to *moneyness* (or delta), not to fixed strikes, so the whole skew slides sideways with spot. Consequence: when spot falls 5%, the entire curve shifts down 5% with it, and the *new* at-the-money strike reads the *same* vol the old at-the-money did. **At-the-money vol is unchanged.** This is closer to the typical regime in trending or stressed markets, where the smile travels with the spot.

Why does this matter in dollars? Because your position's vol exposure — your vega, covered in [vega](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) — depends on which regime holds. In a sticky-strike world, a falling market hands a long-vol position an extra tailwind: ATM vol rises as spot falls, so being long options pays twice (long gamma into the drop *and* long vega into the rising vol). In a sticky-delta world, that second leg doesn't show up at the at-the-money. The regimes are idealizations — real markets blend them and switch between them — but knowing which one is roughly in force tells you whether a sell-off will inflate or merely move your vols, which is the difference between a hedge that overperforms and one that disappoints.

#### Worked example: the same 5% drop, two regimes

Stock falls from \$100 to \$95. You hold a 30-day option that was at-the-money (the 100 strike) at 17% vol.

- **Sticky-strike:** the 100-strike vol is unchanged at 17%, but the 100 strike is now ≈5% OTM-put on the curve, so it reads up the skew toward ≈19%. Your once-at-the-money option's implied vol *rose* about 2 points purely from the spot move. If that option carried, say, \$0.05 of vega per vol point, that's a +\$0.10 vega tailwind on top of the directional move.
- **Sticky-delta:** the whole curve slides down 5%, so the *new* at-the-money (95 strike) sits at 17% — the same level the old at-the-money had — and your option's implied vol is roughly unchanged from the spot move alone.

Same crash, same skew, opposite consequence for your vol P&L. The intuition: the skew is not just a shape; it is a shape that *moves*, and how it moves when spot moves is its own risk you carry the moment you hold any option off the at-the-money.

## Common misconceptions

**"Implied volatility is one number per stock."** No — it is a *curve* (really a surface, once you add expiries). The "the VIX is 17" or "AAPL's IV is 30%" shorthand refers to a single at-the-money-ish summary, but every strike has its own implied vol. On our representative SPX skew the 85 put implies 26% and the 115 call implies 14.5% — a 11.5-vol-point spread *on the same name, same expiry*. Pricing any non-at-the-money option off the single headline number is the mistake that made the hook trader's \$2,100 estimate come back at \$10,000.

**"The skew predicts a crash."** No — the skew is a standing *price*, not a dated forecast. It is elevated essentially all the time on equity indices, in bull markets and bear markets alike, because the structural demand for puts and supply of calls never goes away. A steep skew tells you crash protection is expensive *right now*; it does not tell you a crash is coming next week. The skew *steepens* when fear rises and flattens when complacency sets in, so *changes* in the skew carry information — but the level itself is a permanent feature, not a prediction. Treating an ever-present skew as a crash signal would have had you bracing for a collapse every single day for forty years.

**"A 10%-OTM call and a 10%-OTM put cost about the same."** No — on an equity index they are wildly different. We priced them: the 90 put is about \$0.103 and the 110 call about \$0.024, a 4.4× gap, because the put sits at 22% vol and the call at 14.8%. Anyone who sizes "cheap OTM options" by distance from spot without checking the vol is systematically overpaying for downside and underpaying for upside relative to their intuition.

**"OTM puts are overpriced, so selling them is free money."** No — they are *richly priced for a reason*. The skew is compensation for being short a risk that is in structural excess demand and that blows up in a violent, correlated, gap-down fashion exactly when you least want to be short it. Selling the rich put vol does capture a premium on average (the variance risk premium is real), but the few times it goes wrong, it goes wrong catastrophically — a fat left tail is fat precisely because the losses out there are enormous. The price is high because the risk is bad, not because the market is dumb.

**"The smile is the same for every asset."** No — the *shape* is asset-specific and reads as a fear map. Equity indices smirk downward (fear of crashes), FX majors smile symmetrically (fear of moves either way), and energy often skews *upward* (fear of supply-shock spikes). Assuming an equity-style put skew on an oil book — or pricing an FX option off an equity intuition — will misprice the wings, because the tail the market fears is on a different side.

## How it shows up in real markets

**The 1987 regime change.** Before October 19, 1987, equity-index options traded close to flat across strikes — the textbook world really did roughly hold. The 20.5% one-day S&P crash was a multi-standard-deviation event under any pre-1987 lognormal model, and it rewrote how the market priced tails. From that day forward, OTM put vols traded persistently above ATM and call vols, and the equity skew became a permanent fixture. The skew is, in a real sense, the market's institutional memory of 1987 — a price that has carried the lesson of that day for nearly four decades. (Full story and mechanics: [the 1987 crash case study](/blog/trading/options-volatility/case-study-the-1987-crash-and-the-birth-of-the-skew).)

**The skew steepens in a sell-off.** During market stress — February 2018's "Volmageddon," March 2020's COVID crash, the August 2024 yen-carry unwind — the whole vol curve does not just lift; it *steepens*. The put wing rises faster than the call wing, because the demand for downside protection spikes exactly when the downside is materializing, while the call overwriters keep supplying calls. So the 25-delta risk reversal blows out — the gap between put and call vol widens — at the worst moment. A trader long a put spread benefits from the vol rise but is partly capped because the short lower-strike put's vol rises even faster (the skew steepens *under* their long leg). Knowing the skew steepens, not just shifts, in a crash is the difference between sizing a hedge correctly and being disappointed by it.

**Why the steepening feeds on itself.** The reason the skew steepens rather than just shifts in a sell-off is partly a feedback loop in dealer hedging. Dealers who are net short OTM puts are short gamma down there — as spot falls toward those strikes, their short puts gain delta fast, and to stay hedged they must *sell* the underlying into the falling market. That selling pushes spot lower, which makes their short puts even more sensitive, which forces more selling. At the same time, the public's demand for fresh downside protection surges exactly when the market is dropping. Both forces hit the put wing simultaneously: more demand for puts and more dealer pain in being short them, so put implied vols gap up faster than the rest of the curve. The skew steepening is, in part, the visible trace of dealers being forced to chase a falling market — which is why the steepest skews print in precisely the disorderly, gap-down sessions the skew was pricing against all along.

**The persistent richness of crash puts.** Studies of long-run index option returns find that systematically *buying* far-OTM index puts has been a money-loser on average — the skew is so persistently rich that the insurance has, over decades, cost more than it paid out. The flip side is that systematically *selling* that vol has earned a premium, punctuated by occasional catastrophic losses. This is the variance risk premium viewed through the skew: the left wing is the most over-insured, and therefore the most expensive, volatility on the board. It is also the most dangerous to be short. Both the premium and the danger are largest exactly where the skew is steepest.

**Single-stock skews around catalysts.** Individual equities usually carry the same downside skew as the index, but it can *invert* into a forward skew around a specific upside catalyst — a pending acquisition where the stock can only gap *up* on a deal, or a biotech awaiting a binary trial readout with a huge upside case. When the feared (or hoped-for) move is to the upside, the call wing bids up and the skew flips. Reading a single name's skew tells you whether the market is bracing for a down-move (normal) or pricing a binary up-move (catalyst) — information you can't get from the at-the-money vol alone.

## The playbook: trading and hedging with the skew

The skew is not a thing you trade in isolation on day one; it is a lens that should sit behind every options decision you make. Here is how to put it to work.

**1. Always price the strike off its own vol.** Before you do anything, pull the implied vol *of the strike you are trading*, never the at-the-money number. The hook trader's five-fold surprise is the failure mode; it is entirely avoidable by reading the right point on the curve. If your platform shows a flat vol, you are flying blind on the wings.

**2. When buying protection, respect the tax — and consider selling some of it back.** Outright OTM puts are the richest vol on the board, so naked downside hedging is structurally expensive. Cut the cost with structures that *sell* expensive vol against your purchase: a **put spread** (sell a lower, even-richer put against your long put) or a **collar** (sell a cheap call to finance the dear put). You will give up some protection or some upside, but you stop paying full freight for the steepest part of the skew. The Greek profile of a long put is long vega and long gamma into a drop; a put spread caps both (you are short the lower-strike vol), and a collar adds a short-call leg that you must manage if the upside runs.

**3. When selling vol, prefer the side you're paid for — and size for the tail.** If your edge is harvesting the variance risk premium, the put side pays more per unit of risk, but it loads you onto the fat left tail. Size as if the tail event *will* happen, because the whole reason the put vol is rich is that the loss out there is enormous and correlated. A reasonable discipline: never be short more downside vega than you can survive a 1987-style gap on, and define the risk (sell *spreads*, not naked puts) so a crash can't take more than the spread width. Invalidation: a regime where the skew is unusually *flat* — the premium for selling the put wing has thinned, so the risk is no longer paying you.

**4. Quote the skew in one number and watch it move.** Track the 25-delta risk reversal (put vol minus call vol) for the names you trade. The *level* tells you how expensive downside is right now; the *change* tells you whether fear is building (steepening) or fading (flattening). A steepening risk reversal into a quiet tape can be an early warning; a flattening one in a panic can mark capitulation. Trading the *change* in skew — via risk reversals and butterflies — is its own discipline; the entry, sizing, and Greek profile of those trades are in [trading skew](/blog/trading/options-volatility/trading-skew-risk-reversals-collars-and-the-shape-of-fear).

**5. Match the structure to the asset's fear map.** Don't assume an equity-style put skew everywhere. On FX, expect a symmetric smile and price two-sided protection accordingly; on energy, expect a forward skew and remember the cheap wing is the *down* side and the rich wing is the *up* side. Building an equity-style collar on a commodity book — selling the cheap puts and buying the dear calls — would have you fighting the skew instead of using it.

**6. Know your sticky regime before you lean on a vol hedge.** If you are relying on a long-vol position to pay off in a sell-off, ask whether the market is sticky-strike (ATM vol rises as spot falls — your hedge overperforms) or sticky-delta (ATM vol roughly unchanged — your hedge pays on gamma but not on a vol pop). The second-order Greeks that govern how your vega and delta morph as spot and the skew move — vanna and friends — are the formal tools for this, and they live in [the second-order Greeks](/blog/trading/options-volatility/rho-dividends-and-the-second-order-greeks-vanna-volga-charm).

The throughline of this whole series holds here: an option is a bet on volatility, and the skew is the market's price for the *kind* of volatility you are betting on. The downside is dear because the downside is feared; the upside is cheap because it is supplied. Every strike you touch quotes you off that curve. Read the curve, price every leg off its own point on it, and you will never again be shocked by a fill — you'll have seen the skew coming.

## Further reading & cross-links

- [What sets an option's price: the five inputs and the intuition](/blog/trading/options-volatility/what-sets-an-options-price-the-five-inputs-and-the-intuition) — where implied volatility enters as the one input you cannot look up. This post takes that input and makes it a curve.
- [Moneyness and the strike: ITM, ATM, OTM and what you're really buying](/blog/trading/options-volatility/moneyness-and-the-strike-itm-atm-otm-and-what-you-are-really-buying) — the strike vocabulary the skew is plotted against.
- [Vega: your exposure to implied volatility and the vol of vol](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) — why a moving skew is a P&L event, and how vega measures it.
- [Rho, dividends, and the second-order Greeks: vanna, volga, charm](/blog/trading/options-volatility/rho-dividends-and-the-second-order-greeks-vanna-volga-charm) — vanna is exactly how your delta morphs as the skew moves.
- [Reading the vol surface like a trader: the 3D map of fear](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) — the skew across every expiry, assembled into the full surface.
- [Case study: the 1987 crash and the birth of the skew](/blog/trading/options-volatility/case-study-the-1987-crash-and-the-birth-of-the-skew) — how one day created the modern equity skew.
- [Vertical spreads: debit and credit, defining your risk](/blog/trading/options-volatility/vertical-spreads-debit-and-credit-defining-your-risk) — how the skew prices into a two-leg vertical.
- [Trading skew: risk reversals, collars, and the shape of fear](/blog/trading/options-volatility/trading-skew-risk-reversals-collars-and-the-shape-of-fear) — trading the *change* in the skew as its own strategy.
- [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) — the constant-volatility pricing derivation the skew departs from.
- [The volatility surface](/blog/trading/quantitative-finance/volatility-surface) — the surface as a no-arbitrage object, and the Breeden-Litzenberger link to the implied distribution.

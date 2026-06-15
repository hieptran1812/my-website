---
title: "Following the Flows: Positioning, the COT Report, and Dealer Hedging"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Prices move because of flows — who is buying, who is selling, and who is forced to. The COT report, fund flows, and options-dealer gamma show where the crowd is leaned and where the pain trade hides."
tags: ["macro", "positioning", "cot-report", "dealer-gamma", "fund-flows", "short-squeeze", "options", "volatility", "market-structure", "trading"]
category: "trading"
subcategory: "Macro Trading"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Prices move because of *flows* — the actual buying and selling — and the most violent moves come from traders who are *forced* to transact. Positioning data (the COT report, fund flows, options-dealer gamma) tells you where the crowd is leaned, and a crowded position is fuel for a move in the opposite direction.
>
> - **Flows set the timing; fundamentals set the tide.** A stock can be cheap for a year and only move the week the inflows arrive. Knowing *who has to trade* is often worth more than knowing what something is worth.
> - **Crowding is fuel, not a signal.** When everyone is already long, there is no one left to buy and a small shock forces the crowd out the same narrow door. Positioning is a *contrarian gauge at extremes*, but it tells you the move could be violent — not when it starts.
> - **Dealer gamma flips markets between calm and chaos.** When options dealers are long gamma they sell rallies and buy dips, pinning price and damping volatility. When they are short gamma they must chase the move, and a quiet tape becomes an air pocket.
> - **The one number to remember:** in late January 2021, GameStop's short interest was around 140% of its free float — more shares were sold short than actually existed to buy back — and the forced covering helped drive the stock from about \$20 to an intraday \$483 in two weeks, on essentially no change in the business.

On 26 January 2021, a struggling video-game retailer was worth more than most of the companies in the S&P 500's bottom third. GameStop had not invented anything, signed a transformational deal, or reported a blowout quarter. Its stores were still half-empty. And yet the stock had gone from roughly \$20 at the start of the month to an intraday high near \$483 a few sessions later — a move of more than 2,000% — while the actual business barely twitched.

What happened had almost nothing to do with the value of GameStop and almost everything to do with *who was positioned how*. A very large number of professional traders had sold the stock short — borrowed shares, sold them, and promised to buy them back later — betting the price would fall. The short interest was so extreme that the number of borrowed-and-sold shares exceeded the entire free float of the company. When the price started rising instead of falling, every one of those shorts faced a brutal arithmetic: their losses were theoretically unlimited, and the only way to stop the bleeding was to *buy the stock back*. So they bought. Their buying pushed the price higher. The higher price forced more shorts to buy. The buying that was supposed to be a free-market expression of opinion had become a fire — self-feeding, mechanical, and indifferent to any notion of fair value.

This post is about that fire and the machinery that produces it. The thesis is blunt and a little uncomfortable: **markets move because of flows — the actual orders hitting the tape — and the most violent moves are driven by participants who are *forced* to trade, not by anyone changing their mind about fundamentals.** We are going to build, from absolute zero, an understanding of how to *see* those flows before they happen: the Commitments of Traders (COT) report that x-rays futures positioning, the fund-flow data that reveals the marginal buyer, and the options-dealer hedging (gamma) that quietly decides whether a market is a trampoline or a trapdoor.

![Flow pyramid with fundamentals tide ordinary flows timing and forced flows violence](/imgs/blogs/following-the-flows-positioning-cot-dealer-hedging-1.png)

The figure above is the whole worldview on one page, and the rest of the post earns the right to read it. At the base, slow and gravitational, sit the fundamentals — earnings, growth, rates, valuation. They set the *direction* over months and years, but they rarely tell you what happens this week. In the middle sit ordinary flows: the inflows into funds, the positioning that builds up over weeks, the marginal buyer. They set the *timing* — when the move that fundamentals justify actually arrives. And at the top, small but explosive, sit forced flows: short covering, margin calls, dealer hedging. They set the *violence* — how far and how fast price overshoots when the crowd all tries to leave at once. Hold these three layers in your head. Everything else is detail.

## Foundations: flows, positioning, and options, from zero

Before we can read a single positioning report, we have to be precise about a handful of ideas that beginners routinely blur together. We will define each from scratch, build intuition with everyday-money analogies, and only then go deep.

### Flows versus fundamentals: price is set by orders, not by worth

Start with the most important and least intuitive fact in all of trading: **a price is not a measure of worth. A price is the level at which the last trade happened.** It moves only when someone is willing to buy more aggressively than someone else is willing to sell, or vice versa. That willingness — expressed as actual orders — is what we call *flow*.

Here is the everyday analogy. Imagine a small town with one house for sale and one hundred families who think it is fairly priced at \$300,000. Nothing happens. The "fundamental value" is \$300,000, but there is no transaction, so there is no price action. Now suppose a single new family moves to town and *must* buy a house this week — relocation, school starting, no choice. Their need to transact, their *flow*, sets the price. If three families bid against them, the house might clear at \$340,000. The fundamental value did not change; the flow did. The marginal buyer — the last, most-motivated participant — set the price for everyone.

Markets work exactly this way, just faster and with more participants. Most of the time, fundamentals and flows agree and you cannot tell them apart. But at turning points they diverge violently, and the trader who is watching flows sees the move coming while the trader watching only fundamentals is left asking why a "cheap" stock keeps falling or an "expensive" one keeps ripping.

The practical takeaway: **fundamentals tell you the destination; flows tell you the schedule.** A cheap asset with no inflows can stay cheap for years. An expensive asset with relentless inflows can stay expensive far longer than you can stay solvent betting against it. To trade, you need both — and most people only watch one.

### Positioning: where the crowd is already leaned

*Positioning* is the answer to a simple question: among the people who can trade this thing, who has already done so, and in which direction? If most large traders are already long, the market is "crowded long." If most are short, it is "crowded short."

Why does positioning matter so much? Because **a position already taken is future flow in reverse.** If a fund is already long, it cannot buy more without adding risk it may not want; its next move is more likely to be a sale. The crowd's current position tells you which orders are *probable* next. When everyone is already long, the marginal buyer is nearly gone, and the only large flows left to come are sells. That is why crowded positions are fragile: they have spent their fuel.

The everyday analogy: think of a crowded theater with one small exit. While everyone is seated and calm, the crowding is invisible — it does not matter. The moment someone yells "fire," the exact same crowding becomes lethal, because everyone wants to leave through a door built for a trickle. Positioning is the seated crowd. A shock is the shout. Forced flow is the stampede.

### Forced flow: the squeeze that turns crowding into a cascade

The theater analogy carries a subtle but crucial point: the danger is not the crowd, it is the *forcing*. A seated crowd that can leave at its leisure is harmless; a crowd that *must* leave through a tiny door, all at once, is a disaster. In markets, the equivalent of "must leave" is the **forced trade** — a transaction the participant has no choice about. And the canonical forced trade is the **short squeeze**.

To see it, you first have to understand short selling. When you *short* a stock, you borrow shares from someone who owns them, sell those borrowed shares into the market, and pocket the cash — betting you can buy the shares back later at a lower price, return them to the lender, and keep the difference. Your profit is capped (the stock can only fall to zero) but your loss is *unlimited* (the stock can rise forever, and you still owe the shares). That asymmetry is the entire engine of the squeeze. A long position that goes wrong is a slow, bounded pain; a short position that goes wrong is an accelerating, unbounded panic.

Now stack the crowding on top. Suppose a stock is heavily shorted — a large fraction of its freely-tradable shares have been borrowed and sold. Every one of those short-sellers is sitting on a position that loses money if the price rises, with no ceiling on how much. They are the seated crowd in the theater. The "fire" is any catalyst that pushes the price up: a good earnings surprise, a coordinated buying wave, even just the first few shorts deciding to cut their losses. Once the price starts rising, the math becomes unbearable for the shorts, and to stop the bleeding they must do the one thing that makes everything worse — *buy the stock back*. That buying is forced; it does not care about valuation; and it pushes the price higher, which forces *more* shorts to buy, which pushes it higher still.

![Positioning squeeze mechanic from crowded through shock and forced covering to a cascade](/imgs/blogs/following-the-flows-positioning-cot-dealer-hedging-3.png)

The figure above is the full mechanic in five steps. It starts with **crowding** — everyone leaned the same way, little fuel left to push the original trend. A **shock** arrives, small and often mundane. The losers face **forced covering** — they *must* buy, not as a choice but as a margin-driven necessity. The covering becomes a **cascade**, a self-feeding loop where each round of buying forces the next. And it ends in an **overshoot**: price flies far above any fair value on no new information, then collapses once the forced buyers are all done. This is the top layer of the flow pyramid in motion, and it is worth working through with numbers.

#### Worked example: a crowded short squeeze

Suppose a stock trades at \$20 and **30% of its free float is sold short.** (Illustrative round numbers chosen to show the mechanism.) A piece of good news — a better-than-feared quarter — lifts the stock 10%, to \$22. That 10% pop is enough to put many of the shorts underwater and trip risk limits at their funds, forcing some of them to cover.

Trace the cascade. Say the float is 10 million shares, so the short interest is 3 million shares that must, eventually, be bought back. The first wave of forced covering — perhaps a third of the shorts, 1 million shares — hits a market that has very little stock for sale near \$22, because longs are holding and the shorts are now buyers, not sellers. With buyers vastly outnumbering willing sellers, the price gaps up to \$26. That higher price puts the *remaining* shorts in even deeper trouble, and the next wave covers, gapping it to \$30. By the time the squeeze burns through the short interest, the stock has run from \$20 to \$28 — a **+40% move** — on a piece of news that, on its own, justified maybe the first 10%.

```
starting price                 = $20
news pop (fundamental)         = +10%  -> $22
short interest                 = 30% of a 10M float = 3M shares to cover
wave 1 covering (1M shares)    -> gap to $26   (thin offer above)
wave 2 covering (1M shares)    -> gap to $30
settle after forced buying done -> ~$28
total move                     = +40%   (vs +10% fundamental)
```

The extra 30% is *pure positioning*: it is the price of a crowded short being forced out through a small exit. Note what makes it explosive — the same arithmetic that the GameStop example pushed to its limit, where short interest reached roughly 140% of the float. The one-sentence intuition: a heavily-shorted stock is pre-loaded with forced buyers, so any push higher can ignite a self-feeding squeeze that dwarfs the news that started it.

The mirror image exists for crowded *longs*: a heavily-owned, over-loved stock has its forced *sellers* pre-loaded, and a small disappointment can trigger a downside cascade as leveraged longs hit margin calls and dump into a market with no bid. Crowding cuts both ways; the squeeze just runs in whichever direction the crowd is leaned.

### The Commitments of Traders (COT) report: an x-ray of futures positioning

We cannot see most positioning directly — it is private. But in the *futures* markets, a US regulator publishes it. The Commodity Futures Trading Commission (CFTC) requires large traders to report their futures positions, and every Friday it releases the **Commitments of Traders (COT) report**, a snapshot of who is long and who is short in each futures market (oil, gold, the S&P 500, currencies, bonds, and dozens more).

The COT splits every market into three groups, and understanding the three groups is the heart of reading it.

![COT breakdown of commercials large speculators and small traders summing to zero](/imgs/blogs/following-the-flows-positioning-cot-dealer-hedging-2.png)

- **Commercials (hedgers).** These are businesses that use the underlying for real: a farmer hedging a corn crop, an airline hedging jet fuel, a miner hedging gold, a bank hedging interest-rate exposure. They are not trying to predict price; they are offloading risk. Crucially, they tend to trade *against* the trend — selling into rallies (locking in good prices for what they produce) and buying into dips. Because they live in the physical market, the commercials are often called the **smart money**, and at extremes they are frequently on the right side.
- **Large speculators (the funds).** Hedge funds, commodity trading advisors (CTAs), and other large financial players who are in it purely to bet on price. They tend to trade *with* the trend — buying strength, selling weakness — which means they pile in as moves mature. The large specs are **the crowd**: often right in the middle of a trend, often catastrophically wrong at the extremes.
- **Small traders (non-reportable).** Everyone too small to report, lumped together. This is the least-informed cohort, and historically the most extreme exactly at tops and bottoms.

The single most important structural fact about the COT — and the one beginners miss — is that **it is a closed, zero-sum system.** A futures contract is an agreement between a buyer and a seller; for every long contract there is exactly one short. So across the three groups, net long plus net short must equal zero. If the large specs are at a record net-long, *someone* has to be at a record net-short on the other side — and it is usually the commercials. The COT does not tell you what an asset is worth. It tells you how the chips are distributed, and whether the crowd has any fuel left.

### Fund flows: following the marginal buyer

The second window onto flow is **fund flows** — the money moving into and out of investment vehicles. When you put \$1,000 into an S&P 500 index fund, that fund must go buy roughly \$1,000 of stocks. Your savings became forced buying. Multiply by millions of savers and you get a river of money that has to be deployed regardless of valuation.

The cleanest version of this is the **ETF creation/redemption** mechanism. An exchange-traded fund (ETF) issues new shares ("creations") when demand is high, and to back those shares it buys the underlying assets. When \$20 billion flows into an equity ETF, roughly \$20 billion of stock gets bought, mechanically, by an authorized participant who delivers the basket. Redemptions work in reverse: outflows force selling. Mutual-fund inflows behave similarly — the manager receives cash and must deploy it.

Fund flows matter because they reveal the **marginal buyer**: the last, most-motivated dollar that sets the price. A market does not need everyone to buy to go up; it needs the *next* buyer to be more eager than the *next* seller. Persistent inflows guarantee a steady stream of next-buyers. When the inflows stop or reverse, the marginal buyer vanishes — and a market that looked unstoppable can stall on no news at all, because the only thing that changed is that the forced buying ended.

### Options, delta, and gamma: the language of dealer hedging

The third and most technical window is options-dealer hedging. To read it we need three terms. They sound intimidating; the ideas are simple.

An **option** is a contract giving the right (not the obligation) to buy or sell an asset at a fixed price (the *strike*) by a date. A **call** profits when the asset rises; a **put** profits when it falls. You pay a *premium* up front for that right, and the most you can lose as the buyer is the premium.

**Delta** answers: if the underlying moves \$1, how much does the option's value move? A delta of 0.50 means the option gains \$0.50 when the stock gains \$1. Delta also equals roughly the probability the option finishes in-the-money, and — most usefully for us — it tells a dealer how many shares to hold to offset the option. If a dealer is short a call with delta 0.50 on 100 shares, they buy 50 shares to be *delta-neutral*: flat to small moves.

**Gamma** answers: how fast does delta *itself* change as the underlying moves? This is the one that matters for flows. Gamma is the *curvature*. If gamma is high, then as the stock rises the dealer's delta grows quickly, and to stay neutral they must keep re-hedging — buying more as it rises, selling more as it falls. Gamma is what turns a static position into a stream of forced orders. The everyday analogy: delta is your speed; gamma is your acceleration. A position with high gamma is one whose required hedging *accelerates* with the move — and that acceleration is exactly what amplifies or damps the market.

It helps to see why gamma forces *continuous* trading where delta alone would not. If a dealer held a position with delta but zero gamma — a plain block of stock, say — they would never need to re-hedge; their exposure does not change as price moves. Options have gamma, so their delta is a moving target: every tick changes how many shares the dealer needs, and the dealer must chase that target all day. The size of that chase scales with gamma and with how much price moves, which is why gamma exposure matters most in volatile markets and near the strikes where the most options sit (gamma is highest for at-the-money options close to expiry). A dealer book stuffed with near-expiry at-the-money options is a hair-trigger; the same book a month earlier, with the options far from expiry, barely twitches.

The crucial point, which the next sections build out: **dealers hedge to stay neutral, and the *sign* of their gamma decides whether that hedging pushes the market back toward calm or out into chaos.** That single fact is one of the most powerful and least-understood flow signals in modern markets — and the reason a stock can trade like a millpond for weeks and then convulse on a Friday afternoon with no headline at all.

## Reading the COT: a crowded crowd is downside fuel

Now we can use the COT report as traders do. The naive reading — "specs are long, so the trend is up, so buy" — is exactly backwards at the moments that matter. The professional reading inverts it: **a heavily one-sided position is fuel for a move in the *opposite* direction, because the crowd has spent its buying power and the only large flows left are exits.**

The logic runs through the zero-sum identity. Suppose the large speculators in crude oil futures push to a record net-long — say they have never, in three years of data, been this lopsidedly bullish. By the closed-system identity, the commercials are at a record net-short on the other side. Two things follow. First, the smart money (commercials) is leaning hard the other way, which historically is a warning. Second, and more mechanically: if every momentum fund is *already* long, who is left to buy? The trend has consumed its own fuel. The next marginal flow is far more likely to be a sale — and if a shock arrives, all those longs head for the same exit at once.

This is why positioning is a **contrarian gauge at extremes**. Not in the middle of the distribution, where it tells you almost nothing, but in the tails, where the crowd has become so one-sided that the trade is fragile. The art is in defining "extreme" — usually as a percentile of the position's own multi-year history, not an absolute number, because every market has its own scale.

The standard way practitioners normalize this is the **COT index**: take the net position of a group, find its minimum and maximum over a lookback window (commonly three years), and rescale today's reading to a 0–100 range. A COT index of 100 means the group is at its most-long in three years; 0 means its most-short. This matters because raw contract counts are not comparable across markets or across time — open interest grows, contract sizes differ, and a "big" gold position is a "small" Treasury position. Normalizing to a percentile makes "extreme" mean the same thing everywhere: the top or bottom decile of the group's *own* history. Most COT-based systems only act when the index is above ~90 or below ~10, and only on the *speculator* and *commercial* lines, because those are the two groups whose extremes have historically carried information.

A second refinement worth knowing: since 2009 the CFTC also publishes a **disaggregated** report that splits the old "commercial" bucket into *producers/merchants* (true physical hedgers) and *swap dealers* (financial intermediaries who lay off risk), and the old "large spec" bucket into *managed money* (hedge funds and CTAs, the purest trend-following crowd) and *other reportables*. The "managed money" line is the one most flow-watchers actually use as the crowd gauge, because it isolates the discretionary and systematic speculators from everyone else. When managed money is at a three-year net-long extreme, that is the cleanest signal that the momentum crowd is all-in. The legacy three-group report is simpler and still widely cited, but the disaggregated lines are sharper.

One more structural nuance that beginners trip on: the COT tells you *net* positioning, but the *gross* numbers and the *open interest* matter too. A market where specs are net-long 250,000 contracts on 1 million total open interest is far less crowded than the same net figure on 400,000 open interest — the second is a much larger fraction of the whole market leaning one way. And a position can become "less crowded" two ways: the longs can sell (price falls), or new shorts can come in against them (open interest rises while net stays flat). Reading the *change* in open interest alongside the net position tells you whether a trend is being driven by new money entering or by the existing crowd simply holding. New money on rising open interest is a healthy trend; a flat-to-falling open interest on a grinding rally is a trend running out of participants — which is exactly the exhaustion the COT extreme is trying to flag.

#### Worked example: reading a COT extreme

Suppose you are tracking the COT for the S&P 500 e-mini futures and you compute the *large-speculator net position* every week for three years. (These are illustrative round numbers chosen to show the mechanics, not a specific historical print.) The position has ranged from net-short 80,000 contracts at the 2022 lows to net-long 250,000 at prior tops. This week it prints **net-long 250,000 contracts — a fresh three-year high.**

What does that tell you, and what is the historical edge?

First, the structural read. By the zero-sum identity, the commercials are now at a three-year net-short extreme: the smart money is positioned for a fall. And the large specs — the crowd — have never been more all-in long. There is essentially no marginal momentum buyer left; the fuel is spent.

Now the historical forward return. Suppose you go back and find that on the *ten* prior occasions the spec net-long hit its top decile (the most-bullish 10% of all weeks), the index's average return over the next four weeks was **-1.8%**, versus a +0.7% average for all weeks — and that seven of the ten episodes were negative. (Again, illustrative figures used to show how you would *measure* the edge.)

```
all-weeks 4-week forward return       = +0.7%   (baseline)
top-decile-long 4-week forward return = -1.8%   (10 episodes)
edge from the extreme                 = -2.5%   (relative)
hit rate (episodes negative)          = 7 / 10  = 70%
```

The read is not "short it today." It is: **the crowd is maximally long, the smart money is maximally short, the forward distribution is skewed down, and the trade is fragile — so I should fade strength with a defined risk and wait for a catalyst, not chase the rally.** Positioning told you the asymmetry, not the timing. The one-sentence intuition: a crowded long is downside fuel because the buyers are already in, so the next big flow can only be a sale.

A vital caveat lives inside this example, and it is the reason positioning humbles people: **the COT is not timely.** The data is collected on Tuesday and released the following Friday — a three-day-old snapshot at best — and in 2020 the report was even delayed at times. Crowded positions can get *more* crowded for weeks before they break. Positioning is a *gauge*, not a *trigger*. You combine it with price action, a catalyst, or a level; you never trade it alone.

## Dealer gamma: long-gamma damps, short-gamma amplifies

The COT is a weekly, lagged x-ray. Dealer gamma is the opposite: a real-time, mechanical force that decides whether the *intraday* tape is calm or violent. It is the single best explanation for why some days a market shrugs off bad news and other days it falls off a cliff on nothing.

Here is the setup. When you buy an option, a market-maker (the dealer) takes the other side. The dealer does not want a directional bet — they make money on the spread, not the direction — so they *hedge* to stay delta-neutral. As the underlying moves, their delta drifts (that is gamma), so they must re-trade the underlying to get back to neutral. The direction of that re-hedging is determined entirely by whether the dealer is **long gamma** or **short gamma**.

![Dealer gamma hedging long gamma damps short gamma amplifies the move](/imgs/blogs/following-the-flows-positioning-cot-dealer-hedging-5.png)

**Long gamma (stabilizing).** When dealers are net long options, they are long gamma. As the underlying rises, their net delta grows positive, so to stay neutral they *sell* into the rally. As it falls, their delta grows negative, so they *buy* the dip. In both directions they trade *against* the move. The effect is to absorb volatility: rallies get sold, dips get bought, and price tends to get "pinned" near the strikes where the most options sit. A long-gamma market is a trampoline — it pushes price back toward the middle.

**Short gamma (destabilizing).** When dealers are net short options — which often happens when the public has been *selling* options to them, or when a lot of puts have been bought — they are short gamma. Now the hedging flips. As the underlying falls, their delta grows *short*, so to stay neutral they must *sell* into the decline. As it rises, they must *buy* into the rally. They trade *with* the move, in the same direction, adding fuel. A short-gamma market is a trapdoor — once price starts moving, the dealers' own hedging pushes it further, and a small down-move can cascade into an air pocket with no natural buyer.

This is why the *sign* of aggregate dealer gamma is one of the most-watched flow signals among professional vol traders. Estimated as "gamma exposure" (often abbreviated GEX), it tells you the regime of the day: positive (long-gamma, expect mean-reversion and pinning) or negative (short-gamma, expect trends and tail risk). It does not predict direction. It predicts *character*: whether a move, once it starts, gets damped or amplified.

#### Worked example: dealer short-gamma forces buying that amplifies the move

Suppose the aggregate dealer position in a single stock is **short gamma of \$5 million per 1% move.** That phrasing means: for every 1% the stock moves, the dealers' net delta changes by \$5 million worth of stock that they must hedge *in the direction of the move*. (Illustrative figures to show the mechanism.)

The stock is at \$100. A piece of good news lifts it 1%, to \$101. The dealers, being short gamma, find their hedge is now \$5 million too short, so they must **buy \$5 million of stock** — which pushes the price up further. Say that extra buying lifts it another 0.5%, to \$101.50. That additional move forces them to buy *more*, roughly \$2.5 million, lifting it again, and so on. The move feeds itself:

```
initial news move:   $100.00 -> $101.00   (+1.0%)
dealer hedge buy:    $5.0M  -> pushes +0.5% -> $101.50
next hedge buy:      $2.5M  -> pushes +0.25% -> $101.75
... geometric tail ...
total move:          roughly +2% on a +1% catalyst
```

The same machine runs in reverse on the way down, and *that* is the dangerous direction: a -1% shock forces dealers to *sell*, which drives -0.5% more, which forces more selling, and a quiet stock becomes a one-way air pocket. The one-sentence intuition: when dealers are short gamma, their forced hedging points the same way as the move, so a \$1 push becomes \$2 and a small shock becomes a cascade.

The trader's read is not "predict the direction." It is "respect the regime." In a deeply short-gamma tape you widen stops (normal levels get blown through), you do not assume support holds (there is no natural dip-buyer; the dealers are selling too), and you size down — because the realized volatility you are about to experience is far higher than the calm tape suggests. In a long-gamma tape you do the opposite: fade extremes toward the pinning strikes, expect chop, and do not chase breakouts that are likely to get sold back.

### Why the sign of dealer gamma is usually negative near a selloff

Here is the second-order insight that makes gamma so useful as a regime signal: dealer gamma is not random — it has a *structural tilt*, and the tilt flips precisely when markets get scared. In normal times, a large share of the option flow is investors *buying puts for protection* and *selling calls for income* on stocks they own (the "covered call" trade). Selling calls to dealers makes the dealers *long* those calls and therefore long gamma at the upside strikes — which is part of why rallies often feel capped and grindy, with price pinned below big call strikes into an expiry. So the resting state of the market skews dealers long gamma, which is stabilizing, which is why most days are quiet.

But protective puts behave differently. When investors *buy* puts, the dealers are *short* those puts and therefore short gamma at the downside strikes. As long as price stays well above those strikes, that short-gamma exposure is dormant — the puts are far out-of-the-money and their gamma is tiny. The moment price falls *toward* the wall of protective puts, that dormant short gamma wakes up: the dealers' gamma turns sharply negative right where the puts are clustered, and their hedging flips from stabilizing to amplifying exactly when the market can least afford it. This is the mechanical reason selloffs accelerate as they deepen — the market falls *into* its own short-gamma zone. It is also why the downside is structurally more dangerous than the upside: the protective-put wall below the market is a short-gamma minefield, while the call wall above is long-gamma quicksand that merely caps gains.

There is also a calendar to it. A huge fraction of options expire on the third Friday of each month, and especially on the quarterly expirations ("triple witching"). As those options expire, the dealer gamma tied to them *vanishes*, and the pinning force that was holding price near a big strike suddenly releases. This is why the days *after* a major monthly options expiration (OPEX) are often more volatile and more trending than the pinned, range-bound days before it — the stabilizing long-gamma hedging that was suppressing movement has rolled off. A flow-aware trader marks the monthly and quarterly OPEX dates on the calendar and expects the character of the tape to change around them, even with no news.

#### Worked example: how a put-protection wall flips the regime

Suppose an index sits at 5,000 and a large amount of protective put buying has concentrated at the 4,800 strike — call it dealer short gamma of \$3 billion per 1% move *if and when* price reaches 4,800, but near zero at 5,000 because the puts are 4% out-of-the-money. (Illustrative figures.) At 5,000, the market is calm: dealers are net long gamma from all the call-selling, so dips get bought and the tape chops sideways. Now a shock drives the index down 3%, to 4,850 — close to the put wall. The dealers' gamma at 4,800 is no longer dormant; their net gamma has swung from positive to sharply negative, and a further 1% drop now forces them to sell roughly \$3 billion:

```
index at 5,000:  dealers NET LONG gamma  -> dips bought, calm
shock: -3% to 4,850 (approaching 4,800 put wall)
index at 4,850:  dealers NET SHORT gamma -> forced selling begins
each further -1% -> ~$3B of forced dealer selling -> drives another -1%+
result: the last 4% down is far faster than the first 3%
```

The first 3% was an orderly news move; the next leg, falling *into* the short-gamma zone, is an accelerating air pocket. The one-sentence intuition: a market is calmest above its put-protection wall and most dangerous as it falls into it, because that is exactly where dealer hedging flips from damping to amplifying.

## Fund flows and the marginal buyer

The third leg is the slowest but the most relentless. Fund flows do not cause intraday fireworks the way gamma does; they set the *background tide* against which everything else happens. And because so much modern saving is automatic — every paycheck, a slice goes into a retirement account that buys an index fund regardless of price — there is a permanent river of forced buying that an asset enjoys simply by being in the index.

![Fund flow funnel from savings through ETFs and funds to the marginal buyer](/imgs/blogs/following-the-flows-positioning-cot-dealer-hedging-6.png)

The funnel above traces it. Paychecks and savings are the source. They pour through wrappers — 401(k)s and pensions on the automatic side, brokerage and trading apps on the discretionary side — into the funds: ETF creations and mutual-fund inflows, each of which translates dollars into mechanical buying of the underlying basket. At the bottom of the funnel sits the **marginal buyer**: the last dollar in, the one whose eagerness sets the clearing price for everyone. When the funnel runs full, the marginal buyer is always present and price grinds higher. When the funnel runs dry — inflows stall or reverse — the marginal buyer disappears, and the same asset can fall on no news, because the only thing that changed is that the forced bid went away.

This is the mechanism behind a lot of "it makes no sense" price action. A mediocre company in a popular index can rise for years on flows alone, because index funds must buy it in proportion to its weight no matter what they make of it. The flow does not care about the story. It cares about the basket.

#### Worked example: a fund-flow tally and the implied buying

Suppose over one month a popular equity ETF reports **\$20 billion of net inflows.** What does that imply for buying pressure, and how do you size its impact?

First, the mechanical buying. Net creations of \$20 billion mean the authorized participants delivered roughly \$20 billion of the underlying basket to the fund. That is \$20 billion of stock that *had* to be bought, irrespective of valuation — forced flow. (Illustrative figures.)

Now scale it against the market it hits. Suppose the basket trades roughly \$200 billion of volume over that month. The forced ETF buying is then about 10% of the dollar volume:

```
ETF net inflows                 = $20B
basket monthly dollar volume    = $200B
forced buying as share of volume = $20B / $200B = 10%
```

A persistent 10% of volume tilted to the buy side is an enormous, structural bid — easily enough to lift price against a neutral fundamental backdrop, and exactly the kind of flow that makes an "expensive" market keep climbing. The reverse is the warning: if next month the \$20 billion of inflows becomes \$5 billion of *outflows*, that same machine becomes \$5 billion of forced *selling*, and the bid you were leaning on vanishes. The one-sentence intuition: inflows are forced buying and outflows are forced selling, so the *change* in flows is often a better timing tool than the level of valuation.

### The second-order effects: passive flows, index inclusion, and reflexivity

Three deeper consequences make fund flows even more powerful than the simple funnel suggests.

The first is the rise of **passive investing**. A growing majority of new equity money goes into index funds that buy stocks in proportion to their market-cap weight, mechanically, with no view on any individual company. This has a strange consequence: the biggest stocks get the most forced buying simply because they are already the biggest, which makes them bigger, which earns them an even larger share of the next inflow. Passive flows are inherently *momentum-amplifying* on the way up — they channel the marginal saver's dollar disproportionately into whatever is already large and popular. The same machine runs in reverse in an outflow: passive selling hits the biggest names hardest, which is part of why mega-cap-led indices can fall fast once the flow turns. The flow does not evaluate; it weights.

The second is **index inclusion**. When a stock is added to a major index, every passive fund tracking that index *must* buy it, on the effective date, regardless of price — a one-time wall of forced buying. Traders have long front-run this: buy the stock when the addition is announced, sell into the forced index-fund buying on the inclusion date. The effect has shrunk as it became well-known and as funds got smarter about minimizing their footprint, but the mechanism is the purest possible illustration of forced flow: a price move driven entirely by *who must transact and when*, with zero new information about the business.

The third, and the deepest, is **reflexivity** — the feedback loop George Soros named, where price and flows feed each other. Rising prices attract inflows (performance-chasing: funds that went up get more money), the inflows force more buying, which lifts prices further, which attracts more inflows. The flow is not an independent input reacting to value; it is partly *caused by* the price it then *causes*. This is why trends in flow-driven assets can run far past any fundamental justification, and why they reverse so violently — the same loop that pulled money in on the way up shoves it out the door on the way down. Positioning, gamma, and flows are all expressions of this single uncomfortable truth: in the short and medium term, the market is not a weighing machine. It is a voting machine wired to a feedback loop, and the votes are dollars that often *have* to be cast.

## Common misconceptions

Positioning analysis is powerful and widely misused. Five myths cause most of the damage.

**Myth 1: "Positioning predicts direction."** It does not, and treating it as a directional signal is the fastest way to lose money. Positioning predicts *fragility* and *asymmetry*, not *timing*. A crowded long tells you the downside, if it comes, could be violent and that the upside fuel is spent — it does *not* tell you the top is in. Crowded markets routinely get more crowded for weeks. The correction: use positioning to size the *risk* and the *asymmetry* of a move, and use price, a catalyst, or a level to time it. In the COT example above, the edge was a -2.5% relative skew over four weeks with a 70% hit rate — a real but probabilistic tilt, not a guarantee.

**Myth 2: "The COT report is timely."** It is structurally late. Positions are measured Tuesday and published Friday — a snapshot that is already three days old when you see it, and stale-er by the time you can act. In fast markets the crowd can have already reversed. The correction: treat the COT as a slow, weekly *gauge* of where the chips sit, never as an entry trigger. If you need real-time positioning, that is what gamma and intraday flow are for.

**Myth 3: "Retail flow is just noise."** For a long time this was a safe assumption; GameStop in 2021 ended it. When small, individually-tiny traders coordinate — through social media, through the same zero-commission app, through the same handful of popular options — their aggregate flow can move large-cap stocks and even bend dealer gamma. Retail buying of short-dated call options, in particular, can force dealers short gamma and into a feedback loop. The correction: retail is noise *until it is correlated*, at which point it is one of the largest flows in the market. Watch for crowding in retail-favorite names and options.

**Myth 4: "If the smart money (commercials) is short, I should short too."** The commercials are *hedgers*, not speculators. A commercial net-short in oil might just be a producer locking in prices for next year's output — not a bet that oil falls. Their positioning is informative at extremes but it is not a directional recommendation; they are indifferent to the price level in a way you are not. The correction: read the commercials as a *contrarian context* (extreme commercial shorts often coincide with froth) rather than as a trade to copy.

**Myth 5: "Gamma tells me which way the market will go."** Gamma is direction-agnostic. It tells you the *character* of moves — damped (long gamma) or amplified (short gamma) — not the sign. A short-gamma tape is dangerous in *both* directions; it just happens to be most dangerous down, because fear moves faster than greed. The correction: use gamma to set your expectation for volatility and your stop placement, never to pick a side.

## How it shows up in real markets

Positioning and flows are not abstractions. They are the explanation for some of the most memorable market events of the last decade.

**The GameStop squeeze, January 2021.** The cleanest forced-flow event in living memory. Short interest had reached roughly 140% of the free float — meaning more shares had been borrowed and sold short than actually traded freely, an arithmetic impossibility to all close at once. When a coordinated wave of retail buying (much of it through short-dated call options) lifted the price, two forced flows stacked on top of each other. First, the shorts had to cover, buying the stock to close losing positions. Second, the dealers who had sold those retail calls were short gamma and had to buy the stock to hedge as it rose — a classic *gamma squeeze* layered on top of the short squeeze. The two mechanical bids fed each other and drove the stock from about \$20 to an intraday \$483 in two weeks, on no change whatsoever in the underlying business. It is the canonical illustration of the flow pyramid's top layer: forced flow setting the violence. (For the full anatomy, see the cross-linked case study below.)

**Crowded carry unwinds, August 2024.** For years one of the most crowded trades on earth was to borrow near-zero-yielding Japanese yen and use it to buy higher-yielding assets everywhere — US dollars, tech stocks, emerging-market currencies. The positioning was enormous and one-sided: everyone leaned the same way. On 5 August 2024, a small Bank of Japan rate nudge plus a soft US jobs report started to close the rate gap that powered the trade, and the crowd all tried to unwind through the same door at once. The yen ripped higher, the assets bought with borrowed yen got dumped, and the Japanese equity market fell 12.4% in a single session — its worst day since 1987. Nothing had *blown up*; a crowded position simply unwound. The VIX, the market's gauge of expected volatility, briefly spiked above 65 intraday. The mechanism was pure positioning: too many people on the same side of the same trade, and an exit too small for the crowd.

![VIX year-end closes near twenty with panic spikes in 2018 2020 and 2024](/imgs/blogs/following-the-flows-positioning-cot-dealer-hedging-4.png)

The chart above is the realized result of positioning unwinds, written in the language of volatility. The VIX — an index of expected 30-day S&P 500 volatility, measured in annualized percentage points — sits near its long-run average of about 19.5 most years, and year-end closes cluster in the low-to-high teens. But the spikes tell the real story. February 2018 (the "Volmageddon" blow-up of short-volatility products) hit 37; March 2020 (the COVID de-risking, the most violent forced-selling event of the era) hit 82.7; August 2024 (the yen carry unwind) hit 65.7 intraday. Each spike is a crowded position unwinding at once. Volatility is not random — it is the price-tag of forced flow, and it explodes precisely when positioning that built up quietly for months has to reverse in days.

**The 2018 short-volatility blow-up.** A subtler case worth its own mention. By early 2018 an enormous crowd had been *selling* volatility — betting, via products like the inverse-VIX ETN known as XIV, that calm would persist. They were collectively very short gamma. When volatility ticked up on 5 February 2018, the products that had to buy volatility to hedge their own losses created a feedback loop: a spike in the VIX forced more buying of volatility, which spiked it more. XIV lost about 90% of its value in a day and was shuttered. The fundamentals of the S&P 500 had barely moved. A crowded short-volatility position, sitting on negative gamma, simply detonated when the regime turned. This is the gamma mechanism on a market-wide scale.

The thread through all three: in each case, *nothing fundamental broke*. A position that had been quietly crowded — short GameStop, long the yen carry, short volatility — met a small shock, and the forced unwinding became the entire event. Watching the flows would have told you the fuel was there. Watching only the fundamentals would have left you baffled.

It is worth being precise about what "watching the flows" would and would not have given you in each case, because this is where most people overclaim. In none of the three did the flow data hand you the exact day. What it gave you was *advance knowledge of the asymmetry*: a GameStop short interest above 100% of float was a flashing sign that the downside for shorts was bounded and the upside pain was unbounded — a setup you simply do not stand in front of as a short, and one a contrarian could lean into with defined risk. The years-long crowding of the yen carry was visible in positioning surveys and in the relentless one-way grind of the trade — a sign that the *unwind*, whenever it came, would be violent and correlated across every asset funded by borrowed yen. The 2018 short-vol crowding was visible in the assets under management of the inverse-VIX products themselves — a public number that had ballooned to the point where a modest vol spike would force a self-feeding buy. In each case the flow data did not predict the trigger; it told you, in advance, that *if* a trigger came, the move would be far larger than the news deserved. That is the entire value proposition of flow analysis: not timing, but the size and direction of the tail.

## How to trade it / The playbook

Everything above converges into a practical discipline. You are not trying to predict; you are trying to read where the crowd is leaned, where the forced flows hide, and where the pain trade lives — then position with the asymmetry rather than against it.

![Positioning playbook of COT extremes dealer gamma and fund flows with the trade for each](/imgs/blogs/following-the-flows-positioning-cot-dealer-hedging-7.png)

The matrix above is the playbook on one page: three flow signals, what each tells you, and the trade each implies. Work through them as a checklist.

**1. Fade COT extremes — but only with a catalyst.** Track the large-speculator net position for the futures you trade (equity index, oil, gold, the major currencies, bonds) as a percentile of its own three-year history. When it hits the top or bottom decile, the crowd is maximally one-sided and the smart money (commercials) is maximally opposite. That is your signal to *lean against* the crowd — but not yet. Positioning is a gauge, not a trigger. Wait for price to confirm (a failed new high, a reversal bar) or for a catalyst (a data print, a policy surprise). The position: a defined-risk fade — a put spread, a short with a hard stop above the extreme — sized for the -2% to -3% kind of edge you measured, not a bet-the-farm reversal. The invalidation: a clean breakout to new highs on rising open interest means the crowd is *adding*, not exhausting — stand aside.

**2. Respect short-gamma; exploit long-gamma.** Check the sign of aggregate dealer gamma (the GEX estimates published by several vol-research shops). In a **short-gamma** regime, the tape is a trapdoor: widen your stops because normal support levels get blown through, do not assume dips get bought (the dealers are selling too), size down because realized volatility will be far above the calm reading, and respect the downside especially — fear hedges faster than greed. In a **long-gamma** regime, the tape is a trampoline: expect chop and pinning near the big strikes, fade moves toward those strikes, and do not chase breakouts that are likely to be sold back. The invalidation in either case is a regime flip — a large options expiry or a big new flow can flip the sign overnight, so re-check it daily.

**3. Watch flows at key levels.** Track ETF and fund flows for the assets you trade, and treat the *change* in flow as the signal. Persistent inflows are a structural bid that can keep an expensive market climbing; a flip to outflows removes the marginal buyer and is a far better top-warning than valuation. The highest-odds setup is *confluence*: a positioning extreme **and** a short-gamma regime **and** a flow inflection **at a key technical level**. When three independent flow signals point the same way at a level price already respects, you have a real edge — not because any one of them predicts the future, but because together they tell you the crowd is trapped and the pain trade is obvious.

**The meta-rule that ties it together:** positioning is both a *contrarian gauge at extremes* and a *momentum amplifier through dealer hedging*, and the skill is knowing which mode you are in. At a crowded extreme with no fresh fuel, fade it. Inside a short-gamma cascade where forced hedging is driving the move, do not stand in front of it — the same flows that make the eventual reversal violent are, right now, the thing carrying price away from you. Read the flows, find the forced participant, and ask the only question that matters: *who has to trade next, and in which direction?* The answer is where the move is going.

A final word of humility. Flows are probabilistic, the data is lagged, and crowded trades can stay crowded far longer than you can stay solvent fighting them. Positioning tells you where the fuel is and how violent the move could be; it does not light the match. Pair it with price, with a catalyst, and with ruthless risk control — and never confuse "the crowd is wrong" with "the crowd is wrong *today*."

## Further reading & cross-links

- [Risk-On, Risk-Off: How Money Rotates Between Assets](/blog/trading/macro-trading/risk-on-risk-off-how-money-rotates) — the regime backdrop that decides when crowded positioning unwinds all at once.
- [What Actually Moves a Currency: Rates, Flows, and the Carry Trade](/blog/trading/macro-trading/what-moves-exchange-rates-rates-flows-carry) — the yen carry trade as the textbook example of crowded positioning and a violent unwind.
- [The GameStop Short Squeeze of 2021](/blog/trading/finance/gamestop-2021-short-squeeze) — the full anatomy of the stacked short-squeeze and gamma-squeeze that drove the canonical forced-flow event.
- [What Liquidity Really Means for Traders](/blog/trading/macro-trading/what-liquidity-means-market-funding-global-traders) — why a crowded exit becomes a stampede when liquidity thins, the plumbing behind every positioning unwind.

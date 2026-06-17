---
title: "Dealer Gamma, Charm, and Vanna: How Options Flows Move the Spot"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How market makers hedging their option books move the underlying: long vs short gamma regimes, the gamma flip, charm flow into OPEX, and the vanna rally after a vol crush."
tags: ["options", "volatility", "dealer-gamma", "gamma-exposure", "vanna", "charm", "market-makers", "delta-hedging", "opex", "0dte"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Dealers are the forced counterparty to your option trades, and the stock they buy and sell to hedge their *aggregate* Greek position is large enough to move the underlying itself. The sign of their net gamma decides whether that flow calms the tape or accelerates it.
>
> - **Long-gamma dealers sell rallies and buy dips** -- their hedging *dampens* moves and pins price to big strikes. **Short-gamma dealers buy rallies and sell dips** -- their hedging *amplifies* moves into air pockets. The level where their net gamma crosses zero is the **gamma flip**.
> - **Charm** (delta decaying with time) creates a mechanical hedge drip into **OPEX Friday**; **vanna** (delta shifting as implied vol falls) fuels the slow **grind-up** after a vol crush.
> - A single dealer desk that is short 50,000 at-the-money S&P contracts has to trade **~462,000 shares** (about **\$0.23bn**) just to re-hedge a routine **1% move** -- that is the size of the tail that wags the dog.
> - **The one rule:** treat dealer-flow (GEX, charm, vanna) as a *pressure*, never a *forecast*. It tells you whether to fade or respect a move and how to size -- it does not tell you the direction.

## A pin, an air pocket, and the tail that wags the dog

On the third Friday of most months, something quietly strange happens to the U.S. stock market. As the afternoon wears on, the S&P 500 stops wandering. It drifts toward a round number where an enormous pile of options is set to expire, and it sticks there -- 5,000, say, or 4,500 -- as if the index were a marble that rolled into a bowl. Volume is heavy, news is light, and the price barely moves off the magic strike. Traders call this a **pin**. It is not a coincidence and it is not manipulation. It is the mechanical residue of thousands of dealers hedging their option books into expiration.

Now run the tape in the other direction. On the morning of **5 August 2024**, the unwinding of the yen carry trade slammed into a market that was already thinly positioned. The VIX spiked to a **38.57** close -- a level it normally only touches in genuine crises -- and the S&P gapped down hard before recovering. The move was wildly out of proportion to the news. Part of the reason was that dealers were caught **short gamma**: every tick down forced them to sell *more* futures to stay hedged, which pushed price lower, which forced them to sell again. The hedge became the move. That is an **air pocket**, and it is the mirror image of the pin.

These two faces -- the calming pin and the accelerating air pocket -- are the same mechanism seen from opposite sides. Behind both sits a population of professional **dealers** (market makers) who never wanted a directional view at all. They took the other side of customer option flow because that is their job, and now they must trade the underlying to neutralize the risk they inherited. Their hedging is not a bet; it is plumbing. But the plumbing is enormous, and when enough of it points the same way, **the options tail wags the dog**. This post is about how that happens, how to estimate its size with the Black-Scholes Greeks, and -- the part most "GEX" commentary skips -- where the story breaks down.

![Feedback loop showing customer option flow forcing dealer hedging that moves the spot and loops back](/imgs/blogs/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot-1.png)

The loop in the figure is the spine of everything that follows. Customers trade options for protection or leverage; dealers absorb the other side and inherit a net Greek position; they delta-hedge in the underlying to flatten the risk; that hedging moves the spot; and the new spot changes their Greeks, which forces more hedging. Read it once, clockwise, and the rest of this post is just filling in the four boxes with numbers.

## Foundations: who the dealer is and why they must hedge

Before any of the flow story makes sense, you need three building blocks: who the dealer is, what *delta-hedging* means, and why the dealer's **aggregate** position -- not any single trade -- is what moves price. If the Greeks are still fuzzy, the [delta](/blog/trading/options-volatility/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral) and [gamma](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) posts build them from scratch; here I will assume you know that delta is the option's sensitivity to the underlying and gamma is the rate at which delta changes.

### The dealer is the forced counterparty

When you buy a call on an exchange, someone sells it to you. Most of the time that someone is not another retail trader with an opposite view -- it is a **market maker**: a firm that quotes both a bid and an ask on thousands of strikes and earns the spread between them. They are happy to sell you the call. What they are *not* happy to do is take your directional bet. A market maker's edge is the bid-ask spread and the [variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt), harvested across an enormous, diversified book. Carrying your view on the direction of the S&P would swamp that edge with noise on the first bad day. So the moment they sell you the call, they immediately try to cancel its directional risk. The mechanics of how a desk thinks about this are the whole subject of the [market-maker post](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade); here we only need the consequence.

If you want the institutional context -- who these firms are, how they connect to the exchange, why high-frequency infrastructure matters -- the [market makers and high-frequency trading](/blog/trading/finance/market-makers-and-high-frequency-trading) piece covers the plumbing. For our purposes, the dealer is simply *the other side*, and the other side has to hedge.

### Delta-hedging in one paragraph

An option's **delta** is how many shares of the underlying it behaves like. A call with a delta of 0.54 moves about \$0.54 for every \$1.00 the stock moves; it is, locally, like owning 54 shares per 100-share contract. When a dealer sells you that call, they are now *short* 0.54 of a share's worth of upside per share of contract. To neutralize it, they **buy** 0.54 shares' worth of the underlying (54 shares per contract). Now if the stock rises \$1, the option they are short loses them \$0.54 and the stock they bought gains them \$0.54. They are flat. That is **delta-hedging**: holding exactly enough of the underlying to cancel the option's first-order directional exposure. Do it across the whole book and the desk is, instant to instant, indifferent to which way the market goes -- which is exactly what they want.

The catch is the word *instant*. Delta is not constant. As the stock moves, the option's delta changes, and the hedge that was perfect at \$500 is wrong at \$505. The rate at which delta changes is **gamma**, and gamma is where the feedback comes from.

#### Worked example: how big is one desk's re-hedge?

Take a single dealer desk that has **sold 50,000 at-the-money call contracts** on a SPY-like underlying trading at **\$500**, with implied vol of **15%** and **30 days** to expiry. Each contract controls 100 shares (the multiplier). From the Black-Scholes model, the **per-share gamma** of one such at-the-money option is about **0.01847**. Gamma is the change in delta per \$1 move, so the desk's *aggregate* gamma -- the change in its net delta in shares per \$1 the stock moves -- is:

```
agg_gamma = gamma_per_share x contracts x multiplier
          = 0.01847 x 50,000 x 100
          = 92,325 shares of delta per $1 move
```

Now the stock rallies **1%**, from \$500 to **\$505** -- a \$5 move. The desk's net delta changes by roughly:

```
delta_change = agg_gamma x move
             = 92,325 x 5
             = 461,627 shares
```

The dealer must trade **~462,000 shares** -- about **\$231 million** of stock at \$500 -- just to re-hedge a single, utterly routine 1% move. Multiply that across dozens of desks all positioned similarly and you have a hedging footprint measured in the *billions of dollars per percent*. **That is the tail.** The intuition: a position you would casually describe as "50,000 contracts" is, in hedging terms, a half-million-share trading program triggered by a move the news barely notices.

### Why aggregate, not individual, positioning matters

One dealer hedging 462,000 shares is meaningful but absorbable. The reason dealer flow *moves* markets is that the customer base is lopsided in a structurally predictable way, so dealers across the street end up positioned the **same way** at the same time. The classic example: index investors and pension funds are chronic *buyers of downside protection* (puts) and chronic *sellers of upside* (covered calls and overwriting). That pushes the aggregate dealer book toward being **long puts hedged, short calls hedged** -- and crucially, toward being **long gamma** in the broad index during calm periods, because they are net long the options that institutions overwrite. The aggregate sign of dealer gamma is therefore not random; it is the fingerprint of how the customer base habitually trades. When the whole street leans one way, individual hedging trades stop cancelling each other and start summing into a directional flow. That sum is what we are going to model.

To make this concrete, trace a single customer trade all the way through to its market footprint. A pension fund buys 10,000 contracts of a six-month index put to protect a portfolio. The dealer who sells it is now short 10,000 puts -- a position with *positive* delta (a short put gains when the market rises), *negative* gamma (the position gets shorter delta as the market falls), *negative* vega (it loses when implied vol rises), and a charm and vanna profile that will matter as expiry and vol shift. The dealer hedges the delta immediately by shorting index futures. But they cannot hedge the gamma, vega, charm, and vanna away with a static futures position -- those are *option* risks, and futures have none of them. So the dealer is left carrying a residual short-gamma, short-vega book that they must dynamically manage, re-hedging the delta every time the market moves. The pension fund's single defensive trade has been converted, on the dealer's side, into a standing program to *sell futures when the market falls and buy them when it rises* -- the destabilizing short-gamma flow -- for the next six months. Now multiply by every fund running the same protection, and the structural short-gamma-below, long-gamma-above shape of the index dealer book emerges. The flow is not a conspiracy; it is the arithmetic of who buys what.

There is one more foundational distinction worth nailing down, because it is where a lot of GEX commentary gets sloppy: **the dealer's gamma sign is the *opposite* of the customer's**, but the *flow* it generates depends on the dealer being the one who hedges. Customers who buy options are long gamma -- but most customers do *not* delta-hedge; they hold the option for its payoff. Dealers, who are short those same options and therefore short gamma, *do* hedge, continuously. So the market-moving flow comes from the short-gamma, hedging side. When you hear "dealers are short gamma," the operational meaning is "the hedging population is short gamma, so their re-hedging will lean *into* moves." That is the whole reason a population of long-gamma *buyers* (customers) coexisting with short-gamma *hedgers* (dealers) produces a destabilizing flow: only one side is mechanically forced to trade the underlying.

## Long gamma versus short gamma: the two dealer regimes

Everything about whether dealer hedging *calms* or *accelerates* the market comes down to one binary: is the aggregate dealer book **long gamma** or **short gamma**? The two regimes produce exactly opposite hedging behavior in response to the same price move, and that difference is the most important single fact in this entire post.

### Long gamma: the dealer is a built-in stabilizer

Suppose the aggregate dealer book is **long gamma** -- they own more gamma than they are short. Long gamma means their delta *increases* when the stock rises and *decreases* when it falls. To stay delta-neutral, they must constantly trade *against* the move: when the stock rallies and their delta climbs, they **sell** the underlying to bring delta back to flat; when it dips and their delta falls, they **buy**. Sell rallies, buy dips. This is the textbook profile of someone running a long [gamma-scalping](/blog/trading/options-volatility/gamma-scalping-turning-a-long-straddle-into-a-vol-harvest) book, just scaled to the whole market.

The consequence is profound: long-gamma dealer hedging **leans against** every move. It supplies liquidity exactly when the market wants to run, dampening realized volatility and tugging price back toward where the dealers are most exposed -- typically the big open-interest strikes. This is the mechanical engine behind **pinning**. The tape feels heavy and mean-reverting; intraday ranges compress; selloffs find a "magic bid" and rallies hit an invisible offer.

### Short gamma: the dealer becomes an accelerant

Now flip the sign. When the aggregate book is **short gamma**, the dealer's delta *falls* as the stock rises and *rises* as it falls. To re-hedge, they must trade *with* the move: when the stock rallies and their delta drops (gets more negative), they **buy** the underlying to catch up; when it dips and their delta climbs (gets more positive), they **sell**. Buy rallies, sell dips. This is precisely the [toxic short-gamma](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) profile that bites option sellers, now operating at market scale.

Short-gamma hedging **amplifies** every move. A move up triggers dealer buying, which pushes price further up, which forces more buying. A move down triggers dealer selling, which deepens the drop, which forces more selling. The flow becomes a positive-feedback loop. Realized volatility *expands*; ranges blow out; and a small shock can detonate into an **air pocket** as the hedging chases price into a vacuum of liquidity. The 5 August 2024 spike, parts of the February 2018 "Volmageddon," and the late-December 2018 selloff all carried this fingerprint.

![Two lines showing dealer shares traded versus spot move, opposite slopes for long and short gamma](/imgs/blogs/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot-2.png)

The chart makes the asymmetry concrete. It plots, for our 50,000-contract desk, the shares the dealer must trade as a function of how far spot has moved. The two lines have *opposite slopes*. The long-gamma desk (green) sells into rallies and buys into dips -- its hedging always pushes back toward zero, a restoring force. The short-gamma desk (red) does the reverse, leaning into every move. Same desk, same gamma magnitude, same 1% move; the only thing that changed is the **sign**, and the sign flips the market's character from stable to unstable.

#### Worked example: the same 1% move, dampened vs amplified

Use the desk from before -- aggregate gamma of **92,325 shares per \$1**, and a **+\$5 (1%)** rally. The *magnitude* of the re-hedge is the same in both regimes: **461,627 shares**, worth about **\$231 million** at \$500. What differs is the *direction*, and therefore the effect on price:

```
LONG gamma:  delta rose on the rally -> SELL 461,627 shares
             -> hedging supplies offers into strength -> move DAMPENED
SHORT gamma: delta fell on the rally -> BUY 461,627 shares
             -> hedging adds demand into strength -> move AMPLIFIED
```

In the long-gamma case the dealer is dumping a quarter-billion dollars of stock into the rally, capping it. In the short-gamma case the dealer is *chasing* the rally with a quarter-billion dollars of buying, extending it. The number 461,627 is identical; its sign is everything. **The intuition: dealer gamma does not predict direction -- it sets whether the existing direction gets faded or fed.**

### The gamma flip: where the regime changes

Here is the part that makes this tradeable rather than merely interesting. The dealer book is not uniformly long or short gamma across all price levels. It depends on *where spot sits relative to the strikes that carry the open interest*. Above a certain level, calls dominate the gamma and dealers are net long gamma (stabilizing). Below it, puts dominate and dealers are net short gamma (destabilizing). The price level where the aggregate dealer gamma crosses **zero** is called the **gamma flip** (or "zero-gamma level"), and it acts like a regime boundary.

Practitioners estimate the dealer book's gamma at every spot level and call the result **gamma exposure**, or **GEX** -- a dollar figure (positive or negative) for how much stock the street must trade per a given percentage move. Positive GEX means net-long-gamma dealers; negative GEX means net-short-gamma dealers. The zero-cross is the flip.

![GEX curve crossing zero, positive and stabilizing above the flip, negative and destabilizing below](/imgs/blogs/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot-3.png)

The figure shows a stylized GEX profile against the index level. Above the flip (the amber line), GEX is positive and shaded green: dealers fade moves, vol is suppressed, the tape pins. Below the flip, GEX is negative and shaded red: dealers chase moves, vol is amplified, the tape gaps. The practical takeaway is that **the same index can behave like two completely different markets on either side of one number.** A market trading 2% above its flip is a sleepy, range-bound, fade-the-extremes tape; the same market 2% below its flip is a treacherous, momentum-driven, do-not-catch-the-knife tape. Knowing roughly where the flip sits tells you which game you are playing -- not where price is going, but *how it will travel.*

A crucial nuance: crossing below the flip is itself destabilizing, because it *increases* the short-gamma hedging that pushes price down further. That is why selloffs through a major flip level so often accelerate: the act of breaking the level recruits more sellers. It is also why these levels are watched -- not as price targets, but as inflection points in market *behavior*.

### How GEX is actually estimated (and why it is fragile)

It helps to know roughly how a GEX number is built, because the construction reveals exactly where it can mislead you. The estimator starts from the publicly visible **open interest** at every strike across the listed expirations of an index or its key components. For each strike it computes the option's gamma from a Black-Scholes-style model at the current spot, multiplies by the open interest and the contract multiplier to get a per-strike *dollar gamma*, and then -- this is the load-bearing assumption -- it *assigns a sign* based on a rule about who is long and who is short. The standard convention is to assume customers are net long calls and net short puts (or some variant), which makes dealers short calls and long puts, and from that you get the familiar "dealers long gamma above, short gamma below" profile. Sum across all strikes and expirations and you have an aggregate GEX curve; find where it crosses zero and you have the gamma flip.

Every step after "open interest" is an *assumption*. The model picks an implied vol per strike. The sign rule guesses positioning that is genuinely unobservable -- the exchange does not publish who is long. The estimator usually ignores OTC and structured-product flow that never shows up in listed open interest. And it is a *snapshot*: open interest updates once a day, while the gamma it implies changes with every tick of spot and vol. The result is a number that is *directionally informative* -- the sign and the rough location of the flip are usually right -- but *quantitatively soft*, and softest precisely in a fast market where positioning is churning. This is the difference between using GEX to say "we are probably in a short-gamma regime, expect amplified moves" (sound) and "GEX is -\$4.7bn so the market will fall 1.3% today" (false precision dressed up as analysis).

#### Worked example: turning open interest into a GEX number

Take one strike to see the arithmetic. Suppose the 4,950 puts on an index near \$5,000 carry **120,000 contracts** of open interest, with a per-share gamma of **0.0011** at the current spot and a 50-multiplier (index style). The dollar gamma at that strike, expressed as the change in dealer delta-dollars per **1%** spot move, is approximately:

```
strike_dollar_gamma = gamma x OI x multiplier x spot x (spot x 0.01)
                    = 0.0011 x 120,000 x 50 x 5,000 x 50
                    = 0.0011 x 120,000 x 50 x 5,000 x (5,000 x 0.01)
```

Working it through: `0.0011 x 120,000 x 50 = 6,600` shares of delta per \$1; times spot \$5,000 gives \$33.0M of delta-dollars per \$1 move; times the \$50 (= 1% of \$5,000) gives **~\$1.65bn of dealer delta-dollars that must be traded per 1% move at this one strike**. Apply the sign rule -- if the convention says dealers are *long* these puts the contribution is negative GEX, if *short* it is positive -- and you have this strike's signed contribution. Repeat for thousands of strikes and sum. **The intuition: a single heavily-traded strike near the money can carry a billion-plus dollars of hedging sensitivity, which is why a few big strikes dominate the whole GEX curve -- and why the sign assumption on those strikes can flip the entire reading.**

### Index flow versus single-name flow

The dealer-gamma story is strongest in the broad index (S&P 500, its futures, and the major index ETFs) and weakest in random single stocks, and the reason is worth understanding because it stops you from over-applying the lens. In the index, three things line up: enormous, concentrated open interest at round-number strikes; a structurally lopsided customer base (everyone hedges *the market*); and deep, liquid futures that let dealers hedge cheaply and continuously. The flow is large relative to the daily volume, so it bites.

In a single mid-cap stock, none of that holds reliably. Open interest is thin and scattered; the customer base is a mix of bulls and bears with no structural tilt; and the dealer's hedge competes with idiosyncratic news, insider flow, and index-rebalancing that swamp the gamma effect. There are exceptions -- a heavily-optioned megacap into earnings, or a meme stock where retail call-buying forces a genuine **gamma squeeze** as dealers chase their short-call hedge upward -- but as a default, apply the dealer-flow lens to *the index and its biggest, most-optioned names*, and be skeptical of GEX overlays on illiquid single stocks. The mechanism is universal; the *magnitude relative to liquidity* is what makes it tradeable, and that magnitude lives in the index.

The gamma squeeze deserves its own note because it is the single-name case where flow *does* dominate. When a crowd of retail traders buys massive quantities of short-dated OTM calls on one stock, the dealers who sold them are short calls -- short gamma, with delta that grows as the stock rises. To hedge, they buy the stock; their buying lifts it; the higher price grows their short-call delta; they buy more. That is a short-gamma feedback loop pointed *up*, and in a stock with limited float it can detonate into a vertical squeeze. It is the same mechanics as the index air pocket, sign-flipped to the upside and concentrated in one name. The lesson is symmetric: short-gamma dealer hedging amplifies whichever direction the move starts, up or down.

## Charm: the delta that bleeds into expiration

Gamma is the dealer's reaction to *price* moving. But two of the most reliable flow patterns come from the dealer's reaction to *time* and *volatility* moving even when price does not. The first is **charm**.

**Charm** is the rate at which an option's delta changes as time passes, holding spot constant. (It is one of the second-order Greeks covered alongside [vanna and volga](/blog/trading/options-volatility/rho-dividends-and-the-second-order-greeks-vanna-volga-charm).) Intuitively: as expiration approaches, an out-of-the-money option becomes more and more certain to expire worthless, so its delta drifts toward zero; an in-the-money option becomes more certain to be exercised, so its delta drifts toward 1.0 (or -1.0 for puts). Either way, with spot pinned, **delta decays toward its terminal value as the clock runs.** That decay is charm, and it forces the dealer to unwind the hedge they put on against that delta.

This is why monthly options expiration -- **OPEX**, the third Friday -- has a recognizable flow signature. Through the week, if spot is roughly flat, the delta the dealer hedged against bleeds away, and the dealer mechanically trims the corresponding stock position. The flow is one-directional, news-free, and concentrated into Thursday afternoon and Friday morning. Traders call the resulting tape the **OPEX drift**.

![Time figure showing dealer delta bleeding day by day toward Friday OPEX with a mechanical sell flow](/imgs/blogs/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot-4.png)

The figure walks the week. On Monday the desk holds a hedge sized to the options' delta. By Wednesday, with spot unchanged, that delta has bled toward zero on its own -- *time did the work, not price.* Thursday the bleed accelerates (charm grows as expiry nears), and into Friday the residual hedge gets dumped. The result is a quiet, persistent drift in the tape that correlates with nothing in the news, because its cause is mechanical.

#### Worked example: estimating the charm flow into OPEX

Suppose a dealer desk is **short 80,000 slightly out-of-the-money call contracts** -- the 505-strike on our \$500 underlying, with **15%** vol -- because customers bought them as a cheap lottery on a rally that has not come. Spot stays flat at \$500. We need the *change in the call's delta over one day*, with two days to expiry.

From Black-Scholes, the 505 call's delta is about **0.192** with two days left and about **0.106** with one day left. Because the customers are *long* the call, the dealer is *short* it, so to hedge their negative delta the dealer holds a *long* stock position equal to the call's delta:

```
hedge held at 2 days = call_delta x contracts x multiplier
                     = 0.192 x 80,000 x 100 = 1,535,339 shares (long)
hedge held at 1 day  = 0.106 x 80,000 x 100 =   845,995 shares (long)
charm sell flow      = 1,535,339 - 845,995  =   689,344 shares
```

Overnight, with spot unchanged, charm pulls the call's delta down by **~0.086 per contract**, and the dealer mechanically **sells ~689,000 shares** to shrink the now-oversized hedge. No view, no news -- just the clock. Stack several such desks and the OPEX drift becomes a multi-million-share, one-way program into Friday's close. **The intuition: charm turns the calendar into a hedging schedule; the dealer is forced to trade time even when price stands still.**

The sign of the charm flow depends on the dealer's positioning. Here, short upside calls, the flow is to *sell* as those calls decay. A desk that is net short downside puts (the more common index case) experiences charm flow that *buys* into expiration as those put deltas decay -- which is one reason the OPEX and post-OPEX window in equity indices has historically leaned mildly bullish. The mechanics are symmetric; only the sign flips with the book.

Two refinements separate the people who understand charm from those who merely repeat the OPEX-drift folklore. First, **charm is largest for near-the-money options close to expiry and for moderately out-of-the-money options.** A deep-in-the-money option already has a delta near 1.0 and a deep-OTM option already has a delta near 0 -- neither has far to travel, so neither bleeds much. The options whose delta is mid-range and *uncertain* are the ones whose delta charm pulls hardest, and those tend to cluster in a band around the money. That is why the charm flow concentrates: it comes from the strikes near where spot actually is, which are also the strikes with the heaviest open interest. Second, **charm and gamma fight or reinforce depending on whether spot moves.** The charm flow described above assumes spot is *flat*. If spot moves during the OPEX week, gamma hedging is layered on top of the charm hedging, and the two can either add (a flat-then-down week amplifies the sell flow) or partly cancel. The clean charm signature only shows up on the quiet, range-bound weeks -- which, conveniently, are also the weeks when there is little else moving the tape, so the drift is visible.

There is also a feedback wrinkle worth flagging. As an option approaches expiry its gamma *spikes* near the money -- a one-day-to-expiry at-the-money option has enormous gamma because its delta swings from near 0 to near 1 over a tiny price range. So OPEX week combines *decaying* charm flow with *exploding* gamma sensitivity near the pinning strike. In a long-gamma regime this is doubly stabilizing: the charm flow drips one way while the soaring gamma clamps price ever harder to the strike. It is the single most powerful pinning setup in the calendar, and it is why so many monthly expirations close glued to a round number.

## Vanna: the delta that shifts when fear drains away

The second time-and-vol pattern is **vanna**, and it is the engine behind one of the most reliable post-event tapes in the market: the slow, relentless **grind-up** after a scare resolves.

**Vanna** is the rate at which an option's delta changes as *implied volatility* changes, holding spot constant. (Again, see the [second-order Greeks](/blog/trading/options-volatility/rho-dividends-and-the-second-order-greeks-vanna-volga-charm) for the formal treatment.) The intuition: an out-of-the-money option's delta is partly a measure of *how likely it is to come into the money*, and higher implied vol means a wider distribution of outcomes, which makes a given OTM strike more likely to be reached. So when implied vol **falls**, the OTM option's delta shrinks toward zero. The dealer hedging that delta must then unwind part of their hedge -- and the direction of that unwind is the vanna flow.

This matters most after **events**. Before an earnings report, an FOMC meeting, or a CPI print, implied vol is elevated because the market is pricing an uncertain jump -- the [expected move](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options) and the [vol crush](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush) posts cover this pricing in depth. Once the event passes and the uncertainty resolves, implied vol **collapses** -- the famous vol crush. That collapse mechanically shifts dealer deltas via vanna, and the resulting hedge unwind is the **vanna rally**.

![Line showing dealer short hedge shrinking as implied vol falls, producing a buy-to-cover grind up](/imgs/blogs/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot-5.png)

The chart plots the dealer's required hedge against implied vol for a desk short a pile of downside puts. Customers hold the puts as protection; the dealer is short them and therefore holds a **short** stock hedge (negative shares) to offset the puts' positive contribution to dealer delta. As implied vol falls, the put deltas shrink, the dealer's short hedge shrinks too, and the dealer **buys back** stock -- a steady bid that lifts the tape with no news behind it. The grind-up feels almost suspicious in its smoothness precisely because it is plumbing, not opinion.

#### Worked example: estimating the vanna flow from a vol crush

A desk is **short 60,000 of the 480-strike puts** (customers bought them as crash insurance) on the \$500 underlying, **20 days** to expiry. Going into a binary event, implied vol on those puts is **30%**. The event passes calmly and vol crushes to **22%** -- an 8-vol-point drop. We need the change in the put's delta from that crush.

From Black-Scholes, the 480 put's delta is about **-0.259** at 30% vol and about **-0.195** at 22% vol. Because the dealer is *short* the put, their contribution to dealer delta is the negative of that: **+0.259** at 30%, **+0.195** at 22%. To neutralize a positive delta the dealer holds a **short** stock hedge equal in size:

```
dealer short hedge at 30% IV = -0.259 x 60,000 x 100 = -1,551,675 shares
dealer short hedge at 22% IV = -0.195 x 60,000 x 100 = -1,167,716 shares
vanna buy-back flow          = -1,167,716 - (-1,551,675)
                             = +383,959 shares (the dealer BUYS to cover)
```

The 8-point crush shrinks each put's delta by **~0.064**, and the dealer **buys back ~384,000 shares** of their short hedge -- a quarter-billion-dollar bid that materializes purely because fear drained out of the options, not because anyone turned bullish. This is why markets so often grind higher in the day or two *after* a feared event passes without incident: the vanna flow does the buying. **The intuition: a vol crush is a delta crush for dealers, and unwinding that delta is mechanical demand.**

Vanna and charm often stack. The window after a calm event, heading into an OPEX, can carry *both* a vanna buy-back (from falling vol) and a charm buy-in (from decaying put deltas) at once -- the combined "vanna-charm" flow that quant desks point to as the structural tailwind behind so many of the market's quiet, news-free advances. They are also why a *failed* crush is so dangerous: if the event disappoints and vol *rises* instead, vanna runs in reverse and dealers sell, adding to a decline.

The deeper reason vanna is a *systematic* tailwind for equities, and not just an event artifact, is the **persistent put skew** in index options. As the [volatility smile and skew](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more) post explains, OTM index puts trade at higher implied vols than OTM calls -- the post-1987 "smirk." Because institutions are forever buying those skewed puts and dealers are forever short them, the dealer book carries a large, standing *negative* vanna exposure to the downside puts. Whenever broad implied vol drifts lower -- which it does most of the time, since vol mean-reverts down from its spikes far more often than it spikes up -- that negative vanna translates into a slow, persistent dealer *buy* of the underlying. This is a structural reason equity indices tend to grind higher during low-vol regimes and grind for *longer* than fundamentals alone would suggest: the vol-down drift is being monetized as upside delta through the dealer hedge. It is the flow-mechanics counterpart to the [variance risk premium](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) -- the same structural short-vol, skew-bid market posture, seen through the lens of hedging flow rather than P&L.

The flip side is the **vanna unwind in a vol spike**, and it is brutal. When a shock hits and implied vol leaps, the standing negative vanna runs in reverse at speed: dealer put deltas *grow*, the dealer's short hedge *grows*, and they sell more underlying into a falling market. Vanna selling, gamma selling (if below the flip), and outright risk reduction all point the same way at once. That triple-stacking is what converts an ordinary 1-2% down day into a 5% air pocket, and it is the reason vol spikes are so much sharper and faster than the rallies that precede them. Markets, the saying goes, take the stairs up and the elevator down -- and a large part of that asymmetry is dealer vanna and gamma flipping from a gentle tailwind into a violent headwind.

## Putting a number on the tail: how big is the hedging footprint?

It is one thing to say dealer hedging "moves the market." It is another to size it. The honest way is the same Black-Scholes machinery: gamma times notional gives shares-to-trade, integrated over the move.

![Bar chart of estimated hedging volume in shares for increasing spot moves](/imgs/blogs/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot-6.png)

The chart sizes the tail for our single 50,000-contract desk across a range of moves. At a 0.5% move the desk re-hedges ~231,000 shares; at 1% it is ~462,000; at 2% it is ~923,000; at 3% it is ~1.4 million shares -- roughly **\$0.69 billion** of stock for one desk on a 3% day. The relationship is close to linear in the move size for small moves (because gamma is roughly constant locally), which is exactly the constant-gamma approximation: shares-to-hedge ≈ aggregate-gamma × move. The point is the **order of magnitude**. One desk's mechanical response to an ordinary day is a hundreds-of-thousands-of-shares program. The whole street, leaning the same way, is the dog being wagged.

This is also where GEX commentary earns its keep *and* overreaches. A reported "GEX of -\$5bn per 1%" is a useful estimate of how much stock dealers must trade per 1% move, and its sign tells you the regime. But it is an *estimate*, built on assumptions about who owns what and how it is hedged, and it is most fragile exactly when it matters most -- in a fast market, where positioning is shifting by the minute. Use the magnitude to calibrate your expectations for realized volatility; do not use it as a price target.

### Why the flow compounds: the second-order feedback

The linear estimate -- shares ≈ gamma × move -- understates the danger in a short-gamma regime, because gamma itself is not constant. As price moves toward the strikes with the heaviest open interest, gamma *rises*, which means each subsequent equal-sized move requires a *larger* re-hedge than the last. In a long-gamma regime this is benign: the stabilizing flow gets stronger the further price strays, which is why pins are so sticky. But in a short-gamma regime it is the opposite -- the destabilizing flow *accelerates* as price moves, so a selloff that starts gently can compound into a rout as it approaches a wall of short-gamma strikes. This convexity-of-the-flow is why air pockets are non-linear events: they are not "1% of selling per 1% down" but something closer to "increasing selling per increasing decline," at least until price clears the dense strike region. The constant-gamma estimate is a floor on the hedging footprint in a fast market, not a ceiling.

There is also a *liquidity interaction* that the pure Greek math misses. The dealer's re-hedge has to be *executed* in the underlying, and execution itself moves price -- the slippage and impact that the [delta-hedging in practice](/blog/trading/options-volatility/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral) post quantifies. On a calm day, 462,000 shares is a manageable program executed over hours with little impact. On a fast day, the same 462,000 shares hitting a thin book *is* the move -- and a thinner book means more impact per share, which means a bigger price move, which means a bigger gamma-driven re-hedge. The flow and the liquidity feed each other. This is why the same nominal GEX produces a gentle drift on a quiet Tuesday and a violent gap on a panicked Monday: the hedging size is similar, but the market's capacity to absorb it has collapsed.

## Common misconceptions

**"GEX predicts where the market will go."** No. GEX is an estimate of the *dealer hedging response to a move*, not a forecast of the move. Positive GEX says "if price moves, dealers will lean against it" -- it makes a *range* more likely, not an up-day. In our worked example the re-hedge was **461,627 shares** in *both* the long- and short-gamma cases; the sign told you whether that flow would dampen or amplify, never which way price would break first. GEX shapes the *character* of the tape (calm vs jumpy), not its direction. Anyone selling you GEX as a directional crystal ball is selling you the wrong product.

**"Pinning is the index being manipulated to the max-pain strike."** No. Pinning is an emergent byproduct of long-gamma dealers selling rallies and buying dips around the strike where their gamma is largest. "Max pain" -- the strike that would expire the most options worthless -- often sits near the pin, but it is a *consequence* of where the open interest piled up, not a level anyone steers toward. The mechanism is each dealer independently flattening their own delta; the pin is what that uncoordinated hedging sums to. See the [pin-risk and expiration mechanics](/blog/trading/options-volatility/assignment-pin-risk-and-expiration-day-mechanics) post for the assignment side of this.

**"Dealers are always short gamma, so they always make markets crash faster."** No -- it is regime-dependent and, in the broad index, dealers are *often net long gamma* in calm periods because institutions habitually overwrite calls and the street ends up long that gamma. The dangerous short-gamma regime tends to appear after spot has fallen through the [gamma flip](/blog/trading/options-volatility/the-net-greeks-of-a-position-building-your-risk-dashboard), or in heavily put-protected names. The whole reason the flip *matters* is that it separates two opposite behaviors; if dealers were permanently short gamma there would be no flip to watch.

**"The charm/vanna flows are big enough to override fundamentals."** No. They are *pressures*, typically meaningful on quiet days and swamped on news days. A 384,000-share vanna buy-back is a real bid on a calm Tuesday; it is a rounding error on a day the Fed surprises with 50 basis points. These flows explain the *grind* between catalysts -- the suspiciously smooth drift up after a calm event -- not the catalysts themselves. When a real shock hits, fundamentals and panic dominate, and the flow merely adds to (short gamma) or leans against (long gamma) the move.

**"0DTE options created this and it is all new."** Partly. Dealer hedging flow is as old as listed options; charm and the OPEX drift have been documented for decades. What [0DTE](/blog/trading/options-volatility/0dte-and-the-rise-of-short-dated-options-the-new-market-structure) options changed is the *tempo*: with enormous volume in options that expire the same day, gamma becomes huge and concentrated near the money, so the hedging response is faster, more intraday, and more sensitive to small moves. The mechanism is unchanged; the clock got faster and the gamma got sharper.

## How it shows up in real markets

**The monthly OPEX pin.** The most reproducible pattern. In a calm, long-gamma regime, the index tends to gravitate toward the strike carrying the largest open interest into the third-Friday expiration, and to stick there through the morning as dealer hedging fades every wiggle. You see it as compressed intraday ranges, heavy volume at a round number, and a close pinned within a few points of a major strike. It dissolves the moment a genuine catalyst overwhelms the hedging flow -- which is precisely why a pin that *breaks* on news often breaks hard: the stabilizing flow was masking how thin the real liquidity underneath was.

**The short-gamma air pocket: August 2024 and December 2018.** When spot falls through a major gamma flip, dealer hedging switches from leaning-against to leaning-into the decline. The **5 August 2024** yen-carry unwind drove the VIX to a **38.57** close on a move that, by fundamental news, looked modest -- the short-gamma feedback turned a wobble into a gap. The late-**December 2018** selloff (the worst December for U.S. stocks since the Depression) carried the same signature: thin holiday liquidity, dealers below their flip, and hedging that sold into every leg down. In both cases the flow did not *cause* the selloff, but it *shaped* it -- converting an orderly decline into an air pocket. Compare these to the genuine crisis spikes in the [VIX](/blog/trading/options-volatility/the-vix-and-vol-products-vix-vxx-uvxy-and-the-cost-of-the-roll) history -- the **82.69** COVID peak in March 2020, the **80.86** GFC peak in November 2008 -- where fundamentals plus forced deleveraging dwarfed any pure hedging flow.

**The post-event vanna grind-up.** After a feared macro event -- an FOMC decision, a CPI print, a megacap earnings report -- resolves without disaster, implied vol crushes and the vanna buy-back appears. The classic tell is a market that gaps up on the relief, then *keeps grinding higher* for a day or two on no fresh news, with shallow, bought dips. That continuation is the vanna and charm flows working through the dealer book as the elevated vol that was priced for the event drains away. The same machinery runs in reverse if the event *disappoints*: vol rises, vanna sells, and the dealer flow adds to the disappointment.

**The Volmageddon detonation (February 2018).** On **5 February 2018** the VIX leapt to a **37.32** close and the short-volatility exchange-traded products (the inverse-VIX notes) imploded after the close. The fuel was a population of short-vol and short-gamma positions that all had to cover at once: as vol spiked, their hedging demanded buying VIX futures and selling equity index futures, which spiked vol further -- a textbook short-gamma feedback loop, just expressed through volatility products instead of single-name options. It is the cleanest cautionary tale that *being on the wrong side of the aggregate gamma is a tail risk, not a daily annoyance.*

**The 0DTE intraday gamma cycle.** The rise of same-day-expiry options changed the *intraday* texture of the index tape. A 0DTE option has, near the money, enormous gamma -- its delta can swing from 0.2 to 0.8 over a fraction of a percent because it has hours, not months, to resolve. That means dealer hedging of the 0DTE book is hyper-sensitive to small moves *within* the session, and the gamma profile resets every morning. On a typical day, dealers often accumulate a long-gamma 0DTE position as the session matures (customers buy lottery-ticket 0DTE calls and puts that dealers hedge), which can produce an afternoon pinning effect toward the day's high-volume strikes -- a daily, miniature version of the monthly OPEX pin. But the same structure can flip: if the customer flow tilts toward selling premium and the dealer ends up short 0DTE gamma into a trending afternoon, the intraday move can accelerate sharply in the final hour. The [0DTE post](/blog/trading/options-volatility/0dte-and-the-rise-of-short-dated-options-the-new-market-structure) covers the market-structure consequences; the flow point here is that 0DTE compresses the entire gamma-flip-and-pin story into a single trading day, on repeat.

**The quiet grind that ends in a gap.** Perhaps the most instructive pattern is the *combination*: a long stretch of low-vol, positive-GEX grinding-up tape -- vanna and charm supplying a steady bid, dealers fading every dip -- that lulls everyone into complacency, followed by a sudden gap-down when a shock pushes spot through the flip. The grind *itself* builds the trap: as the market rises and vol falls, more protection gets bought and more short-gamma exposure accumulates below the market, so the eventual break has more fuel. The calm and the crash are not separate regimes; the calm *manufactures* the conditions for the crash. This is the single most important thing to internalize about dealer flow -- it does not just describe the weather, it loads the spring.

## The honest limits: correlation, causation, and what breaks the model

Before the playbook, a section that most flow commentary skips entirely, because it is the difference between a useful tool and a superstition. The dealer-flow models -- GEX, the gamma flip, charm and vanna estimates -- are *behavioral models of one population's forced hedging*. They are right about a real mechanism. They are wrong, or at least incomplete, in several specific ways that you must hold in mind every time you use them.

First, **correlation is not causation, and here it is genuinely mixed.** When the market pins to a big strike on OPEX and the GEX model "predicted" a long-gamma regime, did the dealer hedging *cause* the pin, or did the same calm conditions that made dealers long gamma also make the market quiet for unrelated reasons? Usually both. The hedging flow is a real causal force, but it operates *inside* a market that has its own reasons to move. On a quiet day the flow can be the dominant marginal driver; on a news day it is a footnote. Attributing every wiggle to dealer flow is the classic error -- the model becomes unfalsifiable, explaining a pin as long-gamma and a breakout as "the pin broke," with no prediction that could ever be wrong.

Second, **the positioning inputs are guesses.** As the GEX-construction example showed, the sign of each strike's contribution rests on an assumption about who is long and who is short -- and that assumption can be flat wrong, especially around big OTC trades, structured-product hedges, and end-of-quarter rebalancing that never appears in listed open interest. A GEX model that assumes dealers are short the puts at a strike where they are actually *long* them will get the regime exactly backwards at that level.

Third, **the model assumes dealers actually hedge mechanically and continuously.** Real desks hedge with discretion: they widen their hedging bands in fast markets to avoid getting run over, they net exposures across products, they sometimes choose to *carry* delta when they have a view on short-term flow, and they pre-position ahead of known events. The clean "re-hedge every \$1 move" assumption is a useful approximation, not a description of any real trader's blotter.

Fourth, **everyone now watches the same models.** When a flow signal becomes widely known, traders front-run it, which can either amplify it (everyone fades the extreme in a long-gamma tape, reinforcing the pin) or neutralize it (everyone expects the vanna grind-up, so they buy early and the flow has less to do). A crowded flow trade is a different animal from the pure mechanical flow. Treat published GEX levels as *consensus expectations*, not private edge.

The correct posture, then, is humility with utility: the flow tells you the *regime* and the *pressures* with decent reliability, and the *magnitudes* with poor reliability. Trade the regime; never bet the magnitude.

## The playbook: trading with the flow, not betting on it

The dealer-flow lens does not give you trades. It gives you *context* -- a read on whether to fade or respect a move, and how aggressively to size. Used that way it is one of the most useful free signals in the market. Used as a forecast, it will eventually run you over. Here is how to put it to work.

![Decision figure for reading the gamma regime, watching OPEX and vanna, with the flow-pressure caveat](/imgs/blogs/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot-7.png)

**Step 1 -- Read the regime.** Find where spot sits relative to the estimated gamma flip and note the sign of GEX. This is the single most important read, because it tells you which *game* you are playing for the session. You are not asking "where is price going" -- you are asking "if price moves, will dealers fade it or feed it?"

**Step 2 -- Set the bias to match the regime.**
- *Long-gamma / positive-GEX tape (above the flip):* expect mean reversion, pinning, and compressed ranges. **Fade the extremes.** This is the environment where selling premium -- [iron condors](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range), [covered calls](/blog/trading/options-volatility/covered-calls-and-the-wheel-selling-premium-on-stock-you-own) -- and buying dips toward the big strike are working *with* the dealer flow. Breakouts are suspect; they tend to get sold back into the range by the hedging.
- *Short-gamma / negative-GEX tape (below the flip):* expect momentum, gaps, and air pockets. **Respect the move; do not catch the knife.** Mean-reversion strategies are dangerous here because the dealer flow is *amplifying*, not damping. If you are long premium ([straddles](/blog/trading/options-volatility/straddles-strangles-and-the-long-volatility-bet), long gamma), this is your environment -- realized vol is being manufactured for you.

**Step 3 -- Watch the calendar.** Mark the monthly and quarterly **OPEX** dates -- the charm-driven drift concentrates there. Mark scheduled **events** (FOMC, CPI, big earnings) -- the post-event vol crush is when the **vanna** grind-up fires. And in index products, respect that **0DTE** flow makes intraday gamma swings sharper than the headline monthly positioning suggests.

**Step 4 -- Size to the flow, then forget it.** Lean a touch larger when the flow is a tailwind and you have an independent reason for the trade; widen your stops in a short-gamma tape because the air pockets are real; and *never* let the flow read be the entire thesis. The flow is the wind, not the destination. For the discipline of turning a read into a position size, the [position-sizing](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading) and [Kelly](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) frameworks apply directly.

**A concrete session read.** Suppose it is an OPEX Friday, the index is trading two percent above its estimated gamma flip, GEX is solidly positive, and there is no scheduled catalyst. The flow read is: *long-gamma, pinning regime.* Your bias is mean-reversion toward the big strike, ranges compressed, breakouts suspect. A reasonable posture is to fade a stretch to the upper end of the morning range back toward the high-open-interest strike, with a tight stop, sizing modestly because the *edge is the regime, not a hard signal.* You also note the charm drift -- if the dealer book is short downside puts, expect a mild bid into the close as those put deltas decay. Now contrast: the next Monday, a soft jobs number gaps the index down through the flip into negative GEX. The flow read flips to *short-gamma, momentum regime.* You tear up the fade plan entirely -- catching the dip now means standing in front of accelerating dealer selling. Either stand aside, or trade *with* the momentum with wide stops, or buy cheap downside gamma to be on the amplifying side. Same instrument, same week, two opposite playbooks, and the gamma regime told you which one was live. Nothing in that read predicted the jobs number; it told you how the market would *travel* once the number hit.

**The caveat, in bold, because it is the whole post:** dealer-flow estimates are a *pressure*, not a *certainty*. GEX is a model of positioning, not a forecast of price. Correlation between the flow read and the tape is real and exploitable on quiet days; causation is partial and breaks down exactly when news arrives. The trader who internalizes that the **461,627-share** re-hedge is identical whether the market dampens or amplifies -- that the *sign of the regime*, not the size of the flow, is the information -- is the trader who uses this lens correctly. Everyone else is mistaking the tail for the dog.

## Further reading and cross-links

Within the Options & Volatility series:
- [Gamma: the Greek that bites](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short) -- the curvature, convexity, and toxic-short profile that scales up into dealer-gamma regimes.
- [Rho, dividends, and the second-order Greeks: vanna, volga, charm](/blog/trading/options-volatility/rho-dividends-and-the-second-order-greeks-vanna-volga-charm) -- the formal definitions of charm and vanna used throughout this post.
- [How an options market maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) -- the desk view of being the forced counterparty.
- [Delta-hedging in practice](/blog/trading/options-volatility/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral) -- the cost and slippage of actually staying neutral.
- [0DTE and the rise of short-dated options](/blog/trading/options-volatility/0dte-and-the-rise-of-short-dated-options-the-new-market-structure) -- how same-day expiries sharpened the gamma and sped up the flow.
- [Assignment, pin risk, and expiration-day mechanics](/blog/trading/options-volatility/assignment-pin-risk-and-expiration-day-mechanics) -- the settlement side of the OPEX pin.

Going deeper on the theory and the institutions:
- [Market makers and high-frequency trading](/blog/trading/finance/market-makers-and-high-frequency-trading) -- who the dealers are and the infrastructure behind the quotes.
- [Black-Scholes](/blog/trading/quantitative-finance/black-scholes) -- the pricing model from which every gamma, charm, and vanna number in this post was computed.

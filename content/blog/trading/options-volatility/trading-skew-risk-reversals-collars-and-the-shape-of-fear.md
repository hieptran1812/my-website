---
title: "Trading Skew: Risk Reversals, Collars, and the Shape of Fear"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "How to monetize the volatility skew with risk reversals and collars: express a view that puts are too rich or too cheap, finance downside protection for free, and understand why being short skew is being short the crash."
tags: ["options", "volatility", "skew", "risk-reversal", "collar", "zero-cost-collar", "put-spread-collar", "tail-risk", "vanna", "hedging"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The skew is not just a shape on a chart; it is a price you can buy, sell, and finance. A risk reversal expresses a pure view on how steep the skew should be, and a collar uses the steep equity skew to pay for your downside protection — but selling the put wing is selling the crash, and the crash is exactly when it bids up.
>
> - **The risk reversal is the skew trade.** Long the out-of-the-money call, short the out-of-the-money put (or the reverse) isolates the *tilt* of the curve. Short skew (sell the rich put, buy the cheap call) collects a credit and wins if the skew flattens; long skew is the mirror.
> - **The collar finances protection with the skew.** Own the stock, buy a put, sell a call. On an equity index the call you sell is cheap and the put you buy is dear, so you give up upside to floor your downside — and you can pick the call strike so its premium exactly funds the put.
> - **Short skew is short the tail.** Selling the put wing earns a small, steady credit and then loses a fortune in a single gap-down, because skew *steepens* in a selloff: the puts you are short bid up just as the market falls.
> - **The number to remember:** a short-skew risk reversal that collects a \$0.11 credit at entry can lose about \$5.49 per share — roughly fifty times the credit — when the stock drops 8% and the put wing gaps up. The skew is rich because the risk is real.

In the late summer of one of those calm, grinding bull markets, two investors looked at the same steep equity skew and drew opposite conclusions. The first ran a large concentrated long position — a single name worth several million dollars that had run up hard and that they did not want to sell for tax reasons. They were nervous about a pullback. Their broker built them a **collar**: keep the stock, buy a downside put, and sell an upside call to pay for it. Because the equity skew makes downside puts expensive and upside calls cheap, the call premium they collected almost exactly covered the put they bought. They locked in a floor for, as the broker put it, "basically free." A few weeks later the name gapped down 18% on a bad print. The collar's put caught the fall; the position barely flinched. The skew had paid for their insurance.

The second investor looked at the same steep skew and saw a trade, not a hedge. The downside puts looked *too* rich — surely 22% implied vol on a 90-strike put when the stock had been realizing 12% was free money. So they sold the rich put and bought the cheap call: a **short-skew risk reversal**, for a small net credit. For three weeks it printed money as the market drifted up and time decayed the put. Then the same kind of bad print hit *their* name, the market sold off, and the puts they were short did not just rise with the spot — they *gapped*, because the skew steepened exactly when the downside materialized. The position that had collected an eleven-cent credit was now down more than five dollars a share. The skew had been rich for a reason, and the reason had just shown up.

This post is about the gap between those two outcomes. We met the skew as a *shape* in [the volatility smile and skew](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more) — why each strike trades its own implied vol and why equity-index puts are structurally dear. Here we do something different: we **monetize** it. We take the skew as given and ask the practitioner's question — *how do you express a view on it, and how do you use it?* The instruments are the risk reversal and the collar, and the recurring danger is that the cheap-looking side of every skew trade is cheap because the expensive side is where the crash lives.

![Profit and loss diagram for a bullish risk reversal at expiry, showing a short put loss region on the downside, a flat zone between strikes, and a long call gain region on the upside, with a breakeven near the put strike](/imgs/blogs/trading-skew-risk-reversals-collars-and-the-shape-of-fear-1.png)

The figure above is the shape we will be trading all post: a **bullish risk reversal** — short the rich out-of-the-money put, long the cheap out-of-the-money call. Notice three things. The payoff slopes up like a piece of stock (it is directionally long), it is *roughly free* to put on (a small \$0.11 credit at entry, because the skew funds the call), and it has a nasty downside: below the short put strike the line dives, because you are short the crash. That picture is the whole story — a cheap directional with a skew tilt, and a tail you cannot ignore. Let us build it from zero.

## Foundations: the skew is a price, not just a shape

Start with one paragraph of recap, because everything here rests on it. The **volatility skew** is the fact that, on equity indices, every strike trades at its own implied volatility, and the downside strikes trade *richer* than the upside. On the series' representative SPX 30-day skew, the 90% put implies 22% vol, the at-the-money sits at 17%, and the 110% call implies just 14.8%. That tilt is structural — it comes from relentless demand for downside protection, a steady supply of overwriting calls, and the market's memory that crashes are violent and gap-prone. The full *why* is in [the volatility smile and skew post](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more); we take it as a given. What that post established as a shape, this post treats as a tradable price.

Here is the reframe that unlocks the rest. When you say "the puts are rich," you are not making an aesthetic observation about a curve — you are saying a specific number, the gap between put vol and call vol, is **higher than it should be**, and that gap is something you can sell. When you say "the puts are cheap," you are saying that gap is too low, and you can buy it. The skew, in other words, has a price, and like any price it can be too high or too low relative to what you think is fair. Trading skew means taking a position on that gap — directly, with a structure that is long one wing and short the other.

### The benchmark: the 25-delta risk reversal

Practitioners compress the whole skew curve into one number so they can quote and trade it. That number is the **25-delta risk reversal**: the implied vol of the 25-delta put minus the implied vol of the 25-delta call. "25-delta" means the strike whose option has a delta of about 0.25 in absolute value — roughly a quarter of the way out of the money in probability terms (see [delta](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio) for what delta measures). Quoting by delta rather than by a fixed dollar strike is the convention because it keeps the comparison stable as spot drifts: the 25-delta strike is always "a quarter OTM," wherever spot happens to be.

On our representative skew, the 25-delta put sits near the 97 strike at about 18.2% implied vol, and the 25-delta call near the 103.5 strike at about 16.0%. The risk reversal — put vol minus call vol — is therefore about **2.2 vol points**. When a trader says "the SPX 25-delta risk reversal is bid 2.2 vols," that single number *is* the steepness of the skew: the market is charging 2.2 extra vol points to be long the downside relative to the upside. The risk reversal is to the skew what the VIX is to the level of vol — one number that captures the thing you actually trade.

### The two instruments

Everything in this post is one of two structures.

- **The risk reversal** is a two-leg options trade: long an OTM call and short an OTM put (a *bullish* risk reversal), or long an OTM put and short an OTM call (a *bearish* risk reversal). It has almost no premium cost — the call you buy and the put you sell roughly offset — so it behaves like a cheap synthetic position in the underlying, *tilted* by the skew. It is the purest way to bet on the skew itself.
- **The collar** is a hedging structure on a position you already own: long stock, long a protective OTM put, short an OTM call to finance the put. It is a risk reversal *bolted onto stock you hold*. Where the risk reversal is a speculative skew bet, the collar is the natural use of the skew by someone with a long position to protect.

The two are siblings. A collar's option legs (long put, short call) are exactly a *bearish* risk reversal — you buy the dear put and sell the cheap call. The difference is only that the collar owner also holds the stock, so the combined position is protected and capped rather than purely directional. Master one and you understand the other; we will treat them together throughout.

## The risk reversal: a cheap directional with a skew tilt

Take the bullish risk reversal from the cover figure and price it from the model, leg by leg, off each strike's own skew vol. The honest way to draw any of these payoffs is to compute them from Black-Scholes (the derivation lives in [the Black-Scholes post](/blog/trading/quantitative-finance/black-scholes); we do not repeat it), feeding each strike the implied vol the skew actually quotes there.

#### Worked example: pricing a 25-delta risk reversal

Stock at \$100, 30 days to expiry (T = 30/365 ≈ 0.082 years), risk-free rate 4%. The bullish risk reversal is: **buy the 25-delta call** (strike ≈ 103.5, vol ≈ 16.0%) and **sell the 25-delta put** (strike ≈ 97, vol ≈ 18.2%).

- **Buy the 103.5 call at 16.0% vol:** Black-Scholes value ≈ **\$0.69** per share — you pay this.
- **Sell the 97 put at 18.2% vol:** Black-Scholes value ≈ **\$0.80** per share — you collect this.
- **Net cash flow:** you collect \$0.80 and pay \$0.69, for a **net credit of about \$0.11** per share.

So before the stock moves at all, the skew has *paid you* eleven cents to take on a position that is directionally long. That is the whole appeal: the rich put you sold more than financed the cheap call you bought. Now imagine the skew didn't exist and both strikes traded at a flat 17%. The 97 put would be worth about \$0.69 and the 103.5 call about \$0.78, so the same structure would cost you a **\$0.09 debit** — you'd have to pay to be long. The skew swings the entry by about \$0.20 per share, from a \$0.09 cost to a \$0.11 credit. The intuition: a risk reversal is long one wing and short the other, so its price *is* the skew distilled to a single trade — the steeper the skew, the more the structure pays you to be long the downside-feared direction.

What did that \$0.11 credit buy you? Look again at the cover figure. Between the two strikes (97 to 103.5) the payoff is roughly flat at the small credit — you keep your eleven cents if the stock goes nowhere. Above 103.5 the long call kicks in and you make money one-for-one with the stock, like owning shares. Below 97 the short put kicks in and you *lose* money one-for-one, again like owning shares — but this is the leg that bites, because the downside on an equity index is where the violent moves happen. The breakeven is about \$96.93, just below the put strike. The risk reversal is, in effect, a synthetic long stock position you got into for free (or a small credit) — *because* you agreed to be short the feared tail.

### Why it is a "pure" skew bet

The reason traders call the risk reversal a *pure* skew trade is that its **vega** — its sensitivity to a parallel shift in overall implied vol — nearly cancels between the legs, while its sensitivity to the *tilt* of the curve does not. You are long vega on the call and short vega on the put; if all vols rise or fall together, the two roughly offset and you barely care. What you *do* care about is the *difference* between the put vol and the call vol — the risk reversal level itself. If the put wing cheapens relative to the call wing (the skew flattens), your short put loses value faster than your long call, and you profit. If the put wing richens relative to the call (the skew steepens), you lose. You have stripped out the level of vol and isolated the slope. That is what "trading skew" literally means: a bet on the slope, holding the level roughly fixed.

There is one more layer of purity available, and professional skew traders use it: **delta-hedging the risk reversal.** As built, the structure is directionally long — it has a positive delta, because the long call and short put both lean bullish. If your view is *only* about the skew and you have no opinion on direction, that delta is unwanted noise; a 3% rally could mask a skew loss, or a 3% drop could swamp a skew gain. So the trader sells a small amount of the underlying stock to zero out the net delta at entry, leaving a position whose P&L responds almost entirely to the *slope* of the curve and to realized volatility, not to which way the stock drifts. The catch is that delta-hedging is not free: it must be rebalanced as spot moves (because the risk reversal's delta itself changes), and each rebalance crosses the bid-ask and may lock in a small realized-vol cost. The fully delta-hedged risk reversal is the cleanest possible expression of "the skew is mispriced" — but it converts a low-maintenance directional-with-a-tilt into an active book that must be tended. Most retail expressions skip the hedge and simply accept the directional tilt; most professional skew desks run the hedge and trade the slope in isolation. Knowing which one you are doing is the difference between a skew bet and a disguised directional bet.

## Expressing a view: too steep, too flat, and which way to lean

Now the actual decision. You have a view on the skew — you think the risk reversal is mispriced — and you want to express it. The logic is symmetric and worth stating cleanly, because it is exactly backwards from how beginners often think about "expensive" options.

![Decision matrix for trading skew, with a too-steep branch leading to short skew by selling the put and buying the call, and a too-flat branch leading to long skew, each annotated with the winning scenario and the tail risk](/imgs/blogs/trading-skew-risk-reversals-collars-and-the-shape-of-fear-4.png)

The matrix above lays it out. Read the live 25-delta risk reversal, compare it to your estimate of fair, and the gap tells you the trade.

- **If you think the skew is too steep** — the puts are *too rich* relative to the calls, the risk reversal is wider than it should be — you want to **short skew**: sell the expensive 25-delta put and buy the cheap 25-delta call (the bullish risk reversal). You collect a credit, and you profit if the skew flattens back toward fair, or if the market grinds up calmly and the rich put you sold decays to nothing. The risk is the tail: a selloff bids the puts you are short.
- **If you think the skew is too flat** — the puts are *too cheap* relative to the calls, the risk reversal is narrower than it should be — you want to **go long skew**: buy the cheap 25-delta put and sell the call (the bearish risk reversal). You pay a debit, and you profit if the skew steepens, especially in a selloff that bids the put you own. The risk is the carry: a calm tape lets the rich put you own decay against you.

The deep point is that **the cheap-looking side is the bet, not free money.** A beginner sees a 90 put at 22% vol next to a stock realizing 12% and thinks "that's obviously overpriced, sell it." Sometimes it is overpriced — that is the variance risk premium, the structural tendency of implied vol to exceed realized, covered in [the variance risk premium post](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt). But the put is *also* compensating you for a real, fat, gap-prone left tail. Selling it because it "looks rich" without sizing for the tail is how the second investor in our hook got carried out. Trading skew well means having an actual view on whether the *current* steepness is rich or cheap relative to the realized skew — how fat the tail has actually been — not just noticing that downside vol is high.

### The risk reversal value as a function of steepness

It helps to see, in dollars, how the price of the skew bet moves as the skew itself steepens. The figure below holds the call wing fixed and walks the put wing up, plotting what it costs to put on a *long-skew* risk reversal (buy the put, sell the call) as the steepness — the put-over-call vol gap — widens.

![Line chart of the cost to put on a long-skew risk reversal rising as the skew steepness increases from zero to eight vol points, with today's 2.2 point skew and a panicked 6.2 point skew marked](/imgs/blogs/trading-skew-risk-reversals-collars-and-the-shape-of-fear-5.png)

At zero steepness — a flat skew, both wings at 16% — the long-skew risk reversal is roughly free or even a small credit (the put and call nearly offset). At today's 2.2-vol-point skew it costs you about \$0.11 to be long the skew (buy the put, sell the call). At a panicked 6.2-vol-point skew — the kind you see when fear has gripped the tape — it costs about \$0.49. The relationship is nearly linear: each extra vol point of steepness adds roughly nine or ten cents to the price of being long the skew. That is the "exchange rate" between skew steepness and dollars, and it is exactly what you are trading. If you short skew today at 2.2 points and it flattens to 1.2, you make about a dime; if it steepens to 6.2, you lose about thirty-eight cents — *plus* whatever the directional move does to you, which on the downside is the dangerous part.

#### Worked example: the skew P&L of a short-skew risk reversal

You put on the bullish (short-skew) risk reversal from before — short the 97 put at 18.2%, long the 103.5 call at 16.0%, for a \$0.11 credit. The 2.2-point skew is your entry. Now suppose, with spot unchanged at \$100, the **skew flattens** to 1.2 points: the put vol drops to 17.2% while the call vol holds at 16.0%.

- **The 97 put you are short** reprices from \$0.80 (at 18.2%) to about \$0.76 (at 17.2%) — a drop of about \$0.04. Since you are short it, that \$0.04 is profit.
- **The 103.5 call you are long** is unchanged at \$0.69 (its vol didn't move).
- **Net skew P&L ≈ +\$0.04** per share, on top of the \$0.11 credit you already banked.

A one-vol-point flattening, with the stock pinned, hands you about four cents per share — modest, but it is the *pure* skew profit, isolated from any directional move. The intuition: shorting skew is a small, grinding bet that the put wing will cheapen relative to the call wing; you win a little when fear fades, and the position bleeds in your favor through decay as long as the tape stays calm. The danger is entirely on the other side of that calm.

## The collar: financing protection with the skew

Now turn the same machinery toward hedging. You own a stock and you are nervous. The textbook hedge is a **protective put** — buy a put, floor your downside, keep your upside. But on an equity index (and most single stocks) the put you want is sitting on the steep, rich part of the skew, so naked protection is expensive. The **collar** solves this by selling the cheap upside to pay for the dear downside.

![Profit and loss diagram comparing naked long stock against a collared position, where the collar floors losses below the put strike and caps gains above the call strike, with the protection band between the strikes shaded](/imgs/blogs/trading-skew-risk-reversals-collars-and-the-shape-of-fear-2.png)

The figure contrasts naked long stock (the dashed line, a straight 45-degree payoff with unlimited upside and ruinous downside) against the collar (the solid line). The collar flattens both tails: below the put strike (95) your losses are floored, and above the call strike (105) your gains are capped. In between — the *protection band* from 95 to 105 — you participate in the stock one-for-one, minus a tiny premium drag. You have traded away the extremes of the long-stock payoff for a defined band. You give up the upside tail (which you may not have wanted anyway) to buy off the downside tail (which you were losing sleep over).

#### Worked example: a skew-financed collar

Own 100 shares bought at \$100. Buy the 95 put for protection, sell the 105 call to finance it. 30 days, r = 4%, each leg priced off its own skew vol.

- **Buy the 95 put at 19.0% vol:** ≈ **\$0.44** per share — you pay this.
- **Sell the 105 call at 15.6% vol:** ≈ **\$0.37** per share — you collect this.
- **Net cost ≈ \$0.07** per share (a small debit). The cheap call you sold financed most of the dear put you bought.

Your position is now collared: below \$95 you are floored (the put pays off dollar-for-dollar with the stock's decline), and above \$105 you are capped (the short call gives up further gains). For seven cents a share you have bounded a \$100 position to roughly the \$95–\$105 range for the month. Now contrast the flat-vol world: if both strikes traded at 17%, the 95 put would cost \$0.31 and the 105 call would fetch \$0.47, so the collar would be a **\$0.16 credit** — you'd be *paid* to put it on. The skew costs this collar about \$0.23 per share (the swing from a \$0.16 credit to a \$0.07 debit) because it makes the put you buy dearer and the call you sell cheaper. The intuition: the skew is a headwind for the *symmetric* collar — but, as we will see, it is precisely what lets you build a *zero-cost* collar by choosing the strikes asymmetrically.

### The zero-cost collar: choosing strikes so the call funds the put

The collar's marketing pitch — "free protection" — comes from the **zero-cost collar**: pick the strikes so the call premium *exactly* equals the put premium, and your net cash outlay is zero. You pay nothing up front; you simply agree to give up upside above the call strike in exchange for protection below the put strike. The skew is what makes this possible on an equity index, because the rich put can be funded by a call that is not very far out of the money.

![Line chart showing call premium collected rising as the short call strike moves closer to spot, crossing the fixed 95 put premium at the zero-cost call strike near 104.5, with the credit region and debit region shaded](/imgs/blogs/trading-skew-risk-reversals-collars-and-the-shape-of-fear-3.png)

The construction is mechanical, and the figure shows it. Hold the put fixed — say the 95 put, which costs \$0.44. Then sweep the short-call strike. A closer call (say 102) collects more premium; a farther call (say 108) collects less. Plot the call premium against the strike and find where the curve crosses the put's \$0.44 line. To the left of that crossing (closer calls) you collect more than the put costs — the collar earns a *credit*, but you cap your upside tightly. To the right (farther calls) you collect less — the collar costs a *debit*, but you keep more upside. The crossing point is the zero-cost pair.

#### Worked example: finding the zero-cost call strike

You hold 100 shares at \$100 and buy the 95 put at 19.0% vol for **\$0.44**. You want to sell a call whose premium is exactly \$0.44, so the collar costs nothing. Sweep the call strike, pricing each off the skew:

- **Sell the 103.5 call (vol ≈ 16.0%):** ≈ \$0.69 collected — more than \$0.44, so this collar earns a \$0.25 credit but caps you at 103.5.
- **Sell the 105 call (vol ≈ 15.6%):** ≈ \$0.37 collected — less than \$0.44, so this collar costs a \$0.07 debit.
- **Sell the ~104.5 call (vol ≈ 15.7%):** ≈ **\$0.46** collected — almost exactly the \$0.44 put cost. **This is the zero-cost pair** (a hair of credit, essentially free).

So the zero-cost collar on this name is roughly **long the 95 put, short the 104.5 call**, for net zero. You have floored your downside at \$95 and capped your upside at \$104.5, paying nothing. Notice what the skew did *for* you here: because the put was so rich, you could fund it by selling a call only 4.5% out of the money rather than, say, 8% out. The give-up is real — you surrender everything above \$104.5 — but the protection genuinely costs nothing in cash. The takeaway: a zero-cost collar is never truly free; you pay in *surrendered upside*, and the steeper the skew, the *closer* the call you must sell, so the more upside the skew quietly makes you give up.

That last point is the one most "free collar" pitches gloss over. On a flat-vol stock, funding a 95 put would let you sell a call much farther out (the put is cheaper and the call dearer), so you'd keep more upside. The steep equity skew makes the put dear and the call cheap, which means the call has to come *in closer* to raise the same premium — so the equity skew, while it makes the collar feasible for "free," also forces you to give up more upside than a naive flat-vol intuition would suggest. The collar is free in cash precisely because it is expensive in surrendered upside, and the skew sets that exchange rate.

### The put-spread collar: cheapening the give-up

There is a refinement worth knowing, because it directly trades the skew against itself. In a plain collar you buy a put outright. But the put you buy sits on the rich part of the skew — so why not sell an even-richer, lower put against it, turning your long put into a **put spread**, and use the credit to need less from the call (so you can sell a farther call and keep more upside)? That structure — long stock, long put, short a lower put, short a call — is the **put-spread collar**.

#### Worked example: a put-spread collar's cost

Own 100 shares at \$100. Build a put-spread collar: buy the 95 put, sell the 90 put (turning the protection into a 95/90 put spread), and sell the 105 call. 30 days, r = 4%, skew vols throughout.

- **Buy the 95 put at 19.0%:** ≈ \$0.44 paid.
- **Sell the 90 put at 22.0%:** ≈ \$0.10 collected (this leg is on the steepest, richest part of the skew — you collect extra because it is so dear).
- **Net put spread cost:** \$0.44 − \$0.10 ≈ **\$0.34** for protection between 95 and 90.
- **Sell the 105 call at 15.6%:** ≈ \$0.37 collected.
- **Net of all four legs:** \$0.44 − \$0.10 − \$0.37 ≈ **−\$0.03**, i.e. a small \$0.03 *credit*.

So the put-spread collar is essentially free *and* lets you keep the upside out to 105 (rather than pulling the call in to 104.5 as the plain zero-cost collar required). The catch is that your downside protection is no longer a hard floor — below \$90 the short put kicks back in and you start losing again. You have a *protection band* from 95 down to 90, then you are unprotected below 90 (you keep the stock's losses past that point, offset only by the \$5 the spread paid). The takeaway: the put-spread collar uses the steepest part of the skew (the deep put you sell) to cheapen your protection, but it converts a floor into a *trapdoor* — you are protected in a normal pullback and exposed again in a true crash, which is the one scenario you most wanted the hedge for. It is a skew trade dressed as a hedge, and you must size it knowing the deep tail is open.

## Skew dynamics: what you are really trading against

A static snapshot of the skew is not the whole game. The skew *moves* — it steepens and flattens — and those dynamics are what a skew trader is positioned for or against. Three of them matter.

### Skew steepens in selloffs (and your delta drifts)

The single most important dynamic: **the skew steepens when the market falls.** As spot drops, demand for downside protection surges (everyone wants puts at once), while the call overwriters keep supplying calls, so the put wing rises faster than the call wing. The 25-delta risk reversal blows out — from 2.2 points to 4, 5, 6 or more — in a real selloff. This is the mechanism that crushed the second investor in our hook: they were short the put wing, and the put wing is exactly what bids up.

This steepening also changes your *delta* as you hold the position, through the second-order Greek **vanna** — the sensitivity of delta to a change in implied vol, covered in [the second-order Greeks post](/blog/trading/options-volatility/rho-dividends-and-the-second-order-greeks-vanna-volga-charm). Here is why it matters for a risk reversal. You are short a put and long a call. As the market falls and the put's vol rises (skew steepens), vanna makes the put's delta grow faster than spot alone would imply — the short put gets *more* short-stock-exposed than you expected. Your position's delta drifts against you precisely when the market is moving against you. You set out with a clean "synthetic long stock for a credit" and discover, in the selloff, that the rising put vol is making you effectively *longer* (more exposed to the drop) than your model said at entry. Vanna is the formal name for "the skew moved, so my delta moved." A skew trader who ignores vanna is repeatedly surprised by how much their hedge ratio drifts in stress.

### The skew "carry": rolling down a steep skew

The flip side of the danger is the **carry**. If the skew is steep and *stable* — it does not steepen further — then a short-skew position earns money simply by the passage of time, because the rich put you are short decays toward its (low) intrinsic value while the stock sits still. This is sometimes called "rolling down the skew": as time passes with spot unchanged, the put you sold is worth less and less, and you pocket the difference. The variance risk premium ([covered here](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt)) is the engine — implied vol on the put wing tends to print above the realized vol that follows, so the seller of that wing earns a structural premium *on average*. The skew carry is just the variance risk premium concentrated on the steepest, richest part of the curve. It is real, it is positive on average, and it is the reason short-skew trades feel like free money for long stretches.

To make the carry concrete in dollars: take the 97 put you are short at 18.2% vol, worth \$0.80 with 30 days to run and spot pinned at \$100. Hold spot exactly still and let two weeks pass, so the put now has 16 days to expiry at the same 18.2% vol. Black-Scholes now values it at roughly \$0.42 — it has shed about \$0.39 of time value while the stock did nothing. Since you are short it, that \$0.39 per share is realized carry, just from the clock. Over a full month, an undisturbed 97 put bleeds essentially all of its \$0.80 to you. That is the seductive part of the trade: a steep, stable skew quietly pays the short-skew position a few cents a day, every day the market refuses to crash. The trap is that the entire month of carry — and several months more — can be wiped out in the single session when the put gaps from \$0.80 to \$5.61, as the earlier tail example showed. The carry is the bait; the gap is the hook.

### When is the skew too steep or too flat to fade?

The hard question. The skew is rich on average, so a naive "always short skew" rule would collect the carry — until it didn't, and the single bad day erased years of it. Skill is in judging when the *current* steepness is extreme enough to fade and when it is too dangerous. A few honest heuristics:

- **Fade a steep skew (short it) only when realized skew has been mild.** If the stock has been grinding up with shallow, slow pullbacks, the fat-left-tail premium is probably overpriced, and selling the put wing into that calm is the variance-risk-premium trade working for you. Size it as if the tail *will* eventually hit.
- **Do not fade a steep skew into a fragile tape.** If the market is already nervous — credit spreads widening, breadth deteriorating, a known catalyst looming — a steep skew may be *correctly* pricing an imminent gap. Selling the put wing there is selling insurance the day before the storm.
- **Go long skew (buy the put wing) when the skew is unusually flat and the tape is complacent.** A 25-delta risk reversal that has compressed to near zero in a quiet bull market is the market pricing almost no crash risk — historically a good moment to *own* cheap downside, because the skew tends to re-steepen violently when complacency breaks.

There is no formula here, only a discipline: the skew is a *risk premium*, not a free lunch, so you fade it when you are being overpaid for a risk that is currently dormant, and you buy it when you are being underpaid for a risk the market has forgotten.

## The recurring tail: short skew is short the crash

Everything in this post comes back to one risk, and it deserves its own section because it is the way skew traders blow up. **Selling the put wing — whether as a short-skew risk reversal, a naked put, or the short put in a put-spread collar — is being short the crash.** The premium you collect is compensation for a risk that is real, fat, and correlated: when the market gaps down, the puts you are short do not rise smoothly, they *gap up*, because the skew steepens at the same time. You lose on the spot move and on the vol move simultaneously, and both are largest in the same disorderly session.

![Line chart of the mark-to-market profit and loss of a short-skew risk reversal as the stock falls, starting at a small credit near spot and turning into a large loss as the stock drops eight percent and the put wing is bid up](/imgs/blogs/trading-skew-risk-reversals-collars-and-the-shape-of-fear-6.png)

The figure traces it. You put on the short-skew risk reversal (short 97 put, long 103.5 call) for a \$0.11 credit at spot \$100. As long as the stock holds or rises, you keep the credit and the long call even adds upside. But walk the stock *down* and let the skew steepen as it falls — the put vol bid up about one vol point for every 1% the stock drops — and the line plunges. By the time the stock is down 8% (to \$92), the position is showing a loss of about \$5.49 per share. The small green credit at the top right and the deep red loss at the bottom left are the same trade.

#### Worked example: the short-skew risk reversal that loses in a selloff

You are short the 97 put at 18.2% and long the 103.5 call at 16.0%, banked at a \$0.11 credit. Spot \$100, 30 days. Now the market sells off: spot falls 8% to \$92 over a few sessions (call it 25 days left), and the skew **steepens** — the 97 put's vol gaps up about 8 points to 26.2%, while the call's vol barely moves to 18.0%.

- **The 97 put you are short** was worth \$0.80; at \$92 spot, 25 days, and 26.2% vol it is now worth about **\$5.61**. You are short it, so this leg loses you \$5.61 − \$0.80 ≈ **−\$4.81** per share.
- **The 103.5 call you are long** was worth \$0.69; with spot down at \$92 it is now nearly worthless, about \$0.01. You are long it, so this leg loses you 0.69 − 0.01 ≈ **−\$0.68** per share.
- **Total position P&L ≈ −\$5.49** per share — about **\$549 per contract** (100 shares).

You collected an eleven-cent credit and lost five and a half dollars — a loss roughly *fifty times* the credit, from a move of just 8%. Most of the damage is the short put, and most of *that* is the vol gap, not the spot move: had the put's vol stayed put, the loss would have been far smaller. The intuition: the skew is rich because the put wing bids up exactly when the market falls, so shorting it means you lose on direction and on vol at once — the textbook short-the-tail blowup, and the reason a short-skew book must be sized for the gap, not the average day. This is the same lesson the 1987 crash taught the entire options market, the story we tell in full in [the 1987 crash case study](/blog/trading/options-volatility/case-study-the-1987-crash-and-the-birth-of-the-skew).

## Collar versus naked stock versus protective put

Before the misconceptions, one comparison that ties the hedging structures together, because the choice between them is the practical decision a long holder actually faces.

![Comparison table of naked long stock, protective put, and collar across up-front cost, downside protection, upside kept, how the skew helps, and when to use each](/imgs/blogs/trading-skew-risk-reversals-collars-and-the-shape-of-fear-7.png)

The three ways to hold a long position differ on exactly one axis: **what you pay, and what upside you surrender, for downside protection.**

- **Naked long stock** costs nothing extra, keeps unlimited upside, and has zero protection — full loss to zero. Use it when you are fully bullish and can ride drawdowns.
- **The protective put** keeps unlimited upside (minus the premium) and floors the downside, but you pay the full put premium up front — about \$0.44 for the 95 put — and the skew *hurts* you here, because the put you must buy sits on the rich wing. Use it when you are bullish but scared and willing to pay cash for the protection while keeping your upside.
- **The collar** floors the downside at the same level for roughly zero cash, because the call you sell funds the put — the skew *helps* you here — but you give up the upside above the call strike. Use it when you are neutral-to-defensive and upside is not the point; you mainly want to bound the position cheaply.

The collar and the protective put share the same floor; they differ entirely in whether you pay in *cash* (protective put) or in *surrendered upside* (collar), and the skew is what tilts that choice. On a steep equity skew, the collar's "free" framing is genuine in cash terms but expensive in upside terms, while the protective put's "keep your upside" framing is genuine but expensive in cash terms. There is no free protection; there is only which currency you pay in.

## Common misconceptions

**"A zero-cost collar is free protection."** No — it is free *in cash*, paid for in *surrendered upside*. The numbers make it concrete: to fund a \$0.44 put on our skew, you must sell a call around the 104.5 strike, giving up everything above \$104.5 for the month. If the stock rips to \$115, your collared position stops at \$104.5 while a naked holder keeps the whole \$15 — a \$10.50 per share opportunity cost. The collar is "free" the way a fixed-rate mortgage is "free" of rate risk: you paid for it, just not in the currency you were watching. The steeper the skew, the closer the call you must sell, so the *more* upside the "free" collar quietly costs you.

**"Sell the rich puts — they're obviously overpriced."** No — they are richly priced for a reason, and the reason is the tail. We priced it: a short-skew risk reversal that collects \$0.11 loses about \$5.49 when the stock drops 8% and the skew steepens — fifty times the credit. The put vol is high because the left tail is fat and gap-prone, and the variance risk premium you earn on average is precisely the fee for warehousing that tail. Selling the put wing can be a good trade, but only when you have a view that the *current* steepness exceeds the realized risk, and only sized so a single gap cannot ruin you. "It looks overpriced" is not that view.

**"A risk reversal is a directional trade, so the skew doesn't really matter."** No — the skew is what sets the *entry price*, and the entry price decides whether the trade is a credit or a debit. The same bullish risk reversal that collects \$0.11 under our 2.2-point skew would *cost* \$0.09 under a flat skew — a \$0.20 swing per share before the stock moves a penny. On a flat-skew or forward-skew asset (an FX pair, an energy contract) the sign can flip entirely, and a structure that pays you to be long on equities might charge you on oil. You are always trading the skew when you trade a risk reversal, whether you priced it that way or not.

**"The collar's protection is a hard floor, so I'm safe below the put strike."** Usually yes for a plain collar — but **not** for a put-spread collar, which converts the floor into a trapdoor. We built one: long the 95 put, short the 90 put, short the 105 call, for a \$0.03 credit. It protects you from \$95 down to \$90 — but below \$90 the short put kicks back in and you are exposed to the stock's losses again, offset only by the \$5 the spread paid. In a true crash — the exact scenario you bought protection for — the put-spread collar leaves the deep tail open. The cheaper "protection" bought by selling the deep put is paid for by giving back the crash protection itself.

**"Skew steepening just means my puts go up — good if I own them."** Partly, but it also drifts your *delta* through vanna, which can wrong-foot you. If you are short the put wing, a steepening skew makes the short put's delta grow faster than the spot move alone — your position gets effectively longer (more exposed to the drop) exactly as the market falls, so you lose more than a static delta would predict. And if you are *long* the put wing as a hedge, the steepening helps your P&L but also means your hedge's delta is moving, so the amount of stock you'd need to sell to stay neutral is itself shifting under you. Skew dynamics are not just a vol-level story; they reshape your directional exposure in real time.

## How it shows up in real markets

**The "free" collar that saved a concentrated holder.** The first investor in our hook is a real, common situation: an executive or early investor sitting on a large, low-basis position they cannot or will not sell. Selling triggers a tax bill; staying naked-long courts a single-name crash. The zero-cost collar is the standard institutional answer — lock a band around the position for no cash outlay, ride out the holding period, and let the put catch any gap-down. When the name does gap (an earnings miss, a regulatory headline, a sector rerating), the collar's put converts the loss into a bounded one. The skew is what makes it nearly free: on a name with a steep put skew, the rich downside put can be funded by a call only modestly out of the money. The trade-off — surrendered upside — is exactly the upside a nervous concentrated holder is happy to give up.

**Volmageddon and the short-vol unwind.** In early February 2018, a crowd of traders and products were effectively short the vol and the put wing — short variance, short VIX futures, short the skew via various structures. The S&P fell a few percent and the vol complex *gapped*: the VIX more than doubled in a day, and the put wing steepened violently. Short-vol products were liquidated, some to near zero. The mechanism was exactly the short-skew tail in this post, at scale: a small, steady premium collected for months, then a single session where the thing you were short bid up faster than you could cover. Anyone running a short-skew risk reversal or a naked short-put book learned the fifty-to-one math the hard way.

**The August 2024 yen-carry unwind.** A more recent rerun: a quiet, grinding tape lulled sellers into shorting vol and the put wing, and then a rapid unwind of the yen carry trade gapped equities down and spiked the VIX to the high 30s intraday. The skew steepened sharply; the puts that had been "obviously overpriced" for months suddenly were not. The lesson recurs every cycle because the structure recurs every cycle: the variance risk premium pays until it doesn't, and "doesn't" is always a gap, never a grind.

**Why dealers steepen the skew in a falling market.** The steepening is not arbitrary — it is partly the visible trace of dealer hedging. Dealers who are net short OTM puts are short gamma down there; as spot falls toward those strikes, their short puts gain delta fast, forcing them to sell the underlying into the drop to stay hedged. That selling pushes spot lower, which steepens the put wing further, which forces more selling. At the same time the public's demand for fresh downside protection surges. Both forces hit the put wing at once, so the skew gaps steeper precisely in the disorderly, gap-down sessions it was pricing against. If you are short the skew, you are on the wrong side of that feedback loop.

**The persistent, modest payoff of being short skew.** Studies of long-run index-option returns find that systematically *selling* the rich put wing has earned a positive premium over decades — the variance risk premium is real and the skew carry is positive on average — but the return stream is punctuated by occasional catastrophic drawdowns. The strategy looks like picking up coins with rare, violent interruptions. That profile is *fine* if you size for the interruption (define risk, never bet the book on one gap) and *fatal* if you mistake the long calm stretches for a riskless edge. The skew is the market's standing price for a real, rare, severe risk, and the people who survive shorting it are the ones who never forget the "rare, severe" part.

## The playbook: how to trade skew

Pull it together into a practitioner checklist. Trading skew is not an everyday speculative trade for most people; it is a lens and a small set of structures you deploy when the skew is mispriced relative to your view, or when you have a position to hedge.

**1. Quote the skew in one number, and have a fair estimate.** Track the 25-delta risk reversal (put vol minus call vol) for the names you trade. The *level* tells you how steep the skew is right now; you need a *fair* estimate — what the steepness should be given how the stock has actually been realizing its tail — to know whether to fade it or buy it. No fair estimate, no trade; you are just guessing the cheap side is free.

**2. To short skew, sell the put / buy the call — only when overpaid, and sized for the gap.** If your view is that the skew is too steep (the put wing is richer than the realized tail warrants), put on the bullish risk reversal: short the 25-delta put, long the 25-delta call, for a credit. The Greek profile is roughly long delta (synthetic long stock), short put-wing vega, and *short the tail*. Entry: when the risk reversal is wide and the tape is calm. Sizing: as if a 1987/2018/2024-style gap *will* happen — never short more put-wing exposure than you can survive a multi-sigma down-gap on, and prefer defined-risk versions (sell a put *spread*, not a naked put). Invalidation: a fragile tape with a known catalyst, where a steep skew may be correctly pricing an imminent gap — stand aside.

**3. To go long skew, buy the put / sell the call — when the skew is too flat.** If the skew has compressed in a complacent bull market, own the cheap downside: the bearish risk reversal, for a debit. The Greek profile is short delta, long put-wing vega, and *long the tail* — it pays off if the market falls and the skew re-steepens. Entry: when the risk reversal is unusually narrow and complacency is high. The cost is carry: the rich put you own decays if the calm persists, so size it as a cheap, time-limited tail hedge, not a position you marry.

**4. Use the collar to finance protection on a position you actually hold.** If you own stock you want to keep but protect, build a collar: long an OTM put, short an OTM call to fund it. On a steep equity skew, target the *zero-cost* pair — find the call strike whose premium equals the put's (around 104.5 against a 95 put on our skew). Know that the steeper the skew, the closer the call you must sell, so the more upside you surrender. Use a collar when you are net long, want a hard floor, and are willing to cap the upside; use a plain protective put instead when keeping the upside matters more than the cash cost.

**5. Reach for the put-spread collar only with eyes open about the trapdoor.** If you want to keep more upside (sell a farther call) or pay less, you can cheapen the put leg by selling a deeper put against it — but that reopens the deep tail below the lower strike. It is a fine structure for hedging a *normal* pullback cheaply; it is the wrong structure if the scenario you fear is a true crash, because that is exactly where the protection runs out. Match the structure to the tail you are actually trying to cover.

**6. Respect the dynamics: skew steepens in selloffs, and vanna drifts your delta.** Whatever skew position you hold, model what happens when the market falls 5–10% *and* the skew steepens 3–6 points at the same time, because those two move together. Short-skew positions lose on direction and on vol simultaneously; long-skew positions win on both but see their delta drift through vanna. The formal tools — vanna, volga, and how your Greeks morph as spot and the skew move — are in [the second-order Greeks post](/blog/trading/options-volatility/rho-dividends-and-the-second-order-greeks-vanna-volga-charm). Hedging a portfolio with these structures at scale — protective puts, collars, and tail overlays — is its own discipline, taken up in [hedging a portfolio with options](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk).

The throughline of the series holds here as sharply as anywhere: an option is a bet on volatility, and the skew is the price of the *kind* of volatility the market fears most. Trading it means selling that fear when you are overpaid for it and buying it when you are underpaid — and never, ever forgetting that the cheap-looking side of a skew trade is cheap because the expensive side is where the crash lives. The first investor in our hook used the skew to make their protection free. The second sold the skew and met the tail. The difference was not cleverness; it was respect for the shape of fear.

## Further reading & cross-links

- [The volatility smile and skew: why OTM puts cost more](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more) — where the skew comes from and why equity-index puts are structurally rich. This post monetizes what that one explains.
- [Rho, dividends, and the second-order Greeks: vanna, volga, charm](/blog/trading/options-volatility/rho-dividends-and-the-second-order-greeks-vanna-volga-charm) — vanna is exactly how your delta drifts as the skew moves; the formal tool for skew dynamics.
- [Vertical spreads: debit and credit, defining your risk](/blog/trading/options-volatility/vertical-spreads-debit-and-credit-defining-your-risk) — the defined-risk building blocks behind the put-spread collar.
- [The variance risk premium: why selling vol pays until it doesn't](/blog/trading/options-volatility/the-variance-risk-premium-why-selling-vol-pays-until-it-doesnt) — the engine behind the skew carry, and the structural reason short-skew trades make money on average.
- [Reading the vol surface like a trader: the 3D map of fear](/blog/trading/options-volatility/reading-the-vol-surface-like-a-trader-the-3d-map-of-fear) — the skew across every expiry, the full surface the risk reversal lives on.
- [Hedging a portfolio with options: protective puts, collars, and tail risk](/blog/trading/options-volatility/hedging-a-portfolio-with-options-protective-puts-collars-and-tail-risk) — collars and tail overlays applied to a whole book.
- [Case study: the 1987 crash and the birth of the skew](/blog/trading/options-volatility/case-study-the-1987-crash-and-the-birth-of-the-skew) — the day that created the skew, and the original short-the-tail blowup.
- [The volatility surface](/blog/trading/quantitative-finance/volatility-surface) — the surface as a no-arbitrage object, the theory behind the curve.
- [Options theory](/blog/trading/quantitative-finance/options-theory) — the pricing fundamentals this series builds the practice on.

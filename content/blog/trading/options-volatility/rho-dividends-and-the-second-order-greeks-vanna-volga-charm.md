---
title: "Rho, Dividends, and the Second-Order Greeks: Vanna, Volga, Charm"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Go beyond delta, gamma, theta, and vega: learn rho and dividends, then meet the second-order Greeks — vanna, volga, and charm — that tell dealers how their hedges drift when vol, rates, and time move."
tags: ["options", "volatility", "rho", "dividends", "vanna", "volga", "charm", "second-order-greeks", "options-greeks", "dealer-hedging"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — Once you know delta, gamma, theta, and vega, two more first-order sensitivities (rho and the dividend effect) and three second-order Greeks (vanna, volga, charm) explain almost everything else your option does — and they are exactly what dealer desks hedge when the simple Greeks lie still.
>
> - **Rho is your sensitivity to the risk-free rate.** It is tiny for short-dated options but real for LEAPS: a one-year at-the-money call gains about **+\$0.52 per share for every 1% the rate rises**, while the matching put loses about **−\$0.44**. Higher rates lift calls and depress puts because of carry.
> - **Dividends pull the other way:** they push call values down and put values up, and they create the one situation where exercising an American *call* early is rational — deep in the money, just before a fat ex-dividend, when the dividend captured beats the time value surrendered.
> - **The second-order Greeks are derivatives of the first-order Greeks.** **Vanna** = how your delta drifts when implied vol moves (∂delta/∂sigma). **Charm** = how your delta bleeds as time passes with spot flat (∂delta/∂time). **Volga** = the convexity of vega (∂vega/∂sigma) — why far-out-of-the-money options gain vega as vol rises.
> - **The one rule to remember:** delta, gamma, theta, and vega tell you where you are; rho, vanna, charm, and volga tell you *how that map is being redrawn underneath you* as rates, vol, and the calendar move. Dealers hedge the second set; that hedging is what powers the "vanna-charm rally" into monthly expiration.

## A LEAPS holder who got an interest-rate surprise

In early 2022 a long-term investor — call him Marcus — held a stack of LEAPS: one-year and two-year call options on a basket of large-cap stocks. He liked the idea. Instead of buying \$50,000 of stock, he had paid a fraction of that for calls that gave him most of the upside with a defined maximum loss. He had read the standard Greek primers. He knew his delta (how much he made per dollar the stocks moved), his theta (the daily time-decay rent), and his vega (his exposure to implied volatility). He felt covered.

Then the Federal Reserve started raising rates — fast. The federal funds rate went from roughly **0.25%** at the start of 2022 to **over 4.5%** by year-end, the most violent tightening cycle in four decades. Marcus watched his stocks chop sideways for a stretch in the summer, and noticed something he could not explain with the Greeks he knew: on a couple of days when the underlying stocks barely moved and implied vol was flat, his **call values crept up anyway**, and a put hedge he held on the same names quietly lost value. Nothing in delta, gamma, theta, or vega accounted for it. The move was small per day, but over the year the rate component alone added meaningful value to his long-dated calls.

What Marcus had run into was **rho** — the sensitivity of an option's price to the risk-free interest rate. For the 30-day options most retail traders live in, rho is so small it is rounding error, and most primers wave it away in a sentence. But Marcus was not holding 30-day options. He was holding one- and two-year LEAPS, and rho grows with the time to expiry. In a regime where rates moved several whole percentage points in a year, the Greek everyone ignores became a Greek that mattered.

![The Greek family tree mapping first-order Greeks to second-order Greeks](/imgs/blogs/rho-dividends-and-the-second-order-greeks-vanna-volga-charm-1.png)

This post is about the Greeks that live beyond the famous four. We will start with **rho** and with **dividends** — both first-order effects (direct slopes of the price) that the introductory posts skipped — and then climb into the **second-order Greeks**: vanna, charm, and volga. These are the derivatives of the derivatives. They are quiet on a calm day and dominant on a stressed one, and they are precisely what a market-maker's risk system watches when the simple Greeks look fine but the desk is still bleeding or printing money. By the end you should be able to read the whole family tree in the figure above and know which branch is moving your P&L on any given day.

If you have not yet internalized the big four, the rest of this series builds them from zero: [delta](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio) is your directional exposure, [time value and theta](/blog/trading/options-volatility/time-value-and-theta-why-an-option-is-a-melting-ice-cube) is the melting-ice-cube of time decay, and [vega](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) is your exposure to implied volatility. This post assumes you know those and pushes past them.

## Foundations: what a Greek actually is

A Greek is a **partial derivative** of the option's price with respect to one of its inputs, holding everything else fixed. That sounds like calculus, but the plain-language version is simple: a Greek answers the question *"if this one thing moves a little, how much does my option's price move?"* The word "partial" matters — it means we change exactly one input at a time and freeze the rest.

An option's price is a function of five inputs. The dedicated post on [what sets an option's price](/blog/trading/options-volatility/what-sets-an-options-price-the-five-inputs-and-the-intuition) walks through all five; here is the short version:

- **S** — the price of the underlying stock.
- **sigma** — the implied volatility (the market's guess at how much the stock will move).
- **t** — time, usually measured as T, the years left until expiration.
- **r** — the risk-free interest rate.
- **K** — the strike price (fixed in the contract, so we do not take a derivative against it).

Each of the four moving inputs gets a first-order Greek — a single derivative of the price:

> **Delta** = ∂V/∂S (price vs spot). **Vega** = ∂V/∂sigma (price vs vol). **Theta** = ∂V/∂t (price vs time). **Rho** = ∂V/∂r (price vs rate).

That is the top two rows of the family tree. The third row — the **second-order Greeks** — comes from differentiating *again*. The most famous is **gamma**, which is ∂delta/∂S: how your delta itself changes as the stock moves. But delta also changes when *vol* moves and when *time* passes, and vega changes when vol moves. Those give us the three Greeks this post is really about:

- **Vanna** = ∂delta/∂sigma = ∂vega/∂S. (The two are equal — a mathematical fact called a cross-partial, or Maxwell, relation.)
- **Charm** = ∂delta/∂t. Also called *delta decay* or *delta bleed*.
- **Volga** (also *vomma*) = ∂vega/∂sigma. The convexity of vega.

The point of the family tree is that there is nothing mysterious here. Every Greek is "the slope of the option's price (or of another Greek) along one axis." The first-order Greeks tell you *where you are* on the surface; the second-order Greeks tell you *how that surface is curving and shifting underneath you* as the inputs move. Dealers care intensely about the second set because they are running thousands of positions that they delta-hedge continuously, and the second-order Greeks tell them how much that hedge is going to drift before they can rebalance it.

Everything below is computed from the Black-Scholes model — the same model used throughout this series. We do not re-derive it here; the full derivation lives in [the Black-Scholes post](/blog/trading/quantitative-finance/black-scholes). We just *read* its outputs.

## Rho: your exposure to interest rates

Start with the first-order Greek the primers skip. **Rho** measures how an option's price changes when the risk-free interest rate changes. Conventionally it is quoted as the dollar change in the option's price per share for a **1 percentage-point** (100 basis-point) move in rates.

### Why rates move option prices at all

The intuition runs through **carry** — the cost or benefit of financing a position. Think about what owning a call really gives you: the right to buy the stock later at a fixed strike K. Compared to buying the stock today, a call lets you *defer* paying for the stock. That deferred payment is money you get to keep in the bank earning interest until you decide to exercise. The higher the interest rate, the more valuable that deferral is — so **a higher rate makes a call worth more.**

A put is the mirror image. A put gives you the right to *sell* the stock at K later. Compared to selling the stock today (and putting the cash in the bank to earn interest), holding a put means you are *delaying* the moment you collect that cash. The higher the rate, the more you are giving up by waiting — so **a higher rate makes a put worth less.**

That is the whole story of rho's sign: **rate up lifts calls, depresses puts.** It is not about volatility or direction; it is purely the time value of money applied to the strike you will eventually pay or receive.

### Why rho is small for short-dated options and large for LEAPS

The size of rho scales with how long you are deferring that cash flow. A 30-day option defers the strike payment by a month — not much interest accrues in a month, so rho is tiny. A two-year LEAPS defers it by two years — now the interest on the strike is substantial, and rho is large.

Mathematically, the call rho in Black-Scholes is `K · T · e^(−rT) · N(d2)`, and you can see the `T` sitting right in front: double the time to expiry and you roughly double rho. Let me make that concrete.

![Rho versus time to expiry for an at-the-money call and put](/imgs/blogs/rho-dividends-and-the-second-order-greeks-vanna-volga-charm-2.png)

The figure plots rho (per 1% rate move) against time to expiry for an at-the-money call and put on a \$100 stock at 20% vol with rates at 4%. Two things jump out. First, the call line is positive and the put line is negative — the carry sign we just derived. Second, both fan out as you move right: rho grows roughly linearly with maturity. At three months it is a rounding error; at one year it is meaningful; at two years it is something you must hedge.

#### Worked example: rho P&L on a one-year LEAP from a 1% rate move

Take a one-year at-the-money call: stock S = \$100, strike K = \$100, time T = 1 year, vol sigma = 20%, rate r = 4%. From the Black-Scholes model the call is worth **\$9.93 per share** (so \$993 for one 100-share contract).

Now ask: what is its rho? The model's `od.rho` returns the sensitivity per a 1.00 (= 100%) change in the rate, so we divide by 100 to get the per-1%-point number. The result is:

> rho (call, per 1% rate move) = **+\$0.5187 per share**, i.e. **+\$51.87 per contract.**

Let's verify by repricing. Bump the rate from 4% to 5% (a full percentage point) and reprice the same call: it goes from \$9.93 to **\$10.45**, a gain of **+\$0.53 per share** (+\$53 per contract). That matches the rho of +\$0.52 almost exactly — the small difference is the curvature of rho itself over a one-point move.

The matching one-year put? It starts at \$6.00 and falls to **\$5.57** on the same rate move, a loss of **−\$0.43 per share**, right in line with its rho of **−\$0.44**.

*The intuition: a one-year call holder makes about half a dollar a share for every full point rates rise, just from carry — small per move, but in a year that moved rates four points (like 2022), that is roughly \$2 of pure rate tailwind on a \$10 call.*

To pin the 2022 hook precisely: a one-year at-the-money call at 25% vol was worth about **\$10.06** when rates sat at 0.25%; with rates at 4.5% the *same* option (same stock, same vol) was worth about **\$12.09** — a **+\$2.02 per share** lift attributable to nothing but the rate path. A two-year version gained about **+\$3.91**. That is what Marcus felt and could not name.

#### Worked example: why a short-dated trader can ignore rho

Now take the same at-the-money call but with only 30 days to expiry instead of a year. Its rho is:

> rho (30-day call, per 1% rate move) = **+\$0.0419 per share**, i.e. about **+\$4.19 per contract.**

Even a dramatic 1% surprise hike — rare in a single meeting — moves this option about four cents a share. Against the option's daily theta and its swings from delta and vega, four cents is in the noise. *This is why every short-dated options primer waves rho away: for the 0-to-45-day options most traders live in, it genuinely does not matter. It only wakes up for LEAPS and in regimes where rates themselves are volatile.*

### A parity check that proves the sign

Here is a clean sanity check. Put-call parity ties a call and a put on the same strike together. Their rhos must differ by exactly the rho of the forward strike payment, `K · T · e^(−rT)`. For our one-year option: `100 × 1 × e^(−0.04) = 96.08`. And indeed, the model's call rho minus put rho (in per-1.00 units) is **96.08**. The numbers are internally consistent — rho is not a vague hand-wave, it is the precise derivative the model defines.

This parity fact also tells you something practical about hedging rate risk. If you hold a call and a put at the same strike (a straddle), your *net* rho is the rho of a forward — you are implicitly long the forward, and your rate exposure is the rate exposure of buying the stock forward at K. If you want a position that is rate-neutral, you can combine options so the rhos cancel; if you want to *express* a rate view through options, you would lean long calls (positive rho) in a rising-rate thesis and long puts (negative rho) in a falling-rate thesis. In practice almost nobody trades rates through equity options — there are far cleaner instruments (rate futures, swaps) — but the exposure is there whether you intend it or not, and on a large LEAPS book it is worth knowing your aggregate rho the same way you know your aggregate delta.

One more nuance that trips people: rho as the model defines it assumes a *flat, constant* rate. In the real world rates have a term structure (the yield curve differs by maturity) and they move in correlated ways across maturities. A desk pricing a two-year option does not use the same rate as one pricing a one-month option, and a parallel shift of the whole curve is only one of many ways rates can move. For a single retail position none of this matters; for an institutional book of long-dated optionality, rho is decomposed across curve tenors exactly the way a bond desk decomposes duration. The single-number rho we computed is the textbook starting point, not the end of the story.

## Call versus put as rates rise

To see rho's two-sided nature in one picture, hold everything fixed and sweep the rate.

![Call value rises and put value falls as the risk-free rate increases](/imgs/blogs/rho-dividends-and-the-second-order-greeks-vanna-volga-charm-3.png)

The figure sweeps the risk-free rate from 0% to 8% for a one-year at-the-money call and put. The call line slopes up and the put line slopes down — they diverge. The slope of each line *is* rho. Notice it is gentle: across the entire 0-to-8% range the call gains only a few dollars. That gentleness is exactly why rho hides. But notice also that the lines are not flat — and for a LEAPS holder in a fast-tightening cycle, "not flat" times "rates moved four points" equals real money.

A subtle practical point: when option desks quote a "risk-free rate" they actually use a financing/repo rate appropriate to the underlying, and for index options the relevant rate interacts with the index's dividend yield. That is our bridge to the next first-order effect.

## Dividends: the other thing that shifts call and put values

Dividends are not one of the canonical five inputs, but they belong in this post because they are a *direct, first-order* shifter of option values — and because they create the single most important real-world exception to a rule beginners are taught as gospel ("never exercise an American option early").

### Why a dividend pushes calls down and puts up

When a stock pays a dividend, its price mechanically drops by roughly the dividend amount on the ex-dividend date (the day the stock starts trading without the right to the upcoming dividend). If a \$100 stock pays a \$2 dividend, on the ex-date it opens around \$98, all else equal. The dividend did not destroy value — it moved value from the share price into a cash payment that goes to whoever held the stock the day before.

But an option holder does **not** receive the dividend. A call holder owns the right to buy the stock, not the stock itself, so they get no dividend — they just watch the share price drop by the dividend on the ex-date. That expected drop makes the call worth **less**. A put holder benefits from the same expected drop (a lower stock makes their right-to-sell more valuable), so a dividend makes a put worth **more.**

In the Black-Scholes machinery, a continuous dividend yield `q` enters as a discount on the stock term — economically it acts like a *negative* interest rate on the stock side. The sign is the exact opposite of rho: where a higher rate lifts calls, a higher dividend yield depresses them. This symmetry is not a coincidence. The two carries — interest on the strike (rho) and dividends on the stock (the dividend effect) — are the two halves of the cost of *holding the underlying forward.* Buying a call instead of the stock lets you keep your cash earning the rate (good for the call) but means you forgo the dividends the stockholder collects (bad for the call). The net of those two is the option's "cost of carry," and it is why the same model handles rates and dividends with mirror-image signs.

There is a subtlety in *how* dividends are modeled. Real companies pay discrete, lumpy dividends on specific dates (a \$0.50 quarterly payment), not a smooth continuous yield. For broad indices, the continuous-yield approximation `q` is fine because the index has hundreds of stocks paying on staggered dates, smoothing into something close to a constant stream. For a single stock with one big dividend before expiration, desks instead subtract the *present value of the known dividend* from the stock price and price the option on the dividend-adjusted spot. Both approaches push call values down and put values up; the discrete method just times the drop to the actual ex-date rather than spreading it. For our intuition, the continuous-yield view is cleaner, and the direction of the effect is identical either way.

#### Worked example: how a dividend yield shifts a six-month option

Take a six-month at-the-money option (S = K = \$100, T = 0.5y, sigma = 20%, r = 4%) and vary the dividend yield `q`:

| Dividend yield q | Call value | Put value |
| --- | --- | --- |
| 0% | \$6.63 | \$4.65 |
| 2% | \$6.06 | \$5.07 |
| 4% | \$5.53 | \$5.53 |
| 6% | \$5.02 | \$6.00 |

Read down the columns: as the dividend yield climbs from 0% to 6%, the call sheds about **\$1.60 of value** (\$6.63 to \$5.02) while the put *gains* about the same (\$4.65 to \$6.00). At q = 4%, with the dividend yield exactly offsetting the 4% rate, the call and put are worth the same — the carry on the strike (rate) and the carry on the stock (dividend) cancel. *The intuition: a fat dividend is a headwind for call buyers and a tailwind for put buyers, because the call holder eats the ex-date price drop without ever collecting the cash that caused it.*

### The one time early exercise of a call is rational

You will read everywhere that exercising an American option early "throws away time value" and is almost never optimal. For puts and for non-dividend-paying calls, that is true. But there is exactly one classic exception, and dividends are the reason.

Consider a **deep in-the-money American call** on a stock that is about to pay a meaningful dividend tomorrow. If you exercise the call *today* (before the ex-date), you convert the call into actual stock, and you own that stock through the record date — so **you capture the dividend.** The cost of doing so is that you give up the option's remaining **time value** (extrinsic value): once you exercise, you hold stock with downside, not a call with a floor.

So the decision is a straight inequality:

> **Exercise early if: dividend captured > time value given up (plus the small interest you'd have earned on the strike by waiting).**

For a deep in-the-money call near expiry, the time value is tiny — the option is almost all intrinsic value, behaving like the stock itself. If the dividend is larger than that sliver of time value, exercising to grab the dividend wins.

![Decision figure for early exercise of an American call before ex-dividend](/imgs/blogs/rho-dividends-and-the-second-order-greeks-vanna-volga-charm-4.png)

#### Worked example: the early-exercise decision with numbers

Take a deep in-the-money American call: stock S = \$110, strike K = \$100, with 15 days left, sigma = 20%, r = 4%. From the model the call is worth **\$10.18 per share.** Its intrinsic value is `110 − 100 = \$10.00`, so its **time value is only \$0.18 per share.**

Now suppose the stock goes ex-dividend tomorrow with a **\$2.00 dividend.**

- **Exercise early:** capture the \$2.00 dividend, give up the \$0.18 of time value. Net benefit = **\$2.00 − \$0.18 = +\$1.82 per share.** Exercise.
- **Hold:** if instead the dividend were only **\$0.10**, the comparison flips: \$0.10 captured versus \$0.18 of time value surrendered = **−\$0.08 per share.** You would rather hold (or simply sell the call to someone else), keeping the optionality.

*The intuition: early exercise of a call is rational only in a narrow window — deep in the money (so the time value is thin), right before a fat ex-dividend (so the dividend is big), with little time left (so there is barely any time value to lose). Outside that window, you sell the call instead of exercising it.* This is exactly why deep-ITM calls on high-dividend stocks see a wave of early assignments the day before ex-dividend — a phenomenon we cover from the short seller's side in the forthcoming post on [assignment, pin risk, and expiration mechanics](/blog/trading/options-volatility/assignment-pin-risk-and-expiration-day-mechanics).

A note for the curious: the time value never goes to *exactly* zero before expiry, so there is always *some* time value being sacrificed. The cleaner way to think about the trade is that you should only consider early exercise when the dividend exceeds that residual time value, and you should compare against simply *selling* the call — which captures the time value the market will still pay you. Most of the time, selling beats exercising.

## The second-order Greeks: derivatives of the derivatives

Now we climb the family tree to the second-order Greeks. These measure how the *first-order* Greeks change. Why should you care how a Greek changes? Because if you are hedging, you set your hedge based on a Greek's current value — and if that Greek is drifting, your hedge is going stale even when nothing dramatic seems to be happening.

The most familiar second-order Greek is **gamma** — ∂delta/∂S — which the [delta post](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio) covers: how your delta changes as the stock moves. This post handles the other three, the ones that move your delta and vega for reasons *other than* the stock price: a change in **vol** (vanna), the passage of **time** (charm), and the convexity of your vol exposure (volga).

### Vanna: how your delta drifts when vol moves

**Vanna** is the rate of change of delta with respect to implied volatility: ∂delta/∂sigma. Equivalently — and this surprises people — it is also the rate of change of *vega* with respect to spot, ∂vega/∂S. These two are mathematically identical (a cross-partial derivative does not care which order you take it in), and that double identity is why vanna sits at the heart of skew trading: it links your directional exposure to your vol exposure.

The everyday way to feel vanna: you have delta-hedged a position so you are "delta neutral." Then, with the stock dead flat, **implied vol jumps**. Suddenly you are not delta neutral anymore — your delta has shifted, and you have an unhedged directional exposure you did not have a minute ago. Vanna is the number that told you, in advance, how big that shift would be.

![Delta versus spot at low and high implied volatility showing the vanna shift](/imgs/blogs/rho-dividends-and-the-second-order-greeks-vanna-volga-charm-5.png)

The figure shows two delta-vs-spot curves for the same call: one at 20% implied vol, one at 35%. They are not the same curve. Where the option is out-of-the-money (left side), raising the vol *lifts* delta — a higher vol gives a struggling option more chance of finishing in the money, so it behaves more like the stock. The vertical gap between the two curves, per unit of vol, *is* vanna. It is largest in the wings (out-of-the-money and in-the-money) and passes near zero right at the money, where delta is already pinned around 0.5 regardless of vol.

#### Worked example: a vanna delta shift from a 5-vol IV move

Take an out-of-the-money call: stock S = \$100, strike K = \$110, T = 90 days, r = 4%, at sigma = 20%. From the model its delta is **0.2088** — for every \$1 the stock moves, the option moves about 21 cents.

Now implied vol rises 5 points, from 20% to 25%, with the stock dead flat at \$100. Reprice the delta: it becomes **0.2656.** Your delta jumped by **+0.0568** with no move in the stock at all.

What does that mean in dollars of hedge? If you were short 100 of these calls and had hedged by holding 21 shares (to offset the 0.21 delta), you are now short 0.27 deltas per contract — your hedge is off by about **0.0568 × 100 = 5.7 shares per contract.** A vol spike just handed you an unhedged short-delta position. To stay neutral you must buy about 6 more shares per contract.

We can also read this as the local vanna: finite-differencing delta against sigma gives a vanna of about **1.31 per 1.00 (= 100 vol points) of sigma**, so ×0.05 predicts roughly **+0.065** — close to the exact +0.057, with the gap being delta's own curvature in vol. *The plain reading: vanna is why a vol-neutral, delta-neutral book can suddenly grow a directional tilt the instant the vol surface moves — and why desks that are long downside puts (negative vanna) get longer delta as markets fall and vol rises, forcing them to sell into the decline.*

#### Worked example: proving vanna equals vega's spot-sensitivity

The claim that ∂delta/∂sigma = ∂vega/∂spot is easy to assert and worth seeing in numbers, because it is the bridge between the directional and the vol world. Take the same OTM call (S = \$100, K = \$110, T = 90 days, r = 4%, sigma = 20%).

- Compute vanna **the delta way**: bump sigma up and down a hair around 20% and measure how delta changes. The model gives ∂delta/∂sigma ≈ **1.307 per 1.00 of sigma.**
- Compute it **the vega way**: bump the *stock* up and down a dollar around \$100 and measure how vega changes. The model gives ∂vega/∂spot ≈ **1.307.**

They match to three decimals — the tiny residual is just finite-difference rounding. *The lesson: there is only one number here, and it has two faces. When you hedge the drift of your delta against vol, you are simultaneously hedging the drift of your vega against spot. A desk does not track these as two separate risks; they track one vanna.*

The double identity of vanna — ∂delta/∂sigma equals ∂vega/∂spot — is worth pausing on because it is the reason vanna matters so much to desks. Read it the second way: it says your *vega* changes as the *stock* moves. So a position that is vega-neutral right now will not stay vega-neutral after the stock moves, and a position that is delta-neutral now will not stay delta-neutral after vol moves. Vanna is the single number that quantifies *both* of those drifts, because they are the same derivative. A desk that is short downside puts (the classic dealer position) has *negative* vanna: as the market falls and vol rises, the puts' deltas become more negative, so the dealer's net delta climbs (they get longer the falling market on the short-put side), and they are forced to **sell stock into the decline** to re-hedge. That forced selling is one of the mechanical accelerants of a sell-off, and it is pure vanna.

The vanna-and-skew connection deserves its own paragraph. In equity indices, out-of-the-money puts trade at higher implied vol than calls — the [volatility skew](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more). Because vanna links delta to vol, a position's vanna tells you how its delta will move as the *skew* shifts, not just as the level of vol shifts. When you trade a risk reversal (long an OTM call, short an OTM put, or vice versa) you are taking a position whose dominant exposure is *not* vega and *not* delta but vanna — you are betting on how the relationship between spot and vol behaves. Skew traders live and die by vanna; it is the Greek of the smile's tilt, just as volga (next) is the Greek of the smile's curvature.

### Charm: how your delta bleeds as time passes

**Charm** is the rate of change of delta with respect to time: ∂delta/∂t. It is often called **delta decay** or **delta bleed**, and it answers a question that catches hedgers off guard: *even if the stock does not move at all, how will my delta have changed by tomorrow simply because a day passed?*

The reason delta moves with time is that, as expiry approaches, an option's delta gets pulled toward its terminal value. At expiration, an in-the-money option has a delta of exactly 1.0 (it *is* the stock) and an out-of-the-money option has a delta of exactly 0 (it expires worthless). So as the clock runs down with spot held flat, an in-the-money call's delta firms up toward 1, and an out-of-the-money call's delta bleeds down toward 0. Charm is the speed of that drift, usually quoted **per day.**

![Delta versus days to expiry for an out-of-the-money and an in-the-money call](/imgs/blogs/rho-dividends-and-the-second-order-greeks-vanna-volga-charm-6.png)

The figure plots delta against days-to-expiry (time running to the right, toward expiry) for an out-of-the-money call (strike \$105, red) and an in-the-money call (strike \$95, green), both on a \$100 stock. The red line slumps toward 0; the green line climbs toward 1. The slope of each line is charm. Notice the curves bend hardest near expiry — charm, like gamma and theta, is most violent in the final days. That is why hedgers obsess over it into monthly expiration (OPEX) and over weekends, when several days of charm accrue before they can next trade.

#### Worked example: charm delta bleed over a week

Take the out-of-the-money call: S = \$100, K = \$105, sigma = 20%, r = 4%, with **30 days** to expiry. Its delta is **0.2222.**

Now let one calendar week (7 days) pass with the stock perfectly flat at \$100, so 23 days remain. Reprice the delta: it is now **0.1850.** The delta bled by **−0.0372 over the week**, or about **−0.0053 per day** of charm.

In hedge terms: if you were long 100 of these calls and hedged by shorting about 22 shares, after a flat week your option delta is only 18.5 — you are now *over-hedged short* by about 3.7 shares per contract, purely from the calendar. To stay neutral you would buy back roughly 4 shares per contract. *The intuition: charm is the reason a delta-hedged book drifts off-neutral over a quiet weekend or into expiration even when the underlying never moves — your delta is silently sliding toward 0 or 1 as the options age.* For the in-the-money call (strike \$95) the same flat week pushes delta the *other* way, up toward 1, because that option is hardening into stock.

There is a practical wrinkle that catches every new hedger: **charm accrues over calendar time, but you can only trade over market time.** Over a weekend, three days of charm pile up (Friday close to Monday open) with no chance to rebalance in between. The same is true across holidays. Desks that run tight delta-neutral books compute the expected charm drift over the closed period and pre-hedge for it on Friday afternoon — buying or selling the shares they *know* the calendar will require by Monday — rather than waking up to an off-neutral book at the Monday open. Into a long three-day weekend ahead of a monthly expiration, that pre-hedging can itself be a meaningful flow.

Charm has a directional asymmetry that matters for desks. Dealers who are net short out-of-the-money options (a common structural position, since the public buys lottery-ticket calls and protective puts) hold positions whose delta drifts predictably as time passes. A short OTM put has positive delta that fades toward zero as the put decays; to stay hedged the dealer, who had sold stock against it, must buy that stock back over time. Multiply across thousands of contracts approaching a big expiration and you get a slow, mechanical, news-free bid for stock — the charm half of the vanna-charm rally we get to shortly. That predictable drift is one of the engines behind the flows we discuss next, and it is why so many desks watch the *gamma and charm profile by strike* in the final week of every monthly cycle.

### Volga: the convexity of vega

**Volga** (also called **vomma**) is the rate of change of vega with respect to implied vol: ∂vega/∂sigma. If vega is your exposure to vol, volga is the *convexity* of that exposure — how much your vega itself grows or shrinks as vol moves. It is the vol-world analogue of gamma (which is the convexity of delta).

Why does this matter? Because vega is not a constant. For an at-the-money option, vega is remarkably stable across different vol levels — its volga is near zero. But for a **far out-of-the-money** option, vega is small when vol is low and grows rapidly as vol rises. That positive convexity is volga, and it is the mathematical fingerprint of the **volatility smile's curvature.**

![Vega versus implied volatility for an at-the-money and a far out-of-the-money call](/imgs/blogs/rho-dividends-and-the-second-order-greeks-vanna-volga-charm-7.png)

The figure plots vega (per 1 vol point) against implied vol for an at-the-money call (strike \$100, blue) and a far out-of-the-money call (strike \$120, amber), both 90 days out. The blue at-the-money line is nearly flat — its vega barely changes whether vol is 10% or 50%. The amber out-of-the-money line is steeply curved — its vega starts near zero and climbs hard as vol rises. The curvature of that amber line *is* volga.

#### Worked example: a far-OTM call gaining vega as vol rises

Take the far out-of-the-money call: S = \$100, K = \$120, T = 90 days, r = 4%. At **20% vol** its vega (per 1 vol point) is **\$0.0478 per share** — almost nothing; a one-point move in vol barely budges it. Now let implied vol rise to **30%.** Reprice the vega: it is now **\$0.1102 per share** — it more than doubled.

Compare the at-the-money call (strike \$100) over the same vol move: its vega goes from **\$0.1959** to **\$0.1961** — essentially unchanged. *The intuition: an at-the-money option has nearly zero volga (its vega is flat in vol), but a far-OTM option has large positive volga — as vol rises, the OTM option's vega balloons, so a long-OTM-option position gets *more* sensitive to vol exactly when vol is rising. That self-reinforcing sensitivity is why tail-hedge buyers (long far-OTM puts) make outsized money in a vol spike, and it is the reason the vol smile bends upward at the wings rather than staying flat.*

It helps to hold the gamma analogy firmly. Gamma is the convexity of the *price* in spot: a long-gamma position makes money from movement in the stock because its delta moves favorably (you get longer as the stock rises, shorter as it falls). Volga is the exact same idea one axis over: a long-volga position makes money from movement in *vol* because its vega moves favorably (you get longer vega as vol rises). Just as you can be delta-neutral but long gamma (neutral to small moves, profiting from big ones), you can be vega-neutral but long volga (neutral to small vol changes, profiting from big vol *swings*). A trader who believes vol itself will become more volatile — that the next vol regime will be jumpy rather than calm — wants positive volga, and they get it by owning the wings of the smile (far-OTM options) rather than the at-the-money straddle.

This is also why the smile exists as a *shape* rather than a flat line. If volga were zero everywhere, all strikes would respond to vol identically and the implied-vol curve across strikes would be flat. Because far-OTM options have high positive volga, the market prices them with extra implied vol — buyers pay up for that convexity, sellers demand compensation for being short it — and the curve bends upward at the wings. The curvature of the smile *is* the market's collective price for volga. Volga is what makes structures like the **risk reversal** and **butterfly** behave the way they do, and it is the second-order Greek that vol-of-vol traders (covered in the [vega and vol-of-vol post](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol)) are really trading. Long volga = long the curvature of the smile = long the convexity of vol.

## How it shows up in real markets

The second-order Greeks would be a footnote if they only lived in textbooks. They do not. They drive some of the most reliable flow patterns in the equity market.

### The vanna-charm rally

Here is the mechanism that ties this whole post together. Options dealers (market-makers) are, in aggregate, on the other side of the public's trades. The retail and institutional crowd tends to **buy puts for protection** and **buy calls for upside**, which leaves dealers net **short** those options. To stay risk-neutral, dealers **delta-hedge** their books continuously — and crucially, they re-hedge as their *second-order* Greeks move their deltas.

Two of those second-order effects line up in the same direction in the back half of a calm month:

- **Charm:** as expiration approaches with the market grinding higher or sideways, the out-of-the-money puts dealers are short bleed delta. A short put has positive delta; as the put decays toward worthless, that delta fades toward zero. To stay hedged, dealers who had shorted stock against those puts must **buy stock back** — a steady, mechanical bid. This is charm-driven flow.
- **Vanna:** in a calm, grinding-higher tape, implied vol typically *drifts down.* Falling vol changes the delta of the puts dealers are short (vanna), and again the hedge adjustment is to **buy stock.** This is vanna-driven flow.

When both fire together — a calm tape into a big monthly expiration, with vol drifting lower — dealers are mechanically buying stock for *both* charm and vanna reasons, with no fundamental news required. Traders call the result a **"vanna-charm rally"**: a grind higher in the days before OPEX that is powered by hedging mechanics, not by anyone deciding the market is cheap. The flip side is just as real: a vol spike reverses the vanna flow violently (dealers must *sell*), which is part of why down moves can feed on themselves. We unpack the full dealer-positioning machine — gamma, charm, and vanna together — in the forthcoming post on [dealer gamma, charm, and vanna](/blog/trading/options-volatility/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot).

The takeaway for a non-dealer: when you see the market drift higher into a monthly expiration on no news, and then chop or reverse the Monday after, you are very likely watching second-order Greek hedging flows wash through and then reset.

A worked sense of scale, without inventing numbers: the effect is strongest when (a) a large notional of options is set to expire (so there is a lot of delta to re-hedge), (b) the tape is calm so vol is drifting down rather than spiking, and (c) dealers are net short the relevant strikes. None of those conditions is exotic — they describe a typical quiet week before a quarterly expiration. The flows are not a conspiracy and not a guaranteed trade; they are the visible footprint of risk-neutral hedging done at scale. The reason this matters to *you*, even if you never trade an option, is that it changes the base rate of what "no news" days do near expiration: the drift is real, but it is also self-extinguishing once the expiring options roll off, which is why the post-OPEX reset is as reliable as the pre-OPEX grind.

### The 2022-2023 rho regime

Marcus's surprise from the opening was not a one-off. For most of the 2010s, rates sat near zero and rho was, correctly, ignored by nearly everyone. Then 2022 changed the regime: the fastest tightening cycle in forty years made the *rate* itself a fast-moving variable. Suddenly, desks pricing long-dated structured products, LEAPS, and convertible bonds had to actively manage rho — a Greek their risk systems had been running on autopilot.

The practical lesson is that **Greeks are not equally important in all regimes.** In a zero-rate, low-vol world, rho and volga are afterthoughts. In a high-rate, high-vol-of-vol world, they move real money. A good options trader does not memorize a fixed ranking of which Greeks matter; they ask *which inputs are moving fast right now* and weight their attention to the matching Greeks.

### Dividend assignment season

Every quarter, around the ex-dividend dates of large, high-yielding stocks, options desks see a predictable surge of early assignments on deep in-the-money calls — exactly the early-exercise mechanic we worked through. Short-call sellers who do not watch their positions into ex-dividend can be assigned, suddenly find themselves short the stock, and owe the dividend. The decision figure earlier in this post is, from the assignment side, a *risk checklist*: if you are short a deep-ITM call on a stock about to pay a dividend bigger than the call's remaining time value, assume you will be assigned and plan for it.

This shows up most acutely for traders running covered-call and short-call-spread positions on high-yield names. A trader who sold a deep-ITM call as part of a spread might assume both legs run to expiration — but if the long counterparty exercises early to grab the dividend, the short leg is assigned days ahead of schedule, the trader is suddenly short stock through the record date, and they owe the dividend on the assigned shares. The position that looked fully defined now has an unexpected cash outflow and a stock position to unwind. None of this is bad luck; it is the predictable output of the inequality in the decision figure, run from the other side of the trade. The defense is simple: in the days before a known ex-dividend, scan your short calls for any that are deep in the money with thin time value, and either roll them, close them, or accept and plan for the assignment. The mechanic is the same whether you are the holder deciding to exercise or the seller bracing to be assigned — only the sign of the cash flow flips.

## Common misconceptions

**"Rho doesn't matter, so I can ignore interest rates entirely."** Half right. Rho is negligible for the short-dated options most retail traders hold — a 30-day at-the-money call has a rho of about **\$0.04 per share per 1% move**, genuine noise. But for LEAPS it is real: a one-year call's rho is **\$0.52 per share per 1%**, and a two-year call's is **\$1.03.** In 2022, rates moved several whole points; a LEAPS holder who ignored rho mis-estimated their P&L by dollars per share. The fix is not to obsess over rho on weeklies — it is to *check the maturity*. Long-dated, rate matters; short-dated, it does not.

**"A dividend just lowers the stock, so it doesn't really change my option."** It changes it directly. The expected ex-date price drop is already baked into the option price *before* the dividend is paid, and it is sign-flipped between calls and puts. Our six-month example showed a call losing **\$1.60** and a put gaining about the same as the dividend yield went from 0% to 6%. If you buy a call on a high-dividend stock and ignore the dividend, you have overpaid relative to what the option can actually deliver, because you will eat the ex-date drop without ever collecting the cash.

**"You should never exercise an American option early — it throws away time value."** True for puts and for non-dividend-paying calls, false for one specific case. A deep in-the-money American call, right before a fat ex-dividend, with little time value left, *should* be exercised early when the dividend captured (e.g. \$2.00) exceeds the time value surrendered (e.g. \$0.18). That is the **+\$1.82 per share** edge from our worked example. The rule is not "never"; it is "only when the dividend beats the residual time value."

**"Vanna and charm are exotic Greeks only quants need."** They are exotic in the sense of being second-order, but they drive flows that move the index you trade. Charm and vanna hedging by dealers is the documented engine behind pre-OPEX grinds and the violence of vol-spike sell-offs. You do not need to *compute* them to trade equities, but if you do not know they exist, the "no-news drift higher into expiration, then reversal" pattern will look like magic. It is mechanics.

**"Vega is a single number, so my vol exposure is fully described by it."** Only locally. Vega is the *first* derivative of price against vol; volga is the second. For an at-the-money option vega is nearly flat in vol (volga ≈ 0), so the single number is fine. But for a far-OTM option, vega *doubled* (from **\$0.0478** to **\$0.1102** in our example) as vol rose from 20% to 30%. If you size a tail-hedge position by its current vega alone, you will badly underestimate how much it pays in a real vol spike — because its vega grows as vol grows. That growth is volga, and it is why far-OTM options are convex bets on volatility, not linear ones.

**"The second-order Greeks are just smaller versions of the first-order ones."** No — they answer different questions, and their *signs* relative to your position can flip your intuition. You can be long gamma and short volga at the same time (a calendar spread does this), or delta-neutral with large negative vanna (a short-skew position). Treating them as "delta but tinier" misses the point: vanna couples your direction to vol, charm couples your direction to the calendar, and volga couples your vol exposure to itself. They are not weaker forces; they are *different* forces that happen to be quiet until the input they track moves. The whole reason dealers maintain a full Greek matrix rather than a single risk number is that these exposures do not collapse into one another.

## The playbook: trading and hedging the Greeks beyond the big four

Here is how to actually use everything above.

**For the directional position-holder (you are long or short calls/puts):**

- **Check rho only on long-dated trades.** If your option has under ~90 days to expiry, set rho aside — it is noise next to theta, delta, and vega. If you hold LEAPS or sell long-dated structures, pull the rho: a one-year ATM call carries about **+\$0.52 per share of rho per 1%**, and you want to know whether you are implicitly long or short rates. Long calls = long rates; long puts = short rates.
- **Account for dividends before you buy.** On a dividend-paying name, the call you are buying has the expected ex-date drop already priced in. Do not be surprised when the stock gaps down on the ex-date and your call does not "lose" — that drop was anticipated. If you are *short* a deep-ITM call into an ex-dividend, assume early assignment when the dividend exceeds the call's time value, and either close the position or be ready to be assigned the stock and owe the dividend.

**For the hedger (you run a delta-neutral or vega-neutral book):**

- **Re-hedge for vanna after a vol move.** When implied vol jumps with spot flat, your delta has shifted by roughly vanna × (vol change). In our example a 5-point vol rise added **+0.057 of delta** to an OTM call — about 6 shares per contract of new exposure. Rebalance, or you are carrying an accidental directional bet.
- **Re-hedge for charm over time, especially over weekends and into OPEX.** Delta drifts toward 0 (OTM) or 1 (ITM) as the clock runs even with spot flat. Our OTM call bled **−0.037 of delta over one flat week** (~4 shares per contract). Going into a long weekend or the final week before expiration, pre-compute the charm drift and adjust, rather than discovering it Monday morning.
- **Size vol exposure by volga, not just vega, for OTM positions.** A long far-OTM option is a convex vol bet — its vega *grows* as vol rises (volga). Our OTM call's vega doubled from **\$0.048 to \$0.110** as vol went 20%→30%. If you are buying tail protection, that convexity is the feature you are paying for; size it knowing the payoff in a real spike is bigger than the linear-vega estimate suggests.

**For reading the tape (you trade the index, not the options):**

- **Expect vanna-charm grinds into monthly expiration** in calm, vol-falling tapes: dealers mechanically buy stock to re-hedge short puts as charm and vanna fade their deltas. Expect a reset after OPEX. Treat a no-news drift higher into a big expiration as a flow signal, not a fundamental one — and respect that the same machinery reverses hard if vol spikes.

**Entry, exit, and invalidation.** None of these are standalone trades — they are *adjustments* and *context* for trades whose primary thesis lives in delta and vega. The entry is "I have a position with material rho/vanna/charm/volga." The exit/adjustment trigger is a move in the matching input: a rate surprise (rho), a vol jump (vanna, volga), or simply the calendar advancing (charm). The invalidation is regime: in a zero-rate, dead-vol world these Greeks sleep; when rates or vol-of-vol wake up, move them to the front of your risk dashboard.

The deepest lesson of this post is the family tree at the top. The big four Greeks are the slopes of your option's price. The second-order Greeks are how those slopes are being *redrawn* under your feet as vol, rates, and time move. The traders who get blindsided are the ones who hedge the slopes once and assume they stay put. The traders who survive know the map is always shifting — and they watch the Greek that is shifting it.

## Further reading & cross-links

- [Delta: direction exposure and the hedge ratio](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio) — the first-order Greek whose drift vanna and charm measure.
- [Vega: your exposure to implied volatility and the vol-of-vol](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol) — the Greek whose convexity is volga.
- [What sets an option's price: the five inputs and the intuition](/blog/trading/options-volatility/what-sets-an-options-price-the-five-inputs-and-the-intuition) — where rate and dividend live among the price drivers.
- [The options chain and contract mechanics](/blog/trading/options-volatility/the-options-chain-and-contract-mechanics-multiplier-expiry-settlement) — multiplier, expiry, and settlement, for converting per-share Greeks to per-contract dollars.
- [Time value and theta: why an option is a melting ice cube](/blog/trading/options-volatility/time-value-and-theta-why-an-option-is-a-melting-ice-cube) — the time-decay context for charm.
- [The volatility smile and skew: why OTM puts cost more](/blog/trading/options-volatility/the-volatility-smile-and-skew-why-otm-puts-cost-more) — the curvature that volga measures and the skew that vanna trades. *(forthcoming)*
- [Dealer gamma, charm, and vanna: how options flows move the spot](/blog/trading/options-volatility/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot) — the full dealer-hedging machine behind the vanna-charm rally. *(forthcoming)*
- [Assignment, pin risk, and expiration-day mechanics](/blog/trading/options-volatility/assignment-pin-risk-and-expiration-day-mechanics) — the short seller's view of early-exercise and dividend assignment. *(forthcoming)*
- [Black-Scholes, from the ground up](/blog/trading/quantitative-finance/black-scholes) — the pricing model every Greek in this post is a derivative of.

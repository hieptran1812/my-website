---
title: "Gamma: The Greek That Bites — Curvature, Convexity, and the Toxic Short"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn what gamma really is, why it makes a long option a friend and a short option a time bomb, and how to read the curvature that turns small moves into big ones near expiry."
tags: ["options", "volatility", "gamma", "convexity", "options-greeks", "delta-hedging", "gamma-scalping", "short-gamma", "pin-risk", "black-scholes"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — Gamma is the rate at which delta changes per \$1 move in the stock. It is the *curvature* of an option's value, and it is the reason a long option helps you more and more as you are right, and a short option hurts you more and more as you are wrong.
>
> - **Gamma = convexity.** If you are long options you have positive gamma: your delta grows when the move helps you and shrinks when it hurts you. That is "buy low, sell high" baked into the position — but you rent it by paying theta.
> - **Short gamma is the toxic position.** Your delta moves *against* you, forcing you to rehedge by selling low and buying high. You collect small premium most days and take a catastrophic loss on a gap. This is the structural reason naked option sellers blow up.
> - **Gamma peaks at the money and explodes into expiry.** A 30-day at-the-money option has gamma around 0.069; the same option with one day left has gamma around 0.38 — more than five times the curvature, all packed into the final hours.
> - **The one rule to remember:** gamma is the second derivative that turns a calm position into a violent one. Long it, and chop pays you; short it, and a single gap can erase a year of premium. Never be short gamma without knowing exactly how big your worst overnight move can be.

## Two traders, one night

On the evening of August 2, 2024, two traders went home holding opposite sides of the same kind of bet on a major equity index. Both had positions worth a few thousand dollars in premium. Both had spent the summer collecting or paying small amounts day after day. Neither thought the weekend would be remarkable.

The first trader was *short* options — she had sold index puts and calls, collected the premium, and hedged herself delta-neutral so she had no obvious directional bet. For months this had been a quiet, profitable grind: the index barely moved, the options she'd sold decayed a little every day, and she banked the decay. On Friday she went home flat and content, short a stack of options that were melting in her favor. What she was actually short, though she may not have framed it this way, was *gamma* — the curvature of those options — and the position was a coiled spring.

The second trader was *long* options. He had paid premium for a straddle on the same index, bled a little theta every quiet day all summer, and watched his position do nothing while the seller collected. He was the one who looked foolish through July. What he owned, again whether or not he said it out loud, was *gamma* — positive curvature, convexity in his favor — and he was waiting for the spring to release.

Over that weekend the yen carry trade unwound. Monday, August 5, the index gapped down hard at the open; the VIX spiked to **38.57** intraday, one of the largest single-day volatility shocks on record. The short-gamma trader watched her quiet, hedged, "neutral" book detonate: as the market fell her sold puts ballooned in value, her delta lurched against her, and every hedge she put on locked in a loss because she was forced to sell into a collapsing market and buy back into the bounce. The premium she'd collected all summer evaporated in a morning, and then some. The long-gamma trader had the mirror experience: as the market crashed his straddle's delta swung sharply negative in his favor, and rehedging *handed* him profit — he was selling strength and buying weakness automatically, banking the violent move he'd been renting all summer.

Same index. Same weekend. Same Greek. One trader was destroyed by it and the other was made whole by it, and the only difference was the *sign* of their gamma. This post is about that Greek: what it is, where it lives, why being long it is a blessing you pay rent for, why being short it is the structural reason naked sellers blow up, and how to feel curvature in your gut before it bites you.

![Gamma versus stock price for a near-dated and a longer-dated option, both peaking at the strike](/imgs/blogs/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short-1.png)

Look at the two bells above. Each curve is gamma — the curvature of an option — plotted against the stock price, for a strike of \$100. The blue curve is a 30-day option; the red one is the same option with 7 days left. Both peak right at the strike (gamma is biggest at the money) and both fade to nothing in the wings. But the near-dated red bell is *taller and narrower* — its curvature is concentrated into a sharp spike right at the money, and it spikes higher the closer you get to expiry. That shape — a bell that grows taller and skinnier as the clock runs out — is the entire story of gamma, and everything below is an unpacking of why it looks like that and what it does to your money.

## Foundations: from delta to its rate of change

Before we can talk about gamma we have to be precise about delta, because **gamma is the rate of change of delta**, and you can't measure the change in a thing you can't pin down. If delta feels fuzzy, the dedicated post [Delta: Direction, Exposure, and the Hedge Ratio](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio) builds it from scratch; here is the one-paragraph recap we need.

**Delta** is how much an option's price moves for a \$1 move in the underlying stock. A call with delta 0.50 gains about \$0.50 when the stock rises \$1; a call with delta 0.90 gains about \$0.90. You can read delta three equivalent ways, and all three matter for what follows:

- **As a slope:** delta is the slope of the option-value curve — the rate at which the premium changes as the stock changes.
- **As a hedge ratio:** delta tells you how many shares of stock to trade to offset the option's directional risk. A call with delta 0.50 behaves like 50 shares (for a 100-share contract), so you'd short 50 shares to be "delta-neutral."
- **As a probability-ish number:** delta roughly tracks the chance the option finishes in-the-money. A deep in-the-money call has delta near 1 (almost certainly finishes ITM); a far out-of-the-money call has delta near 0.

Here is the crucial fact that makes gamma necessary: **delta is not constant.** As the stock moves, the option's delta changes. A call that was 0.50 delta at a \$100 stock might be 0.60 delta after the stock rises to \$101, and 0.83 delta after it rises to \$105. The hedge ratio you set up this morning is wrong by this afternoon, because the stock moved and your delta moved with it. The Greek that measures *how fast delta moves* is gamma.

> **Gamma is the rate of change of delta per \$1 move in the stock.** If delta is the speedometer (how fast the option's value changes as the stock changes), gamma is the accelerator (how fast that speed itself is changing).

In the language of calculus, if delta is the first derivative of the option price with respect to the stock price, gamma is the second derivative. You do not need calculus to trade it, but the picture the calculus gives is exactly right: **delta is the slope of a curve, and gamma is the curvature of that curve** — how much it bends.

### Two ways to define the same number

There are two equivalent definitions of gamma, and holding both makes it concrete.

**Definition 1 — gamma is the change in delta.** If the stock moves \$1 and your delta goes from 0.50 to 0.57, your gamma is about 0.07. Gamma is quoted in "delta per \$1 of stock," so a gamma of 0.07 means "every \$1 the stock moves, I pick up (or lose) 0.07 of delta." For one standard 100-share contract, that's 7 shares of equivalent stock exposure appearing or disappearing for every \$1 the underlying moves.

**Definition 2 — gamma is the curvature of the value curve.** Plot the option's value against the stock price and you get a curved line. Delta is the slope of that line at any point. Gamma is how much the slope is bending — the curvature. A straight line has zero curvature (zero gamma); a sharply bending curve has high curvature (high gamma).

Both definitions describe the same quantity, and our pricer computes it directly. Let me make it dollars-and-cents.

### The units, and why the sign is what matters

Gamma is quoted as *delta per dollar of underlying move*, which is a slightly awkward unit until you anchor it. A gamma of 0.069 means: for every \$1 the stock moves, your delta changes by 0.069. Per standard 100-share contract, multiply by 100 — so that same option gains or loses about 6.9 shares of equivalent stock exposure for each \$1 the underlying travels. Run the stock up \$10 and you've silently acquired roughly 69 shares of extra length (with the correction that gamma fades as you go, as we saw). The number looks small on paper precisely because it's a *second*-order effect — but second-order effects are the ones that surprise you, because they're invisible until the move is large.

The sign of gamma is the single most important thing about it, more important than the magnitude. **Long an option → positive gamma. Short an option → negative gamma.** This is true for calls and for puts, with no exceptions. A long call and a long put both have positive gamma; a short call and a short put both have negative gamma. The reason is that gamma measures curvature, and *owning* optionality always curves your payoff favorably (you keep the good tail, you cap the bad one), while *selling* optionality always curves it unfavorably (you cap your gain, you keep the bad tail). Positive gamma is the geometric fingerprint of "I own the right to choose"; negative gamma is the fingerprint of "I sold someone else that right."

A subtle, useful consequence: **a call and a put at the same strike and expiry have identical gamma.** This falls straight out of put-call parity — a call minus a put equals a forward position in the stock, and a forward has zero curvature (it's linear in the stock price), so the call and the put must have the *same* curvature as each other. (The parity relationship itself is proved in [Put-Call Parity and No-Arbitrage](/blog/trading/quantitative-finance/put-call-parity-no-arbitrage-quant-interviews).) That's why a straddle — long a call and a long put at the same strike — has exactly *twice* the gamma of either leg, a fact we'll lean on heavily when we scalp it. It also means that when a trader talks about "buying gamma," the call-versus-put choice is about delta and skew, not about the gamma itself; the curvature you're buying is the same either way.

#### Worked example: computing gamma and watching delta move

Take our running setup — a \$100 stock, a \$100-strike call, 20% annualized volatility, a 4% risk-free rate, and 30 days to expiration. Using the Black-Scholes model from this series' pricer:

- **Delta at \$100:** `delta(100, 100, 30/365, 0.04, 0.20, kind="call")` = **0.5343**. The call behaves like 53.4 shares.
- **Gamma at \$100:** `gamma(100, 100, 30/365, 0.04, 0.20)` = **0.06932**. So a \$1 move should change delta by about 0.069.

Now let's check that the gamma number actually predicts the new delta. Push the stock up \$1 to \$101:

- **Delta at \$101:** `delta(101, ...)` = **0.6024**. The actual change in delta is 0.6024 − 0.5343 = **+0.0681**.

Gamma predicted +0.0693; the true change was +0.0681. Nearly identical — the tiny discrepancy is gamma *itself* changing as the stock moved (the curve's bend is not perfectly uniform). For a small \$1 move, the gamma estimate nails it. The intuition: gamma is the dial that told you, before the stock even moved, that your 53-share-equivalent position was about to become a 60-share-equivalent position. You got *longer* as the stock went up — without trading anything.

#### Worked example: the same gamma over a bigger move

The estimate `Δdelta ≈ gamma × ΔS` is only exact for an infinitesimal move; over a big move, curvature compounds. Take the same 30-day ATM call and push the stock up **5%**, from \$100 to \$105:

- **Delta at \$100:** 0.5343.
- **Delta at \$105:** `delta(105, ...)` = **0.8256**. The actual change is **+0.2913**.
- **The naive gamma estimate** says: gamma × ΔS = 0.06932 × 5 = **+0.3466**.

The linear estimate (+0.347) overshoots the true change (+0.291), because gamma itself *falls* as the stock pulls away from the strike — you can see it on the gamma bell, where curvature drops off past \$100. The honest way to say it: over a \$1 move, "delta changes by gamma" is essentially exact; over a \$5 move, it's a first guess that you correct for the fact that gamma is shrinking on the way. Either way, the message is the same — as the stock rose 5%, this call went from behaving like 53 shares to behaving like 83 shares. The position got dramatically longer *because* you were right, and that automatic lengthening, this convexity, is what gamma gives you.

## Where gamma lives: the bell, and how it sharpens into expiry

Gamma is not a single number — it is a *surface* over stock price and time. Two facts organize the whole surface, and both are visible on the cover figure.

**Fact 1: gamma peaks at the money.** An option's curvature is greatest when the stock sits right at the strike, and it fades to near zero deep in-the-money and far out-of-the-money. The reason is intuitive once you see it through delta. Deep ITM, delta is pinned near 1 and barely moves — the option already behaves like stock, so its delta has nowhere to go, hence almost no curvature. Far OTM, delta is pinned near 0 and barely moves — the option is nearly worthless and a small wiggle in the stock changes little. It's right *at* the strike, where the option is poised on the knife edge between "will finish worthless" and "will finish in-the-money," that a small move in the stock swings delta the most. That swinginess is gamma, and it lives at the money.

**Fact 2: near-dated options have more gamma, concentrated more tightly at the money.** The cover figure shows the 7-day bell towering over the 30-day bell — taller and skinnier. With little time left, an at-the-money option's delta is hypersensitive: a tiny move can flip it from "probably finishes worthless" to "probably finishes in-the-money," so delta whipsaws and gamma is enormous. With more time, that same move is a smaller fraction of the stock's plausible range, so delta moves gently and gamma is mild and spread out across a wider band of strikes.

![Delta versus stock price with tangent slopes showing high gamma at the money and low gamma in the wings](/imgs/blogs/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short-3.png)

The figure above makes the "gamma is the slope of delta" definition literal. The blue S-curve is delta as a function of the stock price for our 30-day call. The short straight segments are the *slope* of that curve — which is gamma — drawn at three points. At the money (the amber segment near \$100) the curve is steep: delta is changing fast, gamma is high (0.069). Out in the wings (the gray segments near \$90 and \$110) the curve is nearly flat: delta is barely changing, gamma is low (0.017 and 0.014). The steepness of the delta curve *is* gamma. Where delta is on its hair-trigger, in the middle of the S, gamma is fat; where delta is saturated at 0 or 1, gamma is thin.

#### Worked example: gamma exploding as expiry approaches

Hold the stock at \$100 (right at the strike) and watch the gamma of the at-the-money call as the calendar runs down. Using `gamma(100, 100, T, 0.04, 0.20)` for shrinking `T`:

- **1 year (T = 365/365):** gamma = **0.0191**
- **90 days:** gamma = **0.0397**
- **30 days:** gamma = **0.0693**
- **7 days:** gamma = **0.1439**
- **1 day:** gamma = **0.3810**

Look at the acceleration. From one year to thirty days, gamma roughly tripled. From thirty days to one day, it grew more than fivefold. The mathematical reason is that at-the-money gamma scales like 1 over the square root of time-to-expiry — as `T` shrinks toward zero, `1/√T` blows up. The practical reason is sharper: with one day left, a \$100 stock sitting exactly on a \$100 strike is genuinely poised on a coin's edge. A \$1 move is the difference between the option being a near-certain winner or a near-certain loser, so its delta swings violently from a small move, and a violent swing in delta from a small move is, by definition, gigantic gamma.

![Gamma of an at-the-money option versus days to expiry, exploding vertically in the final week](/imgs/blogs/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short-7.png)

The figure above plots that explosion. Reading left (120 days out) to right (expiry), the gamma of the at-the-money option is a gentle slope for most of the option's life and then turns nearly vertical in the final week (the red zone). This is the single most dangerous feature of options near expiry: the curvature you can mostly ignore for a 90-day option becomes the dominant risk in the last few days of a short-dated one. It is why "0DTE" (zero-days-to-expiry) options behave so wildly, why dealers fear the final hours of a big expiration, and why pin risk — which we'll get to — is a thing. The bell from the cover doesn't just sit there; it *grows a spike* as the clock runs out.

## Long gamma: convexity in your favor

Now we can say precisely why the long-gamma trader from the hook was made whole. **Owning options gives you positive gamma**, and positive gamma is *convexity in your favor* — a structural asymmetry where your position automatically gets more bullish when the stock rises and more bearish when it falls. Your exposure leans into every move that helps you and away from every move that hurts you, without you lifting a finger.

![Option value as a curved line sitting above its straight delta tangent, the gap being convexity](/imgs/blogs/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short-2.png)

This is the picture of convexity. The blue curve is the true value of a long call as the stock moves; the dashed gray line is the *delta tangent* — the straight-line estimate you'd get if you assumed delta stayed fixed at its \$100 value of 0.534. Notice that the true curve sits *above* the straight line on **both** sides. Up at \$110 the option is worth \$10.43, but the delta-only estimate said \$7.79 — the curve is **\$2.64 richer** than the linear guess (the green wedge). Down at \$90 the option is worth \$0.08, but the delta-only estimate said *negative* \$2.89 — the curve is again well above the line. That "always above its own tangent" property is convexity, and it is gamma made visual. When you are long gamma, reality is always at least as kind as the linear approximation, and usually kinder.

Translate that into delta. As the stock rises, your delta rises (you get longer, leaning into the gain). As the stock falls, your delta falls (you get shorter — or less long — cushioning the loss). Your position is forever adjusting itself to be more right and less wrong. That is what the curve bending upward on both sides *means* in trading terms.

### Harvesting the convexity: the rehedge scalp

The convexity is latent in the option, but you can *realize* it as cash by delta-hedging. Here's the mechanism. Suppose you own a long straddle (a call plus a put at the same strike) and you want no directional bet, so you hold an offsetting stock position to keep your total delta near zero. As the stock moves, your option delta changes (that's gamma), so to stay neutral you must rebalance the stock — and here is the magic of being long gamma: **the rebalancing always makes you sell stock high and buy stock low.**

When the stock rallies, your long-gamma delta rises, so to get back to neutral you *sell* shares — at the higher price. When the stock dips, your delta falls, so to get back to neutral you *buy* shares — at the lower price. Sell high, buy low, mechanically, every time you rehedge. That is the long-gamma "scalp," and it is a profit you collect simply for holding convexity and keeping your hedge current.

![Two-column figure showing long gamma rehedging by selling high and buying low, short gamma doing the opposite](/imgs/blogs/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short-4.png)

The left column above traces it: a rally grows your delta, so you sell stock high; a dip shrinks your delta, so you buy stock low; the result is "buy low, sell high" baked into every rehedge, and the price you pay for this lovely machine is theta — the decay rent on the options you own. The right column is the toxic mirror, which we'll get to in a moment.

#### Worked example: a single long-gamma rehedge round-trip

Own one ATM straddle on our \$100 stock (the \$100 call + the \$100 put, 30 days, 20% vol). The straddle's gamma is the sum of the call and put gammas — and since they share a strike, each contributes the same 0.06932, for a **straddle gamma of 0.13864**. Start delta-neutral. Now walk a round trip: the stock rises to \$102, you rehedge, then it falls back to \$100.

- **Stock 100 → 102.** Your straddle gains value and its delta climbs from about +0.069 to about +0.334 — you picked up roughly **+0.27 of delta** just from the move (that's gamma at work). You started neutral by being short 0.069 shares; the straddle's value rose by \$0.407 and your tiny stock hedge lost \$0.137, netting **+\$0.27 per share** so far.
- **Rehedge at \$102.** To get neutral again you sell stock — now short about 0.334 shares — locking your hedge in *at the high*.
- **Stock 102 → 100.** The straddle gives back its \$0.407 of value, but your short 0.334-share hedge, bought back as the stock fell, earns **+\$0.668**. Net for this leg: **+\$0.26 per share.**

Add it up: the full round trip nets about **+\$0.53 per share, or +\$53 per contract** — and the stock ended exactly where it started. You made money on *motion*, not direction. The textbook says a gamma scalp earns about `½ × gamma × (ΔS)²` per rehedge interval; here `½ × 0.13864 × 2² × 2 intervals ≈ \$0.55`, matching our \$0.53 simulation almost exactly. The intuition: being long gamma turns chop into cash, because every wiggle forces you to sell high and buy low.

### The price of convexity: theta

If long gamma were free money, everyone would own it. It isn't free — you pay for it with **theta**, the daily time decay on the options you hold. The two Greeks are bound together: positive gamma comes with negative theta, always. The convexity that lets you scalp the chop is the same optionality that melts a little every day. (The full treatment of theta as a Greek you actively trade is in [Time Value and Theta: Why an Option Is a Melting Ice Cube](/blog/trading/options-volatility/time-value-and-theta-why-an-option-is-a-melting-ice-cube) and the forward post [Theta: Trading the Clock](/blog/trading/options-volatility/theta-trading-the-clock-and-the-price-of-being-long-options); here we just need the trade-off.)

So the long-gamma trader's daily question is: **did the stock move enough that my gamma scalp beat my theta bleed?** There is a break-even move where the two exactly cancel.

#### Worked example: the break-even move (gamma vs theta)

Our ATM straddle has gamma 0.13864 and a theta of about **−\$0.0762 per share per day** (the call and put thetas summed, divided by 365). The gamma scalp from a daily move of size `ΔS` is about `½ × gamma × (ΔS)²`. Set that equal to the day's theta cost and solve:

- Break-even move: `ΔS = √(2 × 0.0762 / 0.13864)` = **\$1.05**, about a **1.05% move**.

If the stock moves *more* than about 1.05% on the day, your scalp gains beat your theta rent and you profit. If it moves *less*, theta wins and you bleed. That 1.05% daily move corresponds to an annualized realized volatility of about `1.05% × √252 ≈ 16.6%`. And here's the punchline: you *paid* for this straddle at an implied volatility of 20%. So you break even if the stock realizes about 16.6% vol — comfortably below the 20% you paid. **You are long gamma at 20% implied; you profit if the world realizes more than ~16.6%, and lose if it realizes less.** That gap between implied and realized is the whole game of being long gamma, and it's why this series insists an option is a bet on volatility, not direction.

![Long-gamma cumulative P&L showing scalp gains outpacing theta bleed over an oscillating path](/imgs/blogs/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short-5.png)

The figure above runs the full simulation: own one ATM straddle, delta-hedge once a day, walk a choppy 10-day path that ends right back at \$100. The green line is the cumulative gamma-scalp gains (sell-high/buy-low rehedges), climbing to about **+\$384 per contract**. The amber line is the cumulative theta bleed, sinking to about **−\$81**. The blue line is the net — about **+\$303 per contract**, earned entirely from chop on a stock that finished exactly where it began. The chop realized more volatility than the 20% you paid, so gamma beat theta and you won. (The dedicated mechanics of doing this as a strategy live in the forward post [Gamma Scalping: Turning a Long Straddle into a Vol Harvest](/blog/trading/options-volatility/gamma-scalping-turning-a-long-straddle-into-a-vol-harvest).)

### Why the scalp is realized volatility, paid in dollars

There's a deeper identity hiding in that simulation, and it's worth surfacing because it is the bridge between gamma and the entire volatility-trading worldview this series is built on. The total profit-and-loss of a delta-hedged long option position, over its life, is approximately:

> **P&L ≈ (the gamma scalp from every move) − (the theta paid every day)**, and that nets out to a bet on **realized volatility minus implied volatility.**

Each day, the scalp you collect is about `½ × gamma × (ΔS)²` — it depends on the *square* of the move, which is exactly what variance is. The theta you pay is, by the no-arbitrage construction of the option's price, calibrated to the *implied* volatility you bought it at. So when the stock's actual squared moves (realized variance) exceed what the option's theta was charging for (implied variance), you profit, and when they fall short, you bleed. The option market sold you a forecast of volatility baked into the premium; delta-hedging strips out the direction and leaves you holding a clean wager on whether the world turns out choppier or calmer than that forecast. This is why practitioners say being long an option and hedging it is "being long realized vol versus implied" — the gamma is the machine that converts each squared price move into cash, and theta is the price of admission.

That identity also explains a counterintuitive fact: **how the stock gets from A to B matters more than where it ends up.** Our simulation ended at \$100, exactly where it started — zero net direction — and still printed +\$303, because the *path* was choppy and every wiggle fed the gamma scalp. A different path that drifted straight from \$100 to \$110 and stopped might realize *less* variance than the violent round trip, and a long-gamma trader would earn less from it despite the bigger net move. Gamma doesn't care about your destination; it cares about how much you wiggled on the way, because variance is built from squared increments.

#### Worked example: the same straddle through a dead, quiet market

To see the other side of the long-gamma bet, replay the 10-day hold on a *boring* tape. Suppose instead of the choppy path, the stock drifts in a tight band — say it never moves more than about \$0.50 on any day, realizing roughly 8% volatility, well under the 20% you paid. Now flip the scalp-versus-theta math:

- **Daily gamma scalp** at a \$0.50 move: `½ × 0.13864 × 0.50² ≈ \$0.017 per share`, about **+\$1.70 per contract per day**.
- **Daily theta cost:** about **−\$7.62 per contract per day** (the straddle's theta we computed earlier).
- **Net per day:** roughly **−\$5.90 per contract**, and over ten quiet days, about **−\$59 per contract** — a steady, grinding loss.

Same straddle, same starting point, opposite outcome: the dead tape realized far less volatility than the 20% you paid, so theta crushed your meager scalp and you bled. The intuition: long gamma is not a free lunch you collect by holding — it is a bet that the market will be *more* volatile than the price implied, and you lose that bet, day after day, in a market that goes nowhere quietly.

### How often should you rehedge?

The scalp lives in the rehedging, which raises a practical question with a surprisingly deep answer: how frequently do you adjust the hedge? There's a genuine trade-off. Rehedge *continuously* (in the idealized theory) and you capture the full `½ × gamma × (ΔS)²` from every infinitesimal wiggle — but in the real world, continuous hedging means infinite trades and infinite transaction costs, which would devour the scalp. Rehedge *too rarely* — say, only once a week — and between adjustments your delta drifts far from neutral, so you're carrying an unintended directional bet that can dwarf the gamma scalp you're trying to harvest. The art is choosing a cadence (or a delta-band trigger, like "rehedge whenever delta drifts past ±0.10") that captures most of the convexity while keeping costs and drift tolerable.

This matters because the *discrete* hedger doesn't capture the smooth theoretical scalp — they capture the realized variance of the actual sampled path, which is noisy. Hedge daily and a stock that gyrated wildly *intraday* but closed flat will look calm to you; you'll miss the intraday chop your daily snapshots never saw. The friction, slippage, and sampling error of real delta-hedging is a rich subject in its own right, treated in [Delta-Hedging in Practice: The Cost and Slippage of Staying Neutral](/blog/trading/options-volatility/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral). For now the takeaway is that the clean `½ × gamma × (ΔS)²` scalp is the *theoretical* prize; what you actually pocket depends on how, and how often, you adjust.

## Short gamma: the toxic position

Now flip every sign. **Selling options gives you negative gamma**, and negative gamma is *convexity against you* — the most dangerous structural feature in all of options trading, and the reason naked option sellers periodically get carried out. The short-gamma trader from the hook didn't get unlucky; she was holding a position that was *designed* to lose catastrophically on a big move, and August 5 simply delivered the move.

Run the rehedge logic in reverse. When you are short gamma, your delta moves *against* you: as the stock rallies, your delta falls (you get shorter into a rising market — exactly the wrong way); as the stock falls, your delta rises (you get longer into a falling market — again the wrong way). To stay neutral you are forced to do the opposite of the long-gamma trader: **buy stock high and sell stock low.** Every rehedge locks in a loss. You are chasing the market, buying after it's already gone up and selling after it's already gone down, bleeding a little on every adjustment.

The right column of the two-column figure above traces it: a rally pushes your delta negative, so you buy stock high to get neutral; a dip pushes your delta positive, so you sell stock low. The result is "buy high, sell low" baked into every rehedge — the exact inverse of the long-gamma machine. The compensation you receive for enduring this is theta: you collected the premium up front, and time decay drips into your account every quiet day. But that compensation is small and bounded, and the loss when a real move comes is large and unbounded.

![Hedged P&L of a short option position versus the size of an overnight gap, a downward parabola](/imgs/blogs/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short-6.png)

This is the picture of negative convexity, and it should be tattooed on the inside of every option seller's eyelids. The red curve is the P&L of a short, delta-hedged at-the-money call as a function of the overnight gap in the stock — a gap you can't hedge through because it happens while you're asleep. The curve is a downward-opening parabola. Right at zero (the stock barely moves overnight), you make a tiny positive amount — your theta credit, about **+\$4.36** for the day. But move in *either* direction and you lose, and the loss accelerates: the curvature works against you, so a bigger gap doesn't just hurt more, it hurts *disproportionately* more. The asymmetry is brutal: a sliver of profit at the top, an ever-steepening cliff on both sides.

#### Worked example: pennies on quiet days, a cliff on the gap

Sell one ATM call (\$100 strike, 30 days, 20% vol). You collect **\$2.4513** in premium (\$245 per contract) and immediately hedge delta-neutral by buying 0.5343 shares against the contract. Now compare the quiet day with the gap.

- **The quiet day (stock flat at \$100).** The call decays by one day of theta. You collect about **+\$0.0436 per share, +\$4.36 per contract**. Pleasant, small, repeatable. Do this 250 trading days a year and it adds up to real money — which is exactly the seduction.
- **The +\$5 overnight gap.** The stock opens at \$105 before you can rehedge. The call you're short jumps from \$2.45 to about \$5.87, a **−\$3.42** hit; your 0.5343-share hedge gains **+\$2.67**; net **−\$0.75 per share, −\$75 per contract.** One gap just wiped out about **17 quiet days** of theta.
- **The +\$10 overnight gap.** The stock opens at \$110. The short call jumps to about \$10.41 (a **−\$7.96** hit), your hedge gains **+\$5.34**, net **−\$2.62 per share, −\$262 per contract.** One gap erased roughly **60 quiet days** of theta — three months of patient premium, gone in one open.
- **The +\$15 overnight gap.** Net **−\$4.86 per share, −\$486 per contract** — about **111 quiet days**, nearly half a year of collected theta, vaporized by a single morning.

The intuition: as a short-gamma seller you are picking up nickels in front of a steamroller. The nickels (theta) are real and steady. The steamroller (a gap) is rare but, because of negative convexity, it doesn't just take back the nickels — it takes a *multiple* of everything you've collected. And note that all of this is the *delta-hedged* loss. The hedge cushions you but cannot save you, because the thing hurting you is the curvature the hedge doesn't capture.

#### Worked example: the naked short and the structural blowup

Now strip away even the hedge, the way a retail seller selling "safe" out-of-the-money calls often does. Sell that same \$100 call naked for \$2.45 and walk away. If the stock gaps to \$120 — an Aug-2024-style or Feb-2018-style shock — your call is now \$20 in-the-money. You sold something for \$2.45 that is now worth \$20. Your loss is **\$20 − \$2.45 = \$17.55 per share, or \$1,755 per contract** — more than *seven times* the premium you collected, on a single contract, from a single gap. Sell ten of them to juice the income and that's **\$17,550** against the \$2,450 you took in.

This is the structural reason naked short options blow up, stated plainly: **your maximum gain is the premium (bounded, small), and your maximum loss is effectively unbounded (the stock can gap arbitrarily far).** Short gamma is the mathematical engine of that asymmetry — it is the curvature that makes the loss accelerate while the gain stays capped. Every famous short-vol disaster, from the funds that imploded in February 2018's "Volmageddon" (the VIX more than doubled in a day, to a 37.32 close) to the carry-trade casualties of August 2024, is, at its core, the same trade: somebody was short gamma, collected the steady premium for a long time, and met the gap that the negative convexity had been promising all along.

### Why the bad day is *worse* than the gamma alone says

There's a cruel compounding that the gap-loss figure understates, because it holds volatility fixed. In a real shock, three things hit the short-gamma seller at once, and they reinforce each other:

- **The gap itself** moves the underlying against your hedged book — that's the gamma loss we computed, the downward parabola.
- **Implied volatility spikes** at the same instant. A seller of options is *short vega* too (vega is the sensitivity to implied volatility), so when fear floods in and IV jumps from 20% to 40%, the options you're short get repriced sharply higher *on top of* the move in the underlying. You're now losing on the spot move and on the vol move simultaneously.
- **Your gamma itself grows** as the move and the falling time-to-expiry sharpen the curvature, so each subsequent rehedge is even more punishing than the last.

This is why a short-vol book doesn't fail gracefully — it fails *convexly*. The gamma loss and the vega loss arrive together because the same event (a crash) that moves the spot is the event that spikes implied vol; they are correlated by construction. A trader who modeled only the delta-hedged gamma loss and felt comfortable with the "−\$262 on a \$10 gap" number can find the real loss is a multiple of that once the vol repricing is added in. The lesson the survivors internalize: when you're short gamma you are also short the very volatility that a gap unleashes, so size for the *combined* blow, not the spot move alone. The vega side of this is its own dimension, but the structural point is that short gamma and short vega travel together, and both bite hardest on the same terrible morning.

#### Worked example: the gamma loss versus the vega loss on the same gap

Stay with the short ATM call, hedged delta-neutral at \$100. We computed that a \$10 overnight gap, *holding implied vol at 20%*, costs about **−\$262 per contract** — pure gamma loss. Now layer on the realistic vol spike. In a shock that gaps the stock 10%, implied vol does not sit still; suppose it jumps from 20% to 35%. Re-price the short call at the new spot *and* the new vol: the call you're short is now worth dramatically more than the 20%-vol reprice, because vega has added a large slug on top. The extra loss from the vol move alone — your short-vega hit — can easily rival or exceed the gamma loss, turning a −\$262 day into a −\$500-or-worse day on the same single contract. The intuition: on a calm day you're short a little gamma and it barely matters; on the one day it matters, you discover you were short gamma *and* vega, and the market collects on both at once.

## Common misconceptions

**Misconception 1: "Gamma only matters near expiry, so I can ignore it on my monthlies."** Half-right, dangerously stated. Gamma is *largest* near expiry, but it is never zero, and on a big move even a 90-day option's gamma moves your delta meaningfully. Our 90-day ATM call has gamma 0.0397 — a 10% move in the stock changes its delta by roughly `0.0397 × 10 ≈ 0.40`, turning a 50-delta position into a 90-delta one. That is not negligible; it's the difference between a hedged book and an accidental directional bet. The honest version: gamma is *manageable* far from expiry and *unmanageable* near it. Don't ignore it; just respect that it grows teeth as the clock runs down. By 7 days the gamma is 0.1439 — 3.6× the 90-day value — and by the final day it's 0.3810.

**Misconception 2: "I'm delta-hedged, so I have no risk."** This is the belief that destroyed the trader in the hook. Delta-hedging removes your *first-order* (linear) risk, but it does nothing about your *second-order* (curvature) risk — gamma. A delta-neutral short-gamma book is neutral only for an instant and only for an infinitesimal move; the moment the stock moves, your delta lurches (that's gamma) and you're no longer neutral, in the wrong direction. Our worked example showed a delta-hedged short call still losing **\$262 per contract** on a \$10 gap. "Delta-neutral" is not "risk-neutral"; it's "neutral to small moves, exposed to big ones," and the size of that exposure is your gamma.

**Misconception 3: "Selling premium is steady income — gamma is just jargon."** The "income" framing hides the negative convexity. Yes, you collect about \$4.36 a day on that short ATM call when nothing happens. But the worked example showed a single \$10 gap costing \$262 — about 60 days of that income — and a naked \$20 gap costing \$1,755, about 400 days of income. The premium is not income; it is *compensation for selling insurance*, and like any insurer you make small steady money right up until the claim arrives. Call it income and you'll size it like income, and sizing a negatively-convex payoff like a steady paycheck is precisely how accounts get to zero.

**Misconception 4: "Long gamma is free money — I just scalp the chop."** No: you pay theta for every day you hold the convexity, and if the stock doesn't move *enough*, theta beats your scalp and you bleed. Our straddle needed about a **1.05% daily move** (≈16.6% realized vol) just to break even against its theta, when we'd paid 20% implied. Long gamma only pays if realized volatility exceeds the implied volatility you bought it at. It's not free money; it's a *bet that the world will be choppier than the option market priced*, and you can absolutely lose that bet by holding convexity through a dead, quiet tape.

**Misconception 5: "Gamma is the same for calls and puts, so it doesn't matter which I trade."** The first clause is true and useful — a call and a put at the same strike, expiry, and vol have *identical* gamma (and identical vega), because gamma depends on the curvature of the value, which is symmetric between the two. That's why a straddle's gamma is just twice the single-leg gamma. But the conclusion is wrong: *which* you trade still matters for delta, for assignment risk, and for the skew you pay (out-of-the-money puts are usually pricier per unit of gamma because of the volatility skew). Same gamma per leg does not mean same trade.

## How it shows up in real markets

**August 5, 2024 — the yen carry unwind.** The setup in the hook was real. After a long, calm summer, an unwinding carry trade and a soft US jobs print collided over a weekend, and global equities gapped down hard at the Monday open; the VIX spiked to a **38.57** intraday print, its biggest one-day jump in years. Short-gamma books — dealers, structured-product desks, and retail premium-sellers alike — were forced to sell into the collapse to rehedge their ballooning negative delta, which mechanically *added* to the selling pressure. This is the dark feedback loop of aggregate short gamma: when the crowd is short gamma and the market falls, their hedging amplifies the fall. The dealer-flow version of this dynamic is its own large subject, covered in [Dealer Gamma, Charm, and Vanna: How Options Flows Move the Spot](/blog/trading/options-volatility/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot).

**February 5, 2018 — "Volmageddon."** The VIX more than doubled in a single session to a 37.32 close, and a family of products that were *structurally short volatility and short gamma* — most infamously an exchange-traded note that sold VIX futures — lost almost all their value overnight, with one notable note effectively terminating. The investors had enjoyed a long, smooth ride collecting the variance risk premium (the same edge that pays option sellers in calm times), and the negative convexity they'd been carrying all along finally delivered its bill in one evening. The lesson traders took: a strategy can look like a low-volatility income machine for years and still be a hidden short-gamma time bomb. The economics of that premium-collection edge are in [Volatility as an Asset: Owning Fear](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear).

**0DTE and the daily gamma circus.** A large fraction of S&P 500 options volume now trades in contracts expiring *the same day*. From the figure of gamma exploding into expiry, you already know why these are wild: a 0DTE at-the-money option has enormous gamma (recall the one-day gamma of 0.38 versus 0.069 at 30 days), so tiny moves in the index produce huge swings in those options' deltas, and the dealers hedging them must trade large amounts of underlying fast. On quiet days this can pin the index near big strikes; on news days it can accelerate moves. The whole 0DTE phenomenon is, mechanically, a story about concentrated near-expiry gamma.

**Pin risk into expiration.** When a heavily-traded stock or index sits very close to a major strike right before expiry, you get *pin risk*: the gamma is so high that the stock gets metaphorically "pinned" to the strike as dealers' hedging flows push back against moves away from it (sell when it ticks up, buy when it ticks down — the long-gamma rehedge, performed in aggregate by the people who are net long that strike). The flip side is that an option holder right at the strike at the close faces real uncertainty about whether they'll be assigned — a few cents either way flips the exercise decision. The strike-level mechanics of being right on the line are built up in [Moneyness and the Strike: ITM, ATM, OTM](/blog/trading/options-volatility/moneyness-and-the-strike-itm-atm-otm-and-what-you-are-really-buying), and the dealer-flow side of pinning is in the dealer-gamma post above.

**Earnings and the long-gamma trap.** A trader who buys a straddle into earnings is buying gamma (and vega) — they want a big move. They often get the move and still lose, because the implied volatility they paid was so inflated by the event that the realized move, however large, didn't beat it (the "vol crush"). This is the long-gamma break-even logic from above, applied to a single event: you can be long gamma, get a 6% earnings move, and still lose because you paid for an 8% move. The event-specific version is in [Event Volatility: Implied vs. Realized and the Vol Crush](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush) and [The Expected Move: Pricing Event Risk with Options](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options).

## The playbook: trading with and against curvature

Everything above collapses into a handful of operating rules. Gamma is not something you "trade" in isolation — it's a property of every options position you hold, and the playbook is about knowing your sign, knowing your worst case, and sizing accordingly.

**Before any options trade, answer: am I long or short gamma?** If you *bought* options (long calls, long puts, straddles, debit spreads), you are long gamma — convexity helps you, chop pays you, and you pay theta for the privilege. If you *sold* options (covered calls, cash-secured puts, credit spreads, condors, naked options), you are short gamma — convexity hurts you, you collect theta, and a gap is your enemy. This sign determines everything that follows, so establish it first, every time.

**If you are long gamma (you bought options):**

- **You need realized volatility to beat what you paid.** Your break-even is the move where the gamma scalp covers the theta — for our ATM straddle, about a 1.05% daily move against the 20% implied you paid. If you expect the stock to chop more than the implied vol suggests, long gamma is your trade. If you expect dead calm, you're paying rent for a machine that won't run.
- **Rehedge to harvest the convexity — but not too often.** Delta-hedging your long-gamma position turns latent convexity into realized cash (sell-high/buy-low). Rehedge too rarely and you leave directional risk on; rehedge too often and transaction costs eat the scalp. The cost-and-slippage mechanics of staying neutral are the subject of [Delta-Hedging in Practice: The Cost and Slippage of Staying Neutral](/blog/trading/options-volatility/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral).
- **Near expiry, your gamma is huge — respect both sides.** Long gamma into expiry means your scalps get juicy (high gamma, big delta swings to harvest) but your theta is brutal (the rent is highest at the end). It's a high-variance window: great if it moves, painful if it pins.

**If you are short gamma (you sold options):**

- **Size for the gap, not the drip.** This is the cardinal rule and the one violated in every blowup. Your daily theta credit is small and seductive; your gap loss is large and accelerating (negative convexity). Set your position size by asking "what's my loss if the underlying gaps 10%, 20% overnight?" — our worked example showed a single \$10 gap costing \$262 against a \$4.36 daily credit. If a plausible overnight gap would do unacceptable damage, you are too big, full stop. The discipline of sizing occasionally-catastrophic payoffs is its own subject — see [Position Sizing and the Kelly Criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion).
- **Prefer defined-risk structures over naked shorts.** A credit spread (sell one option, buy a cheaper one further out) caps your loss and therefore caps your short gamma in the tail — you're still short gamma near the strikes, but the long wing kills the unbounded part. Naked short options keep the unbounded tail; that's what turns a bad week into a terminal one.
- **Buy back or roll before the gamma spike.** The accelerating gamma of the final week is where short sellers get hurt most per dollar of remaining premium. Many premium-sellers harvest the gentle middle of the decay curve (around 30–45 days out) and close or roll before the last gamma-dangerous days, rather than squeezing the final pennies out of a position whose curvature is going vertical.
- **Never read "positive theta" as "safe."** Collecting theta means you are short gamma and usually short vega too — a fast move and a vol spike both hurt you at once, and the worst days deliver both. The steady credit is real; it is also exactly the bait.

**The portfolio view: know your net gamma.** Across a book, gammas add up. A position can be delta-neutral and still carry large net short gamma — the most treacherous combination, because it *looks* hedged and *behaves* like a short-vol bomb. Pull up your net gamma alongside your net delta and ask the same question every day: if the underlying gaps hard tonight, which way does my book lurch, and how much does it cost? If you can answer that in dollars before you go home, you understand gamma better than most people who trade it.

#### Worked example: re-running the hook with the playbook

Replay the short-gamma trader's summer with the playbook in hand. Her mistake wasn't selling premium — the variance risk premium is a real edge, and most quiet days it pays. Her mistakes were three, and each maps to a rule:

1. **She sized for the drip.** Collecting, say, \$4.36 per contract per day across a large book, she sized as if that steady income were the risk. The playbook says size for the gap: had she asked "what does a 10% overnight index gap cost me?" the answer (hundreds of dollars per contract, times her contract count) would have forced a smaller, survivable position. The August 5 gap would have stung instead of ended her.
2. **She sold naked, not spread.** A defined-risk version — selling the put spread instead of the naked put — would have capped her tail loss. She'd still have had a rough Monday, but a bounded one. The unbounded tail is optional; she chose to keep it.
3. **She mistook delta-neutral for safe.** Her book was hedged to small moves and wide open to the big one, because she was net short gamma. The playbook's portfolio rule — read net gamma, not just net delta — would have flagged the coiled spring before the weekend.

The long-gamma trader, by contrast, followed the long-gamma rules almost by accident: he was sized small (long premium can only lose the premium), he held convexity through a choppy-then-violent tape, and when the gap came his curvature paid him. The intuition: the playbook wouldn't have changed *what* either trader believed about the market — it would have changed their *sign awareness and their sizing*, which is exactly where the short seller lost everything and the long holder was made whole.

**The single number to remember:** for our 30-day at-the-money option, gamma is about **0.069**; for the same option with one day left it's about **0.38** — more than five times the curvature, all of it loaded into the final hours. If you can feel that one fact — that curvature is small and slow far from expiry and violent and fast near it — you will never again be surprised by why a "neutral" book detonated on a quiet Monday. Know your sign, know your worst gap, and never be short the bite without respecting the teeth.

## Further reading & cross-links

- **[Delta: Direction, Exposure, and the Hedge Ratio](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio)** — the first-order Greek gamma is the rate of change of; read this first if delta as slope and hedge ratio isn't yet second nature.
- **[Time Value and Theta: Why an Option Is a Melting Ice Cube](/blog/trading/options-volatility/time-value-and-theta-why-an-option-is-a-melting-ice-cube)** — the rent you pay for long gamma (and collect for short gamma); the inseparable trade-off partner to this post.
- **[Theta: Trading the Clock and the Price of Being Long Options](/blog/trading/options-volatility/theta-trading-the-clock-and-the-price-of-being-long-options)** — the full tactical treatment of decay as a Greek you actively manage against your gamma.
- **[Gamma Scalping: Turning a Long Straddle into a Vol Harvest](/blog/trading/options-volatility/gamma-scalping-turning-a-long-straddle-into-a-vol-harvest)** — the dedicated strategy post on realizing long-gamma convexity as cash, building on the rehedge mechanics here.
- **[Delta-Hedging in Practice: The Cost and Slippage of Staying Neutral](/blog/trading/options-volatility/delta-hedging-in-practice-the-cost-and-slippage-of-staying-neutral)** — what it actually costs to keep the hedge current, the friction that eats the scalp.
- **[Dealer Gamma, Charm, and Vanna: How Options Flows Move the Spot](/blog/trading/options-volatility/dealer-gamma-charm-and-vanna-how-options-flows-move-the-spot)** — the market-structure consequence of aggregate gamma: how dealers' hedging of their gamma feeds back into the price.
- **[Moneyness and the Strike: ITM, ATM, OTM](/blog/trading/options-volatility/moneyness-and-the-strike-itm-atm-otm-and-what-you-are-really-buying)** — where gamma peaks and the strike-level mechanics behind pin risk.
- **[Black-Scholes](/blog/trading/quantitative-finance/black-scholes)** — the pricing model these gamma, delta, and value curves are computed from, with the full derivation we deliberately did not repeat here.

---
title: "What Sets an Option's Price? The Five Inputs and the Intuition"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Build intuition for the five inputs that price every option — spot, strike, time, volatility, and rates plus dividends — then watch each one move the price, and see why volatility is the input you can't see."
tags: ["options", "volatility", "black-scholes", "option-pricing", "implied-volatility", "greeks", "put-call-parity", "derivatives", "carry"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — An option's price is set by exactly five inputs (six if you count dividends), and four of them you can read straight off a screen. The fifth — volatility — is the only one you cannot observe, and it is the one that does most of the work. Learning to price options is mostly learning to feel how each input pushes the price.
>
> - The five inputs: **spot price**, **strike**, **time to expiry**, **volatility**, and the **risk-free rate** (plus **dividends** as a sixth). Each pushes a call and a put in a fixed, knowable direction.
> - Four inputs are observable. Only **volatility is an estimate**, so the entire game of options trading is really a game about whether the volatility *priced in* turns out higher or lower than the volatility that *actually shows up*.
> - Move any one input and hold the rest still, and the price traces out a curve. The *slope* of each of those curves has a name — delta, theta, vega, rho — and those are the Greeks we trade in the next track.
> - The one number to remember: at the money, every extra **1 volatility point adds about \$0.20** to a \$100, three-month option. Volatility is not a footnote. It is the price.

A trader once showed me two option quotes side by side and asked which was the better buy. Both were three-month call options. Both were on \$100 stocks. Both were struck at \$100 — exactly at the money. One was offered at \$4.09. The other was offered at \$11.39, nearly three times as much. To the untrained eye they looked identical: same stock price, same strike, same expiry, both calls. So why was one almost triple the price of the other?

It was not a mistake, and it was not because one stock was "better." Every visible number was the same. The single difference lived in a number you cannot see on the chart at all: the market's estimate of how much each stock would *move*. The cheap one was on a sleepy utility the market thought would drift, with volatility priced at 18%. The expensive one was on a biotech the market thought might double or halve on a trial result, with volatility priced at 55%. Same spot, same strike, same clock — but one stock was expected to thrash around three times as violently, and an option is, at its core, a bet on movement. The price difference *was* the volatility difference.

That is the whole story of this post, and in a sense the whole story of the series. An option's price is not a mysterious thing conjured by a formula. It is the output of a small, fixed list of inputs, and the most important of them is the one you can't read off a screen. Before we touch a single equation, we are going to build a physical, everyday feel for each of those inputs — what it is, which way it shoves a call, which way it shoves a put, and *why*. Then we will turn the crank on a real pricing model, vary one input at a time, and watch the price respond. By the end you will be able to look at any option and roughly explain its price out loud. The exact derivation of the pricing formula lives in [the Black-Scholes post](/blog/trading/quantitative-finance/black-scholes); this post is the intuition layer that sits underneath it.

![Five inputs feed a pricing model that outputs a single option price; each input is labeled with its direction of effect on a call and put](/imgs/blogs/what-sets-an-options-price-the-five-inputs-and-the-intuition-1.png)

## Foundations: what an option is, and the inputs that price it

Let's define everything from zero, because the rest of the post leans on it.

An **option** is a contract that gives you the *right, but not the obligation*, to buy or sell a stock (the **underlying**) at a fixed price (the **strike**) on or before a fixed date (**expiry**). A **call** is the right to *buy* at the strike. A **put** is the right to *sell* at the strike. You pay money up front for that right — the **premium** — and that premium is the "price" of the option we are going to dissect.

The word "right, but not the obligation" is the heart of it. If you own a \$100-strike call and the stock finishes at \$130, you exercise: you buy at \$100 and you are instantly \$30 richer per share. If the stock finishes at \$70, you simply walk away — you are not forced to buy a \$70 stock for \$100. You only ever lose the premium you paid. That asymmetry — unlimited upside on a call, capped downside at the premium — is exactly what makes an option valuable, and it is also exactly what makes pricing it interesting. You are paying for the good outcomes while being shielded from the bad ones.

Now, what determines how much that right is worth? It turns out a complete list is surprisingly short. To price a plain European option you need:

1. **The spot price (S)** — what the underlying stock trades at right now.
2. **The strike (K)** — the fixed price written into the contract.
3. **The time to expiry (T)** — how long until the option dies, measured in years.
4. **The volatility (σ, "sigma")** — how much the stock is expected to move, per year, in percentage terms.
5. **The risk-free rate (r)** — the interest rate you could earn on cash with no risk.

And one more, which matters for stocks that pay them:

6. **Dividends (q)** — the cash the stock pays out to holders while you hold the option but don't own the stock.

That's the entire input list. Plug those six numbers into a pricing model and out comes a single fair price for the call and a single fair price for the put. The model most people mean is **Black-Scholes**, and we will use a working implementation of it throughout — but we are deliberately *not* deriving it here. The derivation (where it comes from, why the famous formula has the shape it does, what assumptions it leans on) is a beautiful piece of math that lives in [the Black-Scholes deep-dive](/blog/trading/quantitative-finance/black-scholes) and [the options-theory primer](/blog/trading/quantitative-finance/options-theory). Our job today is to develop a feel for how the *price responds* when you wiggle each input — which, conveniently, is also the path to understanding the Greeks in the next track.

### Intrinsic value and time value: the two pieces of any premium

Before we go input by input, one decomposition makes everything click. The premium of any option splits into two parts:

**Intrinsic value** is what the option would be worth if it expired *right now*. For a call, that's `max(S − K, 0)` — how far in-the-money it is, floored at zero. A \$100-strike call with the stock at \$108 has \$8 of intrinsic value. A \$100-strike call with the stock at \$95 has \$0 of intrinsic value (you'd never exercise the right to buy at \$100 when you can buy at \$95 in the market).

**Time value** (also called **extrinsic value**) is everything else — the part of the premium that exists *because the option hasn't expired yet*. It is the price of all the things that *could still happen* before expiry. It is pure optionality, and it is where volatility and time live. An at-the-money option has zero intrinsic value, so its entire premium is time value.

When we say a call is worth \$4.49, what we are really saying is: \$0.00 of that is intrinsic (it's at the money) and \$4.49 is time value — the market's price for three months of "the stock might rise." Hold that decomposition in your head. Almost every input either changes the intrinsic part or the time-value part, and once you can say which, you can predict the direction of the price move without any arithmetic.

### Why these inputs and not others

It's worth pausing on what's *missing* from the list, because the omissions are as instructive as the inclusions. Notice there's no input for "where I think the stock is going," no input for the company's earnings growth, no input for whether the analyst community is bullish. The pricing model is deliberately *agnostic about direction*. It does not ask whether the stock will rise; it asks only how far it might travel, in either direction, and how the carry tilts the expected endpoint. That's a profound and counterintuitive fact: the fair price of an option does not depend on your forecast of the stock's return. Two traders who violently disagree about whether a stock will go up or down should still agree on the option's price, as long as they agree on its volatility. The reason is the no-arbitrage logic of [risk-neutral pricing](/blog/trading/quantitative-finance/risk-neutral-pricing-martingale-measure-quant-interviews) — a market-maker can hedge away the directional exposure by trading the stock, so what they charge you can't depend on a direction they've neutralized. The full machinery is in the Black-Scholes derivation; the takeaway for intuition is that **an option is a bet on the *magnitude* of movement, not the *sign*.**

The other thing worth flagging: the model assumes a few things to keep the input list this short — that the stock moves smoothly without sudden jumps, that volatility is a single constant number over the option's life, that you can trade and hedge continuously without friction. None of those is exactly true in the real world (stocks gap on news, volatility itself moves around, and trading costs money). The gaps between the model's tidy assumptions and messy reality are where a lot of the *edge* in real options trading hides — but they don't change the directional intuitions we're about to build. The directions are robust; the precise prices are model-dependent. We're after the directions.

## Input 1 — Spot price: the engine under a call

The **spot price** is the underlying stock's current price. It is the most intuitive input, because the option is *about* the stock.

**Direction.** Raise the spot, and a **call gets more valuable** — you have the right to buy at a fixed strike, and the thing you can buy is now worth more. Raise the spot, and a **put gets less valuable** — you have the right to sell at a fixed strike, and the thing you'd be selling is worth more, so the right to dump it at the old strike matters less.

**The everyday analogy.** Think of a call as a *coupon for a fixed-price burger*. The strike is the price on the coupon — say \$10. The spot is what the burger actually costs at the counter today. If burgers are running \$15, your \$10 coupon is worth \$5 in pure intrinsic value, plus a bit more because prices might rise further before the coupon expires. If burgers cost \$6, nobody wants a coupon to pay \$10 — it's worthless intrinsically, though it still carries a sliver of "prices might spike" value. As the burger price climbs, your fixed-price coupon climbs with it. The put is the mirror image: a coupon that lets you *sell* a burger for \$10. The cheaper burgers get, the more that selling-coupon is worth.

This sensitivity of the option price to the spot has a name. The slope of "option price as a function of spot" is **delta**, the first Greek, and we'll devote a whole post to it. For now, just notice that delta for a call is positive (price moves *with* spot) and delta for a put is negative (price moves *against* spot).

#### Worked example: a call's price as the stock climbs

Let's price a 100-strike call with three months to expiry, 20% volatility, a 4% risk-free rate, and no dividends, at three spot prices. Using the Black-Scholes pricer in our `data_options` module — `od.bs_price(S, K=100, T=0.25, r=0.04, sigma=0.20, kind="call")` — we get:

- Stock at **\$95** (below the strike, out-of-the-money): call = **\$2.46**
- Stock at **\$100** (right at the strike, at-the-money): call = **\$4.49**
- Stock at **\$105** (above the strike, in-the-money): call = **\$5.92**

Watch the pattern. Moving the stock up \$5 from \$95 to \$100 added \$2.03 to the call. Moving it up another \$5, from \$100 to \$105, added only \$1.43. The call gets *more responsive* to the stock as it goes in-the-money — but never one-for-one until it's deeply in-the-money, because part of the premium is still time value that doesn't move with spot. **The intuition: a call's price rides up with the stock, but the ride is gentler than the stock's because you're holding a right, not the shares themselves.**

The full curve makes this obvious. Below is the call's price plotted against spot, for two different expiries, with the dashed line showing the option's intrinsic value (its value at expiry).

![Call price versus stock price for a one-month and six-month call, with the intrinsic-value line for reference](/imgs/blogs/what-sets-an-options-price-the-five-inputs-and-the-intuition-2.png)

Notice two things in that chart. First, both curved lines sit *above* the dashed intrinsic line everywhere — that vertical gap is the time value, the price of "the stock might still move in my favor." Second, the longer-dated (six-month) call sits higher than the shorter-dated (one-month) call at every spot, because more time means more chance for the stock to travel. That gap *is* the next input.

## Input 2 — Strike: the line you draw in the sand

The **strike** is the fixed price written into the contract — the price at which a call holder can buy or a put holder can sell. Unlike the spot, you don't observe the strike; you *choose* it when you open the trade. Each available strike is a different contract with a different price.

**Direction.** Raise the strike, and a **call gets cheaper** — you've committed to paying more for the stock if you exercise, so the right is worth less. Raise the strike, and a **put gets more expensive** — you've secured the right to sell at a higher price, which is more valuable.

**The everyday analogy.** The strike is the *deductible on an insurance policy*, run in reverse for the two sides. For a put — which really is insurance on a stock you own — a higher strike is like a *lower* deductible: you're protected closer to today's price, so you pay more for the policy. For a call, the strike is the price you've locked in to buy; the higher you set it, the further the stock has to travel before the right is worth anything, so it's cheaper.

Spot and strike almost always travel together in your thinking, because what really matters is the *relationship* between them — the **moneyness**. A call is **in-the-money** when spot is above strike, **at-the-money** when they're roughly equal, and **out-of-the-money** when spot is below strike. (For a put, flip it.) When you raise the strike with the spot fixed, you're walking the option from in-the-money toward out-of-the-money, and the price falls accordingly. The strike's effect is the mirror image of the spot's — which is exactly why on a pricing screen the call prices fall as you read down the strike ladder while the put prices rise.

#### Worked example: walking up the strike ladder

Hold the stock at \$100, three months out, 20% vol, 4% rate. Price a call at three strikes with `od.bs_price`:

- Strike **\$95** (in-the-money call): call = **\$8.39**, put = **\$2.45**
- Strike **\$100** (at-the-money): call = **\$4.49**, put = **\$3.49**
- Strike **\$110** (out-of-the-money call): call = **\$1.46**, put = **\$10.41**

Read across and the mirror is exact in direction: as the strike rises, the call price collapses (\$8.39 → \$4.49 → \$1.46) while the put price climbs (\$2.45 → \$3.49 → \$10.41). The \$95-strike call costs \$8.39 because \$5 of that is already intrinsic — the stock is \$5 above the strike — and the rest is time value. The \$110 call has zero intrinsic value and is all hope, so it's cheap. **The intuition: the strike is the bar the stock has to clear; set it lower and a call is closer to a sure thing, set it higher and you're buying a lottery ticket.**

## Input 3 — Time to expiry: the melting ice cube

The **time to expiry** is how long the option has left to live, measured in years (so three months is `T = 0.25`). It is observable — it's just the calendar — but it's the input that *only ever runs in one direction*: down. Every day that passes, T shrinks, and the option marches toward its expiry.

**Direction.** More time generally makes **both calls and puts more valuable**. The reason is pure optionality: the longer the option has to live, the more opportunities the stock has to move in your favor, and because your downside is capped at the premium, more "chances to move" can only help you. (The lone wrinkle is deep in-the-money European puts, where the time value of money can make a longer put *slightly* cheaper — a second-order effect we'll wave at later. For everything you'll trade day to day, more time means more value.)

**The everyday analogy.** Time value is a **melting ice cube**. The day you buy the option, the cube is at its biggest — it's full of "anything could happen between now and expiry." Every day, a little melts off. And — this is the cruel part — it doesn't melt at a constant rate. It melts *faster* the closer you get to expiry, the way the last sliver of ice vanishes quickly. At expiry, the cube is gone entirely: time value is exactly zero, and the option is worth only its intrinsic value (what it's in-the-money by), or nothing if it's out-of-the-money.

This decay of time value has a name too. The rate at which the option loses value purely from the clock ticking is **theta**, the Greek of time decay, and it's the headwind every option *buyer* fights and every option *seller* collects. We'll devote a full post to it.

#### Worked example: an at-the-money call melting toward zero

Take a 100-strike call, stock at \$100, 20% vol, 4% rate, and shrink the time to expiry. Pricing each with `od.bs_price`:

- **1 year** to expiry (`T = 1.0`): call = **\$9.93**
- **6 months** (`T = 0.5`): call = **\$6.63**
- **3 months** (`T = 0.25`): call = **\$4.49**
- **1 month** (`T = 0.083`): call = **\$2.47**
- **1 week** (`T = 0.019`): call = **\$1.14**
- **At expiry** (`T = 0`): call = **\$0.00** (it's at-the-money, so zero intrinsic value)

Look at the *acceleration*. Going from 1 year to 6 months — losing six months — cost \$3.30. But going from 1 month to 1 week — losing just three weeks — cost \$1.33. The decay is brutal at the end. A position that was bleeding a couple of cents a day at six months out can bleed twenty cents a day in its final week. **The intuition: time value doesn't trickle out evenly — it pours out in the final stretch, which is why option *buyers* hate holding into expiry and option *sellers* love it.**

Here is the whole decay, for an out-of-the-money, an at-the-money, and an in-the-money call, with time counting down to zero as you read left to right:

![Option price versus time to expiry for three calls, all decaying toward intrinsic value as expiry approaches](/imgs/blogs/what-sets-an-options-price-the-five-inputs-and-the-intuition-3.png)

Every curve in that chart lands on its **intrinsic value** at the right edge, where time runs out. The at-the-money and out-of-the-money calls land on \$0 (no intrinsic value). The in-the-money call lands on \$8 — it's \$8 above the strike, and that \$8 is real value the clock can't take away. Everything *above* those landing points, all the way up the curve, is time value evaporating. The curvature — steepest near expiry — is the accelerating melt.

## Input 4 — Volatility: the star, and the one you can't see

Now we get to the input that does most of the work and earns the series its name. **Volatility** is the expected size of the stock's moves, quantified as the annualized standard deviation of its returns, in percent. A "20% vol" stock is one the market expects to move, in a one-standard-deviation year, by about 20% up or down. A "55% vol" stock is expected to move nearly three times as violently.

**Direction.** Higher volatility makes **both calls and puts more valuable** — unambiguously, always. This is the most important "both up" in all of options. Whether you hold the right to buy or the right to sell, a wilder stock helps you, because your downside is capped at the premium while your upside scales with how far the stock travels. A bigger range of outcomes, with the bad tail chopped off by the cap, is pure gold.

**The everyday analogy.** Volatility is the **wildness of the weather** when you've bought insurance against extreme days. If you own a policy that pays out on any day hotter than 90°F or colder than 20°F, you desperately want a *volatile* climate — wild swings — not a placid, always-72°F one. The placid climate never triggers the payout. The wild one triggers it constantly. An option is exactly that policy: it pays out on the extreme moves, so it's worth more when extreme moves are likely. Buying a call or a put is, fundamentally, **buying volatility**. Selling one is **selling volatility**.

And here is the thing that makes volatility the star of this whole series: **it is the only input you cannot observe.** Spot, strike, time, rate, dividends — you can read every one of them off a screen or a calendar. But nobody can tell you what the stock's volatility *will be* over the next three months. So traders run the logic backwards. They take the option's *actual market price*, and they ask: what volatility number, fed into the model, would *produce* this price? That number is the **implied volatility (IV)** — the volatility the market's price *implies*. It's the market's collective forecast of future movement, expressed as a single number.

That backwards step is the whole game. Implied volatility is a forecast, and forecasts can be wrong. What actually happens — the volatility the stock *delivers* over the option's life — is **realized volatility (RV)**. The entire edge in options trading, the spine of this series, lives in the gap between the two:

> You make money trading the gap between **implied** and **realized** volatility. If you buy an option and the stock moves *more* than its implied vol said it would, you win. If you sell an option and the stock moves *less* than implied, you win. Direction is almost a sideshow; the real bet is whether the priced-in movement is too high or too low.

We'll spend whole posts on the IV-versus-RV theme — the [variance risk premium](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear) (the structural tendency for implied to print *above* realized, which is why selling options has an edge on average), the [vol crush around earnings](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush), and the [vol surface](/blog/trading/quantitative-finance/volatility-surface) (how implied vol differs across strikes and expiries). For now, the headline is simply: of the five inputs, this is the one that is *estimated*, the one that *moves the most* day to day, and the one you are really trading.

The sensitivity of the option price to volatility is the Greek **vega**. And vega has a lovely property at the money: it's nearly constant, which makes the price-versus-vol relationship almost a straight line.

#### Worked example: the price of an extra volatility point

Take our at-the-money 100-strike call, three months out, 4% rate, and crank the volatility:

- **10% vol**: call = **\$2.52**
- **20% vol**: call = **\$4.49**
- **30% vol**: call = **\$6.46**
- **40% vol**: call = **\$8.43**

Every 10-point jump in volatility adds almost exactly the same \$1.97 to the price. That straight-line behavior is the at-the-money vega, and it's why traders quote it as a rule of thumb: at the money, **about \$0.20 of price per 1 volatility point** for a \$100, three-month option. Want the call to be worth \$2 more? You need the market to re-price volatility up by roughly 10 points. **The intuition: at the money, the option price is a near-linear meter that reads out the market's volatility forecast — move the forecast, move the price, almost one-for-one along a fixed slope.**

This is the picture, and it's the single most important chart in the series. Both the call and the put rise nearly in a straight line with implied vol:

![Option price versus implied volatility for an at-the-money call and put, rising in a nearly straight line](/imgs/blogs/what-sets-an-options-price-the-five-inputs-and-the-intuition-4.png)

That near-straight line is the **vega preview**. The slope is vega; we'll spend a whole post turning it into a tradable quantity. And now you can answer the question from the hook precisely: the two calls were identical in spot, strike, and time, but one was priced at 18% vol (\$4.09) and the other at 55% vol (\$11.39). Walk up the line from 18% to 55% and you've added \$7.30 of price — *all of it* volatility. The expensive option wasn't expensive because the stock was special. It was expensive because the market was pricing in three times the movement.

## Input 5 — The risk-free rate: the cost of carry

The **risk-free rate** is the interest you could earn on cash with no risk — think short-term government bills. It feels like it should be a footnote; it's the smallest of the effects in normal markets. But it's real, it's directional, and understanding *why* it moves option prices is the cleanest possible introduction to the idea of **carry** that runs through all of derivatives.

**Direction.** Higher rates make a **call more valuable** and a **put less valuable**. This catches people off guard — why would interest rates have anything to do with a stock option? The answer is about *what owning the option saves you or costs you* compared to owning the stock.

**The everyday analogy.** A call is a way to get the *upside of owning a stock without paying for the stock today*. If you buy a \$100 call instead of \$100 of stock, you've kept your \$100 in the bank, earning interest, while still being exposed to the stock's rise. The higher the interest rate, the more valuable that "I get to keep my cash earning interest" feature is — so the call is worth more. It's like the difference between **buying a house outright versus putting down a deposit and paying the rest later**: when interest rates are high, the option to defer the big payment is worth a lot, because your money earns more while it waits. A put is the reverse: holding a put is a substitute for being *short* the stock, and a short position *earns* you cash to invest; high rates make that alternative more attractive than the put, so the put is worth less.

A cleaner way to see it: the model doesn't really price off today's spot, it prices off the **forward price** — where the stock is expected to be at expiry, which is roughly `S × e^(rT)`. Higher rates push the forward *up*, and a higher forward helps a call (the stock is "expected" higher) and hurts a put. The dividends input, next, pushes the forward the other way.

The sensitivity of the price to the rate is the Greek **rho**. It's the Greek nobody loses sleep over in normal times — for short-dated options it's tiny — but it becomes meaningful for long-dated options (LEAPS, multi-year warrants) and in regimes where rates swing hard.

#### Worked example: rates pulling a call and put apart

Hold spot, strike, time, and vol fixed (\$100, \$100, three months, 20%) and vary only the rate, pricing both legs with `od.bs_price`:

- **0% rate**: call = **\$3.99**, put = **\$3.99** (identical!)
- **2% rate**: call = **\$4.23**, put = **\$3.73**
- **4% rate**: call = **\$4.49**, put = **\$3.49**
- **6% rate**: call = **\$4.75**, put = **\$3.26**
- **8% rate**: call = **\$5.02**, put = **\$3.04**

Two things to savor. First, at a 0% rate the at-the-money call and put are *exactly equal* — there's no carry to tilt them, so the right to buy and the right to sell at the same strike are worth the same. Second, as the rate rises, they fan apart symmetrically: the call gains roughly what the put loses. Going from 4% to 8% lifted the call by \$0.53 and dropped the put by \$0.45. **The intuition: the rate doesn't change *how much movement* you're paying for; it shifts *where the stock is expected to be* at expiry, and that shift helps the buy-side right and hurts the sell-side right.**

Here is the divergence, plotted across a range of rates:

![Call and put prices versus the risk-free rate, the call rising and the put falling as rates increase](/imgs/blogs/what-sets-an-options-price-the-five-inputs-and-the-intuition-5.png)

That diverging fan is the **rho preview**. The call line slopes up, the put line slopes down, and they pivot around the no-carry point. The slope of each line is rho.

## Input 6 — Dividends: carry, but in reverse

**Dividends** are the cash a stock pays out to its shareholders. If you own a call, you *don't* own the stock, so you *don't* collect those dividends — and that's exactly why they matter to your option's price.

**Direction.** Higher dividends make a **call less valuable** and a **put more valuable** — the exact opposite of the risk-free rate. Dividends are a *drag* on the stock's price (the stock typically drops by roughly the dividend amount when it goes ex-dividend), and they're cash you're missing out on by holding the option instead of the shares.

**The everyday analogy.** Going back to the house: dividends are like a **rental income the property throws off that you only collect if you actually own the place, not if you just hold the option to buy it**. The bigger the rent you're forgoing by holding the option instead of the house, the less attractive the option to buy is — and the more attractive the option to sell (the put), because the property's value is leaking out as cash you'd rather not be exposed to on the downside.

Mechanically, dividends push the **forward price down** — the stock is expected to be lower at expiry by the dividends it'll pay out — and a lower forward hurts calls and helps puts, mirror-image to what rates do. This is why, in practice, traders net the two together into the **cost of carry**, `(r − q)`: the rate lifts the forward, dividends drag it down, and what matters is the difference. For a high-dividend stock, the dividend drag can completely swamp the small rate lift — a 5%-yielding utility with a 4% rate has *negative* net carry, which flips the usual call-pricier-than-put relationship on its head.

The mechanism is concrete and dated. A stock that pays a dividend has an **ex-dividend date**: buy the shares before it and you collect the dividend, buy after and you don't. On that morning the stock price *opens lower* by roughly the dividend amount, because the cash has left the company. An options market-maker knows this date and amount in advance, so they bake the expected drop into the forward and therefore into every option's price *today* — the option doesn't suddenly re-price on the ex-date, because the drop was never a surprise. What the call holder loses isn't a shock; it's the steady, anticipated leak of value to shareholders they aren't. The only genuine surprise is when a company *changes* its dividend — a cut or a special dividend re-prices the whole option chain, because it moves an input the model treated as known.

#### Worked example: a dividend haircut on a call

Take our \$100, three-month, 20%-vol, 4%-rate option and add a 3% annual dividend yield, pricing with `od.bs_price(..., q=0.03)`:

- **No dividend** (q = 0%): call = **\$4.49**, put = **\$3.49**
- **3% dividend** (q = 3%): call = **\$4.08**, put = **\$3.83**

Adding the dividend knocked \$0.41 off the call and added \$0.34 to the put. The call holder is being penalized for the cash they won't collect; the put holder is being compensated because the stock will drift lower as it pays out. **The intuition: dividends are negative carry for a call — every dollar the company pays out is a dollar the call holder misses, so the right to buy is worth a little less.** This is also why call holders sometimes *exercise early* on American options right before a juicy dividend — to capture the payout — a wrinkle we'll cover in the early-exercise post.

## Putting it together: comparative statics and the Greeks they become

We've now walked all six inputs. The trick we used each time — *move one input, hold the rest fixed, watch the price* — has a formal name: **comparative statics**. And the slope of the price as you wiggle each input *is* a Greek. That's not a coincidence; the Greeks are literally defined as the partial derivatives of the option price with respect to each input. Everything you just built intuition for becomes a tradable, hedgeable number in the next track.

Here's the whole table, the direction of each effect on a call and a put, and the Greek it foreshadows:

![Comparative-statics grid showing the direction each input moves a call and a put, with the Greek each becomes](/imgs/blogs/what-sets-an-options-price-the-five-inputs-and-the-intuition-6.png)

Let me read that grid out loud, because the pattern is worth memorizing:

- **Spot up** → call **up**, put **down**. This slope is **delta**.
- **Strike up** → call **down**, put **up**. (No standalone Greek — you don't trade the strike; you pick it.)
- **Time up** → both **up** (usually). The *decay* of this is **theta**, and it works against the buyer.
- **Vol up** → both **up**. This slope is **vega** — the star.
- **Rate up** → call **up**, put **down**. This slope is **rho**.
- **Dividend up** → call **down**, put **up**. The dividend version of rho.

Three patterns jump out. First, **spot, rate, and dividends are "directional"** — they help one side of the contract and hurt the other, because they're all really about *where the stock ends up*. Second, **time and volatility are "both up"** — they help *both* calls and puts, because they're about *how much the stock could move*, and more potential movement helps any optionality you hold. That second pattern is the soul of options: you can be long an option and not care which way the stock goes, only *how far*. Third, the two "both up" inputs are the two everyone fights over — time decay (theta) bleeds the buyer every day, and volatility (vega) is the thing the buyer and seller are really disagreeing about.

## A full pricing, start to finish — and put-call parity

Let's do one complete pricing, plug in real numbers, and then verify the whole thing is internally consistent with a beautiful no-arbitrage relationship.

#### Worked example: pricing a call and a put, then proving put-call parity

Take the canonical contract: spot \$100, strike \$100, three months to expiry (T = 0.25), 20% volatility, 4% risk-free rate, no dividends. Price both legs:

- Call: `od.bs_price(100, 100, 0.25, 0.04, 0.20, kind="call")` = **\$4.49**
- Put: `od.bs_price(100, 100, 0.25, 0.04, 0.20, kind="put")` = **\$3.49**

The call is worth more than the put — that's the carry from the 4% rate, exactly as the rate section predicted. Now here's the elegant part. There's a relationship that *must* hold between any call and put with the same strike and expiry, regardless of what volatility is or what the stock does, on pain of free money. It's called **put-call parity**:

```
C - P = S - K * exp(-r * T)
```

In words: the call price minus the put price must equal the spot price minus the *present value* of the strike (the strike discounted back at the risk-free rate). Let's check our numbers:

- Left side: `C − P = 4.49 − 3.49 = ` **\$1.00**
- Right side: `S − K·e^(−rT) = 100 − 100 × e^(−0.04 × 0.25) = 100 − 99.00 = ` **\$1.00**

They match — \$1.00 = \$1.00 (to the penny, \$0.995 before rounding). **The intuition: a call minus a put is, in payoff terms, just owning the stock with the purchase deferred to expiry — so it must be worth the stock today minus the discounted cost of that deferred purchase. The \$1.00 gap is purely the carry on the strike.** This relationship is *model-free*: it doesn't care whether Black-Scholes is right; it holds by pure arbitrage. (The full proof — how you'd construct a risk-free profit if it *didn't* hold — is in [the put-call parity post](/blog/trading/quantitative-finance/put-call-parity-no-arbitrage).)

And the relationship doesn't just hold at the money — it holds at *every* strike, which is a strong consistency check on any pricer. Here's `C − P` plotted against `S − K·e^(−rT)` across a ladder of strikes; every point lands exactly on the 45-degree line:

![Put-call parity check: call minus put plotted against stock minus discounted strike, all points on the 45-degree line](/imgs/blogs/what-sets-an-options-price-the-five-inputs-and-the-intuition-7.png)

Every dot on that line is a different strike, and every one satisfies the relationship perfectly. When you see a real options chain where this *doesn't* quite hold, you're either looking at stale quotes, hard-to-borrow stock (which messes with the carry), or a genuine fleeting arbitrage that market-makers will close in seconds. Parity is one of the most reliable sanity checks in all of trading.

## The inputs don't move alone: second-order intuition

So far we've moved one input at a time, which is the right way to *learn* the directions. But the deeper intuition — the part that separates someone who can recite the table from someone who can actually feel a price — is understanding how the inputs *interact*. The sensitivity to one input is itself a function of the others. This is exactly why the second-order Greeks exist, and a little intuition for them now will pay off across the whole series.

**Volatility and time are really one input wearing two hats.** Look back at the time-decay chart and the volatility chart. They're both pricing the same thing: the *total amount of movement the stock can do before expiry*. More volatility means bigger steps; more time means more steps. What actually drives the time value is the product — roughly, `σ × √T`, the expected total travel. This is why a 30-day option at 40% vol can be worth almost the same as a 60-day option at 28% vol: the "movement budget" is similar. It's also why, as expiry nears, the option becomes *less* sensitive to volatility — there's no time left for the vol to express itself, so a vol change moves the price less. Vega shrinks toward zero as `T → 0`. A trader who buys a high-vol option a day before expiry is paying for movement that has almost no time to happen.

**The spot sensitivity (delta) is biggest where the option is most uncertain.** A deep in-the-money call moves almost dollar-for-dollar with the stock (delta near 1.00) — it's basically a stock substitute. A deep out-of-the-money call barely moves (delta near 0) — the stock is so far from the strike that a small wiggle changes nothing. The action is *at the money*, where a small move in the stock can flip the option from worthless to valuable. That's where the price is most *curved* with respect to spot — the rate at which delta itself changes is **gamma**, and it peaks at the money and near expiry. The pin-risk and 0DTE dramas you'll read about later are all gamma stories: tiny stock moves causing huge swings in delta right at the strike, right at expiry.

**Carry's importance scales with everything else's irrelevance.** The rate and dividend effects looked small in our three-month examples — fractions of a dollar. But stretch the option to two years and the discounting term `e^(−rT)` has four times the runway to bite; rho grows roughly with maturity. This is the general shape of second-order thinking: every input's *importance* depends on the regime. In a calm, low-rate, short-dated world, volatility is almost the only thing that matters and you can nearly ignore rho. In a high-rate world of long-dated index options, carry climbs back onto the stage. The skilled trader doesn't memorize "rho is small" — they know *when* it stops being small.

#### Worked example: why a longer option is less vol-sensitive per day

Compare two at-the-money calls on a \$100 stock at 20% vol and a 4% rate. The one-month call (`T = 0.083`) prices at **\$2.47**; the six-month call (`T = 0.5`) prices at **\$6.63**. Now bump volatility from 20% to 21% on each. The one-month call rises to about **\$2.58** — up roughly \$0.12 for the vol point. The six-month call rises to about **\$6.90** — up roughly \$0.28 for the same vol point. **The intuition: the same one-point change in volatility is worth more than twice as much to the longer option, because there's more than twice the time for that extra volatility to actually move the stock — vega grows with the square root of time, so longer-dated options are the natural home for a pure volatility view.**

This interaction view is why the Greeks are not just five separate dials but a coupled system — change the spot and your vega changes, let time pass and your delta and gamma shift, and a vol spike re-prices everything at once. The next track takes each Greek and makes it a live, hedgeable number; everything there is built on the directions and the interactions you just internalized here.

## Common misconceptions

A handful of beliefs about option pricing are not just wrong but *expensively* wrong. Each one is corrected with a number.

**Misconception 1: "This option is expensive because the stock is expensive."** No. The stock's *level* is almost irrelevant to whether an option is "expensive" — what matters is the *volatility* priced in. Our two hook options were on \$100 stocks with \$100 strikes, identical in every observable way, yet one was \$4.09 and the other \$11.39. The \$100 spot told you nothing. The 18%-vs-55% implied vol told you everything. A \$10 option on a \$50 stock can be far "cheaper" (in vol terms) than a \$2 option on a \$30 stock. **Price in dollars is not the same as price in volatility, and only the latter tells you whether you're overpaying.**

**Misconception 2: "Buying out-of-the-money calls is cheap leverage."** They're cheap in *dollars*, not cheap in *odds*. That \$110-strike call we priced at \$1.46 looks like a steal next to \$100 of stock — but it's all time value, and it bleeds. The at-the-money call we priced loses about \$0.03 a day to theta at the start; an out-of-the-money call loses a larger *fraction* of its value daily and dies worthless unless the stock makes a real move *before expiry*. The lottery-ticket calls that look cheapest are the ones with the worst expected value, because you're paying for volatility *and* fighting time decay *and* needing the move to happen on a deadline. **"Cheap" in dollars often means "low probability" in disguise.**

**Misconception 3: "Higher implied volatility means the option is overpriced."** Not necessarily — high IV might be *correct*. A biotech into a binary trial *should* have 55% implied vol; the stock really might double or halve. The question is never "is IV high?" but "is IV higher than the volatility that will actually realize?" That's the implied-versus-realized gap. An option at 55% IV is a *bargain* if the stock is about to move 70%, and a *rip-off* if it only moves 30%. **You can't judge an option's value from its IV alone — only from IV relative to your forecast of realized vol.**

**Misconception 4: "Time decay hurts me evenly, so I have plenty of time."** No — theta accelerates. Our at-the-money call lost \$3.30 over the first six months but \$1.33 in just the final three weeks. If you're long an option, the last third of its life is where the bleeding is worst. **An option held to expiry doesn't melt linearly; it falls off a cliff at the end, which is why long-option traders take profits or roll well before expiry.**

**Misconception 5: "Interest rates and dividends don't matter for options."** They're small for short-dated options, but they're not zero, and they're *directional*. Our 4%-rate call was worth \$0.50 more than the same call at 0%. Add a 3% dividend and the call drops \$0.41. For a one-year LEAPS or a high-dividend stock, those effects compound into real money, and ignoring them is how you misprice a long-dated position or get surprised by an early-exercise assignment right before an ex-dividend date. **Carry is a footnote on a weekly option and a headline on a two-year one.**

## How it shows up in real markets

The clean comparative statics above are exactly what's happening, in real time, on every options screen — they just don't show up one input at a time. In the wild, several inputs move at once, and learning to *attribute* a price change to the right input is the core skill of an options trader.

**Earnings, where volatility moves on a schedule.** The most vivid real-market example of the volatility input is the **earnings vol crush**. In the days before a company reports, its implied volatility ramps up — the market knows a big move is coming and prices it in, so options get expensive in *vol* terms even if the stock barely moves. Then, the morning after earnings, the uncertainty resolves: the number is out, the big move (or non-move) has happened, and implied vol *collapses* — often from 60% back to 30% overnight. Walk that down our vega chart and you can see the damage: a 30-point IV drop on a \$100 stock vaporizes roughly \$6 of an at-the-money option's value *instantly*, regardless of which way the stock went. Traders who bought calls "to play the earnings beat" routinely watch the stock rise on a good number and their calls *lose money anyway*, because the vol crush overwhelmed the directional gain. This is the IV-versus-RV story made painfully concrete, and it's covered in depth in [the event-volatility post](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush) and [the expected-move post](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options).

**The vol skew, where strike and volatility interact.** On a real chain, you'll notice that out-of-the-money puts carry *higher* implied vol than at-the-money or out-of-the-money calls — the so-called **skew** or **smirk**. That's the market pricing in the empirical fact that stocks crash down faster than they melt up, so downside insurance (puts) is bid. It means the "one volatility number" we used all post is really a *surface* of vol that varies by strike and expiry. The intuition you built — higher vol means a higher price — still holds at each strike; there's just a different vol at each one. [The volatility-surface post](/blog/trading/quantitative-finance/volatility-surface) treats this as a no-arbitrage object.

**Rate regimes, where rho wakes up.** For most of the 2010s, with rates near zero, rho was genuinely ignorable — the 0%-rate case where calls and puts price almost equally. But in 2022–2023, as policy rates jumped from near-zero to over 5%, the carry term suddenly mattered, especially for long-dated index options. A two-year call's price has meaningful rho; the move in rates re-priced those options by amounts large enough that desks that had been ignoring rho got a rude reminder. The lesson: the "small" inputs aren't permanently small — they scale with the rate level and the option's maturity.

**The variance risk premium, where the edge lives.** Across decades, S&P 500 implied volatility has tended to print *above* the volatility that subsequently realized — historically by roughly 3 to 4 volatility points on average (implied around 19.5, realized around 15.8 over long samples). That structural gap is the **variance risk premium**: option *sellers* get paid, on average, for providing insurance, the same way insurance companies profit on average from selling policies. It's why systematic option-selling strategies have a real edge — and why they blow up spectacularly when realized vol spikes past implied, as it did in February 2018's "Volmageddon" and March 2020's COVID crash. Selling vol is picking up pennies that usually keep coming, in front of an occasional steamroller. We treat this as a tradable asset class in [the volatility-as-an-asset post](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear).

The throughline: in real markets you are never moving one input at a time. The stock ticks (spot), the clock runs (time), and — dominating both — the market constantly re-prices its volatility forecast (vega). Becoming a competent options trader is learning to *decompose* a price change back into those moving parts, and to recognize that most of the day-to-day P&L noise in an option is the vol input being re-marked, not the stock moving.

## The playbook: how to think about an option's price

You came here to build intuition; here's how to *use* it. This is the checklist to run on any option before you trade it.

**1. Read the four observable inputs off the screen first, in order.** Spot relative to strike (the moneyness — is it in, at, or out of the money?), time to expiry (how many days of melting ice cube are left?), and then mentally note the rate and dividend (small, but know their direction). These four are facts, not opinions. They tell you the option's intrinsic value instantly and roughly how fast it'll decay.

**2. Then back out the one input you can't see — the implied volatility.** Every broker shows it. This is the market's forecast, and it's the number you're actually trading. Compare it to the stock's recent *realized* volatility and to its own history. Is the option pricing in more movement than the stock has been delivering, or less? That comparison — not the dollar price — is your entry signal.

**3. Frame the trade as a volatility view, not just a direction.** "I think the stock goes up" is a weak reason to buy a call, because you can be right on direction and still lose to vol crush and theta. The stronger frame: "implied vol is 25% but I think this stock realizes 40% over the next month" → you want to be *long* options (long vega), and you'll make money on the movement regardless of which way it breaks. "Implied vol is 50% into earnings but I think the actual move will be small" → you want to be *short* options (short vega), collecting the rich premium. Direction is a separate, smaller bet you layer on top with your choice of call versus put versus spread.

**4. Know which Greek you're long or short before you click.** Long any option = long vega (you win if vol rises), long gamma (you win on movement), and *short* theta (you bleed every day). Short any option = the mirror: you collect theta daily, but you're short vega and gamma — one fast move can hurt you far more than the premium you collected. The comparative-statics grid you memorized *is* your Greek exposure map. The full mechanics of trading each Greek live in the next track of this series.

**5. Size for the input you can't control.** Because volatility is the dominant, unobservable input, it's also the one that surprises you. Long-option positions can decay to zero (your max loss is the premium — survivable). Short-option positions have the premium as your max *gain* and a potentially much larger loss — so size them as if realized vol will, occasionally, blow through implied. The variance risk premium says selling vol pays on average; the blow-up history says it pays until it very suddenly doesn't. Position sizing is its own discipline — see [the position-sizing and Kelly post](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) — and it matters most precisely because the star input is the one you can't see.

**The invalidation.** Your view is wrong, and you should be out, when the implied-versus-realized gap closes against you: you bought options expecting big moves and the stock goes quiet (realized vol falls below implied, theta wins), or you sold options expecting calm and the stock erupts (realized blows past implied, your short vega bleeds). The single sentence to carry out of this post: **an option's price is five visible inputs and one invisible one, and the invisible one — volatility — is both the most important and the one you're really betting on.** Everything else in this series is built on that.

## Further reading & cross-links

- [Black-Scholes, derived](/blog/trading/quantitative-finance/black-scholes) — where the pricing formula we used all post actually comes from, and why it has the shape it does.
- [Option pricing fundamentals](/blog/trading/quantitative-finance/options-theory) — the theory layer beneath this intuition layer.
- [Put-call parity and no-arbitrage](/blog/trading/quantitative-finance/put-call-parity-no-arbitrage) — the full proof of the `C − P = S − K·e^(−rT)` relationship we verified.
- [The volatility surface](/blog/trading/quantitative-finance/volatility-surface) — why "one volatility number" is really a surface that varies by strike and expiry.
- [Event volatility: implied vs realized and the vol crush](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush) — the IV-vs-RV theme made concrete around earnings.
- [The expected move: pricing event risk with options](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options) — how the volatility input encodes the market's expected move.
- [Volatility as an asset: owning fear](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear) — the variance risk premium as a tradable edge.
- [Position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) — sizing for the input you can't control.

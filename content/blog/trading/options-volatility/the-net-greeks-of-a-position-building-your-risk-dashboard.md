---
title: "The Net Greeks of a Position: Building Your Risk Dashboard"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn how a trader reads a whole position or book through its net Greeks, converts them to dollar-Greeks, runs scenario shocks, and budgets which risks to carry instead of trying to zero them all."
tags: ["options", "volatility", "options-greeks", "net-greeks", "dollar-greeks", "risk-management", "delta", "gamma", "theta", "vega", "scenario-analysis", "position-sizing"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A single option has Greeks; a *position* has *net* Greeks, the simple sum of every leg's delta, gamma, theta, and vega, and a trader reads risk through that one dashboard rather than through any single contract.
>
> - **Net Greeks add.** Sum each leg's delta/gamma/theta/vega (with the right long/short sign), then convert to *dollar*-Greeks: dollar-delta (P&L per 1% spot move), dollar-gamma, dollar-vega (per vol point), and theta-per-day. The book's risk is the sum, not the scariest-looking line.
> - **Every structure has a Greek fingerprint.** A long straddle is +gamma / +vega / −theta; a short iron condor is −gamma / −vega / +theta; a covered call is +delta / −gamma / −vega / +theta. Learn to read the structure from its signs.
> - **You cannot zero every Greek.** Trading is choosing *which* risk to carry. Hedge what you didn't mean to hold, keep the one that is your view, and size the residual to a pre-set Greek budget.
> - **The one habit to build:** before every trade, ask what it does to the *book's* net Greeks — not whether the single position looks fine on its own. A delta-neutral book can still be massively short vega and lose a fortune the day implied vol spikes.

## A book that was short the one thing nobody was watching

A trader I'll call Marcus ran a small "income" options book — nothing exotic, just the bread-and-butter premium-selling structures that look sensible one position at a time. He was short some at-the-money straddles on an index-like name sitting at \$100, short a batch of out-of-the-money calls against a long stock holding, and short a stack of cash-secured puts a little below the market. Each trade, examined alone, had a tidy logic. The straddles harvested theta. The calls were "just an overwrite" on shares he was happy to sell higher. The puts were "just getting paid to maybe buy the stock cheaper." Every line item passed the smell test in isolation.

What Marcus never did was add them up. He watched each position on its own little screen — its own profit-and-loss, its own delta, its own days-to-expiry — and felt diversified because he had a dozen different tickers and strikes on. He even delta-hedged: he'd glance at each position's delta, buy or sell a few shares here and there, and tell himself the book was "market-neutral." On any quiet afternoon, that was true. The book made a steady few hundred dollars a day in decay, the delta was a rounding error, and Marcus slept fine.

Then implied volatility spiked. Not a crash — just one of those ordinary fear-flares where some headline hits, the index barely moves, and the VIX jumps eight points in an afternoon. Marcus's spot was flat. His *delta* was flat. And his book lost nearly **\$10,000 in a few hours**, with the underlying basically unchanged. He had been so focused on the one Greek he could see on each screen — delta — that he never measured the Greek that actually defined his book. Summed across every leg, his position was short roughly **\$1,163 of vega for every single point** implied vol could rise. He wasn't running an income book. He was running a giant short-volatility bet wearing an income book's clothes, and he found out the size of it only when the market sent him the invoice.

This post is about the dashboard Marcus didn't build. A single option is read through its Greeks; a *whole position or book* is read through its **net Greeks** — the sum of every leg's sensitivities — and that sum is where the real risk lives. We'll learn to add the Greeks up with the right signs, turn them into dollars, recognize every common structure by its Greek "fingerprint," scan a book the way a professional risk desk does (scenario shocks, not gut feel), and confront the central truth that Marcus missed: you cannot set every Greek to zero. Trading *is* choosing which risks to carry.

![A position risk dashboard showing net dollar-delta, dollar-gamma, dollar-vega, and theta of a small book as labeled bars](/imgs/blogs/the-net-greeks-of-a-position-building-your-risk-dashboard-1.png)

The figure above is the dashboard Marcus needed and didn't have — the single screen that would have screamed at him. Four bars, one for each net dollar-Greek of his book. Dollar-delta sits at about zero: he *was* delta-hedged, that part was real. But look at the rest. Dollar-gamma is deeply negative (−\$735), dollar-vega is the longest red bar on the board (−\$1,163 per vol point), and theta is the lone green bar (+\$326 a day). That combination — short vega, short gamma, long theta — has a name: it's a short-volatility book, and its whole character is "collect a little every day, lose a lot when the world moves or gets scared." Everything in this article is in service of being able to read that one dashboard at a glance, for any position you'll ever put on.

## Foundations: what a "net Greek" is and why it's just a sum

Let's build this from the ground up, assuming only that you've met the individual Greeks before (and if any feel shaky, each has a dedicated post linked at the end). We'll start with the single most important — and most reassuring — fact in all of position risk management.

### The Greeks are additive

A **Greek** is a *sensitivity*: a number that tells you how much an option's price changes when one input moves a little, holding everything else fixed. Delta is the sensitivity to the underlying stock price. Gamma is the sensitivity of delta itself to the stock price (the curvature). Theta is the sensitivity to the passage of time. Vega is the sensitivity to implied volatility. Each is a derivative — literally, in the calculus sense — of the option's value with respect to one variable.

Here is the property that makes a whole-position dashboard even possible: **the derivative of a sum is the sum of the derivatives.** If your position is leg A plus leg B plus leg C, then the value of the position is `V = V_A + V_B + V_C`, and the position's delta is just `delta_A + delta_B + delta_C`. The same is true for gamma, theta, and vega. There is no interaction term, no cross-correction, no clever blending. You add them up. That's it.

> **Net Greek of a position = the sum, across every leg, of (that leg's Greek) × (its quantity) × (+1 if long, −1 if short).**

This is worth pausing on because it is genuinely a gift. It means a book of fifty positions across twenty tickers collapses, for first-order risk purposes, into *four numbers* per underlying: net delta, net gamma, net theta, net vega. (Plus rho, which we mostly set aside; in low-rate-sensitivity equity options it's a minor character, and the second-order Greeks like vanna and volga refine the picture but don't change the additivity. Both are covered in the [rho and second-order Greeks](/blog/trading/options-volatility/rho-dividends-and-the-second-order-greeks-vanna-volga-charm) post.) A human cannot hold fifty positions in their head. A human can absolutely hold four numbers.

### The sign convention: long, short, calls, puts

The only thing you can get wrong when summing Greeks is the *sign*, and getting it wrong is exactly how people blow up. So let's nail the conventions cold. We always quote a Greek for a *single long contract* first, then flip the sign if we're short.

For a **long call**, the Greek signs are:

- **Delta: positive.** The stock going up helps you (you own the right to buy low).
- **Gamma: positive.** As the stock rises, your delta grows — gains accelerate, losses decelerate.
- **Theta: negative.** Time decay bleeds you; the option is a melting asset.
- **Vega: positive.** Rising implied vol fattens the premium and helps you.

For a **long put**:

- **Delta: negative.** The stock going *down* helps you (you own the right to sell high).
- **Gamma: positive.** Same curvature benefit — long options of either type are long gamma.
- **Theta: negative.** Still a melting asset; you still pay rent for time.
- **Vega: positive.** Still long volatility; rising IV still helps.

Notice the symmetry: **anything you are long gives you positive gamma and positive vega and negative theta.** Calls and puts differ only in the *sign of delta*. Gamma, vega, and theta don't care whether it's a call or a put — they care whether you're long or short the *optionality*.

And **shorting flips every sign.** Short a call and your delta is negative, gamma negative, theta positive, vega negative. Short a put and your delta is positive, gamma negative, theta positive, vega negative. The rule of thumb that falls out of this is the single most useful sentence in the whole subject:

> **Long options → long gamma, long vega, short theta (you pay to own convexity). Short options → short gamma, short vega, long theta (you collect to sell convexity).** Direction (delta) is the only thing the call/put choice sets independently.

Hold that, and you can sign any leg in your head. Owning a put? Short theta, long gamma, long vega, short delta. Selling a put? The exact opposites. There is no fifth case to memorize.

### Dollar-Greeks: turning sensitivities into money

The raw Greeks from a pricing model come in awkward units. Delta is "change in option price per \$1 change in the stock." Gamma is "change in delta per \$1 of stock." Vega, in this series' pricer, is "change in price per 1.00 change in sigma" (i.e., per 100 vol points), which we divide by 100 to get the trader-friendly per-one-vol-point number. Theta is per year, which we divide by 365 for the per-day bleed. None of those are dollars of P&L on your actual position. To run a book, you convert each to a **dollar-Greek** — how many real dollars you make or lose for a *standard-sized move* in the relevant variable.

The conversions, for equity options with the standard 100-share multiplier:

- **Dollar-delta** = net delta × multiplier × spot × 1%. This answers "how many dollars do I make or lose if the stock moves 1%?" A net delta of +0.50 on one contract at a \$100 stock is `0.50 × 100 × 100 × 0.01 = +\$50` per 1% up-move. (Some desks quote dollar-delta per \$1 move instead, which is just net delta × multiplier; pick one convention and stick to it. We'll use per-1% throughout, because comparing risk across a \$30 stock and a \$400 stock only makes sense in percentage terms.)
- **Dollar-gamma** = net gamma × multiplier × (spot × 1%). This is "how much does my dollar-delta *change* for a 1% move?" — the curvature, in dollars. It's what tells you whether your delta hedge will still be a good hedge after the stock moves.
- **Dollar-vega** = net vega-per-point × multiplier. "How many dollars do I make or lose if implied vol moves one point?" This is the number Marcus never looked at.
- **Theta-per-day** = net theta-per-year ÷ 365 × multiplier. "How many dollars does time hand me (or take from me) overnight, with everything else still?"

The reason we bother is that raw Greeks lie about *relative* size. A net vega of 0.23 and a net theta of −0.076 look comparable as bare numbers. But once you convert — and especially once you scale by how *big* a realistic move in each variable is — you discover that the vega risk dwarfs the theta. The dollar-Greek dashboard puts every risk in the same currency so you can actually compare apples to apples: dollars of P&L.

#### Worked example: summing the net Greeks of a long straddle

Let's do the simplest multi-leg position from scratch. A **long straddle** is a long call plus a long put at the same strike and expiry — the classic "I think it's going to move, I don't know which way" trade. Take a \$100 stock, the \$100 strike, 30 days to expiry, 20% implied vol, 4% rate. Using this series' Black-Scholes pricer, here are the two legs:

- **Long \$100 call:** price \$2.4513, delta +0.5343, gamma 0.06932, vega +0.1140 per point, theta −\$0.0436/day.
- **Long \$100 put:** price \$2.1230, delta −0.4657, gamma 0.06932, vega +0.1140 per point, theta −\$0.0326/day.

Now sum, leg by leg (both legs are long, so every sign stays as-is):

- **Net premium (cost):** 2.4513 + 2.1230 = **\$4.5743** per share → **\$457.43** to buy one straddle (one call + one put, ×100).
- **Net delta:** +0.5343 + (−0.4657) = **+0.0685**. Almost flat — the call's positive delta nearly cancels the put's negative delta. A straddle at-the-money is roughly delta-neutral, which is the whole point: it's a pure bet on *movement*, not direction.
- **Net gamma:** 0.06932 + 0.06932 = **0.13864**. Strongly positive — you're *long gamma* on both legs.
- **Net vega:** +0.1140 + 0.1140 = **+0.2279** per point. Strongly positive — *long vega*.
- **Net theta:** −0.0436 + (−0.0326) = **−0.0762** per day. Negative on both legs — you *pay theta*.

Convert to dollar-Greeks (×100 multiplier, spot \$100): dollar-delta is about `0.0685 × 100 × 100 × 0.01 = +\$6.85` per 1% — negligible. Dollar-vega is `0.2279 × 100 = +\$22.79` per vol point. Theta-per-day is `−0.0762 × 100 = −\$7.62`. So the dashboard reads: roughly delta-flat, long \$22.79 of vega per point, paying \$7.62 a day in decay, and (from the gamma) your delta swings by about `0.13864 × 100 × 1 = +13.9` shares-equivalent for every \$1 the stock moves.

The intuition: a long straddle's Greek fingerprint is **+gamma, +vega, −theta, ≈0 delta** — it is the purest expression of "long volatility." You're paying \$7.62 a day (theta) to own the right to profit from movement (gamma) and from rising fear (vega). It is the mirror image of everything Marcus was short.

## The risk graph: the picture behind the net Greeks

The net Greeks are the *local* description of a position — they tell you how P&L behaves for *small* moves, right where the stock sits now. To see the whole behavior, across every possible spot price, you draw the **risk graph** (also called the payoff diagram or P&L profile): the position's profit-and-loss on the vertical axis against the underlying price on the horizontal axis. And crucially, you draw it at *two dates*: today (with time value still in the options) and at expiry (intrinsic value only). The gap between the two curves is the time value you still own or owe.

You build the risk graph the same way you build the net Greeks — by summing the legs. At each candidate spot price, you reprice every leg with the pricing model (for the "now" curve) or with its expiry intrinsic value (for the "expiry" curve), apply the long/short sign, add them up, and subtract what you paid (or add what you collected). The resulting curve *is* the position, and its local slope at today's spot is the net delta, its curvature is the net gamma, the vertical gap between the two-date curves encodes the theta you'll bleed or bank, and the way the whole picture shifts if you re-draw it at a higher vol is the vega.

![Risk graph showing a long-strangle position profit and loss versus spot price at two dates, now and at expiry](/imgs/blogs/the-net-greeks-of-a-position-building-your-risk-dashboard-2.png)

The figure plots a slightly more interesting position than a plain straddle so the shape is vivid: a **long strangle with a directional tilt** — long two \$105 calls and one \$95 put, 45 days out, 20% vol, on the \$100 stock. The blue curve is the position's P&L *today* (45 days left); the gray curve is its P&L *at expiry*. Both are simply the sum of the three legs' values, leg by leg through the pricer, minus the \$314.50 net debit paid.

Read the picture the way the Greeks describe it. Near the strikes the expiry curve dips into a loss valley (the red-shaded region) — at expiry, if the stock just sits between 95 and 105, the options expire near-worthless and you lose most of the premium. But the curve turns sharply upward in both wings (the green-shaded profit regions): a big move either way prints, and because there are *two* calls versus one put, the right wing climbs faster — the position is tilted bullish. The today-curve sits *above* the expiry curve almost everywhere, and that vertical gap is the time value you still own. As the days pass with the stock flat, the blue curve sinks toward the gray one — that downward drift *is* your negative theta made visible. The convex, smile-like shape of the curve *is* your positive gamma. And if you re-drew the whole blue curve at 25% vol instead of 20%, it would lift up off the page — that lift is your positive vega.

This is the deep link between the two ways of seeing a position: **the risk graph is the global view; the net Greeks are the local view.** The Greeks are the slope, curvature, and shift of the graph evaluated at today's spot. If you understand that one curve and the four numbers that describe its behavior at the current price, you understand the position completely.

### Net delta is itself a function of spot

Here's a subtlety that trips up beginners and is obvious to professionals: **the net Greeks are not constants.** They are themselves functions of spot, vol, and time. The net delta you read today is the delta *at today's price*; move the stock and the delta changes (that change is precisely the gamma). For a multi-leg position the net delta can do genuinely surprising things — including *flipping sign*.

![Net delta of a multi-leg position versus spot price, crossing from negative to positive](/imgs/blogs/the-net-greeks-of-a-position-building-your-risk-dashboard-3.png)

The figure plots the net delta of that same long-strangle position (two \$105 calls + one \$95 put) as the spot price sweeps from \$78 to \$122. Watch what happens. Far below the market, around \$80, the net delta is about **−1.0**: down there the \$95 put is deep in-the-money and dominates, so the position behaves *short* the stock (it gains as the stock falls further). As the stock rises, the put loses delta and the calls gain it, and somewhere around **\$97** the net delta crosses zero — that's the amber dot, the point where the position is momentarily direction-neutral. Above that, the two calls take over and net delta climbs steeply, reaching almost **+2.0** up near \$120, where the position behaves like being long two shares.

The lesson is that "what's my delta?" has no single answer for an option position — it has an answer *at each spot price*, and the whole reason you carry gamma is that your delta is going to change as the stock moves. A position that's delta-neutral today is not delta-neutral after a 5% move; that's the entire content of gamma. Reading a book means knowing not just where your Greeks are *now*, but how they'll *evolve* as the market moves — which is exactly what the scenario analysis later in this post is for.

## The Greek fingerprint of common structures

Once you can sum Greeks and sign them, you can recognize every standard options structure by its *signature*: the pattern of signs across net delta, gamma, theta, and vega. This is how experienced traders think — not "I sold an iron condor," but "I'm short gamma, short vega, long theta, flat delta." The structure name is shorthand; the Greek signature is the actual risk. Learning to translate between the two is what makes the strategy track (verticals, condors, straddles) feel like variations on a theme rather than a list to memorize.

![A matrix table of common option structures and the sign of each net Greek showing the fingerprint of each structure](/imgs/blogs/the-net-greeks-of-a-position-building-your-risk-dashboard-4.png)

The matrix above is the cheat sheet. Each row is a structure; each column is a net Greek; each cell is the sign, colored green where the Greek helps you in that structure's intended scenario, red where it works against you, gray where it's roughly flat. Let's read the rows, because each one is a worked example of summing Greeks.

- **Long straddle / strangle (buy vol):** ≈0 delta, **+gamma, −theta, +vega**. We computed this one above. You're long convexity and long vol, paying theta for the privilege. The bet: a big or fast move, or a vol spike, before decay grinds you down. The whole [long-volatility playbook](/blog/trading/options-volatility/straddles-strangles-and-the-long-volatility-bet) lives in this signature.
- **Short iron condor / strangle (sell vol):** ≈0 delta, **−gamma, +theta, −vega**. The mirror image. You collect theta daily, but you're short convexity (a big move hurts) and short vol (a spike hurts). This is Marcus's core position, and the signature is exactly why it bit him.
- **Covered call (stock + short call):** **+delta, −gamma, +theta, −vega**. You're long the stock (positive delta) but you've sold a call against it, which subtracts delta, adds short gamma, adds positive theta, and adds short vega. The covered call quietly turns a pure stock position into a short-volatility position with a delta tilt — a fact most people who sell covered calls never articulate.
- **Bull call vertical / debit spread:** **+delta, +gamma (small), −theta (small), +vega (small)**. You buy a call and sell a higher one. The short leg cancels *most* of the long leg's gamma, theta, and vega, leaving a position that's mostly a *defined-risk directional bet* (positive delta) with only modest exposure to the other Greeks. That's the appeal of the [vertical spread](/blog/trading/options-volatility/vertical-spreads-debit-and-credit-defining-your-risk): you keep the direction and strip out most of the volatility and time risk.
- **Cash-secured short put (sell put):** **+delta, −gamma, +theta, −vega**. Selling a put is bullish (positive delta), collects theta, and is short gamma and short vega. Same fingerprint as the covered call — which is no coincidence; by put-call parity they're closely related, as the [parity proof](/blog/trading/quantitative-finance/put-call-parity-no-arbitrage-quant-interviews) makes precise.

The pattern jumps out once you've read a few rows: **buying volatility costs theta and earns gamma and vega; selling volatility earns theta and owes gamma and vega.** Delta is the free dimension you can dial independently with the choice of strikes and calls-versus-puts. Almost every structure in the strategy track is just a particular point in this space — a chosen delta, a chosen sign of gamma/vega/theta, and a chosen amount of each. When you put on a new trade, the right first question is never "what's it called?" It's "what does this do to my net Greeks?"

#### Worked example: the covered call's hidden short-vol position

Let's compute the covered-call fingerprint, because it's the most under-appreciated structure on the board — millions of people run covered calls thinking they own "stock with a little extra income," not realizing they've become volatility sellers.

Take 100 shares of the \$100 stock (long stock has delta +1.00 per share, and zero gamma, theta, and vega — stock is linear, with no optionality). Against it, sell one \$105 call, 45 days out, 20% vol. The short call's Greeks (the *long* call's Greeks with every sign flipped):

- Short \$105 call: delta −0.2778, gamma −0.04775, vega −0.1177 per point, theta +\$0.0291/day, and you collect \$1.1628 (\$116.28) in premium.

Sum with the stock (per share, then note the share leg dominates delta):

- **Net delta:** +1.000 (stock) + (−0.2778) (short call) = **+0.7222**. Still net long the stock, but the short call has shaved off about 28% of your directional exposure.
- **Net gamma:** 0 + (−0.04775) = **−0.04775**. Now *short gamma* — your delta will *shrink* as the stock rises (you're capping your upside) and *grow* as it falls (your losses accelerate on the way down, exactly the wrong direction).
- **Net theta:** 0 + (+0.0291) = **+\$0.0291/day**. You're now *collecting* theta — that's the "income."
- **Net vega:** 0 + (−0.1177) = **−0.1177 per point**, or **−\$11.77 per vol point** on the contract. You're *short volatility*.

The intuition: a covered call is not "stock plus income." It is a *short-volatility position with a long-delta tilt*. You've traded away your upside above \$105 and your positive convexity in exchange for a daily theta drip and a one-time premium — and you've quietly become someone who *loses* when implied vol rises. If you run a hundred covered calls, you have a hundred little short-vega positions stacked up, and on the day fear spikes, every one of them moves against you at once. That is the same trap Marcus fell into, dressed in friendlier clothes.

## Scanning a book: scenario shocks, not gut feel

Net Greeks are a brilliant *local* summary, but they're a first-order (and for gamma, second-order) approximation. They tell you what happens for *small* moves. For the moves that actually hurt — a 5% gap, an 8-point vol spike, a week disappearing off the calendar — the linear Greek estimate drifts away from the truth, because the Greeks themselves change as the market moves (that's the gamma-of-the-gamma, the vanna, the volga, all the higher-order effects). So professionals don't only stare at the Greeks. They run **scenario analysis**, also called **shock analysis** or **stress testing**: they re-price the *entire book* under a grid of hypothetical moves and read off the actual P&L.

The recipe is mechanical and honest. Pick a grid of shocks — say, spot from −8% to +8% and implied vol from −8 to +8 points. For each cell in the grid, re-price every leg of the book at the shocked spot and shocked vol (and, if you like, with some days elapsed), sum the legs with their signs, and subtract the book's current value. The result is the book's P&L *in that scenario*. No approximation, no Greek extrapolation — you just ask the pricing model "what would this book be worth if the world looked like that?" and difference it against today.

![Scenario shock heatmap of a short-volatility book P&L across spot move and implied-vol move](/imgs/blogs/the-net-greeks-of-a-position-building-your-risk-dashboard-5.png)

The heatmap above is Marcus's book run through exactly this grid. Columns are the spot move (−8% to +8%); rows are the implied-vol move (+8 points at the top down to −8 points at the bottom). Each cell is the book's full-reprice P&L, green for a gain and red for a loss. The book is delta-hedged, so the center column (spot unchanged) is symmetric-ish, and the dead-center cell (no move, no vol change) is exactly \$0 by construction.

Now read the colors, because the colors *are* the risk. The entire top of the grid — wherever implied vol rises — is red, and it gets *more* red as vol rises further, regardless of what spot does. That vertical red wash is the short-vega risk made visible: this book hates a vol spike. The entire perimeter — wherever spot moves a lot in *either* direction — is also red, deepening toward the corners. That's the short-gamma risk: a big move hurts no matter which way it goes. The only green island is the calm center-bottom: small spot move, *falling* vol. The book makes money exactly when nothing happens and fear recedes — and loses money in almost every other corner of the world. That asymmetry is the whole signature of selling volatility, and it's the kind of thing a scenario grid shows you instantly and a single delta number hides completely.

#### Worked example: the vol-spike that bit Marcus, scenario by scenario

Let's put real dollars on the hook. Marcus's book, delta-hedged with +165 shares at spot \$100 and 18% implied vol, repriced under specific shocks (full reprice, all legs summed via the pricer):

- **Nothing happens (spot flat, vol flat, no time):** P&L = **\$0**. The baseline.
- **One quiet week passes (spot flat, vol flat, 7 days elapsed):** P&L = **+\$2,386**. This is the theta harvest — the reason the book exists. About \$326/day × 7 days, give or take the curvature. On a calm week, Marcus banks a couple thousand dollars for doing nothing. Seductive.
- **Spot flat, implied vol +8 points (the fear-flare), no time:** P&L = **−\$9,834**. The stock didn't move *at all*. Delta was flat *and stayed flat*. And the book still lost nearly ten grand, purely because implied vol jumped and the book was short \$1,163 of vega per point — eight points of spike is roughly `−1,163 × 8 ≈ −\$9,300`, plus a bit more from the vega itself growing as vol rose. This is the cell Marcus lived.
- **Spot −5% *and* vol +8 points (the realistic scary day, since vol and down-moves come together):** P&L = **−\$18,010**. Now the short-gamma and short-vega risks stack: the down-move hurts the gamma *and* the correlated vol spike hammers the vega. Nearly four months of \$2,386-a-week theta harvest, gone in one afternoon.

The intuition: a single quiet week (+\$2,386) feels like the book "works." But it would take roughly *four* of those quiet weeks to recover from one ordinary scary day (−\$9,834), and roughly *seven and a half* quiet weeks to recover the bad-day-with-a-drop (−\$18,010). That is the precise shape of "picking up pennies in front of a steamroller," and you can only *see* the steamroller if you run the scenario grid. The net-vega number on the dashboard told Marcus the slope; the scenario grid told him the size of the cliff.

### What each Greek tells you to hedge

The point of the dashboard and the scenario grid isn't to admire the risk — it's to *act* on it. Each net Greek is a specific instruction about what to hedge and how:

- **Net delta** tells you your directional exposure. Hedge it by buying or selling the *underlying* (shares or futures), which is pure delta with no other Greeks attached — the cleanest hedge there is. Marcus's +165-share hedge zeroed his delta without touching his vega or gamma, which is exactly what a delta hedge does (and exactly why it didn't save him). The mechanics of this are in [delta and the hedge ratio](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio).
- **Net gamma** tells you how *stable* your delta hedge is. High positive gamma means your delta hedge stays good as the stock moves (gamma helps you). High *negative* gamma — Marcus's situation — means your delta hedge *decays* as the stock moves: you have to keep re-hedging, always buying high and selling low, bleeding money on every swing. You can only reduce gamma by trading *options* (buying back some of what you're short), never with shares. The full treatment of why short gamma is "toxic" is in [the gamma post](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short).
- **Net theta** tells you which side of the clock you're on. Positive theta (Marcus) means time is your friend; negative theta means you're paying rent and need the move to come. You don't usually "hedge" theta directly — it's the price (or the income) of the gamma/vega you've chosen — but you watch it to know your daily carry. See [theta and the price of being long options](/blog/trading/options-volatility/theta-trading-the-clock-and-the-price-of-being-long-options).
- **Net vega** tells you your exposure to *fear itself*. This is the Greek Marcus ignored. You hedge vega by trading options (or VIX products, or variance swaps), never with the underlying — shares have zero vega. If your book is short \$1,163 of vega and that scares you, you buy back some optionality to bring it toward your comfort level. The deep dive is [vega and your exposure to implied volatility](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol).

The deep insight here is that **delta and vega are hedged with completely different instruments.** You can flatten delta all day with shares and never touch your vega by one cent. That's *why* a "delta-neutral" book can be a screaming vega bet — the delta hedge is orthogonal to the vega risk. Marcus diligently did the easy, visible hedge (delta, with shares) and never did the harder, invisible one (vega, which would require trading options back). The dashboard exists precisely to stop you from confusing "I hedged the Greek I could see" with "I hedged the Greek that will hurt me."

#### Worked example: a delta-neutral book that loses on a quiet day

To make the orthogonality concrete, let's isolate it. Take Marcus's book and confirm it's delta-neutral: net option delta across all legs is about −164.5 deltas, and his +165 shares contribute +165, leaving net delta ≈ **+0.5 shares** — flat for all practical purposes. So if the stock ticks up or down 0.5%, his P&L barely moves. He's hedged. He's neutral. He's safe — to first order, in *spot*.

Now hold spot dead flat and move *only* implied vol, up 5 points:

- The 165 shares: P&L = `165 × (100 − 100) = \$0`. Shares have no vega; the delta hedge does nothing here.
- The options: every short leg gains value as vol rises (bad for a seller). Full reprice: the option book loses about **−\$6,043**.
- Total: **−\$6,043**, on a day the stock didn't move at all.

The intuition: his delta hedge was a perfect hedge against the risk he *measured* (spot) and a perfect *non*-hedge against the risk he *carried* (vol). A delta-neutral book is neutral to small spot moves and to *nothing else*. The whole reason to build a net-Greek dashboard with *all four* Greeks, not just the one that's easy to see, is that the market will happily collect on whichever risk you forgot to put on the board.

## The interaction: you cannot zero every Greek

Here is the conceptual heart of the post, and the thing that separates traders who understand risk from traders who think risk management means "make all the numbers zero." **You cannot, in general, set every Greek to zero.** Worse, you usually don't *want* to — because a position with all Greeks at zero is a position with no view and no edge, and (after costs) no reason to exist.

Why can't you zero them all? Because the Greeks are *coupled*. You have four risks (delta, gamma, theta, vega) but they don't move independently when you trade the instruments available to you:

- **Shares** move only delta. They're a pure delta knob, zero of everything else. Great for flattening delta, useless for anything else.
- **Options** move delta, gamma, theta, *and* vega *together*, in fixed proportions set by the model. You cannot buy "just gamma" or sell "just vega." Every option you trade to adjust one Greek drags the others along with it. Buy a call to add gamma and you've also added vega, subtracted theta, and added delta — all at once, in proportions you don't control independently.

So the moment you try to neutralize gamma by trading options, you change your vega and theta too, and fixing *those* requires trading *more* options, which re-disturbs your gamma. With enough different strikes and expiries you can engineer a position that's simultaneously delta-, gamma-, and vega-neutral at a single point — market-making desks do exactly this — but it's expensive (you cross a lot of bid-ask spreads), it's only neutral *at that instant and that spot* (gamma and vanna immediately knock it off as the market moves), and a fully-neutral book by construction earns only the theta edge minus costs. The deeper truth is that **risk is not something you eliminate; it's something you choose.**

![A decision figure showing the Greek budget, keeping the Greek that is your view and hedging the rest](/imgs/blogs/the-net-greeks-of-a-position-building-your-risk-dashboard-7.png)

The figure lays out how a professional actually thinks about it — not "zero everything," but a **Greek budget**. Start from the *view*: say you believe realized volatility will come in below what's implied (a short-vol view). That view *is* a bet on specific Greeks: short vega, long theta. So you **keep** those — that's the position paying you, the reason you're in the trade. Then you look at the Greeks you *didn't* mean to hold: you probably didn't intend a directional bet, so you **hedge out** the delta (buy or sell shares — cheap, clean, no side effects). And then there are the Greeks you can't cleanly separate from your bet: short vega comes welded to short gamma, so you **accept the residual** short gamma as the unavoidable cost of expressing your vol view, and you *size it* so that even in a bad scenario, the loss fits inside a number you decided in advance — your budget.

That budget is the discipline. It's a set of hard limits: max dollar-vega of −\$1,500, max dollar-gamma of −\$900, a delta band of ±50, and so on. You don't trade to zero; you trade *to the budget*. Every new position is evaluated by what it does to the book's net Greeks relative to those limits. If adding a trade would push net vega past the cap, you don't add it (or you add an offsetting option). The budget converts the vague goal "manage risk" into a concrete, checkable rule: *does the book, after this trade, still sit inside every Greek limit, and does the worst-case scenario-grid cell still fit inside my maximum acceptable loss?* That second clause — sizing to the worst-case scenario, not the typical day — is the one that would have saved Marcus, and it's the subject of [position sizing and risk of ruin](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading).

#### Worked example: setting and checking a Greek budget on the book

Let's run Marcus's book through a budget the way a desk would, and see exactly where it breaks the rules. Suppose his risk limits are: dollar-vega within ±\$800 per point, dollar-gamma within ±\$500, delta band ±\$2,000 of dollar-delta, and a maximum scenario loss of \$8,000 in the worst grid cell of a ±5% spot / ±5-point vol shock.

Check the book (delta-hedged, S=100, 18% vol) against each limit:

- **Dollar-delta:** ≈\$0 (he's hedged). **Within budget.** ✓
- **Dollar-gamma:** −\$735. Budget is ±\$500. **Over by \$235.** ✗
- **Dollar-vega:** −\$1,163 per point. Budget is ±\$800. **Over by \$363.** ✗
- **Worst scenario cell (within ±5% / ±5pt):** from the scenario run, spot −5% with vol +5 points = **−\$14,673**. Budget is a \$8,000 max loss. **Over by \$6,673 — nearly double the limit.** ✗

So the book fails three of four limits, and badly. The fix isn't "hedge delta harder" (delta's the one thing that's fine). The fix is to *reduce the short-vol bet itself*: buy back some of the short straddles or short puts to bring net vega from −\$1,163 toward −\$800 and net gamma from −\$735 toward −\$500. Buying back, say, a third of the short straddles would cut vega and gamma by roughly a third — pulling vega to about −\$780 and gamma to about −\$490, both inside budget — and would shrink the worst-case scenario cell proportionally, toward the \$8,000 limit. It costs Marcus some of his daily theta (he's harvesting less now), but that's the point: **the budget forces him to trade a little less income for a lot less tail risk.** The intuition: a Greek budget turns "I feel diversified" into an arithmetic check that either passes or fails — and Marcus's book fails it loudly, which is exactly the warning he never got because he never built the dashboard.

### The two risks the book actually carries

To close the loop on "which risks am I choosing," it helps to plot the chosen Greeks across spot, not just read them at the current price. Marcus's *real* bet — once you strip out the delta he hedged away — is short vega and short gamma. How big are those, and where?

![Net vega and net gamma of a book versus spot price shown as two lines](/imgs/blogs/the-net-greeks-of-a-position-building-your-risk-dashboard-6.png)

The figure plots the book's net dollar-vega (per point) and net dollar-gamma against spot, from \$85 to \$115. Both curves are deeply negative and both are *most* negative right around \$100 — exactly where the stock sits now. That's because the short straddles and short puts are clustered near the money, and an option's gamma and vega peak at-the-money. So the book is *maximally* short vol precisely at the current price, and its short-vol exposure *fades* as the stock moves into the wings (the options go further in- or out-of-the-money, where gamma and vega shrink). 

There's a cruel twist hidden in that shape. The book is least dangerous (smallest short vega/gamma) when the stock has already moved far away — but to *get* there, the stock has to traverse the region where the book is most short gamma, losing money the whole way. And a vol spike, which is what a big move usually triggers, slams the vega exposure that's largest right where he sits. The risk graph and the Greek-vs-spot curves together tell the full story that no single snapshot number can: this book is a coiled short-vol spring, tightest right at the money, and it pays a steady \$326 a day right up until the day it doesn't.

## Common misconceptions

### Misconception 1: "If I'm delta-neutral, I'm market-neutral."

This is the exact belief that cost Marcus ten thousand dollars, and it's the most common error in all of options risk. Delta-neutral means *neutral to small moves in the underlying price* — and to nothing else. It says nothing about your exposure to a *vol* change (vega), to a *large* move (gamma), or to *time* (theta). Marcus's book was delta-neutral to within half a share and still lost **−\$9,834** on a day the stock didn't move, purely from an 8-point vol spike against his −\$1,163-per-point vega. "Market-neutral" is a four-dimensional claim; "delta-neutral" checks exactly one of the four boxes. A truly neutral book would have to be flat on delta, gamma, *and* vega at once — and as we saw, that's expensive, fleeting, and usually pointless. The correction in numbers: zero delta, −\$1,163 vega, −\$735 gamma is *not* a neutral book. It's a short-vol book with a delta hedge.

### Misconception 2: "I'm diversified because I have a dozen different positions."

Diversification is about *uncorrelated* risks, not a *count* of positions. Marcus had a dozen tickers and felt spread out, but every one of his positions was short volatility — short straddles, short calls, short puts. When you sum the Greeks, a dozen short-vol positions don't diversify; they *concentrate*. They all have the same sign on vega and gamma, so they all move against him on the same day, in the same direction, for the same reason. The net-Greek dashboard reveals this instantly: a "diversified" book whose net vega is −\$1,163 has one big bet, not twelve small independent ones. The correction: diversification is measured in the *summed* Greeks (and the scenario grid), not in the number of line items. Twelve copies of the same bet is one bet, twelve times as large.

### Misconception 3: "Positive theta means the position is safe."

Positive theta is seductive because it pays you every single day the world does nothing — and "nothing" is the most common thing that happens, so positive-theta strategies win *most* of the time. But "wins most days" is not "safe." Positive theta is mechanically welded to *negative* gamma and *negative* vega (you can't collect the rent without selling the convexity). So a positive-theta book is, by construction, a book that loses on a big move and loses on a vol spike. Marcus collected +\$326 a day and lost \$9,834 in an afternoon — a positive-theta book behaving exactly as the sign of its *other* Greeks promised it would. The correction: theta's sign tells you your daily carry, never your safety. To know if you're safe, you must read gamma and vega, and run the scenario grid for the bad day.

### Misconception 4: "Net Greeks tell me everything I need about my risk."

Net Greeks are a *local linear* (and for gamma, local quadratic) approximation. They're superb for small moves and completely trustworthy for the next tick. But they systematically *understate* the risk of large moves, because the Greeks themselves change as the market moves — gamma changes (that's "speed" or the third-order Greek), vega changes with spot (that's vanna), vega changes with vol (that's volga). For the moves that actually blow up accounts, the linear Greek estimate drifts. That's precisely why the scenario grid exists: it *fully re-prices* the book under each shock instead of extrapolating from today's Greeks. The correction: use net Greeks for the local read and the daily dashboard, but size your position off the *full-reprice scenario grid*, where the worst-case cell is computed exactly, not approximated. Greeks are the speedometer; the scenario grid is the crash test.

### Misconception 5: "Good risk management means hedging every Greek to zero."

If you zero every Greek, you've zeroed your *view* — and after you've paid the bid-ask spread on every option you traded to get there, you have a position that, by construction, can only lose to costs. A neutral book has no edge to harvest. Professional risk management isn't zeroing; it's *budgeting* — deciding which Greek is your intended bet (and keeping it), which Greeks are accidental (and hedging those down), and how large the residual you can't separate is allowed to get (the budget). The correction: the goal is not "all Greeks at zero," it's "the Greek that is my view sized to my conviction, the Greeks I didn't mean to hold hedged toward flat, and the coupled residuals sized so the worst-case scenario fits my maximum loss." Trading is choosing which risks to carry, never carrying none.

## How it shows up in real markets

### The short-vol blowups: Volmageddon and the yen carry unwind

Marcus's story is a miniature of the most famous accidents in modern options markets, which are almost always *the same trade at scale*: a giant book that's short volatility, delta-neutral, collecting steady theta, and catastrophically short vega when the spike comes. On February 5, 2018 — "Volmageddon" — a cluster of short-volatility products and strategies that had been quietly harvesting decay for years met a single day where the VIX spiked to a **37.32** close, more than doubling intraday. Funds and exchange-traded products engineered to be short vol weren't wrong about direction (the S&P fell only a few percent); they were short vega into a vol explosion, and several were effectively wiped out overnight. The net-Greek lesson is exact: their dashboards, had anyone read them honestly, showed enormous negative vega, and the scenario grid for "VIX doubles" showed a loss that exceeded their capital. The steady theta they'd collected for years didn't matter; the one bad cell did.

The August 5, 2024 yen-carry-unwind episode rhymed: the VIX spiked to **38.57** as a crowded, leveraged, implicitly-short-vol set of positions unwound at once. Again, the move in the underlying indices was real but not historic; the damage came from the *vol* spike against books that were structurally short vega and had sized to the calm-day theta, not the bad-day scenario. Both episodes are the macro version of the single screen in this post's cover: short vega + short gamma + long theta, sized to the income and not to the tail.

### The earnings vol-crush, seen as a Greek trade

Flip to the other side and the net-Greek lens explains the most common retail trade around earnings. A trader who buys a straddle the day before earnings is putting on the long-volatility fingerprint: +gamma, +vega, −theta, ≈0 delta. They're long vega into an event — and the catch is that pre-earnings implied vol is *elevated* precisely because everyone wants that long-vega exposure. The morning after, two things hit their dashboard at once: the stock gaps (good for their gamma if the move is big enough) but implied vol *collapses* — the "vol crush" — and their large positive vega turns that IV drop into a loss. Many earnings straddle buyers are baffled that "the stock moved and I still lost money." The net Greeks dissolve the mystery: their position's P&L is `(gamma gain from the move) + (vega loss from the crush) − (theta paid)`, and if the vega loss from a multi-point IV collapse exceeds the gamma gain from the move, they lose despite being directionally "right" about a move happening. The full mechanics are in [the expected move and pricing event risk](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options) and [event volatility and the vol crush](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush).

#### Worked example: the earnings straddle that "should have" worked

Use our long straddle (long \$100 call + long \$100 put, 30 days, but priced into earnings at an inflated **35%** implied vol). At 35% vol the straddle costs more — say it's worth about \$8.00 (call + put) going in, with a net vega of roughly +0.23 per point (×100 = **+\$23/point**) and roughly delta-flat. Earnings hit, the stock gaps from \$100 to \$104 (a 4% move), and implied vol *crushes* from 35% back to a normal 20% — a **15-point** drop.

- **Gamma/spot gain:** the 4% move is worth something to a long straddle — the call gains more than the put loses. Call it roughly +\$1.50 of intrinsic-and-time gain on the move, in the right ballpark for a \$4 move on a 30-day straddle.
- **Vega/crush loss:** `−15 points × \$23/point ≈ −\$345` on the contract... but vega itself shrinks as vol falls, so the *realized* crush loss on this straddle is more like **−\$3.50 to −\$4.00** per share of premium evaporating — easily swamping the \$1.50 directional gain.
- **Net:** the straddle that cost \$8.00 is now worth perhaps \$5.50 even though the stock moved 4% in a clean direction. The buyer was right about the move and still lost roughly **−\$2.50 per share** (−\$250 on the contract), because their dominant Greek going in was *long vega* and the event *crushed* vega.

The intuition: around a known event, the long-volatility fingerprint means your biggest exposure is to *implied vol*, not to the move. The dashboard would have told the trader "you're long \$23 of vega per point into an event that reliably crushes vol 15 points" — which is a different, and much less appealing, trade than "I think the stock will move."

### How a real desk uses the dashboard

On a professional options desk, the net-Greek dashboard and the scenario grid aren't a nice-to-have; they're the primary risk screen, refreshed in real time. A market-maker running thousands of positions across hundreds of strikes doesn't think in individual contracts at all — they think in the *aggregated* Greeks per underlying and the scenario P&L matrix for the whole book. Their delta is hedged near-continuously with futures (cheap, pure delta). Their gamma and vega are managed to *limits* — a desk will have a hard cap on net vega per name and per book, and a risk manager whose entire job is to make sure no trader's scenario grid has a cell that exceeds the firm's appetite. When a desk is "too short gamma" into a Friday, it buys back optionality even at a loss, because the budget says so. This is the institutional version of everything in this post: read the book through summed Greeks, stress it with full-reprice scenarios, and trade to a budget rather than to a feeling. The dealer-positioning consequences of all those desks being long or short gamma together is itself a market force, explored in [volatility as an asset](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear).

## The playbook: building and reading your risk dashboard

Everything above collapses into an operating routine. This is the synthesis of the whole Greeks track — the point where delta, gamma, theta, and vega stop being five separate lessons and become one dashboard you read before, during, and after every trade.

**Build the dashboard first, for the whole book, in dollars.** For each underlying, sum every leg's delta, gamma, theta, and vega with the correct long/short sign, then convert to dollar-Greeks: dollar-delta (per 1% spot), dollar-gamma, dollar-vega (per point), theta-per-day. One screen, four numbers per name, plus a book-level total. If you can't produce this screen for your current positions in under a minute, you don't actually know your risk — you know your *positions*, which is not the same thing.

**Read the book by its fingerprint, not its names.** Before you ask "what strategies do I have on," ask "what's the sign and size of each net Greek?" Long vega + long gamma + short theta = you're long volatility and need a move or a spike. Short vega + short gamma + long theta = you're short volatility and need calm. The names (condor, straddle, covered call) are shorthand; the summed signs are the truth. Marcus's "income book" was a short-vol book — the fingerprint said so even though the names didn't.

**Always run the scenario grid for the bad day, not the typical day.** Net Greeks are the local slope; they understate large-move risk. Re-price the whole book under a grid of shocks — at minimum spot ±5%, IV ±5 points, and "one week passes" — and find the worst cell. Size the position so that worst cell is a loss you can survive and have decided to accept *in advance*. Marcus's typical day was +\$2,386 a week; his bad day was −\$18,010. He sized to the first and got killed by the second. Size to the scenario, never to the carry.

**Hedge the Greek you didn't mean to hold; keep the one that's your view.**

- If a Greek is *accidental* (usually delta, when your view is about vol or time), hedge it with the cleanest instrument: **shares/futures for delta** (zero side effects). Don't let an unintended directional bet ride just because it's small today — gamma will grow it as the stock moves.
- If a Greek *is* your view (vega for a vol trade, delta for a directional trade), **keep it, sized to your conviction**, and put it on the dashboard as the intended bet.
- Remember the orthogonality: **delta is hedged with the underlying; gamma and vega are hedged only with options.** A delta hedge does nothing to your vega. If your vega scares you, you must trade options back, not shares.

**Run a Greek budget, and check every new trade against it.** Set hard limits — max dollar-vega, max dollar-gamma, a delta band, and a maximum scenario loss. Before adding any position, compute what it does to the book's net Greeks and re-run the worst-case scenario cell. If the trade pushes any limit past its cap, either don't do it or add an offsetting option. The budget turns "manage risk" into an arithmetic gate that passes or fails. The discipline of sizing the residual you can't hedge away is the bridge to [position sizing and risk of ruin in options trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading).

**Re-read the dashboard after the market moves, not just when you trade.** Because the net Greeks are functions of spot, vol, and time, a book you set up neutral this morning can be far from neutral by the afternoon. A short-gamma book gets *more* short delta as the stock falls and *more* long delta as it rises (it leans into the move, the wrong way), so its delta hedge decays and must be refreshed. Watching the dashboard *evolve* — not just its opening snapshot — is the difference between managing a position and being managed by it.

#### Worked example: re-running Marcus's book with the playbook

Let's close the loop on the hook by replaying Marcus's situation with the dashboard he should have had. Same positions, same \$100 spot, same 18% vol going in.

1. **Build the dashboard.** Summed, the book reads: dollar-delta ≈ \$0 (after his +165-share hedge), dollar-gamma −\$735, dollar-vega −\$1,163/point, theta +\$326/day. The very first glance shows the longest red bar is *vega*, not delta — the risk lives in the Greek he never watched.
2. **Read the fingerprint.** Short vega + short gamma + long theta = short volatility. This isn't an income book; it's a leveraged bet that vol stays low. Naming it correctly changes how he'd size it.
3. **Run the scenario grid.** The worst cell within a ±5% / ±5-point shock is spot −5% with vol +5 points = **−\$14,673**, and a sharper "spot −5%, vol +8" is **−\$18,010**. He'd see, in dollars, that one bad day erases roughly two months of theta harvest.
4. **Apply a budget.** Against limits of ±\$800 vega, ±\$500 gamma, and an \$8,000 max scenario loss, the book fails three of four. He buys back about a third of his short straddles, pulling vega to ≈−\$780 and gamma to ≈−\$490 (inside budget) and shrinking the worst scenario cell toward the \$8,000 cap. He keeps a *smaller* short-vol bet — the one that's actually his view — and gives up some daily theta to fit the budget.
5. **Survive the spike.** Now the same 8-point vol flare on flat spot costs him about `−780 × 8 ≈ −\$6,200` instead of −\$9,834 — a loss inside what he decided in advance he could take, not a gut-punch that threatens the account.

The intuition: the playbook wouldn't have changed Marcus's *view* — being short vol into a calm regime is a legitimate, even profitable, bet most of the time. It would have changed his *measurement and his size*. He'd have seen the vega on the dashboard, named the book correctly, priced the bad day with the scenario grid, and sized the residual to a budget. The difference between a manageable −\$6,200 and an account-threatening −\$18,010 was never the trade idea. It was whether he built the dashboard before the market sent the invoice.

**The single habit to build:** before every trade, and again every time the market moves, ask one question — *what does this do to my book's net Greeks, and where's the worst cell in my scenario grid?* If you can answer that in dollars, for the whole book, in under a minute, you're reading the dashboard the way a professional does. If you can't, you're Marcus on a quiet afternoon: comfortable, delta-hedged, and short the one thing nobody's watching.

## Further reading & cross-links

- **[Delta: Direction, Exposure, and the Hedge Ratio](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio)** — the first Greek, the one you hedge with shares, and why net delta is the slope of the risk graph.
- **[Gamma: The Greek That Bites — Curvature, Convexity, and the Toxic Short](/blog/trading/options-volatility/gamma-the-greek-that-bites-curvature-convexity-and-the-toxic-short)** — why a short-gamma book's delta hedge decays and bleeds on every swing, the residual Marcus had to accept.
- **[Theta: Trading the Clock and the Price of Being Long Options](/blog/trading/options-volatility/theta-trading-the-clock-and-the-price-of-being-long-options)** — the daily carry on the dashboard, and why positive theta is income, never safety.
- **[Vega: Your Exposure to Implied Volatility and the Vol of Vol](/blog/trading/options-volatility/vega-your-exposure-to-implied-volatility-and-the-vol-of-vol)** — the Greek Marcus ignored; how to measure and hedge exposure to fear itself.
- **[Rho, Dividends, and the Second-Order Greeks: Vanna, Volga, Charm](/blog/trading/options-volatility/rho-dividends-and-the-second-order-greeks-vanna-volga-charm)** — why the net Greeks aren't constant, and the higher-order effects the scenario grid captures and the linear Greeks miss.
- **[Vertical Spreads: Debit and Credit, Defining Your Risk](/blog/trading/options-volatility/vertical-spreads-debit-and-credit-defining-your-risk)** — the structure that keeps direction and strips out most gamma/vega/theta; a point in the fingerprint space.
- **[Iron Condors and Credit Spreads: Selling the Range](/blog/trading/options-volatility/iron-condors-and-credit-spreads-selling-the-range)** — the short-vol fingerprint as a defined-risk structure, the safer cousin of Marcus's naked short straddles.
- **[Straddles, Strangles, and the Long Volatility Bet](/blog/trading/options-volatility/straddles-strangles-and-the-long-volatility-bet)** — the long-volatility fingerprint in full, the mirror image of this post's book.
- **[Position Sizing and Risk of Ruin in Options Trading](/blog/trading/options-volatility/position-sizing-and-risk-of-ruin-in-options-trading)** — how to set the Greek budget and size the residual so the worst scenario cell is survivable.
- **[Black-Scholes](/blog/trading/quantitative-finance/black-scholes)** — the pricing model every Greek and every scenario reprice in this post is computed from, with the full derivation we deliberately did not repeat.

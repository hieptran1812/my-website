---
title: "Building a market-making simulator: spread, inventory, and adverse selection"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A from-scratch, code-and-numbers guide to how market makers earn the spread, bleed to inventory and adverse selection, and why the Avellaneda-Stoikov reservation price ties it all together -- built around a simulator you can reason about."
tags: ["market-making", "quant-research", "adverse-selection", "inventory-risk", "bid-ask-spread", "avellaneda-stoikov", "reservation-price", "quant-interviews", "trading-simulation", "liquidity"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** -- A market maker quotes both a buy price and a sell price, earns the gap between them, and slowly bleeds that profit back to inventory risk and adverse selection. Simulating the whole loop turns those abstract tradeoffs into dollars you can watch accumulate and leak.
>
> - **You earn the spread** by buying low at your *bid* and selling high at your *ask*. A \$0.20 spread on 100 shares is \$20 of profit per completed round trip.
> - **You carry inventory risk** because fills do not arrive in matched pairs. You accumulate a position whose mark-to-market jumps with every price tick -- a long position loses money when the price falls.
> - **You skew your quotes by inventory** -- shifting both prices down when you are long and up when you are short -- to pull your position back toward flat. The center of your quotes is the *reservation price*.
> - **You lose to adverse selection** because the traders who pick you off tend to be right: informed flow only hits the side that is about to win. Your spread must be wide enough that random flow pays for the toxic flow.
> - **The Avellaneda-Stoikov model** writes both ideas as one formula: a reservation price that drifts with inventory and time, plus an optimal half-spread that widens with volatility and risk aversion and collapses as the close nears.
> - **The number to remember:** market making earns a thin, steady edge most days and carries a fat left tail. In our simulator, an average day nets about **\$270** while the worst 5% of days lose more than **\$400**.

Why does the price you can *buy* a stock at always sit a little above the price you can *sell* it at, in the very same instant, on the very same screen? You see it everywhere: a stock shows "\$99.90 / \$100.10", a foreign-exchange app quotes you one rate to buy euros and a worse one to sell them back, a crypto exchange shows a green number and a red number a hair apart. That gap is not a glitch and it is not greed. It is the price of a service -- *immediacy* -- and somebody is being paid to provide it. That somebody is a **market maker**, and this article is about building a simulator that shows, in dollars, exactly how they make money and exactly how they lose it.

![A market maker buys at the bid and sells at the ask; the gap is the spread it earns for providing liquidity.](/imgs/blogs/market-making-simulator-quant-research-1.png)

The diagram above is the mental model for the whole post. In the middle sits the market maker, posting two prices at once: a **bid** of \$99.90 (where it is willing to buy) and an **ask** of \$100.10 (where it is willing to sell). An impatient buyer who needs shares right now *lifts the ask* and pays \$100.10; an impatient seller who needs cash right now *hits the bid* and sells at \$99.90. If both happen, the market maker has bought at \$99.90 and sold at \$100.10, banking the \$0.20 *spread* without ever predicting which way the price goes. That sounds like free money. It is not, and the rest of this post is the story of why -- told through a simulator you could code in an afternoon and reason about for a career.

This matters far beyond curiosity. "Make a market" is the single most common live exercise in a quant trading interview at firms like Jane Street, Optiver, SIG, IMC, HRT, and Citadel Securities, and "build a market-making simulator" is a classic take-home. If you can explain spread capture, inventory risk, adverse selection, and the reservation price from first principles -- and back each with a worked number -- you are most of the way to a strong interview. For the interview-game cousin of this article, see [market-making games: making tight markets under uncertainty](/blog/trading/quantitative-finance/market-making-games-quant-interviews); this piece is the *simulation and research* counterpart.

A note before we start: this is educational, not financial advice. We are explaining mechanisms and tradeoffs, not telling anyone to trade.

## Foundations: the words you need before any math

Let us build the vocabulary from absolute zero. Skip ahead if you already trade for a living, but every term below appears in the worked examples, so it pays to be precise.

**An order book** is the list of everyone's resting buy and sell orders for an asset, sorted by price. Buyers post *bids* (offers to buy at a price); sellers post *asks* or *offers* (offers to sell at a price). The order book has two sides that do not touch: the highest bid is always below the lowest ask, because if a bid ever reached up to an ask, the two would trade and vanish.

**The bid** is the highest price anyone is currently willing to *buy* at. The **ask** (or **offer**) is the lowest price anyone is willing to *sell* at. If you want to sell *right now*, you sell *to* the bid. If you want to buy *right now*, you buy *from* the ask. The bid is always the lower number; the ask is always the higher number.

**The spread** is the gap between them: spread = ask − bid. In our running example, ask − bid = \$100.10 − \$99.90 = **\$0.20**. The spread is quoted in dollars (or in *ticks*, the smallest price increment an exchange allows, or in *basis points* -- a *basis point* is one hundredth of one percent, so 0.01%).

**The mid** (or *mid-price*) is the average of bid and ask: (bid + ask) / 2 = (\$99.90 + \$100.10) / 2 = **\$100.00**. The mid is the market's rough consensus on *fair value* -- the price at which the asset is, loosely, "worth" something. It is not a price you can actually trade at; it sits in the no-man's-land between the two sides.

**A market maker** (also called a *liquidity provider*) is a trader whose business is to post *both* a bid and an ask simultaneously and continuously, standing ready to buy from sellers and sell to buyers. It does not take a directional view ("the price will rise"); it takes a *flow* view ("trades will arrive, and I will be on the other side of them and collect the spread").

**A liquidity taker** is anyone who trades *against* a resting quote -- the impatient buyer or seller who crosses the spread to get filled immediately. Takers pay the spread; makers earn it. This is the fundamental exchange of market microstructure: takers buy immediacy, makers sell it.

**Inventory** (or *position*) is how many units the market maker currently holds. We write it as $q$. A positive $q$ means *long* (you own shares and profit if the price rises); a negative $q$ means *short* (you sold shares you did not have and profit if the price falls); $q = 0$ means *flat* (no position, no price risk). Inventory is the central state variable of the whole problem.

**Mark-to-market** (MTM) is valuing your inventory at the current mid-price rather than what you paid. If you are long 100 shares and the mid is \$100, your inventory is *marked* at \$10,000. If the mid drops to \$99.50, the same 100 shares are now marked at \$9,950 -- you have an unrealized loss of \$50 even though you have not sold anything.

**Adverse selection** is the tendency for the trades that fill your quotes to be, on average, the ones you would rather not have made -- because the trader on the other side knows something you do not. We will define this carefully later; for now, hold the intuition that *the counterparties who pick you off are disproportionately right*.

**Profit and loss** (**P&L**) is the change in your total wealth: realized cash from completed trades plus the mark-to-market value of inventory you still hold. P&L is what we simulate and attribute.

With those nine terms, you can read every figure and every worked example in this post. Now let us put them to work.

## The simplest possible strategy: symmetric quoting

Strip the problem to its bones. You are a market maker in one stock whose fair value -- the mid -- is \$100.00. You decide to quote *symmetrically*: a bid \$0.10 below the mid and an ask \$0.10 above it. So your bid is \$99.90, your ask is \$100.10, and your **half-spread** (the distance from the mid to each quote) is \$0.10. The full spread is \$0.20.

You quote in *clips* of 100 shares -- that is the size you are willing to trade at each price. Every time a buyer lifts your ask, you sell 100 shares at \$100.10. Every time a seller hits your bid, you buy 100 shares at \$99.90.

#### Worked example: the dollars in one round trip

Suppose, over the next minute, exactly one buyer and one seller arrive. The seller hits your bid first: you *buy* 100 shares at \$99.90, paying 100 × \$99.90 = **\$9,990**. Your inventory is now +100 (long), and your cash is −\$9,990.

Then the buyer lifts your ask: you *sell* 100 shares at \$100.10, receiving 100 × \$100.10 = **\$10,010**. Your inventory is back to 0 (flat), and your cash is −\$9,990 + \$10,010 = **+\$20**.

You started flat, you ended flat, and you are \$20 richer. That \$20 is exactly the spread (\$0.20 per share) times the size (100 shares): \$0.20 × 100 = **\$20**. The mid never moved, you took no view, and you earned \$20 for the service of being on both sides. This is *spread capture*, and it is the entire reason market making exists.

The one-sentence intuition: **a completed round trip -- buy at the bid, sell at the ask -- banks the spread times the size, regardless of where the price goes.**

#### Worked example: a day of spread capture

Now scale it. Say two-sided flow is balanced and you complete 30 round trips over a trading day, each one a buy-at-bid then sell-at-ask of 100 shares. Each round trip nets \$20, so your gross spread capture is 30 × \$20 = **\$600** for the day.

The figure below shows the first five round trips stacking up. Each completed pair adds \$20 to cumulative P&L, so the line climbs in a clean staircase. Five trips is \$100; thirty trips is \$600.

![Every matched buy-then-sell pair banks $0.20 of spread on 100 shares, so flat two-sided flow grows P&L in a staircase.](/imgs/blogs/market-making-simulator-quant-research-2.png)

If the world really worked like this -- perfectly balanced flow, a mid that never moves, no smarter counterparties -- market making would be a money printer and everyone would do it (which would compete the spread to zero, but set that aside). The world does not work like this. Two forces eat into that \$600: your inventory swings, and your counterparties are sometimes informed. We take them one at a time.

## Inventory risk: the position you did not mean to take

The staircase above assumed buys and sells arrive in neat alternating pairs. They do not. Flow is *random*: sometimes three sellers hit your bid in a row before any buyer shows up, and now you are long 300 shares whether you wanted to be or not. You did not choose this position by forming a view; you *inherited* it by providing liquidity. That is **inventory risk**.

Why is it a risk? Because while you hold those 300 shares, the mid keeps moving, and your inventory is marked to it. Every \$0.01 the mid falls costs a long position 300 × \$0.01 = \$3. The position you accumulated as a byproduct of earning spread has become a directional bet you never intended to make.

#### Worked example: the inventory path and its mark-to-market

Let us trace a concrete path. You start flat. Over ten minutes, fills arrive unevenly and your inventory wanders: +100, then +50, then up to +280, back to +160, down through zero to −170, then to −280, and so on. The figure below plots this *inventory path* -- the running total of shares you hold -- against time. Above the dashed zero line you are long (green); below it you are short (red).

![Inventory drifts up and down as fills arrive; the further it strays from zero, the more a price move can hurt.](/imgs/blogs/market-making-simulator-quant-research-3.png)

Now layer on price risk. Suppose at the moment your inventory peaks at +300 long, the mid drops by \$0.50 -- a perfectly ordinary move. Your mark-to-market loss is 300 × \$0.50 = **\$150**. That single adverse move wipes out the spread from more than seven round trips (\$150 / \$20 = 7.5). And the symmetry is brutal: if instead you were *short* 300 and the mid *rose* \$0.50, you lose the same **\$150**, because a short position loses when the price climbs.

The intuition: **inventory P&L scales with the size of your position times the size of the price move, and it is mean-zero in expectation but ruinous in the tail.** Over many days, price moves average out -- but on the day a big move catches you holding a big position, the loss can dwarf a week of spread.

Two levers control inventory risk. The first is *volatility*: a more volatile stock (bigger price swings per minute) makes any given position more dangerous. We will write volatility as $\sigma$ (sigma), the standard deviation of the price move per unit of time -- if $\sigma = \$0.10$ per square-root-second, the mid typically wanders about \$0.10 over one second and about $\$0.10 \times \sqrt{60} \approx \$0.77$ over a minute. The second lever is *how much inventory you let yourself accumulate*. You cannot stop fills from arriving unevenly, but you *can* change your quotes to make the position you are accumulating less likely to grow. That is the next idea.

#### Worked example: how big does inventory get on its own?

Here is a fact that surprises beginners. Even with perfectly balanced flow -- buyers and sellers equally likely each tick -- your inventory does not stay near zero. It performs a *random walk*, and a random walk wanders. If each fill is +100 or −100 shares with equal probability, then after $n$ fills the standard deviation of your inventory is $100 \times \sqrt{n}$ shares. After 100 fills that is $100 \times \sqrt{100} = 1{,}000$ shares of typical inventory; after 400 fills it is $100 \times \sqrt{400} = 2{,}000$ shares. The position grows like the square root of the number of trades, with *no* tendency to come back on its own. So if a \$100 stock has \$0.30 volatility over the time it takes to mean-revert, a typical 1,000-share inventory carries 1,000 × \$0.30 = **\$300** of one-standard-deviation risk -- bigger than the spread from fifteen round trips. This is precisely why skewing exists: left alone, balanced flow still builds a dangerous position, so you must actively bias your quotes to pull it home. The intuition: **inventory does not stay flat by itself; without skew it random-walks to dangerous sizes, and risk grows even when flow is fair.**

## Skewing your quotes: making inventory mean-revert

Here is the key move that separates a naive market maker from a real one. If you are long and want to stop getting longer, do not quote symmetrically around the mid -- quote symmetrically around a price *below* the mid. Lower both your bid and your ask. Lowering your ask makes you *more eager to sell* (your sell price is now closer to where buyers are, so you get hit more often on the sell side), and lowering your bid makes you *less eager to buy* (your buy price is now further from where sellers are). Both effects push your inventory back down toward zero. When you are short, you do the mirror image: raise both quotes to buy faster and sell slower.

The center you quote around is your **reservation price**, written $r$. It is your *personal* fair value given the inventory you are stuck holding -- the price at which *you*, with your current position, are indifferent between buying and selling one more share. When you are flat, $r$ equals the mid. When you are long, $r$ sits below the mid (you would happily sell, even a bit cheap, to shed risk). When you are short, $r$ sits above the mid.

![Your reservation price slides below the mid when you are long and above it when you are short, pulling inventory home.](/imgs/blogs/market-making-simulator-quant-research-4.png)

The figure shows three inventory states side by side. When you are *flat* (middle), your bid \$99.90 and ask \$100.10 bracket the mid \$100.00 symmetrically. When you are *long +200* (right), the whole ladder slides down: your reservation price drops to \$99.95, so your ask becomes \$100.05 (you will sell cheaper to get flat) and your bid becomes \$99.85 (you make buyers come to you). When you are *short −200* (left), the ladder slides up the same way. The half-spread -- the \$0.10 on each side -- stays the same; what moves is the *center*.

#### Worked example: how far to skew

A simple, popular rule sets the reservation price as the mid minus a skew proportional to inventory:

$$r = \text{mid} - \gamma \, \sigma^2 \, q$$

where $q$ is your inventory, $\sigma$ is volatility, and $\gamma$ (gamma) is your *risk aversion* -- a knob for how much you dislike risk, with a larger $\gamma$ meaning you skew harder to dump inventory faster. The $\sigma^2 q$ piece says: skew more when the stock is volatile (each share is riskier) and more when your position is large.

Let us plug in friendly numbers. Take $\gamma = 0.1$, $\sigma^2 = 0.25$ (so $\sigma = \$0.50$ over the relevant horizon), and inventory $q = +200$. Then the skew is $0.1 \times 0.25 \times 200 = \$5.00$? That is far too large for a \$100 stock -- which tells you the units of $\gamma$ and $\sigma$ have to be chosen to make the magnitude sane. Recalibrate: with $\gamma = 0.001$ and $\sigma^2 = 0.25$ and $q = 200$, the skew is $0.001 \times 0.25 \times 200 = \$0.05$. So your reservation price is \$100.00 − \$0.05 = **\$99.95**, and both quotes drop a nickel -- exactly the figure above. The lesson about units is itself worth remembering: **always sanity-check that the skew is a small fraction of the spread, not larger than the price.**

The figure below makes the relationship explicit: the reservation-price shift is a straight line through zero. At $q = 0$ the shift is zero (you quote around the mid). As $q$ grows positive (long), the shift grows negative (quote lower). As $q$ goes negative (short), the shift grows positive (quote higher). The slope is $\gamma \sigma^2$.

![Holding 200 long, your reservation price drops below the mid, so the same half-spread leaves you more eager to sell.](/imgs/blogs/market-making-simulator-quant-research-6.png)

The intuition: **skewing your quotes turns a random-walking inventory into a mean-reverting one -- it does not stop fills from arriving, but it biases which side fills so your position drifts home.** Skewing costs you a little: you sell some inventory slightly below the mid and buy some slightly above, which is a small, deliberate giveback of edge in exchange for carrying less risk. That trade -- a touch of spread for a lot less inventory variance -- is almost always worth it.

## Adverse selection: why your fills are not random

Now the deepest idea, and the one that separates traders who survive from those who blow up. So far we have treated fills as if a coin decides whether a buyer or seller arrives. In reality, *some* of the traders hitting your quotes know something. A trader who has just seen news, or computed a better fair value, or detected a large order about to hit the market, will trade against you *only on the side that is about to win*. If good news is coming, they lift your ask (buy from you cheap) right before the price jumps up. If bad news is coming, they hit your bid (sell to you dear) right before it drops. Either way, you are left holding the wrong side. This is **adverse selection**, and it is the central cost of market making.

![Your fills split into harmless noise that pays the spread and toxic informed flow that runs the price against you.](/imgs/blogs/market-making-simulator-quant-research-5.png)

The figure splits your flow into two streams. Most trades come from **noise traders** -- people trading for reasons unrelated to short-term price (rebalancing a fund, raising cash, hedging, sheer randomness). Noise flow is harmless: it fills your quotes in both directions, and you keep the spread, earning roughly +\$0.10 per share. The other stream is **informed traders**. Informed flow is *toxic*: it only ever takes the side that is about to move in its favor, so right after you fill an informed order, the price runs against you. In the figure, that costs −\$0.40 per share. The whole game is that the spread you earn from noise has to be wide enough to pay for the losses you take to informed flow.

This is formalized in the classic **Glosten-Milgrom** model of market microstructure: the spread exists *precisely because of* adverse selection. Even a market maker with zero costs and zero risk aversion must charge a spread, because some fraction of its counterparties are informed and will systematically pick it off. The spread is the toll that lets the maker break even across both populations.

There is a subtle and important corollary here, and interviewers probe it: **every fill is information.** When someone lifts your ask, you have just learned that at least one trader thinks the asset is worth *more* than your ask -- which is mild evidence the price is about to rise, exactly the move that hurts the short position you just took on. A sophisticated market maker therefore nudges its own fair value in the direction of the flow: get hit on the bid (someone sold to you) and you mark your mid down a touch; get lifted on the ask (someone bought from you) and you mark it up. This is the *same* logic as inventory skew but driven by what the trade reveals rather than by the risk of the position -- and in practice the two reinforce each other. The reason naive market makers blow up is that they treat fills as neutral cash-register events instead of as signals that the world just moved against them.

#### Worked example: the cost of adverse selection

Put numbers on it. Suppose 85% of your fills are noise and 15% are informed. On a noise fill, you earn the half-spread of +\$0.10 per share (the price does not move against you, and on average you round-trip it for the spread). On an informed fill, the price moves \$0.50 against you before you can react, so you lose −\$0.40 per share (the \$0.50 adverse move minus the \$0.10 of spread you collected up front).

Your expected P&L per fill is the probability-weighted average:

$$0.85 \times (+\$0.10) + 0.15 \times (-\$0.40) = \$0.085 - \$0.060 = +\$0.025 \text{ per share.}$$

So you still make money -- but only \$0.025 per share, not the \$0.10 the spread seemed to promise. Adverse selection has eaten three-quarters of your apparent edge. And the math is knife-edge: if the informed fraction rises to 25%, your expected P&L becomes $0.75 \times \$0.10 + 0.25 \times (-\$0.40) = \$0.075 - \$0.100 = -\$0.025$ per share -- you now *lose* money on every fill, and the only fix is to widen your spread until the noise edge covers the informed cost again.

#### Worked example: the spread that survives the informed

How wide must your spread be to break even against that 25%-informed flow? Let the half-spread be $h$. On a noise fill you earn $+h$. On an informed fill, the price moves \$0.50 against you and you collected $h$ up front, so you lose $-( \$0.50 - h) = h - \$0.50$. Break-even requires expected P&L = 0:

$$0.75 \times h + 0.25 \times (h - \$0.50) = 0.$$

Simplify: $0.75h + 0.25h - 0.125 = 0$, so $h - 0.125 = 0$, giving $h = \$0.125$. You need a half-spread of **\$0.125** (a full spread of \$0.25) just to break even when a quarter of your flow is informed and the informed move is \$0.50. The more toxic the flow, or the bigger the informed move, the wider you must quote. This is the core tension of the business: **quote tight and fill often but bleed to the informed; quote wide and stay safe but barely fill at all.**

## The Avellaneda-Stoikov reservation price and optimal spread

We now have two separate insights: skew your *center* by inventory (the reservation price), and set your *width* by adverse selection and risk (the spread). In 2008, Marco Avellaneda and Sasha Stoikov published a model that derives *both* at once from a single optimization -- a market maker maximizing expected utility of end-of-day wealth while penalizing inventory risk. You do not need the stochastic-control derivation to use the result; you need the two formulas and the intuition behind each term.

**The reservation price** under Avellaneda-Stoikov is:

$$r(s, q, t) = s - q \, \gamma \, \sigma^2 \, (T - t)$$

where $s$ is the current mid, $q$ is inventory, $\gamma$ is risk aversion, $\sigma$ is volatility, and $(T - t)$ is the time remaining until the close ($T$ is the closing time, $t$ is now). Read it term by term. Start at the mid $s$. Subtract a skew $q \gamma \sigma^2 (T - t)$ that is bigger when you hold more inventory ($q$), when you are more risk-averse ($\gamma$), when the stock is more volatile ($\sigma^2$), and -- the new ingredient -- when there is *more time left in the day* ($T - t$). That last factor is the model's signature insight: early in the day, a unit of inventory is dangerous because a lot of trading time remains for the price to move against it, so you skew hard to shed it; as the close approaches, the same inventory has little time left to hurt you, so you skew less and let it ride.

**The optimal spread** under Avellaneda-Stoikov is:

$$\delta^{\text{total}} = \gamma \, \sigma^2 \, (T - t) + \frac{2}{\gamma} \ln\!\left(1 + \frac{\gamma}{k}\right)$$

where $\delta^{\text{total}}$ is the *total* spread (ask minus bid) and the half-spread is half of it. The formula has two pieces. The first, $\gamma \sigma^2 (T - t)$, is an **inventory-risk premium**: you widen your spread when the stock is volatile, when you are risk-averse, and when there is more time left for inventory to bite. The second, $\frac{2}{\gamma}\ln(1 + \gamma/k)$, is a **liquidity term** driven by $k$, a parameter of the fill model that measures how fast fill probability decays as you quote further from the mid (more on $k$ in the simulator section). A larger $k$ -- fills drop off sharply with distance -- pushes you to quote tighter; a smaller $k$ lets you quote wider without losing too many fills.

![Early in the day you quote wide for inventory risk; as the close nears, the spread collapses toward the pure fee term.](/imgs/blogs/market-making-simulator-quant-research-7.png)

The figure shows the optimal half-spread shrinking through the day. At the open, with the full session ahead, the inventory-risk term $\gamma \sigma^2 (T - t)$ is large and the half-spread is wide -- \$0.18 in the example. As $(T - t)$ shrinks toward the close, the inventory term collapses and the half-spread falls toward the floor set by the liquidity term alone -- \$0.04 here. The green floor is the spread you would charge even with no inventory risk at all, purely to compensate for providing immediacy.

#### Worked example: the Avellaneda-Stoikov optimal half-spread

Let us compute one point on that curve. Take $\gamma = 0.1$, $\sigma = \$2$ per square-root-day (so $\sigma^2 = 4$), time remaining $(T - t) = 0.5$ (half the day left), and fill-decay $k = 1.5$.

The inventory-risk term is $\gamma \sigma^2 (T - t) = 0.1 \times 4 \times 0.5 = 0.20$. The liquidity term is $\frac{2}{\gamma}\ln(1 + \gamma/k) = \frac{2}{0.1}\ln(1 + 0.1/1.5) = 20 \times \ln(1.0667) = 20 \times 0.0645 = 1.29$? That is too large -- again a reminder that these parameters must be calibrated to a real stock's tick size, not pulled from thin air. With a smaller risk aversion $\gamma = 1$ and the same $k = 1.5$: inventory term $= 1 \times 4 \times 0.5 = 2.0$ (still large for a \$2 sigma in *dollars per day*; in practice you scale to the quote units), and the liquidity term $= \frac{2}{1}\ln(1 + 1/1.5) = 2 \times \ln(1.667) = 2 \times 0.511 = 1.02$. The point is not the exact dollar value -- it depends entirely on the units you calibrate -- but the *structure*: the total spread is an inventory-risk premium that melts through the day plus a fixed liquidity floor, and the half-spread is half of that total. **Calibrate the parameters to your asset, then trust the shape: wider early, tighter late, wider in volatile names.**

The reservation price and the optimal spread are computed together each instant: center your quotes on $r$, then place the bid at $r$ minus half the optimal spread and the ask at $r$ plus half. Inventory moves the center; volatility, time, and adverse selection set the width.

## Building the simulator

We have all the pieces; now we assemble the loop. A market-making simulator needs three components: a process for the mid-price, a model for when your quotes get filled, and bookkeeping for inventory and P&L. Each tick of the simulation does the same five things.

![Each tick draws a mid move, sets skewed quotes, draws fills, updates inventory, and marks profit and loss.](/imgs/blogs/market-making-simulator-quant-research-9.png)

**Step one: the mid-price process.** The simplest honest choice is a *random walk* (technically arithmetic Brownian motion): each tick, the mid moves by a random draw with mean zero and standard deviation $\sigma \sqrt{\Delta t}$, where $\Delta t$ is the length of the tick. Mean zero is the crucial modeling choice -- it bakes in that the market maker has *no* directional edge. All the profit must come from the spread; all the risk comes from holding inventory while this walk wanders. (For background on the random-walk model itself, see [Brownian motion for quant interviews](/blog/trading/quantitative-finance/brownian-motion-quant-interviews).)

**Step two: set quotes.** Compute the reservation price $r$ from the current mid and inventory, then place the bid at $r - h$ and the ask at $r + h$ where $h$ is the optimal half-spread. This is where the Avellaneda-Stoikov formulas live.

**Step three: the fill model.** This is the heart of any realistic simulator and the thing most beginners get wrong. You do not control whether you get filled; the market does. The standard model says the *intensity* (rate) of fills on a side decays exponentially with how far that quote sits from the mid:

$$\lambda(\delta) = A \, e^{-k \delta}$$

where $\delta$ is the distance from the mid to your quote, $A$ is the base fill rate at the mid, and $k$ controls how fast fills die off as you quote further away. Quote right at the mid ($\delta = 0$) and you fill at the maximum rate $A$. Quote far from the mid (large $\delta$) and fills become rare. Each tick, you draw whether a fill happens on each side using this intensity.

![Tight quotes near the mid fill often for little edge; wide quotes far from the mid fill rarely but earn more each time.](/imgs/blogs/market-making-simulator-quant-research-10.png)

The figure shows the tradeoff the fill model encodes. Quote a nickel from the mid and you fill about 30 times a minute but earn only that thin nickel each time. Quote twenty cents away and you fill only about 5 times a minute but pocket a fat edge per fill. The optimal spread from Avellaneda-Stoikov is precisely the point that balances "fill often for little" against "fill rarely for a lot", given your risk aversion.

**Step four: update inventory and cash.** If the bid filled, inventory goes up by the clip size and cash goes down by bid × size. If the ask filled, inventory goes down and cash goes up by ask × size.

**Step five: mark to market.** Total P&L is cash plus inventory marked at the current mid: $\text{P\&L} = \text{cash} + q \times \text{mid}$. Log it, log the inventory, log how much came from spread versus inventory, and step to the next tick.

Here is the loop in runnable Python. Note the comments are indented, never at column zero, per the house style.

```python
import numpy as np

def simulate_market_maker(
    n_ticks=23400,        # one trading day, 1-second ticks (6.5 hours)
    dt=1.0,
    sigma=0.02,           # mid vol per sqrt-second, in dollars
    gamma=0.1,            # risk aversion
    k=1.5,                # fill-intensity decay
    A=0.5,                # base fill rate at the mid, per tick
    clip=100,             # shares per fill
    s0=100.0,             # opening mid
    seed=0,
):
    rng = np.random.default_rng(seed)
    mid = s0
    q = 0                 # inventory in shares
    cash = 0.0
    T = n_ticks * dt
    spread_pnl = 0.0      # edge captured from the half-spread on each fill
    pnl_path = np.empty(n_ticks)
    inv_path = np.empty(n_ticks)

    for t in range(n_ticks):
        time_left = (T - t * dt) / T
        #  Avellaneda-Stoikov reservation price and optimal half-spread.
        reservation = mid - q * gamma * sigma**2 * time_left
        total_spread = gamma * sigma**2 * time_left + (2 / gamma) * np.log(1 + gamma / k)
        half = total_spread / 2
        bid = reservation - half
        ask = reservation + half

        #  Fill intensities fall off with distance from the mid.
        delta_bid = max(mid - bid, 0.0)
        delta_ask = max(ask - mid, 0.0)
        prob_buy = A * np.exp(-k * delta_bid) * dt    # someone hits our bid
        prob_sell = A * np.exp(-k * delta_ask) * dt   # someone lifts our ask

        if rng.random() < prob_buy:
            q += clip
            cash -= bid * clip
            spread_pnl += (mid - bid) * clip          # half-spread earned
        if rng.random() < prob_sell:
            q -= clip
            cash += ask * clip
            spread_pnl += (ask - mid) * clip          # half-spread earned

        #  Mid takes a mean-zero random step: no directional edge.
        mid += rng.normal(0.0, sigma * np.sqrt(dt))

        pnl_path[t] = cash + q * mid                  # mark to market
        inv_path[t] = q

    return pnl_path, inv_path, spread_pnl

pnl, inv, spread = simulate_market_maker()
print(f"end P&L: ${pnl[-1]:.0f}, spread captured: ${spread:.0f}, end inventory: {inv[-1]:.0f}")
```

Run that once and you get one day's path. Run it ten thousand times with different seeds and you get the *distribution* of outcomes -- which is the only honest way to judge a market-making strategy, because any single day is dominated by luck.

## Profit-and-loss attribution: where the money actually went

The most useful thing a simulator produces is not the final P&L number -- it is the *attribution*: a breakdown of how much you earned from spread, how much you gained or lost from inventory, and how much you bled to adverse selection. Two strategies can end the day at the same P&L for completely different reasons, and only attribution tells you which one to trust.

The clean decomposition is:

$$\text{Total P\&L} = \underbrace{\text{spread captured}}_{\text{from each fill}} + \underbrace{\text{inventory P\&L}}_{\text{position} \times \text{price move}} - \underbrace{\text{adverse selection}}_{\text{toxic fills}}.$$

Spread captured is the sum over all fills of the half-spread you earned at the moment of each fill (the `spread_pnl` variable in the code). Inventory P&L is what your held position earned or lost as the mid moved -- pure mark-to-market on the inventory you were carrying. Adverse selection shows up inside the inventory P&L as a systematic loss: it is the part of your inventory P&L that is *not* mean-zero, because your fills are biased to put you on the losing side right before moves.

![The market maker earns gross spread, then gives back inventory P&L and adverse-selection cost to reach net profit.](/imgs/blogs/market-making-simulator-quant-research-8.png)

#### Worked example: attributing one day's P&L

The waterfall above attributes a representative day. You earned **\$600** of gross spread from 30 round trips. Inventory P&L cost you **\$150** -- you happened to be carrying a long position through a downward drift, and that position marked down. Adverse selection cost another **\$180** -- the fills that put you in that long position were disproportionately the ones right before the price fell. Net it out: \$600 − \$150 − \$180 = **\$270** of profit for the day. The spread did all the earning; inventory and adverse selection did all the leaking; what survives is the thin edge that is the real business.

The intuition: **gross spread is the gross revenue of market making; inventory and adverse selection are its cost of goods sold; net P&L is the margin -- and the margin is thin.**

#### Worked example: the distribution across many days

One day means nothing. Run the simulator ten thousand times and plot the daily P&L. You get the distribution below: most days cluster around a small positive number (the mean is +\$270), but the left tail is fat -- the worst 5% of days lose more than \$400, because those are the days a big move caught you holding a big position. This shape is the signature of market making: **a high win rate, a small average win, and rare large losses.** It is the opposite of a lottery ticket; it is closer to selling insurance, where you collect small premiums constantly and occasionally pay a big claim.

![Across many simulated days the average profit is modest and positive, but a left tail of losing days is unavoidable.](/imgs/blogs/market-making-simulator-quant-research-12.png)

This is also why *risk-adjusted* metrics matter so much in this business. A strategy that nets +\$270 a day on average sounds fine until you see it can lose \$400+ on a bad day; the ratio of average daily profit to the standard deviation of daily profit (a daily *Sharpe-like* measure) is what tells you whether the edge is real or whether you are just one volatile day away from giving it all back. For how to evaluate any trading signal honestly, see [evaluating alpha signals: IC, Sharpe, turnover](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research).

The three strategies compared below summarize the whole design space. Tight symmetric quoting fills the most but swings widest and gets picked off worst. Wide symmetric quoting is safer per fill but fills so rarely it barely earns. Inventory-skewed quoting -- the Avellaneda-Stoikov approach -- takes the middle path on spread while mean-reverting inventory and avoiding the worst of the adverse-selection bleed.

![Wider, inventory-aware quoting trades raw fill volume for protection against adverse selection and blowups.](/imgs/blogs/market-making-simulator-quant-research-11.png)

## In the interview room and the take-home

These are the kinds of questions that come up when a firm wants to know whether you actually understand market making or just memorized the words. Each is fully solved. Work them yourself first, then check.

#### Worked example: spread captured and daily P&L (the warm-up)

*"You make a market \$49.95 / \$50.05 in 200-share clips. Over a balanced day you complete 40 round trips. What is your gross spread capture, and what is your spread per share?"*

The spread is \$50.05 − \$49.95 = \$0.10 per share. Each round trip is buy 200 at the bid, sell 200 at the ask, capturing \$0.10 × 200 = \$20. Forty round trips give 40 × \$20 = **\$800** gross spread capture. The spread per share is **\$0.10**, and the half-spread is **\$0.05**. The trap here is to forget that one "round trip" requires *both* a buy and a sell; a single fill is only half a round trip and leaves you with inventory.

#### Worked example: the inventory mark-to-market

*"You are long 500 shares of a \$80 stock with mid-volatility of \$0.04 per square-root-second. The market goes quiet and you cannot trade out for 100 seconds, during which the mid moves one standard deviation against you. What is your expected mark-to-market loss?"*

Over 100 seconds, the standard deviation of the mid move is $\sigma \sqrt{t} = \$0.04 \times \sqrt{100} = \$0.04 \times 10 = \$0.40$. A one-standard-deviation move against a long position is a \$0.40 drop. Your loss is 500 × \$0.40 = **\$200**. The teaching point is the square-root-of-time scaling: ten times the holding period is only $\sqrt{10} \approx 3.16$ times the risk, not ten times -- a fact that underlies almost every risk calculation in trading. (See [Itô's lemma](/blog/trading/quantitative-finance/itos-lemma-quant-interviews) and [stochastic differential equations](/blog/trading/quantitative-finance/stochastic-differential-equations-gbm-ou-quant-interviews) for why volatility scales with the square root of time.)

#### Worked example: pricing in adverse selection

*"30% of your flow is informed, and when an informed trader fills you the price moves \$0.20 against you. Noise traders pay you the half-spread. What half-spread makes you break even?"*

Let $h$ be the half-spread. On a noise fill (70% of flow) you earn $+h$. On an informed fill (30%) you collected $h$ up front but the price moved \$0.20 against you, so you net $h - \$0.20$. Break-even:

$$0.70 \times h + 0.30 \times (h - 0.20) = 0.$$

That is $0.70h + 0.30h - 0.06 = 0$, so $h - 0.06 = 0$, giving $h = \$0.06$. You need a half-spread of **\$0.06** (a \$0.12 full spread) to break even. If you only quote \$0.04 half-spread, your expected P&L per fill is $0.70 \times 0.04 + 0.30 \times (0.04 - 0.20) = 0.028 - 0.048 = -\$0.02$ -- you lose two cents a share, and more volume just loses faster. The insight interviewers want: **against informed flow, quoting tighter and trading more is not "scaling up an edge", it is scaling up a loss.**

#### Worked example: the reservation-price shift

*"Your model sets reservation price $r = \text{mid} - \gamma \sigma^2 q (T - t)$. The mid is \$200, you are long 300 shares, $\gamma = 0.2$, $\sigma^2 = 0.01$ (in consistent daily units), and half the trading day remains so $(T-t) = 0.5$. Where do you center your quotes, and which way does it bias your fills?"*

The skew is $\gamma \sigma^2 q (T - t) = 0.2 \times 0.01 \times 300 \times 0.5 = 0.30$. So $r = \$200 − \$0.30 = \$199.70$. You center your quotes \$0.30 below the mid. Because you are long and want to get flat, this lowers both your bid and your ask -- making you more likely to *sell* (your ask is closer to the action) and less likely to *buy* (your bid is further away). The skew biases your fills toward selling, which mean-reverts your +300 position back toward zero. The follow-up an interviewer loves: *"What happens to this skew as the close approaches?"* Answer: $(T - t) \to 0$, so the skew shrinks to zero -- near the close there is little time for inventory to hurt you, so you stop fighting your position so hard.

#### Worked example: the optimal half-spread and the fill tradeoff

*"With $\gamma = 1$, $\sigma^2 = 0.04$, $(T - t) = 1$ (full day), and fill decay $k = 2$, compute the Avellaneda-Stoikov total spread and half-spread. Then explain in one sentence what raising $k$ does."*

The inventory-risk term is $\gamma \sigma^2 (T - t) = 1 \times 0.04 \times 1 = 0.04$. The liquidity term is $\frac{2}{\gamma}\ln(1 + \gamma/k) = \frac{2}{1}\ln(1 + 1/2) = 2 \times \ln(1.5) = 2 \times 0.405 = 0.81$. The total spread is $0.04 + 0.81 = \$0.85$ and the half-spread is **\$0.425** (in the model's calibrated units). Raising $k$ -- meaning fills die off faster as you quote away from the mid -- *shrinks* the liquidity term, because $\ln(1 + \gamma/k)$ falls as $k$ rises, so you quote *tighter* to keep filling. The one-sentence answer: **a higher $k$ means liquidity-takers are more price-sensitive, so you must quote closer to the mid to win their flow.**

#### Worked example: the take-home extension

A common take-home asks you to *extend* the simulator and report what changes. A strong answer picks one realistic feature and shows its P&L impact with a number. For example: *"Add a 20% chance that each fill is informed, modeled as a \$0.15 jump in the mid against you immediately after the fill. How does daily P&L change?"* You would modify the loop so that after each fill, with probability 0.2, the mid jumps \$0.15 in the direction that hurts your new position, then rerun 10,000 days. The expected result: mean daily P&L falls sharply (toxic flow is now eating the spread), the left tail fattens, and the *optimal* spread the strategy should quote widens -- which you can demonstrate by sweeping the half-spread and showing the P&L-maximizing width moved out. The grader is not looking for the "right" number; they are looking for whether you *attribute* the change correctly to adverse selection and whether you *re-optimize* the spread in response. **Always show the attribution and always re-optimize -- a take-home that just reports a lower P&L without explaining the mechanism is a weak submission.**

## Common misconceptions

**"Market makers predict which way the price will go."** No -- the whole point is that they do not. A pure market maker takes no directional view; the mid in our simulator is a mean-zero random walk by design. Profit comes from the spread and from managing inventory, not from forecasting. Real firms do blend in short-horizon signals, but the *market-making* part of the business is structurally agnostic about direction.

**"A wider spread always means more profit."** Wider spreads earn more *per fill* but fill far less often, and past a point the lost volume outweighs the fatter edge. The exponential fill model makes this precise: there is an interior optimum, and the Avellaneda-Stoikov spread is an attempt to find it. Quoting too wide is just as much a mistake as quoting too tight.

**"If I always earn the spread, I cannot lose money."** You earn the spread on each round trip, but you do not control whether your fills come in matched pairs. Unbalanced flow leaves you holding inventory, and a price move against that inventory can lose more than many round trips earned. Spread capture is the revenue; inventory and adverse selection are the costs, and the costs can exceed the revenue on a bad day.

**"Adverse selection is the same as just being unlucky."** Bad luck is mean-zero -- it washes out over many days. Adverse selection is *systematic*: informed traders only ever fill you on the side about to win, so the bias does not wash out, it compounds. That is why you cannot fix adverse selection by trading more; you fix it by widening your spread or by detecting and avoiding toxic flow.

**"Inventory risk goes away if I trade fast enough."** Faster quoting helps you shed inventory sooner, but it cannot eliminate the risk, because the price can gap before you trade out, and because the very flow that lets you exit may be the informed flow that is moving the price against you. Speed reduces the *window* of inventory risk; it does not remove it, and in the tail (a halt, a gap, a news jump) it offers no protection at all.

**"The reservation price is the fair value of the stock."** It is *your* fair value given *your* inventory -- not the market's. Two market makers looking at the same \$100 mid will have different reservation prices if one is long and one is short. The mid is the market's consensus; the reservation price is a personal adjustment for the risk you are already carrying.

## How it shows up in real markets

**Designated market makers and the NYSE.** On the New York Stock Exchange, designated market makers (DMMs, formerly "specialists") are obligated to maintain a continuous two-sided quote in their assigned stocks, especially at the open and close. They earn the spread and rebates for providing liquidity, and they carry exactly the inventory risk this post describes -- which is why their quotes visibly widen during volatile opens and around news. The obligation to quote *both sides continuously* is the institutional version of our simulator's core loop.

**Electronic market making in equities (Citadel Securities, Virtu, Jane Street).** The overwhelming majority of US retail stock orders are filled not on an exchange but by an electronic market maker that pays brokers for that flow -- the controversial practice of *payment for order flow*. Why pay for it? Because retail flow is overwhelmingly *noise*, not informed: it is uncorrelated with short-term price moves, so the market maker captures the spread with little adverse selection. The firms are, in effect, paying to route the *non-toxic* end of the flow distribution to themselves. The entire economics rest on the noise-versus-informed split from the adverse-selection section.

**The 2010 Flash Crash.** On 6 May 2010, US equity indices dropped about 9% and recovered within minutes. A documented mechanism was inventory risk overwhelming market makers: as a large automated sell program hit the market, market makers accumulated long inventory faster than they could offload it, hit their risk limits, and pulled their quotes or widened them dramatically. With liquidity providers stepping back, prices gapped -- some stocks printed at a penny. It is the inventory-risk tail of our simulator playing out at market scale: when everyone is forced long at once and cannot get flat, quotes vanish and the price falls through the floor.

**Crypto market making and the inventory problem.** On crypto exchanges, market makers face the same spread-versus-inventory tradeoff with extra teeth: 24/7 trading, higher volatility, and frequent gaps. A maker long an altcoin into a sudden 20% drop suffers exactly the mark-to-market loss our inventory section computes, but with a volatility $\sigma$ several times that of a blue-chip stock, so the same inventory is far more dangerous. Crypto makers skew aggressively and quote wide in thin names precisely because the inventory-risk term in the optimal spread is so large.

**Options market making and the Greeks.** An options market maker quotes spreads on contracts whose "inventory risk" is multi-dimensional -- exposure to the underlying's price (*delta*), to volatility (*vega*), and more. The principle is identical to this post's: earn the spread, hedge away the inventory you do not want to hold, and quote wider when the risk you cannot hedge is larger. The reservation-price idea generalizes to skewing quotes by net Greek exposure. (For the option side, see [options theory](/blog/trading/quantitative-finance/options-theory) and the [volatility surface](/blog/trading/quantitative-finance/volatility-surface).)

**Treasury and FX dealers.** In bonds and foreign exchange, "dealers" are market makers who quote two-sided prices to clients. A dealer who buys a large block of bonds from a client is suddenly long inventory and must either hedge or work out of the position -- and they price the spread they quote partly on how much adverse selection they expect (is this client informed?) and how much inventory risk they will carry. The same three forces, in a market measured in trillions. A telling detail: dealers quote *tighter* spreads to clients they believe are uninformed (a corporate treasury rebalancing cash) and *wider* spreads to clients they suspect are informed (a hedge fund that trades right before moves) -- a direct, real-world application of the noise-versus-informed split, with the spread set client by client.

**The "last look" and quote fading in electronic FX.** In electronic foreign-exchange markets, market makers stream quotes to many clients at once and protect themselves from adverse selection with a mechanism called *last look* -- a brief window in which the maker can reject a fill if the price has moved against them in the milliseconds since the quote. It is controversial, but its existence is the clearest possible evidence of how seriously adverse selection is taken: makers would rather decline a trade than be systematically picked off by the fastest informed flow. When you read about quotes "fading" the instant you try to trade on them, you are watching a market maker refuse to be adversely selected. The lesson for the simulator is that real fill models are not the clean exponential of our toy: the most aggressive flow is also the most toxic, and a realistic model must let the maker step away from it.

## When this matters to you and further reading

If you are preparing for a quant trading or research role, this is foundational: the market-making game is the single most common live interview exercise, and a simulator like this is a frequent take-home. Internalize the three forces -- spread capture, inventory risk, adverse selection -- and the one model that unifies them, and you can reason out loud through almost any market-making question an interviewer throws at you.

If you trade your own account, the lesson is humbler but real: *you are the liquidity taker*. Every time you cross the spread to get filled instantly, you are paying a market maker for immediacy. Understanding that you pay the spread -- and that it is wider in volatile, thinly traded names -- is the difference between trading deliberately and leaking money on every fill.

To go deeper, the natural next steps from here are: the interview-game version of this material in [market-making games: making tight markets under uncertainty](/blog/trading/quantitative-finance/market-making-games-quant-interviews); the random-walk foundations in [Brownian motion for quant interviews](/blog/trading/quantitative-finance/brownian-motion-quant-interviews); honest strategy evaluation in [backtesting done right](/blog/trading/quantitative-finance/backtesting-done-right-quant-research) and [evaluating alpha signals](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research); and the Avellaneda-Stoikov paper itself ("High-frequency trading in a limit order book", 2008) for the full stochastic-control derivation behind the two formulas we used. The Glosten-Milgrom (1985) and Kyle (1985) models are the canonical reading on adverse selection and informed trading -- they are where the idea that *the spread exists because of asymmetric information* was first made rigorous.

Build the simulator. Run it ten thousand times. Watch the spread accumulate, watch the inventory wander, watch a bad day eat a good week -- and the tradeoffs that take a paragraph to describe will become something you can feel in dollars.

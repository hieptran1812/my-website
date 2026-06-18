---
title: "AMMs vs Arbitrageurs: Toxic Flow and Loss-Versus-Rebalancing"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Why an automated market maker is a passive sitting duck that arbitrageurs pick off every time the price moves, and how to measure that loss with impermanent loss and loss-versus-rebalancing."
tags: ["game-theory", "trading", "amm", "uniswap", "impermanent-loss", "loss-versus-rebalancing", "arbitrage", "market-making", "adverse-selection", "defi"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — An automated market maker (AMM) is a market maker that has agreed, in advance and in code, never to use any of a market maker's defenses. It quotes a price from a fixed formula, it does not skew that price to shed unwanted inventory, and it does not widen when the flow turns informed. So whenever the real price moves, an arbitrageur trades against the stale pool and pockets the difference — and that profit is, dollar for dollar, the liquidity provider's loss.
>
> - **The arbitrageur is the informed counterparty.** The pool quotes the same price to everyone; the arb is the trader who shows up *only* when that price is wrong, always trading the profitable direction. That is textbook adverse selection.
> - **Impermanent loss** measures the gap between the pool and just holding. For a constant-product pool, a price ratio $r$ gives $\text{IL} = \frac{2\sqrt{r}}{1+r} - 1$ — a 2x or a halving both cost about 5.7%.
> - **Loss-versus-rebalancing (LVR)** is the sharper framing: it is the arbitrageur's profit, marked against the true price, and it equals the LP's adverse-selection cost made precise.
> - **The one rule:** an honest LP yield is *fees minus LVR*. Providing liquidity is only +EV when the fees a pool earns are larger than the LVR its volatility implies.

A trader I know once described providing liquidity on a decentralized exchange as "earning yield while you sleep." The pool collects a fee on every swap, the fees compound, and the dashboard shows a tidy annual percentage. It feels like a savings account that pays you for owning two assets you wanted to own anyway.

Then he watched, block by block, what actually happened to his position during a single volatile afternoon. Ethereum's price ran up about 9% on a centralized exchange in a couple of hours. Every few blocks, a bot reached into his pool, bought the now-underpriced ETH the pool was still quoting too cheaply, and sold it on the exchange where it was already worth more. By the end of the day the fees had indeed accrued — and his position was worth meaningfully *less* than if he had simply held the two assets in his wallet and done nothing. The "yield while you sleep" had a counterparty, and that counterparty had been wide awake, picking his pocket on a schedule.

This post is about who that counterparty is, why the AMM hands them the money so reliably, and how to put a number on the bleed. The figure below is the mental model for the whole post: in calm markets the pool and the outside world agree, so there is nothing to exploit; the instant the outside price moves, the pool's quote goes stale, an arbitrageur trades against it, and the realignment is paid for out of the liquidity provider's pocket.

![AMM as a sitting-duck market maker, before and after a price move](/imgs/blogs/amms-vs-arbitrageurs-toxic-flow-and-loss-versus-rebalancing-1.png)

The deep claim of this series applies cleanly here: a trade is a strategic interaction, not a bet against nature. The LP is not losing to "volatility" as an abstract force. The LP is losing to a specific, identifiable opponent — the arbitrageur — who is reasoning one level deeper, who only ever trades when they are right, and whom the AMM has been programmed to serve at the worst possible price. Once you see the AMM as a market maker with its hands tied, the rest is arithmetic.

## Foundations: what an AMM is and why it can't defend itself

Let's build every piece from zero. By the end of this section you'll be able to derive the pool's price by hand and see exactly where the leak is.

### What a market maker is, and what it normally does to survive

A *market maker* (MM) is anyone who stands ready to both buy and sell an asset continuously, quoting a *bid* (the price they'll buy at) and an *ask* (the price they'll sell at). The gap between them is the *bid-ask spread* — the market maker's gross margin. You, the trader, cross the spread when you want to transact right now; that crossing is the MM's pay.

But the spread is not free money, because some of the people trading against a market maker know something the MM doesn't. This is *adverse selection*: a market maker's quotes get hit disproportionately by traders who are right. If a stock is about to jump on news you haven't seen, you'll lift my ask before I can pull it. I sold low to someone who knew. This is the central problem of market making, and a real MM has three defenses against it, all of which a constant-product AMM lacks:

1. **Inventory skew.** If I'm a real MM and I've just been forced to buy a lot of an asset, I tilt my quotes lower to encourage someone to buy it back from me. I manage my inventory by moving my price.
2. **Adverse-selection widening.** When I sense that the flow hitting me is informed — when fills are coming fast and all in one direction — I widen my spread so that the price the informed trader pays compensates me for the ones who pick me off.
3. **The right to refuse.** I can pull my quotes entirely, fade size, or simply stop quoting a name when it looks dangerous.

We covered the theory of how a spread is set to survive adverse selection in [the bid-ask spread as an adverse-selection game](/blog/trading/game-theory/the-bid-ask-spread-as-an-adverse-selection-game-glosten-milgrom) — the Glosten-Milgrom model, where the spread exists precisely to cover the losses to informed traders. Keep that model in mind, because the AMM is going to violate every lesson in it.

### What an AMM is, and the one formula that runs it

An *automated market maker* is a smart contract — a program living on a blockchain — that plays the role of market maker without a human or an order book. Instead of a book of resting buy and sell orders, it holds a *pool* of two assets and quotes a price by a fixed mathematical rule. Anyone can swap one asset for the other against the pool at the price the rule dictates. Anyone can also become a *liquidity provider* (LP) by depositing both assets into the pool, in exchange for a share of the fees and a share of the pool's value.

The most common rule, used by Uniswap v2 and a hundred clones, is the *constant-product* formula. The pool holds a reserve $x$ of one token (say ETH) and a reserve $y$ of another (say a dollar stablecoin, USDC). The contract enforces one invariant:

$$x \cdot y = k$$

where $k$ is a constant. When you trade, you add to one reserve and remove from the other in whatever amounts keep the product $x \cdot y$ equal to $k$. That single constraint defines the entire price curve. The pool's *marginal price* — the price for an infinitesimally small trade — is simply the ratio of the reserves:

$$P = \frac{y}{x}$$

That's it. There is no inventory model, no volatility input, no view on who is trading. The price is a deterministic function of the reserves, and the reserves only change when someone trades. The pool quotes the same curve to a retail user buying \$50 of ETH for their wallet and to a \$10 million arbitrage bot. It has, by design, surrendered all three of a real market maker's defenses.

#### Worked example: deriving the pool price by hand

Suppose a pool starts with $x_0 = 100$ ETH and $y_0 = 200{,}000$ USDC. Then $k = 100 \times 200{,}000 = 20{,}000{,}000$, and the quoted price is $P_0 = y_0 / x_0 = 200{,}000 / 100 = \$2{,}000$ per ETH. The pool is "worth" $x_0 \times P_0 + y_0 = 100 \times \$2{,}000 + \$200{,}000 = \$400{,}000$, split 50/50 between the two assets.

Now suppose someone wants to buy 1 ETH from the pool. After the trade the pool holds $99$ ETH, so to keep the product at $20{,}000{,}000$ it must hold $y = 20{,}000{,}000 / 99 = \$202{,}020.20$ in USDC. The buyer therefore pays $\$202{,}020.20 - \$200{,}000 = \$2{,}020.20$ for that 1 ETH — slightly more than the \$2,000 marginal price, because buying moves the price up along the curve. That extra \$20.20 over the starting price is *slippage*, and it is the AMM working exactly as intended for an ordinary trade.

The intuition: the pool's price is just the ratio of what it holds, and every trade rebalances that ratio, so trading *into* the pool always pushes the price against you. This is fair and benign for a retail swap — the trouble starts only when the *outside* price moves first and leaves the pool's price stale.

### The bonding curve, and why the pool always holds 50/50 by value

It helps to see the $x \cdot y = k$ rule as a *curve*. Plot the ETH reserve $x$ on one axis and the USDC reserve $y$ on the other, and the set of all allowed states is a hyperbola — the *bonding curve*. The pool always sits somewhere on this curve. Every trade slides the pool to a new point on the curve; the price at any point is the (negative) slope of the curve there. When ETH is expensive relative to its starting price, the pool slides toward holding little ETH and lots of USDC; when ETH is cheap, the pool slides the other way. The curve is the pool's entire memory and its entire pricing engine.

A consequence worth pinning down: at every point on a constant-product curve, the *value* of the two reserves is split exactly 50/50. If the pool holds $x$ ETH at price $P = y/x$, then the ETH is worth $x \cdot P = x \cdot y / x = y$ — exactly the USDC reserve. So the dollar value of the ETH side always equals the dollar value of the USDC side. This is why a constant-product LP is, in effect, holding a continuously rebalanced 50/50 portfolio: the formula forces it. That forced rebalancing is the source of both the impermanent loss (you're always selling the winner) and the comforting property that the pool can never fully run out of either asset — the curve never touches the axes.

This 50/50-by-value fact is also what makes the impermanent-loss algebra short, as we'll see: both the "hold" benchmark and the "pool" value have clean closed forms in the price ratio, so their difference does too.

### Where the leak is, in one sentence

A market maker's whole job is to update its quotes faster than informed traders can exploit the old ones. The AMM updates its quote *only when someone trades against it*. So when the real price moves on a fast centralized exchange (CEX), the pool's quote does not move — until an arbitrageur trades against the stale quote to capture the gap, which is the very act that updates it. The AMM cannot move first. It can only be moved, and only by someone profiting from the move.

That is the entire thesis. Everything below is making it precise and putting dollar signs on it.

## How the arbitrageur picks off the pool

Let's meet the opponent properly. An *arbitrageur* in this context is a trader (almost always a bot) that watches both the AMM pool and a deep, fast external market — a major CEX, or another, more liquid pool. When the two prices diverge, the arb trades against the cheaper one and unwinds against the dearer one, banking the spread. They take essentially no directional risk; they are harvesting a price *difference*, not betting on a price *direction*.

This is the same family of trader we met in [adverse selection and the winner's curse](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news): the counterparty whose fill is bad news for you precisely because they only trade when they're right. Against an AMM, the arbitrageur's edge is mechanical and almost risk-free, because the pool's mispricing is observable and the pool cannot refuse the trade.

Why "almost risk-free"? Three properties stack in the arbitrageur's favor, and each one is a defense the AMM lacks. First, the mispricing is *observable* — the pool publishes its reserves on-chain, so the arb can compute the pool's exact quote and compare it to the CEX price with no guesswork. A real market maker's quotes are also visible, but a real MM moves them; the pool can't. Second, the trade is *atomic* in the sense that the arb can size it precisely — they solve for the exact trade that drags the pool to the fair price and captures the maximum gap, no more, no less. There's no slippage surprise, because the curve is deterministic. Third, the arb takes *no inventory risk* on the legged trade if they unwind immediately on the CEX: they buy ETH from the pool and sell it on the exchange in the same logical instant, so they're never exposed to a directional move. They're not betting on ETH; they're harvesting a price gap that already exists. Put those together and the arbitrageur is closer to a risk-free skimmer than a speculator — and the skim comes out of the LP.

The contrast with a directional trader is worth stating plainly. A speculator who thinks ETH is going up takes on real risk — they can be wrong, and the price can fall. The arbitrageur against an AMM takes on essentially none of that risk; their "view" is just that \$2,200 on the CEX is more than \$2,000 in the pool, which is not a forecast but an observed fact. That's what makes the flow toxic in the precise sense of [adverse selection](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news): the LP is trading against someone with strictly better information and no risk, every single time the price moves.

### The realignment loop

Here is the loop that runs, block after block, for the life of the pool.

![The arbitrage-realignment loop from a price move to a booked LP loss](/imgs/blogs/amms-vs-arbitrageurs-toxic-flow-and-loss-versus-rebalancing-7.png)

Read it left to right. The external price moves. The pool's quote is now stale. The arbitrageur spots the gap and computes the exact trade that drags the pool back to the new fair price. They execute that trade against the pool — buying the underpriced asset or selling the overpriced one — and the pool realigns. The realignment is not free: the value the arbitrageur extracted comes straight out of the pool's reserves, and the LP owns the reserves. The loop closes with a booked loss for the LP that equals, to the penny, the arbitrageur's profit.

Crucially, the arbitrageur is the *only* party who can realign the pool, and they will only do it when there's a profit in it for them. So the LP doesn't get a kindly realignment that costs nothing; they get the realignment that maximizes the arb's take. The pool is always rebalanced at the worst price for the LP and the best price for the informed trader.

### The single move, computed end to end

Let's run the loop once with real numbers, on the pool from before: $100$ ETH and $200{,}000$ USDC, $k = 20{,}000{,}000$, quoting \$2,000.

#### Worked example: one arbitrage trade, from price move to LP loss

The external price jumps to \$2,200 — a 10% move up. The pool still quotes \$2,000, because no one has traded yet. Now the arbitrageur acts. To realign the pool to \$2,200, the new reserves must satisfy two conditions at once: the product is still $k$, and the new ratio $y/x$ equals \$2,200. Solving, $x_1 = \sqrt{k / 2200} = \sqrt{20{,}000{,}000 / 2200} = 95.346$ ETH, and $y_1 = \sqrt{k \times 2200} = \sqrt{44{,}000{,}000{,}000} = \$209{,}761.77$.

So the arbitrageur *buys* $100 - 95.346 = 4.654$ ETH out of the pool, and *pays in* $\$209{,}761.77 - \$200{,}000 = \$9{,}761.77$ of USDC for it. Their average price paid to the pool is $\$9{,}761.77 / 4.654 = \$2{,}097.62$ per ETH — somewhere between the stale \$2,000 and the new \$2,200, because they walk the price up the curve as they buy. They then sell those 4.654 ETH on the CEX at \$2,200. Their profit:

$$4.654 \times (\$2{,}200 - \$2{,}097.62) \approx \$476.46$$

Now the LP's side. Before the move, holding would have left the LP with 100 ETH and \$200,000. At the new \$2,200 price, *holding* is worth $100 \times \$2{,}200 + \$200{,}000 = \$420{,}000$. But the LP doesn't hold — they're in the pool, which now contains 95.346 ETH and \$209,761.77, worth $95.346 \times \$2{,}200 + \$209{,}761.77 = \$419{,}523.54$. The LP is worse off by $\$420{,}000 - \$419{,}523.54 = \$476.46$.

Look at the two numbers: the arbitrageur made \$476.46 and the LP lost \$476.46. They are the same number. The intuition: the arbitrageur's entire profit is the LP's entire loss — this is a zero-sum transfer wearing the costume of "providing liquidity," and the LP is on the losing side every single time the price moves.

That equality is not a coincidence of these numbers; it is the core identity of the whole topic, and it has a name we'll get to: loss-versus-rebalancing.

## Impermanent loss: the loss versus holding

Before LVR became the standard framing, the LP's pain had an older, fuzzier name: *impermanent loss* (IL). It answers a slightly different question — not "how much did the arbitrageur take?" but "how much worse off am I in the pool than if I'd just held the two assets?"

### Building the IL formula from x·y=k

Start from the constant-product invariant and ask: when the price ratio changes from its starting value to a new value, what fraction of value does the LP give up versus holding?

Let $r = P_\text{new} / P_\text{old}$ be the price ratio — $r = 1$ means no move, $r = 2$ means the price doubled, $r = 0.5$ means it halved. Two facts about a constant-product pool make the derivation short. First, the value the LP would have by *holding* the original $x_0, y_0$ scales as $(1 + r)/2$ relative to the starting value (half the value was in the asset whose price moved by $r$, half in the numéraire that didn't). Second, the value of the *pool*, which auto-rebalances to stay 50/50 by value, scales as $\sqrt{r}$. Taking the ratio of pool value to holding value and subtracting 1 gives the loss:

$$\text{IL}(r) = \frac{2\sqrt{r}}{1 + r} - 1$$

This is always less than or equal to zero, with equality only at $r = 1$. Any move in either direction costs the LP. The chart shows the shape.

![Impermanent loss as a function of the price ratio, computed from the constant-product formula](/imgs/blogs/amms-vs-arbitrageurs-toxic-flow-and-loss-versus-rebalancing-2.png)

The curve has three features worth internalizing. It is *symmetric in a logarithmic sense* — a doubling and a halving cost the same. It is *zero and flat at the center* — small moves cost very little, because $\sqrt{r}/(1+r)$ is flat near $r = 1$. And it *accelerates in the tails* — big moves hurt disproportionately. A 2x move costs about 5.7% of your value; a 4x move costs a full 20%.

#### Worked example: impermanent loss when ETH doubles

You deposit \$200 of ETH and \$200 of USDC into a pool, for \$400 total, when ETH is \$2,000. So you put in 0.1 ETH and \$200. The constant is $k = 0.1 \times 200 = 20$.

ETH then doubles to \$4,000, so $r = 2$. Plug into the formula: $\text{IL}(2) = \frac{2\sqrt{2}}{1 + 2} - 1 = \frac{2.828}{3} - 1 = -0.0572$, i.e. a 5.72% loss versus holding.

Let's verify it directly. If you'd held, you'd have 0.1 ETH now worth \$400 plus your \$200, for \$600. Inside the pool, the reserves rebalance: $x_1 = \sqrt{k / 4000} = \sqrt{20/4000} = 0.0707$ ETH and $y_1 = \sqrt{k \times 4000} = \sqrt{80{,}000} = \$282.84$. The pool is worth $0.0707 \times \$4{,}000 + \$282.84 = \$565.69$. Holding gives \$600; the pool gives \$565.69; you are down $\$600 - \$565.69 = \$34.31$, which is $34.31 / 600 = 5.72\%$. The formula and the direct computation agree.

The intuition: in a pool you are automatically selling the asset that's going up and buying the one that's going down, so you always end up holding *less* of the winner and *more* of the loser than you would by doing nothing — and that systematic "sell winners, buy losers" is the impermanent loss.

### Why "impermanent" is a dangerous word

The name suggests the loss evaporates if the price comes back, and arithmetically it does: if $r$ returns to 1, IL returns to zero. This is the source of endless LP cope — "it's only impermanent, I'll be fine when the price recovers."

But that framing hides the real cost, and this is exactly where the newer LVR framing earns its keep. The trip *out* to $r = 2$ and back to $r = 1$ was not free. On the way out, arbitrageurs bought your cheap ETH; on the way back, arbitrageurs sold it back to you expensively. The round trip generated real, *permanent* arbitrage profit at your expense even though your IL ended at zero. "Impermanent loss" measures only the endpoints; it is blind to the path, and the path is where the bleeding happens. To see the bleeding we need a measure that accrues on every move, not just on the net displacement. That measure is LVR.

## Loss-versus-rebalancing: the modern, sharper framing

*Loss-versus-rebalancing* (LVR, pronounced "lever") was introduced in a 2022 paper by Milionis, Moallemi, Roughgarden, and Zhang. It reframes the LP's loss not as a comparison to holding, but as a comparison to a smarter benchmark: a *rebalancing strategy* that holds the same position the pool holds at every instant, but trades it at the true external price rather than at the pool's stale price.

### The benchmark that isolates adverse selection

Here's the conceptual move. Impermanent loss compares the pool to *holding*, which conflates two different things: the cost of having a 50/50 rebalanced exposure at all, and the cost of trading that exposure at bad prices. LVR strips out the first and isolates the second — the pure adverse-selection cost.

The rebalancing benchmark is a hypothetical trader who, at every moment, holds exactly the same amounts of ETH and USDC the pool holds, but whenever they need to adjust that position, they trade at the *current external market price*, not at the pool's curve. The LP, in contrast, can only adjust by being arbitraged — i.e., by trading at a price the arbitrageur chose. LVR is the difference between these two, and it is exactly the arbitrageur's profit.

In the worked example above, the LP and the rebalancing benchmark both ended up holding 95.346 ETH and \$209,761.77. The only difference was the price at which the position got there: the benchmark would have sold/bought at \$2,200, but the LP transacted with the arb at an average of \$2,097.62. That price gap, applied to the 4.654 ETH that changed hands, *is* the \$476.46. LVR makes that the headline number.

![Loss-versus-rebalancing equals the arbitrageur's profit, computed from the pool math](/imgs/blogs/amms-vs-arbitrageurs-toxic-flow-and-loss-versus-rebalancing-3.png)

The chart computes LVR for a \$1,000,000 pool as a function of the external move multiple $m = P_\text{new} / P_\text{old}$. For a constant-product pool, a single move from $P_\text{old}$ to $P_\text{new} = m \cdot P_\text{old}$ hands the arbitrageur

$$\text{LVR}(m) = V_0 \left( \frac{1 + m}{2} - \sqrt{m} \right)$$

where $V_0$ is the pool's value before the move. This is zero only at $m = 1$ and strictly positive otherwise — every move, up or down, transfers value to the arb. Notice the shape: like IL, it's flat for tiny moves and accelerates for large ones, because the arbitrageur's edge grows with the size of the mispricing they get to exploit.

#### Worked example: LVR on a 20% move, and why the LP can't avoid it

Take a \$1,000,000 pool ($V_0 = \$1{,}000{,}000$) and a 20% upward move, $m = 1.2$. Then

$$\text{LVR}(1.2) = \$1{,}000{,}000 \left( \frac{1 + 1.2}{2} - \sqrt{1.2} \right) = \$1{,}000{,}000 \,(1.1 - 1.09545) = \$4{,}554.$$

So a single 20% move costs the LP \$4,554 in LVR. A 20% *down* move ($m = 0.8$) costs $\$1{,}000{,}000\,(0.9 - \sqrt{0.8}) = \$5{,}573$ — slightly more, because of the convexity of the curve. A 50% up move ($m = 1.5$) costs \$25,255.

Here is the part that traps LPs: there is *nothing the LP can do during the move to avoid this*. The LP is passive by construction. They cannot pull the quote, cannot skew, cannot widen. The only way to avoid the LVR on a move is to not be in the pool when it happens — which means timing volatility, which no one can do reliably. The intuition: LVR is the rent the pool pays to volatility, collected by arbitrageurs, and a passive LP has pre-agreed to pay it.

### The continuous-time version: LVR as a volatility tax

If you let the moves get small and frequent, the LVR per unit time has a beautifully simple form. The instantaneous LVR rate for a 50/50 constant-product pool is

$$\text{LVR rate} = \frac{\sigma^2}{8}$$

where $\sigma$ is the *annualized volatility* of the asset pair. Read it carefully: the LP's adverse-selection cost is proportional to *variance* (volatility squared). Double the volatility and you quadruple the bleed. This is the single most important number for an LP, because it tells you that liquidity provision is a short-volatility position — you are, in effect, selling a continuous strip of options to arbitrageurs, and getting paid in fees.

The connection to the [Glosten-Milgrom](/blog/trading/game-theory/the-bid-ask-spread-as-an-adverse-selection-game-glosten-milgrom) spread is exact in spirit. In that model the dealer sets a spread to cover expected losses to the informed; LVR is the AMM's version of those losses, except the AMM has no spread-setting mechanism to cover them, so the LP eats the cost and hopes the fee tier was set high enough. LVR is the AMM's adverse selection, made precise and continuous.

The $\sigma^2/8$ form also makes precise *why* providing liquidity behaves like selling options. An option seller is short *gamma* — they lose money when the underlying moves a lot in either direction, and the loss grows with the square of the move. The LP's LVR has exactly that signature: it's positive for any move, up or down, and it scales with variance. A constant-product LP is, mathematically, running a continuously-delta-hedged short-options book, and LVR is the premium they're *paying* to arbitrageurs for the privilege — except the LP is on the wrong side, paying the premium rather than collecting it. The fees are the LP's attempt to collect a premium back. So the LP's profit-and-loss is the eternal options-seller's question dressed in DeFi clothes: *does the premium I collect (fees) exceed the gamma cost I pay (LVR)?* If you've read about how an [options market maker thinks about the other side of your trade](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade), this is the same short-gamma bookkeeping — the AMM is just a short-gamma desk that never adjusts its quotes.

#### Worked example: the LVR rate as an annual bleed

Suppose you provide \$50,000 to an ETH/USDC pool with an annualized volatility of $\sigma = 70\% = 0.70$. The continuous LVR rate is $\sigma^2 / 8 = 0.70^2 / 8 = 0.49 / 8 = 0.06125$, i.e. about 6.1% a year. On \$50,000 of liquidity that's roughly $0.06125 \times \$50{,}000 = \$3{,}063$ flowing to arbitrageurs over a year, *before* a single dollar of fees. If the pool earns you a 9% fee yield, you net $9\% - 6.1\% = 2.9\%$, or about \$1,450 — a real but thin profit. If the pool earns 5% in fees, you net $5\% - 6.1\% = -1.1\%$, losing about \$550 even though the dashboard cheerfully showed a 5% "APY." The intuition: the advertised yield is the gross premium you collect; the $\sigma^2/8$ rate is the gamma you pay, and only the difference is yours to keep.

## The honest LP yield: fees minus LVR

Now we can write down the only equation an LP actually needs. The pool earns fees on every swap and pays LVR on every arbitrage. The LP's true expected return is the difference:

$$\text{LP yield} = \text{fee income} - \text{LVR}$$

Both sides scale with how much the asset moves and trades, but they scale *differently*, and that difference is the whole game.

### When is providing liquidity actually +EV?

Fee income depends on *volume*: more swaps, more fees. LVR depends on *volatility*: more price movement, more arbitrage. The question of whether an LP is +EV is the question of whether the pool's volume earns enough in fees to outrun the LVR its volatility implies. The chart makes the tradeoff visible.

![Honest LP yield as fee income minus LVR, with the breakeven volatility marked](/imgs/blogs/amms-vs-arbitrageurs-toxic-flow-and-loss-versus-rebalancing-4.png)

The amber curve is annual LVR as a percentage of pool value, $100 \times \sigma^2 / 8$, rising with volatility. The dashed lines are two fixed fee-income levels: a thin pool earning 6% a year in fees, and a busy pool earning 20%. Wherever the LVR curve sits *below* a pool's fee line, that pool is +EV; wherever LVR rises above the fee line, the LP is paying more in adverse selection than they earn in fees, and they'd be better off holding. The breakeven volatility for a given fee tier is $\sigma^* = \sqrt{8 \times \text{fee}}$.

#### Worked example: is this stablecoin pool worth it?

You're weighing two pools. Pool A is a USDC/USDT stablecoin pair — two dollar tokens that barely move against each other, so annualized volatility is tiny, say $\sigma = 2\% = 0.02$. Pool B is an ETH/USDC pair with $\sigma = 80\% = 0.80$, a realistic figure for crypto.

Pool A's LVR rate is $\sigma^2 / 8 = 0.02^2 / 8 = 0.00005$, i.e. **0.005% a year**. Essentially nothing. Even a thin 0.3% annual fee yield swamps it, so a stablecoin LP is comfortably +EV — there's almost no mispricing for an arb to harvest, so almost no LVR. This is why stablecoin and other low-volatility pools are where passive LPing genuinely works.

Pool B's LVR rate is $0.80^2 / 8 = 0.08$, i.e. **8% a year**. For this pool to break even, fees must clear 8% annually. If the pool earns a fat 20% in fees because it's heavily traded, the LP nets $20\% - 8\% = 12\%$ — genuinely +EV. If it earns only 6% in fees because volume is thin, the LP nets $6\% - 8\% = -2\%$ — they are paying \$2 per \$100 per year to provide liquidity, and would have done better holding.

The intuition: an LP is a casino selling a fixed-price product (the swap) to two kinds of customers — gamblers who pay the fee for fun (volume) and card-counters who pay the fee but take the house's money (LVR) — and the LP is only profitable when the gamblers outspend the card-counters.

This is also why the headline "APY" on an LP dashboard is a half-truth. It usually shows fee income — the gross — and quietly omits the LVR, which is the cost of goods sold. A 20% fee APY on an 80%-vol pair is really a 12% net; a 6% fee APY on the same pair is a *loss*. Always subtract the LVR the volatility implies before you believe a yield.

### Concentrated liquidity: leverage on the same bet

The most important evolution of the AMM since the original constant-product design is *concentrated liquidity*, introduced by Uniswap v3 in 2021. Instead of spreading your capital across the entire price curve from zero to infinity, you choose a price *range* — say \$1,800 to \$2,200 for ETH — and your capital only provides liquidity within that band. Inside the band, your liquidity is far deeper than it would be if spread everywhere, so you earn a much larger share of the fees on trades that happen there. The marketing pitch is "capital efficiency": the same dollars earn many times the fees.

The catch is that concentration is leverage, and leverage cuts both ways. If you concentrate your liquidity into a band that's, say, 10x tighter than the full range, you earn roughly 10x the fees on flow inside the band — *and* you suffer roughly 10x the LVR on price moves inside the band, because you're now a 10x-larger fraction of the liquidity exactly where the arbitrage happens. Concentration multiplies both terms of fees-minus-LVR by the same factor; it amplifies the magnitude of the bet without changing its sign. If a position was marginally +EV spread wide, concentrating it makes it more +EV (and more volatile); if it was -EV, concentrating it makes it lose faster.

#### Worked example: concentrated liquidity multiplies both sides

You have \$10,000 to provide on an ETH/USDC pool. Spread across the full curve, suppose it earns 8% in fees and pays 6% in LVR over a year, netting $8\% - 6\% = 2\%$, or \$200.

Now you concentrate it into a band roughly 5x tighter, so (while the price stays in the band) you earn about 5x the fee yield, $5 \times 8\% = 40\%$, but you also pay about 5x the LVR, $5 \times 6\% = 30\%$. Your net is $40\% - 30\% = 10\%$, or \$1,000 — five times the \$200, exactly as the leverage analogy predicts. The catch: if the price *leaves* your band entirely, your position converts fully into the worse-performing asset and stops earning fees, locking in the loss at the band's edge — the concentrated version of impermanent loss, hit harder and faster. The intuition: concentrated liquidity is a margin account on the LP trade, and like any margin account it magnifies a good bet and ruins a bad one; it is not a way to dodge the fees-minus-LVR arithmetic.

## Toxic flow vs benign flow: the AMM's two customers

We can now make the "two kinds of customer" idea precise, because it's the same toxic-versus-benign distinction that governs every market maker — see [adverse selection and the winner's curse](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news) for the general version. Two very different flows hit the exact same pool, and the LP's fate depends entirely on the mix.

![Toxic arbitrage flow versus benign retail flow into the same pool](/imgs/blogs/amms-vs-arbitrageurs-toxic-flow-and-loss-versus-rebalancing-6.png)

*Benign retail flow* is the swap from someone who actually wants the token — a person buying ETH for their wallet, a DAO paying a contributor, a trader rebalancing for reasons unrelated to the pool's price being wrong. This flow has no information edge. It trades both directions roughly at random over time, so its trades don't systematically move the price against the LP. For the LP, benign flow is *pure fee income*: the LP collects the fee and keeps it, because the trade was not predicated on the pool being mispriced.

*Toxic arbitrage flow* is the swap from the bot watching the CEX, trading only when the pool is stale, and always in the profitable direction. This flow is informed by definition. It pays the same fee as everyone else — but it takes more value than the fee covers, and the net gap is LVR. The fee is a small tax on the arbitrageur; the LVR is the LP's real loss.

The LP can't tell the two apart at the moment of the trade — the AMM, lacking the right to refuse, serves both at the same curve. But over time the *composition* of flow determines everything. A pool dominated by benign retail flow (lots of volume, little of it informed) is a money machine for LPs. A pool whose flow is mostly arbitrage (little genuine demand, frequent mispricing) is a slow drain. This is why the most profitable pools to provide for are high-volume pairs on assets that don't move much: maximum benign flow, minimum toxic flow.

### The fee as a (weak) defense

The fee is the *one* defense an AMM has, and it's a blunt one. By charging, say, 0.30% on every swap, the pool makes the arbitrageur wait for a slightly bigger mispricing before it's worth trading — the arb only acts once the price gap exceeds the fee. This creates a *no-arbitrage band* around the pool price: within the band, the mispricing isn't big enough to overcome the fee, so the pool is left alone and the LP keeps the spread on benign flow. The wider the fee, the wider the band, the less often the arb strikes — but also the less benign volume the pool attracts, because high fees scare off retail. The fee tier is the only knob, and it trades off LVR reduction against volume.

#### Worked example: how the fee creates a no-arbitrage band

Take our \$2,000 pool with a 0.30% fee. For an arbitrageur to profit from a CEX move, the price gap has to exceed the round-trip cost — at minimum the pool's fee. A 0.30% move on \$2,000 is \$6. So if ETH ticks from \$2,000 to \$2,003 on the CEX, the 0.15% gap doesn't clear the 0.30% fee, and the rational arb does nothing; the pool stays at \$2,000 and the LP loses nothing to arbitrage on that tick.

But once ETH reaches roughly \$2,006 or higher (a gap past the fee), arbitrage becomes profitable and the pool gets dragged up. The fee bought the LP a small dead-zone of protection — about \$6 wide here — but it does nothing for the big moves that generate most of the LVR. On our earlier 10% move to \$2,200, the \$6 band is a rounding error against the \$476 of LVR. The intuition: the fee protects the LP from the small, frequent noise but is nearly useless against the large moves that do the real damage — which is exactly why a thin fee on a high-volatility pair is a losing proposition.

## The AMM has none of a market maker's defenses

Let's make the contrast with a real market maker explicit, because it's the cleanest way to see *why* the AMM bleeds. A human (or professional algorithmic) market maker is engaged in a constant, adaptive game against the informed; the AMM has opted out of that game entirely.

![AMM versus a real market maker, side by side, showing the missing defenses](/imgs/blogs/amms-vs-arbitrageurs-toxic-flow-and-loss-versus-rebalancing-5.png)

Walk the two columns. The AMM quotes one formula price with no inventory skew — it cannot tilt its quote to shed a position it's accumulating, so it keeps quoting the wrong price until an arb corrects it. It charges a fixed fee that never widens for risk — when adverse selection spikes, a real MM widens its spread to charge the informed more, but the AMM's fee is hard-coded. And it quotes to everyone equally with no right to refuse — it cannot pull quotes or fade size when the flow turns toxic. The sum of these missing defenses is the LVR: the AMM is *systematically* picked off in a way a defended market maker is not.

A real market maker, by contrast, treats every fast one-directional fill as a warning and reacts: skew to dump inventory, widen to price in adverse selection, fade or pull when it senses information. Its spread is set, in equilibrium, to *cover* the adverse-selection cost — that's the entire content of the Glosten-Milgrom model. The AMM's "spread" (its fee) is set once by governance and never adapts. The professional MM is playing the repeated game; the AMM is a stationary target.

This is not a flaw to be embarrassed about — it's the *deal*. The AMM trades away the market maker's defenses in exchange for being permissionless, trustless, and always-on. Anyone can provide liquidity with no relationship, no credit check, no trading desk. That openness is genuinely valuable. But it has a price, and the price is LVR, paid by whoever's capital is sitting in the pool.

## Common misconceptions

**"Impermanent loss isn't real — it reverses when the price comes back."** The IL number does return to zero if the price round-trips, but the *path* generated real, permanent arbitrage profit at your expense. Every wiggle out and back was a pair of arbitrage trades that paid an arbitrageur. IL measures only endpoints; LVR measures the path, and the path is where the money left. A position that ends with zero IL can still have bled substantial LVR.

**"The fees more than make up for it — look at the APY."** The displayed APY is almost always *gross fee income*, not net of LVR. On a low-volatility pair it's roughly honest; on a high-volatility pair it can be wildly misleading. A 20% fee APY on an 80%-vol pool is a 12% net; the same fee APY on a 150%-vol pool ($\text{LVR rate} = 1.5^2/8 = 28\%$) is a *negative* return. Always subtract the volatility-implied LVR before you trust the number.

**"I'll just provide liquidity in a tight range to earn more (concentrated liquidity)."** Concentrating your liquidity in a price band (Uniswap v3 style) does multiply your fee income within that band — but it multiplies your LVR by the same factor, because you're a larger fraction of the pool exactly where the arbitrage happens. Concentration raises both the fee income *and* the adverse-selection cost; it changes the magnitude, not the sign of fees-minus-LVR. It is leverage on the LP bet, not a way to escape it.

**"Arbitrageurs are stealing from LPs — it should be banned."** Arbitrageurs are doing the LP's pricing for them. Without arbitrage, the pool's price would drift arbitrarily far from reality and the LP would be exploited even worse by the first informed trader to notice. Arbitrage is the *mechanism* that keeps the pool roughly correct; the LVR is the fee for that service. The honest fix is not to ban arbitrage but to design AMMs that capture some of the arbitrage value back for LPs (auctions, dynamic fees, oracle-based pricing) — an active research frontier.

**"Stablecoin pools are also exposed to big IL."** Stablecoin pairs barely move against each other, so $\sigma$ is tiny and LVR scales with $\sigma^2$. The IL/LVR on a USDC/USDT pool is a rounding error in normal times. The real risk there isn't LVR — it's the *depeg*, the rare event where one "stable" coin stops being worth \$1, at which point the price ratio moves violently and the LP is left holding the broken one. Different risk, not the LVR risk.

**"If I keep the price in a tight band, I avoid IL entirely."** Some LPs believe that providing only in a narrow range, or only on a pair that "always reverts," sidesteps impermanent loss. It does not — it changes when the loss is realized, not whether it accrues. Within the band, every wiggle still feeds the arbitrageur exactly as before; you've just bet that the price won't leave the band. If it does, you're converted fully into the worse asset at the edge, and the loss you postponed arrives all at once. Range-bound LPing is a bet on realized volatility staying low, not a hedge against it.

**"Arbitrageurs and MEV searchers are the same villain double-dipping."** They overlap but aren't identical. The arbitrageur captures the price gap (the LVR); MEV is the broader game of *ordering* transactions to extract value, of which AMM arbitrage is one slice. In practice much of an arbitrage opportunity's value is competed away in the block-builder auction — searchers bid most of it to validators as priority fees. So from the LP's seat the loss is the LVR regardless of who ultimately pockets it; the MEV supply chain just decides how the arb's profit is split between searcher, builder, and validator, not how much the LP loses.

## How it shows up in real markets

**Uniswap's launch and the IL reckoning (2020–2021).** When Uniswap v2 brought constant-product AMMs to the mainstream in 2020, the "DeFi summer" yields looked spectacular and LPs piled in. As volatility ripped through 2021, many discovered that their fee earnings had been more than eaten by impermanent loss on volatile pairs — a flood of "I provided liquidity and lost money while the token went up" posts. The lesson the market learned the hard way was exactly fees-minus-LVR: gross fee APY was not net return, and on high-vol pairs the difference was often negative.

**The LVR paper reframes the field (2022).** The Milionis-Moallemi-Roughgarden-Zhang paper, "Automated Market Making and Loss-Versus-Rebalancing," gave the field the precise number it had been missing. Empirical estimates that followed put LVR on major Ethereum Uniswap pools at a meaningful fraction of pool value annually — on the order of several percent to low double digits for volatile pairs, with the $\sigma^2/8$ rate matching observed losses reasonably well. This turned a fuzzy complaint ("IL hurts") into a measurable cost that protocol designers could target.

**MEV and the arbitrage auction.** On Ethereum, the right to be the arbitrageur who realigns a stale pool is itself valuable and contested — it's a slice of *maximal extractable value* (MEV). Searchers compete in block-builder auctions to land the arbitrage transaction, and a large share of the LVR they capture gets bid away to validators and builders as priority fees. So the LVR doesn't even all go to "the arbitrageur" as profit; much of it is competed away up the supply chain. For the on-chain mechanics of how this ordering game works, see [analyzing DEX and AMM activity](/blog/trading/onchain/analyzing-dex-and-amm-activity), and for how to read a pool's depth and composition, [reading DEX liquidity and pools](/blog/trading/onchain/reading-dex-liquidity-and-pools).

**Dynamic-fee and oracle AMMs as the defense.** The design response has been to give the AMM back some of a market maker's defenses. Dynamic-fee pools widen the fee when volatility spikes (re-introducing adverse-selection widening). Oracle-based AMMs price off an external feed so the pool isn't stale (removing the gap the arb exploits). LVR-rebate and auction designs (like protocol-run arbitrage auctions) try to capture the LVR for LPs instead of leaking it. Each is an attempt to claw back one of the three defenses the constant-product AMM threw away — and each trades off some of the simplicity and trustlessness that made AMMs valuable in the first place.

**The 2022–2023 stablecoin depegs.** When USDC briefly traded to about \$0.88 in March 2023 during the Silicon Valley Bank scare, and when UST collapsed in May 2022, stablecoin LPs learned the difference between LVR risk and tail risk. In calm times those pools had near-zero LVR and looked like free yield. In the depeg, the price ratio moved violently, arbitrageurs drained the good asset and left LPs holding the broken one, and the "low-risk" pool delivered a large, sudden loss. The $\sigma^2/8$ rate still described the loss — it's just that $\sigma$ is not constant, and the tail is where the LVR concentrates: a year of calm with near-zero LVR can be erased by a single day of 30% movement on a "stable" asset.

**Concentrated-liquidity LPs and the 2021–2022 ETH drawdown.** When Uniswap v3 launched in mid-2021, sophisticated LPs concentrated their liquidity into tight ranges around the prevailing ETH price to chase the higher capital efficiency. Through the calm stretches this worked beautifully — fee yields on concentrated positions ran into the tens of percent. But as ETH fell through late 2021 and 2022, those tight ranges were exactly where the price moved, so the LVR scaled up by the same concentration factor as the fees. Many concentrated LPs found their positions had been pushed entirely out of range (fully converted into the depreciating asset, earning no fees) and that the realized impermanent loss on the way down had swamped the fat fees earned on the way up. The lesson was the leverage lesson: concentration is a magnifier, and a magnifier of a position that turns -EV in a downtrend is a fast loss.

**Professional LPs run the fees-minus-LVR calculation explicitly.** The market's response to all of this has been the professionalization of liquidity provision. Sophisticated market-making firms now model each pool's realized volatility, estimate its LVR rate, and provide liquidity only where the realized fee yield clears that rate with margin — actively rebalancing, hedging the directional exposure with perps or options elsewhere, and avoiding pools where the toxic-flow share is too high. The naive "set and forget" LP has largely been competed out of the volatile pools, leaving them to firms that treat the position as the short-gamma book it actually is. Passive LPing survives mainly where it was always safe: deep, low-volatility, high-volume pairs.

## The playbook: how to play it

**Who's on the other side.** If you provide liquidity to a constant-product pool, your counterparty on every price move is an arbitrageur who is informed, fast, and trading only when they're right. You are the passive, defenseless market maker; they are the professional. The pool serves them at the worst price for you, by design.

**The game you're in.** You are short volatility and long volume. You collect a fee on flow and pay LVR on movement. Your expected return is fee income minus LVR, and LVR scales with the *square* of volatility ($\sigma^2/8$ as an annual rate of pool value). There is no version of passive LPing that escapes this identity — only versions that change the magnitudes.

**Your edge, if you have one.** The only honest LP edge is being in pools where benign flow dwarfs toxic flow: high-volume pairs on assets that don't move much. Stablecoin pairs, correlated-asset pairs (two staked-ETH variants, say), and the deepest blue-chip pools are where fees-minus-LVR is reliably positive. The further you get from that — thin pools, volatile pairs, exotic tokens — the more you're paying arbitrageurs for the privilege of providing liquidity.

**Sizing and the breakeven.** Before you provide, estimate the pair's annualized volatility, compute the implied LVR rate ($\sigma^2/8$), and compare it to the pool's *actual realized* fee yield (fees collected divided by liquidity, not the advertised APY). If the fee yield doesn't clear the LVR rate with margin to spare, don't provide — you're paying to play. For an 80%-vol pair, you need the pool to be earning roughly 8%+ a year in fees just to break even.

How do you estimate the volatility? Take the pair's daily price changes over a recent window, compute the standard deviation of the daily log returns, and annualize by multiplying by $\sqrt{365}$ (since variance adds linearly over time). A pair whose daily moves have a standard deviation of about 4% annualizes to roughly $4\% \times \sqrt{365} \approx 76\%$ — squarely in the "needs a fat fee tier" zone. A pair whose daily moves are 0.1% annualizes to under 2% and is essentially LVR-free. The crude rule of thumb: if the asset routinely moves several percent a day, you need the pool to be very busy for LPing to clear; if it barely moves day to day, even a thin pool can be +EV. And remember that volatility is not constant — size for the regime you might enter, not just the calm one you're in, because the $\sigma^2$ scaling means a doubling of volatility quadruples your bleed overnight.

**The invalidation.** Your LP thesis is invalidated the moment realized volatility rises faster than fee income — for example, when a calm pair becomes volatile (a stablecoin starts to wobble, a correlated pair decorrelates). LVR rises with $\sigma^2$, so a doubling of volatility quadruples your bleed while fees rise at best linearly with volume. The exit signal is structural: if the pair's volatility regime changes, the math flips against you fast.

**The honest mindset.** Treat an LP position as what it is — a sold strip of options to arbitrageurs, paid for in fees. That is a real, sometimes profitable, business. But it is *market making with the defenses removed*, and it only works when you're paid enough for the adverse selection you can't avoid. The number to keep on a sticky note: **fee yield minus $\sigma^2/8$ must be positive, or you're the sitting duck.** This is educational, not financial advice — but the arithmetic is the same whoever runs it.

## Further reading & cross-links

- [The bid-ask spread as an adverse-selection game (Glosten-Milgrom)](/blog/trading/game-theory/the-bid-ask-spread-as-an-adverse-selection-game-glosten-milgrom) — the model that explains why a market maker's spread exists to cover informed losses; LVR is the AMM's version of that cost, but the AMM has no spread mechanism to cover it.
- [Adverse selection and the winner's curse: why a fast fill is bad news](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news) — the general theory of the informed counterparty whose trade is bad news for you; the arbitrageur is the on-chain incarnation.
- [Analyzing DEX and AMM activity](/blog/trading/onchain/analyzing-dex-and-amm-activity) — the on-chain mechanics of how swaps, arbitrage, and MEV actually move through a pool block by block.
- [Reading DEX liquidity and pools](/blog/trading/onchain/reading-dex-liquidity-and-pools) — how to read a pool's depth, composition, and flow so you can judge the benign-versus-toxic mix before you provide.

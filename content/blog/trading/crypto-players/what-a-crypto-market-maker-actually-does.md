---
title: "What a Crypto Market Maker Actually Does: Quotes, Spreads, and Inventory"
date: "2026-07-22"
publishDate: "2026-07-22"
description: "A build-from-zero guide to what a crypto market maker really does — two-sided quoting, spread capture, inventory, and adverse selection — with worked dollar examples and how a maker's presence or absence shows up in the price you trade."
tags: ["crypto", "market-makers", "market-microstructure", "bid-ask-spread", "liquidity", "adverse-selection", "inventory-risk", "maker-taker-fees", "crypto-players", "order-book"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — A market maker (MM) is a firm that continuously posts *both* a buy price and a sell price on the same token at the same time, and earns the tiny gap between them — the spread — over and over across thousands of trades. It is not a directional bet on the token; it is a volume business.
>
> - **Two-sided quoting** is the whole job: rest a *bid* (a standing buy order) just below the mid-price and an *ask* (a standing sell order) just above it, and let the market trade against both.
> - **Spread capture** is small per trade and huge in aggregate: buy at \$99.50, sell at \$100.50, pocket \$1.00 — do that a thousand times a day and you have \$1,000 of gross profit from a spread most traders never notice.
> - The two things that can wreck the business are **inventory** (every fill leaves the MM holding tokens it didn't choose to own, and the price can move against that position) and **adverse selection** (the risk that the trader lifting your quote knows something you don't).
> - The spread is not pure profit — it is *priced* to cover adverse selection, inventory risk, and operating cost first. That is why a thin, illiquid token has a wide spread and a deep, liquid one has a razor-thin spread.
> - **How it shows up in the price you trade:** tight spreads and deep books mean a maker is present and confident; when spreads suddenly gap wide and depth vanishes, the maker has cancelled its quotes — often right before the news that would have cost it. A token with *no* maker barely trades at all.
> - The number to anchor on: one large algorithmic maker, Wintermute, quotes more than 1,000 assets across 50+ venues with daily volumes that frequently exceed \$5 billion — proof that the edge is in the *count* of trades, not the size of any one spread (source: Wintermute, *OTC 2024 in review*, Jan 2025).

You place a buy order for a token. A tenth of a second later, it fills. You place a sell order. It fills too. Who was on the other side — both times, instantly, ready to trade in whichever direction you chose?

Almost always, the answer is the same firm: a **market maker**. It was already there, quietly offering to buy from you at one price and sell to you at a slightly higher one, and it will still be there for the next thousand people who trade after you. It did not care which way you wanted to go. It cared only about the gap between its two prices, and about not getting run over while it collected that gap a few cents at a time.

This post is about what that firm actually *does*, built from zero. We will define every term — bid, ask, spread, quote, resting order, fill, inventory, adverse selection, maker versus taker — before we lean on it, and we will ground each idea in a worked example with round dollar numbers you can check in your head. By the end you should be able to look at any token's order book and read, from the width of the spread alone, whether a professional is standing behind it or whether you are about to trade into a vacuum.

The diagram below is the mental model for everything that follows. A buyer and a seller each trade against the market maker's resting quotes; the maker's cut is the spread it captures when the round trip completes. Every later section is a tour of one corner of this picture.

![A market maker posts both a bid and an ask, so every buyer and seller trades against it and it captures the spread on the round trip.](/imgs/blogs/what-a-crypto-market-maker-actually-does-1.webp)

This is the mechanics companion to the series hub, [Crypto VC and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers), and it sits directly on top of [How Crypto Prices Actually Move](/blog/trading/crypto-players/how-crypto-prices-actually-move), which builds the order book itself from scratch. If you have not met the order book before, skim that post first; here we assume it and climb up to the seat of the person *quoting* it. This is educational, not financial advice — the goal is to let you see the plumbing so you know who is on the other side of your trade.

## Foundations: the building blocks

Forget crypto for a second. Start with a currency-exchange booth at an airport. The booth posts two numbers on its board: it will *buy* your dollars at one rate and *sell* you dollars at a slightly worse one. It makes its living on the difference. It does not have a view on whether the dollar is going up or down this week — it just wants to turn over as much currency as possible and keep the gap. A market maker is that booth, running at the speed of light, on a token instead of a currency.

Let us name the parts.

### Bid, ask, spread, and mid

- A **bid** is a standing order to *buy*: "I will pay up to \$99.50 for this token." The highest bid in the market is the **best bid** — the most anyone is currently willing to pay.
- An **ask** (also called an **offer**) is a standing order to *sell*: "I will sell at \$100.50 or higher." The lowest ask is the **best ask** — the cheapest place you can buy right now.
- The **spread** is the gap between them: `spread = best ask − best bid`. In our example, \$100.50 − \$99.50 = **\$1.00**.
- The **mid-price** is just the average of the two, `(best ask + best bid) / 2` = **\$100.00**. It is a convenient single number for "where the market is," even though you cannot actually trade at the mid.

A **quote** is simply a price a market maker is showing. A **two-sided quote** is the defining act of market making: showing a bid *and* an ask at the same time, on the same token. The airport booth's board is a two-sided quote. The maker is saying, in effect, "I'll buy from you at \$99.50, or sell to you at \$100.50 — your choice, right now."

Two words in that definition carry the whole job. *Two-sided* means the maker commits to both directions — it does not get to only buy when it feels bullish; it must also stand ready to sell. And *at the same time* means the quote is **continuous**: the maker re-posts, adjusts, and refreshes its bid and ask thousands of times a second, all day, so that whenever *you* arrive there is already a price waiting. This is why a token "with a market maker" feels liquid and a token "without one" feels broken. Without a maker continuously quoting both sides, a buyer who shows up has to wait for a seller to happen to show up at the same moment and agree on a price — which, most of the time, simply doesn't happen. The maker's standing offer is the thing that lets you trade *now* instead of *whenever a natural counterparty materializes*. In that sense a market maker doesn't just tighten the market; on a young or thin token it is the reason there is a tradeable market at all.

### Resting orders, fills, and the two ways to trade

When the maker posts its bid and ask, those orders **rest** in the order book — they sit there, visible, waiting for someone to trade against them. An order that rests and waits is providing *liquidity*: it is the thing other people get to trade against.

When someone does trade against a resting order, that order gets a **fill** — a completed trade. If the maker's \$100.50 ask gets filled, the maker has just *sold* a token at \$100.50.

There are exactly two ways any trader — you, me, the maker — can place an order, and the difference is the difference between controlling *price* and controlling *timing*:

- A **limit order** names a price and waits. "Buy at \$99.50 or better." It joins the book as a resting order. You control the price you get, but not *whether* you trade — if the market never comes to you, you sit there. This is how you *provide* liquidity.
- A **market order** names a quantity and takes whatever price is available *right now*. "Buy one token, immediately." You are guaranteed to trade, but you take the price the book offers — you cross the spread and pay it. This is how you *consume* liquidity.

The maker almost always uses limit orders (it *provides*), and the impatient trader who wants in *now* uses a market order (it *consumes*). Hold onto that distinction — it is exactly what "maker" and "taker" mean, and it decides who pays whom in fees. We come back to it in section 5.

### Inventory: the tokens you're left holding

Here is the twist that makes market making genuinely hard. Every time the maker's bid gets filled, it has *bought* tokens — it is now holding them. Every time its ask gets filled, it has *sold* tokens — it now owns fewer, and if it sold more than it had, it is *short* (it owes tokens it must buy back later).

The running net position — how many tokens the maker is long or short at any moment — is its **inventory**. Inventory is not a choice the maker makes trade-by-trade; it is the *residue* of whoever happened to trade against it. And inventory is risk, because while the maker holds it, the price can move.

That sensitivity to price has a name worth learning, because it comes straight from the options world: **delta**, or **directional exposure**. Delta is simply how much money you make or lose per \$1 move in the underlying price. If the maker is long 5,000 tokens, its delta is +5,000: a \$1 rise earns it \$5,000, a \$1 fall costs it \$5,000. A maker that is *flat* — zero inventory — has zero delta and does not care which way the price goes. Staying near flat is the goal; we unpack the mechanics (and the cost) of getting there in [Delta: Direction, Exposure, and the Hedge Ratio](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio).

### Adverse selection: the trader who knows more

The last building block is the maker's deepest fear, and we will define it fully in section 3 — but you need the word now. **Adverse selection** is the risk that the person trading against your quote knows something you don't. When the maker rests a two-sided quote, it is making an *open offer to the entire market*. Most of the people who take that offer are trading for reasons unrelated to the token's near-term value — they need to rebalance, they're paying for something, they have a hunch. But some of them are trading precisely *because* they have information that the price is about to move, and they are choosing the side that will hurt the maker. Being systematically picked off by the informed is adverse selection, and pricing for it is why the spread exists at all.

That is the whole vocabulary. Now the first worked example — the simplest possible thing a market maker does.

![A two-sided quote is a resting bid below the mid and a resting ask above it; the gap between them is the spread the maker collects on the round trip.](/imgs/blogs/what-a-crypto-market-maker-actually-does-2.webp)

The figure is the anatomy of a two-sided quote for a token trading around \$100. The maker's ask (\$100.50, in red — it is *selling* there) sits above the mid; its bid (\$99.50, in green — it is *buying* there) sits below; the spread (\$1.00, in amber) is the gap in between. Other traders' orders stack above and below. On the right, the maker's inventory — the net tokens it is left holding — changes with every fill. A buyer who wants in *now* lifts the ask; a seller who wants out *now* hits the bid.

#### Worked example: one round trip

Suppose over the next minute, two people trade against our maker's quotes:

1. An impatient **seller** wants out immediately, so they send a market sell. It hits the maker's resting **bid**. The maker *buys* 1 token at **\$99.50**. Its inventory is now +1 token; it has spent \$99.50.
2. A moment later an impatient **buyer** wants in immediately, so they send a market buy. It lifts the maker's resting **ask**. The maker *sells* 1 token at **\$100.50**. Its inventory is back to 0; it has received \$100.50.

Net result for the maker: it bought at \$99.50 and sold at \$100.50, ending flat (zero inventory) with **\$100.50 − \$99.50 = \$1.00** of profit. It never had a view on the token. It never held the position for more than a minute. It just stood in the middle, quoted both sides, and collected the spread when a buyer and a seller each crossed it.

> The maker does not get paid for being right about the price. It gets paid for being *present* — for standing willing to trade both ways when nobody else will.

That single dollar is the atom of the entire business. Everything else in this post is about (a) how you turn one dollar into a real income by repeating it, and (b) the two forces — inventory and adverse selection — that keep it from being free money.

## 1. The spread-capture engine

The intuition first: a market maker is a *volume* business, not a *margin* business. It is closer to a supermarket than to a hedge fund. A supermarket makes a few cents on a can of beans, but it sells a million cans, and the few cents times a million is a fortune. A maker makes a fraction of a percent on one round trip, but it does thousands of round trips a day.

The formal version is almost embarrassingly simple. If the maker captures the spread `s` on each completed round trip (one buy on the bid, one matching sell on the ask), and it completes `N` round trips in a day, its gross spread income is:

`gross P&L = N × s`

That's it. The art is not in the formula; it is in making `N` large while keeping `s` from having to be large — and in surviving the days when inventory and adverse selection turn some of those round trips into losses.

#### Worked example: a day of round trips

Take our \$100 token with a \$1.00 spread. Assume the maker completes one full round trip per token and turns over 1,000 round trips over a trading day:

- Gross spread income = `N × s` = 1,000 × \$1.00 = **\$1,000 per day**.

Now add the wrinkle that makes it realistic: fees and slippage eat into every trip. Say the all-in cost of doing business — exchange fees on the taker side of some trips, the occasional trip where the maker has to pay up to rebalance, infrastructure — averages about **\$0.10 per round trip**. Then:

- Net income = `N × (s − cost)` = 1,000 × (\$1.00 − \$0.10) = **\$900 per day**.

The figure below plots this: the gross line rises \$1.00 per round trip to \$1,000, and the net-of-fees line rides just underneath it to about \$900. Neither line is dramatic — the point is precisely that it *isn't* dramatic. No single trade matters. The slope is gentle. What matters is that the maker keeps the engine running all day, every day, and that the slope stays *positive*.

![Spread capture is tiny on any one round trip but compounds: a thousand round trips at a dollar each stack into a thousand dollars of gross daily profit, about nine hundred net of fees.](/imgs/blogs/what-a-crypto-market-maker-actually-does-3.webp)

Two things are worth pausing on.

First, **the maker wants the spread to be as *tight* as it can profitably make it, not as wide as possible.** A tighter spread is more attractive to traders, so it wins more of the flow — a bigger `N`. There is a competition here: on a busy pair, several makers undercut each other's quotes until the spread is razor-thin, and the one willing to quote tightest while still covering its costs wins the queue. Wide spreads are a symptom of a market *nobody wants to make*, not of a greedy maker. We will see exactly why in section 4.

Second, **the whole model breaks if the round trips stop completing.** In the clean example, every buy on the bid is soon matched by a sell on the ask, so inventory returns to zero and the maker banks the spread. In the real world, the flow is lumpy: sometimes ten sellers hit the bid in a row and no buyer shows up. Now the maker is sitting on a growing pile of tokens it must eventually offload, and the price is moving the whole time. That pile is inventory, and it is the subject of section 2.

## 2. Inventory: the position you didn't choose

Recall the definition: inventory is the maker's running net position — the tokens it is long or short as a residue of whoever traded against it. In the perfect round trip, inventory pings up to +1 and back to 0 within a minute. In reality it wanders, and while it is away from zero, the maker has a *directional bet it never wanted*.

Here is the asymmetry that makes inventory dangerous. When flow is balanced — buyers and sellers arriving in roughly equal numbers — inventory stays near zero and the maker just collects spread. But flow is rarely balanced when it matters most. If the price is quietly sliding, sellers keep hitting the maker's bid and buyers stay away. The maker keeps *buying* — accumulating a bigger and bigger long position — into a falling market. It is being handed exactly the position it least wants, at exactly the wrong time. (When the price is ripping upward the mirror image happens: buyers lift the ask, the maker keeps selling, and it builds a growing *short* into a rising market.)

So a real maker does not passively let inventory pile up. It *skews* its quotes to lean against the position — shifting both its bid and its ask in the direction that will bleed inventory back toward zero.

![When inventory piles up long, the maker shifts both its bid and its ask downward — cheapening its ask to attract buyers and lowering its bid to discourage more sellers — so the position bleeds back toward flat.](/imgs/blogs/what-a-crypto-market-maker-actually-does-4.webp)

Read the figure left to right. On the left, the maker is flat: 0 tokens, a symmetric quote of \$99.50 / \$100.50 centered on a \$100.00 mid. On the right, it has been buying — it is now long 5,000 tokens — so it shifts *both* quotes down by \$0.20, to \$99.30 / \$100.30. Why down? Because a *lower ask* (\$100.30 instead of \$100.50) is a better deal for buyers, so it pulls in buy orders that will *sell the maker's inventory back out*; and a *lower bid* (\$99.30 instead of \$99.50) is a worse deal for sellers, so it discourages even more tokens from being dumped onto the maker. The skew makes the maker's quote deliberately lopsided to fix a lopsided position.

#### Worked example: an inventory loss

Skewing helps, but it does not make inventory free — sometimes the price moves faster than the maker can bleed the position off. Suppose:

- The maker has accumulated a long inventory of **5,000 tokens** at an average cost of **\$100.00** each (it kept buying on the bid as sellers hit it).
- Before it can offload them, bad news drops and the token falls **\$2.00**, from \$100 to \$98.

Its inventory is now worth \$98 × 5,000 = \$490,000, against the \$500,000 it paid. The loss is:

- 5,000 tokens × \$2.00 = **\$10,000**.

To put that in perspective: a \$10,000 inventory loss wipes out *ten full days* of the \$1,000/day gross spread income from section 1. This is the central tension of the whole business — **the spread income is a slow trickle, but the inventory risk is a sudden gush.** A maker can quote profitably for weeks and give it all back in one violent hour if it is caught holding the wrong side.

This is why serious makers spend enormous effort staying **delta-neutral** — keeping their net directional exposure near zero even while they are forced to hold inventory. The usual tool is to *hedge*: the instant the maker gets long spot tokens, it sells an offsetting amount in a related market — most often a **perpetual futures contract** (a "perp"), a derivative that tracks the token's price and lets you take a short position without holding the token itself.

#### Worked example: hedging the inventory

Take the same situation, but now the maker hedges.

- The maker gets long **5,000 tokens** on the spot market at \$100.00 (inventory it was handed by sellers).
- Immediately, it *shorts* 5,000 tokens' worth of the perpetual — a bet that profits \$1 for every \$1 the price *falls*.
- The token falls **\$2.00**, from \$100 to \$98.

Now tally both legs:

- Spot inventory: down \$2.00 × 5,000 = **−\$10,000** (the loss from before).
- Perp short: up \$2.00 × 5,000 = **+\$10,000** (the hedge gains exactly what the spot lost).
- Net directional P&L: **\$0.**

The maker held the tokens, the price moved \$2 against them, and it lost *nothing* on direction — because the perp short carried the opposite exposure. Its delta was near zero the whole time. What it keeps is the spread it earned quoting; what it *doesn't* keep is a directional bet it never wanted.

Hedging is not perfectly free — the spot and perp prices can drift apart (basis risk), the perp charges a periodic **funding rate**, and putting the hedge on costs its own slippage. Those frictions are exactly the "inventory carry" cost we fold into the spread in the next section, and the full mechanics are the subject of the next post in this series. But the one-sentence version is: **a good maker tries to earn the spread while owning as little of the token's direction as it possibly can.**

> A market maker's dream is to touch every token that trades and *keep* none of it. Inventory is the gap between that dream and reality.

## 3. Adverse selection: trading against someone who knows more

Inventory is the risk that the price moves *while* you hold a position. Adverse selection is a subtler and, in the long run, more dangerous risk: it is the risk that the price moves *because* of the very trade that gave you the position — because the person on the other side knew it was about to.

Start with the intuition. When the maker rests a two-sided quote, it is making a blanket offer to everyone. Two very different kinds of trader take that offer:

- **Noise traders** (also called uninformed flow): people trading for reasons unrelated to the token's next tick. They rebalance a portfolio, they cash out to pay a bill, they buy on a whim. Crucially, *which side they take is roughly random* — some buy, some sell — and after they trade, the price is no more likely to go up than down. The maker *loves* this flow: it captures the spread and the price doesn't punish it.
- **Informed traders**: people trading *because* they know something — a big buyer is about to move in, an unlock is coming, a listing is imminent, a wallet just did something telling. They don't trade randomly. They trade the side that is about to be right. If they know the price is about to jump, they *lift the maker's ask* (buy cheap from the maker just before it gets expensive). If they know it's about to fall, they *hit the maker's bid* (dump onto the maker just before it gets cheap).

The word "adverse" is doing real work: the maker is *selected against*. It wins its little spread from the harmless traders, but it is systematically handed the losing side by the dangerous ones. The figure below lays the two rows side by side.

![Adverse selection: the maker keeps the spread from noise traders whose side is random, but loses to informed traders who lift the ask right before a rise, leaving the maker short into a rally.](/imgs/blogs/what-a-crypto-market-maker-actually-does-5.webp)

#### Worked example: getting picked off

Our maker is quoting \$99.50 / \$100.50 on a \$100 token.

**The good case (noise trader).** A trader with no special information sends a market buy and lifts the maker's ask. The maker sells 1 token at \$100.50. Nothing happens next — the price stays around \$100. The maker later buys a token back on its bid at \$99.50 and books the \$1.00 spread. Textbook.

**The bad case (informed trader).** A trader who *knows* a large buyer is about to sweep the market lifts the maker's ask — but for 100 tokens, at \$100.50. The maker is now short 100 tokens. Seconds later the price rips to **\$105**, exactly as the informed trader expected. To get flat, the maker must buy those 100 tokens back at the new price:

- It sold 100 at \$100.50 = \$10,050 received.
- It must buy 100 back at \$105.00 = \$10,500 paid.
- Loss = 100 × (\$105.00 − \$100.50) = **\$450**.

That single informed trade cost the maker **\$450**, or 450 of its \$1.00 spreads. The maker didn't do anything wrong — it quoted a fair two-sided market. It simply had the misfortune of being *the* offer standing in the way when someone with better information decided to trade.

Now hold both worked examples together, because their tension defines the craft:

- Against noise traders, the maker earns \$1.00 per round trip.
- Against informed traders, it loses \$450 in one hit.

The maker cannot tell them apart at the moment of the trade. All it can do is *price* for the mix — set its spread wide enough that the money it earns from the harmless majority covers the money it loses to the dangerous minority. That pricing calculation is the bridge to section 4, and it is the reason spreads are wide exactly where you'd least want them to be.

### How a maker defends itself

Pricing the spread wide is the slow, standing defense — but a maker has three faster ones, and recognizing them on a chart is a genuine edge:

- **Widen the spread.** If the maker senses the risk of informed flow rising — around a scheduled announcement, an unlock date, a spike in volatility — it pulls its bid down and its ask up, charging more for the privilege of trading with it. A spread that suddenly balloons is often a maker bracing for information it can smell coming.
- **Cut its size.** The maker can keep the spread tight but shrink how many tokens it rests at each price. It still quotes, but it exposes far less to any single trade, so a pickoff hurts less. The book *looks* nearly as tight, but the depth behind the top price has quietly thinned.
- **Cancel entirely.** In the extreme, the maker simply deletes its quotes. Its orders are resting limit orders, and cancelling them takes milliseconds. When the risk of being run over is high enough, the rational move is to step aside and stop quoting until the dust settles.

All three are defenses against adverse selection, and all three degrade the market *for you* exactly when you might want it most — the spread you trade against is only as good as the maker's current appetite for risk. This is the behavior you will see rendered in the price later in this post: the same book, moments apart, with the maker present and then pulled.

## 4. Why the spread has to be wider on a thin market

Beginners often assume a wide spread means a greedy or lazy market maker. Almost the opposite is true. **The spread is not a profit margin the maker chooses; it is a break-even price it is forced to charge.** The spread has to cover, in order, the cost of adverse selection, the cost of carrying inventory, and the plain operating cost of running the machine — and only what's left over is profit. On a thin market all three of those costs are high, so the spread *has* to be wide, or the maker loses money and simply stops quoting.

Let's decompose the spread into what it actually pays for.

![The spread is priced to cover adverse selection, inventory risk, and operating cost before any profit is left; on a thin market those first blocks are large, so the whole spread must be wide.](/imgs/blogs/what-a-crypto-market-maker-actually-does-6.webp)

On the left, a thin market's \$1.00 spread breaks into four blocks: the largest, adverse selection (\$0.50), then inventory risk (\$0.30), then operating cost (\$0.10), and only then profit (\$0.10). On the right, a deep, liquid market: many makers compete for the flow, informed traders are a smaller fraction of it, and any inventory clears in seconds because someone is always trading — so the adverse-selection and inventory blocks shrink to almost nothing and the whole spread collapses to about \$0.10, roughly ten times tighter.

#### Worked example: the break-even spread

Let's build the thin-market spread from the ground up, using the adverse-selection loss we just computed.

Suppose that on this thin token, **1 in 20 trades** comes from an informed trader, and when it does, it costs the maker about **\$4.50** per token (our section-3 pickoff). The *expected* adverse-selection cost the maker must eat on *every* trade — informed or not — is:

- (1/20) × \$4.50 = **\$0.225 per token** of expected loss to informed flow.

On top of that, say carrying and hedging its inventory costs the maker about **\$0.15 per token** on average (the price drifts against it while it holds; hedging isn't free), and operating cost — fees, infrastructure — runs about **\$0.05 per token**. Add them up:

- Adverse selection: \$0.225
- Inventory carry: \$0.15
- Operating cost: \$0.05
- **Break-even cost: about \$0.425 per token.**

Now the bridge to the figure. The maker earns only about *half* the spread on each one-sided fill — a buy on the bid, then eventually a sell on the ask, each captures roughly half the round-trip spread — so the spread has to be about **twice** the per-token cost to break even: 2 × \$0.425 ≈ **\$0.85**. It will quote a touch wider than that, near **\$1.00**, to leave a thin profit. Double the per-token costs and you get exactly the full-spread blocks in the figure above (adverse selection near \$0.50, inventory near \$0.30, operating near \$0.10, and about \$0.10 of profit left over). The thin token trades at a \$1.00 spread not out of greed but out of arithmetic.

Now run the same arithmetic on a deep, liquid token. Informed traders are a much smaller share of a huge, diverse flow — say **1 in 200** trades — and a pickoff costs less because the maker can offload fast. If the expected adverse-selection cost drops to \$0.02, inventory carry to \$0.01 (it clears in seconds), and operating cost to \$0.02, the break-even is about \$0.05 per token — a spread near **\$0.10**. Same maker, same skill, ten-times-tighter spread, purely because the market underneath is deeper and safer to quote.

> A wide spread is the market maker telling you the truth about the token: *"This is dangerous to stand behind, so standing behind it costs you."* A tight spread is the same honesty in reverse.

This is the single most useful thing a retail trader can take from understanding market makers. The spread is a *live readout of risk*. When you see a token with a 2% spread and a thin book, the professionals are pricing in real danger — low volume, high adverse selection, the constant threat of a violent move. You are not getting a bargain by trading it; you are paying, in slippage, exactly the risk premium the maker refused to absorb for free.

## 5. Maker rebates: getting paid to provide liquidity

There is one more piece of the maker's economics, and it flips a fee you probably think of as a cost into, sometimes, a *payment*. It comes back to the maker-versus-taker distinction from the foundations.

Recall: a **maker** posts a resting limit order and *provides* liquidity; a **taker** sends a market order that *consumes* it. Exchanges love makers — a book full of resting orders is what makes an exchange usable, and a deep book attracts more traders. So most exchanges deliberately tilt their fee schedule to *reward* making and *charge* taking. This is the **maker-taker fee model**, and it can even run to a **maker rebate**: a *negative* fee, where the exchange literally pays you to leave resting liquidity on its book.

![The maker rests a limit order and is paid a rebate for the liquidity; the taker crosses the spread with a market order and pays the fee, and the exchange settles the split.](/imgs/blogs/what-a-crypto-market-maker-actually-does-7.webp)

Follow the flow: the maker rests a limit order, which tightens the book; a taker sends a market order that crosses the spread; the exchange matches them and settles the fees — charging the taker and, at the richest tiers, *crediting* the maker.

The real numbers, as of 2026 (fee schedules change constantly, so treat these as illustrative of the *structure*, not fixed forever):

- On Binance spot, the base fee is **0.10%** for both maker and taker, falling with volume; at the top VIP tier the spot maker fee drops to roughly **0.00825%** (source: Binance fee schedule).
- On derivatives venues the incentive goes further into negative territory: OKX advertises a maker fee as low as **−0.01%** at its top VIP tier (a rebate), against a taker fee around 0.013%, and Bybit runs a market-maker program paying up to a **−0.01%** maker rebate (sources: OKX and Bybit fee documentation, 2026).

A −0.01% rebate sounds trivial. For a professional maker doing enormous volume, it is not.

#### Worked example: rebate economics

Suppose our maker trades a notional volume of **\$500 million** in a day as a maker (all resting limit orders), on a venue paying a **0.01%** maker rebate.

- Rebate income = 0.01% × \$500,000,000 = **\$50,000 per day**, *before any spread capture at all.*

The rebate is not the main event — the spread is — but it changes the arithmetic in two important ways. First, it *lowers the maker's break-even spread*: if the exchange is paying you 0.01% to post, you can afford to quote a little tighter than a maker who pays to trade, which helps you win the queue (bigger `N`). Second, it explains a behavior that looks strange from the outside: makers will sometimes quote at a spread so thin it barely covers their costs, because the *rebate* is a meaningful slice of the total return. The visible spread understates what the maker actually earns.

The taker — you, when you hit the button that fills immediately — is on the other side of all of this. You pay the spread *and* the taker fee. That is the price of immediacy, and it is entirely fair: the maker took on the inventory and adverse-selection risk of standing there waiting, and you paid it to be there so you didn't have to wait. But it is worth knowing that when you cross the spread with a market order, you are simultaneously paying the maker its spread and paying the exchange its taker fee — two tolls, both flowing to the parties who provided the liquidity you just consumed. If those tolls matter to you, the fix is to *become a maker yourself*: post a limit order and wait, and you flip from paying the fee to (on some venues) earning the rebate. The catch, of course, is that you give up immediacy and take on the maker's risks — you might not get filled, or you might get filled right before the price moves against you. There is no free lunch; there is only choosing which side of the spread you want to stand on.

## How a maker shows up in the price you trade

Everything above happens off-screen. What you actually *see*, as a trader staring at a chart or an order book, is the shadow the maker casts on the price. Learning to read that shadow is the practical payoff of this whole post.

The single clearest signal is the spread itself, together with the depth behind it.

![With a maker quoting, the spread is a few cents and depth is stacked; the instant it cancels, the spread gaps and the next small order jumps the price.](/imgs/blogs/what-a-crypto-market-maker-actually-does-8.webp)

On the left, a maker is present: the spread is a couple of cents (0.02%), tens of thousands of tokens rest near the top of the book, and a \$10,000 order barely moves the price — about a tenth of a percent. On the right, the same token the instant the maker *pulls* its quotes: the spread gaps to \$2.00 (2%), depth collapses to a few hundred tokens, and now that same \$10,000 order jumps the price around 30%, because there is almost nothing underneath it to absorb the trade. (That last number is not hypothetical hand-waving; it is exactly the thin-book slippage math worked out in [How Crypto Prices Actually Move](/blog/trading/crypto-players/how-crypto-prices-actually-move).)

Nothing about the token changed between those two panels. The only thing that changed is whether a professional was willing to stand there. And here is the part that should make you cautious: **the maker can flip from the left panel to the right panel in milliseconds.** Its quotes are resting limit orders, and it can cancel every one of them in the blink of an eye. It does exactly that whenever the risk of being picked off spikes — the moment before a scheduled news release, during a burst of volatility, when it detects informed flow arriving. The liquidity that made the token look deep and safe evaporates precisely when you would most want it.

So the practical reads are:

- **A tight spread and a deep book** mean a maker is present and *confident* — it judges adverse selection and inventory risk to be low right now. The token is cheap to trade and unlikely to lurch on a small order.
- **A suddenly widening spread and thinning depth** mean makers are pulling back — they are pricing in danger, or stepping away entirely. Treat it as a warning light, not an invitation.
- **A permanently wide spread** means no professional wants to make this market at a tight price at all. Every trade you do will pay that spread in slippage. This is the default state of most small, illiquid tokens, and it is why "the chart looks cheap" and "you can actually get out at that price" are two completely different claims.

## Common misconceptions

**"The market maker is betting against me."** Usually not. A well-run maker is deliberately *delta-neutral* — it does not want the token to go up or down; it wants it to trade. When you buy from its ask, it is not taking a short position it hopes will profit from your loss; it is capturing a spread and, moments later, hedging or offloading the inventory you just handed it. Its P&L comes from the *volume* of your trading, not the *direction*. (Firms that *do* take directional bets while also making markets exist, and that conflict is real and serious — but it is a separate hat, covered in the "designated versus principal market making" post later in this series, not the base case.)

**"A wide spread means the maker is gouging me."** The spread is a break-even price, not a margin (section 4). A wide spread is the maker telling you the token is expensive to stand behind — thin volume, high adverse-selection risk, hard-to-clear inventory. If the spread were pure profit, competitors would undercut it to zero. The fact that it stays wide means the *costs* are real.

**"Market makers create fake volume / manipulate the price."** Legitimate market making is the opposite of manipulation: it *dampens* volatility by absorbing imbalances and it *narrows* spreads by competing. There is a genuine dark side — wash trading and manufactured volume — but that is a distinct, and in many places illegal, activity, and we treat it separately and carefully later in this series. Do not confuse the honest business of quoting two sides with the abuse of faking trades.

**"If I just post limit orders, I'm a market maker and it's free money."** You would be *providing* liquidity, and you might earn a rebate — but you would also be taking on the maker's two risks with none of its tools. You'd be picked off by informed traders (adverse selection) and left holding inventory as the price moves against you, with no hedging infrastructure and no speed to cancel before the news. The rebate is small; the risks are not. Professionals win this game on technology, speed, and risk management, not on the rebate.

**"The price on the screen is the price I can trade."** The *last-trade* price is one number; the price you can actually transact depends on the spread and the depth behind it. On a thin book, the displayed price and the price you'd get for any real size are far apart — the difference is slippage, and it is the maker's absence made visible.

## How it shows up in real markets

### 1. The scale of a modern maker — Wintermute

To feel how much the volume-not-margin model compounds, look at one of the largest algorithmic makers. In its own 2024 review, Wintermute reported providing liquidity across **more than 50 centralized and decentralized exchanges**, quoting **over 1,000 digital assets**, with daily volumes that **frequently exceed \$5 billion**; its single-day OTC spot volume hit a record of about **\$2.24 billion** in November 2024 (source: Wintermute, *OTC 2024 in review*, Jan 2025, and Finance Magnates). The spread it captures on any one of those trades is tiny — often a fraction of a basis point on the most liquid pairs. The business works because the *count* is astronomical. A thousand assets, fifty venues, billions of dollars turned over a day: that is section 1's `N × s` engine running at industrial scale, and it is why a small number of firms provide a very large share of all the liquidity you trade against.

### 2. When the maker pulls — the 2017 GDAX ETH flash crash

The clearest real-world demonstration of the "maker pulls, spread gaps" panel happened on **21 June 2017** on GDAX (now Coinbase). A single large multi-million-dollar market sell hit the ETH-USD book. On a normal day, resting maker liquidity would have absorbed most of it. But the sell was large enough to blow through the thin resting depth, the price dropped about **29.4%** from \$317.81 in one sweep, and as it fell it triggered a cascade of roughly **800 stop-loss and margin liquidations** that drove ETH momentarily to **\$0.10** before recovering (source: GDAX / Adam White post-mortem, June 2017). The mechanism is exactly section 2 and the price panel above: once the resting liquidity was gone and makers had no reason to stand in front of a falling knife, there was nothing underneath the price, and a single order re-priced the whole book by hundreds of times the cash that actually traded. The lesson is not that makers are unreliable — it is that maker liquidity is *conditional*, and it is thinnest exactly when a violent move is underway.

### 3. A token with no maker at all

Walk onto the order book of a brand-new, un-listed-anywhere microcap token — the kind that launched last week with a handful of holders. There is often *no* professional maker present at all. The spread might be 10% or 30%. There might be a few hundred dollars of depth on each side. Every buy jumps the price; every sell craters it. This is what a market looks like with the maker subtracted: barely a market at all, just a scattering of resting orders from other retail traders. It is the strongest possible illustration of the post's thesis — that a token **literally cannot trade smoothly without someone willing to make it.** The first thing a serious project does before a major listing is *hire* a market maker precisely to fix this, which is the subject of the loan-plus-options deal covered next in this series.

### 4. The maker with no humans — automated market makers

Crypto did something traditional finance never did: it turned the market maker into a piece of code. An **automated market maker (AMM)** — Uniswap is the canonical example — replaces the human maker's quoting engine with a formula. Anyone can deposit two tokens into a shared *pool*, and a mathematical rule automatically quotes a two-sided price against that pool: as buyers drain one token, the formula raises its price; as sellers add it, the formula lowers it. There is no order book and no firm cancelling quotes — just a curve that always offers a bid and an ask.

And here is the beautiful part: the AMM faces *exactly the same economics as a human maker*, just expressed in code. The people who deposit into the pool earn the **spread** in the form of trading fees. They carry **inventory** — the pool's balance shifts toward whichever token is being dumped into it. And they suffer **adverse selection** under a name the DeFi world coined for it: **impermanent loss**, which is precisely the loss a passive liquidity provider takes when informed traders and arbitrageurs pick off the pool's stale formula price against the true market price. Same spread, same inventory, same adverse selection — no humans in the loop. It is the cleanest possible proof that the three forces in this post are not quirks of one trading desk; they are the fundamental physics of quoting a two-sided market. The full mechanics live in [DeFi Protocols: Uniswap, Aave, and MakerDAO](/blog/trading/crypto/defi-protocols-uniswap-aave-makerdao).

### 5. The fee model as a competitive weapon

Exchanges know the maker economics of section 5 cold, and they compete on them. When a venue wants to bootstrap liquidity on a new market, it dangles a **maker rebate** to lure professional makers in — pay them to post, and the book fills, and takers follow the depth. When Coinbase, Binance, OKX, Bybit and others tune their VIP maker-taker schedules, they are effectively bidding for the world's makers to point their quoting engines at *their* books rather than a competitor's. The visible result, for you, is which venues have the tightest spreads on which pairs. The tightness you enjoy as a taker is downstream of a rebate war you never see — and it is the clearest sign that liquidity is a service that has to be *paid for*, one way or another.

## When this matters to you

You do not need to run a quoting engine to use any of this. The value is in what it lets you *see* before you click:

- **Read the spread before you trade, every time.** It is the market maker's own estimate of how dangerous the token is to stand behind. A tight spread is a green light on cost; a wide one is a toll you will pay in slippage whether you notice it or not.
- **Check the depth, not just the price.** The last-trade number tells you nothing about whether you can get *size* in or out. Thin depth behind a nice-looking price means the exit is narrower than the entrance.
- **Treat a suddenly gapping spread as a warning.** If the spread widens and depth thins out on a token you hold, makers are pricing in risk or stepping away. That is often the calm before a move, not a buying opportunity.
- **Know which side of the spread you're on.** Crossing with a market order buys you immediacy and costs you the spread plus the taker fee. Resting a limit order saves that cost (and may earn a rebate) but gives up certainty of filling. Neither is "better" — they are different trades, and the right one depends on whether you are paying for speed or for price.
- **Respect the maker without romanticizing it.** It provides a real service — it is why liquid tokens are cheap and smooth to trade — but its liquidity is conditional and its interests are its own. When it matters most, it can and will step aside.

The market maker is the most important player you will never see named in a headline. It sets the spread you pay, the depth you rely on, and the smoothness of every fill you take for granted — and it does so not out of charity but because, across millions of tiny round trips, standing in the middle and quoting both sides is a business. Understanding that business is the difference between trading *through* the plumbing and trading *blind to* it.

To go deeper on the mechanics one layer down, see the order-book-and-slippage companion, [How Crypto Prices Actually Move](/blog/trading/crypto-players/how-crypto-prices-actually-move); for where market makers sit in the broader hierarchy of who moves crypto, [The Hidden Power Structure of Crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto); and for the traditional-markets version of the exact same role, [Market Makers and the Spread: Who Provides Liquidity](/blog/trading/capital-markets/market-makers-and-the-spread-who-provides-liquidity).

## Sources & further reading

- Wintermute, [*OTC 2024 in review & 2025 outlook*](https://www.wintermute.com/insights/market-color/reports/wintermute-otc-2024-in-review-2025-outlook) — the 50+ venues, 1,000+ assets, and daily volume figures (Jan 2025).
- Finance Magnates, [*Crypto Market Maker Wintermute Sees Record \$2.24 Billion Daily Trading Volume*](https://www.financemagnates.com/cryptocurrency/crypto-market-maker-wintermute-sees-record-224-billion-daily-trading-volume/) — the November 2024 single-day OTC spot record (Jan 2025).
- Binance, [spot trading fee schedule](https://www.binance.com/en/fee/schedule) — the 0.10% base and VIP-tier maker/taker rates (as of 2026; schedules change).
- OKX and Bybit fee documentation — negative maker fees / maker-rebate programs at top VIP tiers (as of 2026; schedules change).
- GDAX (Coinbase), Adam White, post-mortem of the 21 June 2017 ETH-USD flash crash — the mechanics of a maker-less book and cascading liquidations.
- This blog: [Crypto VC and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers) (series hub) · [How Crypto Prices Actually Move](/blog/trading/crypto-players/how-crypto-prices-actually-move) (order book, slippage, thin float) · [The Hidden Power Structure of Crypto](/blog/trading/crypto-players/the-hidden-power-structure-of-crypto) (who moves the market) · [Market Makers and the Spread](/blog/trading/capital-markets/market-makers-and-the-spread-who-provides-liquidity) (the TradFi analog) · [Delta: Direction, Exposure, and the Hedge Ratio](/blog/trading/options-volatility/delta-direction-exposure-and-the-hedge-ratio) (directional exposure and hedging).

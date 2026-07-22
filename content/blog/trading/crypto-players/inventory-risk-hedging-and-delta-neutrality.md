---
title: "Inventory Risk, Hedging, and Delta-Neutrality: How Market Makers Stay Flat"
date: "2026-07-22"
publishDate: "2026-07-22"
description: "A build-from-zero guide to how a crypto market maker manages the tokens it is forced to hold: inventory risk, hedging spot with perpetual and dated futures, staying delta-neutral, and what the hedge costs in funding — with worked dollar examples and sourced market figures."
tags: ["crypto", "market-makers", "hedging", "delta-neutral", "perpetual-swaps", "funding-rate", "basis-trade", "inventory-risk", "derivatives", "crypto-players"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 38
---

> [!important]
> **TL;DR** — A market maker wants the *spread*, not the *coin*. Every trade it fills leaves it holding inventory — tokens it never wanted a bet on — and if the price moves against that inventory, the loss can dwarf the tiny spread it just earned. So it hedges, and the whole game becomes staying flat.
>
> - **Inventory risk** is the exposure created by the coins you are forced to hold from quoting. Hold 100,000 tokens and the price drops 10%, and you lose \$10,000 no matter how tightly you quoted.
> - The fix is a **hedge**: for every token you are long on spot, go **short** the same amount on a **perpetual future**, so the two P&Ls cancel. Net exposure to price falls to about zero — **delta-neutral**.
> - The hedge is not free. You pay (or receive) the **funding rate** on a perp — the "default" is 0.01% every 8 hours, about **10.95% a year** — plus the **basis** on a dated future. That carry is the real cost of staying flat.
> - The number to remember: at the 0.01%-per-8h default, funding runs about **10.95% annualized**; in the 2021 euphoria it reached **0.15–0.20% per 8h (roughly 160–220% a year)** before Bitcoin fell from about \$64,000 to about \$30,000. The cost of the hedge is a live, moving number.
> - When you see a maker "dumping" on you or a big short appear, it is usually a hedge against inventory, not a directional bet against you. Learning to read that is a retail-defense skill.

You watch a token you own tick down and you see it on the tape: right as your buy order fills, someone sells the same size back into the market, and a fresh short position pops up on the perpetual exchange. It *feels* like a whale betting against you, like someone knows something you don't.

Almost always, it is something far more boring and far more important to understand: a market maker got handed your tokens, doesn't want them, and is hedging. The maker is not making a bet on the price going down. It is doing everything it can to make *no bet at all* — because its business is collecting a fee for providing liquidity, and a directional position on the coin is a distraction that can wipe out weeks of those fees in a single afternoon.

This post is about that hedge: where it lives, what it costs, and why a professional liquidity provider spends most of its energy trying to hold as close to *nothing* as possible while still quoting billions of dollars of two-sided markets. The diagram below is the mental model for the whole article — a maker fills both sides, accumulates a coin it never wanted, cancels the price risk with an offsetting short, and is left holding only the spread minus a small carrying cost. Everything else here is a tour of that picture.

![The market maker's flat book: fill both sides, accumulate unwanted inventory, hedge it with an offsetting short, stay delta-neutral, and keep only the spread minus carry.](/imgs/blogs/inventory-risk-hedging-and-delta-neutrality-1.webp)

We will build every term from zero — inventory, long and short, delta, spot, mark-to-market, notional, hedge, perpetual swap, funding rate, dated future, basis, and delta-neutral — and ground each one in a worked example with round dollar numbers. This is educational, not financial advice; the goal is to let you *see the plumbing* so you know what the flow you're watching actually means. It is the natural next chapter after [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) and the [loan-plus-options deal that pays market makers](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid), and it leans on the price mechanics from [how crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move). If you want the wider map of who these players are, start at the series hub, [Crypto VC and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers).

## Foundations: inventory, exposure, and the tools to cancel it

Before we can talk about hedging, we need the vocabulary. None of this assumes any finance background — we will define every term the first time it appears and anchor it to something concrete.

### What a market maker is actually trying to do

A **market maker** (MM) is a firm that stands ready to both *buy* and *sell* a token at all times. It posts a **bid** (the price it will buy at) and an **ask** (the price it will sell at) simultaneously, and it profits from the gap between them — the **spread**. If it buys at \$0.99 and sells at \$1.01, it pockets \$0.02 per token, over and over, thousands of times a day. That two-sided quoting is a genuine service: without it, a buyer and a seller who show up at different moments could never trade with each other, and the token's price would gap around wildly. (We built this from the ground up in [what a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does); the TradFi version is [market makers and the spread](/blog/trading/capital-markets/market-makers-and-the-spread-who-provides-liquidity).)

The crucial point for *this* post: the market maker wants to earn that spread **without taking a view on where the price goes**. It is not trying to guess whether the token rises or falls. It just wants to be the toll booth in the middle, collecting a few cents every time someone crosses. The moment it is forced to hold a pile of the coin, it has accidentally become a *speculator* — and that is exactly what it is trying to avoid.

### Inventory: the coin you're forced to hold

**Inventory** is the stock of tokens a market maker is holding at any given moment as a byproduct of quoting. Here is how it accumulates. Suppose the market is falling and everyone is selling. The MM's bid is sitting there, so *it gets hit* — sellers keep trading against its buy order. Fill after fill, the MM ends up *long* a growing pile of the token while the price keeps dropping. It never chose to buy; it just kept its promise to provide a bid, and the market handed it the bag.

That accumulated pile is inventory. It is unavoidable: you cannot make a two-sided market without sometimes ending up net long or net short. The whole discipline of market making is *managing* that inventory so it never becomes a big directional bet.

> Inventory is the market maker's occupational hazard: the price you pay for standing in the middle is that you keep catching whatever everyone else is throwing away.

### Why the maker can't just refuse the inventory

The obvious question is: if inventory is so dangerous, why not just *stop quoting* when it starts to pile up? Two reasons. First, many makers are contractually obligated to quote — a project or exchange pays them (often via the [loan-plus-options deal](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid)) precisely to keep a tight two-sided market up during the exact conditions — a sell-off, a listing, a news shock — when quoting is most dangerous and most valuable. Pulling quotes when it hurts is how a maker loses the mandate.

Second, the alternative to holding inventory is to *widen the spread* so aggressively that no one trades — which earns nothing and defeats the entire business. A maker's edge is quoting *tight enough* to win the flow while quoting *smart enough* to not drown in inventory. It cannot avoid inventory; it can only manage it. And the tool for managing it is not to refuse the coin — it is to accept the coin and immediately cancel its directional risk with a hedge. That is why hedging, not quote-pulling, is the maker's real defense.

### Long, short, and directional exposure

Two words you must own:

- **Long** means you own the asset (or a position that gains when the price rises). If you are long 100,000 tokens, you profit \$1,000 for every \$0.01 the price goes up, and you lose \$1,000 for every \$0.01 it goes down.
- **Short** means you have a position that gains when the price *falls*. You can be short a token by borrowing it and selling it (planning to buy it back cheaper), or — as we'll see — by taking a short position in a derivative. A short 100,000 tokens loses \$1,000 when the price rises \$0.01 and gains \$1,000 when it falls \$0.01. Short is the mirror image of long.

**Directional exposure** (also called **delta**, which we define precisely in a moment) is simply how much your total P&L moves when the price moves. If a \$0.01 move in the token changes your wealth by \$1,000, you have directional exposure. A market maker with a big pile of inventory has big directional exposure — and that is the problem, because it did not want to bet on direction at all.

### Delta: the number that measures the bet

**Delta** is the amount your position's value changes for a one-unit change in the price of the underlying. For a plain spot holding it is beautifully simple: **if you hold N tokens, your delta is +N.** Own 100,000 tokens, and your delta is +100,000 — for every \$1 the token moves, your position moves \$100,000 in the same direction. Short 100,000 tokens, and your delta is −100,000.

Delta is just a precise word for "how big is my directional bet, and which way." A market maker's target is to keep the delta of its *whole book* — spot inventory plus every hedge — as close to **zero** as it can. Zero delta means the price can do whatever it wants and your wealth barely notices. That state has a name, **delta-neutral**, and reaching it is the entire subject of this article.

### Spot, mark-to-market, and notional

Three quick, load-bearing definitions:

- **Spot** means the actual asset, bought and settled right now, at the current ("spot") price. When you buy a token on an exchange and it lands in your wallet, that's a spot purchase. Spot is distinct from a *derivative*, which is a contract whose value merely *references* the token's price (futures, perps, options).
- **Mark-to-market** (MTM) means valuing a position at the current market price *right now*, even if you haven't sold. If you bought 100,000 tokens at \$1.00 and the price is now \$0.90, your position is *marked* at \$90,000 — an unrealized loss of \$10,000. You haven't sold, so the loss isn't "locked in," but it is absolutely real: it is what you'd get if you sold this instant, and it is how every professional desk tracks its P&L minute by minute.
- **Notional** is the total face value of a position: price × quantity. A position of 100,000 tokens at \$1.00 has a notional of \$100,000. Notional matters because fees, funding, and risk limits are all measured against it.

#### Worked example: what "inventory risk" even means

Let's make MTM concrete before we go further. Suppose you are a market maker and, over a busy morning, sellers hit your bid until you are holding **100,000 tokens** that you accumulated at an average price of **\$1.00**. Your inventory notional is 100,000 × \$1.00 = **\$100,000**. Your delta is **+100,000**.

Now the price ticks to \$0.98.

- Your inventory is marked at 100,000 × \$0.98 = **\$98,000**.
- Unrealized P&L on the inventory: \$98,000 − \$100,000 = **−\$2,000**.

You didn't sell anything. You didn't *decide* to bet the token would rise. But because you were forced to hold 100,000 tokens with a delta of +100,000, a 2-cent drop just cost you \$2,000 on paper. **That is inventory risk**: the exposure you carry simply from holding the coins your quoting hands you. The one-sentence intuition: *the moment you hold inventory, the market's direction becomes your P&L, whether you wanted the bet or not.*

The rest of this post is the market maker's answer to that problem.

## 1. Inventory risk: why holding the coin is the problem

Let's see how bad inventory risk can get, and why it is the single thing that keeps a market-making desk up at night.

The spread a maker earns per trade is *tiny*. On a liquid token it might be a fraction of a percent; even on a thin one it's a few percent at most. To make real money, the MM needs *volume* — thousands of round trips, each capturing a sliver. The problem is that a single adverse price move on accumulated inventory can be enormous compared to a day's worth of those slivers.

Here is the asymmetry drawn to scale.

![Unhedged, one bad move erases days of spread: a maker's captured spread is tiny next to the price risk on the coin it is forced to hold.](/imgs/blogs/inventory-risk-hedging-and-delta-neutrality-2.webp)

#### Worked example: an unhedged inventory loss when the price drops

Take our maker holding **100,000 tokens** at an average cost of **\$1.00** — a **\$100,000** notional, delta **+100,000**. Say that over the same session it captured a spread averaging \$0.02 per token across the flow it quoted, and the day's volume earned it about **\$2,000** in gross spread. A good day.

Then the token falls 10%, from \$1.00 to **\$0.90** — an utterly ordinary daily move in crypto.

- Inventory now marked at 100,000 × \$0.90 = **\$90,000**.
- Loss on inventory: \$90,000 − \$100,000 = **−\$10,000**.
- Net for the day: +\$2,000 spread − \$10,000 inventory loss = **−\$8,000**.

The maker did its actual *job* perfectly — quoted tightly, captured the spread, provided liquidity — and still lost \$8,000, because it was carrying a directional bet it never wanted. The \$10,000 hole swallowed five days of \$2,000 spread in a single 10% move. And 10% is nothing; crypto tokens routinely move 20–40% in a day, and a newly listed token can halve in an hour.

This is why an unhedged market maker in crypto is not really a market maker — it's a leveraged directional gambler wearing a liquidity-provider costume. The spread is the business; the inventory is the risk that can end the business. So the maker's entire operational focus becomes: **cancel the inventory's directional exposure as fast as it appears.** That cancellation is hedging, and it is where we go next.

*What this costs / when it breaks:* the danger scales with how much inventory you're forced to hold and how fast the price can move. On a deep, liquid token you accumulate slowly and can offload quickly; on a thin, freshly listed token — exactly the kind of token whose project *pays* a maker to quote it, per the [loan-plus-options deal](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid) — inventory piles up fast and the price gaps hard. Thin books are where inventory risk is most lethal.

## 2. The hedge: shorting a perp against spot inventory

A **hedge** is a second position taken specifically to cancel the risk of a first one. If your first position gains when the price rises and loses when it falls, a hedge is something that does the *opposite* — loses when the price rises, gains when it falls — in equal measure. Put them together and the price moves cancel out. You are left with whatever *isn't* price risk: in the maker's case, the spread.

The market maker's inventory is **long** spot (delta +100,000). To cancel it, the maker needs a **short** position of the same size (delta −100,000). Add them up and the net delta is zero. The question is *what* to short. Shorting the token on spot means borrowing 100,000 real tokens and selling them — possible, but borrow is often unavailable, expensive, or slow for the kind of small-cap tokens makers quote. So in crypto the hedge of choice is almost always a **derivative**, and specifically the **perpetual swap**.

### The perpetual swap, from zero

A **future** is a contract to settle the difference between an agreed price and the actual price of an asset at some later time. If you are **short** one token of futures at \$1.00 and the price ends at \$0.90, you receive \$0.10; if it ends at \$1.10, you pay \$0.10. Crucially, a futures short gives you the exact mirror of a spot long *without needing to borrow and sell the real coin* — you just enter a contract.

A **perpetual swap** (a "perp") is crypto's dominant invention: a future *with no expiry date*. It never settles and never rolls over; you can hold the position forever. That's wildly convenient for hedging, because your inventory doesn't have an expiry date either — you want a hedge you can hold open as long as you're carrying the coin. Perps are the single most-traded instrument in crypto; the mechanism was pioneered by BitMEX and is now offered by every major venue.

But "no expiry" creates a problem the perp has to solve with a clever trick, which we'll get to in Section 3. First, let's see the hedge work.

### Building the hedged book

The maker holds **+100,000 tokens** on spot (delta +100,000). It opens a **short perpetual position on 100,000 tokens** at the current price of \$1.00 (delta −100,000). Now look at the combined book:

- **Total delta = +100,000 − 100,000 = 0.** The book is **delta-neutral**.

The payoff picture below is the heart of the whole post. The long-spot leg (solid line) gains as the price rises and loses as it falls. The short-perp leg (dashed line) does the exact opposite. Add them together and you get the thick, flat line: a horizontal payoff that doesn't care about the price at all, sitting at the level of the spread the maker already captured.

![Spot plus short perp: the long-spot gain and short-perp loss offset at every price, so the combined book lands on a flat +$2,000 spread line.](/imgs/blogs/inventory-risk-hedging-and-delta-neutrality-3.webp)

#### Worked example: the same position, hedged

Same maker, same 100,000 tokens bought at \$1.00, same \$2,000 of spread captured — but this time it opened a short perp on 100,000 tokens at \$1.00 the moment the inventory built up. The token falls to **\$0.90**.

- **Spot leg:** 100,000 × (\$0.90 − \$1.00) = **−\$10,000** (the inventory loss, unchanged).
- **Short perp leg:** you were short at \$1.00, price is \$0.90, so you gain 100,000 × (\$1.00 − \$0.90) = **+\$10,000**.
- **Price P&L: −\$10,000 + \$10,000 = \$0.** It cancels exactly.
- **Net for the day: +\$2,000 spread + \$0 = +\$2,000.**

Run it the other way to be sure. The token *rises* to \$1.10:

- Spot leg: 100,000 × (\$1.10 − \$1.00) = **+\$10,000**.
- Short perp leg: 100,000 × (\$1.00 − \$1.10) = **−\$10,000**.
- Price P&L: **\$0**. Net: still **+\$2,000**.

Whatever the price does, the hedged maker keeps its spread and nothing else. That is the flat line in the figure, and it is *exactly* what a market maker wants: to be paid for providing liquidity, not to be paid or punished for guessing direction. The one-sentence intuition: *the hedge converts a directional gamble back into a fee business by making the coin's price irrelevant.*

### Delta-neutral, stated precisely

We can now define the term the whole business orbits. A book is **delta-neutral** when the sum of the deltas of all its positions is (about) zero, so that a small change in the underlying price produces (about) no change in the book's value. The balance below is the accountant's view of our hedged maker: a long leg of +100,000 delta on the left, a short leg of −100,000 delta on the right, and every price scenario cancelling row by row.

![Delta-neutral: the long spot leg and the short perp leg carry equal and opposite delta, so their P&L cancels on every move and net exposure falls to about zero.](/imgs/blogs/inventory-risk-hedging-and-delta-neutrality-4.webp)

Notice the word "about." Real delta-neutrality is never perfect and never permanent — the hedge drifts, the two instruments aren't identical, and holding the short costs money. Those imperfections are the subject of Sections 3, 4, and 5. But the core is now in place: **a market maker stays flat by pairing every unit of spot inventory with an offsetting unit of short derivative.**

### Where the hedge actually lives

It's worth being concrete about *where* the two legs sit, because the split is the source of a lot of the residual risk. The **spot inventory** lives on a spot exchange or in a wallet — real tokens the maker owns. The **hedge** lives somewhere else entirely: a short position on a derivatives venue (a perpetual on Binance, Bybit, OKX, or an on-chain venue like Hyperliquid; a dated future on the CME or a crypto exchange). The two legs are on different order books, often at different firms, settled in different collateral.

That separation matters. The short perp is a *margined* position: the maker posts collateral (margin) and the exchange marks it to market continuously. If the price rips upward, the short's mark-to-market loss grows and the exchange demands more margin; the maker must move collateral fast or risk liquidation. Meanwhile the offsetting *gain* is sitting in the spot leg on a different venue, where it can't instantly be used to meet the perp's margin call. So a maker running a delta-neutral book is also running a *treasury* operation — shuttling collateral between venues, keeping buffers, and making sure the profitable leg can always fund the losing leg's margin. The hedge cancels the price, but it creates a plumbing problem, and plumbing problems are what break desks in a fast market.

*What this costs / when it breaks:* the hedge is not magic — it moves the risk, it doesn't delete it. You've swapped "price risk on the coin" for a smaller set of frictions: the cost of holding the short (funding), the risk that spot and the derivative don't move perfectly together (basis risk), and the risk that a violent move forces your short to be liquidated — on one venue — before the spot gain on another venue can be mobilized. We spend the rest of the post on exactly those.

## 3. What the hedge costs: the funding rate

Here is the puzzle the perpetual swap had to solve. A normal future has an expiry date, and at expiry its price is *forced* to equal the spot price (the contract settles against spot). That anchor keeps the future tethered to reality. A perp has no expiry — so what stops its price from drifting far away from spot forever?

The answer is the **funding rate**: a small periodic payment exchanged directly between the longs and the shorts to keep the perp's price glued to spot.

### How funding works

Every few hours (on most venues, every **8 hours** — at 00:00, 08:00, and 16:00 UTC), the exchange looks at whether the perp is trading *above* or *below* the spot price and makes one side pay the other:

- If the perp trades **above** spot (too many aggressive longs pushing it up), the **funding rate is positive**, and **longs pay shorts**. This makes being long more expensive, nudging longs to close and pulling the perp back down toward spot.
- If the perp trades **below** spot (too many aggressive shorts), the **funding rate is negative**, and **shorts pay longs**, nudging the perp back up.

The payment is a percentage of your position's notional. The **"default" funding rate baked into most venues is 0.01% every 8 hours** — the number the rate reverts to when the perp is trading right at spot ([Coinbase](https://www.coinbase.com/learn/perpetual-futures/understanding-funding-rates-in-perpetual-futures); [The Block funding data](https://www.theblock.co/data/crypto-markets/futures/btc-funding-rates)). Because it's paid three times a day, every day, it annualizes to roughly:

`0.01% × 3 payments/day × 365 days ≈ 10.95% per year`

That is the carrying cost of a perp position at baseline — small per tick, meaningful over a year. The timeline below shows how those little payments accumulate.

![The funding-rate meter: a $100,000 perp hedge exchanges about $10 every 8 hours at the default rate, roughly 11% a year, in whichever direction the rate points.](/imgs/blogs/inventory-risk-hedging-and-delta-neutrality-5.webp)

#### Worked example: the funding cost of carrying the hedge

Our maker is short a perp on **100,000 tokens** at **\$1.00** — a **\$100,000** notional hedge. Assume funding sits at the 0.01%-per-8h default and, for now, that it's *positive* is a cost to the side that pays.

- **Per funding tick:** 0.01% × \$100,000 = **\$10** exchanged.
- **Per day:** 3 ticks × \$10 = **\$30**, i.e. 0.03% of notional.
- **Over 30 days:** 90 ticks × \$10 = **\$900**, i.e. 0.9% of notional.
- **Annualized:** 0.01% × 3 × 365 = 10.95%, i.e. about **\$10,950** on a \$100,000 hedge held all year.

So carrying the hedge for a month costs on the order of \$900 in funding at baseline — a real drag, but small next to the \$10,000 inventory loss the hedge saved us from in Section 2. That trade is almost always worth making.

### The sign flips — and sometimes funding *pays* the hedger

Here is the part that trips people up. Our maker is **short** the perp. When funding is **positive** (longs pay shorts), the maker is on the *receiving* side — **funding is income, not a cost.** When funding is **negative** (shorts pay longs), the maker pays. Which way it goes depends entirely on crowd positioning, not on anything the maker did.

In crypto, the crowd is usually net long and over-eager, so funding is *positive* most of the time — which means the short hedge frequently gets *paid* to exist. A maker hedging long inventory with a short perp in a bullish, high-funding market can actually earn carry on its hedge on top of the spread. That is not a footnote; it is a core reason delta-neutral "cash-and-carry" strategies are a real business (more on that in the real-markets section). The funding cost of the hedge is a *live, two-sided number* — sometimes a fee, sometimes a rebate — and a serious desk models it continuously.

Why is crypto funding structurally biased positive? Because the marginal crypto participant is a retail trader who wants *leveraged long* exposure and expresses it on perps. That chronic demand to be long pushes perp prices above spot and keeps funding positive, so the natural *counterparty* — the patient short who provides that leverage — gets paid a premium for it. A hedging market maker is, almost by accident, exactly that counterparty: it needs to be short to cancel its long inventory, and the crowd pays it to take the other side. This is the same structural long-bias that makes the [basis](/blog/trading/crypto-players/how-crypto-prices-actually-move) trade profitable, and it means the hedge a maker is *forced* to hold often turns into an income stream rather than a cost. It is one of the quiet reasons market-making in crypto can be so profitable: you get paid the spread for quoting, and frequently paid *again* — via funding — for holding the very hedge that lets you quote safely.

#### Worked example: when funding turns extreme

Funding is not always 0.01%. In euphoric bull markets it can run an order of magnitude higher. During the 2021 bull run, Bitcoin perpetual funding reached **0.15–0.20% per 8 hours** at peak; in April 2021 it sat above roughly 0.15% for a sustained stretch — which annualizes to **roughly 160–220% a year** across that range — right before Bitcoin fell from about **\$64,000 to about \$30,000** ([FXStreet](https://www.fxstreet.com/cryptocurrencies/news/bitcoin-funding-rates-jump-to-100-sparking-opportunity-for-savvy-traders-202402270607); [OKX](https://www.okx.com/en-us/learn/bitcoin-funding-rates-market-sentiment)). Let's price a hedge in that environment.

At **0.15% per 8h** on our **\$100,000** short perp:

- Per tick: 0.15% × \$100,000 = **\$150**.
- Per day: 3 × \$150 = **\$450**, i.e. 0.45%/day.
- Annualized: 0.15% × 3 × 365 ≈ **164%**.

But remember the sign: in that overheated market the crowd was wildly net long, so funding was strongly *positive* and the **short** side was *receiving* it. A maker short-hedging its inventory would have been collecting roughly \$450 a day per \$100,000 of hedge — a spectacular rebate — precisely because everyone else was crowding the long side. High positive funding is simultaneously a warning that the market is overheated and a reason the hedgers and cash-and-carry desks are getting paid to lean against it. The one-sentence intuition: *funding is the price of crowd conviction, and the hedger is usually on the other side of the crowd, collecting it.*

*What this costs / when it breaks:* funding risk is real and it swings. A maker that assumed it would *receive* funding can find the sign flip against it in a sudden risk-off move, turning a rebate into a bill. And extreme funding — in either direction — is itself a signal that a violent move (and a wave of liquidations) may be coming, which is exactly when a hedge is most likely to be stress-tested.

## 4. Dated futures, basis, and the roll

Perps aren't the only hedging tool. The other is the **dated future** — a future with an actual expiry date (say, quarterly: the last Friday of March, June, September, December). Dated futures are how a lot of institutional hedging happens, especially on regulated venues like the CME, and they introduce one more concept you must own: the **basis**.

### The basis, from zero

The **basis** is the difference between a future's price and the spot price of the same asset: `basis = futures price − spot price`. Because a dated future settles against spot *at expiry*, its price is *forced* to converge to spot as expiry approaches. If today a 3-month future trades at \$1.02 while spot is \$1.00, the basis is \$0.02 — a 2% premium — and that entire premium must melt to zero by expiry.

A positive basis (futures above spot) is called **contango**; a negative basis (futures below spot) is called **backwardation**. Crypto futures are *usually* in contango — the future trades at a premium — because demand for leveraged long exposure is chronically high and traders will pay up for it. The chart below shows the convergence: the future starts above spot and slides down to meet it at expiry.

![The basis: a dated future starts at a premium and its price melts to zero basis by expiry, so shorting the future while holding spot locks in the initial basis.](/imgs/blogs/inventory-risk-hedging-and-delta-neutrality-6.webp)

### Why the basis is a locked-in return for a hedger

Here's the beautiful part for someone who is hedging. If you *hold spot* and *short the future* against it, you are delta-neutral (long spot + short future = zero delta, same as with a perp). But because the future is trading *above* spot and must converge *down* to it, your short future is guaranteed to gain the basis as it converges — regardless of what the price does. This is the famous **cash-and-carry** trade.

#### Worked example: a basis / cash-and-carry walkthrough

Suppose spot is **\$1.00** and the 3-month future trades at **\$1.02** (a \$0.02, or 2%, basis; annualized, that 2% over a quarter is roughly **8% a year**). You hold **100,000 tokens** on spot and short **100,000 tokens** of the future at \$1.02. Consider two scenarios at expiry:

**Scenario A — price rises to \$1.20 by expiry.**

- Spot leg: 100,000 × (\$1.20 − \$1.00) = **+\$20,000**.
- Short future: you were short at \$1.02, it settles at \$1.20, so 100,000 × (\$1.02 − \$1.20) = **−\$18,000**.
- Total: +\$20,000 − \$18,000 = **+\$2,000**.

**Scenario B — price falls to \$0.80 by expiry.**

- Spot leg: 100,000 × (\$0.80 − \$1.00) = **−\$20,000**.
- Short future: 100,000 × (\$1.02 − \$0.80) = **+\$22,000**.
- Total: −\$20,000 + \$22,000 = **+\$2,000**.

In *both* cases you make exactly **\$2,000** — which is the basis (\$0.02 × 100,000) you locked in at the start. The price move is irrelevant; your return is the basis, captured as the future converges to spot. That is why a hedged inventory book on dated futures earns (or pays) the basis, and why "the basis" is quoted as an annualized yield the way a bond is. The one-sentence intuition: *shorting a premium future against spot turns a directional coin into a fixed, price-independent yield equal to the basis.*

### The roll

Dated futures expire, but your inventory doesn't. So when the near contract approaches expiry, the maker **rolls** the hedge: it closes the expiring short and opens a new short in the next contract. This is the one real cost of using dated futures instead of perps — each roll crosses a spread and re-prices the basis, which may have moved. If the basis has compressed (the new contract's premium is smaller), your locked-in yield on the next leg is lower; if it has widened, higher. Rolling is the dated-future analog of paying funding on a perp: it is the recurring friction of keeping a hedge alive on an instrument that keeps expiring.

*What this costs / when it breaks:* the basis can go *negative* (backwardation), which flips the economics — a short-future hedger in backwardation *pays* the basis instead of earning it, because the future must converge *up* to spot. Backwardation is rarer in crypto and usually appears in sharp sell-offs, when everyone wants short exposure and futures trade below spot. We'll see a real, dated example of exactly that in the real-markets section.

### Perp vs dated future: two ways to hold the same hedge

A maker can hedge the same inventory with either instrument, and the choice shapes what the hedge costs and how it can go wrong. The two are mirror images of the same idea — a short that cancels long spot — differing mainly in *how* they stay tethered to spot and *what* recurring friction that tether creates.

| Attribute | Perpetual swap | Dated future |
|---|---|---|
| Expiry | None — hold it open forever | Fixed date (e.g. quarterly) |
| Tether to spot | Funding paid every 8 hours | Forced convergence at expiry |
| Recurring friction | Funding (you pay *or* receive it) | The roll — re-pricing the basis each expiry |
| Typical carry | ~10.95%/yr at the 0.01%/8h default; swings hard with the crowd | The basis; roughly 15–30%/yr annualized in bull runs |
| Main hedge risk | Funding sign flips against you; short can be liquidated | Basis moves or goes negative; roll slippage |
| Who leans on it | On-chain, DeFi, and retail-facing makers | Institutional and CME hedgers, ETF basis desks |

The practical rule of thumb: **perps are the convenient default** — no expiry, deep liquidity, hedge-and-forget — but their funding is a live number that can turn from rebate to bill overnight. **Dated futures lock in a known basis** for a fixed window, which institutions prefer for planning, but they force you to roll and expose you to whatever the basis has become at each roll. Many desks run both: perps for the fast, continuously-rehedged inventory book, dated futures for the slower, size-able positions where a locked basis is worth the roll.

## 5. Staying delta-neutral in practice

The worked examples above are clean because they freeze the world. Real hedging is a *continuous* activity, because the neutrality you set up at 9:00 a.m. degrades all day. Here is what actually breaks and how desks manage it.

### Delta drift and rehedging

The maker's inventory changes every second — every fill adds or removes tokens. A hedge sized for +100,000 delta is wrong the instant the next 5,000 tokens come in. So desks **rehedge** constantly: they monitor net delta in real time and top up or trim the short whenever it drifts past a threshold.

#### Worked example: a rehedge

Start delta-neutral: **+100,000** spot, **−100,000** short perp, net **0**. Over the next hour, sellers hit the maker's bid and it accumulates another **20,000 tokens**. Now:

- Spot delta: **+120,000**.
- Perp delta: **−100,000**.
- **Net delta: +20,000** — the book is now long, exposed to a drop on 20,000 tokens (a \$2,000 hit per \$0.10 move).

To restore neutrality, the maker sells another **20,000 tokens** of perp short, taking the perp leg to −120,000 and net delta back to 0. Multiply this by thousands of fills a day and you see why market-making is an *engineering* problem: the hedge is a control loop that has to run faster than the inventory changes. The intuition: *delta-neutral is not a state you reach, it's a state you continuously re-impose.*

### The residual risks that survive the hedge

Even a perfectly rehedged book is not risk-free. The hedge trades one big risk for a handful of smaller ones:

- **Basis / funding risk.** The hedge instrument (perp or future) does not track spot *perfectly*. Funding can turn against you; the basis can move. Your delta is neutral, but your *carry* is not, and it can cost more than the spread you're earning.
- **Liquidation risk on the short.** The short perp is a leveraged position posted with margin. In a violent *upward* move, the short loses money fast, and if the maker can't post margin quickly enough, the exchange can **liquidate** the short — leaving the maker suddenly long, unhedged, at the worst possible moment. The spot leg is fine on paper, but the hedge got blown out from under it.
- **Gap / execution risk.** The two legs live on different venues (spot on one exchange, perp on another). If one venue halts, lags, or the price gaps between fills, the hedge can slip. On thin tokens, selling the perp to rehedge can itself move the perp price.
- **Correlation risk on cross-hedges.** Sometimes there's no liquid perp for the exact token, so the maker hedges with a *correlated* asset (say, a basket or a larger-cap proxy). That hedge is only as good as the correlation, which breaks exactly when it matters most.

#### Worked example: a cross-hedge that slips

Say the maker holds **100,000 units** of a small token, "\$SMOL," at **\$1.00** (\$100,000 notional), but \$SMOL has no liquid perp. The maker hedges with a short on a larger, correlated token, "\$BIG," that historically moves about 1-for-1 with \$SMOL — it shorts **\$100,000** of \$BIG perp as a proxy hedge.

A market-wide sell-off hits. \$BIG falls 10%, so the short gains **+\$10,000**. But \$SMOL, being smaller and thinner, falls *20%* — its inventory loses **−\$20,000**.

- Spot leg (\$SMOL): **−\$20,000**.
- Cross-hedge (short \$BIG): **+\$10,000**.
- Net: **−\$10,000.**

The hedge covered only *half* the loss, because the correlation the maker relied on broke exactly when it was needed: in a panic, small tokens fall harder than large ones. The book was "delta-neutral" on paper against \$BIG, but *not* against its own inventory. The one-sentence intuition: *a hedge on a different asset is only ever as good as a correlation that tends to fail in the very moments you're hedging against.* This is why a real perp on the exact token — even an expensive one — beats a cheap proxy, and why makers charge more (or refuse) to quote tokens with no direct hedge.

The takeaway: hedging is not the elimination of risk, it's the *transformation* of a large, obvious risk (price) into a bundle of smaller, subtler ones (funding, liquidation, basis, execution). A good desk is one that manages the bundle better than its competitors. A blown-up desk is usually one whose hedge failed under stress — the short got liquidated, or the correlation broke — and it was suddenly, involuntarily, holding the coin.

## Common misconceptions

**"If a market maker is shorting the token, it must be bearish."** Almost never. The short is a *hedge* against long spot inventory, sized to cancel it, not a bet. A maker with +100,000 spot and −100,000 perp has a *net* view of exactly zero. Reading its short as a directional signal is like assuming someone who bought fire insurance is hoping their house burns down.

**"Delta-neutral means no risk."** No. Delta-neutral means no *first-order price* risk. It leaves funding risk, basis risk, liquidation risk on the leveraged leg, execution/gap risk, and (for options books) second-order "gamma" risk. Delta-neutral desks blow up regularly — not from the price, but from the frictions the hedge introduced.

**"The hedge is free / the maker just pockets the spread."** The hedge has a running cost (or benefit): funding on a perp, the basis-and-roll on a dated future. Sometimes that carry is positive (the short gets *paid*); sometimes it's a drag that eats the spread. A maker that quotes a token whose perp funding is deeply negative may find the hedge costs more than the spread is worth — which is one reason some tokens are expensive or impossible to make markets in.

**"Funding is a fee the exchange charges."** Funding is paid *between traders* — longs to shorts or shorts to longs — not to the exchange. The exchange just computes and routes it. It exists to tether the perp's price to spot, not to generate revenue for the venue.

**"Market makers want the price to go up because they hold the tokens."** A *hedged* maker is indifferent to the price by construction — that's the whole point of this article. The player who wants the price up is the one holding *unhedged* tokens with a call-option kicker, which is a different arrangement (the [loan-plus-options deal](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid)). Don't conflate the market-making function with the option position; a firm can wear both hats, but they are different bets.

**"A big short print means someone knows something."** In a market dominated by hedgers, a large short is more often the offsetting leg of a large *long* held somewhere you can't see — a maker's inventory, a fund's spot position, an ETF's holdings. The short is visible; the long it balances often isn't.

## How it shows up in real markets

Hedging flow is not abstract — it is one of the largest, most persistent forces in crypto derivatives. Here is where you can actually *see* it, with real, dated figures.

### 1. The perpetual funding market is the maker's hedge bill, in public

Perps are the most-traded instrument in crypto, and their open interest is enormous. As of January 2026, the total open interest in Ethereum futures alone was about **\$37.9 billion** according to CoinGlass — led by Binance (~\$8.7bn), the CME (~\$5.7bn), and Gate (~\$3.8bn) — and the on-chain perp venue Hyperliquid had reached roughly **\$15 billion** of perp open interest by October 2025 ([CoinGlass](https://www.coinglass.com/open-interest/ETH)). A large share of that open interest is *not* directional conviction; it is hedgers — market makers, funds, and cash-and-carry desks — carrying offsetting shorts against spot they hold elsewhere. The funding rate on all that open interest is, quite literally, the aggregate hedging bill (or rebate) being settled every 8 hours.

### 2. The 2021 euphoria: funding as an overheating gauge

In the 2021 bull run, Bitcoin perp funding reached **0.15–0.20% per 8 hours** at its peak — roughly 160–220% annualized — as the crowd piled into leveraged longs ([FXStreet](https://www.fxstreet.com/cryptocurrencies/news/bitcoin-funding-rates-jump-to-100-sparking-opportunity-for-savvy-traders-202402270607)). That extreme positive funding did two things at once: it *paid* every short-hedger and cash-and-carry desk handsomely for leaning against the crowd, and it flashed a giant warning that the market was overheated. In April 2021, with funding sustained above ~0.15%, Bitcoin then fell from roughly **\$64,000 to about \$30,000**. The lesson makers live by: when funding is screaming, the hedge is a rebate *and* a signal.

### 3. February 2024: the funding spike

In late February 2024, as Bitcoin rallied hard, perpetual funding rates surged to their highest levels in over two years, with the jump equating to roughly **100% annualized** at the peak ([FXStreet, 27 Feb 2024](https://www.fxstreet.com/cryptocurrencies/news/bitcoin-funding-rates-jump-to-100-sparking-opportunity-for-savvy-traders-202402270607)). For anyone hedging long inventory with a short perp, that was a window of unusually large carry income — and, again, a tell that leveraged longs were crowding in.

### 4. The spot-ETF basis trade: hedging flow you can measure

The launch of US spot Bitcoin ETFs in January 2024 created a clean, regulated way to hold spot, and institutions immediately paired it with short CME futures to harvest the basis. By late Q1 2024, leveraged funds held a **record net short in CME Bitcoin futures of about 16,102 contracts** (each contract = 5 BTC, so roughly **80,500 BTC**) — the largest since the CME launched Bitcoin futures in 2017 — and added around **\$1.6 billion** of short notional in March 2024 alone. Analysts attributed roughly an **80% surge in CME Bitcoin futures open interest** to these basis trades ([The Block](https://www.theblock.co/post/301041/spot-etf-basis-trades-driving-80-surge-in-cme-bitcoin-futures-open-interest-analyst-says); [CME OpenMarkets](https://www.cmegroup.com/openmarkets/equity-index/2025/Spot-ETFs-Give-Rise-to-Crypto-Basis-Trading.html)). This is the single clearest example that a giant "short" position can be almost entirely *hedging*, not bearishness: those funds were not betting against Bitcoin; they were long spot via the ETF and short futures to lock the basis, delta-neutral. During bull phases the annualized CME basis has run roughly **15–30%** — a rich yield that pulls enormous hedged capital into the trade.

### 5. Late 2025: when the basis went negative

The basis is not always a gift. In early December 2025, the CME Bitcoin annualized basis fell to about **−2.35%** — its deepest **backwardation** since the dislocations of the FTX collapse in November 2022, when it briefly approached **−50%** ([CoinDesk, 3 Dec 2025](https://www.coindesk.com/markets/2025/12/03/bitcoin-futures-return-to-deepest-backwardation)). Backwardation flips the cash-and-carry economics: with the future *below* spot, a short-future hedger *pays* the basis instead of earning it, and the trade that printed money in 2024 becomes a drag. That is exactly the "funding/basis risk" from Section 5 showing up at market scale — and it is why hedging desks watch the sign of the basis as closely as its size.

### 6. Reading the tape: what the hedging flow looks like to you

Put it all together and you get a practical, defensive skill: most of the maker selling, shorting, and open-interest you watch on the tape is *hedging*, not a directional bet against you. The matrix below decodes the signals a retail trader sees into what they usually mean — and the "tell" that distinguishes a real directional move from a hedge.

![What you see on the tape vs. what the market maker is actually doing: most maker selling and shorting decodes to a hedge against inventory, not a bet against you.](/imgs/blogs/inventory-risk-hedging-and-delta-neutrality-7.webp)

## When this matters to you

If you trade tokens — especially newly listed, thin ones — the hedging machinery in this post is on the other side of a lot of your fills, and understanding it changes how you read the market.

- **Don't mistake a hedge for a signal.** When you see a maker sell into your buy, or a large short appear, or open interest spike, your first question should be "is this a directional bet, or a hedge against a position I can't see?" Usually it's the latter. The tell is *simultaneity and size-matching*: a spot sale paired with an equal-size perp short opened in the same window is a hedge, not a forecast. Rising open interest while price stays flat is neutral, hedged flow — not conviction.
- **Funding is a crowd-positioning gauge, not a direction.** Positive funding means the crowd is net long and paying to be there; deeply negative funding means shorts are crowded and paying. Extreme funding in *either* direction is a warning that the market is one-sided and a violent reversion (and liquidation cascade) is more likely. You can read the public funding rate the way a maker does — as the temperature of the crowd.
- **Thin tokens are where hedging breaks — and where you're most exposed.** The tokens whose projects *pay* makers to quote them are exactly the ones with thin spot books, no deep perp, and violent moves. That's where hedges slip, shorts get liquidated, and the maker can get caught long into a crash — dumping inventory into you as it unwinds. If a token has a huge perp funding rate or a wild basis, that's the market telling you the hedging is hard and the risk is high.
- **This is educational, not advice.** None of this tells you whether to buy or sell anything. It tells you how to *see* the flow, so that when the tape looks scary, you can ask whether you're watching a bet or a hedge — and price your own risk accordingly.

The deepest point is the one we started with: a market maker wants the spread, not the coin. Everything it does around your trade — the shorting, the funding, the basis, the constant rehedging — is machinery built to hold *nothing*, to be paid for providing liquidity while betting on nothing at all. Once you can see that machinery, a lot of "someone is dumping" panic resolves into "someone is hedging," and that is a calmer, more accurate way to read a market where the biggest players are usually trying their hardest to stay flat.

## Sources & further reading

Primary and market-data sources behind the figures in this post (as-of dates noted inline where the number is a moving one):

- [Understanding funding rates in perpetual futures — Coinbase Learn](https://www.coinbase.com/learn/perpetual-futures/understanding-funding-rates-in-perpetual-futures) — the 8-hour funding mechanism and the 0.01% default.
- [Bitcoin perpetual futures funding rates — The Block](https://www.theblock.co/data/crypto-markets/futures/btc-funding-rates) — live and historical BTC perp funding data.
- [Bitcoin funding rates and market sentiment — OKX](https://www.okx.com/en-us/learn/bitcoin-funding-rates-market-sentiment) and [FXStreet, 27 Feb 2024](https://www.fxstreet.com/cryptocurrencies/news/bitcoin-funding-rates-jump-to-100-sparking-opportunity-for-savvy-traders-202402270607) — the 2021 euphoria (0.15–0.20% per 8h) and the February 2024 spike (~100% annualized).
- [Spot ETF basis trades driving CME open interest — The Block](https://www.theblock.co/post/301041/spot-etf-basis-trades-driving-80-surge-in-cme-bitcoin-futures-open-interest-analyst-says) and [Spot ETFs give rise to crypto basis trading — CME OpenMarkets](https://www.cmegroup.com/openmarkets/equity-index/2025/Spot-ETFs-Give-Rise-to-Crypto-Basis-Trading.html) — the record ~16,102-contract net short, the ~\$1.6bn March 2024 short notional, the ~80% OI surge, and 15–30% bull-market basis.
- [Bitcoin futures return to deepest backwardation since FTX — CoinDesk, 3 Dec 2025](https://www.coindesk.com/markets/2025/12/03/bitcoin-futures-return-to-deepest-backwardation) — CME annualized basis at −2.35% (late 2025) vs ~−50% in November 2022.
- [Ethereum futures open interest — CoinGlass](https://www.coinglass.com/open-interest/ETH) — ~\$37.9bn ETH futures OI (Jan 2026) and venue breakdown; Hyperliquid ~\$15bn perp OI (Oct 2025).

Sibling posts in this series:

- [What a crypto market maker actually does](/blog/trading/crypto-players/what-a-crypto-market-maker-actually-does) — two-sided quoting, the spread, and inventory from zero.
- [The loan-plus-options deal: how market makers get paid](/blog/trading/crypto-players/the-loan-plus-options-deal-how-market-makers-get-paid) — the crypto-native MM contract and the call-option kicker.
- [How crypto prices actually move](/blog/trading/crypto-players/how-crypto-prices-actually-move) — order books, thin float, and slippage.
- [Crypto VC and market makers](/blog/trading/crypto/crypto-vc-and-market-makers) — the series hub on who moves crypto prices.

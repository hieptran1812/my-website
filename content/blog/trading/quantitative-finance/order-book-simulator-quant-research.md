---
title: "Building a limit order book simulator from scratch"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A hands-on, first-principles build of the core data structure of modern markets: the limit order book and its matching engine. We define every term from zero, then code add, match, partial-fill, and cancel; walk a 500-share market order through three price levels for a $5,007.00 fill; compute order-flow imbalance and the microprice on a live book; reconstruct the book from a message feed; and solve five interview and take-home problems the way Jane Street, Optiver, HRT, Jump, and Citadel Securities actually ask them."
tags:
  [
    "limit-order-book",
    "matching-engine",
    "market-microstructure",
    "price-time-priority",
    "order-flow-imbalance",
    "microprice",
    "quant-interviews",
    "quant-research",
    "high-frequency-trading",
    "data-structures",
    "execution",
    "python",
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — The limit order book (LOB) is the central data structure of every modern electronic market, and building a matching engine for it teaches you order types, price-time priority, and the microstructure that execution desks and HFT firms trade against.
>
> - A book is two sorted stacks of resting orders: **bids** (buyers) below, **asks** (sellers) above, with an untraded **spread** between the best bid and best ask.
> - The matching rule almost everywhere is **price-time priority**: better prices fill first, and within one price, the order that arrived first fills first (FIFO).
> - A fast book is a **price-sorted map of levels**, each level holding a **FIFO queue** of orders; that combination gives you the best price in `O(log n)` and correct time priority for free.
> - A market buy for 500 shares **walks the asks**: 300 at \$10.01, 150 at \$10.02, 50 at \$10.03, for a total of **\$5,007.00** and a volume-weighted price of **\$10.014** — \$4.50 of slippage versus the \$10.005 mid.
> - Two signals fall straight out of the book: **order-flow imbalance** (here +0.60, a buy lean) and the **microprice** (\$10.008, leaning toward the thinner ask) — both are real, tradeable microstructure features.

Here is a question that sounds simple and isn't: when you click "buy 100 shares of Apple at market," where does your order go, who decides what price you pay, and why do you sometimes pay more than the price you saw a half-second ago? The answer to all three is one object — the **limit order book** — and one piece of software that operates on it, the **matching engine**. Almost every stock, futures contract, option, and crypto pair on Earth trades through some version of this exact machine. Learn it once and you understand the plumbing under trillions of dollars a day.

This post builds that machine from nothing. We will define every term as it appears, write real Python you can run, and ground every idea in worked examples with concrete share counts and dollar figures. By the end you will be able to code a matching engine, walk an order through the book by hand, reconstruct a book from a raw message feed, and compute the two microstructure signals interviewers love to ask about. That is, not coincidentally, exactly the surface area that quant trading and dev interviews and take-homes at firms like Jane Street, Optiver, HRT, Jump, and Citadel Securities probe.

![Order-book ladder showing asks above the spread in red and bids below in green, with price, size, and side columns and a one-cent spread](/imgs/blogs/order-book-simulator-quant-research-1.png)

The ladder above is the mental model to hold the whole way through: a stack of prices, sell orders resting on top, buy orders resting underneath, and a thin gap in the middle where no trade has happened yet. Everything else in this article is a rule for how that stack changes when a new order arrives.

## Foundations: what an order book actually is

Let's build the vocabulary from zero. No prior finance knowledge is assumed; if you have traded before, skim this, but every later section leans on these exact definitions.

An **order** is an instruction to buy or sell a quantity of something at a price. The "something" is the **instrument** — a stock, a futures contract, a coin. We will use a fictional stock and round numbers throughout. The **quantity** is measured in **shares** (or contracts, or coins); we will say shares. The **side** is either **buy** (you want to own more) or **sell** (you want to own less).

A **limit order** names a worst acceptable price. "Buy 100 shares at \$10.00 limit" means: buy up to 100 shares, but never pay more than \$10.00 a share. If nobody will sell to you at \$10.00 or below right now, your order doesn't vanish — it **rests** in the book, waiting, advertising to the world that you'll buy 100 at \$10.00 whenever a seller shows up. A resting limit order is a standing offer; it is **liquidity** that other people can trade against.

A **market order** names no price. "Buy 100 at market" means: buy 100 shares right now, whatever it costs, sweeping the cheapest available sellers. A market order does not rest — it **takes** liquidity immediately and is gone. This is the first crucial split in all of trading: **makers** post resting limit orders and provide liquidity; **takers** send marketable orders that consume it.

Now stack the resting limit orders by price. All the resting **buy** orders form the **bid** side. All the resting **sell** orders form the **ask** side (also called the **offer** side). Each distinct price with at least one resting order is a **price level**. The **best bid** is the highest price any buyer is willing to pay; the **best ask** is the lowest price any seller is willing to accept. In the ladder figure, the best bid is \$10.00 (500 shares want to buy there) and the best ask is \$10.01 (300 shares want to sell there).

The gap between them has two names you must know cold. The **spread** is the difference: best ask minus best bid, here \$10.01 − \$10.00 = **\$0.01**, one cent. The **mid** (or **midprice**) is the simple average: (\$10.00 + \$10.01) / 2 = **\$10.005**. The spread is the cost of immediacy — the round-trip toll you pay to buy now and sell now — and the mid is the market's rough "fair" reference price. Notice that no trade can happen *inside* the spread right now: the highest bidder won't pay as much as the lowest seller demands. A trade only happens when someone **crosses the spread** — a buyer willing to pay the ask, or a seller willing to accept the bid.

![Vertical price axis showing asks offering down toward the spread, bids pushing up toward it, with the one-cent spread and 10.005 mid highlighted in amber between best bid and best ask](/imgs/blogs/order-book-simulator-quant-research-2.png)

The figure above shows the same book turned on its side, with price increasing upward. Read it as a tug-of-war: sellers keep posting offers *downward* toward the spread to attract buyers, buyers keep posting bids *upward* toward the spread to attract sellers, and the amber band in the middle — the spread — is the no-trade zone they're squeezing from both ends. The narrower that band, the cheaper it is to trade; a one-cent spread on a \$10 stock is tight and liquid, while a \$0.50 spread would mean the instrument is thin and expensive to get in and out of.

One more foundational term: **depth**. The depth at a price level is how many shares rest there; the depth of the book is the full ladder of sizes at each level. A book that is 500 shares at the best bid and 5,000 shares two cents down is **deep** at the back but **thin** at the top. Depth is what a large order eats through, and it is the raw material for every signal we compute later.

### A note on units and money

Throughout, prices are in dollars per share, quantities in shares, and a trade's notional value is `price × quantity` in dollars. A fill of 300 shares at \$10.01 is a notional of 300 × \$10.01 = \$3,003.00. Keep that arithmetic visible — half of order-book reasoning is just multiplying a price by a size and summing.

## Price-time priority: the matching rule

We have a book. Now: when a marketable order arrives, *which* resting order does it trade against first? The answer is a rule called **price-time priority**, and it is the law in the overwhelming majority of the world's exchanges (equities, futures, most crypto). Two tie-breakers, applied in order.

**Price first.** A better price always wins. For a buyer crossing the spread, "better" for the resting sellers means *lower* ask — the incoming buy fills against the cheapest seller available, then the next cheapest, and so on up the ladder. For a seller crossing, the incoming sell fills against the *highest* bid first. This is just rational: the engine gives the aggressor the best available price and gives the resting order at the best price the first chance to trade.

**Time second.** When several resting orders sit at the *same* price level, the one that arrived earliest fills first. This is **FIFO** — first in, first out. Your position in that line is your **queue position**, and it matters enormously: at a busy price level, being first in the queue versus tenth can be the difference between getting filled and watching the price move away while you wait.

![Price-time priority inside one level: a FIFO queue at 10.01 with Order A at the head, then B, then C, then the tail where new orders join, and an incoming sell of 250 filling Order A first](/imgs/blogs/order-book-simulator-quant-research-3.png)

The figure shows one price level, \$10.01, as a FIFO queue. Three buy orders rest there: Order A (300 shares, arrived 09:30:01), Order B (200 shares, 09:30:04), Order C (100 shares, 09:30:09). They are lined up in arrival order. When a sell order for 250 shares crosses down to \$10.01, it fills against the *head* of the queue — Order A — first, taking 250 of A's 300 shares. A now has 50 left and stays at the head; B and C haven't been touched. New buy orders at \$10.01 join the *tail*, behind C. The rule is mechanical and fair, and it is why "time" in "price-time priority" is not a formality: it literally orders who gets paid.

#### Worked example: who fills first at a shared price level

You and two other traders all rest buy limit orders at \$10.01. You arrive third, with 100 shares; ahead of you sit 300 shares and 200 shares, in that order. A 400-share market sell crosses down to your level. Walk it through:

- The 400-share sell hits the head first: it takes all 300 of the first order. 100 shares of the sell remain.
- It moves to the second order: takes all 200? No — only 100 shares of the sell are left. It takes 100 of that 200-share order, leaving 100 there.
- The sell is now exhausted. Your order, third in line, **never traded at all.** You're still resting, now at the head of a depleted queue with one 100-share order ahead of you.

The one-sentence intuition: at a shared price, your fill depends entirely on how much size sits *ahead* of you in the queue — price-time priority turns "when did you arrive" into "do you get paid."

There is one important variation worth naming, because interviewers do. A few markets — notably some interest-rate futures — use **pro-rata** matching instead of pure time priority: at a price level, an incoming order is split across resting orders *in proportion to their size*, not strictly by arrival. A 100-share taker against two resters of 300 and 100 shares would fill 75 against the big one and 25 against the small one under pro-rata, regardless of who came first. We build the FIFO version here because it's the default; if you're asked about pro-rata, the difference is exactly this allocation rule.

## Order types: limit, market, cancel, modify, and the conditional taker orders

A matching engine doesn't just see "buy" and "sell." It sees a stream of message types, and it must handle each one's semantics exactly. Here is the full taxonomy a real engine processes.

![Order-type taxonomy as a branching graph: an incoming message splits into new order, cancel, and modify; new order splits into limit, market, IOC, and FOK](/imgs/blogs/order-book-simulator-quant-research-5.png)

- **Limit order.** Buy/sell up to a quantity at a named price or better. If it can match immediately it does; whatever can't match **rests** in the book. This is the liquidity-providing default.
- **Market order.** Buy/sell a quantity immediately at whatever prices are available, sweeping levels until filled. Names no price. Provides no resting liquidity. Risk: in a thin book it can fill at a terrible average price (this is **slippage**, which we quantify below).
- **Cancel.** Remove a resting order (or some of its quantity) from the book. The trader changed their mind, or their model moved, or they're just churning quotes. Cancels are by far the most common message on real feeds — most posted orders are cancelled, not filled.
- **Modify (a.k.a. amend).** Change a resting order's price or quantity. Crucially, most exchanges treat a *price change* — or a quantity *increase* — as a **loss of time priority**: the modified order goes to the *back* of the new level's queue, as if freshly submitted. A quantity *decrease* usually keeps priority. This rule is why aggressive quoters often cancel-and-replace rather than modify, and why "does a modify keep my queue position" is a classic interview gotcha.

Two conditional **taker** order types matter enough to name:

- **Immediate-or-cancel (IOC).** Fill as much as you can *right now* against resting liquidity, then **cancel** any unfilled remainder — never rest. It's a market order with a price cap. Use it to take liquidity without leaving a resting footprint.
- **Fill-or-kill (FOK).** Fill the order *completely and immediately*, or do nothing at all — cancel the whole thing if it can't be filled in full in one shot. All-or-nothing. Use it when a partial fill is useless to you.

The mental model: every message either creates resting liquidity (a limit that doesn't fully cross), consumes it (market, IOC, FOK, or the crossing part of an aggressive limit), or edits the book (cancel, modify). Your engine is a giant dispatch on message type, and getting the IOC-vs-FOK and the modify-priority rules right is most of what separates a correct engine from a buggy one.

## The data structures behind a fast book

Now the engineering question that take-homes are really testing: how do you store the book so the hot operations are fast? The hot operations are (1) find the best bid and best ask, (2) match an incoming order against the best level, walking deeper if needed, and (3) add, cancel, or modify a resting order. You want all of these to be cheap even when the book holds millions of orders.

The canonical answer is two layers:

1. A **price-sorted map of levels**. The bid side is sorted so the *highest* price is first (best bid); the ask side so the *lowest* price is first (best ask). A balanced tree (`std::map` in C++, `SortedDict` / a tree in Python) gives you the best price in `O(1)` to peek and `O(log n)` to insert a new level. In ultra-low-latency engines this is often replaced by a flat array indexed by price ticks, but the *semantics* are identical.
2. At each price level, a **FIFO queue** of orders. A doubly linked list or a deque holds the orders in arrival order; the head is the next to fill, the tail is where new orders join. You also keep a hash map from `order_id` to the order's node, so a cancel is `O(1)` — you find the node by id and unlink it, no scanning.

![The book data structure: an asks sorted map and a bids sorted map, each pointing at price levels, each level holding a FIFO queue of orders with ids and sizes](/imgs/blogs/order-book-simulator-quant-research-6.png)

The figure shows the whole structure. The asks map (sorted low-price-first) points at level \$10.01, whose FIFO queue is `[A:300] → [B:150]`, then level \$10.02 with `[C:150]`. The bids map (sorted high-price-first) points at \$10.00 with `[D:500] → [E:200]`, then \$9.99 with `[F:250]`. To find the best ask you read the front of the asks map; to match against it you pop from the head of its queue; to cancel order E you jump straight to its node via the id map and unlink it. Every operation is either `O(1)` or `O(log n)`, and the FIFO queue gives you time priority for free because you always fill the head.

Here is that structure in Python. We will reuse it for every later example.

```python
from collections import deque
from dataclasses import dataclass, field
from sortedcontainers import SortedDict  # pip install sortedcontainers


@dataclass
class Order:
    order_id: int
    side: str          # "buy" or "sell"
    price: float
    qty: int           # remaining shares
    ts: int            # arrival sequence, for FIFO / debugging


class OrderBook:
    def __init__(self):
        self.bids = SortedDict()          # price -> deque[Order], high price = best
        self.asks = SortedDict()          # price -> deque[Order], low price = best
        self.orders = {}                  # order_id -> Order, for O(1) cancel
        self._seq = 0

    def best_bid(self):
        return self.bids.peekitem(-1)[0] if self.bids else None   # highest bid

    def best_ask(self):
        return self.asks.peekitem(0)[0] if self.asks else None    # lowest ask

    def _level(self, side, price):
        book = self.bids if side == "buy" else self.asks
        if price not in book:
            book[price] = deque()
        return book[price]
```

A few choices worth defending, because an interviewer will poke at them. We use one `SortedDict` per side rather than one for the whole book so "best bid" and "best ask" are each a single peek. We store remaining quantity *on the order object* so a partial fill is one subtraction. We keep the `orders` dict so cancel never scans a queue. And we keep an arrival sequence `ts` so we can reason about and test FIFO order explicitly. None of this is exotic — it's the standard shape, and being able to draw it and defend it is the bar.

## Processing an order: add, match, partial fill, cancel

Now the heart of the engine: the function that takes one incoming order and does the right thing. The control flow is a small decision tree.

![The add, cancel, and match control flow: an order arrives, the engine checks cancel-or-new, then whether it crosses the best opposite price, then matches the top of queue or rests at the back, looping on leftover quantity before publishing an update](/imgs/blogs/order-book-simulator-quant-research-7.png)

Read the flow left to right. An order arrives. If it's a **cancel**, we remove the resting quantity and we're done. If it's a **new** order, we ask: does it **cross** the best opposite price — is this incoming buy priced at or above the best ask, or this incoming sell at or below the best bid? If yes, we **match** against the top of the opposite queue, repeatedly, peeling off shares until either the incoming order is fully filled or it stops crossing (the next level is too expensive). If the incoming order is a limit and still has quantity left after it stops crossing, that remainder **rests** at the back of its own level's queue. A market or IOC remainder instead gets cancelled. Then we **publish** the resulting book update. Here is the matching loop in code.

```python
def add_limit(self, side, price, qty):
    self._seq += 1
    incoming = Order(self._seq, side, price, qty, self._seq)
    fills = []                              # list of (price, qty) we traded
    book = self.asks if side == "buy" else self.bids
    crosses = (lambda p: price >= p) if side == "buy" else (lambda p: price <= p)

    while incoming.qty > 0 and book:
        best_price = book.peekitem(0 if side == "buy" else -1)[0]
        if not crosses(best_price):
            break                           # no longer marketable; stop
        queue = book[best_price]
        resting = queue[0]                  # head of FIFO queue
        traded = min(incoming.qty, resting.qty)
        fills.append((best_price, traded))
        incoming.qty -= traded
        resting.qty  -= traded
        if resting.qty == 0:                # fully consumed: pop and forget
            queue.popleft()
            self.orders.pop(resting.order_id, None)
            if not queue:                   # level empty: drop the level
                del book[best_price]

    if incoming.qty > 0:                    # leftover rests as liquidity
        self._level(side, price).append(incoming)
        self.orders[incoming.order_id] = incoming
    return fills


def cancel(self, order_id):
    o = self.orders.pop(order_id, None)
    if o is None:
        return False                        # already filled or never existed
    book = self.bids if o.side == "buy" else self.asks
    queue = book.get(o.price)
    if queue:
        queue.remove(o)                     # O(level size); see note below
        if not queue:
            del book[o.price]
    return True
```

A market order is the same loop without the price cap — its `crosses` predicate is always true, so it walks levels until it runs out of quantity or the book runs dry. An IOC is the limit loop but skips the "leftover rests" step (it cancels the remainder instead). An FOK first *checks* whether the full quantity is available across the crossing levels and only then executes; otherwise it does nothing.

One honest caveat on the cancel: `queue.remove(o)` on a `deque` is `O(level size)`. For a production engine you'd use a doubly linked list of order nodes and store the node in `self.orders`, making the unlink `O(1)`. The Python above favors readability; in the interview, say out loud that you'd swap the deque for an intrusive linked list to get `O(1)` cancels, because cancels dominate the message stream.

#### Worked example: add-and-match by hand and in code

Start with an empty book. Submit, in order:

1. `add_limit("buy", 10.00, 500)` — nothing to cross (no asks), so 500 rests at the \$10.00 bid. Best bid \$10.00 / 500.
2. `add_limit("sell", 10.02, 300)` — \$10.02 doesn't cross the \$10.00 bid, so 300 rests at the \$10.02 ask. Best ask \$10.02 / 300. Spread is now \$0.02.
3. `add_limit("buy", 10.02, 100)` — this buy is priced at \$10.02, which *crosses* the \$10.02 ask. It matches the head of that ask queue, trading 100 shares at \$10.02 (notional 100 × \$10.02 = \$1,002.00). The incoming buy is fully filled; the \$10.02 ask shrinks from 300 to 200. No remainder rests.

After these three messages the book is: best bid \$10.00 / 500, best ask \$10.02 / 200, and one trade has printed (100 @ \$10.02). The intuition: a limit order is only a "passive" resting order when its price *doesn't* cross — price it through the spread and it behaves exactly like a marketable order until it can't cross anymore.

## Walking the levels: a market order and the dollars it pays

Here is the example every microstructure interview wants you to do in your head. A market order doesn't pay one price — it pays a **blend** of prices as it eats through depth. That blend is the **volume-weighted average price (VWAP)** of its fills, and the gap between that and the mid is **slippage**, the hidden cost of demanding immediacy.

Take this ask side of the book:

| Price | Resting size | Cumulative size |
|---|---|---|
| \$10.01 | 300 | 300 |
| \$10.02 | 150 | 450 |
| \$10.03 | 400 | 850 |

A trader sends a **market buy for 500 shares**. Price-time priority says: fill the cheapest sellers first, walking up the ladder.

![A 500-share market buy eats three ask levels: 300 filled at 10.01, 150 at 10.02, 50 of 400 at 10.03, total cost 5007 dollars and average 10.014 per share, with 4.50 dollars of slippage versus the 10.005 mid](/imgs/blogs/order-book-simulator-quant-research-4.png)

The figure traces it. The order takes all 300 shares at \$10.01 (the cheapest), all 150 at \$10.02, then needs 50 more and takes 50 of the 400 resting at \$10.03 — at which point it's filled and stops. The deepest level still has 350 shares resting, and because the \$10.01 and \$10.02 levels are now empty, the **new best ask is \$10.03**: the book has moved up.

#### Worked example: the total dollars and the VWAP fill

Compute the cost leg by leg:

- 300 shares × \$10.01 = \$3,003.00
- 150 shares × \$10.02 = \$1,503.00
- 50 shares × \$10.03 = \$501.50
- **Total = \$5,007.50** for 500 shares.

The volume-weighted average price is total dollars ÷ total shares = \$5,007.50 ÷ 500 = **\$10.015 per share.** That single number — the VWAP — is the price the trader *actually* paid, and it's the only honest way to score a multi-level fill; quoting just the best level (\$10.01) would understate the cost of every share above the first 300.

Now the slippage. The mid before the order was \$10.005. Had you magically filled all 500 shares at the mid, you'd have paid 500 × \$10.005 = \$5,002.50. You actually paid \$5,007.50. The difference — **\$5.00** — is your slippage, the price of eating through real depth instead of a frictionless mid. On a per-share basis that's \$5.00 ÷ 500 = \$0.01 per share, a full cent, which on a one-cent-spread stock is a meaningful tax.

![One marketable order walking three ask levels as a pipeline: buy 500 at market fills 300 at 10.01, 150 at 10.02, 50 at 10.03, reaches an average of 10.015 per share, and leaves the best ask at 10.03](/imgs/blogs/order-book-simulator-quant-research-8.png)

The pipeline above sequences the same fill as a flow rather than a stack of bars: the 500-share order enters, peels off 300 at \$10.01, carries 200 to the next level and takes 150 at \$10.02, carries 50 to the third level and takes 50 at \$10.03, lands on the \$10.015 VWAP, and leaves the book with its best ask repriced to \$10.03. Reading it left to right makes the *sequence* explicit — depth is consumed in strict price order, and each leg hands its leftover quantity to the next — which is exactly the loop the matching code runs.

The one-sentence intuition: a market order's true price is the VWAP of the depth it consumes, and the thinner the book, the further that VWAP drifts above the mid — slippage is depth-dependent, not a fixed fee.

Two follow-ups that interviewers chain onto this. First: *what if the book only had 450 shares of asks total?* Then a 500-share market buy fills 450 and the remaining 50 either rest (if the venue converts the unfilled market remainder to a limit, which most don't) or, far more commonly, get cancelled — and on some venues the order is rejected outright if it can't fill. Always ask "what does this venue do with the unfilled remainder of a market order," because the answer is not universal. Second: *how would a smart trader avoid that \$5.00?* By **not** sending one aggressive market order — by slicing it into pieces over time, or by posting passive limits and waiting, trading immediacy for a better price. That trade-off is the entire field of **execution**, and the book is where it plays out.

## The book evolving under order flow

A static snapshot is a teaching aid; a real book is a movie. Every microsecond, messages arrive — adds, cancels, trades — and the book updates. Understanding how the *top of book* (best bid, best ask, and their sizes) dances under this flow is the core skill, because that's the surface most signals and most execution logic actually watch.

Walk through one sequence on our book (best bid \$10.00 / 500, best ask \$10.01 / 300):

- A **cancel** removes 200 of the 500 at the best bid. Best bid is still \$10.00, but now only 300 deep. The book got thinner on the bid side without the price moving.
- A new **buy limit** for 400 shares arrives at \$10.00, joining the *back* of that level's queue. Best bid \$10.00 is now 700 deep. The bid side thickened.
- A **market sell** for 700 shares crosses down and eats the entire \$10.00 bid level (700 shares at \$10.00). The level empties; the **best bid drops** to the next level down, say \$9.99. The price moved *because depth was consumed*, not because anyone "set" a new price.

That last point is the deepest idea in microstructure and the one beginners miss: **prices move when liquidity is taken or pulled, not by decree.** The best bid falls to \$9.99 not because someone announced it but because the orders at \$10.00 are gone — either filled by an aggressor or cancelled by their owners. The "price of a stock" at any instant is just an emergent summary of which orders currently rest where. A matching engine never sets a price; it only enforces the rules by which orders meet, and the price is whatever the surviving orders imply.

This is also why the same trade can have very different *price impact* depending on the book it lands in. A 700-share market sell that hits a 700-deep bid level moves the price one tick; the same sell hitting a 7,000-deep level barely dents it. **Price impact is a function of order size relative to available depth** — a relationship execution algorithms model carefully and HFT market makers exploit constantly.

Put a number on it. Suppose the bid side reads \$10.00 / 700, \$9.99 / 5,000, \$9.98 / 6,000. A 700-share market sell empties exactly the top level and the best bid steps to \$9.99 — a one-cent move, and the seller's VWAP is a clean \$10.00. Now send a *7,000*-share sell into the same book: it takes 700 at \$10.00 (\$7,000.00), 5,000 at \$9.99 (\$49,950.00), and 1,300 at \$9.98 (\$12,974.00), for \$69,924.00 on 7,000 shares — a VWAP of \$9.989, two full ticks below where it started, and the best bid is now \$9.98. The same instrument, the same direction, ten times the size, and the price impact went from one tick to two while the slippage versus the \$10.005 mid ballooned from roughly \$3.50 to about \$112.00. That convexity — impact growing faster than linearly as you exhaust thin levels — is precisely why large orders are sliced over time rather than fired in one shot, and it's the quantitative core of every execution algorithm.

## Reconstructing the book from a message feed

So far we've *operated* the book. In practice, as a quant researcher or dev, you usually receive the book as a **stream of messages** from the exchange and have to **rebuild** it yourself to know its state at any moment. Exchanges publish a market-data feed: a sequence of messages — `ADD` (a new resting order appeared), `CANCEL` (a resting order was removed), `TRADE` / `EXECUTE` (a match happened, consuming resting quantity), and `MODIFY`. Each message carries a **sequence number** so you can detect gaps and replay in exact order.

Reconstruction — also called **book building** — is deterministic: start from an empty (or snapshot) book, apply messages in sequence-number order, and you recover the exact book state the exchange had after each message. This is the single most common data-engineering task in microstructure research, and it shows up constantly in take-homes.

![Reconstructing the book from a message feed: a timeline of add, add, add, trade, cancel messages by sequence number, ending in the rebuilt book state of 100 at 10.01 and 200 at 10.02](/imgs/blogs/order-book-simulator-quant-research-11.png)

The figure replays a tiny feed. Seq 1: `ADD bid 200 @ \$10.00`. Seq 2: `ADD ask 300 @ \$10.02`. Seq 3: `ADD bid 100 @ \$10.01`. Seq 4: `TRADE 100 @ \$10.02` — a buyer crossed and took 100 of the 300-share ask, so that ask drops to 200. Seq 5: `CANCEL 200 @ \$10.00` — the original bid is pulled. After replaying all five in order, the book is **bid 100 @ \$10.01, ask 200 @ \$10.02**. Get the order wrong, or drop a message, and your reconstructed book diverges from reality — which is why sequence numbers and gap detection are not optional.

```python
def apply_feed(book: "OrderBook", messages):
    """Replay an ordered message feed to reconstruct book state.
    messages: list of dicts with keys: seq, type, side, price, qty, order_id
    """
    expected = None
    for m in sorted(messages, key=lambda x: x["seq"]):
        if expected is not None and m["seq"] != expected:
            raise ValueError(f"gap: expected seq {expected}, got {m['seq']}")
        expected = m["seq"] + 1

        if m["type"] == "ADD":
            book.add_limit(m["side"], m["price"], m["qty"])
        elif m["type"] == "CANCEL":
            book.cancel(m["order_id"])
        elif m["type"] == "TRADE":          # remove executed qty from resting side
            book.reduce_at(m["side"], m["price"], m["qty"])
        # MODIFY = cancel + re-add (loses time priority on price change)
    return book
```

Two subtleties that separate a working reconstruction from a broken one. First, the **`TRADE` message tells you a match already happened** on the exchange — you don't re-run your matching engine on it, you just *remove* the executed quantity from the resting side, because the exchange already did the matching. (Some feeds are "order-by-order" and give you adds and cancels only, leaving you to infer trades by watching resting quantity disappear; others give explicit trades. Know which kind of feed you have.) Second, **gap handling**: if your sequence numbers skip, you've lost a message and your book is now wrong — the correct response is to stop, request a fresh snapshot, and resync, never to silently continue. A book that's quietly out of sync produces signals that are subtly, dangerously wrong.

#### Worked example: rebuild the top of book from five messages

Replay this feed by hand (each `ADD` is a fresh resting order; assume distinct order ids):

1. `ADD buy 200 @ \$10.00` → bid \$10.00 / 200. Best bid \$10.00.
2. `ADD sell 300 @ \$10.02` → ask \$10.02 / 300. Best ask \$10.02. Spread \$0.02.
3. `ADD buy 100 @ \$10.01` → bid \$10.01 / 100. **Best bid is now \$10.01** (higher than \$10.00). Spread tightens to \$0.01.
4. `TRADE 100 @ \$10.02` → remove 100 from the \$10.02 ask: it drops to 200. Best ask still \$10.02.
5. `CANCEL the \$10.00 bid (200)` → that level empties. Best bid falls back to \$10.01 / 100.

Final book: **best bid \$10.01 / 100, best ask \$10.02 / 200, mid \$10.015, spread \$0.01.** The intuition: reconstruction is just disciplined replay — apply each message's effect to the exact level it names, in sequence order, and the top of book falls out correctly at every step.

## Microstructure metrics: depth, order-flow imbalance, and the microprice

Once you can hold the book in memory, it becomes a feature factory. Two signals fall directly out of the top of book and show up in both research and interviews. Both are built on the same raw inputs: the best bid price and size, and the best ask price and size.

### Depth and order-flow imbalance

We've already met **depth** (size at a level). The simplest *predictive* feature built on it is **order-flow imbalance (OFI)** — sometimes called queue imbalance or book imbalance. The intuition first: if there are far more shares wanting to buy at the best bid than to sell at the best ask, buying pressure outweighs selling pressure, and the next price move is *more likely* to be up than down. OFI turns that intuition into a number between −1 and +1.

The top-of-book version is:

$$\text{OFI} = \frac{Q_{\text{bid}} - Q_{\text{ask}}}{Q_{\text{bid}} + Q_{\text{ask}}}$$

where $Q_{\text{bid}}$ is the size resting at the best bid and $Q_{\text{ask}}$ the size at the best ask. It's +1 if there's only bid size (all buy pressure), −1 if only ask size, 0 if perfectly balanced.

![Order-flow imbalance at the top of book: a tall green bid bar of 800 shares versus a short red ask bar of 200 shares, with the OFI formula computing plus 0.60 and a scale showing plus 1 all bid to minus 1 all ask](/imgs/blogs/order-book-simulator-quant-research-9.png)

#### Worked example: order-flow imbalance from the top of book

Best bid \$10.00 with 800 shares resting; best ask \$10.01 with 200 shares resting. Plug in:

$$\text{OFI} = \frac{800 - 200}{800 + 200} = \frac{600}{1000} = +0.60$$

A +0.60 reading is a strong **buy lean**: there's four times as much size waiting to buy at \$10.00 as to sell at \$10.01. Empirically, on short horizons (the next few trades or the next handful of seconds), a heavily positive imbalance is associated with an *upward* next move in the mid — the thin ask side gets consumed and the price ticks up more often than not. That's not a guarantee on any single instance; it's a statistical edge that market makers and short-horizon signals lean on. The one-sentence intuition: OFI reads the *relative weight of resting buyers versus sellers at the top of the book* as a short-horizon directional tilt.

A caveat the honest version of this answer includes: top-of-book OFI is noisy and easily gamed — large resting orders can be spoofed (posted to create a false imbalance, then cancelled before they fill, which is illegal in regulated markets but still happens). Robust implementations use **changes** in queue sizes (the order-flow *imbalance* of Cont, Kukanov, and Stoikov is defined on increments — additions and cancellations at the touch — not levels), weight multiple price levels, and validate against realized moves. But the top-of-book ratio above is the right place to start and the version you'll be asked to derive cold.

### The microprice: a smarter "fair value" than the mid

The mid, (bid + ask) / 2, has a blind spot: it ignores *sizes*. If there are 800 shares bid and only 200 offered, the next trade is more likely to lift the offer (consuming the thin ask) than to hit the bid — so the true "fair value" is closer to the ask than to the mid. The **microprice** captures this by weighting each side's price by the *opposite* side's size.

$$\text{microprice} = \frac{Q_{\text{ask}} \cdot P_{\text{bid}} + Q_{\text{bid}} \cdot P_{\text{ask}}}{Q_{\text{bid}} + Q_{\text{ask}}}$$

The cross-weighting is the whole trick and the part people get backwards, so internalize *why*: the bid price gets weighted by the **ask** size, and the ask price by the **bid** size. The reasoning: a *large* bid size means the bid is "strong" and unlikely to be the side that gives way — so the fair price should sit *away* from a heavy bid, i.e. toward the ask. Weighting the ask price by the (large) bid size pulls the microprice toward the ask exactly when the bid is heavy. The microprice leans toward the **thinner**, more vulnerable quote.

![The microprice leans toward the heavier side: a book with best bid 10.00 size 800 and best ask 10.01 size 200, midprice 10.005 ignoring sizes, microprice 10.008 leaning toward the thin ask](/imgs/blogs/order-book-simulator-quant-research-10.png)

#### Worked example: the microprice between bid and ask in dollars

Same book: best bid \$10.00 with 800 shares, best ask \$10.01 with 200 shares. Compute:

$$\text{microprice} = \frac{200 \times \$10.00 + 800 \times \$10.01}{800 + 200} = \frac{\$2000.00 + \$8008.00}{1000} = \frac{\$10008.00}{1000} = \$10.008$$

The plain mid is \$10.005. The microprice is **\$10.008** — it has moved 80% of the way from the mid toward the ask, because the bid is four times heavier than the ask and the thin ask is the likely side to give way. In code it's a one-liner:

```python
def microprice(p_bid, q_bid, p_ask, q_ask):
    return (q_ask * p_bid + q_bid * p_ask) / (q_bid + q_ask)

print(microprice(10.00, 800, 10.01, 200))   # -> 10.008
```

The one-sentence intuition: the microprice is a size-aware fair value that leans toward whichever quote is thinner, because that's the side the next trade is most likely to consume. It's a better short-horizon "true price" than the mid, and it's a building block in market-making and execution models — the connection to building tradeable features is the whole subject of [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) and how you'd then validate one in [backtesting done right](/blog/trading/quantitative-finance/backtesting-done-right-quant-research).

## In the interview room and the take-home

This is the section to actually rehearse. Quant trading, dev, and research interviews lean on the order book because it tests data structures, careful reasoning, and microstructure intuition all at once. Below are five fully worked problems in the exact shapes that come up — coding tasks, hand calculations, and "explain the rule" questions. Solve each one before reading the solution.

#### Worked example: design the matching engine (the classic take-home)

**Prompt.** "Design and implement a limit order book that supports `add_limit(side, price, qty)`, `market_order(side, qty)`, and `cancel(order_id)`. It must respect price-time priority and return the list of fills for each aggressive order. Discuss complexity."

**Solution.** This is the engine we built above. The design answer the grader wants: two `SortedDict`s (one per side) keyed by price, each value a FIFO queue (deque or intrusive linked list) of orders; a hash map `order_id → order` for `O(1)` cancel. `best_bid`/`best_ask` are `O(1)` peeks. Matching pops from the head of the best opposite level repeatedly (`O(1)` per fill), dropping emptied levels. Adding a resting order is `O(log n)` to find/create the level plus `O(1)` to append. Cancel is `O(1)` with a linked list (`O(level)` with a deque — call this out and offer the linked-list fix).

The bugs they're watching for: (1) forgetting to drop an emptied level (leaves a ghost best price); (2) not handling a limit that crosses *and then rests* the remainder; (3) losing FIFO order on a partial fill (the partially filled resting order must stay at the *head*, not go to the back); (4) a cancel that scans the whole book instead of jumping via the id map. Hit all four and the implementation question is essentially won. The intuition: a matching engine is a dispatch-on-message-type over two sorted-map-of-FIFO-queues structures, and the entire difficulty is in the bookkeeping of partial fills and emptied levels.

#### Worked example: walk a market order and price the fill

**Prompt.** "Asks are 200 @ \$50.00, 300 @ \$50.05, 500 @ \$50.10. A market buy for 600 shares arrives. What's the total cost, the average fill price, and the new best ask?"

**Solution.** Walk the ladder cheapest-first: 200 @ \$50.00 = \$10,000.00; then 300 @ \$50.05 = \$15,015.00 (now 500 filled, 100 to go); then 100 @ \$50.10 = \$5,010.00 (600 filled, stop). Total = \$10,000.00 + \$15,015.00 + \$5,010.00 = **\$30,025.00.** Average fill = \$30,025.00 ÷ 600 = **\$50.0417 per share.** The \$50.10 level had 500 and gave up 100, so 400 remain — the **new best ask is \$50.10** (the first two levels are empty). The intuition: a market order's average price is the VWAP of consumed depth, and it almost always prints worse than the inside quote because it eats deeper, pricier levels.

#### Worked example: does a modify keep your queue position?

**Prompt.** "You have a resting buy of 100 @ \$10.00, third in the queue. You (a) lower your size to 60, (b) raise your size to 150, (c) reprice to \$10.01. For each, what happens to your queue position?"

**Solution.** (a) **Size decrease keeps priority** — you stay third, now with 60 shares. (b) **Size increase loses priority** on most venues — the added quantity (and usually the whole order) goes to the *back* of the \$10.00 queue, because letting people jump the line by inflating size would be unfair. (c) **Reprice always loses priority** — your order is effectively cancelled at \$10.00 and re-submitted at \$10.01, landing at the *back* of the \$10.01 queue with a fresh timestamp. The intuition: exchanges protect time priority by penalizing any change that could let you cut the line — only a pure size *reduction* is "free." This is why aggressive quoters care so much about queue position and often prefer to rest early and hold, rather than chase the book with modifies.

#### Worked example: compute imbalance and microprice and predict the tick

**Prompt.** "Best bid \$99.98 size 1,200; best ask \$100.00 size 400. Compute the order-flow imbalance, the mid, and the microprice. Which way is the next tick more likely?"

**Solution.** OFI = (1200 − 400) / (1200 + 400) = 800 / 1600 = **+0.50** — a clear buy lean. Mid = (\$99.98 + \$100.00) / 2 = **\$99.99.** Microprice = (400 × \$99.98 + 1200 × \$100.00) / 1600 = (\$39,992.00 + \$120,000.00) / 1600 = \$159,992.00 / 1600 = **\$99.995.** The microprice sits *above* the mid, three-quarters of the way toward the ask, because the bid is three times heavier. With a +0.50 imbalance and the microprice leaning up, the **next move is more likely up** — the thin 400-share ask is the vulnerable side and tends to get lifted. State the caveat: it's a probabilistic tilt, not a certainty, and it's noisy on any single observation. The intuition: imbalance and microprice are two views of the same fact — a heavy bid against a thin ask tilts the short-horizon move upward.

#### Worked example: reconstruct the book and detect a gap

**Prompt.** "You receive: seq 1 `ADD bid 500 @ \$20.00`; seq 2 `ADD ask 500 @ \$20.02`; seq 4 `CANCEL the \$20.00 bid`. Rebuild the book and flag any problem."

**Solution.** Apply seq 1: bid \$20.00 / 500. Apply seq 2: ask \$20.02 / 500; spread \$0.02. Now seq jumps from 2 to **4 — you're missing seq 3.** The correct action is to **halt and resync**: do not apply seq 4 as if nothing happened, because the missing message might have changed the very level you're about to cancel, or added liquidity you now can't see. A reconstruction that silently swallows a gap produces a book that's quietly wrong, and every downstream signal inherits that error. If forced to proceed (some take-homes want the "best effort" book), you'd note the gap, apply seq 4 (the \$20.00 bid cancels, leaving only ask \$20.02 / 500), and *flag the result as unreliable*. The intuition: sequence numbers are the integrity check of a feed — a gap means "stop and resync," never "keep going and hope."

A grab-bag of shorter ones interviewers fire off, with one-line answers: *Why is the spread a cost?* — it's the round-trip toll; buy at the ask, sell at the bid, you lose the spread. *Why do most orders get cancelled, not filled?* — market makers continuously requote as their fair value moves, so the cancel rate dwarfs the fill rate. *What's the difference between a market order and a marketable limit?* — a market order has no price floor and can fill arbitrarily deep; a marketable limit crosses but stops at its limit price, protecting you from a thin book. *Why single-threaded engines?* — determinism: one thread processing one ordered message stream guarantees a reproducible, fair sequence with no race conditions over queue position.

## Common misconceptions

**"The price of a stock is a number the exchange sets."** No — there is no authoritative "price." There's a best bid, a best ask, and a stream of trade prints. The "price" you see on a quote screen is usually the last trade or the mid, both of which are *summaries* of where orders currently rest or last met. The engine never sets a price; it enforces matching rules and the price emerges from surviving orders.

**"A market order fills at the price I saw."** Only if the entire order fits at the best level. Beyond that it walks deeper, pricier levels, and your average fill is the VWAP of consumed depth — worse than the inside quote, by an amount (slippage) that grows with your size relative to the book's depth. The price you "saw" was the *best* level, not a guarantee for your whole order.

**"Bigger resting orders always fill first."** Under price-time priority, no — *earlier* orders fill first at a given price, regardless of size. A 100-share order that arrived at 09:30:01 fills before a 10,000-share order that arrived at 09:30:02 at the same price. Size only governs allocation under the minority *pro-rata* model. Confusing the two is a classic interview slip.

**"The mid is the fair price."** The mid ignores sizes. When the book is lopsided — heavy bid, thin ask — the size-aware microprice is a better short-horizon fair value, and it can sit well off the mid. Treating the mid as gospel throws away the information sitting in the depth.

**"Order-flow imbalance is a money printer."** It's a noisy, short-horizon, easily-gamed statistical tilt — not a guarantee. Resting size can be spoofed; the signal decays in seconds; transaction costs eat naive implementations. It's a real feature, but it's an *ingredient*, not a strategy.

**"Cancels are rare and suspicious."** On real feeds, cancels vastly outnumber fills — often 20-to-1 or more — because legitimate market makers continuously requote as their fair value moves. High cancel rates are the *normal* texture of a liquid market, not evidence of manipulation. (Spoofing is a specific, illegal pattern — posting with intent to cancel before filling — not "cancelling a lot.")

## How it shows up in real markets

**Exchange matching engines are single-threaded and deterministic.** The matching cores at Nasdaq, CME, NYSE, and the major crypto venues process their order stream on a single logical thread precisely so the sequence of events is reproducible and fair — every participant's order is handled in arrival order with no race conditions. This is why "latency to the engine" is everything: the engine is a serial bottleneck, and being a microsecond earlier into the queue can win you priority at a price level. Whole businesses exist to shave nanoseconds off the path to that one thread.

![Where the book lives in a real exchange: a co-located trader sends an order over fiber through an order gateway and pre-trade risk check into a single-threaded matching engine, which updates the limit order book, prints trades, and fans both out on a public market-data feed to all other traders](/imgs/blogs/order-book-simulator-quant-research-12.png)

**Co-location and the latency race.** Because queue position is decided by arrival time, firms pay exchanges to place their servers in the *same building* as the matching engine — co-location — and run dedicated fiber (or microwave links between cities) to shave microseconds. The figure traces the path: a co-located trader's order crosses fiber to an order gateway, passes a mandatory pre-trade risk check, reaches the single-threaded engine, updates the book, and the resulting trades and book deltas fan back out on the public market-data feed that *every other* trader is also racing to read. The entire HFT industry is, in large part, a competition to traverse that loop faster than anyone else. The 2010s microwave-tower arms race between Chicago and New York — building line-of-sight towers to beat fiber's speed by milliseconds — was about exactly this path.

**Order-book signals drive market making.** Automated market makers — the firms quoting both sides of thousands of instruments — continuously compute features like the microprice and order-flow imbalance off the live book to decide where to quote and how to skew. If the book leans heavily bid, a market maker shades its quotes up (microprice logic) to avoid being run over by the likely up-move, and manages inventory accordingly. The signals we derived by hand are, in production, computed millions of times a second.

**Reconstruction is the daily grind of microstructure research.** A quant researcher studying execution or signals starts by rebuilding the book from raw exchange feeds — Nasdaq ITCH, CME MDP, or a crypto venue's order-by-order feed — replaying billions of messages to recover the historical book state at every microsecond, then computing features on it. Getting reconstruction *exactly* right (gap handling, trade-vs-add semantics, the modify-priority rule) is a prerequisite to any honest analysis; a subtly wrong book produces subtly wrong research, which is worse than no research. This is the unglamorous foundation under [the data-quality biases](/blog/trading/quantitative-finance/market-data-eda-biases-quant-research) that trip up naive studies.

**The May 2010 Flash Crash and the role of depth.** On May 6, 2010, U.S. equity indices plunged roughly 9% and recovered within minutes. A large automated sell program hit a thin book; as liquidity providers pulled their resting bids (cancelling, as is their right), the bid side of many books emptied, and market sells walked down through near-empty levels — printing some trades at absurd prices (a few stocks traded at a penny). It's the textbook demonstration of the lesson from the "book evolving" section: **price impact is a function of size relative to depth, and when depth vanishes, even a moderate order can move the price catastrophically.** The book isn't just where prices are set; it's where liquidity crises *happen*.

**Pro-rata venues and queue strategy.** On the CME's short-term interest-rate futures (Eurodollar/SOFR), matching is partly pro-rata, not pure FIFO. That single rule change rewires trader behavior: under FIFO you race to be first in the queue; under pro-rata you post *large* size to win a bigger slice of each incoming order, because allocation is proportional to size. The matching rule is not a detail — it shapes the entire strategy around the book, which is exactly why interviewers ask whether you know the difference.

## When this matters and where to go next

If you ever click "buy at market" and pay more than the quote you saw, you've met the book directly — that surprise *is* slippage, the VWAP of the depth your order consumed. If you place a limit order and watch it sit unfilled while the price dances around it, that's queue position and price-time priority doing their job. And if you're preparing for a quant trading, dev, or research role, the order book is the single most reliable topic to over-prepare: it shows up as a coding take-home, a hand calculation, and a microstructure-intuition question, often in the same loop.

The natural next steps build on this foundation. To turn book features like the microprice and imbalance into something tradeable, see [building an alpha signal](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) and then [evaluating alpha signals with IC, Sharpe, and turnover](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research). To test any strategy honestly against historical books, [backtesting done right](/blog/trading/quantitative-finance/backtesting-done-right-quant-research) covers the pitfalls. And because the matching engine is fundamentally a low-latency data-structures problem, [C++ for low-latency](/blog/trading/quantitative-finance/cpp-for-low-latency-quant-interviews) and the [data structures and algorithms](/blog/trading/quantitative-finance/coding-interview-quant-data-structures-algorithms) reference are where the engine we sketched in Python becomes the nanosecond-grade machine real exchanges run.

The best way to learn this remains the one we started with: open an editor, type out the `OrderBook` class above, feed it a sequence of orders, and watch the ladder change. Once you can predict every fill and every price move before your code prints it, you understand the core machine of modern markets — and you're ready for any version of the question an interviewer can throw at it.

*This article is educational, not trading advice. Order-book mechanics described here are general; specific venues differ in their exact rules (matching algorithm, order types, modify-priority semantics), so always read the rulebook of the venue you actually trade.*

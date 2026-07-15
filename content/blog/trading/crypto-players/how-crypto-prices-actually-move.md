---
title: "How Crypto Prices Actually Move: Order Books, Thin Float, and Slippage"
date: "2026-07-15"
publishDate: "2026-07-15"
description: "A build-from-zero guide to the order book, resting liquidity, slippage, and thin float — the exact mechanics by which a small order can move a large headline number, with worked dollar examples and a real flash-crash case study."
tags: ["crypto", "order-book", "liquidity", "slippage", "market-microstructure", "market-makers", "thin-float", "reflexivity", "perpetual-swaps", "crypto-players"]
category: "trading"
subcategory: "Crypto Players"
author: "Hiep Tran"
featured: true
readTime: 39
---

> [!important]
> **TL;DR** — The "price" of a token is nothing more than the last trade: one marginal buyer meeting one marginal seller at the top of the order book. Multiply that single print across the whole supply and you get the headline "market cap." When the book underneath is thin, a small order moves the print a lot — and the headline moves by hundreds of times the cash that actually changed hands.
>
> - An **order book** is a public ladder of resting buy orders (**bids**) and sell orders (**asks**); the gap between the best of each is the **spread**, and the size stacked behind each price is the **depth**.
> - A **market order** takes whatever price the book offers, eating levels cheapest-first. The gap between the price you saw and the average you paid is **slippage**, and it grows *faster* than your order size.
> - On a **thin book**, a \$10,000 buy can move the price 35% and re-price a 10-million-token float by \$7,000,000 on paper — while only \$10,000 of real cash traded. The same \$10,000 barely moves a **deep book**.
> - Market makers supply most of the depth you see, and they can cancel it in milliseconds — right before the news that would have needed it.
> - The number to remember: on 21 June 2017 a single multi-million-dollar market sell drove ETH from \$317.81 down 29.4% in one sweep, then a cascade of ~800 liquidations took it to \$0.10 — because the resting liquidity underneath was thin (source: GDAX / Adam White).

You have probably seen a headline like this: *"Token X adds \$400 million in market cap overnight."* It sounds like \$400 million of real money poured in. It almost never does. Very often the *entire* move is bootstrapped by a few tens of thousands of dollars of actual buying, landing on a book so thin that each order jumps the price to the next rung — and every token in existence gets marked up to that new rung on paper.

This is the single most important mechanical fact in all of crypto, and once you see it you cannot unsee it. It is why a "small" player — a market maker, a well-timed whale, a coordinated group of buyers — can move a "large" number. It is why market cap is a fiction that behaves like a fact. And it is the machinery behind almost every pump, every flash crash, and every "it's mooning" narrative you will ever read.

The diagram below is the mental model for the whole post: your order goes into the book, the last fill becomes "the price," and that one price is multiplied across the supply to manufacture the headline. Everything else here is a tour of that picture.

![How a small market order becomes a large headline market cap: the last trade sets the price, which is multiplied across the whole supply.](/imgs/blogs/how-crypto-prices-actually-move-1.webp)

We will build every term from zero — order book, bid, ask, spread, depth, market and limit orders, slippage, float, market cap, and a first look at perpetual swaps — and ground each one in a worked example with round dollar numbers. This is educational, not financial advice; the goal is to let you *see the plumbing* so you know who is on the other side of your trade. This post is the mechanics companion to the series overview, [Crypto VC and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers), and it leans on the same microstructure that traditional exchanges run — if you want the TradFi version, see [Inside an Exchange: the Matching Engine and the Order Book](/blog/trading/capital-markets/inside-an-exchange-the-matching-engine-and-the-order-book).

## Foundations: the order book from zero

Forget crypto for a second. Picture the simplest possible market: a room full of people who want to buy a thing and people who want to sell it, each holding up a sign with a price and a quantity.

An **order book** is exactly that room, written down and sorted. It has two sides:

- **Bids** — resting *buy* orders. Each bid says "I will buy up to N tokens at price P (or lower)." Buyers want to pay *less*, so bids are stacked from the highest price down.
- **Asks** (also called **offers**) — resting *sell* orders. Each ask says "I will sell up to N tokens at price P (or higher)." Sellers want to receive *more*, so asks are stacked from the lowest price up.

The **best bid** is the highest price any buyer is currently willing to pay. The **best ask** is the lowest price any seller is currently willing to accept. Naturally the best bid is *below* the best ask — if a buyer were willing to pay what a seller wants, they would already have traded. That gap between them is the **spread**:

`spread = best ask − best bid`

The **mid price** is just the average of the two, `(best ask + best bid) / 2` — a convenient single number for "where the market is," even though you can't actually trade at the mid.

The **depth** at a price level is how many tokens are resting there. A book with a lot of size at every price near the top is *deep*; a book with only a few tokens at each price is *thin*. Depth is the whole game, and almost nobody checks it before they trade.

The figure below is the anatomy of a small book for an imaginary token we'll call \$MOON. Read it top to bottom: sellers (asks, in red) stacked above, buyers (bids, in green) stacked below, and the spread (in amber) sitting in the gap between the best ask and the best bid.

![Anatomy of an order book: asks stacked above, bids below, the spread in the middle, and the size at each price is the depth.](/imgs/blogs/how-crypto-prices-actually-move-2.webp)

#### Worked example: reading the best bid, best ask, and spread

Look at the \$MOON book in the figure. The best ask is **\$2.01** (the cheapest place you can buy right now), and the best bid is **\$1.99** (the highest place you can sell right now). So:

- Spread = \$2.01 − \$1.99 = **\$0.02**.
- As a percentage of the mid price (\$2.00), that's 0.02 / 2.00 = **1.0%**.
- If you *bought* at the best ask (\$2.01) and immediately *sold* at the best bid (\$1.99), you would lose \$0.02 per token — the spread is a round-trip cost, paid to whoever is quoting both sides.

The intuition: **the spread is the toll you pay for immediacy.** Want to trade *right now*, no waiting? You cross the spread and pay it. That toll usually lands in the pocket of a market maker, the professional who posts both the bid and the ask — a topic we cover in depth in [Market Makers and the Spread](/blog/trading/capital-markets/market-makers-and-the-spread-who-provides-liquidity).

### Two ways to place an order: market vs limit

There are two fundamental order types, and the difference between them is the difference between *price certainty* and *execution certainty*. You can have one, not both.

- A **limit order** names a price and waits. "Buy 1,000 \$MOON at \$1.98 or better." It joins the book as *resting liquidity* — it becomes one of the bids or asks other people trade against. You control the price, but you are not guaranteed to trade at all; if the market never comes to your price, you sit there forever. Limit orders are how you *provide* liquidity.

- A **market order** names a quantity and takes whatever price the book offers *right now*. "Buy \$10,000 of \$MOON, immediately." You are guaranteed to trade, but you do *not* control the price — you get filled against the resting orders on the other side, starting from the best price and walking up (for a buy) or down (for a sell) until your order is filled. Market orders are how you *consume* liquidity.

That "walking up the book" is where the whole story lives. A market buy doesn't get one price — it gets a *sequence* of prices, one per level it eats, and you pay the blended average. If you want the deeper taxonomy of order types and how an order physically reaches a venue, see [Order Types and How an Order Travels to the Market](/blog/trading/capital-markets/order-types-and-how-an-order-travels-to-the-market).

### How the match actually happens: price-time priority

When your market buy walks the book, the exchange isn't choosing which resting orders to fill arbitrarily — it follows a strict rule called **price-time priority**. Orders are matched best-price-first, and among orders sitting at the *same* price, the one that arrived *earlier* is filled first. The resting liquidity at each level is really a queue: to get filled sooner, you either offer a better price or you got there earlier.

This matters twice over for our story. First, it's precisely why a market order fills at a *worsening sequence* of prices — it drains the best-priced queue, then the next, then the next, paying more at each step. Second, it's why *posting* a limit order puts you at the back of a queue you don't control: your resting buy at \$1.98 fills only if every earlier order at \$1.98 fills first *and* the price actually falls to you. An exchange, then, is not a vault of tokens — it is a matching engine running this queue millions of times a second, the machinery we pull apart in [Inside an Exchange](/blog/trading/capital-markets/inside-an-exchange-the-matching-engine-and-the-order-book).

### The units that make headlines: float, market cap, and FDV

Three more terms and you have the full vocabulary.

- **Circulating supply** (or **float**) — the number of tokens actually tradeable right now: not locked, not vested, not sitting in a team wallet with a two-year cliff. This is the supply that meets the order book.
- **Market cap** — the headline number, defined as `last price × circulating supply`. Note what it multiplies: the *last trade* times *every circulating token*, even though only a handful of those tokens actually traded at that price.
- **Fully diluted valuation (FDV)** — `last price × maximum supply`, i.e. what the market cap *would* be if every token that will ever exist were already circulating. For a token with 10% of supply unlocked, FDV is 10× the market cap. We unpack why that gap matters so much in the companion post on [why a token is not a stock](/blog/trading/crypto/crypto-vc-and-market-makers).

Hold onto one idea from this section: **market cap is not money that exists.** It is an *extrapolation* — the last price, projected across a supply that could never all sell at that price. The rest of this post is really about how violently that extrapolation swings when the book is thin.

## 1. Market orders eat the book: slippage and market impact

Here is the mechanism at the heart of everything. When you send a market buy, the exchange fills it against the cheapest ask first, then the next-cheapest, and so on, until your order is done. Two things happen as it walks:

1. **Market impact** — each level you consume *removes* that resting liquidity, so the next fill is at a higher price. By the time your order finishes, the best ask has moved up. You *pushed the price* just by trading.
2. **Slippage** — because you paid a rising sequence of prices, your *average* fill price is worse than the price you saw when you clicked. Slippage is that difference: `slippage = (average price you paid − price you expected) / price you expected`.

Put concretely: the price on your screen is the price of the *first* token, but you're buying *thousands* of them, each a little more expensive than the last.

Watch it happen. In the animation below, a single \$10,000 market buy sweeps up a thin ask ladder, lighting each level as it's consumed and dragging the "last price" pointer up the rungs from \$2.00 to \$2.70.

<figure class="blog-anim">
<svg viewBox="0 0 720 440" role="img" aria-label="A market buy sweeps up five ask levels from $2.00 to $2.70, lighting each rung as it is consumed while the last-price pointer climbs" style="width:100%;height:auto;max-width:760px">
<title>A $10,000 market buy sweeps up a thin ask ladder</title>
<style>
.ob-rung{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}
.ob-eat{fill:#e8590c;opacity:0}
.ob-lbl{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.ob-sz{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.ob-axis{font:600 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.ob-ptr{fill:var(--accent,#6366f1)}
.ob-ptrtxt{font:700 14px ui-sans-serif,system-ui;fill:var(--accent,#6366f1)}
@keyframes ob-e1{0%,4%{opacity:0}10%,100%{opacity:.85}}
@keyframes ob-e2{0%,20%{opacity:0}26%,100%{opacity:.85}}
@keyframes ob-e3{0%,38%{opacity:0}44%,100%{opacity:.85}}
@keyframes ob-e4{0%,56%{opacity:0}62%,100%{opacity:.85}}
@keyframes ob-e5{0%,74%{opacity:0}80%,100%{opacity:.55}}
@keyframes ob-climb{0%,4%{transform:translateY(0)}10%,20%{transform:translateY(-64px)}26%,38%{transform:translateY(-128px)}44%,56%{transform:translateY(-192px)}62%,100%{transform:translateY(-256px)}}
.ob-r1{animation:ob-e1 11s ease-in-out infinite alternate}
.ob-r2{animation:ob-e2 11s ease-in-out infinite alternate}
.ob-r3{animation:ob-e3 11s ease-in-out infinite alternate}
.ob-r4{animation:ob-e4 11s ease-in-out infinite alternate}
.ob-r5{animation:ob-e5 11s ease-in-out infinite alternate}
.ob-pointer{animation:ob-climb 11s ease-in-out infinite alternate}
@media (prefers-reduced-motion:reduce){.ob-r1,.ob-r2,.ob-r3,.ob-r4{animation:none;opacity:.85}.ob-r5{animation:none;opacity:.55}.ob-pointer{animation:none;transform:translateY(-256px)}}
</style>
<text class="ob-axis" x="60" y="28">thin ask ladder (cheapest at the bottom)</text>
<text class="ob-axis" x="500" y="28">your $10,000 buy &#8594;</text>
<rect class="ob-rung" x="60" y="336" width="360" height="40" rx="6"/>
<rect class="ob-eat ob-r1" x="60" y="336" width="360" height="40" rx="6"/>
<text class="ob-lbl" x="76" y="361">$2.00</text><text class="ob-sz" x="300" y="361">1,000 tokens</text>
<rect class="ob-rung" x="60" y="272" width="360" height="40" rx="6"/>
<rect class="ob-eat ob-r2" x="60" y="272" width="360" height="40" rx="6"/>
<text class="ob-lbl" x="76" y="297">$2.10</text><text class="ob-sz" x="300" y="297">1,000 tokens</text>
<rect class="ob-rung" x="60" y="208" width="360" height="40" rx="6"/>
<rect class="ob-eat ob-r3" x="60" y="208" width="360" height="40" rx="6"/>
<text class="ob-lbl" x="76" y="233">$2.25</text><text class="ob-sz" x="300" y="233">1,000 tokens</text>
<rect class="ob-rung" x="60" y="144" width="360" height="40" rx="6"/>
<rect class="ob-eat ob-r4" x="60" y="144" width="360" height="40" rx="6"/>
<text class="ob-lbl" x="76" y="169">$2.45</text><text class="ob-sz" x="300" y="169">1,000 tokens</text>
<rect class="ob-rung" x="60" y="80" width="360" height="40" rx="6"/>
<rect class="ob-eat ob-r5" x="60" y="80" width="360" height="40" rx="6"/>
<text class="ob-lbl" x="76" y="105">$2.70</text><text class="ob-sz" x="300" y="105">only $1,200 filled</text>
<g class="ob-pointer" transform="translate(440,356)">
<path class="ob-ptr" d="M0 0 L22 -10 L22 10 Z"/>
<text class="ob-ptrtxt" x="30" y="5">last price</text>
</g>
</svg>
<figcaption>A $10,000 market buy walks up the thin book cheapest-first. It clears four full levels ($8,800) and part of the fifth, and the last fill at $2.70 becomes the new quoted price — a 35% jump on $10,000 of buying.</figcaption>
</figure>

#### Worked example: a \$10,000 market buy through a thin book

Suppose \$MOON has a thin ask side. Best ask \$2.00, and only 1,000 tokens resting at each of the next few price levels:

| Ask level | Price | Tokens resting | Cost to clear it | Cumulative spent |
|---|---|---|---|---|
| 1 | \$2.00 | 1,000 | \$2,000 | \$2,000 |
| 2 | \$2.10 | 1,000 | \$2,100 | \$4,100 |
| 3 | \$2.25 | 1,000 | \$2,250 | \$6,350 |
| 4 | \$2.45 | 1,000 | \$2,450 | \$8,800 |
| 5 | \$2.70 | 1,000 | \$2,700 | \$11,500 |

You send a **\$10,000 market buy**. Walk it level by level:

1. Levels 1–4 clear completely: that's 4,000 tokens for **\$8,800**.
2. You have \$10,000 − \$8,800 = **\$1,200** left, which buys into level 5 at \$2.70: 1,200 / 2.70 ≈ **444 tokens**.
3. Total tokens received: 4,000 + 444 = **4,444 tokens**.
4. Average price paid: \$10,000 / 4,444 ≈ **\$2.25**.

Now tally the damage:

- The price you *saw* when you clicked was \$2.00 (the best ask). The price you *paid on average* was \$2.25. Slippage = (2.25 − 2.00) / 2.00 = **12.5%**.
- The **last fill** printed at \$2.70. That is now the quoted price of \$MOON. The market just moved from \$2.00 to \$2.70 — **up 35%** — on a single \$10,000 order.

The figure below is this exact example: five bars, one per price level, each labeled with the dollars needed to clear it. The order eats the four cheapest (red) and part of the fifth (amber), and the last fill at \$2.70 becomes the new price.

![A $10,000 market buy clears the four cheapest ask levels ($8,800) and part of the fifth; the last fill at $2.70 is the new price.](/imgs/blogs/how-crypto-prices-actually-move-3.webp)

The one-sentence intuition: **on a thin book, your own order is the news.** Nothing happened in the world — no announcement, no flow of real capital worth mentioning — yet the price rose 35% purely because there was almost nothing resting above \$2.00 to absorb the order.

## 2. The same order in a deep book barely moves

Now change *one* variable — depth — and hold everything else fixed. Same token, same \$10,000 buy, same \$2.00 starting price. But this time the book is **deep**: real size resting at every tick.

#### Worked example: the identical \$10,000 buy in a deep book

Say the deep book has **4,000 tokens resting at \$2.00**, then **20,000 tokens at \$2.001**, and so on — tight ticks, heavy size. Walk the same \$10,000 order:

1. First it takes 4,000 tokens at \$2.00 = **\$8,000**.
2. It has \$2,000 left, which buys 2,000 / 2.001 ≈ **999 tokens** at \$2.001.
3. Total: ~4,999 tokens for \$10,000 → average price ≈ **\$2.0004**.
4. The last fill printed at \$2.001.

Compare the two worlds side by side:

- **Thin book:** average \$2.25 (**12.5%** slippage), last print \$2.70 (**+35%**).
- **Deep book:** average \$2.0004 (**~0.02%** slippage), last print \$2.001 (**+0.05%**).

Same dollars, same intention. The *only* thing that differed was how much liquidity was resting to absorb the order — and it changed the price move by nearly a thousand-fold. The figure makes the contrast literal.

![The identical $10,000 order: it detonates a thin book (+35%) but barely dents a deep book (+0.05%).](/imgs/blogs/how-crypto-prices-actually-move-4.webp)

> Depth, not size of order, decides how much you move the price. A tiny order on a thin book moves more than a huge order on a deep one.

This is why the *same* dollar amount is a rounding error on Bitcoin and a market-moving event on a freshly-listed micro-cap: Bitcoin's book is thousands of times deeper. The lesson for a real trader is uncomfortable but simple — **before you size an order, look at the depth, not just the price.** A price with nothing behind it is a trap.

### Why "how much can I move it" cuts both ways

Depth is symmetric. If you can push the price *up* 35% by buying into a thin book, then someone else can push it *down* 35% by selling into a thin book — and *you* are then the one holding tokens marked to a price that evaporates the moment you try to exit. Thin liquidity is not a friend that only shows up when you're buying; it is a property of the market that hurts whoever needs to trade size against it. When we get to the real flash crashes later, this symmetry is the entire story.

## 2b. Where the depth actually comes from

One more foundational point before the math gets sharper: **who puts all those resting orders there?** On most crypto books, the overwhelming majority of the depth near the top is posted by **market makers** — firms (or bots) whose business is to quote both a bid and an ask continuously and earn the spread as buyers and sellers cross it. A token with an active market maker has a tight spread and real depth; a token without one has a wide spread and a book you can punch a hole through with pocket change. This is the operating model we profile across the [Crypto VC and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers) series, and the reason a project will *pay* a market maker to show up on day one.

We will see in Section 4 exactly how much that posted depth is worth — and how fast it can disappear.

## 3. The slippage curve: why size is nonlinear

Here is the property that traps almost everyone: **slippage is convex in order size.** Doubling your order does not double your slippage — it more than doubles it, because each extra dollar has to reach deeper into the book, where prices are worse. The cost accelerates.

#### Worked example: slippage at \$1k, \$10k, and \$50k on the same book

Take the thin \$MOON book from Section 1 and run three order sizes through it:

- **\$1,000 buy.** This fits *inside* the first level (\$2,000 of tokens rest at \$2.00). You get 500 tokens all at \$2.00. Average = \$2.00, slippage = **~0%**, price unchanged. A small order on a thin book can still be fine — if it's small *enough*.
- **\$10,000 buy.** As computed above: average \$2.25, **12.5%** slippage, last print \$2.70 (+35%).
- **\$50,000 buy.** This runs clean off the visible book. Once the cheap levels are gone, the order keeps climbing into ever-thinner, ever-higher asks; by the time it's filled it has paid an average well north of \$4 and dragged the last print to three or four times the starting price. On this kind of book a \$50,000 order can eat **over 100%** slippage.

Notice the shape: from \$1k to \$10k (10× the size), slippage went from ~0% to 12.5%. From \$10k to \$50k (5× the size), it went from 12.5% to over 100% — an eight-fold jump for a five-fold increase in size. That acceleration *is* convexity. The figure plots it.

![Slippage is convex in order size: a $1k order pays ~0%, a $10k order 12.5%, and a $50k order over 100% on the same thin book.](/imgs/blogs/how-crypto-prices-actually-move-5.webp)

The practical takeaway is the reason professionals almost never send large market orders: **they break big orders into small pieces over time** (a practice called *order slicing* or using a *TWAP/VWAP* execution algorithm — spreading the order across time so the book can refill between clips). A \$50,000 order sent all at once pays the convex penalty; the same \$50,000 fed in as fifty \$1,000 clips over a few hours, letting fresh liquidity replenish between them, can pay a small fraction of that. If you ever see a token gap 40% on a single candle and snap most of the way back, you are very often looking at one impatient market order that should have been sliced — and the liquidity refilling behind it.

## 4. Market makers add and pull depth — and that changes everything

We said the depth you see is mostly posted by market makers. Now the crucial part: **that depth is optional.** A resting limit order is a standing offer, not a commitment. The market maker can cancel every one of its orders in a millisecond, and it will, whenever the risk of getting run over rises — for example, right before a scheduled news event, a token unlock, or a listing announcement that could move the price against it.

#### Worked example: the same order with the market maker in, then out

Start with a market maker actively quoting \$MOON. It has posted **20,000 tokens across \$2.00–\$2.02** — a deep, tight book.

- Your **\$10,000 buy** fills almost entirely against orders near \$2.00. Average ≈ \$2.005, slippage ≈ **0.25%**, and the price ticks up maybe **0.5%**. Smooth.

Now imagine it's thirty seconds before a big exchange listing announcement. The market maker, not wanting to be the sucker holding the bag if the price gaps, **cancels its resting orders.** The book instantly reverts to the thin state from Section 1.

- The *identical* \$10,000 buy now sweeps four price levels. Average \$2.25, slippage **12.5%**, price **+35%**.

Same order. Same token. The only thing that changed is whether the market maker's depth was there or not — and it swung the outcome from a 0.5% move to a 35% gap. The figure shows both states.

![A market maker's posted depth is the difference between a 0.5% move and a 35% gap — and it can be cancelled in milliseconds.](/imgs/blogs/how-crypto-prices-actually-move-6.webp)

The one-sentence intuition: **the liquidity you see is a promise that can be un-promised.** This is not a bug or a scandal by itself — a market maker managing its risk by pulling quotes ahead of a volatile event is doing exactly what a rational risk manager should. But it means the "depth" on your screen is *conditional*, and it is thinnest at precisely the moments you are most likely to want to trade (big news, fast moves). When we discuss defense at the end, "is this depth real or is it a market maker who will vanish under stress?" is the central question.

There is also a darker version of this, where the depth was never real to begin with — orders posted only to *look* like liquidity, then cancelled before they can be hit (*spoofing*), or trades bounced between related accounts to fake volume (*wash trading*). Those are contested, sometimes illegal practices, and we treat them carefully in a dedicated post later in the series; here the point is narrower and uncontroversial: **even entirely legitimate depth is temporary.**

## 5. Thin float and market cap: the leverage of the last trade

Now we can answer the question from the very top: how does a small order move a *large* headline number?

Recall that market cap = last price × circulating supply, and that the last price is set by the *last trade*. Put those together and something remarkable falls out: **when your \$10,000 order moves the last price, every single circulating token gets re-priced to the new level — on paper.**

#### Worked example: \$10,000 of buying, \$7,000,000 of "market cap"

Give \$MOON some realistic tokenomics for a new listing:

- **Circulating float:** 10,000,000 tokens (10 million actually tradeable).
- **Maximum supply:** 100,000,000 tokens (the other 90% locked with the team, investors, and treasury — a classic *low-float* launch).
- **Starting price:** \$2.00.

At the start:

- Market cap = 10,000,000 × \$2.00 = **\$20,000,000**.
- FDV = 100,000,000 × \$2.00 = **\$200,000,000**.

Now your thin-book \$10,000 buy from Section 1 drags the last price to **\$2.70**. Recompute:

- Market cap = 10,000,000 × \$2.70 = **\$27,000,000** — a **\$7,000,000** increase.
- FDV = 100,000,000 × \$2.70 = **\$270,000,000** — a **\$70,000,000** increase.
- Actual cash that changed hands: **\$10,000.**

Sit with those numbers. Ten thousand real dollars added seven million dollars to the market cap and seventy million to the FDV. That is roughly **700×** paper leverage on the headline market cap, and **7,000×** on FDV, from a single order — because the last trade re-priced 10 million tokens (or 100 million, for FDV) that never actually traded. The table figure lays out the before and after.

![$10,000 of real buying adds $7M to the headline market cap and $70M to FDV, because the last trade re-prices every circulating token.](/imgs/blogs/how-crypto-prices-actually-move-7.webp)

This is the mechanical engine behind every "Token X adds \$400M overnight" headline. The \$400M is real *arithmetic* — last price times supply genuinely went up by that much — but it is not \$400M of anything you could withdraw. If holders tried to actually *sell* their 10 million tokens into that thin book, the price would collapse back through the same levels it climbed, and the "market cap" would evaporate on the way down. The number you can actually realize is far smaller than the headline; there is even a metric built to capture this gap, called **realized value** (or *realized cap*), which sums what every coin *last actually traded for* rather than marking the whole supply to the latest print.

#### Worked example: what your exit is actually worth

The headline says \$MOON is a \$27,000,000 token. Suppose you own **1,000,000 tokens** — 10% of the float — marked at \$2.70, so your position "shows" **\$2,700,000**. What happens if you try to sell it into the same thin book?

Your sell order walks *down* the bid side the way the buy walked *up* the asks. Say the bids mirror the thin asks: 1,000 tokens resting at \$2.00, another 1,000 at \$1.90, 1,000 at \$1.75, and thinning from there. Selling 1,000,000 tokens into that would blow straight through every visible bid and keep falling — you would realize an average of well under \$1.00 per token, not \$2.70. Your \$2,700,000 "position" converts to perhaps **\$700,000–\$900,000** of actual cash, and on the way down you have printed a 60%-plus crash that marks *everyone else's* holdings lower too.

The lesson to carry: **the marked value of a position and the cash you can extract from it are different numbers, and the gap widens with the size you hold and the thinness of the book.** This is exactly why large holders route size through over-the-counter (OTC) desks — negotiating a block trade off the public book — instead of dumping into the order book, a mechanism we cover later in the series.

The one-sentence intuition: **market cap is the last trade wearing a costume.** A thin float turns that costume into a megaphone — the smaller the tradeable supply relative to the total, the more violently a small order swings the headline. That is not an accident of a few bad tokens; it is the deliberate design of the modern low-float, high-FDV launch, which we dissect in [why a token is not a stock](/blog/trading/crypto/crypto-vc-and-market-makers).

## 6. Reflexivity: the price move that writes its own story

So far the mechanics are cold and physical: orders, levels, arithmetic. But markets are made of people, and people react to prices. This is where a small mechanical move becomes a large *sustained* one.

The financier George Soros gave the loop its name — **reflexivity** — in *The Alchemy of Finance* (1987). The plain-English version: **the price doesn't just reflect reality; it changes the reality that then feeds back into the price.** In crypto this loop is unusually tight and fast:

1. A thin-book order ticks the price up (mechanical, as in Section 1).
2. That green candle becomes a *story*: "\$MOON is breaking out," screenshotted and posted.
3. The story pulls in FOMO buyers — people who buy *because it went up*, not because anything fundamental changed.
4. Their market orders hit the same thin book, moving the price up again.
5. Which makes a louder story, which pulls in more buyers... and around it goes — until one large sell hits the same thin depth and the whole thing runs *in reverse* just as fast.

![Reflexivity: a mechanical price move manufactures a narrative that pulls in buyers whose orders move the price again — and it unwinds just as fast.](/imgs/blogs/how-crypto-prices-actually-move-8.webp)

Reflexivity is why the thin-book mechanic matters *beyond* the first order. The first \$10,000 doesn't have to do all the work; it just has to start the candle that recruits the next hundred buyers. This is also why narratives in crypto feel self-fulfilling for a while and then aren't: the loop runs on new buyers, and when they run out, the same thin liquidity that magnified the rise magnifies the fall. If you want the behavioural-finance machinery behind step 3 — herding, momentum, and the psychology of buying-because-it-went-up — the [Crypto VC and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers) overview and the broader game-theory of coordination posts are the place to go.

Here is the loop in one concrete arc. A token sits quietly at \$2.00 on a thin book. A single \$10,000 buy walks it to \$2.70 (+35%) — pure Section 1 mechanics, no news. Someone screenshots the candle: "\$MOON breaking out, up 35%." A few hundred people who follow that account buy small amounts *because it went up*; their market orders hit the still-thin book and push it to \$3.50. Now it's up 75% and trending, so a bigger account amplifies it, and the price tags \$5.00 — up 150% on what may be, in total, a couple hundred thousand dollars of real buying spread across the climb. Then the early investor whose tokens just unlocked sells into the crowd, the fresh demand is exhausted, and the price falls back through \$3.50, \$2.70, \$2.00 on the same thin depth — faster than it rose. Nobody lied about a fundamental; the price wrote its own story, recruited an audience, and then took their liquidity.

The point is not that every rally is fake. Plenty of moves are driven by genuine demand and survive. The point is that on a thin float you *cannot tell the difference from the price alone* — a real rally and a reflexive one produce the same green candle. You have to look at the plumbing underneath.

## 7. Perpetual swaps and funding: a second channel into the price

One more mechanism, introduced briefly because it deserves its own post (and gets one later in the series). So far we've talked only about the **spot** market — buying and selling the actual token. But a huge share of crypto trading volume happens in **perpetual swaps** (or **perps**), and they feed back into the spot price in a way worth knowing about.

A **perpetual swap** is a derivative — a contract that tracks the price of a token *without an expiry date*, letting traders take leveraged long or short positions. Popularised by the exchange BitMEX around 2016, the perp has become the dominant instrument in crypto by volume. Because it never expires, it needs a mechanism to keep its price tethered to the underlying spot price. That mechanism is the **funding rate**:

- When perps trade *above* spot (too many leveraged longs), longs pay a small periodic **funding** payment to shorts. This makes being long expensive and nudges the perp price back down toward spot.
- When perps trade *below* spot (too many shorts), shorts pay longs, nudging it back up.

Why does this touch the spot order book we've been studying? Two ways. First, **arbitrage**: when the perp and spot diverge, arbitrageurs trade *both* to capture the gap — and their spot leg lands as real orders on the spot book. Second, **liquidations**: leveraged perp positions get force-closed when the price moves against them, and those forced closes are market orders that hit the book, often cascading (a falling price triggers long liquidations, which sell, which push the price down further, which triggers more liquidations). That cascade is a reflexive loop of its own, running through the same thin depth we've been examining — leverage pouring fuel on the fire.

The scale is not small. On volatile days, derivatives-tracking services have reported *billions of dollars* of leveraged positions liquidated across the crypto market in a single 24-hour window — for example around the 19 May 2021 sell-off discussed below. Every one of those liquidations is a forced market order landing on a book, and on the thin books of smaller tokens the effect is proportionally far more violent than on bitcoin. (Liquidation totals are estimates from third-party derivatives trackers, not audited exchange figures, and are best read as order-of-magnitude.)

You don't need the full mechanics today. Just hold the shape: **leverage and funding are a second faucet that pours orders onto the spot book**, and on a thin book that extra flow moves the price exactly as hard as any other order. We come back to perps, funding, and the perpetual-swap machinery in a dedicated post; the [BitMEX and the perp](/blog/trading/crypto/crypto-vc-and-market-makers) thread in the series has the deep version.

## 8. How the players use thin books

Everything so far is neutral mechanics. But once you know that a small order moves a thin book — and that the last trade re-prices the whole float — you also know the *toolkit* that sophisticated players have available. None of this requires a conspiracy; it falls straight out of the arithmetic, which is why the rest of this series spends its time on *who* runs these plays and how their business models reward it.

**Whales paint the tape.** A large holder who wants to *sell* has a problem: dumping into a thin book craters the price against them. So the pattern is often the reverse of what you'd expect — they *buy* first, in small clips, to walk the price up and light the reflexive candle from Section 6. The story recruits new buyers, the book above fills with fresh demand, and *that* new liquidity is what the whale sells into. The buying was never the goal; it was the bait that manufactured the exit. "Painting the tape" — trading to create a misleading impression of price or activity — is, in regulated markets, a recognised form of manipulation; the narrow point here is only that a thin float makes it *cheap to attempt*.

**Market makers time their depth.** A market maker with a token-loan-and-options deal (the crypto-native contract we dissect in the market-maker posts) has a view on when the price *should* move. It can post generous depth to keep a launch looking liquid and orderly, then thin its quotes ahead of a known catalyst — a token unlock, an exchange listing, a vesting cliff — so it is not the one absorbing the flow when the price gaps. The depth is a dial, and the firm quoting your book is the hand on it.

**Order "walls" steer sentiment.** A single very large *visible* limit order — a "buy wall" just below the price or a "sell wall" just above — is often placed not to trade but to be *seen*, nudging other traders' behaviour ("there's huge support at \$2, it can't fall below"). Because a limit order can be cancelled instantly, a wall can vanish the moment the price approaches it. When a resting order is posted with no intent to execute and pulled before it can be hit, that is **spoofing**, which is illegal and actively prosecuted in traditional markets; in crypto it is widely *reported* and far harder to police.

**Manufactured volume.** Liquidity and volume look reassuring, so there is a standing incentive to fake them. **Wash trading** — a trader, or a colluding pair, buying and selling with themselves to inflate reported volume without changing their net position — makes a thin token look actively traded and can lift it up an exchange's rankings. Several analyses have *alleged* that large fractions of reported volume on some venues and around some tokens are wash-traded; these are contested, firm-specific claims that this series treats as **reported / alleged**, never as settled fact, and always with the source attached. The defensive point stands regardless of any single allegation: **reported volume is not the same as real, two-sided depth**, and only the latter protects your fill.

Put together, these are not exotic tricks. They are the natural moves available to anyone who can see that the price is just the last trade and the float is thin. Knowing the mechanics doesn't make you a manipulator — but it does let you recognise the shape of one, and to ask, of any suspiciously clean rally, *who benefits if I buy here?*

## Common misconceptions

**"Market cap is how much money is in a coin."** No. Market cap is the *last trade price* multiplied by the *circulating supply*. It is an extrapolation, not a pool of cash. As the worked example showed, \$10,000 of real buying can add \$7,000,000 to market cap — the two numbers are not the same thing and are not even close.

**"A big market cap means I can sell my position at that price."** No. You can only sell into the *depth that's resting below you*. If you hold 1% of a token whose entire book is thin, trying to sell your stake will walk the price down through every level, and you'll realize a fraction of the marked value. Big market cap plus thin float is the most dangerous combination there is for an exit.

**"The price is set by supply and demand across the whole market."** Only loosely. At any instant the price is set by exactly *two* orders: the marginal buyer and the marginal seller at the top of the book. Everyone else — every long-term holder, every locked token — is a spectator whose paper wealth is being marked to those two orders' handshake.

**"A large price move must mean large money moved."** No, and this is the whole thesis. On a thin book the price move and the money moved are almost unrelated. A 35% candle can be \$10,000 of buying or \$10,000,000 — the candle alone can't tell you which. You have to look at the depth and the volume.

**"Slippage is just the trading fee."** No. The fee is a small fixed percentage the exchange charges (often ~0.1%). Slippage is a *separate* and often much larger cost that comes from *moving the price against yourself* as you eat the book. On a thin book, slippage can dwarf the fee by a hundred to one.

**"Limit orders always get me a better deal than market orders."** Not always — a limit order gives you price control but *no guarantee of execution*. If you post a limit buy below the market and the price runs away up, you never get filled and you miss the move entirely. Market orders guarantee execution at the cost of price; limit orders guarantee price at the cost of execution. Which is "better" depends entirely on whether you need to trade *now*.

**"High trading volume means the token is liquid."** Not necessarily. Volume can be manufactured through wash trading, and even genuine volume is backward-looking — it tells you what *has* traded, not what is resting on the book *right now*. A token can show millions of dollars of 24-hour volume and still have a thin book at this instant, because that volume happened in a burst that has since evaporated. Depth, not volume, is what your order actually trades against.

**"A green candle means real buyers showed up."** A candle tells you only that the price moved and roughly how much traded — not *why*, and not whether the buying was one motivated whale, a market maker's own flow, or genuine broad demand. On a thin book the very same candle can be produced by wildly different causes, and the reflexive loop means the candle itself recruits the next wave of buyers regardless of what set it off. Read the depth and the source of the flow, not just the colour of the bar.

## How it shows up in real markets

The mechanics above are not hypothetical. Here are documented episodes where thin resting liquidity — not any change in fundamentals — produced enormous price moves.

### 1. The GDAX ether flash crash — 21 June 2017

This is the canonical demonstration that a market order eats the book. On 21 June 2017, at roughly 12:30pm PT, a **multi-million-dollar market sell order** hit the ETH-USD book on GDAX (the exchange operated by Coinbase, later renamed Coinbase Pro). According to GDAX's own post-mortem, written by then-VP Adam White, the order filled from **\$317.81 down to \$224.48** in a single sweep — a **29.4%** book slippage on the first leg alone, exactly the "walk down the book" mechanic from Section 1, in reverse.

That 29.4% drop then triggered a cascade: roughly **800 stop-loss orders and margin-funding liquidations** fired automatically, each one a fresh market sell hitting an increasingly empty book, and ETH momentarily traded as low as **\$0.10** — a more than 99.9% crash from the day's high, in a matter of milliseconds. GDAX's investigation found no exchange malfunction and no market manipulation: the matching engine worked exactly as designed. The cause was simply that **liquidity at exchanges is limited**, and a large market order into a thin book, plus a cascade of forced liquidations, is enough to momentarily erase the price. Coinbase subsequently credited affected customers who were stopped out or liquidated. (Sources: GDAX / Adam White post-mortem; CoinDesk; CNBC, June 2017.)

The lesson maps one-to-one onto everything above: the price is the last trade, the book underneath was thin, and forced market orders (Section 7's liquidation cascade) took it to a price nobody would have quoted.

### 2. The Binance.US bitcoin flash crash — 21 October 2021

On 21 October 2021 at 11:34 UTC, the price of bitcoin on **Binance.US** plunged from around **\$65,760 to as low as \$8,200** — an **87%** drop — and snapped back within about a minute. Crucially, this happened on *one venue only*: on Bitstamp, the price dipped roughly **2.3%** and stayed above \$63,600 over the same span. Binance.US attributed the crash to a **bug in an institutional trader's algorithm** that dumped a large sell order onto their book. (Sources: CoinDesk; Bloomberg, October 2021.)

This is the cleanest possible illustration of the depth argument from Section 2. The *same* asset, at the *same* moment, moved 87% on a thin single-venue book and 2.3% on a deep one. Nothing about bitcoin the asset changed; the only variable was how much resting liquidity sat under the sell order on each exchange. A market cap of hundreds of billions did not stop one venue's book from being punched through to \$8,200.

### 3. The May 2021 leverage-cascade crash

On 19 May 2021, the broad crypto market fell sharply intraday — bitcoin dropped from around \$43,000 toward \$30,000 at the lows before partially recovering, and many altcoins fell much more. A major driver was the **liquidation cascade** described in Section 7: as prices fell, leveraged long positions on perpetual swaps were force-liquidated, and those forced market sells hit the spot and perp books, pushing prices lower and triggering the next wave of liquidations. Industry trackers estimated several billion dollars of positions were liquidated in a single day. (Sources: contemporaneous reporting, Reuters and The Block, May 2021 — liquidation totals are estimates from derivatives-tracking services, not audited figures, and should be read as order-of-magnitude.)

The mechanism is reflexivity plus leverage running through thin depth: each liquidation is a market order, each market order moves the price, and each move triggers more liquidations. The same loop that magnifies a rally magnifies a rout.

### 4. The everyday version: a fresh low-float listing

You don't need a famous crash to see this. Every week, a token launches with a small circulating float against a huge FDV, a market maker posts a thin book, and the price whips 20–50% on modest volume in the first hours of trading — up on a few coordinated buys, down when an early investor's unlocked allocation hits the book. There's rarely a news article; it's just the plumbing doing what plumbing does.

Watch a specific pattern and you'll spot it repeatedly: a new listing opens, spikes hard on day one (thin book, reflexive candle, everyone chasing), grinds sideways for a few weeks, and then steps down sharply on a *scheduled* unlock date as freshly-vested supply meets the same shallow book. The price move looks like "the market losing interest," but the mechanism is supply hitting depth — the calendar told you it was coming. The [lifecycle of a token from seed to unlock](/blog/trading/crypto/crypto-vc-and-market-makers) walks through exactly who is entering and exiting at each of those moments, and why launch-day retail so often ends up as the exit liquidity for the insiders who bought years earlier at a fraction of the price.

## When this matters to you

If you take one habit from this post, make it this: **check the depth before you size an order.** Concretely, before buying or selling any token, especially a small one:

- **Look at the order book, not just the price.** How much size is resting within a few percent of the current price? If a \$5,000 order would walk the price several percent, the book is thin — size down or slice your order.
- **Estimate your own slippage.** Mentally (or with the exchange's preview) walk your order up the book. If the average fill is meaningfully worse than the top-of-book price, that gap is a cost you pay every time, in and out.
- **Distrust the headline market cap.** Ask what the *float* is. A \$500M market cap on a 5%-circulating float is a very different (and more fragile) thing than the same market cap on a fully-circulating token. High market cap plus thin float is the profile most prone to violent moves in both directions.
- **Remember the depth is conditional.** The comforting book you see in calm markets is partly a market maker who may pull quotes exactly when volatility spikes — which is exactly when you might panic-trade. Liquidity is thinnest when you want it most.
- **Respect the convexity.** A big market order pays far more than a proportional share of the slippage. If you must move size, break it up over time and let the book refill.
- **Use the spread as a first-glance tell.** A tight spread (a few basis points — hundredths of a percent) usually signals an active market maker and real depth; a wide spread (a percent or more) is a flag that the book is thin and your slippage will be high. It's the fastest single check you can make before you even look at the depth.

In practice, reading depth is not hard. Most exchange interfaces show the order book right next to the chart, and many show a **depth chart** — the cumulative-liquidity staircase — where you can literally see how far the price would move for a given order size. Before you size a trade, do the thirty-second version of the worked examples above: glance at how much is resting within a few percent of the current price, and use the exchange's order preview (most spot venues estimate your average fill and slippage before you confirm). If the preview says a modest order will move the price several percent, that is the book telling you it is thin — believe it, and size down or slice.

None of this is a prediction about any token going up or down — it's educational, not advice. It is simply the ability to look at a price and ask the right question: *how thin is the book underneath this number, and who is on the other side of my trade?* Once you can see the order book, the headline market cap stops looking like a fact and starts looking like what it is — the last trade, wearing a costume. The rest of this series is about the players who know that better than anyone, and who trade accordingly.

## Sources & further reading

- **GDAX / Adam White**, "ETH-USD Trading Update" (post-mortem of the 21 June 2017 ether flash crash) — the source for the \$317.81 → \$224.48 first-sweep (29.4% slippage), the ~800 stop-loss and margin liquidations, the \$0.10 low, and the customer reimbursement.
- **CoinDesk**, ["\$13: Ether Prices Plunge in GDAX Exchange Flash Crash"](https://www.coindesk.com/markets/2017/06/21/13-ether-prices-plunge-in-gdax-exchange-flash-crash) (21 June 2017) and **CNBC**, "Ethereum price crashed from \$319 to 10 cents on GDAX after huge trade" (22 June 2017).
- **CoinDesk**, ["Bitcoin Price Flash Crash on Binance.US Attributed to Trader Algorithm Bug"](https://www.coindesk.com/markets/2021/10/21/bitcoin-price-flash-crash-on-binanceus-attributed-to-trader-algorithm-bug) (21 October 2021) — the \$65,760 → \$8,200 (87%) single-venue crash and the ~2.3% move on Bitstamp; **Bloomberg**, "Bitcoin Crashed 87% on Binance's U.S. Exchange Due to Algo Bug."
- **George Soros**, *The Alchemy of Finance* (1987) — the original statement of reflexivity.
- On this blog: [Crypto VC and Market Makers](/blog/trading/crypto/crypto-vc-and-market-makers) (the series hub), [Inside an Exchange: the Matching Engine and the Order Book](/blog/trading/capital-markets/inside-an-exchange-the-matching-engine-and-the-order-book), [Market Makers and the Spread](/blog/trading/capital-markets/market-makers-and-the-spread-who-provides-liquidity), and [Order Types and How an Order Travels to the Market](/blog/trading/capital-markets/order-types-and-how-an-order-travels-to-the-market).

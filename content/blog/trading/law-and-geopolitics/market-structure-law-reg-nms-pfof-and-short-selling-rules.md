---
title: "Market-structure law: Reg NMS, payment for order flow, and short-selling rules"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How the rules for routing, matching, and settling orders quietly shape your fills, who profits from your trade, and how a short squeeze detonates."
tags: ["regulation", "market-structure", "reg-nms", "payment-for-order-flow", "short-selling", "reg-sho", "gamestop", "best-execution", "trading"]
category: "trading"
subcategory: "Law & Geopolitics"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — The rules that govern how your order is routed, matched, and settled silently shape every fill you get; learn them and you stop being the slow money at the table.
>
> - **Reg NMS** built one national best price (the NBBO) and made it illegal to trade through it. That protects you on price but fragments your order across dozens of venues.
> - **Payment for order flow (PFOF)** is why your commission is \$0: a wholesaler like Citadel Securities pays your broker to fill your order, profiting from the spread and handing you a sliver of price improvement. Free trading is not free — the cost moved into the spread.
> - **Short selling** is borrow-sell-cover-return, and the legal plumbing (Reg SHO's locate and close-out, the borrow fee, the recall) is exactly what turns a crowded short into a squeeze.
> - **The number to remember:** GameStop went from \$17.25 on January 4, 2021 to \$347.51 on January 27 — a 20x move in 17 trading days — when forced short covering met a clearinghouse margin spike and brokers restricted buying.

On the morning of January 28, 2021, millions of retail traders opened their brokerage apps to buy more GameStop and found the buy button gone. Robinhood, Webull, and others had restricted purchases of GME and a handful of other names to "closing only" — you could sell, but you couldn't add. The stock, which had closed at \$347.51 the day before, cratered. Conspiracy theories bloomed instantly: the hedge funds had pulled the plug; the system was rigged for the rich.

The truth was both more boring and more interesting. No villain flipped a switch. A chain of *rules* — settlement timing, clearinghouse margin formulas, broker net-capital requirements, the mechanics of who must post how much collateral and when — tightened all at once and forced brokers to choke off the very buying that was driving the squeeze. The "rigging" was the plumbing. And almost nobody buying GameStop understood that the plumbing existed.

That is the whole point of this post. Below the price you see on the screen sits a dense layer of law and regulation that decides where your order goes, who gets to match it, what price you are guaranteed, who profits from handling it, and — at the extremes — whether a stock can melt up 20x or whether your broker has to pull the plug. You do not have to be a market-structure quant to trade well, but if you do not know these rules exist, you are the person at the poker table who does not know there is a rake.

![Order routing flow from a retail trader through a broker to wholesaler, exchange, or dark pool before the fill](/imgs/blogs/market-structure-law-reg-nms-pfof-and-short-selling-rules-1.png)

## Foundations: the order book, the NBBO, and the rules that bind them

Start with the simplest object in all of trading: the **order book**. For every stock, an exchange keeps two lists. On one side are **bids** — orders to buy, each with a price and a size ("buy 500 shares at \$10.00"). On the other are **offers** or **asks** — orders to sell ("sell 300 shares at \$10.02"). The highest bid and the lowest offer are the **best bid** and **best offer**. The gap between them is the **bid-ask spread**, and it is the single most important number you have never looked at. If the best bid is \$10.00 and the best offer is \$10.02, the spread is 2 cents.

A **market order** says "fill me now at the best available price." A **limit order** says "fill me only at \$10.00 or better." When you hit "buy" with a market order, you cross the spread: you pay the offer (\$10.02), not the bid. When you sell with a market order, you hit the bid (\$10.00). That round-trip — buy at the offer, sell at the bid — costs you the spread, every time, before any commission. Hold that thought; it is the hidden cost that makes "free" trading possible.

The order book is also a *queue*, and the queue has rules. Most venues match on **price-time priority**: the best price fills first, and within a price level, the order that arrived first fills first. This is why a resting limit order at the front of the queue is valuable — it gets filled before later orders at the same price — and why high-speed firms invest fortunes in shaving microseconds off the time to *post* a quote. When you place a limit buy at \$10.00 and there are already 5,000 shares bid at \$10.00 ahead of you, you are at the back of the line; sellers will hit the earlier orders first, and you may never fill if the price ticks up before your turn. The depth of the book — how many shares rest at each price level — is what determines whether a large order can fill at one price or has to **walk the book**, eating progressively worse prices as it consumes each level. A thin book is a wide, fragile market; a deep book absorbs size with little movement. None of this is visible on a simple price chart, but all of it is in the order book the rules govern.

There are also order types built specifically around these mechanics. A **marketable limit order** is a limit order priced aggressively enough to fill immediately (a buy limit at or above the offer) — it crosses the spread like a market order but caps your worst price. A **mid-point peg** order rests at the midpoint of the NBBO, splitting the spread. An **immediate-or-cancel (IOC)** order takes whatever is available right now and cancels the rest. Sophisticated traders pick the order type to control the trade-off between *certainty of execution* (market and marketable-limit orders fill now) and *price* (resting limit orders fill better but may not fill at all). The default "buy" button on most retail apps sends a plain market order — the most certain and the most expensive option on a wide spread.

### The NBBO: one best price across a fragmented market

Here is the complication that drives everything else. There is no single "the stock exchange" in the United States. There are sixteen-plus registered **exchanges** (NYSE, Nasdaq, Cboe's four markets, IEX, MEMX, and more), plus dozens of off-exchange venues. Each runs its own order book. So at any instant, the best offer for a stock might be \$10.02 on Nasdaq and \$10.03 on NYSE. Which one is "the" best price?

The answer is the **National Best Bid and Offer**, or **NBBO**: the single highest bid and lowest offer across *all* of the protected venues, consolidated into one feed. The NBBO is the official "best price in the country" at any moment. It is the benchmark your broker is measured against. When people say a stock is "trading at \$10.02," they mean the NBBO offer is \$10.02.

### Reg NMS: the rulebook that built the modern market

The law that stitched this fragmented mess into one logical market is **Regulation National Market System**, or **Reg NMS**, adopted by the U.S. Securities and Exchange Commission (the **SEC**, the federal agency that regulates securities markets) in 2005 and fully live by 2007. Three of its rules matter for you:

1. **The Order Protection Rule (Rule 611), a.k.a. the trade-through rule.** A "trade-through" is when a trade prints at a worse price than a protected quote available elsewhere. Rule 611 makes that illegal. If the NBBO offer is \$10.02 on Nasdaq, no venue may sell shares to you at \$10.03 while \$10.02 is sitting right there. Your order must be routed to (or matched at) the best price, or the venue must "satisfy" the better quote. This is the rule that *guarantees you the NBBO* on a market order.

2. **The Sub-Penny Rule (Rule 612).** For stocks priced \$1 and above, quotes must be in whole cents — you cannot post a bid at \$10.0001. This stops a tactic called "penny-jumping," where a high-speed trader steps in front of your \$10.00 bid by quoting \$10.0001 and grabbing the trade for a hundredth of a cent of improvement. (As you will see, the SEC's recent reforms partly *reverse* this for the most liquid names.)

3. **The Access Fee Cap (Rule 610).** Exchanges run a **maker-taker** model: they pay a rebate to traders who post resting limit orders (who "make" liquidity) and charge a fee to traders who take it with market orders (who "take" liquidity). Rule 610 caps that access fee — historically at \$0.0030 per share, now being reduced for many stocks. This fee is invisible to you but shapes where your broker routes your order, because rebates are real money to whoever captures them.

The maker-taker model creates its own conflict, distinct from PFOF, and it's worth understanding because it's the lit-market cousin of the same problem. A broker routing a *marketable* (liquidity-taking) order pays the access fee, but a broker routing a *resting* (liquidity-making) order *collects* the rebate. So a broker has a financial incentive to route to the exchange that pays the highest rebate or charges the lowest fee — which is not necessarily the venue that fills the customer best. This is exactly the kind of conflict best-execution duty is meant to police, and it's why the SEC's reforms target the access-fee cap directly: lower the rebate, and you weaken the incentive to route for the broker's benefit rather than the customer's. IEX, the exchange made famous by Michael Lewis's *Flash Boys*, was founded partly as a reaction to these incentives — it uses a "speed bump" and a flat, near-zero fee model specifically to neutralize the rebate-chasing and latency arbitrage that maker-taker pricing encourages. The point for you: even on lit exchanges, where there's no wholesaler paying for flow, the *fee structure* quietly tugs your order toward venues chosen for the broker's economics. Best execution is the rule standing between you and that tug, in both the PFOF world and the maker-taker world.

![Reg NMS before and after, showing a fragmented quote versus a protected national best bid and offer](/imgs/blogs/market-structure-law-reg-nms-pfof-and-short-selling-rules-2.png)

The figure above is the whole idea of Reg NMS in one picture. Before an order-protection rule, a venue could fill your buy at \$10.05 even while a better \$10.02 offer sat on another exchange — you simply overpaid because nobody was forced to look across venues. After Rule 611, the \$10.02 quote is *protected*: trading through it is banned, so your buy must execute at \$10.02 or better. The cost of that protection is fragmentation. To honor one national best price, your order may be sliced and routed across many venues — which is exactly the seam that payment for order flow slips into.

Two subtleties make Rule 611 less of a guarantee than it sounds, and you should understand both before you trust the "best price" on your screen. First, the rule only protects **top-of-book** quotes — the single best bid and best offer at each exchange — not the depth behind them. If 100 shares are offered at \$10.02 and 50,000 are offered at \$10.03, a buyer of 10,000 shares is *trading through* nothing illegal when most of the order fills at \$10.03; the protected \$10.02 quote was only 100 shares deep. The rule protects the *displayed best price*, not the price you'll actually pay for size. Second, the rule protects only **automated, immediately-accessible quotes** — a manual quote that can't be hit electronically isn't protected, which was part of the original justification for letting fast electronic markets ignore slower floor-based ones. The practical upshot: Rule 611 stops the crudest form of getting ripped off, but it is a floor, not a promise of a great fill.

There is also a deep debate about whether Reg NMS *over*-fragmented the market. By forcing every venue to honor every other venue's best quote, the rule made it viable to launch a new exchange with almost no liquidity — it would still receive routed orders whenever it happened to post the NBBO. The result is sixteen-plus exchanges and dozens of dark pools competing for the same flow, each with its own fee schedule and order types. Critics argue this complexity benefits the fastest intermediaries (who can see and act across all venues in microseconds) at the expense of ordinary investors. Defenders point out that spreads are historically tight and explicit costs near zero. Both can be true: the *visible* cost of trading collapsed, while the *structure* grew labyrinthine enough that only specialists can navigate it fully. That gap between cheap-and-simple-looking and structurally-complex is the recurring theme of this entire post.

### Market makers, exchanges, and dark pools

Three kinds of venue can match your order, and the distinction is load-bearing:

- **Lit exchanges** (NYSE, Nasdaq, Cboe) publish their order books *before* trades happen. The quotes are visible; price discovery happens here. They are **lit** because everyone can see the bids and offers.
- **Dark pools**, formally **Alternative Trading Systems (ATS)**, are private venues that do *not* publish pre-trade quotes. A large institution can try to buy a million shares in a dark pool without telegraphing its intent to the whole market (which would push the price up). They are "dark" because the order book is hidden until after a trade prints.
- **Wholesalers / market makers** are firms that stand ready to buy and sell continuously, quoting a bid and an offer and earning the spread. The dominant retail wholesalers are **Citadel Securities** and **Virtu Financial**. They are not exchanges; they are dealers who *internalize* your order — filling it from their own book rather than sending it to a lit exchange.

Dark pools exist for a real reason, and it is not sinister. Imagine a pension fund that needs to sell two million shares. If it posts that order on a lit exchange, every high-speed trader sees a giant seller and front-runs it, driving the price down before the fund can finish — the order *moves the market against itself*. A dark pool lets the fund seek a counterparty without broadcasting its intent, reducing this **market impact**. The trade still prints publicly *after* it happens (so the tape is complete), but the *pre-trade* quote is hidden. The trade-off is transparency: because dark-pool liquidity isn't displayed, it doesn't contribute to price discovery, and a market where too much volume moves into the dark can leave the lit quotes thin and unrepresentative. Regulators watch the share of off-exchange volume for exactly this reason. Dark pools are also where some of the conflict-of-interest enforcement has landed — several operators have paid penalties for misrepresenting how their pools worked or who could trade in them.

That last category — the wholesaler — is where most retail orders actually go, and it is the bridge to the second pillar of this post.

## Payment for order flow: why your trading is "free"

Until around 2013, buying 100 shares of stock cost you a commission — \$7 to \$10 at a discount broker, more at a full-service one. Today most brokers charge \$0. What changed? Not generosity. The business model changed, and the engine of that change is **payment for order flow (PFOF)**.

PFOF is exactly what it sounds like: a wholesaler *pays your broker* for the right to execute your orders. When you buy 100 shares of Apple on a zero-commission app, your broker does not send that order to Nasdaq. It sends it to Citadel Securities or Virtu, who fills it from their own inventory and pays your broker a small rebate — on the order of \$0.0015 per share — for sending the flow.

Why would a wholesaler pay for your order? Because **retail order flow is profitable to handle**. Retail traders are, on average, *uninformed* — they are not trading because they have a model that says Apple is about to move; they are buying because they read an article or got a tip. A market maker that fills a stream of uninformed orders earns the spread without getting picked off by smarter traders. This is the opposite of trading against a hedge fund that knows something you don't. So the wholesaler is happy to (a) fill you slightly *inside* the spread — giving you **price improvement** — and (b) still pay your broker for the flow, because what's left of the spread is pure profit.

The key concept underneath PFOF is **adverse selection**, and it is worth slowing down on because it explains the entire economics. A market maker's nightmare is trading against someone who *knows more than the price reflects* — an "informed" trader. If you sell stock to someone right before bad news, you've been adversely selected: you're now holding inventory that's about to drop. On a lit exchange, a market maker quotes to the whole world and cannot tell who is on the other side; some of the takers are informed, and the losses to them eat into the spread earned from the uninformed. **Retail flow is special precisely because it's pre-sorted to be uninformed.** A wholesaler that only fills retail orders sidesteps most adverse selection, so its realized spread (after losses to informed traders) is far higher per share than a lit exchange's. That higher realized spread is the pie that gets split into your price improvement, the broker's PFOF rebate, and the wholesaler's profit. PFOF is the mechanism that *separates* uninformed retail flow from the toxic flow on lit venues — and that separation is exactly why critics worry it leaves lit markets with a worse, more adversely-selected mix.

This filling-from-your-own-book practice is called **internalization**: the wholesaler doesn't route your order to an exchange at all; it takes the other side itself. Internalization is legal and ubiquitous — well over a third of total U.S. equity volume trades off-exchange, much of it internalized by a handful of wholesalers. The concentration is striking: **Citadel Securities and Virtu Financial together handle a large majority of internalized retail equity flow**, with smaller players like G1 Execution, Two Sigma Securities, and Jane Street rounding out the field. That concentration is itself a market-structure question — when two firms see the lion's share of retail order flow, they have an information advantage (they see what retail is doing in aggregate, in real time) that no exchange or ordinary trader has.

![Payment for order flow money diagram showing the wholesaler capturing the spread while rebating the broker and improving the trader's price](/imgs/blogs/market-structure-law-reg-nms-pfof-and-short-selling-rules-3.png)

The figure traces the money. You send a buy order; your broker (commission \$0) routes it to a wholesaler; the wholesaler fills it off the spread and splits the value three ways: a PFOF rebate (~\$0.0015/share) to the broker, price improvement (~\$0.003/share) to you, and the rest of the spread kept as profit. Three parties profit from your trade. You are one of them — but you are also the source of the value.

### Best execution: the rule that's supposed to protect you

Here is the obvious tension. If your broker is *paid* to route your order to a particular wholesaler, what stops the broker from routing to whoever pays the most rather than whoever fills you best? The legal answer is the **duty of best execution**: a broker must seek the most favorable terms reasonably available for a customer order — best price, speed, and likelihood of execution — not just the venue that pays the broker the most. This duty comes from FINRA rules and SEC interpretation, and it is the legal counterweight to PFOF.

In practice, best execution is measured through public disclosures. Under **SEC Rule 605**, market centers publish execution-quality statistics (effective spreads, price-improvement rates, speed). Under **Rule 606**, brokers disclose where they route orders and how much PFOF they receive. These reports are the docket you can actually read to see whether your broker is getting you a good deal — most retail traders never open them.

Best execution has teeth, and the enforcement record proves it. In December 2020 the SEC charged Robinhood with **misleading customers about how it made money**: Robinhood had downplayed PFOF in its disclosures while accepting unusually high payments that, the SEC alleged, came at the cost of *worse* execution prices for customers — even after accounting for the savings from \$0 commissions. Robinhood paid a **\$65 million** settlement without admitting or denying the findings. The case crystallized the core conflict: a broker is legally obligated to seek your best execution, yet it is *paid more* when it routes to whoever pays the most, and those two pulls do not always point the same way. The lesson for you is not that PFOF is fraud — it usually delivers genuine price improvement — but that the incentive structure requires you to verify, not trust, and the 605/606 reports are how you verify.

The debate over PFOF is genuinely two-sided, and you should hold both ideas at once:

- **The pro-PFOF case:** competition among wholesalers has driven *real* price improvement. Retail market orders frequently fill *inside* the NBBO, meaning you pay slightly less than the official best offer. And commissions went to zero. For a small trader, the all-in cost of a trade is arguably lower than it has ever been.
- **The anti-PFOF case:** the cost did not vanish; it moved into the spread, where you can't see it. A broker paid for flow has a conflict of interest. And the price improvement you get may be smaller than what you'd get in a more competitive auction for your order. This is why the SEC proposed an **order-competition rule** (more below).

Let's put numbers on the "free isn't free" claim, because that is the single most important misconception to kill.

#### Worked example: the hidden cost of a wide spread on a "free" \$10,000 order

You want to buy \$10,000 of a stock trading with a 2-cent spread: best bid \$49.99, best offer \$50.01, midpoint \$50.00. The "fair" price is the \$50.00 midpoint. Your zero-commission market order fills at the \$50.01 offer.

- Shares bought: \$10,000 / \$50.01 ≈ **199.96 shares** (call it 200).
- You paid \$50.01 vs. a \$50.00 mid. The half-spread cost = \$0.01 per share.
- On 200 shares: 200 × \$0.01 = **\$2.00 of hidden cost** on the buy.
- A full round trip (buy at the offer, later sell at the bid) costs the *full* spread: 200 × \$0.02 = **\$4.00**.

Now suppose the wholesaler gives you 0.2 cents of price improvement: you fill at \$50.008 instead of \$50.01. Your buy-side hidden cost drops to 200 × \$0.008 = **\$1.60**. Better — but still real, and still invisible on your "\$0 commission" confirmation.

Compare to the old world: a \$7 commission would have dwarfed this \$2–\$4 spread cost on a single \$10,000 trade. **So for an occasional trader, free-plus-PFOF is genuinely cheaper.** The lesson is not "PFOF is a scam"; it's "your real cost is the spread, so trade liquid names with tight spreads and prefer limit orders when the spread is wide."

#### Worked example: PFOF economics from the wholesaler's seat

Walk through the wholesaler's profit on your 200-share buy in the same 2-cent-spread stock. The wholesaler quotes \$49.99 bid / \$50.01 offer and fills your buy at \$50.008 (giving you 0.2 cents of improvement).

- The wholesaler captures the spread minus what it gives away. It bought (from a seller) near the bid, say \$49.99, and sold to you at \$50.008.
- Gross capture per share: \$50.008 − \$49.99 = **\$0.018**.
- On 200 shares: 200 × \$0.018 = **\$3.60** gross.
- It pays the broker PFOF of \$0.0015/share: 200 × \$0.0015 = **\$0.30**.
- It gave you \$0.002/share of improvement vs. the offer: 200 × \$0.002 = **\$0.40** (this is your benefit, already reflected in the fill).
- Net to the wholesaler ≈ \$3.60 − \$0.30 = **\$3.30** before its own hedging and tech costs.

Run that across millions of orders a day and you see why Citadel Securities and Virtu pay billions in PFOF and still profit handsomely. **The wholesaler's edge is volume of uninformed flow, not a big margin per trade — which is also why it can afford to give you a little improvement and still pay your broker.**

### The SEC's order-competition and tick-size reforms

In late 2022 the SEC proposed the most significant market-structure changes since Reg NMS. Two are worth knowing:

- **The order-competition rule (proposed):** instead of a broker handing your order straight to one wholesaler, certain retail orders would be exposed to a brief **auction** where multiple firms compete to fill you. The theory: competition extracts more of the spread for *you*. The pushback: auctions add complexity and latency, and the existing system already delivers improvement.
- **Tick-size and access-fee reforms (adopted 2024):** the SEC moved to allow **sub-penny quoting** (smaller tick sizes, e.g. half-cent) for the most liquid, tight-spread stocks, and to *lower the access-fee cap* from \$0.0030 toward \$0.0010 for those names. Smaller ticks can narrow spreads (good for takers) but shrink rebates (changing maker incentives). It is a direct tweak to the Rule 610/612 machinery you met above.

You don't need to memorize the rulemaking. You need to know that **the dials that set your spread are being adjusted by regulators**, and that a change to tick size or the access-fee cap can quietly widen or narrow the cost of every trade you make in a given stock. That is the transmission chain in miniature: a rule change → a spread change → your fill.

## Short selling: the legal plumbing of betting against a stock

The third pillar is the one that makes headlines. **Short selling** is how you profit when a stock *falls*. The mechanics are simple to state and surprisingly intricate underneath, and the intricacy is exactly what sets up a squeeze.

To short a stock you:

1. **Borrow** the shares from someone who owns them (via your broker, who sources them from a custodian or another client's margin account).
2. **Sell** the borrowed shares into the market at today's price.
3. Later, **buy them back** ("buy to cover") — hopefully at a lower price.
4. **Return** the shares to the lender and keep the difference.

If you short at \$50 and cover at \$40, you make \$10 per share. If the stock rises to \$60, you lose \$10 per share — and unlike a long position, your loss is theoretically *unlimited*, because a stock can rise forever but can only fall to zero.

That asymmetry is the defining feature of short selling and the reason it is so much riskier than going long. When you buy a stock at \$50, the worst case is it goes to zero: you lose \$50, capped. Your upside is unbounded. When you short at \$50, your *gain* is capped at \$50 (the stock can't go below zero) while your *loss* is unbounded. You are selling a lottery ticket: you collect a limited premium and accept a small chance of a catastrophic payout. This is precisely backwards from the risk profile most investors are comfortable with, and it is why short sellers tend to be specialists — hedge funds, dedicated short funds, and market makers — rather than retail traders. It is also why a short position requires constant management: a long position you can buy and forget, but a short can blow through your account if you stop watching.

Who actually lends the shares you borrow? The supply comes from a **securities-lending** market that sits mostly out of public view. The big lenders are **index funds, pension funds, and ETFs** — long-term holders who own shares anyway and earn extra yield by lending them out. Your broker sources borrow from these lenders (often through a custodian bank or a prime broker) and passes you a fee. Crucially, when you hold stock in a **margin account**, you typically grant your broker the right to lend *your* shares to short sellers — a practice called **rehypothecation**. So the GameStop bulls holding shares in margin accounts were, in many cases, unknowingly supplying the very shares being shorted against them. The amount of stock available to borrow — the **lendable supply** — is finite, and when it runs dry, the borrow fee explodes. That scarcity is the hinge on which every squeeze turns.

![Short sale lifecycle from locate and borrow through sell, recall risk, buy to cover, and return](/imgs/blogs/market-structure-law-reg-nms-pfof-and-short-selling-rules-4.png)

The lifecycle figure shows the legal gates. You can't just sell shares you don't have — that would be **naked shorting** (selling without first borrowing or arranging to borrow), which is restricted. Before the sale you need a **locate**: under **Regulation SHO (Reg SHO)**, the SEC's short-selling rulebook, Rule 203(b) requires your broker to have *reasonable grounds to believe the shares can be borrowed and delivered* before executing a short sale. Then you actually borrow (paying a **borrow fee**), sell, and hold an open position you must eventually close.

### The borrow fee, fails-to-deliver, and threshold securities

Two pieces of plumbing decide how dangerous a short is:

- **The borrow fee (the "cost to borrow").** Lendable shares are a market. For a stock that's easy to borrow (lots of float, few shorts), the fee is a fraction of a percent per year — trivial. For a **hard-to-borrow** stock (heavily shorted, little float left to lend), the fee can spike to *tens or even hundreds of percent annualized*. A 100% borrow fee means you pay 100% of the position's value per year just to maintain the short — roughly 0.27% per *day*. That carrying cost is a slow bleed that can force you out long before your thesis plays out.
- **Fails-to-deliver (FTDs).** When a short sale doesn't deliver the shares by the settlement date, it's a "fail." Reg SHO Rule 204 imposes a **close-out requirement**: a broker with a persistent fail must buy in (force-purchase) the shares to close it. Stocks with large, persistent fails land on the **threshold securities** list, which triggers stricter close-out rules. Persistent fails plus a forced buy-in are a squeeze accelerant.

### The recall: the trapdoor under every short

The most underappreciated risk is the **recall**. The shares you borrowed belong to someone else, who can demand them back *at any time* — for example, to vote them, or because they're selling. When your lender recalls, your broker must find replacement shares to borrow. If none are available (because the stock is hard-to-borrow and everyone's scrambling), you get a **forced buy-in**: your broker buys shares in the open market to return to the lender, closing your short *whether you want to or not, at whatever price the market demands.* In a squeeze, recalls cascade exactly when prices are spiking — forced buyers piling in at the worst moment.

### The uptick rule, old and new

For decades, U.S. law tried to stop shorts from "piling on" during a decline. The original **uptick rule (Rule 10a-1, 1938–2007)** said you could only short on an **uptick** — at a price higher than the last trade — so shorts couldn't hammer a falling stock straight down. The SEC repealed it in 2007, judging it obsolete in a decimalized, high-speed market.

After the 2008 crash reignited the debate, the SEC adopted the **alternative uptick rule (Rule 201, 2010)**, also called the **short-sale circuit breaker**. It works differently: it does nothing on a normal day, but if a stock drops **10% in a single day**, it flips on for the rest of that day and the next, and from then on you may only short at a price *above* the current best bid. The goal is to prevent short selling from accelerating a crash *after* a stock is already in distress. (For how broader trading halts and circuit breakers work, see the companion post on [circuit breakers and the legal plumbing of a crash](/blog/trading/law-and-geopolitics/circuit-breakers-halts-and-the-legal-plumbing-of-a-crash).)

## How a short squeeze detonates

Now combine the pieces. A **short squeeze** is what happens when a heavily shorted stock starts rising and the shorts are *forced* to buy back into a market that has no shares to give them.

![Short squeeze feedback loop where a rising price forces margin calls, recalls, and dealer hedging that drive more buying](/imgs/blogs/market-structure-law-reg-nms-pfof-and-short-selling-rules-5.png)

The feedback loop runs like this. A trigger — a buying surge, a short-thesis breaking, a coordinated retail push — lifts the price above a key level. That does three things at once. First, **margin calls**: as the stock rises, shorts' paper losses grow, and their brokers demand more cash collateral; those who can't post it are liquidated (forced to buy to cover). Second, **recalls**: lenders pull their shares back, and the borrow fee spikes, forcing more shorts out. Third, **dealer gamma**: traders who *sold* call options to the surging crowd must hedge by *buying* the underlying stock as it rises, adding fuel. All three streams converge on **forced buying** into thin supply, which drives the price *higher* — trapping the next cohort of shorts and restarting the loop.

The crucial insight: in a squeeze, the buyers are not optimists. They are shorts and dealers who *have* to buy, at any price, because the rules (margin, recall, hedging) compel them. That's why squeezes overshoot wildly past any sane valuation and then collapse just as fast once the forced buying exhausts.

The **gamma** leg deserves its own paragraph because it was the accelerant that made GameStop unique. When you buy a **call option** (the right to buy a stock at a set price), someone sold it to you — usually a market maker. The seller is now short the call and must **hedge** by buying some shares of the underlying, so that if the stock rises and the call moves against them, they're already covered. How many shares they buy depends on the option's **delta**, and as the stock rises toward and past the strike, the delta climbs toward 1.0 — meaning the dealer must buy *more* shares the higher the stock goes. The rate at which the dealer's hedge demand grows is the option's **gamma**. So a wave of call buying forces dealers into a self-reinforcing buy program: stock up → delta up → dealers buy more → stock up. This is a **gamma squeeze**, and when it stacks on top of a short squeeze — as it did in GameStop, where retail bought both shares *and* out-of-the-money calls — the two feedback loops compound. (The mechanics of how dealers price and hedge this convexity live in the [volatility surface](/blog/trading/quantitative-finance/volatility-surface).)

It is worth being clear about what a squeeze is *not*. A squeeze is not a verdict on the company's value — GameStop's business did not improve 20x in seventeen days. It is a *liquidity* event: too many forced buyers chasing too few available shares. That distinction is the whole edge. A fundamental investor asks "what is this worth?" A squeeze trader asks "who is forced to buy or sell next, and is there anything for them to trade against?" Those are different questions, and conflating them is how people buy the top of a squeeze believing they've found a misunderstood growth story.

#### Worked example: a short squeeze P&L with the borrow fee

You short 1,000 shares of a stock at \$20.00, collecting \$20,000 of proceeds. The borrow fee is a punishing 50% annualized because the stock is hard-to-borrow.

- **Daily borrow cost:** 50% / 365 ≈ 0.137% per day on the position value. On day one: 0.00137 × \$20,000 ≈ **\$27/day** — and it climbs as the price rises.
- The stock starts squeezing. After two weeks it's at \$60.00. Your mark-to-market loss: (\$60 − \$20) × 1,000 = **−\$40,000**, three times your initial proceeds.
- Your broker issues a **margin call** for additional collateral. Say you can't meet it. The broker force-covers: buys 1,000 shares at \$60 to close. Realized loss = **−\$40,000**, plus ~\$400 of accumulated borrow fees over the two weeks.
- Had the lender **recalled** at \$80 on a spike, your forced buy-in would crystallize a (\$80 − \$20) × 1,000 = **−\$60,000** loss — you don't get to wait for the pullback.

**The short's loss is uncapped and the rules can force you to realize it at the worst possible price — that asymmetry is the entire danger of being short a squeezing stock.**

## Common misconceptions

**"Free trading means there's no cost."** There is always a cost; on a zero-commission trade it's the **spread** plus any market impact. In the worked example, a \$10,000 buy in a 2-cent-spread stock carried \$1.60–\$2.00 of hidden one-way cost even after price improvement — invisible on a "\$0 commission" confirmation. The fix is not to avoid free brokers but to **trade liquid, tight-spread names and use limit orders when the spread is wide**, so you control the price you pay rather than crossing a fat spread blindly.

**"Short sellers crashed the stock."** This conflates mechanics with narrative. Short selling adds *sell* pressure, yes — but a short sale is also a *future guaranteed buyer*, because every share sold short must eventually be bought back to cover. In a squeeze, the shorts are the ones being crushed: GameStop's shorts lost an estimated **\$5–\$6 billion in January 2021** as they were forced to cover. The popular story ("hedge funds shorted it to zero") had the causality backwards — the shorts were the fuel, not the arsonists.

**"You always get the best price."** Reg NMS Rule 611 guarantees you won't be *traded through* the protected NBBO — but the NBBO itself is a snapshot of one-cent ticks across a fragmented market, and on a fast-moving or thinly-quoted stock the "best" displayed price can be stale or thin. A large market order can **walk the book**, filling progressively worse as it eats through each price level. Rule 611 protects you from one specific abuse; it does not promise that the displayed best price is deep enough to fill your whole order at that level.

**"Naked shorting is rampant and it's why stocks fall."** True naked shorting — selling without locating shares — is restricted by Reg SHO's locate (Rule 203) and close-out (Rule 204) rules, and persistent fails-to-deliver land a stock on the threshold list with mandatory buy-ins. Fails happen for benign operational reasons too. The data on FTDs is public (the SEC publishes it twice monthly); for almost every large stock the fails are a tiny fraction of volume, not a hidden conspiracy.

## How it shows up in real markets

### GameStop, January 2021: a market-structure stress test

The GameStop episode is the cleanest case study in market-structure law ever handed to retail investors, because every layer of plumbing showed up at once.

![GameStop share price in January 2021 rising from 17 dollars to 347 then collapsing](/imgs/blogs/market-structure-law-reg-nms-pfof-and-short-selling-rules-6.png)

The price path tells the story: GME closed at \$17.25 on January 4, ground to \$76.79 by January 25, then exploded to \$347.51 on January 27 before brokers restricted buying and it collapsed to \$193.60 on January 28 and \$53.50 by February 4. (These are raw, pre-split prices; the stock did a 4-for-1 split in 2022.) The setup was a stock with **short interest reported above 100% of its float** — more shares had been sold short than actually existed to lend, because the same shares had been re-lent. When buying overwhelmed the available supply, the squeeze loop above ran at full force: margin calls, recalls with borrow fees spiking into the hundreds of percent, and dealer call-hedging all forcing buyers in.

Then the *settlement* plumbing bit. U.S. equities settled on **T+2** in 2021 (trade date plus two business days; the U.S. moved to T+1 in May 2024). Between trade and settlement, brokers must post collateral to the **clearinghouse** — the **NSCC** (National Securities Clearing Corporation, part of the DTCC) — to cover the risk that a trade fails. The clearinghouse's margin formula scales with **volatility and concentration**. As GME's volatility exploded and trading concentrated in a few names, the NSCC's collateral demand on Robinhood reportedly spiked into the **billions of dollars** overnight. Robinhood didn't have the capital on hand. To reduce its collateral requirement, it restricted *opening* buys in the volatile names — which is why the buy button vanished on January 28. (The SEC's later staff report found the restrictions traced to these clearing and capital requirements, not to a conspiracy to protect hedge funds.)

To see *why* the collateral demand exploded, you have to understand what the clearinghouse is insuring against. When you buy GME on Monday, you don't actually own the shares until settlement two days later; in between, your broker has promised the clearinghouse it will deliver the cash, and the seller's broker has promised to deliver the shares. The clearinghouse stands in the middle and guarantees both legs — if your broker fails, the NSCC eats the loss. To protect itself, the NSCC collects margin sized to the *worst plausible price move* over the settlement window. The formula has a **volatility component** (a wildly swinging stock could move far against an unsettled trade) and a **concentration add-on** (if a broker's unsettled trades pile into one volatile name, the risk isn't diversified away). GME was the perfect storm for this formula: extreme volatility *and* extreme concentration, because so much of Robinhood's volume was that one stock. The margin requirement is mechanical — no human at the NSCC decided to punish Robinhood; the volatility and concentration inputs simply spat out a number in the billions.

Robinhood faced a brutal choice: post collateral it didn't have, or shrink the requirement. It chose to shrink it the only way it quickly could — by halting *opening* buys in the volatile names, which capped the new unsettled risk it was accumulating. (Selling reduces a customer's position and the associated risk, which is why selling stayed open; buying adds to it.) Robinhood scrambled to raise over **\$3 billion** in emergency capital that week to meet the demands and reopen buying. The episode exposed a structural fragility in the zero-commission model: a broker carrying enormous volume in volatile names is one volatility spike away from a capital crisis, because the clearing system's margin scales faster than the broker's balance sheet.

So the trading halt that looked like sabotage was the settlement-and-margin machinery doing exactly what it's designed to do under stress. The regulatory aftermath fed directly into the reforms you met above. The headline change was the move to **T+1 settlement** in May 2024: compressing the settlement window from two days to one roughly *halves* the time over which collateral risk can accumulate, which mechanically lowers the clearing margin a broker must post during a volatility spike — a direct structural response to the GameStop lesson. The SEC's PFOF and order-competition proposals, the renewed scrutiny of "gamification" in trading apps, and the debate over real-time settlement (T+0) all trace back to the January 2021 stress test. This is the transmission chain at full length: a market-structure shock → a regulatory review → a rule change (T+1) → a quieter, less collateral-hungry plumbing the next time volatility spikes.

### A volatility and structure stress map

GameStop wasn't an isolated glitch; it sits in a lineage of episodes where market structure itself was the stress.

![VIX close at market structure and macro stress events from the 2010 flash crash to the 2024 yen carry unwind](/imgs/blogs/market-structure-law-reg-nms-pfof-and-short-selling-rules-7.png)

The chart shows the **VIX** (the market's expected-volatility index, the "fear gauge") at a series of stress events. The amber bars are market-*structure* stresses: the **2010 flash crash** (VIX 32.8), when fragmented routing and a withdrawal of liquidity sent the Dow down ~1,000 points in minutes; **2018's "Volmageddon"** (VIX 37.3), when leveraged short-volatility products imploded; and the **2024 yen-carry unwind** (VIX 38.6), a leverage-and-liquidity cascade. The slate bars are macro shocks (COVID at VIX 82.7, Ukraine, SVB). The point: a meaningful share of the biggest volatility spikes were *plumbing* events — fragmentation, leverage, and forced liquidation — not fundamental news. That is precisely why market-structure rules (circuit breakers, the alternative uptick rule, clearing margin) exist, and why understanding them is edge.

### A spread change after a rule

The clearest everyday example of structure moving prices is the **2001 decimalization** of U.S. stocks: when quotes shifted from fractions (sixteenths, or 6.25 cents) to pennies, spreads collapsed for liquid names from ~6 cents toward 1 cent. Trading got cheaper for retail almost overnight — and harder for traditional market makers, whose per-trade margins shrank, accelerating the rise of high-speed electronic firms. The same dynamic is in motion with the 2024 tick-size reforms: change the minimum quoting increment and you change the spread, the rebate, and who profits from making markets. Rule change → spread change → your fill.

Decimalization is also a cautionary tale about second-order effects, because the spread compression did not come for free. As per-trade margins shrank, market makers responded by **quoting less size** — posting fewer shares at each price level so they weren't exposed when the spread no longer compensated them for the risk. Some researchers argue this hollowed out displayed depth, especially in smaller stocks, contributing to the long-running concern that small-cap liquidity is worse than the tight headline spreads suggest. It is a clean illustration of why you should never read a single market-structure metric in isolation: a rule that tightened the *visible* spread simultaneously thinned the *invisible* depth behind it. The same caution applies to the 2024 reforms — narrower ticks may tighten spreads in liquid megacaps while doing little, or even harming depth, in the long tail of less-traded names. Whenever a regulator turns a market-structure dial, ask not just "what happens to the spread?" but "what happens to the depth, the rebate, and the incentive to make markets at all?"

## How to trade it: the market-structure playbook

You will not out-route Citadel Securities. But knowing the rules turns several invisible costs and risks into manageable decisions.

**1. Mind the spread, not the commission.** Your real cost is the spread plus impact. Before trading, glance at the bid-ask. A penny spread on a \$50 megacap is noise; a 30-cent spread on a \$5 small-cap is a 6% round-trip tax. **Trade liquid names when you can, and size down in wide-spread names.**

**2. Use limit orders when the spread is wide or the stock is fast.** A market order crosses the spread and can walk the book on a thin name. A limit order caps the price you pay. On megacaps with penny spreads, market orders are fine; on small-caps, illiquid hours, or right after news ([liquidity gaps around news](/blog/trading/event-trading/liquidity-and-gaps-around-news) are real), limit orders protect you from a terrible fill.

**3. Know where your flow goes — and read the 605/606 reports.** If you trade enough that execution quality matters, your broker's Rule 606 report tells you who they route to and how much PFOF they get; Rule 605 data shows the execution quality. For most people the right takeaway is simpler: PFOF makes occasional trading cheap, but if you're trading *size* or in *illiquid* names, a broker that routes to lit markets or offers direct routing may fill you better.

**4. Read short interest, borrow, and days-to-cover for squeeze risk.** Three signals, in order of usefulness:
- **Short interest as a percentage of float** — above ~20% is crowded; near or above 100% (as GME was) is a powder keg.
- **The borrow fee / cost-to-borrow** — a fee spiking from 1% to 50%+ annualized means lenders are scarce and shorts are under pressure *now*.
- **Days-to-cover (the short interest ratio)** — shares short divided by average daily volume; it estimates how many days of normal trading the shorts would need to all buy back. High days-to-cover means the exit door is narrow.

**5. Know what *invalidates* a squeeze thesis.** A squeeze needs forced buyers and scarce supply. It fails when: short interest has already collapsed (the shorts covered — the fuel is gone); the borrow fee normalizes (lenders returned, no pressure); the company issues new shares (an **ATM offering** floods supply at the worst moment for longs — companies often do this *into* a squeeze, as AMC did in 2021); or the broader tape turns risk-off and the speculative bid evaporates. If you're long a squeeze, the borrow fee and short interest *falling* is your exit signal, not a dip to buy. For sizing speculative, fat-tailed positions like this, see the work on [positioning and the pain trade](/blog/trading/event-trading/positioning-and-the-pain-trade).

There is a subtle data trap worth flagging: short-interest figures are **stale**. U.S. exchanges report short interest only twice a month, with a lag of several days, so the number you see may be a week or more old — and in a fast squeeze, the entire short base can cover in *hours*, long before the next official print. By the time a headline screams "short interest still over 100%," the shorts may already be out and the fuel gone. The borrow fee and the *availability* of shares to borrow update far faster and are the better real-time gauge: when borrow goes from impossible-and-expensive back to easy-and-cheap, the lenders have returned, the forced-buying pressure has bled off, and the squeeze is over regardless of what the stale short-interest number says. Treat short interest as the slow-moving setup and the borrow fee as the fast-moving trigger and exit.

**6. Respect the role of options and the broader tape.** A pure short squeeze on borrowed shares is powerful; a short squeeze *stacked with a gamma squeeze* — heavy retail call buying forcing dealers to hedge — is what produces the truly violent melt-ups. So when you scan for squeeze candidates, look at the **options activity** too: a surge in near-dated, out-of-the-money call buying is the gamma fuel. And remember that squeezes are risk-on phenomena that need a willing speculative crowd; they tend to die when the macro tape turns defensive and that crowd retreats to safety. A squeeze is a bet on *flow and positioning*, not value — size it as the asymmetric, time-limited gamble it is, never as a long-term investment, and decide your exit before you enter, because the collapse is as fast as the melt-up.

#### Worked example: days-to-cover and the GameStop setup

Use round numbers from GameStop's January 2021 setup to compute the squeeze pressure.

- Shares short reported around **70 million** in early January 2021 — against a float of roughly **50 million** shares, i.e. **short interest > 100% of float** (shares were re-lent and shorted multiple times).
- Average daily volume in the quiet pre-squeeze period was roughly **10 million shares/day**.
- **Days-to-cover** = shares short / avg daily volume = 70M / 10M = **7 days**.

Seven days-to-cover means that even at normal volume, the shorts would need a full week of *being the entire buy side* just to close — physically impossible without driving the price up violently. Now layer on >100% short interest (more shorts than shares to buy back) and a borrow fee spiking toward triple digits, and you have the structural recipe for a 20x melt-up. **Days-to-cover plus short-interest-over-float is the dashboard: the higher both run, the harder the squeeze if a trigger arrives — and the GME reading was off the charts on both.** (For the math of building positions and probabilities around binary, structure-driven events, the [quantitative-finance order-book and execution toolkit](/blog/trading/quantitative-finance/order-book-simulator-quant-research) is the natural next step.)

## The bottom line

Market-structure law is the invisible architecture of every trade you make. **Reg NMS** built one national best price and made trading through it illegal — protecting you on price while fragmenting your order across venues. **Payment for order flow** turned that fragmentation into a business: wholesalers pay your broker for your (profitable, uninformed) flow, give you a sliver of price improvement, and keep the spread — which is why "free" trading is real but not actually free. And the **short-selling rules** — Reg SHO's locate and close-out, the borrow fee, the recall, the alternative uptick rule — are the exact plumbing that turns a crowded short into a 20x squeeze and, when settlement margin spikes, can make your broker pull the plug.

You can't beat the wholesalers at their own game. But you can stop paying the invisible tax: trade tight-spread names, use limit orders when it matters, read short interest and the borrow fee before you bet for or against a squeeze, and remember that the scariest market "rigging" stories are usually just the plumbing under stress.

## Further reading & cross-links

- [Circuit breakers, halts, and the legal plumbing of a crash](/blog/trading/law-and-geopolitics/circuit-breakers-halts-and-the-legal-plumbing-of-a-crash) — how LULD bands and market-wide circuit breakers shape crash dynamics, the natural companion to this post.
- [Securities law 101: the '33 and '34 Acts and the SEC](/blog/trading/law-and-geopolitics/securities-law-101-the-33-and-34-acts-and-the-sec) — the disclosure regime that underpins price discovery and gives the SEC its authority to write Reg NMS and Reg SHO.
- [Insider trading, Reg FD, and what is actually illegal](/blog/trading/law-and-geopolitics/insider-trading-reg-fd-and-what-is-actually-illegal) — the other half of who gets an edge from information.
- [Liquidity and gaps around news](/blog/trading/event-trading/liquidity-and-gaps-around-news) — why spreads blow out and fills get ugly exactly when you most want to trade.
- [Positioning and the pain trade](/blog/trading/event-trading/positioning-and-the-pain-trade) — how crowded positioning (including crowded shorts) sets up forced-flow reversals.
- [The order-book and execution-algorithm toolkit](/blog/trading/quantitative-finance/order-book-simulator-quant-research) — the quantitative mechanics of matching, routing, and minimizing impact.
- [The volatility surface](/blog/trading/quantitative-finance/volatility-surface) — how option dealers' hedging (the "gamma" leg of the squeeze loop) prices into the underlying.

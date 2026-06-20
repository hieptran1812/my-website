---
title: "The Trading Book: Market-Making, Flow vs Prop, and the Volcker Rule"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How a bank's markets division actually makes money: market-making and the bid-ask spread, the difference between agency flow and proprietary trading, why the trading book is a separate ledger from the banking book, why the Volcker rule banned prop trading at deposit banks, and how VaR limits cap the risk."
tags: ["banking", "trading-book", "market-making", "bid-ask-spread", "proprietary-trading", "volcker-rule", "value-at-risk", "flow-trading", "market-risk", "london-whale"]
category: "trading"
subcategory: "Banking"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A bank's markets division mostly earns a small, repeatable margin called the *bid-ask spread* for standing ready to buy and sell securities on demand, the way a shop earns a markup for keeping goods in stock; the danger is the inventory it has to hold to do that job.
>
> - **Market-making** earns the spread by quoting a price to buy (the *bid*) and a price to sell (the *ask*) at the same time, and pocketing the gap — but it carries *inventory risk*, because between buying and selling the price can move.
> - **Flow vs prop** is the core distinction: *agency flow* passes a client's order straight to the market and earns a fee with almost no risk; *proprietary (prop) trading* uses the bank's own money to bet on prices, with all the risk on the bank.
> - The **trading book** is a separate ledger from the **banking book**: it is marked to market daily, attracts *market-risk* capital based on *Value at Risk (VaR)*, and is meant for short-term resale, not for holding to maturity.
> - The **Volcker rule** (US, effective 2014-15) banned deposit-taking banks from prop trading while still permitting market-making and hedging — because a bank with insured deposits should not be gambling its capital.
> - **The one number to remember:** a market-maker quoting a 0.20-wide spread on a \$100 bond earns about \$0.10 of edge on each side, so on \$50 million of two-way volume in a day, roughly \$25,000 of spread — tiny per trade, large at scale, and instantly erased by one bad inventory position.

In April and May of 2012, a single trading desk inside JPMorgan's Chief Investment Office in London built a derivatives position so large the market gave its trader a nickname: the *London Whale*. The desk was supposed to be hedging the bank's risk. Instead it had quietly turned into one of the biggest one-way bets in the credit market, and when the position moved against it, the bank lost more than \$6 billion. The official report later showed the desk's own risk model had been changed in a way that made the position look half as risky as it was, and that its *Value at Risk* limit — the cap on how much it was allowed to lose on a bad day — had been breached and then quietly raised rather than enforced.

That episode is the whole subject of this post in miniature. A bank's markets division is supposed to do something genuinely useful and genuinely low-margin: stand in the middle of the market, quote a price to anyone who wants to trade, and earn a sliver of the *spread* for the service. That is *market-making*, and it is a respectable, repeatable business. The trouble starts when the same desk stops serving clients and starts betting the bank's own capital — *proprietary trading* — because then a single position can swallow years of patient spread income, and the money at risk is, ultimately, depositors' money. The London Whale is what happens when the line between the two blurs.

This is also where our running theme for the whole series shows up in a new disguise. A bank is a leveraged, confidence-funded machine that borrows short and lends long, earns the spread, and survives only as long as its thin equity cushion absorbs losses faster than they arrive. The trading book is a second place that same spread game gets played — and a second place that same thin cushion can get eaten, much faster than a loan book ever could, because a trading position can lose a year's profit in an afternoon. The diagram below is the mental model for the whole post: a market-maker sitting between sellers and buyers, quoting both sides and keeping the difference.

![Market-maker between sellers and buyers earning the spread](/imgs/blogs/the-trading-book-market-making-flow-vs-prop-and-the-volcker-rule-1.png)

## Foundations: the trading book, market-making, and the words you need

Before we go deep, let's build every term from zero. If you have never read a bank's annual report, none of this should require prior knowledge.

### The banking book vs the trading book

A bank actually keeps two separate ledgers, and the difference between them is the most important idea in this post.

The **banking book** is everything the bank intends to *hold*. A mortgage it made and plans to keep for thirty years, a corporate loan it expects to be repaid over five years, a portfolio of government bonds it bought to park cash and collect the coupon — all of that lives in the banking book. The intent is to hold the asset and earn the *spread* between what the asset pays and what the bank's funding costs. (If "spread" is new: it is simply the gap between two rates — say, earning 6% on a loan while paying 2% on deposits, a 4% spread.) The banking book is accounted for mostly at *amortized cost*, meaning the bank carries the loan at roughly what it lent, and only takes a hit when a borrower actually looks likely to default.

The **trading book** is everything the bank intends to *resell*. When a bank's markets desk buys a bond not to keep it but to sell it on to a client an hour later, that bond sits in the trading book. The intent is short-term: buy low, sell slightly higher, collect the *turn*. The trading book is accounted for at *mark to market* — every position is repriced at the current market price every single day, and the resulting gain or loss flows straight into the bank's profit and loss (its "P&L") that day. There is no waiting for a default; the loss is recognized the moment the price moves.

That single difference in *intent* — hold vs resell — cascades into opposite accounting and opposite capital rules, as the comparison below lays out.

![Banking book versus trading book intent accounting and capital](/imgs/blogs/the-trading-book-market-making-flow-vs-prop-and-the-volcker-rule-2.png)

The reason regulators care so much about which book a position lives in is that the two attract completely different *capital charges* — the amount of the bank's own equity it must set aside against possible losses. The banking book attracts *credit-risk* capital (will the borrower pay me back?). The trading book attracts *market-risk* capital (will the price move against me before I can sell?). A bank that wanted to dodge a heavy capital charge has, historically, been tempted to park a position in whichever book charges less — and a lot of regulation since 2008 has been about closing that door, which is exactly why the *boundary* between the two books is now policed so tightly.

### What "market-making" actually means

Here is the cleanest way to understand market-making: **a market-maker is the shopkeeper of securities.** A corner shop buys bread at wholesale and sells it at retail, keeping the markup. It does not consume the bread or bet on bread prices; it earns its living by *being there* — by holding stock on the shelf so that when you walk in wanting bread, it is available immediately. The markup is the price you pay for that convenience.

A market-maker does the same thing with financial instruments. It continuously quotes two prices for, say, a particular bond:

- the **bid** — the price at which it is willing to *buy* the bond from you, and
- the **ask** (or *offer*) — the slightly higher price at which it is willing to *sell* the bond to you.

The gap between them is the **bid-ask spread**, and it is the market-maker's markup. If the bid is 99.90 and the ask is 100.10, the spread is 0.20. A seller who needs cash now sells to the dealer at 99.90; a buyer who needs the bond now buys from the dealer at 100.10; the dealer captures the 0.20 difference for providing *immediacy* — the ability to trade right now without waiting for a matching counterparty to appear. That service is called *providing liquidity*, and it is genuinely valuable: without market-makers, you might wait hours or days for someone who wants the exact opposite of your trade.

### Flow vs prop: who is the bank trading for?

The single most important distinction in a markets division is *who the trade is for*.

- **Flow trading** means the bank is trading to serve a *client's* order. Within flow there are two flavors. *Agency flow* means the bank acts purely as a middleman — it takes your order and routes it to the market, never holding the position itself, and earns a commission. *Principal flow* (which is what market-making is) means the bank takes the other side of your trade onto its own book, then works the position off later — earning the spread, but carrying the risk in between.
- **Proprietary (prop) trading** means the bank is trading for *itself* — putting its own capital at risk to bet on where prices are going, with no client involved at all. There is no spread to earn and no commission; the only way to make money is to be right about the direction of the market.

That distinction — flow for clients vs prop for the house — is the line the Volcker rule draws. We will return to it in depth.

### Value at Risk in one paragraph

The last term you need is **Value at Risk (VaR)**. It answers one question: *on a normal bad day, how much could this book lose?* More precisely, a one-day 99% VaR of \$7 million means: "on 99 out of 100 days, the book should lose no more than \$7 million; only on the worst 1 day in 100 should it lose more." VaR is the standard *limit* regulators and risk managers put on a trading book — a cap on how much loss the desk is allowed to expose the bank to. It is enormously useful and, as the London Whale showed, enormously gameable. We will give it a full section, and link out to the proper statistics. For now, just hold the one-sentence version: VaR is a line drawn in the loss tail.

## Earning the spread: the economics of market-making

Let's make the market-maker's business concrete, because once you see the arithmetic, the whole division clicks into place.

The market-maker's income per trade is *half the spread* on average, not the whole spread. Why half? Because the *fair* mid-price sits in the middle of the bid and the ask. When a seller hits the bid at 99.90, the dealer buys at 0.10 below the 100.00 mid — that 0.10 is the edge. When a buyer lifts the ask at 100.10, the dealer sells at 0.10 above the mid — another 0.10 of edge. If, over a day, equal numbers of buyers and sellers come through, the dealer earns 0.10 per trade and ends the day *flat* (holding no net position). The full 0.20 spread is only captured on a *round trip* — one buy and one matching sell.

#### Worked example: a market-maker's bid-ask P&L on a day's volume

You run a market-making desk in a \$100 corporate bond. You quote a bid of 99.90 and an ask of 100.10, so the mid is 100.00 and your edge is \$0.10 per \$100 of face value on each side.

Over the day, clients trade \$50 million of face value with you — and suppose the flow is roughly balanced: about \$25 million sold to you (hitting your bid) and \$25 million bought from you (lifting your ask). On each \$100 of face you earn \$0.10, which is 0.10% of face value.

- Edge per dollar of face traded: \$0.10 / \$100 = 0.001, i.e. 0.10%.
- Total volume: \$50,000,000.
- Gross spread income: \$50,000,000 × 0.001 = **\$50,000**.

But wait — that \$0.10 per side, times \$50 million, double counts. Let's be careful: the desk earns \$0.10 of edge on the \$25 million it bought *and* \$0.10 on the \$25 million it sold. So income = (\$25,000,000 × 0.001) + (\$25,000,000 × 0.001) = \$25,000 + \$25,000 = **\$50,000**? No — re-check the unit. The edge is \$0.10 per \$100 of face, i.e. 0.10% of face, applied once to each leg. \$25m × 0.10% = \$25,000 per leg, two legs, \$50,000. That is the gross. The conservative number to remember is the per-leg figure: roughly **\$25,000 of clean spread on \$50 million of one-directional volume**, doubling to \$50,000 only when the day's buys and sells balance into full round trips.

The intuition: market-making income is *volume times a tiny fraction*. The edge per trade is almost nothing; the business only works because the volume is enormous and the desk does it thousands of times a day. This is the spread business all over again — the same engine as the bank's loan book, just measured in basis points per trade instead of basis points per year.

A *basis point* (bp), for the record, is one hundredth of a percent: 0.01%. A 0.20 spread on a 100.00 bond is 20 bps wide. Spreads on the most liquid government bonds can be under 1 bp; on an illiquid corporate or emerging-market bond they can be 50-100 bps. Wider spread, more edge per trade — but also fewer trades, because nobody wants to cross a wide spread.

## Inventory risk: the part that can hurt

If market-making were only about earning half the spread on balanced flow, it would be a quiet annuity. It is not quiet, because flow is almost never balanced, and the dealer is left *holding inventory*.

Picture the day going one way. A wave of sellers hits your bid all morning and almost nobody lifts your ask. You have now *bought* \$30 million of the bond and sold almost none of it. You are long \$30 million of inventory, and you did not choose to be — you became long because your job is to buy from anyone who wants to sell. Now the price matters enormously. If the bond ticks down 0.50 before you can offload your inventory, you lose \$0.50 per \$100 of face on \$30 million — \$150,000 — which wipes out six full days of the \$25,000 spread income from the example above.

That is **inventory risk**: the risk that the price moves against the position the market-maker is forced to hold while waiting to sell it on. It is the central tension of the whole business. The dealer *wants* to be flat — holding no net position, just collecting spread — but providing liquidity means absorbing whatever clients throw at it, which constantly pushes it off flat. The chart below shows the two forces pulling against each other across a single trading day: spread income accruing steadily upward, inventory marks swinging the running total around, and the net being the jagged sum.

![Market-maker daily P&L spread accruing and inventory swinging](/imgs/blogs/the-trading-book-market-making-flow-vs-prop-and-the-volcker-rule-3.png)

#### Worked example: inventory risk on a position the dealer was forced to hold

You are flat at the open. Through the morning, sellers hit your 99.90 bid and you accumulate a *long* position of \$30 million face value of the bond, bought at an average price of 99.90.

Now two scenarios for the afternoon:

**Scenario A — the price holds.** You manage to sell the \$30 million back out at an average of 100.10 (lifting your own ask to buyers). Your profit is the full spread: (100.10 − 99.90) / 100 × \$30,000,000 = 0.0020 × \$30,000,000 = **+\$60,000**. The business worked exactly as designed.

**Scenario B — the price falls.** Before you can sell, news hits and the bond drops to 99.40. You are still long \$30 million, now marked at 99.40 against your 99.90 cost. Your mark-to-market loss is (99.40 − 99.90) / 100 × \$30,000,000 = −0.0050 × \$30,000,000 = **−\$150,000**.

The asymmetry is the whole point. The *upside* of the position is a few basis points of spread — \$60,000 on a good day. The *downside* of being caught long into a 0.50 move is \$150,000, two and a half times the good-day profit. A market-maker who lets inventory build is no longer running a spread business; it is running a directional bet it never decided to make. The intuition: the spread is the wage, inventory is the risk, and a good desk is judged on how flat it stays, not how much it earns on any one position.

This is why real market-makers *hedge* their inventory the instant it builds — selling a related instrument, a futures contract, or an index to neutralize the directional exposure while they work the position off. Hedging is not prop trading; it is the opposite. It is the dealer trying to get back to flat. Keep that distinction in mind, because the Volcker rule turns on it.

## What sets the width of the spread

If the spread is the market-maker's wage, the natural question is: what decides how wide it is? Why is the spread on a German government bond under a basis point while the spread on a small-company stock or an exotic derivative can be 2% of the price? The answer is that the spread is not a markup the dealer picks at will — it is the price the market forces on the dealer to compensate for three specific costs, and understanding them tells you almost everything about why some markets are deep and liquid and others are treacherous.

The first cost is **order-processing cost** — the plain expense of running the desk: the technology, the exchange fees, the clearing and settlement, the salaries of the traders and the risk team behind them. This is small and roughly fixed, and on a high-volume instrument it amounts to a fraction of a basis point spread over millions of trades. It is the least interesting component.

The second cost is **inventory cost** — the very risk we just worked through. The dealer must be paid for the danger of being caught holding a position while the price moves. The more volatile the instrument, the more the price can swing while the dealer waits to offload inventory, and so the wider the spread must be to compensate. This is why spreads *widen automatically when volatility spikes*: a dealer quoting a tight spread in a calm market will pull those quotes apart the instant the market gets jumpy, because the inventory risk it is being asked to bear has just multiplied. You see this vividly in a crisis — spreads that were pennies wide blow out to dollars, not because dealers are greedy but because the cost of holding inventory has genuinely exploded.

The third cost is the subtle one, and it is the deepest idea in market-making: **adverse selection**, sometimes called the *information cost*. When someone trades with you, you have to worry about *why* they are trading. Most counterparties are *uninformed* — a pension fund rebalancing, a corporation hedging, a saver buying a bond — and trading with them is safe, because they have no special information about where the price is going. But some counterparties are *informed* — they know something you do not, and they are trading *because* of it. If a hedge fund that has done deep research on a company sells you its bond aggressively, there is a real chance it knows the bond is about to fall, and you have just bought something from someone who knows more than you. On average, the dealer loses money to informed traders and makes it back from uninformed ones — so the spread has to be wide enough that the profit from the uninformed crowd covers the losses to the informed few.

#### Worked example: how adverse selection forces the spread wider

You make markets in a stock at a mid of \$100. Suppose that out of every 100 people who trade with you, 90 are uninformed (they trade for reasons unrelated to the price direction) and 10 are informed (they trade only when they know the price is about to move by \$2 in their favor — meaning \$2 against you).

If you quoted a *zero* spread (buy and sell both at \$100), here is your expected result per 100 trades:

- From the 90 uninformed traders: you earn the spread, which is zero. Net: \$0.
- From the 10 informed traders: each one costs you \$2 because they only trade when the price is about to move against your position. Net: 10 × (−\$2) = **−\$20**.

A zero spread is a losing business — you bleed \$20 per 100 trades to the informed crowd. To break even, you need a spread wide enough that the edge you earn from all 100 traders covers the \$20 you lose to the informed 10. If you set a spread that earns you, say, \$0.25 of edge per trade from everyone who crosses it, then across 100 trades you collect 100 × \$0.25 = \$25, more than enough to cover the \$20 loss to the informed traders, leaving \$5 of profit. The intuition: the spread is not a fee for nothing — a meaningful chunk of it is the dealer's insurance premium against the traders who know more than it does. In a market where informed traders dominate, no spread is wide enough, and dealers simply stop quoting — which is exactly how a market "loses liquidity" in a panic.

This is why spreads are tight on instruments where information is *symmetric* and widely shared — major government bonds, large-cap stocks, currency pairs — and wide on instruments where someone might plausibly know something you don't: small companies, distressed debt, illiquid corporate bonds. The width of the spread is, in a real sense, a measure of how dangerous it is to trade with the person on the other side.

## How a dealer hedges its way back to flat

We have said repeatedly that a good market-maker hedges its inventory rather than holding a directional bet. It is worth seeing how that actually works, because hedging is the mechanical heart of the difference between market-making and prop trading, and it is the activity the Volcker rule had to carve out as explicitly permitted.

When a dealer is forced long \$30 million of a specific corporate bond, it does not have to wait passively for buyers. It can *neutralize the direction* by taking an offsetting position in a closely related, more liquid instrument. For a corporate bond, the dealer might sell short a Treasury futures contract (because the corporate bond's price moves largely with interest rates, and shorting the future cancels most of that exposure) and buy protection in a credit index (to offset the credit-spread risk). After hedging, the dealer is no longer betting on whether the bond rises or falls; it is left holding only the *residual* — the bond's price relative to its hedges — which is precisely the small, idiosyncratic piece it actually expects to earn the spread on.

#### Worked example: hedging an inventory position back toward flat

You are long \$30 million of a corporate bond, bought at 99.90, and you want to stay in business as a market-maker without betting on the direction of interest rates. The bond's price moves about \$0.80 for every \$1.00 the equivalent Treasury moves (its *hedge ratio*), so to neutralize the rate exposure you sell short \$24 million of Treasury futures (\$30 million × 0.80).

Now take the same bad-news afternoon from the earlier example, where the bond falls 0.50:

- **Unhedged:** you lose \$0.50 per \$100 on \$30 million = **−\$150,000**, as before.
- **Hedged:** suppose the broad rate move accounts for 0.35 of that 0.50 fall. Your short Treasury position *gains* roughly \$0.35 per \$100 on the \$24 million-equivalent hedge — about +\$84,000 — offsetting most of the loss. Your net loss collapses to roughly **−\$66,000**, and the part that remains is the bond's *credit-specific* move, which is the risk you are actually paid to take as a credit market-maker.

The intuition: hedging does not make money — it *removes the bet you didn't want* so that what's left is only the spread-earning service. A market-maker that hedges is keeping its inventory risk small and idiosyncratic; a "market-maker" that lets a huge unhedged directional position ride is not hedging at all — it is running a prop book, which is exactly the disguise the London Whale used. This is why the Volcker rule had to write a careful definition of permitted hedging: the activity is essential to real market-making, but the word is the perfect cover for the thing the rule was trying to ban.

## Agency flow vs principal risk: the same desk, two business models

Step back and notice that a markets division actually runs two quite different businesses under one roof, and they have opposite risk profiles.

In **agency flow**, the bank never owns anything. A client says "buy me 1,000 shares of company X," the bank routes that order to the exchange, the shares go straight into the client's account, and the bank earns a commission for the service. The bank's capital is never at risk because the bank never holds the position. This is often called *riskless principal* or *agency* trading, and its economics look like a toll booth: small fee, enormous volume, almost no risk.

In **principal risk** — which includes market-making — the bank takes the position onto its own book. A client says "I need to sell 1,000 shares right now," and rather than make the client wait for a buyer, the bank *buys the shares itself*, onto its own inventory, and works them off later. Now the bank owns the shares, and if the price falls before it can resell them, the bank eats the loss. The economics look like the shopkeeper's: it earns the spread for providing immediacy, but it carries inventory risk to do so.

![Agency flow passes through versus principal trading holds inventory](/imgs/blogs/the-trading-book-market-making-flow-vs-prop-and-the-volcker-rule-4.png)

#### Worked example: flow revenue vs prop revenue side by side

Let's compare what each business earns on the same \$100 million of trading in a quarter.

**Agency flow.** The desk routes \$100 million of client orders and charges a commission of, say, 5 bps (0.05%). Revenue = \$100,000,000 × 0.0005 = **\$50,000**, and the risk to the bank's capital is essentially zero — the bank held nothing. If the market crashes that quarter, the agency desk still earns its \$50,000, because it never owned anything.

**Market-making (principal flow).** The desk makes markets in the same names and turns over \$100 million of two-way volume at a 10 bp average captured spread. Gross spread = \$100,000,000 × 0.0010 = \$100,000 — twice the agency revenue. But it carried inventory along the way. Suppose its inventory positions cost it \$40,000 in adverse marks over the quarter. Net = \$100,000 − \$40,000 = **\$60,000**, with real risk to capital and a number that could just as easily have been *negative* in a bad quarter.

**Prop trading.** The desk takes \$100 million of the bank's *own* capital and bets on the direction of those same names. There is no spread and no commission — only the bet. If it is right and the market rises 3%, it makes \$3,000,000. If it is wrong and the market falls 3%, it loses \$3,000,000. The expected revenue is whatever edge the traders genuinely have; the *variance* is enormous, and every dollar of it is the bank's own.

The intuition: as you move from agency to market-making to prop, the revenue per dollar of volume rises and so does the risk to the bank's capital — and at the prop end, the "revenue" is just the outcome of a bet placed with depositors' money standing behind it. That escalation is precisely what regulators decided a deposit-taking bank should not be allowed to climb all the way up.

## The trading book as a ledger: mark-to-market and daily P&L

We said the trading book is marked to market every day. It is worth dwelling on what that actually does to a bank, because it is a double-edged sword.

On the banking book, a loan that is quietly going bad can sit at full value for quarters before the bank is forced to recognize the loss — the accounting lets it wait until default looks likely. That can hide problems, but it also smooths the bank's earnings. On the trading book, there is no hiding: every position is repriced at the closing market price each day, and the change flows straight into that day's reported profit. A desk that is up \$2 million on Monday and down \$5 million on Tuesday reports exactly that. The book tells the truth daily, which is good for transparency and brutal for stability.

This is why the trading book is the place a bank's losses can appear *fastest*. A loan book deteriorates over months; a trading book can print a nine-figure loss in a single session. When you read that a bank "took a trading loss" of some dramatic number in one quarter, you are reading mark-to-market doing its job. And because the loss hits capital immediately, the trading book is where the thin equity cushion at the heart of this whole series can be eaten in the time it takes for a market to gap.

#### Worked example: a daily mark-to-market move hitting the P&L

Your desk holds a \$200 million portfolio of bonds in the trading book, carried at yesterday's close of 100.00, so book value \$200 million.

Overnight, yields rise and the bonds reprice to 98.50. The portfolio is now worth \$200,000,000 × (98.50 / 100.00) = \$197,000,000. The mark-to-market loss is:

- \$200,000,000 − \$197,000,000 = **−\$3,000,000**, recognized *today*, straight into the bank's P&L.

There was no default, no missed coupon, nothing "wrong" with the bonds — only that the market price fell 1.5%. On a banking-book holding of the very same bonds classified as held-to-maturity, the bank might recognize *none* of that loss in its reported earnings, because it intends to hold them to par. Same bonds, same price move, opposite accounting — entirely because of which book they sit in. The intuition: the trading book is honest to a fault, and that honesty is exactly why a bank's most sudden losses tend to come from it.

That last point — same asset, very different treatment depending on the book — is not academic. It is the precise mechanism that destroyed Silicon Valley Bank in 2023, where the relevant distinction was between held-to-maturity and available-for-sale classification in the banking book, but the underlying lesson is the same: *where you put an asset determines when its losses become visible.* For the full anatomy of how a thin equity cushion gets eaten, the [bank capital and leverage](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) piece walks through the arithmetic.

## Value at Risk: drawing a line in the loss tail

Now we can give VaR the treatment it deserves, because it is the master control on the whole trading book.

A trading book's daily P&L is a *distribution*. Most days the desk makes or loses a little; rarely it makes or loses a lot. If you plotted every day's P&L for a year, you would get a roughly bell-shaped pile centered near zero, with thin tails stretching out to the big-loss and big-gain days. **Value at Risk picks a point in the loss tail and says: this is the worst we expect to do on all but the unluckiest days.**

The standard form is a *one-day 99% VaR*. To compute it, you look at the distribution of daily losses and find the loss level that you would only exceed 1% of the time — the 1st percentile. If that level is \$7 million, then your one-day 99% VaR is \$7 million: you expect to lose more than \$7 million on roughly 1 trading day in 100 (about 2-3 days a year), and less than that on the other 99%.

![VaR as a cut point in the loss distribution tail](/imgs/blogs/the-trading-book-market-making-flow-vs-prop-and-the-volcker-rule-5.png)

#### Worked example: computing a VaR number from volatility

Your desk's daily P&L has a *standard deviation* (a measure of typical day-to-day swing) of \$3 million, and we will assume — as the simplest models do — that daily P&L is roughly *normally distributed* (the familiar bell curve).

For a normal distribution, the 1st percentile sits about 2.33 standard deviations below the mean. So:

- One-day 99% VaR ≈ 2.33 × \$3,000,000 = **\$6.99 million**, call it **\$7 million**.

If the risk team instead wanted a 95% VaR (the loss exceeded 1 day in 20, a less extreme line), they would use 1.65 standard deviations: 1.65 × \$3,000,000 = **\$4.95 million**. And to convert a one-day VaR to a ten-day VaR — the horizon regulators often require, on the logic that it might take ten days to unwind a position in a stressed market — you scale by the square root of time: \$7 million × √10 ≈ **\$22 million**.

The intuition: VaR turns "this desk feels risky" into a single dollar number a risk manager can put a limit on. But notice every assumption we just made — normal distribution, a stable standard deviation, that the past predicts the future. Each of those is wrong in exactly the moments that matter most, which is why the next section is about how VaR lies. For the full statistics — why the normal assumption understates tail risk, what *expected shortfall* fixes, and how the 2008 crisis broke VaR models outright — see [value at risk and exactly how VaR lies](/blog/trading/risk-management/value-at-risk-and-exactly-how-var-lies).

## VaR limits: how the trading book is actually controlled

A VaR number on its own does nothing. What makes it bite is the *limit framework* built on top of it. Every desk in a markets division is given a VaR limit — a hard cap on the VaR it is allowed to run. The fixed-income desk might have a \$10 million daily VaR limit, the equities desk \$8 million, and the whole division an aggregate limit set by the board's risk appetite. A trader who wants to put on a bigger position must check it against the limit first; a position that would push the desk's VaR over the cap simply cannot be done without sign-off from risk management, and often not even then.

The limit is enforced by *daily monitoring*. Every morning, an independent risk team — not the traders themselves — recomputes each desk's VaR and compares it to the limit. A *breach* (VaR over the limit) triggers an escalation: the desk must cut the position to get back under the cap, usually within a day. The chart below shows the normal life of a desk's VaR against its limit — quiet for weeks, then a position builds, the VaR climbs through the limit, the breach is flagged, and the risk team forces the position down.

![Desk daily VaR climbing through its limit then being cut](/imgs/blogs/the-trading-book-market-making-flow-vs-prop-and-the-volcker-rule-7.png)

#### Worked example: a desk approaching and breaching its VaR limit

Your fixed-income desk has a daily 99% VaR limit of \$10 million. It runs comfortably for weeks around \$5-6 million. Then a trader builds a large position in long-dated bonds, and the desk's VaR climbs:

- Week 8: VaR = \$9 million — under the limit, but the risk team flags it as "approaching."
- Week 9: VaR = \$11 million — **breach.** It is \$1 million over the \$10 million cap. Escalation: the trader is told to reduce.
- Week 10: the trader, convinced the position is about to pay off, *adds* instead. VaR = \$13 million — a worse breach. Now it goes to the head of risk and the desk is *forced* to cut, regardless of the trader's conviction.

What should have happened in week 9 is that the position came down. What actually happened at JPMorgan's CIO in 2012 — the London Whale — is that the desk's VaR *model was changed* so the position looked like it fit under the limit, turning a real breach into an invisible one. The intuition: a VaR limit only protects the bank if it is computed by people who do not profit from the position and enforced by people who can overrule the trader. The number is only as good as the independence of the person who owns it. The deeper problem of limits being gamed is its own subject; [how risk limits get gamed](/blog/trading/risk-management/risk-limits-and-how-they-get-gamed) catalogs the tricks.

There is also a *back-testing* discipline attached to VaR. If a desk's one-day 99% VaR is honest, it should be exceeded by an actual daily loss about 1% of the time — roughly two or three days a year. Regulators count these *exceptions*. If a desk blows through its VaR far more often than the model predicts, the model is understating risk, and the bank is penalized with a higher capital multiplier. Back-testing is the reality check that keeps a desk from setting its VaR artificially low to free up risk-taking room.

## The Volcker rule: banning the bet, keeping the service

Now we arrive at the rule that reshaped this entire business. To understand it, you have to understand *why* anyone cared whether a bank traded for clients or for itself.

Before 2008, the biggest US banks ran enormous proprietary trading desks — teams whose entire job was to bet the bank's own capital on markets, like an in-house hedge fund. When the crisis hit, those bets blew up alongside everything else, and taxpayers found themselves backstopping banks whose losses came partly from speculative trading that had nothing to do with serving customers. The political objection was sharp and simple: *a bank that takes government-insured deposits and can borrow from the central bank in a crisis should not be using that cheap, subsidized money to gamble in the markets.* If the bet wins, the bank keeps the profit; if it loses, the public safety net catches the fall. That is a one-way bet against the taxpayer.

The **Volcker rule** — named for former Federal Reserve chairman Paul Volcker, who proposed it, and enacted as part of the 2010 Dodd-Frank Act with compliance phased in through 2014-2015 — was the answer. Its core is a single prohibition: a banking entity that takes insured deposits may not engage in *proprietary trading*. It also sharply restricted such banks from owning or sponsoring hedge funds and private equity funds. The principle is to wall the public safety net off from speculation.

But — and this is the subtle, essential part — the rule did *not* ban market-making. It could not, because market-making is a genuine service the economy needs: someone has to provide liquidity so that pension funds, corporations, and other banks can trade when they need to. So the Volcker rule had to thread a needle: ban the bet, keep the service. The pipeline below shows how it sorts a trade by *purpose*.

![How the Volcker rule sorts a trade by purpose permitted or banned](/imgs/blogs/the-trading-book-market-making-flow-vs-prop-and-the-volcker-rule-8.png)

The test the rule uses is *intent*, and intent is hard to prove, which is what makes the rule so complicated in practice. A trader holding a bond could claim it is *market-making inventory* (permitted) or be *betting it will rise* (banned), and the position looks identical either way. To draw the line, the rule introduced a standard with the unlovely name **RENTD** — "the Reasonably Expected Near-Term Demand of clients." A market-making desk is permitted to hold inventory, but only an amount reasonably justified by the demand it expects from its customers. Hold more than your clients could plausibly want, and you are no longer making markets; you are running a prop book in disguise.

The rule also explicitly permits **hedging** — trading to *reduce* the bank's own risk rather than to take new risk. If a bank holds a portfolio of loans and buys a credit derivative to offset some of that risk, that is permitted, because it shrinks risk rather than placing a bet. But "hedging" became the favorite hiding place for prohibited trading, because almost any position can be dressed up as a hedge of *something*. The London Whale position was, on paper, a "portfolio hedge." In reality it had grown into a directional bet many times larger than anything it was supposedly hedging — which is exactly why the episode became Exhibit A in the argument that the hedging exemption was being abused.

#### Worked example: the same trade, permitted or banned

A trader at a deposit-taking bank buys \$500 million of a corporate bond. Is it allowed under Volcker? It depends entirely on the *purpose*, and the same \$500 million position falls on different sides of the line:

- **As market-making:** the desk regularly quotes this bond for clients and expects, based on recent client flow, to sell most of the \$500 million to customers within days. The inventory is justified by *reasonably expected near-term demand*. **Permitted.**
- **As a hedge:** the bank holds \$500 million of loans to the same issuer, and the bond position is structured to offset the credit risk on those loans. It *reduces* the bank's net exposure. **Permitted.**
- **As prop:** the trader has no client demand for anything like \$500 million and no offsetting exposure to hedge — he simply believes the bond will rally and is betting the bank's capital on it. **Banned.**

The intuition: under Volcker, the legality of a trade is not in the trade itself but in the *reason* for it — which is why compliance with the rule is less about blocking positions and more about documenting intent. Banks now staff entire teams to evidence that their inventory ties to client demand, precisely because the position alone cannot tell a regulator which side of the line it is on.

## Where market-making, flow, and prop actually sit

It helps to see the three businesses laid side by side on the dimensions that matter: who the desk trades for, who carries the risk, what it earns, and whether a deposit bank is allowed to do it at all.

![Market-making flow and prop compared on risk earnings and Volcker](/imgs/blogs/the-trading-book-market-making-flow-vs-prop-and-the-volcker-rule-6.png)

Notice what the table makes obvious. Agency flow is the safest business — no bank risk, a clean fee — and is fully permitted. Market-making sits in the middle — real inventory risk, but justified by a service clients need — and is permitted *as long as it stays a service*. Prop trading is pure risk-taking with the bank's own capital and is banned at deposit banks. The Volcker rule did not abolish prop trading; it pushed it *out* of deposit-taking banks and into entities that do not enjoy the public safety net — hedge funds, proprietary trading firms, and the trading arms of non-bank institutions. The risk did not vanish; it moved to a place where, if it blows up, depositors and taxpayers are not on the hook.

This connects directly to the broader question of how an investment bank earns its keep across all its divisions. Market-making and flow are the engine of the *sales and trading* business; for how that sits alongside the advisory, underwriting, and other fee businesses, [inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) zooms out to the whole firm.

## Common misconceptions

**"Market-making is gambling with depositors' money."** No — well-run market-making is close to the opposite. A market-maker *wants* to be flat, earning the spread for providing liquidity, not betting on direction. It carries inventory risk as a byproduct of the service and hedges to minimize it. The gambling is *prop trading*, which is precisely the thing the Volcker rule removed from deposit banks. Conflating the two misses the entire point of the rule: it kept the service and banned the bet.

**"The bid-ask spread is the bank ripping you off."** The spread is the price of *immediacy* — the ability to trade right now without waiting. If you are willing to wait and post your own order, you can often trade inside the spread. The dealer earns the spread for taking the other side instantly and bearing the inventory risk until it can offload the position. On the most liquid instruments that spread is under a basis point; it is wide only where liquidity is genuinely scarce, which is exactly where immediacy is most valuable.

**"VaR tells you the most you can lose."** This is the dangerous one. VaR tells you the loss you would *exceed* only 1% (or 5%) of the time — it says nothing about *how bad* the other 1% gets. A one-day 99% VaR of \$7 million is fully consistent with a \$70 million loss on the worst day; VaR caps the frequency of a big loss, not its size. Treating VaR as a worst case is how desks get blindsided by the tail. The proper fix is *expected shortfall*, which averages the losses *beyond* the VaR cut — covered in [CVaR, expected shortfall, and asking how bad is bad](/blog/trading/risk-management/cvar-expected-shortfall-and-asking-how-bad-is-bad).

**"The Volcker rule ended prop trading."** It ended prop trading *at deposit-taking banks*. The activity itself moved to non-bank firms — standalone proprietary trading shops, hedge funds, and market-makers that do not take insured deposits. The total amount of speculation in the market did not fall; its *location* changed, to entities outside the safety net. Whether that made the system safer or simply moved the risk somewhere less visible is a live debate.

**"A trading loss means someone did something wrong."** Often it just means the market moved. The trading book is marked to market daily, so a price decline becomes a reported loss instantly, with no default and no error. A desk can do everything right — quote tight, hedge its inventory, stay within limits — and still print a loss on a day the market gaps. The error is not *having* a loss; it is having a loss *bigger than the limits were supposed to allow*, which points to a control failure, not a trading one.

## How it shows up in real banks

**The London Whale, JPMorgan, 2012.** The cleanest case study of every theme in this post. JPMorgan's Chief Investment Office in London held a vast credit-derivatives position that began as a "portfolio hedge" and metastasized into a one-way directional bet — prop trading wearing a hedge's clothing. As losses mounted, the desk's VaR model was changed to a new methodology that roughly *halved* the reported VaR, making a position that had breached its limit appear to fit within it. When the position finally blew up, the bank disclosed more than \$6 billion in losses. The episode validated the Volcker rule's central worry — that the hedging and market-making exemptions could be stretched to cover speculation — and it showed that a VaR limit is worthless if the people running the desk can quietly rewrite the model that computes it.

**The Volcker rule takes effect, 2014-2015.** When compliance kicked in, the largest US banks wound down or spun off their dedicated prop desks. Goldman Sachs, Morgan Stanley, and others closed in-house proprietary trading units, and a wave of traders left to join or start hedge funds. The visible effect was exactly as designed: speculation with the bank's own capital left the deposit-taking institutions. The contested effect was on *liquidity* — some argued that by constraining how much inventory market-makers could hold, the rule made markets thinner and more prone to sharp moves when everyone needed to trade at once. That trade-off between safety and liquidity is the rule's enduring tension.

**Market-making in a crisis: March 2020.** When the pandemic hit, even the world's deepest market — US Treasuries — briefly seized up. Sellers flooded in, and dealers, constrained by both regulation and their own risk limits, could not or would not expand their inventory fast enough to absorb the flow. Bid-ask spreads blew out, prices gapped, and the Federal Reserve had to step in as buyer of last resort. The lesson is that market-making is *not* an unlimited backstop: a dealer provides liquidity until its inventory risk and capital constraints say stop, and in a true panic that limit is reached quickly. The very inventory risk we worked through earlier is what caps how much liquidity a market-maker can supply on the worst day — exactly when it is needed most.

**The 2008 prop-trading losses.** In the years before the crisis, large banks ran prop books that took on mortgage and credit risk indistinguishable from outright bets. When those markets collapsed, the losses landed on banks that also held insured deposits and ultimately needed public support. This is the historical episode the Volcker rule was written to prevent from recurring — the moment when "the bank's trading desk lost money" became "the taxpayer is backstopping the bank's trading desk." Everything about the rule's design traces back to severing that link.

**Knight Capital, 2012 — when the machinery breaks.** Not every trading-book disaster is a bet gone wrong; some are the plumbing failing. Knight Capital, a major US market-maker, deployed faulty trading software one August morning in 2012 that began firing off unintended orders into the market at a rate of thousands per second. In about 45 minutes the firm accumulated a giant unwanted inventory of stocks it never meant to buy, and when it unwound the position it booked a loss of roughly \$440 million — more than the firm was worth — and had to be rescued days later. The lesson sits alongside the others: a market-maker's defining risk is the inventory it ends up holding, and that inventory can build not only from a one-way market or a rogue bet but from a single broken line of code. The control framework around a trading book has to be as much about operational reliability as about VaR limits, because the book can fill up with risk faster than any human can react.

## The takeaway: how to read a bank's markets division

Once you understand the trading book, you can read a bank's markets division the way an analyst does, and you can spot the failure mode before it becomes a headline.

Start by separating the *service* from the *bet*. A healthy markets division earns most of its money from market-making and flow — patient, repeatable spread and commission income that grows with client volume and barely flinches when the market falls, because the desk is mostly flat. That income is high-quality: it is a *fee for a service*, not a wager. When you read a bank's results, the markets revenue you want to see is the kind that shows up reliably quarter after quarter, scaling with activity rather than with the bank's directional luck.

The warning sign is the opposite: markets revenue that is *lumpy* — huge in some quarters, negative in others — because that is the signature of directional risk-taking, of a book that is making money by being right about the market rather than by serving it. That is also where the thin equity cushion at the heart of this whole series is most exposed, because the trading book is the one place a bank can lose a year's profit in an afternoon. The banking book deteriorates over quarters; the trading book can gap in a single session, and mark-to-market accounting hits capital that same day.

This is why the entire apparatus exists — the separate ledger, the daily marks, the VaR limits computed by an independent risk team, the back-testing of exceptions, the Volcker prohibition on prop trading. None of it is bureaucratic box-ticking. Every piece is there to keep the markets division a *spread business* — a shopkeeper earning a markup for keeping the shelves stocked — and to stop it from quietly becoming a *betting business* with depositors' money behind the chips. When that machinery fails, as it did with the London Whale, the result is not a rounding error; it is billions of dollars and a hard lesson in why the line between flow and prop is drawn exactly where it is.

The single sentence to carry away: **market-making earns a small, honest margin for providing liquidity and carries inventory risk as the cost of doing so; the moment a desk holds more risk than its clients' demand can justify, it has crossed from a service into a bet — and at a deposit-taking bank, that crossing is exactly what the rules are built to catch.**

## Further reading & cross-links

- [Bank capital and leverage: why equity is the thin cushion](/blog/trading/banking/bank-capital-and-leverage-why-equity-is-the-thin-cushion) — the arithmetic of how a trading loss eats the thin equity layer that stands between a bank and insolvency.
- [Inside an investment bank: how they make money](/blog/trading/finance/inside-an-investment-bank-how-they-make-money) — how sales and trading sits alongside advisory, underwriting, and the other fee businesses across the whole firm.
- [Value at Risk and exactly how VaR lies](/blog/trading/risk-management/value-at-risk-and-exactly-how-var-lies) — the full statistics behind VaR, the normal-distribution trap, and why the tail is worse than the model says.
- [CVaR, expected shortfall, and asking how bad is bad](/blog/trading/risk-management/cvar-expected-shortfall-and-asking-how-bad-is-bad) — the fix for VaR's blind spot: measuring the average loss *beyond* the cut point.
- [Risk limits and how they get gamed](/blog/trading/risk-management/risk-limits-and-how-they-get-gamed) — the practical ways a desk works around the limits that are supposed to contain it, the London Whale among them.

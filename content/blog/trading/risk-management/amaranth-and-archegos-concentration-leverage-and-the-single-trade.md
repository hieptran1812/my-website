---
title: "Amaranth and Archegos: Concentration, Leverage, and the Single Trade That Ends You"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "How two famous funds proved the same lesson — that one concentrated, levered bet, sized past what any margin of safety could absorb, can erase everything in a single week."
tags: ["risk-management", "concentration", "leverage", "case-study", "amaranth", "archegos", "position-limits", "counterparty-risk", "blow-up"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **One concentrated, levered bet — sized past what any margin of safety could absorb — is how the largest blow-ups happen, and the same fix stops all of them: cap the single trade.**
> - Amaranth Advisors lost about **\$6.6 billion in a single week** in September 2006 on concentrated natural-gas calendar spreads in a book so illiquid it could not be sold into a bid.
> - Archegos Capital lost its entire **~\$10 billion** of equity in March 2021 on a few single stocks held through total-return swaps at roughly **5x leverage**, and dealt prime brokers **over \$10 billion** of losses (Credit Suisse alone ~\$5.5 billion).
> - The arithmetic is brutal and simple: a position's hit to your capital equals **leverage × position weight × the adverse move**. At 5x leverage, a name at 60% of the book, down 35%, erases **105% of your equity** — the book is gone, and then some.
> - Archegos worked because **no single prime broker could see the total** — each saw only its own slice, each was within its own limit, and the aggregate was several times what anyone would have allowed.
> - The survival fix is a **hard max-loss budget per name and per theme**: on a \$10,000,000 book, a 2% single-name budget turns the same 35% gap from a wipeout into a survivable \$200,000 dent. You only compound if you are still in the game.

There is a particular kind of fund death that does not look like a slow bleed. It is not a strategy quietly decaying, or a year of mediocre returns, or a manager losing his touch. It is sudden. On a Friday the fund is one of the most successful in its category, the founder is rich, the marketing deck is full of green. By the following Friday the fund is being wound down and the founder's name is a verb for what not to do. Nothing changed about the strategy in between. What happened is that one position — the one position the entire book had quietly become — moved against them, and there was no margin of safety left to absorb it.

Two funds, fifteen years and two asset classes apart, tell exactly this story. **Amaranth Advisors** was a multi-strategy hedge fund that, by 2006, had become a giant natural-gas trading desk wearing a multi-strategy costume. **Archegos Capital Management** was a family office that, by 2021, had become a leveraged bet on a handful of media and tech stocks held through derivatives that hid its true size from the very banks financing it. Amaranth lost roughly \$6.6 billion in a week. Archegos lost everything it had and stuck its lenders with more than \$10 billion of losses. Different markets, different instruments, different decades — and the same three ingredients in the same order: **concentration, then leverage, then a single adverse move**. Figure 1 lays the two blow-ups side by side so you can see they are the same picture drawn twice.

![Two blowups, one lesson: Amaranth bet the fund on natural-gas calendar spreads and Archegos on a handful of swap-financed stocks, both held one giant levered position no margin of safety could survive, both gone in days](/imgs/blogs/amaranth-and-archegos-concentration-leverage-and-the-single-trade-1.png)

This post is a case study, but the point is not to gawk at two famous failures. The point is to make the abstract risk-management ideas you have met elsewhere in this series — concentration and position limits, marginal and component risk, leverage, counterparty exposure — *concrete*, by watching what happens when each one is ignored. Amaranth is what concentration plus illiquidity does to you. Archegos is what leverage plus hidden counterparty exposure does to you. Both are what happens when a single trade is allowed to grow large enough to vote on whether your firm survives. The survival thesis of this whole series — *you can only compound if you are still in the game, and ruin is absorbing* — is not a slogan. It is the exact thing that both of these funds violated, and it is the thing a few hard limits would have enforced.

## Foundations: the building blocks of a single-trade wipeout

Before we walk through the two blow-ups, let us define every term from zero, because the whole story is built from four ideas that each sound mundane and become lethal in combination.

**Concentration** is how much of your capital depends on a single bet. If you have a \$10,000,000 book and one position is 5% of it, that position is \$500,000, and the worst it can do to you — even if it goes to zero — is a 5% hit. If instead one position is 60% of the book, that single name now decides most of your fate. Concentration is not inherently bad: a high-conviction investor with a real edge *wants* to put more weight on his best ideas. But concentration is the dial that decides how much a single surprise can hurt you, and surprises are the one thing markets reliably deliver. The sibling post on [concentration and position limits](/blog/trading/risk-management/concentration-and-position-limits-the-one-trade-that-can-end-you) develops this dial in detail; here we watch two funds turn it to the maximum.

**Leverage** is using borrowed money — or borrowed exposure — to control a position larger than your capital. If you have \$10,000,000 of equity and you control \$50,000,000 of exposure, you are running 5x leverage. Leverage multiplies *both* directions: a 10% gain on \$50,000,000 of exposure is \$5,000,000, which is a 50% gain on your \$10,000,000 of equity. The same 10% the other way is a 50% loss. Leverage does not change your edge; it changes the size of every outcome relative to the capital that has to survive the bad ones. The [arithmetic of ruin under leverage](/blog/trading/risk-management/leverage-and-the-arithmetic-of-ruin) is the dedicated treatment; for our purposes the one fact that matters is: **leverage L turns a move of x in your position into a move of L × x in your equity.**

**Liquidity** is whether you can actually get out at something near the quoted price. A liquid market lets you sell a large position quickly without moving the price much. An illiquid one does not: if you are the largest holder, *you are the market*, and trying to sell pushes the price against you precisely when you most need it to hold still. Liquidity is invisible in calm times — every position looks sellable when nobody is selling — and it vanishes exactly when everyone wants out at once. Amaranth's gas spreads were a textbook case: the fund's position was so large relative to the market that there was no one on the other side to sell to without crushing the price.

**Counterparty exposure** is the risk that runs through whoever you trade *with* or *through* — the prime broker that finances your positions, the bank that writes your swaps, the clearinghouse that stands between you and the market. When you hold exposure through a derivative rather than by owning the underlying outright, your counterparty is carrying the position on its own books and lending you the economics. That has two consequences: the counterparty can call you for more collateral (margin) when the position moves against you, and — crucially for Archegos — *the counterparty only sees the slice of your book that it is financing.* If you spread one giant bet across five banks, no single bank knows how big the whole thing is.

Now the combination. Take a concentrated position. Add leverage, so the position's notional is several times your capital. Make it illiquid or hold it through counterparties who can pull funding. Then let a single adverse move arrive. The hit to your equity is:

**equity hit = leverage × position weight × adverse move.**

Each of the three factors on the right is something a trader controls *before* the move arrives, and the move itself is the one thing nobody controls. That is the entire lesson of both case studies in one line. When all three factors are large at once, you do not need a once-in-a-century event to be wiped out — a perfectly ordinary 20–35% move in a single name, multiplied by 5x leverage and a 60% weight, is enough to take you through zero. The rest of this post is that equation, worked four ways with real dollars and two real funds.

#### Worked example: the equity-hit equation on a \$10,000,000 book

You run a **\$10,000,000 book**. You hold one name at a weight of 60% of the book, financed at 5x leverage so the notional exposure on this single name is large relative to your capital. The name gaps down 35% overnight — a move that is severe but, for a single stock around an earnings miss or a fraud allegation, entirely within the realm of an ordinary bad week.

- Position weight: **60%** of the book.
- Leverage: **5x**.
- Adverse move: **−35%**.
- Equity hit = leverage × weight × move = 5 × 0.60 × 0.35 = **1.05 = 105% of equity**.

A 105% hit means the position's loss is larger than your entire capital. Your \$10,000,000 is gone — all of it — and you are \$500,000 in the hole to your lenders on top. The same name, the same 35% gap, would have cost an *unlevered, 5%-weight* book just 1 × 0.05 × 0.35 = 1.75% — a \$175,000 dent on \$10,000,000, a Tuesday. The move was identical. What differed was the two dials the trader set in advance.

*The move that ruins you is rarely extraordinary; what is extraordinary is the leverage and concentration you stacked on top of an ordinary move.*

## Amaranth: concentration and illiquidity in a natural-gas book

Amaranth Advisors began life as a diversified, multi-strategy hedge fund — convertible arbitrage, merger arbitrage, statistical strategies, and energy trading all under one roof. The idea of a multi-strategy fund is that the strategies are uncorrelated, so a bad month in one is offset by a good month in another, and the whole thing is steadier than any part. That is genuine diversification, and it works — until one of the strategies grows so large that it stops being one strategy among many and becomes the whole fund.

By 2006, that is exactly what had happened. Amaranth's energy book, run by a star trader, had been extraordinarily profitable, and success bred size. The fund kept allocating more capital to the trade that was working, until the natural-gas position was not a sleeve of a diversified fund — it was the fund, with some other strategies attached for decoration. According to the CFTC and a 2007 U.S. Senate Permanent Subcommittee on Investigations report, the fund lost roughly **\$6.6 billion**, the bulk of it in a single week in September 2006, on concentrated **natural-gas calendar spreads** (`dr.CRISES["amaranth_2006"]`). Figure 2 puts the scale on one chart: the loss in that one week was on the order of the *entire fund's capital.*

![Amaranth 2006 shown as horizontal bars, with the concentrated natural-gas position notional far larger than the fund capital, and the roughly 6.6 billion dollar loss in a single week equal to most of the fund's capital](/imgs/blogs/amaranth-and-archegos-concentration-leverage-and-the-single-trade-2.png)

### What a calendar spread is, and why this one was a trap

A **calendar spread** in natural gas is a bet on the *difference* in price between two delivery months — for example, buying the March contract and selling the April contract. Natural gas has a strong seasonal pattern: March is the tail of winter heating demand, April is the start of the shoulder season when demand collapses and storage refills. The March–April spread is a famous expression of a view on winter weather and storage: a cold, tight winter widens it; a mild winter or full storage collapses it. It is a legitimate trade with real economic logic.

The problem was not the trade's logic. The problem was three of our four building blocks at once:

- **Concentration.** The position was enormous relative to the fund and relative to the market. Amaranth at times held a dominant share of the open interest in the relevant natural-gas contracts. When one trader's book is a large fraction of an entire futures market, that book is no longer a participant in the market — it *is* the market for that spread.
- **Leverage.** Futures are inherently levered: you post a margin deposit that is a small fraction of the contract's notional value. A commodity-spread book can run notional exposure many times its capital, which is why a spread move that looks small in percentage-of-notional terms can be enormous in percentage-of-capital terms.
- **Illiquidity.** This is the factor that turned a bad bet into a catastrophe. Because the position was so large, there was no one on the other side big enough to take it off Amaranth's hands at anything near the screen price. When the spread moved against them, the fund could not exit. Every attempt to sell pushed the spread further against the remaining position. The fund was trapped inside its own size.

When the March–April spread collapsed in September 2006 — winter risk receded, storage looked comfortable, the trade's thesis evaporated — Amaranth was holding a position it could neither defend nor escape. The book bled the bulk of \$6.6 billion in days. The energy positions were ultimately transferred, at a steep discount, to JPMorgan and the hedge fund Citadel, who had the balance sheet to warehouse them. Amaranth itself was wound down. A fund that had been celebrated as a multi-strategy powerhouse was, in the end, a single concentrated, levered, illiquid bet — and the single bet was wrong.

#### Worked example: how a "small" spread move erases a concentrated futures book

Suppose a fund runs a **\$10,000,000** energy book but, because futures are levered, controls a notional exposure of **\$120,000,000** in a single calendar spread — that is **12x** leverage on the spread's notional. Spreads are quoted in dollars per unit and move in what look like tiny increments, so let us say the spread moves against the position by an amount equal to **8% of the notional value of the legs**.

- Notional exposure: **\$120,000,000**.
- Adverse move: **8%** of notional.
- Dollar loss = 0.08 × \$120,000,000 = **\$9,600,000**.
- As a fraction of the \$10,000,000 capital: \$9,600,000 / \$10,000,000 = **96% of the book**.

An 8% move in the *notional* — which a spread trader might have called a routine bad week — is a 96% loss of *capital*, because the notional was 12x the capital. And that assumes you can actually get out at that 8% level. In an illiquid book you cannot: the act of selling pushes the spread further, so the realized loss overshoots. That overshoot is how an 8%-of-notional move becomes a more-than-100% loss of capital and a fund that no longer exists.

*Leverage does not just magnify your return; it converts the language of "a small move" into the language of "the whole fund," and illiquidity makes the realized loss worse than the screen ever showed.*

### Why a winning strategy grows into a concentration trap

It is tempting to read Amaranth as a story of a reckless trader, but the more uncomfortable truth is that the concentration was the *natural result of success*, and that is what makes it a pattern worth understanding rather than a one-off villain. A strategy that is working attracts capital — from the fund's own profits compounding, from the manager's conviction rising as the thesis keeps paying, and from the institutional incentive to put more behind a winner. Each of those forces is individually reasonable. Together they ratchet a position larger and larger until, almost without anyone deciding it, the trade that was one good idea among many has become *the firm*. There is rarely a single meeting where someone says "let us bet the fund on natural gas." There is a series of small, locally sensible decisions to add to the winner, and the concentration accretes silently in the gap between them.

This is precisely why a hard limit has to be a *number*, set in advance and enforced mechanically, rather than a judgment call made in the moment. Judgment in the moment is exactly what fails here, because in the moment the trade is working, the conviction is high, and every reason to trim looks like leaving money on the table. The trader who built Amaranth's gas book was not stupid; he was *winning*, and winning is the most dangerous state for a concentrated position, because it removes every emotional and institutional brake at the exact moment the size is becoming lethal. A pre-committed cap does not require the trader to be wise during the euphoria. It requires the firm to have been wise once, beforehand, when it set the number — and then to refuse to move it.

There is a liquidity twist specific to *being the market*. When your position is a dominant share of the open interest, the size that makes you powerful on the way up is the size that traps you on the way down. On the way up, controlling the market lets you push the spread in your favor; on the way down, there is no one of comparable size to sell to, so the same dominance that amplified your gains amplifies the price impact of your exit. The position's liquidity is not a fixed property of the market — it is a function of *your own size relative to it*, and it evaporates exactly in proportion to how concentrated you have become. A 5%-of-market position can be sold in a day; a 40%-of-market position cannot be sold at all without being the thing that crashes the price. Amaranth's risk was never visible in the daily volatility of the spread. It was hidden in the answer to a question nobody on the desk wanted to ask: *if this goes wrong, who exactly do we sell to?*

### The aftermath: what the post-mortems concluded

The 2007 Senate Permanent Subcommittee on Investigations report on Amaranth did not conclude that the fund was the victim of an unforecastable market. It concluded that the fund had accumulated positions so large that they distorted the natural-gas market itself, and that the concentration and the resulting illiquidity were the proximate causes of the collapse — a position that could not be exited because its own size had eliminated the other side. The regulatory response focused on position limits and the transparency of large traders, which is the institutional version of the single rule this post keeps returning to: *no participant should be allowed to grow a single position to a size that makes it both systemically dangerous and impossible to unwind.* The market-structure fix and the fund-level fix are the same idea at different scales — cap the single trade.

*A concentrated position is most dangerous not when it is losing but when it is winning, because success is what grows it past the point of no return, and a number set in advance is the only brake that still works once the euphoria has disabled judgment.*

## Archegos: leverage and the counterparty who could not see the whole

Archegos Capital Management was a family office — a private investment vehicle managing the personal fortune of one man, Bill Hwang. It managed no outside client money, which mattered, because it meant lighter regulatory disclosure than a public hedge fund. By early 2021 it had turned a few billion dollars of equity into tens of billions of dollars of single-stock exposure, concentrated in a small number of media and technology names. In March 2021 it collapsed in days, taking with it an entire family fortune and inflicting more than **\$10 billion** of losses on the global banks that had financed it — Credit Suisse alone reported roughly **\$5.5 billion** (`dr.CRISES["archegos_2021"]`).

The mechanism was different from Amaranth's in its plumbing but identical in its skeleton: a concentrated bet, multiplied by leverage, hit by a single adverse move, with no margin of safety to absorb it. The leverage and the hidden size both ran through one instrument: the **total-return swap.**

### What a total-return swap is, and why it hid the size

When you want exposure to a stock, the obvious thing is to buy the stock. You pay cash, you own shares, your name shows up on the ownership records, and the whole world can see your position. A **total-return swap** lets you get the *economics* of owning the stock without owning it. You sign a contract with a bank: the bank buys (and holds) the actual shares, and you agree to pay the bank a financing fee and to exchange the stock's total return — if the stock goes up you receive the gain, if it goes down you pay the loss. You post collateral (margin) against the position. The result is that you have all the upside and downside of owning the stock, at a fraction of the cash outlay, and the shares are on the *bank's* books, not yours.

Two things follow, and both were central to the blow-up:

- **Leverage.** Because you only post a margin fraction rather than the full purchase price, a swap is leverage by construction. Archegos ran roughly **5x or more** exposure relative to its equity through these swaps (`dr.CRISES["archegos_2021"]`). Figure 3 shows the leverage stack on the left: a few billion of equity controlling tens of billions of single-stock exposure.
- **Invisibility.** Because the bank holds the shares, *Archegos's* name does not appear on public ownership disclosures the way a direct buyer's would. And because Archegos used *several* prime brokers, each bank saw only the swaps it had written — its own slice — and none saw the aggregate. Figure 3's right panel makes this concrete: every broker's view was a fraction of the true total exposure, and the true total was a line none of them could see.

![Archegos 2021 in two panels, left showing equity controlling roughly five times its size in swap-financed exposure, right showing each prime broker seeing only its own slice while the true aggregate was far larger than any single view](/imgs/blogs/amaranth-and-archegos-concentration-leverage-and-the-single-trade-3.png)

### The collapse: a margin call no margin of safety could meet

In late March 2021, one of Archegos's largest holdings (and then several) gapped down sharply — a stock issuance and a series of declines in the concentrated names. With ~5x leverage, a drawdown in the names that would have been merely painful for an unlevered holder was fatal for Archegos: the loss, multiplied by leverage, exceeded the collateral it had posted. The prime brokers issued margin calls — demands for more collateral to cover the losses. Archegos could not meet them.

What happens next is the part that distinguishes a private loss from a systemic event. When a leveraged client defaults on a margin call, the prime broker seizes the collateral and *liquidates the position* — it sells the underlying shares it has been holding to recover what it is owed. But every broker was holding the *same* concentrated names, and every broker tried to sell on the *same* days. The selling pushed the prices down further, which deepened the losses, which triggered more selling. The banks that moved fastest (notably Goldman Sachs and Morgan Stanley) got out with smaller losses; the banks that hesitated (Credit Suisse, Nomura) wore enormous ones. Figure 4 shows the shape every one of these collapses shares: a long grind upward while the concentrated bet works, then a cliff — equity falling through zero in a handful of days once the bet turns.

![Equity curve of a concentrated levered book grinding upward for months then falling off a cliff to zero in days, with an unlevered version surviving the same drawdown and the levered version wiped out overnight](/imgs/blogs/amaranth-and-archegos-concentration-leverage-and-the-single-trade-4.png)

This is the asymmetry of losses in its most violent form. The fund spent years grinding up and a week going to zero, and zero is *absorbing* — there is no recovery from it, no matter how good the original thesis was, because there is no capital left to express the thesis with. A −20% drawdown in the names needs a +25% recovery to undo; a −100% wipeout of the fund needs an infinite recovery, which is to say it needs a recovery that cannot happen.

It is worth being precise about *why* the collapse accelerates rather than stops once it starts, because this is the feature that turns a large loss into a total one. When a leveraged client cannot meet a margin call, the broker does not politely wait — it sells the collateral immediately, because the broker's own capital is now at risk. But the broker is selling into a market where the other brokers holding the same names are doing the same thing on the same day. Each block of forced selling pushes the price down, which deepens the losses on every remaining position, which triggers the next margin call and the next forced sale. The mechanism feeds on itself. This is a deleveraging cascade, the same fire-sale dynamic that recurs in every crisis: the act of reducing risk *increases* the loss for everyone still holding, because everyone is reducing the same risk through the same narrow exit at once. A position that could have been unwound calmly over weeks, if anyone had known to do it, instead got liquidated over days at prices far below where it started — and the fund, sitting at the bottom of the leverage stack, absorbed the entire gap. The lesson nests inside the leverage lesson: leverage not only magnifies the loss, it removes your control over *when* you exit, because past a margin call the decision belongs to your lender, not to you.

There is a second-order point hiding in who survived. Goldman Sachs and Morgan Stanley, which moved fastest to liquidate, took comparatively small losses; Credit Suisse and Nomura, which hesitated, took enormous ones. The brokers were playing the same exit game against each other that traders play in a crowded trade — the first one out gets a price, the last one out gets whatever is left. So even among the *lenders*, the structure rewarded speed and punished hesitation, which means the cascade was not an accident of one bank's incompetence but the predictable Nash outcome of many lenders facing the same forced unwind with no coordination. The single client's hidden concentration did not just endanger the client; it set the lenders against each other.

#### Worked example: how 5x leverage turns a survivable drawdown into a wipeout

Take the recurring **\$10,000,000 book**. It holds a concentrated basket of a few single stocks, financed at **5x** leverage so the notional exposure is **\$50,000,000**. The basket of names draws down **22%** from its peak — severe for a concentrated single-stock basket, but the kind of move that happens in any given year.

First, what the drawdown does to an *unlevered* version of the same book:

- Unlevered exposure = equity = \$10,000,000.
- Loss = 22% × \$10,000,000 = **\$2,200,000**.
- Remaining equity = \$10,000,000 − \$2,200,000 = **\$7,800,000** — a 22% drawdown, painful but survivable; recovery needed is 0.22 / (1 − 0.22) = **+28%**.

Now the same 22% on the **5x-levered** book:

- Levered exposure = \$50,000,000.
- Loss = 22% × \$50,000,000 = **\$11,000,000**.
- Remaining equity = \$10,000,000 − \$11,000,000 = **−\$1,000,000**.

The loss is \$11,000,000 against \$10,000,000 of capital. Equity goes *negative* — the fund is wiped out and owes its lenders \$1,000,000 on top. The same 22% move that left the unlevered book at \$7,800,000 and needing a +28% recovery left the levered book at less than zero and needing a recovery that does not exist. Leverage did not change the bet or the market; it changed which side of the line between "bad year" and "the end" the same drawdown landed on.

*A drawdown the unlevered version of you survives easily can be the exact drawdown that kills the levered version of you — same market, same names, only the leverage dial moved.*

## The hidden-size problem: why each counterparty was blind

The most important structural feature of Archegos — the thing that made it a *systemic* event rather than just one rich man's bad week — is that **no single counterparty could see the total position.** This is worth sitting with, because it is the failure mode that ordinary risk limits do not catch. Every prime broker that financed Archegos had its own risk limits, and as far as each bank could tell, Archegos was within them. The slice *that bank* saw looked like a large but manageable client. What no bank saw was that the *same client* had a nearly identical slice at four or five other banks, and the aggregate of those slices was a concentration that no risk committee on earth would have signed off on. Figure 5 is the structure: one client, many brokers, each under its own limit, the aggregate far over.

![Graph of one client splitting a concentrated bet across several prime brokers, each seeing only its own slice within its own limit, while the true aggregate exposure is far over any safe total and every broker calls margin on the same day](/imgs/blogs/amaranth-and-archegos-concentration-leverage-and-the-single-trade-5.png)

This is a coordination failure dressed up as a risk-management success. Each bank optimized locally — "is *my* exposure to this client acceptable?" — and the answer was yes for each one. But risk is not additive in the comforting way the banks assumed. The relevant number was never any single bank's exposure; it was the *total* concentration in the same handful of names, because when those names moved, every bank's slice moved together and every bank tried to exit through the same door at the same moment. The local limits were satisfied and the global outcome was catastrophic. It is the strategic-risk version of a crowded trade, where everyone holds the same position and the exit is too small for all of them at once — the game-theory post on [crowded trades and the exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game) develops that dynamic, and Archegos is the clean case where the *crowd was one client* multiplied across many lenders.

The Credit Suisse / Paul Weiss report into the collapse was blunt about the cause: it was not a sophisticated, unforeseeable event. It was a failure to ask basic questions about the *total* size, the concentration, and the liquidity of the underlying names — to insist on transparency from a client whose business was lucrative enough that the bank did not want to push. The risk function existed; it was simply overruled by the desire to keep a profitable client happy. That is a governance failure, and the [hedge-fund failure taxonomy](/blog/trading/hedge-funds/how-hedge-funds-die-the-failure-taxonomy) catalogs how often the proximate cause of a blow-up is not a market event but a risk control that someone chose to ignore.

#### Worked example: the aggregate the brokers could not see

Suppose a single client, with **\$10,000,000,000** of equity, wants ~5x exposure — **\$50,000,000,000** of single-stock positions — and splits it across five prime brokers. Each broker writes swaps for roughly one-fifth of the book.

- Broker A's view: **\$10,000,000,000** of exposure to this client.
- Broker B's view: **\$10,000,000,000**. Broker C: **\$10,000,000,000**. And so on.

Each broker checks: is \$10,000,000,000 of exposure to a \$10,000,000,000-equity client acceptable? With collateral posted and the client's track record, each bank says yes — roughly 1x exposure-to-equity *as that bank sees it.* But the aggregate across all five brokers is:

- Total exposure = 5 × \$10,000,000,000 = **\$50,000,000,000** against \$10,000,000,000 of equity = **5x leverage**, all of it concentrated in the *same few names*.

No broker ever saw the \$50,000,000,000 number or the 5x leverage. Each saw 1x and approved it. The number that mattered — the one that determined whether a 20% move in the names would wipe the client and stick the lenders with the loss — was visible to *no one with the power to stop it.* The fix is not more sophisticated risk models; it is the boring discipline of demanding that a client disclose its total exposure across all counterparties before you finance any of it.

*A risk limit you satisfy locally tells you nothing about a risk that is only dangerous in aggregate; if no one adds up the slices, the most dangerous position in the market can be the one every individual lender thinks is fine.*

## The wipeout grid: reading the danger before the move

We can now put the two case studies into a single picture and use it as a forward-looking risk tool rather than a post-mortem. The equity hit from a single position is **leverage × position weight × adverse move**. Fix the adverse move at a fairly ordinary −35% single-name gap, and sweep the two dials the trader actually sets in advance: leverage (how many times your capital you control) and position weight (how much of the book the single name is). Figure 6 is the result — the wipeout grid. Every cell is the percent of your equity erased; the white contour is the −100% line, beyond which the book is gone.

![Heatmap grid with leverage on the vertical axis and position weight on the horizontal axis, each cell showing the percent of equity erased by a 35 percent single-name gap, with a white contour marking the 100 percent total wipeout line](/imgs/blogs/amaranth-and-archegos-concentration-leverage-and-the-single-trade-6.png)

Read the grid the way a risk manager should. Down in the bottom-left — low leverage, small weight — a −35% gap is a bad day, a single-digit dent, the kind of thing a diversified book shrugs off. As you move up and to the right, the numbers go past −50%, then −100%, and the cells turn dark. The Archegos cell — 5x leverage, the name at 60% of the book — sits deep in the wipeout zone: 5 × 0.60 × 0.35 = **1.05**, a 105% hit, the book gone with a debt left over. The grid makes the lesson visual and falsifiable: you do not need a tail event to be wiped out if you have placed yourself in the dark corner of the grid. An ordinary move is sufficient. The only thing that keeps you out of the dark corner is a limit on the two dials you control, set *before* the move, because after the move there is nothing left to set.

Notice what the grid does *not* depend on: the size of the move is held fixed at a perfectly normal −35%. The catastrophe is entirely a function of the leverage and concentration the trader chose. This is the single most important reframing in risk management: **blow-ups are not primarily caused by extreme markets; they are caused by ordinary markets meeting extreme positioning.** Amaranth and Archegos did not lose because natural gas or a few stocks did something impossible. They lost because they had arranged their books so that something perfectly possible would be fatal.

#### Worked example: walking two cells of the grid

Same **\$10,000,000 book**, same −35% gap in the single name. Compare two points on the grid.

*Cell A — the survivor's corner.* Leverage 1x (no borrowing), the name at 8% of the book.

- Equity hit = 1 × 0.08 × 0.35 = **0.028 = 2.8%**.
- Dollar loss = 0.028 × \$10,000,000 = **\$280,000**.
- Remaining equity = **\$9,720,000**. Recovery needed: 0.028 / (1 − 0.028) = **+2.9%**. A bad afternoon.

*Cell B — the Archegos corner.* Leverage 5x, the name at 60% of the book.

- Equity hit = 5 × 0.60 × 0.35 = **1.05 = 105%**.
- Dollar loss = 1.05 × \$10,000,000 = **\$10,500,000**.
- Remaining equity = \$10,000,000 − \$10,500,000 = **−\$500,000**. The book is gone and you owe \$500,000. Recovery needed: undefined — there is no capital left to recover with.

The market did the same thing in both cells — a −35% gap. The difference between "a bad afternoon" and "the firm no longer exists" was set entirely by where on the grid the trader chose to stand. That choice is the only part of this you control, which is exactly why it is the only part a limit can govern.

*The market hands everyone the same move; the grid shows you that survival was decided long before the move, by the leverage and concentration you signed up for.*

## The limit that would have saved them

Everything above is diagnosis. Here is the cure, and it is almost insultingly simple: **size every position to a hard maximum-loss budget, and never let any single name or single theme exceed it.** Instead of thinking "how much do I want to own of this?", you think "how much am I willing to *lose* on this in a bad scenario?", and you back out the position size from there. The budget is a fixed fraction of equity — say 2% per single name — and it is non-negotiable, enforced before the trade, not renegotiated during the drawdown.

The arithmetic is the reverse of the wipeout equation. If your worst-case adverse move on a name is `g`, your leverage on it is `L`, and your max-loss budget is `B` (as a fraction of equity), then the largest weight `w` the rule allows is the `w` that makes `L × w × g = B`, i.e. `w = B / (L × g)`. You let the budget and the assumed bad move *determine* the size, rather than letting your conviction determine the size and discovering the loss afterward. Figure 7 runs the exact same adverse event — a concentrated, 5x-levered single name hit by a −35% gap on a \$10,000,000 book — two ways: uncapped, the way Archegos ran it, and under a hard 2% max-loss-per-name budget.

![Bar chart comparing book equity after the same 35 percent single-name gap, the uncapped 5x and 60 percent weight version wiped to zero, the hard 2 percent max-loss cap version surviving with only a 200000 dollar dent](/imgs/blogs/amaranth-and-archegos-concentration-leverage-and-the-single-trade-7.png)

The uncapped book is gone — 5 × 0.60 × 0.35 = 105% of equity, the wipeout we have now computed three times. The capped book takes a **\$200,000** loss — 2% of the \$10,000,000 book — and survives with \$9,800,000, ready to trade the next day. The cap did not require predicting the gap, or being smarter than the market, or having better information than the other side. It required one decision, made in advance, about how much of the firm a single bet was allowed to put at risk. That is the entire difference between the funds that blow up and the funds that are still here.

#### Worked example: sizing a concentrated levered name to a 2% budget

You want to put on a high-conviction single-name trade, levered 5x, on the **\$10,000,000 book**. You judge that a realistic bad-scenario gap for this name is **−35%**. Your firm's rule is a **2% of equity** max-loss budget per single name. How big can the position be?

- Max-loss budget B = 2% of \$10,000,000 = **\$200,000**.
- Worst-case move g = **35%**; leverage L = **5x**.
- Largest allowed weight w = B / (L × g) = 0.02 / (5 × 0.35) = 0.02 / 1.75 = **0.0114 = 1.14% of the book**.
- That is a position with notional exposure of 5 × 1.14% × \$10,000,000 = **\$571,000** of the single name.
- Check: a −35% gap on \$571,000 of exposure = **\$200,000** loss = exactly the 2% budget. ✓

The cap does not say "never trade your best idea." It says your best idea, levered 5x, can be about 1.1% of the book — and if you want it bigger, you must *reduce the leverage* or *accept a smaller assumed bad move only if you can defend it.* Archegos at 60% weight and 5x was running this name at roughly fifty times what a 2% budget would have allowed. The rule would have made the March 2021 gap a \$200,000 event instead of a \$10,000,000,000 one.

*Sizing to a max-loss budget flips the question from "how much do I want" to "how much can I afford to lose," and it is the single discipline that turns a wipeout into a dent.*

## Detecting the pattern in your own book

You will never name your concentration "the bet that ends me." Nobody does. So you need a few mechanical questions whose answers reveal whether you have quietly drifted into the Amaranth/Archegos corner of the grid, regardless of how diversified the book *feels*.

The first question is a stress test, not a forecast: **what is my worst single-day loss if my largest exposure gaps against me, and is that loss survivable?** Take your biggest theme — not your biggest line item, your biggest *theme*, because the names that move together are one exposure — and ask what a −35% move in it does to your equity, with your actual leverage applied. If the answer is more than your drawdown tolerance, you are concentrated whether or not the position list looks long. This is the wipeout grid (Figure 6) run on your own book: you are finding which cell you are standing in *before* the market tells you. The number you compute is not a prediction of what will happen; it is a measurement of how much of your survival you have handed to a single move.

The second question is about aggregation across counterparties and accounts: **if I added up every place I have this exposure — every broker, every account, every instrument — how big is the total, and does any single venue's view of me reflect it?** Archegos is the warning that the dangerous number is the aggregate, and that the aggregate can be invisible to every individual party including, if you are not careful, yourself. The same applies to a retail trader with positions spread across three brokerages, or a desk whose exposure to one theme is split across cash equities, options, and swaps. Add it up in one place. The act of summing is the control.

The third question is about liquidity under stress: **if I had to exit this position in a day, at what price, and to whom?** If you cannot name a buyer of comparable size, you do not have a liquid position — you have a position whose liquidity exists only as long as nobody needs it, which is to say it does not exist when it matters. Size the position as though you will have to ride the full adverse move with no exit, because in the scenario that kills you, that is exactly what you will have to do.

#### Worked example: the survival stress test on your own \$100,000 account

You run a **\$100,000 retail account** and you feel diversified — eight positions, none looking dominant. But six of them are megacap tech names, which is one theme: when the sector sells off, they move together. Your tech exposure is **70% of the account**, and you hold it through a leveraged product giving you **2x** exposure on that sleeve. You run the survival stress test with a −35% sector gap.

- Theme weight: **70%** of the \$100,000 account.
- Leverage on the sleeve: **2x**.
- Adverse move: **−35%**.
- Equity hit = 2 × 0.70 × 0.35 = **0.49 = 49% of the account**.
- Dollar loss = 0.49 × \$100,000 = **\$49,000**. Remaining equity = **\$51,000**.
- Recovery needed: 0.49 / (1 − 0.49) = **+96%** just to get back to even.

"Eight positions" felt diversified. The stress test reveals one 49%-of-account bet hiding behind eight tickers, needing a near-double to recover from an ordinary sector gap. A 2% single-theme max-loss budget — \$2,000 — would have capped that sleeve at roughly 2.9% of the account at 2x leverage, turning the same gap into a \$2,000 dent. The diversification was an illusion of the position *list*; the stress test measured the diversification of the *risk*, and they were not the same.

*Diversification you can count in line items is not diversification you can rely on in a crisis; the only honest measure is the worst-day loss when your biggest theme gaps, and you have to compute it before the market does.*

## Common misconceptions

**"It was a tail event — a once-in-a-lifetime move nobody could have seen."** No. The natural-gas spread move that broke Amaranth and the single-stock declines that broke Archegos were both ordinary in magnitude. The wipeout grid (Figure 6) holds the move fixed at a perfectly normal −35% and still produces a 105% loss at 5x leverage and 60% weight. The "tail event" was the *positioning*, not the *market*. A −35% single-name move happens somewhere every few weeks; what does not happen every few weeks is a fund arranged so that such a move is fatal.

**"Diversification protects you — they just got unlucky in one name."** Both funds *looked* diversified on paper. Amaranth was a multi-strategy fund with several sleeves; Archegos held more than one stock. But diversification is about *effective* exposure, not the number of line items. When one trade is 60% of the book, or one strategy is 90% of the risk, you hold one position with decorations. A −50% move in a 5%-weight name costs 2.5% of the book; the same move in a 60%-weight name costs 30%. The arithmetic of [marginal and component VaR](/blog/trading/risk-management/marginal-and-component-var-where-the-risk-actually-lives) is precisely the tool that reveals when your "diversified" book is secretly one bet — it measures where the risk *actually* lives, not where the names are listed.

**"Leverage is fine as long as your thesis is right."** Leverage is fine right up until the moment the path of prices — not the final destination — takes your equity through zero. You can be *ultimately correct* and still be wiped out, because a leveraged position has to survive every intermediate drawdown to collect on the eventual gain. Archegos's names were not worthless; several recovered later. But Archegos was liquidated at the bottom, so being right "eventually" was worth nothing. Leverage converts "you need to be right" into "you need to be right *and* survive every wobble on the way," and the second condition is the one that kills.

**"A family office with no outside investors is lower-risk."** Archegos managed only one man's money, which meant lighter disclosure and less external scrutiny — and that made it *more* dangerous, not less. There were no outside investors demanding transparency, no allocator running due diligence on concentration, fewer regulatory eyes on the aggregate position. The discipline that outside accountability imposes is a feature; removing it removed a check that might have flagged the 5x concentration before it became \$10 billion of losses.

**"The banks were the victims of a rogue client."** The banks were the victims of their own failure to demand transparency. Each prime broker had the right to ask Archegos for its total exposure across all counterparties before financing any of it, and each chose not to push a lucrative client. The Credit Suisse / Paul Weiss report described a risk function that saw warning signs and was overruled by the business. The client's concentration was the loaded gun; the banks' incuriosity was the hand that ignored the safety. Counterparty transparency is something the lender controls, and they declined to use it.

**"A hard limit just caps your upside — it makes you worse."** A max-loss budget caps the size of any *single* bet, not your total risk-taking, and it does so precisely to keep you in the game long enough for your edge to compound. Figure 7 is the whole argument: the capped book lost \$200,000 and lived; the uncapped book lost everything and died. Over any long horizon, the compounding penalty of the occasional capped winner is trivially small next to the compounding catastrophe of a single uncapped wipeout, because the wipeout is *absorbing* — it does not just cost you that bet, it costs you every future bet you will never get to make.

## How it shows up in real markets

**Amaranth Advisors, September 2006.** A multi-strategy hedge fund whose energy book had grown into the entire fund lost roughly **\$6.6 billion**, the bulk of it in one week, on concentrated natural-gas calendar spreads (`dr.CRISES["amaranth_2006"]`, source: CFTC / Senate PSI report 2007). The position was so large relative to the market that it was illiquid — there was no counterparty big enough to take it off the fund's hands at the screen price — so when the March–April spread collapsed, the fund could neither defend nor exit the trade. The energy book was sold at a steep discount to JPMorgan and Citadel, and the fund was wound down. **Lesson: concentration plus illiquidity means you cannot get out when you most need to, and a position you cannot exit is a position that can take all of you.**

**Archegos Capital Management, March 2021.** A family office turned a few billion dollars of equity into tens of billions of single-stock exposure through total-return swaps at roughly **5x+ leverage**, spread across multiple prime brokers so that each saw only its slice (`dr.CRISES["archegos_2021"]`, source: Credit Suisse / Paul Weiss report 2021 and bank disclosures). When the concentrated names declined, margin calls hit, Archegos could not meet them, and the brokers liquidated the same names into the same days — a fire sale that cost the banks more than **\$10 billion**, Credit Suisse alone around **\$5.5 billion**. **Lesson: leverage turns a survivable drawdown into a wipeout, and hidden aggregate size turns a private loss into a systemic event; the fix is transparency on the total before financing any of it.**

**Long-Term Capital Management, 1998 — the same skeleton, one rung up.** LTCM ran roughly **25:1 balance-sheet leverage** on convergence trades whose correlations all went to 1 in the 1998 flight to quality, losing about **\$4.6 billion** in four months and requiring a Fed-organized recapitalization (`dr.CRISES["ltcm_1998"]`). It is the same disease — concentrated risk (in a *factor* rather than a single name), enormous leverage, and a single regime shift that hit every position at once — at the scale where the survival of the financial system, not just the fund, was the question. The dedicated [LTCM case study](/blog/trading/risk-management/concentration-and-position-limits-the-one-trade-that-can-end-you) and the [crowded-genius-trade analysis](/blog/trading/game-theory/crowded-trades-and-the-exit-game) trace how diversification *and* liquidity failed together, which is the LTCM-specific twist on the Amaranth/Archegos pattern.

The thread through all three is that none of them was killed by a market doing something impossible. Amaranth's spread moved within its historical range. Archegos's stocks fell within ordinary single-name bounds. LTCM's spreads widened in a flight to quality that had happened before. Each fund had simply arranged itself — through concentration, leverage, illiquidity, or hidden size — so that an ordinary move would be fatal. The market did not have to be extraordinary. The positioning was.

## The risk playbook: never let one trade vote on your survival

The discipline that would have saved all three funds fits on an index card. Here it is, as concrete rules.

- **Set a hard single-name max-loss budget — and size to it, not to conviction.** Pick a fraction of equity (commonly 1–2%) that any one name is allowed to lose in a defined bad scenario, and back the position size out of it: `weight = budget / (leverage × bad-case move)`. On a \$10,000,000 book at 2%, no single name should be able to cost you more than \$200,000 even in a −35% gap. Conviction sizes you *up to* the cap; it never sizes you past it.

- **Set a single-theme limit, not just a single-name limit.** Five "different" names that are all the same bet — same sector, same factor, same macro driver — are one position wearing five labels. Cap your *theme* exposure (e.g. no more than 10–15% of risk in any one driver), because the names will move together exactly when it hurts. This is what would have caught Amaranth's "diversified multi-strategy" book that was really one gas bet, and what marginal/component risk measurement exists to reveal.

- **Cap leverage with the worst path in mind, not the expected outcome.** Leverage L must let your equity survive every *intermediate* drawdown, not just the final destination. Ask: at my leverage, what single-name or basket drawdown takes my equity to zero? If the answer is a move that happens in any given year, your leverage is too high regardless of how right you are about the destination.

- **Demand counterparty transparency — and provide it.** If you are a lender, require a client to disclose its *total* exposure across all counterparties before you finance any of it; a slice within your limit tells you nothing about a concentration that is only dangerous in aggregate. If you are the client, recognize that hiding your size from your lenders hides it from yourself — the aggregate you don't track is the aggregate that kills you.

- **Price illiquidity into the size.** Before you put on a large position, ask how many days it would take to exit at a price you would accept. If the answer is "I am a large fraction of this market," you cannot get out, and a position you cannot exit must be sized as though you will have to ride the full adverse move with no escape. Amaranth's true risk was not the spread's volatility; it was that the fund *was* the market for that spread.

- **Make the limits non-negotiable in a drawdown.** Every blow-up has a moment where someone overrode the control because the client was profitable, the conviction was high, or "this time is different." The limit only works if it is enforced *before* the trade and not renegotiated during the loss. The risk function that gets overruled is, in the post-mortem, indistinguishable from the risk function that never existed.

The single sentence under all of it is the survival thesis of this series: **you can only compound if you are still in the game, and one concentrated, levered, hidden, or illiquid bet is the fastest way to leave it.** Amaranth and Archegos were not unsophisticated. They had talented people, real edges, and years of success. What they lacked was a hard limit on the single trade — a number that said *no position, however good it looks, is allowed to be big enough to end us.* That number is the cheapest insurance in finance, and the two most expensive lessons in this post are what it costs to skip it.

### Further reading

- [Concentration and position limits: the one trade that can end you](/blog/trading/risk-management/concentration-and-position-limits-the-one-trade-that-can-end-you) — the general theory of the concentration dial and how to cap it, with the position-impact arithmetic these two funds ignored.
- [Leverage and the arithmetic of ruin](/blog/trading/risk-management/leverage-and-the-arithmetic-of-ruin) — why leverage multiplies the path, not just the destination, and how it converts survivable drawdowns into wipeouts.
- [Marginal and component VaR: where the risk actually lives](/blog/trading/risk-management/marginal-and-component-var-where-the-risk-actually-lives) — the measurement tool that reveals when a "diversified" book is secretly one bet.
- [How hedge funds die: the failure taxonomy](/blog/trading/hedge-funds/how-hedge-funds-die-the-failure-taxonomy) — the GP-seat view of why blow-ups so often trace to a risk control someone chose to override.
- [Crowded trades and the exit game](/blog/trading/game-theory/crowded-trades-and-the-exit-game) — the strategic dynamic where everyone holds the same position and the exit is too small, which is exactly what Archegos's many-broker concentration became.

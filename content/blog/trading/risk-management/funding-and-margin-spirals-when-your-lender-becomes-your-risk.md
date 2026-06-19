---
title: "Funding and Margin Spirals: When Your Lender Becomes Your Risk"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "When you borrow to trade, the lender becomes your biggest risk. Here is how margin, haircuts, and the margin call combine into a self-reinforcing spiral that can wipe you out even when your thesis is right."
tags: ["risk-management", "funding-liquidity", "margin-call", "haircuts", "leverage", "deleveraging", "repo", "forced-selling", "survival"]
category: "trading"
subcategory: "Risk Management"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **One sentence:** The moment you borrow to trade, the most dangerous counterparty in your life is not the market — it is the lender who can demand their money back at the worst possible moment, and the mechanism by which they do it can feed on itself until you are wiped out even though your thesis was right.
> - **Borrowing turns your lender into your risk.** A levered position has two ways to kill you: the trade can be wrong, or the financing can be pulled. The second one does not care whether you are right, and it is the one that ends careers.
> - **The haircut is the lever, and it widens when you can least afford it.** The same collateral that funds a position at a 5% haircut in calm markets funds far less at a 30% haircut in a crisis — your borrowing capacity collapses exactly when prices are falling and you need it most.
> - **The margin call is a forced sale, not a polite request.** When equity drops below the maintenance line, the lender makes you post cash you do not have or sells your collateral at the market price. You stop choosing when you exit.
> - **The spiral feeds itself.** A price fall cuts levered equity faster, triggers a call, forces a sale, and the sale drives the price lower — which triggers the next call. The loop ignores whether you are correct.
> - **Funding liquidity is not market liquidity.** "Can I borrow against this?" and "Can I sell this?" are two different taps, and a crisis shuts both at once. Survival means keeping a margin buffer, stressing your haircuts, and terming your funding so the loop can never start with you.

The fastest way to be removed from a trade you would have won is to borrow the money to put it on.

This is the part of risk management that most surprises people, because it has nothing to do with being wrong. You can have the right thesis, the right entry, the right time horizon, and still be carried out feet-first — not because the market disagreed with you, but because the entity that financed your position decided, on its own schedule and for its own reasons, that it no longer wanted to. When you trade with your own money, your only opponent is the market. When you trade with borrowed money, you acquire a second opponent who is far more dangerous, because this one can act *before* the market resolves your thesis and can force you to act with it. That second opponent is your lender.

Long-Term Capital Management, a fund run by two Nobel laureates and the best fixed-income arbitrageurs alive, was financed by virtually every major bank on Wall Street. In the summer of 1998 their trades were, in the long run, correct — most of their convergence positions did eventually converge. But they were levered roughly 25 to 1, and when Russia defaulted and their spreads moved the wrong way, the financing structure turned a survivable mark-to-market loss into the destruction of the firm in about four months and roughly \$4.6 billion of capital. Archegos, in 2021, held its positions through swaps financed by a handful of prime brokers; when a few of its concentrated stocks fell, the brokers' forced unwind drove those same stocks down further, and the loop vaporised more than \$10 billion of bank capital in days. In both cases the lender was the risk, and the mechanism was a spiral.

![A closed loop where a price fall cuts levered equity, triggers a margin call, forces a sale, and the sale drives the price lower into the next call](/imgs/blogs/funding-and-margin-spirals-when-your-lender-becomes-your-risk-1.png)

Look at the loop above before reading another word, because it is the whole post in one picture. A price falls — an ordinary move. Because you are levered, your equity falls faster than the price. That drops you below the maintenance line, so the lender issues a margin call. You have no spare cash, so the call resolves as a forced sale. The forced sale — yours and everyone else's in the same trade — pushes the price down again. And a lower price triggers the next call. The arrow closes back on itself: this is a *spiral*, a feedback loop that can run several times in a single day, and it does not stop because your thesis is good. It stops when you have nothing left to sell or when someone outside the loop steps in to break it.

We are going to build this from absolute zero — what margin is, what a haircut is, what a margin call mechanically does, why the haircut widens exactly when it hurts most, and how the loop runs leg by leg — and tie every piece back to the survival spine of this whole series: *you can only compound if you are still in the game*, and your lender is the single counterparty most able to take you out of it.

## Foundations: margin, haircuts, and the maintenance line

Before any spiral, we need five ideas defined from scratch. Every later section rests on these, so skip nothing.

### What it means to borrow against a position

When you buy an asset with borrowed money, you are not borrowing against your general creditworthiness — you are borrowing *against the asset itself*, which serves as **collateral**. The lender hands you cash; you buy the asset; the asset sits in your account as the lender's security. If you fail to pay them back, they take the asset and sell it. This is the same structure as a mortgage: the bank lends you money to buy a house, and the house secures the loan. In trading, the asset is a stock, a bond, a futures contract, or a basket of them, and the lender is a broker, a prime broker, or a repo counterparty.

The key consequence is that your loan and your collateral are *the same trade*. The thing securing the loan is the thing whose price you are betting on. When the collateral falls in value, two bad things happen at once: you lose money, and the security backing your loan shrinks. The lender notices the second one immediately, and that is where the danger lives.

### Margin: the slice that is yours

**Margin** is the portion of a position funded by your own capital rather than by the loan. If you buy \$100,000 of an asset and put up \$50,000 of your own money, borrowing the other \$50,000, your margin is 50% — half the position is yours, half is the lender's. Put up \$33,333 and borrow \$66,667, and your margin is one-third. The smaller your margin, the more you have borrowed, and the more leveraged you are.

There are two margin numbers that matter, and confusing them is how people get surprised:

- **Initial margin** is how much of your own money you must put up to *open* the position. A 50% initial margin means you can buy \$100,000 of the asset with \$50,000 of equity.
- **Maintenance margin** is the minimum equity percentage you must *keep* in the position to hold it open. A 25% maintenance margin means that if your equity ever falls below a quarter of the position's value, the lender acts.

The gap between these two is your room to lose. You open at 50% margin; you are allowed to ride the position down until your equity hits the 25% maintenance line; below that, the lender intervenes. That intervention is the margin call, and we will get to its mechanics in a moment. For now, fix the idea: **maintenance margin is the trapdoor under your levered position, and the price drop that opens it is much smaller than most people expect.**

### The haircut: how much the lender will not lend against

A **haircut** is the discount a lender applies to your collateral when deciding how much to lend against it. If you post \$100 of a bond as collateral and the lender applies a 5% haircut, they will lend you \$95 against it — they hold back \$5 as a cushion against the collateral falling in value before they can sell it. The haircut is the lender protecting *themselves*, not you. It is their estimate of how much the collateral could drop in the time it would take them to liquidate it if you defaulted.

The haircut and your leverage are two sides of one coin. If the haircut is `h`, then per dollar of collateral the lender finances `1 − h`, and the maximum leverage you can run is `1 / h`:

- A 5% haircut finances \$0.95 of every \$1.00 of collateral, supporting up to 20× leverage.
- A 30% haircut finances \$0.70, supporting at most about 3.3×.
- A 50% haircut finances \$0.50, capping you at 2×.

The haircut is therefore the master lever on how much you can borrow. And the central, lethal fact of this entire post — the one we will return to again and again — is that **the haircut is not a constant.** It is set by the lender's fear, and it widens precisely when markets are falling, which is precisely when you have the least ability to absorb it.

![Stacked bars showing the same one hundred dollars of collateral financing ninety-five dollars at a five percent haircut and only fifty dollars at a fifty percent haircut, with the implied maximum leverage falling from twenty times to two times](/imgs/blogs/funding-and-margin-spirals-when-your-lender-becomes-your-risk-2.png)

The figure above shows the same \$100 of identical collateral across four funding regimes. In calm markets a 5% haircut lets it finance a \$95 loan and support 20× leverage. As stress builds, the haircut widens to 15%, then 30%, then 50% in a true freeze — and the same collateral now finances only \$50, capping you at 2×. Nothing about the collateral changed. The bond is the same bond, the stock the same stock. What changed is how much the lender is willing to lend against it, and that single change can force you to either post more capital or shrink the position — often both, at once, when you can least afford either.

#### Worked example: how a haircut sets your borrowing capacity

Take our recurring \$100,000 retail account and suppose you want to buy a high-quality bond portfolio.

- **In a calm market, the haircut is 5%.** For every \$1.00 of bonds you post, the lender finances \$0.95 and you must fund \$0.05 from your own equity. With \$100,000 of equity available to cover haircuts, you can support a position of roughly \$100,000 / 0.05 = **\$2,000,000** — that is 20× leverage, and \$1,900,000 of it is the lender's money.
- **A stress event widens the haircut to 30%.** Now every \$1.00 of bonds requires \$0.30 of your own equity. The same \$100,000 of equity supports only \$100,000 / 0.30 ≈ **\$333,000** of position — about 3.3×.
- **The change in your borrowing capacity is brutal.** From \$2,000,000 down to \$333,000 is a collapse of more than 83% in how much you can finance, and the bonds themselves may have barely moved. If you were holding the \$2,000,000 position when the haircut widened, you must find roughly \$500,000 of fresh equity to keep it — or sell about \$1,670,000 of bonds to shrink down to what the new haircut allows.

*The haircut, not the price, is often the thing that actually forces you to sell — and it widens fastest in exactly the market where selling is most expensive.*

### Who the lender actually is, and the channels through which you borrow

It helps to be concrete about *who* holds this power over you, because the spiral works the same way whether the lender is a retail broker or a tier-one investment bank — only the scale changes. Leverage reaches a trader through several channels, and each one installs a different lender as a counterparty who can call you:

- **A margin loan from a broker.** The most direct form: your broker lends you cash to buy more of an asset than your equity alone could, holding the asset as collateral. The broker sets the initial and maintenance margins and can change them. This is the channel most retail and many professional traders use, and it is the one behind the classic stock-market margin call.
- **Futures.** A futures contract lets a small deposit — the initial margin — control a large notional position. The leverage is built into the instrument, and the clearinghouse marks you to market daily via variation margin. You never took out a "loan" in the everyday sense, but the leverage and the daily cash demands are identical in effect, and the clearinghouse is the counterparty that can force you out.
- **Repo (repurchase agreements).** The plumbing of the leveraged bond world: you sell a bond to a counterparty and agree to buy it back later at a slightly higher price, which is economically a secured loan with the bond as collateral and the haircut baked in. Repo financing is typically very short-term — often overnight — which is exactly what makes it fragile: the lender can decline to roll it tomorrow, widen the haircut, or raise the rate, and an entire levered bond book can be defunded in days. This was central to 1998, 2008, and 2020.
- **Total-return swaps.** Here you never even hold the asset. A bank holds it and pays you its total return in exchange for a financing fee, while you post margin against the position's moves. The leverage can be very high, and — crucially — because the bank holds the asset, each of your several swap counterparties may see only the slice of your exposure that they finance, and none sees the total. This is precisely the structure that let Archegos build a position no single lender understood the full size of, until it unwound.

The unifying point across all four channels is that **someone other than you holds the power to demand cash or force a sale when the position moves against you.** The vocabulary differs — margin loan, variation margin, repo haircut, swap financing fee — but the arithmetic is the one we have been building, and the danger is the same: the lender's terms can tighten, on the lender's schedule, regardless of whether your thesis is right. Whenever you are levered, the first question is not "what is my downside if I'm wrong?" but "who can force me to sell, and when can they do it?"

### The margin call: what mechanically happens

A **margin call** is the lender's demand that you restore your equity to the required minimum. Here is the precise sequence, with no hand-waving:

1. **Your equity falls below the maintenance margin.** Because the asset dropped, or the haircut widened, or both, the percentage of the position that is *yours* has fallen under the line.
2. **The lender notifies you of the shortfall** and gives you a window — sometimes a day, sometimes hours, sometimes minutes in a fast market — to make it good.
3. **You either post fresh cash to top up your equity, or you sell collateral to shrink the position** until your equity is back above the line.
4. **If you do neither in time, the lender does it for you** — they liquidate enough of your collateral, at whatever price the market is offering, to bring you back into compliance. You do not choose what is sold, when, or at what price.

There is a related mechanism worth naming, because it is the daily heartbeat of the call: **variation margin**. In many levered structures — futures, swaps, cleared derivatives — you do not wait for a single dramatic call. Instead, your account is marked to market every single day, and each day's loss is *swept out of your account as cash* and paid to the counterparty, while each day's gain is swept in. This is variation margin, and it means a levered position is bleeding or gaining real cash continuously, not just on paper. A string of losing days drains your cash reserve one day at a time, and when the reserve runs dry the position must be cut. The margin call is the dramatic version; variation margin is the slow, relentless version that drains the buffer before the dramatic call ever arrives. Both are the lender (or the clearinghouse) reaching into your account and taking cash when the position moves against you — and both accelerate exactly when the market is most violent.

The fourth step is where the wipeout happens, and it follows from a fact about leverage that almost everyone underestimates: when you are levered, your spare cash is *already deployed* — that is what being levered means. The whole point of leverage is that you have used your capital plus borrowed money to control more exposure. So the cash to meet a call is exactly the cash you do not have. For most levered players, step 3 is not "post fresh cash"; it is "sell." And the sale lands at the worst possible price, because a margin call is triggered by falling prices, which means you are forced to sell into a market that is already going down. The call converts a paper drawdown you might have ridden out into a realised, permanent loss.

This is the bridge to the survival spine. A margin call walks you from the survivable side of the [recovery-asymmetry math](/blog/trading/risk-management/leverage-and-the-arithmetic-of-ruin) — where a drawdown is painful but recoverable because you still hold the position to recover *with* — onto the absorbing side, where the loss is locked in and the position is gone. You cannot compound a recovery on exposure you were forced to sell.

## How small a fall it takes: leverage and the margin buffer

The single most useful number to internalise is how far the price can fall before a margin call hits, as a function of your leverage. Most people radically overestimate it. They assume they have plenty of room. They do not.

Here is the math, built from the maintenance-margin definition. Let `E0` be your own capital, `L` your leverage, so your gross position is `P = L × E0` and your fixed debt is `D = (L − 1) × E0`. After a price drop of fraction `x`, the position is worth `P × (1 − x)` and your equity is `P × (1 − x) − D`. A maintenance call fires when your equity falls to the maintenance fraction `m` of the position value. Setting equity equal to `m` times position value and solving for `x` gives the price drop that triggers the call:

`x_call = (1 − m × L) / (L × (1 − m))`

That formula is worth keeping. With a standard 25% maintenance margin (`m = 0.25`), it produces results that should make any over-levered trader uncomfortable.

![A curve showing the price drop that triggers a margin call falling sharply as leverage rises, with markers showing a thirty-three percent drop at two times, an eleven percent drop at three times, and already under margin at five times](/imgs/blogs/funding-and-margin-spirals-when-your-lender-becomes-your-risk-3.png)

The figure above plots `x_call` against leverage for two maintenance margins. The curve is steep and unforgiving. At 2× leverage with a 25% maintenance margin, you can absorb a 33% fall before the call — genuine room. At 3×, that collapses to an 11% fall. And at 5×, with a 25% maintenance requirement, you are *already* under margin the moment you open the position, because 5× means only 20% of the position is your equity and the line is at 25%. The shaded thin-cushion zone at the bottom is the danger band: any leverage above about 4× leaves you with less than a 10% buffer before the lender takes the wheel. The thing to feel in your gut is how *small* an ordinary market move has to be to call a moderately levered book — and how the very buffer that protects you shrinks fastest exactly as you add the leverage that makes the position lucrative.

#### Worked example: the margin buffer at three leverage levels

Take the \$100,000 account again, with a 25% maintenance margin, and put on a position at three different leverage levels. How far can the price fall before the call?

- **At 2× (a \$200,000 position, \$100,000 borrowed):** `x_call = (1 − 0.25 × 2) / (2 × 0.75) = 0.5 / 1.5 = 33.3%`. The price can fall a full third before you are called. You have real room to ride out an ordinary correction.
- **At 3× (a \$300,000 position, \$200,000 borrowed):** `x_call = (1 − 0.75) / (3 × 0.75) = 0.25 / 2.25 = 11.1%`. A move smaller than a routine 10-day pullback in many assets puts you on the call line. Your buffer has shrunk from a third to roughly a tenth.
- **At 5× (a \$500,000 position, \$400,000 borrowed):** your own equity is only 20% of the position, already below the 25% maintenance line. You cannot hold this position at all under a 25% maintenance margin without a lower requirement; the first tick down calls you.

The position is the same asset in all three cases. The only thing that changed is how much you borrowed, and that single choice walked your buffer from a comfortable 33% down to a hair-trigger, and then to no buffer at all.

*Leverage does not change the market's volatility; it changes how much of that volatility your buffer can absorb before someone else makes your decisions for you.*

## The spiral: why a small shock becomes a wipeout

Everything so far has been static — one position, one drop, one call. The spiral is what happens when the call triggers an action that triggers the next call. This is the heart of the post, and it is what separates a survivable drawdown from a blow-up.

Here is the mechanism, leg by leg. Start with a levered position and an ordinary exogenous shock — say a 6% price fall driven by some piece of news. Because you are levered, your equity falls faster than 6%, and if your buffer was thin, that drop pushes you below the maintenance line. The lender calls. You have no spare cash, so you sell collateral to comply. But your selling has *price impact* — every sale you and everyone else in the same crowded trade execute pushes the price down a little further. And a lower price means your remaining position is worth less, which puts you back below the line, which triggers *another* call, which forces *another* sale, which moves the price *again*. Each leg of the loop feeds the next.

![A two-panel figure where the top shows a price walking down from one hundred through successive legs and the bottom shows cumulative forced selling rising at each leg](/imgs/blogs/funding-and-margin-spirals-when-your-lender-becomes-your-risk-4.png)

The figure above traces a stylised spiral leg by leg. The top panel is the price: it starts at \$100, takes the exogenous 6% shock down to \$94, and then — instead of stabilising — keeps walking down, \$94 to \$89 to \$82 and on toward \$68, because each margin call forces a sale and each sale knocks the price lower into the next call. The bottom panel shows the cumulative forced selling that drives it: a wave of liquidation that swells while the position is under-margined and only tapers once enough has been sold that the margin ratio is finally restored. The crucial point is the causal direction. The first leg was exogenous — the market did it to you. Every leg after that was *endogenous* — you and your fellow forced sellers did it to yourselves. That is the difference between a shock and a spiral: a shock is a single push, a spiral is a push that recruits your own forced selling to keep pushing.

This is why the spiral is so dangerous and so hard to reason about in advance. When you size a position, you ask "how much can the market move against me?" But the spiral means the answer is partly determined by *your own forced selling and that of everyone in the same trade* — a quantity that does not exist until the loop starts, and then grows with each turn. Your true downside is not the exogenous shock; it is the exogenous shock amplified by the deleveraging it sets off.

#### Worked example: a margin-call cascade on a \$10,000,000 book

Now scale up to the \$10,000,000 book and walk through a cascade with explicit dollar math. Suppose you run a \$10,000,000 position financed at 3× — that means \$3,333,333 of your own equity and \$6,666,667 borrowed, with a 25% maintenance margin.

![A chart of a ten million dollar book at three times leverage, plotting position value, equity, and the maintenance requirement as the price falls, with the margin call firing where equity crosses the maintenance line at an eleven percent drop](/imgs/blogs/funding-and-margin-spirals-when-your-lender-becomes-your-risk-7.png)

The figure above shows the three lines that matter as the price falls. The blue line is the position value, sliding down from \$10,000,000. The green line is your equity — position value minus the fixed \$6,666,667 of debt — which falls *three times faster* than the price because the debt does not move. The dashed lavender line is the maintenance requirement, 25% of the position value. Watch where green crosses lavender: at an 11.1% price drop, your equity has fallen to the maintenance line and the call fires.

Here is the arithmetic at that crossing:

- **At an 11.1% drop**, the position is worth \$10,000,000 × 0.889 = **\$8,888,889**.
- Your equity is \$8,888,889 − \$6,666,667 = **\$2,222,222**.
- The maintenance requirement is 25% × \$8,888,889 = **\$2,222,222**.
- Equity exactly equals the requirement. The next tick down puts you under, and the call is issued.

Now the cascade. You have no spare cash, so you sell. To restore a 25% equity ratio after the call, you must sell a large slice of the book — and that selling pushes the price down further. Say it drops another 3% from your selling and the broader rush. The position is now worth roughly \$8,888,889 × 0.97 ≈ \$8,622,000, your equity has fallen below \$2,000,000, and you are under margin *again* — even though you just sold to comply. A second call. More selling. Another leg down. The book that looked solid at \$10,000,000 with \$3,333,333 of equity can lose its entire equity cushion in a cascade triggered by a price move — an 11% drop — that an unlevered holder would have shrugged off as a bad fortnight.

*A levered book does not have one downside number; it has a cascade whose depth depends on how much forced selling its own margin calls unleash into a falling market.*

### Why the spiral is a crowd, not a solo act

The cascade above is destructive enough run by one trader, but the real-world version is far worse because you are almost never alone in the trade. A good trade attracts capital; capital arrives levered; and so the same position is held, in size, by many players financed in similar ways. When the exogenous shock hits, it does not call only you — it calls *everyone* in the trade at roughly the same time, because they all crossed their maintenance lines together. And now every one of them is forced to sell the same asset into the same falling market at the same moment.

This is what turns a manageable individual deleveraging into a market-wide spiral. Your forced selling has price impact; so does everyone else's; and the combined impact drives the price down far faster and further than your own selling alone would. Each participant is reacting rationally to their own margin call, and the sum of those rational individual reactions is a collective rout. The price impact you should have stress-tested against was never just *your* forced selling — it was the forced selling of the entire crowd that shares your financing structure. This is why crowded, uniformly-levered trades are so dangerous: the crowding is invisible and free in calm markets, and it is the amplifier that converts a shock into a spiral in a crisis. The strategic version of this — the race to be the first one out the door — is the [exit game](/blog/trading/game-theory/case-study-ltcm-1998-the-crowded-genius-trade) that every crowded leveraged trade eventually plays.

### Procyclicality: the system is built to make this worse

There is a second-order effect baked into the way margin and haircuts are set, and it has a name: **procyclicality**. Margin requirements and haircuts are typically tied to recent volatility — when markets are calm and volatility is low, haircuts are small and you can lever up; when markets are turbulent and volatility spikes, haircuts widen and leverage is forced down. That sounds prudent, and from each individual lender's risk-management seat it is. But aggregated across the whole system it is destabilising, because it means the rules *loosen in the boom* (encouraging more leverage exactly when assets are expensive) and *tighten in the bust* (forcing deleveraging exactly when assets are cheap and selling is most damaging).

The result is that the financing system amplifies the cycle instead of damping it. In the good times, cheap leverage inflates positions and prices. In the bad times, the same mechanics that fed the boom now demand cash, widen haircuts, and force sales — pouring fuel on the fire. The funding spiral is not a bug that occasionally appears; it is the predictable consequence of a system that sizes leverage to recent calm and then withdraws it at the first sign of storm. Knowing this is itself protective: it tells you that the moment everyone is comfortably levered against a low-volatility backdrop is exactly the moment the next forced-deleveraging spiral is being loaded.

#### Worked example: a haircut widening that forces a sale all by itself

Here is the subtle case that catches even careful traders: a forced sale triggered not by the price falling, but by the *haircut* widening on a position whose price has barely moved. Take the \$10,000,000 book again, but this time financed against high-grade bonds at a 5% haircut, so you have posted \$500,000 of equity against the haircut and borrowed \$9,500,000.

- **Calm state:** haircut 5%, so your \$500,000 of equity supports the \$10,000,000 position exactly. You are fully utilised but compliant.
- **A funding stress hits and the lender widens the haircut to 15%** — note the bond price has barely moved, maybe down 1%. The new requirement is 15% × \$10,000,000 = **\$1,500,000** of equity. You have only \$500,000 posted.
- **You are now \$1,000,000 short of the new requirement**, purely because the haircut tripled. To comply without posting fresh cash, you must sell bonds: shrinking the position to where your \$500,000 of equity meets a 15% haircut means a position of \$500,000 / 0.15 ≈ **\$3,333,000**. You must sell roughly **\$6,667,000** of bonds — two-thirds of the book — into a stressed market, all because the lender changed one number.
- **And your selling pushes the price down**, which widens the haircut further and pushes the bonds toward the next round of selling. The price did almost nothing; the haircut did everything.

*A spiral does not need the price to fall first — a widening haircut can start the forced selling all on its own, and the selling then supplies the price fall that the spiral needs to continue.*

## Funding liquidity is not market liquidity

There is a distinction here that is easy to blur and expensive to miss, and it connects this post directly to the [companion piece on market liquidity](/blog/trading/risk-management/liquidity-risk-you-cant-sell-what-no-one-will-buy). There are two completely different questions you can ask about a position, and they have different answers:

- **Market liquidity** asks: *can I sell this?* It is about the depth of buyers waiting on the other side, the bid-ask spread, how much the price moves when you try to exit. A market-liquid asset is one you can sell in size without crushing the price.
- **Funding liquidity** asks: *can I borrow against this?* It is about whether a lender will finance the position, at what haircut, and whether they will keep financing it tomorrow. A funding-liquid position is one you can keep borrowing against.

These are genuinely separate taps. You can have a position that is market-liquid (deep buyers, tight spread) but funding-illiquid (no lender will finance it, or only at a punishing haircut). And you can have the reverse. They are controlled by different people — market liquidity by the buyers, funding liquidity by the lenders — and they fail for different reasons.

![Two columns comparing funding liquidity as the lender's tap and market liquidity as the buyer's tap, each shown going from calm to crisis and both shutting at the bottom](/imgs/blogs/funding-and-margin-spirals-when-your-lender-becomes-your-risk-5.png)

The figure above lays the two taps side by side. On the left, funding liquidity: the lender's tap, fine at a 5% haircut in calm markets, choked off at a 30%+ haircut or pulled entirely in a crisis — and it shuts on the *lender's* fear, regardless of whether your thesis is right. On the right, market liquidity: the buyer's tap, deep and tight in calm markets, evaporating in a crisis as bids disappear and each sale moves the price — and it shuts on the *buyers'* fear, right when you are forced to sell. The lethal fact is at the bottom of both columns: **in a real crisis, both taps shut at the same time.** The lender widens your haircut and forces you to sell exactly when the buyers have vanished and selling is most expensive. You are pushed through a door that someone else is simultaneously closing.

This is the deep reason funding spirals are so destructive. The margin call forces you to access *market* liquidity (you have to sell) precisely at the moment that *funding* liquidity has dried up (the lender pulled your line). The two failures are correlated — not by coincidence, but because the same crisis fear drives both. A levered trader is therefore exposed to a compound failure: the financing vanishes, which forces a sale, into a market where the buyers have also vanished. The treatment of how the *market* side of this fails — the vanishing bids, the widening spread, the price impact of being forced to sell — is the subject of the [liquidity-risk post](/blog/trading/risk-management/liquidity-risk-you-cant-sell-what-no-one-will-buy); here the point is that your lender is the one who chooses *when* you are forced into that illiquid market, and they choose the worst possible moment.

The two taps also feed each other through a reflexive loop that is worth tracing explicitly. When funding tightens — haircuts widen, lines get pulled — levered holders are forced to sell, which consumes market liquidity and pushes prices down. Falling prices and rising volatility then make lenders *more* nervous, so they widen haircuts and pull lines *further*, which forces *more* selling. Market illiquidity feeds funding illiquidity, which feeds market illiquidity, in the same circular way the margin spiral feeds itself. This is why crises in funding markets and crises in trading markets are almost always the same event: they are two faces of one reflexive loop, and a levered book sits at the exact point where the two loops intersect. The trader who keeps both taps in mind — who asks not only "can I sell this?" but "will my lender still finance this tomorrow, and at what haircut?" — is the one who is not surprised when both shut together. The deeper version of how an illiquid position can tip into outright insolvency once the whole book is marked to the fire-sale price is the subject of the [liquidity-solvency doom loop](/blog/trading/risk-management/the-liquidity-solvency-doom-loop-illiquid-can-become-insolvent).

## The cost of carry: when financing itself gets expensive

There is one more way the lender becomes your risk, subtler than the outright margin call but just as corrosive: the *price* of financing can spike even when the lender does not pull your line entirely. Most levered trades are, at bottom, a carry trade — you borrow at one rate and earn a higher return on the asset, pocketing the difference. The trade is profitable only as long as your funding stays cheap. When the cost of borrowing jumps, a position that was comfortably positive-carry can flip to negative-carry overnight, and you are now *paying* to hold a trade that no longer earns its keep.

![A line chart showing an overnight financing rate sitting near two percent through a calm period and then spiking sharply to about nine and a half percent during a stress window before subsiding](/imgs/blogs/funding-and-margin-spirals-when-your-lender-becomes-your-risk-6.png)

The figure above shows an illustrative path of overnight secured financing — the kind of repo rate a levered book pays to roll its borrowing each day. For most of the window it sits placidly near 2%, and the carry looks free. Then a stress event hits, and the rate spikes toward 9.5% as lenders pull back and the demand for cash overwhelms the supply. The pattern is real and recurring: secured-funding rates blew out in the September 2019 repo spike and in the 2008 dash-for-cash, and they do so for the same reason haircuts widen — lenders' fear and a scramble for cash converge at the worst moment. The cruelty is in the timing. Your financing gets most expensive exactly when your position is also falling and your buffer is thinnest, so the rising cost of carry piles onto the mark-to-market loss and the margin pressure all at once.

#### Worked example: carry flips from free to ruinous

Take the \$10,000,000 book financed with \$6,666,667 of borrowed money. Suppose the asset yields 4% a year and your overnight financing costs 2% a year in calm markets.

- **In calm markets**, you pay 2% on \$6,666,667 = **\$133,333 a year** in financing, and you earn 4% on the \$10,000,000 position = **\$400,000 a year**. Your net carry is +\$266,667 a year. The trade pays you to hold it.
- **A funding stress hits and your overnight rate spikes to 9.5%.** You now pay 9.5% on \$6,666,667 = **\$633,333 a year** — an annualised rate; over a multi-week stress window the daily cost balloons accordingly. Against the same \$400,000 of asset yield, your carry has flipped to roughly −\$233,333 a year. The trade now *costs* you to hold it.
- **And this happens while the price is falling.** So you are simultaneously taking a mark-to-market loss, facing a widening haircut, and bleeding negative carry — three pressures converging, each one feeding the temptation to sell into the very market where selling is most expensive.

*A carry trade is a bet that your funding stays cheap; when the funding spikes, the trade can turn into a money-loser before the underlying thesis has even been tested.*

## Common misconceptions

**"A margin call just means I have to add a little cash — it's an inconvenience, not a catastrophe."** This is the most dangerous misunderstanding, and it rests on imagining you have spare cash. The defining feature of being levered is that your capital is *already deployed* — that is what leverage means. The cash to meet a call is precisely the cash you do not have. For almost everyone, the call resolves not as a top-up but as a forced sale at the worst price. On the \$10,000,000 book at 3×, an 11% drop does not ask you for "a little cash"; it puts your entire \$2,222,222 of remaining equity at the mercy of a cascade.

**"As long as my thesis is right, I'll be fine — I can wait it out."** The spiral does not care whether you are right. LTCM's convergence trades were, in the long run, correct, and most did converge — but the firm was destroyed in four months because being right *eventually* is worthless if you are forced to sell *now*. The margin call severs the link between your thesis and your survival: it forces the exit before the thesis can resolve. Being right is necessary but it is not sufficient; you also have to still be holding the position when the rightness shows up.

**"My haircut is 5%, so I have a comfortable buffer."** Your haircut is 5% *today, in calm markets*. It is set by the lender's fear, and it will widen to 15%, 30%, or more exactly when prices fall — collapsing your borrowing capacity from \$2,000,000 to \$333,000 on the same \$100,000 of equity, as the worked example showed. Sizing to a calm-market haircut is sizing to a number that exists only when you do not need a buffer.

**"A stop-loss protects me from the margin call."** A stop-loss caps your loss only if the market trades through your stop in an orderly way. A margin call in a fast, spiraling market does not honour your plan — the lender liquidates at the available price, which in a cascade can be far below your intended exit. And a gap or a limit-down open skips your stop entirely. The forced sale of a margin call and the orderly exit of a stop-loss are different events; the spiral produces the former, not the latter.

**"Funding can't just vanish overnight — there are contracts."** Much short-term financing is exactly that: short-term. Overnight repo, daily margin, prime-broker financing that is callable on short notice — these are not multi-year commitments. The lender can widen the haircut, raise the rate, or decline to roll the financing tomorrow, and there is often little you can do about it. Archegos's prime brokers did not need to breach any contract to unwind it; the financing structure simply allowed them to act. Funding *does* vanish overnight, and the assumption that it cannot is how levered books are caught flat-footed.

**"Big funding blow-ups only happen to reckless amateurs."** The opposite is closer to true. LTCM was run by Nobel laureates and financed by every major bank. Archegos was a professional family office levered through swaps that hid its total size from each prime broker. Sophistication does not protect you from the spiral; if anything it provides the confidence to use more leverage and more complex financing, which is exactly what makes the eventual cascade larger. The mechanism does not care how smart you are — it cares how levered you are and how thin your buffer is.

## How it shows up in real markets

**Long-Term Capital Management, August–September 1998.** LTCM carried roughly **25 to 1** balance-sheet leverage — about \$125 billion of assets on around \$4.7 billion of equity — plus around \$1.25 trillion of gross derivatives notional. Their convergence trades had a real edge in normal conditions. But when Russia defaulted and a flight to quality drove their spreads the wrong way, the leverage turned a few percent of mark-to-market loss into a threat to the firm's existence. Critically, their lenders and counterparties widened margin and demanded collateral exactly as the positions moved against them, and the trades were so crowded that the forced unwind had nowhere to go — funding liquidity and market liquidity failed together, precisely the compound failure this post describes. They lost about **\$4.6 billion** of capital in roughly four months and required a Fed-organised consortium recapitalisation of about \$3.6 billion. The strategic anatomy — why the trade was crowded and the exit impossible — is dissected in the [LTCM case study](/blog/trading/game-theory/case-study-ltcm-1998-the-crowded-genius-trade).

**Archegos Capital Management, March 2021.** Bill Hwang's family office held enormous, concentrated single-stock positions through total-return swaps, levered around **5×**, with the structure deliberately hiding the total size from each individual prime broker. When a few of his concentrated names fell, the swap losses hit his equity at the leverage multiple, the prime brokers issued margin calls, and — exactly as the spiral figure shows — the brokers' forced unwind of the positions drove those same stocks down further, triggering more calls and a race among the brokers to sell first. The banks absorbed more than **\$10 billion** in aggregate, with Credit Suisse alone losing about **\$5.5 billion**. This is the funding spiral at institutional scale: concentrated, swap-financed leverage met an ordinary adverse move, and the forced sale by the lenders crystallised a catastrophe. The firm-level autopsy of how financing structures like this kill institutions is in [how hedge funds die](/blog/trading/hedge-funds/how-hedge-funds-die-the-failure-taxonomy).

**The COVID crash and dash-for-cash, February–March 2020.** The S&P 500 fell about **34%** peak to trough in roughly a month — the fastest bear market on record — and the VIX hit a record **82.69** close on 16 March. Beneath the equity move was a textbook funding spiral: a scramble for cash drove haircuts wider and financing rates higher across markets, even in normally pristine collateral like Treasuries, forcing levered holders to sell into a market where everyone else was also selling. Correlations went to 1, both liquidity taps shut at once, and the forced deleveraging fed the very crash that was triggering the margin calls. It took central-bank intervention on a historic scale to break the loop — the same role the Fed played in 1998, stepping in from *outside* the spiral because no participant inside it could.

**The yen-carry unwind, 5 August 2024.** A crowded, levered funding-carry trade — borrow cheaply in yen, invest in higher-yielding assets — unwound in a matter of days when the yen reversed and the funding leg moved against everyone at once. The Nikkei fell **12.4%** in a single session, its worst day since 1987, and the VIX spiked intraday to **65.7**. The trade had paid a small, steady carry for a long time, and the cheap funding made it feel free — right up until the funding cost and the currency turned together, and the reflexive deleveraging compressed years of accumulated carry into a few catastrophic days. It is the pattern of this entire post: cheap funding that looks free in calm conditions is the same funding that detonates when the lender's terms turn against you all at once.

## The funding-survival playbook

Borrowing is not forbidden — used within strict limits it is a legitimate tool. The discipline is to make sure the lender can never start the spiral with you: to keep a buffer big enough that an ordinary shock never forces your hand, and to structure your financing so it cannot be yanked at the worst moment. Here is the concrete system.

**Keep a margin buffer — and treat dry powder as a position.** The margin call is lethal only because you have no cash to meet it. Hold a deliberate cash reserve so that an ordinary drawdown resolves as a survivable top-up, not a forced sale. The buffer preserves the one thing the lender tries to strip from you: the ability to choose when you exit. Cash earning nothing looks like a drag in a calm market; it is the difference between surviving and being liquidated in a spiral. A useful default: size so that you can absorb the largest plausible single-day move — a −20% day, a −34% month — with positive equity *and* without breaching maintenance margin, which from the buffer figure caps a typical book near 2×, not 3× or 5×.

**Stress your haircuts, not just your prices.** When you size a levered position, do not assume today's haircut. Re-run the position at a crisis haircut — 30% where today it is 5% — and ask whether you could meet the resulting margin requirement without a forced sale. If a haircut widening alone would force you to liquidate, the position is too large regardless of what the price does. The haircut is the lever the lender pulls first; stress it explicitly.

**Term your funding — match the maturity of your financing to the maturity of your trade.** The deepest fragility in a funding spiral is borrowing short to hold long: financing a multi-month thesis with overnight repo or callable margin that can be pulled tomorrow. Where you can, lock in longer-term financing, negotiate term repo, or use instruments whose leverage is embedded and cannot be margin-called daily. The more of your funding that is callable on short notice, the more of your survival you have handed to the lender's daily mood.

**Pre-commit to de-grossing, and do it before the lender does.** Decide in advance, in writing, at what drawdown you cut leverage — and cut it mechanically. A margin call, a widening haircut, a counterparty pulling your line: these all do the same thing, force you to sell into weakness. The entire game is to be the one who *chooses* to reduce, ahead of the ones who are *forced* to. The trader who halves their gross at a −15% drawdown survives to the rebound; the one who holds and hopes meets the forced sale at the bottom of the spiral.

**Diversify your lenders and watch for crowding.** A spiral is far worse when everyone in the same trade is financed the same way and gets called at the same time — that is what turns your forced selling into everyone's forced selling. Be wary of crowded, uniformly-financed positions, and do not concentrate all your financing with a single counterparty whose withdrawal would take you out in one move. The [exit game in crowded trades](/blog/trading/game-theory/case-study-ltcm-1998-the-crowded-genius-trade) is the strategic version of this same risk.

The thread back to the survival spine is direct. Leverage hands the timing of your exit to your lender, and the funding spiral is the mechanism by which they take it. You can only compound if you are still in the game, and the single most reliable way to be removed from the game — even when your thesis is right — is to let a price fall, a widening haircut, and a forced sale link up into a loop. Keep the buffer real, stress the haircut, term the funding, and pre-commit to de-grossing, and you keep the one thing that matters: the ability to wait, to choose your own exit, and to survive to trade tomorrow.

### Further reading

- [Liquidity risk: you can't sell what no one will buy](/blog/trading/risk-management/liquidity-risk-you-cant-sell-what-no-one-will-buy) — the market-liquidity side of the same crisis: vanishing bids, widening spreads, and the price impact of being forced to sell.
- [The liquidity-solvency doom loop: illiquid can become insolvent](/blog/trading/risk-management/the-liquidity-solvency-doom-loop-illiquid-can-become-insolvent) — how a funding problem becomes a solvency problem when forced selling marks the whole book to the fire-sale price.
- [Leverage and the arithmetic of ruin](/blog/trading/risk-management/leverage-and-the-arithmetic-of-ruin) — the recovery math the margin call pushes you up, from survivable into terminal.
- [Case study: LTCM 1998, the crowded genius trade](/blog/trading/game-theory/case-study-ltcm-1998-the-crowded-genius-trade) — the strategic anatomy of the most famous funding-spiral blow-up.
- [How hedge funds die: the failure taxonomy](/blog/trading/hedge-funds/how-hedge-funds-die-the-failure-taxonomy) — the firm-level view of how leverage, concentration, and forced selling end institutions.

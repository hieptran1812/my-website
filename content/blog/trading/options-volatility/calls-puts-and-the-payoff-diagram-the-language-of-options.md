---
title: "Calls, Puts, and the Payoff Diagram: The Language of Options"
date: "2026-06-17"
publishDate: "2026-06-17"
description: "Learn to read the four basic option payoffs at expiry, split a premium into intrinsic and time value, find every breakeven, and see why all structures are just sums of these four legs."
tags: ["options", "volatility", "payoff-diagram", "call-option", "put-option", "breakeven", "intrinsic-value", "time-value", "options-basics", "derivatives"]
category: "trading"
subcategory: "Options & Volatility"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A payoff diagram is the single picture that encodes everything an option does at expiry: where you make money, where you lose it, and the most you can win or lose. Learn to draw the four basic shapes and you can read any options strategy ever built, because every one of them is just these four added together.
>
> - There are only four single legs: **long call, short call, long put, short put.** Each is a hockey stick that kinks at the strike. Memorize the four shapes and the rest of options is arithmetic.
> - An option's premium splits into **intrinsic value** (what it is worth if expiry were today) plus **extrinsic value** (everything else — time and volatility). Extrinsic value is always at least zero and always decays to zero at expiry.
> - **Breakeven** is the strike adjusted by the premium: a call breaks even at strike **+** premium, a put at strike **−** premium. Below that you have not made money yet; you have only earned back what you paid.
> - **The one rule to remember:** when you *buy* an option your maximum loss is the premium and your upside can be large; when you *sell* one your maximum gain is the premium and your downside can be enormous — a naked short call has *unlimited* loss. Know which side of that trade you are on before you click.

In February 2018, a fund called LJM Preservation and Growth had spent years quietly selling options on the S&P 500. The strategy looked beautiful on a spreadsheet: collect a little premium every month, pocket it as the options expired worthless, repeat. For most of 2017 the market barely moved and the premiums rolled in. Then, over two trading days in early February 2018 — the episode the desks still call *Volmageddon* — volatility exploded. The VIX more than doubled, closing at 37.32 on February 5. LJM was *short* options, which means it was on the wrong side of exactly the payoff shape we are about to draw: the one where your gain is capped at the premium you collected and your loss is not capped at all. The fund lost about 80% of its value in those two days and shut down. The investors who lost their money had, in almost every case, never drawn the payoff diagram of what they owned.

That is the whole reason this post exists. An option is a contract whose value at a future date depends entirely on where one number — the price of some underlying thing — ends up. A *payoff diagram* turns that contract into a picture: stock price on the horizontal axis, your profit or loss on the vertical axis, and a line that tells you exactly what happens to your money at every possible ending price. Once you can read that picture fluently, you can never again be the person who sells a naked call without understanding that the loss extends off the top of the page. Every later topic in this series — the Greeks, volatility trading, spreads, hedging — is drawn on top of this one diagram. So we are going to make it your native language.

![Profit and loss at expiry for long call, short call, long put, and short put in a two by two panel](/imgs/blogs/calls-puts-and-the-payoff-diagram-the-language-of-options-1.png)

Look at the four panels above and notice the symmetry. The top row is calls; the bottom row is puts. The left column is the buyer (long); the right column is the seller (short). Every short payoff is its matching long payoff flipped upside down — that is the deep fact that makes options a zero-sum game between buyer and seller. Where the buyer's line is in the green, the seller's is in the red by exactly the same amount, and vice versa. Hold that mirror image in your head and you already understand half of options. The rest of this post is filling in the numbers.

## Foundations: what an option actually is

Before any diagram, the contract. An **option** is the *right, but not the obligation,* to buy or sell a fixed quantity of an underlying asset at a fixed price on or before a fixed date. Four nouns do all the work, so let us define each from zero.

The **underlying** is the thing the option is written on — a stock like Apple, an index like the S&P 500, a commodity, a currency. We will use a single stock trading at \$100 throughout this post and call its price *S*.

The **strike price** (or strike, written *K*) is the fixed price at which the contract lets you transact. A "100-strike call" lets you *buy* the stock at \$100 no matter where it actually trades. A "95-strike put" lets you *sell* the stock at \$95.

The **expiration** (or expiry) is the date the right runs out. After that the option is worthless or has been exercised; it no longer exists. We will use three months to expiry, written *T = 0.25* years, for most examples.

The **premium** is the price you pay (if you are buying the option) or receive (if you are selling it) for the contract today. This is the option's market price. It is small relative to the stock — a few dollars per share against a \$100 stock — which is exactly where the leverage, and the danger, comes from.

A **call** is the right to *buy* the underlying at the strike. You buy a call when you think the price will rise: the right to buy at \$100 becomes valuable if the stock goes to \$120, because you can buy at \$100 and immediately the position is worth \$20.

A **put** is the right to *sell* the underlying at the strike. You buy a put when you think the price will fall, or when you want insurance: the right to sell at \$95 becomes valuable if the stock crashes to \$60, because you can sell at \$95 something that is only worth \$60.

That is the entire vocabulary. One more convention: standard equity options are quoted *per share* but trade in *contracts of 100 shares.* A premium quoted as "\$4.49" means \$4.49 per share, so one contract costs \$449. Throughout this post we work in per-share dollars to keep the arithmetic clean; multiply by 100 (and by the number of contracts) to get the dollars that actually leave your account.

> [!note]
> **A quick analogy that will not lead you astray.** A call is a *deposit* that locks in a purchase price — like putting \$5,000 down to reserve the right to buy a house at \$500,000 next year. If the house soars to \$600,000 you exercise and pocket the difference; if it sags to \$400,000 you walk away and lose only the deposit. A put is *insurance* — like paying a premium so that if your \$100,000 boat sinks, the insurer pays you its agreed value. You hope you never use it, and the premium is gone either way, but it caps your loss. Buyers of options pay a deposit or a premium and risk only that; sellers collect it and take on the obligation. Hold those two images and the four payoffs below will feel inevitable.

### Exercise, assignment, and the two-sided nature of every contract

One subtlety to settle before we draw, because it confuses beginners and quietly shapes the diagrams. Every option has two parties: the **buyer** (the holder, who is *long*) and the **seller** (the writer, who is *short*). They are bound by the same contract and their P&L is exactly equal and opposite — which is why, in the very first figure, each short payoff was its long payoff flipped about the horizontal axis. The exchange and its clearinghouse sit in the middle guaranteeing the trade, but in payoff terms, every dollar the buyer makes comes from the seller and vice versa.

To **exercise** is to use the right: the call holder buys the stock at the strike; the put holder sells it at the strike. When a holder exercises, a seller somewhere is **assigned** — chosen, usually at random by the clearinghouse, to fulfill the obligation. As a buyer you control exercise; as a seller you cannot choose when you are assigned, only hope the option expires worthless so you never are. Almost nobody exercises early in practice — selling the option back to the market captures both its intrinsic *and* its remaining extrinsic value, while exercising throws the extrinsic value away — so for the purposes of the at-expiry diagram you can assume options are simply settled at their intrinsic value on the expiration date.

A second distinction you will hear: **American-style** options can be exercised any day up to expiry; **European-style** options only at expiry. US single-stock options are American; most cash-settled index options (like those on the S&P 500) are European. The difference rarely matters for the at-expiry payoff — which is identical for both — and it never changes the four shapes. It matters for *during-life* pricing and early-exercise edge cases, which is theory we [cross-link rather than re-derive](/blog/trading/quantitative-finance/options-theory). For this post, every diagram is the value *at expiry*, where American and European options agree.

### Moneyness: where the stock sits relative to the strike

The last bit of vocabulary you need to read a diagram fluently is **moneyness** — a one-word answer to "where is the stock relative to my strike right now?" It is the coordinate that tells you which part of the hockey stick you are sitting on.

- A call is **in-the-money (ITM)** when the stock is *above* the strike (S > K) — exercising would be profitable today. A put is ITM when the stock is *below* the strike (S < K).
- An option is **at-the-money (ATM)** when the stock sits right at the strike (S ≈ K) — at the kink.
- A call is **out-of-the-money (OTM)** when the stock is *below* the strike; a put is OTM when the stock is *above* it — exercising today would be pointless.

Moneyness governs almost everything about how an option behaves. Deep-ITM options act nearly like the underlying stock itself (a 70-strike call on a \$100 stock moves roughly dollar-for-dollar with the stock); deep-OTM options act like cheap lottery tickets (mostly worthless, occasionally explosive); and ATM options are the most sensitive, most actively traded, and carry the most time value — which is the hump we will see in the intrinsic/extrinsic figure later. When a trader says "I'm buying the 105 calls," the first thing to ask is the moneyness: with the stock at \$100, those are OTM, cheap, and a pure bet on a move; with the stock at \$120 they would be ITM and behave very differently.

### The payoff at expiry: where the diagram comes from

At expiry, time has run out and uncertainty collapses. The option is now worth exactly its **intrinsic value** — what you would get by exercising it right now. For a call that is `max(S − K, 0)`: if the stock is above the strike you capture the difference, otherwise you let it expire and get nothing. For a put it is `max(K − S, 0)`: if the stock is below the strike, your right to sell high is worth the gap, otherwise it is worthless.

Your *profit or loss*, the thing the diagram plots, is that intrinsic value minus what you paid (if you are long) or the premium you kept minus the intrinsic value you now owe (if you are short). That single subtraction — payoff minus premium — is what shifts the hockey stick up or down so that it crosses zero at the breakeven. Everything in the four-panel figure above is one of these two formulas with a premium subtracted or added. Let us walk each one.

A note on *why* we draw the diagram at expiry first, even though almost no one holds an option to its dying second. The at-expiry payoff is the **anchor** — the one shape that is pinned down by the contract terms alone, with no assumptions about volatility, interest rates, or time remaining. It is the same for an American and a European option, the same regardless of what the market thinks tomorrow, and it is what every pricing model must converge to as the clock runs out. The during-life value — the smooth curve — wobbles with every tick of implied volatility and every passing day, but it is always tethered to that fixed expiry shape, hovering above it (for longs) by the extrinsic value and collapsing onto it at the end. So we learn the anchor cold, then study how the curve floats above it. Get the four anchors into muscle memory and the rest of options is the study of one moving curve relative to four fixed lines.

The four positions below are organized along two binary choices, and that 2×2 is worth saying out loud because it is the whole taxonomy: **call or put** (do you have the right to buy or to sell?) crossed with **long or short** (did you buy the contract or sell it?). Buy a call, buy a put, sell a call, sell a put — four combinations, four shapes, and nothing else exists at the single-leg level. We will take them in that order.

## The long call: defined risk, large upside

You **buy a 100-strike call for \$4.49** because you think the stock, now at \$100, is going up. (That \$4.49 is the fair Black-Scholes price for a three-month at-the-money call when annualized volatility is 20% and the risk-free rate is 4% — we will use it consistently so the numbers tie out, and the [Black-Scholes derivation](/blog/trading/quantitative-finance/black-scholes) is one click away if you want the why.) Here is what happens at expiry across the range of ending prices.

If the stock finishes *below* \$100, your right to buy at \$100 is worthless — why exercise the right to pay \$100 for something worth \$95? You let it expire and you are out the \$4.49 premium. That is your **maximum loss**, and it is the same whether the stock ends at \$99 or at \$0. On the diagram this is the flat floor on the left: a horizontal line sitting at −\$4.49.

If the stock finishes *above* \$100, your right to buy at \$100 has intrinsic value. At \$105 the call is worth \$5; you paid \$4.49, so your profit is \$0.51. At \$110 the call is worth \$10 and your profit is \$5.51. The line rises one-for-one with the stock — every dollar the stock gains above the strike is a dollar in your pocket. There is no ceiling, which is why we say the long call has **unlimited upside** (bounded only by the stock going to infinity).

The point where the line crosses zero — where you have neither made nor lost money — is the **breakeven**. For a call it is the strike plus the premium, because you need the intrinsic value to first repay what you spent: `breakeven = K + premium = 100 + 4.49 = $104.49`. Below that you are still underwater even if the option has *some* intrinsic value, because that intrinsic value has not yet covered the premium.

![Long call payoff at expiry compared with the smooth during life value curve showing extrinsic value as the gap](/imgs/blogs/calls-puts-and-the-payoff-diagram-the-language-of-options-2.png)

The figure above adds the second thing you must understand about every option: the sharp hockey stick is only the picture *at expiry.* Before expiry — the dashed lavender curve — the P&L is a smooth bend, not a kink. The vertical gap between the two lines, shaded amber, is **extrinsic value**, also called time value. It exists because while time remains, the stock might still move your way; that possibility is worth something, and you paid for it. As expiry approaches, the curve sinks down onto the hockey stick and the gap closes. By the final second, the curve *is* the hockey stick. This is the single most important dynamic in the whole series, and it is why we keep saying an option is a bet on volatility and time, not just direction: you can be right about direction and still lose, because the time value you paid for bled away faster than the stock moved.

Look closely at the *slope* of that smooth curve and you meet the first Greek, the one that gives this series its backbone. Near the far left, where the stock is well below the strike, the curve is nearly flat — the option barely moves when the stock does, because it is unlikely ever to pay off. Near the far right, deep in-the-money, the curve runs at almost 45 degrees — the option moves nearly dollar-for-dollar with the stock. At the strike it is somewhere in the middle, roughly a half-slope. That slope is **delta**: the rate at which the option's value changes per \$1 move in the underlying, and equivalently the number of shares the option currently behaves like. Our ATM call has a delta of about 0.56, meaning that right now it gains about \$0.56 for every \$1 the stock rises and behaves like 56 shares of long stock. As the stock climbs, delta climbs toward 1.0 (the curve steepens); as the stock falls, delta sinks toward 0 (the curve flattens). The fact that delta *itself changes* as the stock moves — the curvature of that line — is the second Greek, *gamma*, and the reason the during-life curve bends instead of going straight. You do not need to compute either yet; just see that they are *features of this picture*, the slope and the bend of the smooth line, not abstract formulas. The next post in the series gives them their proper treatment.

One more reading of the same figure, because it pays off the whole-series thesis. The amber gap is widest at the strike and narrows as you move either way — exactly the hump we will formalize shortly. That gap is what you are *really* buying when you buy an option: not the stock's direction, but the *optionality* — the asymmetric right to participate in an upside while capping your downside. The price of that optionality is set by how much the market thinks the stock will move (implied volatility) and how long you have (time). When you hear that an option is "expensive" or "cheap," it almost never refers to the dollar premium — a \$0.50 lottery-ticket call can be wildly expensive and a \$40 deep-ITM call dirt cheap — it refers to whether the *extrinsic* portion, that amber gap, is rich or thin relative to the movement that actually shows up. Trading that gap is the entire game.

#### Worked example: pricing and the P&L of a long call

You buy one 100-strike call for a \$4.49 premium, so \$449 leaves your account for the contract (100 shares × \$4.49). Three months later:

- Stock at **\$95**: call expires worthless. P&L = `0 − 4.49 = −$4.49` per share, or **−\$449** on the contract. You lost the entire premium.
- Stock at **\$104.49** (the breakeven): call is worth `104.49 − 100 = $4.49`. P&L = `4.49 − 4.49 = $0`. Exactly flat.
- Stock at **\$115**: call is worth `115 − 100 = $15`. P&L = `15 − 4.49 = +$10.51` per share, or **+\$1,051** on the contract — a 234% return on the \$449 you risked.

Notice the asymmetry: you risked \$449 to make \$1,051, and you could have made far more if the stock ran higher. *That* is the leverage a long call gives you — but it only pays if the move is large enough and fast enough to beat the premium and the time decay. The intuition: buying a call is a deposit on the stock's upside, and the premium is what that deposit costs.

## The short call: capped gain, unlimited risk

Now flip to the other side of that same contract. You **sell the 100-strike call and collect \$4.49.** You have taken on the *obligation* to deliver the stock at \$100 if the buyer exercises. Your payoff is the exact mirror of the long call — the top-right panel in the first figure.

If the stock finishes below \$100, the buyer does not exercise, the option expires worthless, and you keep the entire \$4.49. That premium is your **maximum profit**, full stop. The best case for a seller is always just "I keep what I was paid."

If the stock finishes above \$100, you are on the hook. At \$110 you must deliver stock worth \$110 for \$100, a \$10 loss, against the \$4.49 you collected — a net loss of \$5.51. At \$130 the loss is `130 − 100 − 4.49 = $25.51`. At \$200 it is \$95.51. There is no ceiling: the more the stock rises, the more you lose, dollar for dollar, forever. This is **undefined risk** — the loss is theoretically unlimited. It is the LJM trade from the opening, and it is the single most dangerous shape in retail options.

The breakeven is the same \$104.49 as the long call, just approached from the other side: above it the seller is losing, below it the seller is winning.

> [!warning]
> A *naked* short call (one where you do not own the underlying stock) has genuinely unlimited loss. This is not a theoretical footnote. In a takeover or a short squeeze a stock can double overnight, and your "I collected \$449" can become "I owe \$10,000" before you can react. Most brokers require a large margin balance and high approval tier to sell naked calls precisely because of this shape. If you are ever tempted, draw the diagram first and find the right edge — there isn't one.

There is, however, a tamed version of this shape that is one of the most common positions in all of investing: the **covered call.** If you already *own* 100 shares of the stock and then sell a call against them, the unbounded loss disappears — because if the stock rockets and you are assigned, you simply deliver the shares you already hold. You give up the upside above the strike (your shares get called away at \$100 even if they are worth \$130), but in exchange you pocket the premium. We will see in the additivity section that a covered call is *literally* a long-stock payoff plus a short-call payoff added together, and the sum is a perfectly defined-risk position. The lesson to carry forward: the short call is dangerous *naked*, but harmless when it is covering stock you own — and the difference is visible the instant you draw the combined diagram.

## The long put: defined risk, large (but bounded) profit

You **buy a 95-strike put for \$1.60** because you fear the stock will fall, or because you own the stock and want a floor under it. (Again, \$1.60 is the fair price for the out-of-the-money three-month put under our standard assumptions.) The put is the right to *sell* at \$95.

If the stock finishes above \$95, why sell at \$95 something worth more? You let the put expire and lose the \$1.60 premium — your **maximum loss**, the flat floor on the bottom-left panel.

If the stock finishes below \$95, your right to sell at \$95 is worth the gap. At \$85 the put is worth \$10; minus the \$1.60 premium, your profit is \$8.40. At \$60 it is worth \$35 and your profit is \$33.40. The line rises as the stock *falls* — a put is a bet that goes up when the world goes down, which is exactly why it works as insurance.

Unlike the call, the put's profit is *large but bounded:* a stock can only fall to zero. The maximum the 95-strike put can ever be worth is \$95 (if the stock goes to \$0), so the maximum profit is `95 − 1.60 = $93.40` per share. Big, but finite.

The breakeven for a long put is the strike **minus** the premium, because the stock has to fall past the strike far enough to repay what you spent: `breakeven = K − premium = 95 − 1.60 = $93.40`. Below \$93.40 you are profiting; between \$93.40 and \$95 the put has intrinsic value but not enough to cover the premium yet.

#### Worked example: a put as a portfolio hedge in a crash

Suppose you own 100 shares of the stock at \$100 (a \$10,000 position) and you buy one 95-strike put for \$1.60 (\$160) as crash insurance. Now a shock hits and the stock falls to \$70 at expiry.

- Your **stock** is down `(70 − 100) × 100 = −$3,000`.
- Your **put** is worth `(95 − 70) = $25` per share, so `(25 − 1.60) × 100 = +$2,340` in P&L.
- **Net loss:** `−3,000 + 2,340 = −$660`, versus the **−\$3,000** you would have suffered unhedged.

The put converted a \$3,000 hole into a \$660 scratch, at a cost of \$160 when you bought it. That is the long-put payoff doing precisely the job the bottom-left panel promises: a cheap, defined-cost line that turns sharply profitable exactly when your other holdings are bleeding. The intuition: a put is insurance you buy on your own portfolio, and like all insurance it costs a small premium you hope to waste.

## The short put: capped gain, large downside

Sell the 95-strike put and **collect \$1.60.** You have taken on the obligation to *buy* the stock at \$95 if the buyer exercises — which they will do precisely when the stock has fallen below \$95 and they want to dump it on you at the higher strike. The payoff is the mirror of the long put (bottom-right panel), and we get a dedicated figure for it because the asymmetry trips up so many newcomers.

![Short put payoff at expiry with breakeven and maximum loss labelled](/imgs/blogs/calls-puts-and-the-payoff-diagram-the-language-of-options-5.png)

If the stock finishes above \$95, the put expires worthless and you keep the \$1.60 — your **maximum profit**, again just the premium. The line is flat across the entire upper range: a short put makes the same modest amount whether the stock ends at \$95 or at \$500. You do not participate in the upside at all; you only ever wanted the stock to stay above your strike.

If the stock finishes below \$95, you must buy at \$95 something now worth less. At \$85 you lose `(95 − 85) − 1.60 = $8.40`. At \$70 you lose \$23.40. In the absolute worst case — the stock goes to zero — you lose `95 − 1.60 = $93.40` per share. This is **large but bounded** downside (a stock cannot go below zero), which is the one mercy a short put has over a short call. But "bounded at \$93.40 against \$1.60 collected" is still a brutal risk-reward, and short puts blow up portfolios in crashes for exactly this reason.

The breakeven is again strike minus premium, `95 − 1.60 = $93.40`, approached from the other side: above it the seller wins, below it the seller loses.

#### Worked example: the short put that gets run over

You sell ten 95-strike puts for \$1.60 each, collecting `10 × 100 × 1.60 = $1,600`. It feels like free money — the stock is at \$100 and would have to fall 5% just to reach your strike. Then a bad earnings report drops the stock to \$78 at expiry.

- Each put is now intrinsic-worth `95 − 78 = $17` per share, against the \$1.60 you collected: a loss of `17 − 1.60 = $15.40` per share.
- Across ten contracts: `15.40 × 100 × 10 = −$15,400`.

You collected \$1,600 and lost \$15,400 — a loss nearly ten times the premium, from a move that was not even a crash. Plot it on the bottom-right panel and the lesson is obvious: the flat top is short and the downhill slope is long. The intuition: selling a put is selling insurance, and insurers get rich slowly and go bankrupt suddenly.

## Intrinsic and extrinsic value: how a premium splits in two

We have used the words; now we make them precise, because the split between intrinsic and extrinsic value is the engine of every options P&L.

The **intrinsic value** of an option is what it would be worth if expiry were *right now* — its immediate exercise value, never less than zero. For a call it is `max(S − K, 0)`; for a put `max(K − S, 0)`. An option with intrinsic value is **in-the-money** (ITM); one with none is **out-of-the-money** (OTM); one sitting right at the strike is **at-the-money** (ATM).

The **extrinsic value** (time value) is *everything the market charges above intrinsic value:* `extrinsic = premium − intrinsic`. It is the price of the *chance* that the option moves further into the money before expiry. Two forces feed it — how much time is left, and how violently the underlying tends to move (its volatility). More time and more volatility mean a wider range of possible ending prices, which means more chance the option pays off big, which means more extrinsic value. This is the seed of the entire series: **extrinsic value is the option's volatility-and-time content, and trading options is mostly trading extrinsic value.**

Two facts about extrinsic value are worth burning in:

1. **It is always at least zero.** A premium can never trade below intrinsic value, because if it did you could buy the option, exercise instantly, and pocket a riskless profit — arbitrageurs erase that gap in microseconds. (The formal statement of these no-free-lunch relationships lives in [put-call parity](/blog/trading/quantitative-finance/put-call-parity-no-arbitrage-quant-interviews).)
2. **It always decays to zero at expiry.** At T = 0 there is no time left and no chance of further movement, so extrinsic value vanishes and the premium equals intrinsic value exactly. That decay — the curve sinking onto the hockey stick — is *theta*, and it is the headwind every option buyer fights and every option seller harvests. We give theta its own post; here just see that it must happen.

![A call premium split into intrinsic and extrinsic value as a stacked area across stock price](/imgs/blogs/calls-puts-and-the-payoff-diagram-the-language-of-options-3.png)

The stacked figure shows the split for a 100-strike call across stock prices. The green band is intrinsic value — zero until the stock crosses \$100, then rising one-for-one. The amber band on top is extrinsic value, and notice its shape: it is fattest **near the money** and thins out as the option goes deep in or deep out. That is not an accident. Deep-OTM options are almost certainly going to expire worthless, so there is little chance to price; deep-ITM options behave almost like the stock itself, so there is little *optionality* left to charge for. The uncertainty — and therefore the time value — is concentrated where the outcome is most in doubt, right around the strike. Remember this hump: it is why at-the-money options are the most sensitive to volatility, a fact we will lean on in every later post.

#### Worked example: splitting three real premiums

Using our standard assumptions (σ = 20%, T = 0.25y, r = 4%), the Black-Scholes pricer gives these 100-strike call prices, which we split by hand:

| Stock price *S* | Premium | Intrinsic = max(S−100,0) | Extrinsic = premium − intrinsic |
|---|---|---|---|
| \$100 (ATM) | \$4.49 | \$0.00 | **\$4.49** |
| \$110 (ITM) | \$11.78 | \$10.00 | **\$1.78** |
| \$130 (deep ITM) | \$31.01 | \$30.00 | **\$1.01** |

The ATM call is *all* extrinsic value — you are paying \$4.49 purely for the chance of a favorable move. The \$110 call is mostly intrinsic with \$1.78 of time value left. The \$130 call is almost entirely intrinsic; only \$1.01 of optionality remains, because the option already moves nearly dollar-for-dollar with the stock. The intuition: the further from the strike you go, the more an option becomes "just the stock" or "just a lottery ticket," and the less you are paying for genuine optionality.

## Reading any payoff diagram in ten seconds

You now have everything you need to read these diagrams at a glance. Here is the checklist that turns a strange-looking chart into a sentence:

- **The axes.** *x* is always the price of the underlying at expiry; *y* is always your profit or loss. The point where *y* = 0 is breakeven; left of zero on *y* is a loss, right is a profit.
- **The kink(s).** Every kink sits at a strike. A single-leg position has one kink; a spread has two; a butterfly has three. Count the kinks and you know how many strikes are involved.
- **The flat sections.** A flat line means your P&L stops changing — you have hit a maximum gain or a maximum loss in that region. A flat *floor* is defined risk (you cannot lose more); a section that keeps sloping off the page is undefined risk.
- **The slopes.** A line sloping up to the right means you are net *long* the underlying's direction in that region (you want the stock higher); sloping down means net *short*. The steepness is roughly your *delta* — how many shares the position behaves like — which is the subject of the next post.
- **At-expiry vs. during-life.** The straight, kinked line is the payoff *at expiry.* Before expiry the line is a smooth curve sitting above it (for long positions) by the amount of remaining extrinsic value. As time passes the curve melts onto the straight line. Always know which one you are looking at.

Run that checklist on the four panels in the first figure and each becomes a one-liner: "long call — lose the premium below \$104.49, profit without limit above." "Short put — keep \$1.60 above \$93.40, lose a lot below." That fluency is the entire goal of this post.

![Breakeven anatomy of a long call showing strike premium and breakeven labelled on the payoff line](/imgs/blogs/calls-puts-and-the-payoff-diagram-the-language-of-options-4.png)

The anatomy figure above labels the three numbers that define any single-leg call: the **strike** (where the kink sits), the **premium** (how far the floor drops below zero — your max loss), and the **breakeven** (strike plus premium, where the line crosses back to zero). Internalize that a call's line is built from exactly these three inputs and you can sketch any call payoff on a napkin from the quote alone.

#### Worked example: reading a payoff you have never seen before

Suppose a colleague shows you a P&L line they have on the screen and says nothing about what it is. You apply the checklist. The line is flat at +\$2.40 for all stock prices below \$95; it then slopes *downward* between \$95 and \$105, passing through zero at \$97.40; and it flattens again at −\$7.60 for all prices above \$105. What is it?

- **Flat sections at both ends** → the maximum gain and maximum loss are both finite, so this is a *defined-risk* position (no unlimited tails).
- **Two kinks** (at \$95 and \$105) → two strikes are involved; this is a two-leg structure, not a single option.
- **Profitable on the left, losing on the right** → it makes money when the stock *falls* and loses when it rises, so it is a bearish position.
- **Max gain +\$2.40, max loss −\$7.60, breakeven \$97.40** → the gain is the net credit received and the loss is the strike width (\$10) minus that credit.

Putting it together: this is a **bear call spread** — sell the 95 call, buy the 105 call, collect a net \$2.40, risking the \$10 gap minus the credit. You decoded an unlabeled structure in four reads, without anyone telling you what it was, purely from the geometry. The intuition: a payoff diagram is self-documenting — the kinks, the flats, and the breakeven *are* the trade, and learning to read them is learning to read options.

## Max profit, max loss, and which side is dangerous

Let us tabulate the four legs precisely, because "defined risk vs. undefined risk" is the distinction that decides whether a position can quietly end your trading career. The numbers use our standard premiums (100-strike call \$4.49, 95-strike put \$1.60).

| Position | Max profit | Max loss | Breakeven | Risk class |
|---|---|---|---|---|
| **Long call** | Unlimited | −\$4.49 (premium) | \$104.49 (K + prem) | Defined risk |
| **Short call** | +\$4.49 (premium) | **Unlimited** | \$104.49 (K + prem) | **Undefined risk** |
| **Long put** | +\$93.40 (K − prem) | −\$1.60 (premium) | \$93.40 (K − prem) | Defined risk |
| **Short put** | +\$1.60 (premium) | −\$93.40 (K − prem) | \$93.40 (K − prem) | Large but bounded |

Three things to take from this table. First, **buyers always have defined risk** — the most a long option can lose is the premium, because you can always just walk away from a right. Second, **sellers always have capped profit** — the most a short option can make is the premium collected, because that is all anyone paid you. Third, and most important, **the short call is the only one with truly unlimited loss,** because a stock's price has no ceiling. The short put is bad but bounded; the short call is the genuinely open-ended one. When someone says "selling options is like picking up nickels in front of a steamroller," the short call is the steamroller.

This is also why brokers gate these positions behind approval tiers: buying calls and puts is the lowest tier (defined risk), cash-secured short puts a step up, and naked short calls the highest tier of all (undefined risk). The payoff diagram is literally the risk-management framework.

## Combining legs: every structure is a sum

Here is the payoff that, once you see it, makes the entire universe of options strategies collapse into something learnable. **A multi-leg position's payoff is just the vertical sum of its legs' payoffs.** At each stock price, add up the P&L of every leg you hold, and the total is your combined payoff. That is all a "strategy" is — a chosen set of the four basic legs whose payoffs add up to a shape you want.

This sounds almost too simple to be the foundation of an entire industry, but it is. The exotic-sounding menagerie of options strategies — verticals, calendars, butterflies, iron condors, collars, ratio spreads, risk reversals — is nothing more than different combinations of the same four legs (and sometimes a position in the underlying stock, which is just a 45-degree line through the origin). A trader who has memorized the four hockey sticks and the rule "add them vertically" can construct or decompose any of those structures with a pencil. There is no fifth shape to learn, ever. When this series gets to iron condors and butterflies, we will not be teaching you new objects; we will be teaching you which legs to add and *why* — what view of volatility and direction each sum expresses. The mechanical part you already know after this section.

The reason the addition works so cleanly is worth stating explicitly: your total profit is, by definition, the sum of the profits on each contract you hold, and each contract's profit at a given expiry price does not depend on the others. So the combined payoff *function* is literally the pointwise sum of the leg payoff functions. There is no interaction term, no cross-effect — \$1 of profit on the call plus \$1 of loss on the put is exactly \$0 of combined P&L at that price. That independence is what lets you reason about a four-leg iron condor by sketching four lines and adding their heights at each x, and it is why the payoff diagram is such a powerful tool: complex positions reduce to simple addition.

![A bull call spread shown as a long call plus a short call payoff adding to a capped profit shape](/imgs/blogs/calls-puts-and-the-payoff-diagram-the-language-of-options-6.png)

The figure above builds a **bull call spread** — a bet that the stock rises modestly — out of two legs we already understand. Buy the 100-strike call (\$4.49, the green hockey stick) and simultaneously sell the 110-strike call (\$1.14, the red mirror). Add the two lines at every price and you get the blue shape: a position that costs a net \$3.35, profits as the stock rises past \$100, but *caps* its profit at \$110 because above that the short call's losses exactly offset the long call's further gains. You gave up the unlimited upside in exchange for a cheaper entry. Every spread, condor, butterfly, collar, and straddle in this series is exactly this operation — pick legs, add the lines, read the resulting shape.

![A long straddle payoff computed by summing a long call leg and a long put leg](/imgs/blogs/calls-puts-and-the-payoff-diagram-the-language-of-options-7.png)

The last figure proves the additivity claim a second way with a **long straddle** — buy the 100 call (\$4.49) *and* the 100 put (\$3.49), a \$7.98 bet that the stock moves a lot in *either* direction. The dashed green and amber lines are the two legs; the solid blue line is their sum at every price. You can read the combined breakevens straight off the chart — \$92.02 on the downside and \$107.98 on the upside (the strike ± the total premium) — and the maximum loss of \$7.98 right at the strike, where neither leg has any intrinsic value. Notice what the straddle is *really* a bet on: not direction, but **magnitude.** It is the purest expression of the series' thesis — a position that is long volatility and short time, profiting only if the stock moves more than the \$7.98 of premium implies. We will trade exactly this structure later when we talk about earnings and the [expected move](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options).

#### Worked example: the bull call spread P&L

You buy the 100 call for \$4.49 and sell the 110 call for \$1.14, a **net debit of \$3.35** (\$335 per spread). At expiry:

- Stock at **\$100 or below**: both calls expire worthless. P&L = `−$3.35` per share, your **max loss** (just the net premium paid).
- Stock at **\$105**: long call worth \$5, short call worthless. P&L = `5 − 3.35 = +$1.65`.
- Stock at **\$110 or above**: long call worth \$10, short call costs you `S − 110`. The two further-out moves cancel, so P&L is capped at `(110 − 100) − 3.35 = +$6.65` per share, your **max profit**.

You risked \$3.35 to make \$6.65 — a clean, defined-risk, roughly 2-to-1 bet on a move to \$110, costing 25% less than buying the call outright (\$3.35 vs. \$4.49). The intuition: selling the upside you do not expect to need pays for most of the call you do want — that is the whole trade in spreads.

#### Worked example: the covered call as long stock plus a short call

The additivity rule also tames the dangerous short call, and seeing it as a sum makes the safety obvious. You own 100 shares bought at \$100 and you sell the 105 call against them, collecting \$2.39 (the fair price of that OTM call under our assumptions). Add the two legs at expiry:

- Stock at **\$95**: stock leg `95 − 100 = −$5.00`; short call expires worthless, you keep `+$2.39`. Combined: `−$5.00 + 2.39 = −$2.61` per share.
- Stock at **\$105**: stock leg `+$5.00`; short call still worthless, keep `+$2.39`. Combined: `+$7.39` per share.
- Stock at **\$130**: stock leg `+$30.00`; but the short 105 call now costs you `(130 − 105) = $25.00` against the \$2.39 collected. Combined: `30.00 − 25.00 + 2.39 = +$7.39` per share — *capped*.

Above \$105 the position is flat at +\$7.39: every extra dollar the stock gains is exactly clawed back by the short call, because your shares get called away at \$105. The unbounded loss of the naked short call has vanished — added to long stock, the dangerous leg becomes a defined-risk income position whose worst case is just owning the stock down to zero, cushioned by the premium. The intuition: a covered call trades away your upside above the strike for cash today, and the summed payoff shows that trade exactly — a capped, slightly cushioned version of just holding the stock.

## Common misconceptions

**"Buying calls is cheap leverage — it's basically free upside."** The premium is not the cost; *time decay* is. That \$4.49 ATM call has a theta of about −\$0.027 *per day* and rising as expiry nears — roughly \$2.70 per contract evaporating daily even if the stock does nothing. Over the three-month life, an option that stays at-the-money loses its entire \$4.49 to decay. Calls are not "free upside"; they are a wager that the stock will move enough, soon enough, to outrun a clock that is always running against you. The number to remember: an ATM option that does not move loses ~100% of its premium by expiry.

**"My option is in the money, so I'm making money."** Being in-the-money means positive *intrinsic* value, not positive *P&L*. Our 100-strike call at a stock price of \$104 is \$4 in-the-money but you paid \$4.49 — you are still down \$0.49 per share. You do not profit until the stock clears the **breakeven** (\$104.49), which is always further out than the strike for a long position. Confusing "in-the-money" with "profitable" is how people hold winners that are still losers.

**"Selling options is safe because most options expire worthless."** It is true that a large majority of options expire worthless — that is the seller's edge, and it is real (see the variance risk premium below). But "wins often" and "safe" are different statements. The short-put example above lost \$15,400 against \$1,600 collected in a single ordinary earnings move; the LJM fund in the intro won for years and then lost 80% in two days. The payoff diagram tells you why: the seller's gain is capped and the loss is not. A strategy can win 90% of the time and still have negative expectancy if the 10% losses are large enough — the math of which is exactly [position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion).

**"A naked short call and a short put are about equally risky."** No — and this is the one that gets people margin-called. The short put's loss is *bounded* because the stock can only fall to zero (max loss \$93.40 here). The naked short call's loss is genuinely *unbounded* because the stock can rise without limit. They look like mirror images on the page, but the call's loss line runs off the top forever while the put's bottoms out at the x-axis. Never treat them as symmetric risks.

**"The payoff diagram tells me what my position is worth today."** It tells you what it will be worth *at expiry.* Today — before expiry — your position sits on the *smooth curve* above (for longs) or below (for shorts) the kinked line, separated by the remaining extrinsic value. A long call that is at-the-money two months before expiry can be down money even though the at-expiry diagram shows it sitting right at the kink, because two months of time value has decayed. Always ask: am I reading the expiry line or the during-life curve?

## How it shows up in real markets

**Volmageddon, February 2018.** The opening story is the cleanest real-world short-volatility blowup of the modern era. Funds like LJM and products like the XIV exchange-traded note were, in payoff terms, *short* options on the S&P 500 — collecting premium, capped gain, unbounded loss. For all of 2017 the VIX averaged just 11.1, the calmest year on record, and the strategy printed money. On February 5, 2018 the VIX closed at 37.32, more than doubling in days. The short-vol positions hit the steep part of their loss curve simultaneously; XIV lost ~96% of its value overnight and was liquidated, and LJM lost ~80% and shut down. Every one of those losses lived on the right edge of a short-option payoff diagram that the holders had never drawn.

**The variance risk premium — why sellers exist at all.** If short options are so dangerous, why does anyone sell them? Because they get paid to. Historically, S&P 500 one-month *implied* volatility (what option premiums price in) averages around 19.5 vol points while the *realized* volatility that actually follows averages around 15.8 — a structural gap of roughly **+3.7 vol points** that option sellers harvest as profit, on average, in exchange for bearing crash risk. This is the **variance risk premium**, the single most important structural edge in options, and it is the reason the seller's "wins often, loses big" payoff can still be a real business when sized correctly. We devote a whole post to trading [implied versus realized volatility](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush); for now, just know that the seller's dangerous payoff is *compensated* danger.

**The GameStop squeeze, January 2021 — the short call's right edge made real.** If you ever doubted that the short call's loss line truly runs off the top of the page, the meme-stock squeeze settled it. Anyone who had sold naked calls on GameStop expecting it to drift, having seen it trade in the low single digits and then around \$20, watched it close near \$347 on January 27, 2021. A trader short the \$60 calls for a few dollars of premium was suddenly facing nearly \$287 of intrinsic value *per share* against them — a loss of roughly 50× to 100× the premium collected, exactly the unbounded geometry of the top-right panel. Some of those positions were force-liquidated by brokers as margin evaporated. The squeeze is remembered as a story about retail traders and short sellers, but for our purposes it is the cleanest live demonstration that "the short call has no right edge" is a literal, account-ending fact, not a textbook caveat.

**Earnings and the long straddle — buying the move, fighting the crush.** The long straddle we summed in the additivity section is the workhorse position for betting on a big move around a catalyst like an earnings report. The trap is that *everyone* knows the catalyst is coming, so the market bids up implied volatility — and therefore the straddle's extrinsic value — *before* the event. You can be exactly right that the stock will gap, watch it gap, and *still lose money* if the move is smaller than the rich premium you paid, because the moment the news is out, implied volatility collapses (the "vol crush") and the extrinsic value you overpaid for evaporates. The straddle's two breakevens, strike ± total premium, are precisely the bar the move has to clear. This is the purest illustration of the series thesis — direction was right, but you were long an overpriced bet on *magnitude* — and we devote a whole post to pricing it via the [expected move](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options).

## The playbook: how to trade the four legs

You came here to read diagrams; you leave able to *trade* them. Here is the practitioner's playbook, leg by leg.

**Before any trade, draw the payoff.** Literally sketch the four numbers: strike, premium, breakeven, and — most importantly — *where is the right edge of my loss?* If the loss line runs off the page (a naked short call), you are taking unbounded risk and must size for it or refuse the trade. This single habit would have saved most of the people in the case studies above.

- **Long call** — when you have a directional-up view *and* expect a move large enough and soon enough to beat the premium and decay. The Greek profile is long delta, long vega (you want volatility to rise), short theta (time is your enemy). Entry: buy when implied volatility is *low* relative to the move you expect. Invalidation: the thesis is wrong if the stock stalls — time decay will grind you out even if you are eventually right.

- **Long put** — when you want downside exposure or, more commonly, a *hedge* on stock you own. Same Greek profile as the long call but short delta. Size it as insurance: you are *supposed* to lose the premium most of the time, and you are paying for the tail. Invalidation: a put is a poor *speculative* short if you are paying a fat volatility premium into a calm market — you may be right on direction and still lose to decay.

- **Short put** — when you are willing to own the stock at the strike and want to get paid to wait, *and* implied volatility is elevated so the premium is rich. This is the famous "cash-secured put." Size it by the **bounded** worst case (you could be forced to buy the stock at the strike), never by the premium collected. Invalidation: stop treating it as free money the moment volatility spikes — that is when the loss curve is steepest.

- **Short call** — *only* covered (you own the underlying), essentially never naked unless you are a professional with the margin and the hedging to manage unbounded risk. The covered call caps your stock's upside in exchange for premium income — itself just a long-stock payoff plus a short-call payoff, summed. Invalidation: if you find yourself selling naked calls "because they expire worthless," re-read the Volmageddon section and the right edge of the payoff diagram.

The through-line: **buyers pay a defined premium for a chance at a large move; sellers collect a defined premium for taking on an open-ended obligation.** The payoff diagram tells you, before you risk a dollar, exactly which of those you are and where the danger lives. Every later post in this series — the Greeks, the volatility surface, spreads and condors, dealer gamma, hedging — is drawn on top of these four shapes. You now speak the language. Next we give those slopes and curves their names: the Greeks.

## Further reading & cross-links

- [Black-Scholes, derived from scratch](/blog/trading/quantitative-finance/black-scholes) — where the \$4.49 fair premium and the smooth during-life value curve actually come from.
- [Put-call parity and no-arbitrage](/blog/trading/quantitative-finance/put-call-parity-no-arbitrage-quant-interviews) — the proof behind "extrinsic value is never negative" and how calls and puts lock together.
- [Options pricing fundamentals](/blog/trading/quantitative-finance/options-theory) — a complementary, theory-first treatment of the same contracts.
- [The expected move: pricing event risk with options](/blog/trading/event-trading/the-expected-move-pricing-event-risk-with-options) — what the straddle we summed is really betting on.
- [Implied vs. realized volatility and the vol crush](/blog/trading/event-trading/event-volatility-implied-vs-realized-and-the-vol-crush) — why option sellers get paid and how the variance risk premium works.
- [Volatility as an asset class](/blog/trading/cross-asset/volatility-as-an-asset-owning-fear) — the VIX, owning fear, and the short-vol blowups in the case studies.
- [Position sizing and the Kelly criterion](/blog/trading/technical-analysis/position-sizing-and-kelly-criterion) — the math that decides whether "wins often, loses big" is a business or a time bomb.

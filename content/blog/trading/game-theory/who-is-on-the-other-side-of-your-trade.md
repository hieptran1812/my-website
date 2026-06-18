---
title: "Who Is on the Other Side of Your Trade?"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "Every fill pairs you with a specific counterparty, and knowing which of the five it is tells you more about your edge than any chart pattern ever will."
tags: ["game-theory", "trading", "market-microstructure", "counterparty", "order-flow", "adverse-selection", "market-making", "informed-trading"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 40
---

> [!important]
> **TL;DR** — A trade is not a bet against the market; it is a deal with one specific person, and your edge is mostly a function of *who* that person is and *why* they showed up.
>
> - Every fill pairs you with one of five counterparties: a **hedger** offloading risk, an **informed trader** who knows something you don't, a **noise/retail trader** acting on whim or cash needs, a **market maker** renting you liquidity for the spread, or a **forced seller/buyer** who simply must transact.
> - Each type is a *different game* with a *different payoff*. Trading against noise and forced flow is where edge lives; trading against the informed is where accounts die.
> - **Your fill quality is a confession.** A resting limit order that fills instantly and the price keeps running was probably taken by someone informed — toxic flow. A fill that sits and waits is more likely benign.
> - The one discipline to keep: after every fill, ask *"who took the other side, and what does that tell me?"* If you can't answer, you don't know your edge — you just have a position.

In late January 2021, thousands of retail traders bought call options on GameStop, a struggling video-game retailer, and the stock ran from around \$20 to an intraday \$483. Most of the story was told as a David-versus-Goliath morality play. But strip away the narrative and ask the only question that actually pays: when a retail trader clicked "buy" on a \$300 call, *who sold it to them?* The answer was usually a market maker — and the moment that market maker sold the call, it was short gamma, which forced it to buy stock to hedge, which pushed the stock higher, which forced it to buy more. The retail buyers thought they were beating "the system." They were, for a few weeks, beating a *specific counterparty* whose hedging rules were mechanical and predictable. When the rules changed — when brokers restricted buying and the forced hedging unwound — the same predictability ran in reverse.

That is the whole lesson of this post in one episode. You never trade against "the market." You trade against a person or a machine with a name, a reason, and a payoff. A *counterparty* is simply whoever takes the other side of your order — the seller when you buy, the buyer when you sell. Your profit is, almost always, their loss or their cost; markets do not print free money. So the most important number in trading is not a price target or a moving average. It is the answer to a question most traders never ask: **who is on the other side, and why are they doing this?**

![Counterparty taxonomy from you to the trade to five types](/imgs/blogs/who-is-on-the-other-side-of-your-trade-1.png)

The diagram above is the mental model for this entire series. Your order goes into the market and gets matched against exactly one of five kinds of counterparty. Each one implies a different game. Get the identity right and you know whether the fill you just got is a gift or a trap. Get it wrong and you will spend years confusing luck for skill, right up until the informed traders quietly take it all back. This is the question that opens the "Markets as Games" track, and it is the question the rest of the series spends its time answering for specific situations: the squeeze, the auction, the dealer's quote, the panic. Let's build the taxonomy from zero.

## Foundations: the five counterparties and the game each one plays

Before we can reason about who is on the other side, we need a few plain definitions. A *market* is a place where buyers and sellers meet to exchange an asset for money. An *order* is your instruction to buy or sell — a *market order* says "fill me now at whatever the going price is," while a *limit order* says "fill me only at this price or better." The *bid* is the highest price someone is currently willing to pay; the *ask* (or *offer*) is the lowest price someone is willing to sell at; the gap between them is the *bid-ask spread*. When your order executes, it is called a *fill*, and whoever was on the other side of that fill is your counterparty.

Game theory is the study of strategic interaction — situations where your best move depends on what other people do, and theirs depends on what you do. A *game* is defined by four things: the **players**, the **strategies** each can choose, the **payoffs** each gets for every combination of choices, and the **information** each player has. A trade is a game in exactly this sense. The players are you and your counterparty (and often a market maker in between). The strategies are buy, sell, wait, widen, pull. The payoffs are the profit or loss each side books. And the information is the crux: do you know something they don't, or do they know something you don't?

The single most useful idea in this whole series is that **the trade you place is not one game — it is five different games wearing the same costume**, and which game you are actually in depends entirely on the type of counterparty you drew. Here are the five.

### The hedger — trading to offload risk, not to win

A *hedger* trades to reduce risk they already carry, not to make a profit on the trade itself. A farmer who has planted wheat and won't harvest for six months is exposed to the price of wheat falling. To remove that risk, the farmer sells wheat futures today, locking in a price. The farmer does not care whether wheat goes up or down afterward — the goal was to stop caring. An airline buys oil futures to fix its fuel cost. A pension fund buys bonds to match its future payouts. A company that earns euros but reports in dollars sells euros forward to neutralize the currency swing.

The defining feature of the hedger is that they are **price-insensitive within reason** — they will pay a little extra to get the hedge done, because the hedge is insurance, not a bet. They are not trading because they think the price is wrong. They are trading because they want to *stop being exposed*. This makes them **benign flow**: when a hedger takes the other side of your trade, they are not doing it because they have information about where the price is going. They are doing it to sleep at night. The game here is close to a simple, fair exchange — you provide the liquidity they need, and you get compensated for it without being on the wrong end of a secret.

### The informed trader — knows something you don't

An *informed trader* trades because they possess information the price has not yet reflected. This is the counterparty to fear. The information can be many things: a hedge fund that has modeled next week's earnings beat, an analyst who just downgraded a stock and is selling ahead of the report, an insider who knows a deal is coming (illegal in most cases, but it happens), or simply a faster, smarter participant who has figured out that order flow is about to turn. When an informed trader takes the other side of your trade, you are, by definition, on the wrong side of better information.

This is *toxic flow* — toxic because trading against it loses money on average. The informed trader only trades with you when the trade is good *for them*, which means it is bad for you. If they are buying from you, they expect the price to rise; if they are selling to you, they expect it to fall. The game against an informed trader is a game of *asymmetric information*, and asymmetric information games have a brutal property: the side with less information should often refuse to play. The whole architecture of market making — the spread, the size limits, the pulling of quotes — is built to survive the constant presence of informed traders. We will spend a great deal of this series on the informed trader, because almost every way you lose money cleanly traces back to having unknowingly traded against one. The next post in the series, on [adverse selection and the winner's curse](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news), is dedicated to exactly this.

It is worth being precise about what "informed" means, because the word is broader than insider trading. An informed trader is anyone whose reason for trading correlates with the price's *future* move. That includes the obvious cases — a fund that has done genuine fundamental research, a quant shop whose model just flipped — but also subtler ones. A high-frequency firm that sees order flow a few milliseconds before you do is informed about the very-short-term direction. A trader who has noticed that a large forced seller is about to hit the market is informed about supply. Even a trader who simply *reacts faster* to public news than you do is, for the seconds that matter, the informed counterparty. Information in markets is not only secret facts; it is any edge in *knowing the next move before the price does*, whether that edge comes from research, speed, or sheer attention. The defining test is always the same: after they trade with you, does the price tend to move *their* way? If yes, they were informed, no matter how they got there.

### The noise/retail trader — trading on whim or need

A *noise trader* trades for reasons unrelated to information about value. The name comes from the idea that their trading is "noise" laid on top of the "signal" of informed flow. A retail trader who buys a stock because a friend mentioned it, or because it was in the news, or because they have \$2,000 of new savings to invest this month, is a noise trader. So is a fund that must sell some holdings to meet a redemption, or an index fund mechanically buying whatever its benchmark holds. Noise traders are not stupid — they may be perfectly rational people pursuing goals that have nothing to do with predicting the next tick.

This is **the flow you want**. When a noise trader takes the other side of your trade, they are not doing it because they know something. They are doing it because they had cash to deploy or cash to raise. On average, you are not adversely selected against this flow, so you can earn the spread, the small edge, the mean reversion — without the secret information lurking on the other side. Market makers love noise flow; their entire business model is to capture the spread from noise traders while defending against informed ones. The challenge, as we will see, is that you cannot easily *tell* noise flow from informed flow in the moment — they both just look like a fill.

### The market maker — renting you liquidity for the spread

A *market maker* (also called a *dealer* or *liquidity provider*) is a participant whose business is to always be willing to buy and sell, quoting both a bid and an ask, and earning the spread between them. When you place a market order to buy, the market maker often sells to you at the ask; when you sell, they buy from you at the bid. They do not (ideally) take a directional view — they want to buy at the bid and sell at the ask many times, pocketing the spread and ending the day flat. They are the *intermediary*: a counterparty whose goal is to facilitate your trade and rent you immediacy, not to bet against you.

The market maker's game is the most interesting of all, because it sits at the center of every other game. The market maker *wants* your trade — but only if you are a noise or hedger. The market maker *fears* your trade if you are informed, because then the spread they earn is smaller than the loss they take when the price moves against the inventory they just acquired from you. The entire craft of market making is reading, in real time, whether the order that just hit them was benign or toxic, and adjusting the spread and size accordingly. We cross-link out to [how an options market maker thinks](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) for the dealer's full inner monologue; for now, hold the idea that the market maker is the counterparty who is *also* trying to figure out who you are.

### The forced seller/buyer — must transact, no matter the price

A *forced* participant must transact for a reason outside their control, on a schedule that is often known in advance. A trader who bought stock on margin — with borrowed money — and whose position has fallen far enough triggers a *margin call*: a demand from the broker to add cash or have the position liquidated. If they cannot add cash, the broker sells the position at market, right now, regardless of price. An index fund must buy a stock the day it is added to the index and sell a stock the day it is removed, at the closing price, because its mandate is to track the index. A fund facing large redemptions must raise cash by a deadline. A volatility-targeting fund must sell when volatility spikes, mechanically.

Forced flow is **predictable and therefore exploitable**. The forced seller is not trading because they think the price is too high — they are trading because a rule, a margin clerk, or a calendar said so. This makes the flow readable: you often know *that* it is coming, *roughly how much*, and *by when*. The game against forced flow is the opposite of the game against the informed: instead of fearing what they know, you anticipate what they must do. Position ahead of a known liquidation or a known rebalance and you can profit from the temporary, information-free pressure they create. This is one of the cleanest edges in markets, precisely because it has nothing to do with predicting value.

### Why this is a poker question, not a forecasting question

Here is the cleanest way to feel why the counterparty matters more than the chart. Take a hand of poker. A beginner stares at their own cards and asks "is my hand good?" A professional barely glances at their cards and instead asks "what does the *betting* tell me about the other players' hands, and what kind of player is each one?" The cards are the chart. The opponents are the counterparties. The beginner who only reads their own cards is the trader who only reads their own chart — and both lose to the player who reads the table.

In poker, when a tight, conservative player who has folded all night suddenly shoves their whole stack, you fold a strong hand, because the *identity of the bettor* tells you their hand is stronger than yours. The bet is benign when a loose, reckless player makes it (they bet on anything) and toxic when the rock makes it (they only bet the nuts). The bet looks identical on the felt; the only difference is who made it. A fill is exactly the same: the order that crossed with yours looks identical whether it came from a noise trader or an informed one. Your entire job is to read, from the available tells, which kind of player just acted.

This is why the best trading firms recruit poker players and run their training like a poker room: the skill being trained is not prediction, it is *opponent modeling*. Predicting where a price goes is forecasting nature, and nature does not adapt to you. Reading who is on the other side is playing a game against an adversary who adapts — and that is a fundamentally different, and more winnable, problem. You do not need to know the future. You need to know who is sitting across the table, and reason one level deeper than they do.

Notice what this reframing does to the idea of "edge." A trading *edge* is not a crystal ball; it is a structural reason that the counterparties you face are, on average, worse-informed or more-constrained than you. A casino's edge over a gambler is not that the casino predicts the dice — it is that the rules guarantee the gambler is on the wrong side of the odds. Your edge as a trader is similar: it is a structural guarantee that you are systematically matched against benign and forced flow rather than informed flow. Lose that structural guarantee and your "strategy" stops working, no matter how good the chart looks, because the chart never knew who was on the other side in the first place.

That is the taxonomy. Five counterparties, five games. Now let's make the central claim of the post concrete: the *same trade* has a completely different expected value depending on which of these five took the other side.

## The same trade, five different expected values

Here is the idea that should reorganize how you think about every fill. Imagine — and the chart below makes this literal — that you place the exact same order, with the exact same chart setup, five times. The only thing that changes is *who* is on the other side. The expected value of that identical trade swings from strongly positive to strongly negative purely as a function of counterparty identity.

To see this we need one piece of arithmetic. The *expected value* (EV) of a trade is the probability-weighted average of its outcomes: multiply each possible profit or loss by its probability, then add them up. If a trade wins \$30 with 55% probability and loses \$20 with 45% probability, its EV is `0.55 × 30 + 0.45 × (−20) = 16.5 − 9 = +7.50` per unit. Positive EV means the trade makes money on average; negative EV means it bleeds. We lean on the EV machinery developed in [expected value, edge, and variance](/blog/trading/game-theory/expected-value-edge-and-variance-thinking-like-the-house) — here we just apply it five times.

![Expected value of a trade by counterparty type](/imgs/blogs/who-is-on-the-other-side-of-your-trade-3.png)

The chart computes the EV of one stylized trade — capture a small edge, occasionally take a loss — against each of the five counterparties. The win/loss outcomes differ because the counterparty's *reason for trading* differs. Read the bars left to right: against a forced seller you make the most, because they hand you a discount for liquidity they desperately need; against noise and hedgers you make a solid, steady edge; the market maker just clips a thin spread; and against the informed, the same trade is deeply negative, because they only trade with you when they are right.

#### Worked example: the same setup, five counterparties

You buy 100 shares at \$50.00 on a setup you like. Let's price the EV of that purchase against each counterparty, in dollars per 100 shares.

- **Against a forced seller** (a margin liquidation dumping shares cheap): you win \$40 with 70% probability (the panic discount reverts) and lose \$25 with 30% probability. EV `= 0.70 × 40 + 0.30 × (−25) = 28 − 7.5 = +\$20.50`.
- **Against a noise/retail seller** (someone raising cash): win \$30 at 55%, lose \$20 at 45%. EV `= 0.55 × 30 + 0.45 × (−20) = +\$7.50`.
- **Against a hedger** (offloading risk): win \$25 at 50%, lose \$15 at 50%. EV `= +\$5.00`.
- **Against a market maker** (renting you the spread): win \$8 at 80%, lose \$22 at 20%. EV `= +\$2.00`.
- **Against an informed seller** (who knows the stock is about to drop): win \$15 at 30%, lose \$45 at 70%. EV `= 0.30 × 15 + 0.70 × (−45) = 4.5 − 31.5 = −\$27.00`.

Same chart, same \$50.00 entry, same 100 shares. The EV ranges from `+\$20.50` to `−\$27.00` — a swing of `\$47.50` per 100 shares — and *nothing changed except the identity of the person who sold to you*. The intuition to burn in: your chart does not know who is on the other side, but your bank account does.

This is why "the other side" is the right unit of analysis. Two traders can take the identical technical setup, and the one who is systematically matched against forced and noise flow gets rich while the one systematically matched against informed flow goes broke — using the *same chart*. The edge was never in the pattern. It was in the counterparty.

## Toxic versus benign: the flow spectrum

The five counterparties are not five isolated boxes; they sort along a spectrum. At one end sits **benign flow** — counterparties trading for reasons that have nothing to do with your edge. At the other end sits **toxic flow** — counterparties trading precisely *because* the price is about to move against you. The single most important skill in trading is reading where on this spectrum your fill came from.

![Flow spectrum from benign to toxic](/imgs/blogs/who-is-on-the-other-side-of-your-trade-2.png)

The grid lays out the spectrum. On the benign (left) side, the noise trader and the hedger are not betting on the price at all; your edge *grows* the more of this flow you trade. On the toxic (right) side, the informed trader knows the price will move; your edge *shrinks* the more you face them. The forced seller sits interestingly in the middle — they trade because they must, not because they know, so their flow is benign in information terms but unusually *aggressive*, which makes it both exploitable (predictable) and dangerous (it can crater the price short-term before reverting).

The word *toxic* is the actual term traders and exchanges use. Exchanges measure "order flow toxicity" with metrics like VPIN — the volume-synchronized probability of informed trading — because for a market maker, toxic flow is the difference between a profitable day and a blown-up book. When you hear a desk say "that flow was toxic," they mean: the orders we filled came from someone who knew more than we did, and the price ran against us right after. Benign flow is the opposite — you fill it and the price doesn't punish you for it.

Here is the uncomfortable truth that makes this hard: **benign and toxic flow look identical at the moment of the fill.** Both are just an order that crossed with yours. You cannot see the counterparty's reason. You can only infer it — from the size, the speed, the time of day, the instrument, and above all from *what the price does immediately after you trade*. That last clue is so important it deserves its own section.

#### Worked example: how much benign flow you need to survive toxic flow

Suppose you are a small market maker earning a half-spread of \$0.02 on each benign trade — you buy at the bid, sell at the ask, and pocket two cents per round-trip share. But some fraction of your fills are informed, and when you trade with the informed, the price moves \$0.50 against you on average before you can react. How toxic can the flow get before you lose money?

You break even when the expected gain from benign flow equals the expected loss from toxic flow. If a fraction `f` of orders are informed, the math is `f × 0.50 = (1 − f) × 0.02`. Solving: `0.50f = 0.02 − 0.02f`, so `0.52f = 0.02`, giving `f = 0.02 / 0.52 ≈ 0.0385`, or about **3.85%**.

That is a startling number. If just **3.85%** of your fills come from informed traders, your two-cent edge on the other 96% is wiped out. A market maker who cannot keep toxic flow below ~4% of volume is dead — and that is why making markets is so much about *avoiding the informed* rather than *predicting the price*. The intuition: a tiny sliver of toxic flow, because each toxic trade hurts 25 times as much as a benign trade helps, dominates the whole P&L.

## Your fill quality is a confession

You cannot see your counterparty's face. But you can see your fill — and the *quality* of that fill leaks information about who took the other side. This is one of the most practical ideas in microstructure, and almost no retail trader uses it.

Start with a resting *limit order* — an order to buy or sell at a specific price that sits in the order book waiting for someone to cross it. When you post a resting limit order to buy at \$50.00, you are offering liquidity to the market: anyone who wants to sell *right now* can hit your bid. The question is, who hits it, and how?

There are two very different ways your order can fill, and they tell opposite stories.

![Fill quality reveals the counterparty as a decision flow](/imgs/blogs/who-is-on-the-other-side-of-your-trade-4.png)

The flow above lays out the inference. **Path one: your order fills instantly, and the price keeps moving in the direction the seller was going.** You posted a bid at \$50.00, it got hit immediately, and within seconds the stock is at \$49.80 and falling. Think about what that means. Someone was so eager to sell *right at your price* that they took your bid the instant it appeared — and then the price kept dropping, which means they were *right* to sell. That is the signature of an informed seller. They wanted out before the move, and you were the convenient buyer standing at \$50.00 with your hand up. Your "good fill" was the informed trader's exit. This is *adverse selection*: the orders that fill fastest are disproportionately the ones you should not have wanted.

**Path two: your order sits in the book, fills slowly, and the price does not run after.** You posted your bid at \$50.00, it sat there for a while, eventually someone sold to you, and the stock stayed around \$50.00 afterward. Nobody was in a hurry to take your price. That patience is the signature of benign flow — a noise trader raising cash, a hedger offloading risk, neither of whom had a reason to rush. The fill that *sits* is the fill you want.

The rule that falls out of this is deeply counterintuitive and worth tattooing on your forearm: **a fill that is too good is a warning, not a gift.** When your limit order fills instantly at a great price and the market immediately keeps going, you did not outsmart anyone — you got selected. The next post covers exactly this dynamic; for now, internalize that fill quality is a confession, and you should read it on every trade.

#### Worked example: pricing the cost of a fast fill

You post a limit buy for 1,000 shares at \$50.00. Consider two scenarios with the same fill price but different stories.

- **The fill that sits.** Your order rests for 90 seconds, then fills. Over the next minute the stock drifts between \$49.98 and \$50.04. The counterparty was benign — say a fund rebalancing. Your mark-to-market a minute later is roughly flat, maybe `+\$10` to `−\$20` of noise on 1,000 shares. No information was traded against you.
- **The instant fill.** Your order fills the moment you post it, and within 60 seconds the stock is at \$49.50. You bought 1,000 shares at \$50.00 that are now worth \$49.50 — a paper loss of `1,000 × (50.00 − 49.50) = \$500`. The counterparty was informed; the speed of the fill was the tell.

Same \$50.00 fill price. One costs you nothing; the other costs you `\$500` in under a minute. The only difference was *how fast and how cleanly the order filled* — which is exactly the variable that encodes the counterparty's information. The intuition: speed of fill is negatively correlated with fill quality, because the most eager counterparty is usually the most informed one.

## The you-versus-informed game has a depressing equilibrium

Let's formalize the worst game in the taxonomy — you facing an informed trader — and find its *equilibrium*. An equilibrium is a stable outcome where no player can improve by unilaterally changing their strategy; it is the resting point a repeated, rational game converges to. The equilibrium of the you-versus-informed game explains why spreads exist at all, and why they widen exactly when you most want to trade.

Set it up as a *payoff matrix* — a grid showing what each player earns for every combination of choices. You are a small liquidity provider. Your two strategies: post a **tight** quote (a narrow spread, good prices, inviting trades) or post a **wide** quote (a fat spread, defensive prices). The informed trader's two strategies: **trade** against your quote, or **pass** and wait. We compute the equilibrium with `data_gametheory.nash_2x2`, which finds the stable strategy combinations of a two-by-two game.

![You versus informed payoff matrix with the Nash equilibrium](/imgs/blogs/who-is-on-the-other-side-of-your-trade-5.png)

Read the matrix. The four cells are the four ways the game can resolve, and the colored cell is the *Nash equilibrium* — the stable outcome. Here is the logic, cell by cell. If you post tight and the informed trades, you are adversely selected: you lose 8 and they win 8 — this is the trap. If you post tight and they pass, you earn the small spread from the noise traders also in the market: you make 2. If you post wide and they trade, your spread was fat enough to still profit: you make 3 (but they would lose 3, so they won't do it). If you post wide and they pass, you collect the wide spread from noise flow undisturbed: you make 4.

#### Worked example: solving the you-versus-informed game

Walk the equilibrium logic with the payoffs in the matrix.

- **Is posting tight ever your best move?** Compare your two rows. If the informed trades, tight gives you `−8` versus wide's `+3` — wide is better by `\$11`. If the informed passes, tight gives you `+2` versus wide's `+4` — wide is better by `\$2`. So **posting wide beats posting tight no matter what the informed does.** In game theory, tight is a *dominated strategy* — you should never play it once the informed might be present.
- **Given that you post wide, what does the informed do?** If they trade, they earn `−3` (they lose); if they pass, they earn `0`. So they **pass**.
- **The equilibrium** is therefore (you post wide, informed passes), the green cell, where you earn `+\$4` and they earn `\$0`. `nash_2x2` confirms this is the unique pure equilibrium.

Now sit with what that means. The *stable* outcome of facing possible informed flow is a **wide spread that the informed never crosses**. You are protected — but everyone else, including the benign noise and hedger flow, now pays your fat spread too. The mere *possibility* of an informed trader forces the spread wide for *everyone*. The informed trader doesn't even have to show up; the threat is enough. This is the deep reason markets are not free: the spread is the toll the benign pay to insure the market maker against the toxic. The intuition: in a game of hidden information, the uninformed side rationally defends, and that defense is a cost borne by the innocent.

## The spread is the price of adverse selection

The you-versus-informed game told us *that* the spread must be wide; the Glosten-Milgrom model tells us *how* wide, as an exact function of how much toxic flow is in the market. This is one of the foundational results in market microstructure, and `data_gametheory` computes it directly.

The setup is the simplest possible: an asset is worth either a high value or a low value — say \$110 or \$90 — with equal odds, so its fair "mid" price is \$100. A fraction of arriving traders are informed (they know which value is correct and trade accordingly); the rest are noise (they buy or sell at random). A competitive market maker, who cannot tell informed from noise in the moment, must set its bid and ask so that it breaks even *given* the toxic flow mixed in. The result: the market maker quotes an ask equal to the expected value *conditional on someone buying*, and a bid equal to the expected value *conditional on someone selling*. Because a buy is slightly more likely to come from an informed trader who knows the value is high, the ask sits above the mid; symmetrically, the bid sits below. The gap between them — the spread — is the market maker's protection against being picked off.

![Glosten-Milgrom spread rises with the share of informed flow](/imgs/blogs/who-is-on-the-other-side-of-your-trade-7.png)

The chart traces the spread the market maker must charge as the fraction of informed traders rises from 0% to 50%. The relationship is clean and linear: with no informed flow the spread is zero — a market maker facing only noise would, in this idealized model, quote for free. As toxic flow climbs, the spread widens in lockstep, because every extra point of informed flow is another point of expected loss the spread has to cover.

#### Worked example: the spread at 30% informed

Set the value at \$110 or \$90, equal odds, and 30% of arriving orders informed. Run `glosten_milgrom(v_high=110, v_low=90, p_high=0.5, frac_informed=0.30)`:

- The model returns a **bid of \$97.00** and an **ask of \$103.00**, a **spread of \$6.00** around the \$100 mid.
- Why? When someone *buys*, the market maker updates toward "this might be an informed buyer who knows it's worth \$110," so the break-even ask is pulled up to \$103. When someone *sells*, it updates toward \$90, pulling the bid down to \$97.
- At 10% informed the spread is \$2.00; at 20% it is \$4.00; at 30% it is \$6.00; at 50% it is \$10.00 — exactly linear in the toxic fraction, as the chart shows.

The takeaway you can use: when you see a spread suddenly widen on a name — say it goes from a penny wide to a dime wide — the market is telling you the *perceived fraction of informed flow just jumped*. Someone, somewhere, is suspected of knowing something. A widening spread is the market maker raising the toll because it smells toxicity. The intuition: the spread is not a fee for a service, it is an insurance premium against the informed, priced in real time.

## Forced sellers are predictable, and predictable means exploitable

We have spent most of this post on flow you should *fear* — the informed. Now flip it: the flow you can *anticipate*. Forced participants are the gift of the taxonomy, because their trades are driven by rules, deadlines, and triggers you can often see coming. The game against forced flow is not about reading hidden information; it is about reading a calendar and a margin schedule.

![Forced selling timeline from trigger to liquidation](/imgs/blogs/who-is-on-the-other-side-of-your-trade-6.png)

The timeline above traces a textbook forced sale. A price drop pushes an account's equity below its *maintenance margin* — the minimum cushion a broker requires on a borrowed-money position. The broker issues a *margin call*. If the trader cannot deposit cash, the broker liquidates the position at market, at a time and size chosen by the rule, not by anyone's view of value. The same shape appears in *index rebalancing*: when a stock is added to or removed from a major index, every fund tracking that index must buy or sell it on the rebalance date, at the closing price, mechanically. And in *redemptions*: a fund hit by withdrawals must raise cash by a deadline.

What unites all of these is that the seller is **not trading because they think the price is wrong** — they are trading because they have no choice. That makes the flow information-free and *direction-known*. You know it is selling pressure, you know roughly how much, and you often know when. The price impact is real but temporary: once the forced flow clears, the price tends to revert, because nothing about the asset's value changed. This is the cleanest non-informational edge in markets.

#### Worked example: fading a forced liquidation

A leveraged trader holds 10,000 shares of a stock bought at \$50.00 on 50% margin, meaning they borrowed \$250,000 of the \$500,000 position. The broker's maintenance margin is 30%. The stock falls to \$36.00.

- Position value is now `10,000 × 36.00 = \$360,000`. The loan is still `\$250,000`, so equity is `360,000 − 250,000 = \$110,000`. Equity as a fraction of position value is `110,000 / 360,000 ≈ 30.6%` — right at the maintenance line.
- The stock dips one more dollar to \$35.00. Now equity is `350,000 − 250,000 = \$100,000`, which is `100,000 / 350,000 ≈ 28.6%`, **below** the 30% requirement. Margin call. The trader can't add cash, so the broker dumps all 10,000 shares at market.
- That forced sale pushes the price down further, say to \$33.50 on the liquidation, as 10,000 shares hit a thin book. But nothing about the company changed — this was pure mechanical selling. A trader who anticipated the liquidation and bought into the panic at \$33.50 might see the stock revert to \$35.50 once the forced flow cleared: a `2.00 / 33.50 ≈ 6%` gain on `\$33,500` of stock per 1,000 shares bought, or roughly `+\$2,000` per 1,000 shares.

The forced seller handed you a discount for liquidity they could not refuse to take. The intuition: when the seller has no choice and no information, the price they accept is a gift, not a signal.

A word of discipline, because this edge has a sharp edge of its own: forced flow is *predictable in direction* but *dangerous in magnitude*. A cascade of margin calls can push a price far below fair value and keep it there longer than your capital can wait — the "the market can stay irrational longer than you can stay solvent" problem. Fading forced flow is profitable *on average* and across many instances, but any single liquidation can overshoot violently. Size for the overshoot, not for the average.

## Common misconceptions

**"I'm trading against the market, not a person."** The market is not a counterparty — it is a *venue* where counterparties meet. Every share you buy was sold by a specific entity with a specific reason. "The market went down" is shorthand for "more participants with reasons to sell than to buy met at a lower price." Treating the market as an impersonal force you can outguess is the first error; it is people, and people have tells.

**"A great fill means I'm a great trader."** Often the opposite. The best fills — instant execution at a price better than you expected — are disproportionately the ones where an informed counterparty was desperate to trade at your price *because they knew it was about to be wrong*. A fill that feels too easy should raise your eyebrows, not your confidence. We made this concrete above: the instant fill that cost `\$500` in a minute looked, at the moment of execution, like a gift.

**"Retail flow is dumb money I can pick off."** Be careful. Yes, individual retail orders are usually noise — but the *aggregate* of retail flow can be informed in disguise, and it can move markets through reflexive mechanisms (the GameStop gamma squeeze was retail flow forcing dealers to hedge). More importantly, you are usually not the one filling retail flow — the market maker is, and they pay for the privilege. If you think you are picking off retail, double-check you are not actually the one being picked off by the desk in the middle.

**"Spreads are just a fee brokers charge to gouge me."** No — the spread is mostly an *insurance premium against adverse selection*, as the Glosten-Milgrom math shows. A market maker quoting a \$6 spread on a \$100 stock with 30% informed flow is not gouging; it is pricing the expected loss from the informed traders mixed into the flow. Where competition is high and toxic flow is low, spreads collapse toward zero (think a penny wide on a megacap). Where toxic flow is high or competition is thin, spreads blow out — not from greed, but from risk.

**"If I just had a better chart pattern, I'd win."** The five-EV figure is the rebuttal. The identical pattern is worth `+\$20.50` against forced flow and `−\$27.00` against informed flow. The pattern is not the edge; *who you systematically trade against* is the edge. Strategies that work do so because they route you toward benign and forced flow and away from informed flow — even when their inventors describe them in the language of chart shapes. The honest version of this is covered in [smart money concepts, honestly](/blog/trading/technical-analysis/smart-money-concepts-honestly).

**"More liquidity is always better for me."** Not when the liquidity is toxic. A market that looks deep and tight can be a trap if much of that apparent depth is informed traders waiting to pick you off the moment you cross the spread. Conversely, a wider market dominated by benign and forced flow can be a far better place to trade. What you want is not maximum liquidity but *the right counterparty mix* — lots of noise and forced flow, little informed flow. Depth and tightness are proxies for liquidity, not for safety; the question that actually matters is who is providing that depth and why.

**"The big institution on the other side must know more than me."** Sometimes — and sometimes the exact opposite. A pension fund rebalancing \$50 million into bonds is a giant, but it is a *hedger/forced* giant with no view on your timing. An index fund buying a newly added stock is enormous and completely uninformed about value. Size is not information. The mistake is to assume that because the counterparty is large, they must be smart money; many of the largest flows in the market are the most benign, precisely because they are mechanical. Judge the counterparty by their *reason* for trading, not their size.

## How it shows up in real markets

**GameStop, January 2021 — retail forcing the dealer's hand.** Retail call buying made market makers short gamma, which mechanically forced them to buy stock as it rose, amplifying the move from ~\$20 to an intraday \$483 on January 28. The retail buyers were, briefly, trading against a counterparty (the dealer) whose hedging was rule-bound and predictable. When the rules of the game changed — Robinhood and others restricted buying on January 28 — the forced flow reversed, and the stock fell back toward \$40 within days. The episode is a clinic in identifying the counterparty: the edge was never the stock, it was the dealer's mechanical hedging. The [dealer gamma](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) mechanics are the engine room of this story.

**Volkswagen, October 2008 — the forced buyers were short sellers.** Porsche had quietly accumulated control of most of Volkswagen's freely-traded shares via options. When it disclosed this on October 26, 2008, the short sellers — who had borrowed and sold VW stock expecting it to fall — discovered there were almost no shares left to buy back. They were *forced buyers* with no choice, and the price exploded from around €210 to over €1,000 intraday on October 28, briefly making VW the most valuable company in the world. The lesson: the most violent moves come when one side of the trade is *forced* and the other side knows it. If you can identify a forced counterparty, you can stand in front of the move — or get out of its way.

**LTCM, 1998 — being the predictable forced seller.** Long-Term Capital Management ran enormous leveraged convergence trades. When Russia defaulted in August 1998 and spreads blew out, LTCM faced margin calls and had to unwind. Because its positions were large, concentrated, and *known* to the street, other desks could see what LTCM would be forced to sell and traded ahead of it, deepening the losses. LTCM was the forced seller in the timeline above, and the rest of the market played the fading game against it. The Fed organized a \$3.6 billion bailout in September 1998. The lesson cuts both ways: being the *identifiable* forced participant is the worst position in markets, because everyone games your liquidations.

**Index rebalancing — the most scheduled flow on earth.** When S&P announces an addition to the S&P 500, index funds must buy the stock by the effective date. Studies have long documented an "index inclusion effect": added stocks tend to rise into the rebalance as front-runners buy ahead of the forced index demand, then give some of it back afterward. This is forced flow at its purest — direction known, size estimable, date on a calendar. Desks build entire strategies around anticipating index buys and sells, with no view on the companies whatsoever.

**The 2010 Flash Crash — when benign liquidity fled and only toxic remained.** On May 6, 2010, a large automated sell program hit a thin market; market makers, unable to distinguish the selling from informed flow, pulled their quotes to avoid adverse selection. With the benign liquidity providers gone, prices for some stocks momentarily collapsed to a penny. The event is a vivid demonstration of the you-versus-informed equilibrium: when the *perceived* fraction of toxic flow spikes, market makers rationally widen to infinity (pull quotes), and the spread the rest of the market pays becomes the entire price range. Liquidity is not a constant — it is the equilibrium of a game, and it vanishes exactly when the game turns toxic.

**Crypto liquidations — forced flow you can watch in real time.** Crypto exchanges offer enormous leverage, and when the market moves, cascades of liquidations fire automatically. Unlike equities, much of this is *visible*: liquidation data, funding rates, and open interest are published, so the forced flow is not just predictable in principle — you can literally watch the margin pressure build. On days like the May 2021 crash, billions of dollars of leveraged long positions were force-liquidated in hours, driving prices far below where they settled once the forced selling exhausted itself. The counterparty on the other side of those liquidations was, definitionally, forced — and the traders who understood that were buying the information-free overshoot, not panicking with the crowd. The lesson generalizes: wherever forced flow is *observable*, the anticipation game is at its most playable.

**The SIG/Susquehanna way — building a firm around counterparty identification.** Proprietary trading firms that grew out of poker culture, like Susquehanna, are explicitly built around the question this post asks. Their training drills the habit of asking, on every trade, *who is on the other side and why would they take this price?* — the same expected-value, opponent-modeling discipline you use at a poker table. We cross-link the full story in [the SIG/Susquehanna playbook](/blog/trading/quant-careers/sig-susquehanna-playbook-poker-game-theory-and-ev); the point here is that the best trading cultures institutionalize the counterparty question rather than treating it as an afterthought.

## The playbook: how to play it

Here is how to turn the taxonomy into a discipline you run on every trade.

**Who is on the other side?** Before and after every fill, name the likely counterparty. Are you trading thin, news-driven flow where the informed lurk? Or deep, scheduled, mechanical flow — an index rebalance, a known liquidation, end-of-month rebalancing — where forced and noise flow dominate? You will rarely *know* for certain, but the act of asking changes which trades you take. If your honest answer is "probably someone who knows more than me," that is a trade to skip or shrink.

**What game are you in?** Map the situation to one of the five games. Facing possible informed flow, you are in the defensive you-versus-informed game: widen, shrink, or stand aside. Facing forced flow, you are in the anticipation game: position ahead of the known liquidation or rebalance and fade the overshoot. Facing noise and hedger flow, you are in the liquidity-provision game: post resting orders and earn the spread. The strategy that wins one game loses another, so identify the game first.

**Read your fills as confessions.** Build the habit of marking what the price did in the 60 seconds after each fill. Fills that filled fast and ran against you are toxic — log them and ask what you missed. Fills that sat and stayed flat are benign — that is your real edge, and you want more of it. Over a few hundred trades, this log will tell you whether you are systematically being adversely selected, which no win-rate statistic can reveal. (A trader can win 70% of trades and still lose money if the 30% are all toxic fast-fills.)

#### Worked example: the fill-quality scoreboard

Keep a simple two-column tally over 100 fills. For each fill, mark whether the price one minute later had moved *against* you by more than a tick (toxic) or stayed flat-to-favorable (benign). Suppose you find 18 toxic and 82 benign fills. Now attach the typical dollar move: the toxic fills averaged a `−\$0.40` adverse move on 1,000 shares (a `−\$400` mark per fill) and the benign fills earned your `+\$0.02` half-spread (`+\$20` per fill).

- Toxic damage: `18 × (−\$400) = −\$7,200`.
- Benign income: `82 × (+\$20) = +\$1,640`.
- Net over 100 fills: `−\$7,200 + \$1,640 = −\$5,560`.

You won 82% of your fills and *still* lost `\$5,560`, because the 18% toxic tail dwarfed everything else — exactly the asymmetry the 3.85% break-even revealed. The scoreboard, not the win rate, is the truth. The intuition: track the *cost* of your toxic fills, not the *count* of your benign ones, because one informed counterparty undoes dozens of noise ones.

**Your edge and where it lives.** Your durable edge is structural: it comes from systematically trading against benign and forced flow while avoiding informed flow. That means favoring liquid, scheduled, mechanical situations over thin, news-driven, information-rich ones. It means earning the spread from noise rather than guessing direction against the informed. It means fading forced sellers, not chasing informed buyers. None of this requires predicting the future — it requires reading the counterparty.

**The invalidation.** Your read on the counterparty is wrong when the price keeps moving against you after the fill and *does not revert*. If you bought from a "forced seller" and the price keeps falling on growing volume with no bounce, you misread it — that was informed flow, not forced flow, and your fade is broken. Cut it. The single cleanest invalidation in this whole framework: *benign and forced flow revert; informed flow does not.* If your fade does not revert, you were the sucker.

**Sizing and exit.** Size inversely to your suspicion of toxicity. When you cannot rule out informed flow, size small or stand aside — the informed only need to be right occasionally to ruin an oversized position, as the 3.85% break-even showed. When you are confident the flow is forced or benign, you can size up, because the downside is temporary impact, not permanent information. Exit forced-flow fades into the reversion (don't get greedy waiting for full mean-reversion — overshoots can recur). Exit any position the moment your counterparty read is invalidated.

The whole discipline reduces to one repeated question, asked on every single trade: **who took the other side, and what does that tell me?** Traders who ask it relentlessly are playing the game on the screen. Traders who never ask it are the counterparty everyone else wants to find. That is the spine of this entire series — a trade is a strategic interaction, not a bet against nature — and this post is where it becomes concrete. Every post that follows zooms into one specific version of this question: the squeeze, the auction, the dealer's quote, the panic, the signal. They are all the same question. Who is on the other side?

## Further reading & cross-links

- [The trade is a game: why markets are strategic, not random](/blog/trading/game-theory/the-trade-is-a-game-why-markets-are-strategic-not-random) — the series opener that frames every trade as a strategic interaction; this post makes its central question concrete.
- [Expected value, edge, and variance: thinking like the house](/blog/trading/game-theory/expected-value-edge-and-variance-thinking-like-the-house) — the EV machinery we applied five times here, developed from first principles.
- [Adverse selection and the winner's curse: why a fast fill is bad news](/blog/trading/game-theory/adverse-selection-and-the-winners-curse-why-a-fast-fill-is-bad-news) — the next post, dedicated entirely to the fill-quality-is-a-confession idea.
- [How an options market maker thinks: the other side of your trade](/blog/trading/options-volatility/how-an-options-market-maker-thinks-the-other-side-of-your-trade) — the dealer's full inner monologue and how they price toxic flow.
- [Smart money concepts, honestly](/blog/trading/technical-analysis/smart-money-concepts-honestly) — an honest look at what "the institutions are on the other side" claims do and don't mean.
- [The SIG/Susquehanna playbook: poker, game theory, and EV](/blog/trading/quant-careers/sig-susquehanna-playbook-poker-game-theory-and-ev) — a firm built entirely around the counterparty question.

*This is educational material about market mechanics and game theory, not financial advice. Every instrument that can make money can lose it; size for the case where your read on the counterparty is wrong.*

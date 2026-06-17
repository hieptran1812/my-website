---
title: "Zero-Sum, Positive-Sum, and the House: Where Trading Profits Come From"
date: "2026-06-18"
publishDate: "2026-06-18"
description: "The single most important question before any trade is which kind of game you are in, because that decides whether time is your ally or the house's."
tags: ["game-theory", "trading", "zero-sum", "positive-sum", "expected-value", "transaction-costs", "investing", "derivatives", "market-structure"]
category: "trading"
subcategory: "Game Theory"
author: "Hiep Tran"
featured: true
readTime: 41
---

> [!important]
> **TL;DR** — Before you ask whether a trade will work, ask what kind of game it is, because the game decides whether the person on the other side has to lose for you to win.
>
> - **Derivatives and FX are zero-sum**: every long is matched by a short, so one side's gain is the other's loss, and the whole table nets to exactly nothing before costs.
> - **Equities are positive-sum over the long run**: companies create real value through earnings, dividends, and buybacks, so buy-and-hold owners can all come out ahead together.
> - **Trading frictions are negative-sum**: spreads, commissions, slippage, borrow fees, and taxes are the house's rake, and they make the average active trader's game *worse* than a coin flip.
> - **The one rule:** in a zero- or negative-sum game you win only by being better than a specific named counterparty; in a positive-sum game, time itself is doing the work for you.

In 1999, a friend of mine put \$10,000 into an S&P 500 index fund and forgot about it. He checked it maybe twice a year, never traded, paid almost no fees. Another friend, same year, same \$10,000, opened a brokerage account and started day-trading the same kinds of stocks — in, out, in, out, chasing every wiggle. He was smart. He read everything. He was, by his own honest accounting, *right* about market direction more often than he was wrong.

By 2024 the index-fund friend had a little over \$80,000 without lifting a finger. The day-trader, despite being right more than half the time, had less than he started with. He had not been unlucky. He had not picked terrible stocks. He had simply spent twenty-five years playing a different game than the one he thought he was playing — and the game he was actually in was rigged against him by a force he never put on his spreadsheet.

That force has a name, and it is the subject of this whole post. The deepest question in trading is not "will this go up?" It is **what kind of game am I in, and who has to lose for me to win?** Get that wrong and no amount of being right about prices will save you. Get it right and you may discover, like my first friend, that you were never really fighting anyone at all.

![Three games hide inside the word trading: a zero-sum derivative, a positive-sum equity, and a negative-sum active trader](/imgs/blogs/zero-sum-positive-sum-and-the-house-where-trading-profits-come-from-1.png)

The diagram above is the mental model for everything that follows. The word "trading" hides three completely different games. In the first, your gain is carved directly out of someone else's loss and the table never grows. In the second, the table itself grows over time, so everyone seated at it can leave richer. In the third — the one most active traders actually sit down to — a quiet rake skims a little off every hand, so the average player loses even when no single opponent beats them. Same word, three different fates. Let us build each one from zero.

## Foundations: what a "sum" of a game even means

Before we can say a game is zero-sum or positive-sum, we need to be precise about three plain words: *players*, *payoffs*, and the *sum*. None of these requires any finance background. We will define each with money you can count.

A **player** is anyone whose choices affect the outcome and who cares how it turns out. In a trade, the players are you, the person who took the other side of your trade (your *counterparty* — the specific party who agreed to the opposite of what you did), and, lurking in the background, the *house*: the broker, the exchange, the market maker, and the tax authority, each of whom takes a cut for letting the game happen at all.

A **payoff** is the dollars a player walks away with, counted *relative to where they started*. If you buy something for \$100 and later it is worth \$130, your payoff from that position is +\$30. If your counterparty sold you that same thing and it rose, their payoff is whatever the mirror of your gain is. The crucial habit — the one this entire series is built on — is to always ask, *when I make \$30, where did the \$30 come from?* That question has three possible answers, and which one is true defines the game.

The **sum of a game** is exactly what it sounds like: add up every player's payoff. Three cases:

- **Zero-sum.** The payoffs add to zero. Every dollar one player gains, another loses. The pot is fixed; players only redistribute it. Poker (ignoring the casino's cut) is the classic example: chips just move around the table. So is a bet between two friends, a coin-flip wager, and — as we will prove with numbers shortly — a derivatives contract and a currency trade.
- **Positive-sum.** The payoffs add to *more* than zero. New value got created during the game, so the pot grew, and players can all end up ahead. A farmer and a baker trading wheat for bread are both better off afterward — that is why they traded. Owning a productive business that earns more money each year is positive-sum: the value isn't taken from anyone, it is *made*.
- **Negative-sum.** The payoffs add to *less* than zero. Value leaked out of the game while it was played — usually to a party who isn't really a "player" so much as a toll collector. A casino's roulette wheel is negative-sum for the gamblers as a group: the house edge guarantees the players collectively hand money to the building. As we will see, the average active trader's game is negative-sum for exactly the same reason.

Here is the single most useful line in the whole post, so read it twice. **The sum of the game tells you what time does to you.** In a zero-sum game, time is neutral — play forever and the average player breaks even, because the pot never changes. In a positive-sum game, time is your *ally* — the pot grows, so just staying seated and patient pays. In a negative-sum game, time is your *enemy* — every round the rake bleeds a little more, so the longer you play, the more certainly you lose. The same trade can flip between these depending on how you hold it.

One more foundational distinction, because it trips up almost everyone. **The market as a whole and your slice of it are different games.** The total value of all the world's companies has grown enormously over a century — that is positive-sum, and it is real. But that does not mean *your* active trading is positive-sum. You can be a net loser inside a market that is, in aggregate, creating value, because your trading is a side game played *on top of* the value creation, and that side game has a rake. Keeping these two layers separate — the value the economy creates versus the redistribution-plus-rake of active trading — is the whole intellectual content of this post.

### The third seat at the table: meet the house

When you picture a trade, you naturally picture two people: you and whoever takes the other side. But almost every real trade has a quiet third party, and that party is the one this post is most worried about. The *house* is the collection of intermediaries who profit from the trade happening at all, regardless of which direction the price goes. Spell out who they are, because you will be paying every one of them:

- **The broker** takes you to market and either charges a commission or, more commonly today, sells your order flow to a market maker. Either way they are paid by *volume* — the more you trade, the more they earn — so their incentive is structurally opposite to yours.
- **The exchange** charges a fee for matching buyers and sellers. Tiny per share, vast in aggregate; exchanges are some of the most profitable businesses in finance precisely because they collect a sliver of every transaction without ever taking a directional risk.
- **The market maker** stands ready to buy at the bid and sell at the ask, pocketing the spread. They are the most sophisticated player at the table, and in a zero-sum instrument they are frequently *your* counterparty — which means they are designed to win the spread off you on average.
- **The lender** rents you shares to short, or rents you cash to trade on margin, and charges interest either way.
- **The tax authority** takes a share of your realized gains, and a bigger share if you traded fast enough to make them short-term.

Notice the common thread: not one of these players is betting on whether the price goes up or down. They win on the *flow*. A casino does not care whether you hit on 17; it cares that you keep playing, because the edge is baked into the rules and paid on volume. The house in markets is identical. When we say a game is "negative-sum," we mean the players-who-take-directional-risk, as a group, are handing money to the players-who-only-take-a-cut. The directional players can fight all they want over who gets the remaining pot — but the pot is shrinking every round, and the shrinkage flows to the house. Holding this third seat in mind is what separates a clear-eyed trader from a hopeful one.

### A vocabulary you'll need

A few terms will recur, so let us gloss them once, plainly:

- A *long* position means you own the thing (or you profit when it rises). A *short* position means you have sold something you don't own (or you profit when it falls). For every long there is, somewhere, an exactly offsetting short — that mirroring is the heart of why derivatives are zero-sum.
- A *spread* (or *bid-ask spread*) is the gap between the price you can buy at and the price you can sell at, right now, for the same thing. The buy price (the *ask*) is always a little higher than the sell price (the *bid*). That gap is a cost you pay just for the privilege of trading.
- *Slippage* is the difference between the price you saw and the price you actually got, because the market moved or your order was too big for the quote.
- *Commission* is the flat or per-share fee the broker charges to execute.
- The *rake* is the gambling term for the cut the house takes out of each pot. We will borrow it for all the trading frictions together, because that is exactly what they are.

With those defined, we can build each of the three games from scratch — starting with the one where you genuinely must take money out of someone's pocket to make a cent.

## The zero-sum game: derivatives, FX, and the conservation of dollars

Start with the cleanest possible example, because the cleanliness is the point. A *derivative* is a contract whose value is *derived* from something else — a stock, an index, a currency, a barrel of oil. A *futures contract*, an *option*, a *swap*, a *contract for difference* — all derivatives. The defining feature, for our purposes, is that a derivative is a contract *between two parties*. It is not a share of a company that exists out in the world earning money. It is a private agreement: I will pay you if the price goes one way, you will pay me if it goes the other.

Because it is a two-party contract, the arithmetic is brutally simple. Whatever I win, you lose, to the penny. There is no third source of money. The contract created nothing; it just defines who pays whom. This is *conservation of dollars*, and it is as ironclad in a derivatives market as conservation of energy is in physics.

![Long call profit, short call profit, and their sum, which stays flat at zero across every expiry price](/imgs/blogs/zero-sum-positive-sum-and-the-house-where-trading-profits-come-from-3.png)

The chart makes conservation visible. Take one call option — a contract giving the buyer the right to buy a stock at a fixed *strike* price of \$100, for which the buyer pays the seller a *premium* of \$5 up front. The blue line is the long (buyer's) profit at every possible stock price on expiry day. The lavender line is the short (seller's) profit. Notice they are perfect mirror images across the horizontal zero line. Now add them together at any price you like — that is the dashed slate line, and it sits flat on zero everywhere. Whatever one side makes, the other loses, exactly. That dashed line *is* zero-sum, drawn.

#### Worked example: who pays whom in a single option contract

Let us put dollars on it. You buy one call on a stock, strike \$100, premium \$5. (One option contract usually covers 100 shares, but let us keep it to one share so the arithmetic stays friendly — the proportions are identical.) The person who sold it to you collected your \$5 today.

Three scenarios on expiry day:

- **Stock ends at \$108.** You exercise: you buy at \$100, the stock is worth \$108, so the option is worth \$8. You paid \$5 for it, so your profit is \$8 − \$5 = +\$3. The seller received \$5 but must hand over a stock worth \$8 for only \$100, a \$8 loss on the option offset by the \$5 they kept: their net is \$5 − \$8 = −\$3. Your +\$3 plus their −\$3 equals **\$0**.
- **Stock ends at \$100 or below.** The option expires worthless. You lose your \$5 premium: −\$5. The seller keeps the \$5 and owes nothing: +\$5. Sum: **\$0**.
- **Stock ends at \$105 (the breakeven).** The option is worth \$5, exactly what you paid. You make \$5 − \$5 = \$0. The seller keeps \$5 but owes \$5: \$0. Sum: **\$0**.

In every single case the two payoffs are equal and opposite. There is no price of the stock, no path it could take, where you both win or both lose. The intuition: a derivative is a sealed envelope of money that two people are fighting over — nobody can win except by taking from the other.

The same logic governs foreign exchange. When you buy euros with dollars, someone is selling you euros for dollars. If the euro rises, you gain and they lose the mirror amount; the global currency market does not "produce" euros the way a factory produces value. FX is a vast zero-sum reshuffling — roughly \$7.5 trillion a day changes hands as of the 2022 BIS survey, and across all of it, the gains and losses sum to zero before anyone pays a spread. (Central banks and corporates hedging real trade are doing something economically useful, but the *speculative P&L* among traders still nets to zero.)

It is worth dwelling on *why* a currency cannot be positive-sum the way a company is. A share of stock has a business behind it — a thing that earns. A unit of currency has nothing behind it that pays you for holding it except, at most, an interest rate, and that interest rate is itself a zero-sum transfer (the borrower pays exactly what the lender receives). There is no factory making euros more valuable while you hold them. So a long euro position is not "renting a seat at a value-creation table"; it is a pure bet that the euro will be worth more dollars later, against a counterparty betting the opposite. One of you is right. The dollars conserve. This is the cleanest possible illustration of the foundational rule: *if there is no underlying value being created, your only source of profit is someone else's loss.*

#### Worked example: the FX spread is a tax you pay twice

Let us see how even a "free" FX trade leaks to the house, and how a round-trip charges you the rake on both ends. You want to bet \$10,000 that the euro will rise against the dollar. The broker quotes EUR/USD at a bid of 1.0998 and an ask of 1.1002 — a spread of 4 *pips* (a pip is the smallest standard price increment, here 0.0001). You buy euros at the ask, 1.1002.

To make any profit you must first climb back over the spread, because to close you will *sell* at the bid. So your true breakeven is not "the euro goes up" — it is "the euro goes up by at least the spread." On \$10,000 the 4-pip round-trip spread costs:

$$\$10{,}000 \times \frac{0.0004}{1.1002} \approx \$3.64 \text{ each way, so } \approx \$7.27 \text{ round-trip.}$$

That is small — about 0.07% — which is exactly why FX feels "free." But trade \$10,000 five times a day, 250 days a year, and you cross that spread 1,250 times: 1,250 × \$7.27 ≈ \$9,090 of pure friction in a year, on an account that might be \$10,000. You can be a brilliant forecaster and still be handing the dealer your entire stake in spreads alone. The intuition: the spread is not a one-time entry fee, it is a toll you pay on every round-trip, and the busier you are, the more of your bankroll you ship to the house.

So what does it *mean* to win a zero-sum game? It means you outplayed a specific counterparty. There is no "the market gave me 8%" in a pure derivative — there is only "I was on the right side of a contract against someone who was on the wrong side." That is why this series keeps hammering the same question: *who is on the other side?* In a zero-sum game, your profit is literally their loss, so you had better know who they are and why they are wrong, because if you cannot name the sucker at the table, the sucker is you.

### Why hedgers make derivatives slightly less brutal — but not positive-sum

A fair objection: "If derivatives are pure zero-sum, why does anyone trade them? Surely it can't be a giant casino." The answer is that for *some* participants, the derivative is not a bet at all — it is insurance. A farmer who sells wheat futures locks in a price and sleeps at night; an airline that buys oil futures caps its fuel cost. These *hedgers* are willing to give up a little expected value in exchange for certainty, the same way you pay an insurance premium you hope to "lose."

That makes the derivative socially useful — risk gets transferred from people who hate it to people who will hold it for a fee. But notice it does *not* make the game positive-sum in dollars. The hedger's expected loss is the speculator's expected gain; the dollars still conserve. What changed is that one player now values certainty more than money, so they enter knowing their expected P&L is slightly negative and are fine with it. That is the closest a derivative comes to "win-win," and it is worth understanding precisely: the *utility* can be positive-sum even when the *dollars* are zero-sum. For your trading P&L, though, only the dollars pay your rent.

## The positive-sum game: why owning businesses is different in kind

Now the game that actually built my friend's \$80,000. When you buy a *share of stock*, you are not signing a contract with a counterparty who must lose for you to win. You are buying a small slice of ownership in a real company — a piece of its factories, its brand, its future profits. That difference is everything.

A company is a value-creating machine. It takes raw materials, labor, and capital, and turns them into products worth more than the inputs. The surplus is *profit*, and as an owner you have a claim on it. The company can hand you that profit directly as a *dividend* (a cash payment to shareholders), reinvest it to grow future profits, or buy back its own shares so your slice of the pie gets bigger (a *buyback*). None of these requires anyone else to lose. The value was *created*, out in the economy, by the business doing business.

![Equity total return broken into dividends, earnings growth, inflation, valuation, and trading against other holders](/imgs/blogs/zero-sum-positive-sum-and-the-house-where-trading-profits-come-from-4.png)

The chart decomposes where long-run stock returns actually come from. Over a century, U.S. equities returned roughly 9–10% per year nominally. Crucially, the lion's share of that — the green bars — comes from *value creation*: dividends paid out and earnings that grew. Inflation (the blue bar) is just the unit of account drifting. The amber bar, valuation change (the market paying a higher or lower multiple for the same earnings), washes out to near zero over long horizons — it is the part that *is* zero-sum, a tug-of-war between buyers and sellers over price. And the slate bar — your trading P&L against other shareholders — contributes essentially nothing to the *average* owner's return, because for every holder who times it well another times it badly.

#### Worked example: a coupon that nobody had to lose

You buy one share of a company for \$100. The company earns \$8 per share this year and pays you \$4 of it as a dividend, reinvesting the other \$4 to grow. Where did your \$4 come from? Not from another shareholder's pocket. It came from customers who bought the company's product for more than it cost to make. The dividend is *new money entering the game from outside the table* — from the real economy.

Now suppose every shareholder holds for the year. The company pays out, say, \$4 per share across all of them. Every single owner is \$4 richer, and no owner is poorer. Add up all the payoffs: a large positive number. That is positive-sum, in cash, and it is why a buy-and-hold owner does not need anyone on the other side to be wrong. The intuition: a dividend is a slice of value the business *made*, so all owners can eat at once.

#### Worked example: time as your ally in a positive-sum game

Watch what compounding does when the pot grows. You invest \$10,000 in a basket of stocks returning 9% a year, of which let us say 7% is the value-creation part (dividends plus earnings growth) and the rest is wash. Reinvest everything.

- After 1 year: \$10,000 × 1.09 = \$10,900.
- After 10 years: \$10,000 × 1.09¹⁰ ≈ \$23,674.
- After 25 years: \$10,000 × 1.09²⁵ ≈ \$86,231.

You did nothing but wait. No counterparty had to lose. The \$76,000 of gains came overwhelmingly from companies earning money year after year and the magic of reinvesting it. Compare this to a zero-sum game played 25 years: your *expected* result there is to break even, because the pot never grew. The intuition: in a positive-sum game, sitting still is a strategy, and a good one — time is on the owner's side.

This is the deepest practical reason index investing works. The index owner is not trying to beat a counterparty. They are simply renting a seat at the value-creation table and letting the economy do the work. They have opted out of the zero-sum side game almost entirely — and, just as importantly, out of the rake, which we turn to now.

### Where does the positive-sum money actually come from?

It is fair to be suspicious of "everyone wins." In a zero-sum game that is impossible by definition, so when someone claims a game is positive-sum your instinct should be to demand: *show me the source of the extra money.* For equities, the source is concrete and external to the table. A company sells a product to a customer for more than it cost to make. That surplus did not come from another shareholder — it came from a customer in the real economy who valued the product more than the price. The company then routes that surplus to its owners. So the "extra" in positive-sum is not magic and it is not a trick; it is the ordinary profit of commerce, flowing from outside the financial market into the pockets of owners.

This is why the *kind* of asset matters so much more than most beginners expect. Three assets can look identical on a price chart — a stock, a stock-index future, and a leveraged ETF tracking that index — and have completely different games underneath. The stock owner is collecting a slice of corporate profit (positive-sum). The futures trader is in a dated contract against a counterparty with no profit accruing to them (zero-sum, and it expires). The leveraged-ETF holder is paying a daily-rebalancing drag that compounds against them (negative-sum by construction). Same underlying index, three different fates over a year — and the only way to tell them apart is to ask where, if anywhere, value is being created and who pays the rake.

#### Worked example: two routes to the same index, one positive-sum and one not

You are bullish on a stock index and have \$10,000. Route A: you buy a low-cost index fund and hold it. Route B: you buy a leveraged ETF that promises twice the index's daily move, and you hold that.

Say the index ends the year exactly flat — but it got there by chopping: down 10% one stretch, up 11% the next, back and forth. Route A, the simple fund, also ends roughly flat: you collected the dividends (call it +2%, so about +\$200) and paid a tiny fee, net maybe +\$190. Positive-sum tailwind, small but real.

Route B suffers *volatility decay*. A 2× fund that loses 10% then gains 11% does not end flat; it ends down. Losing 20% then gaining 22% leaves you at 0.80 × 1.22 = 0.976 of your start — a roughly −2.4% loss, about −\$240, on an index that went *nowhere*. Add the fund's ~1% expense, and Route B has quietly cost you ~\$340 to track a flat index, while Route A *made* you ~\$190. Same view, same year, a \$530 gap — entirely explained by which game each instrument plays. The intuition: in a positive-sum instrument, going nowhere still pays you a little; in a negative-sum instrument, going nowhere bleeds you, because the rake never sleeps.

### The catch: positive-sum on average is not positive-sum for everyone

Before we leave equities, one honest caveat that the game-theory lens demands. Equities are positive-sum *in aggregate and over the long run*. That does not mean every stock, or every holding period, is positive-sum *for you*. A company can go bankrupt and the value you owned can evaporate — that loss was real, not transferred. And over short horizons, the price you pay and the price you sell at are set by a zero-sum tug-of-war among traders, so a poorly-timed buy-and-sell can lose even as the underlying business thrives. The positive-sum nature is a *tailwind*, not a guarantee. Diversification (owning many companies) and time (holding through cycles) are how you actually collect the positive-sum part instead of getting chopped up in the zero-sum part on top of it.

## The negative-sum game: the house's rake

Here is where most active traders actually live, and where my day-trading friend lost a quarter-century. The moment you start trading frequently, you are no longer playing the clean positive-sum game of ownership, nor even the clean zero-sum game of a single derivative. You are playing a zero-sum game *with a rake on every hand*, and a rake turns a fair fight into a guaranteed slow loss for the average player.

![A waterfall showing a gross trading edge eaten down to a net loss by spreads, commissions, slippage, borrow, and taxes](/imgs/blogs/zero-sum-positive-sum-and-the-house-where-trading-profits-come-from-2.png)

The waterfall chart shows the anatomy. Start with the green bar — a *gross* edge, the profit you would make if trading were free. Suppose you are genuinely good and your gross expected value is +\$3,000 a year on a \$100,000 account. Now subtract, one amber wedge at a time, every friction: the spread you cross on each trade, the commissions, the slippage when your order moves the price, the cost to borrow shares for shorts plus the tax on your gains. Each is small. Together they are not. By the time the rake has taken its cut, that +\$3,000 gross edge can become a *negative* net result — the red bar at the right. You were right. You still lost.

### Where each piece of the rake comes from

Let us make each friction concrete, because their smallness is exactly what makes them dangerous — they hide.

- **Spread.** Every time you buy at the ask and later sell at the bid, you pay the gap. On a liquid stock it might be a penny on a \$50 share — 0.02%. On a thin small-cap or a far-out-of-the-money option it can be 1% or worse. You pay it *coming and going*, and you pay it whether you win or lose.
- **Commission.** Even in the "zero-commission" era, brokers are paid — often through *payment for order flow*, where a market maker pays your broker to route your order to them, then earns the spread off you. The commission didn't vanish; it moved inside the spread.
- **Slippage.** The quote said \$50.00, but by the time your order filled the price was \$50.03. On large orders this dwarfs the explicit costs.
- **Borrow cost.** To short a stock you must borrow it, and you pay a fee — trivial for big liquid names, brutal (sometimes >50% annualized) for hard-to-borrow squeeze candidates.
- **Taxes.** In most jurisdictions, short-term gains are taxed at a higher rate than long-term. Churning a position into a short-term gain can hand 30–40% of the profit to the tax authority, while the buy-and-hold owner defers tax for decades and pays a lower rate.

Each of these is a leak. The leaks go to the house — the broker, the exchange, the market maker, the government — none of whom are betting on direction. They win regardless. That is the structural definition of a rake: a party who profits from the *volume* of play, not the *outcome*.

#### Worked example: the rake turns a winning record into a loss

Let us prove the headline claim with numbers. You make 250 round-trip trades a year, \$10,000 of stock each time. You are good: you win 55% of your trades, and your average win equals your average loss before costs — a +5% move when right, a −5% move when wrong, so \$500 either way.

Gross expected value per trade, before any costs:

$$\text{EV}_{\text{gross}} = 0.55 \times \$500 + 0.45 \times (-\$500) = \$275 - \$225 = +\$50.$$

Over 250 trades that is a gross edge of 250 × \$50 = +\$12,500. Looks great. Now the rake. Suppose each round-trip costs you, all in (spread both ways + slippage + a sliver of commission), about 0.6% of the \$10,000 traded, or \$60 per round-trip. That is not extreme — for a fast trader in mid-liquidity names it is conservative.

$$\text{EV}_{\text{net}} = \$50 - \$60 = -\$10 \text{ per trade}.$$

Over 250 trades: 250 × (−\$10) = **−\$2,500 a year.** You won 55% of your trades — a genuinely good hit rate — and you still lost \$2,500, because the rake of \$60 was bigger than your \$50 edge. The intuition: in a negative-sum game, being right more often than wrong is not enough; you must be right by *more than the rake*, and the rake is charged on every hand whether you win it or not.

This is the single most important calculation in the post, and it is why win rate lies. We treat this in depth in the cross-linked piece on [why win rate lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) and on building [risk-reward and expectancy in practice](/blog/trading/technical-analysis/risk-reward-and-expectancy-in-practice) — the short version is that your *expectancy* (average dollars per trade after costs) is what matters, and the rake is subtracted from it every single time.

![Expected cumulative P&L over up to 250 trades for a fair game with no rake, a four-dollar rake, and an eight-dollar rake](/imgs/blogs/zero-sum-positive-sum-and-the-house-where-trading-profits-come-from-5.png)

The chart drives it home for a perfectly *fair* underlying game — a 50/50 bet where the gross expected value is exactly zero, so nobody has any edge at all. With no rake (the flat slate line) you break even forever, as a fair zero-sum game should. Add a \$4 rake per trade and your expected wealth slides down the amber line; add an \$8 rake and you bleed down the red line to −\$2,000 after 250 trades. Nothing about your skill changed — there was no skill to change. The rake alone, applied to a fair game, manufactures a certain loss. This is mathematically identical to a casino: roulette is *almost* fair, and the "almost" is the entire business model.

#### Worked example: the expected value of a coin flip with a toll

Let us compute the EV of a single fair flip with a toll, to see the negative number appear from nothing. You flip a fair coin: heads you win \$100, tails you lose \$100. The expected value is:

$$\text{EV} = 0.5 \times \$100 + 0.5 \times (-\$100) = \$0.$$

A fair game — zero-sum, time-neutral. Now the house charges \$8 to play each round (the rake). Your expected value per round becomes:

$$\text{EV}_{\text{net}} = \$0 - \$8 = -\$8.$$

Play it 250 times and your expected loss is 250 × \$8 = \$2,000, exactly the red line on the chart. There was never any edge to lose — the rake created the loss out of thin air. The intuition: a toll on a fair game is not a fair game minus a little; it is a *losing* game, full stop, and the only way to win it is to refuse to play most hands.

### The rake compounds, which is worse than it sounds

The rake is not a one-time subtraction. Because it hits *every* trade, and the surviving balance is what funds the next trade, the drag *compounds* — it eats a percentage of a shrinking base, round after round.

![Expected account balance over a year for buy-and-hold, active, and heavy-churn cost levels, all in a fair game](/imgs/blogs/zero-sum-positive-sum-and-the-house-where-trading-profits-come-from-7.png)

The chart shows three traders in the same fair game (zero gross edge) over a year of trading, differing only in turnover and cost. The buy-and-hold owner who rarely trades pays almost nothing and stays flat at \$10,000. The active trader paying about 15 basis points (a *basis point* is one hundredth of a percent, 0.01%) per round-trip drifts down the amber line. The heavy churner paying 35 basis points per round-trip falls hardest, ending well below \$9,000 after ~260 trading days — purely from costs, with no bad luck and no bad calls. The gap between the green and red lines is the entire difference between my two friends, drawn as a single year. Stretch it to twenty-five years and the lines diverge into the \$80,000-vs-broke gulf the post opened with.

### Why the rake is so easy to miss

If the rake is this destructive, why does almost everyone underestimate it? Three reasons, each a behavioral trap worth naming so you can defend against it.

First, **the rake is small per trade and large per year.** A \$7 spread on a \$10,000 trade feels like a rounding error in the moment — and it is, in the moment. But you do not feel the annual sum of a thousand rounding errors until you tally them, and most traders never tally them. The house designed it this way; a casino's edge on any single roulette spin is small enough to ignore, which is precisely why people keep spinning.

Second, **the rake hides inside outcomes you attribute to luck.** When a trade loses, you blame the market, the news, your timing — never the spread you crossed to get in. The cost got bundled into a result that *looks* like a directional loss, so you learn the wrong lesson ("I was wrong about direction") instead of the right one ("I was charged a toll I never priced in"). The friction is invisible because it wears the costume of a bad call.

Third, **the rake is asymmetric against frequency, which is the opposite of intuition.** Most people believe that trading more — being more active, more engaged, more "on top of it" — should improve results, because effort improves results everywhere else in life. In a negative-sum game the relationship inverts: more activity means more rakes paid, so effort actively *destroys* value. This is the single most counterintuitive fact in retail trading, and it is why the data so reliably shows the least active accounts beating the most active ones.

#### Worked example: the breakeven win rate the rake forces on you

Here is the rake translated into the win rate you actually need. Suppose your average winning trade makes \$500 and your average losing trade loses \$500 (symmetric), and the round-trip rake is \$40. To break even, your wins must cover your losses *plus* the rake on every trade. Let $w$ be your win rate. Per trade:

$$\text{EV} = w \times (\$500 - \$40) + (1 - w) \times (-\$500 - \$40) = 0.$$

Solving: \$460w − \$540(1 − w) = 0, so \$1{,}000w = \$540, giving $w = 0.54$. You need to win **54%** of trades just to break even, even though your wins and losses are the same size — a fair coin (50%) loses. Raise the rake to \$80 and the breakeven win rate climbs to 58%. The intuition: the rake silently raises the bar you must clear, so a strategy that looks like a fair coin on paper is a guaranteed loser in practice unless it is meaningfully better than a coin.

#### Worked example: turnover times cost times time

Make the compounding precise. You start with \$10,000 in a fair game and pay 0.15% (15 bps) on each round-trip. After $T$ round-trips your expected balance is:

$$B_T = \$10{,}000 \times (1 - 0.0015)^{T}.$$

- After 50 trades: \$10,000 × 0.9985⁵⁰ ≈ \$9,277.
- After 250 trades: \$10,000 × 0.9985²⁵⁰ ≈ \$6,873.

A 15-basis-point cost — fifteen hundredths of one percent, an amount you would never notice on a single trade — quietly vaporizes about 31% of your money over a year of active trading, before any losing *bets* at all. The buy-and-hold owner trading once paid 0.0015 once and kept essentially all \$10,000. The intuition: the rake's damage is turnover × cost × time, and active trading maxes out all three terms while ownership minimizes them.

## Common misconceptions

**"The stock market is a zero-sum game — for me to win, someone has to lose."** This is the most common and most expensive confusion, and it conflates the two layers. The *trading* of stocks among holders is roughly zero-sum (minus the rake, so negative-sum). But *owning* stocks is positive-sum, because companies create value that flows to all owners via earnings and dividends. The index investor who never trades is collecting the positive-sum part and skipping the zero-sum side game. Believing the whole thing is zero-sum talks people *out* of the one genuinely win-win game available to them.

**"I win more than half my trades, so I'm profitable."** Win rate is not edge. As the worked example showed, a 55% hit rate loses money once the rake exceeds your per-trade edge. What matters is *expectancy* — average dollars per trade after every cost. You can win 70% of trades and go broke (if your losers are big and your winners small and the rake nibbles the rest), or win 35% and get rich (if your few winners are enormous). The house cares only that you keep playing; it makes its rake on your volume regardless of your record.

**"Commissions are basically zero now, so costs don't matter."** The explicit commission shrank, but the rake just changed costume. Zero-commission brokers are typically paid through payment for order flow — the market maker pays for the right to fill your order and earns the spread. You still pay the spread, you still pay slippage, you still pay borrow on shorts, and you very much still pay taxes. The total friction for an active retail trader is routinely 0.3–1% per round-trip even in the "free" era. "Free" describes the line item you can see, not the cost you actually bear.

**"If derivatives are zero-sum, the people on the other side are dumb, so I should win."** The opposite is usually true. In liquid derivatives the counterparty is frequently a professional market maker or a hedger with a real economic reason — not a sucker, but the most informed, best-capitalized player at the table. In a zero-sum game your profit comes from *their* loss, so you are choosing to play against the people least likely to be wrong. If you cannot articulate why your specific counterparty is mispriced, you are the one supplying the edge. We unpack this dealer's-eye view at length elsewhere in the series.

**"I'll just hold the derivative long-term like a stock and let time work."** Time does not work for you in a zero-sum instrument — there is no underlying value creation accruing to you, and many derivatives actively *decay* (options lose time value; leveraged and inverse ETFs bleed from daily rebalancing; futures roll). Holding a stock for ten years lets the company earn for you; holding most derivatives for ten years just exposes you to ten more years of a fair-or-worse bet plus carrying costs. The "time is my ally" rule belongs to the positive-sum game only.

## How it shows up in real markets

**The S&P 500 versus the average active fund (the SPIVA scorecard).** S&P's twice-yearly SPIVA report is the cleanest real-world demonstration of the rake. Across most multi-year windows, roughly 80–90% of actively managed U.S. equity funds underperform the simple index after fees. These are professionals — full-time, well-resourced, often genuinely skilled. They lose to a no-brain index not because they pick worse stocks on average, but because the index is collecting the positive-sum return while paying almost no rake, and the active funds are paying their own fees plus trading costs on top. It is the two-friends story at industrial scale.

**The day-trader survival studies.** Academic studies of retail day-traders — most famously the long-running work on Brazilian and Taiwanese day-traders, and Barber and Odean's U.S. brokerage data from the late 1990s — consistently find that the large majority lose money over time, and that the more frequently an account trades, the worse it does. Barber and Odean's memorable finding was that the households that traded the most underperformed those that traded the least by several percentage points a year, almost entirely explained by transaction costs. The single best predictor of a retail account's returns was, perversely, how *little* it traded. That is the negative-sum rake, measured in the wild.

**Foreign exchange as a zero-sum arena.** The FX market is enormous and astonishingly competitive precisely because it is zero-sum. With about \$7.5 trillion traded daily (2022 BIS Triennial Survey) and no underlying "FX dividend," every speculative dollar of profit is a dollar of someone else's loss before spreads, and after spreads the speculative crowd as a whole loses to the dealers. Retail FX brokers' own regulatory disclosures routinely show 70–80% of client accounts losing money in a given quarter. There is no value-creation tailwind to bail anyone out — just a pure contest against, often, the very dealer quoting your price.

**Options expiry: a sealed pot fought over.** On a major options expiration, the conservation of dollars is almost tactile. Every dollar an option buyer makes on a winning call is a dollar the seller loses, and vice versa; the open interest nets to zero by construction. Market makers, who are typically on the other side of retail flow and hedge their direction away, are not betting on the stock — they are collecting the spread and the edge in the pricing, hand after hand, like the casino. Retail buyers of short-dated options, studied repeatedly, lose money in aggregate; the structure guarantees that the rake plus the informed counterparties extract value from the uninformed crowd.

**The buyback era and positive-sum mechanics.** A concrete positive-sum example: when a profitable company like Apple buys back tens of billions of dollars of its own shares a year, it shrinks the share count, so each remaining owner's slice of future earnings grows — without any selling shareholder having to be "wrong." Over the 2010s, buybacks plus dividends returned trillions of dollars of real corporate profit to shareholders collectively. No counterparty lost for owners to gain; the value came from the businesses earning it. That is the engine behind the green bars in the return-decomposition chart, and it is why patient ownership is structurally different from trading.

**Lotteries, sports books, and the purest negative-sum games.** It is clarifying to look at the games that are *openly* negative-sum, because they are what an over-traded portfolio quietly resembles. A typical national lottery returns roughly 50 cents on the dollar to players in aggregate — a −50% expected value, the most brutal rake in legal finance. A regulated sports book holds a "vig" of around 4–5% of the money wagered, baked into the odds, so the bettors as a group hand the book that slice no matter who wins. Neither game has any value creation; both are pure redistribution with a heavy toll. The reason this matters here is that nothing about the *instrument* — a stock, a future, a lottery ticket — tells you the game; the toll-plus-no-value-creation structure does. A trader churning a brokerage account at 1% round-trip costs has, mathematically, built themselves a private sports book and seated themselves on the losing side of the vig.

**The cost of "free" via payment for order flow.** When the major U.S. brokers went to zero commissions around 2019, retail volume exploded — and so did payment for order flow, the practice of selling customer orders to wholesale market makers. In 2021, U.S. regulators reported that the largest such market maker paid brokers well over a billion dollars for order flow, money that ultimately comes out of the spread retail traders cross. The headline "commission" went to zero while the *real* rake — the spread captured by the wholesaler — quietly grew with volume. It is the cleanest modern proof that the house does not need a visible fee to take its cut; it only needs you to keep trading.

**The meme-stock squeezes: a vivid zero-sum settlement.** The 2021 GameStop episode is, underneath the drama, a zero-sum settlement at scale. Some short sellers lost billions; those exact billions were won by the people on the other side of those shorts. No new value was created by the squeeze itself — shares of a struggling retailer do not become more productive because they changed hands violently. It was a redistribution, and a brutal one, with a heavy rake skimmed by everyone facilitating the frenzy. We dig into the coordination dynamics of squeezes in the [prisoner's-dilemma piece in this series](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once); here the lesson is narrower: when you see a fortune made fast in a derivative or a squeeze, look immediately for the matching fortune lost, because in a zero-sum game it is always there.

![A two-by-two matrix contrasting a zero-sum derivative bet with a positive-sum business stake](/imgs/blogs/zero-sum-positive-sum-and-the-house-where-trading-profits-come-from-6.png)

The matrix contrasts the two pure games side by side. In the top row, the zero-sum bet, every cell has one winner and one loser and the row sums to zero — you only eat by beating a named counterparty. In the middle row, the positive-sum stake, *both* players can come out ahead because the business created value, so even your worse outcome is still a gain. The bottom row is the takeaway: in the zero-sum world the forecast is everything and a miss is fatal; in the positive-sum world even a mediocre year can be a positive one, because the game itself is paying you to stay. Which row you are standing in is the first thing to decide before any trade.

## The playbook: how to play it

Everything above collapses into one discipline: **identify the game before you size the position.** Here is how to actually use it.

**First, name the game.** For any position you are about to take, answer three questions. (1) Is there an underlying that creates value over time and pays it to owners — earnings, dividends, buybacks, coupons? If yes, you have a positive-sum tailwind. (2) Is this a two-party contract where my gain is literally a counterparty's loss — a derivative, an FX bet, a CFD? If yes, you are in a zero-sum fight and time is not on your side. (3) How much rake will I pay, in spread plus slippage plus commission plus borrow plus tax, *per round-trip*, and how many round-trips will I make? Multiply those out; that is your hurdle.

**Who is on the other side.** In the positive-sum game, "the other side" of your buy is just another investor reallocating — you are not really competing with them, you are both renting seats at the value-creation table. In the zero-sum game, the other side is frequently a professional dealer or a better-informed hedger, and your profit is their loss, so you must be able to state *why they are wrong*. If you cannot, do not take the trade — you are the edge they are harvesting. We unpack the dealer's reasoning in the series' market-maker material and the EV-and-edge mindset in [thinking like the house](/blog/trading/game-theory/expected-value-edge-and-variance-thinking-like-the-house).

**Your edge and the hurdle it must clear.** In a positive-sum game your "edge" can be as simple as *patience plus diversification plus low costs* — you do not need to outsmart anyone, you need to stay invested and not bleed to the rake. In a zero-sum or negative-sum game, your edge must clear the rake *and then some*, on every hand. Use the expectancy test: average dollars per trade after all costs must be positive. If your gross edge per trade is \$50 and your round-trip rake is \$60, you have no business taking that trade however confident you feel — the math, not the feeling, is your invalidation.

**Sizing and exits driven by the game.** Because time is your ally in the positive-sum game, sizing should favor *staying in* — broad, diversified, long-horizon, rebalanced rarely so the rake stays tiny. Because time is your enemy in the negative-sum game, sizing there should favor *playing fewer, higher-conviction hands* — every avoided marginal trade saves a full rake, so the discipline is to trade *less*, not more. The exit in a zero-sum trade is set by your thesis on the counterparty being wrong; when that thesis is realized or invalidated, you are done, because there is no tailwind to "hold and hope" into.

**The one-line rule to carry.** In a zero- or negative-sum game you win only by being better than a specific, named, often professional counterparty — so know who they are or stay out. In a positive-sum game, time is the edge — so the winning move is frequently to do almost nothing, cheaply, for a very long time. My two friends are the whole lesson: one played the game where staying still pays, the other played the game where the house rakes every hand, and twenty-five years told the truth.

**A checklist you can run in ten seconds.** Before any position, walk these in order. Is there an underlying that earns and pays owners? If yes, you have a positive-sum tailwind and patience is a strategy. Is my profit literally a named counterparty's loss? If yes, I must be able to say why they are wrong, or I am the one being harvested. What is my all-in cost per round-trip, and how many round-trips will my plan require? Multiply them; that product is the hurdle my edge must clear, and it is charged whether I win or lose. If the honest answer to the last question is "the rake is bigger than any edge I can defend," the correct trade is no trade — and recognizing that is itself an edge, because most of the table never runs the check.

This is educational, not individualized advice — but if you take one habit from it, let it be the reflex to ask, before any trade, *which of the three games is this, and where is the rake?*

## Further reading & cross-links

- [The Trade Is a Game: Why Markets Are Strategic, Not Random](/blog/trading/game-theory/the-trade-is-a-game-why-markets-are-strategic-not-random) — the series opener on treating every trade as a strategic interaction against an adaptive opponent rather than a bet against nature.
- [Expected Value, Edge, and Variance: Thinking Like the House](/blog/trading/game-theory/expected-value-edge-and-variance-thinking-like-the-house) — how the house actually reasons about EV and edge, the engine behind why the rake wins.
- [The Prisoner's Dilemma in Markets: Why Everyone Sells at Once](/blog/trading/game-theory/the-prisoners-dilemma-in-markets-why-everyone-sells-at-once) — the coordination dynamics behind squeezes and panics, the zero-sum settlements writ large.
- [The SIG / Susquehanna Playbook: Poker, Game Theory, and EV](/blog/trading/quant-careers/sig-susquehanna-playbook-poker-game-theory-and-ev) — how a top prop firm builds its whole culture around expected value and knowing the counterparty.
- [Expectancy: Why Win Rate Lies](/blog/trading/technical-analysis/expectancy-why-win-rate-lies) and [Risk-Reward and Expectancy in Practice](/blog/trading/technical-analysis/risk-reward-and-expectancy-in-practice) — the practitioner math for why a high win rate loses to the rake, and how to compute the expectancy that doesn't lie.

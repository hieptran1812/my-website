---
title: "Order book imbalance: the shortest-horizon signal there is"
date: "2026-08-31"
publishDate: "2026-08-31"
description: "The ratio of resting size on the two sides of the book predicts the next move in price. It is one of the most robust effects in market microstructure, and it is dead within seconds, which is exactly what makes it interesting."
tags: ["order-book-imbalance", "market-microstructure", "market-making", "adverse-selection", "limit-order-book", "micro-price", "high-frequency-trading", "signal-decay", "quantitative-finance", "math-for-quants"]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: false
readTime: 19
---

> [!important]
> **TL;DR:** Count the shares resting on each side of the top of the order book. The side with more size tends to win the next move. That is order book imbalance, and it is simultaneously one of the most reliable predictors in all of finance and one of the least tradeable.
>
> - Imbalance at the touch is a single number between -1 and +1. The version computed from deeper levels barely improves it, because only the top row is mechanically binding.
> - It works through two channels at once: the thin side of the book is cheaper to clear, and liquidity chose where to sit for a reason.
> - The imbalance-weighted fair value sits at mid plus **half a spread times the imbalance**. The signal's entire range is one spread wide, so a trade that crosses the spread can never be paid for it.
> - The same imbalance that predicts the move is what makes your resting order fill at the worst moment. At an imbalance of -0.50 when you get filled, half your spread capture is already gone: \$1,000 a day instead of \$2,000 on \$10m of notional.

## The one number a market maker checks first

Open any stock's order book and you will see two stacks of resting orders. Buyers waiting on one side, sellers waiting on the other. Most of the time, the two stacks are not the same size.

Suppose the best buyers are offering to take 12,000 shares and the best sellers are offering only 3,000. Nothing has traded yet. No news has come out. And yet, over the next few seconds, the price is more likely to go up than down.

That asymmetry has a name, **order book imbalance**, and it is about as close to a free lunch as microstructure gets. It is stable across stocks, across venues, across decades. It survives every robustness check anyone has thrown at it. It is also, for almost everyone reading this, completely untradeable. Understanding why both of those things are true at once is the entire point of this post.

![A two-sided depth ladder showing 3,000 shares resting at the best ask of 50.02 and 12,000 at the best bid of 50.00, with the imbalance computed as plus 0.60 and the three-level version as plus 0.07](/imgs/blogs/order-book-imbalance-short-horizon-prediction-math-for-quants-1.webp)

The figure above is the mental model. Two stacks, one number that summarises their asymmetry, and a strong hint about where the price goes next.

## Foundations: the book, the touch, and the queue

You need four ideas before any of this makes sense. If you already know what a limit order book is, skim to the next section, but do read the part about queues.

### The limit order book

Most modern exchanges run a **continuous double auction**. Anyone can post a *limit order*: a promise to buy up to N shares at no more than some price, or to sell at no less. Those promises sit in a public list, sorted by price, until they are filled or cancelled. That list is the **limit order book**.

The other order type is a *market order*, which says "fill me now at whatever price is available". A market order does not join the book. It walks into the book and consumes resting limit orders until it is filled. Every trade is one side crossing into the other's resting size. If you want the mechanics in full, with code, see [the order book simulator post](/blog/trading/quantitative-finance/order-book-simulator-quant-research).

### The touch, the spread and the mid

The **best bid** is the highest price anyone is currently willing to buy at. The **best ask** (or best offer) is the lowest price anyone is willing to sell at. Together they are called **the touch**, or the top of book.

Say the best bid is \$50.00 and the best ask is \$50.02. The gap between them is the **bid-ask spread**, here 2 cents, or 4 basis points of the price. A *basis point* is one hundredth of a percent. The **mid-price** is the average of the two, \$50.01. Nobody can trade at the mid. It is a convention, not a price, and remembering that turns out to matter enormously.

### The queue

Behind each of those two prices sits a pile of orders, and that pile has an order. Exchanges almost always allocate fills by **price-time priority**: at a given price, the order that arrived first gets filled first. So if 12,000 shares are resting at \$50.00 and you add 1,000 more, you are behind all 12,000. A seller has to hit through the entire queue ahead of you before you trade.

**Queue position is the asset.** That single fact drives everything that follows.

### Resting size

Finally, **resting size** is just how many shares are sitting at a given price level. It is the raw material of the signal: at any instant, the book tells you exactly how many shares are waiting on each side.

## Defining imbalance

Let $Q_b$ be the number of shares resting at the best bid and $Q_a$ the number at the best ask. The **touch imbalance** is

$$
I \;=\; \frac{Q_b - Q_a}{Q_b + Q_a}
$$

The numerator is the raw asymmetry and the denominator normalises it, so $I$ always lands in the interval from -1 to +1. At $I = +1$ the ask side is empty and every resting share is a buyer. At $I = -1$ the reverse. At $I = 0$ the two sides are exactly matched.

There is also a multi-level version, which sums resting size over the first $K$ price levels on each side, usually with weights $w_k$ that decay with distance from the touch:

$$
I_K \;=\; \frac{\sum_{k=1}^{K} w_k \left(Q_b^{(k)} - Q_a^{(k)}\right)}{\sum_{k=1}^{K} w_k \left(Q_b^{(k)} + Q_a^{(k)}\right)}
$$

You would expect more data to help. It mostly does not, and the reason is structural rather than statistical. Only the touch is mechanically binding: the price cannot move until a touch queue is exhausted or a new order appears inside the spread, so only touch size sits on the causal path. Orders three levels deep are cheap to post and free to cancel, they carry far less commitment, and they can be withdrawn faster than anyone can trade against them. Depth away from the touch describes intent. Depth at the touch describes obligation.

#### Worked example 1: reading one book

Take the book in the figure above. Best bid \$50.00 with 12,000 shares. Best ask \$50.02 with 3,000 shares.

Step 1, the touch imbalance:

$$
I = \frac{12{,}000 - 3{,}000}{12{,}000 + 3{,}000} = \frac{9{,}000}{15{,}000} = +0.60
$$

Step 2, turn it into a probability. Adopt the simplest possible mapping, and note clearly that this is an **assumption we are making, not a measured coefficient**: assume the chance the next mid move is upward equals the bid's share of total touch size,

$$
\Pr(\text{next mid move is up}) = \frac{Q_b}{Q_b + Q_a} = \frac{1 + I}{2} = 0.80
$$

Step 3, look one level deeper. Add 4,000 shares bid at \$49.99 and 6,000 at \$49.98, against 9,000 offered at \$50.03 and 7,000 at \$50.04. Now the equally weighted three-level imbalance is ${(22{,}000 - 19{,}000)/41{,}000 = +0.07}$.

Same book, same instant. The touch says strongly up, the three-level view says essentially nothing. Step 4 puts money on it: if you are marking a 10,000 share position, \$500,000 of notional, the difference between valuing it at mid and valuing it at the imbalance-adjusted fair value we derive below is 0.6 cents a share, or \$60.

**The intuition:** the signal lives at the touch, it is worth basis points rather than percent, and averaging in deeper levels dilutes it rather than sharpening it.

## Why it predicts: two channels, not one

![Two panels, the left showing the queue race where 3,000 shares of buying clears the thin ask while 12,000 of selling is needed to clear the bid, the right showing the information channel where sellers who expect a higher price stop posting asks](/imgs/blogs/order-book-imbalance-short-horizon-prediction-math-for-quants-2.webp)

Candidates in interviews usually name one of these two mechanisms. The complete answer is that both operate, and that they reinforce each other.

### The queue race

The mid-price moves when a touch queue empties out, or when someone posts inside the spread. Consider our book. To clear the ask, the market needs to absorb 3,000 shares. To clear the bid it needs 12,000. If buy and sell pressure arrive at roughly similar rates, the thin side simply runs out first, far more often than not. When the ask clears, the best ask steps up to \$50.03 and the mid rises with it.

This is a race between two depleting queues, and the shorter queue wins more often. It requires no information, no view, no informed trader. It is arithmetic about arrival times. Cont, Stoikov and Talreja formalised exactly this as a stochastic queueing model of the book, and Lipton, Pesavento and Sotiropoulos worked out the probability of the next price move as a function of quote imbalance in the same spirit.

### Where liquidity chose to sit

The second channel is that the book's shape is not exogenous. It is a record of choices.

If participants broadly expect the price to rise, sellers become reluctant to post at \$50.02 and start cancelling; buyers become eager and pile into \$50.00. The thin ask is not an accident of arrival times. It is a vote. Under this reading, imbalance is a real-time poll of who wants to transact and at what price, and the fact that it predicts is unsurprising.

The two channels are entangled and cannot be cleanly separated in data. They also push the same way, which is precisely why the effect is so robust. A purely mechanical effect could be arbitraged away by anyone willing to post on the thin side. A purely informational one would vanish once the information was public. Because both are running, imbalance keeps predicting even though every serious participant knows about it.

## The micro-price, and why the spread eats it

Here is the piece that turns "the price will probably go up" into a number you can act on.

If imbalance tells you the ask is likelier to break than the bid, then fair value is not the mid. It is somewhere between mid and ask. The natural first estimator is a **size-weighted mid**, where each price gets the *opposite* side's size as its weight:

$$
P_w \;=\; \frac{Q_a P_b + Q_b P_a}{Q_a + Q_b}
$$

The cross-weighting looks backwards until you see why. A big bid queue means a lot of demand waiting, so fair value should sit near the ask, and so the ask price needs the big weight. Substituting the definition of $I$ and writing $s$ for the spread and $P_m$ for the mid gives a clean identity:

$$
P_w \;=\; P_m + \frac{I}{2}\,s
$$

Every term is observable. With $I = +0.60$ and a 2 cent spread, fair value sits 0.6 cents above the mid, at \$50.016. Sasha Stoikov's *micro-price* is the refined version of this estimator, correcting the fact that the naive weighted mid is not a martingale; the first-order intuition is the identity above.

![A number line from the best bid at 50.00 to the best ask at 50.02, with the micro-price at 50.016 marked inside, showing that half a spread is the entire range the signal can move fair value while a round trip costs a full spread](/imgs/blogs/order-book-imbalance-short-horizon-prediction-math-for-quants-3.webp)

Now look at what that identity forbids. Since $|I| \le 1$, the displacement $\tfrac{I}{2}s$ can never exceed **half a spread**. The whole signal, at its theoretical maximum, moves fair value by exactly the amount a crossing trade pays to get in, and pays again to get out.

#### Worked example 2: crossing the spread when the signal is right

You see $I = +0.60$ and decide to buy 10,000 shares, about \$500,000 of notional. Give yourself the best case: the signal is exactly right and fully realised.

1. You cross and buy at the ask, \$50.02.
2. Fair value moves to the micro-price, \$50.016, and the book re-forms around it: new bid \$50.006, new ask \$50.026.
3. You exit by crossing back, selling at the new bid of \$50.006.
4. P&L per share: ${50.006 - 50.02 = -0.014}$ dollars, so 1.4 cents of loss.
5. On 10,000 shares that is **-\$140**, before any exchange fee.

You were right, and you lost \$140. To break even you would need the mid to move a full 2 cents, which by the identity requires $I = 2$. There is no such book.

**The intuition:** order book imbalance is not an alpha signal for anyone who crosses the spread. It is a fair-value correction, and it only pays the participants who never pay the spread in the first place.

## How fast the edge dies

The second reason this is not a strategy is that the prediction is about *the next mid move*, not about a drift you can sit in. The book refreshes continuously. Queues refill, cancels arrive, the imbalance that pointed up thirty seconds ago has been replaced several times over.

![A chart with holding horizon in seconds on the x-axis and cents per share on the y-axis, showing the noise curve rising from 0.65 to 5.06 cents while the signal curve falls from 0.6 to 0.1 cents, both far below the 2 cent round-trip cost line](/imgs/blogs/order-book-imbalance-short-horizon-prediction-math-for-quants-4.webp)

Two things happen as the horizon stretches, and they work against you independently. The signal decays, because the state that generated it is gone. And the noise grows with the square root of time, because that is what diffusion does, a point developed at length in [the post on the central limit theorem](/blog/trading/math-for-quants/law-large-numbers-central-limit-theorem-math-for-quants). The ratio of the two collapses from both ends at once.

#### Worked example 3: the horizon test, in dollars per round trip

Same \$50 stock, same 2 cent spread, same 10,000 share clip.

**The noise side is derived, not assumed.** Take annualised behaviour equivalent to 2% daily volatility, so a one-day standard deviation of \$1.00 on a \$50 stock. A US equity session runs 6.5 hours, which is 23,400 seconds. Scaling by the square root of time gives a one-second standard deviation of ${1.00/\sqrt{23{,}400} = 0.0065}$ dollars, so 0.65 cents. At 10 seconds it is 2.07 cents; at 60 seconds, 5.06 cents.

**The signal side is assumed and labelled as such.** The literature does not hand you a universal decay constant, so take an illustrative path that starts at the full micro-price displacement and halves roughly every ten seconds: 0.6 cents at 1 second, 0.3 at 10 seconds, 0.1 at 60 seconds.

| Horizon | Expected move | Gross on 10,000 shares | 1 sigma noise | Round-trip spread cost | Net |
| --- | --- | --- | --- | --- | --- |
| 1 second | 0.60 cents | \$60 | \$65 | \$200 | -\$140 |
| 10 seconds | 0.30 cents | \$30 | \$207 | \$200 | -\$170 |
| 60 seconds | 0.10 cents | \$10 | \$506 | \$200 | -\$190 |

At one second the expected gain is smaller than one standard deviation of noise. By one minute it is smaller by a factor of fifty, and the cost line has not moved at all. Every row loses money.

**The intuition:** state the horizon or the number is meaningless. "Imbalance predicts returns" is true at one second and false at one minute, and the cost of crossing is constant across both.

## Adverse selection: the same number prices your fill

So the signal belongs to whoever posts rather than crosses. Now comes the part candidates almost always miss, and the reason the desk asks about it.

![A two-column comparison showing that when the book is bid-heavy your resting bid sits behind 12,000 shares and rarely fills while the price rises without you, and when the book is ask-heavy you fill quickly and the price then falls](/imgs/blogs/order-book-imbalance-short-horizon-prediction-math-for-quants-5.webp)

You are a market maker resting a bid at \$50.00. When do you actually get filled?

When the book is bid-heavy, you are behind 12,000 shares. Sellers have to chew through all of them before reaching you, so you rarely fill, and the price rises without you. When the book is ask-heavy, the bid queue is short, sellers reach you quickly, and you fill. But an ask-heavy book is precisely the state in which imbalance says the price is about to fall.

Your fills are therefore not a random sample of book states. They concentrate in the states where the signal points against you. **Conditional on being filled, the imbalance leans the wrong way**, and that conditional expectation is exactly the adverse selection cost. This is the queue-based cousin of the classic Glosten-Milgrom picture developed in [the market making simulator post](/blog/trading/quantitative-finance/market-making-simulator-quant-research): the price you get is informative about the trade you just did.

#### Worked example 4: a maker quoting \$10m of notional

You quote both sides of the \$50 stock and get filled on \$10m of notional over the day, which is 200,000 shares.

The naive P&L is easy. Buy at \$50.00 against a mid of \$50.01, and you capture half a spread, 1 cent a share. On 200,000 shares that is **\$2,000 a day**, gross.

But the mid is not fair value. The micro-price is. If the average imbalance at the moment your bid fills is $I = -0.50$, then fair value at your fill is

$$
P_w = 50.01 + \frac{-0.50}{2}\times 0.02 = 50.005
$$

That is \$50.005. You paid \$50.00 for something worth \$50.005. Your true capture is 0.5 cents, not 1 cent. Sweeping the assumed conditional imbalance across a range, always on the same 200,000 shares:

| Imbalance at fill | Fair value at fill | Capture per share | Gross daily P&L |
| --- | --- | --- | --- |
| 0.00 | \$50.010 | 1.00 cents | \$2,000 |
| -0.30 | \$50.007 | 0.70 cents | \$1,400 |
| -0.50 | \$50.005 | 0.50 cents | \$1,000 |
| -0.80 | \$50.002 | 0.20 cents | \$400 |
| -1.00 | \$50.000 | 0.00 cents | \$0 |

The last row is the whole business in one line. At $I = -1$ nothing is left on the bid except you, fair value has collapsed onto your own price, and you capture nothing at all while carrying full inventory risk.

**The intuition:** imbalance is not a signal a maker trades on top of quoting. It is a term inside the quote. Skewing your quotes with it is defence, not offence.

## The trade-offs

| The simple view | What actually happens |
| --- | --- |
| Imbalance predicts returns | It predicts the *next mid move*, on a horizon of seconds |
| More book levels means more information | Deeper levels are cancellable and mostly dilute the signal |
| A strong signal is tradeable | Its maximum displacement is half a spread, and crossing costs a full one |
| Post on the heavy side to ride the move | The heavy side is where you never get filled |
| Adverse selection is a separate risk | It is the same number, evaluated conditional on your fill |

## Common misconceptions

**"A high R-squared means a profitable strategy."** Order flow imbalance explains short-horizon price changes well, and Cont, Kukanov and Stoikov document a linear relation between order flow imbalance and price changes with a slope inversely proportional to depth. Explaining a move is not capturing it. Costs live entirely outside the regression.

**"Imbalance is a leading indicator, so it predicts the next hour."** It predicts the next queue depletion. The state variable is refreshed constantly, and correlation with returns falls away over seconds to tens of seconds. Quoting an unqualified number without a horizon is the single most common error in this topic.

**"Big size on the bid means real buyers."** Resting size is free to cancel and can be withdrawn in microseconds. The touch imbalance predicts anyway because it is the queue that is mechanically binding, but treating deep-book size as a commitment is how people get spoofed.

**"If everyone knows it, it should be arbitraged away."** It cannot be, because the mechanical channel is not an inefficiency. Somebody has to be at the front of the thin queue. Being paid for that position is the compensation for adverse selection, not an anomaly.

**"I can use it to time my retail order."** Directionally it might shave a fraction of a cent. Against a 2 cent spread and a several-second decay, that is inside the noise on any order you can actually place.

## Sources and further reading

- Rama Cont, Arseniy Kukanov and Sasha Stoikov, "The Price Impact of Order Book Events", *Journal of Financial Econometrics* 12(1), 2014, 47-88. [Journal](https://academic.oup.com/jfec/article-abstract/12/1/47/816163) and [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1712822). Establishes the linear relation between order flow imbalance and price changes across fifty US stocks, with a slope inversely proportional to market depth, robust across stocks and time scales.
- Martin Gould, Mason Porter, Stacy Williams, Mark McDonald, Daniel Fenn and Sam Howison, "Limit Order Books", *Quantitative Finance* 13(11), 2013, 1709-1742. [arXiv:1012.0349](https://arxiv.org/abs/1012.0349). The standard survey of empirical regularities and models of the book.
- Alexander Lipton, Umberto Pesavento and Michael Sotiropoulos, "Trade arrival dynamics and quote imbalance in a limit order book", 2013. [arXiv:1312.0514](https://arxiv.org/abs/1312.0514). Derives the probability of the next price move as a function of quote imbalance, and reports imbalance as a strong predictor of average mid-price movement.
- Sasha Stoikov, "The micro-price: a high-frequency estimator of future prices", *Quantitative Finance* 18(12), 2018. [SSRN](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694). The martingale-corrected refinement of the weighted mid.

On measuring decay in your own data, see [evaluating alpha signals](/blog/trading/quantitative-finance/evaluating-alpha-signals-ic-sharpe-turnover-quant-research), which covers information coefficients by horizon and the turnover cost of fast signals.

One honesty note on the numbers above. The qualitative empirical claims, that imbalance predicts short-horizon mid moves and that order flow imbalance relates linearly to price change with a depth-dependent slope, come from the sources listed. Everything with a dollar sign on it, including the 0.6 cent displacement, the assumed decay path and the conditional imbalance at fill, is illustrative arithmetic on assumed inputs, computed on a hypothetical \$50 stock. No coefficient, R-squared or decay half-life in this post is presented as measured.

## In the interview room and on the desk

Jump and Citadel Securities weight this material heavily, and some version of it turns up in every market-making round. The question is usually open: *what would you use to predict the next tick?*

A weak answer names momentum, or a moving average, or something with a lag measured in minutes. A merely acceptable answer says "order book imbalance" and stops. A strong answer does three things in sequence, and the sequence is the signal the interviewer is reading.

First, name it and define it precisely: touch imbalance, $(Q_b - Q_a)/(Q_b + Q_a)$, computed at the top of book. Second, state the horizon before you are asked. "It predicts the next mid move on a horizon of seconds for a liquid name" is a completely different answer from "it predicts returns", and only one of them sounds like someone who has looked at data. Third, and this is the step that separates candidates, raise adverse selection yourself. Say out loud that the same number pricing the move also prices your fill, that your resting bid fills disproportionately when imbalance is negative, and that a maker therefore uses imbalance to skew quotes rather than to take positions.

The trap is presenting it as tradeable alpha. If you say you would buy when imbalance is positive, the next question is what you pay to do that, and the arithmetic in worked example 2 answers it: half a spread of maximum displacement against a full spread of round-trip cost. Whoever is on the other side of your resting order is reading the same number off the same feed, and they were reading it before you crossed.

On the desk, the practical use is narrower than the theory suggests and more valuable. Imbalance goes into fair-value estimation, into quote skew, into the decision to cancel and re-post, and into execution: a child order routed with the imbalance rather than against it saves fractions of a cent, which on institutional volume is real money. It does not go into a signal book as alpha. If you are building one, [the alpha signal construction post](/blog/trading/quantitative-finance/building-an-alpha-signal-quant-research) is the better starting point, and treat imbalance as a cost model input rather than a return forecast.

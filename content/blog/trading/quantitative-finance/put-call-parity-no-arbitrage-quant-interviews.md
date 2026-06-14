---
title: "Put-call parity and no-arbitrage bounds for quant interviews"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A whole class of derivatives interview questions needs no pricing model at all. Build put-call parity, model-free option price bounds, the convexity-in-strike rule, and the early-exercise rule from one law of no-arbitrage, then solve the exact problems Jane Street, Optiver, SIG, IMC, and Citadel Securities ask."
tags:
  [
    "put-call-parity",
    "no-arbitrage",
    "options",
    "static-replication",
    "quant-interviews",
    "derivatives",
    "early-exercise",
    "synthetic-forward",
    "box-spread",
    "quantitative-finance"
  ]
category: "trading"
subcategory: "Quantitative Finance"
author: "Hiep Tran"
featured: true
readTime: 42
---

> [!important]
> **TL;DR** — A surprising number of derivatives-trader interview questions need no pricing model, no Black-Scholes, no normal distribution. They need exactly one idea: two portfolios that pay the same thing in every future state must cost the same today.
>
> - **Put-call parity is the staple**: for European options on a non-dividend stock, $C - P = S - K e^{-rT}$. A call minus a put equals the stock minus the discounted strike. That is an *identity*, not an approximation — it holds whatever your volatility view.
> - **When parity breaks, you arbitrage it**: the conversion (when the call is rich) and the reversal (when the put is rich) assemble a riskless box that banks cash today and pays exactly $0 at expiry. We will lock in **\$1** from a \$6 call and a \$4 put.
> - **Prices are boxed in model-free**: a call is worth at most the stock ($C \le S$) and at least its discounted intrinsic ($C \ge S - K e^{-rT}$). A put has mirror bounds. Any quote outside the box is free money.
> - **Price is convex in strike**: $C(K)$ falls and bows downward as $K$ rises, so a butterfly spread can never cost a negative amount. Violate it and you collect a riskless profit.
> - **The early-exercise rule falls out for free**: you never exercise an American *call* early on a non-dividend stock (selling it always beats it), but an American *put* can be optimal to exercise early. This is the single most common "gotcha" on the floor.
> - **Why desks ask this**: conversions, reversals, and box spreads are real trades, and parity is a hard constraint every market-maker's option quotes must satisfy. They are testing whether you can reason about replication under pressure.

Here is a question that has tripped up more derivatives-desk candidates than almost any other, and it has nothing to do with the Black-Scholes formula everyone crams the night before. A stock trades at **\$100**. A one-year **\$100-strike call** on it trades at **\$6**. A one-year **\$100-strike put** at the same strike trades at **\$4**. The one-year interest rate is **5%**. The interviewer slides this across the table and asks: *is there a free lunch here, and if so, how big?*

Most candidates reach for a model. They try to guess the stock's volatility, plug numbers into Black-Scholes, and check whether \$6 and \$4 are "fair." That is the wrong move, and a good interviewer is delighted when you make it, because it tells them you do not yet see the structure. The right move uses no model at all. It uses one law — *no-arbitrage* — and one technique — *static replication*. With those two tools you will find that this particular set of prices contains a guaranteed, riskless **\$1** of profit, and you will be able to write down the exact trade that captures it.

![Static replication plus the no-arbitrage law produces put-call parity, option price bounds, the convexity rule, and the early-exercise rule with no pricing model](/imgs/blogs/put-call-parity-no-arbitrage-quant-interviews-1.png)

The diagram above is the mental model for this entire article. On the left sits the one law — *no-arbitrage*: if two portfolios deliver identical payoffs in every possible future, they must trade at the same price today, because otherwise you could buy the cheap one, sell the expensive one, and pocket the difference for free. From that law, plus *static replication* (building a complicated payoff out of simpler pieces you set up once and never touch again), four results cascade out: put-call parity, model-free price bounds, convexity of price in the strike, and the early-exercise rule. None of them needs a probability distribution. All of them show up in interviews. By the end of this post you will derive each one yourself and solve the exact problems the top trading firms ask.

This matters far beyond the interview room. The firms that lean hardest on these questions — Jane Street, Optiver, SIG, IMC, Citadel Securities, Jump, DRW — are market-makers. Their literal business is quoting two-sided prices on thousands of options at once and never getting picked off. A market-maker whose call and put quotes violate parity has just printed a banknote for whoever notices first. So these are not trivia questions. They are a direct test of the reasoning the job runs on all day.

We will build the whole thing from absolute zero. No options background is assumed. Every term is defined the first time it appears. By the end you will be able to derive parity on a napkin, bound any option price without a model, spot a butterfly arbitrage across three strikes, and explain — out loud, the way an interviewer wants to hear it — exactly why you never exercise that American call early.

## Foundations: calls, puts, payoffs, and the forward price

Before any clever no-arbitrage arguments, we need a small vocabulary, defined precisely. Skip nothing here; everything that follows rests on these definitions.

### What an option actually is

An **option** is a contract that gives its owner the *right, but not the obligation*, to trade an underlying asset at a fixed price on or by a fixed date. That little phrase "but not the obligation" is the whole point — the owner only acts when it pays to act, and walks away otherwise.

There are two flavors:

- A **call option** gives the owner the right to **buy** the underlying at a fixed price. You buy a call when you want exposure to the stock going *up*.
- A **put option** gives the owner the right to **sell** the underlying at a fixed price. You buy a put when you want to profit from, or protect against, the stock going *down*.

Four numbers define a vanilla option:

- The **underlying** — the asset the option is on. We will use a single share of stock, currently priced at $S$ (the *spot price*, meaning the price for immediate delivery right now). In our running example $S = \$100$.
- The **strike price** $K$ — the fixed price at which the owner may trade. In our example $K = \$100$.
- The **expiry** (or *expiration*) $T$ — the date the right ends, measured in years from now. Our example uses $T = 1$ year.
- The **premium** — the price you pay *today* to own the option. That is the \$6 for the call and \$4 for the put.

One more distinction that will matter enormously later:

- A **European option** can be exercised *only at expiry*, on the final date and not before.
- An **American option** can be exercised *at any time* up to and including expiry.

Despite the names, both trade all over the world; the labels are pure jargon for "exercise only at the end" versus "exercise whenever you like." Most of this article uses European options, because their tidy "only at expiry" structure makes the no-arbitrage arguments clean. We return to the American case — and why early exercise is sometimes, but rarely, worth it — near the end.

### The payoff at expiry

To *exercise* an option is to use the right it grants. The **payoff** is the cash value of exercising at expiry, ignoring the premium you already paid. Let $S_T$ be the stock price *at expiry* (the subscript $T$ means "at time $T$"). Then:

- A call is worth $\max(S_T - K, 0)$ at expiry. If the stock finished at $S_T = \$120$ and your strike is \$100, you exercise: buy at \$100, the share is worth \$120, you made \$20. If the stock finished at $S_T = \$90$, you do *not* exercise — why buy at \$100 something worth \$90? — so the call expires worthless, paying \$0. The payoff is therefore "the bigger of $S_T - K$ and zero," written $(S_T - K)^+$ (the superscript plus means "take the positive part, floor at zero").
- A put is worth $\max(K - S_T, 0) = (K - S_T)^+$. If the stock finished at \$70 and your strike is \$100, you exercise the right to sell at \$100 something worth \$70, making \$30. If it finished at \$120, you let the put expire worthless.

These two payoff shapes are the atoms of everything that follows. Let us draw them.

![A $5 premium $100-strike call caps the buyer loss at $5, breaks even at $105, and has unlimited upside above the strike](/imgs/blogs/put-call-parity-no-arbitrage-quant-interviews-2.png)

The figure above is a **payoff diagram**: the stock price at expiry runs along the horizontal axis (in dollars), and your profit or loss runs up the vertical axis (also in dollars). The green region is profit; the red region is loss — a convention we will keep across every chart in this post. This particular line is a *long call* (you bought it) struck at \$100 for a \$5 premium. Below \$100 the call expires worthless, so you lose exactly your \$5 premium — a flat red floor. Above \$100 the call pays \$1 for every \$1 the stock rises, but you are still down your \$5 premium until the stock clears **\$105**, the *breakeven*. Above \$105 you are in pure profit, and because a stock can in principle rise without limit, your upside is unlimited. The single most important features: **your loss is capped at the premium, and your upside is open-ended.**

The put is the mirror image.

![A $4 premium $100-strike put caps the buyer loss at $4, breaks even at $96, and gains as the stock falls toward zero](/imgs/blogs/put-call-parity-no-arbitrage-quant-interviews-3.png)

Here is a *long put* struck at \$100 for a \$4 premium. To the right of \$100 the put expires worthless and you lose your \$4 premium — the flat red floor on the right. As the stock falls below \$100 the put gains \$1 for every \$1 of decline; you break even at **\$96** (strike minus premium) and profit all the way down. The most the put can be worth is \$100 — if the stock goes to zero, your right to sell at \$100 is worth the full \$100 — so unlike the call, the put's upside is large but *bounded*. Again: **loss capped at the premium, gain large as the stock falls.**

### Intrinsic value and time value

An option's premium splits cleanly into two pieces, and interviewers love to probe whether you understand the split.

The **intrinsic value** is what the option would be worth if it expired *right now*: $\max(S - K, 0)$ for a call, $\max(K - S, 0)$ for a put. It is the payoff evaluated at today's spot. With $S = \$100$ and $K = \$100$, both our options have intrinsic value \$0 — they are *at-the-money* (strike equals spot). A call with $S > K$ is *in-the-money* (it has positive intrinsic value); with $S < K$ it is *out-of-the-money*.

The **time value** is everything else: premium minus intrinsic value. Our \$6 call has \$0 intrinsic value, so all \$6 is time value — the market's price for the *chance* the stock rallies before expiry. Time value comes from two sources: the *optionality* (the stock might move favorably, and you are protected if it does not) and, for the call, the *interest* you earn by deferring payment of the strike. Time value is always non-negative and decays to zero at expiry, where only intrinsic value remains.

![A call price equals its kinked intrinsic-value floor plus a green time-value gap that is largest at-the-money and shrinks to zero at expiry](/imgs/blogs/put-call-parity-no-arbitrage-quant-interviews-11.png)

The figure above makes the split visible. The kinked dashed line is the *intrinsic value* — flat at zero below the \$100 strike, then rising at 45 degrees. The smooth curve above it is the actual *call price* before expiry. The green gap between them is the *time value*, and notice that it is fattest right around the strike (at-the-money) and thins out as the option goes deep in- or out-of-the-money. As expiry approaches, that smooth curve sags down onto the kinked floor and the green gap vanishes. We will use the fact that the curve always sits *on or above* the floor as one of our model-free bounds.

### The forward price: the fair price for future delivery

One last foundation, and it is the bridge to parity. A **forward contract** is an agreement to buy an asset at a fixed price on a future date — no optionality, you *must* transact. The fixed price written into the contract is the **forward price**, $F$. Unlike an option, a forward costs nothing to enter; the price $F$ is set so the deal is fair to both sides at inception.

What is the fair forward price for delivery of a non-dividend stock in $T$ years? Here is the no-arbitrage argument in full, because it is the template for every argument in this article. Suppose you want to own a share at time $T$. You have two ways to guarantee that:

1. **Enter a forward now** at price $F$, pay nothing today, and pay $F$ at time $T$ to receive the share.
2. **Buy the share today** for $S$, borrowing the \$$S$ at the risk-free rate $r$. At time $T$ you owe $S e^{rT}$ on the loan and you hold the share.

Both routes leave you holding exactly one share at time $T$. Route 1 costs you $F$ at time $T$; route 2 costs you $S e^{rT}$ at time $T$. Since the *outcome is identical*, the *cost must be identical*, or someone is leaving free money on the table. Therefore:

$$F = S e^{rT}.$$

Here $e^{rT}$ is the **continuously-compounded growth factor**: a dollar invested at rate $r$ for $T$ years grows to $e^{rT}$ dollars. (If you prefer simple annual compounding, replace $e^{rT}$ with $(1+r)^T$; the logic is identical and the numbers barely differ for one year. We use $e^{rT}$ because it is the market convention and keeps the algebra clean.) The mirror quantity $e^{-rT}$ is the **discount factor**: a dollar *to be received* in $T$ years is worth only $e^{-rT}$ dollars today.

A quick number to anchor it. With $S = \$100$, $r = 5\%$, $T = 1$:

$$F = 100 \cdot e^{0.05 \cdot 1} = 100 \cdot 1.0513 = \$105.13.$$

And the discounted strike — the present value of \$100 to be paid in one year — is

$$K e^{-rT} = 100 \cdot e^{-0.05} = 100 \cdot 0.9512 = \$95.12.$$

Hold onto that \$95.12. It is the single most important number in this whole article. **The discounted strike is the cash you set aside today that grows to exactly the strike by expiry.** Every no-arbitrage result below is built around it.

> **The one-sentence intuition:** an option is a one-sided bet (loss capped at the premium, asymmetric payoff), a forward is a two-sided obligation priced so it costs nothing to enter, and the discounted strike $K e^{-rT}$ is the bridge that ties them together.

## Deriving put-call parity by building two portfolios

Now we earn the headline result. Put-call parity is a *relationship* — an identity, not a forecast — that ties together the call price $C$, the put price $P$, the spot $S$, and the discounted strike $K e^{-rT}$. We will derive it with nothing but the no-arbitrage law and a pair of carefully chosen portfolios.

The trick of *static replication* is to find two different bundles of instruments that pay exactly the same amount in every possible future state, then invoke the law: equal payoffs imply equal price.

Consider two portfolios, both using our \$100 strike and one-year expiry.

**Portfolio A: one call, plus cash $K e^{-rT}$.** You own one \$100-strike call, and you hold \$95.12 in cash invested at the risk-free rate. That cash grows to exactly \$100 by expiry. What does Portfolio A pay at time $T$?

- If $S_T > 100$ (stock finished above the strike): you exercise the call, paying \$100 (which you have, from the grown cash) to receive a share worth $S_T$. You end with the share, worth $S_T$.
- If $S_T \le 100$ (stock finished at or below the strike): the call expires worthless, and you keep your \$100 cash.

So Portfolio A pays $\max(S_T, 100)$ — the *greater* of the stock price and the strike.

**Portfolio B: one put, plus one share of stock.** You own one \$100-strike put and one share. What does Portfolio B pay at time $T$?

- If $S_T < 100$: you exercise the put, selling your share for \$100. You end with \$100 cash.
- If $S_T \ge 100$: the put expires worthless, and you keep your share, worth $S_T$.

So Portfolio B *also* pays $\max(S_T, 100)$.

![Portfolio A of a call plus cash and Portfolio B of a put plus a share both pay max of the stock and the strike at expiry, so they must cost the same today](/imgs/blogs/put-call-parity-no-arbitrage-quant-interviews-4.png)

The figure traces both portfolios state by state and lands on the same payoff, $\max(S_T, \$100)$, in every case. This is the heart of the argument. Two portfolios that pay *identically* in *every* future — whether the stock soars, crashes, or sits still — must cost *identically* today. If they did not, you would buy the cheap one, short the expensive one, and the future payoffs would cancel perfectly, leaving you with the price difference as riskless profit. The no-arbitrage law forbids that, so the prices are equal:

$$\underbrace{C + K e^{-rT}}_{\text{Portfolio A cost today}} = \underbrace{P + S}_{\text{Portfolio B cost today}}.$$

Rearrange, and you have **put-call parity**:

$$\boxed{\,C - P = S - K e^{-rT}\,}$$

In words: *a call minus a put equals the stock minus the discounted strike.* Notice what is **not** in this equation: volatility, the probability the stock rises, any distribution, any model. Parity holds for *any* arbitrage-free market, whatever the true dynamics. That is what makes it bulletproof in an interview — you can derive it from scratch in ninety seconds and you never have to defend an assumption about how the stock moves.

#### Worked example: checking parity on the \$100 stock

Let us plug in the running numbers. $S = \$100$, $K = \$100$, $r = 5\%$, $T = 1$, so $K e^{-rT} = \$95.12$. The right-hand side of parity is

$$S - K e^{-rT} = 100 - 95.12 = \$4.88.$$

So parity *demands* that $C - P = \$4.88$. The market is quoting $C = \$6$ and $P = \$4$, giving $C - P = \$6 - \$4 = \$2.00$.

But \$2.00 is **not** \$4.88. The call-minus-put the market is charging is \$2.88 *too small* — the call is too cheap, or the put is too expensive, or some mix, relative to the iron law of parity. Parity is violated by \$2.88, and a violation of parity is, by construction, a free lunch. The next section shows exactly how to eat it.

> **The one-sentence intuition:** put-call parity is not a pricing model — it is an accounting identity forced by no-arbitrage, so any market where $C - P \ne S - K e^{-rT}$ is literally giving money away.

## The arbitrage when parity breaks: conversions and reversals

When parity is violated, the trade that captures the gap has a name on every options desk: a **conversion** when the call is rich (overpriced) relative to the put, and a **reversal** (or *reverse conversion*) when the put is rich. Together with the bond leg they form a **box** — a position whose payoff at expiry is a known constant, so it carries zero market risk. Let us build the conversion from our numbers.

Recall parity says $C - P$ *should* be \$4.88 but the market has it at \$2.00. The synthetic "call minus put" is too cheap. So we want to **buy** the cheap synthetic (buy the call, sell the put) and **sell** the expensive real thing it should equal (sell the stock, lend out the discounted strike) — or, equivalently for a conversion, do the reverse legs so the cash lands today. The mechanical recipe: trade *every* leg of parity so that the future payoffs cancel and the mispricing falls out as cash now.

Here are the four legs and what each one does, today and at expiry. We will use the version that books the profit immediately.

To capture "the synthetic call-minus-put is too cheap," we **buy** the underpriced side. The clean construction: buy the call, sell the put, short the stock, and lend the discounted strike. Let us tabulate.

![A conversion table: sell the rich call, buy the put, buy the stock, borrow the discounted strike, for zero risk at expiry and one dollar banked today](/imgs/blogs/put-call-parity-no-arbitrage-quant-interviews-5.png)

The grid above lays out the classic **conversion** — the trade you do when the *call is rich*. Read it row by row, watching the two columns: cash today, and net payoff at expiry. The beauty is in the right column: every state-dependent piece ($(S_T-100)^+$, $(100-S_T)^+$, $S_T$, the \$100 owed) cancels to a constant, leaving zero risk at expiry, while the left column nets out to cash in your pocket today.

But in *our* market the call is **cheap**, not rich (parity wants $C-P=\$4.88$ but the market has \$2.00, so the call is underpriced relative to the put). So we run the conversion *in reverse* — a **reversal**. Buy the cheap call, sell the rich put, short the stock, and lend the discounted strike. Let us walk the cash flows explicitly.

#### Worked example: locking in \$1 from the \$6 call and \$4 put

We have $C = \$6$, $P = \$4$, $S = \$100$, $K = \$100$, $r = 5\%$, $T = 1$. Parity is violated by \$2.88. We claimed \$1 of profit in the TL;DR — let us see exactly where it comes from, and why it is \$2.88 of *parity gap* but a specific dollar amount of *captured profit* once we account for the cost of every leg.

Set up the **reversal** today:

- **Buy the call** for \$6. Cash flow today: $-\$6$.
- **Sell (write) the put** for \$4. Cash flow today: $+\$4$.
- **Short the stock** at \$100 (borrow a share, sell it). Cash flow today: $+\$100$.
- **Lend** the present value of the strike, \$95.12, at the risk-free rate so it grows to exactly \$100 at expiry. Cash flow today: $-\$95.12$.

Net cash today:

$$-6 + 4 + 100 - 95.12 = +\$2.88.$$

You have **\$2.88 in your pocket right now**, before anything has happened. Now check that the future is risk-free. At expiry, two cases:

- **If $S_T \ge 100$:** your call is in-the-money — exercise it, pay \$100 (you have it: the loan grew to \$100), receive a share. Use that share to close your short. The put you wrote expires worthless. Net at expiry: you spent your \$100 loan proceeds and your short is covered — exactly \$0.
- **If $S_T < 100$:** the put you wrote is exercised against you — the buyer sells *you* a share for \$100, which you pay from your \$100 loan proceeds. Use that share to close your short. Your call expires worthless. Net at expiry: again exactly \$0.

Either way, the expiry payoff is **\$0** — no market risk, no dependence on where the stock lands. You banked **\$2.88** at inception and owe nothing later. That \$2.88 is the full parity gap, captured as riskless profit.

So why did the TL;DR say \$1? Because in any realistic version of this problem there are *frictions* — the bid-ask spread on each leg, the borrow cost to short the stock, exchange fees, and the haircut on the cash you post. A \$2.88 theoretical edge routinely shrinks to a *realized* dollar or less after costs, which is exactly why these arbitrages are rare and fleeting in liquid markets: market-makers compete them down to the friction floor. The clean, frictionless answer the interviewer wants first is **\$2.88**; the grown-up follow-up — "and after costs you'd realize maybe a dollar" — is what separates a candidate who has only read a textbook from one who has thought about the trade.

![A reversal trade laid out on a timeline: buy the cheap call, sell the rich put, short the stock, lend the discounted strike, bank cash today and net zero at expiry](/imgs/blogs/put-call-parity-no-arbitrage-quant-interviews-12.png)

The timeline above sequences the reversal: at $t=0$ you assemble a synthetic long stock (long call, short put) against a real short stock, fund it by lending the discounted strike, and bank the gap. At expiry the synthetic long and the real short cancel exactly, so your profit was locked the moment you set up — nothing about the stock's path can touch it.

> **The one-sentence intuition:** a parity violation is not a "trading opportunity" in the risky sense — it is a riskless box that pays you the gap today and exactly zero at expiry, which is why desks hunt these gaps to the penny.

## Model-free no-arbitrage price bounds

Parity is an *equality*. The next family of results are *inequalities* — they do not pin down an option's price exactly, but they fence it into a region, again with no model. Interviewers love these because they reward clean replication arguments and punish formula-cramming.

### The upper bound: a call is never worth more than the stock

A call gives you the right to buy a share for $K$. The most that right could ever be worth is the share itself — owning the call can never be better than owning the share outright, because the share *is* the thing the call lets you (conditionally, and at a cost) acquire. Formally:

$$C \le S.$$

The arbitrage if it were violated: suppose a call traded for $C > S$. You would **sell the call** for $C$ and **buy the stock** for $S$, banking $C - S > 0$ today. If the call is ever exercised against you, you already own the share to deliver — you hand it over and keep the \$$K$ strike on top. You can never lose. A price $C > S$ is therefore impossible.

### The lower bound: a call is worth at least its discounted intrinsic value

The richer bound comes straight from parity. Since a put price $P$ can never be negative (an option is a right, never an obligation, so it is worth at least \$0), put-call parity $C = P + S - K e^{-rT}$ with $P \ge 0$ gives

$$C \ge S - K e^{-rT}.$$

And since a call price also can never be negative, we combine:

$$C \ge \max\!\big(S - K e^{-rT},\, 0\big).$$

The arbitrage if the lower bound were violated: suppose $C < S - K e^{-rT}$. You would **buy the call** for $C$, **short the stock** for $S$, and **lend** $K e^{-rT}$ (growing to $K$ at expiry). Net cash today is $S - K e^{-rT} - C > 0$, banked immediately. At expiry your loan returns $K$, which you use to buy the share back (via the call if it is cheaper, or in the market) and close the short. Riskless profit again.

Putting the two bounds together fences the call into a wedge.

![A European call must sit in the wedge between the floor S minus the discounted strike and the 45-degree line C equals S](/imgs/blogs/put-call-parity-no-arbitrage-quant-interviews-6.png)

The figure plots call value against stock price, both in dollars on the same scale. The upper black line is $C = S$ — the call can never poke above this 45-degree line. The lower black line is the floor $\max(S - K e^{-rT}, 0)$, here kinking at the discounted strike \$95. The shaded green wedge between them is the *only* region a no-arbitrage call price can live in. Any quote above the top line or below the bottom line is a free lunch, capturable with the exact trades above — no view on volatility required.

### The put's bounds

The put has mirror bounds, derivable the same two ways. A put can be worth at most its strike, and in fact at most the *discounted* strike for a European put (the most you can ever collect is \$$K$ at expiry, worth $K e^{-rT}$ today):

$$P \le K e^{-rT}.$$

And from parity $P = C - S + K e^{-rT}$ with $C \ge 0$:

$$P \ge \max\!\big(K e^{-rT} - S,\, 0\big).$$

#### Worked example: bounding a call price model-free

Suppose an interviewer says: *the stock is \$50, there is a one-year \$45-strike European call, and the one-year rate is 4%. Without any model, how tightly can you box the call's price?* You compute:

- Upper bound: $C \le S = \$50$.
- Discounted strike: $K e^{-rT} = 45 \cdot e^{-0.04} = 45 \cdot 0.9608 = \$43.24$.
- Lower bound: $C \ge S - K e^{-rT} = 50 - 43.24 = \$6.76$.

So the call must trade between **\$6.76 and \$50** — no distribution assumed, no Black-Scholes. If someone quoted you that call at \$6.50, you would know *instantly*, with certainty, that it is mispriced and arbitrageable, even without knowing its "fair" value. The interviewer is not testing whether you can value the option; they are testing whether you can *fence* it, which is the skill a market-maker actually uses to avoid getting picked off.

> **The one-sentence intuition:** you do not need to know an option's exact price to make money off it — you only need to catch it outside the model-free box, and the box comes entirely from "a portfolio can't cost less than a strictly-better portfolio."

## Monotonicity and convexity in the strike: the butterfly must be non-negative

So far we have varied the stock price. Now hold the stock fixed and vary the *strike*. Two more model-free laws fall out, and the second one — convexity — is a perennial interview favorite because the arbitrage trade has a memorable name: the **butterfly**.

### Monotonicity: higher strike, cheaper call

A call with a higher strike gives you a *worse* right — you have to pay more to buy the same share. So call value must *fall* as the strike rises:

$$K_1 < K_2 \implies C(K_1) \ge C(K_2).$$

The arbitrage if violated (a higher-strike call costing *more*): sell the expensive high-strike call, buy the cheaper low-strike call. This is a **bull call spread** bought for a *credit* — you are paid to put it on, and its worst-case payoff is \$0, so you can never lose. Impossible, hence monotonicity. (There is also a bound on *how fast* the price can fall: $C(K_1) - C(K_2) \le (K_2 - K_1) e^{-rT}$, because the spread's payoff can never exceed the strike difference, discounted.)

### Convexity: the price curve bows downward

The deeper law is that $C(K)$ is **convex** in the strike — the price curve bends *upward* (bows downward as a falling curve), so a chord connecting two strikes lies *above* the curve in between. Concretely, for three equally-spaced strikes $K_1 < K_2 < K_3$ with $K_2$ the midpoint:

$$C(K_2) \le \tfrac{1}{2}\big(C(K_1) + C(K_3)\big).$$

The combination on the right minus the left — *buy one $K_1$ call, buy one $K_3$ call, sell two $K_2$ calls* — is exactly a **butterfly spread**. Its payoff at expiry is a little tent: zero outside $[K_1, K_3]$, peaking at $K_2 - K_1$ when the stock lands on the middle strike, and never negative anywhere. A position whose payoff is *never negative* must cost a *non-negative* amount today — otherwise you would be paid to hold something that can only ever pay you more. Therefore the butterfly's cost,

$$C(K_1) - 2\,C(K_2) + C(K_3) \ge 0,$$

is the convexity condition. Flip the inequality and you have found an arbitrage.

![A convex call-price curve in the strike: the chord between the $95 and $105 strikes lies above the curve at $100, so the butterfly costs a non-negative $0.25](/imgs/blogs/put-call-parity-no-arbitrage-quant-interviews-7.png)

The figure plots call price against strike. The solid curve falls (monotonicity) and bows downward (convexity). The three blue dots are three strikes: $C(\$95) = \$8.50$, $C(\$100) = \$6.00$, $C(\$105) = \$4.00$. The dashed chord connects the outer two; its midpoint value is $\tfrac{1}{2}(8.50 + 4.00) = \$6.25$, which sits *above* the actual \$6.00 of the middle strike. The gap, \$0.25, is exactly the butterfly's cost — non-negative, as convexity demands. If instead the middle call had been quoted at, say, \$6.40, the butterfly would cost $8.50 - 2(6.40) + 4.00 = -\$0.30$: a *negative* cost, meaning you are paid \$0.30 to put on a position that can only ever pay you more. That is a textbook convexity arbitrage.

#### Worked example: spotting a butterfly arbitrage across three strikes

An interviewer gives you three one-year European calls on the same stock: the \$95-strike at \$8.50, the \$100-strike at \$6.50, and the \$105-strike at \$4.00. Is there an arbitrage?

Compute the butterfly cost:

$$C(95) - 2\,C(100) + C(105) = 8.50 - 2(6.50) + 4.00 = 8.50 - 13.00 + 4.00 = -\$0.50.$$

It is **negative** — convexity is violated. So you put on the butterfly to *collect* the \$0.50: **buy one \$95 call** ($-\$8.50$), **buy one \$105 call** ($-\$4.00$), and **sell two \$100 calls** ($+\$13.00$). Net cash today: $+\$0.50$, banked immediately. Now check the payoff at expiry across the regions:

- $S_T \le 95$: all three calls expire worthless. Payoff \$0.
- $95 < S_T \le 100$: only the \$95 call pays, worth $S_T - 95 \ge 0$. Payoff $\ge 0$.
- $100 < S_T \le 105$: the \$95 call pays $S_T - 95$, and the two short \$100 calls cost you $2(S_T - 100)$. Net: $(S_T - 95) - 2(S_T - 100) = 105 - S_T \ge 0$.
- $S_T > 105$: all three pay; net $(S_T - 95) + (S_T - 105) - 2(S_T - 100) = 0$.

The payoff is **non-negative in every region** and you were *paid* \$0.50 to put it on. Riskless profit, no model. The lesson the interviewer is checking: convexity in the strike is not an abstract property — it is enforced by a specific, nameable trade, and a market-maker who lets their strike grid go non-convex is handing out free butterflies.

> **The one-sentence intuition:** option prices must fall and curve convexly as the strike rises, because the spreads that bet against monotonicity or convexity have payoffs that can never go negative — and nothing with a never-negative payoff can cost a negative amount.

## American versus European: the early-exercise rule

Now the most famous gotcha on any derivatives desk, and the one candidates get wrong most often. Recall that an **American** option can be exercised any time, a **European** only at expiry. The question that follows is irresistible to interviewers: *given the freedom to exercise early, when should you?*

### You never exercise an American call early on a non-dividend stock

Take an American call on a stock that pays no dividends. The claim, which sounds wrong to most people the first time they hear it, is that you should **never** exercise it before expiry. Not "rarely" — *never*. Here is why, and the argument is pure no-arbitrage.

Suppose you are tempted to exercise early when the stock is at $S$. Exercising means paying the strike $K$ now to grab the share, capturing $S - K$ of intrinsic value today. But compare that to simply **selling the call** in the market. We proved above that any call satisfies $C \ge S - K e^{-rT}$. And since interest is positive, $K e^{-rT} < K$, which means

$$C \ge S - K e^{-rT} > S - K.$$

The call is worth *strictly more* than the $S - K$ you would capture by exercising. Selling it always beats exercising it. You give up two things when you exercise early: the **interest** on the strike (you pay \$$K$ now instead of at expiry, forfeiting the time value of that money) and the **downside protection** (after exercising you own the share outright, exposed to a crash; while you held the call your loss was capped at the premium). Both are positive, so early exercise is strictly worse.

![Exercising an American call early forfeits the strike interest and downside protection, so selling the call always dominates exercise](/imgs/blogs/put-call-parity-no-arbitrage-quant-interviews-8.png)

The figure contrasts the two choices. The red column on the left is *exercise early*: you pay the \$100 strike now, lose the interest on it until expiry, lose the downside protection, and capture only $S - \$100$. The green column on the right is *sell the call instead*: you collect $C \ge S - K e^{-rT}$, keep the interest, keep the time value, and — the punchline — $C > S - \$100$ always. The dominant choice is always to sell, never to exercise. The corollary every interviewer wants you to state: for a non-dividend stock, an American call is worth *exactly the same* as a European call, because the early-exercise right is worthless.

### An American put can be optimal to exercise early

The put does *not* enjoy this protection, and understanding the asymmetry is what separates a strong candidate from a memorizer. Consider a deep in-the-money American put — the stock has crashed toward zero. Exercising it now sells your share for the full strike $K$ today. If you wait, the most you can *ever* collect is still $K$ (the stock cannot go below zero, so the put cannot be worth more than $K$), but you collect it *later*, forgoing the interest on that $K$ in the meantime.

So for a put, early exercise *captures interest* on the strike rather than forfeiting it — the exact opposite of the call. When the put is deep enough in-the-money that the interest you would earn on the strike outweighs the remaining optionality (the small chance the stock bounces back), it is optimal to exercise early. The American put is therefore worth *strictly more* than the European put, and pinning down the optimal early-exercise boundary is genuinely hard (it has no closed form, which is why American puts are priced numerically). But the *qualitative* rule — call never, put maybe — you can state instantly, and you should.

The asymmetry in one line: exercising a call early means *paying* the strike early (you lose interest), while exercising a put early means *receiving* the strike early (you gain interest). That single sign flip is the whole story.

#### Worked example: should you exercise this American call early?

An interviewer offers: *a non-dividend stock is at \$130, you hold an American \$100-strike call expiring in six months, the rate is 5%. The stock just jumped and you are tempted to lock in the \$30 of intrinsic value. Exercise?*

No. Exercising captures $S - K = 130 - 100 = \$30$. But the discounted strike is $K e^{-rT} = 100 \cdot e^{-0.05 \cdot 0.5} = 100 \cdot 0.9753 = \$97.53$, so the call's lower bound is

$$C \ge S - K e^{-rT} = 130 - 97.53 = \$32.47.$$

The call is worth *at least* \$32.47 if you sell it — \$2.47 more than the \$30 you would capture by exercising. The \$2.47 is the interest you would forfeit on the \$100 strike over six months, plus whatever sliver of downside protection remains. Selling (or just holding) strictly dominates. The right answer is *never exercise the call early on a non-dividend stock — sell it instead* — and the right *follow-up* is "the only thing that could change this is a dividend large enough to make capturing the stock worth more than the forfeited interest," which is exactly the next section.

> **The one-sentence intuition:** the call-versus-put early-exercise asymmetry is just the sign of the strike cash flow — exercising a call pays the strike early and loses interest, exercising a put receives the strike early and gains interest.

## Put-call parity adjusted for dividends

Everything so far assumed the stock pays no dividends. Real stocks do, and a **dividend** — a cash payment the company makes to shareholders — changes the picture, because *holding the stock* collects the dividend while *holding the option* does not. That asymmetry shifts parity.

A **dividend** is a per-share cash distribution paid on the **ex-dividend date** (the date on which a buyer no longer receives the upcoming payment). On that date the stock price mechanically drops by roughly the dividend amount, because the cash leaves the company. The option holder, who owns a *right* on the share rather than the share itself, never sees that cash.

To re-derive parity, go back to Portfolio B (put plus share). The share now throws off a dividend $D$ at some point before expiry. So Portfolio B's holder pockets an extra $D$ that Portfolio A's holder does not. To make the two portfolios pay identically again, we have to *handicap* Portfolio B by the present value of that dividend. The cleanest way: in Portfolio B, hold the share but immediately "pre-spend" the present value of the dividend, $D e^{-rt}$ (where $t$ is the time until the dividend is paid). Equivalently, parity becomes

$$\boxed{\,C - P = S - D e^{-rt} - K e^{-rT}\,}$$

The stock leg is reduced by $D e^{-rt}$, the present value of the dividend the option holder misses. Everything else is identical. (If there are several dividends, subtract the present value of each.)

![A timeline of parity with dividends: at the ex-dividend date the stock pays $2 to the shareholder, so the stock leg of parity drops by the present value of that dividend](/imgs/blogs/put-call-parity-no-arbitrage-quant-interviews-10.png)

The timeline shows the sequence: you set up the two portfolios today, the stock pays its \$2 dividend at the ex-date to whoever holds the share (Portfolio B's holder, not the option holder), and so parity must be adjusted by subtracting the present value of that \$2 from the stock leg. At expiry both legs again pay $\max(S_T, K)$, as before.

#### Worked example: adjusting parity for a \$2 dividend

Take the running numbers — $S = \$100$, $K = \$100$, $r = 5\%$, $T = 1$ — but now the stock pays a **\$2 dividend** in six months ($t = 0.5$). The present value of that dividend is

$$D e^{-rt} = 2 \cdot e^{-0.05 \cdot 0.5} = 2 \cdot 0.9753 = \$1.95.$$

Dividend-adjusted parity now demands

$$C - P = S - D e^{-rt} - K e^{-rT} = 100 - 1.95 - 95.12 = \$2.93.$$

Without the dividend, parity wanted $C - P = \$4.88$; the \$2 dividend pulls that down to **\$2.93**. The intuition: a dividend makes calls *less* valuable (the stock will drop on the ex-date, hurting the call holder who does not collect the cash) and puts *more* valuable (that same drop helps the put holder), so the call-minus-put gap shrinks. Concretely, the call cheapens and the put richens, and the required spread falls from \$4.88 to \$2.93.

Notice the practical sting: our market quotes $C - P = \$2.00$, which was a \$2.88 violation under no-dividend parity but is only a \$0.93 violation once you account for the dividend. *If you forget the dividend, you will overstate the arbitrage and put on a trade that loses money the moment the stock goes ex.* This is precisely the trap interviewers set with dividend questions — they want to see whether you remember to adjust.

> **The one-sentence intuition:** a dividend leaks value out of the stock that the option holder never sees, so it lowers the stock leg of parity by the present value of the payment — forget it and your "arbitrage" evaporates on the ex-date.

## The synthetic forward, and how parity lets you build anything

Before the interview problems, one more construction that interviewers reach for constantly: parity does not just *constrain* prices, it lets you *manufacture* one instrument out of others. This is the most useful practical consequence of the whole framework.

Rearrange parity as $C - P = S - K e^{-rT}$ and read the left side as a *position*: **long one call, short one put, same strike.** What does that position pay at expiry?

- If $S_T > K$: the call pays $S_T - K$, the put you are short expires worthless. You net $S_T - K$.
- If $S_T < K$: the call expires worthless, the put is exercised against you, costing $K - S_T$. You net $-(K - S_T) = S_T - K$.

In *both* cases the payoff is $S_T - K$ — a straight line. That is exactly the payoff of a **forward contract** to buy the stock at $K$: you are obligated to buy at $K$, so you gain $S_T - K$ whatever happens. Long call plus short put *is* a synthetic forward.

![Long call minus short put at the same strike is a straight line paying the stock price at expiry minus the strike, identical to a forward contract](/imgs/blogs/put-call-parity-no-arbitrage-quant-interviews-9.png)

The figure plots the payoff of "long the \$100 call, short the \$100 put" against the stock price at expiry. The two kinked option payoffs add up to a single straight line, $S_T - \$100$, sloping through the strike: green (profit) above \$100 where the long call carries you, red (loss) below \$100 where the short put bites. No kink, no cap, no floor — exactly a forward's linear obligation. This is why desks call long-call-short-put a *synthetic long*, and short-call-long-put a *synthetic short*: you can take a pure directional stock position using only options, which is invaluable when the options are more liquid than the stock, or when shorting the stock directly is hard or expensive.

#### Worked example: building a synthetic forward from a call and a put

An interviewer asks: *you want a forward to buy the stock at \$100 in one year, but only the options market is liquid. The \$100 call is \$6, the \$100 put is \$4, the rate is 5%. How do you build the forward, and what does it effectively cost?*

Build it by going **long the \$100 call (\$6) and short the \$100 put (\$4)**. Net premium today: $-6 + 4 = -\$2$ (you pay \$2 to put it on). At expiry this pays $S_T - \$100$ exactly, the forward payoff. To check the implied forward price, use parity: the synthetic costs $C - P = \$2$ today and locks in delivery at $K = \$100$. The *fair* synthetic would have $C - P = S - K e^{-rT} = 100 - 95.12 = \$4.88$, so paying only \$2 for it means you are getting the synthetic forward cheap — the same \$2.88 edge we found at the very start, now seen through the lens of "I am buying future stock delivery for less than its no-arbitrage cost." Every one of these problems is the same identity wearing a different hat.

> **The one-sentence intuition:** put-call parity is a recipe as much as a constraint — long call plus short put manufactures a forward, so you can take a clean directional position with options alone and price it off the same identity.

## In the interview room

Time to put it together the way it actually happens: a person across the table, no formula sheet, narrate your reasoning out loud. Below are five fully-solved problems in the style of Jane Street, Optiver, SIG, IMC, and Citadel Securities. The goal is not just the answer — it is the *clean chain of reasoning* that gets there, because that is what is being graded.

#### Worked example: the classic parity arbitrage (the opener)

**Problem.** A stock is \$100. A one-year \$100 European call is \$6, the one-year \$100 European put is \$4, the rate is 5%, no dividends. Is there an arbitrage? Quantify it and give the trade.

**Solution, narrated.** "First I'll write down parity: $C - P = S - K e^{-rT}$. The right side is $100 - 100 e^{-0.05} = 100 - 95.12 = \$4.88$. The market's left side is $6 - 4 = \$2.00$. They disagree by \$2.88, so yes — arbitrage. The synthetic call-minus-put is too cheap at \$2.00 versus the \$4.88 it should be, so I buy the cheap synthetic and sell the expensive real package. Concretely: buy the call ($-\$6$), sell the put ($+\$4$), short the stock ($+\$100$), and lend the discounted strike $\$95.12$ ($-\$95.12$). Net cash today $= +\$2.88$, banked now. At expiry, whether the stock is above or below \$100, my synthetic long stock exactly cancels my real short, and the \$100 loan covers the strike — net \$0. So I lock in \$2.88 riskless, modulo transaction costs and borrow." That last clause — acknowledging frictions — is what makes the answer sound like a trader rather than a textbook.

#### Worked example: which option is mispriced?

**Problem.** Same stock at \$100, same 5% rate, no dividends, one year. The \$100 call is \$6, but now you are told the *true* fair call value is \$6 — the call is correctly priced. The put trades at \$4. Which is mispriced, and what is the fair put?

**Solution, narrated.** "Parity must hold in a fair market: $C - P = S - K e^{-rT} = \$4.88$. If the call is fair at \$6, then the fair put is $P = C - 4.88 = 6 - 4.88 = \$1.12$. The market has the put at \$4 — it's roughly \$2.88 *too expensive*. So I'd sell the rich put and hedge it with the parity-replicating package: sell the \$4 put, and to neutralize, buy the call, short the stock, lend the discounted strike — the reversal. The mispricing lives in the put, and the fair value is \$1.12." The interviewer is checking whether you can isolate *which* leg is off once you're handed a reference price, rather than just flagging that something is wrong.

#### Worked example: the no-model price bound

**Problem.** A non-dividend stock is \$80. A one-year \$70-strike European call is quoted at \$9. The one-year rate is 6%. Without a model, can you say the quote is wrong?

**Solution, narrated.** "I'll bound it. Upper bound, the call can't exceed the stock: $C \le \$80$ — \$9 passes that easily. Lower bound is the binding one: $C \ge S - K e^{-rT}$. The discounted strike is $70 e^{-0.06} = 70 \cdot 0.9418 = \$65.93$, so the floor is $80 - 65.93 = \$14.07$. The quote is \$9, which is *below* the \$14.07 floor — so yes, it's arbitrageable with certainty, no model needed. I'd buy the \$9 call, short the stock at \$80, and lend \$65.93; that banks $80 - 65.93 - 9 = \$5.07$ today and the loan covers buying the share back at expiry. The quote violates the floor by \$5.07." This problem rewards knowing the *floor* cold — most candidates only remember $C \le S$ and miss the lower bound, which is where the money usually is.

#### Worked example: the box spread and the implied rate

**Problem.** You can trade four European options on a \$100 stock expiring in one year: \$90 and \$110 calls and puts. The \$90 call is \$15, the \$90 put is \$3, the \$110 call is \$4, the \$110 put is \$12. What riskless payoff can you build, what does it cost, and what interest rate does that imply?

**Solution, narrated.** "A **box spread** locks in a fixed payoff regardless of the stock. I'll buy the \$90/\$110 *bull call spread* (long \$90 call, short \$110 call) and buy the \$90/\$110 *bear put spread* (long \$110 put, short \$90 put). The bull call spread pays $\max(S_T - 90, 0) - \max(S_T - 110, 0)$, the bear put spread pays $\max(110 - S_T, 0) - \max(90 - S_T, 0)$. Add them and the payoff is a constant **\$20** — the strike difference — whatever the stock does. The cost today: long \$90 call $-\$15$, short \$110 call $+\$4$, long \$110 put $-\$12$, short \$90 put $+\$3$, netting $-\$15 + 4 - 12 + 3 = -\$20$. So I pay \$20 today to receive \$20 in a year for sure. That implies a discount factor of $20/20 = 1$, i.e. an interest rate of **0%**. A box is just a synthetic zero-coupon bond, and its price reveals the market's implied financing rate. If the box cost only \$19 for a sure \$20, the implied rate would be $20/19 - 1 \approx 5.3\%$, and I'd compare that to my actual cost of funds to see if it's worth doing." Boxes are how desks borrow and lend through the options market, so this question is testing real machinery, not a toy.

#### Worked example: the dividend trap

**Problem.** A stock is \$100, the one-year rate is 5%. A one-year \$100 European call is \$7, the \$100 put is \$4. You're about to call it an arbitrage — but then the interviewer mentions the stock pays a \$3 dividend in three months. Recompute.

**Solution, narrated.** "Without the dividend I'd say parity wants $C - P = 100 - 95.12 = \$4.88$ and the market has $7 - 4 = \$3.00$, a \$1.88 gap. But the dividend changes the stock leg. The present value of the \$3 dividend, paid in three months, is $3 e^{-0.05 \cdot 0.25} = 3 \cdot 0.9876 = \$2.96$. Dividend-adjusted parity wants $C - P = S - D e^{-rt} - K e^{-rT} = 100 - 2.96 - 95.12 = \$1.92$. The market's \$3.00 is now only \$1.08 away — and in the *opposite* direction from what I first thought. If I'd traded on the no-dividend number I'd have sized the arbitrage wrong and lost money when the stock dropped \$3 on the ex-date. The real, much smaller, edge is \$1.08, and I'd want to double-check my dividend forecast and the ex-date before touching it." Stating the trap explicitly — "I almost forgot the dividend and it would have flipped my trade" — is exactly the self-aware reasoning these desks hire for.

#### Worked example: the early-exercise gotcha

**Problem.** You hold an American \$50-strike call on a non-dividend stock now trading at \$80. The stock has rallied hard and a colleague says "lock in the \$30, exercise now." Six months to expiry, 4% rate. What do you do, and how would your answer change if the stock paid a large dividend tomorrow?

**Solution, narrated.** "On a non-dividend stock I never exercise a call early. Exercising captures $S - K = \$30$. But the call is worth at least $S - K e^{-rT} = 80 - 50 e^{-0.04 \cdot 0.5} = 80 - 50 \cdot 0.9802 = 80 - 49.01 = \$30.99$ — more than \$30 — so I sell or hold rather than exercise; I'd be throwing away the \$0.99 of strike interest and downside protection. *However*, if the stock pays a large dividend tomorrow, the calculus can flip: exercising just before the ex-date lets me own the share and *collect the dividend*, which the call holder doesn't get. I'd exercise early only if that dividend exceeds the interest I'd forfeit on the strike plus the remaining time value — that's the one case where early call exercise is rational, and it's always right before an ex-dividend date, never at a random moment." Naming the *exception* — and that it only ever happens at an ex-dividend date — is the signal you understand the mechanism rather than a memorized rule.

## Common misconceptions

These are the beliefs that get candidates rejected. Each is wrong; here is the why.

**"Put-call parity needs Black-Scholes."** No — this is the single biggest misunderstanding, and stating it confidently is a fast way to fail. Parity is derived purely from no-arbitrage and static replication, with *no* assumption about how the stock moves, no volatility, no distribution. Black-Scholes is a *model* that prices the call and put individually; parity is a model-free *relationship between them* that holds no matter what model (if any) is right. In fact, you can use parity to *check* a model: if a Black-Scholes call and put don't satisfy parity, the implementation is buggy.

**"You should exercise an American call early to lock in profit when it's deep in-the-money."** Wrong for a non-dividend stock — you *never* exercise it early, however deep in-the-money, because selling the call always captures more than $S - K$ (you'd be forfeiting the strike interest and the downside protection). Exercising early is strictly dominated by selling. The only exception is to capture a dividend, and even then only right before the ex-date.

**"Parity tells you whether the call is over- or under-priced on its own."** No — parity is a relationship between *four* quantities ($C$, $P$, $S$, $K e^{-rT}$). A parity violation tells you the *package* is mispriced, but not which single leg is the culprit. To isolate a mispriced option you need an external reference (a model value, or another quote you trust). Confusing "parity is violated" with "the call is overpriced" is a subtle error interviewers probe for.

**"A higher-strike call could be worth more if the market expects a big rally."** Never — call value is *monotonically decreasing* in the strike, full stop, regardless of any rally expectation, because a higher strike is a strictly worse right to the same upside. If $C(K_2) > C(K_1)$ for $K_2 > K_1$, you sell the high-strike call and buy the low-strike one for a credit that can never turn into a loss. Expectations don't override no-arbitrage.

**"The arbitrage profit is huge, so these trades are everywhere."** In liquid, listed options, parity holds to within the bid-ask spread and the borrow cost almost all the time — market-makers compete the gap down to the friction floor in milliseconds. The textbook \$2.88 becomes a realized dollar or less, and often zero after costs. These arbitrages are real but rare and fleeting; the *skill* is in recognizing the structure and pricing the frictions, not in expecting to get rich on free lunches.

**"Convexity in strike is just a Black-Scholes artifact."** No — convexity in the strike is model-free, enforced by the butterfly spread whose payoff is a never-negative tent. *Any* arbitrage-free set of prices must be convex in the strike, whatever the true dynamics. A non-convex strike grid is an arbitrage in any world.

## How it shows up on a real desk

The classroom version is clean; the desk version is where these ideas earn their keep. Here is how the framework actually lives in a trading operation.

**Conversions and reversals as bread-and-butter trades.** Options market-makers run conversion and reversal books continuously. Whenever a customer's order pushes a call or put slightly off parity, the desk does the conversion or reversal to flatten the risk and pocket the tiny edge. These are not glamorous home-run trades; they are the high-frequency, low-margin grind that keeps a market-making book balanced. A desk's risk system flags any strike where the synthetic (call-minus-put) drifts from $S - K e^{-rT}$ by more than the cost of doing the box, and the trade fires automatically. Understanding parity *is* understanding how a market-maker neutralizes inventory.

**Box spreads as synthetic financing.** As the box-spread problem showed, a box is a synthetic zero-coupon bond: a fixed payoff at expiry for a fixed price today, with the implied interest rate baked into the price. Large desks and even sophisticated retail traders use box spreads to *borrow and lend* through the options market, sometimes at rates better than their bank offers, because the options market's implied financing rate can diverge from cash rates. This came to public attention in 2018 when a well-known retail trader lost a large sum on short box spreads during a volatility spike — a reminder that a "riskless" box is only riskless if you hold European options to expiry and don't get assigned early on the American versions. The mechanism is exactly the parity algebra above; the risk is in the fine print (American-style assignment, dividends, financing).

**Parity as a constraint on the volatility surface.** Modern desks quote options not as dollar prices but as *implied volatilities* — the volatility number that, plugged into a model, reproduces the market price. A crucial fact that falls out of parity: *a call and a put at the same strike and expiry must have the same implied volatility.* Why? Because parity ties their prices together exactly, and the model prices both off the same volatility input, so if they didn't share an implied vol, parity would be violated and a conversion/reversal would print money. This is why traders speak of "the implied vol at the \$100 strike" without specifying call or put — parity guarantees they're identical. The entire *volatility surface* (implied vol as a function of strike and expiry) inherits structure from no-arbitrage: it must produce call prices that are monotone and convex in the strike (no butterfly arbitrage) and that respect calendar bounds across expiries. A surface that violates these is not just "mispriced" — it is internally arbitrageable, and surface-fitting algorithms enforce exactly the no-arbitrage conditions from this article as hard constraints.

**Pricing the synthetic when the cash instrument is hard to trade.** Synthetic forwards (long call, short put) let a desk take a directional stock position when the stock itself is hard to short — because of borrow scarcity, a short-sale ban, or a hard-to-access market. During the 2008 short-selling bans on financial stocks, the cash short was illegal but the synthetic short (short call, long put) was not, and the relationship between the synthetic forward and the (frozen) cash market became a live, watched spread. Parity is what let traders read the "true" implied stock price out of the options when the stock market itself was constrained.

**The single most-asked structural question.** Across Jane Street, Optiver, SIG, IMC, Citadel Securities, Jump, and DRW, some version of "derive put-call parity" or "when do you exercise early" or "bound this option price" appears constantly in first-round and superday interviews — not because the firms expect you to run conversions by hand on day one, but because these questions cleanly separate candidates who *reason from replication* from those who *memorize formulas*. A candidate who can build parity from two portfolios in ninety seconds, spot a butterfly arbitrage from three quotes, and explain the call/put early-exercise asymmetry from the sign of the strike cash flow has demonstrated the exact cognitive move the job runs on all day: see a complicated payoff, decompose it into pieces you can price, and check that the prices are mutually consistent.

## When this matters and where to go next

If you are prepping for a quant-trader or derivatives interview, this is among the highest-leverage topics you can master, precisely because it is *model-free*. You cannot be stumped by a volatility assumption you don't know, because there isn't one. Drill until you can derive parity from the two portfolios without notes, recite the four price bounds, state the butterfly condition, and explain the early-exercise asymmetry in one breath. Then practice *narrating* — the answer matters less than the visible chain of replication reasoning that produces it.

Beyond interviews, this framework is the foundation the rest of derivatives sits on. Every pricing model you'll meet — Black-Scholes and its descendants — is required to *respect* these no-arbitrage relationships; the model adds a distributional assumption on top of the model-free skeleton you now own. When you study the [Black-Scholes model](/blog/trading/quantitative-finance/black-scholes), notice that its prices automatically satisfy parity and the bounds — that's a feature, not a coincidence. When you study the [volatility surface](/blog/trading/quantitative-finance/volatility-surface), notice that its no-arbitrage constraints (monotone and convex in strike) are exactly the ones you derived here with three calls and a butterfly. The broader landscape of [options theory](/blog/trading/quantitative-finance/options-theory) and [derivatives pricing](/blog/trading/quantitative-finance/derivatives-pricing) builds outward from this same no-arbitrage core.

A closing note in the spirit of honesty: nothing here is trading advice, and the "free lunches" above are textbook-frictionless. In real markets, listed options trade efficiently enough that pure parity arbitrage is essentially gone for anyone without a market-maker's cost structure. The lasting value is not the trade — it's the *way of thinking*. Once you internalize "two portfolios with the same payoff must cost the same," a whole class of problems that looks like it needs heavy math collapses into a few lines of clean replication. That collapse is the entire reason these questions are asked.
